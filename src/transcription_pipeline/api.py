"""
HTTP API: asynchronous job interface over the pipeline.

POST   /v1/transcriptions                 submit (202 + job id)
GET    /v1/transcriptions                 list jobs
GET    /v1/transcriptions/{id}            status
GET    /v1/transcriptions/{id}/transcript rendered output (json|srt|vtt|txt)
GET    /v1/transcriptions/{id}/segments   time-range query
POST   /v1/transcriptions/{id}/retry      recover failed chunks
DELETE /v1/transcriptions/{id}            remove audio + transcript + segments
GET    /healthz, /readyz
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__, jobs
from .config import get_settings
from .formatting import FORMATS, MEDIA_TYPES
from .retry import retry_failed_transcription
from .storage import delete_transcription, store_transcription

PUBLIC_FIELDS = (
    "job_id",
    "status",
    "filename",
    "size_bytes",
    "duration_sec",
    "is_video",
    "warnings",
    "language",
    "model",
    "created_at",
    "started_at",
    "finished_at",
    "error",
    "attempts",
)


def _public(job: dict) -> dict:
    out = {k: job.get(k) for k in PUBLIC_FIELDS}
    out["status_url"] = f"/v1/transcriptions/{job['job_id']}"
    result = job.get("result")
    if result:
        out.update(
            {
                "detected_language": result.get("language"),
                "mean_logprob": result.get("mean_logprob"),
                "needs_review": result.get("needs_review"),
                "chunks": result.get("chunks"),
                "failed_chunks": result.get("failed_chunks", []),
                "transcript_url": f"/v1/transcriptions/{job['job_id']}/transcript",
            }
        )
    return out


def create_api(start_workers: bool = True) -> FastAPI:
    """Build the FastAPI app. Workers run in-process as threads for the mock/local case."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_workers:
            jobs.start_workers()
        yield
        if start_workers:
            jobs.stop_workers()

    app = FastAPI(title="Transcription API", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.post("/v1/transcriptions", status_code=202)
    async def create(
        file: UploadFile = File(...),
        language: str | None = Query(default=None, description="ISO code; omit for auto-detect"),
        model: str | None = Query(default=None, description="tiny/base/small/medium/large-v3"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        try:
            job = jobs.submit_upload(
                file.file,
                file.filename or "upload.bin",
                language=language,
                model=model,
                idempotency_key=idempotency_key,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "30"}) from exc
        return _public(job)

    @app.get("/v1/transcriptions")
    async def list_all(status: str | None = None):
        return {"jobs": [_public(j) for j in jobs.list_jobs(status)]}

    @app.get("/v1/transcriptions/{job_id}")
    async def status(job_id: str):
        job = jobs.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return _public(job)

    @app.get("/v1/transcriptions/{job_id}/transcript")
    async def transcript(job_id: str, format: str = Query("json", pattern=f"^({'|'.join(FORMATS)})$")):
        job = jobs.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] in (jobs.STATUS_QUEUED, jobs.STATUS_PROCESSING):
            raise HTTPException(409, f"Job is {job['status']}", headers={"Retry-After": "10"})
        if job["status"] == jobs.STATUS_FAILED:
            raise HTTPException(422, job.get("error") or "Transcription failed")
        body = jobs.render_job(job, fmt=format)
        if format == "json":
            return JSONResponse(content=json.loads(body))
        return PlainTextResponse(body, media_type=MEDIA_TYPES[format])

    @app.get("/v1/transcriptions/{job_id}/segments")
    async def segments(job_id: str, start: float = 0.0, end: float | None = None):
        job = jobs.get_job(job_id)
        if not job or not job.get("result"):
            raise HTTPException(404, "Transcript not available")
        segs = [s for s in job["result"]["segments"] if s["end"] >= start and (end is None or s["start"] <= end)]
        return {"job_id": job_id, "start": start, "end": end, "segments": segs}

    @app.post("/v1/transcriptions/{job_id}/retry")
    async def retry(job_id: str):
        job = jobs.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] == jobs.STATUS_FAILED:
            # Whole-job failure: re-queue from scratch.
            job["status"] = jobs.STATUS_QUEUED
            job["error"] = None
            try:
                jobs.JOB_QUEUE.put(job, block=False)
            except Exception as exc:
                raise HTTPException(503, "Service at capacity; retry later", headers={"Retry-After": "30"}) from exc
            return _public(job)
        if job["status"] != jobs.STATUS_DEAD_LETTER:
            raise HTTPException(409, f"Job is {job['status']}; nothing to retry")
        if not os.path.exists(job["path"]):
            raise HTTPException(410, "Original audio no longer available")
        recovered = retry_failed_transcription(
            {**job["result"], "model": job["model"], "language": job.get("language")}, job["path"]
        )
        store_transcription(job["path"], job["content_hash"], recovered)
        job["result"] = recovered
        job["status"] = jobs.STATUS_DEAD_LETTER if recovered["failed_chunks"] else jobs.STATUS_DONE
        job["attempts"] = job.get("attempts", 0) + 1
        return _public(job)

    @app.delete("/v1/transcriptions/{job_id}", status_code=204)
    async def delete(job_id: str):
        job = jobs.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        content_hash = job["content_hash"]
        jobs.delete_job(job_id)
        # Only drop stored data if no other job references the same content.
        if not jobs.find_job_by("content_hash", content_hash):
            delete_transcription(content_hash)
        return None

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "version": __version__}

    @app.get("/readyz")
    async def readyz():
        settings = get_settings()
        return {
            "ok": True,
            "queue_depth": jobs.JOB_QUEUE.qsize(),
            "queue_capacity": jobs.JOB_QUEUE.maxsize,
            "workers": len(jobs._WORKERS),
            "model": settings["model_size"],
            "device": settings["device"],
        }

    return app


app = create_api()


def serve() -> None:
    """Console entry point: ``transcribe-api``."""
    import uvicorn

    uvicorn.run(
        "transcription_pipeline.api:app",
        host=os.environ.get("TP_HOST", "0.0.0.0"),
        port=int(os.environ.get("TP_PORT", "8000")),
        workers=1,  # job registry is in-process; scale with a shared queue instead
    )
