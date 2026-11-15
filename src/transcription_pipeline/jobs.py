"""
Job handling: accept uploads fast, process them later in a bounded worker pool.

The in-memory queue and job registry are deliberately simple stand-ins. In
production ``JOB_QUEUE`` becomes Redis/SQS/RabbitMQ and ``JOBS`` becomes a
database table; the function bodies do not otherwise change.
"""

from __future__ import annotations

import hashlib
import os
import queue
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone

from .audio import load_audio, probe_audio_format
from .config import get_settings
from .formatting import format_timestamped_transcript
from .storage import store_transcription
from .stt import transcribe_audio, transcribe_long_audio

JOB_QUEUE: queue.Queue = queue.Queue(maxsize=get_settings()["queue_capacity"])
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
_WORKERS: list[threading.Thread] = []
_STOP = threading.Event()

STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_DEAD_LETTER = "dead_letter"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def reset_job_state(queue_capacity: int | None = None) -> None:
    """Clear the in-memory queue and registry (tests, or a fresh process)."""
    global JOB_QUEUE
    with JOBS_LOCK:
        JOBS.clear()
    JOB_QUEUE = queue.Queue(maxsize=queue_capacity or get_settings()["queue_capacity"])


def get_job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def list_jobs(status: str | None = None) -> list[dict]:
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    return sorted(jobs, key=lambda j: j["created_at"])


def find_job_by(field: str, value) -> dict | None:
    with JOBS_LOCK:
        return next((j for j in JOBS.values() if j.get(field) == value), None)


def submit_upload(
    stream,
    filename: str,
    language: str | None = None,
    model: str | None = None,
    idempotency_key: str | None = None,
    upload_dir: str | None = None,
) -> dict:
    """
    Accept an upload stream: write to disk with a size cap, dedupe by content hash,
    probe, enqueue, and return a job record. Returns immediately; a worker pool
    drains JOB_QUEUE and runs load_audio -> transcribe -> store.

    Raises ValueError for bad input (→ HTTP 400) and RuntimeError when the queue
    is full (→ HTTP 503 with Retry-After).
    """
    settings = get_settings()
    upload_dir = upload_dir or settings["upload_dir"]
    model = model or settings["model_size"]

    if idempotency_key:
        existing = find_job_by("idempotency_key", idempotency_key)
        if existing:
            return existing

    os.makedirs(upload_dir, exist_ok=True)
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(upload_dir, job_id)
    os.makedirs(job_dir)
    safe_name = os.path.basename(filename) or "upload.bin"
    dest = os.path.join(job_dir, safe_name)

    digest = hashlib.sha256()
    written = 0
    try:
        with open(dest, "wb") as fh:
            while chunk := stream.read(1024 * 1024):
                written += len(chunk)
                if written > settings["max_file_bytes"]:
                    raise ValueError(f"Upload exceeds {settings['max_file_bytes']} bytes")
                digest.update(chunk)
                fh.write(chunk)
        if written == 0:
            raise ValueError("Upload is empty")
        content_hash = digest.hexdigest()

        # Idempotency on content: identical bytes + same model → same job.
        with JOBS_LOCK:
            dup = next(
                (
                    j
                    for j in JOBS.values()
                    if j["content_hash"] == content_hash and j["model"] == model and j["status"] != STATUS_FAILED
                ),
                None,
            )
        if dup:
            shutil.rmtree(job_dir, ignore_errors=True)
            return dup

        probe = probe_audio_format(dest)  # rejects non-media before it hits the queue

        job = {
            "job_id": job_id,
            "content_hash": content_hash,
            "filename": safe_name,
            "path": dest,
            "size_bytes": written,
            "status": STATUS_QUEUED,
            "language": language,
            "model": model,
            "idempotency_key": idempotency_key,
            "duration_sec": probe["duration_sec"],
            "is_video": probe["is_video"],
            "warnings": probe["warnings"],
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
            "attempts": 0,
        }
        with JOBS_LOCK:
            JOBS[job_id] = job

        try:
            JOB_QUEUE.put(job, block=False)
        except queue.Full:
            with JOBS_LOCK:
                JOBS.pop(job_id, None)
            raise RuntimeError("Service at capacity; retry later") from None
        return job

    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise


def process_job(job: dict) -> dict:
    """
    Run the full pipeline for one job: normalise → transcribe (chunked when long)
    → persist. Mutates and returns the job record. Never raises; failures land
    in job["status"] / job["error"] so the worker loop keeps draining.
    """
    settings = get_settings()
    job["status"] = STATUS_PROCESSING
    job["started_at"] = _now()
    job["attempts"] = job.get("attempts", 0) + 1
    normalised = None
    try:
        loaded = load_audio(job["path"])
        normalised = loaded["normalised_path"]

        if loaded["duration_sec"] > settings["long_threshold_sec"]:
            result = transcribe_long_audio(
                normalised,
                model_size=job["model"],
                language=job.get("language"),
                chunk_max_sec=settings["long_threshold_sec"],
            )
        else:
            result = transcribe_audio(normalised, model_size=job["model"], language=job.get("language"))
            result = {**result, "chunks": 1, "failed_chunks": []}

        store_transcription(job["path"], job["content_hash"], result)
        job["result"] = result
        job["status"] = STATUS_DEAD_LETTER if result.get("failed_chunks") else STATUS_DONE
        job["error"] = None
    except Exception as exc:
        job["status"] = STATUS_FAILED
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["finished_at"] = _now()
        if normalised and os.path.exists(normalised):
            os.remove(normalised)
    return job


def render_job(job: dict, fmt: str = "json") -> str:
    """Render a finished job's transcript in the requested format."""
    if not job.get("result"):
        raise ValueError("Job has no result to render")
    return format_timestamped_transcript(job["result"], fmt=fmt)


def delete_job(job_id: str) -> bool:
    """Remove a job and its upload directory. Returns False if unknown."""
    with JOBS_LOCK:
        job = JOBS.pop(job_id, None)
    if not job:
        return False
    shutil.rmtree(os.path.dirname(job["path"]), ignore_errors=True)
    return True


def run_worker(stop: threading.Event, poll_sec: float = 0.5) -> None:
    """Drain JOB_QUEUE until ``stop`` is set. One model instance per worker thread."""
    while not stop.is_set():
        try:
            job = JOB_QUEUE.get(timeout=poll_sec)
        except queue.Empty:
            continue
        try:
            process_job(job)
        finally:
            JOB_QUEUE.task_done()


def start_workers(count: int | None = None) -> list[threading.Thread]:
    """Start background worker threads (idempotent)."""
    count = count or get_settings()["worker_count"]
    if _WORKERS:
        return _WORKERS
    _STOP.clear()
    for i in range(count):
        t = threading.Thread(target=run_worker, args=(_STOP,), name=f"stt-worker-{i}", daemon=True)
        t.start()
        _WORKERS.append(t)
    return _WORKERS


def stop_workers(timeout: float = 5.0) -> None:
    """Signal workers to stop and wait briefly for them."""
    _STOP.set()
    deadline = time.monotonic() + timeout
    for t in _WORKERS:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
    _WORKERS.clear()


def wait_for_job(job_id: str, timeout: float = 60.0, poll_sec: float = 0.1) -> dict:
    """Block until a job leaves queued/processing or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] not in (STATUS_QUEUED, STATUS_PROCESSING):
            return job
        time.sleep(poll_sec)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")
