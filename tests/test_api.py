import pytest
from fastapi.testclient import TestClient

from transcription_pipeline import config, jobs
from transcription_pipeline.api import create_api

from .conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


@pytest.fixture
def client(patch_stt):
    # Workers off: tests drive process_job directly for determinism.
    app = create_api(start_workers=False)
    with TestClient(app) as c:
        yield c


def _upload(client, path, **params):
    with open(path, "rb") as fh:
        return client.post("/v1/transcriptions", files={"file": ("clip.wav", fh, "audio/wav")}, params=params)


def test_health(client):
    assert client.get("/healthz").json()["ok"] is True
    ready = client.get("/readyz").json()
    assert ready["queue_capacity"] > 0
    assert "X-Request-ID" in client.get("/healthz").headers


def test_submit_status_transcript_flow(client, make_wav):
    path = make_wav(duration_sec=4.0)
    r = _upload(client, path, language="en")
    assert r.status_code == 202
    body = r.json()
    job_id = body["job_id"]
    assert body["status"] == "queued"
    assert body["status_url"].endswith(job_id)

    # Not ready yet → 409 with Retry-After
    r = client.get(f"/v1/transcriptions/{job_id}/transcript")
    assert r.status_code == 409
    assert "Retry-After" in r.headers

    jobs.process_job(jobs.get_job(job_id))

    status = client.get(f"/v1/transcriptions/{job_id}").json()
    assert status["status"] == "done"
    assert status["detected_language"] == "en"
    assert status["needs_review"] is False

    js = client.get(f"/v1/transcriptions/{job_id}/transcript").json()
    assert js["text"] == "seg0 seg1"

    srt = client.get(f"/v1/transcriptions/{job_id}/transcript", params={"format": "srt"})
    assert srt.headers["content-type"].startswith("application/x-subrip")
    assert "00:00:00,000 -->" in srt.text

    vtt = client.get(f"/v1/transcriptions/{job_id}/transcript", params={"format": "vtt"})
    assert vtt.text.startswith("WEBVTT")

    bad = client.get(f"/v1/transcriptions/{job_id}/transcript", params={"format": "docx"})
    assert bad.status_code == 422

    segs = client.get(f"/v1/transcriptions/{job_id}/segments", params={"start": 2.5}).json()
    assert [s["text"] for s in segs["segments"]] == ["seg1"]

    listed = client.get("/v1/transcriptions").json()["jobs"]
    assert [j["job_id"] for j in listed] == [job_id]


def test_idempotency_header(client, make_wav):
    path = make_wav(duration_sec=2.0)
    with open(path, "rb") as fh:
        a = client.post(
            "/v1/transcriptions", files={"file": ("a.wav", fh, "audio/wav")}, headers={"Idempotency-Key": "abc"}
        ).json()
    other = make_wav("other.wav", 3.0)
    with open(other, "rb") as fh:
        b = client.post(
            "/v1/transcriptions", files={"file": ("b.wav", fh, "audio/wav")}, headers={"Idempotency-Key": "abc"}
        ).json()
    assert a["job_id"] == b["job_id"]


def test_bad_upload_is_400(client, tmp_path):
    bogus = tmp_path / "x.wav"
    bogus.write_text("hello")
    r = _upload(client, str(bogus))
    assert r.status_code == 400


def test_capacity_is_503(client, make_wav):
    jobs.reset_job_state(queue_capacity=1)
    assert _upload(client, make_wav("a.wav", 2.0)).status_code == 202
    r = _upload(client, make_wav("b.wav", 2.5))
    assert r.status_code == 503
    assert r.headers["Retry-After"] == "30"


def test_unknown_job_404(client):
    assert client.get("/v1/transcriptions/nope").status_code == 404
    assert client.get("/v1/transcriptions/nope/transcript").status_code == 404
    assert client.delete("/v1/transcriptions/nope").status_code == 404
    assert client.post("/v1/transcriptions/nope/retry").status_code == 404


def test_failed_job_transcript_422_and_retry_requeues(client, make_wav, patch_stt):
    patch_stt(fail_on={"stt_"})
    job_id = _upload(client, make_wav(duration_sec=2.0)).json()["job_id"]
    jobs.process_job(jobs.get_job(job_id))
    assert client.get(f"/v1/transcriptions/{job_id}").json()["status"] == "failed"
    assert client.get(f"/v1/transcriptions/{job_id}/transcript").status_code == 422

    patch_stt()  # model healthy again
    r = client.post(f"/v1/transcriptions/{job_id}/retry")
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
    jobs.process_job(jobs.get_job(job_id))
    assert client.get(f"/v1/transcriptions/{job_id}").json()["status"] == "done"


def test_dead_letter_retry_recovers(client, make_wav, patch_stt, monkeypatch):
    monkeypatch.setenv("TP_LONG_THRESHOLD_SEC", "30")
    monkeypatch.setenv("TP_CHUNK_TARGET_SEC", "20")
    config.reset_settings_cache()
    patch_stt(fail_on={"chunk001"})
    job_id = _upload(client, make_wav(duration_sec=60.0)).json()["job_id"]
    jobs.process_job(jobs.get_job(job_id))
    status = client.get(f"/v1/transcriptions/{job_id}").json()
    assert status["status"] == "dead_letter"
    assert len(status["failed_chunks"]) == 1

    patch_stt()  # retry succeeds
    r = client.post(f"/v1/transcriptions/{job_id}/retry")
    assert r.status_code == 200
    assert r.json()["status"] == "done"
    assert r.json()["failed_chunks"] == []
    js = client.get(f"/v1/transcriptions/{job_id}/transcript").json()
    assert js["segments"][-1]["end"] == pytest.approx(60.0, abs=0.1)


def test_retry_on_done_job_is_409(client, make_wav):
    job_id = _upload(client, make_wav(duration_sec=2.0)).json()["job_id"]
    jobs.process_job(jobs.get_job(job_id))
    assert client.post(f"/v1/transcriptions/{job_id}/retry").status_code == 409


def test_delete_removes_job_and_storage(client, make_wav):
    from transcription_pipeline.storage import load_transcription

    job_id = _upload(client, make_wav(duration_sec=2.0)).json()["job_id"]
    job = jobs.get_job(job_id)
    jobs.process_job(job)
    assert load_transcription(job["content_hash"]) is not None
    assert client.delete(f"/v1/transcriptions/{job_id}").status_code == 204
    assert client.get(f"/v1/transcriptions/{job_id}").status_code == 404
    assert load_transcription(job["content_hash"]) is None
