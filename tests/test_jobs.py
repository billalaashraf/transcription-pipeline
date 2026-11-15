import io
import os

import pytest

from transcription_pipeline import config, jobs

from .conftest import requires_ffmpeg


def _stream(path):
    with open(path, "rb") as fh:
        return io.BytesIO(fh.read())


@requires_ffmpeg
def test_submit_upload_enqueues_job(make_wav):
    path = make_wav(duration_sec=3.0)
    job = jobs.submit_upload(_stream(path), "clip.wav")
    assert job["status"] == jobs.STATUS_QUEUED
    assert job["duration_sec"] == pytest.approx(3.0, abs=0.05)
    assert os.path.exists(job["path"])
    assert jobs.JOB_QUEUE.qsize() == 1
    assert jobs.get_job(job["job_id"]) is job


@requires_ffmpeg
def test_submit_upload_dedupes_identical_content(make_wav):
    path = make_wav(duration_sec=3.0)
    a = jobs.submit_upload(_stream(path), "one.wav")
    b = jobs.submit_upload(_stream(path), "two.wav")
    assert a["job_id"] == b["job_id"]
    assert jobs.JOB_QUEUE.qsize() == 1
    # Only one upload directory remains
    upload_dir = os.path.dirname(os.path.dirname(a["path"]))
    assert len(os.listdir(upload_dir)) == 1


@requires_ffmpeg
def test_submit_upload_idempotency_key(make_wav):
    a = jobs.submit_upload(_stream(make_wav("a.wav", 2.0)), "a.wav", idempotency_key="k1")
    b = jobs.submit_upload(_stream(make_wav("b.wav", 2.5)), "b.wav", idempotency_key="k1")
    assert a is b


def test_submit_upload_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        jobs.submit_upload(io.BytesIO(b""), "empty.wav")
    assert jobs.JOB_QUEUE.qsize() == 0


def test_submit_upload_enforces_size_cap(monkeypatch):
    monkeypatch.setenv("TP_MAX_FILE_BYTES", "10")
    config.reset_settings_cache()
    with pytest.raises(ValueError, match="exceeds"):
        jobs.submit_upload(io.BytesIO(b"x" * 100), "big.wav")


@requires_ffmpeg
def test_submit_upload_rejects_non_media():
    with pytest.raises(ValueError, match="decodable"):
        jobs.submit_upload(io.BytesIO(b"not audio at all"), "fake.wav")
    assert jobs.list_jobs() == []


@requires_ffmpeg
def test_submit_upload_backpressure(make_wav):
    jobs.reset_job_state(queue_capacity=1)
    jobs.submit_upload(_stream(make_wav("a.wav", 2.0)), "a.wav")
    with pytest.raises(RuntimeError, match="capacity"):
        jobs.submit_upload(_stream(make_wav("b.wav", 2.5)), "b.wav")
    assert len(jobs.list_jobs()) == 1


@requires_ffmpeg
def test_process_job_end_to_end(make_wav, patch_stt):
    path = make_wav(duration_sec=4.0)
    job = jobs.submit_upload(_stream(path), "clip.wav")
    jobs.process_job(job)
    assert job["status"] == jobs.STATUS_DONE
    assert job["result"]["text"] == "seg0 seg1"
    assert job["attempts"] == 1
    assert job["finished_at"]
    assert jobs.render_job(job, "txt").startswith("[00:00:00.000")


@requires_ffmpeg
def test_process_job_records_failure(make_wav, patch_stt):
    patch_stt(fail_on={"stt_"})  # every normalised temp file matches
    job = jobs.submit_upload(_stream(make_wav(duration_sec=2.0)), "clip.wav")
    jobs.process_job(job)
    assert job["status"] == jobs.STATUS_FAILED
    assert "simulated failure" in job["error"]


@requires_ffmpeg
def test_process_job_dead_letters_partial(make_wav, patch_stt, monkeypatch):
    monkeypatch.setenv("TP_LONG_THRESHOLD_SEC", "30")
    monkeypatch.setenv("TP_CHUNK_TARGET_SEC", "20")
    config.reset_settings_cache()
    patch_stt(fail_on={"chunk001"})
    job = jobs.submit_upload(_stream(make_wav(duration_sec=60.0)), "long.wav")
    jobs.process_job(job)
    assert job["status"] == jobs.STATUS_DEAD_LETTER
    assert len(job["result"]["failed_chunks"]) == 1


@requires_ffmpeg
def test_worker_drains_queue(make_wav, patch_stt):
    job = jobs.submit_upload(_stream(make_wav(duration_sec=3.0)), "clip.wav")
    jobs.start_workers(1)
    done = jobs.wait_for_job(job["job_id"], timeout=15)
    assert done["status"] == jobs.STATUS_DONE
    jobs.stop_workers()


@requires_ffmpeg
def test_delete_job_removes_upload(make_wav):
    job = jobs.submit_upload(_stream(make_wav(duration_sec=2.0)), "clip.wav")
    job_dir = os.path.dirname(job["path"])
    assert jobs.delete_job(job["job_id"]) is True
    assert not os.path.exists(job_dir)
    assert jobs.delete_job("missing") is False
