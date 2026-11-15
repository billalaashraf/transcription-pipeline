import pytest

from transcription_pipeline.retry import backoff_delay, classify_error, escalation_plan, retry_failed_transcription


def _job_with_failed_chunk():
    return {
        "model": "small",
        "language": None,
        "mean_logprob": -0.3,
        "needs_review": True,
        "text": "seg0",
        "segments": [{"id": 0, "start": 0.0, "end": 20.0, "text": "seg0", "avg_logprob": -0.3}],
        "failed_chunks": [{"chunk": 1, "start": 20.0, "end": 40.0, "error": "RuntimeError: boom"}],
    }


def test_classify_error():
    assert classify_error(ValueError("bad input")) == "permanent"
    assert classify_error(FileNotFoundError()) == "permanent"
    assert classify_error(RuntimeError("oom")) == "transient"
    assert classify_error(TimeoutError()) == "transient"


def test_escalation_plan_steps_up_model():
    plan = escalation_plan("small", "en")
    assert plan[0] == {"model_size": "small"}
    assert plan[1] == {"model_size": "small", "language": "en"}
    assert plan[2] == {"model_size": "medium", "language": "en"}


def test_backoff_bounded():
    assert backoff_delay(0, 30, 600, jitter=False) == 30
    assert backoff_delay(3, 30, 600, jitter=False) == 240
    assert backoff_delay(10, 30, 600, jitter=False) == 600
    d = backoff_delay(1, 30, 600)
    assert 30 <= d <= 90


def test_retry_recovers_chunk(make_wav):
    wav = make_wav(duration_sec=40.0)
    calls = []

    def run_chunk(path, cfg):
        calls.append(cfg)
        return {"segments": [{"id": 0, "start": 0.0, "end": 20.0, "text": "recovered", "avg_logprob": -0.5}]}

    out = retry_failed_transcription(_job_with_failed_chunk(), wav, run_chunk=run_chunk, sleep=lambda s: None)
    assert out["status"] == "done"
    assert out["failed_chunks"] == []
    assert out["recovered_chunks"] == 1
    assert [s["text"] for s in out["segments"]] == ["seg0", "recovered"]
    assert out["segments"][1]["start"] == 20.0  # offset applied
    assert out["segments"][1]["end"] == 40.0
    assert out["text"] == "seg0 recovered"
    assert out["gaps"] == []
    assert calls == [{"model_size": "small"}]


def test_retry_escalates_then_dead_letters(make_wav):
    wav = make_wav(duration_sec=40.0)
    calls, sleeps = [], []

    def run_chunk(path, cfg):
        calls.append(cfg)
        raise RuntimeError("transient failure")

    out = retry_failed_transcription(
        _job_with_failed_chunk(), wav, run_chunk=run_chunk, sleep=sleeps.append, max_attempts=3
    )
    assert out["status"] == "dead_letter"
    assert len(out["failed_chunks"]) == 1
    history = out["failed_chunks"][0]["attempts"]
    assert len(history) == 3
    assert [h["config"]["model_size"] for h in history] == ["small", "small", "medium"]
    assert all(not h["ok"] for h in history)
    assert len(sleeps) == 2  # no sleep after the final attempt
    assert out["gaps"] == [{"start": 20.0, "end": 40.0}]
    assert out["needs_review"] is True


def test_retry_stops_on_permanent_error(make_wav):
    wav = make_wav(duration_sec=40.0)
    calls = []

    def run_chunk(path, cfg):
        calls.append(cfg)
        raise ValueError("corrupt region")

    out = retry_failed_transcription(_job_with_failed_chunk(), wav, run_chunk=run_chunk, sleep=lambda s: None)
    assert len(calls) == 1
    assert out["status"] == "dead_letter"
    assert "permanent" in out["failed_chunks"][0]["attempts"][0]["error"]


def test_retry_resumes_from_attempt_history(make_wav):
    wav = make_wav(duration_sec=40.0)
    job = _job_with_failed_chunk()
    job["failed_chunks"][0]["attempts"] = [{"attempt": 0, "config": {"model_size": "small"}, "ok": False}]
    calls = []

    def run_chunk(path, cfg):
        calls.append(cfg)
        return {"segments": []}

    retry_failed_transcription(job, wav, run_chunk=run_chunk, sleep=lambda s: None)
    assert calls == [{"model_size": "small", "language": None}]  # started at attempt 1


def test_retry_nothing_to_do(make_wav):
    wav = make_wav(duration_sec=4.0)
    job = {**_job_with_failed_chunk(), "failed_chunks": [], "needs_review": False}
    out = retry_failed_transcription(job, wav, run_chunk=lambda p, c: pytest.fail("should not run"))
    assert out["status"] == "done"
    assert out["segments"] == job["segments"]
