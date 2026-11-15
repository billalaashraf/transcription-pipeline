import os

from transcription_pipeline.storage import (
    delete_transcription,
    hash_file,
    load_transcription,
    query_segments,
    store_transcription,
)

RESULT = {
    "model": "small",
    "language": "en",
    "duration_sec": 4.0,
    "mean_logprob": -0.3,
    "needs_review": False,
    "text": "a b",
    "segments": [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "a", "avg_logprob": -0.2},
        {"id": 1, "start": 2.0, "end": 4.0, "text": "b", "avg_logprob": -0.4},
    ],
}


def test_store_and_load_roundtrip(make_wav):
    path = make_wav(duration_sec=4.0)
    h = hash_file(path)
    stored = store_transcription(path, h, RESULT)
    assert stored["segments"] == 2
    assert os.path.exists(stored["audio_key"])
    assert h in stored["audio_key"]

    loaded = load_transcription(h)
    assert loaded["model"] == "small"
    assert loaded["text"] == "a b"
    assert [s["text"] for s in loaded["segments"]] == ["a", "b"]
    assert loaded["segments"][1]["start"] == 2.0


def test_store_is_idempotent_per_model(make_wav):
    path = make_wav(duration_sec=4.0)
    h = hash_file(path)
    first = store_transcription(path, h, RESULT)
    updated = {**RESULT, "text": "a b revised", "segments": RESULT["segments"][:1]}
    second = store_transcription(path, h, updated)
    assert first["transcript_id"] == second["transcript_id"]
    loaded = load_transcription(h, model="small")
    assert loaded["text"] == "a b revised"
    assert len(loaded["segments"]) == 1


def test_different_models_coexist(make_wav):
    path = make_wav(duration_sec=4.0)
    h = hash_file(path)
    a = store_transcription(path, h, RESULT)
    b = store_transcription(path, h, {**RESULT, "model": "medium"})
    assert a["transcript_id"] != b["transcript_id"]
    assert load_transcription(h, model="medium")["model"] == "medium"


def test_query_segments_time_range(make_wav):
    path = make_wav(duration_sec=4.0)
    h = hash_file(path)
    store_transcription(path, h, RESULT)
    assert [s["text"] for s in query_segments(h, start=2.5)] == ["b"]
    assert [s["text"] for s in query_segments(h, start=0, end=1.0)] == ["a"]
    assert query_segments("deadbeef") == []


def test_delete_removes_everything(make_wav):
    path = make_wav(duration_sec=4.0)
    h = hash_file(path)
    stored = store_transcription(path, h, RESULT)
    out = delete_transcription(h)
    assert out == {"transcripts_deleted": 1, "blobs_deleted": 1}
    assert not os.path.exists(stored["audio_key"])
    assert load_transcription(h) is None
    assert query_segments(h) == []


def test_load_missing_returns_none():
    assert load_transcription("0" * 64) is None
