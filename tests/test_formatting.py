import json

import pytest

from transcription_pipeline.formatting import format_timestamp, format_timestamped_transcript, merge_segments

SAMPLE = {
    "language": "en",
    "duration_sec": 7.5,
    "model": "small",
    "mean_logprob": -0.4,
    "needs_review": False,
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.2, "text": "Hello", "avg_logprob": -0.2},
        {"id": 1, "start": 1.3, "end": 2.5, "text": "world", "avg_logprob": -0.6},
        {"id": 2, "start": 4.0, "end": 5.0, "text": "", "avg_logprob": -0.1},
        {"id": 3, "start": 5.5, "end": 7.5, "text": "Bye", "avg_logprob": -0.3},
    ],
}


def test_format_timestamp_exact():
    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(59.9999) == "00:01:00.000"
    assert format_timestamp(3661.5, ",") == "01:01:01,500"
    assert format_timestamp(-1) == "00:00:00.000"


def test_merge_segments_joins_close_and_drops_empty():
    merged = merge_segments(SAMPLE["segments"], merge_gap_sec=0.3, max_segment_sec=15)
    assert [m["text"] for m in merged] == ["Hello world", "Bye"]
    assert merged[0]["avg_logprob"] == -0.6  # worst of the merged parts
    assert [m["id"] for m in merged] == [1, 2]


def test_merge_segments_respects_max_length():
    segs = [{"start": i * 1.0, "end": i * 1.0 + 1.0, "text": f"w{i}", "avg_logprob": -0.1} for i in range(10)]
    merged = merge_segments(segs, merge_gap_sec=0.5, max_segment_sec=3.0)
    assert all(m["end"] - m["start"] <= 3.0 for m in merged)
    assert len(merged) == 4


def test_json_output():
    out = json.loads(format_timestamped_transcript(SAMPLE, "json"))
    assert out["language"] == "en"
    assert out["text"] == "Hello world Bye"
    assert len(out["segments"]) == 2
    assert out["failed_chunks"] == []


def test_srt_output():
    srt = format_timestamped_transcript(SAMPLE, "srt")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nHello world\n")
    assert "2\n00:00:05,500 --> 00:00:07,500\nBye" in srt


def test_vtt_output():
    vtt = format_timestamped_transcript(SAMPLE, "vtt")
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:02.500\nHello world" in vtt


def test_txt_output():
    txt = format_timestamped_transcript(SAMPLE, "txt")
    assert txt.splitlines() == [
        "[00:00:00.000 --> 00:00:02.500] Hello world",
        "[00:00:05.500 --> 00:00:07.500] Bye",
    ]


def test_unknown_format():
    with pytest.raises(ValueError):
        format_timestamped_transcript(SAMPLE, "docx")


def test_empty_transcription():
    assert json.loads(format_timestamped_transcript({"segments": []}, "json"))["segments"] == []
    assert format_timestamped_transcript({"segments": []}, "srt") == ""
