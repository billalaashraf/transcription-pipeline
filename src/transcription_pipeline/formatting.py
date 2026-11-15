"""
Stage 3: output rendering.

The STT stage produces one canonical structure; this module projects it into
whatever a consumer wants (json / srt / vtt / txt). Rendering never touches
model code, and the same transcription can be rendered several ways without
re-running inference.
"""

from __future__ import annotations

import json

FORMATS = ("json", "srt", "vtt", "txt")
MEDIA_TYPES = {
    "json": "application/json",
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "txt": "text/plain",
}


def format_timestamp(seconds: float, sep: str = ".") -> str:
    """HH:MM:SS<sep>mmm using integer millisecond arithmetic (no float drift)."""
    ms = int(round(max(seconds, 0.0) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def merge_segments(segments: list[dict], merge_gap_sec: float = 0.3, max_segment_sec: float = 15.0) -> list[dict]:
    """
    Merge fragmented STT segments into readable blocks. Adjacent segments closer
    than ``merge_gap_sec`` are joined until a block would exceed ``max_segment_sec``.
    Confidence is carried through as the minimum of the merged parts.
    """
    merged: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        logprob = seg.get("avg_logprob")
        if (
            merged
            and seg["start"] - merged[-1]["end"] <= merge_gap_sec
            and seg["end"] - merged[-1]["start"] <= max_segment_sec
        ):
            prev = merged[-1]
            prev["end"] = seg["end"]
            prev["text"] = f"{prev['text']} {text}"
            if logprob is not None:
                prev["avg_logprob"] = logprob if prev["avg_logprob"] is None else min(prev["avg_logprob"], logprob)
        else:
            merged.append({"start": seg["start"], "end": seg["end"], "text": text, "avg_logprob": logprob})

    for i, seg in enumerate(merged, start=1):
        seg["id"] = i
    return merged


def format_timestamped_transcript(
    transcription: dict,
    fmt: str = "json",
    merge_gap_sec: float = 0.3,
    max_segment_sec: float = 15.0,
) -> str:
    """
    Render the output of transcribe_audio / transcribe_long_audio as a
    timestamped transcript in json, srt, vtt, or txt.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format '{fmt}'. Use one of {FORMATS}")

    merged = merge_segments(transcription.get("segments", []), merge_gap_sec, max_segment_sec)

    if fmt == "json":
        return json.dumps(
            {
                "language": transcription.get("language"),
                "duration_sec": transcription.get("duration_sec"),
                "model": transcription.get("model"),
                "mean_logprob": transcription.get("mean_logprob"),
                "needs_review": bool(transcription.get("needs_review", False)),
                "failed_chunks": transcription.get("failed_chunks", []),
                "text": " ".join(s["text"] for s in merged),
                "segments": merged,
            },
            ensure_ascii=False,
            indent=2,
        )

    if fmt == "srt":
        return "\n".join(
            f"{s['id']}\n{format_timestamp(s['start'], ',')} --> {format_timestamp(s['end'], ',')}\n{s['text']}\n"
            for s in merged
        )

    if fmt == "vtt":
        blocks = ["WEBVTT\n"] + [
            f"{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n{s['text']}\n" for s in merged
        ]
        return "\n".join(blocks)

    # txt
    return "\n".join(f"[{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}] {s['text']}" for s in merged)
