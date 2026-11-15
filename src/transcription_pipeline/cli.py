"""
Command-line entry point: ``transcribe <file> [--format srt] [--output out.srt]``.

Wires the stages together for one file, synchronously. For many files or a
service, use the API (``transcribe-api``).
"""

from __future__ import annotations

import argparse
import os
import sys

from .audio import load_audio, probe_audio_format
from .config import get_settings
from .formatting import FORMATS, format_timestamped_transcript
from .storage import hash_file, store_transcription
from .stt import transcribe_audio, transcribe_long_audio


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    p = argparse.ArgumentParser(prog="transcribe", description="Audio -> text transcription pipeline")
    p.add_argument("input", help="Path to audio/video file (wav, mp3, m4a, flac, ogg, mp4, ...)")
    p.add_argument("--format", default="json", choices=FORMATS)
    p.add_argument(
        "--model", default=settings["model_size"], help="faster-whisper model size (tiny/base/small/medium/large-v3)"
    )
    p.add_argument("--language", default=None, help="ISO language code; omit for auto-detect")
    p.add_argument(
        "--long-threshold-sec",
        type=float,
        default=settings["long_threshold_sec"],
        help="Files longer than this are chunked",
    )
    p.add_argument("--output", "-o", default=None, help="Write result to this path instead of stdout")
    p.add_argument("--store", action="store_true", help="Persist audio + transcript to the local store")
    p.add_argument("--probe-only", action="store_true", help="Print format probe and exit")
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    normalised_path = None
    try:
        probe = probe_audio_format(args.input)
        if args.probe_only:
            import json

            print(json.dumps(probe, indent=2))
            return 0
        for w in probe["warnings"]:
            _log(f"[warn] {w}")
        if probe["is_video"]:
            _log("[info] Video container detected; extracting audio track")

        loaded = load_audio(args.input)
        normalised_path = loaded["normalised_path"]
        _log(
            f"[info] Normalised {loaded['duration_sec']}s of audio "
            f"({loaded['original_format']}, {loaded['original_sample_rate']} Hz, "
            f"{loaded['original_channels']}ch)"
        )

        if loaded["duration_sec"] > args.long_threshold_sec:
            result = transcribe_long_audio(
                normalised_path,
                chunk_max_sec=args.long_threshold_sec,
                model_size=args.model,
                language=args.language,
            )
            _log(f"[info] Transcribed in {result['chunks']} chunks, {len(result['failed_chunks'])} failed")
        else:
            result = transcribe_audio(normalised_path, model_size=args.model, language=args.language)

        if result.get("needs_review"):
            _log("[warn] Low-confidence transcription — flagged for review")

        if args.store:
            stored = store_transcription(args.input, hash_file(args.input), result)
            _log(f"[info] Stored transcript id={stored['transcript_id']} blob={stored['audio_key']}")

        rendered = format_timestamped_transcript(result, fmt=args.format)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(rendered)
            _log(f"[info] Wrote {args.output}")
        else:
            print(rendered)
        return 0

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _log(f"[error] {exc}")
        return 1
    finally:
        if normalised_path and os.path.exists(normalised_path):
            os.remove(normalised_path)


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
