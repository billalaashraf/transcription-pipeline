"""
Stage 0/1: format detection and normalisation.

All formats go through one decode path (ffmpeg via pydub) and come out as
16 kHz mono 16-bit PCM WAV. Nothing downstream knows that other formats exist.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

from pydub import AudioSegment

from .config import get_settings

SUPPORTED_FORMATS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".webm",
    ".mp4",
    ".mkv",
    ".mov",
}
TELEPHONY_CODECS = {"amr_nb", "amr_wb", "gsm", "g729", "speex"}
LOSSLESS_CODECS = {"pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "flac", "alac"}


def probe_audio_format(file_path: str) -> dict:
    """
    Identify what a file actually is (not what its extension claims) and decide
    how the pipeline should treat it, using ffprobe's container/stream analysis.

    Returns a routing decision: container, codec, whether it needs re-encoding,
    whether it's a video file with an audio track, and any quality warnings.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found on PATH — install ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Not a decodable media file: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("ffprobe timed out — file may be truncated or malformed") from exc

    info = json.loads(out.stdout or "{}")
    streams = info.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video_streams = [s for s in streams if s.get("codec_type") == "video"]

    if not audio_streams:
        raise ValueError("File contains no audio stream")

    # Prefer the stream flagged default, else the first.
    primary = next(
        (s for s in audio_streams if s.get("disposition", {}).get("default") == 1),
        audio_streams[0],
    )

    fmt = info.get("format", {})
    container = fmt.get("format_name", "unknown")
    codec = primary.get("codec_name", "unknown")
    sample_rate = int(primary.get("sample_rate", 0) or 0)
    channels = int(primary.get("channels", 0) or 0)
    duration = float(fmt.get("duration", 0) or primary.get("duration", 0) or 0)
    bit_rate = int(primary.get("bit_rate", 0) or fmt.get("bit_rate", 0) or 0)

    warnings: list[str] = []
    if sample_rate and sample_rate < 16000:
        warnings.append(f"Low sample rate ({sample_rate} Hz) — telephony-grade, expect reduced accuracy")
    if codec in TELEPHONY_CODECS:
        warnings.append(f"Telephony codec '{codec}' — heavily compressed speech")
    if bit_rate and bit_rate < 32000 and codec not in LOSSLESS_CODECS:
        warnings.append(f"Low bitrate ({bit_rate} bps) — lossy artefacts likely")
    if len(audio_streams) > 1:
        warnings.append(f"{len(audio_streams)} audio tracks found; using stream index {primary.get('index')}")
    if channels > 2:
        warnings.append(f"{channels} channels — downmix may bury quiet speakers; consider per-channel transcription")

    target_sr = get_settings()["target_sample_rate"]
    already_canonical = (
        codec == "pcm_s16le" and sample_rate == target_sr and channels == 1 and container.startswith("wav")
    )

    return {
        "container": container,
        "codec": codec,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_sec": round(duration, 3),
        "bit_rate": bit_rate,
        "is_video": bool(video_streams),
        "audio_stream_index": primary.get("index"),
        "needs_transcode": not already_canonical,
        "warnings": warnings,
    }


def load_audio(file_path: str, target_sr: int | None = None, out_dir: str | None = None) -> dict:
    """
    Accept an audio file, validate it, and normalise it to 16 kHz mono PCM WAV
    (the canonical input format for Whisper-family and most STT models).

    Returns metadata the rest of the pipeline needs: the normalised WAV path,
    duration, original format, and sample rate — so downstream stages never
    have to re-inspect the file.
    """
    settings = get_settings()
    target_sr = target_sr or settings["target_sample_rate"]

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_FORMATS)}")

    size = os.path.getsize(file_path)
    if size == 0:
        raise ValueError("Audio file is empty")
    if size > settings["max_file_bytes"]:
        raise ValueError(f"File too large ({size} bytes). Limit is {settings['max_file_bytes']} bytes")

    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as exc:  # pydub raises a mix of CouldntDecodeError / OSError
        raise ValueError(f"Could not decode audio: {exc}") from exc

    duration_sec = len(audio) / 1000.0
    if duration_sec < 0.1:
        raise ValueError("Audio is shorter than 100 ms — nothing to transcribe")
    if duration_sec > settings["max_duration_sec"]:
        raise ValueError(f"Audio is {duration_sec:.0f}s; limit is {settings['max_duration_sec']}s")

    normalised = audio.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="stt_", dir=out_dir)
    os.close(out_fd)
    normalised.export(out_path, format="wav")

    return {
        "source_path": file_path,
        "normalised_path": out_path,
        "original_format": ext.lstrip("."),
        "original_sample_rate": audio.frame_rate,
        "original_channels": audio.channels,
        "sample_rate": target_sr,
        "duration_sec": round(duration_sec, 3),
        "size_bytes": size,
    }
