"""
Stage 2: speech-to-text.

``transcribe_audio`` handles a single file; ``transcribe_long_audio`` splits on
silence, fans out to ``transcribe_audio`` in parallel, and stitches the result
back into one timeline with the same output shape.

faster-whisper is imported lazily and models are cached per process so a
warm worker pays the load cost once, and tests can swap ``get_model`` for a
stub without installing the model runtime.
"""

from __future__ import annotations

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from pydub import AudioSegment
from pydub.silence import detect_silence

from .config import get_settings

_MODEL_CACHE: dict[tuple[str, str], object] = {}
_MODEL_LOCK = threading.Lock()


def get_model(model_size: str, device: str = "auto"):
    """
    Return a cached faster-whisper model for (size, device). Loading is ~2 s and
    hundreds of MB, so it happens once per worker process, not once per request.
    """
    key = (model_size, device)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - environment specific
                raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper") from exc
            compute_type = "int8" if device == "cpu" else "auto"
            try:
                _MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
            except Exception as exc:
                raise RuntimeError(f"Failed to load STT model '{model_size}': {exc}") from exc
        return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    """Drop cached models (frees memory; mainly for tests and graceful shutdown)."""
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


def transcribe_audio(
    wav_path: str,
    model_size: str | None = None,
    language: str | None = None,
    device: str | None = None,
) -> dict:
    """
    Transcribe a 16 kHz mono WAV (output of load_audio) into text.

    Returns the full transcript plus timestamped segments, detected language,
    and confidence signals so downstream stages can decide how much to trust it.
    """
    settings = get_settings()
    model_size = model_size or settings["model_size"]
    device = device or settings["device"]

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    model = get_model(model_size, device)

    segments_iter, info = model.transcribe(
        wav_path,
        language=language,  # None → auto-detect
        beam_size=5,
        vad_filter=True,  # stops hallucination over silence
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=False,
        condition_on_previous_text=False,  # prevents repetition loops on long audio
    )

    # Materialise the lazy iterator here so errors surface in this function.
    segments = []
    for seg in segments_iter:
        segments.append(
            {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
            }
        )

    base = {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_sec": round(info.duration, 3),
        "model": model_size,
    }

    if not segments:
        return {
            **base,
            "text": "",
            "segments": [],
            "mean_logprob": None,
            "needs_review": True,
            "warning": "No speech detected",
        }

    mean_logprob = sum(s["avg_logprob"] for s in segments) / len(segments)
    return {
        **base,
        "text": " ".join(s["text"] for s in segments if s["text"]),
        "segments": segments,
        "mean_logprob": round(mean_logprob, 4),
        "needs_review": mean_logprob < -0.8 or info.language_probability < 0.7,
    }


def plan_chunks(
    audio: AudioSegment,
    chunk_target_sec: float,
    chunk_max_sec: float,
    silence_thresh_db: int = -40,
    min_silence_ms: int = 700,
) -> list[tuple[int, int]]:
    """
    Decide chunk boundaries (in ms) for a long recording: advance to roughly the
    target length, snap to the nearest silence, and hard-cut at the maximum if
    no silence exists in range (music beds, continuous speech).
    """
    total_ms = len(audio)
    target_ms, max_ms = int(chunk_target_sec * 1000), int(chunk_max_sec * 1000)

    silences = detect_silence(audio, min_silence_len=min_silence_ms, silence_thresh=silence_thresh_db)
    cut_candidates = [(s + e) // 2 for s, e in silences]

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < total_ms:
        if total_ms - start <= max_ms:
            bounds.append((start, total_ms))
            break
        window = [c for c in cut_candidates if start + target_ms * 0.5 <= c <= start + max_ms]
        end = min(window, key=lambda c: abs(c - (start + target_ms))) if window else start + max_ms
        if end <= start:  # defensive: never produce an empty chunk
            end = min(start + max_ms, total_ms)
        bounds.append((start, end))
        start = end
    return bounds


def transcribe_long_audio(
    wav_path: str,
    chunk_target_sec: float | None = None,
    chunk_max_sec: float | None = None,
    overlap_sec: float | None = None,
    silence_thresh_db: int = -40,
    min_silence_ms: int = 700,
    max_workers: int | None = None,
    model_size: str | None = None,
    language: str | None = None,
) -> dict:
    """
    Transcribe long audio by splitting on natural pauses, transcribing chunks
    in parallel via transcribe_audio, and stitching results back into a single
    timeline. Short files (<= chunk_max_sec) bypass chunking entirely.
    """
    settings = get_settings()
    chunk_target_sec = chunk_target_sec or settings["chunk_target_sec"]
    chunk_max_sec = chunk_max_sec or settings["long_threshold_sec"]
    overlap_sec = settings["chunk_overlap_sec"] if overlap_sec is None else overlap_sec
    max_workers = max_workers or settings["chunk_workers"]
    model_size = model_size or settings["model_size"]

    audio = AudioSegment.from_file(wav_path)
    total_ms = len(audio)

    if total_ms <= chunk_max_sec * 1000:
        result = transcribe_audio(wav_path, model_size=model_size, language=language)
        return {**result, "chunks": 1, "failed_chunks": []}

    bounds = plan_chunks(audio, chunk_target_sec, chunk_max_sec, silence_thresh_db, min_silence_ms)

    overlap_ms = int(overlap_sec * 1000)
    chunk_files = []
    for i, (s, e) in enumerate(bounds):
        e_ov = min(e + overlap_ms, total_ms)
        fd, path = tempfile.mkstemp(suffix=f"_chunk{i:03d}.wav", prefix="stt_")
        os.close(fd)
        audio[s:e_ov].export(path, format="wav")
        chunk_files.append((i, s / 1000.0, e / 1000.0, path))

    def _run(item):
        idx, offset_sec, hard_end_sec, path = item
        try:
            result = transcribe_audio(path, model_size=model_size, language=language)
            return idx, offset_sec, hard_end_sec, result, None
        except Exception as exc:
            return idx, offset_sec, hard_end_sec, None, f"{type(exc).__name__}: {exc}"
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        outcomes = sorted(pool.map(_run, chunk_files), key=lambda o: o[0])

    segments, failed, logprobs, languages = [], [], [], {}
    last_idx = len(outcomes) - 1
    for idx, offset, hard_end, result, err in outcomes:
        if err:
            failed.append({"chunk": idx, "start": offset, "end": hard_end, "error": err})
            continue
        if result.get("language"):
            languages[result["language"]] = languages.get(result["language"], 0) + 1
        if result.get("mean_logprob") is not None:
            logprobs.append(result["mean_logprob"])
        for seg in result["segments"]:
            abs_start = seg["start"] + offset
            if abs_start >= hard_end and idx < last_idx:
                continue  # overlap region — owned by the next chunk
            segments.append({**seg, "start": round(abs_start, 3), "end": round(seg["end"] + offset, 3)})

    for i, seg in enumerate(segments):
        seg["id"] = i

    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else None
    return {
        "text": " ".join(s["text"] for s in segments if s["text"]),
        "segments": segments,
        "language": max(languages, key=languages.get) if languages else None,
        "language_probability": None,
        "duration_sec": round(total_ms / 1000.0, 3),
        "model": model_size,
        "mean_logprob": round(mean_logprob, 4) if mean_logprob is not None else None,
        "chunks": len(bounds),
        "failed_chunks": failed,
        "needs_review": bool(failed) or (mean_logprob is not None and mean_logprob < -0.8),
    }
