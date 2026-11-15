"""
Retry and recovery for failed transcriptions.

Classify the failure first, retry at the smallest unit that failed (a chunk,
never the whole file), escalate configuration on each attempt, and give up
deterministically into a dead-letter state with full attempt history.
"""

from __future__ import annotations

import os
import random
import tempfile
import time
from collections.abc import Callable

from pydub import AudioSegment

from .config import get_settings
from .stt import transcribe_audio

TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (MemoryError, TimeoutError, OSError, RuntimeError)
PERMANENT_ERRORS: tuple[type[BaseException], ...] = (ValueError, FileNotFoundError)


def classify_error(exc: BaseException) -> str:
    """'permanent' for input problems (retrying is pointless), 'transient' otherwise."""
    if isinstance(exc, PERMANENT_ERRORS):
        return "permanent"
    return "transient"


def escalation_plan(base_model: str, language: str | None) -> list[dict]:
    """
    Per-attempt configuration: same model → same model with language pinned and
    smaller windows → one size up with language pinned.
    """
    bigger = {"tiny": "base", "base": "small", "small": "medium", "medium": "large-v3"}.get(base_model, base_model)
    return [
        {"model_size": base_model},
        {"model_size": base_model, "language": language},
        {"model_size": bigger, "language": language},
    ]


def backoff_delay(attempt: int, base_delay_sec: float, max_delay_sec: float, jitter: bool = True) -> float:
    """Exponential backoff with jitter so a burst of failures does not retry in lockstep."""
    delay = min(base_delay_sec * (2**attempt), max_delay_sec)
    return delay * random.uniform(0.5, 1.5) if jitter else delay


def retry_failed_transcription(
    job: dict,
    original_audio_path: str,
    run_chunk: Callable[[str, dict], dict] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int | None = None,
) -> dict:
    """
    Recover a transcription that has ``failed_chunks``. Re-exports only the failed
    time ranges from the original audio, retries each with backoff and escalating
    configuration, merges recovered segments into the existing timeline, and
    dead-letters what still fails.

    ``run_chunk(wav_path, config) -> transcription dict`` is injectable so the same
    logic drives a local transcribe_audio call or a remote worker; ``sleep`` is
    injectable so tests do not wait.
    """
    settings = get_settings()
    max_attempts = max_attempts or settings["retry_max_attempts"]
    base_delay, max_delay = settings["retry_base_delay_sec"], settings["retry_max_delay_sec"]

    if run_chunk is None:

        def run_chunk(path: str, cfg: dict) -> dict:
            return transcribe_audio(path, model_size=cfg.get("model_size"), language=cfg.get("language"))

    plan = escalation_plan(job.get("model") or settings["model_size"], job.get("language"))
    audio = AudioSegment.from_file(original_audio_path)
    recovered: list[dict] = []
    dead: list[dict] = []

    for chunk in job.get("failed_chunks", []):
        start_ms, end_ms = int(chunk["start"] * 1000), int(chunk["end"] * 1000)
        fd, path = tempfile.mkstemp(suffix=f"_retry{chunk['chunk']:03d}.wav", prefix="stt_")
        os.close(fd)
        audio[start_ms:end_ms].export(path, format="wav")
        history = list(chunk.get("attempts", []))
        succeeded = False

        try:
            for attempt in range(len(history), max_attempts):
                cfg = plan[min(attempt, len(plan) - 1)]
                try:
                    result = run_chunk(path, cfg)
                    for seg in result.get("segments", []):
                        recovered.append(
                            {
                                **seg,
                                "start": round(seg["start"] + chunk["start"], 3),
                                "end": round(seg["end"] + chunk["start"], 3),
                            }
                        )
                    history.append({"attempt": attempt, "config": cfg, "ok": True})
                    succeeded = True
                    break
                except Exception as exc:
                    kind = classify_error(exc)
                    history.append(
                        {
                            "attempt": attempt,
                            "config": cfg,
                            "ok": False,
                            "error": f"{kind}: {type(exc).__name__}: {exc}",
                        }
                    )
                    if kind == "permanent":
                        break
                    if attempt + 1 < max_attempts:
                        sleep(backoff_delay(attempt, base_delay, max_delay))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        if not succeeded:
            dead.append({**chunk, "attempts": history})

    segments = sorted(list(job.get("segments", [])) + recovered, key=lambda s: s["start"])
    for i, seg in enumerate(segments):
        seg["id"] = i

    logprobs = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
    mean_logprob = round(sum(logprobs) / len(logprobs), 4) if logprobs else job.get("mean_logprob")

    return {
        **job,
        "segments": segments,
        "text": " ".join(s["text"] for s in segments if s.get("text")),
        "mean_logprob": mean_logprob,
        "failed_chunks": dead,
        "recovered_chunks": len(job.get("failed_chunks", [])) - len(dead),
        "status": "dead_letter" if dead else "done",
        "needs_review": bool(dead) or bool(job.get("needs_review", False)),
        "gaps": [{"start": d["start"], "end": d["end"]} for d in dead],
    }
