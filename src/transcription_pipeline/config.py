"""
Runtime configuration, read from environment variables with sane defaults.

Everything is a plain function so the rest of the code can call
``get_settings()`` without importing a settings object at module load time
(which makes tests able to override via ``monkeypatch.setenv``).
"""

from __future__ import annotations

import os
from functools import lru_cache


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw not in (None, "") else default


@lru_cache(maxsize=1)
def get_settings() -> dict:
    """Return the resolved settings dictionary (cached for the process lifetime)."""
    data_dir = _env_str("TP_DATA_DIR", os.path.join(os.getcwd(), "data"))
    return {
        # Limits
        "max_file_bytes": _env_int("TP_MAX_FILE_BYTES", 200 * 1024 * 1024),
        "max_duration_sec": _env_int("TP_MAX_DURATION_SEC", 3 * 60 * 60),
        # STT
        "model_size": _env_str("TP_MODEL_SIZE", "small"),
        "device": _env_str("TP_DEVICE", "auto"),
        "target_sample_rate": _env_int("TP_TARGET_SR", 16000),
        # Long-file handling
        "long_threshold_sec": _env_float("TP_LONG_THRESHOLD_SEC", 900.0),
        "chunk_target_sec": _env_float("TP_CHUNK_TARGET_SEC", 600.0),
        "chunk_overlap_sec": _env_float("TP_CHUNK_OVERLAP_SEC", 2.0),
        "chunk_workers": _env_int("TP_CHUNK_WORKERS", 2),
        # Jobs / concurrency
        "queue_capacity": _env_int("TP_QUEUE_CAPACITY", 100),
        "worker_count": _env_int("TP_WORKER_COUNT", 1),
        # Retry
        "retry_max_attempts": _env_int("TP_RETRY_MAX_ATTEMPTS", 3),
        "retry_base_delay_sec": _env_float("TP_RETRY_BASE_DELAY_SEC", 30.0),
        "retry_max_delay_sec": _env_float("TP_RETRY_MAX_DELAY_SEC", 600.0),
        # Storage
        "data_dir": data_dir,
        "upload_dir": _env_str("TP_UPLOAD_DIR", os.path.join(data_dir, "uploads")),
        "blob_dir": _env_str("TP_BLOB_DIR", os.path.join(data_dir, "blobs")),
        "db_path": _env_str("TP_DB_PATH", os.path.join(data_dir, "transcripts.db")),
    }


def reset_settings_cache() -> None:
    """Clear the cached settings (used by tests after changing environment variables)."""
    get_settings.cache_clear()
