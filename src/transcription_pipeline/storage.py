"""
Persistence: content-addressed blob storage for audio, relational store for
transcripts and segments.

SQLite stands in for Postgres and a local folder stands in for S3. The shape
(immutable blob keyed by hash, job header row, segment rows, UNIQUE on
(content_hash, model)) is what carries over to production unchanged.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id            INTEGER PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    audio_key     TEXT NOT NULL,
    model         TEXT NOT NULL,
    language      TEXT,
    duration_sec  REAL,
    mean_logprob  REAL,
    needs_review  INTEGER NOT NULL DEFAULT 0,
    full_text     TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (content_hash, model)
);
CREATE TABLE IF NOT EXISTS segments (
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    seq           INTEGER NOT NULL,
    start_sec     REAL NOT NULL,
    end_sec       REAL NOT NULL,
    text          TEXT NOT NULL,
    avg_logprob   REAL,
    PRIMARY KEY (transcript_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_segments_time ON segments(transcript_id, start_sec);
CREATE INDEX IF NOT EXISTS idx_transcripts_hash ON transcripts(content_hash);
"""


def hash_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """Open (and initialise) the transcript database."""
    db_path = db_path or get_settings()["db_path"]
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def blob_key_for(content_hash: str, ext: str, blob_dir: str | None = None) -> str:
    """Content-addressed path: blobs/audio/<hh>/<hash><ext>."""
    blob_dir = blob_dir or get_settings()["blob_dir"]
    ext = ext if ext.startswith(".") else f".{ext}"
    return os.path.join(blob_dir, "audio", content_hash[:2], f"{content_hash}{ext}")


def store_transcription(
    audio_path: str,
    content_hash: str,
    transcription: dict,
    db_path: str | None = None,
    blob_dir: str | None = None,
) -> dict:
    """
    Persist audio to content-addressed blob storage and the transcript to the
    relational store. Idempotent on (content_hash, model): re-running replaces
    the transcript rather than duplicating it.
    """
    ext = os.path.splitext(audio_path)[1].lower() or ".bin"
    key = blob_key_for(content_hash, ext, blob_dir)
    os.makedirs(os.path.dirname(key), exist_ok=True)
    if not os.path.exists(key):
        shutil.copyfile(audio_path, key)

    segments = transcription.get("segments", [])
    conn = connect(db_path)
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO transcripts
                    (content_hash, audio_key, model, language, duration_sec, mean_logprob, needs_review, full_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_hash, model) DO UPDATE SET
                    audio_key=excluded.audio_key,
                    language=excluded.language,
                    duration_sec=excluded.duration_sec,
                    mean_logprob=excluded.mean_logprob,
                    needs_review=excluded.needs_review,
                    full_text=excluded.full_text
                RETURNING id
                """,
                (
                    content_hash,
                    key,
                    transcription["model"],
                    transcription.get("language"),
                    transcription.get("duration_sec"),
                    transcription.get("mean_logprob"),
                    int(bool(transcription.get("needs_review"))),
                    transcription.get("text", ""),
                ),
            )
            transcript_id = cur.fetchone()[0]
            conn.execute("DELETE FROM segments WHERE transcript_id = ?", (transcript_id,))
            conn.executemany(
                "INSERT INTO segments (transcript_id, seq, start_sec, end_sec, text, avg_logprob) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (transcript_id, i, s["start"], s["end"], s["text"], s.get("avg_logprob"))
                    for i, s in enumerate(segments)
                ],
            )
    finally:
        conn.close()

    return {"transcript_id": transcript_id, "audio_key": key, "segments": len(segments)}


def load_transcription(content_hash: str, model: str | None = None, db_path: str | None = None) -> dict | None:
    """
    Rebuild a transcription dict (same shape transcribe_audio returns) from the
    store, so it can be rendered or retried without re-running inference.
    """
    conn = connect(db_path)
    try:
        if model:
            row = conn.execute(
                "SELECT * FROM transcripts WHERE content_hash = ? AND model = ?", (content_hash, model)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM transcripts WHERE content_hash = ? ORDER BY created_at DESC LIMIT 1",
                (content_hash,),
            ).fetchone()
        if not row:
            return None
        segs = conn.execute(
            "SELECT seq, start_sec, end_sec, text, avg_logprob FROM segments WHERE transcript_id = ? ORDER BY seq",
            (row["id"],),
        ).fetchall()
    finally:
        conn.close()

    return {
        "transcript_id": row["id"],
        "content_hash": row["content_hash"],
        "audio_key": row["audio_key"],
        "model": row["model"],
        "language": row["language"],
        "duration_sec": row["duration_sec"],
        "mean_logprob": row["mean_logprob"],
        "needs_review": bool(row["needs_review"]),
        "text": row["full_text"] or "",
        "created_at": row["created_at"],
        "segments": [
            {
                "id": s["seq"],
                "start": s["start_sec"],
                "end": s["end_sec"],
                "text": s["text"],
                "avg_logprob": s["avg_logprob"],
            }
            for s in segs
        ],
    }


def query_segments(
    content_hash: str,
    start: float = 0.0,
    end: float | None = None,
    model: str | None = None,
    db_path: str | None = None,
) -> list[dict]:
    """Time-range query over stored segments (for players / search UIs)."""
    conn = connect(db_path)
    try:
        params: list = [content_hash]
        sql = "SELECT t.id FROM transcripts t WHERE t.content_hash = ?"
        if model:
            sql += " AND t.model = ?"
            params.append(model)
        sql += " ORDER BY t.created_at DESC LIMIT 1"
        row = conn.execute(sql, params).fetchone()
        if not row:
            return []
        q = "SELECT seq, start_sec, end_sec, text, avg_logprob FROM segments WHERE transcript_id = ? AND end_sec >= ?"
        p: list = [row["id"], start]
        if end is not None:
            q += " AND start_sec <= ?"
            p.append(end)
        q += " ORDER BY seq"
        rows = conn.execute(q, p).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": r["seq"],
            "start": r["start_sec"],
            "end": r["end_sec"],
            "text": r["text"],
            "avg_logprob": r["avg_logprob"],
        }
        for r in rows
    ]


def delete_transcription(content_hash: str, db_path: str | None = None, blob_dir: str | None = None) -> dict:
    """
    Delete audio blob, transcript rows, and segments together (data-deletion
    requests must remove everything, not just the index).
    """
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT id, audio_key FROM transcripts WHERE content_hash = ?", (content_hash,)).fetchall()
        with conn:
            conn.execute("DELETE FROM transcripts WHERE content_hash = ?", (content_hash,))
    finally:
        conn.close()

    removed_blobs = 0
    for r in {row["audio_key"] for row in rows}:
        if os.path.exists(r):
            os.remove(r)
            removed_blobs += 1
    return {"transcripts_deleted": len(rows), "blobs_deleted": removed_blobs}
