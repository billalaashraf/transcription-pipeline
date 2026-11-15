"""
transcription_pipeline — audio in, structured text out.

Stages (all plain functions, no classes):

    audio.probe_audio_format            content-based format detection and routing
    audio.load_audio                    validate + normalise to 16 kHz mono PCM WAV
    stt.transcribe_audio                speech-to-text with timestamps and confidence
    stt.transcribe_long_audio           silence-aware chunking, parallel STT, stitching
    formatting.format_timestamped_transcript   json / srt / vtt / txt rendering
    storage.store_transcription         content-addressed blobs + relational transcript store
    jobs.submit_upload                  streaming upload, dedupe, enqueue with backpressure
    retry.retry_failed_transcription    per-chunk retry with escalation and dead-lettering
    api.create_api                      async job HTTP API (FastAPI)
"""

from .audio import load_audio, probe_audio_format
from .formatting import format_timestamped_transcript
from .jobs import submit_upload
from .retry import retry_failed_transcription
from .storage import store_transcription
from .stt import transcribe_audio, transcribe_long_audio

__all__ = [
    "probe_audio_format",
    "load_audio",
    "transcribe_audio",
    "transcribe_long_audio",
    "format_timestamped_transcript",
    "store_transcription",
    "submit_upload",
    "retry_failed_transcription",
]

__version__ = "0.1.0"
