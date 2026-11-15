"""
Shared fixtures.

The STT model is replaced with a deterministic fake so the suite runs without
downloading Whisper weights or needing a GPU. Audio fixtures are synthesised
with pydub (tone bursts separated by silence) so chunking has real pauses to
find. ffmpeg/ffprobe are still required for format probing and MP3 tests.
"""

from __future__ import annotations

import os
import shutil
import wave
from types import SimpleNamespace

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from transcription_pipeline import config, jobs, stt

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point every stage at a temp data dir and reset in-memory state."""
    monkeypatch.setenv("TP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TP_UPLOAD_DIR", str(tmp_path / "data" / "uploads"))
    monkeypatch.setenv("TP_BLOB_DIR", str(tmp_path / "data" / "blobs"))
    monkeypatch.setenv("TP_DB_PATH", str(tmp_path / "data" / "transcripts.db"))
    monkeypatch.setenv("TP_MODEL_SIZE", "small")
    monkeypatch.setenv("TP_DEVICE", "cpu")
    monkeypatch.setenv("TP_RETRY_BASE_DELAY_SEC", "0")
    monkeypatch.setenv("TP_RETRY_MAX_DELAY_SEC", "0")
    config.reset_settings_cache()
    jobs.reset_job_state()
    stt.clear_model_cache()
    yield
    jobs.stop_workers(timeout=1.0)
    config.reset_settings_cache()


def synth_audio(
    duration_sec: float, sample_rate: int = 16000, channels: int = 1, burst_sec: float = 3.0, gap_sec: float = 1.0
) -> AudioSegment:
    """Tone bursts separated by silence; total length == duration_sec."""
    total_ms = int(duration_sec * 1000)
    burst_ms, gap_ms = int(burst_sec * 1000), int(gap_sec * 1000)
    out = AudioSegment.silent(duration=0, frame_rate=sample_rate)
    tone = Sine(440, sample_rate=sample_rate).to_audio_segment(duration=burst_ms).apply_gain(-6)
    gap = AudioSegment.silent(duration=gap_ms, frame_rate=sample_rate)
    while len(out) < total_ms:
        out += tone + gap
    out = out[:total_ms].set_channels(channels).set_sample_width(2)
    return out


@pytest.fixture
def make_wav(tmp_path):
    def _make(
        name: str = "clip.wav", duration_sec: float = 5.0, sample_rate: int = 16000, channels: int = 1, **kw
    ) -> str:
        path = str(tmp_path / name)
        synth_audio(duration_sec, sample_rate, channels, **kw).export(path, format="wav")
        return path

    return _make


@pytest.fixture
def make_mp3(tmp_path):
    def _make(name: str = "clip.mp3", duration_sec: float = 5.0, sample_rate: int = 44100, channels: int = 2) -> str:
        path = str(tmp_path / name)
        synth_audio(duration_sec, sample_rate, channels).export(path, format="mp3", bitrate="64k")
        return path

    return _make


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def fake_model(
    segment_sec: float = 2.0,
    logprob: float = -0.3,
    language: str = "en",
    lang_prob: float = 0.98,
    fail_on: set[str] | None = None,
):
    """
    Build a stand-in for faster_whisper.WhisperModel that emits one segment every
    ``segment_sec`` with deterministic text, so stitching/offsets are checkable.
    ``fail_on`` is a set of substrings; a wav path containing one raises RuntimeError.
    """

    def transcribe(wav_path, **kwargs):
        if fail_on and any(token in os.path.basename(wav_path) for token in fail_on):
            raise RuntimeError(f"simulated failure for {os.path.basename(wav_path)}")
        duration = wav_duration(wav_path)
        segs = []
        i, t = 0, 0.0
        while t < duration - 1e-6:
            end = min(t + segment_sec, duration)
            segs.append(
                SimpleNamespace(
                    id=i,
                    start=t,
                    end=end,
                    text=f" seg{i} ",
                    avg_logprob=logprob,
                    no_speech_prob=0.05,
                )
            )
            i, t = i + 1, end
        info = SimpleNamespace(language=language, language_probability=lang_prob, duration=duration)
        return iter(segs), info

    return SimpleNamespace(transcribe=transcribe)


@pytest.fixture
def patch_stt(monkeypatch):
    """Install a fake model; returns a function to customise it per test."""

    def _install(**kw):
        model = fake_model(**kw)
        monkeypatch.setattr(stt, "get_model", lambda size, device="auto": model)
        return model

    _install()
    return _install
