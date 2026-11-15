import os
import shutil

import pytest

from transcription_pipeline import config
from transcription_pipeline.audio import load_audio, probe_audio_format

from .conftest import requires_ffmpeg, wav_duration


@requires_ffmpeg
def test_probe_identifies_canonical_wav(make_wav):
    path = make_wav(duration_sec=3.0)
    probe = probe_audio_format(path)
    assert probe["codec"] == "pcm_s16le"
    assert probe["sample_rate"] == 16000
    assert probe["channels"] == 1
    assert probe["needs_transcode"] is False
    assert probe["is_video"] is False
    assert probe["warnings"] == []
    assert abs(probe["duration_sec"] - 3.0) < 0.05


@requires_ffmpeg
def test_probe_detects_real_format_despite_extension(make_mp3, tmp_path):
    mp3 = make_mp3()
    disguised = str(tmp_path / "actually_mp3.wav")
    shutil.copyfile(mp3, disguised)
    probe = probe_audio_format(disguised)
    assert probe["codec"] == "mp3"
    assert probe["needs_transcode"] is True


@requires_ffmpeg
def test_probe_warns_on_low_sample_rate(make_wav):
    path = make_wav(duration_sec=2.0, sample_rate=8000)
    probe = probe_audio_format(path)
    assert any("Low sample rate" in w for w in probe["warnings"])


@requires_ffmpeg
def test_probe_rejects_non_media(tmp_path):
    bogus = tmp_path / "notes.wav"
    bogus.write_text("this is not audio")
    with pytest.raises(ValueError):
        probe_audio_format(str(bogus))


def test_probe_missing_file():
    with pytest.raises(FileNotFoundError):
        probe_audio_format("/nonexistent/file.wav")


def test_load_audio_normalises_stereo_44k(make_wav):
    path = make_wav(duration_sec=2.0, sample_rate=44100, channels=2)
    meta = load_audio(path)
    try:
        assert meta["original_sample_rate"] == 44100
        assert meta["original_channels"] == 2
        assert meta["sample_rate"] == 16000
        assert abs(wav_duration(meta["normalised_path"]) - 2.0) < 0.05
        assert meta["normalised_path"].endswith(".wav")
    finally:
        os.remove(meta["normalised_path"])


@requires_ffmpeg
def test_load_audio_handles_mp3(make_mp3):
    path = make_mp3(duration_sec=2.0)
    meta = load_audio(path)
    try:
        assert meta["original_format"] == "mp3"
        assert abs(meta["duration_sec"] - 2.0) < 0.2  # mp3 encoder padding
    finally:
        os.remove(meta["normalised_path"])


def test_load_audio_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="Unsupported format"):
        load_audio(str(p))


def test_load_audio_rejects_empty_file(tmp_path):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        load_audio(str(p))


def test_load_audio_rejects_oversize(make_wav, monkeypatch):
    path = make_wav(duration_sec=1.0)
    monkeypatch.setenv("TP_MAX_FILE_BYTES", "100")
    config.reset_settings_cache()
    with pytest.raises(ValueError, match="too large"):
        load_audio(path)


def test_load_audio_rejects_too_long(make_wav, monkeypatch):
    path = make_wav(duration_sec=3.0)
    monkeypatch.setenv("TP_MAX_DURATION_SEC", "2")
    config.reset_settings_cache()
    with pytest.raises(ValueError, match="limit is 2s"):
        load_audio(path)


def test_load_audio_rejects_corrupt(tmp_path):
    p = tmp_path / "corrupt.wav"
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    with pytest.raises(ValueError):
        load_audio(str(p))
