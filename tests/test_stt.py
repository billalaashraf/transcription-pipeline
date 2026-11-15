import pytest
from pydub import AudioSegment

from transcription_pipeline.stt import plan_chunks, transcribe_audio, transcribe_long_audio

from .conftest import synth_audio


def test_transcribe_audio_returns_segments_and_confidence(make_wav, patch_stt):
    path = make_wav(duration_sec=6.0)
    result = transcribe_audio(path, model_size="small")
    assert result["language"] == "en"
    assert result["model"] == "small"
    assert len(result["segments"]) == 3
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][-1]["end"] == pytest.approx(6.0, abs=0.01)
    assert result["text"] == "seg0 seg1 seg2"
    assert result["needs_review"] is False
    assert result["mean_logprob"] == pytest.approx(-0.3)


def test_transcribe_audio_flags_low_confidence(make_wav, patch_stt):
    patch_stt(logprob=-1.2)
    result = transcribe_audio(make_wav(duration_sec=4.0))
    assert result["needs_review"] is True


def test_transcribe_audio_flags_uncertain_language(make_wav, patch_stt):
    patch_stt(lang_prob=0.4)
    assert transcribe_audio(make_wav(duration_sec=4.0))["needs_review"] is True


def test_transcribe_audio_no_speech(make_wav, monkeypatch):
    from types import SimpleNamespace

    from transcription_pipeline import stt

    def empty(wav_path, **kw):
        return iter([]), SimpleNamespace(language="en", language_probability=0.5, duration=2.0)

    monkeypatch.setattr(stt, "get_model", lambda size, device="auto": SimpleNamespace(transcribe=empty))
    result = transcribe_audio(make_wav(duration_sec=2.0))
    assert result["text"] == ""
    assert result["segments"] == []
    assert result["needs_review"] is True
    assert "warning" in result


def test_transcribe_audio_missing_file(patch_stt):
    with pytest.raises(FileNotFoundError):
        transcribe_audio("/nope.wav")


def test_plan_chunks_snaps_to_silence():
    audio = synth_audio(60.0, burst_sec=3.0, gap_sec=1.0)
    bounds = plan_chunks(audio, chunk_target_sec=20.0, chunk_max_sec=30.0)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(audio)
    for (_s0, e0), (s1, _) in zip(bounds, bounds[1:]):
        assert e0 == s1
    # Every internal cut should land in a silent region (the 1 s gaps),
    # i.e. not inside a 3 s tone burst. Bursts start at multiples of 4000 ms.
    for _, end in bounds[:-1]:
        assert 3000 <= end % 4000 < 4000, f"cut at {end} ms is inside a tone burst"


def test_plan_chunks_hard_cuts_without_silence():
    audio = synth_audio(50.0, burst_sec=50.0, gap_sec=0.0)
    bounds = plan_chunks(audio, chunk_target_sec=10.0, chunk_max_sec=15.0)
    assert bounds[0] == (0, 15000)
    assert bounds[-1][1] == 50000


def test_transcribe_long_audio_short_file_bypasses_chunking(make_wav, patch_stt):
    path = make_wav(duration_sec=10.0)
    result = transcribe_long_audio(path, chunk_max_sec=30.0)
    assert result["chunks"] == 1
    assert result["failed_chunks"] == []
    assert len(result["segments"]) == 5


def test_transcribe_long_audio_stitches_timeline(make_wav, patch_stt):
    path = make_wav(duration_sec=60.0)
    result = transcribe_long_audio(path, chunk_target_sec=20.0, chunk_max_sec=30.0, overlap_sec=2.0, max_workers=2)
    assert result["chunks"] >= 2
    assert result["failed_chunks"] == []
    segs = result["segments"]
    # Monotonic, no gaps, ids re-sequenced, ends at file end.
    assert [s["id"] for s in segs] == list(range(len(segs)))
    for a, b in zip(segs, segs[1:]):
        assert b["start"] >= a["start"]
        assert b["start"] <= a["end"] + 0.01, "gap in stitched timeline"
    assert segs[0]["start"] == 0.0
    assert segs[-1]["end"] == pytest.approx(60.0, abs=0.05)
    # A segment straddling a cut is kept whole, so a little overlap per boundary
    # is allowed, but never more than the configured overlap window.
    covered = sum(s["end"] - s["start"] for s in segs)
    assert 60.0 <= covered + 0.05 <= 60.0 + 2.0 * (result["chunks"] - 1) + 0.05
    assert result["language"] == "en"
    assert result["duration_sec"] == pytest.approx(60.0, abs=0.01)


def test_transcribe_long_audio_isolates_chunk_failure(make_wav, patch_stt):
    patch_stt(fail_on={"chunk001"})
    path = make_wav(duration_sec=60.0)
    result = transcribe_long_audio(path, chunk_target_sec=20.0, chunk_max_sec=30.0)
    assert len(result["failed_chunks"]) == 1
    failed = result["failed_chunks"][0]
    assert failed["chunk"] == 1
    assert "simulated failure" in failed["error"]
    assert result["needs_review"] is True
    # Other chunks still produced text
    assert result["segments"]
    # No segment falls inside the failed range
    for s in result["segments"]:
        assert not (failed["start"] <= s["start"] < failed["end"])


def test_transcribe_long_audio_cleans_temp_chunks(make_wav, patch_stt, tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    path = make_wav(duration_sec=40.0)
    transcribe_long_audio(path, chunk_target_sec=10.0, chunk_max_sec=15.0)
    leftovers = [p for p in tmp_path.iterdir() if "_chunk" in p.name]
    assert leftovers == []


def test_synth_audio_shape():
    a = synth_audio(5.0, sample_rate=16000, channels=1)
    assert isinstance(a, AudioSegment)
    assert len(a) == 5000
    assert a.channels == 1
