import json

from transcription_pipeline.cli import run

from .conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


def test_cli_json_to_stdout(make_wav, patch_stt, capsys):
    rc = run([make_wav(duration_sec=4.0)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["text"] == "seg0 seg1"


def test_cli_srt_to_file(make_wav, patch_stt, tmp_path):
    out = tmp_path / "out.srt"
    rc = run([make_wav(duration_sec=4.0), "--format", "srt", "--output", str(out)])
    assert rc == 0
    assert out.read_text().startswith("1\n00:00:00,000")


def test_cli_probe_only(make_wav, capsys):
    rc = run([make_wav(duration_sec=2.0), "--probe-only"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["codec"] == "pcm_s16le"


def test_cli_store_flag(make_wav, patch_stt, capsys):
    from transcription_pipeline.storage import hash_file, load_transcription

    path = make_wav(duration_sec=4.0)
    assert run([path, "--store"]) == 0
    assert load_transcription(hash_file(path))["text"] == "seg0 seg1"


def test_cli_long_file_chunks(make_wav, patch_stt, capsys):
    rc = run([make_wav(duration_sec=40.0), "--long-threshold-sec", "15"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "chunks" in captured.err
    assert json.loads(captured.out)["segments"]


def test_cli_missing_file(capsys):
    assert run(["/no/such.wav"]) == 1
    assert "[error]" in capsys.readouterr().err


def test_cli_bad_file(tmp_path, capsys):
    p = tmp_path / "bad.wav"
    p.write_text("nope")
    assert run([str(p)]) == 1
