"""Unit tests for the audio stage — TTS backend and ffmpeg fully mocked."""

import sys
from pathlib import Path

import pytest

from exec_producer import audio, telemetry

VOICES = {
    "provider": "elevenlabs",
    "defaults": {"model_id": "test-tts-model", "gap_seconds": 0.4},
    "voices": {"alpha": {"voice_id": "va"}, "beta": {"voice_id": "vb"}},
}

CONVO = [
    {"id": "alpha", "name": "Alpha", "text": "Hello there."},
    {"id": "beta", "name": "Beta", "text": "[stares into the distance, jaw clenched]"},
    {"id": "alpha", "name": "Alpha", "text": "*sips drink* Try me."},
]


class FakeBackend:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = fail_on

    def synthesize(self, text, voice):
        self.calls.append((text, voice["voice_id"]))
        if text in self.fail_on:
            raise RuntimeError("tts boom")
        return b"MP3" + text.encode()


@pytest.fixture
def no_ffmpeg(monkeypatch):
    """Capture stitch calls instead of shelling out to ffmpeg."""
    stitched = {}

    def fake_stitch(line_files, out_path, gap_seconds=0.4, workdir=None):
        stitched["files"] = list(line_files)
        stitched["out"] = out_path
        stitched["gap"] = gap_seconds
        Path(out_path).write_bytes(b"TRACK")
        return out_path

    monkeypatch.setattr(audio, "stitch_lines", fake_stitch)
    return stitched


# ── spoken_text ──────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Hello there.", "Hello there."),
    ("[whole line is an action]", ""),
    ("*leans in* You're kidding me.", "You're kidding me."),
    ("Wait [beat] what?", "Wait what?"),
    ("  ", ""),
])
def test_spoken_text_strips_action_markup(raw, expected):
    assert audio.spoken_text(raw) == expected


# ── render_scene_audio ───────────────────────────────────

def test_render_routes_voices_and_stitches(tmp_path, no_ffmpeg):
    backend = FakeBackend()
    out = audio.render_scene_audio(CONVO, str(tmp_path / "scene"),
                                   voices=VOICES, backend=backend)

    # Beta's fully-bracketed fallback line is not spoken; Alpha's markup is stripped
    assert backend.calls == [("Hello there.", "va"), ("Try me.", "va")]
    line_files = sorted((tmp_path / "scene_audio").glob("line_*.mp3"))
    assert [p.name for p in line_files] == ["line_000_alpha.mp3", "line_002_alpha.mp3"]
    assert line_files[0].read_bytes() == b"MP3Hello there."
    assert out == str(tmp_path / "scene_audio.mp3")
    assert no_ffmpeg["files"] == line_files and no_ffmpeg["gap"] == 0.4

    # One JSONL record per synthesized line, unknown cost by design
    tts_records = [r for r in telemetry.get_run_log().records
                   if r.get("caller", "").startswith("tts:")]
    assert len(tts_records) == 2
    assert tts_records[0]["provider"] == "elevenlabs"
    assert tts_records[0]["model"] == "test-tts-model"
    assert tts_records[0]["characters"] == len("Hello there.")
    assert tts_records[0]["cost_usd"] is None


def test_render_skips_character_without_voice(tmp_path, no_ffmpeg):
    backend = FakeBackend()
    convo = [{"id": "omega", "name": "Omega", "text": "I exist."}] + CONVO[:1]
    audio.render_scene_audio(convo, str(tmp_path / "s"), voices=VOICES, backend=backend)
    assert backend.calls == [("Hello there.", "va")]


def test_render_line_failure_degrades_not_crashes(tmp_path, no_ffmpeg, capsys):
    backend = FakeBackend(fail_on=("Hello there.",))
    out = audio.render_scene_audio(CONVO, str(tmp_path / "scene"),
                                   voices=VOICES, backend=backend)
    # The failed line is skipped; the track still renders from the survivor
    assert out == str(tmp_path / "scene_audio.mp3")
    assert len(no_ffmpeg["files"]) == 1
    assert "TTS failed" in capsys.readouterr().out
    errors = [r for r in telemetry.get_run_log().records if r.get("error")]
    assert errors and errors[0]["error"] == "RuntimeError"
    assert telemetry.get_run_log().failures["elevenlabs/test-tts-model"] == 1


def test_render_nothing_spoken_returns_none(tmp_path, no_ffmpeg):
    convo = [{"id": "alpha", "name": "Alpha", "text": "[glares silently]"}]
    out = audio.render_scene_audio(convo, str(tmp_path / "s"),
                                   voices=VOICES, backend=FakeBackend())
    assert out is None
    assert "files" not in no_ffmpeg


# ── ffmpeg assembly ──────────────────────────────────────

def test_silence_cmd_matches_stream_params():
    cmd = audio.silence_cmd("/x/_gap.mp3", 0.25, 44100, 1)
    joined = " ".join(cmd)
    assert cmd[0] == "ffmpeg" and "anullsrc=r=44100:cl=mono" in joined
    assert cmd[cmd.index("-t") + 1] == "0.25"
    stereo = audio.silence_cmd("/x/g.mp3", 0.4, 22050, 2)
    assert stereo[stereo.index("-i") + 1] == "anullsrc=r=22050:cl=stereo"


def test_concat_list_gap_between_not_after(tmp_path):
    a, b, gap = tmp_path / "a.mp3", tmp_path / "b.mp3", tmp_path / "_gap.mp3"
    listing = audio.concat_list([a, b], gap).strip().split("\n")
    assert len(listing) == 3
    assert listing[0].endswith("a.mp3'") and listing[2].endswith("b.mp3'")
    assert "_gap.mp3" in listing[1]
    # single line: no gap at all
    assert audio.concat_list([a], gap).strip().split("\n") == [f"file '{a.resolve()}'"]


def test_stitch_lines_generates_matched_gap_then_concats(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(audio, "probe_stream", lambda p: (44100, 1))
    monkeypatch.setattr(audio, "_run_ffmpeg", lambda cmd: calls.append(cmd))
    f1, f2 = tmp_path / "a.mp3", tmp_path / "b.mp3"
    f1.write_bytes(b"x")
    f2.write_bytes(b"y")

    out = audio.stitch_lines([f1, f2], str(tmp_path / "out.mp3"),
                             gap_seconds=0.25, workdir=tmp_path)

    assert out == str(tmp_path / "out.mp3")
    assert len(calls) == 2
    assert "anullsrc=r=44100:cl=mono" in " ".join(calls[0])   # gap matches probe
    assert calls[1][calls[1].index("-f") + 1] == "concat"
    listing = (tmp_path / "_concat.txt").read_text().strip().split("\n")
    assert len(listing) == 3 and "_gap.mp3" in listing[1]


# ── preflight ────────────────────────────────────────────

@pytest.fixture
def audio_ready(monkeypatch):
    """Baseline where every preflight check passes."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setattr(audio.shutil, "which", lambda t: f"/usr/bin/{t}")
    monkeypatch.setattr(audio.importlib.util, "find_spec", lambda n: object())
    monkeypatch.setattr(audio, "load_voices", lambda: VOICES)


def test_preflight_all_good(audio_ready):
    assert audio.preflight(["alpha", "beta"]) == []


def test_preflight_collects_all_problems(audio_ready, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    monkeypatch.setattr(audio.shutil, "which", lambda t: None)
    problems = audio.preflight(["alpha"])
    assert len(problems) == 3  # key + ffmpeg + ffprobe, reported together
    assert any("ELEVENLABS_API_KEY" in p for p in problems)
    assert any("ffprobe" in p for p in problems)


def test_preflight_missing_sdk(audio_ready, monkeypatch):
    monkeypatch.setattr(audio.importlib.util, "find_spec", lambda n: None)
    assert any("uv sync --extra audio" in p for p in audio.preflight(["alpha"]))


def test_preflight_missing_voices_file(audio_ready, monkeypatch):
    def boom():
        raise FileNotFoundError("voices.json")
    monkeypatch.setattr(audio, "load_voices", boom)
    assert any("voices.json not found" in p for p in audio.preflight(["alpha"]))


def test_preflight_uncovered_cast(audio_ready):
    problems = audio.preflight(["alpha", "gamma", "delta"])
    assert len(problems) == 1
    assert "gamma, delta" in problems[0]


def test_preflight_rejects_unknown_provider(audio_ready, monkeypatch):
    monkeypatch.setattr(audio, "load_voices",
                        lambda: {**VOICES, "provider": "acme-tts"})
    problems = audio.preflight(["alpha"])
    assert any("acme-tts" in p and "only 'elevenlabs'" in p for p in problems)


def test_concat_entry_escapes_single_quotes():
    line = audio.concat_entry("/Users/o'brien/scene_audio/line_000.mp3")
    # ffmpeg concat spec: a quoted ' is written as '\''
    assert line == "file '/Users/o'\\''brien/scene_audio/line_000.mp3'\n"


# ── CLI flag guards ──────────────────────────────────────

def _cli_main(monkeypatch, argv):
    from exec_producer import cli
    monkeypatch.setattr(sys, "argv", ["exec-producer"] + argv)
    cli.main()


def test_cli_audio_rejects_no_save(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "x", "--audio", "--no-save"])
    assert "--no-save" in capsys.readouterr().out


def test_cli_audio_rejects_director_only(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "x", "--audio", "--director-only"])
    assert "redaction" in capsys.readouterr().out
