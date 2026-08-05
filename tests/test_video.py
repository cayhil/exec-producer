"""Unit tests for the video stage — ffmpeg fully mocked; card rendering
tests run against real Pillow (skipped if the video extra isn't installed)."""

import sys

import pytest

from exec_producer import video

CONVO = [
    {"id": "alpha", "name": "Alpha", "tagline": "The A", "text": "Hello there."},
    {"id": "beta", "name": "Beta", "tagline": "The B", "text": "[glares silently]"},
    {"id": "alpha", "name": "Alpha", "tagline": "The A", "text": "*sips* Try me."},
]

VISUALS = {"defaults": dict(video.DEFAULTS),
           "characters": {"alpha": {"color": "#123456"}}}


# ── timing helpers ───────────────────────────────────────

def test_srt_timestamp():
    assert video.srt_timestamp(0) == "00:00:00,000"
    assert video.srt_timestamp(3661.5) == "01:01:01,500"
    assert video.srt_timestamp(1.234) == "00:00:01,234"


def test_build_srt_cues_track_segment_layout():
    srt = video.build_srt([CONVO[0], CONVO[2]], [2.0, 3.0], gap_seconds=0.5)
    cues = srt.strip().split("\n\n")
    assert len(cues) == 2
    assert "00:00:00,000 --> 00:00:02,000" in cues[0]
    assert "Hello there." in cues[0]
    # second cue starts after line 1 + gap; markup stripped from the text
    assert "00:00:02,500 --> 00:00:05,500" in cues[1]
    assert "Try me." in cues[1] and "*" not in cues[1]


def test_character_color_configured_then_palette():
    assert video.character_color("alpha", {"alpha", "beta"}, VISUALS) == "#123456"
    # unconfigured characters get deterministic palette slots by sorted cast order
    assert video.character_color("beta", {"alpha", "beta"}, VISUALS) == video.PALETTE[1]
    assert video.character_color("beta", {"alpha", "beta"}, VISUALS) == \
        video.character_color("beta", {"beta", "alpha"}, VISUALS)


# ── card rendering (real Pillow, skipped without the extra) ──

def test_render_card_writes_image(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    out = tmp_path / "card.png"
    video.render_card(CONVO[0], "Hello there.", "#123456", video.DEFAULTS, out)
    img = Image.open(out)
    assert img.size == (video.DEFAULTS["width"], video.DEFAULTS["height"])


def test_render_card_wraps_long_dialogue(tmp_path):
    pytest.importorskip("PIL")
    long_line = "word " * 300  # must wrap and truncate, not crash or overflow
    out = tmp_path / "card.png"
    video.render_card(CONVO[0], long_line.strip(), "#123456", video.DEFAULTS, out)
    assert out.exists()


def test_load_font_falls_back_when_configured_path_is_bad():
    pytest.importorskip("PIL")
    font = video._load_font("/nonexistent/font.ttf", 40)
    assert font is not None


# ── encoder detection + command construction ─────────────

def _fake_ffmpeg_stdout(monkeypatch, stdout):
    class P:
        def __init__(self):
            self.stdout = stdout
    monkeypatch.setattr(video.subprocess, "run", lambda *a, **k: P())


def test_pick_h264_encoder_prefers_libx264(monkeypatch):
    _fake_ffmpeg_stdout(monkeypatch, " libx264 ... h264_videotoolbox ")
    assert video.pick_h264_encoder() == "libx264"


def test_pick_h264_encoder_falls_back_then_none(monkeypatch):
    _fake_ffmpeg_stdout(monkeypatch, " ... h264_videotoolbox ...")
    assert video.pick_h264_encoder() == "h264_videotoolbox"
    _fake_ffmpeg_stdout(monkeypatch, "nothing useful")
    assert video.pick_h264_encoder() is None
    # configured override skips detection entirely
    assert video.pick_h264_encoder("libopenh264") == "libopenh264"


def test_segment_cmd_shape():
    cmd = video.segment_cmd("/w/card_000.png", "/x/line_000_alpha.mp3",
                            "/w/seg_000.mp4", 2.4, video.DEFAULTS, "libx264")
    joined = " ".join(cmd)
    assert cmd[cmd.index("-loop") + 1] == "1"          # card image looped
    assert cmd[cmd.index("-t") + 1] == "2.400"
    assert "apad" in joined                            # audio padded to segment length
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"


def test_finalize_cmd_concats_without_filters():
    cmd = video.finalize_cmd("/w/_concat.txt", "/out.mp4", "libx264")
    assert cmd[cmd.index("-f") + 1] == "concat"
    assert "-vf" not in cmd                            # minimal builds: no text filters


# ── file collection ──────────────────────────────────────

def test_collect_line_files_pairs_by_embedded_index(tmp_path):
    for name in ("line_000_alpha.mp3", "line_002_alpha.mp3", "_gap.mp3"):
        (tmp_path / name).write_bytes(b"x")
    pairs = video.collect_line_files(tmp_path, CONVO)
    # index 1 had no file (TTS skipped it); the gap helper file is ignored
    assert [(e["id"], p.name) for e, p in pairs] == \
        [("alpha", "line_000_alpha.mp3"), ("alpha", "line_002_alpha.mp3")]


# ── render_scene_video ───────────────────────────────────

@pytest.fixture
def fake_media(monkeypatch):
    """Mock everything that shells out or needs Pillow."""
    state = {"ffmpeg": [], "cards": []}
    monkeypatch.setattr(video, "_run_ffmpeg", lambda cmd: state["ffmpeg"].append(cmd))
    monkeypatch.setattr(video, "audio_duration", lambda p: 2.0)
    monkeypatch.setattr(video, "pick_h264_encoder", lambda configured="": "libx264")
    monkeypatch.setattr(video, "render_card",
                        lambda entry, dialogue, color, cfg, out:
                        state["cards"].append((entry["id"], dialogue, color)) or out)
    return state


def test_render_scene_video_full_flow(tmp_path, fake_media):
    prefix = str(tmp_path / "scene")
    audio_dir = tmp_path / "scene_audio"
    audio_dir.mkdir()
    (audio_dir / "line_000_alpha.mp3").write_bytes(b"x")
    (audio_dir / "line_002_alpha.mp3").write_bytes(b"y")

    out = video.render_scene_video(CONVO, prefix, visuals=VISUALS, gap_seconds=0.5)

    assert out == f"{prefix}_video.mp4"
    workdir = tmp_path / "scene_video"
    srt = (workdir / "_subs.srt").read_text()
    assert "Hello there." in srt and "Try me." in srt
    # one card per surviving line, in the character's configured color
    assert fake_media["cards"] == [("alpha", "Hello there.", "#123456"),
                                   ("alpha", "Try me.", "#123456")]
    # two segment renders + one finalize
    assert len(fake_media["ffmpeg"]) == 3
    seg0, seg1, final = fake_media["ffmpeg"]
    # gap trails every segment except the last
    assert seg0[seg0.index("-t") + 1] == "2.500"
    assert seg1[seg1.index("-t") + 1] == "2.000"
    assert final[final.index("-f") + 1] == "concat"
    listing = (workdir / "_concat.txt").read_text().strip().split("\n")
    assert len(listing) == 2 and "seg_000.mp4" in listing[0]


def test_render_scene_video_requires_audio_stage(tmp_path):
    with pytest.raises(RuntimeError, match="--audio"):
        video.render_scene_video(CONVO, str(tmp_path / "scene"), visuals=VISUALS,
                                 gap_seconds=0.4)


def test_render_scene_video_nothing_to_compose(tmp_path, fake_media):
    (tmp_path / "scene_audio").mkdir()
    out = video.render_scene_video(CONVO, str(tmp_path / "scene"), visuals=VISUALS,
                                   gap_seconds=0.4)
    assert out is None and fake_media["ffmpeg"] == []


def test_render_scene_video_errors_without_encoder(tmp_path, fake_media, monkeypatch):
    monkeypatch.setattr(video, "pick_h264_encoder", lambda configured="": None)
    (tmp_path / "scene_audio").mkdir()
    (tmp_path / "scene_audio" / "line_000_alpha.mp3").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="H.264"):
        video.render_scene_video(CONVO, str(tmp_path / "scene"), visuals=VISUALS,
                                 gap_seconds=0.4)


# ── preflight ────────────────────────────────────────────

def test_preflight_ok(monkeypatch):
    pytest.importorskip("PIL")  # video extra — skip on a bare clone
    monkeypatch.setattr(video, "pick_h264_encoder", lambda configured="": "libx264")
    assert video.preflight() == []


def test_preflight_reports_missing_encoder(monkeypatch):
    pytest.importorskip("PIL")  # video extra — skip on a bare clone
    monkeypatch.setattr(video, "pick_h264_encoder", lambda configured="": None)
    problems = video.preflight()
    assert len(problems) == 1 and "H.264" in problems[0]


# ── CLI guards ───────────────────────────────────────────

def _cli_main(monkeypatch, argv):
    from exec_producer import cli
    monkeypatch.setattr(sys, "argv", ["exec-producer"] + argv)
    cli.main()


def test_cli_video_rejects_no_save(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "x", "--video", "--no-save"])
    assert "--no-save" in capsys.readouterr().out


def test_cli_video_rejects_director_only(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "x", "--video", "--director-only"])
    assert "redaction" in capsys.readouterr().out


def test_cli_video_implies_audio(monkeypatch, tmp_path, capsys):
    from exec_producer import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "SCENES", {"test": {"name": "T", "prompt": "p", "mood": "m"}})
    monkeypatch.setattr(cli, "check_api_keys", lambda ids: (["a", "b"], []))
    monkeypatch.setattr(cli, "resolve_cast", lambda arg: ["a", "b"])
    monkeypatch.setattr(cli, "director_key_or_die", lambda: None)
    preflights = {}
    monkeypatch.setattr(cli, "media_preflight_or_die",
                        lambda ids, audio=False, video=False, render_mode="tts":
                        preflights.update(audio=audio, video=video, render_mode=render_mode))
    captured = {}
    monkeypatch.setattr(cli, "run_scene",
                        lambda scene_id, cast, opts: captured.update(opts=opts))

    _cli_main(monkeypatch, ["--scene", "test", "--video"])

    assert "implies --audio" in capsys.readouterr().out
    assert preflights == {"audio": True, "video": True, "render_mode": "tts"}
    assert captured["opts"].audio is True and captured["opts"].video is True


def test_cli_render_mode_veo_requires_video(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "x", "--render-mode", "veo"])
    assert "--video" in capsys.readouterr().out


def test_cli_veo_mode_does_not_imply_audio(monkeypatch, tmp_path, capsys):
    from exec_producer import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "SCENES", {"test": {"name": "T", "prompt": "p", "mood": "m"}})
    monkeypatch.setattr(cli, "check_api_keys", lambda ids: (["a", "b"], []))
    monkeypatch.setattr(cli, "resolve_cast", lambda arg: ["a", "b"])
    monkeypatch.setattr(cli, "director_key_or_die", lambda: None)
    preflights = {}
    monkeypatch.setattr(cli, "media_preflight_or_die",
                        lambda ids, audio=False, video=False, render_mode="tts":
                        preflights.update(audio=audio, render_mode=render_mode))
    captured = {}
    monkeypatch.setattr(cli, "run_scene",
                        lambda scene_id, cast, opts: captured.update(opts=opts))

    _cli_main(monkeypatch, ["--scene", "test", "--video", "--render-mode", "veo"])

    assert "implies --audio" not in capsys.readouterr().out
    assert preflights == {"audio": False, "render_mode": "veo"}
    opts = captured["opts"]
    assert opts.audio is False and opts.video is True and opts.render_mode == "veo"


# ── Veo render path ──────────────────────────────────────

BEATS = [{"beat_number": 1, "veo3_video_prompt": "beat one prompt"},
         {"beat_number": 2, "veo3_video_prompt": "beat two prompt"},
         {"beat_number": 3, "veo3_video_prompt": "  "}]  # empty → filtered


@pytest.fixture
def fake_veo(monkeypatch):
    state = {"ffmpeg": [], "prompts": [], "fail_on": set()}

    def fake_clip(client, model, prompt, poll_seconds, timeout_seconds):
        state["prompts"].append(prompt)
        if prompt in state["fail_on"]:
            raise RuntimeError("veo boom")
        return b"CLIP:" + prompt.encode()

    monkeypatch.setattr(video, "_generate_veo_clip", fake_clip)
    monkeypatch.setattr(video, "_run_ffmpeg", lambda cmd: state["ffmpeg"].append(cmd))
    monkeypatch.setattr(video, "pick_h264_encoder", lambda configured="": "libx264")
    monkeypatch.setattr(video, "_load_visuals_or_defaults",
                        lambda: {"defaults": dict(video.DEFAULTS), "characters": {}})
    return state


def test_render_veo_video_generates_and_concats(tmp_path, fake_veo):
    from exec_producer import telemetry
    out = video.render_veo_video(BEATS, str(tmp_path / "scene"), client=object())

    assert out == str(tmp_path / "scene") + "_video.mp4"
    assert fake_veo["prompts"] == ["beat one prompt", "beat two prompt"]
    clips = sorted((tmp_path / "scene_veo").glob("beat_*.mp4"))
    assert [p.name for p in clips] == ["beat_000.mp4", "beat_001.mp4"]
    assert clips[0].read_bytes() == b"CLIP:beat one prompt"
    assert len(fake_veo["ffmpeg"]) == 1  # single concat, no per-beat encode
    records = [r for r in telemetry.get_run_log().records if r.get("caller") == "veo"]
    assert len(records) == 2 and records[0]["cost_usd"] is None


def test_render_veo_video_beat_failure_degrades(tmp_path, fake_veo, capsys):
    from exec_producer import telemetry
    fake_veo["fail_on"].add("beat one prompt")
    out = video.render_veo_video(BEATS, str(tmp_path / "scene"), client=object())

    assert out is not None  # surviving beat still assembled
    assert "beat 1 failed" in capsys.readouterr().out
    errors = [r for r in telemetry.get_run_log().records if r.get("error")]
    assert errors and errors[0]["caller"] == "veo"
    assert telemetry.get_run_log().failures[f"gemini/{video.VEO_MODEL}"] == 1


def test_render_veo_video_nothing_survives(tmp_path, fake_veo):
    fake_veo["fail_on"].update({"beat one prompt", "beat two prompt"})
    assert video.render_veo_video(BEATS, str(tmp_path / "s"), client=object()) is None
    assert fake_veo["ffmpeg"] == []
    assert video.render_veo_video([], str(tmp_path / "s2"), client=object()) is None


def test_generate_veo_clip_polls_until_done():
    pytest.importorskip("google.genai")
    from types import SimpleNamespace

    pending = SimpleNamespace(done=False)
    finished = SimpleNamespace(
        done=True,
        response=SimpleNamespace(generated_videos=[SimpleNamespace(video="handle")]))
    client = SimpleNamespace(
        models=SimpleNamespace(generate_videos=lambda **k: pending),
        operations=SimpleNamespace(get=lambda op: finished),
        files=SimpleNamespace(download=lambda file: b"VIDEO"),
    )
    out = video._generate_veo_clip(client, "veo-x", "prompt", poll_seconds=0,
                                   timeout_seconds=60)
    assert out == b"VIDEO"


def test_generate_veo_clip_times_out():
    pytest.importorskip("google.genai")
    from types import SimpleNamespace
    pending = SimpleNamespace(done=False)
    client = SimpleNamespace(
        models=SimpleNamespace(generate_videos=lambda **k: pending),
        operations=SimpleNamespace(get=lambda op: pending),
        files=None,
    )
    with pytest.raises(TimeoutError):
        video._generate_veo_clip(client, "veo-x", "prompt", poll_seconds=0,
                                 timeout_seconds=0)


def test_poll_veo_operation_surfaces_operation_error():
    # A done-but-failed operation must raise the API's own message, not die
    # on op.response.generated_videos[0] — the beat receipt records the
    # exception name, and "RuntimeError" beats "AttributeError".
    from types import SimpleNamespace
    failed = SimpleNamespace(done=True, error={"code": 8, "message": "quota exhausted"},
                             response=None)
    with pytest.raises(RuntimeError, match="quota exhausted"):
        video.poll_veo_operation(object(), failed, poll_seconds=0, timeout_seconds=60)


def test_poll_veo_operation_rejects_empty_result():
    # Success shell, no videos (safety-filtered): a legible error, no crash.
    from types import SimpleNamespace
    empty = SimpleNamespace(done=True, error=None,
                            response=SimpleNamespace(generated_videos=[]))
    with pytest.raises(RuntimeError, match="no video"):
        video.poll_veo_operation(object(), empty, poll_seconds=0, timeout_seconds=60)


def test_veo_preflight(monkeypatch):
    pytest.importorskip("google.genai")  # veo extra — skip on a bare clone
    monkeypatch.setattr(video, "pick_h264_encoder", lambda configured="": "libx264")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert video.veo_preflight() == []
    monkeypatch.delenv("GEMINI_API_KEY")
    assert any("GEMINI_API_KEY" in p for p in video.veo_preflight())
