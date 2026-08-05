"""--render-mode vlog: clip planning, delivery markers, cost estimate,
bring-your-own-clips folder contract, reference-image aspect handling,
voice-swap/voiceover treatment, assembly, and CLI guards. All Veo,
ElevenLabs, and ffmpeg calls mocked."""

import json
import sys
from types import SimpleNamespace

import pytest

from exec_producer import engine, vlog
from tests.test_engine import FAKE_CAST

HOST = {"id": "host", "name": "Host", "emoji": "🎙", "tagline": "The Voice"}
CONV = [
    {**HOST, "text": '[waves] "Welcome back to the channel."', "delivery": "on_camera"},
    {**HOST, "text": '"The city looks different at night."', "delivery": "voiceover"},
    {**HOST, "text": "[stares out the window]"},  # action-only — no beat
]
SCENE = {"name": "Night Walk", "prompt": "neon streets after rain", "mood": "quiet"}
VOICES = {"provider": "elevenlabs", "defaults": {"gap_seconds": 0.4},
          "voices": {"host": {"voice_id": "v-host"}}}


def cfg16(**vlog_over):
    return vlog.vlog_config({"vlog": {"price_per_second": 0.5, **vlog_over}})


class FakeBackend:
    def __init__(self, sts_fails=False):
        self.defaults = {}
        self.sts_fails = sts_fails
        self.synth_calls, self.sts_calls = [], []

    def synthesize(self, text, voice):
        self.synth_calls.append(text)
        return b"tts-bytes"

    def convert_speech(self, audio_path, voice):
        self.sts_calls.append(str(audio_path))
        if self.sts_fails:
            raise RuntimeError("sts down")
        return b"sts-bytes"


# ── clip plan ────────────────────────────────────────────

def test_plan_one_beat_per_spoken_line():
    plan = vlog.build_clip_plan(CONV, SCENE, cfg16())
    assert [b["beat"] for b in plan] == [1, 2]
    assert [b["delivery"] for b in plan] == ["on_camera", "voiceover"]
    assert plan[0]["host_in_frame"] is True and plan[1]["host_in_frame"] is False
    assert '"Welcome back to the channel."' in plan[0]["prompt"]
    assert "No one speaks" in plan[1]["prompt"]
    assert "neon streets" in plan[0]["shot"] and "neon streets" in plan[1]["shot"]


def test_plan_defaults_missing_delivery_to_on_camera():
    plan = vlog.build_clip_plan([{**HOST, "text": '"hi"'}], SCENE, cfg16())
    assert plan[0]["delivery"] == "on_camera"


def test_plan_respects_custom_shot_and_host_in_frame():
    conv = [{**HOST, "text": '"hi"', "delivery": "voiceover",
             "shot": "The host walks away from camera", "host_in_frame": True}]
    plan = vlog.build_clip_plan(conv, SCENE, cfg16())
    assert plan[0]["shot"] == "The host walks away from camera"
    assert plan[0]["host_in_frame"] is True


def test_plan_rejects_bad_delivery():
    with pytest.raises(ValueError, match="delivery must be one of"):
        vlog.build_clip_plan([{**HOST, "text": '"hi"', "delivery": "narrated"}],
                             SCENE, cfg16())


def test_vertical_show_gets_center_crop_suffix():
    plan = vlog.build_clip_plan(CONV, SCENE, cfg16(aspect="9:16"))
    assert "9:16 center crop" in plan[0]["prompt"]      # host in frame
    assert "center crop" not in plan[1]["prompt"]        # b-roll, no host


def test_vlog_config_resolves_reference_paths_against_show_dir(monkeypatch, tmp_path):
    # "refs/host.png" in visuals.json means <show dir>/refs/host.png — the
    # documented contract — regardless of the process's cwd.
    from exec_producer import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    cfg = vlog.vlog_config({"characters": {"host": {
        "reference_images": ["refs/host.png", "/abs/other.png"]}}})
    assert cfg["reference_images"]["host"] == [str(tmp_path / "refs" / "host.png"),
                                              "/abs/other.png"]


def test_vlog_config_validates_aspect():
    with pytest.raises(ValueError, match="aspect"):
        vlog.vlog_config({"vlog": {"aspect": "4:3"}})
    cfg = vlog.vlog_config({"vlog": {"aspect": "9:16"}})
    assert (cfg["width"], cfg["height"]) == (1080, 1920)


def test_estimate_cost():
    plan = vlog.build_clip_plan(CONV, SCENE, cfg16())
    assert vlog.estimate_cost(plan, cfg16()) == (16, 8.0)
    no_price = vlog.vlog_config({})
    assert vlog.estimate_cost(plan, no_price) == (16, None)
    assert "ESTIMATED cost ~$8.00" in vlog.describe_plan(plan, cfg16())


def test_describe_plan_totals_mixed_forced_durations():
    # A reference-image beat bills 8s regardless of clip_seconds — the plan
    # header must print the true total, not clips × clip_seconds arithmetic.
    cfg = vlog.vlog_config({"vlog": {"clip_seconds": 5},
                            "characters": {"host": {
                                "reference_images": ["/abs/host.png"]}}})
    plan = vlog.build_clip_plan(CONV, SCENE, cfg)
    assert [b["seconds"] for b in plan] == [8, 5]  # ref-forced vs configured
    desc = vlog.describe_plan(plan, cfg)
    assert "2 clips" in desc and "~13s" in desc and "×" not in desc


def test_clip_prompt_strips_curly_quotes():
    # LLM-written scripts often carry smart quotes — the template's own
    # quoting must not double up around them.
    conv = [{**HOST, "text": "“Welcome back.”", "delivery": "on_camera"}]
    plan = vlog.build_clip_plan(conv, SCENE, cfg16())
    assert '"Welcome back."' in plan[0]["prompt"]
    assert "“" not in plan[0]["prompt"] and "”" not in plan[0]["prompt"]


def test_prompt_pack(tmp_path):
    plan = vlog.build_clip_plan(CONV, SCENE, cfg16())
    vlog.write_prompt_pack(plan, tmp_path / "pack.txt", cfg16())
    text = (tmp_path / "pack.txt").read_text()
    assert "beat_001" in text and "[on_camera]" in text and "[voiceover]" in text
    assert "save as beat_002.mp4" in text
    assert "re-voiced" in text  # the on-camera any-voice note


# ── bring-your-own-clips folder contract ─────────────────

def make_clips(tmp_path, beats=(1, 2), junk=(), empty=()):
    d = tmp_path / "clips"
    d.mkdir(exist_ok=True)
    for n in beats:
        (d / f"beat_{n:03d}.mp4").write_bytes(b"" if n in empty else b"x" * 32)
    for name in junk:
        (d / name).write_bytes(b"x")
    return d


def test_clips_dir_valid(tmp_path):
    plan = vlog.build_clip_plan(CONV, SCENE, cfg16())
    assert vlog.validate_clips_dir(make_clips(tmp_path), plan) == []


def test_clips_dir_lists_every_problem(tmp_path):
    plan = vlog.build_clip_plan(CONV, SCENE, cfg16())
    d = make_clips(tmp_path, beats=(1, 3), junk=("notes.txt",), empty=(1,))
    problems = "\n".join(vlog.validate_clips_dir(d, plan))
    assert "missing clip for beat 2" in problems
    assert "extra clip beat 3" in problems
    assert "unexpected file: notes.txt" in problems
    assert "beat_001.mp4 is empty" in problems
    assert "not a directory" in vlog.validate_clips_dir(tmp_path / "nope", plan)[0]


def test_clips_dir_ignores_os_metadata(tmp_path):
    # .DS_Store / ._AppleDouble files appear in any Finder-touched folder —
    # they must not fail the whole render as "unexpected file".
    plan = vlog.build_clip_plan(CONV, SCENE, cfg16())
    d = make_clips(tmp_path, junk=(".DS_Store", "._beat_001.mp4"))
    assert vlog.validate_clips_dir(d, plan) == []


def test_clips_dir_rejects_symlinks(tmp_path):
    # A symlinked "clip" would splice a file from OUTSIDE the folder into
    # the final video — refuse it.
    plan = vlog.build_clip_plan(CONV, SCENE, cfg16())
    d = make_clips(tmp_path, beats=(1,))
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x" * 32)
    (d / "beat_002.mp4").symlink_to(outside)
    problems = "\n".join(vlog.validate_clips_dir(d, plan))
    assert "beat_002.mp4 is not a regular file" in problems


# ── API route ────────────────────────────────────────────

class FakeVeoClient:
    """Records generate_videos configs; returns a done op with clip bytes."""

    def __init__(self):
        self.configs = []
        client = self
        self.models = SimpleNamespace(generate_videos=self._generate)
        self.operations = SimpleNamespace(get=lambda op: op)
        self.files = SimpleNamespace(download=lambda file: b"clip-bytes")

    def _generate(self, model, prompt, config):
        self.configs.append(config)
        video = SimpleNamespace(video="fake-handle")
        return SimpleNamespace(done=True,
                               response=SimpleNamespace(generated_videos=[video]))


def test_generate_clips_reference_and_aspect_rules(tmp_path, monkeypatch):
    pytest.importorskip("google.genai")
    ref = tmp_path / "host.png"
    ref.write_bytes(b"\x89PNG-fake")
    cfg = vlog.vlog_config({"vlog": {"aspect": "9:16"},
                            "characters": {"host": {"reference_images": [str(ref)]}}})
    plan = vlog.build_clip_plan(CONV, SCENE, cfg)
    client = FakeVeoClient()
    pairs = vlog._generate_clips(plan, cfg, tmp_path, "veo-test", client=client)
    assert len(pairs) == 2 and pairs[0][1].read_bytes() == b"clip-bytes"
    on_cam, voiceover = client.configs
    # host in frame → references attached → forced 16:9 (API rejects 9:16 with refs)
    assert on_cam.reference_images and on_cam.aspect_ratio == "16:9"
    # b-roll beat → no references → native 9:16
    assert not getattr(voiceover, "reference_images", None)
    assert voiceover.aspect_ratio == "9:16"


def test_generate_clips_enforces_max_clips_cap(tmp_path):
    cfg = cfg16(max_clips=1)
    plan = vlog.build_clip_plan(CONV, SCENE, cfg)
    with pytest.raises(RuntimeError, match="max_clips"):
        vlog._generate_clips(plan, cfg, tmp_path, "veo-test", client=object())


def test_generate_clips_failed_beat_recorded_and_skipped(tmp_path, monkeypatch, capsys):
    # A failed Veo beat is receipted and dropped — never retried (respend)
    # and never allowed to crash the run and strand the paid clips.
    def flaky(client, model, beat, cfg, ref_paths):
        if beat["beat"] == 2:
            raise RuntimeError("veo 500")
        return b"clip-bytes"
    monkeypatch.setattr(vlog, "_generate_clip", flaky)
    cfg = cfg16()
    plan = vlog.build_clip_plan(CONV, SCENE, cfg)
    pairs = vlog._generate_clips(plan, cfg, tmp_path, "veo-test", client=object())
    assert [b["beat"] for b, _ in pairs] == [1]
    assert "beat 2 failed" in capsys.readouterr().out
    from exec_producer import telemetry
    veo_records = [r for r in telemetry.get_run_log().records if r.get("caller") == "veo"]
    assert len(veo_records) == 2  # one success receipt + one error receipt
    assert veo_records[1]["error"] == "RuntimeError"


# ── ffmpeg command builders ──────────────────────────────

def test_mix_cmd_ducks_pads_and_crops():
    cmd = " ".join(vlog.mix_cmd("c.mp4", "t.mp3", "o.mp4", cfg16(aspect="9:16"),
                                "libx264", clip_seconds=8.0, tts_seconds=9.0))
    assert "volume=0.2" in cmd and "amix=inputs=2" in cmd
    assert "tpad=stop_mode=clone:stop_duration=1.400" in cmd  # 9.0+0.4-8.0
    assert "crop=1080:1920" in cmd and "-t 9.400" in cmd


def test_mix_cmd_without_ambient_stream():
    cmd = " ".join(vlog.mix_cmd("c.mp4", "t.mp3", "o.mp4", cfg16(), "libx264",
                                clip_seconds=8.0, tts_seconds=3.0, has_ambient=False))
    assert "amix" not in cmd and "anull" in cmd


def test_remux_cmd_reconciles_durations():
    # STS audio slightly longer than the clip: video holds its last frame to
    # cover the drift, nothing is truncated (-shortest must be gone).
    cmd = " ".join(vlog.remux_cmd("c.mp4", "s.mp3", "o.mp4", cfg16(), "libx264",
                                  clip_seconds=8.0, audio_seconds=8.3))
    assert "-map 0:v -map 1:a" in cmd and "-shortest" not in cmd
    assert "tpad=stop_mode=clone:stop_duration=0.300" in cmd
    assert "apad" in cmd and "-t 8.300" in cmd
    assert "crop=1920:1080" in cmd
    assert "-ar 44100 -ac 2" in cmd  # uniform params for the concat demuxer


def test_segment_audio_params_are_uniform():
    # Every segment path and the mix path must carry the same -ar/-ac so the
    # final concat demuxer sees uniform streams even from mixed BYO sources.
    mix = " ".join(vlog.mix_cmd("c.mp4", "t.mp3", "o.mp4", cfg16(), "libx264",
                                clip_seconds=8.0, tts_seconds=3.0))
    assert "-ar 44100 -ac 2" in mix


# ── the stage, bring-your-own-clips end-to-end ───────────

def byo_harness(tmp_path, monkeypatch, backend):
    monkeypatch.chdir(tmp_path)
    cmds = []
    monkeypatch.setattr(vlog, "_run_ffmpeg", lambda cmd: cmds.append(cmd))
    monkeypatch.setattr(vlog, "_clip_duration", lambda p: 8.0)
    monkeypatch.setattr(vlog, "audio_duration", lambda p: 3.0)
    monkeypatch.setattr(vlog, "_has_audio_stream", lambda p: True)
    monkeypatch.setattr(vlog, "load_voices", lambda: VOICES)
    monkeypatch.setattr(vlog, "get_backend", lambda provider, defaults: backend)
    monkeypatch.setattr(vlog, "pick_h264_encoder", lambda configured="": "libx264")
    return cmds


def test_render_vlog_byo_end_to_end(tmp_path, monkeypatch):
    backend = FakeBackend()
    cmds = byo_harness(tmp_path, monkeypatch, backend)
    clips = make_clips(tmp_path)
    out = vlog.render_vlog_video(CONV, SCENE, str(tmp_path / "pfx"),
                                 clips_dir=str(clips),
                                 visuals={"vlog": {"price_per_second": 0.5}})
    assert out.endswith("pfx_video.mp4")
    # on-camera beat went through the voice swap, voiceover through TTS
    assert len(backend.sts_calls) == 1 and backend.synth_calls == [
        '"The city looks different at night."']
    srt = (tmp_path / "pfx_vlog" / "_subs.srt").read_text()
    assert "Welcome back to the channel." in srt and srt.count("-->") == 2
    joined = [" ".join(map(str, c)) for c in cmds]
    assert any("concat" in c for c in joined)      # final assembly
    assert not any("card" in c for c in joined)    # never a name card


def test_render_vlog_sts_failure_keeps_clip_voice(tmp_path, monkeypatch, capsys):
    backend = FakeBackend(sts_fails=True)
    byo_harness(tmp_path, monkeypatch, backend)
    clips = make_clips(tmp_path)
    out = vlog.render_vlog_video(CONV, SCENE, str(tmp_path / "pfx"),
                                 clips_dir=str(clips), visuals={})
    assert out is not None
    assert "keeping the clip's own voice" in capsys.readouterr().out
    # The printed promise must match reality: the beat is genuinely IN the
    # final assembly, not just mentioned on the console.
    concat_list = (tmp_path / "pfx_vlog" / "_concat.txt").read_text()
    assert "seg_001" in concat_list and "seg_002" in concat_list


def test_render_vlog_voiceover_tts_failure_gets_receipt(tmp_path, monkeypatch, capsys):
    # A failed voiceover synth drops the beat AND leaves a JSONL receipt —
    # the same discipline as every other paid call.
    class TTSDown(FakeBackend):
        def synthesize(self, text, voice):
            raise RuntimeError("tts down")
    byo_harness(tmp_path, monkeypatch, TTSDown())
    clips = make_clips(tmp_path)
    out = vlog.render_vlog_video(CONV, SCENE, str(tmp_path / "pfx"),
                                 clips_dir=str(clips), visuals={})
    assert out is not None  # the on-camera beat still renders
    assert "Beat 2 audio treatment failed" in capsys.readouterr().out
    from exec_producer import telemetry
    tts_records = [r for r in telemetry.get_run_log().records
                   if r.get("caller") == "tts:host"]
    assert len(tts_records) == 1 and tts_records[0]["error"] == "RuntimeError"


def test_render_vlog_mismatched_folder_raises_before_any_tts(tmp_path, monkeypatch):
    backend = FakeBackend()
    byo_harness(tmp_path, monkeypatch, backend)
    clips = make_clips(tmp_path, beats=(1,))
    with pytest.raises(RuntimeError, match="clips folder doesn't match"):
        vlog.render_vlog_video(CONV, SCENE, str(tmp_path / "pfx"),
                               clips_dir=str(clips), visuals={})
    assert not backend.synth_calls and not backend.sts_calls


def test_engine_video_stage_routes_vlog_and_degrades(monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(vlog, "render_vlog_video",
                        lambda conversation, scene, prefix, clips_dir="": calls.update(
                            clips_dir=clips_dir) or "out.mp4")
    opts = engine.RunOptions(video=True, render_mode="vlog", clips_dir="/clips")
    track, production = engine.render_video_stage(opts, SCENE, CONV, [], ["host"],
                                                  "pfx", None)
    assert track == "out.mp4" and production is None and calls["clips_dir"] == "/clips"

    def boom(*a, **k):
        raise RuntimeError("veo exploded")
    monkeypatch.setattr(vlog, "render_vlog_video", boom)
    track, _ = engine.render_video_stage(opts, SCENE, CONV, [], ["host"], "pfx", None)
    assert track is None
    assert "Vlog rendering failed" in capsys.readouterr().out


# ── CLI ──────────────────────────────────────────────────

def _cli_main(monkeypatch, argv):
    from exec_producer import cli
    monkeypatch.setattr(sys, "argv", ["exec-producer"] + argv)
    cli.main()


def test_cli_vlog_requires_video(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "x", "--render-mode", "vlog"])
    assert "--video" in capsys.readouterr().out


def test_cli_clips_dir_and_dry_run_require_from_script_vlog(monkeypatch, capsys):
    for argv in (["--scene", "x", "--video", "--render-mode", "vlog", "--clips-dir", "d"],
                 ["--from-script", "f.json", "--video", "--dry-run"],
                 ["--scene", "x", "--dry-run"]):
        with pytest.raises(SystemExit):
            _cli_main(monkeypatch, argv)
        assert "vlog render path" in capsys.readouterr().out


def scene_file(tmp_path):
    payload = {"scene": {"name": "N", "prompt": "p", "mood": "m"},
               "intensity": "low",
               "conversation": [
                   {"id": "alpha", "text": '"on cam line"', "delivery": "on_camera"},
                   {"id": "alpha", "text": '"vo line"', "delivery": "voiceover"}]}
    p = tmp_path / "walk.json"
    p.write_text(json.dumps(payload))
    return p


@pytest.fixture
def cli_world(monkeypatch):
    monkeypatch.setattr(engine, "CAST", FAKE_CAST)
    monkeypatch.setattr(engine, "SHOW_SLUG", "test_show")
    monkeypatch.setattr(engine, "SHOW_NAME", "TEST SHOW")
    monkeypatch.setattr(engine, "MAX_INTENSITY", None)


def test_cli_audio_rejected_with_vlog(monkeypatch, capsys):
    # --audio would render a full TTS track the vlog stage then discards —
    # pure double-billing, so the CLI refuses the combination outright.
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "x", "--video",
                                "--render-mode", "vlog", "--audio"])
    assert "double-bill" in capsys.readouterr().out


def test_cli_dry_run_writes_pack_and_bills_nothing(tmp_path, monkeypatch, capsys,
                                                   cli_world):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vlog, "load_visuals", lambda: {"vlog": {"price_per_second": 0.5}})
    # Billing tripwires: if the dry-run early-exit ever regresses, these fire
    # BEFORE any real Veo/ElevenLabs call could happen on a keyed machine.
    monkeypatch.setattr(vlog, "_generate_clips",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("billed Veo")))
    monkeypatch.setattr(vlog, "get_backend",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("billed ElevenLabs")))
    path = scene_file(tmp_path)
    with pytest.raises(SystemExit) as e:
        _cli_main(monkeypatch, ["--from-script", str(path), "--video",
                                "--render-mode", "vlog", "--dry-run"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "2 clips" in out and "ESTIMATED cost" in out and "Nothing was billed" in out
    assert (tmp_path / "walk_vlog_prompts.txt").is_file()
    assert not (tmp_path / "scripts").exists()  # no outputs, no spend


def test_cli_byo_bad_folder_fails_before_preflight(tmp_path, monkeypatch, capsys,
                                                   cli_world):
    from exec_producer import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vlog, "load_visuals", lambda: {})
    monkeypatch.setattr(cli, "media_preflight_or_die",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("preflight ran")))
    clips = make_clips(tmp_path, beats=(1,))
    with pytest.raises(SystemExit) as e:
        _cli_main(monkeypatch, ["--from-script", str(scene_file(tmp_path)), "--video",
                                "--render-mode", "vlog", "--clips-dir", str(clips)])
    assert e.value.code == 1
    assert "missing clip for beat 2" in capsys.readouterr().out


# ── preflight ────────────────────────────────────────────

def test_preflight_collects_problems(monkeypatch):
    from exec_producer import audio
    monkeypatch.setattr(audio, "preflight", lambda cast_ids: ["audio problem"])
    monkeypatch.setattr(vlog, "pick_h264_encoder", lambda configured="": None)
    monkeypatch.setattr(vlog, "load_visuals", lambda: {})
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    problems = vlog.preflight(["host"], api_route=True)
    assert "audio problem" in problems
    assert any("H.264" in p for p in problems)
    assert any("GEMINI_API_KEY" in p for p in problems)
    # BYO route: no Veo requirements
    byo = vlog.preflight(["host"], api_route=False)
    assert not any("GEMINI" in p for p in byo)


def test_preflight_missing_reference_files(monkeypatch, tmp_path):
    from exec_producer import audio
    monkeypatch.setattr(audio, "preflight", lambda cast_ids: [])
    monkeypatch.setattr(vlog, "pick_h264_encoder", lambda configured="": "libx264")
    monkeypatch.setattr(vlog, "load_visuals", lambda: {
        "characters": {"host": {"reference_images": [str(tmp_path / "ghost.png")]}}})
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    problems = vlog.preflight(["host"], api_route=True)
    assert any("reference image" in p and "ghost.png" in p for p in problems)
