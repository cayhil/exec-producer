"""Unit tests for the guardrail + audit pieces: content-rating gate, config
manifest, and the run-log audit header."""

import json
from types import SimpleNamespace

import pytest

from exec_producer import config, engine, telemetry
from tests.test_engine import FAKE_CAST, SCENE


@pytest.fixture
def fake_cast(monkeypatch):
    """Only the loop tests need a stand-in cast — the rest of this module
    deliberately asserts against the really-loaded show."""
    monkeypatch.setattr(engine, "CAST", FAKE_CAST)
    monkeypatch.setattr(engine, "SHOW_NAME", "TEST SHOW")
    monkeypatch.setattr(engine, "SHOW_SLUG", "test_show")
    monkeypatch.setattr(engine, "MAX_INTENSITY", None)
    # The engine paces itself between turns; the loop tests don't need it,
    # and without this they are the two slowest tests in the suite.
    monkeypatch.setattr(engine.time, "sleep", lambda _s: None)


# ── content-rating gate ──────────────────────────────────

def test_no_cap_passes_through(monkeypatch):
    monkeypatch.setattr(engine, "MAX_INTENSITY", None)
    assert engine.effective_intensity("maximum") == "maximum"


def test_cap_lowers_intensity(monkeypatch, capsys):
    monkeypatch.setattr(engine, "MAX_INTENSITY", "low")
    assert engine.effective_intensity("high") == "low"
    assert "Content rating cap" in capsys.readouterr().out


def test_cap_does_not_raise_intensity(monkeypatch, capsys):
    monkeypatch.setattr(engine, "MAX_INTENSITY", "high")
    assert engine.effective_intensity("low") == "low"
    assert "Content rating cap" not in capsys.readouterr().out


def test_unknown_values_pass_through(monkeypatch):
    monkeypatch.setattr(engine, "MAX_INTENSITY", "low")
    assert engine.effective_intensity("weird") == "weird"
    monkeypatch.setattr(engine, "MAX_INTENSITY", "bogus")
    assert engine.effective_intensity("maximum") == "maximum"


# ── spend cap: config wiring ─────────────────────────────
# The value has to survive the trip from world.json to the gate. Every other
# cap test monkeypatches telemetry.MAX_RUN_USD, which would keep passing if
# the key were renamed or the import dropped — and a cap that silently stops
# existing is unbounded spend.

def test_the_budget_key_reaches_the_gate_from_world_json():
    assert config.read_max_run_usd({"budget": {"max_run_usd": 2.5}}) == 2.5


def test_an_absent_budget_block_means_no_cap():
    assert config.read_max_run_usd({}) is None
    assert config.read_max_run_usd({"budget": {}}) is None


def test_a_zero_cap_means_spend_nothing_not_spend_freely():
    # `if not MAX_RUN_USD` would read 0 as "unconfigured" — the exact
    # inversion of what someone typing 0 wants.
    assert config.read_max_run_usd({"budget": {"max_run_usd": 0}}) == 0


def test_a_cap_of_zero_stops_the_run_immediately(monkeypatch):
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 0)
    assert telemetry.spend_exhausted() is True


def test_an_unusable_cap_is_rejected_at_load_not_mid_run():
    # A string cap used to pass the truthiness check and then raise TypeError
    # at the first turn boundary — after the run had already billed.
    for bad in ("2.50", [2.5], {"usd": 2.5}, True, -1):
        problems = config.validate_budget({"budget": {"max_run_usd": bad}})
        assert problems, f"{bad!r} should be rejected"
        assert "max_run_usd" in problems[0]


def test_a_budget_block_of_the_wrong_shape_is_rejected():
    assert config.validate_budget({"budget": 2.5})


def test_a_valid_budget_block_reports_no_problems():
    assert config.validate_budget({"budget": {"max_run_usd": 2.5}}) == []
    assert config.validate_budget({}) == []


# ── spend cap ────────────────────────────────────────────

def spend(amount):
    telemetry.get_run_log().record(caller="director", model="m", cost_usd=amount)


def test_total_cost_sums_recorded_calls():
    spend(0.01)
    spend(0.02)
    assert telemetry.get_run_log().total_cost_usd() == pytest.approx(0.03)


def test_calls_with_unknown_cost_do_not_break_the_total():
    # TTS and external-render calls record cost_usd=None by design.
    spend(0.01)
    telemetry.get_run_log().record(caller="tts:alpha", model="eleven", cost_usd=None)
    assert telemetry.get_run_log().total_cost_usd() == pytest.approx(0.01)


def test_no_cap_configured_never_stops_a_run(monkeypatch):
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", None)
    spend(1000.0)
    assert telemetry.spend_exhausted() is False


def test_spend_under_the_cap_keeps_running(monkeypatch):
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)
    spend(0.25)
    assert telemetry.spend_exhausted() is False


def test_spend_at_the_cap_stops_the_run(monkeypatch):
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)
    spend(1.0)
    assert telemetry.spend_exhausted() is True


def test_scene_loop_stops_at_the_cap_and_keeps_what_it_generated(monkeypatch, capsys, fake_cast):
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(engine, "get_director_choice",
                        lambda *a, **k: {"next_speaker": "alpha", "direction": "OPEN"})
    monkeypatch.setattr(engine, "inject_scene_update", lambda *a, **k: None)

    # Each turn costs $0.60 against a $1.00 cap: turn 1 runs, turn 2 runs
    # (the cap can only be observed after a call), turn 3 must not start.
    def bill_a_line(*a, **k):
        spend(0.60)
        return "a line"

    monkeypatch.setattr(engine, "generate_line_session", bill_a_line)
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)

    conversation, _ = engine.run_scene(
        "test", ["alpha", "beta"],
        engine.RunOptions(num_lines=10, save=False, intensity="low"),
        sessions={"alpha": SimpleNamespace(), "beta": SimpleNamespace()})

    assert len(conversation) == 2  # stopped early, kept both finished lines
    assert "Spend cap" in capsys.readouterr().out


def test_episode_stops_between_scenes_once_the_cap_is_reached(monkeypatch, capsys, fake_cast):
    monkeypatch.setattr(engine, "EPISODES", {
        "ep": {"name": "Ep", "description": "d", "intensity": "low",
               "scenes": [{"scene": "test", "lines": 1},
                          {"scene": "test", "lines": 1},
                          {"scene": "test", "lines": 1}]}})
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(engine, "get_scene_cast", lambda *a, **k: ["alpha", "beta"])
    monkeypatch.setattr(engine, "build_transition_text", lambda *a, **k: "next scene")
    monkeypatch.setattr(engine, "CharacterSession",
                        lambda cast_id, *a, **k: SimpleNamespace(
                            cast_id=cast_id, inject_context=lambda _: None,
                            trim_if_needed=lambda *_a, **_k: None))
    monkeypatch.setattr(engine, "get_director_choice",
                        lambda *a, **k: {"next_speaker": "alpha", "direction": "OPEN"})
    monkeypatch.setattr(engine, "inject_scene_update", lambda *a, **k: None)

    scenes_run = []

    def bill_a_scene(*a, **k):
        scenes_run.append(1)
        spend(0.60)
        return "a line"

    monkeypatch.setattr(engine, "generate_line_session", bill_a_scene)
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)

    engine.run_episode("ep", ["alpha", "beta"],
                       engine.RunOptions(save=False, intensity="low"))
    assert len(scenes_run) == 2  # third scene never started
    assert "Spend cap" in capsys.readouterr().out


# ── spend cap: the media stages ──────────────────────────
# These are the expensive ones (Veo bills per second of footage), they run
# AFTER the gated turn loop inside save_scene_outputs, and they are per-line
# / per-beat rather than single-shot. Ungated, a capped run could spend
# orders of magnitude more than its cap on video.

def test_tts_stops_at_the_cap_instead_of_billing_every_line(monkeypatch, tmp_path):
    from exec_producer import audio
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)
    spend(1.5)
    calls = []
    monkeypatch.setattr(audio, "get_backend",
                        lambda *a, **k: SimpleNamespace(
                            synthesize=lambda text, voice: calls.append(text) or b"mp3"))
    monkeypatch.setattr(audio, "load_voices", lambda: {
        "defaults": {}, "voices": {"alpha": {"voice_id": "v1"}}})
    conversation = [{"id": "alpha", "name": "Alpha", "text": f"line {i}"}
                    for i in range(5)]

    assert audio.render_scene_audio(conversation, str(tmp_path / "s")) is None
    assert calls == []


def test_veo_stops_at_the_cap_instead_of_generating_every_beat(monkeypatch, tmp_path):
    from exec_producer import video
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)
    spend(1.5)
    generated = []
    monkeypatch.setattr(video, "_generate_veo_clip",
                        lambda *a, **k: generated.append(a) or b"mp4")
    monkeypatch.setattr(video, "pick_h264_encoder", lambda *a, **k: "libx264")
    beats = [{"veo3_video_prompt": f"beat {i} prompt"} for i in range(4)]

    assert video.render_veo_video(beats, str(tmp_path / "s"), client=object()) is None
    assert generated == []


def test_the_media_stages_run_normally_when_under_the_cap(monkeypatch, tmp_path):
    # The gate must not break the ordinary path — including a --from-script
    # render, which makes no generation calls and so never accrues cost.
    from exec_producer import audio
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)
    monkeypatch.setattr(audio, "get_backend",
                        lambda *a, **k: SimpleNamespace(synthesize=lambda t, v: b"mp3"))
    monkeypatch.setattr(audio, "load_voices", lambda: {
        "defaults": {}, "voices": {"alpha": {"voice_id": "v1"}}})
    monkeypatch.setattr(audio, "stitch_lines", lambda *a, **k: None)
    conversation = [{"id": "alpha", "name": "Alpha", "text": "a line"}]

    assert audio.render_scene_audio(conversation, str(tmp_path / "s")) is not None


# ── config manifest / audit block ────────────────────────

def test_manifest_covers_loaded_files():
    assert "world.json" in config.CONFIG_MANIFEST
    assert "cast.json" in config.CONFIG_MANIFEST
    assert "prompts/critical_rules.txt" in config.CONFIG_MANIFEST
    for digest in config.CONFIG_MANIFEST.values():
        assert len(digest) == 64 and int(digest, 16) >= 0  # sha256 hex


def test_audit_block_shape():
    assert config.AUDIT["engine_version"]
    assert config.AUDIT["director_model"] == config.DIRECTOR_MODEL
    assert config.AUDIT["judge_model"] == config.JUDGE_MODEL
    assert config.AUDIT["config_files"] is config.CONFIG_MANIFEST


# ── run-log audit header ─────────────────────────────────

def test_header_written_before_first_record(tmp_path):
    rl = telemetry.RunLog(log_dir=tmp_path, header={"audit": {"engine_version": "x"}})
    rl.record(caller="claude", model="m", total_tokens=10, cost_usd=0.01)
    rl.record(caller="claude", model="m", total_tokens=20, cost_usd=0.01)

    lines = [json.loads(line) for line in rl.path.read_text().strip().split("\n")]
    assert len(lines) == 3
    assert lines[0]["event"] == "run_start"
    assert lines[0]["audit"] == {"engine_version": "x"}
    assert lines[1]["caller"] == "claude"
    # header is file-only: summary math still counts 2 calls
    assert rl.summary()["calls"] == 2


def test_no_records_means_no_file(tmp_path):
    rl = telemetry.RunLog(log_dir=tmp_path, header={"audit": {}})
    assert not rl.path.exists()  # --list style runs leave no log behind


def test_run_log_filename_includes_pid(tmp_path):
    import os
    rl = telemetry.RunLog(log_dir=tmp_path)
    assert f"_{os.getpid()}_calls.jsonl" in rl.path.name
