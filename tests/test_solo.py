"""Solo-host support: a cast of ONE runs end-to-end. Engine relaxations
(cast sheet, casting short-circuit, director-choice guards), CLI key gates
(--solo, one-character shows), and the solo scene loop — classic and
planned. All LLM calls mocked."""

import json
import sys

import pytest

from exec_producer import engine
from tests.test_engine import (CAST_IDS, FAKE_CAST, FAKE_TEMPLATE, SCENE,
                               director_returning)

SOLO_CAST = {"alpha": FAKE_CAST["alpha"]}


@pytest.fixture(autouse=True)
def fake_world(monkeypatch):
    monkeypatch.setattr(engine, "CAST", FAKE_CAST)
    monkeypatch.setattr(engine, "CHARACTER_TEMPLATE", FAKE_TEMPLATE)
    monkeypatch.setattr(engine, "WORLD", "THE_WORLD")
    monkeypatch.setattr(engine, "INTENSITY_GUIDE", {"low": "INT_LOW", "high": "INT_HIGH"})
    monkeypatch.setattr(engine, "EMBODIMENT", {
        "base": "EMB_BASE",
        "gender": {"guy": "EMB_GUY", "girl": "EMB_GIRL"},
        "character_overrides": {},
        "intensity": {"low": "EMB_INT_LOW", "high": "EMB_INT_HIGH"},
    })
    monkeypatch.setattr(engine, "TONE_FACE_TO_FACE", "TONE_F2F")
    monkeypatch.setattr(engine, "TONE_CONFESSIONAL_EARLY", "TONE_CONF_EARLY")
    monkeypatch.setattr(engine, "TONE_CONFESSIONAL_LATE", "TONE_CONF_LATE")
    monkeypatch.setattr(engine, "CRITICAL_RULES", "CRIT_RULES")
    monkeypatch.setattr(engine, "SHOW_NAME", "TEST SHOW")
    monkeypatch.setattr(engine, "SHOW_SLUG", "test_show")
    monkeypatch.setattr(engine.time, "sleep", lambda s: None)


# ── cast sheet ───────────────────────────────────────────

def test_cast_sheet_solo_phrasing():
    sheet = engine.build_cast_sheet(["alpha"])
    assert "solo show" in sheet
    assert "Alpha (he/him) — The A" in sheet
    assert "people in the house" not in sheet


def test_cast_sheet_multi_byte_identical():
    # Multi-cast output is pinned exactly: solo support must not touch it.
    assert engine.build_cast_sheet(["alpha", "beta"]) == (
        "THE CAST (memorize — these are the people in the house with you):\n"
        "  - Alpha (he/him) — The A\n"
        "  - Beta (she/her) — The B"
    )


# ── scene casting ────────────────────────────────────────

def test_scene_cast_solo_skips_director(monkeypatch):
    def boom(system, prompt):
        raise AssertionError("director must not be called for a solo cast")
    monkeypatch.setattr(engine, "call_director", boom)
    assert engine.get_scene_cast(SCENE, ["alpha"]) == ["alpha"]


def test_scene_cast_multi_still_asks_director(monkeypatch):
    calls = director_returning(monkeypatch, [json.dumps({"cast": ["alpha", "beta"]})])
    assert engine.get_scene_cast(SCENE, CAST_IDS) == ["alpha", "beta"]
    assert calls["n"] == 1


# ── director choice with one speaker ─────────────────────

def test_director_choice_solo_monologue_keeps_speaker(monkeypatch):
    note = {"next_speaker": "alpha", "direction": "MONOLOGUE", "target": None}
    calls = director_returning(monkeypatch, [json.dumps(note)])
    result = engine.get_director_choice(SCENE, [], ["alpha"], "alpha")
    assert result["next_speaker"] == "alpha"
    assert calls["n"] == 1


def test_director_choice_solo_repeat_speaker_without_monologue(monkeypatch):
    # A non-MONOLOGUE same-speaker pick lands in the reassign branch; with
    # one member the guard must return that member, direction intact.
    note = {"next_speaker": "alpha", "direction": "RESPOND_TO", "target": None}
    director_returning(monkeypatch, [json.dumps(note)])
    result = engine.get_director_choice(SCENE, [], ["alpha"], "alpha")
    assert result["next_speaker"] == "alpha"
    assert result["direction"] == "RESPOND_TO"


def test_director_choice_solo_api_error_fallback(monkeypatch):
    # The random-pick fallback's empty-options guard: [c != last] is [].
    director_returning(monkeypatch, [RuntimeError("api down")])
    result = engine.get_director_choice(SCENE, [], ["alpha"], "alpha")
    assert result["next_speaker"] == "alpha"
    assert result["reasoning"] == "director fallback"


def test_director_choice_solo_malformed_json_fallback(monkeypatch):
    calls = director_returning(monkeypatch, ["not json"])
    result = engine.get_director_choice(SCENE, [], ["alpha"], "alpha")
    assert result["next_speaker"] == "alpha"
    assert calls["n"] == 3  # malformed responses retry, then fall back


# ── solo scene loop, classic ─────────────────────────────

def test_run_scene_solo_end_to_end(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(
        engine, "get_director_choice",
        lambda *a, **k: {"next_speaker": "alpha", "direction": "MONOLOGUE"})
    lines = {"n": 0}

    def fake_line(session):
        lines["n"] += 1
        return f"line {lines['n']}"
    monkeypatch.setattr(engine, "generate_line_session", fake_line)

    conversation, notes = engine.run_scene(
        "test", ["alpha"], engine.RunOptions(num_lines=2, intensity="low"))

    assert [e["id"] for e in conversation] == ["alpha", "alpha"]
    assert len(notes) == 2
    jsons = [p for p in (tmp_path / "scripts").glob("test_show_test_*.json")
             if not p.name.endswith("_eval.json")]
    assert len(jsons) == 1
    data = json.loads(jsons[0].read_text())
    assert list(data["cast"].keys()) == ["alpha"]
    assert data["audit"]["config_files"]
    assert "telemetry" in data


# ── solo scene loop, planned ─────────────────────────────

def test_run_scene_planned_solo(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    from exec_producer import planner
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(planner.time, "sleep", lambda s: None)
    monkeypatch.setattr(planner, "load_planner_prompt", lambda: "PLANNER_SYS")
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "scene_test_solo")
    monkeypatch.setattr(planner, "call_model_simple", lambda *a, **k: json.dumps(
        {"turns": [{"turn": 1, "phase": "settle-in", "energy": "low", "guidance": "ease in"},
                   {"turn": 2, "phase": "simmer", "energy": "simmer", "guidance": "go deeper"}]}))
    plan_notes = []

    def fake_choice(scene, conversation, cast_ids, last_speaker,
                    intensity="high", plot="", plan_note=""):
        plan_notes.append(plan_note)
        return {"next_speaker": "alpha", "direction": "MONOLOGUE"}
    monkeypatch.setattr(engine, "get_director_choice", fake_choice)
    monkeypatch.setattr(engine, "generate_line_session", lambda s: "spoken line")

    conversation, notes = planner.run_scene_planned(
        "test", ["alpha"], engine.RunOptions(num_lines=2, intensity="low"))

    assert [e["id"] for e in conversation] == ["alpha", "alpha"]
    assert "settle-in" in plan_notes[0] and "simmer" in plan_notes[1]


# ── CLI key gates ────────────────────────────────────────

def _cli_main(monkeypatch, argv):
    from exec_producer import cli
    monkeypatch.setattr(sys, "argv", ["exec-producer"] + argv)
    cli.main()


def _patch_cli(monkeypatch, cast, available):
    from exec_producer import cli
    monkeypatch.setattr(cli, "CAST", cast)
    monkeypatch.setattr(cli, "SCENES", {"test": {"name": "T", "prompt": "p", "mood": "m"}})
    monkeypatch.setattr(cli, "EPISODES", {"test_ep": {}})
    monkeypatch.setattr(cli, "check_api_keys", lambda ids: (available, []))
    monkeypatch.setattr(cli, "director_key_or_die", lambda: None)
    captured = {}
    monkeypatch.setattr(cli, "run_scene",
                        lambda scene_id, cast_ids, opts: captured.update(cast=cast_ids))
    monkeypatch.setattr(cli, "run_episode",
                        lambda episode_id, cast_ids, opts: captured.update(cast=cast_ids))
    return captured


def test_cli_one_character_show_runs_with_one_key(monkeypatch, tmp_path):
    # No --solo needed: a one-character show can only be solo.
    monkeypatch.chdir(tmp_path)
    captured = _patch_cli(monkeypatch, SOLO_CAST, ["alpha"])
    _cli_main(monkeypatch, ["--scene", "test", "--lines", "2"])
    assert captured["cast"] == ["alpha"]


def test_cli_rejects_nonpositive_lines(monkeypatch, capsys):
    # --lines 0 would save an empty script on the classic path and still
    # direct-and-bill one line on the planner path — reject it up front.
    _patch_cli(monkeypatch, FAKE_CAST, list(FAKE_CAST))
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "test", "--lines", "0"])
    assert "--lines" in capsys.readouterr().out


def test_cli_multi_show_still_needs_two_keys(monkeypatch, capsys):
    _patch_cli(monkeypatch, FAKE_CAST, ["alpha"])
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "test"])
    assert "at least 2 API keys" in capsys.readouterr().out


def test_cli_solo_flag_on_multi_show(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    captured = _patch_cli(monkeypatch, FAKE_CAST, ["alpha"])
    _cli_main(monkeypatch, ["--scene", "test", "--cast", "alpha", "--solo"])
    assert captured["cast"] == ["alpha"]


def test_cli_solo_flag_requires_single_cast(monkeypatch, capsys):
    _patch_cli(monkeypatch, FAKE_CAST, list(FAKE_CAST))
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "test", "--solo"])
    assert "--cast <id>" in capsys.readouterr().out


def test_cli_solo_flag_missing_key_still_dies(monkeypatch, capsys):
    _patch_cli(monkeypatch, FAKE_CAST, [])
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "test", "--cast", "alpha", "--solo"])
    assert "cast member's API key" in capsys.readouterr().out


def test_cli_episode_solo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    captured = _patch_cli(monkeypatch, SOLO_CAST, ["alpha"])
    _cli_main(monkeypatch, ["--episode", "test_ep"])
    assert captured["cast"] == ["alpha"]


# ── interactive gate ─────────────────────────────────────

def test_interactive_solo_show_passes_key_gate(monkeypatch, tmp_path):
    from exec_producer import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "CAST", SOLO_CAST)
    monkeypatch.setattr(cli, "SCENES", {"test": {"name": "T", "prompt": "p", "mood": "m"}})
    monkeypatch.setattr(cli, "SHOW_NAME", "TEST SHOW")
    monkeypatch.setattr(cli, "check_api_keys", lambda ids: (["alpha"], []))
    monkeypatch.setattr(cli, "director_key_or_die", lambda: None)
    captured = {}
    monkeypatch.setattr(cli, "run_scene",
                        lambda scene_id, cast_ids, opts: captured.update(cast=cast_ids))
    answers = iter(["all", "1", "2", "low", "n", "n"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    cli.interactive_mode()

    assert captured["cast"] == ["alpha"]


def test_interactive_multi_show_still_needs_two(monkeypatch, capsys):
    from exec_producer import cli
    monkeypatch.setattr(cli, "CAST", FAKE_CAST)
    monkeypatch.setattr(cli, "SHOW_NAME", "TEST SHOW")
    monkeypatch.setattr(cli, "check_api_keys", lambda ids: (["alpha"], []))
    monkeypatch.setattr(cli, "director_key_or_die", lambda: None)
    with pytest.raises(SystemExit):
        cli.interactive_mode()
    assert "Need at least 2 APIs" in capsys.readouterr().out
