"""Unit tests for the LangGraph planner — all LLM calls mocked; the graph
and its sqlite checkpointing run for real (langgraph is a dev dependency
via the planner extra)."""

import json
import sys

import pytest

from exec_producer import engine, planner
from tests.test_engine import CAST_IDS, FAKE_CAST, FAKE_TEMPLATE, SCENE


@pytest.fixture(autouse=True)
def fake_world(monkeypatch):
    monkeypatch.setattr(engine, "CAST", FAKE_CAST)
    monkeypatch.setattr(engine, "CHARACTER_TEMPLATE", FAKE_TEMPLATE)
    monkeypatch.setattr(engine, "WORLD", "THE_WORLD")
    monkeypatch.setattr(engine, "INTENSITY_GUIDE", {"low": "INT_LOW", "high": "INT_HIGH"})
    monkeypatch.setattr(engine, "EMBODIMENT", {
        "base": "EMB_BASE",
        "gender": {"guy": "EMB_GUY", "girl": "EMB_GIRL"},
        "character_overrides": {"gamma": "EMB_GAMMA_OVERRIDE"},
        "intensity": {"low": "EMB_INT_LOW", "high": "EMB_INT_HIGH"},
    })
    monkeypatch.setattr(engine, "TONE_FACE_TO_FACE", "TONE_F2F")
    monkeypatch.setattr(engine, "TONE_CONFESSIONAL_EARLY", "TONE_CONF_EARLY")
    monkeypatch.setattr(engine, "TONE_CONFESSIONAL_LATE", "TONE_CONF_LATE")
    monkeypatch.setattr(engine, "CRITICAL_RULES", "CRIT_RULES")
    monkeypatch.setattr(engine, "SHOW_NAME", "TEST SHOW")
    monkeypatch.setattr(engine, "SHOW_SLUG", "test_show")
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(engine.time, "sleep", lambda s: None)
    monkeypatch.setattr(planner.time, "sleep", lambda s: None)
    monkeypatch.setattr(planner, "load_planner_prompt", lambda: "PLANNER_SYS")


PLAN_JSON = json.dumps({"turns": [
    {"turn": 1, "phase": "settle-in", "energy": "low", "guidance": "small talk"},
    {"turn": 2, "phase": "simmer", "energy": "simmer", "guidance": "seed tension"},
    {"turn": 3, "phase": "spark", "energy": "spark", "guidance": "first jab"},
]})


def planner_harness(monkeypatch, tmp_path, lines_fail_on=None):
    """Mock every LLM touchpoint; graph + checkpointing run for real."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(planner, "call_model_simple",
                        lambda *a, **k: PLAN_JSON)

    calls = {"director": 0, "lines": 0, "plan_notes": []}

    def fake_choice(scene, conversation, cast_ids, last_speaker,
                    intensity="high", plot="", plan_note=""):
        calls["director"] += 1
        calls["plan_notes"].append(plan_note)
        return {"next_speaker": "alpha" if calls["director"] % 2 else "beta",
                "direction": "OPEN"}

    def fake_line(session):
        calls["lines"] += 1
        if lines_fail_on and calls["lines"] in lines_fail_on:
            raise RuntimeError("provider died mid-scene")
        return f"line {calls['lines']}"

    monkeypatch.setattr(engine, "get_director_choice", fake_choice)
    monkeypatch.setattr(engine, "generate_line_session", fake_line)
    return calls


# ── engine seam: the plan note in the director prompt ────

def test_plan_note_reaches_director_prompt(monkeypatch):
    captured = {}

    def fake_director(system, prompt):
        captured["prompt"] = prompt
        return json.dumps({"next_speaker": "alpha", "direction": "OPEN"})

    monkeypatch.setattr(engine, "call_director", fake_director)
    engine.get_director_choice(SCENE, [], CAST_IDS, None, "low",
                               plan_note="phase: settle-in | energy: low")
    assert "THE SCENE PLAN FOR THIS TURN" in captured["prompt"]
    assert "phase: settle-in" in captured["prompt"]
    assert "do not escalate beyond" in captured["prompt"]

    engine.get_director_choice(SCENE, [], CAST_IDS, None, "low")
    assert "THE SCENE PLAN" not in captured["prompt"]


# ── session state round-trip ─────────────────────────────

def test_session_state_round_trip():
    live = engine.CharacterSession("alpha", SCENE, "low", "plot", CAST_IDS)
    live.inject_context("something happened")
    live.add_own_response("my line")
    state = live.to_state()
    json.dumps(state)  # must be checkpoint-serializable

    rebuilt = engine.CharacterSession.from_messages("alpha", **state)
    assert rebuilt.messages == live.messages
    assert rebuilt.model == live.model and rebuilt.line_count == 1
    rebuilt.messages.append({"role": "user", "content": "x"})
    assert len(live.messages) != len(rebuilt.messages)  # no shared mutation


# ── plan node parse + degradation ────────────────────────

def _plan_state():
    return {"scene_queue": [{"scene_id": "test", "lines": 3, "plot": ""}],
            "scene_index": 0, "scene_cast": ["alpha", "beta"], "intensity": "low"}


def test_plan_scene_parses_arc(monkeypatch, tmp_path):
    monkeypatch.setattr(planner, "call_model_simple", lambda *a, **k: PLAN_JSON)
    result = planner.plan_scene(_plan_state())
    assert [t["phase"] for t in result["plan"]] == ["settle-in", "simmer", "spark"]


def test_plan_scene_degrades_on_garbage(monkeypatch):
    monkeypatch.setattr(planner, "call_model_simple", lambda *a, **k: "not json at all")
    assert planner.plan_scene(_plan_state()) == {"plan": []}


def test_plan_scene_degrades_on_api_error(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("api down")
    monkeypatch.setattr(planner, "call_model_simple", boom)
    assert planner.plan_scene(_plan_state()) == {"plan": []}
    assert "without a plan" in capsys.readouterr().out


# ── routing ──────────────────────────────────────────────

def test_routes():
    state = {"scene_queue": [{"scene_id": "test", "lines": 3}], "scene_index": 0,
             "turn_index": 2}
    assert planner.route_after_speak(state) == "continue"
    state["turn_index"] = 3
    assert planner.route_after_speak(state) == "wrap"
    state["scene_index"] = 1
    assert planner.route_after_wrap(state) == "end"
    state["scene_queue"].append({"scene_id": "test", "lines": 2})
    assert planner.route_after_wrap(state) == "next_scene"


# ── full graph runs ──────────────────────────────────────

def test_run_scene_planned_end_to_end(monkeypatch, tmp_path):
    calls = planner_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "scene_test_1")

    conversation, notes = planner.run_scene_planned(
        "test", ["alpha", "beta"],
        engine.RunOptions(num_lines=3, intensity="low"))

    assert [e["text"] for e in conversation] == ["line 1", "line 2", "line 3"]
    assert len(notes) == 3
    # the plan actually reached the director, turn by turn
    assert "Turn 1/3" in calls["plan_notes"][0] and "settle-in" in calls["plan_notes"][0]
    assert "Turn 3/3" in calls["plan_notes"][2] and "first jab" in calls["plan_notes"][2]

    jsons = [p for p in (tmp_path / "scripts").glob("test_show_test_*.json")
             if not p.name.endswith("_eval.json")]
    data = json.loads(jsons[0].read_text())
    assert data["planner"]["thread_id"] == "scene_test_1"
    assert [t["phase"] for t in data["planner"]["plan"]] == ["settle-in", "simmer", "spark"]
    assert (tmp_path / "scripts" / "planner_state.sqlite").exists()


def test_run_episode_planned_transitions_and_casting(monkeypatch, tmp_path):
    planner_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "episode_ep_1")
    monkeypatch.setattr(engine, "SCENES", {"s1": {"name": "One", "prompt": "p", "mood": "m"},
                                           "s2": {"name": "Two", "prompt": "p", "mood": "m"}})
    monkeypatch.setattr(engine, "EPISODES", {"ep": {
        "name": "Ep", "description": "d", "intensity": "low",
        "scenes": [{"scene": "s1", "lines": 2, "plot": "p1"},
                   {"scene": "s2", "lines": 2, "plot": "p2"}]}})
    monkeypatch.setattr(engine, "get_scene_cast", lambda *a, **k: ["alpha", "beta"])

    all_scenes = planner.run_episode_planned("ep", CAST_IDS,
                                             engine.RunOptions(intensity="low"))

    assert len(all_scenes) == 2
    # forced-cast merge: gamma never spoke in scene 1 → forced into scene 2
    assert set(all_scenes[1]["cast"]) == {"alpha", "beta", "gamma"}
    ep_jsons = [p for p in (tmp_path / "scripts").glob("test_show_episode_ep_*.json")
                if not p.name.endswith("_eval.json")]
    data = json.loads(ep_jsons[0].read_text())
    assert data["planner"]["thread_id"] == "episode_ep_1"
    assert len(data["scenes"]) == 2
    assert all(s["plan"] for s in data["scenes"])  # per-scene arc in the audit trail


def test_resume_from_checkpoint_skips_paid_turns(monkeypatch, tmp_path):
    # Crash on the third character line, then resume the same thread: the
    # first two turns must come from the checkpoint, not fresh LLM calls.
    calls = planner_harness(monkeypatch, tmp_path, lines_fail_on={3})
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "scene_test_r1")

    # generate_line_session in the real engine never raises (it degrades), but
    # a hard crash (KeyboardInterrupt, OOM, power) still kills a run mid-graph;
    # the RuntimeError here stands in for that.
    with pytest.raises(RuntimeError):
        planner.run_scene_planned("test", ["alpha", "beta"],
                                  engine.RunOptions(num_lines=3, intensity="low"))
    director_calls_before = calls["director"]
    assert director_calls_before == 3  # turns 1-3 directed; turn 3's line crashed

    calls["lines"] = 10  # past the failure trigger; next fake line is "line 11"
    conversation, _ = planner.run_scene_planned(
        "test", ["alpha", "beta"],
        engine.RunOptions(num_lines=3, intensity="low"), thread_id="scene_test_r1")

    # Turns 1-2 replayed from checkpoint (texts preserved), only turn 3 re-ran
    assert [e["text"] for e in conversation] == ["line 1", "line 2", "line 11"]
    # No additional director calls: the checkpoint already held turn 3's note
    assert calls["director"] == director_calls_before


def test_resume_rejects_wrong_run_type(monkeypatch, tmp_path):
    planner_harness(monkeypatch, tmp_path)
    from exec_producer.errors import ThreadResumeError
    with pytest.raises(ThreadResumeError, match="was not created by this command"):
        planner.run_scene_planned("test", ["alpha", "beta"],
                                  engine.RunOptions(num_lines=3, intensity="low"),
                                  thread_id="episode_pilot_20260101_1")


def test_resume_rejects_unknown_thread(monkeypatch, tmp_path):
    planner_harness(monkeypatch, tmp_path)
    from exec_producer.errors import ThreadResumeError
    with pytest.raises(ThreadResumeError, match="unknown planner thread"):
        planner.run_scene_planned("test", ["alpha", "beta"],
                                  engine.RunOptions(num_lines=3, intensity="low"),
                                  thread_id="scene_test_never_ran")


# ── CLI ──────────────────────────────────────────────────

def _cli_main(monkeypatch, argv):
    from exec_producer import cli
    monkeypatch.setattr(sys, "argv", ["exec-producer"] + argv)
    cli.main()


def test_cli_thread_requires_planner(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "x", "--thread", "t1"])
    assert "--planner" in capsys.readouterr().out


def test_cli_planner_dispatch_and_preflight(monkeypatch, tmp_path, capsys):
    from exec_producer import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "SCENES", {"test": SCENE})
    monkeypatch.setattr(cli, "check_api_keys", lambda ids: (["a", "b"], []))
    monkeypatch.setattr(cli, "resolve_cast", lambda arg: ["a", "b"])
    monkeypatch.setattr(cli, "director_key_or_die", lambda: None)
    captured = {}
    monkeypatch.setattr(planner, "run_scene_planned",
                        lambda scene_id, cast, opts, thread_id=None:
                        captured.update(scene=scene_id, thread=thread_id))

    _cli_main(monkeypatch, ["--scene", "test", "--planner", "--thread", "t9"])
    assert captured == {"scene": "test", "thread": "t9"}

    monkeypatch.setattr(planner, "preflight", lambda: ["langgraph not installed — x"])
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "test", "--planner"])
    assert "langgraph not installed" in capsys.readouterr().out
