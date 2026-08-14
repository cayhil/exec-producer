"""Unit tests for the LangGraph planner — all LLM calls mocked; the graph
and its sqlite checkpointing run for real (langgraph is a dev dependency
via the planner extra)."""

import json
import sys

import pytest

from exec_producer import engine, planner, telemetry
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
    # Host independence (AGENTS.md): a contributor's private world.json may
    # set budget.max_run_usd, and the graph has real budget gates — pin the
    # cap off unless a test opts in.
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", None)
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
    assert planner.plan_scene(_plan_state())["plan"] == []


def test_plan_scene_degrades_on_api_error(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("api down")
    monkeypatch.setattr(planner, "call_model_simple", boom)
    assert planner.plan_scene(_plan_state())["plan"] == []
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


def test_the_arc_writer_pauses_instead_of_billing_past_the_cap(monkeypatch):
    # plan_scene is the most expensive single call in the planner
    # (max_tokens=2000, three attempts). Raising — not returning an empty
    # plan — keeps the checkpoint's `next` pointed here, so --thread
    # re-enters at this node instead of the thread reading as finished.
    from exec_producer.errors import BudgetPause
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)
    telemetry.get_run_log().record(caller="director", model="m", cost_usd=1.5)
    monkeypatch.setattr(planner, "call_model_simple",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("billed the planner past the cap")))
    with pytest.raises(BudgetPause):
        planner.plan_scene({"scene_queue": [{"scene_id": "test", "lines": 3}],
                            "scene_index": 0, "intensity": "low",
                            "cast_ids": ["alpha"], "plan": None})


def test_the_casting_call_pauses_instead_of_billing_past_the_cap(monkeypatch):
    # cast_scene is the between-scenes pause point: wrap_scene has archived
    # everything and spends nothing, so a resume starts the next scene
    # clean. The gate must fire before the casting call bills the director.
    from exec_producer.errors import BudgetPause
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 0)
    monkeypatch.setattr(engine, "get_scene_cast",
                        lambda *a, **k: pytest.fail("billed the casting call past the cap"))
    with pytest.raises(BudgetPause):
        planner.cast_scene({"scene_queue": [{"scene_id": "test", "lines": 1}],
                            "scene_index": 1, "cast_ids": ["alpha"],
                            "intensity": "low", "sessions": {},
                            "completed_scenes": []})


def test_the_routers_no_longer_consult_the_budget(monkeypatch):
    # Routing must be purely structural: a budget check in a router steers
    # to wrap/END, which truncates the arc or completes the thread — both
    # are what the node-top gates exist to replace. Failing on the call
    # itself (not on spend state) pins that the routers don't even ask.
    monkeypatch.setattr(planner.telemetry, "spend_exhausted",
                        lambda: pytest.fail("router consulted the budget gate"))
    state = {"scene_queue": [{"scene_id": "test", "lines": 3}], "scene_index": 0,
             "turn_index": 1}
    assert planner.route_after_speak(state) == "continue"
    state = {"scene_queue": [{"scene_id": "a", "lines": 1},
                             {"scene_id": "b", "lines": 1}],
             "scene_index": 1, "turn_index": 1}
    assert planner.route_after_wrap(state) == "next_scene"


def test_every_billing_node_writes_cumulative_spend(monkeypatch):
    # The checkpointed spend is what a resume seeds its budget from; any
    # billing node that forgets the key silently reopens the N-resumes
    # overspend. speak's write is pinned separately by the cumulative-
    # budget test in the pause section (the second pause's seed IS speak's
    # written value).
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", None)
    telemetry.get_run_log().record(caller="director", model="m", cost_usd=0.25)
    telemetry.get_run_log().prior_spend_usd = 0.5
    monkeypatch.setattr(engine, "get_director_choice",
                        lambda *a, **k: {"next_speaker": "alpha", "direction": "OPEN"})
    monkeypatch.setattr(engine, "get_scene_cast", lambda *a, **k: ["alpha"])
    monkeypatch.setattr(planner, "call_model_simple", lambda *a, **k: PLAN_JSON)

    direct = planner.direct_turn({"scene_queue": [{"scene_id": "test", "lines": 3}],
                                  "scene_index": 0, "turn_index": 0,
                                  "scene_cast": ["alpha"], "conversation": [],
                                  "last_speaker": None, "intensity": "low", "plan": []})
    plan = planner.plan_scene({"scene_queue": [{"scene_id": "test", "lines": 3}],
                               "scene_index": 0, "scene_cast": ["alpha"],
                               "intensity": "low"})
    cast = planner.cast_scene({"scene_queue": [{"scene_id": "test", "lines": 3,
                                                "plot": ""}],
                               "scene_index": 0, "cast_ids": ["alpha"],
                               "intensity": "low", "sessions": {},
                               "completed_scenes": []})
    assert direct["spent_usd"] == plan["spent_usd"] == cast["spent_usd"] == 0.75


# ── full graph runs ──────────────────────────────────────

def test_run_scene_planned_end_to_end(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
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
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
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
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
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


def test_resume_rejects_prefix_overlapping_scene(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    # "test" is a valid prefix of "test_2", so the thread-id prefix guard
    # alone would accept resuming a test_2 checkpoint under --scene test and
    # save the wrong scene under this run's identity. The checkpoint's own
    # scene queue is the authority.
    calls = planner_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(engine, "SCENES", {
        "test": SCENE, "test_2": {"name": "Two", "prompt": "p", "mood": "m"}})
    monkeypatch.setattr(planner, "_new_thread_id",
                        lambda prefix: f"{prefix}_20260101_120000_1")
    planner.run_scene_planned("test_2", ["alpha", "beta"],
                              engine.RunOptions(num_lines=1, intensity="low"))
    lines_before = calls["lines"]

    from exec_producer.errors import ThreadResumeError
    with pytest.raises(ThreadResumeError, match="checkpointed for scenes"):
        planner.run_scene_planned("test", ["alpha", "beta"],
                                  engine.RunOptions(num_lines=1, intensity="low"),
                                  thread_id="scene_test_2_20260101_120000_1")
    assert calls["lines"] == lines_before  # rejected before any dialogue ran


def test_resume_rejects_unknown_thread(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    planner_harness(monkeypatch, tmp_path)
    from exec_producer.errors import ThreadResumeError
    with pytest.raises(ThreadResumeError, match="unknown planner thread"):
        planner.run_scene_planned("test", ["alpha", "beta"],
                                  engine.RunOptions(num_lines=3, intensity="low"),
                                  thread_id="scene_test_never_ran")


# ── budget pause + resume ────────────────────────────────

def billing_harness(monkeypatch, tmp_path, cost_per_line=0.5):
    """planner_harness plus a cost per character line, so a cap can trip
    mid-scene exactly the way a real run's recorded spend does."""
    calls = planner_harness(monkeypatch, tmp_path)
    fake_line = engine.generate_line_session  # the harness's fake

    def billed_line(session):
        telemetry.get_run_log().record(caller="character", model="m",
                                       cost_usd=cost_per_line)
        return fake_line(session)

    monkeypatch.setattr(engine, "generate_line_session", billed_line)
    return calls


def fresh_process(tmp_path, tag):
    # A production resume is always a new process with an empty RunLog; in
    # one test process we simulate that (a shared log would double-count
    # the seed). A subdir keeps its JSONL separate from the paused run's,
    # the way a real second process's timestamp+pid filename would be.
    telemetry._run_log = telemetry.RunLog(log_dir=tmp_path / tag)


def test_a_budget_pause_saves_a_partial_scene_and_resumes(monkeypatch, tmp_path, capsys):
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    calls = billing_harness(monkeypatch, tmp_path)   # $0.50/line
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "scene_test_bp1")
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)  # cap trips after line 2

    conversation, _ = planner.run_scene_planned(
        "test", ["alpha", "beta"], engine.RunOptions(num_lines=4, intensity="low"))

    assert [e["text"] for e in conversation] == ["line 1", "line 2"]
    out = capsys.readouterr().out
    # Distinctive pause-reporter text — the banner already prints "--thread"
    # on every run, so that substring alone proves nothing.
    assert "Budget pause" in out and "--thread scene_test_bp1" in out
    # Partial artifact content asserted BEFORE resuming: in one process a
    # same-second resume can reuse the timestamp+pid prefix and overwrite
    # this file (production saves come from different processes/pids).
    jsons = [p for p in (tmp_path / "scripts").glob("test_show_test_*.json")
             if not p.name.endswith("_eval.json")]
    partial = json.loads(jsons[0].read_text())
    assert partial["partial"] is True
    assert len(partial["conversation"]) == 2
    assert partial["planner"]["thread_id"] == "scene_test_bp1"
    lines_before, directed_before = calls["lines"], calls["director"]

    fresh_process(tmp_path, "resume1")
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 10.0)  # the user raised the cap
    conversation, _ = planner.run_scene_planned(
        "test", ["alpha", "beta"], engine.RunOptions(num_lines=4, intensity="low"),
        thread_id="scene_test_bp1")

    # The full arc, with only the unpaid turns billed
    assert [e["text"] for e in conversation] == ["line 1", "line 2", "line 3", "line 4"]
    assert calls["lines"] == lines_before + 2
    assert calls["director"] == directed_before + 2
    jsons = [p for p in (tmp_path / "scripts").glob("test_show_test_*.json")
             if not p.name.endswith("_eval.json")]
    final = json.loads(sorted(jsons)[-1].read_text())
    assert "partial" not in final
    assert len(final["conversation"]) == 4


def test_resuming_without_raising_the_cap_declines_without_billing_or_resaving(
        monkeypatch, tmp_path, capsys):
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    calls = billing_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "scene_test_bp2")
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)
    planner.run_scene_planned("test", ["alpha", "beta"],
                              engine.RunOptions(num_lines=4, intensity="low"))
    lines_before = calls["lines"]
    saved_before = sorted((tmp_path / "scripts").glob("test_show_test_*"))

    fresh_process(tmp_path, "resume1")  # same cap: seeded spend already exhausts it
    planner.run_scene_planned("test", ["alpha", "beta"],
                              engine.RunOptions(num_lines=4, intensity="low"),
                              thread_id="scene_test_bp2")
    # Declined BEFORE entering the graph: the checkpointed node could be a
    # gateless one (speak, on a crash-resume), so entering could bill. And
    # nothing new was generated, so nothing re-saves — a user retrying the
    # resume N times must not get N duplicate partial artifacts.
    assert calls["lines"] == lines_before
    assert sorted((tmp_path / "scripts").glob("test_show_test_*")) == saved_before
    assert "Budget pause" in capsys.readouterr().out


def test_resumes_share_one_cumulative_budget(monkeypatch, tmp_path):
    # Cap $2 total. Run 1 spends $1 and pauses. The resume seeds that $1,
    # so it affords two more lines — not four — then pauses again. The
    # second pause's seed value IS speak's checkpointed spent_usd write.
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    calls = billing_harness(monkeypatch, tmp_path)   # $0.50/line
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "scene_test_bp3")

    def run(thread_id=None):
        return planner.run_scene_planned(
            "test", ["alpha", "beta"],
            engine.RunOptions(num_lines=6, intensity="low"), thread_id=thread_id)

    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)
    conversation, _ = run()
    assert len(conversation) == 2                    # $1.00 spent, paused

    fresh_process(tmp_path, "resume1")
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 2.0)
    conversation, _ = run(thread_id="scene_test_bp3")
    assert len(conversation) == 4                    # +$1.00 → $2.00 cumulative, paused

    fresh_process(tmp_path, "resume2")
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 10.0)
    conversation, _ = run(thread_id="scene_test_bp3")
    assert [e["text"] for e in conversation] == [f"line {n}" for n in range(1, 7)]
    assert calls["lines"] == 6                       # every turn billed exactly once


def test_pause_saves_skip_the_judge_until_completion(monkeypatch, tmp_path):
    # The pause-save must not bill downstream stages on a half-finished
    # script; they run (at most) once, at resumed completion. Guarding the
    # money path explicitly, per AGENTS.md — without this test, dropping
    # without_billed_stages() from the pause-save would go unnoticed.
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    billing_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "scene_test_bp5")
    judged = []
    monkeypatch.setattr(engine, "evaluate_scene",
                        lambda *a, **k: judged.append(1) or None)
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)
    opts = engine.RunOptions(num_lines=4, intensity="low", evaluate=True)
    planner.run_scene_planned("test", ["alpha", "beta"], opts)
    assert judged == []          # paused at 2 lines; judge not billed

    fresh_process(tmp_path, "resume1")
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 10.0)
    planner.run_scene_planned("test", ["alpha", "beta"], opts,
                              thread_id="scene_test_bp5")
    assert judged == [1]         # once, on the completed scene


def test_a_cap_zero_planner_run_declines_before_the_banner(monkeypatch, tmp_path, capsys):
    # Regression, two layers: (1) an exhausted budget at a scene's first
    # turn used to fall through direct_turn's empty update into speak,
    # which crashed on the never-written current_note; (2) the banner used
    # to print a --thread hint before the decline could stop it — a thread
    # id with no checkpoint behind it, dead-ending in ThreadResumeError.
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    calls = planner_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "scene_test_z1")
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 0)
    conversation, notes = planner.run_scene_planned(
        "test", ["alpha"], engine.RunOptions(num_lines=2, intensity="low"))
    assert (conversation, notes) == ([], [])
    assert calls["lines"] == 0 and calls["director"] == 0
    out = capsys.readouterr().out
    assert "raise it in world.json" in out
    assert "--thread" not in out   # no checkpoint exists — no dead-end hint


def test_a_pause_before_any_line_saves_nothing():
    # Cap trips between plan_scene and the first line: a direct_turn pause
    # with an empty conversation. The or-None keeps the zero-line-artifact
    # rule (precedent: run_episode drops zero-line scenes).
    outcome = planner.GraphOutcome(values={"conversation": []}, paused=True,
                                   next_node="direct_turn")
    assert planner._paused_partial(outcome) is None


def test_a_completed_thread_still_resaves_when_its_spend_exceeds_the_cap(
        monkeypatch, tmp_path, capsys):
    # A finished thread's checkpoint carries spent_usd >= cap. The decline
    # gate must not hijack the "already completed — re-save" path: invoking
    # a completed thread runs zero graph nodes (verified on langgraph
    # 1.2.9). NOTE the seed assertion is what makes this red before the
    # pause plumbing exists — the re-save itself already works today.
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    calls = billing_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "scene_test_bp4")
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 100.0)
    planner.run_scene_planned("test", ["alpha", "beta"],
                              engine.RunOptions(num_lines=2, intensity="low"))
    lines_before = calls["lines"]

    fresh_process(tmp_path, "resume1")
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 0.5)  # below the $1 already spent
    conversation, _ = planner.run_scene_planned(
        "test", ["alpha", "beta"], engine.RunOptions(num_lines=2, intensity="low"),
        thread_id="scene_test_bp4")
    assert len(conversation) == 2
    assert calls["lines"] == lines_before
    out = capsys.readouterr().out
    assert "already completed" in out
    assert "This thread has already spent" in out   # seeding ran on this path too


def _episode_world(monkeypatch):
    monkeypatch.setattr(engine, "SCENES", {"s1": {"name": "One", "prompt": "p", "mood": "m"},
                                           "s2": {"name": "Two", "prompt": "p", "mood": "m"}})
    monkeypatch.setattr(engine, "EPISODES", {"ep": {
        "name": "Ep", "description": "d", "intensity": "low",
        "scenes": [{"scene": "s1", "lines": 2, "plot": "p1"},
                   {"scene": "s2", "lines": 2, "plot": "p2"}]}})
    monkeypatch.setattr(engine, "get_scene_cast", lambda *a, **k: ["alpha", "beta"])


def test_an_episode_budget_pause_saves_completed_scenes_and_resumes(
        monkeypatch, tmp_path):
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    calls = billing_harness(monkeypatch, tmp_path)   # $0.50/line
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "episode_ep_bp1")
    _episode_world(monkeypatch)
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)  # gone after scene 1's 2 lines

    completed = planner.run_episode_planned("ep", CAST_IDS,
                                            engine.RunOptions(intensity="low"))

    # Scene 1 wrapped and archived; the pause hit scene 2's cast_scene
    # gate, so there is no in-flight partial — just the completed scene.
    assert len(completed) == 1
    ep_jsons = [p for p in (tmp_path / "scripts").glob("test_show_episode_ep_*.json")
                if not p.name.endswith("_eval.json")]
    paused_save = json.loads(ep_jsons[0].read_text())
    assert paused_save["partial"] is True
    assert len(paused_save["scenes"]) == 1

    fresh_process(tmp_path, "resume1")
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 10.0)
    all_scenes = planner.run_episode_planned("ep", CAST_IDS,
                                             engine.RunOptions(intensity="low"),
                                             thread_id="episode_ep_bp1")
    assert len(all_scenes) == 2
    assert calls["lines"] == 4  # scene 1's two lines were never re-billed


def test_an_episode_mid_scene_pause_saves_the_partial_scene_marked(
        monkeypatch, tmp_path):
    # The other pause shape: the cap dies inside a scene, so the in-flight
    # dialogue joins the episode save as an entry marked partial (without
    # the wrap treatment — wrapping would advance state the resume needs).
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    calls = billing_harness(monkeypatch, tmp_path)   # $0.50/line
    monkeypatch.setattr(planner, "_new_thread_id", lambda prefix: "episode_ep2_bp1")
    monkeypatch.setattr(engine, "SCENES", {"s1": {"name": "One", "prompt": "p", "mood": "m"}})
    monkeypatch.setattr(engine, "EPISODES", {"ep2": {
        "name": "Ep2", "description": "d", "intensity": "low",
        "scenes": [{"scene": "s1", "lines": 4, "plot": ""}]}})
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.5)  # dies after line 3 of 4

    completed = planner.run_episode_planned("ep2", CAST_IDS,
                                            engine.RunOptions(intensity="low"))
    assert len(completed) == 1
    assert completed[0]["partial"] is True
    assert len(completed[0]["conversation"]) == 3
    ep_jsons = [p for p in (tmp_path / "scripts").glob("test_show_episode_ep2_*.json")
                if not p.name.endswith("_eval.json")]
    paused_save = json.loads(ep_jsons[0].read_text())
    assert paused_save["scenes"][-1]["partial"] is True

    fresh_process(tmp_path, "resume1")
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 10.0)
    all_scenes = planner.run_episode_planned("ep2", CAST_IDS,
                                             engine.RunOptions(intensity="low"),
                                             thread_id="episode_ep2_bp1")
    assert len(all_scenes) == 1
    assert len(all_scenes[0]["conversation"]) == 4
    assert "partial" not in all_scenes[0]
    assert calls["lines"] == 4  # lines 1-3 never re-billed


def test_a_cap_zero_planner_episode_declines_before_the_banner(
        monkeypatch, tmp_path, capsys):
    pytest.importorskip("langgraph")  # planner extra — skip on a bare clone
    planner_harness(monkeypatch, tmp_path)
    _episode_world(monkeypatch)
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 0)
    assert planner.run_episode_planned("ep", CAST_IDS,
                                       engine.RunOptions(intensity="low")) == []
    assert list((tmp_path / "scripts").glob("test_show_episode_*")) == []
    assert "--thread" not in capsys.readouterr().out


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
    # Mock the happy-path preflight too: this test is about CLI dispatch, and
    # must not depend on langgraph being installed (bare-clone runs).
    monkeypatch.setattr(planner, "preflight", lambda: [])

    _cli_main(monkeypatch, ["--scene", "test", "--planner", "--thread", "t9"])
    assert captured == {"scene": "test", "thread": "t9"}

    monkeypatch.setattr(planner, "preflight", lambda: ["langgraph not installed — x"])
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "test", "--planner"])
    assert "langgraph not installed" in capsys.readouterr().out
