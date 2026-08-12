"""How the spend cap behaves at the edges a user actually hits: the
interactive shell, a planner run that can't be resumed, and the empty scene
an episode produces when the cap trips at a scene boundary."""

import pytest

from exec_producer import cli, engine, telemetry
from tests.test_engine import FAKE_CAST, SCENE

SOLO_CAST = {"alpha": FAKE_CAST["alpha"]}


@pytest.fixture
def shell(monkeypatch, tmp_path):
    """The interactive demo shell, wired to a stub cast and no real calls."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "CAST", SOLO_CAST)
    monkeypatch.setattr(cli, "SCENES", {"test": {"name": "T", "prompt": "p", "mood": "m"}})
    monkeypatch.setattr(cli, "SHOW_NAME", "TEST SHOW")
    monkeypatch.setattr(cli, "check_api_keys", lambda ids: (["alpha"], []))
    monkeypatch.setattr(cli, "director_key_or_die", lambda: None)
    scenes_run = []
    monkeypatch.setattr(cli, "run_scene",
                        lambda scene_id, cast_ids, opts: scenes_run.append(scene_id))
    return scenes_run


def answer(monkeypatch, *responses):
    replies = iter(responses)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))


def spend(amount):
    telemetry.get_run_log().record(caller="director", model="m", cost_usd=amount)


# ── interactive shell: one budget per session ────────────

def test_the_shell_declines_a_second_scene_once_the_session_budget_is_spent(
        shell, monkeypatch, capsys):
    # The shell runs many scenes in ONE process, and the budget is per
    # process. Before this, scene 2 onward printed a banner and produced
    # nothing — silently, and saving an empty artifact each time.
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)
    answer(monkeypatch, "all", "1", "2", "low", "n", "y")

    def bill_the_scene(scene_id, cast_ids, opts):
        shell.append(scene_id)
        spend(1.5)

    monkeypatch.setattr(cli, "run_scene", bill_the_scene)
    cli.interactive_mode()

    assert shell == ["test"]  # asked for a second, never started it
    out = capsys.readouterr().out
    assert "budget" in out.lower()
    assert "world.json" in out  # tells them how to raise it


def test_the_shell_keeps_going_while_the_budget_holds(shell, monkeypatch):
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 100.0)
    answer(monkeypatch, "all", "1", "2", "low", "n", "y", "1", "n")
    cli.interactive_mode()
    assert shell == ["test", "test"]


def test_the_shell_refuses_to_start_at_all_with_a_zero_budget(
        shell, monkeypatch, capsys):
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 0)
    answer(monkeypatch, "all", "1", "2", "low", "n")
    cli.interactive_mode()
    assert shell == []
    assert "budget" in capsys.readouterr().out.lower()


def test_no_budget_configured_leaves_the_shell_alone(shell, monkeypatch):
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", None)
    answer(monkeypatch, "all", "1", "2", "low", "n", "y", "1", "n")
    cli.interactive_mode()
    assert shell == ["test", "test"]


# ── planner + budget: warn before the surprise ───────────

def test_a_planned_run_warns_that_a_budget_stop_cannot_be_resumed(
        monkeypatch, capsys):
    # A budget stop ends a planner run for good: the graph reaches END, so
    # the checkpoint records it as finished and --thread refuses to resume.
    # The user re-pays for turns they already bought. Until that's fixed,
    # they should at least hear about it first.
    monkeypatch.setattr(cli, "MAX_RUN_USD", 2.5)
    cli.warn_if_budget_blocks_resume(planner=True)
    out = capsys.readouterr().out
    assert "--thread" in out and "resume" in out.lower()


def test_no_warning_without_a_budget(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_RUN_USD", None)
    cli.warn_if_budget_blocks_resume(planner=True)
    assert capsys.readouterr().out == ""


def test_no_warning_for_a_classic_run(monkeypatch, capsys):
    # The classic loop has no checkpoints to lose.
    monkeypatch.setattr(cli, "MAX_RUN_USD", 2.5)
    cli.warn_if_budget_blocks_resume(planner=False)
    assert capsys.readouterr().out == ""


def test_a_zero_budget_still_warns(monkeypatch, capsys):
    # 0 is a configured cap, not an absent one. A truthiness test here
    # silently skipped the warning for the strictest setting there is.
    monkeypatch.setattr(cli, "MAX_RUN_USD", 0)
    cli.warn_if_budget_blocks_resume(planner=True)
    assert "--thread" in capsys.readouterr().out


def test_a_zero_budget_is_shown_in_the_doctor_report(monkeypatch):
    from exec_producer import doctor
    monkeypatch.setattr(doctor, "MAX_RUN_USD", 0)
    details = " ".join(next(c for c in doctor.run_checks() if c.id == "show").details)
    assert "max_run_usd" in details


def test_a_planned_turn_is_not_billed_when_the_budget_is_already_gone(monkeypatch):
    # The graph's first routing check happens AFTER speak, so an exhausted
    # budget still paid for one director call plus one character call. The
    # classic loop gates at the top of its turn; this must match.
    from exec_producer import planner

    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 0)
    monkeypatch.setattr(planner.engine, "get_director_choice",
                        lambda *a, **k: pytest.fail("billed the director past the cap"))
    state = planner.direct_turn({
        "scene_queue": [{"scene_id": "test", "lines": 3}], "scene_index": 0,
        "turn_index": 0, "scene_cast": ["alpha"], "conversation": [],
        "last_speaker": None, "intensity": "low", "plan": [],
    })
    assert state.get("director_note") in (None, {})


# ── episodes: no empty scenes ────────────────────────────

def test_an_episode_scene_the_budget_killed_is_not_saved_or_judged(
        monkeypatch, capsys):
    # The cap can trip during a scene's casting call, after the per-scene
    # gate but before its first line. That left a zero-line scene in the
    # episode — and at wrap the judge billed a call to score an empty script.
    from types import SimpleNamespace

    monkeypatch.setattr(engine, "CAST", FAKE_CAST)
    monkeypatch.setattr(engine, "SHOW_NAME", "TEST SHOW")
    monkeypatch.setattr(engine, "SHOW_SLUG", "test_show")
    monkeypatch.setattr(engine, "MAX_INTENSITY", None)
    monkeypatch.setattr(engine.time, "sleep", lambda _s: None)
    monkeypatch.setattr(engine, "EPISODES", {
        "ep": {"name": "Ep", "description": "d", "intensity": "low",
               "scenes": [{"scene": "test", "lines": 1}, {"scene": "test", "lines": 1}]}})
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(engine, "build_transition_text", lambda *a, **k: "next")
    monkeypatch.setattr(engine, "CharacterSession",
                        lambda cast_id, *a, **k: SimpleNamespace(
                            cast_id=cast_id, inject_context=lambda _: None,
                            trim_if_needed=lambda *_a, **_k: None))
    monkeypatch.setattr(engine, "get_director_choice",
                        lambda *a, **k: {"next_speaker": "alpha", "direction": "OPEN"})
    monkeypatch.setattr(engine, "inject_scene_update", lambda *a, **k: None)
    monkeypatch.setattr(engine, "generate_line_session", lambda *a, **k: "a line")

    # Scene 1 runs; the casting call for scene 2 blows the budget, so scene
    # 2 is gated at its first turn and produces nothing.
    def cast_and_bill(*a, **k):
        spend(5.0)
        return ["alpha", "beta"]

    monkeypatch.setattr(engine, "get_scene_cast", cast_and_bill)
    judged = []
    monkeypatch.setattr(engine, "evaluate_scene",
                        lambda *a, **k: judged.append(1) or None)
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)

    conversations = engine.run_episode(
        "ep", ["alpha", "beta"],
        engine.RunOptions(save=False, evaluate=True, intensity="low"))

    assert all(c["conversation"] for c in conversations), "kept an empty scene"
    assert len(judged) == 1, "billed the judge for a scene with no dialogue"
