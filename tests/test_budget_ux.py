"""How the spend cap behaves at the edges a user actually hits: the
interactive shell, a planner run that pauses, and the empty scene
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


# ── cumulative spend: resumes share one budget ───────────

def test_seeded_prior_spend_counts_toward_the_cap(monkeypatch):
    # Each resume is a fresh process with an empty RunLog. Without seeding,
    # N resumes of one thread could each spend the full cap; the checkpoint
    # carries cumulative spend and the gate must count it. Binary-exact
    # amounts (0.75 + 0.125 + 0.125 == 1.0) so the >= trip point is tested
    # by construction, not by rounding luck.
    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 1.0)
    telemetry.seed_prior_spend(0.75)
    spend(0.125)
    assert not telemetry.spend_exhausted()
    spend(0.125)
    assert telemetry.spend_exhausted()


def test_seeding_zero_is_a_no_op(capsys):
    # Pre-feature checkpoints have no spent_usd and seed 0 — no banner, no
    # budget effect (same exposure as before the feature; the resume path
    # prints its own "fresh budget" notice, tested in test_planner.py).
    telemetry.seed_prior_spend(0.0)
    assert telemetry.get_run_log().prior_spend_usd == 0.0
    assert capsys.readouterr().out == ""


def test_seeding_never_lowers_the_floor():
    # A second, smaller (or negative) seed must not loosen the budget.
    telemetry.seed_prior_spend(1.5)
    telemetry.seed_prior_spend(0.5)
    telemetry.seed_prior_spend(-3.0)
    assert telemetry.get_run_log().prior_spend_usd == 1.5


def test_seeding_announces_the_carried_spend(capsys):
    telemetry.seed_prior_spend(1.25)
    assert "1.25" in capsys.readouterr().out


def test_the_run_summary_separates_carried_from_this_process():
    # total_cost_usd stays the audit record of THIS process's JSONL;
    # the carried and cumulative numbers get their own keys so a resumed
    # scene's true cost is recoverable from the artifact.
    telemetry.seed_prior_spend(5.0)
    spend(0.5)
    s = telemetry.get_run_log().summary()
    assert s["total_cost_usd"] == 0.5
    assert s["prior_spend_usd"] == 5.0
    assert s["cumulative_cost_usd"] == 5.5


# ── planner + budget: pause behavior lives in test_planner.py ──

def test_a_zero_budget_is_shown_in_the_doctor_report(monkeypatch):
    from exec_producer import doctor
    monkeypatch.setattr(doctor, "MAX_RUN_USD", 0)
    details = " ".join(next(c for c in doctor.run_checks() if c.id == "show").details)
    assert "max_run_usd" in details


def test_a_planned_turn_pauses_instead_of_billing_when_the_budget_is_gone(monkeypatch):
    # direct_turn used to return {} here, which let `speak` run next — with
    # a stale note mid-episode, or a KeyError at a scene's first turn
    # (current_note never written). Raising pauses before either can happen.
    from exec_producer import planner
    from exec_producer.errors import BudgetPause

    monkeypatch.setattr(telemetry, "MAX_RUN_USD", 0)
    monkeypatch.setattr(planner.engine, "get_director_choice",
                        lambda *a, **k: pytest.fail("billed the director past the cap"))
    with pytest.raises(BudgetPause):
        planner.direct_turn({
            "scene_queue": [{"scene_id": "test", "lines": 3}], "scene_index": 0,
            "turn_index": 0, "scene_cast": ["alpha"], "conversation": [],
            "last_speaker": None, "intensity": "low", "plan": [],
        })


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
