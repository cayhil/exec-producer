"""`exec-producer doctor` — standalone health checks over the existing
per-stage preflights. No network, no keys."""

import json
import sys
from pathlib import Path

import pytest

from exec_producer import doctor
from tests.test_engine import FAKE_CAST

FAKE_ENV_KEYS = {"openai": "OPENAI_API_KEY", "xai": "XAI_API_KEY"}


@pytest.fixture(autouse=True)
def fake_show(monkeypatch):
    monkeypatch.setattr(doctor, "CAST", FAKE_CAST)
    monkeypatch.setattr(doctor, "PROVIDER_ENV_KEYS", FAKE_ENV_KEYS)
    monkeypatch.setattr(doctor, "DIRECTOR_MODEL", "xai/grok-4.20")
    monkeypatch.setattr(doctor, "JUDGE_MODEL", "m-alpha")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-value")
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key-value")
    # The real stage preflights answer questions about THIS machine (is
    # ffmpeg installed? is the vlog extra synced?), so leaving them live
    # would make every verdict assertion depend on the developer's box.
    # Tests that care about stages install their own; one test below
    # exercises the real registry.
    monkeypatch.setattr(doctor, "STAGE_PREFLIGHTS", {})


def by_id(checks, check_id):
    return next(c for c in checks if c.id == check_id)


# ── core checks ──────────────────────────────────────────

def guardrail_lines(check):
    """The guardrail details only. `details[0]` is the show's path, which
    can contain any word at all — asserting against the whole blob made
    this suite fail for anyone whose checkout lived under a directory
    named, say, `capstone/`."""
    return [d for d in check.details if not d.startswith("loaded from ")]


def test_the_loaded_show_is_reported_with_its_contents_and_path():
    check = by_id(doctor.run_checks(), "show")
    assert check.status == "pass"
    # `status=PASS` is hardcoded, so asserting only on it proves nothing —
    # the summary is what tells the user WHICH show is about to run.
    assert "3 cast members" in check.summary
    assert any(d.startswith("loaded from ") for d in check.details)


def test_active_guardrails_are_reported_so_a_capped_run_is_never_a_surprise(monkeypatch):
    monkeypatch.setattr(doctor, "MAX_RUN_USD", 2.5)
    monkeypatch.setattr(doctor, "MAX_INTENSITY", "low")
    details = " ".join(guardrail_lines(by_id(doctor.run_checks(), "show")))
    assert "2.50" in details and "low" in details


def test_absent_guardrails_add_no_noise(monkeypatch):
    monkeypatch.setattr(doctor, "MAX_RUN_USD", None)
    monkeypatch.setattr(doctor, "MAX_INTENSITY", None)
    assert guardrail_lines(by_id(doctor.run_checks(), "show")) == []


def test_missing_director_key_is_a_core_failure(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY")
    check = by_id(doctor.run_checks(), "director-key")
    assert check.status == "fail"
    assert "XAI_API_KEY" in check.summary + " ".join(check.details)


def test_present_director_key_passes():
    assert by_id(doctor.run_checks(), "director-key").status == "pass"


def test_cast_key_check_names_each_member_that_cannot_run(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    check = by_id(doctor.run_checks(), "cast-keys")
    assert check.status == "fail"
    joined = " ".join(check.details)
    assert "alpha" in joined and "beta" in joined and "gamma" in joined


def test_one_usable_cast_key_is_a_warning_because_solo_runs_work(monkeypatch):
    # `--solo --cast alpha` runs fine with a single key, so calling the
    # install broken would make doctor disagree with the CLI it describes.
    monkeypatch.setattr(doctor, "PROVIDER_ENV_KEYS",
                        {"openai": "OPENAI_API_KEY", "xai": "XAI_API_KEY"})
    monkeypatch.setattr(doctor, "CAST", {
        "alpha": {**FAKE_CAST["alpha"], "provider": "xai"},
        "beta": FAKE_CAST["beta"],
    })
    monkeypatch.delenv("OPENAI_API_KEY")
    check = by_id(doctor.run_checks(), "cast-keys")
    assert check.status == "warn"
    assert "--solo" in check.summary + check.remedy


def test_a_solo_show_with_its_one_key_passes(monkeypatch):
    monkeypatch.setattr(doctor, "CAST", {"alpha": FAKE_CAST["alpha"]})
    assert by_id(doctor.run_checks(), "cast-keys").status == "pass"


def test_a_solo_show_without_its_key_fails(monkeypatch):
    monkeypatch.setattr(doctor, "CAST", {"alpha": FAKE_CAST["alpha"]})
    monkeypatch.delenv("OPENAI_API_KEY")
    assert by_id(doctor.run_checks(), "cast-keys").status == "fail"


def test_a_show_with_enough_cast_keys_passes():
    assert by_id(doctor.run_checks(), "cast-keys").status == "pass"


def test_judge_key_is_only_a_warning_since_eval_is_opt_in(monkeypatch):
    # --eval is opt-in; a missing judge key must not make a healthy install
    # look broken.
    monkeypatch.delenv("OPENAI_API_KEY")
    check = by_id(doctor.run_checks(), "judge-key")
    assert check.status == "warn"


# ── optional stages ──────────────────────────────────────

def test_optional_stage_problems_are_warnings_not_failures(monkeypatch):
    monkeypatch.setattr(doctor, "STAGE_PREFLIGHTS", {
        "audio": lambda cast_ids: ["ELEVENLABS_API_KEY is not set in .env"],
    })
    checks = doctor.run_checks()
    stage = by_id(checks, "stage:audio")
    assert stage.status == "warn"
    assert "ELEVENLABS_API_KEY is not set in .env" in stage.details
    # An un-installed optional extra is not a broken install.
    assert doctor.exit_code(checks) == 0


def test_a_ready_stage_passes(monkeypatch):
    monkeypatch.setattr(doctor, "STAGE_PREFLIGHTS", {"audio": lambda cast_ids: []})
    assert by_id(doctor.run_checks(), "stage:audio").status == "pass"


def test_every_optional_render_path_has_a_registered_stage_check(monkeypatch):
    # Guards the wiring itself: a new render mode that forgets to register
    # here would silently be undiagnosable. Runs the REAL preflights, so it
    # asserts only on shape — the verdict depends on the host.
    monkeypatch.undo()
    assert set(doctor.STAGE_PREFLIGHTS) == {"audio", "video", "veo", "vlog", "planner"}
    for name, preflight in doctor.STAGE_PREFLIGHTS.items():
        check = doctor.check_stage(name, preflight)
        # Not `in (PASS, WARN, ERROR)` — check_stage has exactly those three
        # returns, so that would be a tautology. What actually matters is
        # that an optional stage can never hard-fail the command.
        assert check.status != doctor.FAIL, name
        assert check.id == f"stage:{name}"
        assert check.summary, name


def test_advisories_a_preflight_prints_are_folded_into_its_check(monkeypatch):
    # vlog's preflight prints advisories (missing reference images) rather
    # than returning them as problems. Left alone they'd land above the
    # report header, detached from the check that produced them.
    def chatty(cast_ids):
        print("  ⚠️  No reference_images configured for marlow")
        return []

    monkeypatch.setattr(doctor, "STAGE_PREFLIGHTS", {"vlog": chatty})
    check = by_id(doctor.run_checks(), "stage:vlog")
    assert check.status == doctor.PASS
    assert any("reference_images" in d for d in check.details)


def test_a_stage_preflight_that_raises_does_not_abort_the_whole_report(monkeypatch):
    def boom(cast_ids):
        raise RuntimeError("ffprobe exploded")

    monkeypatch.setattr(doctor, "STAGE_PREFLIGHTS", {"video": boom})
    checks = doctor.run_checks()
    # The remaining checks still ran...
    assert by_id(checks, "show").status == "pass"
    # ...and the broken check is surfaced, not swallowed.
    assert by_id(checks, "stage:video").status == "error"
    assert doctor.exit_code(checks) == 2


# ── verdicts and output ──────────────────────────────────

def test_exit_code_is_zero_when_everything_passes():
    assert doctor.exit_code(doctor.run_checks()) == 0


def test_exit_code_is_one_when_a_core_check_fails(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY")
    assert doctor.exit_code(doctor.run_checks()) == 1


def test_a_definite_core_failure_outranks_an_unrunnable_optional_check(monkeypatch):
    # Exit 2 means "couldn't determine the answer". A missing director key
    # IS a determined answer, and it's the more actionable one — a broken
    # visuals.json must not downgrade it to a shrug.
    monkeypatch.delenv("XAI_API_KEY")
    monkeypatch.setattr(doctor, "STAGE_PREFLIGHTS", {
        "video": lambda cast_ids: (_ for _ in ()).throw(RuntimeError("boom")),
    })
    assert doctor.exit_code(doctor.run_checks()) == 1


def test_an_unrunnable_check_alone_exits_two(monkeypatch):
    monkeypatch.setattr(doctor, "STAGE_PREFLIGHTS", {
        "video": lambda cast_ids: (_ for _ in ()).throw(RuntimeError("boom")),
    })
    assert doctor.exit_code(doctor.run_checks()) == 2


def test_an_optional_stage_exception_does_not_report_the_install_as_failed(monkeypatch):
    # "A missing optional extra is not a broken install" has to hold for a
    # stage that BLEW UP too, not just one that reported problems.
    monkeypatch.setattr(doctor, "STAGE_PREFLIGHTS", {
        "video": lambda cast_ids: (_ for _ in ()).throw(RuntimeError("boom")),
    })
    assert doctor.overall(doctor.run_checks()) == "degraded"


def test_a_core_check_that_raises_becomes_a_finding_not_a_traceback(monkeypatch):
    # run_checks() promises it never raises. This is the command someone
    # runs BECAUSE their install is broken.
    monkeypatch.setattr(doctor, "MAX_RUN_USD", "2.50")  # a string in world.json
    checks = doctor.run_checks()
    assert by_id(checks, "show").status == "error"
    assert doctor.exit_code(checks) == 2


def test_every_core_check_is_individually_isolated(monkeypatch):
    monkeypatch.setattr(doctor, "JUDGE_MODEL", None)
    checks = doctor.run_checks()
    # The judge check blew up, but the others still answered.
    assert by_id(checks, "judge-key").status == "error"
    assert by_id(checks, "show").status == "pass"
    assert by_id(checks, "director-key").status == "pass"


def test_overall_verdict_tracks_the_worst_check(monkeypatch):
    assert doctor.overall(doctor.run_checks()) == "ok"
    monkeypatch.setattr(doctor, "STAGE_PREFLIGHTS", {"audio": lambda cast_ids: ["nope"]})
    assert doctor.overall(doctor.run_checks()) == "degraded"
    monkeypatch.delenv("XAI_API_KEY")
    assert doctor.overall(doctor.run_checks()) == "failed"


def test_json_report_carries_a_schema_version_and_verdict():
    report = doctor.to_json(doctor.run_checks())
    assert report["schema_version"] == "1"
    assert report["overall"] == "ok"
    assert {"id", "status", "summary"} <= set(report["checks"][0])


def test_human_output_shows_the_remedy_for_a_failing_check(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY")
    out = doctor.format_human(doctor.run_checks())
    assert "XAI_API_KEY" in out
    assert ".env" in out  # tells the user where to put it


# ── CLI wiring ───────────────────────────────────────────

def _cli_main(monkeypatch, argv):
    from exec_producer import cli
    monkeypatch.setattr(sys, "argv", ["exec-producer"] + argv)
    cli.main()


def test_cli_doctor_prints_a_report_and_exits_zero_when_healthy(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _cli_main(monkeypatch, ["doctor"])
    assert exit_info.value.code == 0
    assert "doctor" in capsys.readouterr().out


def test_cli_doctor_exits_one_when_a_core_check_fails(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY")
    with pytest.raises(SystemExit) as exit_info:
        _cli_main(monkeypatch, ["doctor"])
    assert exit_info.value.code == 1


def test_cli_doctor_json_emits_a_machine_readable_report(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "1"
    assert report["overall"] == "ok"


def test_cli_doctor_needs_no_api_keys_to_run(monkeypatch):
    # The command exists for the broken-install case; demanding a working
    # install to run it would defeat the point.
    from exec_producer import cli
    monkeypatch.setattr(cli, "director_key_or_die",
                        lambda: (_ for _ in ()).throw(AssertionError("no key gate")))
    monkeypatch.delenv("XAI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["doctor"])


def test_the_data_dir_fallback_notice_does_not_corrupt_stdout(capsys):
    # config prints this at IMPORT time, before any command runs, so it
    # landed on the --json payload's first line. data/ is gitignored, which
    # makes this the state of every fresh clone of the public repo.
    from exec_producer import config
    config.announce_data_dir_fallback(Path("examples/sunset_terrace"))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "starter show" in captured.err


def test_cli_doctor_json_stays_parseable_when_a_preflight_prints(monkeypatch, capsys):
    # `exec-producer doctor --json | jq` must not choke on stray preflight
    # chatter interleaved with the payload.
    def chatty(cast_ids):
        print("⚠️  advisory noise from a preflight")
        return []

    monkeypatch.setattr(doctor, "STAGE_PREFLIGHTS", {"vlog": chatty})
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["overall"] == "ok"


def test_cli_scene_named_doctor_is_not_sugar(monkeypatch, capsys):
    # "doctor" as the VALUE of a value-taking flag stays a value — same
    # guarantee the render sugar makes.
    from exec_producer import cli
    monkeypatch.setattr(cli, "SCENES", {"other": {"name": "O", "prompt": "p", "mood": "m"}})
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "doctor"])
    assert "Unknown scene: doctor" in capsys.readouterr().out


def test_cli_rejects_json_without_doctor(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _cli_main(monkeypatch, ["--json", "--list"])
    assert exit_info.value.code == 1
    assert "--json" in capsys.readouterr().out
