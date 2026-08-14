"""--from-script ingestion: boundary validation (hard, all problems at
once, before any spend) and the render-only pipeline. All LLM/media calls
mocked."""

import json
import sys

import pytest

from exec_producer import engine, render
from exec_producer.errors import ConfigError
from tests.test_engine import FAKE_CAST, SCENE

VALID = {
    "scene": {"name": "Test Kitchen", "prompt": "the kitchen", "mood": "tense"},
    "intensity": "low",
    "conversation": [
        {"id": "alpha", "text": '[leans in] "First line."', "direction": "OPEN"},
        {"id": "beta", "text": '"Second line."'},
    ],
    "director_notes": [{"stage_direction": "the kettle screams"}, {}],
}


@pytest.fixture(autouse=True)
def fake_world(monkeypatch):
    monkeypatch.setattr(engine, "CAST", FAKE_CAST)
    monkeypatch.setattr(engine, "SHOW_NAME", "TEST SHOW")
    monkeypatch.setattr(engine, "SHOW_SLUG", "test_show")
    monkeypatch.setattr(engine, "MAX_INTENSITY", None)


def write_scene(tmp_path, payload, name="my_scene.json"):
    p = tmp_path / name
    p.write_text(json.dumps(payload) if isinstance(payload, dict) else payload)
    return p


# ── validation ───────────────────────────────────────────

def test_load_valid_scene_normalizes(tmp_path):
    data = render.load_external_scene(write_scene(tmp_path, VALID))
    assert data["scene_id"] == "my_scene"
    assert data["cast_ids"] == ["alpha", "beta"]
    assert data["intensity"] == "low"
    # display fields filled in from the loaded show's cast
    assert data["conversation"][0]["name"] == "Alpha"
    assert data["conversation"][0]["emoji"] == "🅰️"
    assert data["conversation"][1]["tagline"] == "The B"
    assert data["conversation"][0]["direction"] == "OPEN"


def test_load_respects_scene_id_and_extra_fields(tmp_path):
    payload = {**VALID, "scene_id": "Kitchen Fight!",
               "conversation": [{"id": "alpha", "text": "hi", "delivery": "voiceover"}]}
    data = render.load_external_scene(write_scene(tmp_path, payload))
    assert data["scene_id"] == "kitchen_fight"  # sanitized for the file prefix
    # unknown per-line fields pass through (vlog delivery markers, etc.)
    assert data["conversation"][0]["delivery"] == "voiceover"


def test_load_lists_every_problem_at_once(tmp_path):
    payload = {"scene": {}, "conversation": [{"id": "alpha"}, {"text": "no id"}],
               "intensity": "nuclear", "director_notes": "not a list"}
    with pytest.raises(ConfigError) as e:
        render.load_external_scene(write_scene(tmp_path, payload))
    msg = str(e.value)
    assert "'scene' must be" in msg
    assert "missing 'text'" in msg
    assert "missing 'id'" in msg
    assert "'intensity' must be" in msg
    assert "'director_notes' must be" in msg


def test_load_unknown_cast_id_names_the_fix(tmp_path):
    payload = {**VALID, "conversation": [{"id": "zeta", "text": "hi"}]}
    with pytest.raises(ConfigError, match="unknown cast id.*zeta.*--data-dir"):
        render.load_external_scene(write_scene(tmp_path, payload))


def test_load_rejects_episode_file(tmp_path):
    with pytest.raises(ConfigError, match="EPISODE file"):
        render.load_external_scene(write_scene(tmp_path, {"scenes": [{}]}))


def test_load_rejects_bad_json_and_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not valid JSON"):
        render.load_external_scene(write_scene(tmp_path, "{nope"))
    with pytest.raises(ConfigError, match="not a file"):
        render.load_external_scene(tmp_path / "ghost.json")


def test_load_applies_content_rating_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "MAX_INTENSITY", "low")
    payload = {**VALID, "intensity": "maximum"}
    data = render.load_external_scene(write_scene(tmp_path, payload))
    assert data["intensity"] == "low"


# ── render pipeline ──────────────────────────────────────

def render_scene(tmp_path, monkeypatch, payload=VALID, **opt_kw):
    monkeypatch.chdir(tmp_path)
    data = render.load_external_scene(write_scene(tmp_path, payload))
    scorecard = render.render_external_scene(data, engine.RunOptions(**opt_kw))
    return data, scorecard


def test_render_writes_outputs_with_external_audit(tmp_path, monkeypatch):
    render_scene(tmp_path, monkeypatch)
    txts = list((tmp_path / "scripts").glob("test_show_my_scene_*.txt"))
    jsons = list((tmp_path / "scripts").glob("test_show_my_scene_*.json"))
    assert len(txts) == 1 and len(jsons) == 1
    assert "First line." in txts[0].read_text()
    saved = json.loads(jsons[0].read_text())
    assert saved["audit"]["generator"] == "external"
    assert saved["audit"]["generation_cost"] == "unknown"
    assert saved["audit"]["config_files"] is not None
    assert [e["id"] for e in saved["conversation"]] == ["alpha", "beta"]
    assert "telemetry" in saved


def test_a_partial_scene_renders_with_its_marker_preserved(tmp_path, monkeypatch, capsys):
    # A budget-paused save is a valid, renderable scene — but silently
    # dropping the marker on re-save would launder a half-finished script
    # into a complete-looking one, and the user should hear that the
    # stages they're about to bill run on a partial scene.
    payload = {**VALID, "partial": True}
    data, _ = render_scene(tmp_path, monkeypatch, payload=payload)
    assert data["partial"] is True
    assert "partial" in capsys.readouterr().out.lower()
    jsons = list((tmp_path / "scripts").glob("test_show_my_scene_*.json"))
    saved = json.loads(jsons[0].read_text())
    assert saved["partial"] is True


def test_render_eval_writes_externally_stamped_scorecard(tmp_path, monkeypatch):
    card = {"overall": {"score": 8}}
    judged = []
    monkeypatch.setattr(engine, "evaluate_scene",
                        lambda *a, **k: judged.append(a) or card)
    _, scorecard = render_scene(tmp_path, monkeypatch, evaluate=True)
    assert scorecard == card and len(judged) == 1
    evals = list((tmp_path / "scripts").glob("test_show_my_scene_*_eval.json"))
    assert len(evals) == 1
    saved = json.loads(evals[0].read_text())
    assert saved["scorecard"] == card
    assert saved["audit"]["generator"] == "external"


def test_render_audio_uses_the_file_conversation(tmp_path, monkeypatch):
    from exec_producer import audio as audio_mod
    rendered = {}
    monkeypatch.setattr(
        audio_mod, "render_scene_audio",
        lambda conversation, prefix: rendered.update(conv=conversation)
        or f"{prefix}_audio.mp3")
    render_scene(tmp_path, monkeypatch, audio=True)
    assert [e["id"] for e in rendered["conv"]] == ["alpha", "beta"]


def test_generated_run_audit_has_no_generator_stamp(tmp_path, monkeypatch):
    # The external stamp must never leak into classic generated runs.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    conversation = [{"id": "alpha", "name": "Alpha", "emoji": "🅰️",
                     "tagline": "The A", "text": "hi", "direction": "OPEN"}]
    engine.save_scene_outputs("test", SCENE, conversation, [],
                              ["alpha"], engine.RunOptions(), "low")
    saved = json.loads(next((tmp_path / "scripts").glob("*.json")).read_text())
    assert "generator" not in saved["audit"]


# ── CLI wiring ───────────────────────────────────────────

def _cli_main(monkeypatch, argv):
    from exec_producer import cli
    monkeypatch.setattr(sys, "argv", ["exec-producer"] + argv)
    cli.main()


def test_cli_from_script_needs_no_director_key(tmp_path, monkeypatch):
    from exec_producer import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "director_key_or_die",
                        lambda: (_ for _ in ()).throw(AssertionError("no director key needed")))
    path = write_scene(tmp_path, VALID)
    _cli_main(monkeypatch, ["--from-script", str(path)])
    assert list((tmp_path / "scripts").glob("test_show_my_scene_*.txt"))


def test_cli_from_script_stamps_run_header(tmp_path, monkeypatch):
    # The JSONL run_start header must carry the external stamp, matching the
    # saved outputs' audit block.
    from exec_producer import cli
    monkeypatch.chdir(tmp_path)
    headers = {}
    monkeypatch.setattr(cli, "start_run", lambda header=None: headers.update(header or {}))
    path = write_scene(tmp_path, VALID)
    _cli_main(monkeypatch, ["--from-script", str(path)])
    assert headers["audit"]["generator"] == "external"
    assert headers["audit"]["generation_cost"] == "unknown"


def test_cli_render_positional_sugar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_scene(tmp_path, VALID)
    _cli_main(monkeypatch, ["render", str(path)])
    assert list((tmp_path / "scripts").glob("test_show_my_scene_*.txt"))


def test_cli_render_sugar_after_leading_flags(tmp_path, monkeypatch):
    # `--data-dir X render f.json` is the natural order per the docs — the
    # sugar must work after leading flags, not only as the first token.
    monkeypatch.chdir(tmp_path)
    path = write_scene(tmp_path, VALID)
    _cli_main(monkeypatch, ["--debug", "render", str(path)])
    assert list((tmp_path / "scripts").glob("test_show_my_scene_*.txt"))


def test_cli_scene_named_render_is_not_sugar(monkeypatch, capsys):
    # "render" as a VALUE of a value-taking flag stays a value.
    from exec_producer import cli
    monkeypatch.setattr(cli, "SCENES", {"other": {"name": "O", "prompt": "p", "mood": "m"}})
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--scene", "render"])
    assert "Unknown scene: render" in capsys.readouterr().out


# ── contract versioning ──────────────────────────────────

def test_a_file_without_a_schema_version_is_read_as_the_current_one(tmp_path):
    # Every scene .json written before versioning existed — including the
    # showrunner skill's output — must keep rendering.
    assert "scene_schema" not in VALID
    data = render.load_external_scene(write_scene(tmp_path, VALID))
    assert data["scene_id"] == "my_scene"


def test_the_current_schema_version_is_accepted(tmp_path):
    payload = {**VALID, "scene_schema": render.SCENE_SCHEMA}
    assert render.load_external_scene(write_scene(tmp_path, payload))


def test_a_schema_version_this_engine_cannot_read_is_rejected(tmp_path):
    # The point of the field: an older engine must refuse a newer file
    # rather than silently misread it.
    payload = {**VALID, "scene_schema": "2"}
    with pytest.raises(ConfigError, match="scene_schema"):
        render.load_external_scene(write_scene(tmp_path, payload))


def test_an_unreadable_schema_version_is_reported_with_the_other_problems(tmp_path):
    # The boundary's contract is "every problem at once", not "first only".
    payload = {**VALID, "scene_schema": "2", "intensity": "nonsense"}
    with pytest.raises(ConfigError) as err:
        render.load_external_scene(write_scene(tmp_path, payload))
    assert "scene_schema" in str(err.value) and "intensity" in str(err.value)


def test_a_generated_scene_round_trips_back_through_the_render_path(tmp_path, monkeypatch):
    # The saved shape IS the render input. A version stamp the engine writes
    # but can't read back would break the documented workflow.
    monkeypatch.chdir(tmp_path)
    source = render.load_external_scene(write_scene(tmp_path, VALID))
    engine.save_scene_outputs(
        source["scene_id"], source["scene"], source["conversation"],
        source["director_notes"], source["cast_ids"],
        engine.RunOptions(), source["intensity"])
    saved = next((tmp_path / "scripts").glob("test_show_my_scene_*.json"))
    payload = json.loads(saved.read_text())
    assert payload["scene_schema"] == render.SCENE_SCHEMA
    assert render.load_external_scene(saved)["scene_id"] == "my_scene"


def test_load_rejects_non_string_display_fields(tmp_path):
    payload = {**VALID, "conversation": [
        {"id": "alpha", "text": "hi", "name": None, "direction": 7}]}
    with pytest.raises(ConfigError, match="name, direction must be strings"):
        render.load_external_scene(write_scene(tmp_path, payload))


def test_cli_from_script_rejects_generation_flags(monkeypatch, capsys):
    for extra in (["--scene", "x"], ["--episode", "x"], ["--planner"],
                  ["--cast", "alpha"], ["--solo"], ["--no-save"], ["--director-only"]):
        with pytest.raises(SystemExit):
            _cli_main(monkeypatch, ["--from-script", "f.json"] + extra)
        assert "doesn't compose with --from-script" in capsys.readouterr().out


def test_cli_invalid_file_fails_before_any_preflight(tmp_path, monkeypatch, capsys):
    from exec_producer import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "media_preflight_or_die",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("preflight ran")))
    path = write_scene(tmp_path, {"scene": {}, "conversation": []})
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--from-script", str(path), "--audio"])
    assert "doesn't satisfy the scene contract" in capsys.readouterr().out


def test_cli_produce_on_render_path_needs_director_key(tmp_path, monkeypatch, capsys):
    from exec_producer import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "director_key_or_die",
                        lambda: (print("❌ director key"), sys.exit(1)))
    path = write_scene(tmp_path, VALID)
    with pytest.raises(SystemExit):
        _cli_main(monkeypatch, ["--from-script", str(path), "--produce"])
    assert "director key" in capsys.readouterr().out
