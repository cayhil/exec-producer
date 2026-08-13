"""Data-contract and --data-dir tests. The contract failure path runs in a
subprocess (config loads at import time); everything else is direct."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from exec_producer import bootstrap, config


def test_an_abbreviated_data_dir_flag_is_rejected_not_silently_ignored(monkeypatch, capsys):
    # bootstrap matches the exact string "--data-dir" to set the env var,
    # but argparse used to accept abbreviations — so `--data-di X --list`
    # ran the DEFAULT show while looking like it selected X. Wrong show,
    # real spend.
    from exec_producer import cli

    monkeypatch.setattr(sys, "argv",
                        ["exec-producer", "--data-di", "examples/night_signal", "--list"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code != 0
    assert "--data-di" in capsys.readouterr().err


def test_validate_data_dir_accepts_the_loaded_show():
    # Whatever show this test process loaded satisfied the contract.
    assert config.validate_data_dir(config.DATA_DIR) == []


def test_validate_data_dir_lists_every_missing_file(tmp_path):
    (tmp_path / "world.json").write_text("{}")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "director.txt").write_text("x")
    missing = config.validate_data_dir(tmp_path)
    assert "cast.json" in missing and "scenes.json" in missing
    assert "prompts/character_template.txt" in missing
    assert "world.json" not in missing and "prompts/director.txt" not in missing
    # all reported at once, not just the first
    assert len(missing) == len(config.REQUIRED_FILES) - 2


def test_validate_data_dir_nonexistent_path(tmp_path):
    missing = config.validate_data_dir(tmp_path / "nope")
    assert len(missing) == 1 and "not a directory" in missing[0]


def test_bootstrap_applies_data_dir_before_import(monkeypatch):
    # setenv FIRST so monkeypatch records the var as previously-absent and
    # teardown deletes it. (A trailing delenv would instead record the value
    # our direct os.environ writes left behind and resurrect it for the rest
    # of the pytest session.)
    monkeypatch.setenv("EXEC_PRODUCER_DATA_DIR", "placeholder")
    bootstrap.apply_data_dir_arg(["--scene", "x", "--data-dir", "/shows/alpha"])
    assert os.environ["EXEC_PRODUCER_DATA_DIR"] == "/shows/alpha"
    bootstrap.apply_data_dir_arg(["--data-dir=/shows/beta"])
    assert os.environ["EXEC_PRODUCER_DATA_DIR"] == "/shows/beta"


def test_cli_reports_broken_data_dir_legibly(tmp_path):
    """The clone-and-point-at-a-broken-show experience: a clean error listing
    what's missing, exit 1, no traceback."""
    (tmp_path / "world.json").write_text("{}")
    proc = subprocess.run(
        [sys.executable, "-m", "exec_producer", "--data-dir", str(tmp_path), "--list"],
        capture_output=True, text=True, timeout=30,
        cwd=Path(config.__file__).resolve().parent.parent,
        env={**os.environ, "EXEC_PRODUCER_DATA_DIR": ""},
    )
    assert proc.returncode == 1
    out = proc.stdout + proc.stderr
    assert "doesn't satisfy the data contract" in out
    assert "cast.json" in out and "CREATING_A_SHOW.md" in out
    assert "Traceback" not in out


def test_safe_id_validation():
    from exec_producer.errors import ConfigError
    import pytest
    config._require_safe_ids("test", ["fine_one", "also-fine", "ok2"])
    with pytest.raises(ConfigError, match="filename-safe"):
        config._require_safe_ids("test", ["../../../../tmp/pwn"])
    with pytest.raises(ConfigError, match="filename-safe"):
        config._require_safe_ids("test", ["has'quote", "has space"])


def test_committed_starter_shows_satisfy_contract():
    # Both public example shows must always ship the full data contract —
    # night_signal is also the proof that a one-character cast is a valid show.
    root = Path(config.__file__).resolve().parent.parent
    for show in ("sunset_terrace", "night_signal"):
        assert config.validate_data_dir(root / "examples" / show) == [], show


def test_validate_show_data_accepts_the_loaded_show():
    assert config.validate_show_data(config.CAST, config.SCENES, config.EPISODES) == []


def test_validate_show_data_catches_broken_episode_refs():
    # The stakes: a bad ref in a later scene would otherwise KeyError
    # mid-episode — after the earlier scenes were fully billed, and before
    # anything is saved (episodes save only at wrap).
    cast = {"alpha": {"name": "A", "provider": "openai", "model": "m",
                      "emoji": "e", "tagline": "t", "gender": "guy",
                      "personality": "p", "triggers": ["x"]}}
    scenes = {"s1": {"name": "One", "prompt": "p", "mood": "m"}}
    episodes = {"ep": {"name": "Ep", "description": "d",
                       "scenes": [{"scene": "s1", "lines": 2},
                                  {"scene": "typo", "lines": 2},
                                  {"scene": "s1"}]}}
    problems = config.validate_show_data(cast, scenes, episodes)
    assert any("unknown scene 'typo'" in p for p in problems)
    assert any("scene 3" in p and "'lines'" in p for p in problems)
    assert len(problems) == 2  # every problem listed at once


def test_validate_show_data_catches_missing_fields():
    cast = {"Alpha": {"name": "A"}}   # uppercase id + missing contract fields
    scenes = {"s1": {"name": "One"}}  # missing prompt/mood
    episodes = {"ep": {"name": "Ep", "description": "d", "scenes": []}}
    problems = config.validate_show_data(cast, scenes, episodes)
    assert any("lowercase" in p for p in problems)
    assert any("cast.json 'Alpha' is missing" in p and "provider" in p
               for p in problems)
    assert any("scenes.json 's1' is missing" in p and "mood" in p
               for p in problems)
    assert any("non-empty 'scenes'" in p for p in problems)


def test_cli_reports_broken_episode_ref_legibly(tmp_path):
    """A scene typo in episodes.json must die at startup with a clean
    ConfigError — not KeyError deep into a paid episode run."""
    root = Path(config.__file__).resolve().parent.parent
    show = tmp_path / "show"
    shutil.copytree(root / "examples" / "sunset_terrace", show)
    episodes = json.loads((show / "episodes.json").read_text())
    next(iter(episodes.values()))["scenes"][1]["scene"] = "no_such_scene"
    (show / "episodes.json").write_text(json.dumps(episodes))
    proc = subprocess.run(
        [sys.executable, "-m", "exec_producer", "--data-dir", str(show), "--list"],
        capture_output=True, text=True, timeout=30,
        cwd=root, env={**os.environ, "EXEC_PRODUCER_DATA_DIR": ""},
    )
    assert proc.returncode == 1
    out = proc.stdout + proc.stderr
    assert "no_such_scene" in out
    assert "Traceback" not in out


def test_show_identity_defaults():
    # The loaded show defines both keys; absence must fall back, not crash —
    # exercised via the same .get defaults the config module uses.
    assert config.SHOW_NAME and config.SHOW_SLUG
    assert " " not in config.SHOW_SLUG  # slug must be filename-safe
