"""Unit tests for the guardrail + audit pieces: content-rating gate, config
manifest, and the run-log audit header."""

import json

from exec_producer import config, engine, telemetry


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
