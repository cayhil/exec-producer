"""Unit tests for telemetry: run log records, summary math, counters."""

import json

from exec_producer import telemetry


def test_record_appends_jsonl_line(tmp_path):
    rl = telemetry.RunLog(log_dir=tmp_path)
    rl.record(caller="claude", model="m1", total_tokens=100, cost_usd=0.01)
    rl.record(caller="director", model="m2", total_tokens=50, cost_usd=0.002)

    lines = rl.path.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["caller"] == "claude"
    assert first["total_tokens"] == 100
    assert "ts" in first
    assert len(rl.records) == 2


def test_summary_aggregates_by_caller(tmp_path):
    rl = telemetry.RunLog(log_dir=tmp_path)
    rl.record(caller="claude", model="m", total_tokens=100, cost_usd=0.01)
    rl.record(caller="claude", model="m", total_tokens=200, cost_usd=0.02)
    rl.record(caller="director", model="d", total_tokens=50, cost_usd=0.005)

    s = rl.summary()
    assert s["calls"] == 3
    assert s["total_tokens"] == 350
    assert s["total_cost_usd"] == 0.035
    assert s["by_caller"]["claude"] == {"calls": 2, "tokens": 300, "cost_usd": 0.03}
    assert s["by_caller"]["director"]["calls"] == 1
    assert s["calls_with_unknown_cost"] == 0


def test_summary_handles_unknown_cost_and_failures(tmp_path):
    # record_failure targets the active run log, so use it (not a fresh one)
    rl = telemetry.get_run_log()
    rl.record(caller="grok", model="m", total_tokens=10, cost_usd=None)
    rl.record(caller="grok", model="m", error="Timeout")
    telemetry.record_failure("m")

    s = rl.summary()
    assert s["calls"] == 2  # failed calls counted in call total
    assert s["calls_with_unknown_cost"] == 1
    assert s["total_cost_usd"] == 0
    # failed call contributes no tokens/cost to by_caller
    assert s["by_caller"]["grok"]["calls"] == 1
    assert s["terminal_failures"] == {"m": 1}


def test_counters(capsys):
    telemetry.record_failure("model-a")
    telemetry.record_failure("model-a")
    telemetry.record_fallback("model-a", "model-b")
    assert telemetry.get_run_log().failures["model-a"] == 2
    assert telemetry.get_run_log().fallbacks["model-a"] == 1


def test_print_run_summary_smoke(capsys):
    telemetry.get_run_log().record(caller="claude", model="m", total_tokens=10, cost_usd=0.01)
    telemetry.print_run_summary()
    out = capsys.readouterr().out
    assert "Run Summary" in out
    assert "claude" in out
    assert "run log:" in out
