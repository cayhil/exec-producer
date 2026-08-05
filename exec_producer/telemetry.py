"""Run telemetry — structured per-call records, JSONL run log, cost tracking,
run summary, logging configuration, and reliability counters."""

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_LOGGER_NAME = "exec_producer"


def setup_logging(debug=False):
    """Configure the package logger. --debug maps to DEBUG level."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("  [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False


def get_logger(name=_LOGGER_NAME):
    return logging.getLogger(name)


# DECISION: per-call retry counts are deliberately absent from telemetry.
# LiteLLM's num_retries loop is opaque to callers — verified empirically
# (litellm 1.91.3): litellm.failure_callback never fires for retried
# attempts, so exact counts would require reimplementing the retry+fallback
# loop in llm.py just to count attempts. That would duplicate ~30 lines of
# LiteLLM logic and subtly change retry-then-fallback semantics, so we keep
# LiteLLM's built-ins and record terminal outcomes only: success, failure
# (with error type), and fallback engagement.

def record_failure(model):
    get_run_log().failures[model] += 1


def record_fallback(model, responded_model):
    get_run_log().fallbacks[model] += 1
    get_logger().warning(f"fallback engaged: requested {model}, answered by {responded_model}")


class RunLog:
    """One record per LLM call, appended to a JSONL file next to the scripts.

    Records are appended (and flushed) per call, so a crashed run still
    leaves a complete audit trail up to the point of failure.
    """

    def __init__(self, log_dir="scripts", header=None):
        self.started = time.time()
        # pid disambiguates two processes started within the same second —
        # without it they would append interleaved records to one file.
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        self.path = Path(log_dir) / f"run_{run_id}_calls.jsonl"
        self.records = []
        # Optional audit header, written as the file's first line. Deferred to
        # the first record so metadata-only invocations (--list) leave no file.
        self.header = header
        self._header_written = False
        # Terminal API failures (after LiteLLM's internal retries) and
        # fallback activations, keyed by requested model. On the run log —
        # not module globals — so a process that starts several runs doesn't
        # bleed counts between them.
        self.failures = Counter()
        self.fallbacks = Counter()

    def record(self, **fields):
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), **fields}
        self.records.append(rec)
        self.path.parent.mkdir(exist_ok=True)
        with open(self.path, "a") as f:
            if self.header and not self._header_written:
                f.write(json.dumps({"ts": rec["ts"], "event": "run_start", **self.header}) + "\n")
                self._header_written = True
            f.write(json.dumps(rec) + "\n")

    def summary(self):
        ok = [r for r in self.records if not r.get("error")]
        by_caller = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        for r in ok:
            c = by_caller[r.get("caller") or "?"]
            c["calls"] += 1
            c["tokens"] += r.get("total_tokens") or 0
            c["cost_usd"] += r.get("cost_usd") or 0.0
        return {
            "run_log": str(self.path),
            "wall_clock_seconds": round(time.time() - self.started, 1),
            "calls": len(self.records),
            "total_tokens": sum(r.get("total_tokens") or 0 for r in ok),
            "total_cost_usd": round(sum(r["cost_usd"] for r in ok if r.get("cost_usd") is not None), 6),
            "calls_with_unknown_cost": sum(1 for r in ok if r.get("cost_usd") is None),
            "by_caller": {k: {**v, "cost_usd": round(v["cost_usd"], 6)}
                          for k, v in sorted(by_caller.items())},
            "terminal_failures": dict(self.failures),
            "fallbacks": dict(self.fallbacks),
        }


_run_log = None


def start_run(header=None):
    """Create this process's run log (called once from the CLI). The JSONL
    file itself is only created when the first call is recorded."""
    global _run_log
    _run_log = RunLog(header=header)
    return _run_log


def get_run_log():
    global _run_log
    if _run_log is None:
        _run_log = RunLog()
    return _run_log


def print_run_summary():
    s = get_run_log().summary()
    unknown = f" (+{s['calls_with_unknown_cost']} calls with unknown cost)" if s["calls_with_unknown_cost"] else ""
    print("\n  ── Run Summary ──")
    print(f"  {s['calls']} calls | {s['total_tokens']:,} tokens | ${s['total_cost_usd']:.4f}{unknown} | {s['wall_clock_seconds']}s wall clock")
    for caller, v in s["by_caller"].items():
        print(f"    {caller:12s} {v['calls']:3d} calls  {v['tokens']:>8,} tokens  ${v['cost_usd']:.4f}")
    failures = sum(s["terminal_failures"].values())
    fallbacks = sum(s["fallbacks"].values())
    if failures or fallbacks:
        print(f"  reliability: {failures} terminal failures, {fallbacks} fallbacks")
    print(f"  run log: {s['run_log']}")
