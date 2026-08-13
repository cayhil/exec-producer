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

from .config import MAX_RUN_USD
from .redact import RedactingFormatter

_LOGGER_NAME = "exec_producer"


def setup_logging(debug=False):
    """Configure the package logger. --debug maps to DEBUG level.

    Every handler installed here formats through RedactingFormatter. Log
    lines carry provider-controlled text and are what users paste into bug
    reports. This is defense in depth, not a fix for a known leak — no
    supported provider currently echoes a rejected key back in its error
    text, but that is their choice to change, not ours."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    # Set the formatter on every call, not just when installing the handler:
    # otherwise a handler attached before this ran (an embedding app, an
    # earlier setup_logging) would keep its un-redacting formatter forever.
    for handler in logger.handlers:
        handler.setFormatter(RedactingFormatter("  [%(levelname)s] %(message)s"))
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    # LiteLLM's own logger prints a full traceback at ERROR for every failed
    # attempt (bad key, quota, timeout). The engine already prints a one-line
    # outcome per call with the reason and records it in the run log — keep
    # the raw spew behind --debug. Nothing else is done to that logger:
    # litellm._logging installs a handler carrying SecretRedactionFilter,
    # which already scrubs its message, args, tracebacks, and extras, so a
    # second redacting handler here would only duplicate every debug line.
    logging.getLogger("LiteLLM").setLevel(logging.DEBUG if debug else logging.CRITICAL)


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

def spend_exhausted():
    """True once this run's recorded spend has reached the configured cap.

    Checked at unit-of-work boundaries — between turns, between scenes,
    per TTS line, per Veo beat, per vlog clip — never inside call_model.
    Two reasons: a call's cost is unknowable until it returns, and stopping
    at a boundary lets the caller save everything it already produced
    instead of losing paid-for work to an exception.

    Two limits, both documented in README and SECURITY:

    - Stop-line, not a ceiling. A classic turn is a director call plus a
      character call, each retried up to 3x on malformed JSON, so a run can
      overshoot by up to four billed calls.
    - Blind to media spend. ElevenLabs bills per plan tier and the Veo API
      reports no per-operation price, so those calls record cost=None,
      which total_cost_usd() counts as zero. The gate still STOPS the media
      stages once text generation trips it, but media alone can exceed the
      cap without ever tripping it — vlog.max_clips and --dry-run are the
      controls there.

    The judge and production-prompt stages deliberately still run (each
    bounded at 3 attempts per scene) so a capped run keeps its artifacts.

    No cap configured (the default) means this is always False.
    """
    # `is None`, not truthiness: a configured cap of 0 means "spend nothing"
    # and must stop the run, not read as "unconfigured".
    if MAX_RUN_USD is None:
        return False
    run_log = get_run_log()
    spent = run_log.total_cost_usd()
    if spent < MAX_RUN_USD:
        return False
    # Announce once per run. The flag lives on the run log, not at module
    # level, for the same reason the reliability counters do: a process that
    # starts several runs must not bleed state between them.
    if not run_log.spend_cap_announced:
        run_log.spend_cap_announced = True
        print(f"\n  💸 Spend cap reached (${spent:.4f} of ${MAX_RUN_USD:.2f} "
              "budget.max_run_usd) — stopping generation, saving what's done")
    return True


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
        # Set once the budget cap stops this run, so the notice prints once
        # rather than at every turn boundary that re-checks it.
        self.spend_cap_announced = False

    def record(self, **fields):
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), **fields}
        self.records.append(rec)
        self.path.parent.mkdir(exist_ok=True)
        with open(self.path, "a") as f:
            if self.header and not self._header_written:
                f.write(json.dumps({"ts": rec["ts"], "event": "run_start", **self.header}) + "\n")
                self._header_written = True
            f.write(json.dumps(rec) + "\n")

    def total_cost_usd(self):
        """Spend recorded so far. Calls whose cost is unknown (TTS bills per
        plan tier, external renders don't generate) count as zero — the only
        honest option, and the reason the cap is a floor, not a guarantee."""
        return sum(r["cost_usd"] for r in self.records
                   if not r.get("error") and r.get("cost_usd") is not None)

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
