"""`exec-producer doctor` — one command that answers "is this machine set
up to run the thing I'm about to pay for?"

The per-stage `preflight()` functions (audio, video, veo, vlog, planner)
already encode every check worth making, but they only run mid-invocation,
for the flags that invocation happened to pass. There was no way to ask
"is --render-mode vlog ready?" without starting a run. This module is that
question, asked standalone: it aggregates the same preflights, adds the
key and show-contract checks no preflight owns, and reports the lot.

Severity is the design decision here. A missing optional extra is not a
broken install — the engine's core path runs fine without Pillow — so an
un-ready stage is a WARNING and exits 0. Only the things every run needs
(the director key, enough cast keys) can fail the command.

Exit codes:
  0  every core check passed
  1  a core check failed — the answer is "no"
  2  a check could not run — the answer is "don't know", which is a
     different thing for the operator to act on
"""

import contextlib
import io
from dataclasses import dataclass, field

from .config import (CAST, CONFIG_MANIFEST, DATA_DIR, DIRECTOR_MODEL, EPISODES,
                     JUDGE_MODEL, MAX_INTENSITY, MAX_RUN_USD,
                     PROVIDER_ENV_KEYS, SCENES, SHOW_NAME)
from .llm import api_key_set, provider_for_model
from .redact import redact

PASS, WARN, FAIL, ERROR = "pass", "warn", "fail", "error"

# The trailing spaces on WARN/ERROR are load-bearing: those emoji render
# one cell narrower than ✅/❌ in most terminals, and without the pad the
# report's columns drift.
_GLYPH = {PASS: "✅", WARN: "⚠️ ", FAIL: "❌", ERROR: "⁉️ "}
# Ascending severity, so `max(statuses, key=_RANK.__getitem__)` is the
# headline. FAIL ranks above ERROR deliberately — see exit_code().
_RANK = {PASS: 0, WARN: 1, ERROR: 2, FAIL: 3}


@dataclass
class Check:
    id: str
    status: str
    summary: str
    details: list = field(default_factory=list)
    remedy: str = ""


# ── optional stages ──────────────────────────────────────
# Every value takes cast_ids and returns a list of problem strings, so the
# stages with a different native signature are adapted here rather than in
# the reporting loop. Imports stay inside the functions: the whole point of
# the media extras being optional is that importing them can fail.

def _audio(cast_ids):
    from .audio import preflight
    return preflight(cast_ids)


def _video(cast_ids):
    from .video import preflight
    return preflight()


def _veo(cast_ids):
    from .video import veo_preflight
    return veo_preflight()


def _vlog(cast_ids):
    from .vlog import preflight
    return preflight(cast_ids)


def _planner(cast_ids):
    from .planner import preflight
    return preflight()


STAGE_PREFLIGHTS = {
    "audio": _audio,
    "video": _video,
    "veo": _veo,
    "vlog": _vlog,
    "planner": _planner,
}

_STAGE_FLAGS = {
    "audio": "--audio",
    "video": "--video",
    "veo": "--video --render-mode veo",
    "vlog": "--video --render-mode vlog",
    "planner": "--planner",
}


# ── core checks ──────────────────────────────────────────

def _count(n, noun):
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _lines(buffer):
    """Non-empty lines a preflight printed, ready to fold into a Check."""
    return [line.strip() for line in buffer.getvalue().splitlines() if line.strip()]


def check_show():
    """The show directory. Reaching this code at all means the contract
    validated (config.py raises at import otherwise), so this check exists
    to tell the user WHICH show they're about to run — the single most
    common confusion with --data-dir and the data/ vs examples/ fallback."""
    details = [f"loaded from {DATA_DIR}"]
    # Guardrails silently change what a run produces (a capped intensity, a
    # run that stops early), so a user diagnosing "why did it stop?" should
    # find the answer here rather than by reading world.json.
    if MAX_RUN_USD is not None:
        details.append(f"budget.max_run_usd: generation stops at ${MAX_RUN_USD:.2f}")
    if MAX_INTENSITY:
        details.append(f"content_rating.max_intensity: capped at {MAX_INTENSITY}")
    return Check(
        id="show",
        status=PASS,
        summary=f"{SHOW_NAME} — {_count(len(CAST), 'cast member')}, "
                f"{_count(len(SCENES), 'scene')}, "
                f"{_count(len(EPISODES), 'episode')}",
        details=details,
    )


def _key_for_model(model):
    """(env key name, is it set) for a model string."""
    provider = provider_for_model(model)
    env_key = PROVIDER_ENV_KEYS.get(provider, "")
    return env_key, provider, bool(env_key) and api_key_set(env_key)


def check_director_key():
    env_key, provider, ok = _key_for_model(DIRECTOR_MODEL)
    if ok:
        return Check(id="director-key", status=PASS,
                     summary=f"{DIRECTOR_MODEL} — {env_key} set")
    need = env_key or f"a provider_env_keys entry for '{provider}'"
    return Check(
        id="director-key", status=FAIL,
        summary=f"{DIRECTOR_MODEL} needs {need}",
        details=[f"the director makes the first call of every run, so no "
                 f"scene can start without {need}"],
        remedy=f"set {need} in .env (copy .env.example if you haven't)")


def check_judge_key():
    """--eval only. A missing judge key must not make an otherwise healthy
    install report as broken."""
    env_key, provider, ok = _key_for_model(JUDGE_MODEL)
    if ok:
        return Check(id="judge-key", status=PASS,
                     summary=f"{JUDGE_MODEL} — {env_key} set")
    need = env_key or f"a provider_env_keys entry for '{provider}'"
    return Check(
        id="judge-key", status=WARN,
        summary=f"--eval unavailable: {JUDGE_MODEL} needs {need}",
        remedy=f"set {need} in .env to score scenes with --eval")


def check_cast_keys():
    """An ensemble scene needs two cast keys to be a conversation; a solo
    run needs one.

    Exactly one usable key is reported as a WARNING, not a failure: it is
    genuinely enough for `--solo --cast <id>`, which cli.cast_keys_or_die
    accepts. Calling that broken would make this command contradict the CLI
    it is describing."""
    ready, blocked = [], []
    for cid, member in CAST.items():
        env_key = PROVIDER_ENV_KEYS.get(member.get("provider", ""), "")
        if env_key and api_key_set(env_key):
            ready.append(cid)
        else:
            blocked.append(f"{cid} needs "
                           + (env_key or f"a provider_env_keys entry for "
                                         f"'{member.get('provider')}'"))
    tally = f"{len(ready)}/{len(CAST)} cast can run"
    if len(ready) >= 2 or (len(ready) == 1 and len(CAST) == 1):
        return Check(id="cast-keys", status=PASS,
                     summary=tally + (f" ({', '.join(ready)})" if blocked else ""),
                     details=blocked)
    if len(ready) == 1:
        return Check(
            id="cast-keys", status=WARN,
            summary=f"{tally} — enough for --solo, not for an ensemble scene",
            details=blocked,
            remedy=f"run --solo --cast {ready[0]}, or set another key in .env")
    return Check(
        id="cast-keys", status=FAIL,
        summary=f"{tally} — no cast member can speak",
        details=blocked,
        remedy="set the missing keys in .env (copy .env.example)")


def check_stage(name, preflight):
    """One optional stage. A preflight that raises becomes an ERROR rather
    than taking the whole report down with it — a doctor that dies on the
    first broken check is the least useful moment to lose the other answers.

    Preflight stdout is captured: some of them (vlog) print advisories
    rather than returning them, and loose prints would otherwise land above
    the report header, detached from the check that produced them — and
    would corrupt the --json payload outright."""
    flag = _STAGE_FLAGS.get(name, name)
    chatter = io.StringIO()
    try:
        with contextlib.redirect_stdout(chatter):
            problems = list(preflight(list(CAST.keys())))
    except Exception as e:
        # str(e) is redacted: this report is the artifact we tell people to
        # paste into a bug report, and a preflight can surface provider text.
        # Advisories printed before the raise are read here too — vlog
        # prints them from inside a per-cast loop, so a raise on a later
        # iteration would otherwise discard the earlier ones.
        return Check(id=f"stage:{name}", status=ERROR,
                     summary=f"{flag} could not be checked ({type(e).__name__})",
                     details=[redact(str(e))] + _lines(chatter))
    advisories = _lines(chatter)
    if not problems:
        return Check(id=f"stage:{name}", status=PASS, summary=f"{flag} ready",
                     details=advisories)
    return Check(id=f"stage:{name}", status=WARN,
                 summary=f"{flag} not ready ({_count(len(problems), 'problem')})",
                 details=problems + advisories)


def _run_core(check_id, fn):
    """Run one core check, turning a raise into a finding.

    Isolation matters most here, not least: this command exists to be run on
    a broken install, so a malformed world.json value must produce a report
    line, not the traceback that made the user reach for doctor in the first
    place."""
    try:
        return fn()
    except Exception as e:
        return Check(id=check_id, status=ERROR,
                     summary=f"check could not run ({type(e).__name__})",
                     details=[redact(str(e))],
                     remedy="this usually means a malformed value in "
                            "world.json — see CREATING_A_SHOW.md")


def run_checks():
    """Every check, in report order. Never raises.

    The audit manifest is snapshotted and restored around the stage checks:
    the audio/video/vlog/planner preflights deliberately load voices.json,
    visuals.json, and planner.txt to hash them into CONFIG_MANIFEST, which
    is correct during a run but wrong for a diagnostic — a library caller
    that ran doctor and then a scene would get an audit block claiming
    files the run never used."""
    checks = [_run_core(check_id, fn) for check_id, fn in (
        ("show", check_show), ("director-key", check_director_key),
        ("judge-key", check_judge_key), ("cast-keys", check_cast_keys))]
    manifest_before = dict(CONFIG_MANIFEST)
    try:
        checks += [check_stage(name, fn) for name, fn in STAGE_PREFLIGHTS.items()]
    finally:
        CONFIG_MANIFEST.clear()
        CONFIG_MANIFEST.update(manifest_before)
    return checks


# ── verdicts and rendering ───────────────────────────────

def overall(checks):
    """The headline verdict.

    ERROR maps to "degraded" rather than "failed" because it can only come
    from an optional stage or a check that couldn't run — neither of which
    means the engine can't run a scene. Only a core FAIL is "failed"."""
    worst = max((c.status for c in checks), key=lambda s: _RANK[s], default=PASS)
    return {PASS: "ok", WARN: "degraded", ERROR: "degraded", FAIL: "failed"}[worst]


def exit_code(checks):
    """0 healthy / 1 a core check failed / 2 a check couldn't run.

    FAIL outranks ERROR: "the director key is missing" is a determined
    answer and the more actionable one, so a broken visuals.json must not
    downgrade it to "don't know"."""
    statuses = {c.status for c in checks}
    if FAIL in statuses:
        return 1
    return 2 if ERROR in statuses else 0


def to_json(checks):
    """Machine-readable report. schema_version is the contract: additive
    changes bump nothing, a changed/removed field bumps the version."""
    return {
        "schema_version": "1",
        "overall": overall(checks),
        "checks": [{"id": c.id, "status": c.status, "summary": c.summary,
                    "details": list(c.details), "remedy": c.remedy}
                   for c in checks],
    }


_VERDICT_GLYPH = {"ok": _GLYPH[PASS], "degraded": _GLYPH[WARN], "failed": _GLYPH[FAIL]}


def format_human(checks):
    width = max((len(c.id) for c in checks), default=0)
    lines = [f"\n🩺 exec-producer doctor — {SHOW_NAME}\n"]
    for c in checks:
        lines.append(f"  {_GLYPH[c.status]} {c.id:<{width}}  {c.summary}")
        for detail in c.details:
            lines.append(f"     {'':<{width}}   - {detail}")
        if c.remedy:
            lines.append(f"     {'':<{width}}   → {c.remedy}")
    counts = {s: sum(1 for c in checks if c.status == s)
              for s in (FAIL, ERROR, WARN)}
    verdict = overall(checks)
    tally = ", ".join(f"{n} {label}" for label, n in
                      (("failing", counts[FAIL]), ("unchecked", counts[ERROR]),
                       ("degraded", counts[WARN])) if n)
    lines.append(f"\n  {_VERDICT_GLYPH[verdict]} {verdict}"
                 + (f" — {tally}" if tally else ""))
    return "\n".join(lines) + "\n"
