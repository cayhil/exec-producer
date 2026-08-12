"""Data loading — all creative content lives in a show directory as
JSON/text files. The directory is swappable: `--data-dir <path>` (or the
EXEC_PRODUCER_DATA_DIR env var) points the engine at any show."""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .errors import ConfigError

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The data contract: a show directory must provide all of these. Optional
# additions: voices.json (--audio), visuals.json (--video),
# prompts/planner.txt (--planner).
REQUIRED_FILES = [
    "world.json", "cast.json", "scenes.json", "episodes.json", "embodiment.json",
    "prompts/character_template.txt", "prompts/director.txt",
    "prompts/director_choice.txt", "prompts/casting.txt",
    "prompts/production.txt", "prompts/production_request.txt",
    "prompts/judge.txt", "prompts/eval_rubric.txt",
    "prompts/critical_rules.txt", "prompts/tone_face_to_face.txt",
    "prompts/tone_confessional_early.txt", "prompts/tone_confessional_late.txt",
]


def announce_data_dir_fallback(starter):
    """Tell the user which show got picked, on STDERR.

    Stderr, not stdout, because this fires at import — before any command
    has run — and stdout is a machine-readable payload for some of them.
    `doctor --json | jq` on a fresh clone (no data/, so this always fires)
    was getting a non-JSON first line."""
    print(f"ℹ️  No data/ directory — using the starter show at {starter}",
          file=sys.stderr)


def resolve_data_dir():
    """--data-dir/env var wins; otherwise the private data/ in the repo
    root; otherwise the committed starter show, so a fresh clone runs
    without any setup."""
    env = os.environ.get("EXEC_PRODUCER_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    private = _REPO_ROOT / "data"
    if private.is_dir():
        return private
    starter = _REPO_ROOT / "examples" / "sunset_terrace"
    if starter.is_dir():
        announce_data_dir_fallback(starter)
        return starter
    return private  # validate_data_dir turns this into the legible error


def validate_data_dir(path):
    """List every contract violation at once. Pure, so a broken show
    directory can be diagnosed without importing the rest of the engine."""
    path = Path(path)
    if not path.is_dir():
        return [f"{path} is not a directory"]
    return [f for f in REQUIRED_FILES if not (path / f).is_file()]


DATA_DIR = resolve_data_dir()
_missing = validate_data_dir(DATA_DIR)
if _missing:
    raise ConfigError(
        f"show directory {DATA_DIR} doesn't satisfy the data contract — missing:\n"
        + "\n".join(f"     - {m}" for m in _missing)
        + "\n   See CREATING_A_SHOW.md for the full contract."
    )

# Audit trail: sha256 of every config/prompt file loaded this run, so any
# saved output can be traced back to the exact inputs that produced it.
CONFIG_MANIFEST = {}


def _load_bytes(filename):
    raw = (DATA_DIR / filename).read_bytes()
    CONFIG_MANIFEST[filename] = hashlib.sha256(raw).hexdigest()
    return raw


def _load_json(filename):
    """Load a JSON file from the data directory."""
    return json.loads(_load_bytes(filename))


def _load_text(filename):
    """Load a text file from the data directory."""
    return _load_bytes(filename).decode()


# Core data
_world_config = _load_json("world.json")
WORLD = _world_config["world"]
INTENSITY_GUIDE = _world_config["intensity_guide"]
DIRECTOR_MODEL = _world_config["director_model"]
# Judge defaults to the director model, but should be a different one:
# a director grading its own decisions is the classic LLM-as-judge
# self-preference trap.
JUDGE_MODEL = _world_config.get("judge_model", _world_config["director_model"])
PROVIDER_ENV_KEYS = _world_config["provider_env_keys"]

# Show identity — banners and output-file prefixes. Defaults keep older
# world.json files (without the keys) working.
SHOW_NAME = _world_config.get("show_name", "UNTITLED SHOW")
SHOW_SLUG = _world_config.get("show_slug", "show")

# Reliability: retries/timeouts/fallbacks for every LiteLLM call.
# Defaults apply when world.json has no "reliability" block, so show data
# written before this feature keeps working unchanged.
_DEFAULT_RELIABILITY = {
    "num_retries": 3,
    "timeout_seconds": 120,
    "fallbacks": {},
}
RELIABILITY = {**_DEFAULT_RELIABILITY, **_world_config.get("reliability", {})}

# Content-rating gate: when world.json sets content_rating.max_intensity,
# every scene/episode intensity is capped at that level regardless of what
# the episode config or CLI asks for — a guardrail for public/demo runs.
MAX_INTENSITY = _world_config.get("content_rating", {}).get("max_intensity")


def validate_budget(world_config):
    """Problems with the optional budget block, in the list-them-all style.

    Validated at LOAD, unlike content_rating (which degrades harmlessly on a
    bad value), because every failure mode here costs money: a string cap
    passes a truthiness check and then raises TypeError at the first turn
    boundary — i.e. after the run has already billed — and a negative cap
    silently stops every run before its first line."""
    budget = world_config.get("budget", {})
    if not isinstance(budget, dict):
        return ["world.json 'budget' must be an object, e.g. "
                '{"budget": {"max_run_usd": 2.50}}']
    if "max_run_usd" not in budget:
        return []
    value = budget["max_run_usd"]
    # bool is an int subclass, and `true` in JSON is a plausible typo for a
    # value someone meant to be numeric.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [f"world.json budget.max_run_usd must be a number (got "
                f"{type(value).__name__} {value!r}) — e.g. 2.50, or remove "
                "the key for no cap"]
    if value < 0:
        return [f"world.json budget.max_run_usd must be >= 0 (got {value}) — "
                "use 0 to forbid all spend, or remove the key for no cap"]
    return []


def read_max_run_usd(world_config):
    """The configured spend cap, or None when there is none.

    Returns 0 as 0, not as None: someone writing `"max_run_usd": 0` means
    "this run must not spend", and a truthiness test would hand them an
    uncapped run instead — the exact inversion of their intent."""
    value = world_config.get("budget", {}).get("max_run_usd")
    return None if value is None else value


# Budget gate, the money-shaped sibling of content_rating: when world.json
# sets budget.max_run_usd, the generation loops AND the media stages stop
# once this run's recorded spend reaches it, saving what was already
# produced. Absent (the default) = no cap. It is a stop-line rather than a
# hard ceiling — a call's cost isn't knowable until it returns — so a run
# can overshoot by the calls already in flight for the current unit of work.
_budget_problems = validate_budget(_world_config)
if _budget_problems:
    raise ConfigError(
        f"show directory {DATA_DIR} doesn't satisfy the data contract:\n"
        + "\n".join(f"     - {p}" for p in _budget_problems))
MAX_RUN_USD = read_max_run_usd(_world_config)

CAST = _load_json("cast.json")
SCENES = _load_json("scenes.json")
EPISODES = _load_json("episodes.json")

# Show identifiers become output paths and ffmpeg concat-list entries — a
# slug or key with '/', '..', quotes, or control characters would let a
# hostile show directory write outside scripts/. Reject at load, listing
# every offender.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _require_safe_ids(source, ids):
    bad = sorted(str(i) for i in ids if not _SAFE_ID.fullmatch(str(i)))
    if bad:
        raise ConfigError(
            f"{source} identifiers must be filename-safe ([A-Za-z0-9_-]): "
            + ", ".join(repr(b) for b in bad))


_require_safe_ids("world.json show_slug", [SHOW_SLUG])
_require_safe_ids("cast.json", CAST.keys())
_require_safe_ids("scenes.json", SCENES.keys())
_require_safe_ids("episodes.json", EPISODES.keys())


def validate_show_data(cast, scenes, episodes):
    """Cross-file shape checks for fields the engine reads mid-run. Pure,
    like validate_data_dir, so a broken show can be diagnosed directly.
    The stakes: an episode entry referencing a missing scene (or lacking
    'lines') would otherwise KeyError at that scene's start — after every
    earlier scene was already billed, with nothing saved until episode wrap."""
    problems = []
    for cid, c in cast.items():
        if not isinstance(c, dict):
            problems.append(f"cast.json '{cid}' must be an object")
            continue
        if cid != cid.lower():
            problems.append(
                f"cast.json id '{cid}' must be lowercase — director casting "
                "replies and --cast values are normalized to lowercase, so an "
                "uppercase id can never be matched")
        missing = [k for k in ("name", "provider", "model", "emoji", "tagline",
                               "gender", "personality", "triggers") if k not in c]
        if missing:
            problems.append(f"cast.json '{cid}' is missing: {', '.join(missing)}")
        elif not isinstance(c["triggers"], list):
            problems.append(f"cast.json '{cid}' triggers must be a list")
    for sid, s in scenes.items():
        if not isinstance(s, dict):
            problems.append(f"scenes.json '{sid}' must be an object")
            continue
        missing = [k for k in ("name", "prompt", "mood") if k not in s]
        if missing:
            problems.append(f"scenes.json '{sid}' is missing: {', '.join(missing)}")
    for eid, ep in episodes.items():
        if not isinstance(ep, dict):
            problems.append(f"episodes.json '{eid}' must be an object")
            continue
        missing = [k for k in ("name", "description") if k not in ep]
        if missing:
            problems.append(f"episodes.json '{eid}' is missing: {', '.join(missing)}")
        entries = ep.get("scenes")
        if not isinstance(entries, list) or not entries:
            problems.append(f"episodes.json '{eid}' needs a non-empty 'scenes' list")
            continue
        for n, sc in enumerate(entries, 1):
            if not isinstance(sc, dict):
                problems.append(f"episodes.json '{eid}' scene {n} must be an object")
                continue
            if "scene" not in sc:
                problems.append(f"episodes.json '{eid}' scene {n} is missing 'scene'")
            elif sc["scene"] not in scenes:
                problems.append(f"episodes.json '{eid}' scene {n} references "
                                f"unknown scene '{sc['scene']}'")
            if not isinstance(sc.get("lines"), int) or sc["lines"] < 1:
                problems.append(f"episodes.json '{eid}' scene {n} needs an "
                                "integer 'lines' >= 1")
    return problems


_data_problems = validate_show_data(CAST, SCENES, EPISODES)
if _data_problems:
    raise ConfigError(
        f"show directory {DATA_DIR} doesn't satisfy the data contract:\n"
        + "\n".join(f"     - {p}" for p in _data_problems)
        + "\n   See CREATING_A_SHOW.md for the full contract."
    )

# Prompt templates
DIRECTOR_SYSTEM = _load_text("prompts/director.txt").replace("__WORLD__", WORLD)
PRODUCTION_SYSTEM = _load_text("prompts/production.txt")
EVAL_RUBRIC = _load_text("prompts/eval_rubric.txt")
CASTING_TEMPLATE = _load_text("prompts/casting.txt")
DIRECTOR_CHOICE_TEMPLATE = _load_text("prompts/director_choice.txt")
PRODUCTION_REQUEST_TEMPLATE = _load_text("prompts/production_request.txt")
JUDGE_SYSTEM = _load_text("prompts/judge.txt")

# Character prompt components
CHARACTER_TEMPLATE = _load_text("prompts/character_template.txt")
CRITICAL_RULES = _load_text("prompts/critical_rules.txt")
TONE_FACE_TO_FACE = _load_text("prompts/tone_face_to_face.txt")
TONE_CONFESSIONAL_EARLY = _load_text("prompts/tone_confessional_early.txt")
TONE_CONFESSIONAL_LATE = _load_text("prompts/tone_confessional_late.txt")
EMBODIMENT = _load_json("embodiment.json")

def load_voices():
    """Load data/voices.json for the --audio stage. On demand, not at import:
    a missing voices file must never break non-audio runs. Going through
    _load_bytes adds its sha256 to CONFIG_MANIFEST, and AUDIT holds that dict
    by reference — so audio runs carry the voices file in their audit trail
    as long as this is called before any output (or the run-log header) is
    written. The CLI preflight does exactly that."""
    return _load_json("voices.json")


def load_planner_prompt():
    """Load the --planner arc-writer system prompt — same on-demand +
    audit-manifest contract as load_voices(): data/ sets without the file
    must not break non-planner runs."""
    return _load_text("prompts/planner.txt").replace("__WORLD__", WORLD)


def load_visuals():
    """Load data/visuals.json for the --video stage — same on-demand +
    audit-manifest contract as load_voices(). The file is optional: the
    video stage falls back to built-in card defaults when it's absent."""
    return _load_json("visuals.json")


# Audit block embedded in every saved output and at the top of the JSONL run
# log: exactly which engine version, models, and config-file contents (by
# hash) produced a given script.
AUDIT = {
    "engine_version": __version__,
    "director_model": DIRECTOR_MODEL,
    "judge_model": JUDGE_MODEL,
    "config_files": CONFIG_MANIFEST,
}
