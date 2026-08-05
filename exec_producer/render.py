"""--from-script stage: render an externally-written scene through the
engine's downstream pipeline (.txt formatting, --eval, --audio, --video,
--produce) with zero generation calls.

The input contract is the engine's own saved-scene JSON shape (what
save_scene_outputs writes): scene, conversation, director_notes, cast,
intensity. A generation run's output file is a valid render input; external
scripts — e.g. written by a Claude Code session via the showrunner skill —
use the same shape. The decoupling from subscription CLIs happens at this
file contract: LiteLLM stays the engine's single model-access layer.

Validation here is the hard boundary: a malformed file must fail legibly,
listing every problem at once, BEFORE any TTS/video spend."""

import json
import re
from pathlib import Path

from .errors import ConfigError
from . import engine

# Stamped into the audit block (and the JSONL run_start header) of every
# --from-script run: the engine didn't generate this dialogue, so its cost
# is unknowable here (precedent: TTS cost is recorded as unknown too).
EXTERNAL_AUDIT = {"generator": "external", "generation_cost": "unknown"}


def _safe_slug(text, fallback="external"):
    slug = re.sub(r"[^a-z0-9_-]+", "_", str(text).lower()).strip("_")
    return slug or fallback


def load_external_scene(path):
    """Parse + hard-validate a scene .json against the engine's saved shape.
    Returns {scene_id, scene, conversation, director_notes, cast_ids,
    intensity} normalized for the downstream stages, or raises ConfigError
    listing every problem at once (the validate_data_dir pattern)."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"{path} is not a file")
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ConfigError(f"{path} is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a JSON object")
    if isinstance(data.get("scenes"), list):
        raise ConfigError(
            f"{path} looks like an EPISODE file (has a 'scenes' list) — "
            "render episode scenes individually")

    problems = []

    scene = data.get("scene")
    if not isinstance(scene, dict) or not str(scene.get("name", "")).strip():
        problems.append("'scene' must be an object with a non-empty 'name'")
        scene = {"name": "?", "prompt": "", "mood": ""}
    else:
        scene = {"name": scene["name"],
                 "prompt": scene.get("prompt", ""),
                 "mood": scene.get("mood", "")}

    conversation = data.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        problems.append("'conversation' must be a non-empty list of line entries")
        conversation = []
    unknown_ids = []
    normalized = []
    for i, entry in enumerate(conversation):
        if not isinstance(entry, dict):
            problems.append(f"conversation[{i}] must be an object")
            continue
        cid = str(entry.get("id", "")).strip().lower()
        text = entry.get("text")
        if not cid:
            problems.append(f"conversation[{i}] is missing 'id'")
            continue
        if cid not in engine.CAST:
            if cid not in unknown_ids:
                unknown_ids.append(cid)
            continue
        if not isinstance(text, str) or not text.strip():
            problems.append(f"conversation[{i}] ({cid}) is missing 'text'")
            continue
        # Display fields crash format_script/.upper() etc. if present but
        # non-string ("name": null) — catch them here, in the one boundary.
        bad_types = [k for k in ("name", "emoji", "tagline", "direction")
                     if k in entry and not isinstance(entry[k], str)]
        if bad_types:
            problems.append(
                f"conversation[{i}] ({cid}): {', '.join(bad_types)} must be strings")
            continue
        c = engine.CAST[cid]
        normalized.append({
            "id": cid,
            "name": entry.get("name", c["name"]),
            "emoji": entry.get("emoji", c["emoji"]),
            "tagline": entry.get("tagline", c["tagline"]),
            "text": text.strip(),
            "direction": entry.get("direction", ""),
            # forward-compatible passthrough (e.g. vlog delivery markers)
            **{k: v for k, v in entry.items()
               if k not in ("id", "name", "emoji", "tagline", "text", "direction")},
        })
    if unknown_ids:
        problems.append(
            f"unknown cast id(s) {', '.join(unknown_ids)} — every conversation 'id' must "
            f"exist in the loaded show's cast.json (available: {', '.join(engine.CAST)}); "
            "use --data-dir to load the right show")

    director_notes = data.get("director_notes", [])
    if not isinstance(director_notes, list) or any(
            not isinstance(n, dict) for n in director_notes):
        problems.append("'director_notes' must be a list of objects (or absent)")
        director_notes = []

    intensity = data.get("intensity", "low")
    if intensity not in engine.INTENSITY_ORDER:
        problems.append(
            f"'intensity' must be one of {', '.join(engine.INTENSITY_ORDER)} (got {intensity!r})")
        intensity = "low"

    if problems:
        raise ConfigError(
            f"{path} doesn't satisfy the scene contract:\n"
            + "\n".join(f"     - {m}" for m in problems)
            + "\n   The contract is the engine's own saved-scene shape — see "
            "CREATING_A_SHOW.md and .claude/skills/showrunner/SKILL.md.")

    # Cast = unique speakers in order of appearance (drives eval sheet,
    # voice coverage preflight, and the saved cast block).
    cast_ids = list(dict.fromkeys(e["id"] for e in normalized))
    scene_id = _safe_slug(data.get("scene_id") or p.stem)

    return {"scene_id": scene_id, "scene": scene, "conversation": normalized,
            "director_notes": director_notes, "cast_ids": cast_ids,
            "intensity": engine.effective_intensity(intensity)}


def render_external_scene(data, opts):
    """Run the downstream stages on a validated external scene. Identical
    output guarantees to a generated run (same save path), with the audit
    block stamped generator=external. Returns the scorecard (or None)."""
    names = ", ".join(engine.CAST[c]["name"] for c in data["cast_ids"])
    print("\n" + "=" * 60)
    print(f"🎞️   {engine.SHOW_NAME} — rendering external script")
    print(f"🎬  Scene: {data['scene']['name']}  ({len(data['conversation'])} lines)")
    print(f"🎭  Cast: {names}")
    print("=" * 60)
    return engine.save_scene_outputs(
        data["scene_id"], data["scene"], data["conversation"],
        data["director_notes"], data["cast_ids"], opts, data["intensity"],
        audit_extra=EXTERNAL_AUDIT)
