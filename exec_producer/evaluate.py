"""Judge-model eval pass — one LLM call per scene scoring the generated
script against explicit criteria: format adherence (critical_rules.txt),
character voice consistency, and pacing appropriate to the intensity level.

The judge is deliberately a different model than the director (configurable
as judge_model in world.json) so the director isn't grading its own work.
Evaluation must never crash a run: any judge failure degrades to "no
scorecard" with a warning.
"""

import json
import time

from .config import (CAST, CRITICAL_RULES, EVAL_RUBRIC, INTENSITY_GUIDE,
                     JUDGE_MODEL, JUDGE_SYSTEM)
from .llm import call_model_simple, parse_json_response

REQUIRED_KEYS = ("format_adherence", "voice_consistency", "pacing", "overall")


def build_script_text(conversation, director_notes):
    """Script as the judge sees it: stage directions + 'Name: line'."""
    lines = []
    for i, entry in enumerate(conversation):
        if i < len(director_notes) and director_notes[i].get("stage_direction"):
            lines.append(f"[{director_notes[i]['stage_direction']}]")
        lines.append(f"{entry['name']}: {entry['text']}")
    return "\n".join(lines)


def build_judge_prompt(conversation, director_notes, cast_ids, intensity):
    cast_sheet = "\n".join(
        f"- {CAST[c]['name']} ({CAST[c]['tagline']}): {CAST[c]['personality']}"
        for c in cast_ids
    )
    prompt = EVAL_RUBRIC
    prompt = prompt.replace("__CRITICAL_RULES__", CRITICAL_RULES)
    prompt = prompt.replace("__CAST_SHEET__", cast_sheet)
    prompt = prompt.replace("__INTENSITY__", INTENSITY_GUIDE.get(intensity, INTENSITY_GUIDE["high"]))
    prompt = prompt.replace("__SCRIPT__", build_script_text(conversation, director_notes))
    return prompt


def evaluate_scene(conversation, director_notes, cast_ids, intensity):
    """One judge call → scorecard dict, or None if the judge fails.

    API errors are already retried inside call_model; only malformed JSON
    is retried here.
    """
    prompt = build_judge_prompt(conversation, director_notes, cast_ids, intensity)
    for attempt in range(3):
        try:
            raw = call_model_simple(JUDGE_MODEL, JUDGE_SYSTEM, prompt,
                                    max_tokens=2000, caller="judge")
        except Exception as e:
            print(f"  ⚠️  Judge unavailable ({type(e).__name__}) — no scorecard")
            return None
        try:
            card = parse_json_response(raw)
            # isinstance guard: any JSON type can come back (an int, a bare
            # string containing the key names) — only a dict is a scorecard.
            if isinstance(card, dict) and all(k in card for k in REQUIRED_KEYS):
                return card
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        time.sleep(0.5)
    print("  ⚠️  Judge returned malformed scorecards 3 times — no scorecard")
    return None


def summarize_scorecard(card):
    """Compact one-line score summary (safe to print in --director-only:
    scores only, no dialogue quotes)."""
    def sc(key):
        section = card.get(key)
        return section.get("score", "?") if isinstance(section, dict) else "?"

    return (f"format {sc('format_adherence')}/10 | voice {sc('voice_consistency')}/10 | "
            f"pacing {sc('pacing')}/10 | overall {sc('overall')}/10")
