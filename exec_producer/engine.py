"""Main engine — director loop, character sessions, scene and episode runners."""

import json
import os
import random
import time
from dataclasses import dataclass, replace
from datetime import datetime

from .config import (
    AUDIT,
    CAST,
    CASTING_TEMPLATE,
    CRITICAL_RULES,
    DIRECTOR_CHOICE_TEMPLATE,
    DIRECTOR_MODEL,
    DIRECTOR_SYSTEM,
    EMBODIMENT,
    EPISODES,
    INTENSITY_GUIDE,
    JUDGE_MODEL,
    MAX_INTENSITY,
    PRODUCTION_REQUEST_TEMPLATE,
    PRODUCTION_SYSTEM,
    PROVIDER_ENV_KEYS,
    SCENES,
    SHOW_NAME,
    SHOW_SLUG,
    TONE_CONFESSIONAL_EARLY,
    TONE_CONFESSIONAL_LATE,
    TONE_FACE_TO_FACE,
    CHARACTER_TEMPLATE,
    WORLD,
)
from .llm import (api_key_set, call_director, call_model, call_model_simple,
                  parse_json_response)
from . import telemetry
from .evaluate import evaluate_scene, summarize_scorecard

INTENSITY_ORDER = ["low", "medium", "high", "maximum"]


@dataclass
class RunOptions:
    """Per-run flags shared by run_scene and run_episode — one object instead
    of a growing tail of keyword arguments. Episode runs ignore num_lines,
    intensity, and plot (the episode config supplies those per scene)."""
    num_lines: int = 15
    save: bool = True
    produce: bool = False
    intensity: str = "high"
    plot: str = ""
    debug: bool = False
    director_only: bool = False
    evaluate: bool = False
    audio: bool = False
    video: bool = False
    render_mode: str = "tts"  # or "veo"/"vlog": generated footage instead of cards
    clips_dir: str = ""       # vlog bring-your-own-clips folder (via --from-script)


def effective_intensity(requested):
    """Apply the content-rating cap (content_rating.max_intensity in
    world.json). A guardrail for public/demo runs: caps intensity regardless
    of what the episode config or CLI asks for."""
    cap = MAX_INTENSITY
    if not cap or requested not in INTENSITY_ORDER or cap not in INTENSITY_ORDER:
        return requested
    if INTENSITY_ORDER.index(requested) > INTENSITY_ORDER.index(cap):
        print(f"  🔒 Content rating cap: intensity '{requested}' → '{cap}' (content_rating.max_intensity)")
        return cap
    return requested


# ─────────────────────────────────────────────
# SESSION MANAGEMENT — persistent per-character conversations
# ─────────────────────────────────────────────

def build_cast_sheet(cast_ids):
    """Build a cast reference sheet with names, genders, and roles."""
    if len(cast_ids) == 1:
        c = CAST[cast_ids[0]]
        gender_label = {"guy": "he/him", "girl": "she/her"}.get(c["gender"], c["gender"])
        return ("THE CAST — just you. This is a solo show; no one else is on camera:\n"
                f"  - {c['name']} ({gender_label}) — {c['tagline']}")
    lines = ["THE CAST (memorize — these are the people in the house with you):"]
    for cid in cast_ids:
        c = CAST[cid]
        gender_label = {"guy": "he/him", "girl": "she/her"}.get(c["gender"], c["gender"])
        lines.append(f"  - {c['name']} ({gender_label}) — {c['tagline']}")
    return "\n".join(lines)


def tone_rules_for(scene, intensity):
    """Scene-appropriate tone rules — group scenes vs confessionals (which
    split early/late season by intensity). Matched on the scene's display
    name containing 'confessional'."""
    is_confessional = "confessional" in scene.get("name", "").lower()
    if is_confessional and intensity in ("high", "maximum"):
        return TONE_CONFESSIONAL_LATE
    if is_confessional:
        return TONE_CONFESSIONAL_EARLY
    return TONE_FACE_TO_FACE


def build_transition_text(scene, plot, scene_cast, intensity):
    """The between-scenes context injection for a persistent session.
    Carries the new scene's tone rules: the system prompt bakes in only the
    FIRST scene's tone, so without this a confessional later in an episode
    would run under group-scene rules. Shared with the planner graph."""
    return (
        f"--- SCENE CHANGE ---\n"
        f"New scene: {scene['name']}\n"
        f"Setting: {scene.get('prompt', '')}\n"
        f"Mood: {scene.get('mood', '')}\n"
        f"What's happening: {plot}\n"
        f"Cast in this scene: {', '.join(CAST[c]['name'] for c in scene_cast)}\n\n"
        f"TONE RULES FOR THIS SCENE:\n{tone_rules_for(scene, intensity)}"
    )


def build_direction_text(director_note):
    """Build the direction instruction text from a director note."""
    direction = director_note.get("direction", "RESPOND_TO")
    target = director_note.get("target", "")
    stage = director_note.get("stage_direction", "")
    emotional = director_note.get("emotional_state", "")

    text = f"The DIRECTOR wants you to: {direction}"
    if target and target in CAST:
        text += f" (focused on {CAST[target]['name']})"
    if stage:
        text += f"\nWhat's happening around you: {stage}"
    if emotional:
        text += f"\nThe room's energy right now: {emotional}"
    return text


class CharacterSession:
    """Maintains persistent conversation history for one character across a scene."""

    def __init__(self, cast_id, scene, intensity, plot, cast_ids):
        self.cast_id = cast_id
        self.character = CAST[cast_id]
        self.model = self.character["model"]
        self.messages = []
        self.line_count = 0
        self.total_tokens = 0

        # Build the system prompt ONCE
        system_prompt = self._build_system_prompt(scene, intensity, plot, cast_ids)
        self.messages.append({"role": "system", "content": system_prompt})

    def _build_system_prompt(self, scene, intensity, plot, cast_ids):
        """Build the full system prompt from template + data files."""
        character = self.character
        cast_id = self.cast_id
        gender = character.get("gender", "guy")

        # Embodiment: check character override first, then gender default
        overrides = EMBODIMENT.get("character_overrides", {})
        if cast_id in overrides:
            embodiment_specific = overrides[cast_id]
        else:
            embodiment_specific = EMBODIMENT["gender"].get(gender, EMBODIMENT["gender"]["guy"])

        # Relationships
        rel_lines = []
        for other_id, rel in character.get("relationships", {}).items():
            if other_id in cast_ids:
                rel_lines.append(f"  Your feelings about {CAST[other_id]['name']}: {rel}")

        # Tone rules (per-scene; transitions re-inject them for later scenes)
        tone_rules = tone_rules_for(scene, intensity)

        # Assemble from template
        prompt = CHARACTER_TEMPLATE
        prompt = prompt.replace("__WORLD__", WORLD)
        prompt = prompt.replace("__CAST_SHEET__", build_cast_sheet(cast_ids))
        prompt = prompt.replace("__PERSONALITY__", character["personality"])
        prompt = prompt.replace("__INTENSITY__", INTENSITY_GUIDE.get(intensity, INTENSITY_GUIDE["high"]))
        prompt = prompt.replace("__EMBODIMENT_BASE__", EMBODIMENT["base"])
        prompt = prompt.replace("__EMBODIMENT_SPECIFIC__", embodiment_specific)
        prompt = prompt.replace("__EMBODIMENT_INTENSITY__", EMBODIMENT["intensity"].get(intensity, EMBODIMENT["intensity"]["high"]))
        prompt = prompt.replace("__RELATIONSHIPS__", "\n".join(rel_lines))
        prompt = prompt.replace("__TRIGGERS__", "\n".join("  - " + t for t in character["triggers"]))
        prompt = prompt.replace("__TONE_RULES__", tone_rules)
        prompt = prompt.replace("__SCENE_PROMPT__", scene.get("prompt", ""))
        prompt = prompt.replace("__SCENE_MOOD__", scene.get("mood", ""))
        prompt = prompt.replace("__PLOT__", f"SCENE PLOT: {plot}" if plot else "")
        prompt = prompt.replace("__CRITICAL_RULES__", CRITICAL_RULES)
        return prompt

    @classmethod
    def from_messages(cls, cast_id, messages, line_count=0, total_tokens=0):
        """Rebuild a session from serialized message dicts — the --planner
        graph keeps sessions as checkpointable JSON state, not live objects,
        so nodes rehydrate a session, use it, and serialize it back."""
        session = cls.__new__(cls)
        session.cast_id = cast_id
        session.character = CAST[cast_id]
        session.model = session.character["model"]
        session.messages = [dict(m) for m in messages]
        session.line_count = line_count
        session.total_tokens = total_tokens
        return session

    def to_state(self):
        """The checkpointable counterpart of from_messages."""
        return {"messages": self.messages, "line_count": self.line_count,
                "total_tokens": self.total_tokens}

    def inject_context(self, events_text):
        """Add a user message describing what happened while this character wasn't speaking."""
        if events_text.strip():
            self.messages.append({"role": "user", "content": events_text})

    def add_own_response(self, text):
        """Record this character's own line as an assistant message."""
        self.messages.append({"role": "assistant", "content": text})
        self.line_count += 1

    def get_messages(self):
        """Return consolidated messages (merges consecutive same-role for Anthropic compat)."""
        if len(self.messages) <= 1:
            return list(self.messages)
        consolidated = [self.messages[0]]  # system always first
        for msg in self.messages[1:]:
            if msg["role"] == consolidated[-1]["role"]:
                consolidated[-1] = {
                    "role": msg["role"],
                    "content": consolidated[-1]["content"] + "\n\n" + msg["content"]
                }
            else:
                consolidated.append(dict(msg))
        return consolidated

    def trim_if_needed(self, max_messages=30):
        """Keep system prompt (index 0) + last N messages. Drop the middle."""
        if len(self.messages) <= max_messages + 1:
            return
        tail = self.messages[-(max_messages):]
        # The Anthropic API requires the first non-system message to be a
        # user turn; an assistant-first tail after trimming would 400.
        while tail and tail[0]["role"] == "assistant":
            tail.pop(0)
        self.messages = [self.messages[0]] + tail


# ─────────────────────────────────────────────
def get_scene_cast(scene, all_cast_ids, plot=""):
    """Ask the director which cast members should be in this scene."""
    # A solo show has nothing to cast — skip the director call entirely.
    if len(all_cast_ids) == 1:
        return list(all_cast_ids)
    cast_list = "\n".join(f"  - {CAST[c]['name']} ({c}): {CAST[c]['tagline']}" for c in all_cast_ids)
    plot_text = f"\nPLOT: {plot}" if plot else ""

    prompt = (CASTING_TEMPLATE
              .replace("__SCENE_NAME__", scene["name"])
              .replace("__SCENE_PROMPT__", scene["prompt"])
              .replace("__SCENE_MOOD__", scene["mood"])
              .replace("__PLOT__", plot_text)
              .replace("__CAST_LIST__", cast_list))

    for attempt in range(3):
        # API errors are already retried inside call_model — don't retry them
        # again here, just degrade to the full cast.
        try:
            raw = call_director(DIRECTOR_SYSTEM, prompt)
        except Exception as e:
            print(f"\n  ⚠️  Director casting unavailable ({type(e).__name__}) — using full cast")
            break
        try:
            result = parse_json_response(raw)
            raw_cast = result.get("cast", [])
            # Normalize + dedupe: display names ("Alpha") and repeated ids
            # would otherwise degrade to full cast or double-inject the
            # scene transition into one session.
            cast = list(dict.fromkeys(
                str(c).lower().strip() for c in raw_cast
                if str(c).lower().strip() in all_cast_ids))
            if len(cast) >= 2:
                return cast
        except (json.JSONDecodeError, AttributeError):
            time.sleep(0.5)

    # Fallback: use everyone — a copy, because episode runs extend the
    # returned list in place (forced-cast merge) and must not mutate the
    # caller's cast list through an alias.
    return list(all_cast_ids)


def get_director_choice(scene, conversation, cast_ids, last_speaker, intensity="high", plot="",
                        plan_note=""):
    """Ask the director who should speak next. plan_note carries the
    --planner arc's entry for this turn; the director is told to pace to it."""

    cast_descriptions = []
    for cid in cast_ids:
        c = CAST[cid]
        triggers = "\n    ".join(c["triggers"])
        cast_descriptions.append(
            f"  {c['name']} ({cid}):\n"
            f"    Personality: {c['tagline']}\n"
            f"    Triggers:\n    {triggers}"
        )

    conv_text = ""
    for entry in conversation[-15:]:  # last 15 lines for context window
        conv_text += f"  {entry['name']}: {entry['text']}\n"

    intensity_text = INTENSITY_GUIDE.get(intensity, INTENSITY_GUIDE["high"])
    plot_text = f"\nPLOT BEATS (things that MUST happen in this scene — weave these into the action):\n{plot}\n" if plot else ""
    plan_text = (f"\nTHE SCENE PLAN FOR THIS TURN (from the scene planner):\n{plan_note}\n"
                 f"Pace to the plan — build at its speed, do not escalate beyond what it calls for.\n"
                 if plan_note else "")
    spoken = set(e["id"] for e in conversation)

    prompt = (DIRECTOR_CHOICE_TEMPLATE
              .replace("__SCENE_NAME__", scene["name"])
              .replace("__SCENE_PROMPT__", scene["prompt"])
              .replace("__SCENE_MOOD__", scene["mood"])
              .replace("__PLOT__", plot_text)
              .replace("__INTENSITY__", intensity_text)
              .replace("__PLAN__", plan_text)
              .replace("__CAST_DESCRIPTIONS__", "\n".join(cast_descriptions))
              .replace("__CONVERSATION__", conv_text if conv_text else "  (No one has spoken yet — pick who breaks the ice)")
              .replace("__LAST_SPEAKER__", last_speaker or "none")
              .replace("__SPEAKERS__", ", ".join(spoken) if spoken else "none")
              .replace("__NOT_YET__", ", ".join(c for c in cast_ids if c not in spoken) or "everyone has spoken"))

    for attempt in range(3):
        # API errors are already retried inside call_model — don't retry them
        # again here, fall through to the random-pick fallback.
        try:
            raw = call_director(DIRECTOR_SYSTEM, prompt)
        except Exception as e:
            print(f"\n  ⚠️  Director unavailable ({type(e).__name__}) — falling back to random pick")
            break
        try:
            result = parse_json_response(raw)

            speaker = result.get("next_speaker", "").lower().strip()
            direction = result.get("direction", "").upper()
            # Non-string targets (a list, a dict) would crash downstream
            # lookups — sanitize once here so every consumer can trust it.
            # String targets get the same normalization as next_speaker: a
            # display-name target would silently drop the focus note.
            if not isinstance(result.get("target"), str):
                result["target"] = None
            else:
                result["target"] = result["target"].lower().strip()
            is_monologue = "MONOLOGUE" in direction
            if speaker in cast_ids and (speaker != last_speaker or is_monologue):
                # Write the normalized id back: the director sometimes returns
                # a display name ("Alpha") that passes validation here but
                # would KeyError in CAST lookups downstream.
                result["next_speaker"] = speaker
                return result
            # If director picked invalid speaker, pick randomly (guard: a
            # 1-member cast, reachable via library use, must not crash here)
            options = [c for c in cast_ids if c != last_speaker] or list(cast_ids)
            result["next_speaker"] = random.choice(options)
            return result
        except (json.JSONDecodeError, AttributeError):
            time.sleep(0.5)

    # Fallback: random pick with generic direction
    options = [c for c in cast_ids if c != last_speaker] or list(cast_ids)
    return {
        "next_speaker": random.choice(options),
        "direction": "RESPOND_TO",
        "target": last_speaker,
        "reasoning": "director fallback",
        "stage_direction": None,
        "emotional_state": "chaotic",
    }


# ─────────────────────────────────────────────
# CHARACTER LINE GENERATION (session-based)
# ─────────────────────────────────────────────

def _unwrap_reply(text):
    """Trim whitespace and unwrap a reply that is entirely quote- or
    asterisk-wrapped. Only matched pairs with no interior occurrence are
    stripped: a blind .strip('*') eats the outer asterisks of balanced
    markup like '*sighs* fine *walks out*', and the survivors re-pair
    downstream into action spans that swallow the dialogue."""
    text = text.strip()
    for ch in ('"', "*"):
        while (len(text) >= 2 and text[0] == ch and text[-1] == ch
               and ch not in text[1:-1]):
            text = text[1:-1].strip()
    return text


def generate_line_session(session):
    """Generate a character's line using their persistent session."""
    try:
        session.trim_if_needed()
        messages = session.get_messages()
        text, usage = call_model(session.model, messages, max_tokens=400, caller=session.cast_id)
        # Unwrap BEFORE the emptiness check: a whitespace-only response is
        # truthy, and recording it as an empty assistant message poisons the
        # session (Anthropic rejects empty content blocks on every later call).
        text = _unwrap_reply(text) if text else ""
        if not text:
            text = f"[{session.character['name']} stares into the distance, jaw clenched]"
        session.add_own_response(text)
        if usage:
            session.total_tokens += getattr(usage, 'total_tokens', 0) or 0
        return text
    except Exception as e:
        character = session.character
        # Exception type only — provider error strings can echo prompt
        # fragments, which must not reach a --director-only console.
        print(f"\n  ⚠️  API ERROR for {character['name']} ({session.model}): {type(e).__name__}")
        fallback = f"[{character['name']} stares into the distance, jaw clenched, clearly holding something back]"
        session.add_own_response(fallback)
        return fallback


def _events_since_last_line(cast_id, conversation):
    """Slice of the conversation after this character's own last line
    (the whole conversation if they haven't spoken)."""
    last_own_idx = None
    for idx in range(len(conversation) - 1, -1, -1):
        if conversation[idx]["id"] == cast_id:
            last_own_idx = idx
            break
    start_idx = (last_own_idx + 1) if last_own_idx is not None else 0
    return conversation[start_idx:]


def inject_scene_update(session, conversation, direction_text):
    """Inject what happened since this character last spoke into their session."""
    new_events = _events_since_last_line(session.cast_id, conversation)

    if new_events:
        event_lines = [f"{e['name']}: {e['text']}" for e in new_events]
        context = (
            "Here's what just happened:\n"
            + "\n".join(event_lines)
            + f"\n\n{direction_text}\n\n"
            f"Now respond as {session.character['name']}. Stay in character."
        )
    else:
        context = (
            f"{'You are speaking first. Set the tone for the scene.' if not conversation else 'You are continuing.'}\n\n"
            f"{direction_text}\n\n"
            f"Now respond as {session.character['name']}. Stay in character."
        )
    session.inject_context(context)


def inject_scene_tail(session, conversation):
    """After a scene ends, give this character the lines they never saw —
    everything after their own last line (or the whole scene if they were
    present but never spoke). Sessions persist across an episode while each
    scene's conversation log starts empty, so without this a character
    enters the next scene unaware of anything said after their final line."""
    missed = _events_since_last_line(session.cast_id, conversation)
    if missed:
        lines = "\n".join(f"{e['name']}: {e['text']}" for e in missed)
        session.inject_context(f"Before the scene ended, this also happened:\n{lines}")


def format_script(scene, conversation, director_notes):
    """Format the conversation as a proper script."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"{SHOW_NAME} — {scene['name']}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Mood: {scene['mood']}")
    lines.append("=" * 60)
    lines.append("")

    for i, entry in enumerate(conversation):
        # Add stage directions if the director gave any
        if i < len(director_notes) and director_notes[i].get("stage_direction"):
            lines.append(f"  [{director_notes[i]['stage_direction']}]")
            lines.append("")

        lines.append(f"{entry['emoji']} {entry['name'].upper()} ({entry['tagline']})")
        lines.append(f"  {entry['text']}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("END SCENE")
    lines.append("=" * 60)

    return "\n".join(lines)


# ─────────────────────────────────────────────
def generate_production_prompts(scene, conversation, director_notes, cast_ids):
    """Take a generated script and produce Gemini image prompts + Veo 3 video prompts."""

    # Build the script text for the LLM
    script_lines = []
    for i, entry in enumerate(conversation):
        stage = ""
        if i < len(director_notes) and director_notes[i].get("stage_direction"):
            stage = f" [{director_notes[i]['stage_direction']}]"
        script_lines.append(f"{entry['name']}: {entry['text']}{stage}")

    script_text = "\n".join(script_lines)

    cast_visual = "\n".join(
        f"  - {CAST[cid]['name']} ({CAST[cid]['emoji']} {CAST[cid]['tagline']})"
        for cid in cast_ids
    )

    prompt = (PRODUCTION_REQUEST_TEMPLATE
              .replace("__SCENE_NAME__", scene["name"])
              .replace("__SCENE_MOOD__", scene["mood"])
              .replace("__SCENE_PROMPT__", scene["prompt"])
              .replace("__CAST_VISUAL__", cast_visual)
              .replace("__SCRIPT__", script_text))

    print("\n  🎨 Generating production prompts (image + video)...", flush=True)

    for attempt in range(3):
        # API errors are already retried inside call_model — don't retry them
        # again here, just report the failure.
        try:
            raw = call_model_simple(DIRECTOR_MODEL, PRODUCTION_SYSTEM, prompt, max_tokens=4000, caller="production")
        except Exception:
            break
        try:
            result = parse_json_response(raw)
            beats = result.get("beats", [])
            # Shape-check now: a malformed beats list would otherwise crash
            # the _prompts.txt writer after the script files are saved.
            if isinstance(beats, list) and beats and all(isinstance(b, dict) for b in beats):
                print(f"  🎨 Generated {len(beats)} visual beats")
                return result
        except (json.JSONDecodeError, AttributeError):
            if attempt < 2:
                time.sleep(1)

    print("  ⚠️  Production prompt generation failed — script still saved")
    return None


def save_production_files(prefix, scene, conversation, director_notes, cast_ids,
                          quiet=False, production=None):
    """Generate production prompts and save the _production.json /
    _prompts.txt pair under the given path prefix. Returns True on success.
    Pass production to reuse an already-generated beat set (the Veo render
    path pays for one) instead of billing a second call."""
    if production is None:
        production = generate_production_prompts(scene, conversation, director_notes, cast_ids)
    if not production:
        return False

    prod_filename = f"{prefix}_production.json"
    with open(prod_filename, "w") as f:
        json.dump({**production, "audit": AUDIT}, f, indent=2)
    if not quiet:
        print(f"🎨 Production prompts saved to: {prod_filename}")

    # Also save individual prompts as easy copy-paste text files
    prompts_filename = f"{prefix}_prompts.txt"
    with open(prompts_filename, "w") as f:
        f.write(f"{SHOW_NAME} — {scene['name']} — PRODUCTION PROMPTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        for beat in production.get("beats", []):
            n = beat.get("beat_number", "?")
            dur = beat.get("duration_seconds", "?")
            f.write(f"--- BEAT {n} ({dur}s) ---\n")
            f.write(f"Dialogue:\n")
            dialogue_lines = beat.get("dialogue_lines", [])
            if not isinstance(dialogue_lines, list):
                dialogue_lines = [str(dialogue_lines)]
            for line in dialogue_lines:
                f.write(f"  {line}\n")
            f.write(f"\n[GEMINI IMAGE PROMPT — copy into Google Gemini]\n")
            f.write(f"{beat.get('gemini_image_prompt', '')}\n")
            f.write(f"\n[VEO 3 VIDEO PROMPT — copy into Google Veo 3 frames-to-video]\n")
            f.write(f"{beat.get('veo3_video_prompt', '')}\n")
            f.write("\n" + "=" * 60 + "\n\n")
    if not quiet:
        print(f"📋 Copy-paste prompts saved to: {prompts_filename}")
    return True


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────

def check_api_keys(cast_ids):
    available = []
    missing = []
    for cid in cast_ids:
        provider = CAST[cid]["provider"]
        env_key = PROVIDER_ENV_KEYS.get(provider, "")
        if env_key and api_key_set(env_key):
            available.append(cid)
        else:
            missing.append((cid, provider))
    return available, missing


def render_video_stage(opts, scene, conversation, director_notes, cast_ids,
                       prefix, audio_track):
    """One scene's video render in whichever mode opts selects. Returns
    (video_track, production) — production is the beat set a Veo render
    generated from (None in tts mode), surfaced so --produce reuses it
    instead of billing a second generation that wouldn't even match the
    rendered clips. Never raises — a dead video stage must not lose the
    outputs already saved."""
    if opts.render_mode == "vlog":
        from .vlog import render_vlog_video  # lazy: media stage
        try:
            return render_vlog_video(conversation, scene, prefix,
                                     clips_dir=opts.clips_dir), None
        except Exception as e:
            print(f"  ⚠️  Vlog rendering failed ({type(e).__name__}: {e}) — earlier outputs still saved")
            return None, None
    if opts.render_mode == "veo":
        production = generate_production_prompts(scene, conversation, director_notes, cast_ids)
        if not production:
            print("  ⚠️  Veo skipped — production beats unavailable")
            return None, None
        from .video import render_veo_video  # lazy: media stage
        try:
            return render_veo_video(production.get("beats", []), prefix), production
        except Exception as e:
            print(f"  ⚠️  Veo rendering failed ({type(e).__name__}) — earlier outputs still saved")
            return None, production
    if not audio_track:
        print("  ⚠️  Video skipped — no audio track to compose against")
        return None, None
    from .video import render_scene_video  # lazy: media stage
    try:
        return render_scene_video(conversation, prefix), None
    except Exception as e:
        print(f"  ⚠️  Video rendering failed ({type(e).__name__}) — earlier outputs still saved")
        return None, None


def run_scene(scene_id, cast_ids, opts=None, sessions=None):
    """Run an autonomous scene. Returns (conversation, director_notes)."""
    opts = opts if opts is not None else RunOptions()
    plot = opts.plot
    director_only = opts.director_only
    scene = SCENES[scene_id]
    intensity = effective_intensity(opts.intensity)
    conversation = []
    director_notes = []
    last_speaker = None
    # Standalone runs own the run summary; episode runs print it at episode wrap
    standalone = sessions is None

    # Create sessions if not provided (standalone scene run)
    if sessions is None:
        sessions = {}
        for cid in cast_ids:
            sessions[cid] = CharacterSession(cid, scene, intensity, plot, cast_ids)

    print("\n" + "=" * 60)
    print(f"🎬  {SHOW_NAME} — {scene['name']}")
    print(f"🎭  Cast: {', '.join(CAST[c]['name'] for c in cast_ids)}")
    print(f"🎯  Target lines: {opts.num_lines}")
    director_display = DIRECTOR_MODEL.split("/")[-1].split("-beta")[0].replace("-", " ").title() if director_only else DIRECTOR_MODEL
    print(f"🎥  Director: {director_display}")
    print(f"💡  Mood: {scene['mood']}")
    print("=" * 60)
    print()

    for _ in range(opts.num_lines):
        # Step 1: Director decides who speaks and how
        print(f"  🎬 Director is reading the room...", end="", flush=True)
        director_note = get_director_choice(scene, conversation, cast_ids, last_speaker, intensity, plot)
        speaker_id = director_note["next_speaker"]
        character = CAST[speaker_id]
        direction = director_note.get("direction", "?")

        print(f"\r  🎬 Director: {character['name']} → {direction}" + " " * 20)

        # Stage direction
        if director_note.get("stage_direction"):
            print(f"  📍 [{director_note['stage_direction']}]")

        # (reasoning and room energy omitted in director-only — stage direction is enough)

        # Step 2: Inject context and generate line using session
        print(f"  {character['emoji']} {character['name']} is thinking...", end="", flush=True)

        direction_text = build_direction_text(director_note)
        inject_scene_update(sessions[speaker_id], conversation, direction_text)
        text = generate_line_session(sessions[speaker_id])

        entry = {
            "name": character["name"],
            "id": speaker_id,
            "emoji": character["emoji"],
            "tagline": character["tagline"],
            "text": text,
            "direction": direction,
        }
        conversation.append(entry)
        director_notes.append(director_note)
        last_speaker = speaker_id

        if director_only:
            bar_len = random.randint(15, 35)
            print(f"\r  {character['emoji']} {character['name']}: {'█' * bar_len} [{direction}]" + " " * 20)
        else:
            print(f"\r  {character['emoji']} {character['name']}: {text}" + " " * 20)
        print()

        time.sleep(0.3)

    print("=" * 60)
    print("🎬  CUT! That's a wrap.")
    print("=" * 60)

    # Decision flow summary (director-only mode)
    if director_only and director_notes:
        print("\n  ─── Decision Flow ───")
        for i, dn in enumerate(director_notes):
            speaker_name = CAST[dn["next_speaker"]]["name"]
            d = dn.get("direction", "?")
            target = dn.get("target")
            if target and target in CAST:
                d = f"{d}:{CAST[target]['name']}"
            print(f"  {i+1:2d}. {speaker_name:10s} → {d}")

    # Debug summary
    if opts.debug:
        print("\n  [DEBUG] ── Scene Token Summary ──")
        total = 0
        for cid in cast_ids:
            if cid in sessions:
                s = sessions[cid]
                print(f"  [DEBUG] {s.character['name']:10s}: {s.line_count} lines, {s.total_tokens:,} tokens, {len(s.messages)} messages")
                total += s.total_tokens
        print(f"  [DEBUG] TOTAL: {total:,} tokens")

    save_scene_outputs(scene_id, scene, conversation, director_notes, cast_ids,
                       opts, intensity, standalone=standalone)

    return conversation, director_notes


def save_scene_outputs(scene_id, scene, conversation, director_notes, cast_ids,
                       opts, intensity, extra_json=None, standalone=True,
                       audit_extra=None):
    """Everything after a scene's dialogue loop: judge eval, script + .json
    saves, media stages, scorecard file, run summary. Shared by the classic
    loop, the --planner graph, and the --from-script render path so all
    three carry identical output guarantees. extra_json merges additional
    top-level keys (e.g. the planner's arc) into the saved .json;
    audit_extra merges into the audit block (e.g. the render path's
    generator stamp). Returns the scorecard (None unless opts.evaluate
    produced one)."""
    audit = {**AUDIT, **(audit_extra or {})}
    director_only = opts.director_only
    # pid in the prefix: two same-second runs must not overwrite each other
    # (the JSONL run log already does this — telemetry.RunLog)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"scripts/{SHOW_SLUG}_{scene_id}_{timestamp}_{os.getpid()}"

    # Judge-model eval pass — runs before the save so the saved telemetry
    # includes the judge call. Scores only on console; details go to the file.
    scorecard = None
    if opts.evaluate:
        print("\n  🧑‍⚖️ Judge is scoring the scene...", flush=True)
        scorecard = evaluate_scene(conversation, director_notes, cast_ids, intensity)
        if scorecard:
            print(f"  🧑‍⚖️ {summarize_scorecard(scorecard)}")

    # Save
    if opts.save:
        os.makedirs("scripts", exist_ok=True)
        filename = f"{prefix}.txt"
        script = format_script(scene, conversation, director_notes)
        with open(filename, "w") as f:
            f.write(script)
        if not director_only:
            print(f"\n📝 Script saved to: {filename}")

        # TTS audio — rendered before the .json save so the embedded
        # telemetry includes the TTS calls. A failure never loses the script.
        audio_track = None
        if opts.audio:
            from .audio import render_scene_audio  # lazy: optional media deps
            try:
                audio_track = render_scene_audio(conversation, prefix)
            except Exception as e:
                print(f"  ⚠️  Audio rendering failed ({type(e).__name__}) — script still saved")
            if audio_track and not director_only:
                print(f"🔊 Scene audio saved to: {audio_track}")

        # Video failure loses nothing above it. TTS mode composes over the
        # audio stage's per-line files; Veo mode generates footage from
        # production beats and needs no audio track.
        video_track = None
        production = None
        if opts.video:
            video_track, production = render_video_stage(
                opts, scene, conversation, director_notes, cast_ids,
                prefix, audio_track)
            if video_track and not director_only:
                print(f"🎞️  Scene video saved to: {video_track}")

        # Also save raw JSON for potential video editing use
        json_filename = f"{prefix}.json"
        with open(json_filename, "w") as f:
            json.dump({
                "scene_id": scene_id,  # round-trip seam: --from-script reuses it
                "scene": scene,
                **({"audio_track": audio_track} if audio_track else {}),
                **({"video_track": video_track} if video_track else {}),
                "intensity": intensity,
                "conversation": conversation,
                "director_notes": director_notes,
                **(extra_json or {}),
                "cast": {cid: {"name": CAST[cid]["name"], "emoji": CAST[cid]["emoji"],
                               "tagline": CAST[cid]["tagline"]} for cid in cast_ids},
                "generated": datetime.now().isoformat(),
                "telemetry": telemetry.get_run_log().summary(),
                "audit": audit,
            }, f, indent=2)
        if not director_only:
            print(f"📊 Raw data saved to: {json_filename}")

        # Production prompts (Gemini images + Veo 3 videos); a Veo render
        # already generated beats — reuse them
        if opts.produce:
            save_production_files(prefix, scene, conversation, director_notes,
                                  cast_ids, quiet=director_only,
                                  production=production)

        if scorecard:
            eval_filename = f"{prefix}_eval.json"
            with open(eval_filename, "w") as f:
                json.dump({
                    "scene_id": scene_id,
                    "intensity": intensity,
                    "judge_model": JUDGE_MODEL,
                    "scorecard": scorecard,
                    "generated": datetime.now().isoformat(),
                    "audit": audit,
                }, f, indent=2)
            if not director_only:
                print(f"📋 Scorecard saved to: {eval_filename}")

    # Run summary (episode runs print one combined summary at episode wrap)
    if standalone:
        telemetry.print_run_summary()

    return scorecard


def run_episode(episode_id, cast_ids, opts=None):
    """Run a full episode — multiple scenes in sequence with shared context."""
    opts = opts if opts is not None else RunOptions()
    director_only = opts.director_only
    episode = EPISODES[episode_id]
    all_conversations = []
    intensity = effective_intensity(episode.get("intensity", "high"))

    # Create persistent sessions for ALL cast members — these persist across scenes
    first_scene = SCENES[episode["scenes"][0]["scene"]]
    first_plot = episode["scenes"][0].get("plot", "")
    sessions = {}
    for cid in cast_ids:
        sessions[cid] = CharacterSession(cid, first_scene, intensity, first_plot, cast_ids)

    print("\n" + "=" * 60)
    print(f"📺  {SHOW_NAME} — {episode['name']}")
    print(f"📝  {episode['description']}")
    print(f"🎭  Cast: {', '.join(CAST[c]['name'] for c in cast_ids)}")
    print(f"🎬  Scenes: {len(episode['scenes'])}")
    director_display = DIRECTOR_MODEL.split("/")[-1].split("-beta")[0].replace("-", " ").title() if director_only else DIRECTOR_MODEL
    print(f"🎥  Director: {director_display}")
    print("=" * 60)

    for i, scene_config in enumerate(episode["scenes"]):
        scene_id = scene_config["scene"]
        lines = scene_config["lines"]
        scene = SCENES[scene_id]
        plot = scene_config.get("plot", "")

        print(f"\n{'─' * 60}")
        print(f"📺  SCENE {i + 1}/{len(episode['scenes'])}: {scene['name']}")

        # First scene always uses full cast
        if i == 0:
            scene_cast = cast_ids
            print(f"  🎬 Full cast for opening scene: {', '.join(CAST[c]['name'] for c in scene_cast)}")
        else:
            print(f"  🎬 Director is casting this scene...", end="", flush=True)
            scene_cast = get_scene_cast(scene, cast_ids, plot)
            appeared = set()
            for prev in all_conversations:
                for entry in prev["conversation"]:
                    appeared.add(entry["id"])
            missing_cast = [c for c in cast_ids if c not in appeared and c not in scene_cast]
            if missing_cast:
                scene_cast.extend(missing_cast)
            print(f"\r  🎬 Director cast: {', '.join(CAST[c]['name'] for c in scene_cast)}" + " " * 20)
        print(f"{'─' * 60}")

        # Inject scene transition into participating sessions (except first scene)
        if i > 0:
            transition = build_transition_text(scene, plot, scene_cast, intensity)
            for cid in scene_cast:
                if cid in sessions:
                    sessions[cid].inject_context(transition)
                    sessions[cid].trim_if_needed()

        # Run the scene with persistent sessions
        scene_sessions = {cid: sessions[cid] for cid in scene_cast if cid in sessions}
        # save/produce/media/eval are episode-wrap concerns — scenes inside
        # an episode must not save, render, or judge individually (the wrap
        # judges every scene once; leaving evaluate on here would double-bill
        # the judge model)
        scene_opts = replace(opts, num_lines=lines, save=False, produce=False,
                             audio=False, video=False, evaluate=False,
                             intensity=intensity, plot=plot)
        conversation, director_notes = run_scene(
            scene_id, scene_cast, scene_opts, sessions=scene_sessions,
        )
        all_conversations.append({
            "scene_id": scene_id,
            "scene": scene,
            "cast": scene_cast,
            "conversation": conversation,
            "director_notes": director_notes,
        })

        # Give every participant the lines they never saw (anything after
        # their own last line) — sessions persist across scenes but each
        # scene's conversation log starts empty.
        for cid in scene_cast:
            if cid in sessions:
                inject_scene_tail(sessions[cid], conversation)

    # Save the full episode
    print("\n" + "=" * 60)
    print(f"📺  EPISODE WRAP: {episode['name']}")
    print("=" * 60)

    save_episode_outputs(episode_id, episode, all_conversations, cast_ids,
                         opts, intensity)

    return all_conversations


def save_episode_outputs(episode_id, episode, all_conversations, cast_ids,
                         opts, intensity, extra_json=None):
    """Everything after an episode's scenes: per-scene judge eval, media
    stages, combined script/.json/scorecard files, run summary. Shared by
    the classic loop and the --planner graph. extra_json merges additional
    top-level keys into the saved .json."""
    director_only = opts.director_only
    # pid in the prefix: two same-second runs must not overwrite each other
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"scripts/{SHOW_SLUG}_episode_{episode_id}_{timestamp}_{os.getpid()}"

    # Judge-model eval pass: one call per scene, before the save so the saved
    # telemetry includes the judge calls
    scene_cards = []
    if opts.evaluate:
        print("\n  🧑‍⚖️ Judge is scoring each scene...")
        for scene_data in all_conversations:
            card = evaluate_scene(scene_data["conversation"], scene_data["director_notes"],
                                  scene_data["cast"], intensity)
            scene_cards.append({"scene_id": scene_data["scene_id"], "scorecard": card})
            scores = summarize_scorecard(card) if card else "no scorecard"
            print(f"  🧑‍⚖️ {scene_data['scene']['name']}: {scores}")

    # Combined script file FIRST — the dialogue is the money; it must be on
    # disk before the (long, interruptible) media stages run. The scene path
    # makes the same guarantee.
    if opts.save:
        os.makedirs("scripts", exist_ok=True)
        _write_episode_script(f"{prefix}.txt", episode, all_conversations,
                              cast_ids, director_only)

    # Media stages — before the .json save so the embedded telemetry
    # includes the TTS/Veo calls
    audio_tracks = []
    video_tracks = []
    productions = []
    if opts.save and opts.audio:
        from .audio import render_scene_audio  # lazy: optional media deps
        for i, scene_data in enumerate(all_conversations):
            track = None
            try:
                track = render_scene_audio(scene_data["conversation"],
                                           f"{prefix}_scene{i + 1}")
            except Exception as e:
                print(f"  ⚠️  Audio rendering failed for scene {i + 1} ({type(e).__name__})")
            audio_tracks.append(track)
            if track and not director_only:
                print(f"🔊 Scene {i + 1} audio saved to: {track}")
    if opts.save and opts.video:
        # Only tts mode composes over the audio stage's files — veo and vlog
        # bring their own audio pipelines and must not require --audio.
        if opts.render_mode == "tts" and not opts.audio:
            print("  ⚠️  Video skipped — TTS-composited video needs the audio stage")
        else:
            for i, scene_data in enumerate(all_conversations):
                vtrack, production = render_video_stage(
                    opts, scene_data["scene"], scene_data["conversation"],
                    scene_data["director_notes"], scene_data["cast"],
                    f"{prefix}_scene{i + 1}",
                    audio_tracks[i] if i < len(audio_tracks) else None)
                video_tracks.append(vtrack)
                productions.append(production)
                if vtrack and not director_only:
                    print(f"🎞️  Scene {i + 1} video saved to: {vtrack}")

    if not opts.save:
        telemetry.print_run_summary()
        return scene_cards

    # Combined JSON file
    json_filename = f"{prefix}.json"
    with open(json_filename, "w") as f:
        json.dump({
            "episode": {
                "id": episode_id,
                "name": episode["name"],
                "description": episode["description"],
            },
            "intensity": intensity,
            **({"audio_tracks": audio_tracks} if audio_tracks else {}),
            **({"video_tracks": video_tracks} if video_tracks else {}),
            **(extra_json or {}),
            "cast": {cid: {"name": CAST[cid]["name"], "emoji": CAST[cid]["emoji"],
                           "tagline": CAST[cid]["tagline"]} for cid in cast_ids},
            "scenes": all_conversations,
            "generated": datetime.now().isoformat(),
            "telemetry": telemetry.get_run_log().summary(),
            "audit": AUDIT,
        }, f, indent=2)
    if not director_only:
        print(f"📊 Full episode data saved to: {json_filename}")

    # Combined scorecard file
    if scene_cards:
        eval_filename = f"{prefix}_eval.json"
        with open(eval_filename, "w") as f:
            json.dump({
                "episode_id": episode_id,
                "intensity": intensity,
                "judge_model": JUDGE_MODEL,
                "scenes": scene_cards,
                "generated": datetime.now().isoformat(),
                "audit": AUDIT,
            }, f, indent=2)
        if not director_only:
            print(f"📋 Scorecards saved to: {eval_filename}")

    # Production prompts, one pair of files per scene (Veo renders already
    # generated beats per scene — reuse instead of re-billing)
    if opts.produce:
        for i, scene_data in enumerate(all_conversations):
            save_production_files(
                f"{prefix}_scene{i + 1}",
                scene_data["scene"], scene_data["conversation"],
                scene_data["director_notes"], scene_data["cast"],
                quiet=director_only,
                production=productions[i] if i < len(productions) else None,
            )

    telemetry.print_run_summary()

    return scene_cards


def _write_episode_script(path, episode, all_conversations, cast_ids, director_only):
    with open(path, "w") as f:
        f.write(f"{SHOW_NAME} — {episode['name']}\n")
        f.write(f"{episode['description']}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Cast: {', '.join(CAST[c]['name'] for c in cast_ids)}\n")
        f.write("=" * 60 + "\n\n")
        for i, scene_data in enumerate(all_conversations):
            scene = scene_data["scene"]
            f.write(f"{'─' * 60}\n")
            f.write(f"SCENE {i + 1}: {scene['name']}\n")
            f.write(f"Mood: {scene['mood']}\n")
            f.write(f"{'─' * 60}\n\n")
            for entry in scene_data["conversation"]:
                f.write(f"{entry['emoji']} {entry['name'].upper()} ({entry['tagline']})\n")
                f.write(f"  {entry['text']}\n\n")
    if not director_only:
        print(f"📝 Full episode script saved to: {path}")
