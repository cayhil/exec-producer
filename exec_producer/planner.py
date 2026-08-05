"""LangGraph scene planner — the pacing layer above the director.

The documented failure mode of the classic loop: per-turn directing is
myopic, so dialogue escalates instantly with zero filler (the --eval judge
reliably scores pacing low — that is the measurement baseline). Behind
--planner, a planner node writes the scene's turn-by-turn emotional arc
BEFORE any dialogue, and the director follows the plan turn by turn.

LangGraph orchestrates THIS layer only. Nodes are plain functions calling
the existing engine/llm helpers — LiteLLM stays the single text-model access
layer, no LangChain model wrappers. Graph state is deliberately all-JSON
(sessions serialized to message dicts) so SqliteSaver checkpointing is real:
an interrupted episode resumes from its last checkpoint via --thread without
re-billing the turns that already ran. Re-rolling from a midpoint is the
same machinery — checkpoint history plus a forked thread; see the LangGraph
get_state_history docs.

The graph:

    START → cast_scene → plan_scene → direct_turn → speak
                              ▲            ▲          │ continue
                              │            └──────────┤
                              │ next scene            │ scene done
                          wrap_scene ◄────────────────┘
                              │ episode done → END

Dialogue results flow into engine.save_scene_outputs / save_episode_outputs,
so planned runs carry the exact same output guarantees (audit block,
telemetry embed, media stages) as classic runs.
"""

import importlib.util
import json
import os
import random
import time
from datetime import datetime
from typing import TypedDict

from . import engine
from .config import load_planner_prompt
from .engine import RunOptions
from .errors import ThreadResumeError
from .llm import call_model_simple, parse_json_response

CHECKPOINT_DB = "scripts/planner_state.sqlite"


class PlannerState(TypedDict, total=False):
    scene_queue: list        # [{"scene_id", "lines", "plot"}]
    scene_index: int
    cast_ids: list           # the full run cast
    scene_cast: list         # current scene's cast
    intensity: str
    director_only: bool
    plan: list               # current scene's turn-by-turn arc
    turn_index: int
    conversation: list
    director_notes: list
    current_note: dict
    last_speaker: str
    sessions: dict           # cast_id -> CharacterSession.to_state()
    completed_scenes: list


def _rebuild(cast_id, session_state):
    return engine.CharacterSession.from_messages(cast_id, **session_state)


# ─────────────────────────────────────────────
# Graph nodes — plain functions over JSON state
# ─────────────────────────────────────────────

def cast_scene(state):
    """Casting, session setup, and transition injection for the next scene.
    Mirrors run_episode's rules: full cast for the opener, director casting
    plus forced-cast merge afterwards."""
    i = state["scene_index"]
    cfg = state["scene_queue"][i]
    scene = engine.SCENES[cfg["scene_id"]]
    plot = cfg.get("plot", "")
    cast_ids = state["cast_ids"]
    sessions = dict(state.get("sessions") or {})

    print(f"\n{'─' * 60}")
    print(f"📺  SCENE {i + 1}/{len(state['scene_queue'])}: {scene['name']}")

    if not sessions:
        for cid in cast_ids:
            live = engine.CharacterSession(cid, scene, state["intensity"], plot, cast_ids)
            sessions[cid] = live.to_state()

    if i == 0:
        scene_cast = list(cast_ids)
        print(f"  🎬 Full cast for opening scene: {', '.join(engine.CAST[c]['name'] for c in scene_cast)}")
    else:
        print("  🎬 Director is casting this scene...", end="", flush=True)
        scene_cast = engine.get_scene_cast(scene, cast_ids, plot)
        appeared = {e["id"] for sc in state.get("completed_scenes", [])
                    for e in sc["conversation"]}
        scene_cast.extend([c for c in cast_ids
                           if c not in appeared and c not in scene_cast])
        print(f"\r  🎬 Director cast: {', '.join(engine.CAST[c]['name'] for c in scene_cast)}" + " " * 20)

        transition = engine.build_transition_text(scene, plot, scene_cast,
                                                  state["intensity"])
        for cid in scene_cast:
            if cid in sessions:
                live = _rebuild(cid, sessions[cid])
                live.inject_context(transition)
                live.trim_if_needed()
                sessions[cid] = live.to_state()

    return {"scene_cast": scene_cast, "sessions": sessions, "conversation": [],
            "director_notes": [], "turn_index": 0, "last_speaker": ""}


def plan_scene(state):
    """One planner call before any dialogue: the scene's turn-by-turn arc.
    Degrades to an empty plan — the directors then behave exactly as the
    classic, plan-less loop."""
    cfg = state["scene_queue"][state["scene_index"]]
    scene = engine.SCENES[cfg["scene_id"]]
    lines = cfg["lines"]
    plot_text = f"\nPLOT BEATS (must happen during the scene):\n{cfg.get('plot', '')}\n" if cfg.get("plot") else ""
    cast_line = ", ".join(f"{engine.CAST[c]['name']} ({engine.CAST[c]['tagline']})"
                          for c in state["scene_cast"])

    prompt = f"""SCENE: {scene['name']}
PREMISE: {scene['prompt']}
MOOD: {scene['mood']}
{plot_text}
{engine.INTENSITY_GUIDE.get(state['intensity'], '')}

CAST IN THIS SCENE: {cast_line}

Write the emotional arc for a {lines}-turn scene. Output JSON only, with exactly {lines} entries in "turns"."""

    print("  🧠 Planner is writing the arc...", end="", flush=True)
    for attempt in range(3):
        # API errors are already retried inside call_model — don't retry them
        # again here, just degrade to plan-less directing.
        try:
            raw = call_model_simple(engine.DIRECTOR_MODEL, load_planner_prompt(),
                                    prompt, max_tokens=2000, caller="planner")
        except Exception as e:
            print(f"\r  ⚠️  Planner unavailable ({type(e).__name__}) — directing without a plan")
            return {"plan": []}
        try:
            turns = parse_json_response(raw).get("turns", [])
            if isinstance(turns, list) and turns and all(isinstance(t, dict) for t in turns):
                arc = " → ".join(str(t.get("phase", "?")) for t in turns)
                print(f"\r  🧠 Arc ({len(turns)} turns): {arc}"[:160] + " " * 10)
                return {"plan": turns}
        except (json.JSONDecodeError, AttributeError):
            time.sleep(0.5)
    print("\r  ⚠️  Planner output unusable — directing without a plan")
    return {"plan": []}


def _plan_note(plan, turn_index, total_lines):
    if turn_index >= len(plan):
        return ""
    turn = plan[turn_index]
    return (f"Turn {turn_index + 1}/{total_lines} — phase: {turn.get('phase', '?')}"
            f" | energy: {turn.get('energy', '?')}\n{turn.get('guidance', '')}")


def direct_turn(state):
    cfg = state["scene_queue"][state["scene_index"]]
    scene = engine.SCENES[cfg["scene_id"]]
    plan = state.get("plan") or []
    ti = state["turn_index"]

    print("  🎬 Director is reading the room...", end="", flush=True)
    note = engine.get_director_choice(
        scene, state["conversation"], state["scene_cast"],
        state.get("last_speaker") or None, state["intensity"], cfg.get("plot", ""),
        plan_note=_plan_note(plan, ti, cfg["lines"]))

    character = engine.CAST[note["next_speaker"]]
    phase = plan[ti].get("phase") if ti < len(plan) else None
    phase_tag = f" [{phase}]" if phase else ""
    print(f"\r  🎬 Director: {character['name']} → {note.get('direction', '?')}{phase_tag}" + " " * 20)
    if note.get("stage_direction"):
        print(f"  📍 [{note['stage_direction']}]")
    return {"current_note": note}


def speak(state):
    note = state["current_note"]
    speaker_id = note["next_speaker"]
    character = engine.CAST[speaker_id]
    sessions = dict(state["sessions"])

    print(f"  {character['emoji']} {character['name']} is thinking...", end="", flush=True)
    live = _rebuild(speaker_id, sessions[speaker_id])
    engine.inject_scene_update(live, state["conversation"], engine.build_direction_text(note))
    text = engine.generate_line_session(live)
    sessions[speaker_id] = live.to_state()

    entry = {"name": character["name"], "id": speaker_id, "emoji": character["emoji"],
             "tagline": character["tagline"], "text": text,
             "direction": note.get("direction", "?")}
    if state.get("director_only"):
        print(f"\r  {character['emoji']} {character['name']}: {'█' * random.randint(15, 35)} [{entry['direction']}]" + " " * 20)
    else:
        print(f"\r  {character['emoji']} {character['name']}: {text}" + " " * 20)
    print()
    time.sleep(0.3)

    return {"conversation": state["conversation"] + [entry],
            "director_notes": state["director_notes"] + [note],
            "sessions": sessions,
            "turn_index": state["turn_index"] + 1,
            "last_speaker": speaker_id}


def wrap_scene(state):
    """Scene complete: archive it (with its plan, for the audit trail) and
    give every participant the lines they never saw."""
    cfg = state["scene_queue"][state["scene_index"]]
    sessions = dict(state["sessions"])
    for cid in state["scene_cast"]:
        if cid in sessions:
            live = _rebuild(cid, sessions[cid])
            engine.inject_scene_tail(live, state["conversation"])
            sessions[cid] = live.to_state()

    completed = state.get("completed_scenes", []) + [{
        "scene_id": cfg["scene_id"],
        "scene": engine.SCENES[cfg["scene_id"]],
        "cast": state["scene_cast"],
        "conversation": state["conversation"],
        "director_notes": state["director_notes"],
        "plan": state.get("plan", []),
    }]
    print("=" * 60)
    print("🎬  CUT! That's a wrap.")
    print("=" * 60)
    return {"completed_scenes": completed, "sessions": sessions,
            "scene_index": state["scene_index"] + 1}


def route_after_speak(state):
    cfg = state["scene_queue"][state["scene_index"]]
    return "continue" if state["turn_index"] < cfg["lines"] else "wrap"


def route_after_wrap(state):
    return "next_scene" if state["scene_index"] < len(state["scene_queue"]) else "end"


# ─────────────────────────────────────────────
# Graph assembly + entry points
# ─────────────────────────────────────────────

def build_graph():
    from langgraph.graph import StateGraph, START, END  # lazy: planner extra
    g = StateGraph(PlannerState)
    g.add_node("cast_scene", cast_scene)
    g.add_node("plan_scene", plan_scene)
    g.add_node("direct_turn", direct_turn)
    g.add_node("speak", speak)
    g.add_node("wrap_scene", wrap_scene)
    g.add_edge(START, "cast_scene")
    g.add_edge("cast_scene", "plan_scene")
    g.add_edge("plan_scene", "direct_turn")
    g.add_edge("direct_turn", "speak")
    g.add_conditional_edges("speak", route_after_speak,
                            {"continue": "direct_turn", "wrap": "wrap_scene"})
    g.add_conditional_edges("wrap_scene", route_after_wrap,
                            {"next_scene": "cast_scene", "end": END})
    return g


def _run_graph(initial_state, thread_id, expect_scene_ids=None):
    """Invoke with durable checkpointing. initial_state=None resumes the
    thread from its last checkpoint (the --thread flag) — after verifying
    the checkpoint actually exists (a typo'd id would otherwise start the
    graph on empty state and crash mid-node) and that its scene queue is the
    invoked run's. The id-prefix guard alone can't guarantee that: scene ids
    are valid inside each other ('hot_tub' prefixes 'hot_tub_2'), so a
    mismatched resume would complete the wrong run at full cost and save it
    under this run's identity."""
    from langgraph.checkpoint.sqlite import SqliteSaver  # lazy: planner extra
    os.makedirs("scripts", exist_ok=True)
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as saver:
        graph = build_graph().compile(checkpointer=saver)
        config = {"configurable": {"thread_id": thread_id},
                  "recursion_limit": 1000}
        if initial_state is None:
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise ThreadResumeError(
                    f"unknown planner thread '{thread_id}' — no checkpoint in {CHECKPOINT_DB}")
            checkpointed = [c.get("scene_id")
                            for c in snapshot.values.get("scene_queue", [])]
            if expect_scene_ids is not None and checkpointed != list(expect_scene_ids):
                raise ThreadResumeError(
                    f"thread '{thread_id}' was checkpointed for scenes "
                    f"{checkpointed}, not this run's {list(expect_scene_ids)} — "
                    "resume it with the same --scene/--episode it was started with")
            if not snapshot.next:
                print("  ℹ️  This thread already completed — re-saving its outputs "
                      "without re-running any dialogue")
        return graph.invoke(initial_state, config)


def _require_thread_prefix(thread_id, expected_prefix):
    """A thread encodes the run that created it; resuming it under a
    different scene/episode would complete the checkpointed run at full
    cost and then save it under the wrong identity (or drop scenes)."""
    if not thread_id.startswith(expected_prefix):
        raise ThreadResumeError(
            f"thread '{thread_id}' was not created by this command — resume it "
            f"with the same --scene/--episode it was started with "
            f"(expected a '{expected_prefix}…' id)")


def _new_thread_id(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def _banner(title, thread_id, resume):
    print("\n" + "=" * 60)
    print(f"🧠  PLANNED RUN — {title}")
    print(f"🧵  Thread: {thread_id}{'  (resuming from checkpoint)' if resume else ''}")
    print(f"    Resume an interrupted run: --planner --thread {thread_id}")
    print("=" * 60)


def run_scene_planned(scene_id, cast_ids, opts=None, thread_id=None):
    """--planner scene run: the graph drives the dialogue, then the exact
    same output path as the classic runner."""
    opts = opts if opts is not None else RunOptions()
    resume = thread_id is not None
    if resume:
        _require_thread_prefix(thread_id, f"scene_{scene_id}_")
    thread_id = thread_id or _new_thread_id(f"scene_{scene_id}")
    _banner(engine.SCENES[scene_id]["name"], thread_id, resume)

    initial = None if resume else {
        "scene_queue": [{"scene_id": scene_id, "lines": opts.num_lines,
                         "plot": opts.plot}],
        "scene_index": 0, "cast_ids": list(cast_ids),
        "intensity": engine.effective_intensity(opts.intensity),
        "director_only": opts.director_only, "sessions": {},
        "completed_scenes": [],
    }
    final = _run_graph(initial, thread_id, expect_scene_ids=[scene_id])
    result = final["completed_scenes"][0]
    engine.save_scene_outputs(
        scene_id, result["scene"], result["conversation"], result["director_notes"],
        result["cast"], opts, final["intensity"],
        extra_json={"planner": {"plan": result["plan"], "thread_id": thread_id}},
        standalone=True)
    return result["conversation"], result["director_notes"]


def run_episode_planned(episode_id, cast_ids, opts=None, thread_id=None):
    """--planner episode run: per-scene arcs, cross-scene session continuity,
    checkpointed so --thread resumes mid-episode."""
    opts = opts if opts is not None else RunOptions()
    episode = engine.EPISODES[episode_id]
    resume = thread_id is not None
    if resume:
        _require_thread_prefix(thread_id, f"episode_{episode_id}_")
    thread_id = thread_id or _new_thread_id(f"episode_{episode_id}")
    _banner(episode["name"], thread_id, resume)

    initial = None if resume else {
        "scene_queue": [{"scene_id": sc["scene"], "lines": sc["lines"],
                         "plot": sc.get("plot", "")} for sc in episode["scenes"]],
        "scene_index": 0, "cast_ids": list(cast_ids),
        "intensity": engine.effective_intensity(episode.get("intensity", "high")),
        "director_only": opts.director_only, "sessions": {},
        "completed_scenes": [],
    }
    final = _run_graph(initial, thread_id,
                       expect_scene_ids=[sc["scene"] for sc in episode["scenes"]])
    all_conversations = final["completed_scenes"]

    print("\n" + "=" * 60)
    print(f"📺  EPISODE WRAP: {episode['name']}")
    print("=" * 60)
    engine.save_episode_outputs(
        episode_id, episode, all_conversations, cast_ids, opts, final["intensity"],
        extra_json={"planner": {"thread_id": thread_id}})
    return all_conversations


def preflight():
    """--planner checks, before any LLM spend."""
    problems = []
    for module, package in (("langgraph", "langgraph"),
                            ("langgraph.checkpoint.sqlite", "langgraph-checkpoint-sqlite")):
        try:
            present = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            present = False
        if not present:
            problems.append(f"{package} not installed — run: uv sync --extra planner")
            break
    try:
        # Fails fast if the show lacks the arc-writer prompt (a planned run
        # would otherwise silently degrade to plan-less directing, per
        # scene, after the user explicitly paid for --planner) — and puts
        # its hash into the audit manifest before any output is written.
        load_planner_prompt()
    except FileNotFoundError:
        problems.append("prompts/planner.txt missing from the show directory — "
                        "the planner needs its arc-writer prompt (see CREATING_A_SHOW.md)")
    return problems
