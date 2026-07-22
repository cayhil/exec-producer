"""CLI entry point and interactive mode."""

import argparse
import os
import random
import sys

from .config import (AUDIT, CAST, DIRECTOR_MODEL, EPISODES, PROVIDER_ENV_KEYS,
                     SCENES, SHOW_NAME)
from .engine import RunOptions, check_api_keys, run_episode, run_scene
from .errors import ThreadResumeError
from .telemetry import setup_logging, start_run


def _director_provider():
    """Provider for the director model string. Bare names (no prefix) are
    resolved through LiteLLM's registry — defaulting to openai would demand
    the wrong key for a bare Anthropic model."""
    if "/" in DIRECTOR_MODEL:
        return DIRECTOR_MODEL.split("/", 1)[0]
    try:
        import litellm
        return litellm.get_llm_provider(DIRECTOR_MODEL)[1]
    except Exception:
        return "openai"


def director_key_or_die():
    """The director is the first spend of every run — fail before it, not
    after N degraded random-pick 'directing' calls."""
    provider = _director_provider()
    env_key = PROVIDER_ENV_KEYS.get(provider, "")
    if not env_key or not os.environ.get(env_key):
        need = env_key or f"a provider_env_keys entry for '{provider}'"
        print(f"❌ Director model {DIRECTOR_MODEL} needs {need} set in .env")
        sys.exit(1)


def interactive_mode():
    """Demo shell: scene runs only (episodes, media stages, and the planner
    are flag-driven — see --help). Scenes run in one process, so each saved
    .json embeds the telemetry summary cumulative to that point."""
    print("\n" + "=" * 60)
    print(f"🔴  {' '.join(SHOW_NAME)}  🔴".center(60))
    print("── AUTONOMOUS EDITION ──".center(60))
    print("UNSCRIPTED  •  UNFILTERED  •  DIRECTOR-DRIVEN".center(60))
    print("=" * 60)

    director_key_or_die()

    # Check cast
    all_ids = list(CAST.keys())
    available, missing = check_api_keys(all_ids)

    print("CAST STATUS:")
    for cid in all_ids:
        c = CAST[cid]
        status = "✅" if cid in available else "❌"
        via = f"{c['model']}"
        print(f"  {status} {c['emoji']} {c['name']:10s} {via}")

    if missing:
        print(f"\n  ⚠️  Missing keys: {', '.join(f'{n} ({p})' for n, p in missing)}")

    if len(available) < 2:
        print("\n❌ Need at least 2 APIs. Set keys and try again.")
        sys.exit(1)

    # Cast selection
    print(f"\nAvailable: {', '.join(available)}")
    inp = input("Cast (comma-separated, or 'all'): ").strip()
    if inp.lower() in ("all", ""):
        cast_ids = available
    else:
        cast_ids = [c.strip().lower() for c in inp.split(",") if c.strip().lower() in available]
        if len(cast_ids) < 2:
            cast_ids = available

    # Scene selection
    print("\nSCENES:")
    scene_ids = list(SCENES.keys())
    for i, sid in enumerate(scene_ids):
        print(f"  [{i+1:2d}] {SCENES[sid]['name']}")

    inp = input(f"\nScene (1-{len(scene_ids)}): ").strip()
    try:
        scene_id = scene_ids[int(inp) - 1]
    except (ValueError, IndexError):
        scene_id = random.choice(scene_ids)
        print(f"Random pick: {SCENES[scene_id]['name']}")

    # Lines
    inp = input("How many lines? (5-30, default 15): ").strip()
    try:
        num_lines = max(5, min(30, int(inp)))
    except ValueError:
        num_lines = 15

    # Intensity
    inp = input("Intensity (low/medium/high/maximum, default high): ").strip().lower()
    intensity = inp if inp in ("low", "medium", "high", "maximum") else "high"

    # Production prompts
    inp = input("Generate production prompts for Gemini/Veo 3? (y/n, default n): ").strip()
    produce = inp.lower() == "y"

    run_scene(scene_id, cast_ids,
              RunOptions(num_lines=num_lines, produce=produce, intensity=intensity))

    while True:
        again = input("\n🎬 Another scene? (y/n): ").strip().lower()
        if again != "y":
            print("\n👋 Season wrap. ✌️\n")
            break
        print("\nSCENES:")
        for i, sid in enumerate(scene_ids):
            print(f"  [{i+1:2d}] {SCENES[sid]['name']}")
        inp = input(f"Scene (1-{len(scene_ids)}): ").strip()
        try:
            scene_id = scene_ids[int(inp) - 1]
        except (ValueError, IndexError):
            scene_id = random.choice(scene_ids)
        run_scene(scene_id, cast_ids,
                  RunOptions(num_lines=num_lines, produce=produce, intensity=intensity))


def media_preflight_or_die(cast_ids, audio=False, video=False, render_mode="tts"):
    """--audio/--video validation before any LLM money is spent: API keys,
    ffmpeg capabilities, SDK installs, and voice coverage for this run's
    cast. Collects every problem instead of stopping at the first."""
    problems = []
    if audio:
        from .audio import preflight as audio_preflight  # lazy imports:
        problems += audio_preflight(cast_ids)            # keep non-media startup light
    if video and render_mode == "veo":
        from .video import veo_preflight
        problems += veo_preflight()
    elif video:
        from .video import preflight as video_preflight
        problems += video_preflight()
    if problems:
        for p in problems:
            print(f"❌ {p}")
        sys.exit(1)


def planner_preflight_or_die():
    from .planner import preflight  # lazy: keeps non-planner startup light
    problems = preflight()
    if problems:
        for p in problems:
            print(f"❌ {p}")
        sys.exit(1)


def resolve_cast(cast_arg):
    """Turn a --cast argument into validated cast ids (all cast when absent)."""
    if not cast_arg:
        return list(CAST.keys())
    cast_ids = [c.strip().lower() for c in cast_arg.split(",") if c.strip()]
    unknown = [c for c in cast_ids if c not in CAST]
    if unknown:
        print(f"❌ Unknown cast member(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(CAST.keys())}")
        sys.exit(1)
    return cast_ids


def main():
    parser = argparse.ArgumentParser(description="EXECUTIVE PRODUCER — Autonomous Reality Show Engine")
    parser.add_argument("--scene", type=str)
    parser.add_argument("--cast", type=str)
    parser.add_argument("--lines", type=int, default=15)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--produce", action="store_true",
                        help="Generate Gemini image prompts + Veo 3 video prompts for production")
    parser.add_argument("--episode", type=str,
                        help="Run a full episode (multiple scenes)")
    parser.add_argument("--list-episodes", action="store_true",
                        help="List available episodes")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose output: token counts, message counts, timing per API call")
    parser.add_argument("--director-only", action="store_true",
                        help="Redact dialogue, show only director decisions (for demos/sharing)")
    parser.add_argument("--intensity", type=str, default="high",
                        choices=["low", "medium", "high", "maximum"],
                        help="Intensity level (default: high). Episodes override this.")
    parser.add_argument("--eval", action="store_true",
                        help="Judge-model eval pass: score the script (format, voice, pacing) and save a scorecard")
    parser.add_argument("--audio", action="store_true",
                        help="Render each dialogue line with its character's persistent TTS voice "
                             "(data/voices.json) and stitch a per-scene track "
                             "(needs ELEVENLABS_API_KEY, ffmpeg, uv sync --extra audio)")

    parser.add_argument("--video", action="store_true",
                        help="Compose a 9:16 MP4 per scene: character name cards + dialogue "
                             "text + TTS audio. Implies --audio (in the default tts mode).")
    parser.add_argument("--render-mode", choices=["tts", "veo"], default="tts",
                        help="Video source: 'tts' composes cards over the TTS track (default — "
                             "persistent voices, editable, cheap); 'veo' generates real footage "
                             "per beat via Veo (needs GEMINI_API_KEY + uv sync --extra veo; "
                             "expensive, voices drift between clips)")
    parser.add_argument("--planner", action="store_true",
                        help="LangGraph pacing layer: a planner writes the scene's turn-by-turn "
                             "emotional arc before dialogue and the director follows it "
                             "(needs uv sync --extra planner)")
    parser.add_argument("--thread", type=str, default=None,
                        help="Resume an interrupted --planner run from its last checkpoint "
                             "(the thread id is printed when a planned run starts)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Show directory to load instead of data/ (applied at startup "
                             "by the CLI bootstrap; also settable as EXEC_PRODUCER_DATA_DIR)")

    args = parser.parse_args()
    if args.thread and not args.planner:
        print("❌ --thread resumes a --planner run — add --planner")
        sys.exit(1)
    for flag in ("audio", "video"):
        if getattr(args, flag) and args.no_save:
            print(f"❌ --{flag} needs saved outputs — drop --no-save")
            sys.exit(1)
        if getattr(args, flag) and args.director_only:
            print(f"❌ --{flag} renders full dialogue, which defeats --director-only redaction")
            sys.exit(1)
    if args.render_mode == "veo" and not args.video:
        print("❌ --render-mode veo only makes sense with --video")
        sys.exit(1)
    if args.produce and args.no_save:
        print("❌ --produce writes production files — drop --no-save")
        sys.exit(1)
    if args.video and args.render_mode == "tts" and not args.audio:
        print("ℹ️  --video implies --audio — enabling the audio stage")
        args.audio = True
    setup_logging(debug=args.debug)
    # Run-log wall clock starts here; the JSONL file (with its audit header)
    # is created on the first LLM call
    start_run(header={"audit": AUDIT})

    if args.list:
        print("\nSCENES:")
        for sid, s in SCENES.items():
            print(f"  {sid:20s} {s['name']:30s} ({s['mood']})")
        sys.exit(0)

    if args.list_episodes:
        print("\nEPISODES:")
        for eid, ep in EPISODES.items():
            scenes = " → ".join(s["scene"] for s in ep["scenes"])
            total_lines = sum(s["lines"] for s in ep["scenes"])
            print(f"  {eid:20s} {ep['name']}")
            print(f"  {'':20s} {ep['description']}")
            print(f"  {'':20s} Scenes: {scenes} ({total_lines} total lines)")
            print()
        sys.exit(0)

    if args.episode:
        if args.episode not in EPISODES:
            print(f"❌ Unknown episode: {args.episode}")
            print(f"Available: {', '.join(EPISODES.keys())}")
            sys.exit(1)
        cast_ids = resolve_cast(args.cast)
        available, _ = check_api_keys(cast_ids)
        if len(available) < 2:
            print("❌ Need at least 2 API keys.")
            sys.exit(1)
        director_key_or_die()
        if args.audio or args.video:
            media_preflight_or_die(available, audio=args.audio, video=args.video,
                                   render_mode=args.render_mode)
        opts = RunOptions(save=not args.no_save,
                          produce=args.produce, debug=args.debug,
                          evaluate=args.eval, director_only=args.director_only,
                          audio=args.audio, video=args.video,
                          render_mode=args.render_mode)
        if args.planner:
            planner_preflight_or_die()
            from .planner import run_episode_planned
            try:
                run_episode_planned(args.episode, available, opts, thread_id=args.thread)
            except ThreadResumeError as e:
                print(f"❌ {e}")
                sys.exit(1)
        else:
            run_episode(args.episode, available, opts)
    elif args.scene:
        if args.scene not in SCENES:
            print(f"❌ Unknown scene: {args.scene}")
            print(f"Available: {', '.join(SCENES.keys())}")
            sys.exit(1)
        cast_ids = resolve_cast(args.cast)
        available, _ = check_api_keys(cast_ids)
        if len(available) < 2:
            print("❌ Need at least 2 API keys.")
            sys.exit(1)
        director_key_or_die()
        if args.audio or args.video:
            media_preflight_or_die(available, audio=args.audio, video=args.video,
                                   render_mode=args.render_mode)
        opts = RunOptions(num_lines=args.lines, save=not args.no_save,
                          produce=args.produce, debug=args.debug,
                          director_only=args.director_only,
                          intensity=args.intensity, evaluate=args.eval,
                          audio=args.audio, video=args.video,
                          render_mode=args.render_mode)
        if args.planner:
            planner_preflight_or_die()
            from .planner import run_scene_planned
            try:
                run_scene_planned(args.scene, available, opts, thread_id=args.thread)
            except ThreadResumeError as e:
                print(f"❌ {e}")
                sys.exit(1)
        else:
            run_scene(args.scene, available, opts)
    else:
        # Interactive mode is a scene-only demo shell that builds its own
        # RunOptions — stage flags would be silently dropped (and billed for
        # nothing), so reject them here instead.
        dropped = [name for name, on in (
            ("--audio", args.audio), ("--video", args.video),
            ("--eval", args.eval), ("--planner", args.planner),
            ("--produce", args.produce), ("--director-only", args.director_only),
        ) if on]
        if dropped:
            print(f"❌ {', '.join(dropped)} require --scene or --episode "
                  "(interactive mode is a scene-only demo shell)")
            sys.exit(1)
        interactive_mode()


if __name__ == "__main__":
    main()
