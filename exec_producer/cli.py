"""CLI entry point and interactive mode."""

import argparse
import json
import random
import sys
from pathlib import Path

from .config import (AUDIT, CAST, DIRECTOR_MODEL, EPISODES, PROVIDER_ENV_KEYS,
                     SCENES, SHOW_NAME)
from .engine import RunOptions, check_api_keys, run_episode, run_scene
from .errors import ConfigError, ThreadResumeError
from .llm import api_key_set
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
    if not env_key or not api_key_set(env_key):
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

    # One-character shows are solo: one cast key + the director key suffice.
    if len(available) < (1 if len(CAST) == 1 else 2):
        print("\n❌ Need at least 2 APIs. Set keys and try again." if len(CAST) > 1
              else "\n❌ Need the cast member's API key. Set it and try again.")
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


def media_preflight_or_die(cast_ids, audio=False, video=False, render_mode="tts",
                           clips_dir=""):
    """--audio/--video validation before any LLM money is spent: API keys,
    ffmpeg capabilities, SDK installs, and voice coverage for this run's
    cast. Collects every problem instead of stopping at the first."""
    problems = []
    if audio:
        from .audio import preflight as audio_preflight  # lazy imports:
        problems += audio_preflight(cast_ids)            # keep non-media startup light
    if video and render_mode == "vlog":
        from .vlog import preflight as vlog_preflight
        # BYO clips = no Veo spend, so the API-route checks don't apply
        problems += vlog_preflight(cast_ids, api_route=not clips_dir)
    elif video and render_mode == "veo":
        from .video import veo_preflight
        problems += veo_preflight()
    elif video:
        from .video import preflight as video_preflight
        problems += video_preflight()
    # audio + vlog both check the voice pipeline — report each problem once
    problems = list(dict.fromkeys(problems))
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


def cast_keys_or_die(available, missing=(), solo=False):
    """Key gate. An ensemble run needs two cast keys; a solo run needs one
    (plus the director key, checked separately). Solo = --solo, or a show
    whose cast.json has exactly one character — nothing else is possible
    there, so no flag is required."""
    needed = 1 if (solo or len(CAST) == 1) else 2
    if len(available) < needed:
        print("❌ Need the cast member's API key." if needed == 1
              else f"❌ Need at least 2 API keys ({len(available)} cast available).")
        for cid, provider in missing:
            env_key = PROVIDER_ENV_KEYS.get(provider)
            print(f"     - {cid} needs {env_key or f'a provider_env_keys entry for {provider!r}'}")
        print("   Copy .env.example to .env and fill in what you have.")
        sys.exit(1)


def solo_cast_or_die(cast_ids):
    """--solo must resolve to exactly one cast member."""
    if len(cast_ids) != 1:
        print("❌ --solo runs exactly one cast member — pick them with --cast <id>")
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
    parser.add_argument("--render-mode", choices=["tts", "veo", "vlog"], default="tts",
                        help="Video source: 'tts' composes cards over the TTS track (default — "
                             "persistent voices, editable, cheap); 'veo' generates real footage "
                             "per beat via Veo (needs GEMINI_API_KEY + uv sync --extra veo; "
                             "expensive, voices drift between clips); 'vlog' is all-generated "
                             "solo-host footage with the speech-to-speech voice swap — "
                             "persistent voice AND real footage (uv sync --extra vlog)")
    parser.add_argument("--clips-dir", type=str, default="", metavar="DIR",
                        help="Vlog bring-your-own-clips route: a folder of beat_<n>.mp4 files "
                             "you generated externally (e.g. the Veo web UI — subscription "
                             "quota, no API cost). Validated against the clip plan before any "
                             "ElevenLabs spend. Requires --from-script --video --render-mode vlog")
    parser.add_argument("--dry-run", action="store_true",
                        help="Vlog only: print the clip plan (count, durations, on-camera vs "
                             "voiceover split, ESTIMATED cost) and write the per-beat prompt "
                             "pack, then exit — nothing is generated, nothing is billed")
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
    parser.add_argument("--solo", action="store_true",
                        help="Solo run: exactly one cast member, so one cast key + the "
                             "director key suffice. A one-character show is solo "
                             "automatically; on a bigger cast pick the member with --cast <id>")
    parser.add_argument("--from-script", type=str, default=None, metavar="SCENE_JSON",
                        help="Render an externally-written scene .json (the engine's own "
                             "saved shape) through the downstream stages only — .txt, --eval, "
                             "--audio, --video, --produce. Makes no generation calls and needs "
                             "no director key (except --produce, which bills the director "
                             "model). Intensity comes from the file. "
                             "Sugar: exec-producer render <file>")

    # `exec-producer render <scene.json>` is sugar for --from-script. The
    # token may follow leading flags (--data-dir X render f.json) — it is
    # only skipped when it's the VALUE of a value-taking option, so
    # `--scene render` still means a scene literally named "render".
    value_flags = {"--scene", "--cast", "--lines", "--episode", "--intensity",
                   "--render-mode", "--thread", "--data-dir", "--from-script",
                   "--clips-dir"}
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == "render" and (i == 0 or argv[i - 1] not in value_flags):
            argv = argv[:i] + ["--from-script"] + argv[i + 1:]
            break

    args = parser.parse_args(argv)
    if args.lines < 1:
        # The classic loop would save an empty script; the planner graph
        # would still direct-and-bill one line (plan → direct → speak runs
        # before the first turn-count check). Reject instead of doing either.
        print("❌ --lines must be at least 1")
        sys.exit(1)
    if args.thread and not args.planner:
        print("❌ --thread resumes a --planner run — add --planner")
        sys.exit(1)
    if args.from_script:
        # Render is downstream-only: flags that shape GENERATION are a
        # contradiction the user should hear about, not a silent no-op.
        for flag in ("scene", "episode", "planner", "cast", "solo",
                     "no_save", "director_only"):
            if getattr(args, flag):
                print(f"❌ --{flag.replace('_', '-')} doesn't compose with --from-script — "
                      "the script is already written; render runs only downstream stages")
                sys.exit(1)
    for flag in ("audio", "video"):
        if getattr(args, flag) and args.no_save:
            print(f"❌ --{flag} needs saved outputs — drop --no-save")
            sys.exit(1)
        if getattr(args, flag) and args.director_only:
            print(f"❌ --{flag} renders full dialogue, which defeats --director-only redaction")
            sys.exit(1)
    if args.render_mode != "tts" and not args.video:
        print(f"❌ --render-mode {args.render_mode} only makes sense with --video")
        sys.exit(1)
    if args.audio and args.render_mode == "vlog":
        print("❌ --audio doesn't compose with --render-mode vlog — vlog renders its own "
              "audio internally (TTS + voice swap); the separate track would just double-bill")
        sys.exit(1)
    for flag, name in ((args.clips_dir, "--clips-dir"), (args.dry_run, "--dry-run")):
        if flag and not (args.from_script and args.render_mode == "vlog"):
            print(f"❌ {name} is part of the vlog render path — "
                  "use with --from-script --video --render-mode vlog")
            sys.exit(1)
    if args.produce and args.no_save:
        print("❌ --produce writes production files — drop --no-save")
        sys.exit(1)
    if args.video and args.render_mode == "tts" and not args.audio:
        print("ℹ️  --video implies --audio — enabling the audio stage")
        args.audio = True
    setup_logging(debug=args.debug)
    # Run-log wall clock starts here; the JSONL file (with its audit header)
    # is created on the first LLM call. Render runs carry the external-
    # generator stamp in the header too, matching their saved outputs.
    if args.from_script:
        from .render import EXTERNAL_AUDIT
        start_run(header={"audit": {**AUDIT, **EXTERNAL_AUDIT}})
    else:
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

    if args.from_script:
        from .render import load_external_scene, render_external_scene
        try:
            # Validation is the hard boundary: it runs before ANY spend —
            # before the media preflight, before any key is even checked.
            data = load_external_scene(args.from_script)
        except ConfigError as e:
            print(f"❌ {e}")
            sys.exit(1)
        if args.video and args.render_mode == "vlog":
            # Plan problems (bad delivery markers, bad vlog config) and a
            # mismatched clips folder must all fail here — before the judge,
            # TTS, or Veo spend a cent. --dry-run stops here by design.
            from .vlog import (build_clip_plan, describe_plan, validate_clips_dir,
                               vlog_config, write_prompt_pack)
            try:
                cfg = vlog_config()
                plan = build_clip_plan(data["conversation"], data["scene"], cfg)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"❌ {e}")
                sys.exit(1)
            if args.dry_run:
                print(describe_plan(plan, cfg, clips_dir=args.clips_dir or None))
                script = Path(args.from_script)  # any extension, not just .json
                pack = str(script.with_name(script.stem + "_vlog_prompts.txt"))
                write_prompt_pack(plan, pack, cfg)
                print(f"📝 Clip prompts written to: {pack}\n"
                      "   Generate the clips (Veo web UI or any tool), drop them in a "
                      "folder, then re-run with --clips-dir <folder>. Nothing was billed.")
                sys.exit(0)
            if args.clips_dir:
                problems = validate_clips_dir(args.clips_dir, plan)
                if problems:
                    for p in problems:
                        print(f"❌ {p}")
                    sys.exit(1)
        if args.produce:
            director_key_or_die()  # --produce bills the director model
        if args.audio or args.video:
            media_preflight_or_die(data["cast_ids"], audio=args.audio,
                                   video=args.video, render_mode=args.render_mode,
                                   clips_dir=args.clips_dir)
        opts = RunOptions(produce=args.produce, debug=args.debug,
                          evaluate=args.eval, audio=args.audio,
                          video=args.video, render_mode=args.render_mode,
                          clips_dir=args.clips_dir)
        render_external_scene(data, opts)
        return

    if args.episode:
        if args.episode not in EPISODES:
            print(f"❌ Unknown episode: {args.episode}")
            print(f"Available: {', '.join(EPISODES.keys())}")
            sys.exit(1)
        cast_ids = resolve_cast(args.cast)
        if args.solo:
            solo_cast_or_die(cast_ids)
        available, missing = check_api_keys(cast_ids)
        cast_keys_or_die(available, missing, solo=args.solo)
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
        if args.solo:
            solo_cast_or_die(cast_ids)
        available, missing = check_api_keys(cast_ids)
        cast_keys_or_die(available, missing, solo=args.solo)
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
