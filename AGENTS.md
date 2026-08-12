# Working on exec-producer

Instructions for coding agents (Claude Code, Codex, Cursor, Aider — whatever
reads this file). Humans: [README.md](README.md) is the tour, this is the
house style.

## What this is

A director LLM orchestrates a cast of character LLMs to generate a scripted
reality-show scene, then renders it — script, TTS audio, video. Every model
call goes through LiteLLM. All creative content lives in a **show directory**
of JSON/text files, never in Python.

## Package management: UV only

```bash
uv sync                  # install
uv sync --all-extras     # ...including the optional media/planner stages
uv add <package>         # add a dependency (never edit pyproject by hand)
uv run pytest            # run the tests
uv run exec-producer …   # run the CLI
```

**Never use `pip`, `python`, or `python3` directly.** If you changed
`pyproject.toml`, run `uv lock` and commit `uv.lock` — CI blocks on drift.

## Before you touch anything

```bash
uv run exec-producer doctor        # what's configured, what's ready
uv run pytest -q                   # seconds, no keys or network
```

## The rules that actually matter

**Spending money is the hazard.** This engine bills real APIs — LLM calls per
line, ElevenLabs per character, Veo per second of video. Consequences:

- **Never run a generation command to test a change.** The suite mocks every
  call at the LiteLLM boundary; keep it that way. To exercise the downstream
  stages, write a scene `.json` and use `--from-script`, which makes no
  generation calls. Note it still bills whatever stages you add to it:
  `--eval`, `--audio`, `--video`, and `--produce` all cost money.
- **Validate before spending, not after.** Every stage has a `preflight()`
  that runs before the first billed call and reports *all* problems at once,
  not just the first. New stage, new preflight — and register it in
  `doctor.py:STAGE_PREFLIGHTS`.
- **Degrade, never crash.** A failed character call becomes a stage direction;
  a failed director call becomes a random pick; a failed media stage still
  saves the script. Losing a paid-for scene to an exception is the worst
  outcome in the codebase.

**LiteLLM is the only text-model layer.** Don't add a provider SDK for text.
(Veo goes through `google-genai` directly because video generation isn't a
text completion; ElevenLabs likewise for speech.)

**Optional dependencies import lazily.** `elevenlabs`, `Pillow`,
`google-genai`, and `langgraph` are extras. Import them *inside* functions so
a core-only install still runs. A module-level import of an extra is a bug.

**Show data is swappable.** Anything a show can define belongs in the show
directory, not in code. `--data-dir` points at any of them; `examples/` holds
the two committed starters. If you add a required file, update `REQUIRED_FILES`
in `config.py` and `CREATING_A_SHOW.md` together.

**Don't commit private show content.** `data/` is gitignored on purpose and
holds unreleased material. Test against `examples/sunset_terrace` (ensemble)
or `examples/night_signal` (solo).

## Tests

Test-first. The suite is fast and fully offline — there is no excuse for
"tested manually".

- Mock at the LiteLLM boundary (`conftest.make_response`), not deeper.
- Monkeypatch module globals (`engine.CAST`, `doctor.STAGE_PREFLIGHTS`) —
  that's the established seam. `tests/test_engine.py` exports `FAKE_CAST`.
- **A test must not depend on the host.** Installed extras, a populated
  `.env`, and ffmpeg all vary; stub anything that reads them. This is why
  `tests/test_doctor.py` empties `STAGE_PREFLIGHTS` by default.
- Guard the money paths (preflight ordering, spend cap, dry-run) explicitly.

## Comments

Match the surrounding density, which is high — but the bar is *why*, not
*what*. Existing comments record decisions and their rationale ("cards are
Pillow-rendered because minimal ffmpeg builds lack drawtext"). Restating the
code in prose is worse than nothing. Where a non-obvious choice was made
deliberately, say so and say why, so the next person doesn't "fix" it.

## Layout

| File | Owns |
|---|---|
| `bootstrap.py` | entry shim — applies `--data-dir` before config imports |
| `cli.py` | argparse, flag compatibility, key gates, interactive mode |
| `config.py` | show loading, data contract, guardrail values |
| `engine.py` | director loop, sessions, scene/episode runners, save path |
| `llm.py` | the single LiteLLM call layer |
| `telemetry.py` | logging, JSONL run log, cost tracking, spend cap |
| `redact.py` | secret masking for everything that reaches a log line |
| `doctor.py` | `exec-producer doctor` — aggregates the stage preflights |
| `render.py` | `--from-script` validation + downstream-only render |
| `audio.py` `video.py` `vlog.py` | the media stages (lazy deps) |
| `planner.py` | `--planner` LangGraph pacing layer (lazy deps) |
| `evaluate.py` | `--eval` judge pass |

## Contracts that outlive this repo

Two file shapes are read by things you don't control, so changing them
breaks other people:

- **The scene `.json`** — written by the engine, read back by `--from-script`,
  authored by third parties. Versioned as `SCENE_SCHEMA` in `engine.py`;
  bump only on a change that would make an older engine misread a newer file.
- **The show directory** — see `CREATING_A_SHOW.md`.

Adding an optional key to either is fine. Renaming or repurposing one isn't.
