---
name: showrunner
description: Turn an episode idea into rendered scenes with zero generation-API spend. Use when the user describes an episode or scene idea for their show and wants this session to WRITE the scripts (on the user's subscription), then have the engine render them — .txt, --eval, --audio, --video. Also use when the user says "write an episode", "script a scene", or "showrunner".
version: 1.0.0
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [uv]
  # The engine itself needs no API key on this path — --from-script makes
  # zero generation calls. The optional stages this skill can invoke do:
  #   --eval   judge model key (world.json judge_model)
  #   --audio  ELEVENLABS_API_KEY + ffmpeg + `uv sync --extra audio`
  #   --video  `uv sync --extra video`
  # Run `exec-producer doctor` to see which of these are ready.
  optional_env_vars: [ELEVENLABS_API_KEY]
metadata:
  exec-producer:
    writes: scene .json (SCENE_SCHEMA 1 — see CREATING_A_SHOW.md)
    spend: none for script generation; downstream stages bill per their flags
---

# Showrunner: write scripts in-session, render with the engine

The engine normally generates dialogue by calling each character's LLM API. This skill
replaces that generation step with YOU: the session writes the scene JSONs (no API cost —
the user's subscription pays for this conversation), and the engine runs only the
downstream stages. LiteLLM stays the engine's only model-access layer; the handoff is a
file contract, never a wired-in provider.

## Workflow

### 1. Load the show

Resolve the show directory (default `data/`; `--data-dir <dir>` or the starters in
`examples/`). Read, at minimum:

- `world.json` — world text, `intensity_guide` (write to the requested level), `show_slug`
- `cast.json` — every speaking character's personality, triggers, relationships, `gender`
- `prompts/critical_rules.txt` — the FORMAT contract every line must obey
- `prompts/tone_face_to_face.txt` / `tone_confessional_early.txt` / `tone_confessional_late.txt`
  — a scene whose display name contains "confessional" uses the confessional rules
  (early at low/medium intensity, late at high/maximum)
- `scenes.json` + `episodes.json` — existing settings and episode shapes to stay consistent with
- `embodiment.json` — physicality rules (base, per-gender, per-character, per-intensity)

### 2. Structure the episode

From the user's idea, propose: scene list (reuse `scenes.json` settings where they fit),
lines per scene (6-12 for a 60-90s clip), intensity level, and plot beats per scene
(physical activities the drama happens DURING, not monologue topics). Confirm with the
user before writing dialogue if the idea leaves real choices open.

### 3. Write one JSON file per scene

Write to `scripts/<show_slug>_<scene>_draft.json` (the `scripts/` dir is gitignored).
Contract below. Writing rules:

- Every line obeys `critical_rules.txt` — 1-2 spoken sentences, `[action]` separate from
  `"dialogue"`, no asterisks, no self-narration.
- Each character speaks in THEIR voice (personality sheet, triggers, relationships).
  On a solo show: one speaker, no invented callers or co-hosts, address the audience.
- Pace to the intensity level: early lines are LOW energy (settle-in, activity, banter),
  tension seeds early but detonates only in the final third — sized to the intensity.
  Wall-to-wall revelation is the classic failure; ration the true moments.
- `director_notes[i]` pairs with `conversation[i]`; `stage_direction` entries become
  bracketed lines in the rendered script — use them for environment beats.

### 4. Self-review against the rubric

Apply `prompts/eval_rubric.txt` as a checklist before rendering: format adherence
(critical rules), voice consistency (per the cast sheet), pacing vs. intensity. Fix what
fails, then show the user the script — the human read is the primary quality gate here.
Optionally run the engine's `--eval` for an independent judge-model scorecard (bills the
judge model's API; a session grading its own script is the self-preference trap the
separate judge exists to avoid).

### 5. Render each scene

```bash
uv run exec-producer render scripts/<file>.json                  # .txt + audited .json only
uv run exec-producer --from-script scripts/<file>.json --eval    # + judge scorecard
uv run exec-producer --from-script scripts/<file>.json --audio   # + per-line TTS track
uv run exec-producer --from-script scripts/<file>.json --video   # + 9:16 MP4 + SRT

# Vlog mode (solo host, all-generated footage — see CREATING_A_SHOW.md):
# ALWAYS dry-run first: clip plan + ESTIMATED cost + per-beat prompt pack, zero spend.
# NEVER run the API route without the user explicitly approving the estimate.
uv run exec-producer --from-script scripts/<file>.json --video --render-mode vlog --dry-run
uv run exec-producer --from-script scripts/<file>.json --video --render-mode vlog --clips-dir <dir>
```

Add `--data-dir <dir>` if the show isn't `data/`. Render needs no director key (except
with `--produce`). Validation is hard and runs before any TTS/video spend — fix what it
lists. The audit block is stamped `generator: "external"`.

### 6. Emit YouTube packaging (first-class output, not an afterthought)

Write `scripts/<show_slug>_<scene-or-episode>_packaging.md` containing:

- **3-5 title options** — curiosity-gap style: open a loop the video closes, ≤60 chars,
  no clickbait the content can't cash
- **Thumbnail image prompt** — one detailed text-to-image prompt (9:16 or 16:9 per the
  user's format): subject, expression, composition, text overlay ≤4 words
- **Description** — 1-2 sentence hook, then context; hashtags only if the user wants them
- **First-5-seconds hook line** — the exact opening line/moment that earns the next 55s

## The scene JSON contract

The engine's own saved-scene shape. Any agent harness that produces this shape can use
the render path.

```json
{
  "scene_schema": "1",
  "scene_id": "kitchen_fight",
  "scene": {
    "name": "🍳 Kitchen Fight",
    "prompt": "The physical setting and situation.",
    "mood": "tense, simmering"
  },
  "intensity": "low",
  "conversation": [
    {
      "id": "marlow",
      "text": "[slides the fader down] \"It's 2 a.m. and you're still up.\"",
      "direction": "MONOLOGUE"
    }
  ],
  "director_notes": [
    {"stage_direction": "the request line blinks once and goes dark",
     "emotional_state": "settled, warm"}
  ]
}
```

Field rules:

- `scene_schema` — optional; the contract version this file is written against, currently
  `"1"`. Omit it and the engine reads the file as the current version, so every scene
  written before versioning existed still renders. A version the engine can't read is
  rejected at the boundary rather than misinterpreted.
- `scene_id` — optional; filename-safe; defaults to the input filename stem. Sets the
  output prefix `scripts/<show_slug>_<scene_id>_<ts>_<pid>`.
- `scene.name` — required, non-empty. `prompt`/`mood` default to `""`. A name containing
  "confessional" selects the confessional tone rules.
- `intensity` — one of `low | medium | high | maximum` (default `low`). The show's
  `content_rating.max_intensity` cap still applies.
- `conversation` — required, non-empty. Each entry needs `id` (a key in the loaded show's
  `cast.json` — validated hard) and non-empty `text`. `name`/`emoji`/`tagline` are filled
  from the cast file; `direction` is optional; unknown extra fields (e.g. `delivery`)
  pass through untouched.
- `director_notes` — optional list of objects, index-paired with `conversation`;
  `stage_direction` and `emotional_state` are the useful keys.

Vlog-mode per-line fields (optional; only read by `--render-mode vlog`):

- `delivery` — `"on_camera"` (default: the host speaks this line to camera; the clip's audio
  gets the speech-to-speech voice swap) or `"voiceover"` (the line is TTS mixed over ambient
  b-roll). Controls AUDIO treatment only — every line becomes a full video clip.
- `shot` — the Veo shot description for this beat (what's on screen). Write these when
  scripting a vlog episode: a specific, filmable moment beats the engine's generic template.
- `host_in_frame` — whether the host appears in a voiceover beat's footage (default: true for
  on_camera, false for voiceover). Governs whether reference images attach to the generation.

## Hard rules

- NEVER call generation APIs to write dialogue — writing scripts is this session's job.
- Never commit anything from `data/` or `scripts/` (both gitignored — leave the guards alone).
- UV only: `uv run exec-producer ...`, never bare python/pip.
