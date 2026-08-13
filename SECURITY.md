# Security

## Reporting a vulnerability

Please report security issues privately through
[GitHub's private vulnerability reporting](https://github.com/cayhil/exec-producer/security/advisories/new)
rather than opening a public issue. Expect an acknowledgement within a week.

If you're unsure whether something qualifies, report it anyway.

## Threat model

`exec-producer` is a local CLI. It holds API keys for up to seven paid
services and spends real money on your behalf, so the things worth worrying
about are **credential exposure** and **unintended spend** — not remote
attackers.

### Credentials

Keys are read from `.env` (gitignored) via `python-dotenv`, and used only to
authenticate the provider calls the run makes.

- **Log output is redacted.** Once `setup_logging()` has run — which the CLI
  does before anything else — every handler on the package logger formats
  through `RedactingFormatter` (`exec_producer/redact.py`), masking
  vendor-prefixed keys, `KEY=value` assignments, JSON credential fields, and
  bearer tokens. Long secrets keep a first-6/last-4 stub; short ones are
  replaced whole. This is defense in depth, not a patch for a known leak: as
  of this writing none of the supported providers echo a rejected key back in
  its error, but that is their choice to change, not ours. Redaction is
  deliberately conservative — it will miss a bare, prefix-less key sitting in
  free prose, because a false positive destroys the diagnostic the log line
  exists to deliver.
- **The JSONL run log records no message bodies** — caller, model, provider,
  latency, token counts, cost, and the *type* of any error. No prompts, no
  completions, no credentials. That holds by field discipline at the call
  sites, not by redaction: the run log is written as raw JSON and does not
  pass through the formatter.

- **Saved scripts contain generated dialogue**, which is the point. Use
  `--director-only` to redact dialogue from console output and saved scripts
  when sharing a run.

Two output paths sit outside the formatter by construction: console `print()`
and the run log. The one place the engine prints an exception's *message*
rather than just its type — the vlog stage, which talks to two paid providers
— redacts explicitly.

Redaction can be disabled with `EXEC_PRODUCER_REDACT=false` for debugging the
redactor itself. The value is read once at import, so nothing can turn it off
mid-run — but note that a `.env` file is loaded before that snapshot, so the
switch works from `.env` as well as from the shell. Don't set it in either if
you intend to share the output.

### Spend

Anything that bills is gated before the first call, not after:

- Per-stage `preflight()` checks run before any spend and report every
  problem at once. `exec-producer doctor` runs them all standalone.
- `budget.max_run_usd` in `world.json` stops the generation loops *and* the
  media stages once the run's recorded spend reaches it, saving whatever it
  produced. Read the limits below before relying on it.
- `--render-mode vlog` supports `--dry-run`, which prints the clip plan and
  an estimated cost while billing nothing, and `--clips-dir`, which renders
  clips you generated elsewhere with no video-API spend at all. Neither is
  enforced — `--dry-run` first is a strong convention, not a gate.
- `--from-script` makes no *generation* calls. It still bills whatever
  stages you add to it: `--eval` (judge), `--audio` (ElevenLabs),
  `--video` (Veo, in the veo/vlog modes), and `--produce` (director).

**Limits of the budget gate, stated plainly:**

- It is a stop-line, not a ceiling. A call's cost isn't knowable until it
  returns, so a run overshoots by whatever was already in flight — up to
  four billed calls in the classic loop (a turn is a director call plus a
  character call, each retried up to three times on malformed JSON).
- **It cannot see media spend.** ElevenLabs bills per plan tier and the Veo
  API reports no per-operation price, so those calls record a cost of
  *unknown*, which counts as zero. The gate still *stops* the media stages
  once text generation has tripped it — but a run that is mostly or entirely
  video can exceed the cap without the cap ever noticing. For those runs the
  real controls are `vlog.max_clips`, `--dry-run`, and `--clips-dir`.
- It is per process, and one interactive shell session is one process — so
  scenes in a session share a budget rather than each getting a fresh one.
- **A budget stop ends a `--planner` run for good.** The graph exits
  normally, so its checkpoint records the run as finished and `--thread`
  refuses to resume it; you start a new thread and re-pay for turns already
  generated. The CLI warns about this at the start of any capped planner
  run. Resuming a *different* thread also starts a fresh budget.

### Untrusted show directories

A show directory is executable configuration in the sense that it controls
model selection, prompt content, and output filenames. **Treat a show
directory from someone else the way you'd treat their shell script.**

The engine defends the file-path boundary specifically: `show_slug` and every
cast/scene/episode id must match `[A-Za-z0-9_-]+`, so a hostile id can't
traverse out of `scripts/` via ffmpeg concat lists or output prefixes. It
does **not** sandbox prompt content — a show directory can make your models
say anything, and can point `director_model` at any provider you hold a key
for.

Every saved output carries an `audit` block with the engine version, the
models used, and a sha256 of every config and prompt file loaded, so a
questionable output can be traced back to the exact inputs that produced it.

### Generated content

Model output is not filtered. `content_rating.max_intensity` in `world.json`
caps the intensity level requested of the models, which shapes but does not
guarantee the result. If you publish generated video, check your platform's
synthetic-media disclosure rules — see the note in the README.

## Supported versions

The latest `main` is the supported version. This is a single-maintainer
project; there are no backported security releases.
