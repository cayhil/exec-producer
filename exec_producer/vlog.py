"""--render-mode vlog: a consistent solo host on camera in generated
footage, selfie-vlog POV mixed with cinematic segments. EVERYTHING on
screen is generated video — no stills, no name cards, ever, in this mode.

Per-line delivery markers ("delivery": "on_camera" | "voiceover") control
AUDIO treatment only — every beat becomes a full video clip either way:

- on_camera: the clip is generated with the host SPEAKING the line
  (reference images + quoted dialogue in the prompt). The clip's audio is
  then extracted, run through ElevenLabs speech-to-speech into the host's
  persistent voice (data/voices.json), and remuxed — same words + same
  timing keeps the lips plausibly synced, and the host sounds identical in
  every clip of every episode.
- voiceover: the clip is generated with NO dialogue (ambient sound only);
  the line renders via the existing TTS backend and is mixed OVER the clip
  with the ambient bed ducked underneath (a static duck, no sidechain).

Host visual consistency rides on reference images: visuals.json maps the
host to reference photos (characters.<id>.reference_images) attached to
every generation with the host in frame. Veo 3.1 API constraints found in
research (2026-07): reference images force 8-second clips and REJECT 9:16
output — so a 9:16 show generates reference beats at 16:9 with a centered-
subject prompt suffix and center-crops in post; 16:9 (the native YouTube
vlog format) is the default and needs no crop.

Two clip routes, equal citizens: the API route generates via the existing
google-genai machinery (telemetry-receipted, budget-capped, dry-run cost
estimate first); the bring-your-own-clips route consumes a folder of
beat-numbered files the user generated anywhere (Veo web UI on
subscription quota, or any other tool) — the downstream pipeline
(voice-swap, voiceover mix, assembly, SRT) is identical either way.
"""

import importlib.util
import json
import subprocess
import time
from pathlib import Path

from . import telemetry
from .audio import get_backend, spoken_text
from .config import load_visuals, load_voices
from .llm import api_key_set
from .video import (H264_ENCODERS, VEO_MODEL, audio_duration, build_srt,
                    concat_entry, pick_h264_encoder, poll_veo_operation,
                    veo_client)

DELIVERIES = ("on_camera", "voiceover")
CLIP_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv")

# visuals.json "vlog" block defaults. price_per_second is an ESTIMATE knob
# (Veo pricing moves; check current rates) — the plan is priced with it and
# clearly marked as an estimate. max_clips caps the API route per run:
# retakes are expected, and a clear error beats an open-ended spend.
VLOG_DEFAULTS = {
    "aspect": "16:9",
    "fps": 30,
    "video_codec": "",
    "clip_seconds": 8,          # reference images force 8s; used for estimates
    "duck_volume": 0.2,         # ambient bed level under a voiceover line
    "price_per_second": None,   # e.g. 0.40 — set to get cost estimates
    "max_clips": 12,
    "poll_seconds": 15,
    "timeout_seconds": 900,
}
ASPECT_SIZES = {"16:9": (1920, 1080), "9:16": (1080, 1920)}


def _run_ffmpeg(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}")


def vlog_config(visuals=None):
    loaded = visuals
    if loaded is None:
        try:
            loaded = load_visuals()
        except FileNotFoundError:
            loaded = {}
    cfg = {**VLOG_DEFAULTS, **loaded.get("vlog", {})}
    if cfg["aspect"] not in ASPECT_SIZES:
        raise ValueError(f"vlog aspect must be one of {', '.join(ASPECT_SIZES)} (got {cfg['aspect']!r})")
    cfg["width"], cfg["height"] = ASPECT_SIZES[cfg["aspect"]]
    # Relative reference-image paths are show-dir-relative (the docs'
    # "refs/host.png" contract) — NOT cwd-relative, which would break the
    # moment a show is run from anywhere but one blessed directory.
    from .config import DATA_DIR  # deferred: keep module import light
    cfg["reference_images"] = {
        cid: [p if Path(p).is_absolute() else str(DATA_DIR / p)
              for p in c.get("reference_images", [])]
        for cid, c in loaded.get("characters", {}).items()
    }
    return cfg


# ─────────────────────────────────────────────
# Clip plan
# ─────────────────────────────────────────────

def build_clip_plan(conversation, scene, cfg):
    """One beat per spoken line. The delivery marker decides audio treatment;
    the shot description decides what's on screen. A line may carry its own
    "shot" (the showrunner path writes these for free); otherwise a
    deterministic template builds one from the scene setting + the line."""
    plan = []
    for i, entry in enumerate(conversation):
        words = spoken_text(entry["text"])
        if not words:
            continue  # action-only line — nothing to say, no beat
        delivery = entry.get("delivery", "on_camera")
        if delivery not in DELIVERIES:
            raise ValueError(
                f"line {i + 1} ({entry['id']}): delivery must be one of "
                f"{', '.join(DELIVERIES)} (got {delivery!r})")
        host_in_frame = bool(entry.get("host_in_frame", delivery == "on_camera"))
        shot = entry.get("shot") or ""
        if not isinstance(shot, str):
            raise ValueError(f"line {i + 1} ({entry['id']}): 'shot' must be a string")
        shot = shot.strip() or _template_shot(entry, scene, delivery)
        prompt = _clip_prompt(shot, words, delivery, host_in_frame, cfg)
        # Reference-image generations are FORCED to 8s by the API regardless
        # of clip_seconds — price the plan with the duration that will bill.
        uses_refs = host_in_frame and bool(cfg["reference_images"].get(entry["id"]))
        plan.append({
            "beat": len(plan) + 1,
            "line_index": i,
            "entry": entry,
            "delivery": delivery,
            "host_in_frame": host_in_frame,
            "shot": shot,
            "prompt": prompt,
            "seconds": 8 if uses_refs else cfg["clip_seconds"],
        })
    return plan


def _template_shot(entry, scene, delivery):
    # Trailing period stripped: the templates add their own, and a prompt
    # ending in "." would otherwise render "hill.. Natural motion".
    setting = scene.get("prompt", "").strip().rstrip(".")
    if delivery == "on_camera":
        return (f"Selfie-vlog POV, handheld: {entry['name']} ({entry.get('tagline', '')}) "
                f"holds the camera at arm's length and talks directly into it. "
                f"Setting: {setting}. Natural motion, ambient life in the background.")
    return (f"Cinematic b-roll of the surroundings: {setting}. "
            f"Moving camera, ambient life, nobody addresses the camera.")


def _clip_prompt(shot, words, delivery, host_in_frame, cfg):
    """The full Veo prompt for a beat. Dialogue rides in the prompt ONLY for
    on-camera beats; voiceover beats explicitly request ambient sound alone
    (their words arrive later via TTS). The centered-subject suffix keeps a
    16:9 generation croppable when the show's format is 9:16."""
    parts = [shot]
    if delivery == "on_camera":
        # scripts carry their own quote marks, straight or curly — strip them
        # so the template's quoting doesn't double up
        parts.append('They say, speaking naturally to the camera: '
                     f'"{words.strip(chr(34) + "“”")}"')
    else:
        parts.append("No one speaks. Ambient sound only — no dialogue, no narration.")
    if cfg["aspect"] == "9:16" and host_in_frame:
        parts.append("Subject centered in frame, composed to survive a vertical 9:16 center crop.")
    return " ".join(parts)


def estimate_cost(plan, cfg):
    """(total_seconds, cost_or_None). An estimate, priced from the
    configurable price_per_second — marked as such everywhere it prints."""
    seconds = sum(b["seconds"] for b in plan)
    price = cfg.get("price_per_second")
    return seconds, (round(seconds * price, 2) if price else None)


def describe_plan(plan, cfg, clips_dir=None):
    """The dry-run / pre-generation report: count, durations, delivery
    split, estimated cost."""
    on_cam = sum(1 for b in plan if b["delivery"] == "on_camera")
    seconds, cost = estimate_cost(plan, cfg)
    # Total from per-beat seconds, no per-clip arithmetic in the display:
    # reference-image beats are forced to 8s regardless of clip_seconds, so
    # "N clips × ~Xs" would not add up whenever the two durations mix.
    lines = [
        f"Clip plan: {len(plan)} clips, ~{seconds}s of footage "
        f"({on_cam} on-camera, {len(plan) - on_cam} voiceover), {cfg['aspect']}",
    ]
    if clips_dir:
        lines.append(f"Route: bring-your-own-clips from {clips_dir} (no Veo API spend)")
    elif cost is not None:
        lines.append(f"Route: Veo API — ESTIMATED cost ~${cost:.2f} "
                     f"(at ${cfg['price_per_second']}/s — configure vlog.price_per_second "
                     "in visuals.json; check current Veo pricing)")
    else:
        lines.append("Route: Veo API — cost estimate unavailable "
                     "(set vlog.price_per_second in visuals.json)")
    return "\n".join(lines)


def write_prompt_pack(plan, path, cfg):
    """The bring-your-own-clips handoff: one prompt per beat, marked
    on_camera/voiceover, with the exact filename each clip must be saved
    as. Generate the clips in the Veo web UI (subscription quota, no API
    cost) or any other tool, then render with --clips-dir."""
    out = [
        "VLOG CLIP PROMPTS — bring-your-own-clips route",
        f"Format: {cfg['aspect']} · ~{cfg['clip_seconds']}s per clip",
        "Generate each clip with the prompt below, save it with the exact",
        "filename shown into one folder, then run the same render command",
        "with --clips-dir <folder>.",
        "NOTE: on_camera clips must show the host speaking the quoted line",
        "(any voice — it gets re-voiced); voiceover clips must have NO speech.",
        "",
    ]
    for b in plan:
        out += [f"── beat_{b['beat']:03d}  [{b['delivery']}]  → save as beat_{b['beat']:03d}.mp4",
                b["prompt"], ""]
    Path(path).write_text("\n".join(out))
    return path


# ─────────────────────────────────────────────
# Bring-your-own-clips route
# ─────────────────────────────────────────────

def validate_clips_dir(clips_dir, plan):
    """Folder contract, checked BEFORE any ElevenLabs spend: exactly one
    readable, non-empty regular file per beat (beat_001.mp4 …), nothing
    extra. Returns a list of problems — all of them, not just the first."""
    problems = []
    d = Path(clips_dir)
    if not d.is_dir():
        return [f"{clips_dir} is not a directory"]
    found = {}
    for p in sorted(d.iterdir()):
        if p.name.startswith("."):
            # OS metadata (.DS_Store, ._* AppleDouble) appears in any folder
            # Finder has touched — never a clip, never a validation failure.
            continue
        if p.is_symlink() or not p.is_file():
            # Clip folders come from anywhere ("any other tool") — a symlink
            # would splice a file from OUTSIDE the folder into the video.
            problems.append(f"{p.name} is not a regular file (symlinks aren't allowed here)")
            continue
        if p.suffix.lower() not in CLIP_EXTENSIONS or not p.stem.startswith("beat_"):
            problems.append(f"unexpected file: {p.name} (expected beat_<n>{CLIP_EXTENSIONS[0]})")
            continue
        try:
            n = int(p.stem.split("_", 1)[1])
        except ValueError:
            problems.append(f"unexpected file: {p.name} (beat number not parseable)")
            continue
        if n in found:
            problems.append(f"duplicate clip for beat {n}: {p.name} and {found[n].name}")
            continue
        found[n] = p
    for b in plan:
        clip = found.pop(b["beat"], None)
        if clip is None:
            problems.append(f"missing clip for beat {b['beat']} ({b['delivery']}: "
                            f"{spoken_text(b['entry']['text'])[:40]}…)")
        elif clip.stat().st_size == 0:
            problems.append(f"{clip.name} is empty")
    for n, p in found.items():
        problems.append(f"extra clip beat {n} ({p.name}) — the plan has no such beat")
    return problems


def _clips_from_dir(clips_dir, plan):
    d = Path(clips_dir)
    by_beat = {}
    for p in d.iterdir():
        if p.suffix.lower() in CLIP_EXTENSIONS and p.stem.startswith("beat_"):
            try:
                by_beat[int(p.stem.split("_", 1)[1])] = p
            except ValueError:
                pass
    return [(b, by_beat[b["beat"]]) for b in plan if b["beat"] in by_beat]


# ─────────────────────────────────────────────
# API route (Veo with reference images)
# ─────────────────────────────────────────────

def _reference_images(paths):
    from google.genai import types  # lazy: optional veo dep
    refs = []
    for p in paths[:3]:  # Veo 3.1 accepts at most three
        data = Path(p).read_bytes()
        mime = "image/png" if str(p).lower().endswith(".png") else "image/jpeg"
        refs.append(types.VideoGenerationReferenceImage(
            image=types.Image(image_bytes=data, mime_type=mime),
            reference_type="asset"))
    return refs


def _generate_clip(client, model, beat, cfg, ref_paths):
    """One beat → one clip via the polled long-running operation (the
    existing Veo machinery's shape). Reference images are attached whenever
    the host is in frame; they force 16:9 generation (the API rejects 9:16
    with references — the post-crop restores the show's format)."""
    from google.genai import types  # lazy: optional veo dep
    use_refs = bool(ref_paths) and beat["host_in_frame"]
    native_aspect = "16:9" if use_refs else cfg["aspect"]
    config_kwargs = {"aspect_ratio": native_aspect}
    if use_refs:
        config_kwargs["reference_images"] = _reference_images(ref_paths)
    op = client.models.generate_videos(
        model=model, prompt=beat["prompt"],
        config=types.GenerateVideosConfig(**config_kwargs))
    return poll_veo_operation(client, op, cfg["poll_seconds"], cfg["timeout_seconds"])


def _generate_clips(plan, cfg, workdir, model, client=None):
    """API route: every clip is real money — the cap is enforced BEFORE the
    first call, each call is telemetry-receipted, and a failed beat is
    recorded and skipped (never silently retried into more spend)."""
    if len(plan) > cfg["max_clips"]:
        raise RuntimeError(
            f"clip plan ({len(plan)}) exceeds vlog.max_clips ({cfg['max_clips']}) — "
            "raise the cap in visuals.json, trim the scene, or use --clips-dir")
    if client is None:
        client = veo_client()
    log = telemetry.get_run_log()
    pairs, actuals = [], []
    for beat in plan:
        ref_paths = cfg["reference_images"].get(beat["entry"]["id"], [])
        start = time.time()
        try:
            clip_bytes = _generate_clip(client, model, beat, cfg, ref_paths)
        except Exception as e:
            telemetry.record_failure(f"veo/{model}")
            log.record(caller="veo", provider="gemini", model=model,
                       latency_s=round(time.time() - start, 2), error=type(e).__name__)
            print(f"  ⚠️  Veo beat {beat['beat']} failed ({type(e).__name__}) — beat skipped")
            continue
        path = workdir / f"beat_{beat['beat']:03d}.mp4"
        path.write_bytes(clip_bytes)
        # cost_usd=None: the API doesn't report Veo pricing per operation;
        # the MEASURED duration goes in the receipt so actual seconds (the
        # billing unit) are reconstructable from the JSONL alone.
        try:
            actual = round(_clip_duration(path), 2)
        except Exception:
            actual = None
        log.record(caller="veo", provider="gemini", model=model,
                   latency_s=round(time.time() - start, 2), cost_usd=None,
                   seconds=actual)
        actuals.append(actual)
        pairs.append((beat, path))
    # Estimate honesty: the plan priced CONFIGURED durations; reconcile
    # against what the API actually returned (the real billing basis).
    measured = [a for a in actuals if a]
    price = cfg.get("price_per_second")
    if price and measured:
        total = sum(measured)
        print(f"  💰 Generated {total:.0f}s of footage — ESTIMATED "
              f"~${total * price:.2f} at ${price}/s (check your billing console)")
    return pairs


# ─────────────────────────────────────────────
# Audio treatment (ffmpeg command builders — pure, testable)
# ─────────────────────────────────────────────

def _normalize_vf(cfg):
    """Scale-to-cover + center-crop: uniforms every clip (any source, any
    aspect) to the show's format — this is also where a 16:9 reference-image
    generation becomes 9:16."""
    w, h = cfg["width"], cfg["height"]
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"


def extract_audio_cmd(clip_path, out_path):
    return ["ffmpeg", "-y", "-v", "error", "-i", str(clip_path),
            "-vn", "-c:a", "libmp3lame", "-b:a", "128k", str(out_path)]


# Uniform audio parameters on every segment and the final concat: the concat
# demuxer needs matching stream parameters, and vlog segments mix sources
# (ElevenLabs 44.1kHz output, BYO clips of arbitrary rates).
AUDIO_NORMALIZE = ["-c:a", "aac", "-ar", "44100", "-ac", "2"]


def remux_cmd(clip_path, audio_path, out_path, cfg, encoder,
              clip_seconds, audio_seconds):
    """on_camera: the swapped voice replaces the clip's audio track; video is
    normalized to the show format in the same pass. Durations are reconciled
    explicitly — the speech-to-speech output can drift a hair from the clip,
    so the shorter stream is padded (last frame held, silence appended)
    rather than truncating dialogue or video with -shortest."""
    total = max(clip_seconds, audio_seconds)
    pad = max(0.0, total - clip_seconds)
    return ["ffmpeg", "-y", "-v", "error",
            "-i", str(clip_path), "-i", str(audio_path),
            "-map", "0:v", "-map", "1:a",
            "-vf", f"{_normalize_vf(cfg)},tpad=stop_mode=clone:stop_duration={pad:.3f}",
            "-af", "apad", "-t", f"{total:.3f}",
            "-r", str(cfg["fps"]),
            "-c:v", encoder, "-pix_fmt", "yuv420p", *AUDIO_NORMALIZE,
            str(out_path)]


def mix_cmd(clip_path, tts_path, out_path, cfg, encoder, clip_seconds, tts_seconds,
            has_ambient=True):
    """voiceover: TTS over the clip with the ambient bed ducked (static duck,
    no sidechain). A line longer than its clip holds the last frame (tpad
    clone) instead of cutting the voice off."""
    total = max(clip_seconds, tts_seconds + 0.4)
    pad = max(0.0, total - clip_seconds)
    vf = f"[0:v]{_normalize_vf(cfg)},tpad=stop_mode=clone:stop_duration={pad:.3f}[v]"
    if has_ambient:
        af = (f"[0:a]volume={cfg['duck_volume']}[amb];"
              f"[amb][1:a]amix=inputs=2:duration=longest:normalize=0[a]")
    else:
        af = "[1:a]anull[a]"
    return ["ffmpeg", "-y", "-v", "error",
            "-i", str(clip_path), "-i", str(tts_path),
            "-filter_complex", f"{vf};{af}",
            "-map", "[v]", "-map", "[a]", "-t", f"{total:.3f}",
            "-r", str(cfg["fps"]),
            "-c:v", encoder, "-pix_fmt", "yuv420p", *AUDIO_NORMALIZE,
            str(out_path)]


def _has_audio_stream(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "json", str(path)],
        capture_output=True, text=True).stdout or "{}"
    return bool(json.loads(out).get("streams"))


def _clip_duration(path):
    return audio_duration(path)  # format-level duration works for video too


# ─────────────────────────────────────────────
# The stage
# ─────────────────────────────────────────────

def render_vlog_video(conversation, scene, out_prefix, clips_dir="",
                      visuals=None, client=None):
    """Compose <out_prefix>_video.mp4 in vlog mode. Clips come from the API
    route (generated per beat) or --clips-dir (bring-your-own). Either way:
    on-camera beats get the speech-to-speech voice swap, voiceover beats get
    TTS over a ducked ambient bed, everything is normalized to the show
    format, concatenated in script order, with an exact-timing SRT written
    alongside. Never renders cards — a failed beat is recorded and dropped."""
    cfg = vlog_config(visuals)
    plan = build_clip_plan(conversation, scene, cfg)
    if not plan:
        print("  ⚠️  Vlog skipped — no spoken lines to shoot")
        return None

    encoder = pick_h264_encoder(cfg.get("video_codec", ""))
    if not encoder:
        raise RuntimeError("no H.264 encoder in this ffmpeg build "
                           f"(looked for: {', '.join(H264_ENCODERS)})")

    voices_cfg = load_voices()
    voice_defaults = voices_cfg.get("defaults", {})
    voice_map = voices_cfg.get("voices", {})
    backend = get_backend(voices_cfg.get("provider", "elevenlabs"), voice_defaults)
    log = telemetry.get_run_log()

    workdir = Path(f"{out_prefix}_vlog")
    workdir.mkdir(parents=True, exist_ok=True)
    print("  " + describe_plan(plan, cfg, clips_dir=clips_dir or None).replace("\n", "\n  "))

    if clips_dir:
        problems = validate_clips_dir(clips_dir, plan)
        if problems:
            raise RuntimeError("clips folder doesn't match the plan:\n    - "
                               + "\n    - ".join(problems))
        pairs = _clips_from_dir(clips_dir, plan)
    else:
        model = cfg.get("veo_model") or VEO_MODEL
        pairs = _generate_clips(plan, cfg, workdir, model, client=client)

    segments, seg_durations, seg_entries = [], [], []
    for beat, clip_path in pairs:
        entry = beat["entry"]
        voice = voice_map.get(entry["id"])
        seg_path = workdir / f"seg_{beat['beat']:03d}.mp4"
        try:
            if beat["delivery"] == "on_camera":
                _render_on_camera(beat, clip_path, seg_path, voice, voice_defaults,
                                  backend, cfg, encoder, workdir, log)
            else:
                _render_voiceover(beat, clip_path, seg_path, voice, backend,
                                  cfg, encoder, workdir, log,
                                  provider=voices_cfg.get("provider", "elevenlabs"))
        except Exception as e:
            print(f"  ⚠️  Beat {beat['beat']} audio treatment failed ({type(e).__name__}) "
                  "— beat dropped")
            continue
        segments.append(seg_path)
        seg_durations.append(_clip_duration(seg_path))
        seg_entries.append(entry)

    if not segments:
        print("  ⚠️  Vlog skipped — no beats survived")
        return None

    (workdir / "_subs.srt").write_text(build_srt(seg_entries, seg_durations, 0.0))
    list_path = workdir / "_concat.txt"
    list_path.write_text("".join(concat_entry(p.resolve()) for p in segments))
    out_path = Path(f"{out_prefix}_video.mp4").resolve()
    _run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                 "-i", str(list_path.resolve()),
                 "-c:v", encoder, "-pix_fmt", "yuv420p", *AUDIO_NORMALIZE,
                 str(out_path)])
    return f"{out_prefix}_video.mp4"


def _render_on_camera(beat, clip_path, seg_path, voice, voice_defaults,
                      backend, cfg, encoder, workdir, log):
    """Voice-normalize an on-camera clip. If the swap fails, the clip keeps
    its generated voice (drift beats a lost beat) — recorded and printed."""
    if not voice:
        raise RuntimeError(f"no voice for {beat['entry']['id']}")
    raw_audio = workdir / f"raw_{beat['beat']:03d}.mp3"
    _run_ffmpeg(extract_audio_cmd(clip_path, raw_audio))
    clip_seconds = _clip_duration(clip_path)
    model_id = {**voice_defaults, **voice}.get("sts_model_id", "eleven_multilingual_sts_v2")
    start = time.time()
    try:
        swapped = backend.convert_speech(raw_audio, voice)
    except Exception as e:
        telemetry.record_failure(f"elevenlabs/{model_id}")
        log.record(caller=f"sts:{beat['entry']['id']}", provider="elevenlabs",
                   model=model_id, latency_s=round(time.time() - start, 2),
                   error=type(e).__name__)
        # Remux BEFORE announcing the outcome: if the fallback itself fails,
        # the exception reaches the beat loop's "beat dropped" handler — the
        # console must never promise a beat that isn't in the final video.
        _run_ffmpeg(remux_cmd(clip_path, raw_audio, seg_path, cfg, encoder,
                              clip_seconds=clip_seconds, audio_seconds=clip_seconds))
        print(f"  ⚠️  Voice swap failed for beat {beat['beat']} ({type(e).__name__}) "
              "— keeping the clip's own voice")
        return
    log.record(caller=f"sts:{beat['entry']['id']}", provider="elevenlabs",
               model=model_id, latency_s=round(time.time() - start, 2), cost_usd=None)
    swapped_path = workdir / f"sts_{beat['beat']:03d}.mp3"
    swapped_path.write_bytes(swapped)
    _run_ffmpeg(remux_cmd(clip_path, swapped_path, seg_path, cfg, encoder,
                          clip_seconds=clip_seconds,
                          audio_seconds=audio_duration(swapped_path)))


def _render_voiceover(beat, clip_path, seg_path, voice, backend, cfg, encoder,
                      workdir, log, provider="elevenlabs"):
    """TTS the line in the host's persistent voice and mix it over the
    clip's ambient bed."""
    if not voice:
        raise RuntimeError(f"no voice for {beat['entry']['id']}")
    words = spoken_text(beat["entry"]["text"])
    model_id = {**backend.defaults, **voice}.get("model_id", "eleven_multilingual_v2")
    start = time.time()
    try:
        tts_bytes = backend.synthesize(words, voice)
    except Exception as e:
        # Same receipt discipline as every other paid call: record, then let
        # the beat loop drop the beat.
        telemetry.record_failure(f"{provider}/{model_id}")
        log.record(caller=f"tts:{beat['entry']['id']}", provider=provider, model=model_id,
                   latency_s=round(time.time() - start, 2), characters=len(words),
                   error=type(e).__name__)
        raise
    log.record(caller=f"tts:{beat['entry']['id']}", provider=provider, model=model_id,
               latency_s=round(time.time() - start, 2), characters=len(words),
               cost_usd=None)
    tts_path = workdir / f"vo_{beat['beat']:03d}.mp3"
    tts_path.write_bytes(tts_bytes)
    _run_ffmpeg(mix_cmd(clip_path, tts_path, seg_path, cfg, encoder,
                        clip_seconds=_clip_duration(clip_path),
                        tts_seconds=audio_duration(tts_path),
                        has_ambient=_has_audio_stream(clip_path)))


# ─────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────

def preflight(cast_ids, api_route=True):
    """Everything --render-mode vlog needs, before any spend. Both routes
    need the voice pipeline (ElevenLabs + ffmpeg + an encoder + voice
    coverage); the API route additionally needs the Veo machinery and warns
    when no reference images are configured (host consistency is the point
    of this mode). Pillow is deliberately NOT required — no cards, ever."""
    from .audio import preflight as audio_preflight
    problems = list(audio_preflight(cast_ids))
    try:
        if not pick_h264_encoder():
            problems.append("no H.264 encoder in this ffmpeg build "
                            f"(looked for: {', '.join(H264_ENCODERS)})")
    except FileNotFoundError:
        pass  # ffmpeg absence already reported by the audio preflight
    try:
        cfg = vlog_config()
    except (ValueError, json.JSONDecodeError) as e:
        problems.append(f"visuals.json vlog config: {e}")
        return problems
    if api_route:
        if not api_key_set("GEMINI_API_KEY"):
            problems.append("GEMINI_API_KEY is not set in .env (vlog clips generate via the Gemini API)")
        try:
            # find_spec on a dotted name raises when the parent namespace
            # package is absent entirely (the fresh-clone, no-extras state)
            sdk_present = importlib.util.find_spec("google.genai") is not None
        except ModuleNotFoundError:
            sdk_present = False
        if not sdk_present:
            problems.append("google-genai not installed — run: uv sync --extra vlog")
        for cid in cast_ids:
            refs = cfg["reference_images"].get(cid, [])
            missing = [p for p in refs if not Path(p).is_file()]
            if missing:
                problems.append(f"reference image(s) for {cid} not found: {', '.join(map(str, missing))}")
            if not refs:
                print(f"  ⚠️  No reference_images configured for {cid} in visuals.json — "
                      "host consistency will rely on prompts alone")
    return problems
