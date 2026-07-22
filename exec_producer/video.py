"""Video stage — per-scene 9:16 MP4 composed from the audio stage's output.

TTS-composited mode: one segment per spoken line — a Pillow-rendered card
(character name, tagline, and the dialogue text itself) over that line's mp3,
padded by the same inter-line gap the audio stitch uses — then the segments
are concatenated. Text is drawn by Pillow, NOT ffmpeg: minimal ffmpeg builds
ship without drawtext/libass (verified on this machine), so the only ffmpeg
features this stage needs are image looping, an h264 encoder, and the concat
demuxer. An SRT with exact per-line timings is still written next to the
segments as an editing artifact (it drops straight into CapCut).

Pillow is an optional extra (uv sync --extra video) and imports lazily —
the core engine runs without it.
"""

import importlib.util
import json
import os
import re
import subprocess
import time
from pathlib import Path

from . import telemetry
from .audio import concat_entry, spoken_text
from .config import load_visuals, load_voices

# Used when data/visuals.json is absent or has no entry for a character:
# card colors are assigned deterministically from the scene's cast order.
DEFAULTS = {
    "text_color": "white",
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "font": "",           # path to a .ttf/.ttc; empty = try common system fonts
    "video_codec": "",    # empty = auto-pick from H264_ENCODERS
}
PALETTE = ["#8a4f2d", "#1f6f8b", "#5b4a8a", "#2d6a4f", "#8b2d44", "#6b6b23"]

# Preference order: libx264 (portable), VideoToolbox (macOS hardware),
# openh264 (some minimal Linux builds).
H264_ENCODERS = ["libx264", "h264_videotoolbox", "libopenh264"]

# Veo-native render mode. Override via visuals.json defaults.veo_model.
VEO_MODEL = "veo-3.1-generate-preview"

FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",                      # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",             # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",     # Debian/Ubuntu
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",                   # Fedora/Alpine
]

_LINE_FILE = re.compile(r"line_(\d+)_(.+)\.mp3$")


def srt_timestamp(seconds):
    ms = round(seconds * 1000)
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(lines, durations, gap_seconds):
    """One cue per spoken line. Cue i starts after all previous lines plus
    their gaps and lasts that line's audio duration — mirroring exactly how
    the segments are laid out, so the text tracks the voice."""
    cues = []
    t = 0.0
    for n, (entry, dur) in enumerate(zip(lines, durations), start=1):
        cues.append(f"{n}\n{srt_timestamp(t)} --> {srt_timestamp(t + dur)}\n"
                    f"{spoken_text(entry['text'])}\n")
        t += dur + gap_seconds
    return "\n".join(cues)


def audio_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def character_color(cast_id, scene_cast, visuals):
    configured = visuals.get("characters", {}).get(cast_id, {}).get("color")
    if configured:
        return configured
    return PALETTE[sorted(scene_cast).index(cast_id) % len(PALETTE)]


# ─────────────────────────────────────────────
# Card rendering (Pillow)
# ─────────────────────────────────────────────

def _load_font(configured_path, size):
    from PIL import ImageFont  # lazy: optional media dep
    candidates = ([configured_path] if configured_path else []) + FONT_CANDIDATES
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _wrap(draw, text, font, max_width, max_lines=9):
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += " …"
    return lines


def render_card(entry, dialogue, color, cfg, out_path):
    """Draw one line's 9:16 card: character name and tagline up top, the
    dialogue text in the lower half. All text is Pillow-drawn so ffmpeg
    never needs a text filter."""
    from PIL import Image, ImageDraw  # lazy: optional media dep
    w, h = cfg["width"], cfg["height"]
    img = Image.new("RGB", (w, h), color)
    draw = ImageDraw.Draw(img)
    fill = cfg["text_color"]
    font_path = cfg.get("font", "")

    name_font = _load_font(font_path, 110)
    name = entry["name"].upper()
    draw.text(((w - draw.textlength(name, font=name_font)) / 2, h * 0.26),
              name, font=name_font, fill=fill)

    tagline = entry.get("tagline", "")
    if tagline:
        tag_font = _load_font(font_path, 44)
        draw.text(((w - draw.textlength(tagline, font=tag_font)) / 2, h * 0.26 + 150),
                  tagline, font=tag_font, fill=fill)

    if dialogue:
        dia_font = _load_font(font_path, 54)
        line_height = 54 * 1.35
        lines = _wrap(draw, dialogue, dia_font, max_width=w * 0.86)
        y = h * 0.52
        for line in lines:
            draw.text(((w - draw.textlength(line, font=dia_font)) / 2, y),
                      line, font=dia_font, fill=fill)
            y += line_height
    img.save(out_path)
    return out_path


# ─────────────────────────────────────────────
# ffmpeg assembly
# ─────────────────────────────────────────────

def pick_h264_encoder(configured=""):
    if configured:
        return configured
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True).stdout
    for enc in H264_ENCODERS:
        if f" {enc} " in out:
            return enc
    return None


def segment_cmd(card_path, audio_file, seg_path, seg_seconds, cfg, encoder):
    """One line's segment: the card image looped for the segment length, the
    line's audio padded with silence to match (the inter-line gap lives at
    each segment's tail)."""
    return ["ffmpeg", "-y", "-v", "error",
            "-loop", "1", "-i", str(card_path),
            "-i", str(audio_file),
            "-af", "apad",
            "-t", f"{seg_seconds:.3f}",
            "-r", str(cfg["fps"]),
            "-c:v", encoder, "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(seg_path)]


def finalize_cmd(list_path, out_path, encoder):
    # Re-encode rather than stream-copy: it normalizes segment timestamps.
    return ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c:v", encoder, "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(out_path)]


def _run_ffmpeg(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}")


def _load_visuals_or_defaults():
    try:
        loaded = load_visuals()
    except FileNotFoundError:
        return {"defaults": dict(DEFAULTS), "characters": {}}
    return {"defaults": {**DEFAULTS, **loaded.get("defaults", {})},
            "characters": loaded.get("characters", {})}


def collect_line_files(audio_dir, conversation):
    """Pair the audio stage's per-line files back to conversation entries by
    the index embedded in the filename. Lines the TTS stage skipped (action-
    only) or lost (failed synth) have no file and produce no segment."""
    pairs = []
    for path in sorted(Path(audio_dir).glob("line_*.mp3")):
        m = _LINE_FILE.match(path.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx < len(conversation):
            pairs.append((conversation[idx], path))
    return pairs


# ─────────────────────────────────────────────
# The stage
# ─────────────────────────────────────────────

def render_scene_video(conversation, out_prefix, visuals=None, gap_seconds=None):
    """Compose <out_prefix>_video.mp4 from the per-line mp3s the audio stage
    left in <out_prefix>_audio/. Returns the video path, or None when there
    is nothing to compose. Raises when the audio stage hasn't run.
    Intermediates (cards, segments, SRT) stay in <out_prefix>_video/."""
    audio_dir = Path(f"{out_prefix}_audio")
    if not audio_dir.is_dir():
        raise RuntimeError(f"{audio_dir} not found — the video stage composes over --audio output")

    pairs = collect_line_files(audio_dir, conversation)
    if not pairs:
        return None

    if gap_seconds is None:
        gap_seconds = load_voices().get("defaults", {}).get("gap_seconds", 0.4)
    visuals = visuals if visuals is not None else _load_visuals_or_defaults()
    cfg = visuals["defaults"]
    encoder = pick_h264_encoder(cfg.get("video_codec", ""))
    if not encoder:
        raise RuntimeError("no H.264 encoder in this ffmpeg build "
                           f"(looked for: {', '.join(H264_ENCODERS)})")
    scene_cast = {entry["id"] for entry, _ in pairs}

    workdir = Path(f"{out_prefix}_video")
    workdir.mkdir(parents=True, exist_ok=True)

    durations = [audio_duration(path) for _, path in pairs]
    (workdir / "_subs.srt").write_text(
        build_srt([entry for entry, _ in pairs], durations, gap_seconds))

    seg_paths = []
    for n, ((entry, audio_file), dur) in enumerate(zip(pairs, durations)):
        # The gap trails every segment except the last — matching the audio
        # stitch, which puts no silence after the final line.
        seg_seconds = dur + (gap_seconds if n < len(pairs) - 1 else 0.0)
        color = character_color(entry["id"], scene_cast, visuals)
        card_path = workdir / f"card_{n:03d}.png"
        render_card(entry, spoken_text(entry["text"]), color, cfg, card_path)
        seg_path = workdir / f"seg_{n:03d}.mp4"
        _run_ffmpeg(segment_cmd(card_path.resolve(), audio_file.resolve(),
                                seg_path, seg_seconds, cfg, encoder))
        seg_paths.append(seg_path)

    list_path = workdir / "_concat.txt"
    list_path.write_text("".join(concat_entry(p.resolve()) for p in seg_paths))
    out_path = Path(f"{out_prefix}_video.mp4").resolve()
    _run_ffmpeg(finalize_cmd(list_path.resolve(), out_path, encoder))
    return f"{out_prefix}_video.mp4"


def preflight():
    """Video-stage checks beyond the audio preflight (which already covers
    ffmpeg/ffprobe existence): Pillow must be installed and the ffmpeg build
    must have an H.264 encoder. Text filters are deliberately NOT required —
    cards are Pillow-rendered precisely so minimal builds work."""
    problems = []
    if importlib.util.find_spec("PIL") is None:
        problems.append("Pillow not installed — run: uv sync --extra video")
    try:
        # Validates the JSON now (not mid-render) and puts the visuals hash
        # into the audit manifest before the run-log header is written
        _load_visuals_or_defaults()
    except json.JSONDecodeError as e:
        problems.append(f"visuals.json is not valid JSON: {e}")
    problems += _encoder_problems()
    return problems


def _encoder_problems():
    try:
        if not pick_h264_encoder():
            return ["no H.264 encoder in this ffmpeg build "
                    f"(looked for: {', '.join(H264_ENCODERS)})"]
    except FileNotFoundError:
        return ["ffmpeg not found on PATH (brew install ffmpeg)"]
    return []


# ─────────────────────────────────────────────
# Veo-native render mode (--render-mode veo)
# ─────────────────────────────────────────────

def veo_client():
    from google import genai  # lazy: optional veo dep
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _generate_veo_clip(client, model, prompt, poll_seconds, timeout_seconds):
    """One beat → one clip: submit, poll the long-running operation, download.
    Veo generation runs minutes per clip; the deadline stops a stuck poll."""
    from google.genai import types  # lazy: optional veo dep
    op = client.models.generate_videos(
        model=model, prompt=prompt,
        config=types.GenerateVideosConfig(aspect_ratio="9:16"))
    deadline = time.time() + timeout_seconds
    while not op.done:
        if time.time() > deadline:
            raise TimeoutError(f"Veo operation still running after {timeout_seconds}s")
        time.sleep(poll_seconds)
        op = client.operations.get(op)
    return client.files.download(file=op.response.generated_videos[0].video)


def render_veo_video(beats, out_prefix, client=None, poll_seconds=15, timeout_seconds=900):
    """Generate one Veo clip per production beat (each carries its dialogue in
    the prompt) and concatenate them into <out_prefix>_video.mp4. EXPENSIVE —
    every beat is a paid video-generation call, and Veo's native voices drift
    between clips; that tradeoff vs. the TTS-composited default is documented
    in the README. A failed beat is recorded and skipped, same as everywhere
    else in the pipeline. Clips stay in <out_prefix>_veo/ for editing."""
    beats = [b for b in beats if (b.get("veo3_video_prompt") or "").strip()]
    if not beats:
        return None
    cfg = _load_visuals_or_defaults()["defaults"]
    model = cfg.get("veo_model", VEO_MODEL)
    encoder = pick_h264_encoder(cfg.get("video_codec", ""))
    if not encoder:
        raise RuntimeError("no H.264 encoder in this ffmpeg build "
                           f"(looked for: {', '.join(H264_ENCODERS)})")
    if client is None:
        client = veo_client()

    workdir = Path(f"{out_prefix}_veo")
    workdir.mkdir(parents=True, exist_ok=True)
    log = telemetry.get_run_log()
    clip_paths = []
    for i, beat in enumerate(beats):
        start_time = time.time()
        try:
            clip_bytes = _generate_veo_clip(client, model, beat["veo3_video_prompt"],
                                            poll_seconds, timeout_seconds)
        except Exception as e:
            telemetry.record_failure(f"gemini/{model}")
            log.record(caller="veo", provider="gemini", model=model,
                       latency_s=round(time.time() - start_time, 2),
                       error=type(e).__name__)
            print(f"  ⚠️  Veo beat {i + 1} failed ({type(e).__name__}) — beat skipped")
            continue
        path = workdir / f"beat_{i:03d}.mp4"
        path.write_bytes(clip_bytes)
        # cost_usd=None: the API doesn't report Veo pricing per operation
        log.record(caller="veo", provider="gemini", model=model,
                   latency_s=round(time.time() - start_time, 2), cost_usd=None)
        clip_paths.append(path)

    if not clip_paths:
        return None
    list_path = workdir / "_concat.txt"
    list_path.write_text("".join(concat_entry(p.resolve()) for p in clip_paths))
    out_path = Path(f"{out_prefix}_video.mp4").resolve()
    _run_ffmpeg(finalize_cmd(list_path.resolve(), out_path, encoder))
    return f"{out_prefix}_video.mp4"


def veo_preflight():
    """--render-mode veo checks, before any LLM or Veo spend."""
    problems = []
    if not os.environ.get("GEMINI_API_KEY"):
        problems.append("GEMINI_API_KEY is not set in .env (Veo renders via the Gemini API)")
    try:
        sdk_present = importlib.util.find_spec("google.genai") is not None
    except ModuleNotFoundError:
        sdk_present = False
    if not sdk_present:
        problems.append("google-genai not installed — run: uv sync --extra veo")
    problems += _encoder_problems()
    return problems
