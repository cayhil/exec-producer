"""Audio stage — per-line TTS rendering and per-scene track assembly.

Voice consistency is the design rationale: Veo's native audio cannot hold a
character's voice across clips, so each character maps to a persistent voice
ID in data/voices.json and every dialogue line renders with it, then the
lines are stitched into one track per scene with ffmpeg.

Media dependencies are deliberately absent at module import — the elevenlabs
SDK loads lazily inside the backend and ffmpeg is an external binary — so
the core engine runs without them; --audio fails fast at CLI preflight
instead of mid-run.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import telemetry
from .config import load_voices
from .llm import api_key_set

_ACTION_MARKUP = re.compile(r"\[[^\]]*\]|\*[^*]*\*")


def spoken_text(text):
    """Strip action markup — [bracketed] stage directions (including the
    API-failure fallback lines, which are fully bracketed) and *asterisk*
    beats — leaving only the words a voice would actually say."""
    return re.sub(r"\s+", " ", _ACTION_MARKUP.sub(" ", text)).strip()


# ─────────────────────────────────────────────
# TTS backends
# ─────────────────────────────────────────────

class ElevenLabsBackend:
    """The one concrete TTS backend. synthesize(text, voice) -> mp3 bytes,
    where voice is that character's entry from voices.json (merged over the
    file's defaults). The SDK import happens here, not at module import."""

    def __init__(self, defaults):
        # Lazy: optional media deps, imported only when the backend is built
        from elevenlabs.client import ElevenLabs
        from elevenlabs.types import VoiceSettings
        self.client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
        self._voice_settings_cls = VoiceSettings
        self.defaults = defaults

    def synthesize(self, text, voice):
        cfg = {**self.defaults, **voice}
        kwargs = {}
        if cfg.get("voice_settings"):
            # The SDK wants its typed model, not a bare dict
            kwargs["voice_settings"] = self._voice_settings_cls(**cfg["voice_settings"])
        chunks = self.client.text_to_speech.convert(
            voice_id=cfg["voice_id"],
            text=text,
            model_id=cfg.get("model_id", "eleven_multilingual_v2"),
            output_format=cfg.get("output_format", "mp3_44100_128"),
            **kwargs,
        )
        return b"".join(chunks)

    def convert_speech(self, audio_path, voice):
        """Speech-to-speech: re-voice existing audio into this character's
        persistent voice, keeping words and timing (the --render-mode vlog
        voice-normalization step — same words + same timing keeps Veo's lip
        movement plausibly synced). Note the sts model id: speech-to-speech
        uses its own model family, not the TTS one."""
        cfg = {**self.defaults, **voice}
        kwargs = {}
        if cfg.get("voice_settings"):
            # Unlike text_to_speech, the STS endpoint takes voice_settings as
            # a JSON string (multipart form field), not the typed model.
            kwargs["voice_settings"] = json.dumps(cfg["voice_settings"])
        with open(audio_path, "rb") as f:
            chunks = self.client.speech_to_speech.convert(
                voice_id=cfg["voice_id"],
                audio=f,
                model_id=cfg.get("sts_model_id", "eleven_multilingual_sts_v2"),
                output_format=cfg.get("output_format", "mp3_44100_128"),
                **kwargs,
            )
            return b"".join(chunks)


def get_backend(provider, defaults):
    """TTS backend factory — the seam where a different provider would plug
    in. Only ElevenLabs is implemented; the contract is one method:
    synthesize(text, voice) -> encoded audio bytes."""
    if provider == "elevenlabs":
        return ElevenLabsBackend(defaults)
    raise ValueError(f"Unknown TTS provider: {provider}")


# ─────────────────────────────────────────────
# ffmpeg assembly
# ─────────────────────────────────────────────

def probe_stream(path):
    """Sample rate and channel count of an audio file (via ffprobe)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    return int(stream["sample_rate"]), int(stream["channels"])


def silence_cmd(gap_path, seconds, sample_rate, channels):
    layout = "mono" if channels == 1 else "stereo"
    return ["ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl={layout}",
            "-t", str(seconds), "-c:a", "libmp3lame", "-b:a", "128k", str(gap_path)]


def concat_entry(path):
    """One concat-demuxer list line. Single quotes are escaped per the
    ffmpeg concat spec ('\\'') — paths contain user-controlled components
    (checkout location, show identifiers), and an unescaped apostrophe
    kills the whole stitch."""
    escaped = str(path).replace("'", "'\\''")
    return f"file '{escaped}'\n"


def concat_list(line_files, gap_path=None):
    """Concat-demuxer list: a gap between consecutive lines, none after the
    last (no gap entries at all when gap_path is None). Absolute paths, so
    the list works regardless of ffmpeg's cwd."""
    entries = []
    for p in line_files:
        entries.append(str(Path(p).resolve()))
        if gap_path is not None:
            entries.append(str(Path(gap_path).resolve()))
    if gap_path is not None:
        entries = entries[:-1]
    return "".join(concat_entry(e) for e in entries)


def concat_cmd(list_path, out_path):
    # Re-encode rather than stream-copy: it normalizes the gap/line joins and
    # produces a clean duration header.
    return ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(list_path), "-c:a", "libmp3lame", "-b:a", "128k", str(out_path)]


def _run_ffmpeg(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}")


def stitch_lines(line_files, out_path, gap_seconds=0.4, workdir=None):
    """Concatenate per-line mp3s with a silence gap between lines. The gap is
    generated to match the first line's sample rate/channels — the concat
    demuxer needs uniform stream parameters across entries.

    gap_seconds <= 0 means no gap file at all: ffmpeg renders a zero-length
    silence as a frameless mp3, and the concat demuxer hits it, errors on
    stderr, yet still EXITS 0 — silently dropping every line after the first
    gap entry. The gap must be absent from the list, not present-but-empty."""
    workdir = Path(workdir) if workdir else Path(line_files[0]).parent
    gap_path = None
    if gap_seconds > 0 and len(line_files) > 1:
        rate, channels = probe_stream(line_files[0])
        gap_path = workdir / "_gap.mp3"
        _run_ffmpeg(silence_cmd(gap_path, gap_seconds, rate, channels))
    list_path = workdir / "_concat.txt"
    list_path.write_text(concat_list(line_files, gap_path))
    _run_ffmpeg(concat_cmd(list_path, out_path))
    return out_path


# ─────────────────────────────────────────────
# The stage
# ─────────────────────────────────────────────

def render_scene_audio(conversation, out_prefix, voices=None, backend=None):
    """Render every spoken line to its character's voice and stitch the
    result into <out_prefix>_audio.mp3. Per-line mp3s stay in
    <out_prefix>_audio/ for manual editing. Returns the track path, or None
    when nothing was spoken. A failed line is recorded and skipped — the
    track still renders, matching the engine's degradation philosophy."""
    voices_cfg = voices if voices is not None else load_voices()
    defaults = voices_cfg.get("defaults", {})
    voice_map = voices_cfg.get("voices", {})
    provider = voices_cfg.get("provider", "elevenlabs")
    if backend is None:
        backend = get_backend(provider, defaults)

    line_dir = Path(f"{out_prefix}_audio")
    line_dir.mkdir(parents=True, exist_ok=True)
    log = telemetry.get_run_log()
    line_files = []
    for i, entry in enumerate(conversation):
        # Budget gate. TTS is per-line and unbounded, and it runs AFTER the
        # generation loop — without this, a run that stopped at its cap would
        # go on to bill ElevenLabs for every line it had already produced.
        if telemetry.spend_exhausted():
            print(f"  💸 Spend cap reached — {len(conversation) - i} line(s) left unvoiced")
            break
        text = spoken_text(entry["text"])
        voice = voice_map.get(entry["id"])
        if not text or not voice:
            continue
        model_id = {**defaults, **voice}.get("model_id", "eleven_multilingual_v2")
        start_time = time.time()
        try:
            audio_bytes = backend.synthesize(text, voice)
        except Exception as e:
            telemetry.record_failure(f"{provider}/{model_id}")
            log.record(caller=f"tts:{entry['id']}", provider=provider, model=model_id,
                       latency_s=round(time.time() - start_time, 2),
                       characters=len(text), error=type(e).__name__)
            print(f"  ⚠️  TTS failed for {entry['name']} line {i + 1} ({type(e).__name__}) — line skipped")
            continue
        path = line_dir / f"line_{i:03d}_{entry['id']}.mp3"
        path.write_bytes(audio_bytes)
        # cost_usd=None on purpose: ElevenLabs bills per character by plan
        # tier, which the API response doesn't expose — the summary reports
        # these as calls with unknown cost rather than guessing.
        log.record(caller=f"tts:{entry['id']}", provider=provider, model=model_id,
                   latency_s=round(time.time() - start_time, 2),
                   characters=len(text), cost_usd=None)
        line_files.append(path)

    if not line_files:
        return None
    out_path = f"{out_prefix}_audio.mp3"
    stitch_lines(line_files, out_path,
                 gap_seconds=defaults.get("gap_seconds", 0.4), workdir=line_dir)
    return out_path


def preflight(cast_ids):
    """Everything --audio needs, checked before any LLM money is spent.
    Returns a list of problems (empty = good to go), collecting all of them
    instead of stopping at the first."""
    problems = []
    if not api_key_set("ELEVENLABS_API_KEY"):
        problems.append("ELEVENLABS_API_KEY is not set in .env")
    if importlib.util.find_spec("elevenlabs") is None:
        problems.append("elevenlabs SDK not installed — run: uv sync --extra audio")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            problems.append(f"{tool} not found on PATH (brew install ffmpeg)")
    try:
        voices_cfg = load_voices()
    except FileNotFoundError:
        problems.append("voices.json not found in the show directory — the audio stage needs a voice map")
        return problems
    except json.JSONDecodeError as e:
        problems.append(f"voices.json is not valid JSON: {e}")
        return problems
    provider = voices_cfg.get("provider", "elevenlabs")
    if provider != "elevenlabs":
        problems.append(f"unknown TTS provider '{provider}' in voices.json — only 'elevenlabs' is implemented")
    voice_map = voices_cfg.get("voices", {})
    missing = [c for c in cast_ids if not voice_map.get(c, {}).get("voice_id")]
    if missing:
        problems.append(f"voices.json has no voice_id for: {', '.join(missing)}")
    return problems
