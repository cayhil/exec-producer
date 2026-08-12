"""Secret redaction for anything that reaches a log line.

The engine prints provider-controlled text: `call_model` surfaces the first
line of a failed call's exception (llm.py), and provider auth errors have a
habit of echoing the credential they rejected. The JSONL run log is also
exactly the artifact a user attaches to a bug report. So everything that
goes through the package logger passes through here first.

Two masking levels, matching the usual practice for credential display:
a token shorter than 18 characters is replaced whole (6+4 of a 12-char
secret IS the secret), while a longer one keeps its first 6 and last 4 so
"which key was that?" stays answerable from the log.

Deliberately conservative on patterns: a false positive eats the very
error message the user needs to read, which is worse than the marginal
case this would have caught."""

import logging
import os
import re

# Snapshot at import so nothing mid-run — including a model-authored
# `export` in some future tool call — can quietly turn redaction off.
ENABLED = os.environ.get("EXEC_PRODUCER_REDACT", "true").strip().lower() not in (
    "0", "false", "no", "off")

_KEEP_PREFIX = 6
_KEEP_SUFFIX = 4
# Below this, first-6 + last-4 would expose (nearly) the whole token.
_MIN_STUB_LEN = 18

# Vendor key shapes. The first five are the providers this engine actually
# calls; the rest are cheap insurance for keys that share a .env with them.
_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",       # OpenAI, Anthropic (sk-ant-*), DeepSeek
    r"sk_[A-Za-z0-9]{10,}",         # ElevenLabs (underscore, not dash)
    r"xai-[A-Za-z0-9-]{20,}",       # xAI / Grok
    r"AIza[A-Za-z0-9_-]{30,}",      # Google / Gemini / Veo
    r"r8_[A-Za-z0-9]{10,}",         # Replicate
    r"hf_[A-Za-z0-9]{10,}",         # Hugging Face
    r"ghp_[A-Za-z0-9]{10,}",        # GitHub PAT
    r"github_pat_[A-Za-z0-9_]{10,}",
]
_PREFIX_RE = re.compile("|".join(_PREFIX_PATTERNS))

# Minimum length for a value to be treated as a credential by the
# name-anchored patterns. Real keys are far longer; this is what keeps
# `AUTH_MODE=oauth` and `Bearer credentials rejected` readable.
_MIN_SECRET_LEN = 12

# KEY=value, for opaque keys with no vendor prefix (Mistral, most
# self-hosted endpoints). Case-insensitive, because provider errors quote
# their config in whatever case they like (`MistralException - api_key=…`)
# — but the name is bounded by (?![A-Za-z0-9]) so TOKEN doesn't match
# inside MAX_TOKENS or TOKENIZER, which are ordinary telemetry.
_SECRET_ENV_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(
    rf"((?<![A-Za-z0-9_])[A-Za-z0-9_]{{0,50}}{_SECRET_ENV_NAMES}(?![A-Za-z0-9])"
    # The value stops at whitespace, quote, or separator: `(\S+)` used to
    # swallow `,model=gpt-5.4,temperature=0` into the mask.
    rf"[A-Za-z0-9_]{{0,50}}\s*=\s*)(['\"]?)([^\s,;)'\"]+)",
    re.IGNORECASE)

# "api_key": "value" — the shape a provider error quotes a request body in.
# Single quotes included: Python dict reprs land in exception text often.
_JSON_FIELD_RE = re.compile(
    r"""(['"](?:x-)?(?:api[_-]?key|token|secret|password|access_token|refresh_token|"""
    r"""auth_token|authorization|private_key)['"]\s*:\s*['"])([^'"]+)(['"])""",
    re.IGNORECASE)

_BEARER_RE = re.compile(
    rf"((?:bearer|basic|token)\s+)([A-Za-z0-9._~+/=-]{{{_MIN_SECRET_LEN},}})",
    re.IGNORECASE)


def mask(secret):
    """Mask one secret value. Public so callers can redact a value they
    already know is sensitive, without pattern-matching for it."""
    secret = str(secret)
    if len(secret) < _MIN_STUB_LEN:
        return "<redacted>"
    return f"<redacted {secret[:_KEEP_PREFIX]}…{secret[-_KEEP_SUFFIX:]}>"


def redact(text):
    """Mask every credential-shaped run of characters in `text`.

    Key-name-anchored patterns (KEY=, "api_key":, Bearer) run before the
    vendor-prefix ones: they are the more specific match, and running them
    first means a value they already masked can't be re-masked by a prefix
    pattern into nested markers.
    """
    if not ENABLED or not text:
        return text
    text = str(text)
    text = _ENV_ASSIGN_RE.sub(_mask_env_assignment, text)
    text = _JSON_FIELD_RE.sub(lambda m: m.group(1) + mask(m.group(2)) + m.group(3), text)
    text = _BEARER_RE.sub(_mask_if_secretlike, text)
    return _PREFIX_RE.sub(lambda m: mask(m.group(0)), text)


def _mask_env_assignment(match):
    """KEY=value, preserving the quote style. A value too short to be a
    credential is left alone: `AUTH_MODE=oauth` and `TIMEOUT_SECRET=30` are
    diagnostics, and masking them costs more than it protects."""
    name, quote, value = match.group(1), match.group(2), match.group(3)
    if len(value) < _MIN_SECRET_LEN:
        return match.group(0)
    return f"{name}{quote}{mask(value)}"


def _mask_if_secretlike(match):
    """Bearer/Basic/Token followed by something. English words after those
    schemes are ordinary 401 prose ("Bearer credentials rejected"), so only
    mask values that don't look like a word."""
    scheme, value = match.group(1), match.group(2)
    if value.isalpha():
        return match.group(0)
    return scheme + mask(value)


class RedactingFormatter(logging.Formatter):
    """Formatter that redacts the fully rendered line — message, args, and
    any exception traceback — rather than the raw msg. A key interpolated
    from an arg, or one buried in a chained provider traceback, is the
    likeliest way it would leak."""

    def format(self, record):
        return redact(super().format(record))
