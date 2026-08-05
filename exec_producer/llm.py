"""API caller — LiteLLM is the single interface for all providers."""

import json
import os
import re
import time

import litellm

from . import telemetry
from .config import DIRECTOR_MODEL, RELIABILITY

# Suppress LiteLLM's noisy logs
litellm.suppress_debug_info = True

log = telemetry.get_logger(__name__)


def api_key_set(env_key):
    """True when env_key holds a usable API key. Whitespace-only values and
    values starting with '#' count as unset: a .env written in the inline-
    comment style (`KEY=   # note`) reaches os.environ as the literal comment
    text, and gating on truthiness alone would wave it through to a doomed,
    confusing auth failure."""
    value = os.environ.get(env_key or "", "").strip()
    return bool(value) and not value.startswith("#")


def parse_json_response(raw):
    """Parse a model's JSON reply, tolerating markdown code fences.

    Raises json.JSONDecodeError on malformed JSON; callers own the retry
    policy. May return any JSON type — callers must isinstance-check."""
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


def _estimate_cost(response):
    """Cost in USD from LiteLLM's price map; None when the model is unknown."""
    try:
        cost = response._hidden_params.get("response_cost")
        if cost is not None:
            return cost
    except Exception:
        pass
    try:
        return litellm.completion_cost(completion_response=response)
    except Exception:
        return None


def call_model(model, messages, max_tokens=400, caller=None):
    """Call any LLM provider through LiteLLM with a full messages array.

    Transient errors (rate limits, timeouts, 5xx) are retried with backoff by
    LiteLLM itself (num_retries). If fallbacks are configured for this model
    in world.json, LiteLLM tries them in order once the primary is exhausted.
    Errors that survive all of that propagate to the caller, where the engine
    degrades gracefully (director → random pick, character → stage direction).

    Every call — success or failure — is recorded in the JSONL run log.
    `caller` tags the record ("director", "production", or a cast id) so the
    run summary can attribute cost per character.
    """
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        num_retries=RELIABILITY["num_retries"],
        timeout=RELIABILITY["timeout_seconds"],
    )
    fallbacks = RELIABILITY["fallbacks"].get(model)
    if fallbacks:
        kwargs["fallbacks"] = fallbacks

    # Gemini 2.5 Pro is a "thinking" model that burns tokens on internal reasoning
    # before generating output. Bump max_tokens so it has room for both.
    # Also relax safety filters for reality TV content.
    if model.startswith("gemini/"):
        kwargs["max_tokens"] = max(max_tokens, 2000)
        kwargs["safety_settings"] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    start_time = time.time()
    try:
        response = litellm.completion(**kwargs)
    except Exception as e:
        telemetry.record_failure(model)
        telemetry.get_run_log().record(
            caller=caller,
            model=model,
            latency_s=round(time.time() - start_time, 2),
            num_retries_configured=RELIABILITY["num_retries"],
            error=type(e).__name__,
        )
        reason = str(e).strip().splitlines()[0][:200] if str(e).strip() else ""
        log.warning(f"call failed after retries: {model}: {type(e).__name__}"
                    + (f" — {reason}" if reason else ""))
        raise
    elapsed = time.time() - start_time

    # Detect fallback by positive match: providers echo canonical names that
    # can differ from the requested string (xAI drops "-beta"), so only count
    # a fallback when the responder matches a configured fallback model. The
    # most specific match wins: a fallback list carrying a truncated variant
    # of the primary ("gpt-5" configured under "gpt-5.4") must not flag the
    # primary's own echo as a fallback.
    fallback_used = False
    responded_model = None
    if fallbacks:
        responded = str(getattr(response, "model", "") or "")
        matched = max((fb.split("/")[-1] for fb in fallbacks
                       if fb.split("/")[-1] in responded), key=len, default=None)
        requested = model.split("/")[-1]
        if matched and (requested not in responded or len(matched) > len(requested)):
            fallback_used = True
            responded_model = responded
            telemetry.record_fallback(model, responded)

    text = response.choices[0].message.content
    usage = getattr(response, 'usage', None)

    pt = (getattr(usage, 'prompt_tokens', 0) or 0) if usage else 0
    ct = (getattr(usage, 'completion_tokens', 0) or 0) if usage else 0
    tt = (getattr(usage, 'total_tokens', 0) or 0) if usage else 0
    cost = _estimate_cost(response)
    try:
        provider = response._hidden_params.get("custom_llm_provider")
    except Exception:
        provider = None
    if not provider:
        # Hidden params don't populate for every provider (observed: direct
        # Anthropic model strings) — fall back to LiteLLM's static resolver.
        try:
            provider = litellm.get_llm_provider(model)[1]
        except Exception:
            provider = None

    record = dict(
        caller=caller,
        model=model,
        provider=provider,
        latency_s=round(elapsed, 2),
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=tt,
        cost_usd=cost,
        fallback_used=fallback_used,
    )
    if responded_model:
        record["responded_model"] = responded_model
    telemetry.get_run_log().record(**record)
    cost_text = f" | ${cost:.4f}" if cost is not None else ""
    log.debug(f"{model} | {len(messages)} msgs | {pt}p+{ct}c tokens | {elapsed:.1f}s{cost_text}")

    return text, usage


def call_model_simple(model, system_prompt, user_message, max_tokens=400, caller=None):
    """Convenience wrapper for stateless calls (director casting, production prompts, etc)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    text, usage = call_model(model, messages, max_tokens, caller=caller)
    return text


def call_director(system_prompt, user_message):
    """Stateless single-turn call to the director model."""
    return call_model_simple(DIRECTOR_MODEL, system_prompt, user_message, caller="director")
