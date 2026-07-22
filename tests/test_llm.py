"""Unit tests for the LiteLLM layer: reliability kwargs, fallback detection,
failure recording, cost estimation. litellm.completion is always mocked —
no network calls."""

import pytest

from exec_producer import llm, telemetry
from .conftest import make_response

MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture
def reliability(monkeypatch):
    config = {"num_retries": 2, "timeout_seconds": 30, "fallbacks": {}}
    monkeypatch.setattr(llm, "RELIABILITY", config)
    return config


def capture_completion(monkeypatch, response=None, error=None):
    """Mock litellm.completion, capturing the kwargs it was called with."""
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        if error is not None:
            raise error
        return response if response is not None else make_response()

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    return captured


def test_reliability_kwargs_passed(monkeypatch, reliability):
    captured = capture_completion(monkeypatch)
    llm.call_model("some-model", MSGS, max_tokens=123)
    assert captured["num_retries"] == 2
    assert captured["timeout"] == 30
    assert captured["max_tokens"] == 123
    assert "fallbacks" not in captured  # none configured for this model


def test_fallbacks_passed_only_when_configured(monkeypatch, reliability):
    reliability["fallbacks"]["some-model"] = ["backup-model"]
    captured = capture_completion(monkeypatch)
    llm.call_model("some-model", MSGS)
    assert captured["fallbacks"] == ["backup-model"]


def test_gemini_gets_token_bump_and_safety_settings(monkeypatch, reliability):
    captured = capture_completion(monkeypatch)
    llm.call_model("gemini/some-gemini", MSGS, max_tokens=400)
    assert captured["max_tokens"] == 2000
    assert len(captured["safety_settings"]) == 4


def test_failure_recorded_and_raised(monkeypatch, reliability):
    capture_completion(monkeypatch, error=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        llm.call_model("bad-model", MSGS, caller="claude")
    assert telemetry.get_run_log().failures["bad-model"] == 1
    rec = telemetry.get_run_log().records[-1]
    assert rec["error"] == "RuntimeError"
    assert rec["caller"] == "claude"


def test_fallback_detected_by_positive_match(monkeypatch, reliability):
    reliability["fallbacks"]["primary/model-x"] = ["backup-model"]
    resp = make_response(model="backup-model-2026-01-01")
    capture_completion(monkeypatch, response=resp)
    llm.call_model("primary/model-x", MSGS)
    assert telemetry.get_run_log().fallbacks["primary/model-x"] == 1
    rec = telemetry.get_run_log().records[-1]
    assert rec["fallback_used"] is True
    assert rec["responded_model"] == "backup-model-2026-01-01"  # who actually answered


def test_provider_resolved_when_hidden_params_missing(monkeypatch, reliability):
    # Observed live: direct Anthropic model strings leave custom_llm_provider
    # unset — the litellm static resolver is the fallback.
    resp = make_response(model="claude-sonnet-4-6")
    resp._hidden_params = {}
    capture_completion(monkeypatch, response=resp)
    llm.call_model("claude-sonnet-4-6", MSGS)
    assert telemetry.get_run_log().records[-1]["provider"] == "anthropic"


def test_canonical_name_drift_is_not_a_fallback(monkeypatch, reliability):
    # Regression: xAI echoes "grok-4.20-0309-..." for a requested
    # "grok-4.20-beta-0309-..." — name drift must not count as a fallback.
    reliability["fallbacks"]["xai/grok-4.20-beta-0309"] = ["backup-model"]
    resp = make_response(model="grok-4.20-0309")
    capture_completion(monkeypatch, response=resp)
    llm.call_model("xai/grok-4.20-beta-0309", MSGS)
    assert telemetry.get_run_log().fallbacks == {}
    assert telemetry.get_run_log().records[-1]["fallback_used"] is False


def test_success_record_fields(monkeypatch, reliability):
    resp = make_response(content="line", prompt_tokens=100, completion_tokens=20, cost=0.005)
    capture_completion(monkeypatch, response=resp)
    text, usage = llm.call_model("some-model", MSGS, caller="deepseek")
    assert text == "line"
    rec = telemetry.get_run_log().records[-1]
    assert rec["caller"] == "deepseek"
    assert rec["prompt_tokens"] == 100
    assert rec["completion_tokens"] == 20
    assert rec["total_tokens"] == 120
    assert rec["cost_usd"] == 0.005
    assert rec["provider"] == "testprov"


def test_cost_none_when_model_unknown(monkeypatch, reliability):
    resp = make_response()
    resp._hidden_params = {}

    def raise_cost(**kwargs):
        raise ValueError("model not in cost map")

    monkeypatch.setattr(llm.litellm, "completion_cost", raise_cost)
    capture_completion(monkeypatch, response=resp)
    llm.call_model("unknown-model", MSGS)
    assert telemetry.get_run_log().records[-1]["cost_usd"] is None


def test_call_model_simple_builds_messages(monkeypatch, reliability):
    captured = capture_completion(monkeypatch, response=make_response(content="out"))
    text = llm.call_model_simple("some-model", "SYS", "USER", caller="production")
    assert text == "out"
    assert captured["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert telemetry.get_run_log().records[-1]["caller"] == "production"
