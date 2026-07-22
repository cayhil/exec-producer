"""Shared fixtures: isolated telemetry per test and fake LiteLLM responses."""

from types import SimpleNamespace

import pytest

from exec_producer import telemetry


@pytest.fixture(autouse=True)
def clean_telemetry(tmp_path):
    """Point the run log at a temp dir so tests never write into scripts/.
    A fresh RunLog per test also means fresh reliability counters — they
    live on the log, not at module level."""
    telemetry._run_log = telemetry.RunLog(log_dir=tmp_path)
    yield
    telemetry._run_log = None


def make_response(content="hello", model="test-model", prompt_tokens=10,
                  completion_tokens=5, cost=0.001):
    """Minimal stand-in for a litellm.completion() response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        model=model,
        _hidden_params={"response_cost": cost, "custom_llm_provider": "testprov"},
    )
