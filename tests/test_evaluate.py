"""Unit tests for the judge eval pass. The judge call is always mocked."""

import json

import pytest

from exec_producer import evaluate

FAKE_CAST = {
    "alpha": {"name": "Alpha", "tagline": "The A", "personality": "P-ALPHA"},
    "beta": {"name": "Beta", "tagline": "The B", "personality": "P-BETA"},
}

CONVERSATION = [
    {"id": "alpha", "name": "Alpha", "text": "hey"},
    {"id": "beta", "name": "Beta", "text": "hi yourself"},
]
NOTES = [{"stage_direction": "Alpha walks in"}, {"stage_direction": None}]

VALID_CARD = {
    "format_adherence": {"score": 8, "notes": "good", "violations": []},
    "voice_consistency": {"score": 7, "notes": "ok", "weakest_character": None},
    "pacing": {"score": 6, "notes": "rushed"},
    "overall": {"score": 7, "summary": "solid"},
}


@pytest.fixture(autouse=True)
def fake_world(monkeypatch):
    monkeypatch.setattr(evaluate, "CAST", FAKE_CAST)
    monkeypatch.setattr(evaluate, "CRITICAL_RULES", "CRIT_RULES")
    monkeypatch.setattr(evaluate, "INTENSITY_GUIDE", {"low": "INT_LOW", "high": "INT_HIGH"})
    monkeypatch.setattr(evaluate, "EVAL_RUBRIC",
                        "CR=__CRITICAL_RULES__|CS=__CAST_SHEET__|I=__INTENSITY__|S=__SCRIPT__")
    monkeypatch.setattr(evaluate.time, "sleep", lambda s: None)


def judge_returning(monkeypatch, payloads):
    calls = {"n": 0}

    def fake(model, system, user, max_tokens=400, caller=None):
        i = min(calls["n"], len(payloads) - 1)
        calls["n"] += 1
        payload = payloads[i]
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr(evaluate, "call_model_simple", fake)
    return calls


def test_build_script_text_includes_stage_directions():
    text = evaluate.build_script_text(CONVERSATION, NOTES)
    assert text == "[Alpha walks in]\nAlpha: hey\nBeta: hi yourself"


def test_build_judge_prompt_fills_placeholders():
    prompt = evaluate.build_judge_prompt(CONVERSATION, NOTES, ["alpha", "beta"], "low")
    assert "CR=CRIT_RULES" in prompt
    assert "Alpha (The A): P-ALPHA" in prompt
    assert "I=INT_LOW" in prompt
    assert "Beta: hi yourself" in prompt
    assert "__" not in prompt  # every placeholder replaced


def test_evaluate_scene_valid_fenced_json(monkeypatch):
    calls = judge_returning(monkeypatch, ["```json\n" + json.dumps(VALID_CARD) + "\n```"])
    card = evaluate.evaluate_scene(CONVERSATION, NOTES, ["alpha", "beta"], "low")
    assert calls["n"] == 1
    assert card["overall"]["score"] == 7


def test_evaluate_scene_malformed_json_retries_then_none(monkeypatch, capsys):
    calls = judge_returning(monkeypatch, ["not json"])
    card = evaluate.evaluate_scene(CONVERSATION, NOTES, ["alpha", "beta"], "low")
    assert calls["n"] == 3
    assert card is None
    assert "malformed" in capsys.readouterr().out


def test_evaluate_scene_missing_keys_retries(monkeypatch):
    incomplete = json.dumps({"format_adherence": {"score": 5}})
    calls = judge_returning(monkeypatch, [incomplete, json.dumps(VALID_CARD)])
    card = evaluate.evaluate_scene(CONVERSATION, NOTES, ["alpha", "beta"], "low")
    assert calls["n"] == 2
    assert card == VALID_CARD


def test_evaluate_scene_non_dict_json_returns_none(monkeypatch):
    # Regression: a bare int crashed the REQUIRED_KEYS check (TypeError),
    # violating the never-crash contract.
    calls = judge_returning(monkeypatch, ["5"])
    card = evaluate.evaluate_scene(CONVERSATION, NOTES, ["alpha", "beta"], "low")
    assert calls["n"] == 3
    assert card is None


def test_evaluate_scene_string_containing_keys_returns_none(monkeypatch):
    # Regression: `k in card` on a string is a substring check, so a quoted
    # string listing the key names used to pass validation.
    sneaky = json.dumps("format_adherence voice_consistency pacing overall")
    calls = judge_returning(monkeypatch, [sneaky])
    card = evaluate.evaluate_scene(CONVERSATION, NOTES, ["alpha", "beta"], "low")
    assert calls["n"] == 3
    assert card is None


def test_evaluate_scene_api_error_degrades_without_retry(monkeypatch, capsys):
    calls = judge_returning(monkeypatch, [RuntimeError("judge down")])
    card = evaluate.evaluate_scene(CONVERSATION, NOTES, ["alpha", "beta"], "low")
    assert calls["n"] == 1  # API errors already retried inside call_model
    assert card is None
    assert "Judge unavailable" in capsys.readouterr().out


def test_summarize_scorecard():
    line = evaluate.summarize_scorecard(VALID_CARD)
    assert line == "format 8/10 | voice 7/10 | pacing 6/10 | overall 7/10"


def test_summarize_scorecard_tolerates_junk():
    assert "?" in evaluate.summarize_scorecard({"overall": "not a dict"})
