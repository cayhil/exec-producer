"""Unit tests for the engine: prompt assembly, session management, director
degradation paths, and the scene loop. All LLM calls are mocked."""

import json
from types import SimpleNamespace

import pytest

from exec_producer import engine

FAKE_CAST = {
    "alpha": {"name": "Alpha", "provider": "openai", "model": "m-alpha", "emoji": "🅰️",
              "tagline": "The A", "gender": "guy", "personality": "P-ALPHA",
              "triggers": ["poked", "ignored"], "relationships": {"beta": "likes her", "omega": "old rival"}},
    "beta": {"name": "Beta", "provider": "openai", "model": "m-beta", "emoji": "🅱️",
             "tagline": "The B", "gender": "girl", "personality": "P-BETA",
             "triggers": ["dismissed"], "relationships": {}},
    "gamma": {"name": "Gamma", "provider": "openai", "model": "m-gamma", "emoji": "🌀",
              "tagline": "The G", "gender": "guy", "personality": "P-GAMMA",
              "triggers": ["copied"], "relationships": {}},
}

FAKE_TEMPLATE = (
    "W=__WORLD__|CS=__CAST_SHEET__|P=__PERSONALITY__|I=__INTENSITY__"
    "|EB=__EMBODIMENT_BASE__|ES=__EMBODIMENT_SPECIFIC__|EI=__EMBODIMENT_INTENSITY__"
    "|R=__RELATIONSHIPS__|T=__TRIGGERS__|TR=__TONE_RULES__"
    "|SP=__SCENE_PROMPT__|SM=__SCENE_MOOD__|PL=__PLOT__|CR=__CRITICAL_RULES__"
)

SCENE = {"name": "Test Kitchen", "prompt": "the kitchen", "mood": "tense"}
CONFESSIONAL = {"name": "Confessional Booth", "prompt": "the booth", "mood": "raw"}


@pytest.fixture(autouse=True)
def fake_world(monkeypatch):
    monkeypatch.setattr(engine, "CAST", FAKE_CAST)
    monkeypatch.setattr(engine, "CHARACTER_TEMPLATE", FAKE_TEMPLATE)
    monkeypatch.setattr(engine, "WORLD", "THE_WORLD")
    monkeypatch.setattr(engine, "INTENSITY_GUIDE", {"low": "INT_LOW", "high": "INT_HIGH"})
    monkeypatch.setattr(engine, "EMBODIMENT", {
        "base": "EMB_BASE",
        "gender": {"guy": "EMB_GUY", "girl": "EMB_GIRL"},
        "character_overrides": {"gamma": "EMB_GAMMA_OVERRIDE"},
        "intensity": {"low": "EMB_INT_LOW", "high": "EMB_INT_HIGH"},
    })
    monkeypatch.setattr(engine, "TONE_FACE_TO_FACE", "TONE_F2F")
    monkeypatch.setattr(engine, "TONE_CONFESSIONAL_EARLY", "TONE_CONF_EARLY")
    monkeypatch.setattr(engine, "TONE_CONFESSIONAL_LATE", "TONE_CONF_LATE")
    monkeypatch.setattr(engine, "CRITICAL_RULES", "CRIT_RULES")
    # Pin the show identity: file-glob assertions must not depend on which
    # show directory (private vs starter) the test process loaded.
    monkeypatch.setattr(engine, "SHOW_NAME", "TEST SHOW")
    monkeypatch.setattr(engine, "SHOW_SLUG", "test_show")
    monkeypatch.setattr(engine.time, "sleep", lambda s: None)


CAST_IDS = list(FAKE_CAST.keys())


# ── prompt assembly ──────────────────────────────────────

def test_build_cast_sheet():
    sheet = engine.build_cast_sheet(CAST_IDS)
    assert "Alpha (he/him) — The A" in sheet
    assert "Beta (she/her) — The B" in sheet


def test_build_direction_text():
    note = {"direction": "SHADE", "target": "beta",
            "stage_direction": "leans in", "emotional_state": "electric"}
    text = engine.build_direction_text(note)
    assert "SHADE" in text
    assert "focused on Beta" in text
    assert "leans in" in text
    assert "electric" in text


def test_system_prompt_assembly():
    s = engine.CharacterSession("alpha", SCENE, "low", "find the beds", CAST_IDS)
    prompt = s.messages[0]["content"]
    assert s.messages[0]["role"] == "system"
    assert "W=THE_WORLD" in prompt
    assert "P=P-ALPHA" in prompt
    assert "I=INT_LOW" in prompt
    assert "ES=EMB_GUY" in prompt          # gender default, no override for alpha
    assert "TR=TONE_F2F" in prompt         # non-confessional scene
    assert "SCENE PLOT: find the beds" in prompt
    assert "likes her" in prompt           # relationship with present cast member
    assert "old rival" not in prompt       # "omega" not in this scene's cast
    assert "- poked" in prompt


def test_character_override_and_confessional_tones():
    s = engine.CharacterSession("gamma", CONFESSIONAL, "low", "", CAST_IDS)
    prompt = s.messages[0]["content"]
    assert "ES=EMB_GAMMA_OVERRIDE" in prompt
    assert "TR=TONE_CONF_EARLY" in prompt

    s = engine.CharacterSession("gamma", CONFESSIONAL, "high", "", CAST_IDS)
    assert "TR=TONE_CONF_LATE" in s.messages[0]["content"]


# ── session management ───────────────────────────────────

def test_get_messages_consolidates_consecutive_roles():
    s = engine.CharacterSession("alpha", SCENE, "low", "", CAST_IDS)
    s.inject_context("event one")
    s.inject_context("event two")
    s.add_own_response("my line")
    assert len(s.messages) == 4
    consolidated = s.get_messages()
    assert [m["role"] for m in consolidated] == ["system", "user", "assistant"]
    assert "event one" in consolidated[1]["content"]
    assert "event two" in consolidated[1]["content"]


def test_trim_keeps_system_prompt_and_recent_messages():
    s = engine.CharacterSession("alpha", SCENE, "low", "", CAST_IDS)
    for i in range(40):
        s.inject_context(f"event {i}")
    s.trim_if_needed(max_messages=30)
    assert len(s.messages) == 31
    assert s.messages[0]["role"] == "system"
    assert s.messages[-1]["content"] == "event 39"


def test_inject_scene_update_only_includes_events_since_last_line():
    s = engine.CharacterSession("alpha", SCENE, "low", "", CAST_IDS)
    conversation = [
        {"id": "alpha", "name": "Alpha", "text": "old alpha line"},
        {"id": "beta", "name": "Beta", "text": "beta reply"},
        {"id": "gamma", "name": "Gamma", "text": "gamma jab"},
    ]
    engine.inject_scene_update(s, conversation, "DIRECTION_TEXT")
    injected = s.messages[-1]["content"]
    assert "old alpha line" not in injected
    assert "Beta: beta reply" in injected
    assert "Gamma: gamma jab" in injected
    assert "DIRECTION_TEXT" in injected


def test_inject_scene_update_first_speaker():
    s = engine.CharacterSession("alpha", SCENE, "low", "", CAST_IDS)
    engine.inject_scene_update(s, [], "DIRECTION_TEXT")
    assert "speaking first" in s.messages[-1]["content"]


# ── character line generation ────────────────────────────

def test_generate_line_strips_wrapping(monkeypatch):
    s = engine.CharacterSession("alpha", SCENE, "low", "", CAST_IDS)
    usage = SimpleNamespace(total_tokens=42)
    monkeypatch.setattr(engine, "call_model", lambda *a, **k: ('"clean line"', usage))
    text = engine.generate_line_session(s)
    assert text == "clean line"
    assert s.line_count == 1
    assert s.total_tokens == 42


def test_generate_line_empty_response_degrades(monkeypatch):
    s = engine.CharacterSession("alpha", SCENE, "low", "", CAST_IDS)
    monkeypatch.setattr(engine, "call_model", lambda *a, **k: (None, None))
    text = engine.generate_line_session(s)
    assert "Alpha" in text and text.startswith("[")


def test_generate_line_whitespace_response_degrades(monkeypatch):
    # Regression: ' \n ' is truthy, stripped to "", and the empty assistant
    # message poisoned the session for every later Anthropic call.
    s = engine.CharacterSession("alpha", SCENE, "low", "", CAST_IDS)
    monkeypatch.setattr(engine, "call_model", lambda *a, **k: ('  "" \n ', None))
    text = engine.generate_line_session(s)
    assert "Alpha" in text and text.startswith("[")
    assert s.messages[-1]["content"] == text  # never records an empty message


def test_trim_never_leaves_assistant_first(monkeypatch):
    # Regression: an assistant-first tail after trimming 400s on Anthropic.
    s = engine.CharacterSession("alpha", SCENE, "low", "", CAST_IDS)
    for i in range(20):
        s.messages.append({"role": "assistant", "content": f"a{i}"})
        s.messages.append({"role": "user", "content": f"u{i}"})
    # messages[-30] is an assistant turn; trim must drop it
    assert s.messages[-30]["role"] == "assistant"
    s.trim_if_needed(max_messages=30)
    assert s.messages[0]["role"] == "system"
    assert s.messages[1]["role"] == "user"


def test_generate_line_api_error_degrades_to_stage_direction(monkeypatch):
    s = engine.CharacterSession("alpha", SCENE, "low", "", CAST_IDS)

    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(engine, "call_model", boom)
    text = engine.generate_line_session(s)
    assert "Alpha" in text and "holding something back" in text
    assert s.messages[-1]["content"] == text  # scene continues with the fallback on record


# ── director degradation paths ───────────────────────────

def director_returning(monkeypatch, payloads):
    """Mock call_director to return each payload in order, tracking calls."""
    calls = {"n": 0}

    def fake(system, prompt):
        i = min(calls["n"], len(payloads) - 1)
        calls["n"] += 1
        payload = payloads[i]
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr(engine, "call_director", fake)
    return calls


def test_director_choice_valid(monkeypatch):
    note = {"next_speaker": "beta", "direction": "SHADE", "target": "alpha"}
    director_returning(monkeypatch, ["```json\n" + json.dumps(note) + "\n```"])
    result = engine.get_director_choice(SCENE, [], CAST_IDS, last_speaker="alpha")
    assert result["next_speaker"] == "beta"
    assert result["direction"] == "SHADE"


def test_director_choice_normalizes_display_name(monkeypatch):
    # Regression: "Beta" passed validation (compared normalized) but the raw
    # dict was returned, so CAST["Beta"] KeyError'd downstream.
    note = {"next_speaker": " Beta ", "direction": "SHADE", "target": "alpha"}
    director_returning(monkeypatch, [json.dumps(note)])
    result = engine.get_director_choice(SCENE, [], CAST_IDS, last_speaker="alpha")
    assert result["next_speaker"] == "beta"


def test_director_choice_sanitizes_non_string_target(monkeypatch):
    # Regression: a list target crashed `target in CAST` (unhashable).
    note = {"next_speaker": "beta", "direction": "SHADE", "target": ["alpha", "gamma"]}
    director_returning(monkeypatch, [json.dumps(note)])
    result = engine.get_director_choice(SCENE, [], CAST_IDS, last_speaker="alpha")
    assert result["target"] is None
    # and the sanitized note flows through the direction-text builder safely
    assert "SHADE" in engine.build_direction_text(result)


def test_director_choice_reassigns_repeat_speaker(monkeypatch):
    note = {"next_speaker": "alpha", "direction": "RESPOND_TO"}
    director_returning(monkeypatch, [json.dumps(note)])
    result = engine.get_director_choice(SCENE, [], CAST_IDS, last_speaker="alpha")
    assert result["next_speaker"] in ("beta", "gamma")


def test_director_choice_malformed_json_retries_then_falls_back(monkeypatch):
    calls = director_returning(monkeypatch, ["not json at all"])
    result = engine.get_director_choice(SCENE, [], CAST_IDS, last_speaker="alpha")
    assert calls["n"] == 3  # JSON problems are retried
    assert result["reasoning"] == "director fallback"
    assert result["next_speaker"] in ("beta", "gamma")


def test_director_choice_api_error_falls_back_without_extra_retries(monkeypatch):
    calls = director_returning(monkeypatch, [RuntimeError("api down")])
    result = engine.get_director_choice(SCENE, [], CAST_IDS, last_speaker="alpha")
    assert calls["n"] == 1  # API errors were already retried inside call_model
    assert result["reasoning"] == "director fallback"


def test_scene_cast_valid(monkeypatch):
    director_returning(monkeypatch, [json.dumps({"cast": ["alpha", "beta", "bogus"]})])
    cast = engine.get_scene_cast(SCENE, CAST_IDS)
    assert cast == ["alpha", "beta"]  # unknown ids filtered out


def test_scene_cast_api_error_uses_everyone(monkeypatch):
    calls = director_returning(monkeypatch, [RuntimeError("api down")])
    cast = engine.get_scene_cast(SCENE, CAST_IDS)
    assert calls["n"] == 1
    assert cast == CAST_IDS


# ── cross-scene tail injection ───────────────────────────

def test_inject_scene_tail_delivers_missed_lines():
    s = engine.CharacterSession("alpha", SCENE, "low", "", CAST_IDS)
    conversation = [
        {"id": "alpha", "name": "Alpha", "text": "my only line"},
        {"id": "beta", "name": "Beta", "text": "talking about alpha"},
        {"id": "gamma", "name": "Gamma", "text": "piling on"},
    ]
    engine.inject_scene_tail(s, conversation)
    injected = s.messages[-1]["content"]
    assert "my only line" not in injected
    assert "Beta: talking about alpha" in injected
    assert "Gamma: piling on" in injected
    assert "Now respond" not in injected  # passive update, not a speaking turn


def test_inject_scene_tail_silent_participant_gets_whole_scene():
    s = engine.CharacterSession("gamma", SCENE, "low", "", CAST_IDS)
    conversation = [{"id": "alpha", "name": "Alpha", "text": "hello"}]
    engine.inject_scene_tail(s, conversation)
    assert "Alpha: hello" in s.messages[-1]["content"]


def test_inject_scene_tail_no_missed_lines_injects_nothing():
    s = engine.CharacterSession("alpha", SCENE, "low", "", CAST_IDS)
    before = len(s.messages)
    engine.inject_scene_tail(s, [{"id": "alpha", "name": "Alpha", "text": "last word"}])
    assert len(s.messages) == before


def test_unwrap_reply_preserves_balanced_markup():
    # A fully-wrapped reply is unwrapped...
    assert engine._unwrap_reply('"Hello there."') == "Hello there."
    assert engine._unwrap_reply("*stares*") == "stares"
    assert engine._unwrap_reply('  "*fine*"  ') == "fine"
    # ...but balanced action markup keeps its outer asterisks — a blind
    # strip would make the survivors re-pair into dialogue-eating spans
    assert engine._unwrap_reply("*sighs* whatever, man *walks out*") == \
        "*sighs* whatever, man *walks out*"


def test_unwrap_reply_interior_quote_not_stripped():
    # Interior occurrences block unwrapping (can't tell wrapping from markup)
    text = '"Wait — she said "yes"?"'
    assert engine._unwrap_reply(text) == text


def test_transition_text_carries_scene_tone(monkeypatch):
    confessional = {"name": "The Confessional", "prompt": "booth", "mood": "raw"}
    t = engine.build_transition_text(confessional, "plot", ["alpha"], "low")
    assert "TONE RULES FOR THIS SCENE" in t and "TONE_CONF_EARLY" in t
    t = engine.build_transition_text(confessional, "plot", ["alpha"], "high")
    assert "TONE_CONF_LATE" in t
    t = engine.build_transition_text(SCENE, "plot", ["alpha"], "low")
    assert "TONE_F2F" in t


# ── scene loop ───────────────────────────────────────────

def test_run_options_defaults_match_legacy_signature():
    # RunOptions replaced run_scene's keyword tail; the defaults must not drift.
    opts = engine.RunOptions()
    assert (opts.num_lines, opts.save, opts.produce, opts.intensity, opts.plot,
            opts.debug, opts.director_only, opts.evaluate) == (
        15, True, False, "high", "", False, False, False)


def test_run_scene_loop(monkeypatch):
    speakers = iter(["alpha", "beta", "alpha"])
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(engine, "get_director_choice",
                        lambda *a, **k: {"next_speaker": next(speakers), "direction": "OPEN"})
    monkeypatch.setattr(engine, "inject_scene_update", lambda *a, **k: None)
    lines = iter(["line one", "line two", "line three"])
    monkeypatch.setattr(engine, "generate_line_session", lambda *a, **k: next(lines))

    sessions = {"alpha": SimpleNamespace(), "beta": SimpleNamespace()}
    conversation, director_notes = engine.run_scene(
        "test", ["alpha", "beta"], engine.RunOptions(num_lines=3, save=False),
        sessions=sessions)

    assert len(conversation) == 3
    assert len(director_notes) == 3
    assert conversation[0] == {"name": "Alpha", "id": "alpha", "emoji": "🅰️",
                               "tagline": "The A", "text": "line one", "direction": "OPEN"}
    assert conversation[1]["text"] == "line two"
    assert conversation[2]["id"] == "alpha"


def test_run_scene_cap_wiring(monkeypatch, capsys):
    # The gate must be applied inside run_scene, not just exist as a function.
    monkeypatch.setattr(engine, "MAX_INTENSITY", "low")
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(engine, "get_director_choice",
                        lambda *a, **k: {"next_speaker": "alpha", "direction": "OPEN"})
    monkeypatch.setattr(engine, "inject_scene_update", lambda *a, **k: None)
    monkeypatch.setattr(engine, "generate_line_session", lambda *a, **k: "hi")
    sessions = {"alpha": SimpleNamespace(), "beta": SimpleNamespace()}
    engine.run_scene("test", ["alpha", "beta"],
                     engine.RunOptions(num_lines=1, save=False, intensity="high"),
                     sessions=sessions)
    assert "Content rating cap" in capsys.readouterr().out


def test_run_scene_save_and_eval_wiring(monkeypatch, tmp_path):
    # Files must exist with the audit/telemetry/intensity keys the docs claim.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(engine, "get_director_choice",
                        lambda *a, **k: {"next_speaker": "alpha", "direction": "OPEN"})
    monkeypatch.setattr(engine, "inject_scene_update", lambda *a, **k: None)
    monkeypatch.setattr(engine, "generate_line_session", lambda *a, **k: "hi there")
    card = {"format_adherence": {"score": 8}, "voice_consistency": {"score": 7},
            "pacing": {"score": 6}, "overall": {"score": 7}}
    monkeypatch.setattr(engine, "evaluate_scene", lambda *a, **k: card)

    engine.run_scene("test", ["alpha", "beta"],
                     engine.RunOptions(num_lines=1, save=True, intensity="low",
                                       evaluate=True))

    txts = list((tmp_path / "scripts").glob("test_show_test_*.txt"))
    jsons = [p for p in (tmp_path / "scripts").glob("test_show_test_*.json")
             if not p.name.endswith("_eval.json")]
    evals = list((tmp_path / "scripts").glob("test_show_test_*_eval.json"))
    assert len(txts) == 1 and len(jsons) == 1 and len(evals) == 1

    data = json.loads(jsons[0].read_text())
    assert data["intensity"] == "low"
    assert data["audit"]["config_files"]
    assert data["telemetry"]["calls"] >= 0
    eval_data = json.loads(evals[0].read_text())
    assert eval_data["scorecard"] == card
    assert eval_data["audit"]["config_files"]


def test_run_scene_audio_wiring(monkeypatch, tmp_path):
    # --audio renders off the same save prefix and embeds the path in the .json
    from exec_producer import audio as audio_mod
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(engine, "get_director_choice",
                        lambda *a, **k: {"next_speaker": "alpha", "direction": "OPEN"})
    monkeypatch.setattr(engine, "inject_scene_update", lambda *a, **k: None)
    monkeypatch.setattr(engine, "generate_line_session", lambda *a, **k: "hi")
    rendered = {}

    def fake_render(conversation, prefix):
        rendered["prefix"] = prefix
        rendered["lines"] = len(conversation)
        return f"{prefix}_audio.mp3"

    monkeypatch.setattr(audio_mod, "render_scene_audio", fake_render)

    engine.run_scene("test", ["alpha", "beta"],
                     engine.RunOptions(num_lines=1, intensity="low", audio=True))

    assert rendered["prefix"].startswith("scripts/test_show_test_")
    assert rendered["lines"] == 1
    jsons = [p for p in (tmp_path / "scripts").glob("test_show_test_*.json")
             if not p.name.endswith("_eval.json")]
    data = json.loads(jsons[0].read_text())
    assert data["audio_track"] == rendered["prefix"] + "_audio.mp3"


def _scene_harness(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(engine, "get_director_choice",
                        lambda *a, **k: {"next_speaker": "alpha", "direction": "OPEN"})
    monkeypatch.setattr(engine, "inject_scene_update", lambda *a, **k: None)
    monkeypatch.setattr(engine, "generate_line_session", lambda *a, **k: "hi")


def _scene_json(tmp_path):
    jsons = [p for p in (tmp_path / "scripts").glob("test_show_test_*.json")
             if not p.name.endswith("_eval.json")]
    return json.loads(jsons[0].read_text())


def test_run_scene_video_wiring(monkeypatch, tmp_path):
    from exec_producer import audio as audio_mod, video as video_mod
    _scene_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(audio_mod, "render_scene_audio", lambda c, p: f"{p}_audio.mp3")
    monkeypatch.setattr(video_mod, "render_scene_video", lambda c, p: f"{p}_video.mp4")

    engine.run_scene("test", ["alpha", "beta"],
                     engine.RunOptions(num_lines=1, intensity="low",
                                       audio=True, video=True))

    data = _scene_json(tmp_path)
    assert data["audio_track"].endswith("_audio.mp3")
    assert data["video_track"].endswith("_video.mp4")


def test_run_scene_video_skipped_when_audio_produced_nothing(monkeypatch, tmp_path, capsys):
    from exec_producer import audio as audio_mod, video as video_mod
    _scene_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(audio_mod, "render_scene_audio", lambda c, p: None)
    monkeypatch.setattr(video_mod, "render_scene_video",
                        lambda c, p: pytest.fail("video must not render without audio"))

    engine.run_scene("test", ["alpha", "beta"],
                     engine.RunOptions(num_lines=1, intensity="low",
                                       audio=True, video=True))

    assert "Video skipped" in capsys.readouterr().out
    data = _scene_json(tmp_path)
    assert "audio_track" not in data and "video_track" not in data


def test_run_scene_video_failure_keeps_script_and_audio(monkeypatch, tmp_path, capsys):
    from exec_producer import audio as audio_mod, video as video_mod
    _scene_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(audio_mod, "render_scene_audio", lambda c, p: f"{p}_audio.mp3")

    def boom(c, p):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(video_mod, "render_scene_video", boom)

    engine.run_scene("test", ["alpha", "beta"],
                     engine.RunOptions(num_lines=1, intensity="low",
                                       audio=True, video=True))

    assert "Video rendering failed" in capsys.readouterr().out
    data = _scene_json(tmp_path)
    assert data["audio_track"].endswith("_audio.mp3")
    assert "video_track" not in data


def test_run_scene_veo_mode_needs_no_audio(monkeypatch, tmp_path):
    from exec_producer import video as video_mod
    _scene_harness(monkeypatch, tmp_path)
    beats = {"beats": [{"veo3_video_prompt": "p1"}]}
    monkeypatch.setattr(engine, "generate_production_prompts", lambda *a, **k: beats)
    seen = {}
    monkeypatch.setattr(video_mod, "render_veo_video",
                        lambda b, p: seen.update(beats=b, prefix=p) or f"{p}_video.mp4")

    engine.run_scene("test", ["alpha", "beta"],
                     engine.RunOptions(num_lines=1, intensity="low",
                                       video=True, render_mode="veo"))

    assert seen["beats"] == beats["beats"]
    data = _scene_json(tmp_path)
    assert data["video_track"].endswith("_video.mp4")
    assert "audio_track" not in data  # veo never touched the audio stage


def test_run_scene_veo_failure_keeps_script(monkeypatch, tmp_path, capsys):
    from exec_producer import video as video_mod
    _scene_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(engine, "generate_production_prompts",
                        lambda *a, **k: {"beats": [{"veo3_video_prompt": "p"}]})

    def boom(b, p):
        raise RuntimeError("veo exploded")

    monkeypatch.setattr(video_mod, "render_veo_video", boom)

    engine.run_scene("test", ["alpha", "beta"],
                     engine.RunOptions(num_lines=1, intensity="low",
                                       video=True, render_mode="veo"))

    assert "Veo rendering failed" in capsys.readouterr().out
    assert len(list((tmp_path / "scripts").glob("test_show_test_*.txt"))) == 1
    assert "video_track" not in _scene_json(tmp_path)


def test_run_scene_audio_failure_keeps_script(monkeypatch, tmp_path, capsys):
    # A dead audio stage must never lose the script that was already written.
    from exec_producer import audio as audio_mod
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "SCENES", {"test": SCENE})
    monkeypatch.setattr(engine, "get_director_choice",
                        lambda *a, **k: {"next_speaker": "alpha", "direction": "OPEN"})
    monkeypatch.setattr(engine, "inject_scene_update", lambda *a, **k: None)
    monkeypatch.setattr(engine, "generate_line_session", lambda *a, **k: "hi")

    def boom(conversation, prefix):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(audio_mod, "render_scene_audio", boom)

    engine.run_scene("test", ["alpha", "beta"],
                     engine.RunOptions(num_lines=1, intensity="low", audio=True))

    assert "Audio rendering failed" in capsys.readouterr().out
    assert len(list((tmp_path / "scripts").glob("test_show_test_*.txt"))) == 1
    jsons = [p for p in (tmp_path / "scripts").glob("test_show_test_*.json")
             if not p.name.endswith("_eval.json")]
    assert "audio_track" not in json.loads(jsons[0].read_text())


# ── episode orchestration ────────────────────────────────

EPISODE = {"test_ep": {"name": "Test Ep", "description": "two-scene test",
                       "intensity": "high",
                       "scenes": [{"scene": "s1", "lines": 2, "plot": "p1"},
                                  {"scene": "s2", "lines": 2, "plot": "p2"}]}}
TWO_SCENES = {"s1": {"name": "Scene One", "prompt": "p", "mood": "m"},
              "s2": {"name": "Scene Two", "prompt": "p", "mood": "m"}}


def episode_harness(monkeypatch, tmp_path):
    """Common mocks: deterministic director, numbered lines, tiny scene cast
    for scene 2 so forced-cast merging is exercised."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "EPISODES", EPISODE)
    monkeypatch.setattr(engine, "SCENES", TWO_SCENES)

    picks = {"n": 0}
    director_cast_seen = []

    def fake_choice(scene, conversation, cast_ids, last_speaker, intensity="high", plot=""):
        director_cast_seen.append(list(cast_ids))
        picks["n"] += 1
        return {"next_speaker": "alpha" if picks["n"] % 2 else "beta", "direction": "OPEN"}

    lines = {"n": 0}

    def fake_line(session):
        lines["n"] += 1
        return f"line {lines['n']}"

    monkeypatch.setattr(engine, "get_director_choice", fake_choice)
    monkeypatch.setattr(engine, "generate_line_session", fake_line)
    monkeypatch.setattr(engine, "get_scene_cast", lambda *a, **k: ["alpha", "beta"])
    return director_cast_seen


def test_run_episode_end_to_end(monkeypatch, tmp_path):
    director_cast_seen = episode_harness(monkeypatch, tmp_path)
    card = {"format_adherence": {"score": 8}, "voice_consistency": {"score": 7},
            "pacing": {"score": 6}, "overall": {"score": 7}}
    judge_calls = []
    monkeypatch.setattr(engine, "evaluate_scene",
                        lambda *a, **k: judge_calls.append(1) or card)
    produced = []
    monkeypatch.setattr(engine, "save_production_files",
                        lambda prefix, *a, **k: produced.append(prefix) or True)
    tails = []
    monkeypatch.setattr(engine, "inject_scene_tail",
                        lambda session, conversation: tails.append(session.cast_id))

    engine.run_episode("test_ep", CAST_IDS,
                       engine.RunOptions(produce=True, evaluate=True))

    # Forced-cast merge: gamma never spoke in scene 1, so scene 2's cast
    # must include gamma even though the director cast only alpha+beta.
    assert set(director_cast_seen[-1]) == {"alpha", "beta", "gamma"}
    # Exactly one judge call per scene — inner scene runs must not also
    # judge (that would double-bill the judge model)
    assert len(judge_calls) == 2
    # Tail injection ran for every participant of both scenes
    assert tails.count("alpha") == 2 and tails.count("gamma") == 2
    # Production files: one per scene, episode-scoped prefixes
    assert len(produced) == 2 and "scene1" in produced[0] and "scene2" in produced[1]

    ep_jsons = [p for p in (tmp_path / "scripts").glob("test_show_episode_test_ep_*.json")
                if not p.name.endswith("_eval.json")]
    ep_evals = list((tmp_path / "scripts").glob("test_show_episode_test_ep_*_eval.json"))
    assert len(ep_jsons) == 1 and len(ep_evals) == 1
    data = json.loads(ep_jsons[0].read_text())
    assert data["intensity"] == "high"
    assert data["audit"]["config_files"]
    assert len(data["scenes"]) == 2
    assert data["scenes"][1]["cast"] == ["alpha", "beta", "gamma"]
    assert len(data["scenes"][0]["director_notes"]) == 2
    eval_data = json.loads(ep_evals[0].read_text())
    assert len(eval_data["scenes"]) == 2 and eval_data["scenes"][0]["scorecard"] == card


def test_run_episode_audio_one_track_per_scene(monkeypatch, tmp_path):
    # Scenes inside an episode must not render individually — one render per
    # scene happens at episode wrap, with episode-scoped prefixes.
    from exec_producer import audio as audio_mod
    episode_harness(monkeypatch, tmp_path)
    rendered = []
    monkeypatch.setattr(
        audio_mod, "render_scene_audio",
        lambda conversation, prefix: rendered.append(prefix) or f"{prefix}_audio.mp3")

    engine.run_episode("test_ep", CAST_IDS, engine.RunOptions(audio=True))

    assert len(rendered) == 2
    assert "scene1" in rendered[0] and "scene2" in rendered[1]
    ep_jsons = [p for p in (tmp_path / "scripts").glob("test_show_episode_test_ep_*.json")
                if not p.name.endswith("_eval.json")]
    data = json.loads(ep_jsons[0].read_text())
    assert data["audio_tracks"] == [p + "_audio.mp3" for p in rendered]


def test_run_episode_video_one_per_scene(monkeypatch, tmp_path):
    from exec_producer import audio as audio_mod, video as video_mod
    episode_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(audio_mod, "render_scene_audio", lambda c, p: f"{p}_audio.mp3")
    rendered = []
    monkeypatch.setattr(video_mod, "render_scene_video",
                        lambda c, p: rendered.append(p) or f"{p}_video.mp4")

    engine.run_episode("test_ep", CAST_IDS, engine.RunOptions(audio=True, video=True))

    assert len(rendered) == 2
    assert "scene1" in rendered[0] and "scene2" in rendered[1]
    ep_jsons = [p for p in (tmp_path / "scripts").glob("test_show_episode_test_ep_*.json")
                if not p.name.endswith("_eval.json")]
    data = json.loads(ep_jsons[0].read_text())
    assert data["video_tracks"] == [p + "_video.mp4" for p in rendered]
    assert len(data["audio_tracks"]) == 2


def test_run_episode_veo_mode_without_audio(monkeypatch, tmp_path):
    from exec_producer import video as video_mod
    episode_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(engine, "generate_production_prompts",
                        lambda *a, **k: {"beats": [{"veo3_video_prompt": "p"}]})
    rendered = []
    monkeypatch.setattr(video_mod, "render_veo_video",
                        lambda b, p: rendered.append(p) or f"{p}_video.mp4")

    engine.run_episode("test_ep", CAST_IDS,
                       engine.RunOptions(video=True, render_mode="veo"))

    assert len(rendered) == 2
    ep_jsons = [p for p in (tmp_path / "scripts").glob("test_show_episode_test_ep_*.json")
                if not p.name.endswith("_eval.json")]
    data = json.loads(ep_jsons[0].read_text())
    assert data["video_tracks"] == [p + "_video.mp4" for p in rendered]
    assert "audio_tracks" not in data


def test_run_episode_no_save_writes_nothing(monkeypatch, tmp_path):
    episode_harness(monkeypatch, tmp_path)
    engine.run_episode("test_ep", CAST_IDS, engine.RunOptions(save=False))
    assert not list((tmp_path / "scripts").glob("*")) if (tmp_path / "scripts").exists() else True


def test_run_episode_script_survives_interrupt_during_media(monkeypatch, tmp_path):
    # The dialogue is the money: the episode .txt must be on disk BEFORE the
    # long media stages run, so a Ctrl-C mid-TTS doesn't lose the script.
    from exec_producer import audio as audio_mod
    episode_harness(monkeypatch, tmp_path)

    def interrupted(conversation, prefix):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio_mod, "render_scene_audio", interrupted)
    with pytest.raises(KeyboardInterrupt):
        engine.run_episode("test_ep", CAST_IDS, engine.RunOptions(audio=True))
    txts = list((tmp_path / "scripts").glob("test_show_episode_test_ep_*.txt"))
    assert len(txts) == 1 and "line 1" in txts[0].read_text()


def test_veo_with_produce_generates_beats_once(monkeypatch, tmp_path):
    from exec_producer import video as video_mod
    _scene_harness(monkeypatch, tmp_path)
    generations = []
    beats = {"beats": [{"beat_number": 1, "veo3_video_prompt": "p",
                        "gemini_image_prompt": "g", "dialogue_lines": ["A: hi"],
                        "duration_seconds": 5}]}
    monkeypatch.setattr(engine, "generate_production_prompts",
                        lambda *a, **k: generations.append(1) or beats)
    monkeypatch.setattr(video_mod, "render_veo_video", lambda b, p: f"{p}_video.mp4")

    engine.run_scene("test", ["alpha", "beta"],
                     engine.RunOptions(num_lines=1, intensity="low", video=True,
                                       render_mode="veo", produce=True))

    # one paid generation, shared by the render and the production files —
    # and the saved beats match what was actually rendered
    assert len(generations) == 1
    prods = list((tmp_path / "scripts").glob("*_production.json"))
    assert len(prods) == 1
    assert json.loads(prods[0].read_text())["beats"] == beats["beats"]


def test_run_episode_director_only_redacts_console(monkeypatch, tmp_path, capsys):
    # Regression: --director-only was silently ignored on episode runs.
    episode_harness(monkeypatch, tmp_path)
    engine.run_episode("test_ep", CAST_IDS, engine.RunOptions(director_only=True))
    out = capsys.readouterr().out
    assert "line 1" not in out            # no dialogue on the console
    assert "█" in out                     # redaction bars shown instead
    assert "saved to" not in out          # file paths suppressed too
    # files are still written, with full dialogue (files are allowed)
    txts = list((tmp_path / "scripts").glob("test_show_episode_test_ep_*.txt"))
    assert len(txts) == 1 and "line 1" in txts[0].read_text()


def test_run_episode_cap_wiring(monkeypatch, tmp_path, capsys):
    episode_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(engine, "MAX_INTENSITY", "low")
    engine.run_episode("test_ep", CAST_IDS)
    assert "Content rating cap" in capsys.readouterr().out
    ep_jsons = [p for p in (tmp_path / "scripts").glob("test_show_episode_test_ep_*.json")
                if not p.name.endswith("_eval.json")]
    assert json.loads(ep_jsons[0].read_text())["intensity"] == "low"
