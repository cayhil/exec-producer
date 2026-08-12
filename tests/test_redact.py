"""Secret redaction on the log path."""

import io
import logging

from exec_producer import telemetry
from exec_producer.redact import RedactingFormatter, redact


SECRET = "sk-abcdefghijklmnopqrstuvwxyz"


def emit_through_configured_logger(message, *args):
    """Log through the real package logger as setup_logging configured it,
    returning what its handler actually wrote.

    Only the destination stream is swapped — the logger, its handler, and
    the formatter under test are the genuine ones. (capsys can't do this
    job: pytest's logging plugin sits between the record and the stream.)
    """
    telemetry.setup_logging()
    logger = telemetry.get_logger()
    # Every StreamHandler, not handlers[0]: the moment setup_logging grows a
    # second sink (a file handler is the obvious next step), indexing would
    # silently assert against the wrong one.
    handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert handlers, "setup_logging installed no stream handler"
    buffer = io.StringIO()
    originals = [h.stream for h in handlers]
    for handler in handlers:
        handler.stream = buffer
    try:
        logger.warning(message, *args)
    finally:
        for handler, original in zip(handlers, originals):
            handler.stream = original
    return buffer.getvalue()


def test_masks_openai_style_key_keeping_a_short_prefix_and_suffix():
    out = redact("Incorrect API key provided: sk-proj-abcdefghijklmnop1234")
    assert "sk-proj-abcdefghijklmnop1234" not in out
    assert "<redacted" in out


def test_masks_every_provider_prefix_this_engine_uses():
    keys = [
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",   # Anthropic
        "sk-proj-abcdefghijklmnopqrstuvwxyz",        # OpenAI / DeepSeek
        "xai-abcdefghijklmnopqrstuvwxyz1234567890",  # xAI
        "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456",   # Google / Gemini
        "sk_abcdefghijklmnopqrstuvwxyz",             # ElevenLabs
    ]
    for key in keys:
        assert key not in redact(f"auth failed with {key} — retry"), key


def test_short_secrets_are_fully_masked():
    # Nothing of a short token survives: 6+4 of a 12-char secret is the
    # whole secret.
    assert redact("KEY=sk-abcdefghij") == "KEY=<redacted>"


def test_long_secrets_keep_a_debuggable_stub():
    out = redact("sk-abcdefghijklmnopqrstuvwxyz")
    assert out == "<redacted sk-abc…wxyz>"


def test_masks_env_assignments_for_opaque_keys_without_a_known_prefix():
    # Mistral keys carry no vendor prefix — the KEY=value shape is the
    # only thing that identifies them.
    out = redact("MISTRAL_API_KEY=9f8e7d6c5b4a39281706abcdef")
    assert "9f8e7d6c5b4a39281706abcdef" not in out
    assert "MISTRAL_API_KEY=" in out


def test_masks_json_credential_fields():
    out = redact('{"api_key": "9f8e7d6c5b4a3928170", "model": "gpt-5.4"}')
    assert "9f8e7d6c5b4a3928170" not in out
    assert "gpt-5.4" in out


def test_masks_bearer_tokens_in_authorization_headers():
    out = redact("Authorization: Bearer 9f8e7d6c5b4a39281706abcdef")
    assert "9f8e7d6c5b4a39281706abcdef" not in out


def test_leaves_ordinary_text_alone():
    # False positives are worse than useless here — the whole point of the
    # log line is to tell the user what broke.
    text = "call failed after retries: gemini/gemini-2.5-pro: RateLimitError"
    assert redact(text) == text


def test_does_not_mask_model_ids_that_look_keylike():
    assert redact("claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert redact("total_tokens=1234") == "total_tokens=1234"


def test_does_not_eat_ordinary_auth_prose():
    # "Basic authentication failed" is what a 401 body actually says. A
    # redactor that mangles it destroys the diagnosis the line exists for.
    for text in ("Basic authentication failed for this endpoint",
                 "bearer authentication required",
                 "Bearer credentials rejected by the proxy"):
        assert redact(text) == text, text


def test_does_not_match_secret_names_as_substrings_of_ordinary_ones():
    # TOKEN inside MAX_TOKENS / TOKENIZER is telemetry, not a credential.
    for text in ("MAX_TOKENS=4000 exceeded the model limit",
                 "TOKENIZER=cl100k_base",
                 "PROMPT_TOKENS=812 COMPLETION_TOKENS=204"):
        assert redact(text) == text, text


def test_a_masked_value_stops_at_the_end_of_the_value():
    # `(\S+)` used to swallow the rest of a comma-separated line, taking the
    # model and temperature with it.
    out = redact("API_KEY=9f8e7d6c5b4a39281706,model=gpt-5.4,temperature=0")
    assert "9f8e7d6c5b4a39281706" not in out
    assert "model=gpt-5.4,temperature=0" in out


def test_an_empty_env_value_with_a_trailing_comment_is_left_readable():
    # This is exactly the .env shape api_key_set() exists to diagnose —
    # redacting it would hide the explanation of the failure.
    text = "OPENAI_API_KEY=  # paste yours here"
    assert redact(text) == text


def test_masks_lowercase_and_single_quoted_credential_shapes():
    # Mistral keys carry no vendor prefix, so the name-anchored patterns are
    # the only thing that can catch them — and provider errors are not
    # uniformly uppercase or double-quoted.
    secret = "vT9kQ2mNp8xR4wL7zB3cY6dF1gH5jK0s"
    for text in (f"MistralException - api_key={secret}",
                 f"headers={{'x-api-key': '{secret}'}}",
                 f"body={{'api_key': '{secret}'}}",
                 f"Authorization: BEARER {secret}"):
        assert secret not in redact(text), text


def test_formatter_redacts_the_rendered_log_line():
    formatter = RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="exec_producer", level=logging.WARNING, pathname=__file__,
        lineno=1, msg="bad key: %s", args=("sk-abcdefghijklmnopqrstuvwxyz",),
        exc_info=None)
    out = formatter.format(record)
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in out
    assert "bad key:" in out


def test_setup_logging_redacts_the_package_logger():
    out = emit_through_configured_logger("call failed: %s", SECRET)
    assert SECRET not in out
    assert "<redacted" in out


def test_setup_logging_upgrades_a_handler_installed_before_it_ran():
    # setup_logging only *adds* a handler once. If it left an existing
    # handler's formatter alone, redaction could never be turned on for a
    # handler some earlier caller had already attached.
    logger = telemetry.get_logger()
    plain = logging.StreamHandler(io.StringIO())
    plain.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(plain)

    out = emit_through_configured_logger("call failed: %s", SECRET)
    assert SECRET not in out
    assert isinstance(plain.formatter, RedactingFormatter)


def test_the_vlog_failure_notice_does_not_print_a_raw_provider_error(capsys):
    # This handler is the only one in the package that interpolates the
    # exception MESSAGE rather than just its type, and it wraps the stage
    # that talks to two paid providers. It's a print(), so the logger's
    # formatter never sees it.
    from exec_producer import engine

    engine.print_vlog_failure(RuntimeError(f"ElevenLabs 401 for key {SECRET}"))
    out = capsys.readouterr().out
    assert SECRET not in out
    assert "RuntimeError" in out  # the diagnosis survives


def test_the_provider_error_line_from_a_failed_call_is_redacted():
    # The concrete leak this guards: llm.call_model logs the first line of a
    # failed call's exception, and provider auth errors echo the key back.
    out = emit_through_configured_logger(
        f"call failed after retries: gpt-5.4: AuthenticationError — "
        f"Incorrect API key provided: {SECRET}")
    assert SECRET not in out
    assert "AuthenticationError" in out  # the diagnosis itself survives
