"""Malformed containers and prefixed encoded fragments from final review."""

import json

import pytest

from mlflow_import.content.redaction import redact_text

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text,forbidden",
    [
        ('{"user":{"empty":{},"name":"fixture-name"},"broken":', ["fixture-name"]),
        ('{"api_key": ["fixture-first", "fixture-second"], "broken":', ["fixture-first", "fixture-second"]),
        ('{"user":{"nested":[{"name":"fixture-name","literal":"} ] \\""}]},"broken":', ["fixture-name"]),
        ('{"api_key": ["fixture-first", {"nested":"fixture-second"}', ["fixture-first", "fixture-second"]),
    ],
)
def test_malformed_sensitive_container_is_consumed_whole(text, forbidden):
    result = redact_text(text).value
    assert all(value not in result for value in forbidden)
    assert "[REDACTED]" in result


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_prefixed_encoded_json_fragment_is_sanitized(layers):
    value = {"user_id": "fixture-user", "password": "fixture-secret"}
    for _ in range(layers):
        value = json.dumps(value)
    result = redact_text("prefix " + value + " suffix").value
    assert result.startswith("prefix ") and result.endswith(" suffix")
    assert "fixture-user" not in result and "fixture-secret" not in result


def test_prefixed_fragment_over_decoder_envelope_is_masked_whole(monkeypatch):
    from mlflow_import.content import redaction

    monkeypatch.setattr(redaction, "_FRAGMENT_BYTES", 128)
    text = "prefix " + json.dumps(json.dumps({"user_id": "fixture-user" + "x" * 256})) + " suffix"
    result = redact_text(text).value
    assert result.startswith("prefix ") and result.endswith(" suffix")
    assert "fixture-user" not in result and "[REDACTED]" in result


@pytest.mark.parametrize("layers", [1, 2, 3, 5])
@pytest.mark.parametrize("truncated", [0, 1, 2, 5])
def test_mixed_prose_and_encoded_json_masks_secrets_at_every_layer(layers, truncated):
    value = json.dumps({"user_id": "fixture-user", "password": "fixture-secret"})
    for layer in range(layers):
        value = json.dumps(f"message {layer}: {value} trailing prose")
    if truncated:
        value = value[:-truncated]
    text = 'prefix "innocent" ' + value + (" suffix" if not truncated else " ambiguous remainder")
    result = redact_text(text).value
    assert result.startswith('prefix "innocent" ')
    assert "fixture-user" not in result and "fixture-secret" not in result
    assert "[REDACTED]" in result
    if truncated:
        assert "ambiguous remainder" not in result
    else:
        assert result.endswith(" suffix")
    assert redact_text(result).value == result


@pytest.mark.parametrize("layers", [1, 2, 4])
@pytest.mark.parametrize("cut", [1, 2, 6])
def test_truncated_inner_quote_inside_complete_prose_wrappers_is_sanitized(layers, cut):
    value = json.dumps(json.dumps({"user_id": "fixture-user", "password": "fixture-secret"}))[:-cut]
    for layer in range(layers):
        value = json.dumps(f"wrapper {layer}: {value}")
    result = redact_text("prefix " + value + " suffix").value
    assert "fixture-user" not in result and "fixture-secret" not in result
    assert result.startswith("prefix ") and result.endswith(" suffix")


@pytest.mark.parametrize("prose", ["", "message "])
@pytest.mark.parametrize("layers", [1, 3])
def test_unterminated_encoded_outer_quote_masks_json_through_eof(prose, layers):
    value = json.dumps({"user_id": "fixture-user", "password": "fixture-secret"})
    for _ in range(layers):
        value = json.dumps(prose + value)
    result = redact_text("prefix " + value[:-1] + " ambiguous remainder").value
    assert result == 'prefix "[REDACTED]"'


@pytest.mark.parametrize("opening", [r"\u007b", r"\u005B"])
def test_unterminated_quote_with_unicode_encoded_json_opener_masks_to_eof(opening):
    text = 'prefix "message ' + opening + r"\"user_id\":\"fixture-user\",\"password\":\"fixture-secret"
    assert redact_text(text).value == 'prefix "[REDACTED]"'


@pytest.mark.parametrize(
    "value",
    [
        'message "hello" and "password" without assignment',
        'message {"role": "user", "count": 3} trailing prose',
        "message [one, two] and an ordinary path C:\\notes",
        "message café and a newline\nwithout credentials",
    ],
)
@pytest.mark.parametrize("layers", [1, 3])
def test_clean_quoted_prose_retains_exact_original_encoding(value, layers):
    for _ in range(layers):
        value = json.dumps("message " + value, ensure_ascii=True)
    text = "prefix " + value + " suffix"
    assert redact_text(text).value == text


@pytest.mark.parametrize("text", ['prefix "ordinary unfinished prose', r'prefix "ordinary C:\\notes unfinished'])
def test_unterminated_non_json_nonmatch_is_unchanged(text):
    assert redact_text(text).value == text


@pytest.mark.parametrize("separator", ["\n", r"\x"])
def test_truncated_json_quote_cannot_borrow_a_later_fragments_quote(separator):
    encoded = json.dumps({"user_id": "fixture-user", "password": "fixture-secret"})
    broken = json.dumps("message " + encoded)[:-1]
    text = 'prefix "innocent" ' + broken + separator + 'later "quoted prose" and ' + json.dumps(encoded)
    assert redact_text(text).value == 'prefix "innocent" "[REDACTED]"'


def test_fragment_decoding_stays_bounded_with_many_prose_wrapped_fragments(monkeypatch):
    from mlflow_import.content import redaction

    original_decode, original_end = redaction.decode_lossless_json, redaction._quoted_end
    decoded_bytes = scanned_characters = 0

    def decode(value):
        nonlocal decoded_bytes
        size = len(value.encode("utf-8"))
        assert size <= 512
        decoded_bytes += size
        return original_decode(value)

    def quoted_end(text, start):
        nonlocal scanned_characters
        result = original_end(text, start)
        scanned_characters += result[0] - start
        return result

    monkeypatch.setattr(redaction, "_FRAGMENT_BYTES", 512)
    monkeypatch.setattr(redaction, "decode_lossless_json", decode)
    monkeypatch.setattr(redaction, "_quoted_end", quoted_end)
    fragment = json.dumps("message " + json.dumps({"user_id": "fixture-user", "password": "fixture-secret"}))
    oversized = json.dumps("message " + json.dumps({"user_id": "fixture-user", "padding": "x" * 1024}))
    text = "prefix " + " separator ".join([fragment] * 256 + [oversized, oversized[:-1]])
    result = redact_text(text).value
    assert "fixture-user" not in result and "fixture-secret" not in result
    assert decoded_bytes + scanned_characters < len(text) * 32
