"""Unit tests for deterministic redaction and protected-identity handling.

``tests/fixtures/payloads/redaction-positive.json`` and
``redaction-negative.json`` are the reviewable detector contract tables.
Positive cases pin the exact redacted value, per-category detector counts,
and affected-field count for every version-one detector: PEM private-key
blocks, ``Authorization``/``Proxy-Authorization`` bearer/basic headers
(including standalone bearer values), structured auth fields, secret-key
assignments (structured and textual), connection-string passwords, known
token prefixes, and opaque binary values. Negative cases pin near misses
— below-threshold tokens, non-secret identifiers, delimiter-protected
prose, and protected paths — that must redact nothing. The positive
table's ``secret_key_spellings`` lists every key spelling the
secret-key detector must mask, exercised one spelling at a time.

The remaining classes pin boundaries the fixture tables cannot express:
value-detector thresholds pinned exactly at their boundaries, detector
precedence for structured values, protected-path copying, replacement
markers, record-level redaction (valid JSON raw text is re-serialized
from the already-redacted decoded value; malformed text is masked in
place), the mode-aware ``DeterministicRedactor`` implementing the shared
``Redactor`` protocol, and the counts-only report contract.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, cast

import pytest

from opencode_mlflow.content import redaction as redaction_module
from opencode_mlflow.content.canonical import DuplicateKeyObject
from opencode_mlflow.content.redaction import (
    DETECTOR_CATEGORIES,
    REDACTED_BINARY_MARKER,
    REDACTED_MARKER,
    DeterministicRedactor,
    RedactionReport,
    redact_record,
    redact_text,
    redact_value,
)
from opencode_mlflow.contracts import RedactionMode, Redactor
from opencode_mlflow.source.model import SourceRecord
from opencode_mlflow.source.snapshot import PROTECTED_IDENTITY_FIELDS

pytestmark = pytest.mark.unit

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "payloads"


def _load_document(filename: str) -> dict[str, Any]:
    document = json.loads((FIXTURES_DIR / filename).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list) or not document["cases"]:
        raise AssertionError(f"the {filename} case table is missing or empty")
    return document


def _materialize(value: Any) -> Any:
    """Turn a fixture value into the Python value it stands for."""
    if isinstance(value, dict):
        if set(value) == {"$$bytes_b64"}:
            return base64.b64decode(value["$$bytes_b64"])
        if set(value) == {"$$latin1"}:
            return value["$$latin1"].encode("latin-1")
        return {key: _materialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item) for item in value]
    return value


POSITIVE_DOCUMENT = _load_document("redaction-positive.json")
NEGATIVE_DOCUMENT = _load_document("redaction-negative.json")
POSITIVE_CASES: list[dict[str, Any]] = POSITIVE_DOCUMENT["cases"]
NEGATIVE_CASES: list[dict[str, Any]] = NEGATIVE_DOCUMENT["cases"]
SECRET_KEY_SPELLINGS: list[str] = POSITIVE_DOCUMENT["secret_key_spellings"]


@pytest.mark.parametrize("case", POSITIVE_CASES, ids=lambda case: str(case["name"]))
def test_positive_fixture_redacts_exactly(case: dict[str, Any]) -> None:
    protected = tuple(case.get("protected", ()))
    if case["kind"] == "text":
        result = redact_text(str(case["input"]))
        expected: object = case["expected"]
    else:
        result = redact_value(_materialize(case["input"]), protected)
        expected = _materialize(case["expected"])

    assert result.value == expected
    assert result.report.detector_counts == case["expected_counts"]
    assert result.report.affected_field_count == case["expected_affected_fields"]
    assert result.report.affected_record_count == 1


@pytest.mark.parametrize("spelling", SECRET_KEY_SPELLINGS)
def test_every_secret_key_spelling_is_masked(spelling: str) -> None:
    result = redact_value({spelling: "value-12345"})

    assert result.value == {spelling: REDACTED_MARKER}
    assert result.report.detector_counts == {"secret_key_value": 1}
    assert result.report.affected_field_count == 1


@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=lambda case: str(case["name"]))
def test_negative_fixture_redacts_nothing(case: dict[str, Any]) -> None:
    protected = tuple(case.get("protected", ()))
    if case["kind"] == "text":
        result = redact_text(str(case["input"]))
        assert result.value == case["input"]
    else:
        materialized = _materialize(case["input"])
        result = redact_value(materialized, protected)
        assert result.value == materialized

    assert result.report.detector_counts == {}
    assert result.report.affected_field_count == 0
    assert result.report.affected_record_count == 0


def test_redaction_masks_secret_but_preserves_reconciliation_identity():
    result = redact_value(
        {"secret": "sk-1234567890123456", "part_id": "part-7"},
        protected_paths={"part_id"},
    )

    assert result.value == {"secret": "[REDACTED]", "part_id": "part-7"}
    assert result.report.detector_counts["secret_key_value"] == 1
    assert "sk-1234567890123456" not in repr(result.report)


class TestKnownTokenThresholds:
    """The value detectors' minimum-length guards, pinned at the boundary."""

    @pytest.mark.parametrize(
        ("prefix", "fill", "minimum"),
        [
            ("sk-", "a", 16),
            ("ghp_", "a", 20),
            ("github_pat_", "a", 20),
            ("xoxb-", "a", 10),
            ("AKIA", "A", 16),
        ],
        ids=["sk", "ghp", "github-pat", "slack", "aws"],
    )
    def test_at_threshold_redacts_and_one_below_does_not(self, prefix: str, fill: str, minimum: int) -> None:
        at_threshold = redact_text(f"tail {prefix}{fill * minimum} head")
        assert at_threshold.value == f"tail {REDACTED_MARKER} head"
        assert at_threshold.report.detector_counts == {"known_token_prefix": 1}

        below_threshold = redact_text(f"tail {prefix}{fill * (minimum - 1)} head")
        assert below_threshold.value == f"tail {prefix}{fill * (minimum - 1)} head"
        assert below_threshold.report.detector_counts == {}


class TestContractSurface:
    def test_reexports_the_shared_contract_types(self) -> None:
        assert redaction_module.RedactionMode is RedactionMode
        assert redaction_module.Redactor is Redactor

    def test_replacement_markers_are_exact(self) -> None:
        assert REDACTED_MARKER == "[REDACTED]"
        assert REDACTED_BINARY_MARKER == "[REDACTED_BINARY]"

    def test_detector_categories_are_the_version_one_set(self) -> None:
        assert DETECTOR_CATEGORIES == (
            "pem_private_key",
            "auth_header",
            "structured_auth_field",
            "secret_key_value",
            "connection_secret",
            "known_token_prefix",
            "opaque_binary",
        )

    def test_detector_constants_are_compiled_patterns(self) -> None:
        for name in dir(redaction_module):
            if name.endswith("_PATTERN"):
                assert isinstance(getattr(redaction_module, name), re.Pattern), name


class TestReportContract:
    def test_report_fields_are_counts_only(self) -> None:
        assert [item.name for item in dataclass_fields(RedactionReport)] == [
            "detector_counts",
            "affected_field_count",
            "affected_record_count",
        ]

    def test_report_never_carries_matched_text(self) -> None:
        result = redact_text("-----BEGIN PRIVATE KEY-----\nMIIEsecretbody\n-----END PRIVATE KEY-----")

        assert "MIIEsecretbody" not in repr(result.report)
        assert "PRIVATE KEY" not in repr(result.report)

    def test_reports_merge_by_accumulating_counts(self) -> None:
        first = redact_value({"password": "hunter2"})
        second = redact_text("sk-1234567890123456")

        merged = first.report.merge(second.report)

        assert merged.detector_counts == {"secret_key_value": 1, "known_token_prefix": 1}
        assert merged.affected_field_count == 2
        assert merged.affected_record_count == 2


class TestStructuredRedaction:
    def test_identity_like_names_nested_in_transcript_content_do_not_protect_secrets(self) -> None:
        value = {
            "content": {
                "id": {"password": "id-secret"},
                "key": {"token": "key-secret"},
                "hash": {"Authorization": "Bearer abcdefghijklmnop"},
                "type": {"secret": "type-secret"},
                "role": {"credential": "role-secret"},
            }
        }

        result = redact_value(value, PROTECTED_IDENTITY_FIELDS)

        assert result.value == {
            "content": {
                "id": {"password": REDACTED_MARKER},
                "key": {"token": REDACTED_MARKER},
                "hash": {"Authorization": "Bearer [REDACTED]"},
                "type": {"secret": REDACTED_MARKER},
                "role": {"credential": REDACTED_MARKER},
            }
        }
        assert result.report.detector_counts == {
            "secret_key_value": 4,
            "structured_auth_field": 1,
        }

    def test_field_names_are_preserved_while_values_are_masked(self) -> None:
        value = {"outer": {"api_key": "k", "note": "n"}, "role": "user"}

        result = redact_value(value)

        redacted = cast(dict[str, object], result.value)
        assert redacted == {"outer": {"api_key": REDACTED_MARKER, "note": "n"}, "role": "user"}
        assert list(redacted) == list(value)

    def test_key_detector_takes_precedence_over_leaf_value_scan(self) -> None:
        result = redact_value({"token": "sk-1234567890123456"})

        assert result.value == {"token": REDACTED_MARKER}
        assert result.report.detector_counts == {"secret_key_value": 1}

    def test_protected_path_copies_values_through_unchanged(self) -> None:
        result = redact_value({"parts": [{"token": "sk-1234567890123456", "note": "n"}]}, {"parts.token"})

        assert result.value == {"parts": [{"token": "sk-1234567890123456", "note": "n"}]}
        assert result.report.detector_counts == {}

    def test_protected_fields_copy_bytes_through_unchanged(self) -> None:
        result = redact_value({"hash": b"\x00\x01"}, {"hash"})

        assert result.value == {"hash": b"\x00\x01"}
        assert result.report.detector_counts == {}

    def test_unprotected_bytes_become_the_binary_marker(self) -> None:
        result = redact_value({"payload": b"\x00\x01"})

        assert result.value == {"payload": REDACTED_BINARY_MARKER}
        assert result.report.detector_counts == {"opaque_binary": 1}

    def test_secret_keyed_bytes_use_the_binary_marker(self) -> None:
        result = redact_value({"password": b"\x00\x01"})

        assert result.value == {"password": REDACTED_BINARY_MARKER}
        assert result.report.detector_counts == {"secret_key_value": 1}

    def test_sequences_recurse_and_tuples_are_preserved(self) -> None:
        result = redact_value({"parts": [{"password": "x"}], "pair": ("sk-1234567890123456", "n")})

        redacted = cast(dict[str, object], result.value)
        assert redacted == {"parts": [{"password": REDACTED_MARKER}], "pair": (REDACTED_MARKER, "n")}
        assert isinstance(redacted["pair"], tuple)
        assert result.report.detector_counts == {"secret_key_value": 1, "known_token_prefix": 1}


class TestDuplicateKeyObjectRedaction:
    def test_pairs_and_raw_text_are_redacted_preserving_the_wrapper(self) -> None:
        value = DuplicateKeyObject(
            pairs=(("password", "hunter2"), ("note", "clean")),
            raw_text='{"password": "hunter2", "note": "clean"}',
        )

        result = redact_value(value)

        redacted = result.value
        assert isinstance(redacted, DuplicateKeyObject)
        assert redacted.pairs == (("password", REDACTED_MARKER), ("note", "clean"))
        assert redacted.raw_text == '{"password":"[REDACTED]","note":"clean"}'
        assert result.report.detector_counts == {"secret_key_value": 1}
        assert result.report.affected_field_count == 1

    def test_root_exact_protected_pair_value_and_derived_raw_text_are_copied(self) -> None:
        value = DuplicateKeyObject(
            pairs=(("token", "sk-1234567890123456"),),
            raw_text='{"token": "sk-1234567890123456"}',
        )

        result = redact_value(value, {"token"})

        redacted = result.value
        assert isinstance(redacted, DuplicateKeyObject)
        assert redacted.pairs == (("token", "sk-1234567890123456"),)
        assert redacted.raw_text == '{"token":"sk-1234567890123456"}'
        assert result.report.detector_counts == {}
        assert result.report.affected_field_count == 0

    def test_structured_auth_and_nested_pairs_follow_field_rules(self) -> None:
        value = DuplicateKeyObject(
            pairs=(
                ("Authorization", "Bearer eyJhbGciOiJIUzI1NiJ9.x"),
                ("auth", {"password": "hunter2"}),
            ),
            raw_text="{}",
        )

        result = redact_value(value)

        redacted = result.value
        assert isinstance(redacted, DuplicateKeyObject)
        assert redacted.pairs == (
            ("Authorization", "Bearer [REDACTED]"),
            ("auth", {"password": REDACTED_MARKER}),
        )
        assert result.report.detector_counts == {"structured_auth_field": 1, "secret_key_value": 1}


class TestUnknownValueTypesFailClosed:
    def test_unknown_types_are_masked_as_opaque(self) -> None:
        result = redact_value({"context": object()})

        assert result.value == {"context": REDACTED_BINARY_MARKER}
        assert result.report.detector_counts == {"opaque_binary": 1}

        result = redact_value({"flags": {1, 2}})

        assert result.value == {"flags": REDACTED_BINARY_MARKER}
        assert result.report.detector_counts == {"opaque_binary": 1}

    def test_root_unknown_type_is_masked_as_opaque(self) -> None:
        result = redact_value(object())

        assert result.value == REDACTED_BINARY_MARKER
        assert result.report.detector_counts == {"opaque_binary": 1}
        assert result.report.affected_record_count == 1

    def test_known_scalars_still_pass_through(self) -> None:
        result = redact_value({"count": 3, "ratio": 0.5, "ok": True, "missing": None})

        assert result.value == {"count": 3, "ratio": 0.5, "ok": True, "missing": None}
        assert result.report.detector_counts == {}


class TestTextPipeline:
    def test_assignment_masking_precedes_value_detectors(self) -> None:
        result = redact_text("password=sk-1234567890123456")

        assert result.value == f"password={REDACTED_MARKER}"
        assert result.report.detector_counts == {"secret_key_value": 1}

    def test_header_detection_precedes_standalone_bearer(self) -> None:
        result = redact_text("Authorization: Bearer abcdefghijklmnop")

        assert result.value == "Authorization: Bearer [REDACTED]"
        assert result.report.detector_counts == {"auth_header": 1}

    def test_clean_text_passes_through_unchanged(self) -> None:
        result = redact_text("an ordinary sentence with no secrets at all")

        assert result.value == "an ordinary sentence with no secrets at all"
        assert result.report.detector_counts == {}
        assert result.report.affected_record_count == 0

    def test_escaped_secret_key_spelling_is_masked_in_malformed_json_text(self) -> None:
        result = redact_text('{"pass\\u0077ord":"truncated-secret"')

        assert result.value == '{"pass\\u0077ord":"[REDACTED]"'
        assert result.report.detector_counts == {"secret_key_value": 1}
        assert result.report.affected_field_count == 1


class TestRecordRedaction:
    def test_typed_identity_survives_while_root_identity_like_transcript_field_is_redacted(self) -> None:
        record = SourceRecord.from_row(
            "part",
            "part",
            "part-structural-id",
            '{"id":{"password":"nested-secret"},"type":"text"}',
            session_id="ses-structural-id",
        )

        result = redact_record(record)

        redacted = _as_source_record(result.value)
        assert redacted.source_id == "part-structural-id"
        assert redacted.raw_columns["session_id"] == "ses-structural-id"
        assert redacted.decoded_value == {
            "id": {"password": REDACTED_MARKER},
            "type": "text",
        }
        assert "nested-secret" not in cast(str, redacted.raw_json)

    def test_escaped_duplicate_secret_keys_redact_every_pair_and_discard_original_raw_text(self) -> None:
        raw = '{"password":"first-secret-sentinel","\\u0070assword":"escaped-secret-sentinel"}'
        record = SourceRecord.from_row("part", "part", "part-escaped-duplicate", raw)

        result = redact_record(record)

        redacted = _as_source_record(result.value)
        assert isinstance(redacted.decoded_value, DuplicateKeyObject)
        assert redacted.decoded_value.pairs == (
            ("password", REDACTED_MARKER),
            ("password", REDACTED_MARKER),
        )
        assert redacted.raw_json == '{"password":"[REDACTED]","password":"[REDACTED]"}'
        assert "first-secret-sentinel" not in redacted.decoded_value.raw_text
        assert "escaped-secret-sentinel" not in redacted.decoded_value.raw_text
        assert "first-secret-sentinel" not in redacted.raw_json
        assert "escaped-secret-sentinel" not in redacted.raw_json

    def test_valid_json_raw_text_is_reserialized_from_the_redacted_decoded_value(self) -> None:
        record = SourceRecord.from_row(
            "session_message",
            "message",
            "msg-7",
            '{"password": "hunter2", "note": "ok"}',
            session_id="ses-1",
        )

        result = redact_record(record)

        redacted = _as_source_record(result.value)
        assert redacted.decoded_value == {"password": REDACTED_MARKER, "note": "ok"}
        assert redacted.raw_json == '{"note":"ok","password":"[REDACTED]"}'
        assert redacted.raw_columns == {"session_id": "ses-1"}
        assert redacted.source_id == "msg-7"
        assert result.report.detector_counts == {"secret_key_value": 1}
        assert result.report.affected_record_count == 1

    def test_clean_records_keep_their_original_raw_text(self) -> None:
        record = SourceRecord.from_row("session_message", "message", "msg-8", '{"b": 1, "a": 2}')

        result = redact_record(record)

        redacted = _as_source_record(result.value)
        assert redacted.raw_json == '{"b": 1, "a": 2}'
        assert redacted.decoded_value == {"b": 1, "a": 2}
        assert result.report.detector_counts == {}
        assert result.report.affected_record_count == 0

    def test_malformed_raw_text_is_masked_in_place(self) -> None:
        record = SourceRecord.from_row("part", "part", "part-9", '{"password": "hunter2"')
        assert record.warnings == ("malformed_json",)

        result = redact_record(record)

        redacted = _as_source_record(result.value)
        assert redacted.raw_json == '{"password": "[REDACTED]"'
        assert redacted.decoded_value is None
        assert result.report.detector_counts == {"secret_key_value": 1}

    def test_duplicate_key_shadowed_secret_is_redacted_in_raw_text(self) -> None:
        record = SourceRecord.from_row(
            "part",
            "part",
            "part-13",
            '{"note": "leaked sk-1234567890123456 here", "note": "clean"}',
        )
        assert record.warnings == ("duplicate_json_key",)

        result = redact_record(record)

        redacted = _as_source_record(result.value)
        assert redacted.raw_json == '{"note":"leaked [REDACTED] here","note":"clean"}'
        assert isinstance(redacted.decoded_value, DuplicateKeyObject)
        assert redacted.decoded_value.pairs == (("note", "leaked [REDACTED] here"), ("note", "clean"))
        assert result.report.detector_counts == {"known_token_prefix": 1}
        assert result.report.affected_field_count == 1
        assert result.report.affected_record_count == 1

    def test_duplicate_key_visible_secret_redacts_decoded_and_raw_text(self) -> None:
        record = SourceRecord.from_row(
            "part",
            "part",
            "part-14",
            '{"password": "hunter2", "password": "second"}',
        )
        assert record.warnings == ("duplicate_json_key",)

        result = redact_record(record)

        redacted = _as_source_record(result.value)
        assert isinstance(redacted.decoded_value, DuplicateKeyObject)
        assert redacted.decoded_value.pairs == (
            ("password", REDACTED_MARKER),
            ("password", REDACTED_MARKER),
        )
        assert redacted.raw_json == '{"password":"[REDACTED]","password":"[REDACTED]"}'
        # Lossless pairs are authoritative; their derived safe raw text does
        # not double-count the same storage projection.
        assert result.report.detector_counts == {"secret_key_value": 2}
        assert result.report.affected_field_count == 2
        assert result.report.affected_record_count == 1

    def test_raw_columns_are_redacted_with_identity_protection(self) -> None:
        record = SourceRecord.from_row("part", "part", "part-10", None, password="hunter2", session_id="ses-2", seq=3)

        result = redact_record(record)

        redacted = _as_source_record(result.value)
        assert redacted.raw_columns == {"password": REDACTED_MARKER, "session_id": "ses-2", "seq": 3}
        assert redacted.raw_json is None

    def test_additive_raw_columns_named_like_structural_fields_are_content(self) -> None:
        record = SourceRecord.from_row(
            "part",
            "part",
            "part-raw-column-boundary",
            None,
            type={"password": "type-secret"},
            role={"token": "role-secret"},
            key={"api_key": "key-secret"},
            hash={"credential": "hash-secret"},
            position={"secret": "position-secret"},
            session_id="ses-raw-column-boundary",
        )

        result = redact_record(record)

        redacted = _as_source_record(result.value)
        assert redacted.raw_columns["type"] == {"password": REDACTED_MARKER}
        assert redacted.raw_columns["role"] == {"token": REDACTED_MARKER}
        assert redacted.raw_columns["key"] == {"api_key": REDACTED_MARKER}
        assert redacted.raw_columns["hash"] == {"credential": REDACTED_MARKER}
        assert redacted.raw_columns["position"] == {"secret": REDACTED_MARKER}
        assert redacted.raw_columns["session_id"] == "ses-raw-column-boundary"
        assert result.report.detector_counts == {"secret_key_value": 5}

    def test_identity_fields_survive_content_redaction(self) -> None:
        record = SourceRecord.from_row(
            "part",
            "part",
            "part-11",
            json.dumps(
                {
                    "id": "inner-11",
                    "type": "text",
                    "text": "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.x",
                }
            ),
        )

        result = redact_record(record)

        redacted = _as_source_record(result.value)
        assert redacted.decoded_value == {
            "id": "inner-11",
            "type": "text",
            "text": "Authorization: Bearer [REDACTED]",
        }
        assert redacted.source_id == "part-11"
        assert result.report.detector_counts == {"auth_header": 1}

    def test_verbatim_record_redaction_is_a_no_op(self) -> None:
        redactor = DeterministicRedactor(RedactionMode.VERBATIM)
        record = SourceRecord.from_row("part", "part", "part-12", '{"password": "hunter2"}')

        result = redactor.redact_record(record)

        assert result.value is record
        assert result.report.detector_counts == {}
        assert result.report.affected_record_count == 0


class TestDeterministicRedactor:
    def test_satisfies_the_shared_redactor_protocol(self) -> None:
        assert isinstance(DeterministicRedactor(), Redactor)

    def test_default_mode_is_verbatim(self) -> None:
        assert DeterministicRedactor().mode is RedactionMode.VERBATIM

    def test_verbatim_mode_is_a_no_op_passthrough(self) -> None:
        redactor = DeterministicRedactor(RedactionMode.VERBATIM)
        value = {"secret": "sk-1234567890123456", "part_id": "part-7"}

        assert redactor.redact(value, {"part_id"}) is value

    def test_verbatim_results_carry_zero_counts(self) -> None:
        redactor = DeterministicRedactor(RedactionMode.VERBATIM)

        result = redactor.redact_value({"secret": "sk-1234567890123456"})

        assert result.value == {"secret": "sk-1234567890123456"}
        assert result.report.detector_counts == {}
        assert result.report.affected_field_count == 0
        assert result.report.affected_record_count == 0

    def test_redacted_mode_delegates_to_the_active_pipeline(self) -> None:
        redactor = DeterministicRedactor(RedactionMode.REDACTED)
        value = {"secret": "sk-1234567890123456"}

        assert redactor.redact(value) == {"secret": REDACTED_MARKER}
        assert redactor.redact_value(value).value == {"secret": REDACTED_MARKER}
        assert redactor.redact_value(value).report.detector_counts == {"secret_key_value": 1}


def _as_source_record(value: object) -> SourceRecord:
    assert isinstance(value, SourceRecord)
    return value
