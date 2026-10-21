"""Unit tests for the shared cross-layer contracts."""

from __future__ import annotations

from collections.abc import Collection

import pytest

from opencode_mlflow.contracts import CanonicalRecordEnvelope, RedactionMode, Redactor

pytestmark = pytest.mark.unit


class _RecordingRedactor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, tuple[str, ...]]] = []

    def redact(self, value: object, protected_paths: Collection[str]) -> object:
        self.calls.append((value, tuple(protected_paths)))
        return value


def test_redaction_mode_exposes_exactly_the_two_policies() -> None:
    assert [mode.name for mode in RedactionMode] == ["VERBATIM", "REDACTED"]


def test_canonical_record_envelope_holds_canonical_and_raw_provenance() -> None:
    envelope = CanonicalRecordEnvelope(
        {"role": "user"},
        '{"role": "user"}',
        {"id": 1},
        ("duplicate-json-key",),
    )

    assert envelope.decoded == {"role": "user"}
    assert envelope.raw_text == '{"role": "user"}'
    assert dict(envelope.raw_columns) == {"id": 1}
    assert envelope.warnings == ("duplicate-json-key",)


def test_redactor_is_a_structural_protocol() -> None:
    redactor = _RecordingRedactor()

    assert isinstance(redactor, Redactor)

    result = redactor.redact({"part_id": "p1"}, protected_paths=("part_id",))

    assert result == {"part_id": "p1"}
    assert redactor.calls == [({"part_id": "p1"}, ("part_id",))]
