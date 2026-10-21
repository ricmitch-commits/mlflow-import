"""Unit tests for the canonical source model."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_args

import pytest

from opencode_mlflow.content.canonical import DuplicateKeyObject, canonical_json
from opencode_mlflow.source.model import (
    AdapterGeneration,
    SchemaDescription,
    SchemaGeneration,
    SessionHeader,
    SourceAdapter,
    SourceRecord,
    TimestampUnit,
    TimestampValue,
    TurnSummary,
    normalize_timestamp,
    source_database_fingerprint,
)
from opencode_mlflow.source.model_identity import extract_model_identity
from opencode_mlflow.source.normalize import normalize_epoch_ms, unknown_json_field_warning
from opencode_mlflow.source.selection import SelectionEvidence

pytestmark = pytest.mark.unit

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def test_source_record_retains_raw_value_when_json_is_malformed():
    record = SourceRecord.from_row(
        source_table="part",
        source_type="text",
        source_id="part-1",
        raw_json="{not-json",
    )

    assert record.decoded_value is None
    assert record.raw_json == "{not-json"
    assert record.warnings == ("malformed_json",)


class TestTimestampNormalization:
    def test_schema_generation_is_limited_to_legacy_current_mixed(self) -> None:
        assert get_args(SchemaGeneration) == ("legacy", "current", "mixed")

    def test_adapter_generation_excludes_mixed(self) -> None:
        assert get_args(AdapterGeneration) == ("legacy", "current")

    def test_milliseconds_retain_original_value_and_unit(self) -> None:
        timestamp = normalize_timestamp(1_772_000_010_000, "ms")

        assert timestamp is not None
        assert timestamp.value == 1_772_000_010_000
        assert timestamp.unit == "ms"
        assert timestamp.utc == _EPOCH + timedelta(milliseconds=1_772_000_010_000)
        assert timestamp.nanoseconds == 1_772_000_010_000 * 1_000_000

    def test_seconds_convert_to_utc(self) -> None:
        timestamp = normalize_timestamp(1_772_000_010, "s")

        assert timestamp is not None
        assert timestamp.value == 1_772_000_010
        assert timestamp.unit == "s"
        assert timestamp.utc == _EPOCH + timedelta(seconds=1_772_000_010)
        assert timestamp.nanoseconds == 1_772_000_010 * 1_000_000_000

    def test_microseconds_convert_to_utc(self) -> None:
        timestamp = normalize_timestamp(1_772_000_010_000_123, "us")

        assert timestamp is not None
        assert timestamp.utc == _EPOCH + timedelta(microseconds=1_772_000_010_000_123)
        assert timestamp.nanoseconds == 1_772_000_010_000_123 * 1_000

    def test_nanoseconds_convert_to_utc(self) -> None:
        timestamp = normalize_timestamp(1_772_000_010_000_000_012, "ns")

        assert timestamp is not None
        assert timestamp.nanoseconds == 1_772_000_010_000_000_012
        # datetime resolution is microseconds; the remainder is truncated.
        assert timestamp.utc == _EPOCH + timedelta(microseconds=1_772_000_010_000_000)

    def test_float_seconds_convert_without_losing_the_instant(self) -> None:
        timestamp = normalize_timestamp(1_772_000_010.5, "s")

        assert timestamp is not None
        assert timestamp.utc == _EPOCH + timedelta(seconds=1_772_000_010, microseconds=500_000)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_epoch_milliseconds_are_malformed_and_absent(self, value: float) -> None:
        timestamp, warnings = normalize_epoch_ms(value)

        assert timestamp is None
        assert warnings == ("malformed_source_value",)

    def test_missing_values_remain_absent(self) -> None:
        assert normalize_timestamp(None, "ms") is None

    @pytest.mark.parametrize("unit", ["seconds", "fortnights", "", "MS", None])
    def test_unknown_units_are_rejected(self, unit: object) -> None:
        with pytest.raises(ValueError):
            normalize_timestamp(1_000, unit)  # type: ignore[arg-type]

    def test_known_unit_values(self) -> None:
        assert get_args(TimestampUnit) == ("s", "ms", "us", "ns")

    def test_timestamp_value_is_directly_constructible(self) -> None:
        timestamp = TimestampValue(value=5, unit="ms")

        assert timestamp.value == 5
        assert timestamp.unit == "ms"
        assert timestamp.utc == _EPOCH + timedelta(milliseconds=5)


class TestSourceRecord:
    def test_nested_duplicate_wrappers_do_not_repeat_the_full_raw_document(self) -> None:
        depth = 32
        raw = '{"leaf":"value"}'
        for _index in range(depth):
            raw = '{"marker":0,"marker":1,"child":' + raw + "}"

        record = SourceRecord.from_row("message", "user", "msg-nested-duplicates", raw)

        assert record.raw_json == raw
        value = record.decoded_value
        for index in range(depth):
            assert isinstance(value, DuplicateKeyObject)
            assert value.raw_text == (raw if index == 0 else "")
            value = value["child"]
        assert value == {"leaf": "value"}
        assert len(canonical_json(record.decoded_value)) < len(raw.encode("utf-8")) * 8

    def test_duplicate_json_keys_are_preserved_recursively_with_raw_text(self) -> None:
        raw = '{"outer":{"token":"first","token":"second"},"clean":{"one":1}}'

        record = SourceRecord.from_row(
            source_table="message",
            source_type="user",
            source_id="msg-duplicate",
            raw_json=raw,
        )

        assert isinstance(record.decoded_value, dict)
        assert record.decoded_value["clean"] == {"one": 1}
        duplicate = record.decoded_value["outer"]
        assert isinstance(duplicate, DuplicateKeyObject)
        assert duplicate.pairs == (("token", "first"), ("token", "second"))
        assert duplicate.raw_text == ""
        assert record.raw_json == raw
        assert record.warnings == ("duplicate_json_key",)

    def test_valid_json_decodes_and_retains_raw_text(self) -> None:
        record = SourceRecord.from_row(
            source_table="message",
            source_type="user",
            source_id="msg-1",
            raw_json='{"role": "user"}',
        )

        assert record.decoded_value == {"role": "user"}
        assert record.raw_json == '{"role": "user"}'
        assert record.warnings == ()

    def test_decoded_value_is_distinct_from_raw_and_identity_fields(self) -> None:
        record = SourceRecord.from_row(
            source_table="part",
            source_type="text",
            source_id="part-1",
            raw_json='{"text": "hello"}',
        )

        assert record.decoded_value != record.raw_json
        assert record.source_id != record.decoded_value
        assert record.source_table != record.decoded_value
        assert record.source_type != record.decoded_value

    def test_duplicate_json_keys_warn_while_retaining_raw_text(self) -> None:
        record = SourceRecord.from_row(
            source_table="message",
            source_type="user",
            source_id="msg-2",
            raw_json='{"text": "first", "text": "second"}',
        )

        assert record.warnings == ("duplicate_json_key",)
        assert record.raw_json == '{"text": "first", "text": "second"}'
        assert isinstance(record.decoded_value, DuplicateKeyObject)
        assert record.decoded_value.pairs == (("text", "first"), ("text", "second"))

    def test_missing_raw_json_stays_absent(self) -> None:
        record = SourceRecord.from_row(
            source_table="message",
            source_type="user",
            source_id=None,
            raw_json=None,
        )

        assert record.decoded_value is None
        assert record.raw_json is None
        assert record.warnings == ()
        assert record.source_id is None

    def test_raw_columns_are_retained(self) -> None:
        record = SourceRecord.from_row(
            source_table="session_message",
            source_type="user",
            source_id="msg-3",
            raw_json='{"content": []}',
            seq=7,
            time_created=1_772_000_010_000,
        )

        assert dict(record.raw_columns) == {"seq": 7, "time_created": 1_772_000_010_000}

    def test_identity_and_ordering_fields_default_to_absent(self) -> None:
        record = SourceRecord.from_row(
            source_table="part",
            source_type="text",
            source_id="part-9",
            raw_json=None,
        )

        assert record.session_id is None
        assert record.sequence_domain is None
        assert record.sequence_value is None
        assert record.parent_source_id is None
        assert record.derived_child_ordinal is None
        assert record.source_identity is None
        assert record.join_projection_key is None
        assert record.timestamp is None
        assert record.classification is None

    def test_identity_and_ordering_fields_round_trip_through_construction(self) -> None:
        reconstructed = SourceRecord(
            source_table="session_message",
            source_type="user",
            source_id="msg-4",
            raw_json=None,
            session_id="ses-1",
            sequence_domain="session_message.seq",
            sequence_value=3,
            parent_source_id="msg-3",
            derived_child_ordinal=0,
            source_identity="current:session_message:msg-4",
            join_projection_key="session_v2.id=session_message.session_id:msg-4",
            timestamp=normalize_timestamp(1_772_000_010_000, "ms"),
            classification="user",
        )

        assert reconstructed.session_id == "ses-1"
        assert reconstructed.sequence_domain == "session_message.seq"
        assert reconstructed.sequence_value == 3
        assert reconstructed.parent_source_id == "msg-3"
        assert reconstructed.derived_child_ordinal == 0
        assert reconstructed.source_identity == "current:session_message:msg-4"
        assert reconstructed.join_projection_key == "session_v2.id=session_message.session_id:msg-4"
        assert reconstructed.timestamp is not None
        assert reconstructed.timestamp.unit == "ms"
        assert reconstructed.classification == "user"


@pytest.mark.parametrize(
    "value",
    [
        {"provider_id": "provider", "modelID": "model"},
        {"provider_id": "provider", "model": "model"},
        {"model": {"provider_id": "provider", "modelID": "model"}},
        {"model": {"provider_id": "provider", "model": "model"}},
        {"providerID": "outer", "model": {"provider_id": "provider", "modelID": "model"}},
    ],
)
def test_provider_id_alias_is_extracted_in_top_level_and_nested_model_forms(value):
    assert extract_model_identity(value) == "provider/model"
    assert unknown_json_field_warning(value, "model") is None


class TestSessionHeader:
    def test_missing_values_remain_absent(self) -> None:
        header = SessionHeader(session_id="ses-1", adapter_generation="current")

        assert header.session_id == "ses-1"
        assert header.adapter_generation == "current"
        for absent in (
            "project_id",
            "workspace_id",
            "parent_id",
            "fork_session_id",
            "fork_boundary",
            "slug",
            "title",
            "directory",
            "path",
            "version",
            "agent",
            "model",
            "metadata",
            "cost",
            "time_created",
            "time_updated",
            "time_compacting",
            "time_archived",
            "source_fingerprint",
        ):
            assert getattr(header, absent) is None
        assert dict(header.raw_columns) == {}
        assert header.warnings == ()

    def test_header_retains_typed_and_raw_values(self) -> None:
        header = SessionHeader(
            session_id="ses-2",
            adapter_generation="legacy",
            directory="/repo/project",
            time_created=normalize_timestamp(1_771_000_000_000, "ms"),
            tokens_input=120,
            raw_columns={"id": "ses-2", "custom_flag": "unknown-value"},
            warnings=("unknown_source_field",),
        )

        assert header.time_created is not None
        assert header.time_created.value == 1_771_000_000_000
        assert header.time_created.unit == "ms"
        assert header.tokens_input == 120
        assert dict(header.raw_columns)["custom_flag"] == "unknown-value"
        assert header.warnings == ("unknown_source_field",)


class TestTurnSummary:
    def test_summary_holds_turn_identity_without_content(self) -> None:
        summary = TurnSummary(
            session_id="ses-1",
            turn_index=1,
            turn_key="turn-abc",
            first_user_source_id="msg-1",
            first_user_source_type="user",
            start_time=normalize_timestamp(1_772_000_010_000, "ms"),
            end_time=normalize_timestamp(1_772_000_020_000, "ms"),
            record_count=4,
            user_record_count=1,
            assistant_record_count=2,
            terminal_outcome="no_assistant_result",
            root_result_source_id=None,
        )

        assert summary.turn_index == 1
        assert summary.turn_key == "turn-abc"
        assert summary.first_user_source_id == "msg-1"
        assert summary.terminal_outcome == "no_assistant_result"
        assert summary.root_result_source_id is None
        assert summary.source_fingerprint is None
        assert not any(isinstance(value, str) and "prompt" in value for value in (summary.first_user_source_id,))


class TestSourceDatabaseFingerprint:
    def test_fingerprint_is_a_sha256_hex_digest(self, tmp_path: Path) -> None:
        source = _touch(tmp_path / "fingerprint-source.db", b"synthetic")
        digest = source_database_fingerprint("current", source)

        assert len(digest) == 64
        int(digest, 16)  # hex

    def test_fingerprint_excludes_the_absolute_path(self, tmp_path: Path) -> None:
        source = _touch(tmp_path / "named-source.db", b"synthetic-content")
        alias = tmp_path / "totally-different-name.db"
        alias.hardlink_to(source)

        assert source_database_fingerprint("current", source) == source_database_fingerprint("current", alias)

    def test_fingerprint_never_reads_source_content(self, tmp_path: Path) -> None:
        # A write-only file can be stat'ed but not read; the fingerprint
        # must still compute (it hashes only filesystem statistics).
        source = _touch(tmp_path / "write-only.db", b"secret-bytes")
        os.chmod(source, 0o200)
        try:
            with pytest.raises(PermissionError):
                source.read_bytes()
            digest = source_database_fingerprint("current", source)
        finally:
            os.chmod(source, 0o600)
        assert len(digest) == 64

    def test_fingerprint_depends_on_schema_generation(self, tmp_path: Path) -> None:
        source = _touch(tmp_path / "generation.db", b"synthetic")

        assert source_database_fingerprint("legacy", source) != source_database_fingerprint("current", source)

    def test_fingerprint_depends_on_file_identity_not_content(self, tmp_path: Path) -> None:
        first = _touch(tmp_path / "first.db", b"same-content")
        second = _touch(tmp_path / "second.db", b"same-content")

        assert source_database_fingerprint("current", first) != source_database_fingerprint("current", second)

    def test_fingerprint_is_stable_across_calls(self, tmp_path: Path) -> None:
        source = _touch(tmp_path / "stable.db", b"synthetic")

        assert source_database_fingerprint("mixed", source) == source_database_fingerprint("mixed", source)


class TestSourceAdapterProtocol:
    def test_protocol_is_structural(self) -> None:
        adapter = _FakeAdapter()

        assert isinstance(adapter, SourceAdapter)
        assert adapter.schema_generation == "current"
        assert list(adapter.iter_session_headers()) == []
        assert list(adapter.iter_records("ses-1")) == []
        assert list(adapter.iter_session_scoped_records("ses-1")) == []

    def test_protocol_rejects_missing_surface(self) -> None:
        assert not isinstance(object(), SourceAdapter)


class _FakeAdapter:
    """Minimal structural SourceAdapter for protocol checks."""

    schema_generation: SchemaGeneration = "current"

    def inspect(self) -> SchemaDescription:
        return SchemaDescription(generation="current", catalog_version="synthetic")

    def iter_session_headers(self) -> Iterator[SessionHeader]:
        return iter(())

    def iter_records(self, session_id: str) -> Iterator[SourceRecord]:
        return iter(())

    def iter_session_scoped_records(self, session_id: str) -> Iterator[SourceRecord]:
        return iter(())

    def selection_evidence(self, session_id: str) -> SelectionEvidence:
        return SelectionEvidence()


def _touch(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path
