"""Unit tests for the current-generation live exporter.

The tests drive :class:`~opencode_mlflow.source.current.CurrentAdapter`
through the real snapshot driver (bind → inspect → export) over sanitized
synthetic fixture databases, then assert on the normalized snapshot
through the replay adapter. Sequence-domain, nested-content, and
session-scoped behavior is pinned here; identity edge cases use ad-hoc
scripts layered on the shared current fixture DDL.
"""

from __future__ import annotations

import itertools
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from opencode_mlflow.source.catalog import CURRENT_CATALOG_V1, SCHEMA_CATALOG_VERSION, TIE_RANKS
from opencode_mlflow.source.current import CurrentAdapter
from opencode_mlflow.source.model import SchemaDescription
from opencode_mlflow.source.normalize import canonical_row_digest, fallback_source_identity
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter
from opencode_mlflow.source.snapshot import (
    SnapshotWriter,
    SourceExporter,
    SourceSnapshot,
    create_normalized_snapshot,
)
from opencode_mlflow.source.turns import ASSISTANT_ERROR_OUTCOME, iter_turn_summaries
from tests.support.fake_source import identity_redaction
from tests.support.sqlite_builder import fixture_script

pytestmark = pytest.mark.unit

SECRET_SENTINELS = (
    "credential-sentinel",
    "ACCOUNT-TOKEN-SENTINEL",
    "ACCOUNT-STATE-SENTINEL",
    "CONTROL-ACCOUNT-SENTINEL",
    "SHARE_URL_FIXTURE_SENTINEL_CURRENT",
    "SHARE-SENTINEL",
)


def _current_exporter(connection: sqlite3.Connection) -> CurrentAdapter:
    return CurrentAdapter(connection, CURRENT_CATALOG_V1)


@contextmanager
def _exported(
    tmp_path: Path,
    script: str,
    make_exporter: Callable[[sqlite3.Connection], SourceExporter] = _current_exporter,
) -> Iterator[tuple[SourceSnapshot, NormalizedSnapshotAdapter]]:
    """Export a script-built current database and yield (snapshot, adapter)."""
    source = tmp_path / f"source-{next(_SOURCE_COUNTER)}.db"
    connection = sqlite3.connect(source)
    try:
        connection.executescript(script)
        exporter = make_exporter(connection)
    finally:
        connection.close()
    with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
        yield snapshot, NormalizedSnapshotAdapter(snapshot)


_SOURCE_COUNTER = itertools.count()

_SNAPSHOT_DDL = """
CREATE TABLE snapshot_sessions (
    session_id TEXT NOT NULL,
    schema TEXT NOT NULL,
    canonical_session_row TEXT NOT NULL,
    directory_filter_match INTEGER NOT NULL,
    model_filter_match INTEGER NOT NULL
);
CREATE TABLE snapshot_records (
    session_id TEXT NOT NULL,
    order_key TEXT NOT NULL,
    source_table TEXT NOT NULL,
    source_id TEXT,
    canonical_record TEXT NOT NULL,
    raw_permitted_fields TEXT NOT NULL
);
"""

_CHILD_TABLES = (
    "session_v2",
    "session_message",
    "session_pending",
    "session_inbox",
    "instruction_entry",
    "instruction_state",
    "instruction_blob",
)


def _child_select_counts(script: str) -> dict[str, int]:
    """Count executed SELECT statements per source table during an export."""
    connection = sqlite3.connect(":memory:")
    writer_connection = sqlite3.connect(":memory:")
    statements: list[str] = []
    try:
        connection.executescript(script)
        connection.set_trace_callback(statements.append)
        writer_connection.executescript(_SNAPSHOT_DDL)
        adapter = CurrentAdapter(connection, CURRENT_CATALOG_V1)
        writer = SnapshotWriter(writer_connection, "current", identity_redaction)
        adapter.export_to_snapshot(writer, identity_redaction)
    finally:
        connection.close()
        writer_connection.close()
    counts = {table: 0 for table in _CHILD_TABLES}
    for statement in statements:
        text = statement.strip()
        # sqlite's trace callback emits parameterized pragma calls as
        # separate `-- PRAGMA ...` comment entries; they are not SQL.
        if not text or text.startswith("--"):
            continue
        # Identifier quoting ("session_v2") is normalized away so the
        # per-table FROM clauses can be matched.
        padded = f"{text.upper().replace(chr(34), '')} "
        if not padded.startswith("SELECT"):
            continue
        for table in _CHILD_TABLES:
            if f"FROM {table.upper()} " in padded:
                counts[table] += 1
    return counts


def _fixture_script(extra: str = "") -> str:
    return fixture_script("current") + extra


class TestCurrentInspect:
    def test_inspect_reports_generation_and_catalog_version(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(_fixture_script())
            adapter = CurrentAdapter(connection, CURRENT_CATALOG_V1)
            description = adapter.inspect()
        finally:
            connection.close()

        assert isinstance(description, SchemaDescription)
        assert description.generation == "current"
        assert description.catalog_version == SCHEMA_CATALOG_VERSION
        assert "session_v2" in description.tables
        assert "session_message" in description.tables

    def test_inspect_exposes_rank_table_timestamp_units_and_classification(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(_fixture_script())
            adapter = CurrentAdapter(connection, CURRENT_CATALOG_V1)
            description = adapter.inspect()
        finally:
            connection.close()

        assert description.tie_ranks == dict(TIE_RANKS)
        assert description.timestamp_units["session_v2.time_created"] == "ms"
        assert description.timestamp_units["session_v2.fork_boundary"] == "ms"
        assert description.timestamp_units["session_message.time_created"] == "ms"
        assert description.timestamp_units["session_v2.time_suspended"] == "ms"
        assert any("session_message.type=user" in rule for rule in description.classifications)
        assert any("synthetic" in rule for rule in description.classifications)

    def test_inspect_describes_safe_session_scoped_columns(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(_fixture_script())
            adapter = CurrentAdapter(connection, CURRENT_CATALOG_V1)
            description = adapter.inspect()
        finally:
            connection.close()

        for table in ("session_pending", "session_inbox", "instruction_entry", "instruction_blob", "instruction_state"):
            assert table in description.tables
        assert set(description.tables["instruction_state"]) >= {"session_id", "epoch_start", "through_seq"}
        assert description.warnings == ("session_share_omitted",)

    def test_inspect_rejects_a_foreign_catalog(self) -> None:
        from dataclasses import replace

        connection = sqlite3.connect(":memory:")
        try:
            with pytest.raises(ValueError):
                CurrentAdapter(connection, replace(CURRENT_CATALOG_V1, generation="legacy"))
        finally:
            connection.close()


class TestCurrentSessionExport:
    def test_share_url_is_never_projected_or_captured(self, tmp_path: Path) -> None:
        sentinel = "SHARE_URL_SENTINEL_CURRENT"
        script = _fixture_script(f"UPDATE session_v2 SET share_url = '{sentinel}';")

        with _exported(tmp_path, script) as (snapshot, adapter):
            headers = list(adapter.iter_session_headers())
            records = [record for header in headers for record in adapter.iter_records(header.session_id)]
            session_rows = snapshot.connection.execute("SELECT canonical_session_row FROM snapshot_sessions").fetchall()
            record_rows = snapshot.connection.execute(
                "SELECT canonical_record, raw_permitted_fields FROM snapshot_records"
            ).fetchall()

        assert "session_share_omitted" in snapshot.schema_description.warnings
        assert all("share_url" not in header.raw_columns for header in headers)
        assert sentinel not in repr(headers)
        assert sentinel not in repr(records)
        assert sentinel not in repr(session_rows)
        assert sentinel not in repr(record_rows)

    def test_mixed_case_security_columns_are_omitted_before_raw_capture(self, tmp_path: Path) -> None:
        script = _fixture_script(
            "ALTER TABLE session_v2 ADD COLUMN ACCESS_TOKEN TEXT;"
            "ALTER TABLE session_v2 ADD COLUMN Share_Secret TEXT;"
            "ALTER TABLE session_v2 ADD COLUMN SESSION_TOKEN TEXT;"
            "UPDATE session_v2 SET future_note = NULL, Share_URL = 'MIXED-SHARE-URL',"
            " ACCESS_TOKEN = 'MIXED-ACCESS-TOKEN', Share_Secret = 'MIXED-SHARE-SECRET',"
            " SESSION_TOKEN = 'MIXED-SESSION-TOKEN';"
        ).replace("share_url TEXT", "Share_URL TEXT")

        with _exported(tmp_path, script) as (snapshot, adapter):
            headers = list(adapter.iter_session_headers())
            records = [
                record
                for header in headers
                for record in (
                    *adapter.iter_records(header.session_id),
                    *adapter.iter_session_scoped_records(header.session_id),
                )
            ]
            snapshot_rows = "\n".join(
                str(row)
                for table in ("snapshot_sessions", "snapshot_records")
                for row in snapshot.connection.execute(f"SELECT * FROM {table}")
            )

        excluded = {"share_url", "share_secret", "access_token", "session_token"}
        assert all(key.casefold() not in excluded for header in headers for key in header.raw_columns)
        assert all(key.casefold() not in excluded for record in records for key in record.raw_columns)
        assert all(
            value not in repr(headers + records)
            for value in (
                "MIXED-SHARE-URL",
                "MIXED-ACCESS-TOKEN",
                "MIXED-SHARE-SECRET",
                "MIXED-SESSION-TOKEN",
            )
        )
        assert all(
            value not in snapshot_rows
            for value in (
                "MIXED-SHARE-URL",
                "MIXED-ACCESS-TOKEN",
                "MIXED-SHARE-SECRET",
                "MIXED-SESSION-TOKEN",
            )
        )
        assert all("unknown_source_field" not in header.warnings for header in headers)
        assert snapshot.schema_description.warnings == ("session_share_omitted",)

    def test_blank_and_whitespace_session_ids_are_missing_and_not_exported(self, tmp_path: Path) -> None:
        script = _fixture_script(
            "INSERT INTO session_v2 (id, time_created) VALUES ('', 1772001000000);"
            "INSERT INTO session_v2 (id, time_created) VALUES ('   ', 1772001000001);"
        )

        with _exported(tmp_path, script) as (snapshot, adapter):
            headers = list(adapter.iter_session_headers())

        assert [header.session_id for header in headers] == ["ses-current-001", "ses-current-002"]
        assert snapshot.stats.session_count == 2
        assert snapshot.stats.warning_count >= 2
        assert snapshot.export_warnings.count("session_missing_id") == 2

    def test_sessions_export_in_creation_order(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            headers = list(adapter.iter_session_headers())

        assert [header.session_id for header in headers] == ["ses-current-001", "ses-current-002"]

    def test_headers_normalize_typed_fields_and_raw_columns(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            headers = {header.session_id: header for header in adapter.iter_session_headers()}

        first = headers["ses-current-001"]
        assert first.adapter_generation == "current"
        assert first.directory == "/repo/project-alpha"
        assert first.title == "Synthetic current session one"
        assert first.model == {"provider": "synth-provider", "model": "synth-model"}
        assert first.cost == 0.125
        assert first.tokens_cache_write == 8
        assert first.time_created is not None
        assert first.time_created.value == 1_772_000_000_000
        assert first.time_viewed is not None
        assert first.time_viewed.value == 1_772_000_800_000
        assert first.raw_columns["future_note"] == "synthetic-future-note"

        second = headers["ses-current-002"]
        assert second.fork_session_id == "ses-current-001"
        assert second.fork_boundary is not None
        assert second.fork_boundary.value == 1_772_000_500_000
        assert second.idle_outcome == "idle-timeout"
        assert second.resume_attempts == 2
        assert second.time_idle is not None
        assert second.time_idle.value == 1_772_000_550_000
        assert "unknown_source_field" not in second.warnings

    def test_unknown_column_value_warns_unknown_source_field(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            headers = {header.session_id: header for header in adapter.iter_session_headers()}

        assert "unknown_source_field" in headers["ses-current-001"].warnings


class TestCurrentMessageExport:
    @pytest.mark.parametrize("statuses", [("failed", "pending"), ("pending", "failed")])
    def test_failed_and_pending_decoded_assistant_statuses_keep_failure_precedence(
        self, tmp_path: Path, statuses: tuple[str, str]
    ) -> None:
        extra = (
            "INSERT INTO session_message VALUES "
            "('msg-current-status-1', 'ses-current-001', 'assistant', 5, 1772000050000, 1772000050000, "
            f'\'{{"status": "{statuses[0]}"}}\');'
            "INSERT INTO session_message VALUES "
            "('msg-current-status-2', 'ses-current-001', 'assistant', 6, 1772000060000, 1772000060000, "
            f'\'{{"status": "{statuses[1]}"}}\');'
        )
        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-current-001")))

        assert summaries[-1].terminal_outcome == ASSISTANT_ERROR_OUTCOME

    def test_unknown_json_keys_warn_and_remain_in_raw_and_decoded_values(self, tmp_path: Path) -> None:
        script = _fixture_script(
            'UPDATE session_message SET data = \'{"content": [], "future_json_key": "synthetic"}\' '
            "WHERE id = 'msg-current-001';"
        )

        with _exported(tmp_path, script) as (_snapshot, adapter):
            record = next(
                record for record in adapter.iter_records("ses-current-001") if record.source_id == "msg-current-001"
            )

        assert "unknown_source_field" in record.warnings
        assert record.raw_json is not None and "future_json_key" in record.raw_json
        assert record.decoded_value["future_json_key"] == "synthetic"

    def test_messages_order_by_authoritative_seq(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            records = [
                r
                for r in adapter.iter_records("ses-current-001")
                if r.source_table == "session_message" and "#" not in str(r.source_id)
            ]

        assert [record.source_id for record in records] == [
            "msg-current-001",
            "msg-current-002",
            "msg-current-003",
            "msg-current-004",
        ]
        assert all(record.sequence_domain == "session_message.seq" for record in records)
        assert [record.sequence_value for record in records] == [1, 2, 3, 4]

    def test_message_classification_follows_the_catalog(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            records = {
                r.source_id: r
                for r in adapter.iter_records("ses-current-001")
                if r.source_table == "session_message" and "#" not in str(r.source_id)
            }

        assert records["msg-current-001"].classification == "user"
        assert records["msg-current-001"].source_type == "user"
        assert records["msg-current-002"].classification == "assistant"
        assert records["msg-current-003"].classification == "user"
        # Catalog rules match in order: the system rule matches before the
        # synthetic rule; the synthetic flag only excludes user records.
        assert records["msg-current-004"].classification == "system"
        assert records["msg-current-004"].source_type == "system"

    def test_synthetic_user_content_never_classifies_user(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO session_message VALUES ('msg-synthetic-user', 'ses-current-001', 'user', 5,"
            " 1772000050000, 1772000050000,"
            ' \'{"synthetic": true, "content": [{"type": "text", "text": "synthetic user content"}]}\');'
        )
        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            records = {r.source_id: r for r in adapter.iter_records("ses-current-001")}

        assert records["msg-synthetic-user"].classification == "synthetic"

    def test_nested_content_derives_child_records_after_the_parent(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-current-001"))

        assert [record.source_id for record in records[:2]] == ["msg-current-001", "msg-current-001#content-0"]
        child = records[1]
        assert child.source_table == "session_message"
        assert child.source_type == "text"
        assert child.parent_source_id == "msg-current-001"
        assert child.derived_child_ordinal == 0
        assert child.sequence_domain == "session_message.seq"
        assert child.sequence_value == 1
        # Nested user content is content, never another turn start.
        assert child.classification is None
        assert child.decoded_value == {"type": "text", "text": "synthetic user prompt one"}

    def test_nested_content_uses_derived_identities_for_every_index(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO session_message VALUES ('msg-two-items', 'ses-current-001', 'user', 5,"
            " 1772000050000, 1772000050000,"
            ' \'{"content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}\');'
        )
        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            records = [
                r for r in adapter.iter_records("ses-current-001") if str(r.source_id).startswith("msg-two-items")
            ]

        assert [record.source_id for record in records] == [
            "msg-two-items",
            "msg-two-items#content-0",
            "msg-two-items#content-1",
        ]
        assert [record.derived_child_ordinal for record in records] == [None, 0, 1]

    def test_message_level_model_identity_is_preserved(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            records = {r.source_id: r for r in adapter.iter_records("ses-current-001")}

        decoded = records["msg-current-001"].decoded_value
        assert isinstance(decoded, dict)
        assert decoded["model"] == {"provider": "synth-provider", "model": "synth-model"}

    def test_missing_id_message_gets_fallback_identity_and_derived_children(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO session_message VALUES (NULL, 'ses-current-001', 'user', 5,"
            " 1772000050000, 1772000050000,"
            ' \'{"content": [{"type": "text", "text": "no id prompt"}]}\');'
        )
        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-current-001"))

        no_id = [r for r in records if r.source_id is None and r.source_table == "session_message"]
        assert len(no_id) == 1
        message = no_id[0]
        assert isinstance(message.source_identity, str) and len(message.source_identity) == 64
        children = [r for r in records if r.parent_source_id == message.source_identity]
        assert [child.source_id for child in children] == [f"{message.source_identity}#content-0"]
        assert children[0].derived_child_ordinal == 0

    def test_nullable_id_messages_use_rowid_provenance_for_tied_rows(self, tmp_path: Path) -> None:
        def exported_identities(name: str, first_secret: str, second_secret: str) -> list[str | None]:
            extra = (
                "INSERT INTO session_message VALUES (NULL, 'ses-current-001', 'user', 5,"
                f' 1772000050000, 1772000050000, \'{{"password":"{first_secret}"}}\');'
                "INSERT INTO session_message VALUES (NULL, 'ses-current-001', 'user', 5,"
                f' 1772000050000, 1772000050000, \'{{"password":"{second_secret}"}}\');'
            )
            with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
                return [
                    record.source_identity
                    for record in adapter.iter_records("ses-current-001")
                    if record.source_table == "session_message" and record.source_id is None
                ]

        baseline = exported_identities("tied-baseline", "password-alpha", "password-beta")
        changed = exported_identities("tied-changed", "password-gamma", "password-delta")

        assert len(baseline) == len(changed) == 2
        assert len(set(baseline)) == len(set(changed)) == 2
        assert changed == baseline

    def test_nullable_id_messages_keep_expected_rowid_order_in_snapshot(self, tmp_path: Path) -> None:
        """The direct current-adapter snapshot preserves tied row order."""
        extra = (
            "INSERT INTO session_message VALUES (NULL, 'ses-current-001', 'user', 5,"
            ' 1772000050000, 1772000050000, \'{"content":[{"type":"text","text":"first"}]}\');'
            "INSERT INTO session_message VALUES (NULL, 'ses-current-001', 'assistant', 5,"
            ' 1772000050000, 1772000050000, \'{"content":[{"type":"text","text":"second"}]}\');'
        )

        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            records = [
                record
                for record in adapter.iter_records("ses-current-001")
                if record.source_table == "session_message" and record.source_id is None
            ]

        assert [record.decoded_value for record in records] == [
            {"content": [{"type": "text", "text": "first"}]},
            {"content": [{"type": "text", "text": "second"}]},
        ]
        assert [record.parent_source_id for record in records] == [None, None]

    def test_tied_current_optional_rows_use_rowid_order_and_identity(self, tmp_path: Path) -> None:
        """Optional no-ID rows do not sort by secret-bearing key/value text."""

        def script_for(first_key: str, second_key: str) -> str:
            return _fixture_script() + (
                "INSERT INTO instruction_entry VALUES "
                f"('ses-current-001', '{first_key}', 'value-first', 0, 1772000070000, 1772000070000);"
                "INSERT INTO instruction_entry VALUES "
                f"('ses-current-001', '{second_key}', 'value-second', 0, 1772000070000, 1772000070000);"
            )

        def scoped(script: str) -> list[tuple[str | None, object, object]]:
            with _exported(tmp_path, script) as (_snapshot, adapter):
                records = [
                    record
                    for record in adapter.iter_session_scoped_records("ses-current-001")
                    if record.source_table == "instruction_entry"
                    and str(record.raw_columns.get("key", "")).startswith("scoped-tied-")
                ]
            return [(record.source_identity, record.raw_columns["key"], record.timestamp) for record in records]

        baseline = scoped(script_for("scoped-tied-z-secret", "scoped-tied-a-secret"))
        changed = scoped(script_for("scoped-tied-z-changed", "scoped-tied-a-changed"))

        assert [item[1] for item in baseline] == ["scoped-tied-z-secret", "scoped-tied-a-secret"]
        assert [item[1] for item in changed] == ["scoped-tied-z-changed", "scoped-tied-a-changed"]
        assert [item[0] for item in changed] == [item[0] for item in baseline]

        expected: list[str] = []
        for rowid, (_identity, _key, timestamp) in zip((4, 5), baseline, strict=True):
            digest = canonical_row_digest(
                "instruction_entry",
                {},
                None,
                session_id="ses-current-001",
                parent_key="ses-current-001",
                sequence_or_timestamp=timestamp,
                row_discriminator=rowid,
            )
            expected.append(
                fallback_source_identity(
                    "current",
                    "instruction_entry",
                    "ses-current-001",
                    "ses-current-001",
                    timestamp,
                    digest,
                    1,
                )
            )
        assert [item[0] for item in baseline] == expected

    def test_message_without_rowid_uses_safe_local_fallback(self, tmp_path: Path) -> None:
        script = (
            _fixture_script()
            .replace(
                "CREATE TABLE session_message (\n    id TEXT PRIMARY KEY,",
                "CREATE TABLE session_message (\n    id TEXT NOT NULL PRIMARY KEY,",
            )
            .replace(
                "    data TEXT\n);\n\nCREATE TABLE session_pending",
                "    data TEXT\n) WITHOUT ROWID;\n\nCREATE TABLE session_pending",
            )
        )

        with _exported(tmp_path, script) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-current-001"))

        assert len(records) == 8
        assert all(record.source_id is not None for record in records)

    def test_null_seq_orders_after_sequenced_messages(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO session_message VALUES ('msg-null-seq', 'ses-current-001', 'user', NULL,"
            " 1772000070000, 1772000070000, '{\"content\": []}');"
        )
        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            messages = [
                r
                for r in adapter.iter_records("ses-current-001")
                if r.source_table == "session_message" and "#" not in str(r.source_id)
            ]

        assert messages[-1].source_id == "msg-null-seq"
        assert messages[-1].sequence_value is None
        assert messages[-1].source_identity is None

    def test_invalid_utf8_blob_data_warns_and_retains_bytes(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO session_message VALUES ('msg-blob', 'ses-current-001', 'user', 5,"
            " 1772000080000, 1772000080000, X'fffe80');"
        )
        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            records = {r.source_id: r for r in adapter.iter_records("ses-current-001")}

        blob = records["msg-blob"]
        assert blob.decoded_value is None
        assert blob.raw_json is None
        assert "malformed_source_value" in blob.warnings
        assert blob.raw_columns["data"] == b"\xff\xfe\x80"


class TestCurrentScopedRecords:
    def test_scoped_records_are_separated_from_turn_records(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            scoped = [(r.source_table, r.source_id) for r in adapter.iter_session_scoped_records("ses-current-001")]
            turn_tables = {r.source_table for r in adapter.iter_records("ses-current-001")}

        assert scoped == [
            ("session_pending", "pending-001"),
            ("session_inbox", "inbox-001"),
            ("instruction_entry", None),
            ("instruction_state", None),
            ("instruction_blob", "synth-blob-hash-001"),
        ]
        assert turn_tables == {"session_message"}

    def test_scoped_record_values_are_normalized(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            scoped = list(adapter.iter_session_scoped_records("ses-current-001"))

        by_table = {}
        for record in scoped:
            by_table.setdefault(record.source_table, []).append(record)

        pending = by_table["session_pending"][0]
        assert pending.source_type == "tool"
        assert pending.classification is None
        assert pending.decoded_value == {"name": "synthetic-tool"}
        assert pending.raw_columns["delivery"] == "polled"

        inbox = by_table["session_inbox"][0]
        assert inbox.source_type == "event"
        assert inbox.decoded_value == {"event": "synthetic-event"}

        entry = by_table["instruction_entry"][0]
        assert entry.source_id is None
        assert isinstance(entry.source_identity, str)
        assert entry.raw_columns["key"] == "synthetic-instruction-key"

        state = by_table["instruction_state"][0]
        assert state.source_id is None
        assert state.decoded_value == {
            "initial_values": {"synthetic-instruction-key": "synth-blob-hash-001"},
            "current_values": {"synthetic-instruction-key": "synth-blob-hash-001"},
        }
        assert state.raw_columns["epoch_start"] == 1
        assert state.raw_columns["through_seq"] == 4

        blob = by_table["instruction_blob"][0]
        assert blob.source_id == "synth-blob-hash-001"
        assert blob.raw_columns["value"] == "synthetic instruction blob content"

    def test_instruction_blob_emitted_once_for_shared_reference(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            first = [r.source_id for r in adapter.iter_session_scoped_records("ses-current-001")]
            second = [r.source_id for r in adapter.iter_session_scoped_records("ses-current-002")]

        assert "synth-blob-hash-001" in first
        # Hash identity is shared, but each owning session needs a projection.
        assert "synth-blob-hash-001" in second
        assert second == [None, "synth-blob-hash-001"]

    def test_unassociated_scoped_rows_are_reported_not_assigned(self, tmp_path: Path) -> None:
        source = tmp_path / "source.db"
        connection = sqlite3.connect(source)
        try:
            connection.executescript(_fixture_script())
            exporter = CurrentAdapter(connection, CURRENT_CATALOG_V1)
        finally:
            connection.close()
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            all_records = [
                record
                for header in adapter.iter_session_headers()
                for record in (
                    *adapter.iter_records(header.session_id),
                    *adapter.iter_session_scoped_records(header.session_id),
                )
            ]

        # One unassociated pending row, one unassociated instruction
        # entry, and one unreferenced instruction blob.
        assert exporter.export_warnings.count("unassociated_record") == 3
        assert all(record.source_id != "pending-orphan-001" for record in all_records)
        assert all(record.raw_columns.get("key") != "synthetic-orphan-instruction-key" for record in all_records)
        assert all(
            record.raw_columns.get("value") != "synthetic orphan instruction blob content" for record in all_records
        )

    def test_large_blob_reference_export_releases_each_session_set_and_uses_anti_join(self, tmp_path: Path) -> None:
        statements: list[str] = []

        class TracingCurrentAdapter(CurrentAdapter):
            def bind_connection(self, connection: sqlite3.Connection) -> None:
                super().bind_connection(connection)
                connection.set_trace_callback(statements.append)

        references = 2_000
        extra = "".join(
            f"INSERT INTO instruction_blob VALUES ('large-hash-{index:04d}', 'large blob {index}');"
            f"INSERT INTO instruction_entry VALUES ('ses-current-001', 'large-key-{index:04d}',"
            f"'large-hash-{index:04d}', 0, {1772010000000 + index}, {1772010000000 + index});"
            for index in range(references)
        )

        exporters: list[TracingCurrentAdapter] = []

        def make_exporter(connection: sqlite3.Connection) -> TracingCurrentAdapter:
            exporter = TracingCurrentAdapter(connection, CURRENT_CATALOG_V1)
            exporters.append(exporter)
            return exporter

        with _exported(
            tmp_path,
            _fixture_script(extra),
            make_exporter,
        ) as (_snapshot, adapter):
            first = list(adapter.iter_session_scoped_records("ses-current-001"))
            second = list(adapter.iter_session_scoped_records("ses-current-002"))

        assert sum(record.source_table == "instruction_blob" for record in first) == references + 1
        assert [record.source_id for record in second].count("synth-blob-hash-001") == 1
        assert exporters
        assert not hasattr(exporters[0], "_session_blob_references")
        assert exporters[0]._active_session_blob_references == set()
        assert any("NOT EXISTS" in statement.upper() for statement in statements)


class TestCurrentSafety:
    def test_secret_values_never_reach_the_snapshot(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (snapshot, adapter):
            text = "\n".join(
                str(row)
                for table in ("snapshot_sessions", "snapshot_records")
                for row in snapshot.connection.execute(f"SELECT * FROM {table}")
            )
            headers = list(adapter.iter_session_headers())
            records = [
                record
                for header in headers
                for record in (
                    *adapter.iter_records(header.session_id),
                    *adapter.iter_session_scoped_records(header.session_id),
                )
            ]

        for sentinel in SECRET_SENTINELS:
            assert sentinel not in text
            assert all(sentinel not in repr(header) for header in headers)
            assert all(sentinel not in repr(record) for record in records)

    def test_secret_row_changes_do_not_affect_selection_or_counts(self, tmp_path: Path) -> None:
        extra_secret_rows = (
            "INSERT INTO account VALUES ('acct-extra', 'extra-account', 'SECRET-EXTRA-SENTINEL');"
            "INSERT INTO credential VALUES ('cred-extra', 'extra-credential-sentinel');"
            "INSERT INTO session_share VALUES ('share-extra', 'ses-current-001',"
            " 'https://share.example.invalid/extra', 'SECRET-EXTRA-SHARE-SENTINEL');"
        )
        with _exported(tmp_path, _fixture_script()) as (baseline, _adapter):
            pass
        with _exported(tmp_path, _fixture_script(extra_secret_rows)) as (extended, _adapter):
            pass

        assert extended.stats.session_count == baseline.stats.session_count
        assert extended.stats.record_count == baseline.stats.record_count

    def test_exported_stats_are_complete_and_honest(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (snapshot, _adapter):
            stats = snapshot.stats

        assert stats.complete is True
        assert stats.schema_generation == "current"
        assert stats.session_count == 2
        # 10 turn records (5 messages + 5 nested content items) plus 7
        # session-scoped records (1 pending, 1 inbox, 2 instruction
        # entries, 1 instruction state, 2 session projections of a shared blob).
        assert stats.record_count == 17

    def test_protocol_shape_matches_the_exporter_contract(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(_fixture_script())
            adapter = CurrentAdapter(connection, CURRENT_CATALOG_V1)
        finally:
            connection.close()

        assert isinstance(adapter, SourceExporter)


class TestCurrentOptionalScopedTables:
    def test_optional_blob_without_hash_does_not_issue_invalid_join(self, tmp_path: Path) -> None:
        script = _fixture_script() + "ALTER TABLE instruction_blob RENAME COLUMN hash TO unknown_hash;"
        with _exported(tmp_path, script) as (_snapshot, adapter):
            records = list(adapter.iter_session_scoped_records("ses-current-001"))
        assert not any(record.source_table == "instruction_blob" for record in records)

    def test_export_succeeds_without_session_scoped_tables(self, tmp_path: Path) -> None:
        """Session-scoped tables are optional for a current match."""
        script = _fixture_script()
        script += (
            "DROP TABLE session_pending;"
            "DROP TABLE session_inbox;"
            "DROP TABLE instruction_entry;"
            "DROP TABLE instruction_blob;"
            "DROP TABLE instruction_state;"
        )
        with _exported(tmp_path, script) as (snapshot, adapter):
            headers = list(adapter.iter_session_headers())
            turn_records = [r for r in adapter.iter_records("ses-current-001")]
            scoped_records = [r for r in adapter.iter_session_scoped_records("ses-current-001")]

        assert [header.session_id for header in headers] == ["ses-current-001", "ses-current-002"]
        assert len(turn_records) == 8
        assert scoped_records == []
        assert snapshot.stats.record_count == 10


class TestBoundedChildQueries:
    """Finding 3: child-table access is a constant number of queries.

    ``session_message``, every session-scoped table, and the transitive
    instruction blobs each stream through ONE merge-joined cursor for the
    whole database, so the number of executed SELECT statements per table
    does not grow with the number of messages.
    """

    @staticmethod
    def _scaled_script(message_count: int) -> str:
        rows = [
            (
                f"INSERT INTO session_message VALUES ('msg-scale-{index}', 'ses-current-001', 'user',"
                f" {10 + index}, {1772001000000 + index}, {1772001000000 + index},"
                f' \'{{"content": [{{"type": "text", "text": "scale {index}"}}]}}\');'
            )
            for index in range(message_count)
        ]
        return _fixture_script("".join(rows))

    def test_child_query_count_is_constant_in_message_count(self) -> None:
        counts = [_child_select_counts(self._scaled_script(size)) for size in (0, 5, 25)]

        assert counts[0] == counts[1] == counts[2]
        # One merge-joined pass per table plus the metadata/rowid probes
        # and the unassociated-record aggregates — a small fixed budget.
        assert sum(counts[0].values()) <= 20

    def test_scaled_export_produces_every_record(self, tmp_path: Path) -> None:
        with _exported(tmp_path, self._scaled_script(200)) as (snapshot, adapter):
            records = list(adapter.iter_records("ses-current-001"))

        # Fixture rows for ses-current-001 (8 turn records) plus 200
        # messages each with one nested content item.
        assert snapshot.stats.record_count == 17 + 400
        assert len(records) == 408


class TestWarningStatsThroughPipeline:
    def test_snapshot_stats_include_exporter_warnings(self, tmp_path: Path) -> None:
        """Exporter-level warnings reach ``SourceSnapshot.stats`` (finding 2).

        The current fixture carries three unassociated rows (an orphan
        pending row, an orphan instruction entry, and an unreferenced
        instruction blob).
        """
        with _exported(tmp_path, _fixture_script()) as (with_orphans, _adapter):
            pass
        cleaned = _fixture_script() + (
            "DELETE FROM session_pending WHERE session_id = 'ses-current-unknown';"
            "DELETE FROM instruction_entry WHERE session_id = 'ses-current-unknown';"
            "DELETE FROM instruction_blob WHERE hash = 'synth-blob-hash-orphan';"
        )
        with _exported(tmp_path, cleaned) as (clean, _adapter):
            pass

        assert with_orphans.stats.complete and clean.stats.complete
        assert with_orphans.stats.session_count == clean.stats.session_count
        assert with_orphans.stats.record_count == clean.stats.record_count
        assert with_orphans.stats.warning_count == clean.stats.warning_count + 3
        assert with_orphans.export_warnings.count("unassociated_record") == 3
        assert clean.export_warnings == ()


class TestWithoutRowidFallback:
    """A session table without a rowid keeps the per-session query paths."""

    @staticmethod
    def _records_by_session(adapter: NormalizedSnapshotAdapter) -> dict[str, list[str | None]]:
        records: dict[str, list[str | None]] = {}
        for header in adapter.iter_session_headers():
            records[header.session_id] = [record.source_id for record in adapter.iter_records(header.session_id)] + [
                record.source_id for record in adapter.iter_session_scoped_records(header.session_id)
            ]
        return records

    def test_session_without_rowid_matches_the_joined_export(self, tmp_path: Path) -> None:
        script = _fixture_script().replace(
            "    time_suspended INTEGER,\n    resume_attempts INTEGER,\n    future_note TEXT\n);",
            "    time_suspended INTEGER,\n    resume_attempts INTEGER,\n    future_note TEXT\n) WITHOUT ROWID;",
        )
        with _exported(tmp_path, _fixture_script()) as (_snapshot, baseline):
            expected = self._records_by_session(baseline)
        with _exported(tmp_path, script) as (_snapshot, fallback):
            actual = self._records_by_session(fallback)

        assert actual == expected


class TestOptionalInstructionEntry:
    """The joined blob pass tolerates an absent or value-less entry table.

    Session-scoped tables are individually optional for a current match,
    and their columns are not probe-validated. A database with
    ``instruction_blob`` present but ``instruction_entry`` absent (or
    present without its ``value`` join column) exported cleanly before
    the merge-join rework; the joined blob pass is gated on the entry
    table's presence and join column, and the scoped-table ordering
    references only observed columns.
    """

    def test_export_succeeds_without_instruction_entry(self, tmp_path: Path) -> None:
        script = _fixture_script() + "DROP TABLE instruction_entry;"
        with _exported(tmp_path, script) as (snapshot, adapter):
            headers = list(adapter.iter_session_headers())
            scoped = [record.source_table for record in adapter.iter_session_scoped_records("ses-current-001")]

        assert [header.session_id for header in headers] == ["ses-current-001", "ses-current-002"]
        assert snapshot.stats.complete is True
        # 10 turn records plus the pending/inbox/state scoped rows; no
        # entries exist, so no blob is referenced or emitted.
        assert snapshot.stats.record_count == 13
        assert scoped == ["session_pending", "session_inbox", "instruction_state"]
        # The writer's own warning (the unknown future_note column) plus
        # the orphan pending row and both now-unreferenced blobs.
        assert snapshot.stats.warning_count == 4

    def test_export_succeeds_when_instruction_entry_lacks_value(self, tmp_path: Path) -> None:
        script = _fixture_script() + (
            "DROP TABLE instruction_entry;"
            "CREATE TABLE instruction_entry ("
            "session_id TEXT NOT NULL, key TEXT, removed INTEGER, time_created INTEGER, time_updated INTEGER);"
            "INSERT INTO instruction_entry VALUES ("
            "'ses-current-001', 'synthetic-instruction-key', 0, 1772000005000, 1772000006000);"
        )
        with _exported(tmp_path, script) as (snapshot, adapter):
            scoped = [
                (record.source_table, record.source_id)
                for record in adapter.iter_session_scoped_records("ses-current-001")
            ]

        assert snapshot.stats.complete is True
        # The value-less entry row emits with a fallback identity; without
        # the value column no blob reference can exist, so no blob is
        # emitted.
        assert ("instruction_entry", None) in scoped
        assert all(table != "instruction_blob" for table, _ in scoped)
        # 10 turn records plus the pending/inbox/entry/state scoped rows.
        assert snapshot.stats.record_count == 14
        assert snapshot.stats.warning_count == 4

    def test_export_succeeds_when_instruction_entry_lacks_session_id(self, tmp_path: Path) -> None:
        script = _fixture_script() + (
            "DROP TABLE instruction_entry;"
            "CREATE TABLE instruction_entry (value TEXT);"
            "INSERT INTO instruction_entry VALUES ('synth-blob-hash-001');"
        )

        with _exported(tmp_path, script) as (snapshot, adapter):
            headers = list(adapter.iter_session_headers())
            scoped = [record.source_table for record in adapter.iter_session_scoped_records("ses-current-001")]

        assert [header.session_id for header in headers] == ["ses-current-001", "ses-current-002"]
        assert snapshot.stats.complete is True
        assert "instruction_entry" not in scoped
        assert "instruction_blob" not in scoped

    def test_scoped_numeric_tiebreaks_put_invalid_values_last_deterministically(self, tmp_path: Path) -> None:
        script = (
            _fixture_script()
            + """
        DELETE FROM instruction_entry;
        INSERT INTO instruction_entry VALUES
            ('ses-current-001', 'entry-valid', 'synth-blob-hash-001', 0, 1772000100000, 10),
            ('ses-current-001', 'entry-text-time', 'synth-blob-hash-001', 0, 1772000100000, 'bad-time'),
            ('ses-current-001', 'entry-null-time', 'synth-blob-hash-001', 0, 1772000100000, NULL),
            ('ses-current-001', 'entry-huge-time', 'synth-blob-hash-001', 0, 1772000100000, 1000000000000000),
            ('ses-current-001', 'entry-infinite-time', 'synth-blob-hash-001', 0, 1772000100000, 1e999),
            ('ses-current-001', 'entry-invalid-removed', 'synth-blob-hash-001', 'bad-removed',
             1772000100000, 'bad-time-2');
        """
        )

        with _exported(tmp_path, script) as (_snapshot, adapter):
            entries = [
                record.raw_columns["key"]
                for record in adapter.iter_session_scoped_records("ses-current-001")
                if record.source_table == "instruction_entry"
            ]

        assert entries == [
            "entry-valid",
            "entry-text-time",
            "entry-null-time",
            "entry-huge-time",
            "entry-infinite-time",
            "entry-invalid-removed",
        ]
