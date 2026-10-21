"""Unit tests for the legacy-generation live exporter.

The tests drive :class:`~opencode_mlflow.source.legacy.LegacyAdapter`
through the real snapshot driver (bind → inspect → export) over sanitized
synthetic fixture databases, then assert on the normalized snapshot
through the replay adapter. Identity and deduplication edge cases use
ad-hoc scripts layered on the shared legacy fixture DDL.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from opencode_mlflow.content.canonical import DuplicateKeyObject
from opencode_mlflow.source.catalog import LEGACY_CATALOG_V1, SCHEMA_CATALOG_VERSION, TIE_RANKS
from opencode_mlflow.source.legacy import LegacyAdapter
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
    "SHARE_URL_FIXTURE_SENTINEL_LEGACY",
    "SHARE-SENTINEL",
)


def _legacy_exporter(connection: sqlite3.Connection) -> LegacyAdapter:
    return LegacyAdapter(connection, LEGACY_CATALOG_V1)


@contextmanager
def _exported(
    tmp_path: Path,
    script: str,
    make_exporter: Callable[[sqlite3.Connection], SourceExporter] = _legacy_exporter,
) -> Iterator[tuple[SourceSnapshot, NormalizedSnapshotAdapter]]:
    """Export a script-built legacy database and yield (snapshot, adapter)."""
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

_CHILD_TABLES = ("session", "message", "part", "todo")


def _direct_export(script: str) -> tuple[LegacyAdapter, object, object]:
    """Export a script-built database through a self-owned writer.

    Returns ``(adapter, returned stats, writer stats)``. The adapter and
    writer run over in-memory connections the test controls, so the
    export can be observed directly (statement traces, returned stats).
    """
    connection = sqlite3.connect(":memory:")
    writer_connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(script)
        writer_connection.executescript(_SNAPSHOT_DDL)
        adapter = LegacyAdapter(connection, LEGACY_CATALOG_V1)
        writer = SnapshotWriter(writer_connection, "legacy", identity_redaction)
        returned = adapter.export_to_snapshot(writer, identity_redaction)
        base = writer.finalize()
    finally:
        connection.close()
        writer_connection.close()
    return adapter, returned, base


def _child_select_counts(script: str) -> dict[str, int]:
    """Count executed SELECT statements per source table during an export."""
    connection = sqlite3.connect(":memory:")
    writer_connection = sqlite3.connect(":memory:")
    statements: list[str] = []
    try:
        connection.executescript(script)
        connection.set_trace_callback(statements.append)
        writer_connection.executescript(_SNAPSHOT_DDL)
        adapter = LegacyAdapter(connection, LEGACY_CATALOG_V1)
        writer = SnapshotWriter(writer_connection, "legacy", identity_redaction)
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
        # Identifier quoting ("session") is normalized away so the
        # per-table FROM clauses can be matched.
        padded = f"{text.upper().replace(chr(34), '')} "
        if not padded.startswith("SELECT"):
            continue
        for table in _CHILD_TABLES:
            if f"FROM {table.upper()} " in padded:
                counts[table] += 1
    return counts


def _fixture_script(extra: str = "") -> str:
    return fixture_script("legacy") + extra


class TestLegacyInspect:
    def test_inspect_reports_generation_and_catalog_version(self, tmp_path: Path) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(_fixture_script())
            adapter = LegacyAdapter(connection, LEGACY_CATALOG_V1)
            description = adapter.inspect()
        finally:
            connection.close()

        assert isinstance(description, SchemaDescription)
        assert description.generation == "legacy"
        assert description.catalog_version == SCHEMA_CATALOG_VERSION
        assert "session" in description.tables
        assert "message" in description.tables
        assert "part" in description.tables

    def test_inspect_exposes_rank_table_timestamp_units_and_classification(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(_fixture_script())
            adapter = LegacyAdapter(connection, LEGACY_CATALOG_V1)
            description = adapter.inspect()
        finally:
            connection.close()

        assert description.tie_ranks == dict(TIE_RANKS)
        assert description.timestamp_units["session.time_created"] == "ms"
        assert description.timestamp_units["message.time_created"] == "ms"
        assert description.timestamp_units["session.time_compacting"] == "ms"
        assert "user" in " ".join(description.classifications)
        assert any("part" in rule and "inherit" in rule for rule in description.classifications)

    def test_inspect_describes_safe_session_scoped_columns(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(_fixture_script())
            adapter = LegacyAdapter(connection, LEGACY_CATALOG_V1)
            description = adapter.inspect()
        finally:
            connection.close()

        assert "todo" in description.tables
        assert set(description.tables["todo"]) >= {"session_id", "content", "status", "position"}
        # The omitted secret-layout table is reported, never selected.
        assert description.warnings == ("session_share_omitted",)

    def test_inspect_rejects_a_foreign_catalog(self) -> None:
        from dataclasses import replace

        connection = sqlite3.connect(":memory:")
        try:
            with pytest.raises(ValueError):
                LegacyAdapter(connection, replace(LEGACY_CATALOG_V1, generation="current"))
        finally:
            connection.close()

    def test_bind_connection_rebinds_the_export_connection(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (snapshot, _adapter):
            assert snapshot.schema_description.generation == "legacy"


class TestLegacySessionExport:
    @pytest.mark.parametrize("statuses", [("failed", "pending"), ("pending", "failed")])
    def test_failed_and_pending_decoded_assistant_statuses_keep_failure_precedence(
        self, tmp_path: Path, statuses: tuple[str, str]
    ) -> None:
        extra = (
            "INSERT INTO message VALUES "
            "('msg-legacy-status-1', 'ses-legacy-001', 1771000030000, 1771000030000, "
            f'\'{{"role": "assistant", "status": "{statuses[0]}"}}\');'
            "INSERT INTO message VALUES "
            "('msg-legacy-status-2', 'ses-legacy-001', 1771000040000, 1771000040000, "
            f'\'{{"role": "assistant", "status": "{statuses[1]}"}}\');'
        )
        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            (summary,) = iter_turn_summaries(adapter.iter_records("ses-legacy-001"))

        assert summary.terminal_outcome == ASSISTANT_ERROR_OUTCOME

    def test_share_url_is_never_projected_or_captured(self, tmp_path: Path) -> None:
        sentinel = "SHARE_URL_SENTINEL_LEGACY"
        script = _fixture_script(f"UPDATE session SET share_url = '{sentinel}';")

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
            "ALTER TABLE session ADD COLUMN ACCESS_TOKEN TEXT;"
            "ALTER TABLE session ADD COLUMN Share_Secret TEXT;"
            "ALTER TABLE session ADD COLUMN SESSION_TOKEN TEXT;"
            "UPDATE session SET custom_flag = NULL, Share_URL = 'MIXED-SHARE-URL',"
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
            "INSERT INTO session (id, time_created) VALUES ('', 1771001000000);"
            "INSERT INTO session (id, time_created) VALUES ('   ', 1771001000001);"
        )

        with _exported(tmp_path, script) as (snapshot, adapter):
            headers = list(adapter.iter_session_headers())

        assert [header.session_id for header in headers] == [
            "ses-legacy-001",
            "ses-legacy-002",
            "ses-legacy-003",
        ]
        assert snapshot.stats.session_count == 3
        assert snapshot.stats.warning_count >= 2
        assert snapshot.export_warnings.count("session_missing_id") == 2

    def test_sessions_export_in_session_sort_key_order(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            headers = list(adapter.iter_session_headers())

        # ses-legacy-003 has no creation timestamp and sorts after the
        # timestamped sessions, by session ID.
        assert [header.session_id for header in headers] == [
            "ses-legacy-001",
            "ses-legacy-002",
            "ses-legacy-003",
        ]

    def test_headers_normalize_typed_fields_and_raw_columns(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            (header,) = [h for h in adapter.iter_session_headers() if h.session_id == "ses-legacy-001"]

        assert header.adapter_generation == "legacy"
        assert header.project_id == "proj-synth-2"
        assert header.workspace_id == "ws-synth-2"
        assert header.directory == "/repo/project-beta"
        assert header.title == "Synthetic legacy session one"
        assert header.agent == "synthetic-legacy-agent"
        assert header.model == {"provider": "synth-provider", "model": "synth-legacy-model"}
        assert header.cost == 0.5
        assert header.tokens_input == 80
        assert header.tokens_output == 210
        assert header.summary_additions == 2
        assert header.time_created is not None
        assert header.time_created.value == 1_771_000_000_000
        assert header.time_created.unit == "ms"
        assert header.time_compacting is None
        # The complete permitted session-row map is preserved, including
        # the additive unknown column.
        assert header.raw_columns["custom_flag"] == "synthetic-legacy-flag"
        assert header.raw_columns["id"] == "ses-legacy-001"

    def test_unknown_column_value_warns_unknown_source_field(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            headers = {h.session_id: h for h in adapter.iter_session_headers()}

        assert "unknown_source_field" in headers["ses-legacy-001"].warnings
        # A NULL value in the unknown column carries nothing to retain.
        assert "unknown_source_field" not in headers["ses-legacy-002"].warnings

    def test_unknown_json_keys_warn_and_remain_in_raw_and_decoded_values(self, tmp_path: Path) -> None:
        script = _fixture_script(
            'UPDATE message SET data = \'{"role": "user", "future_json_key": "synthetic"}\' '
            "WHERE id = 'msg-legacy-001';"
        )

        with _exported(tmp_path, script) as (_snapshot, adapter):
            record = next(
                record for record in adapter.iter_records("ses-legacy-001") if record.source_id == "msg-legacy-001"
            )

        assert "unknown_source_field" in record.warnings
        assert record.raw_json is not None and "future_json_key" in record.raw_json
        assert record.decoded_value["future_json_key"] == "synthetic"

    def test_records_round_trip_with_raw_json_and_columns(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-001"))

        by_id = {record.source_id: record for record in records}
        message = by_id["msg-legacy-001"]
        assert message.source_table == "message"
        assert message.source_type == "user"
        assert message.classification == "user"
        assert message.session_id == "ses-legacy-001"
        assert message.decoded_value == {"role": "user", "content": "synthetic legacy user prompt"}
        assert message.raw_json == '{"role": "user", "content": "synthetic legacy user prompt"}'
        assert message.raw_columns["session_id"] == "ses-legacy-001"
        assert message.timestamp is not None
        assert message.timestamp.value == 1_771_000_010_000

    def test_parts_attach_immediately_after_their_parent(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-001"))

        assert [record.source_id for record in records] == [
            "msg-legacy-001",
            "part-legacy-001",
            "part-legacy-002",
            "part-legacy-006",
            "msg-legacy-002",
            "part-legacy-003",
            "part-legacy-004",
            "part-legacy-005",
        ]
        parts = [record for record in records if record.source_table == "part"]
        assert all(record.parent_source_id for record in parts)
        assert {record.parent_source_id for record in parts} == {"msg-legacy-001", "msg-legacy-002"}

    def test_part_classification_inherits_the_parent_message(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            records = {record.source_id: record for record in adapter.iter_records("ses-legacy-001")}

        # A text part under a user message inherits the user class.
        assert records["part-legacy-001"].classification == "user"
        # A tool part is content, never a standalone user prompt; under an
        # assistant message it inherits assistant.
        assert records["part-legacy-004"].classification == "assistant"
        # Synthetic step markers classify synthetic.
        assert records["part-legacy-003"].classification == "synthetic"
        assert records["part-legacy-003"].source_type == "step-start"
        assert records["part-legacy-005"].classification == "synthetic"
        assert records["part-legacy-005"].source_type == "step-finish"

    def test_part_level_model_identity_is_preserved(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            records = {record.source_id: record for record in adapter.iter_records("ses-legacy-001")}

        decoded = records["part-legacy-005"].decoded_value
        assert isinstance(decoded, dict)
        assert decoded["model"] == {"provider": "synth-provider", "model": "synth-legacy-model"}

    def test_malformed_part_json_retains_raw_text(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            records = {record.source_id: record for record in adapter.iter_records("ses-legacy-001")}

        malformed = records["part-legacy-006"]
        assert malformed.decoded_value is None
        assert malformed.raw_json == '{"broken'
        assert "malformed_json" in malformed.warnings

    def test_duplicate_json_keys_survive_adapter_snapshot_and_replay(self, tmp_path: Path) -> None:
        raw = '{"role":"assistant","payload":{"note":"first","note":"second"}}'
        extra = (
            "INSERT INTO message VALUES "
            "('msg-duplicate-json', 'ses-legacy-001', 1771000017500, 1771000017500, "
            f"'{raw}');"
        )

        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            record = next(
                record for record in adapter.iter_records("ses-legacy-001") if record.source_id == "msg-duplicate-json"
            )

        assert isinstance(record.decoded_value, dict)
        duplicate = record.decoded_value["payload"]
        assert isinstance(duplicate, DuplicateKeyObject)
        assert duplicate.pairs == (("note", "first"), ("note", "second"))
        assert duplicate.raw_text == ""
        assert record.raw_json == raw

    def test_todos_are_session_scoped_with_fallback_identity(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (_snapshot, adapter):
            scoped = list(adapter.iter_session_scoped_records("ses-legacy-001"))
            turn_ids = [record.source_id for record in adapter.iter_records("ses-legacy-001")]

        assert "todo-synth" not in turn_ids
        assert len(scoped) == 1
        (todo,) = scoped
        assert todo.source_table == "todo"
        assert todo.source_type == "todo"
        assert todo.source_id is None
        # A missing ID never yields a None dedup key: the fallback identity
        # is a stable digest including the occurrence ordinal.
        assert isinstance(todo.source_identity, str) and len(todo.source_identity) == 64
        assert todo.classification is None
        assert todo.raw_columns["content"] == "synthetic legacy todo content"

    def test_unassociated_todo_is_reported_not_assigned(self, tmp_path: Path) -> None:
        source = tmp_path / "source.db"
        connection = sqlite3.connect(source)
        try:
            connection.executescript(_fixture_script())
            exporter = LegacyAdapter(connection, LEGACY_CATALOG_V1)
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

        assert "unassociated_record" in exporter.export_warnings
        assert all(record.raw_columns.get("content") != "synthetic orphan todo content" for record in all_records)


class TestLegacyIdentityAndDeduplication:
    @pytest.mark.parametrize("fallback_path", ["merge", "message_without_rowid", "session_without_rowid"])
    def test_tied_missing_id_parts_use_physical_rowid_for_order_and_identity(
        self, tmp_path: Path, fallback_path: str
    ) -> None:
        """A tied part's physical row keeps its occurrence identity stable.

        The content is deliberately the only differing value and is changed
        between two source copies.  It must not become the tie-breaker for
        either the whole-table merge or the per-parent fallback query.
        """

        def script_for(first_text: str, second_text: str) -> str:
            script = _fixture_script().replace(
                "CREATE TABLE part (\n    id TEXT PRIMARY KEY,",
                "CREATE TABLE part (\n    id TEXT,",
            )
            if fallback_path == "message_without_rowid":
                script = script.replace(
                    "CREATE TABLE message (\n    id TEXT PRIMARY KEY,",
                    "CREATE TABLE message (\n    id TEXT NOT NULL PRIMARY KEY,",
                ).replace(
                    "    data TEXT\n);\n\nCREATE TABLE part",
                    "    data TEXT\n) WITHOUT ROWID;\n\nCREATE TABLE part",
                )
            elif fallback_path == "session_without_rowid":
                script = script.replace(
                    "    time_archived INTEGER,\n    custom_flag TEXT\n);",
                    "    time_archived INTEGER,\n    custom_flag TEXT\n) WITHOUT ROWID;",
                )
            first_json = json.dumps({"type": "text", "text": first_text})
            second_json = json.dumps({"type": "text", "text": second_text})
            return script + (
                "INSERT INTO part VALUES "
                f"(NULL, 'msg-legacy-001', 'ses-legacy-001', 1771000014000, 1771000014000, '{first_json}');"
                "INSERT INTO part VALUES "
                f"(NULL, 'msg-legacy-001', 'ses-legacy-001', 1771000014000, 1771000014000, '{second_json}');"
            )

        def source_identities(script: str) -> list[tuple[str | None, str | None, object, object]]:
            with _exported(tmp_path, script) as (_snapshot, adapter):
                records = [
                    record
                    for record in adapter.iter_records("ses-legacy-001")
                    if record.source_table == "part"
                    and record.source_id is None
                    and "part-tied-" in str(record.raw_columns.get("data", ""))
                ]
            return [
                (record.source_identity, record.parent_source_id, record.raw_columns["data"], record.timestamp)
                for record in records
            ]

        baseline = source_identities(script_for("part-tied-first", "part-tied-second"))
        changed = source_identities(script_for("part-tied-first-changed", "part-tied-second-changed"))

        assert [item[2] for item in baseline] == [
            '{"type": "text", "text": "part-tied-first"}',
            '{"type": "text", "text": "part-tied-second"}',
        ]
        assert [item[2] for item in changed] == [
            '{"type": "text", "text": "part-tied-first-changed"}',
            '{"type": "text", "text": "part-tied-second-changed"}',
        ]
        expected_baseline: list[tuple[str, str, object, object]] = []
        for rowid, (_identity, _parent, data, record_timestamp) in zip((7, 8), baseline, strict=True):
            digest = canonical_row_digest(
                "part",
                {},
                None,
                session_id="ses-legacy-001",
                parent_key="msg-legacy-001",
                sequence_or_timestamp=record_timestamp,
                row_discriminator=rowid,
            )
            expected_baseline.append(
                (
                    fallback_source_identity(
                        "legacy",
                        "part",
                        "ses-legacy-001",
                        "msg-legacy-001",
                        record_timestamp,
                        digest,
                        1,
                    ),
                    "msg-legacy-001",
                    data,
                    record_timestamp,
                )
            )
        assert baseline == expected_baseline
        assert [(item[0], item[1]) for item in changed] == [(item[0], item[1]) for item in baseline]

    @pytest.mark.parametrize("session_without_rowid", [False, True])
    def test_tied_missing_id_todos_use_rowid_not_secret_content_for_order_and_identity(
        self, tmp_path: Path, session_without_rowid: bool
    ) -> None:
        """Session-scoped no-ID rows use rowid in both query and identity."""

        def script_for(first_content: str, second_content: str) -> str:
            script = _fixture_script()
            if session_without_rowid:
                script = script.replace(
                    "    time_archived INTEGER,\n    custom_flag TEXT\n);",
                    "    time_archived INTEGER,\n    custom_flag TEXT\n) WITHOUT ROWID;",
                )
            return script + (
                "INSERT INTO todo VALUES "
                "('ses-legacy-001', ?, 'pending', 'high', 7, 1771000015000, 1771000015000);"
                "INSERT INTO todo VALUES "
                "('ses-legacy-001', ?, 'pending', 'high', 7, 1771000015000, 1771000015000);"
            ).replace("?", "'" + first_content + "'", 1).replace("?", "'" + second_content + "'", 1)

        def scoped(script: str) -> list[tuple[str | None, object, object]]:
            with _exported(tmp_path, script) as (_snapshot, adapter):
                records = [
                    record
                    for record in adapter.iter_session_scoped_records("ses-legacy-001")
                    if record.raw_columns.get("content", "").startswith("todo-tied-")
                ]
            return [(record.source_identity, record.raw_columns["content"], record.timestamp) for record in records]

        baseline = scoped(script_for("todo-tied-z-secret", "todo-tied-a-secret"))
        changed = scoped(script_for("todo-tied-z-changed", "todo-tied-a-changed"))

        assert [item[1] for item in baseline] == ["todo-tied-z-secret", "todo-tied-a-secret"]
        assert [item[1] for item in changed] == ["todo-tied-z-changed", "todo-tied-a-changed"]
        assert [item[0] for item in changed] == [item[0] for item in baseline]
        assert [item[0] for item in baseline] == [
            fallback_source_identity(
                "legacy",
                "todo",
                "ses-legacy-001",
                "ses-legacy-001",
                timestamp,
                canonical_row_digest(
                    "todo",
                    {},
                    None,
                    session_id="ses-legacy-001",
                    parent_key="ses-legacy-001",
                    sequence_or_timestamp=timestamp,
                    row_discriminator=rowid,
                ),
                1,
            )
            for rowid, timestamp in ((3, baseline[0][2]), (4, baseline[1][2]))
        ]

    @pytest.mark.parametrize(
        "insertion_order",
        [
            (None, "", "msg-tied"),
            ("msg-tied", None, ""),
            ("", "msg-tied", None),
        ],
    )
    def test_tied_nullable_message_ids_keep_parts_through_normalized_snapshot(
        self,
        tmp_path: Path,
        insertion_order: tuple[str | None, ...],
    ) -> None:
        """SQL/group ordering must not drop the unique NULL parent's part."""
        timestamp = 1_771_000_090_000
        message_rows = {
            None: (
                "INSERT INTO message VALUES "
                f"(NULL, 'ses-legacy-001', {timestamp}, {timestamp}, "
                '\'{"role": "assistant", "content": "null parent"}\');'
            ),
            "": (
                "INSERT INTO message VALUES "
                f"('', 'ses-legacy-001', {timestamp}, {timestamp}, "
                '\'{"role": "assistant", "content": "empty parent"}\');'
            ),
            "msg-tied": (
                "INSERT INTO message VALUES "
                f"('msg-tied', 'ses-legacy-001', {timestamp}, {timestamp}, "
                '\'{"role": "assistant", "content": "explicit parent"}\');'
            ),
        }
        extra = "".join(message_rows[parent_id] for parent_id in insertion_order)
        extra += (
            "INSERT INTO part VALUES "
            f"('part-empty-parent', '', 'ses-legacy-001', {timestamp + 1}, {timestamp + 1}, "
            '\'{"type": "text", "text": "empty child"}\');'
            "INSERT INTO part VALUES "
            f"('part-null-parent', NULL, 'ses-legacy-001', {timestamp + 2}, {timestamp + 2}, "
            '\'{"type": "text", "text": "null child"}\');'
        )
        script = _fixture_script(extra).replace("message_id TEXT NOT NULL", "message_id TEXT")

        with _exported(tmp_path, script) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-001"))

        tied_parts = [
            record.source_id for record in records if record.source_id in {"part-empty-parent", "part-null-parent"}
        ]
        assert tied_parts == ["part-empty-parent", "part-null-parent"]
        by_id = {record.source_id: record for record in records}
        assert by_id["part-empty-parent"].parent_source_id == ""
        assert by_id["part-empty-parent"].source_identity is None
        assert by_id["part-null-parent"].parent_source_id is not None
        assert by_id["part-null-parent"].parent_source_id == next(
            record.source_identity
            for record in records
            if record.source_table == "message" and record.source_id is None
        )

    def test_missing_id_identity_and_join_projection_ignore_secret_json(self, tmp_path: Path) -> None:
        def script(name: str, secret: str) -> Path:
            path = tmp_path / name
            connection = sqlite3.connect(path)
            try:
                connection.executescript(_fixture_script().replace("message_id TEXT NOT NULL", "message_id TEXT"))
                connection.execute(
                    "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000, ?)",
                    (json.dumps({"role": "user", "password": secret}),),
                )
                connection.execute(
                    "INSERT INTO part VALUES (NULL, NULL, 'ses-legacy-001', 1771000016000, 1771000016000, ?)",
                    (json.dumps({"type": "text", "password": secret}),),
                )
                connection.commit()
            finally:
                connection.close()
            return path

        baseline = script("legacy-identity-baseline.db", "password-alpha")
        changed = script("legacy-identity-changed.db", "password-beta")

        exported: list[dict[str, object]] = []
        for path in (baseline, changed):
            connection = sqlite3.connect(path)
            try:
                exporter = LegacyAdapter(connection, LEGACY_CATALOG_V1)
            finally:
                connection.close()
            with create_normalized_snapshot(path, tmp_path, exporter, identity_redaction) as snapshot:
                adapter = NormalizedSnapshotAdapter(snapshot)
                records = list(adapter.iter_records("ses-legacy-001"))
                exported.append(
                    {
                        "messages": [
                            (record.source_identity, record.parent_source_id)
                            for record in records
                            if record.source_table == "message" and record.source_id is None
                        ],
                        "parts": [
                            (record.source_identity, record.join_projection_key, record.parent_source_id)
                            for record in records
                            if record.source_table == "part" and record.source_id is None
                        ],
                    }
                )

        assert exported[0] == exported[1]

    def test_missing_id_message_gets_fallback_identity(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "synthetic no-id prompt"}\');'
        )
        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            no_id = [r for r in adapter.iter_records("ses-legacy-001") if r.source_id is None]

        assert len(no_id) == 1
        record = no_id[0]
        assert record.source_table == "message"
        assert isinstance(record.source_identity, str) and len(record.source_identity) == 64
        assert record.classification == "user"

    def test_identical_missing_id_rows_remain_distinct(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "identical"}\');'
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "identical"}\');'
        )
        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            no_id = [r for r in adapter.iter_records("ses-legacy-001") if r.source_id is None]

        assert len(no_id) == 2
        identities = {record.source_identity for record in no_id}
        assert len(identities) == 2

    def test_ambiguous_missing_id_parts_are_preserved_without_parent_assignment(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "identical"}\');'
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "identical"}\');'
            "INSERT INTO part VALUES (NULL, NULL, 'ses-legacy-001', 1771000016000, 1771000016000,"
            ' \'{"type": "text", "text": "shared orphan part"}\');'
        )
        # The shared part carries a NULL message_id, so its parent linkage
        # only exists through the NULL-safe message_id join.
        script = _fixture_script(extra).replace("message_id TEXT NOT NULL", "message_id TEXT")
        with _exported(tmp_path, script) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-001"))

        no_id_messages = [r for r in records if r.source_table == "message" and r.source_id is None]
        parts = [
            r
            for r in records
            if r.source_table == "part"
            and r.raw_columns.get("data") == '{"type": "text", "text": "shared orphan part"}'
        ]
        # Two distinct no-ID parents...
        assert len(no_id_messages) == 2
        # There is no defensible parent assignment within this session.
        assert len(parts) == 1
        assert parts[0].parent_source_id is None
        assert parts[0].source_identity != parts[0].join_projection_key
        assert "ambiguous_missing_message_id" in parts[0].warnings
        assert all("ambiguous_missing_message_id" in r.warnings for r in no_id_messages)

    def test_identical_missing_id_parts_stay_distinct(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO message VALUES ('msg-no-id-parent', 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "parent"}\');'
            "INSERT INTO part VALUES (NULL, 'msg-no-id-parent', 'ses-legacy-001', 1771000016000, 1771000016000,"
            ' \'{"type": "text", "text": "duplicate part"}\');'
            "INSERT INTO part VALUES (NULL, 'msg-no-id-parent', 'ses-legacy-001', 1771000016000, 1771000016000,"
            ' \'{"type": "text", "text": "duplicate part"}\');'
        )
        with _exported(tmp_path, _fixture_script(extra)) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-001"))

        duplicates = [
            r
            for r in records
            if r.source_table == "part" and r.raw_columns.get("data") == '{"type": "text", "text": "duplicate part"}'
        ]
        assert len(duplicates) == 2
        assert len({record.source_identity for record in duplicates}) == 2
        assert len({record.join_projection_key for record in duplicates}) == 2

    def test_duplicate_explicit_ids_collapse_to_first_projection(self, tmp_path: Path) -> None:
        script = _fixture_script().replace(" PRIMARY KEY", "")
        script += (
            "INSERT INTO part VALUES ('part-dup', 'msg-legacy-001', 'ses-legacy-001',"
            ' 1771000014000, 1771000014000, \'{"type": "text", "text": "first"}\');'
            "INSERT INTO part VALUES ('part-dup', 'msg-legacy-001', 'ses-legacy-001',"
            ' 1771000014000, 1771000014000, \'{"type": "text", "text": "second"}\');'
        )
        with _exported(tmp_path, script) as (_snapshot, adapter):
            records = [r for r in adapter.iter_records("ses-legacy-001") if r.source_id == "part-dup"]

        assert len(records) == 1
        assert records[0].decoded_value == {"type": "text", "text": "first"}

    def test_session_without_id_is_skipped_and_reported(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO session VALUES (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,"
            " NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,"
            " NULL, NULL, NULL, NULL, NULL, NULL);"
        )
        with _exported(tmp_path, _fixture_script(extra)) as (snapshot, adapter):
            headers = list(adapter.iter_session_headers())

        assert [header.session_id for header in headers] == [
            "ses-legacy-001",
            "ses-legacy-002",
            "ses-legacy-003",
        ]
        assert snapshot.stats.session_count == 3


class TestLegacySafety:
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
            "INSERT INTO session_share VALUES ('share-extra', 'ses-legacy-001',"
            " 'https://share.example.invalid/extra', 'SECRET-EXTRA-SHARE-SENTINEL');"
        )
        with _exported(tmp_path, _fixture_script()) as (baseline, _adapter):
            pass
        with _exported(tmp_path, _fixture_script(extra_secret_rows)) as (extended, _adapter):
            pass

        assert extended.stats.session_count == baseline.stats.session_count
        assert extended.stats.record_count == baseline.stats.record_count
        assert extended.stats.complete and baseline.stats.complete

    def test_exported_stats_are_complete_and_honest(self, tmp_path: Path) -> None:
        with _exported(tmp_path, _fixture_script()) as (snapshot, _adapter):
            stats = snapshot.stats

        assert stats.complete is True
        assert stats.schema_generation == "legacy"
        assert stats.session_count == 3
        # 4 message rows + 6 part rows + 1 associated todo row; the
        # unassociated todo is reported, never written.
        assert stats.record_count == 11

    def test_protocol_shape_matches_the_exporter_contract(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript(_fixture_script())
            adapter = LegacyAdapter(connection, LEGACY_CATALOG_V1)
        finally:
            connection.close()

        assert isinstance(adapter, SourceExporter)


class TestLegacyOptionalScopedTables:
    def test_export_succeeds_without_the_todo_table(self, tmp_path: Path) -> None:
        """The session-scoped todo table is optional for a legacy match."""
        script = _fixture_script() + "DROP TABLE todo;"
        with _exported(tmp_path, script) as (snapshot, adapter):
            headers = list(adapter.iter_session_headers())
            scoped_records = [r for r in adapter.iter_session_scoped_records("ses-legacy-001")]

        assert [header.session_id for header in headers] == [
            "ses-legacy-001",
            "ses-legacy-002",
            "ses-legacy-003",
        ]
        assert scoped_records == []
        assert snapshot.stats.record_count == 10

    def test_todo_table_missing_tiebreak_columns_exports_cleanly(self, tmp_path: Path) -> None:
        """The optional todo table's columns are not probe-validated.

        A todo table without its ``position`` tiebreak column exported
        cleanly before the merge-join rework: ordering references only
        observed columns, so the missing tiebreak degrades to ordering by
        the remaining columns instead of failing identifier resolution.
        """
        script = _fixture_script() + (
            "DROP TABLE todo;"
            "CREATE TABLE todo ("
            "session_id TEXT NOT NULL, content TEXT, status TEXT, priority TEXT,"
            " time_created INTEGER, time_updated INTEGER);"
            "INSERT INTO todo VALUES ("
            "'ses-legacy-001', 'synthetic legacy todo content', 'pending', 'high',"
            " 1771000005000, 1771000006000);"
        )
        with _exported(tmp_path, script) as (snapshot, adapter):
            scoped = [record.source_table for record in adapter.iter_session_scoped_records("ses-legacy-001")]

        assert snapshot.stats.complete is True
        assert scoped == ["todo"]
        # 4 messages + 6 parts + the one associated todo.
        assert snapshot.stats.record_count == 11

        unjoinable = _fixture_script() + (
            "DROP TABLE todo;"
            "CREATE TABLE todo (content TEXT, status TEXT, position INTEGER, time_created INTEGER);"
            "INSERT INTO todo VALUES ('orphan-without-session-id', 'pending', 1, 1771000005000);"
        )
        with _exported(tmp_path, unjoinable) as (unjoinable_snapshot, unjoinable_adapter):
            unjoinable_scoped = list(unjoinable_adapter.iter_session_scoped_records("ses-legacy-001"))
        assert unjoinable_snapshot.stats.complete is True
        assert unjoinable_scoped == []
        assert unjoinable_snapshot.stats.record_count == 10
        assert unjoinable_snapshot.export_warnings == ("unassociated_record",)

    def test_scoped_numeric_tiebreaks_put_invalid_values_last_deterministically(self, tmp_path: Path) -> None:
        script = (
            _fixture_script()
            + """
        DELETE FROM todo;
        INSERT INTO todo VALUES
            ('ses-legacy-001', 'todo-valid', 'pending', 'high', 1, 1771000100000, 10),
            ('ses-legacy-001', 'todo-text-time', 'pending', 'high', 'bad-position', 1771000100000, 20),
            ('ses-legacy-001', 'todo-null-time', 'pending', 'high', NULL, 1771000100000, NULL),
            ('ses-legacy-001', 'todo-huge-time', 'pending', 'high', 1e999, 1771000100000, 30),
            ('ses-legacy-001', 'todo-invalid-position', 'pending', 'high', 'bad-position-2',
             1771000100000, 'bad-time');
        """
        )

        with _exported(tmp_path, script) as (_snapshot, adapter):
            todos = [
                record.raw_columns["content"]
                for record in adapter.iter_session_scoped_records("ses-legacy-001")
                if record.source_table == "todo"
            ]

        assert todos == [
            "todo-valid",
            "todo-text-time",
            "todo-huge-time",
            "todo-null-time",
            "todo-invalid-position",
        ]

    def test_missing_primary_time_created_still_guards_timestamp_tiebreaks(self, tmp_path: Path) -> None:
        script = (
            _fixture_script()
            + """
        DROP TABLE todo;
        CREATE TABLE todo (
            session_id TEXT NOT NULL, content TEXT, status TEXT, position INTEGER,
            time_updated INTEGER
        );
        INSERT INTO todo VALUES
            -- Insert the later timestamp first: timestamp order must beat
            -- physical insertion order when position is the primary key.
            ('ses-legacy-001', 'todo-later', 'pending', 1, 1771000101000),
            ('ses-legacy-001', 'todo-earlier', 'pending', 1, 1771000099000),
            ('ses-legacy-001', 'todo-valid', 'pending', 1, 1771000100000),
            ('ses-legacy-001', 'todo-text-time', 'pending', 1, 'bad-time'),
            ('ses-legacy-001', 'todo-null-time', 'pending', 1, NULL),
            ('ses-legacy-001', 'todo-huge-time', 'pending', 1, 1000000000000000),
            ('ses-legacy-001', 'todo-infinite-time', 'pending', 1, 1e999);
        """
        )

        with _exported(tmp_path, script) as (_snapshot, adapter):
            todos = [
                record.raw_columns["content"]
                for record in adapter.iter_session_scoped_records("ses-legacy-001")
                if record.source_table == "todo"
            ]

        assert todos == [
            "todo-earlier",
            "todo-valid",
            "todo-later",
            "todo-text-time",
            "todo-null-time",
            "todo-huge-time",
            "todo-infinite-time",
        ]


class TestLegacyEnvelope:
    def test_canonical_envelope_carries_record_provenance(self, tmp_path: Path) -> None:
        """The fallback identity and occurrence ordinal ride the protected envelope."""
        extra = (
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "no id"}\');'
        )
        with _exported(tmp_path, _fixture_script(extra)) as (snapshot, _adapter):
            row = snapshot.connection.execute(
                "SELECT canonical_record FROM snapshot_records WHERE source_id IS NULL AND source_table = 'message'"
            ).fetchone()

        assert row is not None
        stored = json.loads(row[0])
        record_section = stored["record"]
        assert record_section["source_identity"] is not None
        assert record_section["source_table"] == "message"
        # The identity fields are persisted verbatim (protected from masking).
        assert stored["raw_text"] == '{"role": "user", "content": "no id"}'

    def test_returned_stats_include_export_level_warnings(self) -> None:
        """Unassociated-record warnings ride the exporter's returned stats.

        The snapshot driver keeps the writer's own stats, so the
        exporter-level warnings (rows that were reported, not written)
        are added on top of the writer's count in the returned stats.
        """
        adapter, returned, base = _direct_export(_fixture_script())

        assert returned.complete is True
        assert returned.session_count == base.session_count == 3
        assert returned.record_count == base.record_count == 11
        # The single unassociated todo is reported on top of the writer's
        # own warning count.
        assert adapter.export_warnings == ("unassociated_record",)
        assert returned.warning_count == base.warning_count + 1

    def test_snapshot_stats_include_exporter_warnings(self, tmp_path: Path) -> None:
        """Exporter-level warnings reach ``SourceSnapshot.stats`` (finding 2).

        The driver merges the exporter's returned warning count into the
        kept stats, so an unassociated record raises the snapshot's
        warning count by exactly one.
        """
        with _exported(tmp_path, _fixture_script()) as (with_orphan, _adapter):
            pass
        cleaned = _fixture_script() + "DELETE FROM todo WHERE session_id = 'ses-legacy-unknown';"
        with _exported(tmp_path, cleaned) as (clean, _adapter):
            pass

        assert with_orphan.stats.complete and clean.stats.complete
        assert with_orphan.stats.session_count == clean.stats.session_count
        assert with_orphan.stats.record_count == clean.stats.record_count
        assert with_orphan.stats.warning_count == clean.stats.warning_count + 1
        assert with_orphan.export_warnings == ("unassociated_record",)
        assert clean.export_warnings == ()


class TestBoundedChildQueries:
    """Finding 3: child-table access is a constant number of queries.

    Messages, parts, and todos stream through ONE merge-joined cursor per
    table for the whole database, so the number of executed SELECT
    statements per table does not grow with the number of messages.
    """

    @staticmethod
    def _scaled_script(message_count: int) -> str:
        rows = [
            (
                f"INSERT INTO message VALUES ('msg-scale-{index}', 'ses-legacy-001',"
                f" {1771001000000 + index}, {1771001000000 + index},"
                f' \'{{"role": "user", "content": "scale {index}"}}\');'
            )
            for index in range(message_count)
        ]
        rows.extend(
            (
                f"INSERT INTO part VALUES ('part-scale-{index}-{part}', 'msg-scale-{index}',"
                f" 'ses-legacy-001', {1771001000000 + index}, {1771001000000 + index},"
                f' \'{{"type": "text", "text": "scale part {part}"}}\');'
            )
            for index in range(message_count)
            for part in range(3)
        )
        return _fixture_script("".join(rows))

    def test_child_query_count_is_constant_in_message_count(self) -> None:
        counts = [_child_select_counts(self._scaled_script(size)) for size in (0, 5, 25)]

        assert counts[0] == counts[1] == counts[2]
        # One merge-joined pass per table plus the metadata/rowid probes
        # and the unassociated-todo aggregate — a small fixed budget.
        assert sum(counts[0].values()) <= 12

    def test_scaled_export_produces_every_record(self, tmp_path: Path) -> None:
        with _exported(tmp_path, self._scaled_script(200)) as (snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-001"))
            order = [r.source_id for r in adapter.iter_records("ses-legacy-001")]

        # Fixture rows (2 messages + 6 parts in ses-legacy-001) plus 200
        # messages each with 3 parts, parents before their children.
        assert snapshot.stats.record_count == 11 + 200 + 600
        assert len(records) == 808
        assert order.index("msg-scale-0") < order.index("part-scale-0-0")
        assert order.index("part-scale-0-2") < order.index("msg-scale-1")


class TestWithoutRowidFallback:
    """Tables without a rowid keep the per-session/per-parent query paths."""

    @staticmethod
    def _records_by_session(adapter: NormalizedSnapshotAdapter) -> dict[str, list[str | None]]:
        return {
            header.session_id: [record.source_id for record in adapter.iter_records(header.session_id)]
            for header in adapter.iter_session_headers()
        }

    def test_message_and_part_without_rowid_match_the_joined_export(self, tmp_path: Path) -> None:
        script = _fixture_script().replace("    data TEXT\n);", "    data TEXT\n) WITHOUT ROWID;")
        with _exported(tmp_path, _fixture_script()) as (_snapshot, baseline):
            expected = self._records_by_session(baseline)
        with _exported(tmp_path, script) as (_snapshot, fallback):
            actual = self._records_by_session(fallback)

        assert actual == expected

    def test_session_without_rowid_falls_back_to_per_session_queries(self, tmp_path: Path) -> None:
        script = _fixture_script().replace(
            "    time_archived INTEGER,\n    custom_flag TEXT\n);",
            "    time_archived INTEGER,\n    custom_flag TEXT\n) WITHOUT ROWID;",
        )
        script += "DROP TABLE todo;"
        with _exported(tmp_path, _fixture_script() + "DROP TABLE todo;") as (_snapshot, baseline):
            expected = self._records_by_session(baseline)
        with _exported(tmp_path, script) as (_snapshot, fallback):
            actual = self._records_by_session(fallback)

        assert actual == expected
