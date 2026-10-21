"""Unit tests for the normalized snapshot and its replay adapter.

This module hosts the Task 3 snapshot behavior tests (temporary snapshot
creation, redaction-before-persistence, failure mapping, cleanup) together
with the replay tests, because the replay adapter is the consumer that
proves the snapshot contract: discovery and replay read the same
normalized temporary tables, never the live source.

WAL-specific concurrency behavior lives in
``tests/integration/test_wal_snapshot.py`` (marker ``integration``).
"""

from __future__ import annotations

import base64
import json
import sqlite3
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from opencode_mlflow.content.redaction import DeterministicRedactor
from opencode_mlflow.contracts import CanonicalRecordEnvelope, RedactionMode
from opencode_mlflow.errors import SnapshotError, SourceCompatibilityError
from opencode_mlflow.source.model import (
    SchemaDescription,
    SessionHeader,
    SourceRecord,
    TurnSummary,
    normalize_timestamp,
    source_database_fingerprint,
)
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter, ReplayTurn
from opencode_mlflow.source.snapshot import (
    ExportStats,
    SourceExporter,
    SourceSnapshot,
    create_normalized_snapshot,
)
from tests.support.fake_source import (
    FixtureExporter,
    ScriptedExporter,
    ScriptedRecord,
    ScriptedSession,
    identity_redaction,
    masking_redaction,
)
from tests.support.sqlite_builder import build_fixture_db

pytestmark = pytest.mark.unit

SECRET_SENTINELS = (
    "credential-sentinel",
    "SECRET-ACCOUNT-TOKEN-SENTINEL",
    "SECRET-ACCOUNT-STATE-SENTINEL",
    "SECRET-CONTROL-ACCOUNT-SENTINEL",
    "SHARE_URL_FIXTURE_SENTINEL_CURRENT",
    "SHARE_URL_FIXTURE_SENTINEL_LEGACY",
    "SECRET-SHARE-SENTINEL",
)


def _session(session_id: str = "ses-1", generation: str = "current") -> ScriptedSession:
    header = SessionHeader(
        session_id=session_id,
        adapter_generation=generation,
        directory="/repo/project",
        title="Synthetic session",
        time_created=normalize_timestamp(1_772_000_000_000, "ms"),
    )
    envelope = CanonicalRecordEnvelope(
        decoded={"directory": "/repo/project", "title": "Synthetic session"},
        raw_text=None,
        raw_columns={"id": session_id, "directory": "/repo/project"},
        warnings=(),
    )
    return ScriptedSession(header=header, canonical_session_row=envelope)


def _record(
    source_id: str,
    session_id: str = "ses-1",
    table: str = "session_message",
    source_type: str = "user",
    raw_json: str | None = '{"content": [{"type": "text", "text": "synthetic prompt"}]}',
    seq: int | None = 1,
    classification: str | None = "user",
) -> ScriptedRecord:
    decoded = SourceRecord.from_row(
        source_table=table,
        source_type=source_type,
        source_id=source_id,
        raw_json=raw_json,
    )
    record = SourceRecord(
        source_table=decoded.source_table,
        source_type=decoded.source_type,
        source_id=decoded.source_id,
        raw_json=decoded.raw_json,
        decoded_value=decoded.decoded_value,
        warnings=decoded.warnings,
        session_id=session_id,
        sequence_domain="session_message.seq" if table == "session_message" else None,
        sequence_value=seq,
        timestamp=normalize_timestamp(1_772_000_010_000, "ms"),
        classification=classification,
    )
    envelope = CanonicalRecordEnvelope(
        decoded=record.decoded_value,
        raw_text=record.raw_json,
        raw_columns={"seq": seq},
        warnings=record.warnings,
    )
    return ScriptedRecord(
        record=record,
        order_key=(seq if seq is not None else -1, source_id if source_id is not None else ""),
        canonical_record=envelope,
        raw_permitted_fields={"seq": seq, "type": source_type},
    )


def _snapshot_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return sorted(str(row[0]) for row in rows)


def _all_snapshot_text(connection: sqlite3.Connection) -> str:
    parts: list[str] = []
    for table in ("snapshot_sessions", "snapshot_records"):
        for row in connection.execute(f"SELECT * FROM {table}"):
            parts.extend(str(value) for value in row)
    return "\n".join(parts)


def _build_current_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source.db"
    connection = build_fixture_db(source, "current")
    connection.close()
    return source


class TestSnapshotCreation:
    def test_snapshot_uses_restrictive_temporary_permissions(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(sessions=[_session()], records=[_record("msg-1")])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            directory_mode = stat.S_IMODE(snapshot.snapshot_path.parent.stat().st_mode)
            file_mode = stat.S_IMODE(snapshot.snapshot_path.stat().st_mode)
            assert directory_mode == 0o700
            assert file_mode == 0o600

    def test_snapshot_lives_under_the_requested_temp_root(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(sessions=[_session()], records=[])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            assert snapshot.snapshot_path.parent.parent == tmp_path
            assert snapshot.snapshot_path.name == "snapshot.db"

    def test_snapshot_contains_only_the_two_normalized_tables(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(sessions=[_session()], records=[_record("msg-1")])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            assert _snapshot_tables(snapshot.connection) == ["snapshot_records", "snapshot_sessions"]

    def test_snapshot_exposes_fingerprint_description_and_stats(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(sessions=[_session()], records=[_record("msg-1"), _record("msg-2", seq=2)])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            assert isinstance(snapshot, SourceSnapshot)
            assert snapshot.fingerprint == source_database_fingerprint("current", source)
            assert snapshot.schema_description.generation == "current"
            assert snapshot.stats.session_count == 1
            assert snapshot.stats.record_count == 2
            assert snapshot.stats.warning_count == 0
            assert snapshot.stats.complete is True
            assert snapshot.stats.schema_generation == "current"

    def test_batched_writes_produce_identical_rows(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        records = [_record(f"msg-{index:02d}", seq=index) for index in range(1, 6)]
        exporter = ScriptedExporter(sessions=[_session()], records=records)
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction, write_batch=1) as snapshot:
            count = snapshot.connection.execute("SELECT count(*) FROM snapshot_records").fetchone()[0]
            ids = [
                row[0] for row in snapshot.connection.execute("SELECT source_id FROM snapshot_records ORDER BY rowid")
            ]
        assert count == 5
        assert ids == [f"msg-{index:02d}" for index in range(1, 6)]

    def test_no_live_source_connection_remains_after_context_setup(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(sessions=[_session()], records=[])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            assert exporter.connection is not None
            with pytest.raises(sqlite3.ProgrammingError):
                exporter.connection.execute("SELECT 1")
            # The replay connection is a different, live connection.
            assert snapshot.connection.execute("SELECT count(*) FROM snapshot_sessions").fetchone() is not None

    def test_replay_connection_is_read_only_in_one_transaction(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(sessions=[_session()], records=[])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            assert snapshot.connection.execute("PRAGMA query_only").fetchone()[0] == 1
            assert snapshot.connection.in_transaction is True
            with pytest.raises(sqlite3.OperationalError):
                snapshot.connection.execute("BEGIN")
            with pytest.raises(sqlite3.Error):
                snapshot.connection.execute("INSERT INTO snapshot_records VALUES ('x','x','x','x','x','x')")

    def test_cleanup_removes_snapshot_files_and_directory(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(sessions=[_session()], records=[_record("msg-1")])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            snapshot_path = snapshot.snapshot_path
        assert not snapshot_path.exists()
        assert not snapshot_path.parent.exists()
        assert list(tmp_path.iterdir()) == [tmp_path / "source.db"]

    def test_cleanup_failure_is_reported_without_paths_or_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(sessions=[_session()], records=[_record("msg-1")])
        original_unlink = Path.unlink

        def failing_unlink(self: Path, *args: object, **kwargs: object) -> None:
            if self.name == "snapshot.db":
                raise OSError(f"simulated failure unlinking {self}")
            original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", failing_unlink)
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            pass

        assert snapshot.cleanup_warnings == ["temporary snapshot cleanup failed"]
        for warning in snapshot.cleanup_warnings:
            assert str(snapshot.snapshot_path) not in warning
            assert "simulated failure" not in warning


class TestSnapshotFailures:
    def test_extraction_failure_maps_to_source_snapshot_failed(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(
            sessions=[_session()],
            records=[_record("msg-1")],
            failure=RuntimeError("synthetic extraction failure"),
        )
        with pytest.raises(SnapshotError) as excinfo:
            with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction):
                pass

        assert excinfo.value.code == "source_snapshot_failed"
        assert excinfo.value.exit_status == 2
        assert "synthetic extraction failure" not in str(excinfo.value)
        # Cleanup still removed the temporary snapshot despite the failure.
        assert [entry.name for entry in tmp_path.iterdir()] == ["source.db"]

    def test_disk_failure_during_population_maps_to_source_snapshot_failed(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(
            sessions=[_session()],
            records=[],
            failure=sqlite3.OperationalError("database or disk is full"),
        )
        with pytest.raises(SnapshotError) as excinfo:
            with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction):
                pass

        assert excinfo.value.code == "source_snapshot_failed"
        assert excinfo.value.exit_status == 2

    def test_population_and_cleanup_failures_are_attached_to_snapshot_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(
            sessions=[_session()],
            records=[_record("msg-1")],
            failure=sqlite3.OperationalError("synthetic population failure"),
        )
        original_unlink = Path.unlink
        original_rmdir = Path.rmdir

        def failing_unlink(self: Path, *args: object, **kwargs: object) -> None:
            if self.name == "snapshot.db":
                raise OSError("synthetic unlink failure")
            original_unlink(self, *args, **kwargs)

        def failing_rmdir(self: Path) -> None:
            if self.name.startswith("opencode-mlflow-snapshot-"):
                raise OSError("synthetic rmdir failure")
            original_rmdir(self)

        monkeypatch.setattr(Path, "unlink", failing_unlink)
        monkeypatch.setattr(Path, "rmdir", failing_rmdir)
        with pytest.raises(SnapshotError) as excinfo:
            with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction):
                pass

        assert excinfo.value.code == "source_snapshot_failed"
        assert excinfo.value.cleanup_warnings == ("temporary snapshot cleanup failed",)
        assert "synthetic population failure" not in str(excinfo.value)

    def test_busy_timeout_failure_maps_to_source_snapshot_failed(self, tmp_path: Path) -> None:
        source = tmp_path / "locked.db"
        connection = sqlite3.connect(source)
        connection.execute("CREATE TABLE session (id TEXT)")
        connection.commit()
        connection.close()

        writer = sqlite3.connect(source, isolation_level=None)
        try:
            writer.execute("BEGIN EXCLUSIVE")
            exporter = ScriptedExporter()
            with pytest.raises(SnapshotError) as excinfo:
                with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction, busy_timeout_ms=200):
                    pass
        finally:
            writer.execute("ROLLBACK")
            writer.close()

        assert excinfo.value.code == "source_snapshot_failed"
        assert excinfo.value.exit_status == 2

    def test_unavailable_source_maps_to_source_unavailable(self, tmp_path: Path) -> None:
        exporter = ScriptedExporter()
        with pytest.raises(SnapshotError) as excinfo:
            with create_normalized_snapshot(tmp_path / "missing.db", tmp_path, exporter, identity_redaction):
                pass

        assert excinfo.value.code == "source_unavailable"
        assert excinfo.value.exit_status == 2

    def test_exporter_returning_incomplete_stats_fails_safely(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(sessions=[_session()], records=[_record("msg-1")], attest_completion=False)
        with pytest.raises(SnapshotError) as excinfo:
            with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction):
                pass

        assert excinfo.value.code == "source_snapshot_failed"
        assert [entry.name for entry in tmp_path.iterdir()] == ["source.db"]

    def test_compatibility_failure_from_inspect_propagates_unwrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "unsupported.db"
        connection = sqlite3.connect(source)
        connection.execute("CREATE TABLE unrelated (id TEXT)")
        connection.commit()
        connection.close()

        original_rmdir = Path.rmdir

        def failing_rmdir(self: Path) -> None:
            if self.name.startswith("opencode-mlflow-snapshot-"):
                raise OSError("synthetic cleanup path failure")
            original_rmdir(self)

        monkeypatch.setattr(Path, "rmdir", failing_rmdir)
        exporter = FixtureExporter()
        with pytest.raises(SourceCompatibilityError) as excinfo:
            with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction):
                pass

        assert excinfo.value.code == "unsupported_schema"
        assert excinfo.value.exit_status == 2
        assert excinfo.value.cleanup_warnings == ("temporary snapshot cleanup failed",)
        assert str(tmp_path) not in str(excinfo.value)
        assert "synthetic cleanup path failure" not in str(excinfo.value)


class TestSnapshotRedaction:
    def test_internal_tag_singletons_round_trip_as_ordinary_redacted_mappings(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        token_bytes = b"sk-abcdefghijklmnop"
        payload = {
            "bytes_collision": {"__opencode_mlflow_bytes_b64__": base64.b64encode(token_bytes).decode("ascii")},
            "duplicate_collision": {
                "__opencode_mlflow_duplicate_keys__": {
                    "pairs": [["ordinary", "source-value"]],
                    "raw_text": "ordinary-source-value",
                }
            },
            "escape_collision": {"__opencode_mlflow_mapping__": [["ordinary", "source-value"]]},
        }
        scripted = _record("msg-tag-collisions", raw_json=None)
        scripted.canonical_record = CanonicalRecordEnvelope(
            decoded=payload,
            raw_text=None,
            raw_columns={"payload": payload},
            warnings=(),
        )
        exporter = ScriptedExporter(sessions=[_session()], records=[scripted])
        redactor = DeterministicRedactor(RedactionMode.REDACTED)

        with create_normalized_snapshot(source, tmp_path, exporter, redactor.redact) as snapshot:
            replayed = next(NormalizedSnapshotAdapter(snapshot).iter_records("ses-1"))

        assert replayed.decoded_value == payload
        assert isinstance(replayed.decoded_value, dict)
        assert token_bytes not in repr(replayed.decoded_value).encode("utf-8")

    def test_source_snapshot_keeps_existing_cleanup_warning_positional_field(self, tmp_path: Path) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            stats = ExportStats(
                session_count=0,
                record_count=0,
                warning_count=0,
                complete=True,
                schema_generation="current",
            )
            snapshot = SourceSnapshot(
                "synthetic-fingerprint",
                tmp_path / "snapshot.db",
                SchemaDescription(generation="current", catalog_version="synthetic"),
                connection,
                stats,
                ["synthetic-cleanup-warning"],
            )
            assert snapshot.cleanup_warnings == ["synthetic-cleanup-warning"]
            assert snapshot.selection_evidence_enabled is False
        finally:
            connection.close()

    def test_callback_runs_before_temporary_persistence(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        record = _record("msg-1")
        # A secret-looking value inside the decoded payload and raw fields.
        record.canonical_record = CanonicalRecordEnvelope(
            decoded={"text": "keep-this-secret", "id": "msg-1"},
            raw_text='{"text": "keep-this-secret", "id": "msg-1"}',
            raw_columns={"seq": 1, "text": "keep-this-secret"},
            warnings=(),
        )
        record.raw_permitted_fields = {"seq": 1, "type": "user", "text": "keep-this-secret"}
        exporter = ScriptedExporter(sessions=[_session()], records=[record])

        with create_normalized_snapshot(source, tmp_path, exporter, masking_redaction) as snapshot:
            row = snapshot.connection.execute(
                "SELECT canonical_record, raw_permitted_fields FROM snapshot_records"
            ).fetchone()
            stored_record = json.loads(row[0])
            stored_fields = json.loads(row[1])

        # Identity-like keys in arbitrary decoded transcript content are not
        # structural envelope fields and therefore receive no exemption.
        assert stored_record["decoded"] == {"text": "[REDACTED]", "id": "[REDACTED]"}
        assert json.loads(stored_record["raw_text"]) == stored_record["decoded"]
        assert stored_record["raw_columns"] == {"seq": 1, "text": "[REDACTED]"}
        assert stored_fields == {"seq": 1, "type": "[REDACTED]", "text": "[REDACTED]"}
        # Typed structural identity fields survive masking; the raw-row
        # projection does not exempt an additive field named ``type``.
        assert stored_record["record"]["source_id"] == "msg-1"
        assert stored_record["record"]["sequence_value"] == 1

    def test_identity_callback_persists_verbatim_values(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(sessions=[_session()], records=[_record("msg-1")])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            row = snapshot.connection.execute("SELECT canonical_record FROM snapshot_records").fetchone()
        stored = json.loads(row[0])
        assert stored["decoded"] == {"content": [{"type": "text", "text": "synthetic prompt"}]}
        assert stored["raw_text"] == '{"content": [{"type": "text", "text": "synthetic prompt"}]}'

    def test_replay_returns_the_redacted_effective_values(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        record = _record("msg-1")
        record.canonical_record = CanonicalRecordEnvelope(
            decoded={"text": "keep-this-secret"},
            raw_text='{"text": "keep-this-secret"}',
            raw_columns={},
            warnings=(),
        )
        exporter = ScriptedExporter(sessions=[_session()], records=[record])
        with create_normalized_snapshot(source, tmp_path, exporter, masking_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            (replayed,) = list(adapter.iter_records("ses-1"))

        assert replayed.decoded_value == {"text": "[REDACTED]"}
        assert json.loads(replayed.raw_json) == replayed.decoded_value


class TestSnapshotSecretExclusion:
    def test_secret_table_rows_never_reach_the_snapshot(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = FixtureExporter()
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            text = _all_snapshot_text(snapshot.connection)
            for sentinel in SECRET_SENTINELS:
                assert sentinel not in text
            assert "account" not in _snapshot_tables(snapshot.connection)

    def test_snapshot_row_counts_reflect_only_permitted_rows(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = FixtureExporter()
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            sessions = snapshot.connection.execute("SELECT count(*) FROM snapshot_sessions").fetchone()[0]
            records = snapshot.connection.execute("SELECT count(*) FROM snapshot_records").fetchone()[0]

        assert sessions == 2
        assert records == 5


class TestReplayRoundTrip:
    def test_headers_round_trip_with_typed_fields(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        sessions = [_session("ses-1"), _session("ses-2")]
        exporter = ScriptedExporter(sessions=sessions, records=[])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            headers = list(adapter.iter_session_headers())

        assert [header.session_id for header in headers] == ["ses-1", "ses-2"]
        assert all(header.adapter_generation == "current" for header in headers)
        first = headers[0]
        assert first.directory == "/repo/project"
        assert first.title == "Synthetic session"
        assert first.time_created is not None
        assert first.time_created.value == 1_772_000_000_000
        assert first.time_created.unit == "ms"
        assert dict(first.raw_columns) == {"id": "ses-1", "directory": "/repo/project"}

    def test_records_round_trip_all_fields(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        scripted = _record("msg-1")
        exporter = ScriptedExporter(sessions=[_session()], records=[scripted])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            (replayed,) = list(adapter.iter_records("ses-1"))

        original = scripted.record
        assert replayed.source_table == original.source_table
        assert replayed.source_type == original.source_type
        assert replayed.source_id == original.source_id
        assert replayed.session_id == original.session_id
        assert replayed.sequence_domain == original.sequence_domain
        assert replayed.sequence_value == original.sequence_value
        assert replayed.timestamp == original.timestamp
        assert replayed.classification == original.classification
        assert replayed.decoded_value == original.decoded_value
        assert replayed.raw_json == original.raw_json
        assert replayed.warnings == original.warnings
        # Replayed raw columns are the union of the envelope's canonical raw
        # columns and the exporter's permitted raw fields.
        assert dict(replayed.raw_columns) == {"seq": 1, "type": "user"}

    def test_records_replay_in_written_order_with_fresh_iterators(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        scripted = [
            _record("msg-b", seq=2),
            _record("msg-a", seq=1),
            _record("msg-c", seq=3),
        ]
        exporter = ScriptedExporter(sessions=[_session()], records=scripted)
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            first_pass = list(adapter.iter_records("ses-1"))
            second_pass = list(adapter.iter_records("ses-1"))

        assert [record.source_id for record in first_pass] == ["msg-b", "msg-a", "msg-c"]
        assert second_pass == first_pass

    def test_records_are_scoped_by_session(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        scripted = [_record("msg-1", session_id="ses-1"), _record("msg-2", session_id="ses-2")]
        exporter = ScriptedExporter(sessions=[_session("ses-1"), _session("ses-2")], records=scripted)
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            assert [record.source_id for record in adapter.iter_records("ses-1")] == ["msg-1"]
            assert [record.source_id for record in adapter.iter_records("ses-2")] == ["msg-2"]
            assert list(adapter.iter_records("ses-missing")) == []

    def test_current_scoped_records_are_separated_from_turn_records(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        turn_record = _record("msg-1", table="session_message", source_type="user")
        scoped_record = _record(
            "pending-1",
            table="session_pending",
            source_type="tool",
            raw_json='{"name": "synthetic-tool"}',
            seq=None,
            classification=None,
        )
        exporter = ScriptedExporter(sessions=[_session()], records=[turn_record, scoped_record])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            turn_ids = [record.source_id for record in adapter.iter_records("ses-1")]
            scoped_ids = [record.source_id for record in adapter.iter_session_scoped_records("ses-1")]

        assert turn_ids == ["msg-1"]
        assert scoped_ids == ["pending-1"]

    def test_legacy_todo_records_are_session_scoped(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        part_record = _record(
            "part-1",
            table="part",
            source_type="text",
            raw_json='{"type": "text", "text": "synthetic part"}',
            seq=None,
            classification=None,
        )
        todo_record = _record(
            "todo-1",
            table="todo",
            source_type="todo",
            raw_json=None,
            seq=None,
            classification=None,
        )
        exporter = ScriptedExporter(
            generation="legacy",
            sessions=[_session(generation="legacy")],
            records=[part_record, todo_record],
        )
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            turn_ids = [record.source_id for record in adapter.iter_records("ses-1")]
            scoped_ids = [record.source_id for record in adapter.iter_session_scoped_records("ses-1")]

        assert turn_ids == ["part-1"]
        assert scoped_ids == ["todo-1"]

    def test_adapter_exposes_generation_and_inspect(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        exporter = ScriptedExporter(sessions=[_session()], records=[])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            assert adapter.schema_generation == "current"
            assert adapter.inspect() == snapshot.schema_description
            assert isinstance(adapter.inspect(), SchemaDescription)

    def test_legacy_generation_snapshot_round_trips(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        session = _session("ses-legacy", generation="legacy")
        exporter = ScriptedExporter(generation="legacy", sessions=[session], records=[])
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            headers = list(adapter.iter_session_headers())

        assert snapshot.stats.schema_generation == "legacy"
        assert headers[0].adapter_generation == "legacy"
        assert adapter.schema_generation == "legacy"

    def test_mixed_generation_snapshot_round_trips(self, tmp_path: Path) -> None:
        source = _build_current_fixture(tmp_path)
        current_record = _record("msg-current-1", table="session_message")
        legacy_record = _record(
            "part-legacy-1",
            table="part",
            source_type="text",
            raw_json='{"type": "text", "text": "synthetic part"}',
            seq=None,
        )
        exporter = ScriptedExporter(
            generation="mixed",
            sessions=[_session("ses-1", generation="current"), _session("ses-2", generation="legacy")],
            records=[current_record, legacy_record],
        )
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            assert adapter.schema_generation == "mixed"
            turn_ids = sorted(
                record.source_id for record in (*adapter.iter_records("ses-1"), *adapter.iter_records("ses-2"))
            )

        assert turn_ids == ["msg-current-1", "part-legacy-1"]


class TestFixtureEdgeValues:
    def test_edge_case_values_round_trip(self, tmp_path: Path) -> None:
        source = tmp_path / "edge.db"
        connection = build_fixture_db(source, "edge_cases")
        connection.close()
        exporter = FixtureExporter()
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            records = {record.source_id: record for record in adapter.iter_records("ses-edge-001")}
            (header,) = list(adapter.iter_session_headers())

        # Duplicate-key JSON keeps its raw text and warns.
        duplicate = records["msg-edge-002"]
        assert duplicate.warnings == ("duplicate_json_key",)
        assert duplicate.raw_json == '{"text": "first", "text": "second"}'

        # A BLOB holding valid UTF-8 JSON decodes normally.
        blob_json = records["msg-edge-003"]
        assert blob_json.decoded_value == {"content": [{"type": "text", "text": "synthetic blob json"}]}

        # A BLOB holding invalid UTF-8 warns and retains the bytes.
        invalid = records["msg-edge-004"]
        assert invalid.decoded_value is None
        assert invalid.warnings == ("malformed_source_value",)
        assert invalid.raw_columns.get("data") == b"\xff\xfe\x80"

        # NULL data stays absent; NULL ids survive as missing identity.
        null_data = records["msg-edge-005"]
        assert null_data.decoded_value is None
        assert null_data.raw_json is None
        assert null_data.sequence_value is None
        assert null_data.timestamp is None
        missing_id = [record for record in records.values() if record.source_id is None]
        assert len(missing_id) == 1

        # NULL session fields stay absent on the replayed header.
        assert header.session_id == "ses-edge-001"
        assert header.directory is None
        assert header.title is None
        assert header.time_created is None

    def test_edge_case_snapshot_stats_and_secret_warning(self, tmp_path: Path) -> None:
        source = tmp_path / "edge.db"
        connection = build_fixture_db(source, "edge_cases")
        connection.close()
        exporter = FixtureExporter()
        with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
            text = _all_snapshot_text(snapshot.connection)

        assert snapshot.stats.session_count == 1
        assert snapshot.stats.record_count == 6
        assert "SECRET-EDGE-SHARE-SENTINEL" not in text
        assert "session_share_omitted" in snapshot.schema_description.warnings


class TestReplayTurnProtocol:
    def test_protocol_is_structural(self) -> None:
        turn = _FakeReplayTurn()

        assert isinstance(turn, ReplayTurn)
        assert turn.session_header.session_id == "ses-1"
        assert turn.summary.turn_key == "turn-synthetic"
        assert list(turn.iter_records()) == []
        assert list(turn.iter_session_records()) == []
        assert list(turn.iter_unassigned_records()) == []
        assert turn.load_root_input(1024) is None
        assert turn.load_root_output(1024) is None

    def test_protocol_rejects_missing_surface(self) -> None:
        assert not isinstance(object(), ReplayTurn)


class _FakeReplayTurn:
    """Minimal structural ReplayTurn for protocol checks."""

    def __init__(self) -> None:
        self._header = SessionHeader(session_id="ses-1", adapter_generation="current")
        self._summary = TurnSummary(session_id="ses-1", turn_index=1, turn_key="turn-synthetic")

    @property
    def session_header(self) -> SessionHeader:
        return self._header

    @property
    def summary(self) -> TurnSummary:
        return self._summary

    def iter_records(self) -> Iterator[SourceRecord]:
        return iter(())

    def iter_session_records(self) -> Iterator[SourceRecord]:
        return iter(())

    def iter_unassigned_records(self) -> Iterator[SourceRecord]:
        return iter(())

    def load_root_input(self, max_bytes: int) -> object | None:
        return None

    def load_root_output(self, max_bytes: int) -> object | None:
        return None


class TestExporterProtocol:
    def test_source_exporter_protocol_is_structural(self) -> None:
        assert isinstance(ScriptedExporter(), SourceExporter)
        assert isinstance(FixtureExporter(), SourceExporter)
