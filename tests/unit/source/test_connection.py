"""Unit tests for the read-only live source connection."""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pytest

from opencode_mlflow.errors import SnapshotError
from opencode_mlflow.source.connection import LiveReadOnlyConnection, open_live_connection
from tests.support.sqlite_builder import build_sqlite_db

pytestmark = pytest.mark.unit

_WRITE_STATEMENTS = ("insert", "update", "delete", "create", "drop", "alter", "replace", "vacuum", "attach", "detach")


def _make_database(path: Path) -> Path:
    connection = build_sqlite_db(path, "CREATE TABLE session (id TEXT); INSERT INTO session VALUES ('ses-1');")
    connection.close()
    return path


class TestPathValidation:
    def test_missing_path_fails_safely(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotError) as excinfo:
            with open_live_connection(tmp_path / "missing.db"):
                pass

        assert excinfo.value.code == "source_unavailable"
        assert excinfo.value.exit_status == 2
        assert str(tmp_path / "missing.db") not in str(excinfo.value)

    def test_directory_path_fails_safely(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotError) as excinfo:
            with open_live_connection(tmp_path):
                pass

        assert excinfo.value.code == "source_unavailable"
        assert str(tmp_path) not in str(excinfo.value)

    def test_non_sqlite_file_fails_safely(self, tmp_path: Path) -> None:
        source = tmp_path / "garbage.db"
        source.write_bytes(b"this is definitely not a sqlite database")

        with pytest.raises(SnapshotError) as excinfo:
            with open_live_connection(source):
                pass

        assert excinfo.value.code == "source_unavailable"
        assert str(source) not in str(excinfo.value)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_unreadable_file_fails_safely(self, tmp_path: Path) -> None:
        source = _make_database(tmp_path / "unreadable.db")
        source.chmod(0o200)
        try:
            with pytest.raises(SnapshotError) as excinfo:
                with open_live_connection(source):
                    pass
        finally:
            source.chmod(0o600)

        assert excinfo.value.code == "source_unavailable"
        assert str(source) not in str(excinfo.value)


class TestReadOnlySafety:
    def test_uri_contains_only_the_normalized_path_and_mode_ro(self, tmp_path: Path) -> None:
        source = _make_database(tmp_path / "source.db")

        with open_live_connection(source) as live:
            assert live.uri.startswith("file://")
            assert live.uri.endswith("?mode=ro")
            assert "immutable" not in live.uri
            assert str(source) not in live.uri or live.uri.startswith("file://")

    def test_query_only_is_enabled(self, tmp_path: Path) -> None:
        source = _make_database(tmp_path / "source.db")

        with open_live_connection(source) as live:
            assert live.connection.execute("PRAGMA query_only").fetchone()[0] == 1

    def test_writes_are_rejected(self, tmp_path: Path) -> None:
        source = _make_database(tmp_path / "source.db")

        with open_live_connection(source) as live:
            with pytest.raises(sqlite3.Error):
                live.connection.execute("INSERT INTO session VALUES ('ses-2')")
            with pytest.raises(sqlite3.Error):
                live.connection.execute("CREATE TABLE forbidden (x)")

    def test_busy_timeout_is_bounded_and_configurable(self, tmp_path: Path) -> None:
        source = _make_database(tmp_path / "source.db")

        with open_live_connection(source) as live:
            assert live.busy_timeout_ms == 5000
            assert live.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

        with open_live_connection(source, busy_timeout_ms=250) as live:
            assert live.busy_timeout_ms == 250
            assert live.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 250

    def test_exactly_one_deferred_begin_is_issued(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _make_database(tmp_path / "source.db")
        statements: list[str] = []
        original_connect = sqlite3.connect

        def tracing_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            connection = original_connect(*args, **kwargs)  # type: ignore[arg-type]
            connection.set_trace_callback(statements.append)
            return connection

        monkeypatch.setattr(sqlite3, "connect", tracing_connect)
        with open_live_connection(source) as live:
            rows = live.connection.execute("SELECT id FROM session").fetchall()
            assert rows == [("ses-1",)]
            assert live.connection.in_transaction is True

        begins = [statement for statement in statements if statement.strip().upper().startswith("BEGIN")]
        assert begins == ["BEGIN"]
        assert not any("IMMEDIATE" in statement.upper() for statement in statements)
        assert not any("EXCLUSIVE" in statement.upper() for statement in statements)

    def test_no_write_statements_are_ever_issued(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _make_database(tmp_path / "source.db")
        statements: list[str] = []
        original_connect = sqlite3.connect

        def tracing_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            connection = original_connect(*args, **kwargs)  # type: ignore[arg-type]
            connection.set_trace_callback(statements.append)
            return connection

        monkeypatch.setattr(sqlite3, "connect", tracing_connect)
        with open_live_connection(source) as live:
            live.connection.execute("SELECT id FROM session").fetchall()

        assert statements
        for statement in statements:
            head = statement.strip().upper().split(None, 1)[0].rstrip(";(")
            assert head in {"SELECT", "PRAGMA", "BEGIN"}, statement
            assert head not in _WRITE_STATEMENTS, statement

    def test_connection_is_closed_on_exit(self, tmp_path: Path) -> None:
        source = _make_database(tmp_path / "source.db")

        with open_live_connection(source) as live:
            connection = live.connection

        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_committed_rows_are_readable(self, tmp_path: Path) -> None:
        source = _make_database(tmp_path / "source.db")

        with open_live_connection(source) as live:
            rows = live.connection.execute("SELECT id FROM session").fetchall()

        assert rows == [("ses-1",)]

    def test_live_connection_reports_regular_file_metadata(self, tmp_path: Path) -> None:
        source = _make_database(tmp_path / "source.db")

        with open_live_connection(source) as live:
            assert stat.S_ISREG(source.stat().st_mode)
            assert isinstance(live, LiveReadOnlyConnection)


class TestBusyTimeoutFailure:
    def test_busy_source_maps_to_source_snapshot_failed(self, tmp_path: Path) -> None:
        source = _make_database(tmp_path / "locked.db")
        writer = sqlite3.connect(source, isolation_level=None)
        try:
            writer.execute("BEGIN EXCLUSIVE")
            with pytest.raises(SnapshotError) as excinfo:
                with open_live_connection(source, busy_timeout_ms=200):
                    pass
        finally:
            writer.execute("ROLLBACK")
            writer.close()

        assert excinfo.value.code == "source_snapshot_failed"
        assert excinfo.value.exit_status == 2
        assert str(source) not in str(excinfo.value)
