"""Unit tests for the source schema probe."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from opencode_mlflow.errors import SourceCompatibilityError
from opencode_mlflow.source.catalog import OPENCODE_SCHEMA_CATALOG_V1, SCHEMA_CATALOG_VERSION
from opencode_mlflow.source.model import SchemaDescription
from opencode_mlflow.source.probe import probe_schema
from tests.support.sqlite_builder import build_fixture_db

pytestmark = pytest.mark.unit


def _build(tmp_path: Path, name: str) -> sqlite3.Connection:
    connection = build_fixture_db(tmp_path / f"{name}.db", name)
    return connection


class TestGenerationClassification:
    def test_legacy_fixture_probes_as_legacy(self, tmp_path: Path) -> None:
        connection = _build(tmp_path, "legacy")
        try:
            description = probe_schema(connection)
        finally:
            connection.close()

        assert isinstance(description, SchemaDescription)
        assert description.generation == "legacy"
        assert description.catalog_version == SCHEMA_CATALOG_VERSION
        assert "session" in description.tables
        assert "message" in description.tables
        assert "part" in description.tables
        assert "session_v2" not in description.tables
        # legacy.sql includes a session_share table: its unknown layout is
        # omitted with a warning.
        assert description.warnings == ("session_share_omitted",)

    def test_current_fixture_probes_as_current(self, tmp_path: Path) -> None:
        connection = _build(tmp_path, "current")
        try:
            description = probe_schema(connection)
        finally:
            connection.close()

        assert description.generation == "current"
        assert "session_v2" in description.tables
        assert "session_message" in description.tables
        assert "session" not in description.tables
        # Session-scoped tables that exist are described too.
        assert "session_pending" in description.tables
        assert "instruction_blob" in description.tables
        assert description.warnings == ("session_share_omitted",)

    def test_mixed_fixture_probes_as_mixed(self, tmp_path: Path) -> None:
        connection = _build(tmp_path, "mixed")
        try:
            description = probe_schema(connection)
        finally:
            connection.close()

        assert description.generation == "mixed"
        for table in ("session", "message", "part", "session_v2", "session_message"):
            assert table in description.tables

    def test_edge_cases_probe_as_current_not_mixed(self, tmp_path: Path) -> None:
        # A legacy `message` table missing its required `data` column must
        # NOT make the database mixed; unknown session_v2 columns and absent
        # optional columns are tolerated.
        connection = _build(tmp_path, "edge_cases")
        try:
            description = probe_schema(connection)
        finally:
            connection.close()

        assert description.generation == "current"
        assert "message" not in description.tables
        observed = description.tables["session_v2"]
        assert "custom_flag" in observed
        assert "time_suspended" not in observed
        assert description.warnings == ("session_share_omitted",)

    def test_empty_database_is_unsupported(self, tmp_path: Path) -> None:
        connection = sqlite3.connect(tmp_path / "empty.db")
        try:
            with pytest.raises(SourceCompatibilityError) as excinfo:
                probe_schema(connection)
        finally:
            connection.close()

        assert excinfo.value.code == "unsupported_schema"
        assert excinfo.value.exit_status == 2
        assert "session" in str(excinfo.value)

    def test_missing_column_diagnostic_names_the_structure(self, tmp_path: Path) -> None:
        connection = sqlite3.connect(tmp_path / "malformed.db")
        try:
            connection.executescript(
                """
                CREATE TABLE session_v2 (
                    id TEXT PRIMARY KEY, project_id TEXT, workspace_id TEXT, parent_id TEXT,
                    fork_session_id TEXT, fork_boundary INTEGER, slug TEXT, directory TEXT,
                    path TEXT, title TEXT, version TEXT, share_url TEXT,
                    summary_additions INTEGER, summary_deletions INTEGER, summary_files INTEGER,
                    summary_diffs TEXT, metadata TEXT, cost REAL,
                    tokens_input INTEGER, tokens_output INTEGER, tokens_reasoning INTEGER,
                    tokens_cache_read INTEGER, tokens_cache_write INTEGER,
                    revert TEXT, permission TEXT, agent TEXT, model TEXT,
                    time_created INTEGER, time_updated INTEGER, time_idle INTEGER,
                    time_viewed INTEGER, idle_outcome TEXT, time_compacting INTEGER,
                    time_archived INTEGER
                );
                CREATE TABLE session_message (
                    id TEXT, session_id TEXT, type TEXT,
                    time_created INTEGER, time_updated INTEGER, data TEXT
                );
                """
            )
            connection.commit()
            with pytest.raises(SourceCompatibilityError) as excinfo:
                probe_schema(connection)
        finally:
            connection.close()

        message = str(excinfo.value)
        assert "session_message" in message
        assert "seq" in message
        # Safe diagnostics name structures, never source content.
        assert "CREATE TABLE" not in message

    def test_default_catalogs_argument_matches_explicit_catalogs(self, tmp_path: Path) -> None:
        connection = _build(tmp_path, "mixed")
        try:
            via_default = probe_schema(connection)
            via_explicit = probe_schema(connection, OPENCODE_SCHEMA_CATALOG_V1)
        finally:
            connection.close()

        assert via_default == via_explicit

    def test_legacy_session_scoped_tables_are_not_required_for_a_match(self, tmp_path: Path) -> None:
        connection = _build(tmp_path, "legacy")
        try:
            connection.execute("DROP TABLE todo")
            connection.commit()
            description = probe_schema(connection)
        finally:
            connection.close()

        assert description.generation == "legacy"
        assert "todo" not in description.tables


class TestProbeSafety:
    def test_probe_reads_only_metadata(self, tmp_path: Path) -> None:
        connection = _build(tmp_path, "mixed")
        statements: list[str] = []
        try:
            connection.set_trace_callback(statements.append)
            probe_schema(connection)
        finally:
            connection.close()

        assert statements
        for statement in statements:
            # sqlite's trace callback emits parameterized pragma calls as a
            # separate `-- PRAGMA table_info='...'` comment entry; skip
            # those — they are not SQL statements.
            if statement.strip().startswith("--"):
                continue
            sql = statement.strip().upper()
            assert sql.startswith("SELECT"), statement
            assert "SQLITE_MASTER" in sql or "PRAGMA_TABLE_INFO" in sql

    def test_diagnostics_never_echo_source_content(self, tmp_path: Path) -> None:
        connection = sqlite3.connect(tmp_path / "secret-unsupported.db")
        try:
            connection.executescript(
                """
                CREATE TABLE unrelated (value TEXT);
                INSERT INTO unrelated VALUES ('credential-sentinel');
                """
            )
            connection.commit()
            with pytest.raises(SourceCompatibilityError) as excinfo:
                probe_schema(connection)
        finally:
            connection.close()

        assert "credential-sentinel" not in str(excinfo.value)
        assert str(tmp_path / "secret-unsupported.db") not in str(excinfo.value)

    def test_observed_columns_match_the_fixture_ddl(self, tmp_path: Path) -> None:
        connection = _build(tmp_path, "legacy")
        try:
            description = probe_schema(connection)
        finally:
            connection.close()

        assert list(description.tables["message"]) == [
            "id",
            "session_id",
            "time_created",
            "time_updated",
            "data",
        ]
