import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from mlflow_import.source.candidates import candidate_session_ids
from mlflow_import.source.model import SchemaDescription
from mlflow_import.source.selection import SelectionFilters


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE session_v2 (id TEXT, time_created INTEGER, directory TEXT, path TEXT)")
    connection.executemany(
        "INSERT INTO session_v2 VALUES (?, ?, ?, ?)",
        (
            ("keep", 1_735_689_600_000, "/repo/app", "/repo/app"),
            ("other", 1_735_689_600_000, "/repo/other", "/repo/other"),
        ),
    )
    return connection


def test_candidate_discovery_uses_header_filters_only() -> None:
    connection = _connection()
    try:
        description = SchemaDescription(generation="current", catalog_version="test")
        filters = SelectionFilters(
            from_time=datetime(2025, 1, 1, tzinfo=UTC),
            to_time=datetime(2025, 12, 31, 23, 59, tzinfo=UTC),
            directory=Path("/repo/app"),
        )
        assert candidate_session_ids(connection, description, filters) == frozenset({"keep"})
    finally:
        connection.close()


def test_candidate_discovery_can_return_no_sessions_without_falling_back_to_all() -> None:
    connection = _connection()
    try:
        description = SchemaDescription(generation="current", catalog_version="test")
        filters = SelectionFilters(directory=Path("/missing"))
        assert candidate_session_ids(connection, description, filters) == frozenset()
    finally:
        connection.close()
