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


def test_explicit_session_stays_scoped_with_record_level_filters() -> None:
    connection = _connection()
    try:
        description = SchemaDescription(generation="current", catalog_version="test")
        assert candidate_session_ids(connection, description, SelectionFilters(min_user_turns=0), "keep") == frozenset(
            {"keep"}
        )
    finally:
        connection.close()


def test_candidate_discovery_includes_ancestors_and_descendants_of_a_match() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE session_v2 "
        "(id TEXT, time_created INTEGER, directory TEXT, path TEXT, parent_id TEXT, fork_session_id TEXT)"
    )
    connection.executemany(
        "INSERT INTO session_v2 VALUES (?, ?, ?, ?, ?, ?)",
        (
            ("grandparent", 1_735_689_600_000, "/repo/other", "/repo/other", None, None),
            ("parent", 1_735_689_600_000, "/repo/other", "/repo/other", "grandparent", None),
            ("match", 1_735_689_600_000, "/repo/app", "/repo/app", "parent", None),
            ("child", 1_735_689_600_000, "/repo/other", "/repo/other", "match", None),
            ("grandchild", 1_735_689_600_000, "/repo/other", "/repo/other", "child", None),
            ("sibling", 1_735_689_600_000, "/repo/other", "/repo/other", "parent", None),
        ),
    )
    try:
        selected = candidate_session_ids(
            connection,
            SchemaDescription(generation="current", catalog_version="test"),
            SelectionFilters(directory=Path("/repo/app")),
        )
        assert selected == frozenset({"grandparent", "parent", "match", "child", "grandchild"})
    finally:
        connection.close()


def test_candidate_discovery_handles_ancestry_cycles() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE session_v2 (id TEXT, directory TEXT, parent_id TEXT)")
    connection.executemany(
        "INSERT INTO session_v2 VALUES (?, ?, ?)",
        (("a", "/repo/app", "b"), ("b", "/repo/other", "a"), ("c", "/repo/other", "b")),
    )
    try:
        selected = candidate_session_ids(
            connection,
            SchemaDescription(generation="current", catalog_version="test"),
            SelectionFilters(directory=Path("/repo/app")),
        )
        assert selected == frozenset({"a", "b", "c"})
    finally:
        connection.close()
