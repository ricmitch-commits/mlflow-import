"""Helpers for building temporary SQLite databases in tests.

Keeping database construction in code (rather than committed binary
fixtures) lets later tasks generate snapshots deterministically and
extend this seam with WAL and fixture-file helpers.

``tests/fixtures/sqlite/*.sql`` hold sanitized synthetic DDL+data (no real
IDs, prompts, or credentials). :func:`fixture_script` reads one of those
files; :func:`build_fixture_db` materializes it as a temporary database.

:func:`build_wal_db` creates a WAL-mode database whose committed rows
live in the ``-wal`` file: the returned writer connection stays open so
the committed-WAL rows (and the ``-wal``/``-shm`` companions) remain
available to a concurrent read-only snapshot. The caller owns the
connection and must close it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "sqlite"


def build_sqlite_db(path: Path, script: str) -> sqlite3.Connection:
    """Create a SQLite database at ``path`` by executing ``script``.

    The caller owns the returned connection and is responsible for closing
    it.
    """
    connection = sqlite3.connect(path)
    try:
        connection.executescript(script)
    except BaseException:
        connection.close()
        raise
    return connection


def fixture_script(name: str) -> str:
    """Read the sanitized synthetic fixture ``tests/fixtures/sqlite/<name>.sql``."""
    return (FIXTURES_DIR / f"{name}.sql").read_text(encoding="utf-8")


def build_fixture_db(path: Path, name: str) -> sqlite3.Connection:
    """Create a temporary database from the named fixture script.

    The caller owns the returned connection and is responsible for closing
    it.
    """
    return build_sqlite_db(path, fixture_script(name))


def build_wal_db(path: Path, script: str) -> sqlite3.Connection:
    """Create a WAL-mode database at ``path`` and commit ``script`` into the WAL.

    WAL is enabled *before* any data is inserted, so every committed row
    lives in the ``-wal`` file rather than the main database file. The
    writer connection is returned still open: committed-WAL rows (and the
    ``-wal``/``-shm`` companions) remain available only while at least one
    connection is open, so the caller must keep the returned connection
    open until the test no longer needs the WAL state, and must close it.
    """
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(script)
    except BaseException:
        connection.close()
        raise
    return connection


def build_wal_fixture_db(path: Path, name: str) -> sqlite3.Connection:
    """Create a WAL-mode database from the named fixture script.

    See :func:`build_wal_db` for the returned connection's ownership and
    the committed-WAL guarantee.
    """
    return build_wal_db(path, fixture_script(name))
