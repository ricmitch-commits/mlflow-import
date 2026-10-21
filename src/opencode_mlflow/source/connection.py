"""Read-only live connections to the OpenCode source database.

Source safety is enforced here, in one place, for every live read:

- the path is validated to be a regular, readable SQLite file before any
  connection is opened (failures raise :class:`SnapshotError` with safe
  messages that never echo the path);
- the database is opened through a URI containing only the normalized path
  and ``mode=ro`` (``uri=True``); ``immutable=1`` is never used, because
  immutable connections can ignore WAL contents, and the ``-wal``/``-shm``
  companions are left in place so committed WAL rows stay readable;
- a bounded busy timeout is applied and ``PRAGMA query_only=ON`` rejects
  every write, schema change, checkpoint, vacuum, and attach through the
  connection;
- exactly one deferred ``BEGIN`` is issued — never ``BEGIN IMMEDIATE`` or
  ``BEGIN EXCLUSIVE`` — so the read transaction shares one consistent view
  with the extraction that follows;
- the connection is closed when the context exits, and the caller must
  close it immediately after extraction anyway (the snapshot driver does).
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from opencode_mlflow.errors import SnapshotError

_DEFAULT_BUSY_TIMEOUT_MS = 5000

_SOURCE_MISSING = "the source database path does not exist"
_SOURCE_NOT_REGULAR = "the source database path is not a regular file"
_SOURCE_NOT_READABLE = "the source database path is not readable"
_SOURCE_NOT_DATABASE = "the source path is not a readable SQLite database"
_SOURCE_BUSY = "the source database remained busy beyond the configured timeout"
_SOURCE_OPEN_FAILED = "the source database could not be opened for a read-only snapshot"


@dataclass(frozen=True)
class LiveReadOnlyConnection:
    """A live source connection restricted to one read-only transaction.

    ``uri`` is the read-only URI the connection was opened with (only the
    normalized path plus ``mode=ro``). The underlying ``connection`` is
    already inside the single deferred read transaction; the context
    manager that produced this object closes it on exit.
    """

    connection: sqlite3.Connection
    busy_timeout_ms: int
    uri: str


@contextmanager
def open_live_connection(
    path: Path, busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS
) -> Iterator[LiveReadOnlyConnection]:
    """Open the source database read-only in one deferred read transaction.

    Path validation and open failures raise :class:`SnapshotError` with
    code ``source_unavailable``; a source that stays busy beyond the
    bounded timeout raises with code ``source_snapshot_failed``. Both exit
    with status ``2`` and safe messages that never echo the path.
    """
    if busy_timeout_ms < 0:
        raise ValueError("busy_timeout_ms must be non-negative")
    _validate_source_path(path)
    uri = _read_only_uri(path)
    try:
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
    except sqlite3.Error as error:
        raise SnapshotError("source_unavailable", _SOURCE_OPEN_FAILED) from error
    try:
        connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        connection.execute("PRAGMA query_only = ON")
        try:
            # First file access: validates the header is a SQLite database
            # (an empty file is a valid empty database) and applies the
            # bounded busy timeout when the file is lock-protected.
            connection.execute("PRAGMA schema_version")
        except sqlite3.OperationalError as error:
            raise SnapshotError("source_snapshot_failed", _SOURCE_BUSY) from error
        except sqlite3.DatabaseError as error:
            raise SnapshotError("source_unavailable", _SOURCE_NOT_DATABASE) from error
        connection.execute("BEGIN")
    except BaseException:
        connection.close()
        raise
    try:
        yield LiveReadOnlyConnection(connection=connection, busy_timeout_ms=busy_timeout_ms, uri=uri)
    finally:
        connection.close()


def _validate_source_path(path: Path) -> None:
    """Fail safely unless ``path`` is a regular, readable file."""
    if not path.exists():
        raise SnapshotError("source_unavailable", _SOURCE_MISSING)
    if not path.is_file():
        raise SnapshotError("source_unavailable", _SOURCE_NOT_REGULAR)
    if not os.access(path, os.R_OK):
        raise SnapshotError("source_unavailable", _SOURCE_NOT_READABLE)


def _read_only_uri(path: Path) -> str:
    """Build a read-only URI from only the normalized path and ``mode=ro``.

    ``Path.as_uri`` absolutizes and percent-encodes the path. No other
    URI parameters are added — in particular never ``immutable=1``.
    """
    normalized = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    return normalized.as_uri() + "?mode=ro"
