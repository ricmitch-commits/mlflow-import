"""Integration test: WAL-aware normalized snapshot against a live writer.

This module is marked ``integration`` and runs only when explicitly
selected (``pytest tests/integration/test_wal_snapshot.py -m integration``).
It builds a WAL-mode SQLite database whose committed rows live in the
``-wal`` file, keeps a writer connection open (so the ``-wal``/``-shm``
companions stay alive), and proves the importer's snapshot contract:

- rows committed before normalized extraction are present in the snapshot;
- rows committed after extraction are absent from it;
- rows committed by ANOTHER connection WHILE the importer is mid-extraction
  (its read cursor open) are absent too: the single deferred read
  transaction pins a consistent snapshot at its first read, so a
  concurrent WAL commit never tears the extraction — while a fresh direct
  reader of the live source does see the committed row, proving the
  invisibility is read consistency, not a failed write;
- discovery and replay read the same normalized snapshot (never the live
  source, which is closed immediately after extraction), and repeated
  discovery/replay passes over the open snapshot return identical results;
- the source, ``-wal``, and ``-shm`` files are never rewritten, deleted,
  or checkpointed by the importer (the main database and WAL bytes are
  unchanged by the importer's read; the ``-shm`` wal-index legitimately
  changes bytes through SQLite's shared read-mark protocol on any read,
  so its size, inode, and existence are asserted instead);
- the importer never issues a source write or checkpoint (statement trace
  plus ``query_only``);
- the temporary snapshot contains no prohibited secret-table row/value;
- temporary snapshot files are cleaned up.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from opencode_mlflow.source.replay import NormalizedSnapshotAdapter
from opencode_mlflow.source.snapshot import create_normalized_snapshot
from tests.support.fake_source import FixtureExporter, identity_redaction
from tests.support.sqlite_builder import build_wal_fixture_db

pytestmark = pytest.mark.integration

SECRET_SENTINELS = (
    "credential-sentinel",
    "SECRET-ACCOUNT-TOKEN-SENTINEL",
    "SECRET-ACCOUNT-STATE-SENTINEL",
    "SECRET-CONTROL-ACCOUNT-SENTINEL",
    "SECRET-SHARE-SENTINEL",
)

_LATE_MESSAGE_ID = "msg-current-late"
_LATE_INSERT = (
    "INSERT INTO session_message VALUES ("
    f"'{_LATE_MESSAGE_ID}', 'ses-current-001', 'user', 9, 1772000090000, 1772000090000, "
    '\'{"content": [{"type": "text", "text": "synthetic late prompt"}]}\''
    ")"
)

_MID_SCAN_MESSAGE_ID = "msg-current-mid-scan"
_MID_SCAN_INSERT = (
    "INSERT INTO session_message VALUES ("
    f"'{_MID_SCAN_MESSAGE_ID}', 'ses-current-001', 'user', 5, 1772000045000, 1772000045000, "
    '\'{"content": [{"type": "text", "text": "synthetic mid-scan prompt"}]}\''
    ")"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_state(path: Path) -> tuple[str, int, int]:
    info = path.stat()
    return (_digest(path), info.st_size, info.st_ino)


def _source_state(source: Path) -> dict[str, tuple[str, int, int] | None]:
    state: dict[str, tuple[str, int, int] | None] = {}
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(source) + suffix)
        state[suffix or "db"] = _file_state(candidate) if candidate.exists() else None
    return state


def _all_snapshot_text(connection: sqlite3.Connection) -> str:
    parts: list[str] = []
    for table in ("snapshot_sessions", "snapshot_records"):
        for row in connection.execute(f"SELECT * FROM {table}"):
            parts.extend(str(value) for value in row)
    return "\n".join(parts)


def _message_ids(connection: sqlite3.Connection) -> set[str | None]:
    rows = connection.execute("SELECT source_id FROM snapshot_records WHERE source_table = 'session_message'")
    return {str(row[0]) for row in rows}


@pytest.fixture
def wal_source(restrictive_tmp_path: Path) -> Path:
    """A WAL-mode fixture database whose writer connection stays open."""
    source = restrictive_tmp_path / "wal-source.db"
    writer = build_wal_fixture_db(source, "current")
    try:
        assert Path(str(source) + "-wal").exists()
        assert Path(str(source) + "-shm").exists()
        yield source
    finally:
        writer.close()


class _MidScanCommitConnection:
    """Connection proxy that commits through ANOTHER connection mid-scan.

    Wraps the live connection the snapshot driver binds into the exporter.
    Once the exporter's ordered ``session_message`` SELECT has yielded
    ``trigger_after`` rows — with the reader's cursor still open and its
    deferred read transaction long established — a separate writer
    connection commits one more row into the WAL, and the scan continues
    over the remaining rows.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        source: Path,
        trigger_after: int,
        insert_sql: str,
        state: dict[str, bool],
    ) -> None:
        self._connection = connection
        self._source = source
        self._trigger_after = trigger_after
        self._insert_sql = insert_sql
        self._state = state

    def execute(self, sql: str, parameters: object = ()) -> object:
        cursor = self._connection.execute(sql, parameters)
        if "FROM session_message" in sql:
            return self._triggering(cursor)
        return cursor

    def _triggering(self, cursor: object) -> object:
        yielded = 0
        for row in cursor:  # type: ignore[union-attr]
            yielded += 1
            if yielded == self._trigger_after and not self._state["committed"]:
                self._commit_mid_scan()
            yield row

    def _commit_mid_scan(self) -> None:
        writer = sqlite3.connect(self._source, isolation_level=None)
        try:
            writer.execute(self._insert_sql)
        finally:
            writer.close()
        self._state["committed"] = True

    def __getattr__(self, name: str) -> object:
        return getattr(self._connection, name)


class _MidScanCommitExporter(FixtureExporter):
    """``FixtureExporter`` whose message scan commits mid-extraction through another connection."""

    def __init__(self, source: Path, trigger_after: int, insert_sql: str) -> None:
        super().__init__()
        self._mid_scan_source = source
        self._trigger_after = trigger_after
        self._insert_sql = insert_sql
        self.state: dict[str, bool] = {"committed": False}

    def bind_connection(self, connection: sqlite3.Connection) -> None:
        super().bind_connection(
            _MidScanCommitConnection(
                connection,
                self._mid_scan_source,
                self._trigger_after,
                self._insert_sql,
                self.state,
            )
        )


def test_wal_snapshot_includes_committed_rows_and_excludes_later_commits(wal_source: Path) -> None:
    exporter = FixtureExporter()
    with create_normalized_snapshot(wal_source, wal_source.parent, exporter, identity_redaction) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)

        # Discovery pass: committed-WAL rows are visible.
        headers = list(adapter.iter_session_headers())
        assert [header.session_id for header in headers] == ["ses-current-001", "ses-current-002"]

        # Replay pass over the SAME snapshot: rows committed before
        # extraction are present.
        first_ids = _message_ids(snapshot.connection)
        replay_ids = {record.source_id for record in adapter.iter_records("ses-current-001")}
        assert "msg-current-001" in first_ids
        assert "msg-current-001" in replay_ids

        # A concurrent writer commits AFTER extraction: the new row must be
        # absent from both discovery and replay views of the snapshot.
        writer = sqlite3.connect(wal_source, isolation_level=None)
        try:
            writer.execute(_LATE_INSERT)
        finally:
            writer.close()

        second_headers = list(adapter.iter_session_headers())
        second_ids = _message_ids(snapshot.connection)
        second_replay = {record.source_id for record in adapter.iter_records("ses-current-001")}
        assert _LATE_MESSAGE_ID not in second_ids
        assert _LATE_MESSAGE_ID not in second_replay
        assert [header.session_id for header in second_headers] == [header.session_id for header in headers]
        # Discovery and replay observe the same normalized snapshot.
        assert second_ids == first_ids


def test_wal_snapshot_leaves_source_wal_and_shm_untouched(wal_source: Path) -> None:
    before = _source_state(wal_source)
    exporter = FixtureExporter()
    with create_normalized_snapshot(wal_source, wal_source.parent, exporter, identity_redaction) as snapshot:
        # The importer's read must not rewrite the main database (no
        # checkpoint), append to or truncate the WAL, or delete/recreate
        # any companion. The -shm wal-index legitimately changes bytes on
        # any SQLite read (read-mark protocol), so only its size, inode,
        # and existence are stable invariants.
        after_extraction = _source_state(wal_source)
        assert after_extraction["db"] == before["db"]
        assert after_extraction["-wal"] == before["-wal"]
        assert after_extraction["-shm"] is not None
        assert after_extraction["-shm"][1] == before["-shm"][1]
        assert after_extraction["-shm"][2] == before["-shm"][2]
        assert snapshot.stats.record_count == 5

    # After the whole snapshot lifecycle (including cleanup), the source
    # files are still present and the main database was never rewritten.
    final = _source_state(wal_source)
    assert final["db"] == before["db"]
    assert final["-wal"] is not None
    assert final["-shm"] is not None


def test_wal_snapshot_never_issues_source_writes_or_checkpoints(wal_source: Path) -> None:
    statements: list[str] = []
    exporter = FixtureExporter(statements=statements)
    with create_normalized_snapshot(wal_source, wal_source.parent, exporter, identity_redaction):
        pass

    assert statements, "the exporter must have read through the bound live connection"
    write_verbs = {
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "DROP",
        "ALTER",
        "REPLACE",
        "VACUUM",
        "ATTACH",
        "DETACH",
        "TRUNCATE",
        "REINDEX",
    }
    for statement in statements:
        # sqlite's trace callback emits parameterized pragma calls as a
        # separate `-- PRAGMA table_info='...'` comment entry; skip those
        # — they are not SQL statements.
        if statement.strip().startswith("--"):
            continue
        normalized = statement.strip().upper()
        assert normalized, statement
        # A statement is safe when its leading verb is a read verb and not
        # a write verb (matching on the first token avoids false positives
        # such as the column name `time_created`).
        head = normalized.split(None, 1)[0].rstrip(";(")
        assert head in {"SELECT", "PRAGMA", "BEGIN"}, statement
        assert head not in write_verbs, statement
        assert "WAL_CHECKPOINT" not in normalized, statement
    # The driver's single deferred BEGIN and its read-only pragmas are
    # pinned by tests/unit/source/test_connection.py; query_only plus the
    # byte-hash test above guarantee no write or checkpoint occurred.


def test_wal_snapshot_excludes_secret_values_and_cleans_up(wal_source: Path) -> None:
    exporter = FixtureExporter()
    with create_normalized_snapshot(wal_source, wal_source.parent, exporter, identity_redaction) as snapshot:
        snapshot_path = snapshot.snapshot_path
        text = _all_snapshot_text(snapshot.connection)
        for sentinel in SECRET_SENTINELS:
            assert sentinel not in text
        # The live source connection is already closed during discovery.
        assert exporter.connection is not None
        with pytest.raises(sqlite3.ProgrammingError):
            exporter.connection.execute("SELECT 1")

    assert not snapshot_path.exists()
    assert not snapshot_path.parent.exists()


def test_wal_snapshot_ignores_a_commit_that_lands_during_extraction(wal_source: Path) -> None:
    """A WAL commit landing WHILE the importer reads must be invisible to it.

    Another SQLite connection commits into the WAL mid-extraction (the
    reader's message cursor is open and its deferred read transaction long
    established), so the single consistent read snapshot must exclude the
    new row — while a fresh direct reader of the live source DOES see it,
    proving the row really committed and its invisibility is the
    snapshot's read consistency, not a failed write.
    """
    before = _source_state(wal_source)
    exporter = _MidScanCommitExporter(wal_source, trigger_after=2, insert_sql=_MID_SCAN_INSERT)
    with create_normalized_snapshot(wal_source, wal_source.parent, exporter, identity_redaction) as snapshot:
        # The concurrent commit really fired while the reader's cursor was open.
        assert exporter.state["committed"] is True
        adapter = NormalizedSnapshotAdapter(snapshot)

        # Discovery and replay over the same snapshot: the mid-scan row is
        # absent from both views, and repeated passes are stable.
        first_headers = [header.session_id for header in adapter.iter_session_headers()]
        first_ids = _message_ids(snapshot.connection)
        first_replay = {record.source_id for record in adapter.iter_records("ses-current-001")}
        assert _MID_SCAN_MESSAGE_ID not in first_ids
        assert _MID_SCAN_MESSAGE_ID not in first_replay
        assert [header.session_id for header in adapter.iter_session_headers()] == first_headers
        assert _message_ids(snapshot.connection) == first_ids
        assert {record.source_id for record in adapter.iter_records("ses-current-001")} == first_replay

        # Every row committed BEFORE extraction is present — exactly the
        # fixture's five message rows, no more and no fewer.
        assert snapshot.stats.record_count == 5
        assert _LATE_MESSAGE_ID not in first_ids
        assert first_ids == {
            "msg-current-001",
            "msg-current-002",
            "msg-current-003",
            "msg-current-004",
            "msg-current-101",
        }

    # The mid-scan commit is durable in the live source: a fresh direct
    # read-only connection sees the row.
    direct = sqlite3.connect(f"file:{wal_source}?mode=ro", uri=True)
    try:
        seen = direct.execute("SELECT COUNT(*) FROM session_message WHERE id = ?", (_MID_SCAN_MESSAGE_ID,)).fetchone()
    finally:
        direct.close()
    assert seen == (1,)

    # The importer never rewrote the main database. The concurrent writer's
    # own commit may legitimately have grown the ``-wal`` file (and the
    # ``-shm`` wal-index changes bytes on any read through the read-mark
    # protocol), so only the main database bytes, existence, and monotonic
    # WAL growth are asserted here.
    after = _source_state(wal_source)
    assert after["db"] == before["db"]
    assert after["-wal"] is not None
    assert after["-wal"][1] >= before["-wal"][1]
    assert after["-shm"] is not None
