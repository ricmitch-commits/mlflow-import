"""Legacy/mixed transcript sorts must fit the unchanged SQLite heap envelope."""

import json
import sqlite3

import pytest

from mlflow_import import resources
from mlflow_import.errors import SourceCompatibilityError
from mlflow_import.source.connection import open_live_connection
from mlflow_import.source.probe import probe_schema
from mlflow_import.source.registry import build_live_exporter
from mlflow_import.source.snapshot import ExportStats
from tests.support.import_fakes import CURRENT_DDL, LEGACY_DDL

pytestmark = pytest.mark.unit


class _CountingWriter:
    """Observe emitted payloads without retaining a transcript or writing a snapshot."""

    def __init__(self, payload, generation, connection):
        self.payload = payload
        self.generation = generation
        self.connection = connection
        self.sessions = self.records = 0
        self.payload_ids = []

    def _observe(self, values):
        assert self.connection.in_transaction
        if self.payload in (values.get("data"), values.get("metadata"), values.get("content")):
            self.payload_ids.append(values.get("id", values.get("position")))

    def write_session(self, header, envelope):
        self.sessions += 1
        self._observe(header.raw_columns)

    def write_record(self, record, order_key, envelope, raw_permitted_fields):
        self.records += 1
        self._observe(raw_permitted_fields)

    def finalize(self):
        return ExportStats(self.sessions, self.records, 0, True, self.generation)


def _source(path, kind, layout, generation, payload):
    ddl = LEGACY_DDL.replace("message_id TEXT NOT NULL", "message_id TEXT")
    if kind == "minimal-todo":
        ddl += "CREATE TABLE todo (id INTEGER PRIMARY KEY, session_id TEXT, content TEXT);"
    else:
        ddl += (
            "CREATE TABLE todo (session_id TEXT, position INTEGER, time_created INTEGER, content TEXT,"
            " PRIMARY KEY(session_id,position));"
        )
    keyed = set()
    if layout in {"keyed-session", "keyed-all"}:
        keyed.add("session")
    if layout in {"keyed-children", "keyed-all"}:
        keyed.update({"part", "todo"})
        if kind != "ambiguous-part":
            keyed.add("message")
    ddl = ";".join(
        statement
        + (" WITHOUT ROWID" if any(statement.strip().startswith(f"CREATE TABLE {t} (") for t in keyed) else "")
        for statement in ddl.split(";")
    )
    connection = sqlite3.connect(path)
    try:
        connection.executescript(ddl)
        if generation == "mixed":
            connection.executescript(CURRENT_DDL)
            connection.execute("INSERT INTO session_v2(id,time_created) VALUES('current-only',0)")
        for index in range(40 if kind == "session" else 1):
            connection.execute(
                "INSERT INTO session(id,time_created,metadata) VALUES(?,?,?)",
                (f"s-{index:02}", index, payload if kind == "session" else None),
            )
        if kind == "message":
            connection.executemany(
                "INSERT INTO message VALUES(?,'s-00',?,?,?)",
                ((f"r-{i:02}", i, i, payload) for i in reversed(range(40))),
            )
        elif kind in {"part", "ambiguous-part"}:
            parent = None if kind == "ambiguous-part" else "parent"
            connection.execute("INSERT INTO message VALUES(?,'s-00',0,0,?)", (parent, '{"role":"user"}'))
            if parent is None:
                connection.execute("INSERT INTO message VALUES(NULL,'s-00',1,1,?)", ('{"role":"user"}',))
            connection.executemany(
                "INSERT INTO part VALUES(?,?,'s-00',?,?,?)",
                ((f"r-{i:02}", parent, i, i, payload) for i in reversed(range(40))),
            )
        elif kind == "todo":
            connection.executemany(
                "INSERT INTO todo VALUES('s-00',?,?,?)", ((i, i, payload) for i in reversed(range(40)))
            )
        elif kind == "minimal-todo":
            connection.executemany("INSERT INTO todo VALUES(?,'s-00',?)", ((i, payload) for i in reversed(range(40))))
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize("generation", ["legacy", "mixed"])
@pytest.mark.parametrize("layout", ["rowid", "keyed-children", "keyed-session", "keyed-all"])
@pytest.mark.parametrize("kind", ["session", "message", "part", "ambiguous-part", "todo", "minimal-todo"])
def test_legacy_payload_ordering_exceeds_input_heap_without_sorting_payloads(tmp_path, generation, layout, kind):
    source = tmp_path / "large-legacy.db"
    payload = json.dumps({"text": "synthetic" + "x" * (2 * 1024 * 1024)})
    _source(source, kind, layout, generation, payload)
    assert len(payload.encode()) * 40 > resources.SQLITE_HEAP_BYTES
    try:
        with resources.temporary_storage(16 * 1024 * 1024, directory=tmp_path) as budget:
            with open_live_connection(source) as live:
                description = probe_schema(live.connection)
                assert description.generation == generation
                exporter = build_live_exporter(live.connection, description)
                assert live.connection.execute("PRAGMA temp_store").fetchone()[0] == 2
                assert 0 < live.connection.execute("PRAGMA hard_heap_limit").fetchone()[0] <= 64 * 1024 * 1024
                assert live.connection.execute("PRAGMA query_only").fetchone()[0] == 1
                writer = _CountingWriter(payload, generation, live.connection)
                stats = exporter.export_to_snapshot(writer, lambda value, protected: value)
                assert stats.complete and not budget.failed
                assert len(writer.payload_ids) == len(set(writer.payload_ids)) == 40
                if kind == "session":
                    assert writer.payload_ids == [f"s-{i:02}" for i in range(40)]
                elif kind in {"message", "part", "ambiguous-part"}:
                    assert writer.payload_ids == [f"r-{i:02}" for i in range(40)]
                elif kind == "todo":
                    assert writer.payload_ids == list(range(40))
                elif layout in {"keyed-children", "keyed-all"}:
                    # Stable declared-key fallback deliberately compares text.
                    assert writer.payload_ids == sorted(range(40), key=str)
                else:
                    # INTEGER PRIMARY KEY aliases rowid, so emission is ascending.
                    assert writer.payload_ids == list(range(40))
                assert stats.session_count == (40 if kind == "session" else 1) + (generation == "mixed")
                assert stats.record_count == {"session": 0, "part": 41, "ambiguous-part": 42}.get(kind, 40)
    finally:
        source.unlink()


def test_without_rowid_locator_cannot_read_an_excluded_credential_column(tmp_path):
    source = tmp_path / "excluded-key.db"
    connection = sqlite3.connect(source)
    try:
        connection.executescript(
            LEGACY_DDL
            + "CREATE TABLE todo(session_id TEXT, content TEXT, share_token TEXT PRIMARY KEY) WITHOUT ROWID;"
            + "INSERT INTO session(id,time_created) VALUES('s',0);"
            + "INSERT INTO todo VALUES('s','synthetic','excluded-credential');"
        )
    finally:
        connection.close()
    reads = []

    def authorize(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ:
            reads.append((table, column))
            if column == "share_token":
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    with resources.temporary_storage(16 * 1024 * 1024, directory=tmp_path):
        with open_live_connection(source) as live:
            live.connection.set_authorizer(authorize)
            exporter = build_live_exporter(live.connection, probe_schema(live.connection))
            with pytest.raises(SourceCompatibilityError, match="permitted stable row locator") as caught:
                exporter.begin_export()
            assert caught.value.code == "unsupported_schema"
            assert ("todo", "share_token") not in reads
