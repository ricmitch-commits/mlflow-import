"""A source larger than the SQLite heap must sort locators, never transcript bodies."""

import json
import sqlite3

import pytest

from mlflow_import import resources
from mlflow_import.source.connection import open_live_connection
from mlflow_import.source.probe import probe_schema
from mlflow_import.source.registry import build_live_exporter
from tests.support.import_fakes import CURRENT_DDL

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("layout", ["rowid", "message-primary-key", "session-primary-key"])
def test_current_joined_export_orders_source_larger_than_heap_by_locators(tmp_path, layout):
    source = tmp_path / "large-source.db"
    payload = json.dumps({"text": "synthetic" + "x" * (2 * 1024 * 1024)})
    writer = sqlite3.connect(source)
    try:
        ddl = CURRENT_DDL
        if layout == "message-primary-key":
            ddl = ddl.replace("    data TEXT\n);", "    data TEXT\n) WITHOUT ROWID;")
        elif layout == "session-primary-key":
            ddl = ddl.replace("    resume_attempts INTEGER\n);", "    resume_attempts INTEGER\n) WITHOUT ROWID;")
        writer.executescript(ddl)
        writer.execute("CREATE INDEX message_session ON session_message(session_id,seq)")
        for session in range(2):
            native = f"synthetic-{session}"
            writer.execute(
                "INSERT INTO session_v2(id,time_created,time_updated) VALUES(?,?,?)", (native, 1000 + session, 2000)
            )
            writer.executemany(
                "INSERT INTO session_message VALUES(?,?,?,?,?,?,?)",
                (
                    (
                        f"message-{session}-{index}",
                        native,
                        "user" if index == 0 else "assistant",
                        index,
                        1000 + index,
                        1001 + index,
                        payload,
                    )
                    for index in range(20)
                ),
            )
        writer.commit()
    finally:
        writer.close()
    assert len(payload.encode()) * 40 > resources.SQLITE_HEAP_BYTES
    with resources.temporary_storage(16 * 1024 * 1024, directory=tmp_path) as budget:
        with open_live_connection(source) as live:
            exporter = build_live_exporter(live.connection, probe_schema(live.connection))
            assert live.connection.execute("PRAGMA temp_store").fetchone()[0] == 2
            assert 0 < live.connection.execute("PRAGMA hard_heap_limit").fetchone()[0] <= 64 * 1024 * 1024
            exporter.begin_export()
            count = 0
            for _, header, key in exporter.iter_exportable_sessions():
                for row in exporter._message_rows(header["id"], key):
                    assert row["data"] == payload
                    count += 1
            assert count == 40
            assert not budget.failed
    assert source.exists()
