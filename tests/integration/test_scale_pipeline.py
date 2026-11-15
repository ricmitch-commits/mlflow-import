"""Synthetic reference-sized end-to-end benchmark; never reads project data."""

import json
import platform
import sqlite3
import time
from contextlib import closing

import pytest

from mlflow_import.config import ImportConfig
from mlflow_import.importer import build_dependencies, exit_status, run_import
from mlflow_import.mlflow.gateway import MlflowGateway
from tests.integration.server import unique_experiment_name
from tests.support.import_fakes import CURRENT_DDL
from tests.support.scale_assertions import assert_imported_workload

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_reference_synthetic_fresh_unchanged_append_edit(live_server, tmp_path, monkeypatch):
    source = tmp_path / "reference-synthetic.db"
    messages = parts = users = byte_count = 0
    with closing(sqlite3.connect(source)) as connection, connection:
        connection.executescript(CURRENT_DDL)
        connection.execute("CREATE INDEX message_session ON session_message(session_id,seq)")
        for session_index in range(307):
            session_id = f"synthetic-session-{session_index:03}"
            connection.execute(
                "INSERT INTO session_v2(id,parent_id,time_created,time_updated) VALUES(?,?,?,?)",
                (
                    session_id,
                    "synthetic-session-000" if session_index in {1, 2} else None,
                    1700000000000 + session_index,
                    1700001000000 + session_index,
                ),
            )
            count = 27 if session_index < 152 else 26
            for ordinal in range(count):
                role = "user" if ordinal == 0 or (session_index < 187 and ordinal == 13) else "assistant"
                content = []
                for part_index in range(6 if messages < 7306 else 5):
                    length = 6600
                    if messages == 0 and part_index == 0:
                        length = 1_206_000
                    elif messages == 0 and part_index == 1:
                        length = 800_000
                    content.append({"type": "text", "text": ("synthetic fixture text " * (length // 23 + 1))[:length]})
                payload = json.dumps({"content": content}, separators=(",", ":"))
                connection.execute(
                    "INSERT INTO session_message VALUES(?,?,?,?,?,?,?)",
                    (
                        f"synthetic-message-{messages:05}",
                        session_id,
                        role,
                        ordinal,
                        1700000000000 + messages,
                        1700000000001 + messages,
                        payload,
                    ),
                )
                messages += 1
                parts += len(content)
                users += role == "user"
                byte_count += len(payload.encode())
    assert (messages, parts, users) == (8134, 47976, 494)
    assert byte_count >= 314_603_929
    print(
        json.dumps(
            {
                "profile": "loopback-mlflow-3.17.2/sqlite/file-artifacts",
                "hardware": platform.uname()._asdict(),
                "sessions": 307,
                "messages": messages,
                "parts": parts,
                "users": users,
                "json_bytes": byte_count,
            }
        )
    )
    config = ImportConfig(
        unique_experiment_name("reference-scale"), source, tracking_uri=live_server.uri, log_level="DEBUG"
    )
    dependencies = build_dependencies(config)
    writes = 0
    mutation_guard = MlflowGateway._require_mutation_allowed

    def observe_mutation(gateway):
        nonlocal writes
        writes += 1
        return mutation_guard(gateway)

    monkeypatch.setattr(MlflowGateway, "_require_mutation_allowed", observe_mutation)
    ids = None
    maps = None
    for phase in ("fresh", "unchanged", "append", "edit"):
        if phase == "append":
            with closing(sqlite3.connect(source)) as connection, connection:
                connection.execute(
                    "INSERT INTO session_message VALUES(?,?,?,?,?,?,?)",
                    (
                        "synthetic-appended-user",
                        "synthetic-session-306",
                        "user",
                        99,
                        1700002000000,
                        1700002000000,
                        '{"text":"synthetic appended prompt"}',
                    ),
                )
        elif phase == "edit":
            with closing(sqlite3.connect(source)) as connection, connection:
                connection.execute(
                    "UPDATE session_message SET data=? WHERE id='synthetic-appended-user'",
                    ('{"text":"synthetic edited prompt"}',),
                )
        started = time.perf_counter()
        writes = 0
        with closing(run_import(config, dependencies)) as result:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "seconds": elapsed,
                        "exit_status": exit_status(result),
                        "summary": result.report["summary"],
                        "errors": result.report["errors"],
                        "mutation_entrypoints": writes,
                    }
                ),
                flush=True,
            )
            assert exit_status(result) == 0, json.dumps(result.report)
            assert elapsed < 600
            current_ids = {
                turn["turn_key"]: turn["trace_id"]
                for session in result.report["sessions"]
                for turn in session["turns"]
                if turn["source_present"]
            }
            if phase == "unchanged":
                assert current_ids == ids
                assert writes == 0
            elif phase == "append":
                assert ids.items() <= current_ids.items()
            elif phase == "edit":
                changed_keys = {key for key in ids if current_ids[key] != ids[key]}
                assert len(changed_keys) == 1
            current_maps = {}
            token = None
            while True:
                page = live_server.client.search_traces(
                    locations=[result.report["experiment"]["id"]], include_spans=False, page_token=token
                )
                current_maps.update({trace.trace_id: trace.info.tags["opencode.artifact_path"] for trace in page})
                token = page.token
                if not token:
                    break
            assert_imported_workload(result, 307, 494 if phase in {"fresh", "unchanged"} else 495, current_maps)
            if maps is not None:
                assert all(
                    current_maps[trace_id] == path for trace_id, path in maps.items() if trace_id in current_maps
                )
            ids, maps = current_ids, current_maps
