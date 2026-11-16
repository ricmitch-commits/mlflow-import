"""Small synthetic sources for production-mapper/recovery acceptance tests."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

from mlflow_import.contracts import RedactionMode
from mlflow_import.mlflow.mapping import MappingContext, map_turn_to_trace
from mlflow_import.source.catalog import CURRENT_CATALOG_V1
from mlflow_import.source.current import CurrentAdapter
from mlflow_import.source.discovery import ARTIFACT_SCHEMA_VERSION, MAPPING_VERSION
from mlflow_import.source.file_snapshot import create_file_snapshot
from mlflow_import.source.replay import NormalizedSnapshotAdapter
from mlflow_import.source.snapshot import create_normalized_snapshot
from mlflow_import.source.turns import build_replay_turn, indexed_turn_summaries
from mlflow_import.sources.jsonl_common import JsonlExporter
from tests.support.import_fakes import build_current_source_db

SESSION = "compact-synthetic"
PROMPT = 'PROMPT:synthetic 雪😀\\"\n' + "p" * 4096


def answer(index, *, final=False, edited=False):
    content = [{"type": "text", "text": f"ANSWER:{index}:" + ("edited" if edited else "a" * 128)}]
    if not final:
        content.append(
            {
                "type": "tool",
                "tool": "synthetic-read",
                "callID": f"call-{index}",
                "state": {
                    "input": {"query": f"QUERY:{index}", "typed": [None, False, 7, "7"]},
                    "output": f"RESULT:{index}:" + "r" * 1024,
                    "status": "completed",
                },
            }
        )
    return {
        "role": "assistant",
        "model": "synthetic/model",
        "content": content,
        "finish": "stop" if final else "tool-calls",
    }


def synthetic_source(directory, calls, *, sibling=False):
    messages = [("u0", SESSION, "user", 1000, json.dumps({"role": "user", "content": PROMPT}))]
    messages.extend(
        (f"a{i}", SESSION, "assistant", 2000 + i, json.dumps(answer(i, final=i == calls - 1))) for i in range(calls)
    )
    if sibling:
        messages.extend(
            [
                ("sibling-u", SESSION, "user", 100000, '{"text":"SIBLING:prompt"}'),
                (
                    "sibling-a",
                    SESSION,
                    "assistant",
                    100001,
                    '{"model":"synthetic/model","text":"SIBLING:answer","finish":"stop"}',
                ),
            ]
        )
    return build_current_source_db(directory, sessions=[(SESSION, 1000)], messages=messages)


@contextmanager
def mapped_source(source, directory):
    """Real exporter, snapshot, replay, and mapper; no gateway or hand-built DTO."""
    with sqlite3.connect(source) as connection:
        exporter = CurrentAdapter(connection, CURRENT_CATALOG_V1)
    with create_normalized_snapshot(source, directory, exporter, lambda value, _: value) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = next(adapter.iter_session_headers())
        turns = [
            build_replay_turn(snapshot, header, summary)
            for summary in indexed_turn_summaries(snapshot, header.session_id)
        ]
        context = MappingContext(
            "0.1.0",
            MAPPING_VERSION,
            ARTIFACT_SCHEMA_VERSION,
            header.adapter_generation,
            RedactionMode.VERBATIM,
            "synthetic-local",
        )
        yield [(turn, map_turn_to_trace(turn, context).to_create_request()) for turn in turns]


@contextmanager
def mapped_scoped_source(directory):
    """Persist a primary Claude conversation and a recorded, associated sidecar."""

    def conversation(prefix, parent=None, delegate=False):
        def row(key, previous, role, content, **extra):
            return {
                "sessionId": "scoped",
                "type": role,
                "uuid": prefix + key,
                "parentUuid": previous,
                "message": {"role": role, "content": content, **extra},
            }

        call = prefix + "call"
        return [
            row("u", parent, "user", prefix + "PROMPT"),
            row(
                "a",
                prefix + "u",
                "assistant",
                [
                    {"type": "text", "text": prefix + "ANSWER"},
                    {
                        "type": "tool_use",
                        "id": call,
                        "name": "Task" if delegate else "read",
                        "input": {"query": prefix + "QUERY"},
                    },
                ],
                id=prefix + "invocation",
                model="synthetic/model",
            ),
            row(
                "r",
                prefix + "a",
                "user",
                [
                    {"type": "tool_result", "tool_use_id": call, "content": prefix + "RESULT"},
                ],
            ),
            row(
                "z",
                prefix + "r",
                "assistant",
                prefix + "FINAL",
                id=prefix + "final",
                model="synthetic/model",
                stop_reason="stop",
            ),
        ]

    path = directory / "scoped.jsonl"
    main = conversation("PRIMARY:", delegate=True)
    main[2]["toolUseResult"] = {"agentId": "child"}
    path.write_text("".join(json.dumps(value) + "\n" for value in main))
    side = directory / "scoped" / "subagents" / "agent-child.jsonl"
    side.parent.mkdir(parents=True)
    child = conversation("CHILD:", parent="PRIMARY:a")
    child[0]["parentToolUseID"] = "PRIMARY:call"
    side.write_text("".join(json.dumps(value) + "\n" for value in child))
    with create_file_snapshot(path, directory, JsonlExporter(path, "claude"), lambda value, _: value) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = next(adapter.iter_session_headers())
        [summary] = list(indexed_turn_summaries(snapshot, header.session_id))
        turn = build_replay_turn(snapshot, header, summary)
        context = MappingContext(
            "0.1.0",
            MAPPING_VERSION,
            ARTIFACT_SCHEMA_VERSION,
            header.adapter_generation,
            RedactionMode.VERBATIM,
            "synthetic-scoped",
        )
        yield turn, map_turn_to_trace(turn, context).to_create_request()
