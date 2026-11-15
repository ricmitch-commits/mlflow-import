"""Literal reference parity through historical ingestion and the SQL backend."""

import json
import sqlite3
from dataclasses import replace

import pytest

from mlflow_import.config import ImportConfig
from mlflow_import.importer import build_dependencies, exit_status, run_import
from tests.integration.server import live_mlflow_server, unique_experiment_name
from tests.support.import_fakes import build_current_source_db

pytestmark = pytest.mark.integration


def _reference_import(live_server, tmp_path):
    session = "ses-reference-synthetic"
    start, completion = 1794580115471, 1794580131437
    messages = [
        ("u", session, "user", 1, json.dumps({
            "time": {"created": start},
            "content": [{"type": "text", "text": "first"},
                        {"type": "text", "text": "synthetic context", "synthetic": True}],
        })),
        ("a1", session, "assistant", 2, json.dumps({
            "time": {"created": start + 1, "completed": start + 100},
            "model": "synthetic/model", "finish": "tool-calls", "cost": 0,
            "tokens": {"input": 36500, "output": 100, "reasoning": 10, "total": 75104,
                       "cache": {"read": 38464, "write": 0}},
            "content": [{"type": "text", "text": "intermediate"}],
        })),
        ("a2", session, "assistant", 3, json.dumps({
            "time": {"created": start + 101, "completed": completion},
            "model": "synthetic/model", "finish": "stop", "cost": 0,
            "tokens": {"input": 36702, "output": 48, "reasoning": 20, "total": 75204,
                       "cache": {"read": 38464, "write": 0}},
            "content": [{"type": "text", "text": "answer"},
                        {"type": "text", "text": " "}, {"type": "text", "text": "last"}],
        })),
    ]
    source = build_current_source_db(tmp_path, sessions=[(session, start)], messages=messages)
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE session_v2 SET parent_id='bare-parent'")
        connection.execute(
            "UPDATE session_message SET time_created=?, time_updated=? WHERE id='u'", (start, completion + 5)
        )
    config = ImportConfig(unique_experiment_name("reference-info"), source, tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    result = run_import(config, dependencies)
    assert exit_status(result) == 0, [turn.get("error") for turn in result.report["sessions"][0]["turns"]]
    return config, dependencies, result


@pytest.mark.parametrize("public_only_handle", [False, True])
def test_reference_projection_retains_raw_accounting_and_reruns(live_server, tmp_path, public_only_handle):
    if public_only_handle:
        # Exercise an external-server-shaped handle using the same disposable
        # URI: public assertions must not depend on local backend access.
        live_server = replace(live_server, run_root=None, process=None)
    config, dependencies, result = _reference_import(live_server, tmp_path)
    session = "ses-reference-synthetic"
    start = 1794580115471
    trace_id = result.report["sessions"][0]["turns"][0]["trace_id"]
    trace = live_server.client.get_trace(trace_id, display=False)
    info = trace.info
    assert info.request_time == start
    assert info.execution_duration == 15966
    assert info.client_request_id is None
    assert info.tags.get("mlflow.traceName") is None
    assert info.request_preview == "first\nsynthetic context"
    assert info.response_preview == "answer\nlast"
    assert info.trace_metadata["mlflow.trace.session"] == session
    assert info.trace_metadata["parent_session_id"] == "bare-parent"
    assert info.token_usage == {"input_tokens": 73202, "output_tokens": 148, "total_tokens": 73380}
    assert info.cost is None
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert root.name == "opencode_conversation"
    assert root.attributes["mlflow.chat.tokenUsage"]["total_tokens"] == 150308
    assert root.attributes["mlflow.chat.tokenUsage"]["cache_read_input_tokens"] == 76928
    assert json.loads(info.tags["mlflow_import.qualified_session_id"])[-1] == session
    again = run_import(config, dependencies)
    assert exit_status(again) == 0, [turn.get("error") for turn in again.report["sessions"][0]["turns"]]
    assert again.report["sessions"][0]["turns"][0]["trace_id"] == trace_id
    assert again.report["summary"]["unchanged_turns"] == 1
    # The immutable ownership tag is rechecked before any reuse/retirement.
    live_server.client.set_trace_tag(trace_id, "mlflow_import.source_namespace", "contradiction")
    rejected = run_import(replace(config, dry_run=True), dependencies)
    assert exit_status(rejected) != 0
    assert live_server.client.get_trace(trace_id, display=False).trace_id == trace_id


@pytest.fixture(scope="module")
def disposable_sql_server():
    """Physical SQL assertions never depend on an externally configured server."""
    with live_mlflow_server(requested_uri="") as server:
        yield server


def test_physical_request_metadata_whitelist(disposable_sql_server, tmp_path):
    _, _, result = _reference_import(disposable_sql_server, tmp_path)
    trace_id = result.report["sessions"][0]["turns"][0]["trace_id"]
    assert disposable_sql_server.run_root is not None
    with sqlite3.connect(disposable_sql_server.run_root / "mlflow.db") as connection:
        keys = {row[0] for row in connection.execute(
            "SELECT key FROM trace_request_metadata WHERE request_id=?", (trace_id,)
        )}
    assert keys <= {"parent_session_id", "mlflow.trace.infoFinalized"}
    assert "parent_session_id" in keys


def _simple_import(server, tmp_path, user, assistant, *, session="review-projection", metadata=None):
    source = build_current_source_db(
        tmp_path, sessions=[(session, 1000)],
        messages=[("u", session, "user", 1, json.dumps(user)),
                  ("a", session, "assistant", 2, json.dumps(assistant))],
    )
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE session_message SET time_created=1000,time_updated=1500 WHERE id='u'")
        connection.execute("UPDATE session_message SET time_created=2000,time_updated=2020 WHERE id='a'")
        if metadata is not None:
            connection.execute("UPDATE session_v2 SET metadata=?", (json.dumps(metadata),))
    config = ImportConfig(unique_experiment_name("review-projection"), source, tracking_uri=server.uri)
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, [turn.get("error") for turn in result.report["sessions"][0]["turns"]]
    return server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)


def test_odd_utf16_preview_boundary_survives_historical_ingestion(live_server, tmp_path):
    text = "a" * 999 + "😀"
    trace = _simple_import(
        live_server, tmp_path, {"text": text},
        {"content": [{"type": "text", "text": text}], "finish": "stop"},
    )
    assert trace.info.request_preview == trace.info.response_preview == "a" * 999 + "\ufffd"
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert text in str(root.inputs) and text in str(root.outputs)


@pytest.mark.parametrize("parent_reasoning,expected", [(30, 140), (0, 110), (None, 115)])
def test_multistep_reasoning_uses_parent_precedence(live_server, tmp_path, parent_reasoning, expected):
    tokens = {"input": 100, "output": 10}
    if parent_reasoning is not None:
        tokens["reasoning"] = parent_reasoning
    trace = _simple_import(
        live_server, tmp_path, {"text": "prompt"},
        {"model": "synthetic/model", "tokens": tokens, "finish": "stop", "content": [
            {"type": "step-finish", "tokens": {"input": 40, "output": 4, "reasoning": 2}},
            {"type": "step-finish", "tokens": {"input": 60, "output": 6, "reasoning": 3}},
            {"type": "text", "text": "answer"},
        ]},
    )
    assert trace.info.token_usage == {"input_tokens": 100, "output_tokens": 10, "total_tokens": expected}


def test_long_qualified_identity_and_git_tag_survive_historical_ingestion(live_server, tmp_path):
    session = "s" * 180
    url = "https://example.test/" + "r" * 260
    trace = _simple_import(
        live_server, tmp_path, {"text": "prompt"}, {"text": "answer", "finish": "stop"},
        session=session, metadata={"git": {"repository_url": url}},
    )
    assert trace.info.trace_metadata["mlflow.trace.session"] == session
    assert len(trace.info.tags["mlflow_import.qualified_session_id"]) > 250
    assert json.loads(trace.info.tags["mlflow_import.qualified_session_id"])[-1] == session
    assert len(url) == 281
    assert trace.info.tags["mlflow.source.git.repoURL"] == url


@pytest.mark.parametrize("invalid", [True, False, 1e100, "malformed"])
def test_invalid_semantic_times_survive_historical_ingestion(live_server, tmp_path, invalid):
    trace = _simple_import(
        live_server, tmp_path, {"text": "prompt", "time": {"created": invalid}},
        {"text": "answer", "finish": "stop", "time": {"created": invalid, "completed": invalid}},
    )
    assert trace.info.request_time == 1000
    assert trace.info.execution_duration == 1020
