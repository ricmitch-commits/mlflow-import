"""Live parity for missing timing and partial-child rollback."""

import sqlite3

import pytest

from opencode_mlflow.mlflow.gateway import MlflowGateway, PublicTraceApi
from tests.integration.import_runner import run_real_import, turn_entries
from tests.integration.server import unique_experiment_name
from tests.support.import_fakes import build_current_source_db

pytestmark = pytest.mark.integration


def test_live_missing_source_timestamps_never_create_a_trace(live_server, restrictive_tmp_path):
    source = build_current_source_db(
        restrictive_tmp_path, sessions=[("missing", 1000)], messages=[("u", "missing", "user", 1, "{}")]
    )
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE session_v2 SET time_created=NULL,time_updated=NULL")
        connection.execute("UPDATE session_message SET time_created=NULL,time_updated=NULL")
    run = run_real_import(
        server_uri=live_server.uri,
        source=source,
        experiment_name=unique_experiment_name("missing-time"),
        tmp_root=restrictive_tmp_path,
    )
    assert run.exit_status == 1
    assert turn_entries(run.report)[0]["error"]["code"] == "source_timestamp_missing"
    assert live_server.client.search_traces(locations=[run.report["experiment"]["id"]]) == []


def test_live_child_failure_ends_and_rolls_back_known_partial_trace(live_server, restrictive_tmp_path):
    source = build_current_source_db(
        restrictive_tmp_path,
        sessions=[("partial", 1000)],
        messages=[
            ("u", "partial", "user", 1, '{"content":[{"type":"text","text":"prompt"}]}'),
            ("a", "partial", "assistant", 2, '{"content":[{"type":"text","text":"reply"}]}'),
        ],
    )

    class FailingApi(PublicTraceApi):
        children = 0

        def start_span_no_context(self, **kwargs):
            if kwargs.get("parent_span") is not None:
                self.children += 1
                if self.children == 2:
                    raise RuntimeError("synthetic second-child failure")
            return super().start_span_no_context(**kwargs)

    retired = []

    class InspectingGateway(MlflowGateway):
        def retire_trace(self, experiment_id, trace_id, **kwargs):
            trace = live_server.client.get_trace(trace_id, display=False, flush=True)
            root = next(s for s in trace.data.spans if s.parent_id is None)
            assert len(trace.data.spans) == 2
            assert root.end_time_ns is not None
            assert root.status.status_code.value == "ERROR"
            super().retire_trace(experiment_id, trace_id, **kwargs)
            retired.append(trace_id)

    run = run_real_import(
        server_uri=live_server.uri,
        source=source,
        experiment_name=unique_experiment_name("partial-create"),
        tmp_root=restrictive_tmp_path,
        gateway_factory=lambda dry_run: InspectingGateway(live_server.client, trace_api=FailingApi(), dry_run=dry_run),
    )
    assert run.exit_status == 1
    turn = turn_entries(run.report)[0]
    assert turn["error"]["code"] == "trace_create_failed"
    assert retired == [turn["trace_id"]]
    assert live_server.client.search_traces(locations=[run.report["experiment"]["id"]]) == []
