"""Production historical gateway, tree/replay and phase-two quota qualification."""

import json
import sqlite3
import tracemalloc
from datetime import UTC, datetime, timedelta

import pytest

from mlflow_import import resources
from mlflow_import.config import ImportConfig, parse_config
from mlflow_import.importer import build_dependencies, exit_status, run_import
from mlflow_import.mlflow import attempts, reconciliation
from mlflow_import.source.replay import NormalizedSnapshotAdapter
from tests.integration.server import unique_experiment_name
from tests.integration.test_attempt_boundaries import _source
from tests.support.temporary_resources import isolate_workspaces

pytestmark = pytest.mark.integration


def _consume_remaining_space():
    budget = resources.current_storage()
    assert budget is not None and budget.phase2
    with resources.open_file(budget.root / "competing-spool", "wb") as handle:
        handle.write(b"x" * (budget.max_bytes - budget.used_bytes - resources.PAGE))


@pytest.mark.parametrize("phase", ["serialization", "download"])
def test_phase_two_quota_exhaustion_preserves_prior_and_recovers_without_leaks(
    live_server, tmp_path, monkeypatch, phase
):
    source = _source(tmp_path)
    config = ImportConfig(unique_experiment_name("quota-phase2"), source, tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    first = run_import(config, dependencies)
    assert exit_status(first) == 0, first.report
    prior = first.report["sessions"][0]["turns"][0]["trace_id"]
    first.close()
    with sqlite3.connect(source) as db:
        db.execute("UPDATE session_message SET data=? WHERE id='a1'", ('{"text":"synthetic quota edit"}',))
    injected = False
    in_ready = False
    with monkeypatch.context() as patch:
        home = isolate_workspaces(patch, tmp_path / "workspaces", 1024 * 1024)
        write = attempts.write_state_json
        download = resources.copy_artifact
        finish = reconciliation._finish_ready

        def serialize(path, value):
            nonlocal injected
            if (
                phase == "serialization"
                and isinstance(value, dict)
                and value.get("state") == "prepared"
                and not injected
            ):
                _consume_remaining_space()
                injected = True
            return write(path, value)

        def copy(*args, **kwargs):
            nonlocal injected
            if phase == "download" and in_ready and not injected:
                _consume_remaining_space()
                injected = True
            return download(*args, **kwargs)

        def ready(*args, **kwargs):
            nonlocal in_ready
            in_ready = True
            try:
                return finish(*args, **kwargs)
            finally:
                in_ready = False

        patch.setattr(attempts, "write_state_json", serialize)
        patch.setattr(resources, "copy_artifact", copy)
        patch.setattr(reconciliation, "_finish_ready", ready)
        result = run_import(config, dependencies)
        assert injected
        assert exit_status(result) == 1, result.report
        codes = [error["code"] for error in result.report["errors"]] + [
            turn["error_code"] for session in result.report["sessions"] for turn in session["turns"]
        ]
        assert "temporary_storage_exhausted" in codes
        assert live_server.client.get_trace(prior, display=False).trace_id == prior
        assert result.report["resources"]["temporary_peak_bytes"] <= 1024 * 1024
        result.close()
        assert list(home.iterdir()) == []
    resumed = run_import(config, dependencies)
    assert exit_status(resumed) == 0, resumed.report
    traces = live_server.client.search_traces(locations=[resumed.report["experiment"]["id"]], include_spans=False)
    assert len(traces) == 1 and traces[0].trace_id != prior
    resumed.close()


def _write_many_turns(path, harness, count):
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    records = (
        [{"type": "session", "version": 3, "id": "resource-tree", "timestamp": stamp.isoformat()}]
        if harness == "pi"
        else [{"sessionId": "resource-replay", "projectHash": "synthetic", "startTime": stamp.isoformat()}]
    )
    parent = None
    for index in range(count):
        when = (stamp + timedelta(seconds=2 * index)).isoformat()
        user, assistant = f"u-{index}", f"a-{index}"
        if harness == "pi":
            records.extend(
                [
                    {
                        "type": "message",
                        "id": user,
                        "parentId": parent,
                        "timestamp": when,
                        "message": {"role": "user", "content": "synthetic prompt"},
                    },
                    {
                        "type": "message",
                        "id": f"discard-{index}",
                        "parentId": user,
                        "message": {"role": "assistant", "content": "inactive branch"},
                    },
                    {
                        "type": "branch_summary",
                        "id": f"branch-{index}",
                        "parentId": user,
                        "fromId": f"discard-{index}",
                        "summary": "synthetic branch",
                    },
                    {
                        "type": "message",
                        "id": assistant,
                        "parentId": f"branch-{index}",
                        "timestamp": when,
                        "message": {"role": "assistant", "content": "synthetic chosen answer"},
                    },
                ]
            )
        else:
            records.extend(
                [
                    {"id": user, "type": "user", "timestamp": when, "content": "synthetic prompt"},
                    {"id": assistant, "type": "gemini", "timestamp": when, "content": "draft"},
                    {"$patch": {"id": assistant, "content": "synthetic replayed answer"}},
                    {"id": f"discard-{index}", "type": "user", "content": "rewound"},
                    {"$rewindTo": f"discard-{index}"},
                ]
            )
        parent = assistant
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


@pytest.mark.parametrize("harness", ["pi", "gemini"])
def test_production_many_small_turn_tree_replay_resource_gate(live_server, tmp_path, monkeypatch, harness):
    home = isolate_workspaces(monkeypatch, tmp_path / "workspaces", 32 * 1024 * 1024)
    original_records = NormalizedSnapshotAdapter.iter_records
    yielded = 0

    def count_records(self, *args, **kwargs):
        nonlocal yielded
        for record in original_records(self, *args, **kwargs):
            yielded += 1
            yield record

    monkeypatch.setattr(NormalizedSnapshotAdapter, "iter_records", count_records)
    measurements = []
    for count in (32, 64):
        source = tmp_path / f"{harness}-{count}.jsonl"
        _write_many_turns(source, harness, count)
        config = parse_config(
            [
                "--experiment-name",
                unique_experiment_name("production-resource"),
                "--harness",
                harness,
                "--source",
                str(source),
                "--tracking-uri",
                live_server.uri,
            ]
        )
        yielded = 0
        tracemalloc.start()
        try:
            result = run_import(config, build_dependencies(config))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert exit_status(result) == 0, result.report
        assert result.report["summary"]["imported_turns"] == count
        telemetry = result.report["resources"]
        assert telemetry["temporary_peak_bytes"] <= 32 * 1024 * 1024
        assert peak < 32 * 1024 * 1024
        measurements.append((telemetry["temporary_sql_statements"], yielded, peak))
        result.close()
        assert list(home.iterdir()) == []
    small, large = measurements
    print(f"{harness} production quota gate (SQL statements, full-session rows, traced heap): {measurements}")
    assert large[0] <= small[0] * 2.5 + 512
    assert large[1] <= small[1] * 2.1 + 16
    assert large[2] <= small[2] + 8 * 1024 * 1024
