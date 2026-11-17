"""Independent session-wide safety checks on disposable synthetic targets."""

import json
import sqlite3
from dataclasses import replace

import pytest

from mlflow_import.config import ImportConfig
from mlflow_import.errors import ItemImportError
from mlflow_import.importer import build_dependencies, exit_status, run_import
from tests.integration.server import unique_experiment_name
from tests.support.import_fakes import build_current_source_db

pytestmark = pytest.mark.integration


def test_multiple_sessions_reconcile_and_rerun_after_execution_store_cleanup(live_server, tmp_path):
    source = build_current_source_db(
        tmp_path,
        sessions=[("first", 1000), ("second", 2000), ("empty", 3000)],
        messages=[
            (f"{session}-{role}", session, role, seq, json.dumps({"text": f"synthetic {role}"}))
            for session in ("first", "second")
            for role, seq in (("user", 1), ("assistant", 2))
        ],
    )
    config = ImportConfig(unique_experiment_name("session-lifecycle"), source, tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    first = run_import(config, dependencies)
    assert exit_status(first) == 0, first.report
    assert [session["session_id"] for session in first.report["sessions"]] == ["first", "second", "empty"]
    assert first.report["summary"]["turns_imported"] == 2

    # Each run's report must survive the destruction of every session store.
    original_ids = [session["turns"][0]["trace_id"] for session in first.report["sessions"] if session["turns"]]
    again = run_import(config, dependencies)
    assert exit_status(again) == 0, again.report
    assert again.report["summary"]["unchanged_turns"] == 2
    assert [session["turns"][0]["trace_id"] for session in again.report["sessions"] if session["turns"]] == original_ids

    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE session_message SET data=? WHERE id='first-assistant'", ('{"text":"edited"}',))
        connection.execute("DELETE FROM session_message WHERE session_id='second'")
    updated = run_import(config, dependencies)
    assert exit_status(updated) == 0, updated.report
    assert updated.report["summary"]["turns_updated"] == 1
    assert updated.report["summary"]["retired_traces"] == 2
    traces = live_server.client.search_traces(locations=[updated.report["experiment"]["id"]], include_spans=False)
    assert len(traces) == 1
    assert not set(original_ids) & {trace.trace_id for trace in traces}


def test_later_turn_failure_preserves_entire_prior_session_and_filtered_pending(live_server, tmp_path):
    source = build_current_source_db(
        tmp_path,
        sessions=[("safety", 1000)],
        messages=[
            ("u1", "safety", "user", 1, '{"text":"synthetic one"}'),
            ("a1", "safety", "assistant", 2, '{"text":"old one"}'),
            ("u2", "safety", "user", 3, '{"text":"synthetic two"}'),
            ("a2", "safety", "assistant", 4, '{"text":"old two"}'),
        ],
    )
    config = ImportConfig(unique_experiment_name("session-safety"), source, tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    first = run_import(config, dependencies)
    assert exit_status(first) == 0, first.report
    before = {turn["trace_id"] for session in first.report["sessions"] for turn in session["turns"]}
    for trace_id in before:
        metadata = live_server.client.get_trace(trace_id, display=False).info.tags
        assert not metadata.get("mlflow.user")
        assert not metadata.get("mlflow.source.git.branch")
        assert metadata["mlflow.source.name"] == "opencode"
    with sqlite3.connect(source) as connection:
        connection.execute(
            "UPDATE session_message SET data=? WHERE id IN ('a1','a2')",
            (json.dumps({"text": "edited synthetic reply"}),),
        )

    def failing_factory(dry_run):
        gateway = dependencies.gateway_factory(dry_run)
        original = gateway.create_trace
        creates = 0

        def create(request):
            nonlocal creates
            creates += 1
            if creates == 2:
                raise ItemImportError("injected_second_turn_failure", "synthetic failure")
            return original(request)

        gateway.create_trace = create
        return gateway

    failed = run_import(config, replace(dependencies, gateway_factory=failing_factory))
    assert exit_status(failed) == 1, failed.report
    for trace_id in before:
        assert live_server.client.get_trace(trace_id, display=False).trace_id == trace_id
    filtered = run_import(replace(config, min_user_turns=99), dependencies)
    assert exit_status(filtered) == 1, filtered.report
    assert filtered.report["summary"]["selected_sessions"] == 0
    assert filtered.report["sessions"][0]["reason"] == "recovery_pending"
    resumed = run_import(config, dependencies)
    assert exit_status(resumed) == 0, resumed.report
    experiment_id = resumed.report["experiment"]["id"]
    traces = list(live_server.client.search_traces(locations=[experiment_id], include_spans=False))
    assert len(traces) == 2


def test_partial_ready_deletions_are_counted_before_completion(live_server, tmp_path):
    source = build_current_source_db(
        tmp_path,
        sessions=[("partial", 1000)],
        messages=[
            ("u1", "partial", "user", 1, '{"text":"one"}'),
            ("a1", "partial", "assistant", 2, '{"text":"old one"}'),
            ("u2", "partial", "user", 3, '{"text":"two"}'),
            ("a2", "partial", "assistant", 4, '{"text":"old two"}'),
        ],
    )
    config = ImportConfig(unique_experiment_name("partial-delete"), source, tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    first = run_import(config, dependencies)
    assert exit_status(first) == 0, first.report
    prior = {turn["trace_id"] for turn in first.report["sessions"][0]["turns"]}
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE session_message SET data=? WHERE type='assistant'", ('{"text":"edited"}',))

    def factory(dry_run):
        gateway = dependencies.gateway_factory(dry_run)
        retire = gateway.retire_trace
        deleted = False

        def partial(*args, **kwargs):
            nonlocal deleted
            if deleted:
                raise ItemImportError("injected_delete_failure", "synthetic failure")
            retire(*args, **kwargs)
            deleted = True

        gateway.retire_trace = partial
        return gateway

    failed = run_import(config, replace(dependencies, gateway_factory=factory))
    assert exit_status(failed) == 1
    assert failed.report["summary"]["retired_traces"] == 1
    reported = {trace_id for turn in failed.report["sessions"][0]["turns"] for trace_id in turn["retired_trace_ids"]}
    assert len(reported) == 1 and reported <= prior
    resumed = run_import(config, dependencies)
    assert exit_status(resumed) == 0, resumed.report
    assert resumed.report["summary"]["retired_traces"] == 2
    traces = live_server.client.search_traces(locations=[resumed.report["experiment"]["id"]], include_spans=False)
    assert len(traces) == 2
    assert not prior & {trace.trace_id for trace in traces}
