"""Recovery at durable session boundaries using only synthetic source content."""

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from mlflow_import.config import ImportConfig
from mlflow_import.errors import ItemImportError
from mlflow_import.importer import build_dependencies, exit_status, run_import
from mlflow_import.mlflow.attempts import AttemptJournal
from tests.integration.server import unique_experiment_name
from tests.support.import_fakes import build_current_source_db

pytestmark = pytest.mark.integration


def _source(tmp_path):
    return build_current_source_db(
        tmp_path,
        sessions=[("boundary", 1000)],
        messages=[
            ("u1", "boundary", "user", 1, '{"text":"synthetic prompt"}'),
            ("a1", "boundary", "assistant", 2, '{"text":"synthetic answer"}'),
        ],
    )


def test_completed_attempt_has_verified_manifest(live_server, tmp_path):
    source = _source(tmp_path)
    config = ImportConfig(unique_experiment_name("manifest"), source, tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    result = run_import(config, dependencies)
    assert exit_status(result) == 0, list(result.report["sessions"][0]["turns"])
    gateway = dependencies.gateway_factory(False)
    experiment_id = result.report["experiment"]["id"]
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    prefix = trace.info.tags["mlflow_import.session_artifact"].split("/verbatim/")[0]
    states = next(AttemptJournal(gateway, experiment_id, prefix).attempts())
    path, checksum, size = states["complete"]["manifest"]
    repository = gateway.experiment_repository(experiment_id)
    downloaded = Path(repository.download_artifacts(path, str(tmp_path)))
    payload = downloaded.read_bytes()
    assert (hashlib.sha256(payload).hexdigest(), len(payload)) == (checksum, size)
    manifest = json.loads(payload)
    assert manifest["schema_version"] == "mlflow-import.session-manifest.v1"
    assert [turn["trace_id"] for turn in manifest["turns"]] == [trace.trace_id]
    assert manifest["session"] == states["prepared"]["session_object"]


@pytest.mark.parametrize("boundary", ["prepared", "ready_to_retire", "complete"])
@pytest.mark.parametrize("policy", ["overwrite", "duplicate", "ignore"])
def test_state_publication_failure_resumes_without_another_trace(live_server, tmp_path, boundary, policy):
    source = _source(tmp_path)
    config = ImportConfig(unique_experiment_name("state-response"), source, tracking_uri=live_server.uri)
    object.__setattr__(config, "_on_conflict", policy)
    dependencies = build_dependencies(config)
    injected = False

    def factory(dry_run):
        gateway = dependencies.gateway_factory(dry_run)
        publish = gateway.publish_experiment_file

        def lose_response(experiment_id, path, local, **kwargs):
            nonlocal injected
            if not injected and boundary == "complete" and Path(path).name.startswith("complete-"):
                injected = True
                raise ItemImportError("injected_state_publication_failure", "synthetic failure")
            result = publish(experiment_id, path, local, **kwargs)
            if not injected and Path(path).name.startswith(boundary + "-"):
                injected = True
                raise ItemImportError("injected_lost_state_response", "synthetic failure")
            return result

        gateway.publish_experiment_file = lose_response
        return gateway

    failed = run_import(config, replace(dependencies, gateway_factory=factory))
    assert injected
    assert exit_status(failed) == 1, failed.report
    resumed = run_import(config, dependencies)
    assert exit_status(resumed) == 0, [turn.get("error") for turn in resumed.report["sessions"][0]["turns"]]
    traces = live_server.client.search_traces(locations=[resumed.report["experiment"]["id"]], include_spans=False)
    assert len(traces) == 1


def test_changed_input_finishes_ready_batch_before_replacement(live_server, tmp_path):
    source = _source(tmp_path)
    config = ImportConfig(unique_experiment_name("ready-changed"), source, tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    first = run_import(config, dependencies)
    assert exit_status(first) == 0
    old_id = first.report["sessions"][0]["turns"][0]["trace_id"]
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE session_message SET data=? WHERE id='a1'", ('{"text":"second answer"}',))

    def factory(dry_run):
        gateway = dependencies.gateway_factory(dry_run)

        def fail_retirement(*args, **kwargs):
            raise ItemImportError("injected_retirement_failure", "synthetic failure")

        gateway.retire_trace = fail_retirement
        return gateway

    failed = run_import(config, replace(dependencies, gateway_factory=factory))
    assert exit_status(failed) == 1
    assert live_server.client.get_trace(old_id, display=False).trace_id == old_id
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE session_message SET data=? WHERE id='a1'", ('{"text":"third answer"}',))
    resumed = run_import(config, dependencies)
    assert exit_status(resumed) == 0, resumed.report
    traces = live_server.client.search_traces(locations=[resumed.report["experiment"]["id"]], include_spans=True)
    assert len(traces) == 1
    assert traces[0].data.spans[0].outputs == {"text": "third answer"}


def test_zero_turn_overwrite_retires_only_after_empty_manifest(live_server, tmp_path):
    source = _source(tmp_path)
    config = ImportConfig(unique_experiment_name("empty-manifest"), source, tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    first = run_import(config, dependencies)
    assert exit_status(first) == 0
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE session_message SET type='system'")
    emptied = run_import(config, dependencies)
    assert exit_status(emptied) == 0, emptied.report
    assert emptied.report["summary"]["selected_turns"] == 0
    assert emptied.report["summary"]["retired_traces"] == 1
    traces = live_server.client.search_traces(locations=[emptied.report["experiment"]["id"]], include_spans=False)
    assert not traces


def test_zero_trace_pending_completion_is_recovered(live_server, tmp_path):
    source = _source(tmp_path)
    config = ImportConfig(unique_experiment_name("empty-ready-resume"), source, tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    first = run_import(config, dependencies)
    assert exit_status(first) == 0
    experiment_id = first.report["experiment"]["id"]
    trace = live_server.client.get_trace(first.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    prefix = trace.info.tags["mlflow_import.session_artifact"].split("/verbatim/")[0]
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE session_message SET type='system'")

    def factory(dry_run):
        gateway = dependencies.gateway_factory(dry_run)
        publish = gateway.publish_experiment_file

        def fail_complete(experiment_id, path, local, **kwargs):
            if Path(path).name.startswith("complete-"):
                raise ItemImportError("injected_completion_failure", "synthetic failure")
            return publish(experiment_id, path, local, **kwargs)

        gateway.publish_experiment_file = fail_complete
        return gateway

    failed = run_import(config, replace(dependencies, gateway_factory=factory))
    assert exit_status(failed) == 1
    assert failed.report["summary"]["retired_traces"] == 1
    resumed = run_import(config, dependencies)
    assert exit_status(resumed) == 0, resumed.report
    journal = AttemptJournal(dependencies.gateway_factory(False), experiment_id, prefix)
    assert all("complete" in states for states in journal.attempts())
