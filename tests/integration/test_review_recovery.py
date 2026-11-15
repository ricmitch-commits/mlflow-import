"""Disposable-target regressions for review findings."""

import sqlite3
from dataclasses import replace

import mlflow
import pytest

from mlflow_import.compatibility import identity_projection
from mlflow_import.config import ImportConfig
from mlflow_import.errors import ItemImportError
from mlflow_import.importer import build_dependencies, exit_status, run_import
from mlflow_import.mlflow.attempts import AttemptJournal
from tests.integration.server import unique_experiment_name
from tests.integration.test_attempt_boundaries import _source

pytestmark = pytest.mark.integration


def test_unrelated_foreign_proof_cannot_authorize_retirement(live_server, tmp_path):
    config = ImportConfig(unique_experiment_name("scope-exploit"), _source(tmp_path), tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    result = run_import(config, dependencies)
    assert exit_status(result) == 0, result.report
    experiment = result.report["experiment"]["id"]
    current = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    unrelated = mlflow.start_span_no_context(name="unrelated-synthetic", experiment_id=experiment)
    unrelated.end()
    trace = live_server.client.get_trace(unrelated.trace_id, display=False)
    gateway = dependencies.gateway_factory(False)
    retirement = {
        "trace_id": unrelated.trace_id,
        "session_id": "boundary",
        "turn_key": "",
        "foreign": True,
        "identity": identity_projection(trace.info.trace_metadata, trace.info.tags),
    }
    ready = {
        "qualified_id": current.info.tags["mlflow_import.qualified_session_id"],
        "session_id": "boundary",
        "policy": "overwrite",
        "instance_id": "",
    }
    with pytest.raises(ItemImportError):
        gateway.authorize_retirement(experiment, retirement, ready)
    assert live_server.client.get_trace(unrelated.trace_id, display=False).trace_id == unrelated.trace_id


def test_mutable_neutral_tags_refresh_without_replacing_trace_or_map(live_server, tmp_path):
    config = ImportConfig(unique_experiment_name("mutable-neutral"), _source(tmp_path), tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    first = run_import(config, dependencies)
    trace_id = first.report["sessions"][0]["turns"][0]["trace_id"]
    original = live_server.client.get_trace(trace_id, display=False).info.tags
    keys = ("mlflow_import.turn_index", "mlflow_import.source_revision", "mlflow_import.metadata_fingerprint")
    for key in keys:
        live_server.client.set_trace_tag(trace_id, key, "stale")
    live_server.client.set_trace_tag(trace_id, "mlflow_import.current.removed", "stale")
    result = run_import(config, dependencies)
    assert exit_status(result) == 0, result.report
    assert result.report["sessions"][0]["turns"][0]["trace_id"] == trace_id
    actual = live_server.client.get_trace(trace_id, display=False).info.tags
    assert all(actual[key] == original[key] for key in keys)
    assert actual["opencode.artifact_path"] == original["opencode.artifact_path"]
    assert "mlflow_import.current.removed" not in actual


def test_missing_session_reference_is_repaired_before_success(live_server, tmp_path):
    config = ImportConfig(
        unique_experiment_name("missing-session-ref"), _source(tmp_path), tracking_uri=live_server.uri
    )
    dependencies = build_dependencies(config)
    first = run_import(config, dependencies)
    old = first.report["sessions"][0]["turns"][0]["trace_id"]
    for key in (
        "mlflow_import.session_artifact",
        "mlflow_import.session_artifact_sha256",
        "mlflow_import.session_artifact_size",
    ):
        live_server.client.delete_trace_tag(old, key)
    result = run_import(config, dependencies)
    assert exit_status(result) == 0, result.report
    turn = result.report["sessions"][0]["turns"][0]
    assert turn["reason"] != "unchanged"
    tags = live_server.client.get_trace(turn["trace_id"], display=False).info.tags
    assert tags["mlflow_import.session_artifact"] and tags["mlflow_import.session_artifact_sha256"]


@pytest.mark.parametrize("policy", ["ignore", "duplicate"])
@pytest.mark.parametrize("boundary", ["prepared", "ready_to_retire"])
def test_changed_to_zero_recovers_before_policy_shortcut(live_server, tmp_path, monkeypatch, policy, boundary):
    source = _source(tmp_path)
    config = ImportConfig(unique_experiment_name("zero-policy-recovery"), source, tracking_uri=live_server.uri)
    object.__setattr__(config, "_on_conflict", policy)
    dependencies = build_dependencies(config)
    publish = AttemptJournal.publish
    prefixes = []

    def interrupt(journal, state, payload):
        prefixes.append(journal.prefix.removesuffix("/attempts"))
        result = publish(journal, state, payload)
        if state == boundary:
            raise ItemImportError("synthetic_interruption", "synthetic failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(AttemptJournal, "publish", interrupt)
        failed = run_import(config, dependencies)
    assert exit_status(failed) == 1
    session = failed.report["sessions"][0]
    assert session["attempt_id"] and session["attempt_state"] == boundary
    assert bool(session["import_instance_id"]) == (policy == "duplicate")
    from mlflow_import.reporting import render_human

    assert "synthetic_interruption" in render_human(failed)
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE session_message SET type='system'")
    result = run_import(config, dependencies)
    assert exit_status(result) == 0, result.report
    gateway = dependencies.gateway_factory(False)
    journal = AttemptJournal(gateway, result.report["experiment"]["id"], prefixes[0])
    assert all("complete" in states or "aborted" in states for states in journal.attempts())
    assert result.report["sessions"][0]["attempt_state"] == ("complete" if boundary == "ready_to_retire" else "aborted")


@pytest.mark.parametrize("marker_location", ["metadata", "tags"])
@pytest.mark.parametrize("contradiction", [False, True])
def test_historical_marker_migrates_with_full_identity_qualification(
    live_server, tmp_path, marker_location, contradiction
):
    import json
    from pathlib import Path

    from mlflow.tracing.client import TracingClient

    from mlflow_import.mlflow.capabilities import CapabilityRequest
    from tests.support.import_fakes import build_current_source_db

    fixture = json.loads((Path(__file__).parents[1] / "fixtures/payloads/historical-36ed44f.json").read_text())
    metadata = fixture["trace"]["metadata"]
    native = metadata["opencode.session_id"]
    source = build_current_source_db(
        tmp_path,
        sessions=[(native, 1000)],
        messages=[
            ("u", native, "user", 1, '{"text":"synthetic historical prompt"}'),
            ("a", native, "assistant", 2, '{"text":"synthetic historical reply"}'),
        ],
    )
    config = ImportConfig(unique_experiment_name("historical-marker"), source, tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    experiment = live_server.client.create_experiment(config.experiment_name)
    span = mlflow.start_span_no_context(name="historical-synthetic", experiment_id=experiment)
    span.end()
    info = live_server.client.get_trace(span.trace_id, display=False).info
    artifact_path = (
        f"opencode-mlflow/v1/sessions/{native}/turns/{metadata['opencode.turn_key']}/{'b' * 64}-{'d' * 64}.json"
    )
    tags = {**info.tags, "opencode.artifact_path": artifact_path}
    if marker_location == "tags":
        tags["opencode.importer"] = metadata.pop("opencode.importer")
    if contradiction:
        tags["opencode.session_id"] = "contradictory-session"
    TracingClient(tracking_uri=live_server.uri).start_trace(replace(info, trace_metadata=metadata, tags=tags))
    local = tmp_path / "historical-artifact.json"
    local.write_text(json.dumps(fixture["artifact"]))
    gateway = dependencies.gateway_factory(False)
    gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True), dry_run=False)
    verification = gateway.upload_named_json(span.trace_id, artifact_path, local)
    gateway.record_artifact_verification(span.trace_id, verification.sha256, verification.size)
    result = run_import(config, dependencies)
    if contradiction:
        assert exit_status(result) != 0
        assert live_server.client.get_trace(span.trace_id, display=False).trace_id == span.trace_id
        assert result.report["summary"]["retired_traces"] == 0
        return
    assert exit_status(result) == 0, result.report
    traces = live_server.client.search_traces(locations=[experiment], include_spans=False)
    assert len(traces) == 1 and traces[0].trace_id != span.trace_id
    assert traces[0].info.tags["mlflow_import.importer"] == "mlflow-import"
    assert result.report["summary"]["retired_traces"] == 1
