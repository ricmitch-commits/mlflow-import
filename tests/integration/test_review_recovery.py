"""Disposable-target regressions for review findings."""

import sqlite3
from dataclasses import replace

import mlflow
import pytest

from mlflow_import.compatibility import decode_ownership, identity_projection, importer_marker
from mlflow_import.config import ImportConfig
from mlflow_import.errors import ConfigurationError, ItemImportError
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
def test_historical_marker_is_recognized_but_unqualified_migration_is_blocked(
    live_server, tmp_path, monkeypatch, marker_location, contradiction
):
    import hashlib
    import json
    from functools import wraps
    from pathlib import Path
    from urllib.parse import unquote, urlsplit

    from mlflow import MlflowClient
    from mlflow.tracing.client import TracingClient

    from mlflow_import.mlflow.capabilities import CapabilityRequest
    from mlflow_import.mlflow.gateway import MlflowGateway
    from mlflow_import.mlflow.reconciliation import OwnedTraceCandidate
    from mlflow_import.mlflow.trace_envelope import PROOF_TAG_KEYS
    from tests.support.import_fakes import build_current_source_db

    assert live_server.process is not None, "legacy migration regression requires an owned disposable server"
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
    tracing = TracingClient(tracking_uri=live_server.uri)
    info = tracing.get_trace_info(span.trace_id)
    artifact_path = (
        f"opencode-mlflow/v1/sessions/{native}/turns/{metadata['opencode.turn_key']}/{'b' * 64}-{'d' * 64}.json"
    )
    tags = {**info.tags, "opencode.artifact_path": artifact_path}
    if marker_location == "tags":
        tags["opencode.importer"] = metadata.pop("opencode.importer")
    if contradiction:
        tags["opencode.session_id"] = "contradictory-session"
    tracing.start_trace(replace(info, trace_metadata=metadata, tags=tags))
    local = tmp_path / "historical-artifact.json"
    local.write_text(json.dumps(fixture["artifact"]))
    gateway = dependencies.gateway_factory(False)
    gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True), dry_run=False)
    verification = gateway.upload_named_json(span.trace_id, artifact_path, local)
    gateway.record_artifact_verification(span.trace_id, verification.sha256, verification.size)
    before = tracing.get_trace_info(span.trace_id)
    assert PROOF_TAG_KEYS.isdisjoint(before.tags), "the historical fixture must remain genuinely unqualified"
    assert importer_marker(before.tags, before.trace_metadata) == "mlflow-import"
    if contradiction:
        with pytest.raises(ConfigurationError) as conflict:
            decode_ownership(before.tags, before.trace_metadata)
        assert conflict.value.code == "ownership_conflict"
    else:
        ownership = decode_ownership(before.tags, before.trace_metadata)
        assert ownership["mlflow_import.importer"] == "mlflow-import"
        assert ownership["mlflow_import.original_importer"] == "opencode-mlflow"
        assert ownership["mlflow_import.session_id"] == native
        assert ownership["opencode.mapping_version"] == "mapping-v4"
        owned = next(item for item in gateway.iter_owned_traces(experiment) if item.trace_id == span.trace_id)
        assert OwnedTraceCandidate.from_owned_trace(owned, experiment).ownership_complete

    artifact_uri = urlsplit(live_server.client.get_experiment(experiment).artifact_location)
    assert artifact_uri.scheme == "file"
    artifact_root = Path(unquote(artifact_uri.path))

    def artifact_snapshot():
        return {
            str(path.relative_to(artifact_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in artifact_root.rglob("*") if path.is_file()
        }

    artifacts_before = artifact_snapshot()
    assert artifacts_before, "the legacy artifact must exist before the blocked import"
    forbidden_calls = []

    def forbid(name, original):
        @wraps(original)
        def forbidden(*args, **kwargs):
            forbidden_calls.append(name)
            raise AssertionError("unqualified historical migration reached " + name)
        return forbidden

    with monkeypatch.context() as patch:
        # Recognition uses only metadata. Neither contradictory ownership nor
        # absent resource proof authorizes speculative payload reads, restamping,
        # automatic replacement, or journal/artifact/retirement mutation.
        for owner in (MlflowClient, TracingClient):
            patch.setattr(owner, "get_trace", forbid(owner.__name__ + ".get_trace", owner.get_trace))
        for name in ("create_experiment", "set_trace_tag", "delete_trace_tag", "delete_traces"):
            patch.setattr(MlflowClient, name, forbid("MlflowClient." + name, getattr(MlflowClient, name)))
        for name in ("start_trace", "log_spans"):
            patch.setattr(TracingClient, name, forbid("TracingClient." + name, getattr(TracingClient, name)))
        for name in (
            "create_trace", "upload_named_json", "publish_experiment_file", "record_artifact_verification",
            "reconcile_tags", "retire_trace", "retire_conflict",
        ):
            patch.setattr(MlflowGateway, name, forbid("MlflowGateway." + name, getattr(MlflowGateway, name)))
        patch.setattr(AttemptJournal, "publish", forbid("AttemptJournal.publish", AttemptJournal.publish))
        result = run_import(config, dependencies)

    assert exit_status(result) == (2 if contradiction else 1), result.report
    assert result.error is not None
    assert result.error.code == ("identity_conflict" if contradiction else "trace_resource_profile_unverified")
    assert forbidden_calls == []
    assert result.report["summary"]["retired_traces"] == 0
    assert result.report["summary"]["turns_imported"] == 0
    assert result.report["summary"]["turns_updated"] == 0
    after = tracing.get_trace_info(span.trace_id)
    assert after.to_dict() == before.to_dict()
    assert PROOF_TAG_KEYS.isdisjoint(after.tags)
    assert artifact_snapshot() == artifacts_before
    traces = live_server.client.search_traces(locations=[experiment], include_spans=False)
    assert [trace.trace_id for trace in traces] == [span.trace_id]
