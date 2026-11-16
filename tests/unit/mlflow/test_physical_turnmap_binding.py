"""Physical map placement, retries, and strict adoption of recorded pointers."""

import json
import re
from dataclasses import replace
from pathlib import PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException

from mlflow_import.contracts import RedactionMode
from mlflow_import.errors import ItemImportError, PartialTraceCreateError
from mlflow_import.mlflow import reconciliation as rec
from mlflow_import.mlflow.capabilities import CapabilityRequest
from mlflow_import.mlflow.gateway import ArtifactVerification, CreatedTrace, OwnedTrace
from mlflow_import.retry import RetryPolicy
from tests.support.fake_gateway import trace_spec

pytestmark = pytest.mark.unit

PATH = "mlflow-import/v2/opencode/namespace/session/verbatim/turns/turn/render-attempt.json"
POLICY = RetryPolicy(max_attempts=2, initial_delay_seconds=0, max_delay_seconds=0)


def request():
    return trace_spec(artifact_relative_path=PATH, root_tags={
        "opencode.artifact_path": PATH,
        "mlflow_import.turn_map": PATH,
        "mlflow_import.attempt_id": "d" * 32,
        "mlflow_import.attempt_state": "staged",
        "opencode.turn_index": "1",
    })


def work_for(mapped):
    key = rec.OwnershipKey(mapped.native_session_id, mapped.metadata["opencode.turn_key"])
    return rec._IdentityWork(
        key=key, entry=SimpleNamespace(session_id=key.session_id, turn_key=key.turn_key, turn_index=1),
        replay=Mock(), request=mapped, redaction_mode=RedactionMode.VERBATIM,
    )


def candidate_for(mapped, trace_id="old"):
    return rec.OwnedTraceCandidate.from_owned_trace(
        OwnedTrace(trace_id, 1, mapped.root_tags, mapped.metadata), mapped.experiment_id,
    )


@pytest.mark.parametrize("client_request_id", [None, "source-request"])
def test_binding_changes_only_physical_placement(client_request_id):
    logical = replace(request(), client_request_id=client_request_id)
    original_tags = dict(logical.root_tags)
    first, second = (rec._bind_physical_turn_map(logical) for _ in range(2))
    assert first.artifact_relative_path != second.artifact_relative_path
    for physical in (first, second):
        path = physical.artifact_relative_path
        assert PurePosixPath(path).parent == PurePosixPath(PATH).parent
        assert re.fullmatch(re.escape(PATH[:-5]) + r"-create-[a-f0-9]{32}\.json", path)
        assert len(PurePosixPath(path).name) < 255
        assert physical.root_tags["opencode.artifact_path"] == path
        assert physical.root_tags["mlflow_import.turn_map"] == path
        assert replace(physical, artifact_relative_path=PATH, root_tags=original_tags) == logical
    assert logical.root_tags == original_tags
    assert logical.artifact_relative_path == PATH


def test_legacy_placement_is_not_allocated(monkeypatch):
    logical = trace_spec()
    monkeypatch.setattr(rec.uuid, "uuid4", Mock(side_effect=AssertionError("unexpected allocation")))
    assert rec._bind_physical_turn_map(logical) is logical


def test_actual_create_retries_use_fresh_paths_without_stacking_suffixes():
    logical = request()
    work = work_for(logical)
    gateway = Mock()
    created = CreatedTrace("new", "root", {}, {}, "OK")
    gateway.create_trace.side_effect = [MlflowException("synthetic", "TEMPORARILY_UNAVAILABLE"), created]
    gateway.iter_owned_traces.return_value = iter(())
    assert rec._create_with_ambiguous_recovery(rec.ReconciliationPlan(), gateway, work, POLICY) == (created, None)
    attempted = [call.args[0] for call in gateway.create_trace.call_args_list]
    assert len(attempted) == 2
    assert attempted[0].artifact_relative_path != attempted[1].artifact_relative_path
    assert all(mapped.artifact_relative_path.count("-create-") == 1 for mapped in attempted)
    assert all(mapped.expected_operation_key == logical.expected_operation_key for mapped in attempted)
    assert work.request is attempted[-1]
    gateway.iter_owned_traces.assert_called_once_with(logical.experiment_id)


@pytest.mark.parametrize("matches", [True, False])
def test_ambiguous_survivor_is_verified_by_operation_not_new_path(matches):
    logical = request()
    work = work_for(logical)
    gateway = Mock()
    gateway.create_trace.side_effect = MlflowException("synthetic", "TEMPORARILY_UNAVAILABLE")
    gateway.iter_owned_traces.return_value = [OwnedTrace("old", 1, logical.root_tags, logical.metadata)]
    gateway.verify_owned_candidate.return_value = rec.CandidateVerification(
        "old", matches=matches, reason=None if matches else "artifact_digest_missing",
    )
    plan = rec.ReconciliationPlan()
    if matches:
        created, winner = rec._create_with_ambiguous_recovery(plan, gateway, work, POLICY)
        assert created is None and winner.trace_id == "old"
        gateway.verify_candidate_payload.assert_called_once_with(winner, work.request)
    else:
        with pytest.raises(ItemImportError) as caught:
            rec._create_with_ambiguous_recovery(plan, gateway, work, POLICY)
        assert caught.value.code == "ownership_conflict"
        gateway.verify_candidate_payload.assert_not_called()
    assert work.request.artifact_relative_path != PATH
    gateway.create_trace.assert_called_once()
    gateway.retire_trace.assert_not_called()


def test_partial_create_cleanup_keeps_bound_request():
    work = work_for(request())
    gateway = Mock()
    created = CreatedTrace("partial", "root", {}, {}, "ERROR")
    error = PartialTraceCreateError(created, ItemImportError("synthetic_failure", "synthetic"))
    gateway.create_trace.side_effect = error
    with pytest.raises(PartialTraceCreateError) as caught:
        rec._create_with_ambiguous_recovery(rec.ReconciliationPlan(), gateway, work, POLICY)
    assert caught.value.rollback_retired
    assert work.request is gateway.create_trace.call_args.args[0]
    assert work.request.artifact_relative_path != PATH
    gateway.retire_trace.assert_called_once_with(
        work.request.experiment_id, created.trace_id,
        session_id=work.key.session_id, turn_key=work.key.turn_key,
        expected_metadata=rec._request_creation_metadata(work.request),
    )


@pytest.mark.parametrize("replacement", [False, True])
def test_import_and_replacement_keep_bound_path_across_upload_retry(replacement):
    logical = request()
    work = work_for(logical)
    gateway = Mock()
    gateway.create_trace.return_value = CreatedTrace("new", "root", {}, {}, "OK")
    uploaded = []

    def upload(replay, created, path, mode):
        uploaded.append((created, path))
        if len(uploaded) == 1:
            raise MlflowException("synthetic publication ambiguity", "TEMPORARILY_UNAVAILABLE")
        return ArtifactVerification(created.trace_id, path, "f" * 64, 2)

    dependencies = rec.ReconciliationDependencies(
        replay=lambda entry: work.replay, map_trace=lambda turn: logical,
        artifact_snapshot=lambda turn: logical.artifact_snapshot_fingerprint,
        upload_artifact=upload, defer_retirement=True,
    )
    plan = rec.ReconciliationPlan()
    action = rec.ReconciliationAction(
        action="updated" if replacement else "imported", turn=work.entry,
        candidates=(candidate_for(logical),) if replacement else (),
    )
    if replacement:
        result = rec._replace_identity(plan, gateway, dependencies, POLICY, work, action)
        assert result.pending_retirement_ids == ("old",)
    else:
        plan.imports.append(action)
        rec._decide_imports(plan, gateway, dependencies, POLICY)
        result = plan.results[0]
    assert result.action == ("updated" if replacement else "imported")
    physical = gateway.create_trace.call_args.args[0]
    assert physical.artifact_relative_path != PATH
    assert uploaded == [(gateway.create_trace.return_value, physical.artifact_relative_path)] * 2
    gateway.verify_trace.assert_called_once_with(gateway.create_trace.return_value, physical)
    gateway.verify_named_json.assert_called_once_with("new", physical.artifact_relative_path, "f" * 64, 2)
    gateway.retire_trace.assert_not_called()


@pytest.mark.parametrize("alias", ["present", "missing", "contradictory"])
def test_reused_payload_adopts_recorded_pointers_without_masking_corruption(
    fake_mlflow_client, tmp_path, monkeypatch, alias,
):
    gateway = fake_mlflow_client.gateway()
    assert gateway.probe(CapabilityRequest()).is_compatible
    original = request()
    tags = dict(original.root_tags)
    if alias == "missing":
        tags.pop("mlflow_import.turn_map")
    elif alias == "contradictory":
        tags["mlflow_import.turn_map"] = PATH + ".different"
    original = replace(original, root_tags=tags)
    created = gateway.create_trace(original)
    stored = fake_mlflow_client._traces[created.trace_id]
    candidate = rec.OwnedTraceCandidate.from_owned_trace(
        OwnedTrace(created.trace_id, 1, stored.info.tags, stored.info.trace_metadata), original.experiment_id,
    )
    artifact = tmp_path / "map.json"
    artifact.write_text(json.dumps({"identity_map": {"trace_id": created.trace_id}}))
    monkeypatch.setattr(gateway, "experiment_repository", lambda experiment_id: object())
    monkeypatch.setattr("mlflow_import.resources.copy_artifact", lambda *args: artifact)
    proposed = rec._bind_physical_turn_map(request())
    mutations = list(fake_mlflow_client.mutations)
    if alias == "contradictory":
        with pytest.raises(ItemImportError, match="the recorded turn map pointers disagree"):
            gateway.verify_candidate_payload(candidate, proposed)
    else:
        gateway.verify_candidate_payload(candidate, proposed)
        stored.data.spans[0].inputs = {"prompt": "corrupted"}
        with pytest.raises(ItemImportError, match="complete root inputs did not round-trip"):
            gateway.verify_candidate_payload(candidate, proposed)
    assert fake_mlflow_client.mutations == mutations
