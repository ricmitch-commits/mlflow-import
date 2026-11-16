"""Immutable-map generations through real prepared recovery on localhost only."""

import hashlib
import json
import urllib.request
from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import urlsplit

import mlflow
import pytest
import requests
from mlflow.exceptions import MlflowException

from mlflow_import import resources
from mlflow_import.contracts import RedactionMode
from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow import gateway as gateway_module
from mlflow_import.mlflow import reconciliation as rec
from mlflow_import.mlflow.attempts import AttemptJournal
from mlflow_import.mlflow.gateway import OwnedTrace, build_gateway
from mlflow_import.retry import RetryPolicy
from tests.integration import import_runner
from tests.integration.import_runner import run_real_import, turn_entries
from tests.integration.server import live_mlflow_server, unique_experiment_name
from tests.support.import_fakes import build_current_source_db

pytestmark = [pytest.mark.integration, pytest.mark.profile]


@pytest.fixture(scope="module")
def map_server():
    with live_mlflow_server(requested_uri="") as server:
        assert server.process is not None and server.uri.startswith("http://127.0.0.1:")
        assert mlflow.__version__ == "3.17.2"
        yield server


@pytest.fixture(autouse=True)
def localhost_only(monkeypatch):
    send, open_url = requests.Session.send, urllib.request.OpenerDirector.open

    def guarded_send(session, request, **kwargs):
        assert urlsplit(request.url).hostname == "127.0.0.1"
        return send(session, request, **kwargs)

    def guarded_open(opener, fullurl, *args, **kwargs):
        assert urlsplit(getattr(fullurl, "full_url", fullurl)).hostname == "127.0.0.1"
        return open_url(opener, fullurl, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "send", guarded_send)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", guarded_open)


@pytest.mark.parametrize("seed,retry_upload,interrupt_recovery", [
    ("autofilled_complete", False, False),
    ("autofilled_complete", True, False),
    ("autofilled_complete", False, True),
    ("uploaded_unrecorded", False, False),
    ("digest_first_tag", False, False),
    ("valid_winner_with_loser", False, False),
])
def test_prepared_map_generation_preserves_immutable_artifacts(
    map_server, tmp_path, monkeypatch, seed, retry_upload, interrupt_recovery,
):
    server = map_server
    source = build_current_source_db(tmp_path, sessions=[("map-session", 1000)], messages=[
        ("u1", "map-session", "user", 1000, json.dumps({
            "content": [{"type": "image", "url": "synthetic-image"}],
        })),
        ("a1", "map-session", "assistant", 2000, json.dumps({
            "finish": "stop", "content": [{"type": "tool", "tool": "synthetic", "call_id": "call-1",
                "state": {"input": {"query": "synthetic"}, "output": {"result": "synthetic"},
                          "status": "completed"}}],
        })),
        ("u2", "map-session", "user", 3000, '{"text":"synthetic prompt"}'),
        ("a2", "map-session", "assistant", 4000, '{"text":"synthetic response","finish":"stop"}'),
    ])
    experiment_name = unique_experiment_name("immutable-map")
    phase = "seed"
    captured = []
    created_requests = []
    prepared = {}
    recorded = []
    uploaded = []
    interrupted = False
    publish_state = AttemptJournal.publish
    runs = []

    def journal_publish(journal, state, values):
        nonlocal interrupted
        if state == "prepared":
            prepared.update(values)
        if state == "ready_to_retire":
            for key in ("attempt_id", "desired", "prior_ids", "session_object", "pricing_bases"):
                assert values[key] == prepared[key]
            journal.validate({**values, "state": state, "schema_version": "mlflow-import.attempt.v1"})
            if phase == "seed":
                if seed == "valid_winner_with_loser":
                    duplicate = captured[0][2](captured[0][1])
                    assert duplicate.trace_id != captured[0][0].trace_id
                raise ItemImportError("synthetic_before_ready", "synthetic interruption")
            if interrupt_recovery and not interrupted:
                interrupted = True
                raise ItemImportError("synthetic_before_ready", "synthetic interruption")
        return publish_state(journal, state, values)

    monkeypatch.setattr(AttemptJournal, "publish", journal_publish)
    monkeypatch.setattr(import_runner, "IMMEDIATE_RETRY_POLICY", RetryPolicy(
        max_attempts=2, initial_delay_seconds=0, max_delay_seconds=0,
    ))

    def factory(dry_run):
        gateway = build_gateway(server.uri, dry_run=dry_run)
        create, verify = gateway.create_trace, gateway.verify_trace
        publish = gateway.publish_experiment_file
        record, retire = gateway.record_artifact_verification, gateway.retire_trace

        def create_trace(request):
            assert request.root_tags["opencode.artifact_path"] == request.artifact_relative_path
            assert request.root_tags["mlflow_import.turn_map"] == request.artifact_relative_path
            if phase == "seed" and not captured and seed == "autofilled_complete":
                with monkeypatch.context() as patch:
                    patch.setattr(gateway_module, "_native_historical_preview", lambda value: value)
                    created = create(request)
            else:
                created = create(request)
            # Retain a bounded synthetic request after the frozen snapshot closes.
            children = tuple(request.child_specs())
            frozen = replace(request, child_specs=lambda children=children: iter(children))
            if phase == "seed":
                captured.append((created, frozen, create))
            created_requests.append(frozen)
            return created

        def verify_trace(created, request):
            if phase == "seed" and seed == "autofilled_complete" and created.trace_id == captured[0][0].trace_id:
                trace = gateway._get_trace(created.trace_id)
                old_projection = replace(request, request_preview=trace.info.request_preview,
                                         response_preview=trace.info.response_preview)
                with monkeypatch.context() as patch:
                    patch.setattr(gateway_module, "_native_historical_preview", lambda value: value)
                    return verify(created, old_projection)
            return verify(created, request)

        def record_digest(trace_id, digest, size):
            recorded.append((phase, trace_id))
            if phase == "seed" and seed in {"uploaded_unrecorded", "digest_first_tag"}:
                if trace_id == captured[0][0].trace_id:
                    if seed == "digest_first_tag":
                        gateway._client.set_trace_tag(trace_id, "opencode.artifact_sha256", digest)
                    raise ItemImportError("synthetic_digest_interruption", "synthetic interruption")
            return record(trace_id, digest, size)

        def retire_trace(experiment_id, trace_id, **kwargs):
            if phase == "seed" and seed in {"uploaded_unrecorded", "digest_first_tag"}:
                raise ItemImportError("synthetic_rollback_interruption", "synthetic interruption")
            return retire(experiment_id, trace_id, **kwargs)

        def publish_file(experiment_id, path, local, **kwargs):
            result = publish(experiment_id, path, local, **kwargs)
            if phase == "recover" and "/turns/" in path:
                uploaded.append((path, result.sha256, result.size, kwargs["trace_id"]))
                if retry_upload and len(uploaded) == 1:
                    raise MlflowException("synthetic post-publication disconnect", "TEMPORARILY_UNAVAILABLE")
            return result

        gateway.create_trace, gateway.verify_trace = create_trace, verify_trace
        gateway.record_artifact_verification, gateway.retire_trace = record_digest, retire_trace
        gateway.publish_experiment_file = publish_file
        if phase == "unchanged":
            def forbidden(*args, **kwargs):
                pytest.fail("unchanged map recovery must not mutate")
            for method in ("create_trace", "retire_trace", "record_artifact_verification",
                           "upload_named_json", "publish_experiment_file", "reconcile_tags"):
                setattr(gateway, method, forbidden)
            for method in ("set_trace_tag", "delete_trace_tag", "delete_traces", "create_experiment"):
                setattr(gateway._client, method, forbidden)
        return gateway

    def run():
        result = run_real_import(server_uri=server.uri, source=source, experiment_name=experiment_name,
                                 tmp_root=tmp_path, gateway_factory=factory)
        runs.append(result)
        return result

    def artifact(gateway, path):
        with resources.temporary_directory(prefix="immutable-map-read-") as directory:
            local = resources.copy_artifact(
                gateway.experiment_repository(experiment_id), path, directory, 8 * 1024 * 1024,
            )
            with resources.open_reader(local) as handle:
                return handle.read(8 * 1024 * 1024 + 1)

    try:
        # Seed the historical fixed-name registry, including an actual uploaded map.
        with monkeypatch.context() as patch:
            patch.setattr(rec, "_bind_physical_turn_map", lambda request: request)
            initial = run()
        assert initial.exit_status == 1 and len(captured) == 2, initial.report
        old_created, old_request, _ = captured[0]
        good_created, good_request, _ = captured[1]
        experiment_id = old_request.experiment_id
        gateway = build_gateway(server.uri, dry_run=True)
        old_map = artifact(gateway, old_request.artifact_relative_path)
        good_map = artifact(gateway, good_request.artifact_relative_path)
        assert json.loads(old_map)["identity_map"]["trace_id"] == old_created.trace_id
        old = server.client.get_trace(old_created.trace_id, display=False)
        if seed == "autofilled_complete":
            assert old.info.request_preview or old.info.response_preview
            assert old.info.tags["opencode.artifact_sha256"] == hashlib.sha256(old_map).hexdigest()
            assert int(old.info.tags["opencode.artifact_size"]) == len(old_map)
            phase = "collision_control"
            with monkeypatch.context() as patch:
                patch.setattr(rec, "_bind_physical_turn_map", lambda request: request)
                collision = run()
            assert collision.exit_status == 1
            assert {turn["error_code"] for turn in turn_entries(collision.report)} == {
                "artifact_verification_failed", "session_staging_incomplete",
            }
            assert gateway.trace_exists(old_created.trace_id) and gateway.trace_exists(good_created.trace_id)
            assert artifact(gateway, old_request.artifact_relative_path) == old_map
        elif seed in {"uploaded_unrecorded", "digest_first_tag"}:
            assert "opencode.artifact_size" not in old.info.tags
            assert bool(old.info.tags.get("opencode.artifact_sha256")) == (seed == "digest_first_tag")

        phase = "recover"
        before = len(created_requests)
        recovered = run()
        if interrupt_recovery:
            assert recovered.exit_status == 1
            assert turn_entries(recovered.report)[0]["attempt_state"] == "prepared"
            creates = len(created_requests)
            recovered = run()
            assert len(created_requests) == creates
        assert recovered.exit_status == 0, recovered.report
        ids = {turn["trace_id"] for turn in turn_entries(recovered.report)}
        assert good_created.trace_id in ids
        assert artifact(gateway, old_request.artifact_relative_path) == old_map
        assert artifact(gateway, good_request.artifact_relative_path) == good_map
        expected_creates = int(seed != "valid_winner_with_loser")
        assert len(created_requests) - before == expected_creates
        if expected_creates:
            replacement_id = next(trace_id for trace_id in ids if trace_id != good_created.trace_id)
            replacement = server.client.get_trace(replacement_id, display=False)
            path = replacement.info.tags["opencode.artifact_path"]
            assert path != old_request.artifact_relative_path
            assert replacement.info.tags["mlflow_import.turn_map"] == path
            assert json.loads(artifact(gateway, path))["identity_map"]["trace_id"] == replacement_id
            assert [trace_id for stage, trace_id in recorded if stage == "recover"] == [replacement_id]
            assert not gateway.trace_exists(old_created.trace_id)
            physical = created_requests[-1]
            assert replace(physical, artifact_relative_path=old_request.artifact_relative_path,
                           root_tags=old_request.root_tags, child_specs=old_request.child_specs) == old_request
        else:
            assert old_created.trace_id in ids
            assert not [trace_id for stage, trace_id in recorded if stage == "recover"]
        if retry_upload:
            assert len(uploaded) == 2 and uploaded[0] == uploaded[1]
        assert all(turn["attempt_state"] == "complete" for turn in turn_entries(recovered.report))
        phase = "unchanged"
        again = run()
        assert again.exit_status == 0, again.report
        assert again.report["summary"]["unchanged_turns"] == 2
        assert {turn["trace_id"] for turn in turn_entries(again.report)} == ids
    finally:
        for result in runs:
            result.result.close()


@pytest.mark.parametrize("alias", ["present", "missing", "contradictory"])
def test_ambiguous_create_adopts_verified_map_at_recorded_location(map_server, tmp_path, monkeypatch, alias):
    # Seed a fully verified real map, then simulate a lost create response whose
    # operation search finds it. No source values or strict payload checks are mocked.
    source = build_current_source_db(tmp_path, sessions=[("survivor", 1000)], messages=[
        ("u", "survivor", "user", 1000, '{"text":"synthetic prompt"}'),
        ("a", "survivor", "assistant", 2000, '{"text":"synthetic reply","finish":"stop"}'),
    ])
    retained = []

    def factory(dry_run):
        gateway = build_gateway(map_server.uri, dry_run=dry_run)
        create = gateway.create_trace
        def capture(request):
            children = tuple(request.child_specs())
            retained.append(replace(request, child_specs=lambda: iter(children)))
            return create(request)
        gateway.create_trace = capture
        return gateway

    # Keep the survivor staged: a real ambiguous same-attempt create expects
    # staging tags, not the completed trace's different state tag.
    publish = AttemptJournal.publish
    def stop(journal, state, values):
        if state == "ready_to_retire":
            raise ItemImportError("synthetic_before_ready", "synthetic interruption")
        return publish(journal, state, values)
    with monkeypatch.context() as patch:
        patch.setattr(AttemptJournal, "publish", stop)
        seeded = run_real_import(server_uri=map_server.uri, source=source,
            experiment_name=unique_experiment_name("map-survivor"), tmp_root=tmp_path, gateway_factory=factory)
    try:
        assert seeded.exit_status == 1
        request = retained[0]
        trace_id = turn_entries(seeded.report)[0]["trace_id"]
        gateway = build_gateway(map_server.uri, dry_run=True)
        if alias == "missing":
            map_server.client.delete_trace_tag(trace_id, "mlflow_import.turn_map")
        elif alias == "contradictory":
            map_server.client.set_trace_tag(trace_id, "mlflow_import.turn_map", "different.json")
        actual = map_server.client.get_trace(trace_id, display=False)
        view = OwnedTrace(trace_id, actual.info.request_time, actual.info.tags, actual.info.trace_metadata)
        work = rec._IdentityWork(
            key=rec.OwnershipKey(request.native_session_id, rec.request_ownership(request)["opencode.turn_key"]),
            entry=SimpleNamespace(), replay=SimpleNamespace(), request=request, redaction_mode=RedactionMode.VERBATIM,
        )
        def ambiguous(proposed):
            assert proposed.artifact_relative_path != request.artifact_relative_path
            assert proposed.expected_operation_key == request.expected_operation_key
            raise MlflowException("synthetic lost response", "TEMPORARILY_UNAVAILABLE")
        monkeypatch.setattr(gateway, "create_trace", ambiguous)
        monkeypatch.setattr(gateway, "iter_owned_traces", lambda experiment_id: iter([view]))
        monkeypatch.setattr(gateway, "publish_experiment_file", lambda *args, **kwargs: pytest.fail("survivor rewrite"))
        if alias == "contradictory":
            with pytest.raises(ItemImportError) as caught:
                rec._create_with_ambiguous_recovery(
                    rec.ReconciliationPlan(), gateway, work, RetryPolicy(max_attempts=1),
                )
            assert caught.value.code == "ownership_conflict"
        else:
            created, survivor = rec._create_with_ambiguous_recovery(
                rec.ReconciliationPlan(), gateway, work, RetryPolicy(max_attempts=1),
            )
            assert created is None and survivor.trace_id == trace_id
        assert gateway.trace_exists(trace_id)
    finally:
        seeded.result.close()
