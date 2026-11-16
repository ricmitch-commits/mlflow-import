"""Strict native absence projection through the pinned localhost transport."""

import hashlib
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import mlflow
import pytest
from mlflow.artifacts import get_artifact_repository
from mlflow.tracing.client import TracingClient

from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.attempts import AttemptJournal
from mlflow_import.mlflow.capabilities import CapabilityRequest
from mlflow_import.mlflow.gateway import MlflowGateway, TraceChildRequest, TraceCreateRequest, build_gateway
from mlflow_import.reporting import render_json
from tests.integration.import_runner import run_real_import, turn_entries
from tests.integration.server import live_mlflow_server, unique_experiment_name
from tests.support.import_fakes import build_current_source_db

pytestmark = [pytest.mark.integration, pytest.mark.profile]


@pytest.fixture(scope="module")
def preview_server():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLFLOW_TRACKING_URI", "http://ambient.invalid:1")
        with live_mlflow_server(requested_uri="") as server:
            assert server.process is not None
            assert server.uri.startswith("http://127.0.0.1:")
            assert mlflow.__version__ == "3.17.2"
            yield server


def _request(experiment_id, prompt, response, status="OK"):
    start = 1_794_580_115_471_000_000
    child = TraceChildRequest(
        source_key="tool", name="tool", span_type="TOOL", parent_source_key=None,
        inputs={"query": "structured request"}, outputs={"result": "structured response"},
        attributes={}, tags={}, start_time_ns=start + 100, end_time_ns=start + 1000, status="UNSET",
    )
    return TraceCreateRequest(
        client_request_id=None, experiment_id=experiment_id, name="synthetic root",
        native_session_id="preview-session",
        root_inputs={"messages": [{"role": "user", "content": [{"type": "image", "url": "synthetic"}]}]},
        root_outputs={"tool_calls": [{"name": "synthetic", "arguments": {"query": "structured"}}]},
        root_attributes={}, root_tags={}, metadata={}, start_time_ns=start, end_time_ns=start + 2_000_000,
        child_specs=lambda: iter((child,)), expected_operation_key="synthetic",
        artifact_snapshot_fingerprint="a" * 64, artifact_relative_path="synthetic.json",
        root_status=status, historical=True, trace_name="opencode",
        request_preview=prompt, response_preview=response,
    )


@pytest.mark.parametrize("prompt,response", list(product((None, "", "literal 雪"), repeat=2)))
@pytest.mark.parametrize("status", ["OK", "ERROR"])
def test_absent_empty_and_nonempty_previews_round_trip(preview_server, monkeypatch, prompt, response, status):
    gateway = MlflowGateway(preview_server.client)
    assert gateway.probe(CapabilityRequest()).is_compatible
    experiment = gateway.resolve_experiment(unique_experiment_name("native-preview"), create=True)
    request = _request(experiment.experiment_id, prompt, response, status)
    starts = []
    start_trace = TracingClient.start_trace

    def capture(client, info):
        starts.append((info.request_preview, info.response_preview))
        return start_trace(client, info)

    monkeypatch.setattr(TracingClient, "start_trace", capture)
    created = gateway.create_trace(request)
    expected = ("" if prompt is None else prompt, "" if response is None else response)
    assert starts == [expected, expected]  # creation AND authoritative finalization
    assert (request.request_preview, request.response_preview) == (prompt, response)
    gateway.verify_trace(created, request)
    trace = preview_server.client.get_trace(created.trace_id, display=False)
    assert (trace.info.request_preview, trace.info.response_preview) == expected
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert root.inputs == request.root_inputs
    assert root.outputs == request.root_outputs
    # Full retrieval on a fresh gateway exercises the same expectation used for reuse.
    MlflowGateway(preview_server.client).verify_trace(created, request)
    for field in ("request_preview", "response_preview"):
        with pytest.raises(ItemImportError, match="the native text previews differ"):
            gateway.verify_trace(created, replace(request, **{field: "different nonempty preview"}))
    # Do not normalize the actual value: historical NULLs must be reconciled too.
    trace.info.request_preview = None
    monkeypatch.setattr(gateway, "_get_trace", lambda _: trace)
    with pytest.raises(ItemImportError, match="the native text previews differ"):
        gateway.verify_trace(created, request)


def test_unprojected_nulls_autofill_and_still_fail_strict_verification(preview_server, monkeypatch):
    gateway = MlflowGateway(preview_server.client)
    assert gateway.probe(CapabilityRequest()).is_compatible
    experiment = gateway.resolve_experiment(unique_experiment_name("preview-control"), create=True)
    request = _request(experiment.experiment_id, None, None)
    with monkeypatch.context() as patch:
        patch.setattr("mlflow_import.mlflow.gateway._native_historical_preview", lambda value: value)
        created = gateway.create_trace(request)
    trace = preview_server.client.get_trace(created.trace_id, display=False)
    assert trace.info.request_preview
    assert trace.info.response_preview
    with pytest.raises(ItemImportError, match="the native text previews differ"):
        gateway.verify_trace(created, request)


@pytest.mark.parametrize("interruption", [None, "prepared", "autofilled_prepared"])
def test_session_import_prepared_resume_and_mutation_free_rerun(preview_server, tmp_path, monkeypatch, interruption):
    # Neither message has display text, but both retain structured source content.
    user = {"content": [{"type": "image", "url": "synthetic-image"}]}
    assistant = {"finish": "stop", "content": [{
        "type": "tool", "tool": "synthetic", "call_id": "synthetic-call",
        "state": {"input": {"query": "synthetic"}, "output": {"result": "synthetic"}, "status": "completed"},
    }]}
    source = build_current_source_db(
        tmp_path, sessions=[("preview-session", 1000)], messages=[
            ("u", "preview-session", "user", 1000, json.dumps(user)),
            ("a", "preview-session", "assistant", 2000, json.dumps(assistant)),
        ],
    )
    name = unique_experiment_name("preview-session")
    runs = []
    captured = {"seed_autofill": interruption == "autofilled_prepared"}
    requests = []

    def factory(dry_run):
        gateway = build_gateway(preview_server.uri, dry_run=dry_run)
        create = gateway.create_trace

        def capture(request):
            if "mlflow_import.attempt_id" in request.root_tags:
                requests.append(request)
                assert request.request_preview is None
                assert request.response_preview is None
            if captured["seed_autofill"] and "mlflow_import.attempt_id" in request.root_tags:
                with monkeypatch.context() as patch:
                    patch.setattr("mlflow_import.mlflow.gateway._native_historical_preview", lambda value: value)
                    created = create(request)
            else:
                created = create(request)
            if "mlflow_import.attempt_id" in request.root_tags:
                captured["created"] = created
            return created

        gateway.create_trace = capture
        if captured["seed_autofill"]:
            def interrupted_rollback(*args, **kwargs):
                raise ItemImportError("injected_rollback_interruption", "synthetic interruption")

            gateway.retire_trace = interrupted_rollback
        return gateway

    def run(gateway_factory=factory):
        result = run_real_import(
            server_uri=preview_server.uri, source=source, experiment_name=name,
            tmp_root=tmp_path, gateway_factory=gateway_factory,
        )
        runs.append(result)
        return result

    publish = AttemptJournal.publish

    def interrupt(journal, state, values):
        if state == "ready_to_retire":
            raise ItemImportError("injected_preview_interruption", "synthetic interruption")
        return publish(journal, state, values)

    try:
        if interruption:
            with monkeypatch.context() as patch:
                patch.setattr(AttemptJournal, "publish", interrupt)
                failed = run()
            assert failed.exit_status == 1, failed.report
            assert turn_entries(failed.report)[0]["attempt_state"] == "prepared"
            assert len(requests) == 1
            old_id = captured["created"].trace_id
            if interruption == "autofilled_prepared":
                old = preview_server.client.get_trace(old_id, display=False)
                assert old.info.request_preview or old.info.response_preview
                assert "opencode.artifact_sha256" not in old.info.tags
            captured["seed_autofill"] = False
        first = run()
        assert first.exit_status == 0, render_json(first.result)
        trace_id = turn_entries(first.report)[0]["trace_id"]
        if interruption:
            assert (trace_id == old_id) == (interruption == "prepared")
            assert len(requests) == (1 if interruption == "prepared" else 2)
            assert all(
                request.artifact_snapshot_fingerprint == requests[0].artifact_snapshot_fingerprint
                and request.expected_operation_key == requests[0].expected_operation_key
                for request in requests
            )
        trace = preview_server.client.get_trace(trace_id, display=False)
        assert (trace.info.request_preview, trace.info.response_preview) == ("", "")
        tags = dict(trace.info.tags)
        repository = get_artifact_repository(
            preview_server.client.get_experiment(trace.info.experiment_id).artifact_location,
            tracking_uri=preview_server.uri,
        )
        retained = {}
        for path_key, digest_key in (
            ("opencode.artifact_path", "opencode.artifact_sha256"),
            ("mlflow_import.session_artifact", "mlflow_import.session_artifact_sha256"),
        ):
            data = Path(repository.download_artifacts(tags[path_key], str(tmp_path))).read_bytes()
            assert hashlib.sha256(data).hexdigest() == tags[digest_key]
            retained[path_key] = data
        session = json.loads(retained["mlflow_import.session_artifact"])
        decoded = [record.get("decoded") for record in session["records"]]
        assert user in decoded
        assert assistant in decoded

        def no_mutations(dry_run):
            gateway = build_gateway(preview_server.uri, dry_run=dry_run)

            def forbidden(*args, **kwargs):
                pytest.fail("unchanged preview rerun must not mutate the target")

            for method in (
                "create_trace", "retire_trace", "record_artifact_verification",
                "upload_named_json", "publish_experiment_file",
            ):
                monkeypatch.setattr(gateway, method, forbidden)
            for method in ("set_trace_tag", "delete_trace_tag", "delete_traces"):
                monkeypatch.setattr(gateway._client, method, forbidden)
            return gateway

        again = run(no_mutations)
        assert again.exit_status == 0, again.report
        assert again.report["summary"]["unchanged_turns"] == 1
        assert turn_entries(again.report)[0]["trace_id"] == trace_id
        reread = preview_server.client.get_trace(trace_id, display=False)
        assert reread.info.tags == tags
        assert {span.span_id for span in reread.data.spans} == {span.span_id for span in trace.data.spans}
        for key, data in retained.items():
            assert Path(repository.download_artifacts(tags[key], str(tmp_path))).read_bytes() == data
        assert len(preview_server.client.search_traces(locations=[trace.info.experiment_id], include_spans=False)) == 1
    finally:
        for result in runs:
            result.result.close()
