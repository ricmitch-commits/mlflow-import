"""Actual mapper + journal acceptance on exclusively owned synthetic targets.

Default: bounded 10-call SQLite/HTTP recovery and 10/100/1000-call pure mapping.
Opt in with MLFLOW_IMPORT_CONTAINER_QUALIFICATION=1 for the same lifecycle at
100 calls on PostgreSQL (1 GiB) + MLflow (8 GiB). No ambient target is adopted.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import urllib.request
from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import requests

from mlflow_import import resources
from mlflow_import.content.bundles import stream_session_object
from mlflow_import.content.canonical import canonical_json
from mlflow_import.content.input_context import INPUT_CONTEXT_ATTRIBUTE, validate_input_contexts
from mlflow_import.contracts import RedactionMode
from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.attempts import AttemptJournal
from mlflow_import.mlflow.gateway import build_gateway
from mlflow_import.mlflow.mapping import SourceIdentity
from mlflow_import.mlflow.trace_envelope import (
    MAX_TRACE_SERIALIZED_SPAN_BYTES,
    TRACE_RESOURCE_BYTES_TAG,
    TRACE_RESOURCE_SPANS_TAG,
    admit_trace,
)
from tests.integration.import_runner import run_real_import, trace_ids_by_key, turn_entries
from tests.integration.server import live_mlflow_server, unique_experiment_name
from tests.support.compact_trace_fixture import (
    PROMPT,
    SESSION,
    answer,
    mapped_scoped_source,
    mapped_source,
    synthetic_source,
)
from tests.support.container_qualification import disposable_container_profile
from tests.support.trace_qualification import assert_exact_value

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("calls", [10, 100, 1000, 702])
def test_compact_production_mapper_complete_linear_source(tmp_path, calls):
    source = synthetic_source(tmp_path, calls)
    assert source.stat().st_size < 16 * 1024**2
    with mapped_source(source, tmp_path) as mapped:
        [(turn, request)] = mapped
        envelope = admit_trace(request)
        children = list(request.child_specs())
        models = [child for child in children if INPUT_CONTEXT_ATTRIBUTE in child.attributes]
        tools = [child for child in children if child.span_type == "TOOL"]
        assert len(models) == calls and len(tools) == calls - 1
        assert len(children) == 3 * calls - 1  # 702 calls + text/tool children reproduce 2105 children.
        assert envelope.span_count == 3 * calls
        assert envelope.serialized_span_upper_bound < MAX_TRACE_SERIALIZED_SPAN_BYTES
        input_bytes = sum(len(canonical_json(model.inputs)) for model in models)
        assert input_bytes < calls * 4000
        assert sum(canonical_json(model.inputs).count(b"PROMPT:") for model in models) == 1
        previous = None
        for index, model in enumerate(models):
            descriptor = model.attributes[INPUT_CONTEXT_ATTRIBUTE]
            assert descriptor["policy"] == "conversation-delta-v1"
            assert descriptor["scope"] == ""
            assert descriptor["previous_context_id"] == previous
            previous = descriptor["context_id"]
            encoded = canonical_json(model.inputs)
            if index:
                assert f"RESULT:{index - 1}:".encode() in encoded
                assert f"QUERY:{index - 1}".encode() in encoded
                if index > 1:
                    assert f"RESULT:{index - 2}:".encode() not in encoded
            else:
                assert model.inputs["messages"] == [{"role": "user", "content": PROMPT}]
        retained = io.BytesIO()
        stream_session_object(turn, retained, RedactionMode.VERBATIM)
        artifact = json.loads(retained.getvalue())
        assert artifact["schema_version"] == "mlflow-import.session.v1"
        records = {record["source_id"]: record for record in artifact["records"]
                   if record.get("source_table") == "session_message"}
        assert {"u0", *(f"a{i}" for i in range(calls))} <= set(records)
        assert records["u0"]["decoded"]["content"] == PROMPT
        for index in range(calls):
            assert_exact_value(records[f"a{index}"]["decoded"], answer(index, final=index == calls - 1))
        print(json.dumps({"mapper_calls": calls, "mapped_children": len(children),
                          "model_input_bytes": input_bytes, "session_bytes": len(retained.getvalue()),
                          "serialized_span_bound": envelope.serialized_span_upper_bound}))


def test_compact_production_contexts_resolve_source_values_within_each_scope(tmp_path):
    with mapped_scoped_source(tmp_path) as (turn, request):
        children = list(request.child_specs())
        models = [child for child in children if INPUT_CONTEXT_ATTRIBUTE in child.attributes]
        assert len(models) == 4
        scopes = {model.attributes[INPUT_CONTEXT_ATTRIBUTE]["scope"] for model in models}
        assert len(scopes) == 2 and "" in scopes
        records = {SourceIdentity.of(record).key: record for record in turn.iter_records()}
        for scope in scopes:
            calls = [model for model in models if model.attributes[INPUT_CONTEXT_ATTRIBUTE]["scope"] == scope]
            prefix, foreign = ("CHILD:", "PRIMARY:") if scope else ("PRIMARY:", "CHILD:")
            assert len(calls) == 2
            assert calls[0].inputs["messages"] == [{"role": "user", "content": prefix + "PROMPT"}]
            assert foreign.encode() not in canonical_json([call.inputs for call in calls])
            assert prefix.encode() + b"RESULT" in canonical_json(calls[1].inputs)
            assert prefix.encode() + b"QUERY" in canonical_json(calls[1].inputs)
            previous = None
            for call in calls:
                descriptor = call.attributes[INPUT_CONTEXT_ATTRIBUTE]
                assert descriptor["previous_context_id"] == previous
                previous = descriptor["context_id"]
                for ref in descriptor["delta"]:
                    source_record = records[ref["source_key"]]
                    assert str(source_record.raw_columns.get("subagent") or "") == scope
                    assert source_record.occurrence_id == ref["occurrence_id"]
                    message = call.inputs["messages"][ref["message_index"]]
                    # Resolve prompt/result values independently from the stored
                    # occurrence, rather than merely trusting the map validator.
                    for marker in (prefix + "PROMPT", prefix + "RESULT"):
                        if marker.encode() in canonical_json(message):
                            assert marker.encode() in canonical_json(source_record.decoded_value)
        agents = {child.source_key: child for child in children if child.span_type == "AGENT"}
        nested = next(call for call in models if call.attributes[INPUT_CONTEXT_ATTRIBUTE]["scope"])
        assert nested.parent_source_key in agents
        tool_keys = {child.source_key for child in children if child.span_type == "TOOL"}
        assert agents[nested.parent_source_key].parent_source_key in tool_keys


@pytest.fixture(params=["sqlite", "postgres"])
def compact_owned_target(request, monkeypatch):
    # Both transports are guarded, including MLflow artifact downloads via urllib.
    send, open_url = requests.Session.send, urllib.request.OpenerDirector.open
    allowed = {"port": None}

    def check(url):
        parsed = urlsplit(url)
        assert parsed.hostname == "127.0.0.1", "non-owned target forbidden"
        if allowed["port"] is not None:
            assert parsed.port == allowed["port"], "non-owned loopback target forbidden"

    def guarded_send(session, prepared, **kwargs):
        check(prepared.url)
        kwargs["proxies"] = {}
        return send(session, prepared, **kwargs)

    def guarded_open(opener, url, *args, **kwargs):
        check(getattr(url, "full_url", url))
        return open_url(opener, url, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "send", guarded_send)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", guarded_open)
    if request.param == "postgres":
        if os.environ.get("MLFLOW_IMPORT_CONTAINER_QUALIFICATION") != "1":
            pytest.skip("explicit owned PostgreSQL container qualification opt-in required")
        with disposable_container_profile() as profile:
            allowed["port"] = urlsplit(profile.uri).port
            yield SimpleNamespace(uri=profile.uri, calls=100, profile=profile)
            metrics_by_role = profile.resources()
            print(json.dumps({"compact_recovery_resources": metrics_by_role}, sort_keys=True))
            for metrics in metrics_by_role.values():
                assert metrics["running"] and not metrics["oom_killed"]
                assert metrics["oom"] == metrics["oom_kill"] == metrics["restarts"] == 0
                assert metrics["cgroup_peak_bytes"] < metrics["limit_bytes"]
    else:
        with live_mlflow_server(requested_uri="") as server:
            assert server.process is not None
            allowed["port"] = urlsplit(server.uri).port
            yield SimpleNamespace(uri=server.uri, calls=10, profile=None)


def _artifact(gateway, experiment, path):
    with resources.temporary_directory(prefix="compact-acceptance-read-") as directory:
        local = resources.copy_artifact(gateway.experiment_repository(experiment), path, directory, 32 * 1024**2)
        return local.read_bytes()


def test_compact_full_journal_crash_resume_and_logical_reuse(compact_owned_target, tmp_path, monkeypatch):
    target = compact_owned_target
    source = synthetic_source(tmp_path, target.calls, sibling=True)
    experiment = unique_experiment_name("compact-production-recovery")
    phase = "crash"
    creates, uploads, publications, runs = [], [], [], []
    publish = AttemptJournal.publish
    prepared = {}
    captured_references = {}
    interrupted_reads = []

    def journal_publish(journal, state, values):
        journal.validate({**values, "state": state, "schema_version": "mlflow-import.attempt.v1"})
        if state == "prepared":
            prepared.update(values)
        if state == "ready_to_retire":
            assert values["attempt_id"] == prepared["attempt_id"]
            assert len(values["replacements"]) == len(values["expected_turns"])
            assert phase != "crash", "ready state was published despite unavailable replacement capture"
            for reference in values["replacements"]:
                assert reference == captured_references[(phase, reference["trace_id"])]
        result = publish(journal, state, values)
        publications.append((phase, state))
        return result

    monkeypatch.setattr(AttemptJournal, "publish", journal_publish)

    def factory(dry_run):
        gateway = build_gateway(target.uri, dry_run=dry_run)
        create, upload = gateway.create_trace, gateway.publish_experiment_file
        capture_reference = gateway.replacement_reference

        def replacement_reference(trace_id):
            # Fail the actual production reference-capture read after both
            # traces/maps are staged, before a ready journal can be published.
            if phase == "crash":
                assert len(creates) == 2

                def unavailable(*args, **kwargs):
                    interrupted_reads.append(trace_id)
                    raise ItemImportError("synthetic_recovery_read_unavailable", "synthetic read interruption")

                with monkeypatch.context() as patch:
                    patch.setattr(gateway._client, "get_trace", unavailable)
                    return capture_reference(trace_id)
            reference = capture_reference(trace_id)
            captured_references[(phase, trace_id)] = reference
            return reference

        def create_trace(request):
            children = tuple(request.child_specs())
            frozen = replace(request, child_specs=lambda: iter(children))
            created = create(request)
            creates.append((phase, created, frozen))
            return created

        def publish_file(experiment_id, path, local, **kwargs):
            content = local.read_bytes()
            result = upload(experiment_id, path, local, **kwargs)
            uploads.append((phase, path, content))
            return result

        gateway.create_trace, gateway.publish_experiment_file = create_trace, publish_file
        gateway.replacement_reference = replacement_reference
        if phase == "noop":
            def forbidden(*args, **kwargs):
                pytest.fail("unchanged import attempted a mutation")
            for method in ("create_trace", "retire_trace", "record_artifact_verification",
                           "upload_named_json", "publish_experiment_file", "reconcile_tags"):
                setattr(gateway, method, forbidden)
            for method in ("set_trace_tag", "delete_trace_tag", "delete_traces", "create_experiment"):
                setattr(gateway._client, method, forbidden)
        return gateway

    def run():
        result = run_real_import(server_uri=target.uri, source=source, experiment_name=experiment,
                                 tmp_root=tmp_path, gateway_factory=factory)
        runs.append(result)
        return result

    try:
        interrupted = run()
        assert interrupted.exit_status == 1, interrupted.report
        assert len(creates) == 2
        assert interrupted_reads
        assert ("crash", "prepared") in publications
        assert ("crash", "ready_to_retire") not in publications
        assert all(turn["attempt_state"] == "prepared" for turn in turn_entries(interrupted.report))
        phase = "resume"
        resumed = run()
        assert resumed.exit_status == 0, resumed.report
        assert len(creates) == 2, "prepared recovery duplicated a verified production-mapped trace"
        assert ("resume", "ready_to_retire") in publications and ("resume", "complete") in publications
        before = trace_ids_by_key(resumed.report)
        gateway = build_gateway(target.uri, dry_run=True)
        experiment_id = resumed.report["experiment"]["id"]
        immutable_maps = {}
        for _, created, request in creates:
            trace = gateway._get_trace(created.trace_id)
            children = list(request.child_specs())
            assert int(trace.info.tags[TRACE_RESOURCE_SPANS_TAG]) == len(trace.data.spans) == len(children) + 1
            assert int(trace.info.tags[TRACE_RESOURCE_BYTES_TAG]) == admit_trace(request).serialized_span_upper_bound
            by_id = {span.span_id: span for span in trace.data.spans}
            root = by_id[created.root_span_id]
            assert root.parent_id is None and root.span_type == "AGENT"
            assert_exact_value(root.inputs, request.root_inputs)
            assert_exact_value(root.outputs, request.root_outputs)
            for child in children:
                span = by_id[created.span_ids_by_source[child.source_key]]
                assert span.span_type == child.span_type
                parent = (created.span_ids_by_source[child.parent_source_key]
                          if child.parent_source_key else created.root_span_id)
                assert span.parent_id == parent
                assert_exact_value(span.inputs, child.inputs)
                assert_exact_value(span.outputs, child.outputs)
            path = trace.info.tags["opencode.artifact_path"]
            assert path == request.artifact_relative_path == trace.info.tags["mlflow_import.turn_map"]
            content = _artifact(gateway, experiment_id, path)
            immutable_maps[path] = content
            assert hashlib.sha256(content).hexdigest() == trace.info.tags["opencode.artifact_sha256"]
            assert len(content) == int(trace.info.tags["opencode.artifact_size"])
            turn_map = json.loads(content)
            assert turn_map["schema_version"] == "mlflow-import.turn-map.v2"
            assert turn_map["identity_map"]["trace_id"] == created.trace_id
            assert validate_input_contexts(turn_map, trace.info.tags)
            assert len(turn_map["input_contexts"]) == sum(INPUT_CONTEXT_ATTRIBUTE in c.attributes for c in children)
            session_bytes = _artifact(gateway, experiment_id, trace.info.tags["mlflow_import.session_artifact"])
            assert hashlib.sha256(session_bytes).hexdigest() == trace.info.tags["mlflow_import.session_artifact_sha256"]
            assert len(session_bytes) == int(trace.info.tags["mlflow_import.session_artifact_size"])
            session = json.loads(session_bytes)
            records = {(r["source_table"], r.get("source_id")): r for r in session["records"]}
            previous = {}
            for entry in turn_map["input_contexts"]:
                descriptor = entry["descriptor"]
                scope = descriptor["scope"]
                assert descriptor["previous_context_id"] == previous.get(scope)
                previous[scope] = descriptor["context_id"]
                span = by_id[entry["span_id"]]
                assert_exact_value(span.attributes[INPUT_CONTEXT_ATTRIBUTE], descriptor)
                for ref in descriptor["delta"] + descriptor["invocation_sources"]:
                    mapped_record = next(r for r in turn_map["records"] if r["source_key"] == ref["source_key"])
                    source_record = records[(mapped_record["source_table"], mapped_record["source_id"])]
                    assert mapped_record["scope"] == scope
                    assert source_record.get("occurrence_id") == ref["occurrence_id"]
            assert PROMPT in [r.get("decoded", {}).get("content") for r in session["records"]]
            for index in range(target.calls):
                assert_exact_value(records[("session_message", f"a{index}")]["decoded"],
                                   answer(index, final=index == target.calls - 1))
        assert len(immutable_maps) == len(creates), "physical creates shared an immutable map path"
        phase = "noop"
        again = run()
        assert again.exit_status == 0 and trace_ids_by_key(again.report) == before, again.report
        assert again.report["summary"]["unchanged_turns"] == 2
        phase = "append"
        with sqlite3.connect(source) as connection:
            connection.executemany(
                "INSERT INTO session_message(id,session_id,type,seq,time_created,data) VALUES(?,?,?,?,?,?)",
                [("append-u", SESSION, "user", 10000, 200000, '{"text":"APPEND:prompt"}'),
                 ("append-a", SESSION, "assistant", 10001, 200001,
                  '{"model":"synthetic/model","text":"APPEND:answer","finish":"stop"}')],
            )
        appended = run()
        assert appended.exit_status == 0, appended.report
        appended_ids = trace_ids_by_key(appended.report)
        assert all(appended_ids[key] == trace_id for key, trace_id in before.items())
        assert len(creates) == 3
        phase = "edit"
        with sqlite3.connect(source) as connection:
            connection.execute("UPDATE session_message SET data=? WHERE id='a0'", (json.dumps(answer(0, edited=True)),))
        edited = run()
        assert edited.exit_status == 0, edited.report
        edited_ids = trace_ids_by_key(edited.report)
        changed = {key for key in appended_ids if appended_ids[key] != edited_ids[key]}
        assert len(changed) == 1 and next(iter(changed)) in before
        assert len(creates) == 4
        assert len({request.artifact_relative_path for _, _, request in creates}) == 4
        for path, content in immutable_maps.items():
            assert _artifact(gateway, experiment_id, path) == content
        phase = "noop"
        final = run()
        assert final.exit_status == 0 and trace_ids_by_key(final.report) == edited_ids, final.report
        # A real metadata-only lookup still succeeds with an absent proof, but
        # the gateway must refuse to issue the subsequent full trace read.
        proof_id = next(iter(edited_ids.values()))
        original_proof = gateway.trace_info(proof_id).tags[TRACE_RESOURCE_SPANS_TAG]
        gateway._client.delete_trace_tag(proof_id, TRACE_RESOURCE_SPANS_TAG)
        try:
            with monkeypatch.context() as patch:
                def forbidden_full(*args, **kwargs):
                    pytest.fail("unqualified trace reached full-payload retrieval")
                patch.setattr(gateway._client, "get_trace", forbidden_full)
                assert gateway.trace_exists(proof_id)
                with pytest.raises(ItemImportError) as caught:
                    gateway._get_trace(proof_id)
                assert caught.value.code == "trace_resource_profile_unverified"
        finally:
            gateway._client.set_trace_tag(proof_id, TRACE_RESOURCE_SPANS_TAG, original_proof)
        print(json.dumps({"backend": "postgres" if target.profile else "sqlite", "calls": target.calls,
                          "physical_creates": len(creates), "publications": publications,
                          "artifact_uploads": len(uploads)}))
    finally:
        for result in runs:
            result.result.close()
