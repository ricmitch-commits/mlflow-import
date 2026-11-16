"""Run explicitly: MLFLOW_IMPORT_CONTAINER_QUALIFICATION=1 .venv/bin/pytest
tests/integration/test_container_trace_qualification.py -s

Requires the local pinned MLflow image and postgres:16-alpine image. This
module never reads MLFLOW_TRACKING_URI or invokes the normal import pipeline.
Only synthetic public gateway requests are sent to its disposable server.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import urlsplit

import mlflow
import pytest
import requests
from mlflow import MlflowClient

from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.capabilities import CapabilityRequest
from mlflow_import.mlflow.gateway import MlflowGateway
from tests.support.container_qualification import MIB, disposable_container_profile
from tests.support.trace_qualification import assert_exact_value, measured_shape_request, qualification_request

pytestmark = [pytest.mark.integration, pytest.mark.profile, pytest.mark.slow]


@pytest.fixture(scope="module")
def container_profile():
    if os.environ.get("MLFLOW_IMPORT_CONTAINER_QUALIFICATION") != "1":
        pytest.skip("set MLFLOW_IMPORT_CONTAINER_QUALIFICATION=1 for owned local container qualification")
    assert mlflow.__version__ == "3.17.2"
    with disposable_container_profile() as profile:
        yield profile


@pytest.fixture
def wire(container_profile, monkeypatch):
    observations = {
        "requests": 0,
        "otlp_requests": 0,
        "otlp_wire_bytes": 0,
        "max_full_response_bytes": 0,
        "full_responses": 0,
        "max_response_bytes": 0,
    }
    send = requests.Session.send
    target = urlsplit(container_profile.uri)

    def guarded_send(session, request, **kwargs):
        parsed = urlsplit(request.url)
        assert (parsed.hostname, parsed.port) == ("127.0.0.1", target.port), "non-owned target forbidden"
        # Synthetic local transport must not inherit ambient proxies.
        kwargs["proxies"] = {}
        observations["requests"] += 1
        if request.method == "POST" and parsed.path.endswith("/v1/traces"):
            assert isinstance(request.body, bytes)
            observations["otlp_requests"] += 1
            observations["otlp_wire_bytes"] += len(request.body)
        response = send(session, request, **kwargs)
        assert response.status_code != 403, "owned localhost request forbidden: " + parsed.path
        if not kwargs.get("stream", False):
            size = len(response.content)
            observations["max_response_bytes"] = max(observations["max_response_bytes"], size)
            if parsed.path.endswith("/mlflow/traces/get"):
                assert response.headers.get("Content-Encoding", "identity") == "identity"
                observations["full_responses"] += 1
                observations["max_full_response_bytes"] = max(observations["max_full_response_bytes"], size)
        return response

    monkeypatch.setattr(requests.Session, "send", guarded_send)
    yield observations


@contextmanager
def memory_observation(profile):
    stop = threading.Event()
    memory = {
        "client_peak_rss_bytes": 0,
        "server_sample_peak_rss_bytes": 0,
        "db_sample_peak_rss_bytes": 0,
        "memory_samples": 0,
    }
    errors = []

    def observe():
        try:
            while not stop.is_set():
                rss = subprocess.run(
                    ["ps", "-p", str(os.getpid()), "-o", "rss="],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                memory["client_peak_rss_bytes"] = max(memory["client_peak_rss_bytes"], int(rss.stdout) * 1024)
                resources = profile.resources()
                for role in ("server", "db"):
                    key = role + "_sample_peak_rss_bytes"
                    memory[key] = max(memory[key], resources[role]["rss_bytes"])
                memory["memory_samples"] += 1
                stop.wait(0.5)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=observe, daemon=True)
    thread.start()
    try:
        yield memory
    finally:
        stop.set()
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert not errors, "resource observer failed: " + repr(errors)
        memory["resources"] = profile.resources()
        high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory["client_process_lifetime_peak_rss_bytes"] = int(high_water * (1 if sys.platform == "darwin" else 1024))


def _gateway(profile):
    gateway = MlflowGateway(MlflowClient(tracking_uri=profile.uri))
    report = gateway.probe(CapabilityRequest())
    assert report.is_compatible, report.safe_error
    experiment = gateway.resolve_experiment("synthetic-container-" + str(time.monotonic_ns()), create=True)
    return gateway, experiment.experiment_id


def _request(experiment_id, shape):
    from mlflow_import.mlflow.trace_envelope import (
        MAX_TRACE_SERIALIZED_SPAN_BYTES,
        MAX_TRACE_SPANS,
        admit_trace,
    )

    if shape == "few_large":
        return qualification_request(experiment_id, 8, 8 * MIB - 32768)
    if shape == "redacted_strict_types":
        from mlflow_import.content.redaction import DeterministicRedactor
        from mlflow_import.contracts import RedactionMode

        base = qualification_request(experiment_id, 8)
        value = DeterministicRedactor(RedactionMode.REDACTED).redact(
            {
                "password": "synthetic-must-never-persist",
                "Authorization": "Bearer synthetic-authorization-value",
                "types": [None, False, True, 0, 7, "7", {"unicode": '雪😀"\\\n'}],
            }
        )
        assert value["password"] == "[REDACTED]"
        assert value["Authorization"] == "Bearer [REDACTED]"
        children = tuple(replace(child, inputs=value, outputs=value) for child in base.child_specs())
        return replace(base, root_inputs=value, root_outputs=value, child_specs=lambda: iter(children))
    if shape == "measured_2105_children":
        return measured_shape_request(experiment_id)
    if shape == "many_small":
        return qualification_request(experiment_id, MAX_TRACE_SPANS - 1)
    children = MAX_TRACE_SPANS - 1
    empty = admit_trace(qualification_request(experiment_id, children, 0))
    # Conservative accounting adds a few bytes per 4096-byte string chunk.
    size = (MAX_TRACE_SERIALIZED_SPAN_BYTES - empty.serialized_span_upper_bound) // children - 64
    request = qualification_request(experiment_id, children, size)
    envelope = admit_trace(request)
    assert envelope.span_count == MAX_TRACE_SPANS
    assert 0.99 * MAX_TRACE_SERIALIZED_SPAN_BYTES <= envelope.serialized_span_upper_bound
    return request


def _artifact_and_read_checks(profile, gateway, created, request):
    """Fresh gateway models restart/read recovery without an import invocation."""
    trace_id = created.trace_id
    bases = {"synthetic-call": {"catalog": "synthetic-pinned", "input_per_million": 1.25}}
    payload = {
        "schema_version": "mlflow-import.turn-map.v1",
        "turn_key": "synthetic-turn",
        "turn_fingerprint": "a" * 64,
        "identity_map": {
            "trace_id": trace_id,
            **{"span:" + key: value for key, value in created.span_ids_by_source.items()},
        },
        "accounting": {"invocations": {key: {"cost": {"basis": value}} for key, value in bases.items()}},
    }
    for kind, content in (("turn", json.dumps(payload).encode()), ("session", b'{"synthetic":true}')):
        path = f"mlflow-import/v2/qualification/{trace_id}/{kind}.json"
        local = profile.directory / (kind + ".json")
        assert local.parent.is_dir()
        local.write_bytes(content)
        proof = gateway.publish_experiment_file(request.experiment_id, path, local)
        assert proof.sha256 == hashlib.sha256(content).hexdigest()
        if kind == "turn":
            for key, value in {
                "opencode.artifact_sha256": proof.sha256,
                "opencode.artifact_size": str(proof.size),
            }.items():
                gateway._client.set_trace_tag(trace_id, key, value)
        else:
            for suffix, value in (("", path), ("_sha256", proof.sha256), ("_size", str(proof.size))):
                gateway._client.set_trace_tag(trace_id, "mlflow_import.session_artifact" + suffix, value)
    gateway.verify_candidate_payload(SimpleNamespace(trace_id=trace_id), request)
    reference = gateway.replacement_reference(trace_id)
    fresh = MlflowGateway(MlflowClient(tracking_uri=profile.uri))
    fresh.verify_replacement_reference(reference)
    assert fresh.read_pricing_donor(reference) == bases
    assert fresh.read_pricing_basis(reference) == bases
    # Full-payload identity is strict: a different digest may not authorize reuse.
    with pytest.raises(ItemImportError) as caught:
        fresh.verify_replacement_reference({**reference, "payload_sha256": "0" * 64})
    assert caught.value.code == "replacement_verification_failed"


@pytest.mark.parametrize(
    "shape",
    [
        "measured_2105_children",
        "few_large",
        "many_small",
        "combined_near_bound",
        "redacted_strict_types",
    ],
)
def test_container_trace_capacity(container_profile, wire, shape):
    from mlflow_import.mlflow.trace_envelope import (
        MAX_TRACE_SERIALIZED_SPAN_BYTES,
        TRACE_RESOURCE_BYTES_TAG,
        TRACE_RESOURCE_SPANS_TAG,
        admit_trace,
    )

    gc.collect()
    gateway, experiment_id = _gateway(container_profile)
    started = time.monotonic()
    with memory_observation(container_profile) as memory:
        request = _request(experiment_id, shape)
        # Physical creation tags, following the historical gateway contract.
        expected_path = "mlflow-import/v2/qualification/PLACEHOLDER/turn.json"
        request = replace(
            request,
            root_tags={
                "mlflow_import.importer": "mlflow-import",
                "mlflow_import.harness": "opencode",
                "mlflow_import.source_namespace": "synthetic-local",
                "mlflow_import.session_id": request.native_session_id,
                "mlflow_import.qualified_session_id": json.dumps(
                    [
                        "v1",
                        "opencode",
                        "synthetic-local",
                        request.native_session_id,
                    ]
                ),
                "mlflow_import.turn_key": "synthetic-turn",
                "opencode.turn_index": "1",
                "opencode.render_fingerprint": "a" * 64,
                "opencode.artifact_path": expected_path,
                "mlflow_import.turn_map": expected_path,
            },
        )
        envelope = admit_trace(request)
        if shape == "measured_2105_children":
            smaller = admit_trace(measured_shape_request(experiment_id, calls=100))
            assert envelope.span_count == 2106
            assert envelope.serialized_span_upper_bound < 11 * smaller.serialized_span_upper_bound
        created = gateway.create_trace(request)
        gateway.verify_trace(created, request)
        trace = gateway._client.get_trace(created.trace_id, display=False)
        assert len(trace.data.spans) == envelope.span_count
        assert trace.info.tags[TRACE_RESOURCE_SPANS_TAG] == str(envelope.span_count)
        assert trace.info.tags[TRACE_RESOURCE_BYTES_TAG] == str(envelope.serialized_span_upper_bound)
        root = next(span for span in trace.data.spans if span.parent_id is None)
        assert_exact_value(root.inputs, request.root_inputs)
        assert_exact_value(root.outputs, request.root_outputs)
        by_id = {span.span_id: span for span in trace.data.spans}
        for child in request.child_specs():
            span = by_id[created.span_ids_by_source[child.source_key]]
            assert_exact_value(span.inputs, child.inputs)
            assert_exact_value(span.outputs, child.outputs)
            parent = created.span_ids_by_source[child.parent_source_key] if child.parent_source_key else root.span_id
            assert span.parent_id == parent
        if shape == "redacted_strict_types":
            serialized = json.dumps(trace.data.to_dict())
            assert "synthetic-must-never-persist" not in serialized
            assert "synthetic-authorization-value" not in serialized
            assert "[REDACTED]" in serialized
        # Bind immutable map pointers once the generated ID is available.
        actual_path = expected_path.replace("PLACEHOLDER", created.trace_id)
        for key in ("opencode.artifact_path", "mlflow_import.turn_map"):
            gateway._client.set_trace_tag(created.trace_id, key, actual_path)
        request = replace(
            request,
            root_tags={
                **request.root_tags,
                "opencode.artifact_path": actual_path,
                "mlflow_import.turn_map": actual_path,
            },
        )
        del trace, by_id
        _artifact_and_read_checks(container_profile, gateway, created, request)
    evidence = {
        "shape": shape,
        "span_count": envelope.span_count,
        "serialized_span_bound_bytes": envelope.serialized_span_upper_bound,
        "seconds": round(time.monotonic() - started, 3),
        **wire,
        **memory,
    }
    print(json.dumps(evidence, sort_keys=True))
    assert wire["otlp_requests"] == 1
    assert wire["full_responses"] >= 8
    assert 0 < wire["otlp_wire_bytes"] <= envelope.serialized_span_upper_bound
    assert 0 < wire["max_full_response_bytes"] <= MAX_TRACE_SERIALIZED_SPAN_BYTES + 2 * MIB
    assert memory["memory_samples"] > 0
    assert memory["client_peak_rss_bytes"] < 8 * 1024**3
    for metrics in memory["resources"].values():
        assert metrics["running"] and not metrics["oom_killed"]
        assert metrics["oom"] == metrics["oom_kill"] == metrics["restarts"] == 0
        assert metrics["cgroup_peak_bytes"] < metrics["limit_bytes"]


@pytest.mark.parametrize("shape", ["too_many", "too_large"])
def test_container_oversize_preflight_zero_calls(container_profile, wire, monkeypatch, shape):
    from mlflow_import.mlflow.trace_envelope import MAX_TRACE_SPANS

    gateway, experiment_id = _gateway(container_profile)
    request = qualification_request(
        experiment_id, MAX_TRACE_SPANS if shape == "too_many" else 10, 16 if shape == "too_many" else 7 * MIB
    )
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("oversized request reached transport")

    monkeypatch.setattr(requests.Session, "send", forbidden)
    before = wire["requests"]
    with pytest.raises(ItemImportError) as caught:
        gateway.create_trace(request)
    assert caught.value.code == "trace_resource_envelope_exceeded"
    assert not attempts and wire["requests"] == before
    print(json.dumps({"shape": shape, "preflight_transport_calls": len(attempts)}))
