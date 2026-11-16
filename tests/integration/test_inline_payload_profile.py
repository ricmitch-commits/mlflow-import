"""Local-only qualification of the pinned historical per-value envelope."""

import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from urllib.parse import urlsplit

import mlflow
import pytest
import requests

from mlflow_import.content.activity import project_activity
from mlflow_import.content.artifacts import load_root_values
from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.capabilities import PINNED_MAX_INLINE_PAYLOAD_BYTES, CapabilityRequest
from mlflow_import.mlflow.gateway import MlflowGateway, TraceChildRequest, TraceCreateRequest
from mlflow_import.mlflow.mapping import _shared_child_specs
from tests.integration import server as server_module
from tests.integration.mlflow_profile import MAX_INLINE_PAYLOAD_BYTES
from tests.integration.server import live_mlflow_server, unique_experiment_name
from tests.support.inline_payloads import (
    QUALIFICATION_SIZES,
    exact_inline_value,
    synthetic_tool_turn,
)

pytestmark = [pytest.mark.integration, pytest.mark.profile, pytest.mark.slow]


@pytest.fixture(scope="module")
def local_inline_server():
    popen = subprocess.Popen

    def one_worker(command, *args, **kwargs):
        if isinstance(command, list) and command[1:4] == ["-m", "mlflow", "server"]:
            command = [*command, "--workers", "1"]
        return popen(command, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLFLOW_TRACKING_URI", "http://remote.invalid:1")
        patch.setenv("MLFLOW_ENABLE_TELEMETRY", "false")
        patch.setattr(server_module.subprocess, "Popen", one_worker)
        with live_mlflow_server(requested_uri="", proxy_artifacts=True) as server:
            assert server.process is not None
            assert server.process.poll() is None
            assert urlsplit(server.uri).hostname == "127.0.0.1"
            assert mlflow.__version__ == "3.17.2"
            yield server


@contextmanager
def _rss_observation(server_pid):
    """Sample whole server process-tree RSS and this importer, in bytes."""
    stop = threading.Event()
    peak = {"server_rss_bytes": 0, "client_rss_bytes": 0, "samples": 0}

    def sample():
        while not stop.is_set():
            result = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, check=True)
            rows = [tuple(map(int, line.split())) for line in result.stdout.splitlines() if line.strip()]
            descendants = {server_pid}
            while True:
                expanded = descendants | {pid for pid, parent, _ in rows if parent in descendants}
                if expanded == descendants:
                    break
                descendants = expanded
            peak["server_rss_bytes"] = max(
                peak["server_rss_bytes"], sum(rss * 1024 for pid, _, rss in rows if pid in descendants)
            )
            peak["client_rss_bytes"] = max(
                peak["client_rss_bytes"], next(rss * 1024 for pid, _, rss in rows if pid == os.getpid())
            )
            peak["samples"] += 1
            stop.wait(0.1)

    observer = threading.Thread(target=sample, daemon=True)
    observer.start()
    try:
        yield peak
    finally:
        stop.set()
        observer.join(timeout=5)
        assert not observer.is_alive()


def _historical_request(experiment_id, value):
    start, end = 1_794_580_115_000_000_000, 1_794_580_116_000_000_000
    child = TraceChildRequest(
        source_key="synthetic-child",
        name="synthetic tool",
        span_type="TOOL",
        parent_source_key=None,
        inputs=value,
        outputs=value,
        attributes={"synthetic.value": value},
        tags={},
        start_time_ns=start + 1,
        end_time_ns=end - 1,
        status="OK",
    )
    return TraceCreateRequest(
        client_request_id=None,
        experiment_id=experiment_id,
        name="synthetic root",
        native_session_id="synthetic-inline-session",
        root_inputs=value,
        root_outputs=value,
        root_attributes={"synthetic.value": value},
        root_tags={},
        metadata={},
        start_time_ns=start,
        end_time_ns=end,
        child_specs=lambda: iter((child,)),
        expected_operation_key="synthetic-inline",
        artifact_snapshot_fingerprint="a" * 64,
        artifact_relative_path="synthetic-inline.json",
        root_status="OK",
        historical=True,
        trace_name="synthetic inline",
        request_preview="synthetic request",
        response_preview="synthetic response",
    )


@pytest.mark.parametrize("size", QUALIFICATION_SIZES)
def test_complete_structured_values_round_trip(local_inline_server, monkeypatch, size):
    _qualify(local_inline_server, monkeypatch, size, structured=True)


def test_complete_string_at_eight_mib_round_trips(local_inline_server, monkeypatch):
    _qualify(local_inline_server, monkeypatch, 8_388_608, structured=False)


@pytest.mark.parametrize("field", ["input", "output"])
def test_above_eight_mib_fails_before_historical_ingest(local_inline_server, monkeypatch, field):
    gateway = MlflowGateway(local_inline_server.client)
    report = gateway.probe(CapabilityRequest())
    assert report.is_compatible
    limit = report.max_inline_payload_bytes
    assert limit == PINNED_MAX_INLINE_PAYLOAD_BYTES == MAX_INLINE_PAYLOAD_BYTES == 8_388_608
    experiment = gateway.resolve_experiment(unique_experiment_name("inline-rejection"), create=True)
    value = exact_inline_value(8_388_609)
    turn = synthetic_tool_turn(value, field)
    activity = project_activity(turn)
    request = replace(
        _historical_request(experiment.experiment_id, "synthetic small"),
        child_specs=lambda: _shared_child_specs(turn, activity, limit),
    )
    requests_seen = []

    def no_transport(*args, **kwargs):
        requests_seen.append(True)
        raise AssertionError("an oversized complete value must fail before transport")

    monkeypatch.setattr(requests.Session, "send", no_transport)
    with pytest.raises(ItemImportError) as caught:
        gateway.create_trace(request)
    assert caught.value.code == "payload_too_large"
    setattr(turn, "root_" + field, value)
    with pytest.raises(ItemImportError) as caught:
        load_root_values(turn, limit)
    assert caught.value.code == "payload_too_large"
    assert not requests_seen


def _qualify(server, monkeypatch, size, *, structured):
    wire = {"max_request_bytes": 0, "otlp_requests": 0, "full_trace_gets": 0}
    send = requests.Session.send

    def localhost_only(session, request, **kwargs):
        parsed = urlsplit(request.url)
        assert parsed.hostname == "127.0.0.1"
        assert parsed.port == urlsplit(server.uri).port
        body = request.body
        if isinstance(body, (bytes, str)):
            wire["max_request_bytes"] = max(wire["max_request_bytes"], len(body))
        if request.method == "POST" and parsed.path.endswith("/v1/traces"):
            wire["otlp_requests"] += 1
        if request.method == "GET" and parsed.path.endswith("/mlflow/traces/get"):
            wire["full_trace_gets"] += 1
        return send(session, request, **kwargs)

    monkeypatch.setattr(requests.Session, "send", localhost_only)
    gateway = MlflowGateway(server.client)
    report = gateway.probe(CapabilityRequest())
    assert report.is_compatible
    limit = report.max_inline_payload_bytes
    assert limit == PINNED_MAX_INLINE_PAYLOAD_BYTES == MAX_INLINE_PAYLOAD_BYTES == 8_388_608
    experiment = gateway.resolve_experiment(unique_experiment_name("inline-profile"), create=True)
    assert server.client.get_experiment(experiment.experiment_id).artifact_location.startswith("mlflow-artifacts:/")
    assert server.process is not None
    with _rss_observation(server.process.pid) as memory:
        started = time.monotonic()
        value = exact_inline_value(size, structured=structured)
        turn = synthetic_tool_turn(value, "input")
        turn.root_input = turn.root_output = value
        loaded = load_root_values(turn, limit)
        assert loaded.root_input == loaded.root_output == value
        if structured:
            activity = project_activity(turn)
            projected = list(_shared_child_specs(turn, activity, limit))
            tool = next(spec for spec in projected if spec.span_type == "TOOL")
            assert tool.inputs == value
        request = _historical_request(experiment.experiment_id, value)
        created = gateway.create_trace(request)
        gateway.verify_trace(created, request)
        trace = server.client.get_trace(created.trace_id, display=False)
        assert len(trace.data.spans) == 2
        root = next(span for span in trace.data.spans if span.parent_id is None)
        child = next(span for span in trace.data.spans if span.parent_id is not None)
        assert child.parent_id == root.span_id
        for span in (root, child):
            for actual in (span.inputs, span.outputs, span.attributes["synthetic.value"]):
                assert type(actual) is type(value)
                assert actual == value
        elapsed = time.monotonic() - started
    assert memory["samples"] > 0
    assert memory["server_rss_bytes"] < 8 * 1024**3
    assert memory["client_rss_bytes"] < 8 * 1024**3
    assert elapsed < 300
    assert wire["otlp_requests"] == 1
    assert wire["max_request_bytes"] < 64 * 1024**2
    assert wire["full_trace_gets"] >= 2
    # Numeric observations only: no generated payloads or source content logged.
    print(
        json.dumps(
            {"canonical_bytes": size, "structured": structured, "seconds": round(elapsed, 3), **wire, **memory},
            sort_keys=True,
        )
    )
