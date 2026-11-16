"""Pinned SQL/OTLP regressions for verbatim JSON-looking source strings."""

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest
from mlflow.artifacts import get_artifact_repository

from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.capabilities import CapabilityRequest
from mlflow_import.mlflow.gateway import MlflowGateway, TraceChildRequest, TraceCreateRequest
from tests.integration.import_runner import run_real_import, turn_entries
from tests.integration.server import live_mlflow_server, unique_experiment_name
from tests.support.import_fakes import build_current_source_db

pytestmark = [pytest.mark.integration, pytest.mark.profile]

_QUOTED = json.dumps('雪 😀 "quoted" \\ path\nnext', ensure_ascii=False)
_VALUES = [
    pytest.param('[1, {"text": "雪 😀", "fraction": 0.30000000000000004}]', id="array-string"),
    pytest.param('{"text":"\\u96ea","nested":"[1]"}', id="object-string"),
    pytest.param(' \n ["雪"] \t', id="whitespace-array-string"),
    pytest.param(_QUOTED, id="quoted-unicode-string"),
    pytest.param(json.dumps(_QUOTED, ensure_ascii=False), id="multilayer-string"),
    pytest.param('""', id="quoted-empty-string"),
    pytest.param('"1"', id="quoted-numeric-string"),
    pytest.param('{"x":1, "x":2}', id="duplicate-key-object-string"),
    pytest.param("plain 雪 😀", id="plain-string"),
    pytest.param("", id="empty-string"),
    pytest.param("[incomplete", id="malformed-json-string"),
    pytest.param("1", id="integer-string"),
    pytest.param("0.30000000000000004", id="float-string"),
    pytest.param("true", id="true-string"),
    pytest.param("false", id="false-string"),
    pytest.param("null", id="null-string"),
    pytest.param("NaN", id="nan-string"),
    pytest.param("Infinity", id="infinity-string"),
    pytest.param(["[1]", {"quoted": _QUOTED}], id="real-list"),
    pytest.param({"array": "[1]", "object": '{"x":1}', "quoted": _QUOTED}, id="real-object"),
    pytest.param(7, id="real-integer"),
    pytest.param(0.30000000000000004, id="real-float"),
    pytest.param(True, id="real-boolean"),
    pytest.param(None, id="real-null"),
]


@pytest.fixture(scope="module")
def local_string_server():
    # An unusable ambient URI must never be consulted by these regressions.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLFLOW_TRACKING_URI", "http://remote.invalid:1")
        with live_mlflow_server(requested_uri="") as server:
            assert server.process is not None
            assert mlflow.__version__ == "3.17.2"
            yield server


@pytest.fixture(scope="module")
def transport_gateway(local_string_server):
    gateway = MlflowGateway(local_string_server.client)
    assert gateway.probe(CapabilityRequest()).is_compatible
    experiment = gateway.resolve_experiment(unique_experiment_name("string-transport"), create=True)
    return gateway, experiment.experiment_id


def _request(experiment_id, value):
    start = 1_794_580_115_471_123_456
    end = start + 12_345_678
    child = TraceChildRequest(
        source_key="child",
        name="synthetic tool",
        span_type="TOOL",
        parent_source_key=None,
        inputs=value,
        outputs=value,
        attributes={"synthetic.value": value},
        tags={},
        start_time_ns=start + 100,
        end_time_ns=end - 100,
        status="OK",
    )
    return TraceCreateRequest(
        client_request_id=None,
        experiment_id=experiment_id,
        name="synthetic root",
        native_session_id="synthetic-string-session",
        root_inputs=value,
        root_outputs=value,
        root_attributes={"synthetic.value": value},
        root_tags={},
        metadata={},
        start_time_ns=start,
        end_time_ns=end,
        child_specs=lambda: iter((child,)),
        expected_operation_key="synthetic",
        artifact_snapshot_fingerprint="a" * 64,
        artifact_relative_path="synthetic.json",
        root_status="OK",
        historical=True,
        trace_name="synthetic",
        request_preview="synthetic request",
        response_preview="synthetic response",
        native_usage={"input_tokens": 7, "output_tokens": 0, "total_tokens": 7},
        native_cost={"total_cost": 0.1 + 0.2},
    )


@pytest.mark.parametrize("value", _VALUES)
def test_root_and_child_values_round_trip_exactly(local_string_server, transport_gateway, value):
    gateway, experiment_id = transport_gateway
    request = _request(experiment_id, value)
    created = gateway.create_trace(request)
    gateway.verify_trace(created, request)
    trace = local_string_server.client.get_trace(created.trace_id, display=False)
    assert len(trace.data.spans) == 2
    for span in trace.data.spans:
        for actual in (span.inputs, span.outputs, span.attributes["synthetic.value"]):
            assert type(actual) is type(value)
            assert actual == value


@pytest.mark.parametrize(
    "location", ["root_inputs", "root_outputs", "root_attributes", "child_inputs", "child_outputs", "child_attributes"]
)
@pytest.mark.parametrize(
    "original,changed",
    [({"kept": 1}, "[2]"), ({"kept": 1}, [1]), ({"kept": 1}, {}), ({"kept": 1}, {"kept": 1, "extra": 2}), ("[1]", [1])],
)
def test_genuinely_different_values_still_fail(transport_gateway, location, original, changed):
    gateway, experiment_id = transport_gateway
    request = _request(experiment_id, original)
    created = gateway.create_trace(request)
    gateway.verify_trace(created, request)
    if location.startswith("root_"):
        replacement = {"synthetic.value": changed} if location == "root_attributes" else changed
        altered = replace(request, **{location: replacement})
    else:
        field = location.removeprefix("child_")
        replacement = {"synthetic.value": changed} if field == "attributes" else changed
        child = replace(next(request.child_specs()), **{field: replacement})
        altered = replace(request, child_specs=lambda: iter((child,)))
    with pytest.raises(ItemImportError) as caught:
        gateway.verify_trace(created, altered)
    assert caught.value.code == "trace_verification_failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_usage", {"input_tokens": 8, "output_tokens": 0, "total_tokens": 8}),
        ("native_usage", {"input_tokens": 7, "total_tokens": 7}),
        ("native_cost", {"input_cost": 0, "total_cost": 0.1 + 0.2}),
        ("native_cost", {"total_cost": 0.3}),
    ],
)
def test_native_analytics_remain_exact(transport_gateway, field, value):
    gateway, experiment_id = transport_gateway
    request = _request(experiment_id, '["unchanged"]')
    created = gateway.create_trace(request)
    gateway.verify_trace(created, request)
    with pytest.raises(ItemImportError) as caught:
        gateway.verify_trace(created, replace(request, **{field: value}))
    assert caught.value.code == "trace_verification_failed"


@pytest.mark.parametrize("field", ["start_time_ns", "end_time_ns"])
def test_fractional_timing_remains_exact(transport_gateway, field):
    gateway, experiment_id = transport_gateway
    request = _request(experiment_id, '["unchanged"]')
    created = gateway.create_trace(request)
    gateway.verify_trace(created, request)
    with pytest.raises(ItemImportError) as caught:
        gateway.verify_trace(created, replace(request, **{field: getattr(request, field) + 1}))
    assert caught.value.code == "trace_verification_failed"


def test_json_looking_tool_result_imports_and_reuses_artifact_identity(local_string_server, tmp_path):
    output = '[{"text":"雪 😀", "quoted":"\\"nested\\"", "fraction":0.30000000000000004}]'
    source = build_current_source_db(
        tmp_path,
        sessions=[("synthetic-json-tool", 1000)],
        messages=[
            ("u", "synthetic-json-tool", "user", 1, json.dumps({"text": "synthetic prompt"})),
            (
                "a",
                "synthetic-json-tool",
                "assistant",
                2,
                json.dumps(
                    {
                        "finish": "stop",
                        "content": [
                            {
                                "type": "tool",
                                "tool": "synthetic",
                                "call_id": "call-json",
                                "state": {
                                    "input": {"query": "synthetic"},
                                    "output": output,
                                    "status": "completed",
                                },
                            },
                            {"type": "text", "text": "synthetic answer"},
                        ],
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
    )
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE session_message SET time_created=1000,time_updated=2000 WHERE id='u'")
        connection.execute("UPDATE session_message SET time_created=2001,time_updated=3000 WHERE id='a'")
    name = unique_experiment_name("json-tool-rerun")
    runs = []
    try:
        first = run_real_import(
            server_uri=local_string_server.uri,
            source=source,
            experiment_name=name,
            tmp_root=tmp_path,
        )
        runs.append(first)
        assert first.exit_status == 0, first.report
        trace_id = turn_entries(first.report)[0]["trace_id"]
        trace = local_string_server.client.get_trace(trace_id, display=False)
        tool = next(span for span in trace.data.spans if span.span_type == "TOOL")
        assert tool.outputs == output
        assert isinstance(tool.outputs, str)
        tags = trace.info.tags
        repository = get_artifact_repository(
            local_string_server.client.get_experiment(trace.info.experiment_id).artifact_location,
            tracking_uri=local_string_server.uri,
        )
        downloaded = Path(repository.download_artifacts(tags["opencode.artifact_path"], str(tmp_path)))
        original_bytes = downloaded.read_bytes()
        assert hashlib.sha256(original_bytes).hexdigest() == tags["opencode.artifact_sha256"]
        document = json.loads(original_bytes)
        assert document["identity_map"] == {
            "trace_id": trace_id,
            **{
                f"span:{span.attributes['mlflow_import.source_key']}": span.span_id
                for span in trace.data.spans
                if span.parent_id is not None
            },
        }
        session_bytes = Path(
            repository.download_artifacts(tags["mlflow_import.session_artifact"], str(tmp_path))
        ).read_bytes()
        assert hashlib.sha256(session_bytes).hexdigest() == tags["mlflow_import.session_artifact_sha256"]
        session = json.loads(session_bytes)
        assert any(
            part.get("state", {}).get("output") == output
            for record in session["records"]
            for part in (record.get("decoded") or {}).get("content", [])
        )
        again = run_real_import(
            server_uri=local_string_server.uri,
            source=source,
            experiment_name=name,
            tmp_root=tmp_path,
        )
        runs.append(again)
        assert again.exit_status == 0, again.report
        assert again.report["summary"]["unchanged_turns"] == 1
        assert turn_entries(again.report)[0]["trace_id"] == trace_id
        reread = local_string_server.client.get_trace(trace_id, display=False)
        assert {span.span_id for span in reread.data.spans} == {span.span_id for span in trace.data.spans}
        assert reread.info.tags["opencode.artifact_path"] == tags["opencode.artifact_path"]
        assert reread.info.tags["opencode.artifact_sha256"] == tags["opencode.artifact_sha256"]
        for key in ("mlflow_import.session_artifact", "mlflow_import.session_artifact_sha256"):
            assert reread.info.tags[key] == tags[key]
        assert (
            Path(repository.download_artifacts(tags["opencode.artifact_path"], str(tmp_path))).read_bytes()
            == original_bytes
        )
    finally:
        for run in runs:
            run.result.close()
