"""Local-only admission, transport-oracle and metadata-gate regressions."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.capabilities import CapabilityRequest
from mlflow_import.mlflow.gateway import MlflowGateway, PublicTraceApi
from mlflow_import.mlflow.mapping import EventSpec
from mlflow_import.mlflow.reconciliation import CandidateContract, OwnedTraceCandidate
from mlflow_import.mlflow.trace_envelope import (
    MAX_TRACE_SERIALIZED_SPAN_BYTES,
    MAX_TRACE_SPANS,
    PROOF_TAG_KEYS,
    TRACE_RESOURCE_BYTES_TAG,
    TRACE_RESOURCE_PROFILE,
    TRACE_RESOURCE_PROFILE_TAG,
    TRACE_RESOURCE_SPANS_TAG,
    SpanBudget,
    TraceEnvelopeLimits,
    admit_trace,
    measure_trace,
    require_trace_proof,
)
from tests.support.fake_gateway import child_spec, fake_version_transport, trace_spec

pytestmark = pytest.mark.unit


def checked_gateway(client, *, real_api=False, trace_limits=None):
    gateway = MlflowGateway(
        client,
        trace_api=PublicTraceApi() if real_api else client.trace_api,
        version_transport=fake_version_transport,
        trace_limits=trace_limits,
    )
    assert gateway.probe(CapabilityRequest()).is_compatible
    return gateway


def test_measure_trace_matches_retaining_admission():
    request = trace_spec(children=[child_spec("c", inputs={"messages": ["x" * 4096]})])
    envelope = admit_trace(request)
    measurement = measure_trace(request)
    assert measurement.serialized_span_upper_bound == envelope.serialized_span_upper_bound
    assert measurement.span_count == envelope.span_count
    assert measurement.control_metadata_upper_bound == envelope.control_metadata_upper_bound


def test_measure_trace_reports_partial_candidate_byte_diagnostics():
    request = trace_spec(root_inputs={"payload": "x" * 4096})
    with pytest.raises(ItemImportError) as error:
        measure_trace(request, max_bytes=1024)
    assert error.value.diagnostics["candidate_span_bytes"] > 1024
    assert "serialized span bytes" in str(error.value)


def test_admission_accounts_event_specs_without_span_type_init_field():
    event = EventSpec(
        source_key="event",
        name="event",
        parent_source_key=None,
        inputs={"payload": "x"},
        attributes={"kind": "event"},
        tags={},
        start_time_ns=1,
        outputs=None,
        status="OK",
        end_time_ns=2,
    )
    envelope = admit_trace(trace_spec(child_specs=lambda: iter([event])))
    [accounted] = list(envelope.request.child_specs())
    assert isinstance(accounted, EventSpec)
    assert accounted.source_key == "event"
    assert accounted.inputs == {"payload": "x"}


def test_span_count_failure_reports_budget_diagnostics():
    budget = SpanBudget(max_spans=1)
    budget.account_spec(child_spec("a"))
    with pytest.raises(ItemImportError) as error:
        budget.account_spec(child_spec("b"))
    assert error.value.diagnostics["span_count"] == 1
    assert error.value.diagnostics["span_limit"] == 1
    assert "span count" in str(error.value)


def test_explicit_limits_raise_span_admission():
    request = trace_spec(child_specs=lambda: (child_spec(str(index)) for index in range(MAX_TRACE_SPANS)))
    with pytest.raises(ItemImportError):
        admit_trace(request)
    envelope = admit_trace(request, limits=TraceEnvelopeLimits(max_spans=MAX_TRACE_SPANS + 1))
    assert envelope.span_count == MAX_TRACE_SPANS + 1


def test_gateway_uses_configured_trace_limits(fake_mlflow_client):
    request = trace_spec(children=[child_spec("a"), child_spec("b")])
    rejected = checked_gateway(fake_mlflow_client, trace_limits=TraceEnvelopeLimits(max_spans=2))
    with pytest.raises(ItemImportError):
        rejected.create_trace(request)
    accepted = checked_gateway(fake_mlflow_client, trace_limits=TraceEnvelopeLimits(max_spans=3))
    created = accepted.create_trace(request)
    assert created.trace_id


def test_incremental_exact_byte_boundary_and_root_inclusive_count():
    child = child_spec("c", inputs={"messages": ["unicode 😀", '"\\\n']})
    measured = SpanBudget()
    measured.account_spec(child)
    exact = SpanBudget(max_bytes=measured.serialized_span_upper_bound, max_spans=1)
    exact.account_spec(child)
    with pytest.raises(ItemImportError, match="finite serialized"):
        exact.account_spec(child)
    with pytest.raises(ItemImportError):
        SpanBudget(max_bytes=measured.serialized_span_upper_bound - 1).account_spec(child)
    request = trace_spec(child_specs=lambda: (child_spec(str(i)) for i in range(MAX_TRACE_SPANS - 1)))
    envelope = admit_trace(request)
    assert envelope.span_count == MAX_TRACE_SPANS
    assert envelope.serialized_span_upper_bound <= MAX_TRACE_SERIALIZED_SPAN_BYTES
    with pytest.raises(ItemImportError):
        admit_trace(replace(request, child_specs=lambda: (child_spec(str(i)) for i in range(MAX_TRACE_SPANS))))


def test_stored_proof_above_default_envelope_remains_verifiable():
    info = SimpleNamespace(
        tags={
            TRACE_RESOURCE_PROFILE_TAG: TRACE_RESOURCE_PROFILE,
            TRACE_RESOURCE_BYTES_TAG: str(96 * 1024 * 1024),
            TRACE_RESOURCE_SPANS_TAG: "5000",
        },
        trace_metadata={},
    )
    assert require_trace_proof(info) == (96 * 1024 * 1024, 5000)
    info.tags[TRACE_RESOURCE_SPANS_TAG] = "40000"
    with pytest.raises(ItemImportError):
        require_trace_proof(info)


def test_old_2105_span_gigabyte_shape_stops_before_full_walk_or_mutation(fake_mlflow_client):
    gateway = checked_gateway(fake_mlflow_client)
    value = "x" * (512 * 1024)
    visited = []

    def children():
        for index in range(2104):
            visited.append(index)
            yield child_spec(str(index), inputs=value)

    before = list(fake_mlflow_client.mutations)
    with pytest.raises(ItemImportError) as error:
        gateway.create_trace(trace_spec(child_specs=children))
    assert error.value.code == "trace_resource_envelope_exceeded"
    assert len(visited) < 150
    assert fake_mlflow_client.mutations == before
    assert fake_mlflow_client.trace_api.calls == []


def test_single_factory_consumption_and_detached_mutable_payload(fake_mlflow_client):
    gateway = checked_gateway(fake_mlflow_client)
    calls = []
    payload = {"messages": ["accepted"]}

    def children():
        calls.append(True)
        assert not fake_mlflow_client.trace_api.calls
        if len(calls) > 1:
            pytest.fail("an admitted producer was invoked again")
        yield child_spec("c", inputs=payload)
        payload["messages"].append("changed after admission")

    created = gateway.create_trace(trace_spec(child_specs=children))
    stored = fake_mlflow_client._traces[created.trace_id]
    assert calls == [True]
    assert stored.data.spans[1].inputs == {"messages": ["accepted"]}
    assert PROOF_TAG_KEYS <= stored.info.tags.keys()
    assert int(stored.info.tags[TRACE_RESOURCE_SPANS_TAG]) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), 1 << 300])
def test_nonfinite_or_unsupported_values_fail_before_production(value):
    def forbidden():
        pytest.fail("children must not be visited after invalid root")
        yield

    with pytest.raises(ItemImportError) as error:
        admit_trace(trace_spec(root_inputs=value, child_specs=forbidden))
    assert error.value.code == "trace_resource_value_invalid"


def integer_transport_span(value):
    """Independent pinned SDK oracle for the historical AnyValue transport."""
    from mlflow.entities import Span

    from mlflow_import.mlflow.gateway import _encode_historical_attribute

    return Span.from_dict(
        {
            "trace_id": "tr-" + "a" * 32,
            "span_id": "b" * 16,
            "parent_span_id": None,
            "name": "integer-transport-oracle",
            "start_time_unix_nano": 1,
            "end_time_unix_nano": 2,
            "status": {"code": "STATUS_CODE_OK"},
            "attributes": {
                "mlflow.traceRequestId": _encode_historical_attribute("tr-" + "a" * 32),
                "probe": _encode_historical_attribute(value),
            },
        }
    )


@pytest.mark.parametrize("value", [(1 << 63) - 1, -(1 << 63)])
def test_int64_boundaries_remain_supported_by_admission_and_pinned_sdk(value):
    for payload in (value, {"nested": [value]}):
        assert integer_transport_span(payload).to_otel_proto().ByteSize() > 0
        admitted = admit_trace(trace_spec(root_inputs=payload, root_attributes={"probe": payload}))
        assert admitted.request.root_inputs == payload


@pytest.mark.parametrize("value", [1 << 63, -(1 << 63) - 1])
@pytest.mark.parametrize(
    "placement", ["root_inputs", "root_outputs", "root_attributes", "child_inputs", "child_outputs", "child_attributes"]
)
def test_otlp_int64_overflow_fails_before_any_gateway_mutation(fake_mlflow_client, value, placement):
    payload = {"nested": [value]}
    with pytest.raises(ValueError, match="out of range"):
        integer_transport_span(payload).to_otel_proto()
    gateway = checked_gateway(fake_mlflow_client)
    if placement.startswith("root_"):
        request = replace(trace_spec(), **{placement: payload})
    else:
        child = replace(child_spec("integer-child"), **{placement.removeprefix("child_"): payload})
        request = trace_spec(children=[child])
    before = list(fake_mlflow_client.mutations)
    with pytest.raises(ItemImportError) as error:
        gateway.create_trace(request)
    assert error.value.code == "trace_resource_value_invalid"
    assert fake_mlflow_client.mutations == before
    assert not fake_mlflow_client.trace_api.calls


def test_timestamps_keep_their_distinct_unsigned_fixed64_transport_range():
    instant = (1 << 64) - 1
    admitted = admit_trace(trace_spec(start_time_ns=instant, end_time_ns=instant))
    assert admitted.request.start_time_ns == instant
    span = integer_transport_span(1).to_otel_proto()
    span.start_time_unix_nano = span.end_time_unix_nano = instant
    assert span.ByteSize() > 0
    for invalid in (-1, 1 << 64, True):
        with pytest.raises(ItemImportError) as error:
            admit_trace(trace_spec(start_time_ns=invalid))
        assert error.value.code == "trace_resource_value_invalid"


def test_oversized_container_stops_before_iterating():
    class Giant(dict):
        def __len__(self):
            return 10**12

        def items(self):
            pytest.fail("giant container must not be walked")

    with pytest.raises(ItemImportError):
        admit_trace(trace_spec(root_inputs=Giant()))


def test_late_control_reservation_and_actual_decorated_overflow(fake_mlflow_client):
    gateway = checked_gateway(fake_mlflow_client)
    request = replace(trace_spec(), request_preview="x" * (1100 * 1024))
    assert admit_trace(request).span_count == 1
    with pytest.raises(ItemImportError):
        admit_trace(request, reserve_late_control=True)
    decorated = replace(request, response_preview="y" * (1100 * 1024))
    before = list(fake_mlflow_client.mutations)
    with pytest.raises(ItemImportError):
        gateway.create_trace(decorated)
    assert fake_mlflow_client.mutations == before


@pytest.mark.parametrize(
    "consumer",
    ["_get_trace", "replacement_reference", "verify_replacement_reference", "read_pricing_basis", "read_pricing_donor"],
)
def test_all_full_read_consumers_gate_unqualified_legacy_metadata_first(fake_mlflow_client, consumer):
    gateway = checked_gateway(fake_mlflow_client)
    created = gateway.create_trace(trace_spec())
    stored = fake_mlflow_client._traces[created.trace_id]
    for key in PROOF_TAG_KEYS:
        stored.info.tags.pop(key)
    fake_mlflow_client.calls.clear()
    argument = (
        created.trace_id if consumer in {"_get_trace", "replacement_reference"} else {"trace_id": created.trace_id}
    )
    with pytest.raises(ItemImportError) as error:
        getattr(gateway, consumer)(argument)
    assert error.value.code == "trace_resource_profile_unverified"
    assert not any(name == "get_trace" for name, _ in fake_mlflow_client.calls)


def test_existence_info_and_artifact_repository_are_metadata_only(fake_mlflow_client):
    gateway = checked_gateway(fake_mlflow_client)
    created = gateway.create_trace(trace_spec())
    for key in PROOF_TAG_KEYS:
        fake_mlflow_client._traces[created.trace_id].info.tags.pop(key)
    fake_mlflow_client.calls.clear()
    assert gateway.trace_exists(created.trace_id)
    assert not gateway.trace_exists("missing")
    assert gateway.trace_info(created.trace_id).trace_id == created.trace_id
    # Location resolution needs identity metadata, never span payloads.
    gateway._sink._repository_factory = lambda *args, **kwargs: SimpleNamespace()
    gateway._sink._trace_artifact_repository(created.trace_id)
    assert not any(name == "get_trace" for name, _ in fake_mlflow_client.calls)


def test_client_only_adapter_uses_exact_metadata_search_and_never_full_fallback():
    searches = []
    info = SimpleNamespace(trace_id="one", experiment_id="exp", tags={}, trace_metadata={})

    def search(**kwargs):
        searches.append(kwargs)
        return [SimpleNamespace(info=info)] if "'one'" in kwargs["filter_string"] else []

    def forbidden(*args, **kwargs):
        pytest.fail("metadata search must never fall back to get_trace")

    client = SimpleNamespace(tracking_uri="https://synthetic.invalid", search_traces=search, get_trace=forbidden)
    gateway = MlflowGateway(client)
    assert gateway.trace_info("one", "exp") is info
    assert gateway.trace_exists("one")
    assert not gateway.trace_exists("absent")
    assert all(item["include_spans"] is False for item in searches)
    assert searches[0]["locations"] == ["exp"]
    assert searches[0]["filter_string"] == "request_id = 'one'"


def test_unqualified_candidate_is_permanent_even_when_mapping_version_differs(fake_mlflow_client):
    gateway = checked_gateway(fake_mlflow_client)
    request = trace_spec()
    created = gateway.create_trace(request)
    owned = next(gateway.iter_owned_traces(request.experiment_id))
    candidate = OwnedTraceCandidate.from_owned_trace(owned, request.experiment_id)
    changed = replace(request, metadata={**request.metadata, "opencode.mapping_version": "old-mapping"})
    contract = CandidateContract.from_create_request(changed, "a" * 64, 1)
    for key in PROOF_TAG_KEYS:
        fake_mlflow_client._traces[created.trace_id].info.tags.pop(key)
    fake_mlflow_client.calls.clear()
    before = list(fake_mlflow_client.mutations)
    with pytest.raises(ItemImportError) as error:
        gateway.verify_owned_candidate(candidate, contract)
    assert error.value.code == "trace_resource_profile_unverified"
    assert not any(name == "get_trace" for name, _ in fake_mlflow_client.calls)
    assert fake_mlflow_client.mutations == before


def test_retirement_identity_checks_are_metadata_only(fake_mlflow_client):
    gateway = checked_gateway(fake_mlflow_client)
    request = trace_spec()
    created = gateway.create_trace(request)
    for key in PROOF_TAG_KEYS:
        fake_mlflow_client._traces[created.trace_id].info.tags.pop(key)
    fake_mlflow_client.calls.clear()
    gateway.retire_trace(
        request.experiment_id,
        created.trace_id,
        session_id=request.native_session_id,
        turn_key=request.metadata["opencode.turn_key"],
    )
    assert not gateway.trace_exists(created.trace_id)
    assert not any(name == "get_trace" for name, _ in fake_mlflow_client.calls)


def test_late_tags_do_not_spend_span_reservation_and_proof_is_idempotent():
    request = trace_spec(root_inputs={"messages": ["bounded"]})
    base = admit_trace(request, reserve_late_control=True)
    decorated = admit_trace(
        replace(
            request,
            root_tags={
                **request.root_tags,
                "mlflow_import.session_artifact": "x" * 4096,
                "mlflow_import.attempt_id": "a" * 32,
                "mlflow_import.import_instance_id": "b" * 32,
            },
        )
    )
    assert decorated.serialized_span_upper_bound == base.serialized_span_upper_bound
    assert admit_trace(decorated.request).proof_tags == decorated.proof_tags


def test_tag_mutations_cannot_stamp_proof_or_overflow_controls(fake_mlflow_client):
    gateway = checked_gateway(fake_mlflow_client)
    created = gateway.create_trace(trace_spec())
    before = list(fake_mlflow_client.mutations)
    with pytest.raises(ItemImportError) as error:
        gateway.reconcile_tags(created.trace_id, {TRACE_RESOURCE_BYTES_TAG: "1"}, prefix="unrelated.")
    assert error.value.code == "trace_resource_proof_immutable"
    with pytest.raises(ItemImportError):
        gateway.reconcile_tags(created.trace_id, {"managed.value": "x" * (3 * 1024 * 1024)}, prefix="managed.")
    assert fake_mlflow_client.mutations == before


def test_returned_count_and_bytes_must_match_initial_proof(fake_mlflow_client):
    gateway = checked_gateway(fake_mlflow_client)
    created = gateway.create_trace(trace_spec())
    gateway._get_trace(created.trace_id)
    stored = fake_mlflow_client._traces[created.trace_id]
    stored.data.spans.append(stored.data.spans[0])
    with pytest.raises(ItemImportError) as error:
        gateway._get_trace(created.trace_id)
    assert error.value.code == "trace_resource_proof_mismatch"
    stored.data.spans.pop()
    stored.info.tags[TRACE_RESOURCE_BYTES_TAG] = "1"
    with pytest.raises(ItemImportError):
        gateway._get_trace(created.trace_id)


@pytest.mark.parametrize(
    "value",
    [
        '"JSON string"',
        '{"k":["escaped\\\\value"]}',
        {"text": '😀\n\t\\"' * 10000, "array": [True, None, 1, 2.5]},
    ],
)
def test_bound_exceeds_independent_pinned_protobuf_and_captured_json_oracles(fake_mlflow_client, monkeypatch, value):
    """Capture the real historical writer's public SDK transport, without I/O."""
    from google.protobuf.json_format import MessageToJson
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    gateway = checked_gateway(fake_mlflow_client, real_api=True)
    captured = {}

    class CaptureClient:
        def __init__(self, **kwargs):
            pass

        def start_trace(self, info):
            captured["info"] = info

        def log_spans(self, *, location, spans):
            captured["spans"] = spans
            body = ExportTraceServiceRequest()
            body.resource_spans.add().scope_spans.add().spans.extend(span.to_otel_proto() for span in spans)
            captured["otlp"] = body.SerializeToString()
            captured["otlp_json"] = MessageToJson(body).encode()
            captured["fullread"] = json.dumps({"spans": [span.to_dict() for span in spans]}).encode()

    monkeypatch.setattr("mlflow.tracing.client.TracingClient", CaptureClient)
    request = replace(
        trace_spec(
            owned=False,
            experiment_id="1",
            root_inputs=value,
            root_outputs=value,
            root_attributes={"historical": value},
            children=[child_spec("c", inputs=value, outputs=value)],
        ),
        historical=True,
    )
    envelope = admit_trace(request)
    created = gateway.create_trace(request)
    assert created.trace_id
    assert len(captured["spans"]) == envelope.span_count
    assert sum(span.to_otel_proto().ByteSize() for span in captured["spans"]) < envelope.serialized_span_upper_bound
    for name in ("otlp", "otlp_json", "fullread"):
        assert len(captured[name]) < envelope.serialized_span_upper_bound
    assert require_trace_proof(captured["info"]) == (envelope.serialized_span_upper_bound, envelope.span_count)
