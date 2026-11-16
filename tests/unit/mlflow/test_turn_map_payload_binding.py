"""A fresh map checksum cannot authorize descriptors bound to different spans."""

import hashlib
import json
from dataclasses import replace

import pytest

from mlflow_import.compatibility import decode_ownership
from mlflow_import.content.canonical import canonical_json
from mlflow_import.content.input_context import context_identity, validate_input_contexts
from mlflow_import.contracts import RedactionMode
from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.capabilities import CapabilityRequest
from mlflow_import.mlflow.reconciliation import upload_artifact
from tests.support.compact_trace_fixture import mapped_source, synthetic_source
from tests.support.trace_resource_gateway import LocalTraceClient, LocalTraceGateway

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("reader", ["verify_replacement_reference", "read_pricing_basis", "read_pricing_donor"])
@pytest.mark.parametrize("damage", ["duplicate-span", "swap-span", "rewrite-invocation"])
def test_checksummed_context_map_must_bind_to_actual_writer_created_spans(tmp_path, reader, damage):
    source = synthetic_source(tmp_path, 2)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    client = LocalTraceClient()
    experiment = client.add_experiment("context-binding", artifact_location=artifacts.as_uri())
    gateway = LocalTraceGateway(client)
    assert gateway.probe(CapabilityRequest()).is_compatible
    with mapped_source(source, tmp_path) as [(turn, request)]:
        request = replace(request, experiment_id=experiment.experiment_id)
        created = gateway.create_trace(request)
        gateway.verify_trace(created, request)
        uploaded = upload_artifact(
            turn, created, request.artifact_relative_path, RedactionMode.VERBATIM,
            gateway=gateway, temporary_directory=tmp_path,
        )
    gateway.record_artifact_verification(created.trace_id, uploaded.sha256, uploaded.size)
    reference = gateway.replacement_reference(created.trace_id)
    gateway.verify_replacement_reference(reference)
    assert isinstance(gateway.read_pricing_donor(reference), dict)

    path = artifacts / reference["map"][0]
    value = json.loads(path.read_bytes())
    first, second = value["input_contexts"]
    trace = client._traces[created.trace_id]
    ownership = decode_ownership(trace.info.tags, trace.info.trace_metadata)
    assert validate_input_contexts(value, ownership, spans=trace.data.spans)
    if damage == "duplicate-span":
        second["span_id"] = first["span_id"]
        value["identity_map"]["span:" + second["source_key"]] = first["span_id"]
    elif damage == "swap-span":
        first["span_id"], second["span_id"] = second["span_id"], first["span_id"]
        for entry in (first, second):
            value["identity_map"]["span:" + entry["source_key"]] = entry["span_id"]
        substitutions = {first["span_id"]: second["span_id"], second["span_id"]: first["span_id"]}
        for record in value["records"]:
            record["span_ids"] = [substitutions.get(span_id, span_id) for span_id in record["span_ids"]]
    else:
        descriptor = first["descriptor"]
        old_context, old_invocation = descriptor["context_id"], descriptor["invocation_id"]
        descriptor["invocation_id"] = "forged-invocation"
        descriptor["context_id"] = context_identity(
            descriptor["qualified_session_id"], descriptor["turn_key"], descriptor["scope"],
            descriptor["invocation_id"],
        )
        for entry in value["input_contexts"]:
            if entry["descriptor"]["previous_context_id"] == old_context:
                entry["descriptor"]["previous_context_id"] = descriptor["context_id"]
        invocations = value["accounting"]["invocations"]
        invocations[descriptor["invocation_id"]] = invocations.pop(old_invocation)
        for record in value["records"]:
            if record["invocation_id"] == old_invocation:
                record["invocation_id"] = descriptor["invocation_id"]
    if damage != "duplicate-span":
        # These corruptions remain internally coherent without the actual
        # payload, so this fails if the gateway omits the spans= argument.
        assert validate_input_contexts(value, ownership)
    assert not validate_input_contexts(value, ownership, spans=trace.data.spans)

    payload = canonical_json(value)
    path.write_bytes(payload)
    gateway.record_artifact_verification(created.trace_id, hashlib.sha256(payload).hexdigest(), len(payload))
    # Capture a fresh, coherent reference after the map and recorded checksum
    # change. Only physical descriptor/span binding can reject it now.
    fresh_reference = gateway.replacement_reference(created.trace_id)
    assert fresh_reference["map"][1:] == [hashlib.sha256(payload).hexdigest(), len(payload)]
    with pytest.raises(ItemImportError) as error:
        getattr(gateway, reader)(fresh_reference)
    assert error.value.code == "replacement_verification_failed"
    assert error.value.safe_message == "the replacement input-context index is invalid"
