"""Finite prospective maps use admitted snapshots and the exact upload builder."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mlflow_import.compatibility import request_ownership
from mlflow_import.content import accounting
from mlflow_import.content.activity import activity_specs, project_activity
from mlflow_import.content.canonical import iter_canonical_json
from mlflow_import.content.input_context import INPUT_CONTEXT_ATTRIBUTE, validate_input_contexts
from mlflow_import.content.turn_map import (
    MAX_GENERATED_SPAN_ID_CHARS,
    MAX_GENERATED_TRACE_ID_CHARS,
    _prospective_turn_map,
    build_turn_map,
    iter_turn_map_json,
    validate_turn_map_admission,
)
from mlflow_import.contracts import RedactionMode
from mlflow_import.errors import ItemImportError
from mlflow_import.identity import qualified_session
from mlflow_import.mlflow.trace_envelope import MAX_TRACE_SERIALIZED_SPAN_BYTES, admit_trace
from tests.support.fake_gateway import trace_spec
from tests.unit.content.test_input_context import conversation, record, replay

pytestmark = pytest.mark.unit


def admitted(records, *, harness="claude", mode=RedactionMode.VERBATIM):
    turn = replay(records, harness)
    activity = project_activity(turn)
    children = [s.to_child_request() for s in activity_specs(activity, turn)]
    header = turn.session_header
    ownership = {
        "mlflow_import.qualified_session_id": qualified_session(header),
        "mlflow_import.harness": harness,
        "mlflow_import.source_namespace": header.source_fingerprint or "unqualified",
        "mlflow_import.session_id": header.session_id,
        "opencode.turn_key": turn.summary.turn_key,
        "opencode.render_fingerprint": "f" * 64,
        "opencode.mapping_version": "mapping-v9",
        "opencode.importer_version": "0.1.0",
        "opencode.redaction_mode": mode.value,
    }
    envelope = admit_trace(trace_spec(
        metadata={}, root_tags=ownership, root_attributes={"mlflow_import.accounting": activity.provenance},
        children=children, owned=False,
    ))
    return turn, envelope


def actual_map(turn, request, *, trace_id="tr-" + "a" * 32, short_spans=False):
    children = tuple(request.child_specs())
    span_ids = {child.source_key: (f"s{i}" if short_spans else f"{i:016x}")
                for i, child in enumerate(children, 1)}
    return build_turn_map(
        turn, children=children, accounting=request.root_attributes["mlflow_import.accounting"],
        ownership=request_ownership(request), trace_id=trace_id,
        root_span_id="root" if short_spans else "f" * MAX_GENERATED_SPAN_ID_CHARS,
        span_ids_by_source=span_ids, mode=RedactionMode.VERBATIM,
    )


def streamed_size(value):
    return sum(len(chunk) for chunk in iter_canonical_json(value))


def test_finite_trace_with_500_long_source_ids_is_rejected_by_turn_map_preflight():
    records = [record("u", "user", "prompt")]
    records.extend(record(f"{i:03}-" + "x" * 1800, "assistant", "answer", model="synthetic/model")
                   for i in range(500))
    turn, envelope = admitted(records)
    assert envelope.span_count == 501
    assert envelope.serialized_span_upper_bound < MAX_TRACE_SERIALIZED_SPAN_BYTES
    # No gateway/target exists: this is the same admitted snapshot the importer
    # hands to global preflight, prior to experiment or attempt mutation.
    assert streamed_size(_prospective_turn_map(turn, envelope.request, RedactionMode.VERBATIM)) > (
        accounting.MAX_PRICING_MAP_BYTES
    )
    with pytest.raises(ItemImportError) as error:
        validate_turn_map_admission(turn, envelope.request, RedactionMode.VERBATIM)
    assert error.value.code == "turn_map_too_large"


@pytest.mark.parametrize("short_spans", [False, True])
@pytest.mark.parametrize("trace_id", ["tr-" + "a" * 32, "short-trace"])
def test_placeholder_reservation_covers_every_actual_identity_occurrence(short_spans, trace_id):
    records = conversation(4)
    records[0] = replace(records[0], source_id='prompt-雪-"\\\n')
    turn, envelope = admitted(records)
    prospective = streamed_size(_prospective_turn_map(turn, envelope.request, RedactionMode.VERBATIM))
    actual = streamed_size(actual_map(turn, envelope.request, trace_id=trace_id, short_spans=short_spans))
    assert prospective >= actual
    if not short_spans and len(trace_id) == MAX_GENERATED_TRACE_ID_CHARS:
        assert prospective == actual


def test_canonical_map_limit_is_exact_and_actual_upload_stream_retains_the_check(monkeypatch):
    turn, envelope = admitted(conversation(2))
    request = envelope.request
    size = streamed_size(_prospective_turn_map(turn, request, RedactionMode.VERBATIM))
    monkeypatch.setattr(accounting, "MAX_PRICING_MAP_BYTES", size)
    assert validate_turn_map_admission(turn, request, RedactionMode.VERBATIM) is None
    assert sum(map(len, iter_turn_map_json(actual_map(turn, request)))) == size
    monkeypatch.setattr(accounting, "MAX_PRICING_MAP_BYTES", size - 1)
    with pytest.raises(ItemImportError) as error:
        validate_turn_map_admission(turn, request, RedactionMode.VERBATIM)
    assert error.value.code == "turn_map_too_large"
    with pytest.raises(ItemImportError) as actual_error:
        list(iter_turn_map_json(actual_map(turn, request)))
    assert actual_error.value.code == "turn_map_too_large"


def test_preflight_uses_the_admitted_snapshot_once_without_reprojecting_or_repricing(monkeypatch):
    import mlflow_import.content.activity as activity

    turn, envelope = admitted(conversation(3))
    calls = []
    frozen = tuple(envelope.request.child_specs())

    def children():
        calls.append(True)
        assert len(calls) == 1
        yield from frozen

    def unexpected(*args, **kwargs):
        pytest.fail("turn-map admission must not re-run the mapper or accounting projector")

    monkeypatch.setattr(activity, "project_activity", unexpected)
    monkeypatch.setattr(activity, "iter_activity_specs", unexpected)
    request = replace(envelope.request, child_specs=children)
    validate_turn_map_admission(turn, request, RedactionMode.VERBATIM)
    assert calls == [True]


def test_late_artifact_nonce_tags_and_session_pointer_do_not_change_prospective_map():
    turn, envelope = admitted(conversation(3))
    request = envelope.request
    decorated = replace(request, artifact_relative_path="changed-create-nonce.json", root_tags={
        **request.root_tags, "opencode.artifact_path": "changed-create-nonce.json",
        "mlflow_import.turn_map": "changed-create-nonce.json", "mlflow_import.session_artifact": "later.json",
        "mlflow_import.attempt_id": "late-attempt",
    })
    before = b"".join(iter_turn_map_json(_prospective_turn_map(turn, request, RedactionMode.VERBATIM)))
    after = b"".join(iter_turn_map_json(_prospective_turn_map(turn, decorated, RedactionMode.VERBATIM)))
    assert before == after


def test_shared_builder_indexes_aliases_contexts_root_targets_and_all_actual_spans():
    records = conversation(2)
    first = records[1]
    records.insert(2, replace(first, source_id="fragment", occurrence_id="effective:fragment",
                              decoded_value={"role": "assistant", "content": "coalesced"}))
    turn, envelope = admitted(records)
    request = envelope.request
    value = json.loads(b"".join(iter_turn_map_json(actual_map(turn, request))))
    spans = [SimpleNamespace(span_id="f" * 16, trace_id="tr-" + "a" * 32, parent_id=None, attributes={})]
    spans.extend(SimpleNamespace(span_id=f"{i:016x}", trace_id="tr-" + "a" * 32,
                                 parent_id="f" * 16, attributes=child.attributes)
                 for i, child in enumerate(request.child_specs(), 1))
    assert validate_input_contexts(value, request_ownership(request), spans=spans)
    assert value["records"][0]["span_ids"] == ["f" * 16]
    by_source = {entry["source_id"]: entry["span_ids"] for entry in value["records"]}
    assert by_source[first.source_id] == by_source["fragment"]
    assert len(value["input_contexts"]) == sum(INPUT_CONTEXT_ATTRIBUTE in s.attributes for s in spans)


def test_context_validation_accepts_real_pinned_sdk_span_entities_without_a_target():
    from mlflow.entities import Span

    turn, envelope = admitted(conversation(2))
    request = envelope.request
    value = json.loads(b"".join(iter_turn_map_json(actual_map(turn, request))))
    trace_id = "tr-" + "a" * 32
    children = tuple(request.child_specs())

    def span(span_id, parent, attributes):
        return Span.from_dict({
            "trace_id": trace_id, "span_id": span_id, "parent_span_id": parent, "name": "local-only",
            "start_time_unix_nano": 1, "end_time_unix_nano": 2, "status": {"code": "STATUS_CODE_OK"},
            "attributes": {key: json.dumps(val) for key, val in {
                **attributes, "mlflow.traceRequestId": trace_id,
            }.items()},
        })

    spans = [span("f" * 16, None, {})]
    spans.extend(span(f"{i:016x}", "f" * 16, child.attributes) for i, child in enumerate(children, 1))
    assert validate_input_contexts(value, request_ownership(request), spans=spans)
