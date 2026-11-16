"""Conversation deltas stay linear and resolve to retained effective source facts."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mlflow_import.content.activity import activity_specs, project_activity
from mlflow_import.content.canonical import canonical_json
from mlflow_import.content.input_context import (
    INPUT_CONTEXT_ATTRIBUTE,
    TURN_MAP_VERSION,
    ConversationDelta,
    validate_input_contexts,
)
from mlflow_import.errors import ItemImportError
from mlflow_import.identity import qualified_session
from mlflow_import.mlflow.mapping import SourceIdentity, _iter_child_specs
from mlflow_import.source.model import SessionHeader, SourceRecord, TurnSummary

pytestmark = pytest.mark.unit


def record(key, role, content, *, scope="", table="jsonl_message", **extra):
    return SourceRecord(
        table, role, key,
        decoded_value={"role": role, "content": content, **extra},
        classification=role if role in {"user", "assistant"} else None,
        raw_columns={"subagent": scope} if scope else {},
        occurrence_id="occurrence:" + scope + key,
        invocation_id=key if role == "assistant" else None,
    )


def replay(records, harness="claude"):
    return SimpleNamespace(
        session_header=SessionHeader("s", "current", catalog_version=harness + "-persisted-v2"),
        summary=TurnSummary("s", 1, "turn-local", record_count=len(records)),
        iter_records=lambda: iter(records),
        iter_session_records=lambda: iter(()),
    )


def project(records, harness="claude"):
    turn = replay(records, harness)
    activity = project_activity(turn)
    return turn, activity, activity_specs(activity, turn)


def indexed(turn, activity, specs):
    contexts = [
        {"source_key": spec.source_key, "span_id": f"span:{i}", "descriptor": spec.attributes[INPUT_CONTEXT_ATTRIBUTE]}
        for i, spec in enumerate(specs) if INPUT_CONTEXT_ATTRIBUTE in spec.attributes
    ]
    return {
        "schema_version": TURN_MAP_VERSION,
        "turn_key": turn.summary.turn_key,
        "mapping_version": "mapping-v9",
        "identity_map": {"span:" + entry["source_key"]: entry["span_id"] for entry in contexts},
        "accounting": activity.provenance,
        "records": [
            {"source_key": SourceIdentity.of(r).key, "occurrence_id": r.occurrence_id,
             "entity_id": r.entity_id or r.source_id, "scope": str(r.raw_columns.get("subagent") or "")}
            for r in turn.iter_records()
        ],
        "input_contexts": contexts,
    }, {"mlflow_import.qualified_session_id": qualified_session(turn.session_header),
        "mlflow_import.turn_key": turn.summary.turn_key}


def conversation(count, *, size=64):
    records = [record("u", "user", "PROMPT:" + "p" * size)]
    for i in range(count):
        records.append(record(f"a{i}", "assistant", f"ANSWER-{i}:" + "a" * size, model="synthetic/model"))
        if i < count - 1:
            records.append(record(f"r{i}", "toolResult", f"RESULT-{i}:" + "t" * size,
                                  toolCallId=f"c{i}"))
    return records


@pytest.mark.parametrize("count", [10, 100, 1000])
def test_payload_and_context_index_grow_linearly(count):
    turn, activity, specs = project(conversation(count))
    calls = [spec for spec in specs if INPUT_CONTEXT_ATTRIBUTE in spec.attributes]
    assert len(calls) == count
    assert calls[0].inputs["messages"] == [{"role": "user", "content": "PROMPT:" + "p" * 64}]
    for i, spec in enumerate(calls[1:], 1):
        messages = spec.inputs["messages"]
        assert [m["role"] for m in messages] == ["assistant", "tool"]
        assert messages[0]["content"][0]["text"] == f"ANSWER-{i - 1}:" + "a" * 64
        assert messages[1]["content"] == f"RESULT-{i - 1}:" + "t" * 64
        assert spec.attributes[INPUT_CONTEXT_ATTRIBUTE]["previous_context_id"] == (
            calls[i - 1].attributes[INPUT_CONTEXT_ATTRIBUTE]["context_id"]
        )
    values = canonical_json([{"input": s.inputs, "output": s.outputs} for s in calls])
    assert values.count(b"PROMPT:") == 1
    assert values.count(b"ANSWER-0:") == 2  # Output + exactly one carry-forward.
    assert values.count(b"RESULT-0:") == 1
    assert len(values) < count * 800
    index, ownership = indexed(turn, activity, specs)
    assert len(canonical_json(index["input_contexts"])) < count * 2300
    assert validate_input_contexts(index, ownership)


def test_2105_child_shape_avoids_quadratic_allocation():
    # 1053 model calls plus 1052 unmatched result spans = 2105 children.
    # A 4 KiB prompt/result on the old cumulative path replicated GiB-scale data.
    turn, activity, specs = project(conversation(1053, size=4096))
    assert len(specs) == 2105
    calls = [s for s in specs if INPUT_CONTEXT_ATTRIBUTE in s.attributes]
    assert sum(len(canonical_json(s.inputs)) for s in calls) < 10 * 1024 * 1024
    index, ownership = indexed(turn, activity, specs)
    assert validate_input_contexts(index, ownership)


@pytest.mark.parametrize("recorded", [None, {}, [], "literal request", {"messages": ["full", "history"]}])
def test_explicit_recorded_request_inputs_are_never_compacted(recorded):
    records = conversation(2)
    records[-1] = replace(records[-1], decoded_value={**records[-1].decoded_value, "input": recorded})
    turn, activity, specs = project(records)
    spec = next(s for s in specs if s.source_key.endswith(":a1"))
    assert spec.inputs == recorded
    descriptor = spec.attributes[INPUT_CONTEXT_ATTRIBUTE]
    assert descriptor["origin"] == "recorded-request-input"
    assert descriptor["provider_request"] == "recorded-input-only"
    assert descriptor["delta"][0]["component"] == "input"
    index, ownership = indexed(turn, activity, specs)
    assert validate_input_contexts(index, ownership)


def test_no_observed_contribution_has_truthful_empty_input():
    _, _, specs = project([record("a", "assistant", "answer", model="synthetic/model")])
    (spec,) = specs
    assert spec.inputs is None
    descriptor = spec.attributes[INPUT_CONTEXT_ATTRIBUTE]
    assert descriptor["empty"] is True
    assert descriptor["delta"] == []
    assert descriptor["provider_request"] == "unknown"


@pytest.mark.parametrize("wrapper", ["message", "payload"])
def test_explicit_inputs_inside_preserved_source_envelopes_are_not_silently_derived(wrapper):
    actual = {"messages": ["full actual request history"], "parameters": {"temperature": 0}}
    records = conversation(2)
    records[-1] = replace(records[-1], decoded_value={**records[-1].decoded_value, wrapper: {"input": actual}})
    turn, activity, specs = project(records)
    call = next(s for s in specs if s.source_key.endswith(":a1"))
    assert call.inputs == actual
    assert call.attributes[INPUT_CONTEXT_ATTRIBUTE]["delta"][0]["component"] == wrapper + ".input"
    index, ownership = indexed(turn, activity, specs)
    assert validate_input_contexts(index, ownership)


def test_scope_isolation_interleaving_and_replay_stability():
    records = [
        record("u", "user", "primary prompt"),
        record("su", "user", "private subagent prompt", scope="child"),
        record("sa", "assistant", "private subagent answer", scope="child", model="synthetic/model"),
        record("a", "assistant", "primary answer", model="synthetic/model"),
        record("sb", "assistant", "private final", scope="child", model="synthetic/model"),
        record("b", "assistant", "primary final", model="synthetic/model"),
    ]
    turn, activity, specs = project(records)
    assert specs == activity_specs(activity, turn)
    calls = [s for s in specs if INPUT_CONTEXT_ATTRIBUTE in s.attributes]
    primary = [s for s in calls if s.attributes[INPUT_CONTEXT_ATTRIBUTE]["scope"] == ""]
    nested = [s for s in calls if s.attributes[INPUT_CONTEXT_ATTRIBUTE]["scope"] == "child"]
    assert b"private" not in canonical_json([s.inputs for s in primary])
    assert b"primary" not in canonical_json([s.inputs for s in nested])
    assert nested[0].parent_source_key == "agent:child"
    assert nested[0].attributes[INPUT_CONTEXT_ATTRIBUTE]["previous_context_id"] is None
    index, ownership = indexed(turn, activity, specs)
    assert validate_input_contexts(index, ownership)
    turn.session_header = replace(turn.session_header, title="changed sibling/session metadata")
    turn.summary = replace(turn.summary, turn_index=999)
    assert calls == [s for s in activity_specs(activity, turn) if INPUT_CONTEXT_ATTRIBUTE in s.attributes]


def test_coalesced_fragments_reference_effective_occurrences_not_provider_history():
    first = record("a", "assistant", "first fragment", model="synthetic/model")
    second = replace(first, source_id="b", occurrence_id="effective:patch:b",
                     decoded_value={"role": "assistant", "content": "effective second fragment"})
    final = record("final", "assistant", "done", model="synthetic/model")
    turn, activity, specs = project([record("u", "user", "prompt"), first, second, final], "gemini")
    calls = [s for s in specs if INPUT_CONTEXT_ATTRIBUTE in s.attributes]
    assert len(calls) == 2
    assert [r["occurrence_id"] for r in calls[0].attributes[INPUT_CONTEXT_ATTRIBUTE]["invocation_sources"]] == [
        "occurrence:a", "effective:patch:b"
    ]
    assert len(calls[1].inputs["messages"]) == 1
    assert calls[1].attributes[INPUT_CONTEXT_ATTRIBUTE]["provider_request"] == "unknown"
    assert b"effective second fragment" in canonical_json(calls[1].inputs)
    index, ownership = indexed(turn, activity, specs)
    assert validate_input_contexts(index, ownership)


def test_legacy_fallback_uses_same_delta_policy_and_preserves_tool_correlation():
    records = [record("u", "user", "prompt", table="message")]
    for i in range(3):
        records.append(record(f"a{i}", "assistant", "", table="message", model="synthetic/model"))
        records.append(SourceRecord(
            "part", "tool", f"t{i}", parent_source_id=f"a{i}", parent_source_table="message",
            decoded_value={"type": "tool", "tool": "read", "callID": f"c{i}",
                           "state": {"input": {"path": f"p{i}"}, "output": f"result{i}"}},
        ))
    turn = replay(records, "opencode")
    turn.load_root_input = lambda _: "prompt"
    specs = list(_iter_child_specs(turn, 10 * 1024 * 1024))
    calls = [s for s in specs if s.span_type == "LLM"]
    assert calls[0].inputs["messages"] == [{"role": "user", "content": "prompt"}]
    assert [m["role"] for m in calls[2].inputs["messages"]] == ["assistant", "tool"]
    assert calls[2].inputs["messages"][0]["tool_calls"][0]["id"] == "c1"
    assert calls[2].inputs["messages"][1] == {"role": "tool", "tool_call_id": "c1", "content": "result1"}
    assert b"result0" not in canonical_json(calls[2].inputs)


def test_pending_is_consumed_by_recorded_inputs_but_preserved_by_unavailable_accounting():
    delta = ConversationDelta("session", "turn")
    delta.observe("", {"role": "user", "content": "old"}, [])
    delta.consume("", "first", [], recorded=({"actual": "request"}, []))
    assert delta.consume("", "second", [])[0] is None
    delta.observe("", {"role": "user", "content": "old"}, [])
    delta.assistant("", {"role": "assistant", "content": "last actual assistant"}, [])
    _, unavailable = delta.consume("", "third", [], available=False)
    inputs, descriptor = delta.consume("", "fourth", [])
    assert inputs == {"messages": [
        {"role": "assistant", "content": "last actual assistant"}, {"role": "user", "content": "old"}
    ]}
    assert descriptor["previous_context_id"] == unavailable["context_id"]
    assert delta.consume("", "fifth", [])[0] is None


@pytest.mark.parametrize("harness", ["pi", "claude", "codex", "gemini", "opencode"])
@pytest.mark.parametrize("operation", ["usage", "compaction", "branch_summary"])
@pytest.mark.parametrize("with_tool", [False, True])
@pytest.mark.parametrize("decoded_type", [True, False])
def test_accounting_contexts_preserve_prompt_or_tool_and_previous_assistant(
    harness, operation, with_tool, decoded_type
):
    records = [record("u", "user", "full observed prompt")]
    if with_tool:
        records.extend([
            record("first", "assistant", "previous actual assistant", model="synthetic/model"),
            record("result", "toolResult", {"actual": [1, 2, 3]}, toolCallId="c"),
        ])
    accounting = SourceRecord(
        "jsonl", operation, "accounting", occurrence_id="accounting-occurrence",
        decoded_value={"type": operation, "model": "synthetic/model", "usage": {"input": 1, "output": 1},
                       "summary": "accounting summary"},
        classification=None if harness == "pi" else "assistant",
    )
    if not decoded_type:
        accounting = replace(
            accounting, decoded_value={k: v for k, v in accounting.decoded_value.items() if k != "type"}
        )
    records.extend([accounting, record("final", "assistant", "answer", model="synthetic/model")])
    turn, activity, specs = project(records, harness)
    accounting_spec = next(s for s in specs if s.source_key == "jsonl:accounting")
    descriptor = accounting_spec.attributes[INPUT_CONTEXT_ATTRIBUTE]
    assert accounting_spec.inputs is None
    assert descriptor["origin"] == "unavailable" and descriptor["delta"] == []
    final = next(s for s in specs if s.source_key == "jsonl_message:final")
    messages = final.inputs["messages"]
    if with_tool:
        assert [message["role"] for message in messages] == ["assistant", "tool"]
        assert messages[0]["content"][0]["text"] == "previous actual assistant"
        assert messages[1] == {"role": "tool", "tool_call_id": "c", "content": {"actual": [1, 2, 3]}}
        assert final.attributes[INPUT_CONTEXT_ATTRIBUTE]["delta"][1]["occurrence_id"] == "occurrence:result"
    else:
        assert messages == [{"role": "user", "content": "full observed prompt"}]
        assert final.attributes[INPUT_CONTEXT_ATTRIBUTE]["delta"][0]["occurrence_id"] == "occurrence:u"
    assert final.attributes[INPUT_CONTEXT_ATTRIBUTE]["previous_context_id"] == descriptor["context_id"]
    index, ownership = indexed(turn, activity, specs)
    assert validate_input_contexts(index, ownership)


def test_unmodeled_accounting_envelope_cannot_replace_the_last_actual_assistant():
    records = [
        record("u", "user", "prompt"),
        record("a", "assistant", "last actual assistant", model="synthetic/model"),
        record("usage", "assistant", "accounting summary", type="usage"),
        record("z", "assistant", "final", model="synthetic/model"),
    ]
    records[2] = replace(records[2], invocation_id=None)
    _, _, specs = project(records)
    final = next(s for s in specs if s.source_key == "jsonl_message:z")
    assert final.inputs["messages"][0]["content"][0]["text"] == "last actual assistant"
    assert b"accounting summary" not in canonical_json(final.inputs)


@pytest.mark.parametrize("damage", [
    "foreign-session", "foreign-turn", "foreign-scope", "unknown-occurrence", "cross-scope-link", "missing-context",
    "span-binding", "bad-version", "boolean-index", "duplicate-context",
])
def test_index_validation_rejects_unresolvable_or_misbound_provenance(damage):
    turn, activity, specs = project(conversation(2))
    index, ownership = indexed(turn, activity, specs)
    assert validate_input_contexts(index, ownership)
    index = copy.deepcopy(index)
    descriptor = index["input_contexts"][1]["descriptor"]
    if damage == "foreign-session":
        descriptor["qualified_session_id"] = "foreign"
    elif damage == "foreign-turn":
        descriptor["turn_key"] = "foreign"
    elif damage == "foreign-scope":
        descriptor["scope"] = "child"
    elif damage == "unknown-occurrence":
        descriptor["delta"][0]["occurrence_id"] = "missing"
    elif damage == "cross-scope-link":
        descriptor["previous_context_id"] = None
    elif damage == "missing-context":
        index["input_contexts"].pop()
    elif damage == "span-binding":
        index["input_contexts"][1]["span_id"] = "foreign"
    elif damage == "bad-version":
        descriptor["version"] = "future"
    elif damage == "boolean-index":
        descriptor["delta"][0]["message_index"] = True
    else:
        index["input_contexts"].append(index["input_contexts"][0])
    assert not validate_input_contexts(index, ownership)


def test_old_turn_map_remains_eligible_for_historical_pricing_decoding_only():
    assert validate_input_contexts({"schema_version": "mlflow-import.turn-map.v1", "mapping_version": "mapping-v8"}, {})
    assert not validate_input_contexts({"schema_version": "mlflow-import.turn-map.v1",
                                        "mapping_version": "mapping-v9"}, {})


@pytest.mark.parametrize("value", [None, True, [], {}, 4])
@pytest.mark.parametrize("field", ["origin", "scope", "invocation_id", "context_id", "delta"])
def test_validator_is_fail_closed_for_malformed_descriptor_fields(field, value):
    turn, activity, specs = project(conversation(2))
    index, ownership = indexed(turn, activity, specs)
    index["input_contexts"][1]["descriptor"][field] = value
    assert not validate_input_contexts(index, ownership)


def test_unmatched_tool_result_id_reuse_is_scope_isolated():
    records = [
        record("u", "user", "primary"),
        record("r", "toolResult", "primary result", toolCallId="same"),
        record("sr", "toolResult", "private result", scope="child", toolCallId="same"),
        record("a", "assistant", "done", model="synthetic/model"),
        record("sa", "assistant", "private done", scope="child", model="synthetic/model"),
    ]
    _, _, specs = project(records)
    tools = [s for s in specs if s.span_type == "TOOL"]
    assert len(tools) == 2
    assert len({s.source_key for s in tools}) == 2
    assert {s.outputs for s in tools} == {"primary result", "private result"}
    primary = next(s for s in specs if s.source_key == "jsonl_message:a")
    assert b"private" not in canonical_json(primary.inputs)


def index_with_actual_spans():
    turn, activity, specs = project(conversation(2))
    index, ownership = indexed(turn, activity, specs)
    index["identity_map"] = {"trace_id": "trace", **{f"span:{s.source_key}": f"span:{i}"
                                                    for i, s in enumerate(specs)}}
    spans = [SimpleNamespace(span_id="root", trace_id="trace", parent_id=None, attributes={})]
    spans.extend(SimpleNamespace(span_id=f"span:{i}", trace_id="trace", parent_id="root",
                                 attributes=copy.deepcopy(dict(spec.attributes))) for i, spec in enumerate(specs))
    ownership["opencode.mapping_version"] = "mapping-v9"
    return index, ownership, spans


def test_context_validation_binds_exactly_to_actual_trace_spans():
    index, ownership, spans = index_with_actual_spans()
    assert validate_input_contexts(index, ownership, spans=spans)


@pytest.mark.parametrize("duplicate", ["source-and-span", "span-only"])
def test_different_invocations_cannot_share_a_context_source_or_span(duplicate):
    index, ownership, _ = index_with_actual_spans()
    first, second = index["input_contexts"]
    second["span_id"] = first["span_id"]
    index["identity_map"]["span:" + second["source_key"]] = first["span_id"]
    if duplicate == "source-and-span":
        second["source_key"] = first["source_key"]
        second["descriptor"]["invocation_sources"].extend(first["descriptor"]["invocation_sources"])
    assert not validate_input_contexts(index, ownership)


@pytest.mark.parametrize("damage", [
    "actual-descriptor", "actual-invocation", "actual-source", "actual-missing-descriptor", "actual-extra-descriptor",
    "map-missing-context", "map-extra-context", "identity-extra", "identity-missing", "identity-wrong-trace",
    "actual-duplicate-span", "actual-duplicate-source", "actual-extra-span", "actual-missing-span", "missing-root",
    "actual-foreign-trace", "actual-descriptor-boolean",
])
def test_context_and_identity_tampering_cannot_be_hidden_by_a_checksummed_index(damage):
    index, ownership, spans = index_with_actual_spans()
    actual = spans[1]
    tool = spans[-1]
    assert INPUT_CONTEXT_ATTRIBUTE not in tool.attributes
    if damage == "actual-descriptor":
        actual.attributes[INPUT_CONTEXT_ATTRIBUTE]["empty"] = True
    elif damage == "actual-descriptor-boolean":
        actual.attributes[INPUT_CONTEXT_ATTRIBUTE]["delta"][0]["message_index"] = False
    elif damage == "actual-foreign-trace":
        actual.trace_id = "foreign"
    elif damage == "actual-invocation":
        actual.attributes["mlflow_import.invocation_id"] = "different invocation"
    elif damage == "actual-source":
        actual.attributes["mlflow_import.source_key"] = "foreign"
    elif damage == "actual-missing-descriptor":
        del actual.attributes[INPUT_CONTEXT_ATTRIBUTE]
    elif damage == "actual-extra-descriptor":
        tool.attributes[INPUT_CONTEXT_ATTRIBUTE] = copy.deepcopy(actual.attributes[INPUT_CONTEXT_ATTRIBUTE])
        tool.attributes["mlflow_import.invocation_id"] = actual.attributes["mlflow_import.invocation_id"]
    elif damage == "map-missing-context":
        removed = index["input_contexts"].pop()
        del index["accounting"]["invocations"][removed["descriptor"]["invocation_id"]]
    elif damage == "map-extra-context":
        index["input_contexts"].append(copy.deepcopy(index["input_contexts"][0]))
    elif damage == "identity-extra":
        index["identity_map"]["span:ghost"] = "ghost"
    elif damage == "identity-missing":
        del index["identity_map"]["span:" + tool.attributes["mlflow_import.source_key"]]
    elif damage == "identity-wrong-trace":
        index["identity_map"]["trace_id"] = "foreign"
    elif damage == "actual-duplicate-span":
        spans.append(copy.deepcopy(actual))
    elif damage == "actual-duplicate-source":
        tool.attributes["mlflow_import.source_key"] = actual.attributes["mlflow_import.source_key"]
    elif damage == "actual-extra-span":
        spans.append(SimpleNamespace(span_id="ghost", trace_id="trace", parent_id="root",
                                     attributes={"mlflow_import.source_key": "ghost"}))
    elif damage == "actual-missing-span":
        spans.remove(tool)
    else:
        spans.pop(0)
    assert not validate_input_contexts(index, ownership, spans=spans)


@pytest.mark.parametrize("schema,mapping,owner,valid", [
    ("mlflow-import.turn-map.v1", "mapping-v8", "mapping-v8", True),
    ("mlflow-import.turn-map.v1", "mapping-v8", "mapping-v9", False),
    ("mlflow-import.turn-map.v1", None, "mapping-v9", False),
    ("mlflow-import.turn-map.v1", "mapping-v9", "mapping-v9", False),
    (TURN_MAP_VERSION, "mapping-v9", "mapping-v8", False),
    (TURN_MAP_VERSION, "mapping-v8", "mapping-v8", False),
    (TURN_MAP_VERSION, "mapping-v9", "mapping-v9", True),
])
def test_map_version_cannot_downgrade_or_disagree_with_trace_ownership(schema, mapping, owner, valid):
    index, ownership, _ = index_with_actual_spans()
    index["schema_version"], index["mapping_version"] = schema, mapping
    ownership["opencode.mapping_version"] = owner
    if schema == "mlflow-import.turn-map.v1":
        del index["input_contexts"]
    assert validate_input_contexts(index, ownership) is valid


def test_projection_version_alias_also_blocks_v1_schema_downgrade():
    value = {"schema_version": "mlflow-import.turn-map.v1", "mapping_version": "mapping-v8"}
    assert not validate_input_contexts(value, {"mlflow_import.projection_version": "mapping-v9"})


def test_v1_schema_cannot_hide_actual_v2_span_descriptors_even_without_owner_version():
    index, ownership, spans = index_with_actual_spans()
    ownership.pop("opencode.mapping_version")
    index["schema_version"] = "mlflow-import.turn-map.v1"
    index["mapping_version"] = "mapping-v8"
    del index["input_contexts"]
    assert not validate_input_contexts(index, ownership, spans=spans)


def test_graph_budget_stops_at_first_excess_spec(monkeypatch):
    import mlflow_import.mlflow.trace_envelope as envelope

    original = envelope.SpanBudget
    budgets = []

    def budget(**kwargs):
        result = original(max_spans=3)
        budgets.append(result)
        return result

    monkeypatch.setattr(envelope, "SpanBudget", budget)
    with pytest.raises(ItemImportError, match="finite serialized resource envelope"):
        project(conversation(10))
    assert budgets[0].span_count == 3


def test_graph_budget_rejects_mapped_bytes_before_graph_is_complete(monkeypatch):
    import mlflow_import.mlflow.trace_envelope as envelope

    original = envelope.SpanBudget
    monkeypatch.setattr(envelope, "SpanBudget", lambda **_: original(max_bytes=16000))
    with pytest.raises(ItemImportError, match="finite serialized resource envelope"):
        project(conversation(10, size=4096))
