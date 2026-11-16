"""Harness-neutral activity projection over one frozen turn (no target APIs)."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from itertools import chain
from typing import Any

from mlflow_import.content.accounting import (
    ACCOUNTING_VERSION,
    cost_projection,
    invocation_key,
    merge_observation,
    normalize_usage,
    object_map,
    pricing_bases,
    sum_fields,
)
from mlflow_import.content.input_context import (
    INPUT_CONTEXT_ATTRIBUTE,
    ConversationDelta,
    recorded_input,
    source_reference,
)
from mlflow_import.identity import harness_name, qualified_session
from mlflow_import.source.model import SourceRecord
from mlflow_import.source.model_identity import extract_model_identity
from mlflow_import.source.replay import ReplayTurn


def enum_text(value: object) -> str:
    """Opaque permitted values remain in artifacts, never become enum labels."""
    return value if isinstance(value, str) else ""


def observed_status(value: Mapping[str, Any]) -> str:
    from mlflow_import.source.outcomes import classify_outcome, decoded_source_outcome

    outcome = decoded_source_outcome(value)
    if outcome is None and value.get("type") == "step-finish" and isinstance(value.get("reason"), str):
        outcome = value["reason"]
    if value.get("isError") is True or value.get("is_error") is True:
        return "ERROR"
    if outcome is None:
        return "UNSET"
    status = classify_outcome(outcome)
    if status != "OK":
        return status
    return (
        "OK"
        if outcome.lower()
        in {
            "stop",
            "end_turn",
            "completed",
            "complete",
            "success",
            "succeeded",
            "done",
            "length",
            "max_tokens",
            "tool_use",
            "tool_calls",
            "tool-calls",
            "tooluse",
        }
        else "UNSET"
    )


@dataclass
class Invocation:
    key: str
    records: list[SourceRecord] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    usage_observation: dict[str, Any] = field(default_factory=dict)
    aggregate_children: list[str] = field(default_factory=list)
    parent_key: str | None = None


@dataclass
class RecordView:
    """Reopen the indexed turn cursor; never retain a second transcript copy."""

    turn: ReplayTurn

    def __iter__(self) -> Iterator[SourceRecord]:
        return self.turn.iter_records()

    def __len__(self) -> int:
        return self.turn.summary.record_count


@dataclass
class Activity:
    records: RecordView
    calls: dict[str, Invocation]
    usage: dict[str, int]
    cost: dict[str, float]
    provenance: dict[str, Any]


def project_activity(turn: ReplayTurn) -> Activity:
    """Coalesce fragments before aggregation; never sum parent/child counters."""
    harness = harness_name(turn.session_header)
    records = RecordView(turn)
    calls: dict[str, Invocation] = {}
    turn_usage: dict[str, int] = {}
    current_model: str | None = None
    cumulative_usage: dict[str, int] = {}
    usage_events: list[dict[str, Any]] = []
    by_source: dict[str, str] = {}
    last_container: SourceRecord | None = None
    steps: dict[str, list[SourceRecord]] = {}
    for record in records:
        decoded = object_map(record.decoded_value)
        previous = record.raw_columns.get("usage_previous_cumulative")
        usage, scope = normalize_usage(
            harness,
            decoded,
            object_map(previous) if previous is not None else None,
            initial=record.raw_columns.get("usage_initial_observation") is True,
        )
        if scope.startswith("cumulative-"):
            cumulative_usage = sum_fields([cumulative_usage, usage])
            usage_events.append(
                {
                    "source_id": record.source_id,
                    "scope": scope,
                    "delta": usage,
                    "previous": previous,
                    "original": decoded.get("info"),
                }
            )
        if harness == "codex" and usage:
            turn_usage = usage
        model = extract_model_identity(decoded)
        if record.source_type == "turn_context" and model:
            current_model = model
        if record.source_table == "part" or record.derived_child_ordinal is not None:
            if usage or "cost" in decoded:
                key = by_source.get(record.parent_source_id or "")
                if (
                    key is None
                    and last_container is not None
                    and last_container.source_id == record.parent_source_id
                    and last_container.classification == "assistant"
                ):
                    key = invocation_key(last_container)
                    call = calls.setdefault(key, Invocation(key, records=[last_container], model=model))
                    by_source[last_container.source_id or ""] = key
                if key is not None:
                    call = calls[key]
                    if decoded.get("type") == "step-finish":
                        steps.setdefault(key, []).append(record)
                    # Message envelopes are authoritative over duplicated part
                    # counters/costs. Parts fill absent fields, never add totals.
                    merge_observation(
                        call.usage_observation, object_map(decoded.get("usage", decoded.get("tokens"))), override=False
                    )
                    call.usage = normalize_usage(harness, {"usage": call.usage_observation})[0]
                    if "cost" in decoded and not call.cost:
                        basis = (pricing_bases.get() or {}).get(turn.summary.turn_key, {}).get(key)
                        call.cost = cost_projection(harness, decoded, call.model, call.usage, basis)
            continue
        if record.source_table in {"message", "session_message", "jsonl_message"}:
            last_container = record
        role = decoded.get("role")
        assistant = record.classification == "assistant" or (isinstance(role, str) and role in {"assistant", "gemini"})
        if record.raw_columns.get("subagent") and (model or record.invocation_id):
            assistant = True
        operation = (
            harness == "pi"
            and record.source_type in {"usage", "compaction", "branch_summary"}
            and isinstance(decoded.get("usage"), Mapping)
        )
        if not (assistant or operation) or (record.source_table == "part" or record.derived_child_ordinal is not None):
            continue
        if (
            model is None
            and current_model is None
            and not usage
            and record.invocation_id is None
            and "cost" not in decoded
        ):
            continue  # Plain assistant activity is an AGENT, not an invented model invocation.
        key = invocation_key(record)
        call = calls.setdefault(key, Invocation(key))
        by_source[record.source_id or ""] = key
        call.records.append(record)
        call.model = model or call.model or current_model
        if scope == "invocation":
            # Fragments are repeated observations, not independent calls.
            # Latest present fields win; missing fields do not erase prior facts.
            merge_observation(call.usage_observation, object_map(decoded.get("usage", decoded.get("tokens"))))
            call.usage = normalize_usage(harness, {"usage": call.usage_observation})[0]
        basis = (pricing_bases.get() or {}).get(turn.summary.turn_key, {}).get(key)
        if "cost" in decoded or "cost" in object_map(decoded.get("usage")):
            call.cost = cost_projection(harness, decoded, call.model, call.usage, basis)
    for parent_key, observations in steps.items():
        if len(observations) < 2:
            continue
        parent = calls[parent_key]
        parent.usage_observation = {}
        parent.cost = {}
        for record in parent.records:
            decoded = object_map(record.decoded_value)
            merge_observation(parent.usage_observation, object_map(decoded.get("usage", decoded.get("tokens"))))
            if "cost" in decoded:
                parent.cost = cost_projection(
                    harness, decoded, parent.model, normalize_usage(harness, {"usage": parent.usage_observation})[0]
                )
        parent.usage = normalize_usage(harness, {"usage": parent.usage_observation})[0]
        for record in observations:
            key = "step:" + invocation_key(record)
            decoded = object_map(record.decoded_value)
            usage = normalize_usage(harness, decoded)[0]
            basis = (pricing_bases.get() or {}).get(turn.summary.turn_key, {}).get(key)
            child = Invocation(
                key,
                records=[record],
                model=extract_model_identity(decoded) or parent.model,
                usage=usage,
                usage_observation=dict(object_map(decoded.get("usage", decoded.get("tokens")))),
                parent_key=parent_key,
            )
            child.cost = cost_projection(harness, decoded, child.model, usage, basis)
            calls[key] = child
            parent.aggregate_children.append(key)
    for call in calls.values():
        if call.aggregate_children:
            children = [calls[key] for key in call.aggregate_children]
            child_usage = sum_fields(child.usage for child in children)
            call.usage = {**child_usage, **call.usage}
            if not call.cost:
                with localcontext() as context:
                    context.prec = 50
                    child_cost = sum_fields(
                        {key: Decimal(amount) for key, amount in child.cost["usd"].items()} for child in children
                    )
                call.cost = {
                    "origin": "importer-derived",
                    "method": "unknown",
                    "usd": {key: str(amount) for key, amount in child_cost.items()},
                    "scope": "sum-observed-steps",
                    "gap": None if all(child.cost["usd"] for child in children) else "partial_step_costs",
                }
            continue
        if not call.cost:
            basis = (pricing_bases.get() or {}).get(turn.summary.turn_key, {}).get(call.key)
            call.cost = cost_projection(harness, {"usage": call.usage_observation}, call.model, call.usage, basis)
    if cumulative_usage:
        turn_usage = cumulative_usage
    usage_total = turn_usage or sum_fields(call.usage for call in calls.values() if call.parent_key is None)
    billable = [call for call in calls.values() if call.parent_key is None]
    with localcontext() as context:
        context.prec = 50
        costs = sum_fields({key: Decimal(value) for key, value in call.cost["usd"].items()} for call in billable)
    cost_complete = {
        key: all(key in call.cost["usd"] and not call.cost.get("gap") for call in billable) for key in costs
    }
    turn_cost = None
    if turn_usage and not any(call.cost.get("origin") == "persisted" for call in calls.values()):
        models = {call.model for call in calls.values() if call.model}
        model = next(iter(models)) if len(models) == 1 else current_model if not models else None
        basis = (pricing_bases.get() or {}).get(turn.summary.turn_key, {}).get("turn-usage")
        turn_cost = cost_projection(harness, {}, model, turn_usage, basis)
        costs = {key: Decimal(value) for key, value in turn_cost["usd"].items()}
        cost_complete = {key: bool(usage_events) for key in costs}
    expected_counts = {"input_tokens", "output_tokens", "total_tokens"}

    def complete_call(call: Invocation) -> bool:
        if call.aggregate_children:
            return expected_counts <= normalize_usage(harness, {"usage": call.usage_observation})[0].keys() or all(
                complete_call(calls[key]) for key in call.aggregate_children
            )
        return expected_counts <= call.usage.keys()

    usage_complete = (
        bool(cumulative_usage)
        and all(
            expected_counts <= event["delta"].keys() and event["scope"] != "cumulative-reset" for event in usage_events
        )
        if usage_events
        else not turn_usage and bool(calls) and all(complete_call(call) for call in billable)
    )
    provenance = {
        "version": ACCOUNTING_VERSION,
        "usage_scope": "cumulative-delta"
        if usage_events
        else "turn-last-observation"
        if turn_usage
        else "coalesced-invocations",
        "invocations": {
            key: {
                "model": call.model,
                "usage": call.usage,
                "original_usage": call.usage_observation,
                "cost": call.cost,
                "parent_invocation": call.parent_key,
                "aggregate_children": call.aggregate_children,
                "usage_complete": complete_call(call),
            }
            for key, call in calls.items()
        },
        "turn_cost": turn_cost,
        "cumulative_observations": usage_events,
        "usage_aggregation": {
            "complete": usage_complete,
            "observed_fields": sorted(usage_total),
            "missing_invocations": [call.key for call in calls.values() if not expected_counts <= call.usage.keys()],
        },
        "cost_aggregation": {
            "scope": "sum-observed-only",
            "complete": cost_complete,
            "missing_invocations": [call.key for call in billable if not call.cost["usd"] or call.cost.get("gap")],
        },
    }
    native_cost = {key: float(value) for key, value in costs.items()}
    if any(not math.isfinite(value) for value in native_cost.values()) or any(
        value > 2**63 - 1 for value in usage_total.values()
    ):
        from mlflow_import.errors import SourceCompatibilityError

        raise SourceCompatibilityError(
            "native_accounting_out_of_range", "source accounting exceeds the qualified native numeric envelope"
        )
    return Activity(records, calls, usage_total, native_cost, provenance)


def content_parts(record: SourceRecord, records: tuple[SourceRecord, ...]) -> list[Mapping[str, Any]]:
    decoded = object_map(record.decoded_value)
    derived = [
        object_map(child.decoded_value)
        for child in records
        if child.parent_source_id == record.source_id
        and (child.source_table == "part" or child.derived_child_ordinal is not None)
    ]
    if derived:
        return derived  # These project the same parent content, not additional copies.
    content = decoded.get("content", decoded.get("parts"))
    parts: list[Mapping[str, Any]] = []
    if isinstance(content, str):
        parts.append({"type": "text", "text": content})
    elif isinstance(content, list):
        parts.extend(item if isinstance(item, Mapping) else {"type": "text", "text": item} for item in content)
    elif isinstance(decoded.get("text"), str):
        parts.append({"type": "text", "text": decoded["text"]})
    for child in records:
        if child.parent_source_id == record.source_id and (
            child.source_table == "part" or child.derived_child_ordinal is not None
        ):
            parts.append(object_map(child.decoded_value))
    return parts


def chat_message(call: Invocation, children: Mapping[str, list[SourceRecord]]) -> dict[str, Any]:
    parts = [
        part
        for record in call.records
        for part in content_parts(record, tuple(children.get(record.source_id or "", [])))
    ]
    message: dict[str, Any] = {"role": "assistant", "content": parts or None}
    tools = []
    reasoning = []
    for record in call.records:
        thoughts = object_map(record.decoded_value).get("thoughts")
        if thoughts is not None:
            reasoning.extend(thoughts if isinstance(thoughts, list) else [thoughts])
        for tool in object_map(record.decoded_value).get("toolCalls", []) or []:
            if isinstance(tool, Mapping):
                tools.append(
                    {
                        "id": tool.get("id"),
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "arguments": json.dumps(tool.get("args"), sort_keys=True),
                        },
                    }
                )
    for part in parts:
        kind = str(part.get("type"))
        if kind in {"thinking", "reasoning"} or part.get("thought"):
            reasoning.append(part.get("thinking", part.get("text")))
        if kind in {"tool_use", "toolCall", "tool"} or part.get("functionCall"):
            function = object_map(part.get("functionCall"))
            state = object_map(part.get("state"))
            tools.append(
                {
                    "id": part.get("id", part.get("callID", part.get("call_id", function.get("id")))),
                    "type": "function",
                    "function": {
                        "name": part.get("name", part.get("tool", function.get("name"))),
                        "arguments": json.dumps(
                            part.get("input", part.get("arguments", state.get("input", function.get("args")))),
                            sort_keys=True,
                        ),
                    },
                }
            )
    if reasoning:
        message["reasoning_content"] = "\n\n".join(
            value
            if isinstance(value, str)
            else "\n\n".join(
                str(item)
                for key in ("subject", "description", "text", "thinking")
                if (item := object_map(value).get(key)) is not None
            )
            or json.dumps(value, ensure_ascii=False, sort_keys=True)
            for value in reasoning
        )
        message["reasoning_details"] = reasoning
    if tools:
        message["tool_calls"] = tools
    return message


def activity_specs(activity: Activity, turn: ReplayTurn) -> list[Any]:
    """Emit AGENT/LLM/TOOL specs with complete structured content and provenance."""
    from mlflow_import.mlflow.mapping import SourceIdentity, SpanSpec, _common_attributes
    from mlflow_import.mlflow.trace_envelope import MAX_TRACE_SPANS, SpanBudget

    budget = SpanBudget(max_spans=MAX_TRACE_SPANS - 1)

    class SpecGraph(list[Any]):
        def append(self, item: Any) -> None:
            from dataclasses import fields

            # EventSpec's span_type is a property, not an init field. Account
            # its equivalent request shape without changing the public spec type.
            budget.account_spec(SpanSpec(**{field.name: getattr(item, field.name) for field in fields(SpanSpec)}))
            super().append(item)

    specs: list[Any] = SpecGraph()
    children_by_parent: dict[str, list[SourceRecord]] = {}
    for record in activity.records:
        if record.parent_source_id is not None and (
            record.source_table == "part" or record.derived_child_ordinal is not None
        ):
            children_by_parent.setdefault(record.parent_source_id, []).append(record)
    agents: dict[str, str] = {}
    root_start = turn.summary.start_time.nanoseconds if turn.summary.start_time else 1
    root_end = turn.summary.end_time.nanoseconds if turn.summary.end_time else root_start

    def times(records: list[SourceRecord]) -> tuple[int, int, str]:
        starts = [r.timestamp.nanoseconds for r in records if r.timestamp]
        ends = [instant.nanoseconds for r in records if (instant := r.completion_timestamp or r.timestamp) is not None]
        return (
            min(starts) if starts else root_start,
            max(ends) if ends else root_end,
            "timestamp-derived" if starts and ends else "synthetic-container",
        )

    def spec(
        key: str,
        name: str,
        kind: str,
        parent: str | None,
        inputs: object,
        outputs: object,
        attrs: dict[str, Any],
        records: list[SourceRecord],
        status: str = "OK",
    ) -> Any:
        start, end, timing = times(records)
        attrs["mlflow_import.timing"] = timing
        attrs["mlflow_import.source_key"] = key
        return SpanSpec(key, name, kind, parent, inputs, attrs, {}, start, outputs, status, max(start, end))

    call_by_record = {
        SourceIdentity.of(record).key: call
        for call in activity.calls.values()
        if call.parent_key is None
        for record in call.records
    }
    context = ConversationDelta(qualified_session(turn.session_header), turn.summary.turn_key)
    call_inputs: dict[str, object] = {}
    call_contexts: dict[str, dict[str, Any]] = {}
    scope_records: dict[str, list[SourceRecord]] = {}
    for record in activity.records:
        scope = str(record.raw_columns.get("subagent") or "")
        scope_records.setdefault(scope, []).append(record)
        decoded = object_map(record.decoded_value)
        accounting_only = record.source_type in {"usage", "compaction", "branch_summary"} or enum_text(
            decoded.get("type")
        ) in {"usage", "compaction", "branch_summary"}
        parts = content_parts(record, tuple(children_by_parent.get(record.source_id or "", [])))
        root_part = (
            decoded
            if enum_text(decoded.get("type")) in {"function_call_output", "tool_result"}
            or decoded.get("role") == "toolResult"
            else None
        )
        # A derived part is already represented by its container's content.
        derived = record.source_table == "part" or record.derived_child_ordinal is not None
        if derived and record.parent_source_id in children_by_parent:
            continue
        results = [
            (ordinal, part)
            for ordinal, part in enumerate(parts)
            if enum_text(part.get("type")) in {"tool_result", "function_call_output"} or part.get("functionResponse")
        ]
        if root_part is not None:
            results.append((-1, root_part))
        container = (
            record.source_table in {"message", "session_message", "jsonl_message"}
            and record.derived_child_ordinal is None
        )
        key = SourceIdentity.of(record).key

        def references(component: str, record: SourceRecord = record, key: str = key) -> list[dict[str, Any]]:
            children = children_by_parent.get(record.source_id or "", [])
            if (
                component == "content"
                and children
                and not isinstance(object_map(record.decoded_value).get("content"), str)
            ):
                return [source_reference(child, SourceIdentity.of(child).key, "content") for child in children]
            if component.startswith("tool:") and children:
                ordinal = int(component.split(":")[1])
                if ordinal < len(children):
                    child = children[ordinal]
                    return [source_reference(child, SourceIdentity.of(child).key, "state.output")]
            return [source_reference(record, key, component)]

        if container and not accounting_only and (record.classification == "user" or decoded.get("role") == "user"):
            content = decoded.get("content")
            result_ordinals = {i for i, _ in results}
            user_parts = [part for ordinal, part in enumerate(parts) if ordinal not in result_ordinals]
            if isinstance(content, str) or user_parts:
                context.observe(
                    scope,
                    {"role": "user", "content": content if isinstance(content, str) else user_parts},
                    references("content"),
                )
        call = call_by_record.get(key)
        if call is not None and call.key not in call_inputs:
            source_keys = [SourceIdentity.of(item).key for item in call.records]
            sources = [
                source_reference(item, item_key, "assistant")
                for item, item_key in zip(call.records, source_keys, strict=True)
            ]
            sources.extend(
                source_reference(child, SourceIdentity.of(child).key, "content")
                for item in call.records
                for child in children_by_parent.get(item.source_id or "", [])
            )
            call_inputs[call.key], call_contexts[call.key] = context.consume(
                scope,
                call.key,
                sources,
                recorded=recorded_input(call.records, source_keys),
                available=not accounting_only,
            )
            if not accounting_only:
                context.assistant(scope, chat_message(call, children_by_parent), sources)
        elif call is None and container and record.classification == "assistant" and not accounting_only:
            context.assistant(
                scope, chat_message(Invocation(key, records=[record]), children_by_parent), references("assistant")
            )
        if call is None and decoded.get("type") == "function_call":
            arguments = decoded.get("arguments")
            context.observe(
                scope,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": decoded.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": decoded.get("name"),
                                "arguments": arguments
                                if isinstance(arguments, str)
                                else json.dumps(arguments, sort_keys=True),
                            },
                        }
                    ],
                },
                references("function_call"),
            )
        for ordinal, part in results:
            response = object_map(part.get("functionResponse"))
            context.observe(
                scope,
                {
                    "role": "tool",
                    "tool_call_id": part.get(
                        "tool_use_id", part.get("call_id", part.get("toolCallId", response.get("id")))
                    ),
                    "content": part.get("output", part.get("content", response.get("response"))),
                },
                references("tool-result" if ordinal < 0 else f"content:{ordinal}:tool-result"),
            )
        # OpenCode and Gemini can retain a tool's actual result on its call part.
        for ordinal, part in enumerate([*parts, *(decoded.get("toolCalls") or [])]):
            if not isinstance(part, Mapping):
                continue
            state = object_map(part.get("state"))
            if "output" not in state and "result" not in part:
                continue
            context.observe(
                scope,
                {
                    "role": "tool",
                    "tool_call_id": part.get("id", part.get("callID", part.get("call_id"))),
                    "content": state.get("output", part.get("result")),
                },
                references(f"tool:{ordinal}:result"),
            )
    # Scope containers express recorded subagent identity, never invented primary turns.
    for record in activity.records:
        scope = str(record.raw_columns.get("subagent") or "")
        if scope and scope not in agents:
            key = "agent:" + scope
            agents[scope] = key
            specs.append(
                spec(
                    key,
                    scope,
                    "AGENT",
                    None,
                    next(
                        (
                            object_map(r.decoded_value).get("content")
                            for r in scope_records[scope]
                            if object_map(r.decoded_value).get("role") == "user" or r.source_type == "user"
                        ),
                        None,
                    ),
                    next(
                        (
                            object_map(r.decoded_value).get("content")
                            for r in reversed(scope_records[scope])
                            if enum_text(object_map(r.decoded_value).get("role")) in {"assistant", "gemini"}
                            or r.classification == "assistant"
                            or r.source_type in {"assistant", "gemini"}
                        ),
                        None,
                    ),
                    {"mlflow_import.agent_scope": scope, "mlflow_import.execution_parentage": "unknown"},
                    scope_records[scope],
                    observed_status(
                        object_map(
                            next(
                                (
                                    r.decoded_value
                                    for r in reversed(scope_records[scope])
                                    if r.source_type in {"assistant", "gemini"}
                                ),
                                None,
                            )
                        )
                    ),
                )
            )
    record_to_call: dict[str, str] = {}
    for call in activity.calls.values():
        first = call.records[0]
        key = SourceIdentity.of(first).key
        for record in call.records:
            record_to_call[record.source_id or ""] = key
        attrs = _common_attributes(first)
        attrs["mlflow_import.invocation_id"] = call.key
        attrs["mlflow_import.source_keys"] = [SourceIdentity.of(record).key for record in call.records]
        attrs["mlflow_import.accounting"] = call.cost
        if call.key not in call_contexts:
            _, call_contexts[call.key] = context.consume(
                str(first.raw_columns.get("subagent") or ""),
                call.key,
                [source_reference(record, SourceIdentity.of(record).key, "assistant") for record in call.records],
                available=False,
            )
        attrs[INPUT_CONTEXT_ATTRIBUTE] = call_contexts[call.key]
        attrs["mlflow_import.inputs_origin"] = call_contexts[call.key]["origin"]
        from mlflow_import.source.outcomes import decoded_source_outcome

        attrs["mlflow_import.finish_state"] = decoded_source_outcome(call.records[-1].decoded_value) or "unknown"
        if call.model and not call.aggregate_children:
            provider, separator, model = call.model.partition("/")
            attrs["mlflow.llm.model"] = model if separator else provider
            if separator and provider:
                attrs["mlflow.llm.provider"] = provider
            attrs["opencode.model"] = call.model
        if call.usage and not call.aggregate_children:
            attrs["mlflow.chat.tokenUsage"] = call.usage
        # An empty convention prevents mutable server-side auto-pricing of gaps.
        attrs["mlflow.llm.cost"] = {name: float(value) for name, value in call.cost["usd"].items()}
        message = chat_message(call, children_by_parent)
        operation = first.source_type in {"usage", "compaction", "branch_summary"}
        output = None if call.parent_key is not None else {"choices": [{"message": message}]}
        if operation:
            summary = object_map(first.decoded_value).get("summary")
            output = {"summary": summary} if summary is not None else None
        parent = (
            SourceIdentity.of(activity.calls[call.parent_key].records[0]).key
            if call.parent_key is not None
            else agents.get(str(first.raw_columns.get("subagent") or ""))
        )
        specs.append(
            spec(
                key,
                "llm_call",
                "AGENT"
                if call.aggregate_children
                else "LLM"
                if call.model or call.usage or first.invocation_id is not None or call.cost.get("origin") == "persisted"
                else "AGENT",
                parent,
                call_inputs.get(call.key) if call.parent_key is None else None,
                output,
                attrs,
                call.records,
                observed_status(object_map(call.records[-1].decoded_value)),
            )
        )
    tools: dict[tuple[str, str], dict[str, Any]] = {}
    for record in activity.records:
        decoded = object_map(record.decoded_value)
        has_derived = (record.source_id or "") in children_by_parent
        parts = [] if has_derived else content_parts(record, ())
        parts.extend(object_map(tool) for tool in decoded.get("toolCalls", []) if isinstance(tool, Mapping))
        if (
            str(decoded.get("type")) in {"tool", "function_call", "function_call_output", "tool_result"}
            or decoded.get("role") == "toolResult"
        ):
            parts.append(decoded)
        for ordinal, part in enumerate(parts):
            kind = str(part.get("type"))
            state = object_map(part.get("state"))
            function = object_map(part.get("functionCall"))
            response = object_map(part.get("functionResponse"))
            call_id = part.get(
                "tool_use_id", part.get("toolCallId", part.get("call_id", part.get("callID", part.get("id"))))
            )
            is_result = (
                kind in {"tool_result", "function_call_output"} or part.get("role") == "toolResult" or bool(response)
            )
            is_call = (
                kind in {"tool", "tool_use", "toolCall", "function_call"}
                or bool(function)
                or ("name" in part and "args" in part)
            )
            if not (is_result or is_call):
                continue
            if not isinstance(call_id, str):
                call_id = "unavailable:" + SourceIdentity.of(record).key + ":" + str(ordinal)
            scope = str(record.raw_columns.get("subagent") or "")
            qualified = scope, call_id
            tool = tools.setdefault(qualified, {"records": [], "id": call_id, "scope": scope})
            tool["records"].append(record)
            tool["error"] = (
                tool.get("error", False)
                or bool(part.get("is_error", part.get("isError")))
                or enum_text(state.get("status")) in {"error", "failed", "cancelled", "interrupted"}
                or state.get("error") is not None
                or enum_text(part.get("status")) in {"error", "failed", "cancelled", "interrupted"}
            )
            if enum_text(state.get("status")) in {"pending", "running", "queued"} or enum_text(part.get("status")) in {
                "pending",
                "running",
                "queued",
                "executing",
            }:
                tool["pending"] = True
            interval = object_map(state.get("time"))
            if interval and record.raw_columns.get("harness", "opencode") == "opencode":
                start, end = interval.get("start"), interval.get("end")
                if (
                    isinstance(start, int)
                    and not isinstance(start, bool)
                    and isinstance(end, int)
                    and not isinstance(end, bool)
                    and end >= start
                ):
                    tool["measured"] = (start * 1_000_000, end * 1_000_000)
            if is_call:
                tool.update(
                    name=part.get("name", part.get("tool", function.get("name", "tool"))),
                    inputs=part.get(
                        "input", part.get("arguments", part.get("args", state.get("input", function.get("args"))))
                    ),
                    parent=(
                        record_to_call.get(record.parent_source_id or "")
                        if record.source_table == "part" or record.derived_child_ordinal is not None
                        else record_to_call.get(record.source_id or "")
                        or record_to_call.get(record.parent_source_id or "")
                    ),
                    source_key=(
                        SourceIdentity.of(record).key
                        if record.source_table == "part" or record.derived_child_ordinal is not None
                        else SourceIdentity.of(record).key + f":tool:{ordinal}"
                    ),
                )
                if "output" in state or "result" in part:
                    tool["outputs"] = state.get("output", part.get("result"))
            if is_result:
                tool["outputs"] = part.get("output", part.get("content", response.get("response")))
    for tool in tools.values():
        if isinstance(tool.get("inputs"), str):
            try:
                tool["inputs"] = json.loads(tool["inputs"])
            except ValueError:
                pass  # Non-JSON tool inputs remain complete source strings.
        attrs = _common_attributes(tool["records"][0])
        attrs["mlflow_import.tool_call_id"] = tool["id"]
        attrs["mlflow_import.pairing"] = "paired" if "inputs" in tool and "outputs" in tool else "unmatched"
        attrs["mlflow_import.execution_parentage"] = "recorded" if tool.get("parent") else "unknown"
        attrs["mlflow_import.source_keys"] = [SourceIdentity.of(record).key for record in tool["records"]]
        parent = tool.get("parent") or agents.get(tool["scope"])
        result_identity = (
            json.dumps([tool["scope"], tool["id"]], separators=(",", ":")) if tool["scope"] else tool["id"]
        )
        tool_key = tool.get("source_key", "tool-result:" + result_identity)
        attrs["mlflow_import.source_key"] = tool_key
        specs.append(
            spec(
                tool_key,
                str(tool.get("name", "unmatched tool result")),
                "TOOL",
                parent,
                tool.get("inputs"),
                tool.get("outputs"),
                attrs,
                tool["records"],
                "ERROR"
                if tool["error"]
                else "UNSET"
                if tool.get("pending")
                else "OK"
                if "outputs" in tool
                else "UNSET",
            )
        )
        if "measured" in tool:
            from dataclasses import replace

            specs[-1] = replace(
                specs[-1],
                start_time_ns=tool["measured"][0],
                end_time_ns=tool["measured"][1],
                attributes={**specs[-1].attributes, "mlflow_import.timing": "measured"},
            )
    represented = {item.source_key for item in specs}
    represented.update(key for item in specs for key in item.attributes.get("mlflow_import.source_keys", []))
    represented.update(SourceIdentity.of(record).key for call in activity.calls.values() for record in call.records)
    from dataclasses import replace

    from mlflow_import.mlflow.mapping import _qualified_parent_source_key, map_source_record

    skip_first = True
    for record in chain(activity.records, turn.iter_session_records()):
        if skip_first:
            skip_first = False
            continue
        key = SourceIdentity.of(record).key
        if key in represented:
            continue
        parent = record_to_call.get(record.parent_source_id or "")
        if parent is None:
            source_parent = _qualified_parent_source_key(record)
            if source_parent in represented:
                parent = source_parent
        item = map_source_record(record, parent)
        start, end, timing = times([record])
        specs.append(
            replace(
                item,
                start_time_ns=start,
                end_time_ns=max(start, end),
                attributes={**item.attributes, "mlflow_import.timing": timing},
                status=observed_status(object_map(record.decoded_value))
                if item.span_type in {"AGENT", "LLM", "TOOL"}
                else item.status,
            )
        )
        represented.add(key)
    links = object_map(object_map(turn.session_header.metadata).get("execution_links"))
    for scope, agent_key in agents.items():
        link = object_map(links.get(scope))
        parent = next(
            (
                item.source_key
                for item in specs
                if item.span_type == "TOOL" and item.attributes.get("mlflow_import.tool_call_id") == link.get("call_id")
            ),
            None,
        )
        if parent is None and link.get("source_id"):
            parent = record_to_call.get(str(link["source_id"]))
        if parent is not None:
            parent_spec = next((item for item in specs if item.source_key == parent), None)
            specs = [
                replace(
                    item,
                    parent_source_key=parent,
                    inputs=item.inputs
                    if item.inputs is not None
                    else parent_spec.inputs
                    if parent_spec is not None
                    else None,
                    attributes={**item.attributes, "mlflow_import.execution_parentage": "recorded"},
                )
                if item.source_key == agent_key
                else item
                for item in specs
            ]
    # Source sidecars may precede their invoking TOOL in source order. Emit a
    # topological order so every public span references a previously logged parent.
    by_key = {item.source_key: item for item in specs}
    ordered: list[Any] = []
    emitted: set[str] = set()
    visiting: set[str] = set()

    def emit(key: str) -> None:
        if key in emitted:
            return
        if key in visiting:
            raise ValueError("cyclic execution parentage")
        visiting.add(key)
        item = by_key[key]
        if item.parent_source_key in by_key:
            emit(item.parent_source_key)
        ordered.append(item)
        visiting.remove(key)
        emitted.add(key)

    for item in specs:
        emit(item.source_key)
    return ordered


def iter_activity_specs(activity: Activity, turn: ReplayTurn) -> Iterator[Any]:
    """Stream plain activity; retain a graph only for actual execution links.

    A turn with 100k ordinary assistant envelopes does not need 100k model-call
    objects, a topological sort, or a second list of full span request objects.
    """
    plain = not activity.calls and all(
        record.source_table in {"message", "session_message", "jsonl_message"}
        and record.derived_child_ordinal is None
        and not record.raw_columns.get("subagent")
        and not object_map(record.decoded_value).get("toolCalls")
        and not object_map(record.decoded_value).get("tool_calls")
        and not isinstance(object_map(record.decoded_value).get("content"), list)
        for record in activity.records
    )
    if not plain:
        yield from activity_specs(activity, turn)
        return
    from dataclasses import replace

    from mlflow_import.mlflow.mapping import SourceIdentity, _qualified_parent_source_key, map_source_record

    emitted: set[str] = set()
    start = turn.summary.start_time.nanoseconds if turn.summary.start_time else 1
    end = turn.summary.end_time.nanoseconds if turn.summary.end_time else start
    for index, record in enumerate(activity.records):
        if index == 0:
            continue
        key = SourceIdentity.of(record).key
        parent = _qualified_parent_source_key(record)
        item = map_source_record(record, parent if parent in emitted else None)
        yield replace(
            item,
            start_time_ns=item.start_time_ns if item.start_time_ns is not None else start,
            end_time_ns=item.end_time_ns if item.end_time_ns is not None else end,
            attributes={
                **item.attributes,
                "mlflow_import.timing": "timestamp-derived" if record.timestamp else "synthetic-container",
            },
            status=observed_status(object_map(record.decoded_value)),
        )
        emitted.add(key)
    for record in turn.iter_session_records():
        key = SourceIdentity.of(record).key
        if key not in emitted:
            item = map_source_record(record, None)
            yield replace(
                item,
                start_time_ns=item.start_time_ns if item.start_time_ns is not None else start,
                end_time_ns=item.end_time_ns if item.end_time_ns is not None else end,
            )
            emitted.add(key)
