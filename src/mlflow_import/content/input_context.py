"""Pure, turn-local conversation deltas and their recoverable source index.

These are observed transcript contributions, not reconstructed provider requests.
The projector retains pending messages and one preceding assistant per scope;
neither payloads nor provenance contain cumulative history prefixes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mlflow_import.content.canonical import canonical_json
from mlflow_import.source.model import SourceRecord

INPUT_CONTEXT_VERSION = "mlflow-import.input-context.v1"
INPUT_CONTEXT_POLICY = "conversation-delta-v1"
INPUT_CONTEXT_ATTRIBUTE = "mlflow_import.input_context"
TURN_MAP_VERSION = "mlflow-import.turn-map.v2"


def context_identity(session: str, turn: str, scope: str, invocation: str) -> str:
    """Logical identity independent of session snapshots and artifact placement."""
    return "context:" + hashlib.sha256(
        canonical_json([INPUT_CONTEXT_VERSION, session, turn, scope, invocation])
    ).hexdigest()


def source_reference(record: SourceRecord, source_key: str, component: str) -> dict[str, Any]:
    """Address an effective occurrence in the full normalized session artifact."""
    return {
        "source_key": source_key,
        "occurrence_id": record.occurrence_id,
        "entity_id": record.entity_id or record.source_id,
        "component": component,
    }


@dataclass
class _Scope:
    pending: list[tuple[dict[str, Any], list[dict[str, Any]]]] = field(default_factory=list)
    assistant: tuple[dict[str, Any], list[dict[str, Any]]] | None = None
    previous: str | None = None


class ConversationDelta:
    """Consume each contribution once, with one adjacent assistant carry-forward."""

    def __init__(self, session: str, turn: str) -> None:
        self.session = session
        self.turn = turn
        self.scopes: dict[str, _Scope] = {}

    def observe(self, scope: str, message: dict[str, Any], sources: list[dict[str, Any]]) -> None:
        self.scopes.setdefault(scope, _Scope()).pending.append((message, sources))

    def consume(
        self,
        scope: str,
        invocation: str,
        sources: list[dict[str, Any]],
        *,
        recorded: tuple[object, list[dict[str, Any]]] | None = None,
        available: bool = True,
    ) -> tuple[object, dict[str, Any]]:
        state = self.scopes.setdefault(scope, _Scope())
        inputs: object = None
        delta: list[dict[str, Any]] = []
        origin = "unavailable"
        if recorded is not None:
            inputs, refs = recorded
            delta = [dict(ref, message_index=None) for ref in refs]
            origin = "recorded-request-input"
        elif available:
            contributions = ([state.assistant] if state.assistant is not None else []) + state.pending
            messages = [message for message, _ in contributions]
            delta = [dict(ref, message_index=index) for index, (_, refs) in enumerate(contributions) for ref in refs]
            inputs = {"messages": messages} if messages else None
            origin = "derived-observed-conversation"
        descriptor: dict[str, Any] = {
            "version": INPUT_CONTEXT_VERSION,
            "policy": INPUT_CONTEXT_POLICY,
            "context_id": context_identity(self.session, self.turn, scope, invocation),
            "qualified_session_id": self.session,
            "turn_key": self.turn,
            "scope": scope,
            "invocation_id": invocation,
            "previous_context_id": state.previous,
            "origin": origin,
            "provider_request": "recorded-input-only" if recorded is not None else "unknown",
            "empty": inputs is None or inputs == {"messages": []},
            "invocation_sources": sources,
            "delta": delta,
        }
        # Accounting observations advance provenance, not the conversation. Their
        # unavailable inputs do not consume a prompt/result needed by the next
        # model call, or replace the last actual assistant message.
        if available:
            state.pending = []
            state.assistant = None
        state.previous = descriptor["context_id"]
        return inputs, descriptor

    def assistant(self, scope: str, message: dict[str, Any], sources: list[dict[str, Any]]) -> None:
        self.scopes.setdefault(scope, _Scope()).assistant = message, sources


def recorded_input(records: Sequence[SourceRecord], keys: Sequence[str]) -> tuple[object, list[dict[str, Any]]] | None:
    """Preserve an explicit request input verbatim, including null/empty values."""
    for record, key in zip(records, keys, strict=True):
        value = record.decoded_value
        if isinstance(value, Mapping) and "input" in value:
            return value["input"], [source_reference(record, key, "input")]
        if isinstance(value, Mapping):
            for wrapper in ("message", "payload"):
                nested = value.get(wrapper)
                if isinstance(nested, Mapping) and "input" in nested:
                    return nested["input"], [source_reference(record, key, wrapper + ".input")]
    return None


def validate_input_contexts(
    mapvalue: object, ownership: Mapping[str, Any], *, spans: Sequence[Any] | None = None
) -> bool:
    """Fail-closed v2 index validation; v1 remains readable for historical pricing.

    This validates structure, logical ownership, source resolution, and scope-local
    predecessor chains. Supplying actual trace spans additionally requires exact
    descriptor, invocation, source-key, and complete identity-map binding. Artifact
    checksum/ownership verification remains the caller's responsibility.
    """
    if not isinstance(mapvalue, Mapping):
        return False
    declared_mapping = mapvalue.get("mapping_version")
    versions = [
        ownership[key]
        for key in ("opencode.mapping_version", "mlflow_import.mapping_version", "mlflow_import.projection_version")
        if key in ownership
    ]
    if any(not isinstance(version, str) or version != declared_mapping for version in versions):
        return False
    if mapvalue.get("schema_version") == "mlflow-import.turn-map.v1":
        if spans is not None:
            if not isinstance(spans, Sequence):
                return False
            for span in spans:
                attrs = getattr(span, "attributes", None)
                if not isinstance(attrs, Mapping) or INPUT_CONTEXT_ATTRIBUTE in attrs:
                    return False
        return declared_mapping != "mapping-v9" and "input_contexts" not in mapvalue
    if mapvalue.get("schema_version") != TURN_MAP_VERSION:
        return False
    if declared_mapping != "mapping-v9":
        return False
    contexts, records = mapvalue.get("input_contexts"), mapvalue.get("records")
    identity = mapvalue.get("identity_map")
    if not isinstance(contexts, list) or not isinstance(records, list) or not isinstance(identity, Mapping):
        return False
    session = ownership.get("mlflow_import.qualified_session_id")
    turn = ownership.get("mlflow_import.turn_key", ownership.get("opencode.turn_key"))
    if not isinstance(session, str) or not session or not isinstance(turn, str) or mapvalue.get("turn_key") != turn:
        return False
    refs: dict[tuple[str, str | None, str | None], str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            return False
        key = record.get("source_key")
        occurrence, entity = record.get("occurrence_id"), record.get("entity_id")
        scope = record.get("scope")
        if not isinstance(key, str) or not isinstance(scope, str) or any(
            value is not None and not isinstance(value, str) for value in (occurrence, entity)
        ):
            return False
        refs[(key, occurrence, entity)] = scope
    previous: dict[str, str] = {}
    seen: set[str] = set()
    invocations: set[str] = set()
    context_sources: set[str] = set()
    context_spans: set[str] = set()
    actual_contexts: dict[str, tuple[str, Mapping[str, Any]]] | None = None
    if spans is not None:
        if not isinstance(spans, Sequence):
            return False
        actual_contexts = _actual_span_contexts(spans, identity)
        if actual_contexts is None:
            return False
    for entry in contexts:
        if not isinstance(entry, Mapping):
            return False
        descriptor = entry.get("descriptor")
        source_key, span_id = entry.get("source_key"), entry.get("span_id")
        if (
            not isinstance(source_key, str)
            or not source_key
            or not isinstance(span_id, str)
            or not span_id
            or source_key in context_sources
            or span_id in context_spans
            or identity.get("span:" + source_key) != span_id
            or not isinstance(descriptor, Mapping)
        ):
            return False
        scope, invocation = descriptor.get("scope"), descriptor.get("invocation_id")
        if not isinstance(scope, str) or not isinstance(invocation, str) or not invocation:
            return False
        context = context_identity(session, turn, scope, invocation)
        origin = descriptor.get("origin")
        if (
            descriptor.get("version") != INPUT_CONTEXT_VERSION
            or descriptor.get("policy") != INPUT_CONTEXT_POLICY
            or descriptor.get("qualified_session_id") != session
            or descriptor.get("turn_key") != turn
            or descriptor.get("context_id") != context
            or context in seen
            or invocation in invocations
            or descriptor.get("previous_context_id") != previous.get(scope)
            or type(descriptor.get("empty")) is not bool
            or not isinstance(origin, str)
            or origin not in {"derived-observed-conversation", "recorded-request-input", "unavailable"}
            or descriptor.get("provider_request") != (
                "recorded-input-only" if origin == "recorded-request-input" else "unknown"
            )
        ):
            return False
        for field_name in ("invocation_sources", "delta"):
            values = descriptor.get(field_name)
            if not isinstance(values, list) or field_name == "invocation_sources" and not values:
                return False
            for ref in values:
                if not isinstance(ref, Mapping):
                    return False
                key, occurrence, entity = ref.get("source_key"), ref.get("occurrence_id"), ref.get("entity_id")
                if (
                    not isinstance(key, str)
                    or any(value is not None and not isinstance(value, str) for value in (occurrence, entity))
                    or refs.get((key, occurrence, entity)) != scope
                    or not isinstance(ref.get("component"), str)
                ):
                    return False
                if field_name == "delta":
                    index = ref.get("message_index")
                    if origin == "recorded-request-input":
                        if index is not None or ref.get("component") not in {"input", "message.input", "payload.input"}:
                            return False
                    elif type(index) is not int or index < 0:
                        return False
        if origin == "unavailable" and (descriptor["delta"] or not descriptor["empty"]):
            return False
        if origin == "derived-observed-conversation":
            indices = {ref["message_index"] for ref in descriptor["delta"]}
            if descriptor["empty"] != (not indices) or indices != set(range(len(indices))):
                return False
        if origin == "recorded-request-input" and len(descriptor["delta"]) != 1:
            return False
        if source_key not in {ref["source_key"] for ref in descriptor["invocation_sources"]}:
            return False
        if actual_contexts is not None:
            actual = actual_contexts.get(span_id)
            if actual is None or actual[0] != source_key:
                return False
            try:
                if canonical_json(actual[1]) != canonical_json(descriptor):
                    return False
            except (TypeError, ValueError, RecursionError):
                return False
        previous[scope] = context
        seen.add(context)
        invocations.add(invocation)
        context_sources.add(source_key)
        context_spans.add(span_id)
    accounting = mapvalue.get("accounting")
    return (
        isinstance(accounting, Mapping)
        and isinstance(accounting.get("invocations"), Mapping)
        and invocations == set(accounting["invocations"])
        and (actual_contexts is None or context_spans == actual_contexts.keys())
    )


def _actual_span_contexts(
    spans: Sequence[Any], identity: Mapping[str, Any]
) -> dict[str, tuple[str, Mapping[str, Any]]] | None:
    """Bind the entire map bijectively to the trace, including non-context spans."""
    actual_identity: dict[str, str] = {}
    contexts: dict[str, tuple[str, Mapping[str, Any]]] = {}
    seen_ids: set[str] = set()
    trace_ids: set[str] = set()
    roots = 0
    for span in spans:
        span_id = getattr(span, "span_id", None)
        attrs = getattr(span, "attributes", None)
        if not isinstance(span_id, str) or not span_id or span_id in seen_ids or not isinstance(attrs, Mapping):
            return None
        trace_id = getattr(span, "trace_id", None)
        if not isinstance(trace_id, str) or not trace_id:
            return None
        trace_ids.add(trace_id)
        seen_ids.add(span_id)
        parent = getattr(span, "parent_id", None)
        if parent is None:
            roots += 1
            actual_identity["trace_id"] = trace_id
            if INPUT_CONTEXT_ATTRIBUTE in attrs:
                return None
            continue
        key = attrs.get("mlflow_import.source_key")
        if not isinstance(key, str) or not key or "span:" + key in actual_identity:
            return None
        actual_identity["span:" + key] = span_id
        if INPUT_CONTEXT_ATTRIBUTE in attrs:
            descriptor = attrs[INPUT_CONTEXT_ATTRIBUTE]
            if not isinstance(descriptor, Mapping) or attrs.get("mlflow_import.invocation_id") != descriptor.get(
                "invocation_id"
            ):
                return None
            contexts[span_id] = key, descriptor
    if roots != 1 or len(trace_ids) != 1 or actual_identity != identity:
        return None
    return contexts
