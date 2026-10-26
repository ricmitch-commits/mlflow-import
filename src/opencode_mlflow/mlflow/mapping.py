"""Pure source-to-trace mapping: ``ReplayTurn`` in, gateway request specs out.

This module maps one logical OpenCode turn (through the
:class:`~opencode_mlflow.source.replay.ReplayTurn` protocol owned
exclusively by :mod:`opencode_mlflow.source.replay`) to the request
specs the MLflow gateway consumes. It never calls MLflow and imports no
MLflow module or client: importing :mod:`opencode_mlflow.mlflow.mapping`
never loads ``mlflow`` (pinned by tests through an AST scan and a fresh
interpreter check). The one sanctioned boundary is
:meth:`TraceSpec.to_create_request`/:meth:`SpanSpec.to_child_request`/
:meth:`EventSpec.to_child_request`, which import the request types
lazily *inside the method body*: conversion happens only when a caller
above the gateway (the write phase) asks for a real request, at which
point MLflow is already loaded.

Boundaries fixed here:

- **Spec shapes.** :class:`TraceSpec`, :class:`SpanSpec`, and
  :class:`EventSpec` carry EXACTLY the gateway request fields
  (:class:`~opencode_mlflow.mlflow.gateway.TraceCreateRequest` and
  ``TraceChildRequest``, contract-tested by field-name parity), so a
  spec is a duck-typed producer of its request; ``to_*_request``
  converts structurally and lazily. Events ride spans typed
  ``event`` (:data:`EVENT_SPAN_TYPE`) because the public trace API has
  no separate event mutation.
- **Children.** :func:`map_turn_to_trace` yields children lazily. Current
  schema rows map every turn record except the first-user prompt through
  :func:`map_source_record`; legacy rows project only native LLM/tool spans
  while preserving all other records in the turn artifact. Each
  ``child_specs()`` call opens fresh replay cursors.
- **Typed children.** Message containers carrying model-call metadata
  map to LLM spans (model identity, provider, typed input/output,
  tokens, cost, finish state); other message containers map to agent
  spans; tool parts map to tool spans (tool name, call ID, state
  input/output, status, error data); text parts map to content spans
  while reasoning/error/step-marker and other annotation parts,
  lifecycle records, and unknown opaque records map to events (unknown
  records carry the normalized record form as an attribute when size
  allows). Source IDs/types/sequences/timestamps are preserved in every
  spec's attributes. Token/cost totals ride namespaced
  ``opencode.tokens.*``/``opencode.cost`` attributes on the LLM spans
  and the root: the compatibility profile pins no native MLflow
  token/cost attribute names, so none are claimed supported.
- **Payload limits.** Root values load through the replay loaders with
  the negotiated inline limit and a COMPLETE value above it raises a
  safe ``payload_too_large`` error BEFORE any trace could be created.
  Oversized child payloads, child attributes, and root attribute values
  are never truncated: they become the Task 8 digest/size reference
  inline while the complete value stays in the turn's artifact. Trace
  metadata values above :data:`METADATA_VALUE_CHAR_LIMIT` (the
  documented server truncation limit) are replaced by the reference
  marker's canonical JSON text for the same reason.
- **Timing.** Start is the first user record's timestamp, else the
  first timestamped record anywhere in the turn, else the source session
  creation instant; end is the last record's timestamp, else the
  resolved start. Every fallback marks ``opencode.timing_estimated``.
  When ALL source timestamps are absent both times are omitted. The
  local wall clock is never consulted.
- **Outcome.** The mapping is authoritative for the root status: a turn
  that ended with a non-error assistant result maps to ``OK`` (the
  explicit-failure case — the final result carrying an error part — maps
  to ``ERROR``), and a turn without an assistant result maps to
  ``UNSET`` with ``opencode.outcome`` recording the source terminal
  outcome. The explicit ``root_status`` rides the request so the
  gateway never re-derives what the source knows.
- **Ownership.** The initial trace-create request carries the complete
  namespaced ownership/operation metadata
  (``opencode.importer`` … ``opencode.operation_key``) plus available
  session/source metadata, so everything survives a crash before trace
  completion. It also records the artifact's exact relative path as the
  ``opencode.artifact_path`` root tag, so a later run can verify a
  candidate's stored artifact at the candidate's own recorded location
  (the path is not an equivalence field). ``operation_key`` is the
  SHA-256 of the exact experiment/session/turn/render tuple with
  versioned separators; the deterministic ``client_request_id`` equals
  it.
- **Identity binding.** :class:`ArtifactIdentityEnvelope` records the
  pre-identity-map artifact identity (render/snapshot fingerprints and
  the deterministic artifact path). :func:`bind_mlflow_identity_map`
  binds generated MLflow IDs to that envelope purely — after every
  fingerprint was fixed — so generated IDs never feed a source, render,
  or artifact-snapshot fingerprint.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from opencode_mlflow.content.artifacts import (
    artifact_relative_path,
    inline_value_or_reference,
    load_root_values,
    resolve_inline_limit,
)
from opencode_mlflow.content.canonical import canonical_json, canonical_json_compatible
from opencode_mlflow.content.fingerprint import (
    artifact_snapshot_fingerprint,
    build_effective_turn,
    canonical_record_value,
    render_fingerprint,
    source_fingerprint,
)
from opencode_mlflow.contracts import IDENTITY_REDACTOR, RedactionMode
from opencode_mlflow.mlflow.reconciliation import ARTIFACT_PATH_TAG
from opencode_mlflow.source.model import SessionHeader, SourceRecord, TurnSummary
from opencode_mlflow.source.model_identity import extract_model_identity
from opencode_mlflow.source.outcomes import (
    SESSION_EVENT_TABLES,
    classify_decoded,
    classify_outcome,
    declared_source_outcome,
    decoded_source_outcome,
    is_explicit_failure,
    record_source_outcome,
    tool_source_outcome,
)
from opencode_mlflow.source.replay import ReplayTurn
from opencode_mlflow.source.turns import (
    ASSISTANT_ERROR_OUTCOME,
    ASSISTANT_RESULT_OUTCOME,
    NO_ASSISTANT_RESULT_OUTCOME,
    PART_ERROR_OUTCOME,
    _is_root_result,
)

if TYPE_CHECKING:
    from opencode_mlflow.mlflow.gateway import TraceChildRequest, TraceCreateRequest

__all__ = [
    "EVENT_SPAN_TYPE",
    "IMPORTER_IDENTITY",
    "METADATA_VALUE_CHAR_LIMIT",
    "OPERATION_KEY_VERSION",
    "OUTCOME_EXPLICIT_FAILURE",
    "ArtifactIdentityEnvelope",
    "EventSpec",
    "MappingContext",
    "MlflowIdentityMap",
    "SourceIdentity",
    "SpanSpec",
    "TraceSpec",
    "TraceStatus",
    "bind_mlflow_identity_map",
    "map_source_record",
    "map_turn_to_trace",
    "operation_key",
]

#: Span type every event converts to. The public trace API has no separate
#: event mutation, so events ride spans typed ``event``.
EVENT_SPAN_TYPE = "event"

#: The importer ownership marker value; equals the gateway's
#: ``OWNERSHIP_IMPORTER_VALUE`` (the contract test pins them together so
#: the two can never drift).
IMPORTER_IDENTITY = "opencode-mlflow"

#: Character limit above which a trace-metadata value is replaced by the
#: digest/reference marker: the pinned server silently truncates metadata
#: values at 250 characters, and a value is never allowed to be silently
#: truncated when the complete value stays in the turn's artifact.
METADATA_VALUE_CHAR_LIMIT = 250

#: Version header of the operation-key hash input (versioned separators:
#: the header plus NUL separators make the tuple framing unambiguous).
OPERATION_KEY_VERSION = "opencode-mlflow-operation-key-v1"

#: Turn outcome when the final assistant result carries an explicit error
#: part: the turn failed explicitly even though a result exists.
OUTCOME_EXPLICIT_FAILURE = "explicit_failure"

#: Span types for the typed child mapping. MLflow's public span-type
#: vocabulary has no content type, so produced-content parts use
#: ``UNKNOWN`` while annotation parts ride the event pseudo-type.
_LLM_SPAN_TYPE = "LLM"
_AGENT_SPAN_TYPE = "AGENT"
_TOOL_SPAN_TYPE = "TOOL"
_CONTENT_SPAN_TYPE = "UNKNOWN"

_MESSAGE_TABLES = frozenset({"message", "session_message"})
_LIFECYCLE_TABLES = SESSION_EVENT_TABLES - {"todo"}
_SESSION_EVENT_TABLES = SESSION_EVENT_TABLES

_TRACE_NAME_PREFIX = "opencode turn"


TraceStatus = Literal["OK", "ERROR", "UNSET"]
"""The span statuses the mapping derives from source terminal outcomes."""


# ---------------------------------------------------------------------------
# Context and identities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappingContext:
    """Deterministic mapping inputs beyond the turn itself.

    Carries the render-contract versions, the source schema/catalog/database
    fingerprint, the redaction mode, the target experiment, and the
    session-level counts that ride the ownership metadata, plus the
    negotiated inline payload limit (``None`` applies Task 8's
    conservative default through
    :func:`~opencode_mlflow.content.artifacts.resolve_inline_limit`).
    Everything here is deterministic input to the request and
    idempotency keys — never runtime state.
    """

    importer_version: str
    mapping_version: str
    artifact_schema_version: str
    source_schema: str
    redaction_mode: RedactionMode
    experiment_id: str
    source_database_fingerprint: str | None = None
    inline_limit_bytes: int | None = None
    logical_turn_count: int | None = None
    source_message_count: int | None = None
    source_part_count: int | None = None
    catalog_version: str = ""


@dataclass(frozen=True)
class SourceIdentity:
    """The stable source identity of one record: its child source key.

    ``stable_key`` is the record's source ID when present, else the
    adapter's fallback identity, else a deterministic digest of the
    canonical record form — so every mapped record has a non-empty key
    for parent linkage and the identity map. ``source_table`` retains
    the record's provenance.
    """

    source_table: str
    stable_key: str

    @classmethod
    def of(cls, record: SourceRecord) -> SourceIdentity:
        if record.source_id is not None:
            stable_key = record.source_id
        elif record.source_identity is not None:
            stable_key = record.source_identity
        else:
            digest = hashlib.sha256(canonical_json(canonical_record_value(record))).hexdigest()
            stable_key = f"record-{digest}"
        return cls(source_table=record.source_table, stable_key=stable_key)

    @property
    def key(self) -> str:
        """The collision-proof child source key qualified by its table."""
        return f"{self.source_table}:{self.stable_key}"


@dataclass(frozen=True)
class ArtifactIdentityEnvelope:
    """The pre-identity-map artifact identity of one imported turn.

    Everything the artifact writer and reconciliation need to identify a
    turn's artifact BEFORE any MLflow identity exists: the source
    session/turn identity, the render fingerprint, the pre-identity-map
    artifact-snapshot fingerprint, and the deterministic relative path
    calculated from them. Generated IDs bind to this envelope only
    afterwards, through :func:`bind_mlflow_identity_map`.
    """

    session_id: str
    turn_key: str
    render_fingerprint: str
    snapshot_fingerprint: str
    artifact_relative_path: str


@dataclass(frozen=True)
class MlflowIdentityMap:
    """Generated MLflow identities for one imported turn, bound to its source.

    Binds AFTER the artifact fingerprints were fixed (through the
    envelope), so generated IDs never feed a source, render, or
    artifact-snapshot fingerprint. :meth:`artifact_section` renders the
    flat string mapping the turn artifact persists as its
    ``identity_map`` section: the generated trace id plus each child
    source key's span id under collision-proof ``span:``-prefixed keys.
    """

    session_id: str
    turn_key: str
    render_fingerprint: str
    snapshot_fingerprint: str
    artifact_relative_path: str
    trace_id: str
    span_ids_by_source: Mapping[str, str] = field(default_factory=dict)

    def artifact_section(self) -> dict[str, str]:
        """The flat ``identity_map`` artifact section (trace + span ids)."""
        section = {"trace_id": self.trace_id}
        for source_key in sorted(self.span_ids_by_source):
            section[f"span:{source_key}"] = self.span_ids_by_source[source_key]
        return section


def bind_mlflow_identity_map(
    source_envelope: ArtifactIdentityEnvelope,
    trace_id: str,
    span_ids_by_source: Mapping[str, str],
) -> MlflowIdentityMap:
    """Bind generated MLflow identities to one turn's artifact envelope (pure).

    Copies the envelope's pre-identity-map identity and the generated
    ids into an :class:`MlflowIdentityMap`; nothing is mutated and no
    fingerprint input changes, because generated ids enter only the
    returned map and the artifact's ``identity_map`` section — never a
    source, render, or artifact-snapshot hash (those were fixed on the
    envelope before any MLflow identity existed).
    """
    return MlflowIdentityMap(
        session_id=source_envelope.session_id,
        turn_key=source_envelope.turn_key,
        render_fingerprint=source_envelope.render_fingerprint,
        snapshot_fingerprint=source_envelope.snapshot_fingerprint,
        artifact_relative_path=source_envelope.artifact_relative_path,
        trace_id=trace_id,
        span_ids_by_source=MappingProxyType(dict(span_ids_by_source)),
    )


def operation_key(experiment_id: str, session_id: str, turn_key: str, render_fingerprint: str) -> str:
    """SHA-256 of the exact experiment/session/turn/render tuple.

    The versioned header and single-NUL separators frame the tuple
    unambiguously, so two different tuples can never concatenate to the
    same hash input. The result is both the trace's operation key
    (``opencode.operation_key``) and its deterministic
    ``client_request_id`` idempotency key.
    """
    material = "\0".join((OPERATION_KEY_VERSION, experiment_id, session_id, turn_key, render_fingerprint))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanSpec:
    """One typed child span mapped from one source record.

    Field parity with the gateway's ``TraceChildRequest`` is exact
    (contract-tested), so the spec is a duck-typed producer of the
    request. :func:`map_source_record` returns specs with COMPLETE
    source values; :func:`map_turn_to_trace` bounds every payload
    against the negotiated inline limit before a spec reaches the
    gateway. The mapping never sets child tags because span-level tags
    update the whole trace on the pinned server.
    """

    source_key: str
    name: str
    span_type: str
    parent_source_key: str | None
    inputs: object | None
    attributes: Mapping[str, object]
    tags: Mapping[str, str]
    start_time_ns: int | None
    outputs: object | None
    status: str
    end_time_ns: int | None

    def to_child_request(self) -> TraceChildRequest:
        """Convert to the real gateway child request (lazy boundary)."""
        from opencode_mlflow.mlflow.gateway import TraceChildRequest

        return TraceChildRequest(
            source_key=self.source_key,
            name=self.name,
            span_type=self.span_type,
            parent_source_key=self.parent_source_key,
            inputs=canonical_json_compatible(self.inputs) if self.inputs is not None else None,
            attributes=_canonical_mapping(self.attributes),
            tags=dict(self.tags),
            start_time_ns=self.start_time_ns,
            outputs=canonical_json_compatible(self.outputs) if self.outputs is not None else None,
            status=self.status,
            end_time_ns=self.end_time_ns,
        )


@dataclass(frozen=True)
class EventSpec:
    """One point-in-time child event mapped from one source record.

    Same field parity as :class:`SpanSpec` except that the span type is
    fixed: :meth:`to_child_request` always converts with
    ``span_type="event"`` (:data:`EVENT_SPAN_TYPE`), because the public
    trace API has no separate event mutation.
    """

    source_key: str
    name: str
    parent_source_key: str | None
    inputs: object | None
    attributes: Mapping[str, object]
    tags: Mapping[str, str]
    start_time_ns: int | None
    outputs: object | None
    status: str
    end_time_ns: int | None

    @property
    def span_type(self) -> str:
        """The fixed event span type."""
        return EVENT_SPAN_TYPE

    def to_child_request(self) -> TraceChildRequest:
        """Convert to the real gateway child request (lazy boundary)."""
        from opencode_mlflow.mlflow.gateway import TraceChildRequest

        return TraceChildRequest(
            source_key=self.source_key,
            name=self.name,
            span_type=EVENT_SPAN_TYPE,
            parent_source_key=self.parent_source_key,
            inputs=canonical_json_compatible(self.inputs) if self.inputs is not None else None,
            attributes=_canonical_mapping(self.attributes),
            tags=dict(self.tags),
            start_time_ns=self.start_time_ns,
            outputs=canonical_json_compatible(self.outputs) if self.outputs is not None else None,
            status=self.status,
            end_time_ns=self.end_time_ns,
        )


@dataclass(frozen=True)
class TraceSpec:
    """One complete turn-to-trace mapping result.

    Field parity with the gateway's ``TraceCreateRequest`` is exact
    (contract-tested), including the explicit ``root_status``: the spec
    is a duck-typed producer of the request and
    :meth:`to_create_request` converts it structurally. ``child_specs``
    yields :class:`SpanSpec`/:class:`EventSpec` objects (which satisfy
    the child-request shape); the conversion wraps each through its own
    ``to_child_request`` lazily, so the gateway still consumes children
    one at a time.
    """

    client_request_id: str
    experiment_id: str
    name: str
    native_session_id: str
    root_inputs: object
    root_outputs: object | None
    root_attributes: Mapping[str, object]
    root_tags: Mapping[str, str]
    metadata: Mapping[str, str]
    start_time_ns: int | None
    end_time_ns: int | None
    child_specs: Callable[[], Iterator[SpanSpec | EventSpec]]
    expected_operation_key: str
    artifact_snapshot_fingerprint: str
    artifact_relative_path: str
    root_status: str | None = None

    def to_create_request(self) -> TraceCreateRequest:
        """Convert to the real gateway trace-create request (lazy boundary)."""
        from opencode_mlflow.mlflow.gateway import TraceCreateRequest

        return TraceCreateRequest(
            client_request_id=self.client_request_id,
            experiment_id=self.experiment_id,
            name=self.name,
            native_session_id=self.native_session_id,
            root_inputs=canonical_json_compatible(self.root_inputs),
            root_outputs=canonical_json_compatible(self.root_outputs) if self.root_outputs is not None else None,
            root_attributes=_canonical_mapping(self.root_attributes),
            root_tags=dict(self.root_tags),
            metadata=dict(self.metadata),
            start_time_ns=self.start_time_ns,
            end_time_ns=self.end_time_ns,
            child_specs=lambda: (child.to_child_request() for child in self.child_specs()),
            expected_operation_key=self.expected_operation_key,
            artifact_snapshot_fingerprint=self.artifact_snapshot_fingerprint,
            artifact_relative_path=self.artifact_relative_path,
            root_status=self.root_status,
        )

    def identity_envelope(self) -> ArtifactIdentityEnvelope:
        """The turn's pre-identity-map artifact identity, from the spec."""
        return ArtifactIdentityEnvelope(
            session_id=self.native_session_id,
            turn_key=self.metadata["opencode.turn_key"],
            render_fingerprint=self.metadata["opencode.render_fingerprint"],
            snapshot_fingerprint=self.artifact_snapshot_fingerprint,
            artifact_relative_path=self.artifact_relative_path,
        )


# ---------------------------------------------------------------------------
# Per-record typed mapping
# ---------------------------------------------------------------------------


def map_source_record(record: SourceRecord, parent_source_key: str | None) -> SpanSpec | EventSpec:
    """Map one source record to its typed child span/event spec (pure).

    Classification is fixed and documented: content parts (``part`` rows
    and derived nested content items) map by their decoded ``type`` —
    tool parts to tool spans, text parts to content spans, and every
    other annotation part (reasoning, error, step markers, …) to events;
    message containers carrying model-call metadata map to LLM spans and
    other message containers to agent spans; lifecycle rows
    (``session_pending``/``session_inbox``) map to lifecycle events; and
    anything else maps to an opaque event carrying the normalized record
    form as an attribute. ``parent_source_key`` is the parent child's
    source key, or ``None`` for a child of the trace root. The returned
    spec carries COMPLETE source values; callers feeding the gateway
    must bound them against the inline limit (``map_turn_to_trace``
    does).
    """
    identity = SourceIdentity.of(record)
    attributes = _common_attributes(record)
    timestamp_ns = record.timestamp.nanoseconds if record.timestamp is not None else None
    completion_timestamp_ns = (
        record.completion_timestamp.nanoseconds if record.completion_timestamp is not None else timestamp_ns
    )
    if _is_content_part(record):
        return _map_content_part(record, identity, parent_source_key, attributes, timestamp_ns, completion_timestamp_ns)
    if _is_message_container(record):
        return _map_message_container(
            record, identity, parent_source_key, attributes, timestamp_ns, completion_timestamp_ns
        )
    if record.source_table in _SESSION_EVENT_TABLES:
        return _map_lifecycle_event(
            record, identity, parent_source_key, attributes, timestamp_ns, completion_timestamp_ns
        )
    return _map_opaque_record(record, identity, parent_source_key, attributes, timestamp_ns, completion_timestamp_ns)


def _is_content_part(record: SourceRecord) -> bool:
    return record.source_table == "part" or record.derived_child_ordinal is not None


def _is_message_container(record: SourceRecord) -> bool:
    return record.source_table in _MESSAGE_TABLES and record.derived_child_ordinal is None


def _map_content_part(
    record: SourceRecord,
    identity: SourceIdentity,
    parent_source_key: str | None,
    attributes: dict[str, object],
    timestamp_ns: int | None,
    completion_timestamp_ns: int | None,
) -> SpanSpec | EventSpec:
    decoded = record.decoded_value if isinstance(record.decoded_value, Mapping) else {}
    part_type = decoded.get("type")
    if isinstance(part_type, str) and part_type:
        attributes["opencode.part_type"] = part_type
    if record.derived_child_ordinal is not None:
        attributes["opencode.part_sequence"] = record.derived_child_ordinal
    name = f"{part_type} part" if isinstance(part_type, str) and part_type else "content part"
    source_outcome = decoded_source_outcome(decoded)
    if source_outcome is not None:
        attributes["opencode.source_outcome"] = source_outcome
    if part_type == "tool":
        return _map_tool_part(decoded, identity, parent_source_key, attributes, timestamp_ns, completion_timestamp_ns)
    if part_type == "error":
        attributes["opencode.error"] = decoded.get("error")
        source_outcome = source_outcome or "error"
    status = classify_outcome(source_outcome)
    if part_type == "error":
        status = "ERROR"
    if part_type == "text":
        return SpanSpec(
            source_key=identity.key,
            name=name,
            span_type=_CONTENT_SPAN_TYPE,
            parent_source_key=parent_source_key,
            inputs=record.decoded_value,
            attributes=MappingProxyType(attributes),
            tags={},
            start_time_ns=timestamp_ns,
            outputs=None,
            status=status,
            end_time_ns=completion_timestamp_ns,
        )
    return EventSpec(
        source_key=identity.key,
        name=name,
        parent_source_key=parent_source_key,
        inputs=record.decoded_value,
        attributes=MappingProxyType(attributes),
        tags={},
        start_time_ns=timestamp_ns,
        outputs=None,
        status=status,
        end_time_ns=completion_timestamp_ns,
    )


def _map_tool_part(
    decoded: Mapping[str, object],
    identity: SourceIdentity,
    parent_source_key: str | None,
    attributes: dict[str, object],
    timestamp_ns: int | None,
    completion_timestamp_ns: int | None,
) -> SpanSpec:
    tool_name = decoded.get("tool")
    if isinstance(tool_name, str) and tool_name:
        attributes["opencode.tool"] = tool_name
    call_id = _optional_str(decoded.get("callID")) or _optional_str(decoded.get("call_id"))
    if call_id is not None:
        attributes["opencode.tool_call_id"] = call_id
    state = decoded.get("state")
    state_map = state if isinstance(state, Mapping) else {}
    tool_status = _optional_str(state_map.get("status"))
    if tool_status is not None:
        attributes["opencode.tool_status"] = tool_status
    error = state_map.get("error")
    if error is not None:
        attributes["opencode.error"] = error
    source_outcome = tool_source_outcome(decoded)
    if source_outcome is None and error is not None:
        source_outcome = "error"
    if source_outcome is not None:
        attributes["opencode.source_outcome"] = source_outcome
    if identity.source_table == "part" and isinstance(tool_name, str) and tool_name:
        attributes["tool_name"] = tool_name
    if identity.source_table == "part" and call_id is not None:
        attributes["tool_id"] = call_id
    if identity.source_table == "part" and tool_status is not None:
        attributes["status"] = tool_status
    return SpanSpec(
        source_key=identity.key,
        name=(f"tool_{tool_name}" if isinstance(tool_name, str) and tool_name else "tool")
        if identity.source_table == "part"
        else (f"tool {tool_name}" if isinstance(tool_name, str) and tool_name else "tool call"),
        span_type=_TOOL_SPAN_TYPE,
        parent_source_key=parent_source_key,
        inputs=state_map.get("input"),
        attributes=MappingProxyType(attributes),
        tags={},
        start_time_ns=timestamp_ns,
        outputs=(
            {"result": state_map.get("output")}
            if identity.source_table == "part" and "output" in state_map
            else state_map.get("output")
        ),
        status=_tool_status(source_outcome, error),
        end_time_ns=completion_timestamp_ns,
    )


def _tool_status(source_outcome: str | None, error: object) -> str:
    if error is not None or classify_outcome(source_outcome) == "ERROR":
        return "ERROR"
    if classify_outcome(source_outcome) == "UNSET":
        return "UNSET"
    return "OK"


def _record_error_value(decoded: Mapping[str, object]) -> bool:
    """Recognize the shared explicit message-level error signals."""
    return is_explicit_failure(decoded)


def _lifecycle_status(outcome: str | None) -> str:
    return classify_outcome(outcome)


def _map_message_container(
    record: SourceRecord,
    identity: SourceIdentity,
    parent_source_key: str | None,
    attributes: dict[str, object],
    timestamp_ns: int | None,
    completion_timestamp_ns: int | None,
) -> SpanSpec:
    decoded = record.decoded_value if isinstance(record.decoded_value, Mapping) else {}
    source_outcome = decoded_source_outcome(decoded)
    if source_outcome is not None:
        attributes["opencode.source_outcome"] = source_outcome
    model_identity = extract_model_identity(decoded)
    if model_identity is None:
        return SpanSpec(
            source_key=identity.key,
            name=f"{record.source_type} message",
            span_type=_AGENT_SPAN_TYPE,
            parent_source_key=parent_source_key,
            inputs=record.decoded_value,
            attributes=MappingProxyType(attributes),
            tags={},
            start_time_ns=timestamp_ns,
            outputs=None,
            status=classify_decoded(decoded),
            end_time_ns=completion_timestamp_ns,
        )
    attributes["opencode.model"] = model_identity
    provider = model_identity.partition("/")[0]
    if provider:
        attributes["opencode.provider"] = provider
    _apply_token_attributes(attributes, decoded.get("tokens"))
    cost = decoded.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        attributes["opencode.cost"] = cost
    finish = _optional_str(decoded.get("finish"))
    if finish is not None:
        attributes["opencode.finish"] = finish
    # A persisted model-call envelope carries its typed request/response
    # components; a message whose data rides alongside its content parts
    # keeps the complete decoded value as the request payload.
    if record.source_table != "message":
        inputs = decoded["input"] if "input" in decoded else record.decoded_value
        outputs = decoded["output"] if "output" in decoded else None
        return SpanSpec(
            source_key=identity.key,
            name=f"model {model_identity}",
            span_type=_LLM_SPAN_TYPE,
            parent_source_key=parent_source_key,
            inputs=inputs,
            attributes=MappingProxyType(attributes),
            tags={},
            start_time_ns=timestamp_ns,
            outputs=outputs,
            status=classify_decoded(decoded),
            end_time_ns=completion_timestamp_ns,
        )
    inputs = {"model": model_identity, "messages": []}
    outputs = {"choices": [{"message": {"role": "assistant", "content": None}}]}
    provider, _, model = model_identity.partition("/")
    attributes["model"] = model or model_identity
    attributes["provider"] = provider
    return SpanSpec(
        source_key=identity.key,
        name="llm_call",
        span_type=_LLM_SPAN_TYPE,
        parent_source_key=parent_source_key,
        inputs=inputs,
        attributes=MappingProxyType(attributes),
        tags={},
        start_time_ns=timestamp_ns,
        outputs=outputs,
        status=classify_decoded(decoded),
        end_time_ns=completion_timestamp_ns,
    )


def _map_lifecycle_event(
    record: SourceRecord,
    identity: SourceIdentity,
    parent_source_key: str | None,
    attributes: dict[str, object],
    timestamp_ns: int | None,
    completion_timestamp_ns: int | None,
) -> EventSpec:
    decoded = record.decoded_value if isinstance(record.decoded_value, Mapping) else {}
    outcome = record_source_outcome(decoded, record.source_type, record.raw_columns)
    attributes["opencode.record"] = canonical_record_value(record)
    if outcome is not None:
        attributes["opencode.lifecycle_outcome"] = outcome
        attributes["opencode.source_outcome"] = outcome
    return EventSpec(
        source_key=identity.key,
        name=f"lifecycle {record.source_type}",
        parent_source_key=parent_source_key,
        inputs=record.decoded_value,
        attributes=MappingProxyType(attributes),
        tags={},
        start_time_ns=timestamp_ns,
        outputs=None,
        status=_lifecycle_status(outcome),
        end_time_ns=completion_timestamp_ns,
    )


def _map_opaque_record(
    record: SourceRecord,
    identity: SourceIdentity,
    parent_source_key: str | None,
    attributes: dict[str, object],
    timestamp_ns: int | None,
    completion_timestamp_ns: int | None,
) -> EventSpec:
    attributes["opencode.record"] = canonical_record_value(record)
    source_outcome = record_source_outcome(record.decoded_value, record.source_type, record.raw_columns)
    if source_outcome is not None:
        attributes["opencode.source_outcome"] = source_outcome
    return EventSpec(
        source_key=identity.key,
        name=f"opaque {record.source_table}",
        parent_source_key=parent_source_key,
        inputs=None,
        attributes=MappingProxyType(attributes),
        tags={},
        start_time_ns=timestamp_ns,
        outputs=None,
        status=classify_outcome(source_outcome),
        end_time_ns=completion_timestamp_ns,
    )


def _common_attributes(record: SourceRecord) -> dict[str, object]:
    """The namespaced source-provenance attributes every child carries."""
    attributes: dict[str, object] = {
        "opencode.source_table": record.source_table,
        "opencode.source_type": record.source_type,
        "opencode.source_key": SourceIdentity.of(record).key,
    }
    if record.source_id is not None:
        attributes["opencode.source_id"] = record.source_id
    if record.sequence_domain is not None:
        attributes["opencode.sequence_domain"] = record.sequence_domain
    if record.sequence_value is not None:
        attributes["opencode.sequence"] = record.sequence_value
    if record.timestamp is not None:
        attributes["opencode.timestamp_ns"] = record.timestamp.nanoseconds
    if record.completion_timestamp is not None:
        attributes["opencode.completion_timestamp_ns"] = record.completion_timestamp.nanoseconds
    if record.parent_source_id is not None:
        attributes["opencode.parent_source_id"] = record.parent_source_id
    if record.parent_source_table is not None:
        attributes["opencode.parent_source_table"] = record.parent_source_table
    if record.classification is not None:
        attributes["opencode.classification"] = record.classification
    return attributes


def _apply_token_attributes(attributes: dict[str, object], tokens: object) -> None:
    """Record token usage aggregates as namespaced attributes when present."""
    if not isinstance(tokens, Mapping):
        return
    _apply_int_attribute(attributes, "opencode.tokens.input", tokens.get("input"))
    _apply_int_attribute(attributes, "opencode.tokens.output", tokens.get("output"))
    _apply_int_attribute(attributes, "opencode.tokens.reasoning", tokens.get("reasoning"))
    cache = tokens.get("cache")
    if isinstance(cache, Mapping):
        _apply_int_attribute(attributes, "opencode.tokens.cache_read", cache.get("read"))
        _apply_int_attribute(attributes, "opencode.tokens.cache_write", cache.get("write"))
    else:
        _apply_int_attribute(attributes, "opencode.tokens.cache_read", tokens.get("cache_read"))
        _apply_int_attribute(attributes, "opencode.tokens.cache_write", tokens.get("cache_write"))


def _apply_int_attribute(attributes: dict[str, object], key: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        return
    attributes[key] = value


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _canonical_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Convert a mapping through canonical-json-v1 while retaining its type."""
    converted = canonical_json_compatible(dict(value))
    if not isinstance(converted, Mapping):  # pragma: no cover - canonical mappings decode as objects
        raise TypeError("a canonical mapping must decode to a JSON object")
    return cast(Mapping[str, object], converted)


# ---------------------------------------------------------------------------
# Turn mapping
# ---------------------------------------------------------------------------


@dataclass
class _TurnScan:
    """The bounded side-products of one streaming pass over a turn's records."""

    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    last_session_timestamp_ns: int | None = None
    root_result_error: bool = False
    source_outcome: str | None = None
    lifecycle_outcome: str | None = None
    lifecycle_error_outcome: str | None = None


def map_turn_to_trace(turn: ReplayTurn, context: MappingContext) -> TraceSpec:
    """Map one logical turn to its complete trace spec (pure, never MLflow).

    Loads the bounded root values FIRST (a COMPLETE root value above the
    negotiated inline limit raises a safe ``payload_too_large`` error
    before any trace could be created), computes the Task 4A
    fingerprints (source, render, and the pre-identity-map artifact
    snapshot) and the deterministic artifact path, derives the
    source-only timing and terminal outcome, assembles the complete
    ownership/operation metadata and typed root attributes, and returns
    the spec with a LAZY child cursor over the turn's records.
    """
    header = turn.session_header
    summary = turn.summary
    inline_limit = resolve_inline_limit(context.inline_limit_bytes)

    root_values = load_root_values(turn, inline_limit)
    effective = build_effective_turn(turn, IDENTITY_REDACTOR, context.redaction_mode)
    source_fp = source_fingerprint(effective, context.redaction_mode)
    render_fp = render_fingerprint(
        source_fp,
        context.artifact_schema_version,
        context.mapping_version,
        context.importer_version,
        context.redaction_mode,
    )
    snapshot_fp = artifact_snapshot_fingerprint(turn, context.redaction_mode)
    relative_path = artifact_relative_path(header.session_id, summary.turn_key, render_fp, snapshot_fp)

    scan = _scan_turn(turn)
    start_ns, start_estimated = _start_time_ns(header, summary, scan)
    summary_end_ns = summary.end_time.nanoseconds if summary.end_time is not None else None
    end_candidates = tuple(
        value for value in (summary_end_ns, scan.last_timestamp_ns, scan.last_session_timestamp_ns) if value is not None
    )
    end_ns = max(end_candidates) if end_candidates else start_ns
    end_estimated = end_ns is not None and (summary_end_ns is None or end_ns != summary_end_ns)
    timing_estimated = start_estimated or end_estimated
    outcome, root_status = _terminal_outcome(summary, scan)
    source_outcome = scan.source_outcome or summary.source_outcome or outcome
    operation = operation_key(context.experiment_id, header.session_id, summary.turn_key, render_fp)

    metadata = _ownership_metadata(
        header, summary, context, source_fp, render_fp, operation, source_outcome, timing_estimated
    )
    root_attributes = _root_attributes(header, summary, context, source_outcome, timing_estimated, inline_limit)

    return TraceSpec(
        client_request_id=operation,
        experiment_id=context.experiment_id,
        name=(
            "opencode_conversation"
            if context.source_schema == "legacy"
            else f"{_TRACE_NAME_PREFIX} {summary.turn_index}"
        ),
        native_session_id=header.session_id,
        root_inputs=(
            _native_root_input(root_values.root_input)
            if context.source_schema == "legacy"
            else root_values.root_input
        ),
        root_outputs=(
            _native_root_output(root_values.root_output)
            if context.source_schema == "legacy"
            else root_values.root_output
        ),
        root_attributes=MappingProxyType(root_attributes),
        # The artifact's exact relative path rides a trace TAG at root
        # creation (crash-survivable): candidate verification reads this
        # RECORDED path and checks integrity at the candidate's own
        # location, since the deterministic path embeds the
        # session-complete artifact-snapshot fingerprint and moves for
        # every sibling turn whenever any turn of the session changes.
        # A tag, not trace metadata: the pinned server silently truncates
        # metadata values at 250 characters while the path is typically
        # longer; tag values round-trip exactly up to 4096.
        root_tags={ARTIFACT_PATH_TAG: relative_path},
        metadata=MappingProxyType(metadata),
        start_time_ns=start_ns,
        end_time_ns=end_ns,
        child_specs=lambda: _iter_child_specs(turn, inline_limit),
        expected_operation_key=operation,
        artifact_snapshot_fingerprint=snapshot_fp,
        artifact_relative_path=relative_path,
        root_status=root_status,
    )


def _scan_turn(turn: ReplayTurn) -> _TurnScan:
    """One streaming pass for the timing fallback and the failure signal.

    Collects source-derived start/end instants from turn records and timed
    session events, plus whether the turn's root result carries an explicit
    error part (the explicit-failure signal); records are consumed one at a
    time and released.
    """
    root_identity = turn.summary.root_result_source_id
    scan = _TurnScan()

    def observe_timestamp(record: SourceRecord, *, session_event: bool = False) -> None:
        """Collect source instants without consulting a local clock."""
        first = record.timestamp or record.completion_timestamp
        if first is not None:
            first_ns = first.nanoseconds
            scan.first_timestamp_ns = (
                first_ns if scan.first_timestamp_ns is None else min(scan.first_timestamp_ns, first_ns)
            )
        last = record.completion_timestamp or record.timestamp
        if last is not None:
            last_ns = last.nanoseconds
            scan.last_timestamp_ns = last_ns if scan.last_timestamp_ns is None else max(scan.last_timestamp_ns, last_ns)
            if session_event:
                scan.last_session_timestamp_ns = (
                    last_ns if scan.last_session_timestamp_ns is None else max(scan.last_session_timestamp_ns, last_ns)
                )

    for record in turn.iter_records():
        observe_timestamp(record)
        if root_identity is not None and record.parent_source_id == root_identity and _is_error_part(record):
            scan.root_result_error = True
            scan.source_outcome = declared_source_outcome(record.decoded_value)
        if root_identity is not None and _is_root_result(record) and _record_identity(record) == root_identity:
            decoded = record.decoded_value
            if isinstance(decoded, Mapping) and _record_error_value(decoded):
                scan.root_result_error = True
                scan.source_outcome = declared_source_outcome(decoded)
            elif isinstance(decoded, Mapping):
                scan.source_outcome = decoded_source_outcome(decoded) or ASSISTANT_RESULT_OUTCOME
        elif root_identity is None and (
            is_explicit_failure(record.decoded_value)
            or classify_outcome(record_source_outcome(record.decoded_value, record.source_type, record.raw_columns))
            == "ERROR"
            or classify_outcome(record.source_type) == "ERROR"
        ):
            scan.root_result_error = True
            scan.source_outcome = (
                record_source_outcome(record.decoded_value, record.source_type, record.raw_columns)
                or record.source_type
            )
    for record in turn.iter_session_records():
        if record.source_table not in _SESSION_EVENT_TABLES:
            continue
        observe_timestamp(record, session_event=True)
        outcome = record_source_outcome(record.decoded_value, record.source_type, record.raw_columns)
        if outcome is not None:
            lifecycle = f"lifecycle_{outcome.lower()}"
            scan.lifecycle_outcome = lifecycle
            if _lifecycle_status(outcome) == "ERROR":
                scan.lifecycle_error_outcome = lifecycle
                scan.source_outcome = lifecycle
            elif scan.lifecycle_error_outcome is None and root_identity is None:
                scan.source_outcome = lifecycle
    return scan


def _is_error_part(record: SourceRecord) -> bool:
    decoded = record.decoded_value
    return isinstance(decoded, Mapping) and decoded.get("type") == "error"


def _record_identity(record: SourceRecord) -> str | None:
    return record.source_id if record.source_id is not None else record.source_identity


def _start_time_ns(header: SessionHeader, summary: TurnSummary, scan: _TurnScan) -> tuple[int | None, bool]:
    """First user timestamp → first timestamped source record → session creation."""
    if summary.start_time is not None:
        return summary.start_time.nanoseconds, False
    if scan.first_timestamp_ns is not None:
        return scan.first_timestamp_ns, True
    if header.time_created is not None:
        return header.time_created.nanoseconds, True
    return None, False


def _terminal_outcome(summary: TurnSummary, scan: _TurnScan) -> tuple[str, TraceStatus]:
    """The authoritative source terminal outcome and its root span status.

    A turn that ended with a non-error assistant result succeeded
    (``OK``); a final result carrying an explicit error part failed
    explicitly (``ERROR``); a turn without an assistant result is
    interrupted/incomplete (``UNSET``), with ``opencode.outcome``
    recording the source terminal outcome verbatim.
    """
    if scan.lifecycle_error_outcome is not None:
        return scan.lifecycle_error_outcome, "ERROR"
    if summary.terminal_outcome == ASSISTANT_RESULT_OUTCOME:
        if scan.root_result_error:
            return OUTCOME_EXPLICIT_FAILURE, "ERROR"
        return ASSISTANT_RESULT_OUTCOME, "OK"
    if summary.terminal_outcome in {ASSISTANT_ERROR_OUTCOME, PART_ERROR_OUTCOME, OUTCOME_EXPLICIT_FAILURE}:
        return summary.terminal_outcome, "ERROR"
    if summary.terminal_outcome is not None and classify_outcome(summary.terminal_outcome) == "UNSET":
        return summary.terminal_outcome, "UNSET"
    if summary.terminal_outcome is not None and summary.terminal_outcome.startswith("lifecycle_"):
        outcome = summary.terminal_outcome.removeprefix("lifecycle_").lower()
        status: TraceStatus = "ERROR" if classify_outcome(outcome) == "ERROR" else "UNSET"
        return summary.terminal_outcome, status
    if summary.root_result_source_id is None and scan.lifecycle_outcome is not None:
        outcome = scan.lifecycle_outcome.removeprefix("lifecycle_").lower()
        status = "ERROR" if classify_outcome(outcome) == "ERROR" else "UNSET"
        return scan.lifecycle_outcome, status
    return NO_ASSISTANT_RESULT_OUTCOME, "UNSET"


def _iter_child_specs(turn: ReplayTurn, inline_limit: int) -> Iterator[SpanSpec | EventSpec]:
    """Lazily map the turn's records to bounded children, fresh per call.

    The turn's first record is its first-user prompt — the root input —
    and is therefore not a child. Every other record maps through
    :func:`map_source_record` keyed by its stable source identity and
    parented under its source parent when that parent is among this
    trace's earlier children (parts are always emitted right after
    their parents) and under the trace root otherwise — which is exactly
    what happens to the root-input prompt's own parts. Each spec's
    payloads are bounded against the negotiated inline limit before it
    is yielded.
    """
    emitted: set[str] = set()
    first_record = next(iter(turn.iter_records()), None)
    if first_record is not None and first_record.source_table != "message":
        first = True
        for record in turn.iter_records():
            if first:
                first = False
                continue
            parent_key = _qualified_parent_source_key(record)
            parent = parent_key if parent_key in emitted else None
            spec = map_source_record(record, parent)
            emitted.add(spec.source_key)
            yield _bound_child(spec, inline_limit)
        for record in turn.iter_session_records():
            if record.source_table not in _SESSION_EVENT_TABLES:
                continue
            if record.timestamp is None and record.completion_timestamp is None:
                continue
            spec = map_source_record(record, None)
            if spec.source_key in emitted:
                continue
            emitted.add(spec.source_key)
            yield _bound_child(spec, inline_limit)
        return

    records = tuple(turn.iter_records())
    if not records:
        return

    prompt = _native_root_input(turn.load_root_input(inline_limit))
    history: list[dict[str, object]] = []
    if isinstance(prompt, Mapping):
        history.append({"role": "user", "content": prompt.get("prompt", "")})

    first = True
    index = 0
    while index < len(records):
        record = records[index]
        if first:
            first = False
            index += 1
            continue

        decoded = record.decoded_value if isinstance(record.decoded_value, Mapping) else {}
        if _is_message_container(record) and extract_model_identity(decoded) is not None:
            children = []
            next_index = index + 1
            while next_index < len(records):
                child = records[next_index]
                if child.parent_source_id != record.source_id:
                    break
                children.append(child)
                next_index += 1
            spec = _native_model_spec(record, children, history)
            emitted.add(spec.source_key)
            yield _bound_child(spec, inline_limit)
            for child in children:
                if child.source_table == "part" and isinstance(child.decoded_value, Mapping):
                    child_value = child.decoded_value
                    if child_value.get("type") == "tool":
                        tool_spec = map_source_record(child, None)
                        emitted.add(tool_spec.source_key)
                        yield _bound_child(tool_spec, inline_limit)
                        state = child_value.get("state")
                        if isinstance(state, Mapping):
                            call_id = child_value.get("callID") or child_value.get("call_id")
                            history.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call_id,
                                    "content": state.get("output"),
                                }
                            )
            index = next_index
            continue

        # Native OpenCode traces expose tool calls and model calls only. The
        # complete text, reasoning, and lifecycle records remain in artifacts.
        if record.source_table == "part" and isinstance(decoded, Mapping) and decoded.get("type") == "tool":
            spec = map_source_record(record, None)
            emitted.add(spec.source_key)
            yield _bound_child(spec, inline_limit)
        index += 1
    # Session-scoped lifecycle rows are not turn boundaries and are retained
    # in the artifact's session-record section. Timestamped lifecycle rows
    # still belong in the trace mapping, exactly once per mapped trace, as
    # root children; untimed instruction/todo state remains artifact-only so
    # the gateway never invents child timing.
    for record in turn.iter_session_records():
        if record.source_table not in _SESSION_EVENT_TABLES:
            continue
        if record.timestamp is None and record.completion_timestamp is None:
            continue
        spec = map_source_record(record, None)
        if spec.source_key in emitted:
            continue
        emitted.add(spec.source_key)
        yield _bound_child(spec, inline_limit)


def _native_model_spec(record: SourceRecord, children: list[SourceRecord], history: list[dict[str, object]]) -> SpanSpec:
    """Project one persisted model message to MLflow's native LLM shape."""
    identity = SourceIdentity.of(record)
    decoded = record.decoded_value if isinstance(record.decoded_value, Mapping) else {}
    attributes = _common_attributes(record)
    model_identity = extract_model_identity(decoded) or ""
    provider, _, model = model_identity.partition("/")
    attributes.update({"opencode.model": model_identity, "model": model or model_identity, "provider": provider})
    source_outcome = decoded_source_outcome(decoded)
    if source_outcome is not None:
        attributes["opencode.source_outcome"] = source_outcome
    _apply_token_attributes(attributes, decoded.get("tokens"))
    if isinstance(decoded.get("cost"), (int, float)) and not isinstance(decoded.get("cost"), bool):
        attributes["opencode.cost"] = decoded["cost"]
    finish = _optional_str(decoded.get("finish"))
    if finish is not None:
        attributes["opencode.finish"] = finish

    message: dict[str, object] = {"role": "assistant", "content": None}
    text_parts = [
        value.get("text")
        for child in children
        if isinstance(child.decoded_value, Mapping)
        for value in [child.decoded_value]
        if value.get("type") == "text" and isinstance(value.get("text"), str)
    ]
    reasoning_parts = [
        value.get("text")
        for child in children
        if isinstance(child.decoded_value, Mapping)
        for value in [child.decoded_value]
        if value.get("type") == "reasoning" and isinstance(value.get("text"), str) and value.get("text")
    ]
    if text_parts:
        message["content"] = "".join(text_parts)
    if reasoning_parts:
        message["reasoning"] = "\n\n".join(reasoning_parts)
    tool_calls = []
    for child in children:
        value = child.decoded_value
        if not isinstance(value, Mapping) or value.get("type") != "tool":
            continue
        tool_name = value.get("tool")
        call_id = value.get("callID") or value.get("call_id")
        state = value.get("state")
        tool_input = state.get("input") if isinstance(state, Mapping) else None
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": canonical_json(tool_input).decode("utf-8")},
            }
        )
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = tool_calls
    outputs = {"choices": [{"message": message}]}
    return SpanSpec(
        source_key=identity.key,
        name="llm_call",
        span_type=_LLM_SPAN_TYPE,
        parent_source_key=None,
        inputs={"model": model_identity, "messages": list(history)},
        attributes=MappingProxyType(attributes),
        tags={},
        start_time_ns=record.timestamp.nanoseconds if record.timestamp is not None else None,
        outputs=outputs,
        status=classify_decoded(decoded),
        end_time_ns=(
            record.completion_timestamp.nanoseconds
            if record.completion_timestamp is not None
            else record.timestamp.nanoseconds
            if record.timestamp is not None
            else None
        ),
    )


def _native_root_input(value: object | None) -> object | None:
    """Convert a source prompt payload to the native conversation shape."""
    text = _text_from_parts(value)
    return {"prompt": text} if text is not None else value


def _native_root_output(value: object | None) -> object | None:
    """Convert a source response payload to the native conversation shape."""
    text = _text_from_parts(value)
    return {"response": text, "status": "completed"} if text is not None else value


def _text_from_parts(value: object | None) -> str | None:
    if isinstance(value, Mapping):
        if isinstance(value.get("text"), str):
            return value["text"]
        content = value.get("content")
        return _text_from_parts(content)
    if isinstance(value, list):
        texts = [item.get("text") for item in value if isinstance(item, Mapping) and isinstance(item.get("text"), str)]
        return "".join(texts) if texts else None
    return value if isinstance(value, str) else None


def _qualified_parent_source_key(record: SourceRecord) -> str | None:
    """Return a parent key in the same namespace as :class:`SourceIdentity`."""
    if record.parent_source_id is None:
        return None
    parent_table = record.parent_source_table
    if parent_table is None and record.derived_child_ordinal is not None:
        parent_table = "message" if record.source_table == "part" else record.source_table
    if parent_table is None:
        # Preserve compatibility for hand-built opaque records from the
        # public pure-mapping seam, which historically passed an already
        # qualified/opaque parent string.
        return record.parent_source_id
    return f"{parent_table}:{record.parent_source_id}"


def _bound_child(spec: SpanSpec | EventSpec, inline_limit: int) -> SpanSpec | EventSpec:
    """Bound one child's payloads against the inline limit (never truncate)."""
    bounded_attributes = {key: _inline_value(value, inline_limit) for key, value in spec.attributes.items()}
    return replace(
        spec,
        inputs=_inline_value(spec.inputs, inline_limit),
        outputs=_inline_value(spec.outputs, inline_limit),
        attributes=MappingProxyType(bounded_attributes),
    )


def _inline_value(value: object | None, inline_limit: int) -> object | None:
    """One value inline-or-reference against the negotiated limit."""
    if value is None:
        return None
    return inline_value_or_reference(value, inline_limit).value


def _ownership_metadata(
    header: SessionHeader,
    summary: TurnSummary,
    context: MappingContext,
    source_fp: str,
    render_fp: str,
    operation: str,
    outcome: str,
    timing_estimated: bool,
) -> dict[str, str]:
    """The complete namespaced ownership/operation/session metadata (strings)."""
    metadata: dict[str, str] = {
        "opencode.importer": IMPORTER_IDENTITY,
        "opencode.session_id": header.session_id,
        "opencode.turn_index": str(summary.turn_index),
        "opencode.turn_key": summary.turn_key,
        "opencode.source_fingerprint": source_fp,
        "opencode.source_schema": context.source_schema,
        "opencode.importer_version": context.importer_version,
        "opencode.mapping_version": context.mapping_version,
        "opencode.artifact_schema_version": context.artifact_schema_version,
        "opencode.redaction_mode": context.redaction_mode.value,
        "opencode.render_fingerprint": render_fp,
        "opencode.operation_key": operation,
        "opencode.outcome": outcome,
    }
    if timing_estimated:
        metadata["opencode.timing_estimated"] = "true"
    if summary.warnings:
        metadata["opencode.warnings"] = ",".join(summary.warnings)

    _add_str(metadata, "opencode.title", header.title)
    catalog_version = context.catalog_version or header.catalog_version or None
    if catalog_version is not None:
        # Catalog version is creation-time ownership metadata, not a
        # descriptive field. It must remain exact so a long value cannot be
        # replaced with an artifact reference and later authorize cleanup.
        metadata["opencode.catalog_version"] = catalog_version
    _add_str(metadata, "opencode.slug", header.slug)
    _add_str(metadata, "opencode.directory", header.directory)
    _add_str(metadata, "opencode.path", header.path)
    _add_str(metadata, "opencode.project_id", header.project_id)
    _add_str(metadata, "opencode.workspace_id", header.workspace_id)
    _add_str(metadata, "opencode.parent_id", header.parent_id)
    _add_str(metadata, "opencode.fork_session_id", header.fork_session_id)
    _add_str(metadata, "opencode.agent", header.agent)
    _add_str(metadata, "opencode.version", header.version)
    model_identity = extract_model_identity(header.model)
    if model_identity is not None:
        _add_str(metadata, "opencode.model", model_identity)

    _add_instant(metadata, "opencode.time_created_ns", header.time_created)
    _add_instant(metadata, "opencode.time_updated_ns", header.time_updated)
    _add_instant(metadata, "opencode.time_compacting_ns", header.time_compacting)
    _add_instant(metadata, "opencode.time_archived_ns", header.time_archived)
    _add_instant(metadata, "opencode.time_idle_ns", header.time_idle)
    _add_instant(metadata, "opencode.time_viewed_ns", header.time_viewed)
    _add_instant(metadata, "opencode.time_suspended_ns", header.time_suspended)

    _add_number(metadata, "opencode.cost", header.cost)
    _add_number(metadata, "opencode.tokens_input", header.tokens_input)
    _add_number(metadata, "opencode.tokens_output", header.tokens_output)
    _add_number(metadata, "opencode.tokens_reasoning", header.tokens_reasoning)
    _add_number(metadata, "opencode.tokens_cache_read", header.tokens_cache_read)
    _add_number(metadata, "opencode.tokens_cache_write", header.tokens_cache_write)

    _add_str(metadata, "opencode.first_user_source_id", summary.first_user_source_id)
    _add_str(metadata, "opencode.root_result_source_id", summary.root_result_source_id)
    _add_number(metadata, "opencode.turn_record_count", summary.record_count)
    _add_number(metadata, "opencode.turn_user_record_count", summary.user_record_count)
    _add_number(metadata, "opencode.turn_assistant_record_count", summary.assistant_record_count)
    _add_number(metadata, "opencode.logical_turn_count", context.logical_turn_count)
    _add_number(metadata, "opencode.source_message_count", context.source_message_count)
    _add_number(metadata, "opencode.source_part_count", context.source_part_count)
    _add_str(
        metadata,
        "opencode.source_database_fingerprint",
        context.source_database_fingerprint
        if context.source_database_fingerprint is not None
        else header.source_fingerprint,
    )
    return metadata


def _root_attributes(
    header: SessionHeader,
    summary: TurnSummary,
    context: MappingContext,
    outcome: str,
    timing_estimated: bool,
    inline_limit: int,
) -> dict[str, object]:
    """The typed root-span attributes: turn identity and source aggregates.

    Trace-level totals come from source aggregates, never from
    truncated input/output previews; every value rides the span inline
    or as a digest reference, so nothing is silently truncated.
    """
    attributes: dict[str, object] = {
        "opencode.session_id": header.session_id,
        "opencode.turn_index": summary.turn_index,
        "opencode.turn_key": summary.turn_key,
        "opencode.outcome": outcome,
    }
    if context.catalog_version:
        attributes["opencode.catalog_version"] = context.catalog_version
    if timing_estimated:
        attributes["opencode.timing_estimated"] = True
    if header.cost is not None:
        attributes["opencode.cost"] = header.cost
    _apply_int_attribute(attributes, "opencode.tokens.input", header.tokens_input)
    _apply_int_attribute(attributes, "opencode.tokens.output", header.tokens_output)
    _apply_int_attribute(attributes, "opencode.tokens.reasoning", header.tokens_reasoning)
    _apply_int_attribute(attributes, "opencode.tokens.cache_read", header.tokens_cache_read)
    _apply_int_attribute(attributes, "opencode.tokens.cache_write", header.tokens_cache_write)
    _apply_int_attribute(attributes, "opencode.record_count", summary.record_count)
    _apply_int_attribute(attributes, "opencode.user_record_count", summary.user_record_count)
    _apply_int_attribute(attributes, "opencode.assistant_record_count", summary.assistant_record_count)
    _apply_int_attribute(attributes, "opencode.logical_turn_count", context.logical_turn_count)
    _apply_int_attribute(attributes, "opencode.source_message_count", context.source_message_count)
    _apply_int_attribute(attributes, "opencode.source_part_count", context.source_part_count)
    return {key: _inline_value(value, inline_limit) for key, value in attributes.items()}


def _add_str(metadata: dict[str, str], key: str, value: str | None) -> None:
    if value is None:
        return
    metadata[key] = _bounded_metadata_value(value)


def _add_number(metadata: dict[str, str], key: str, value: int | float | None) -> None:
    if value is None:
        return
    metadata[key] = str(value)


def _add_instant(metadata: dict[str, str], key: str, value: object) -> None:
    if value is None:
        return
    nanoseconds = getattr(value, "nanoseconds", None)
    if isinstance(nanoseconds, int):
        metadata[key] = str(nanoseconds)


def _bounded_metadata_value(value: str) -> str:
    """Keep short metadata verbatim; replace long values with the reference marker.

    The pinned server silently truncates trace-metadata values above
    :data:`METADATA_VALUE_CHAR_LIMIT` characters; a value above the
    limit is therefore replaced by the canonical JSON text of the Task 8
    digest/size reference marker, while the complete value stays in the
    turn's artifact.
    """
    if len(value) <= METADATA_VALUE_CHAR_LIMIT:
        return value
    reference = inline_value_or_reference(value, METADATA_VALUE_CHAR_LIMIT)
    return canonical_json(reference.value).decode("utf-8")
