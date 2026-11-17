"""Pure, bounded admission for the pinned trace transport.

The bound covers both OTLP protobuf and JSON full reads, not just canonical
source JSON. No SDK, target, artifact, or journal operations occur here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from typing import TYPE_CHECKING, Any, NoReturn, cast

from mlflow_import.errors import ItemImportError

if TYPE_CHECKING:
    from mlflow_import.mlflow.gateway import TraceCreateRequest

MAX_TRACE_SERIALIZED_SPAN_BYTES = 64 * 1024 * 1024
# Short spelling shared with incremental source projectors.
MAX_TRACE_BYTES = MAX_TRACE_SERIALIZED_SPAN_BYTES
MAX_TRACE_SPANS = 4096
MAX_TRACE_CONTROL_BYTES = 2 * 1024 * 1024
LATE_CONTROL_RESERVATION_BYTES = 1024 * 1024
TRACE_ENVELOPE_MESSAGE = "the trace is outside the finite serialized resource envelope"
ABSOLUTE_MAX_TRACE_BYTES = 2 * 1024 * 1024 * 1024
ABSOLUTE_MAX_TRACE_SPANS = 32768
DIAGNOSTIC_MAX_TRACE_BYTES = ABSOLUTE_MAX_TRACE_BYTES
DIAGNOSTIC_MAX_TRACE_SPANS = ABSOLUTE_MAX_TRACE_SPANS
DIAGNOSTIC_MAX_TRACE_CONTROL_BYTES = 16 * 1024 * 1024
TRACE_RESOURCE_PROFILE = "mlflow-import.trace-envelope.v1"
TRACE_RESOURCE_PROFILE_TAG = "mlflow_import.trace_resource_profile"
TRACE_RESOURCE_BYTES_TAG = "mlflow_import.trace_resource_bound_bytes"
TRACE_RESOURCE_SPANS_TAG = "mlflow_import.trace_resource_span_count"
PROOF_TAG_KEYS = frozenset({TRACE_RESOURCE_PROFILE_TAG, TRACE_RESOURCE_BYTES_TAG, TRACE_RESOURCE_SPANS_TAG})


@dataclass(frozen=True)
class TraceEnvelopeLimits:
    max_bytes: int = MAX_TRACE_SERIALIZED_SPAN_BYTES
    max_spans: int = MAX_TRACE_SPANS
    max_control_bytes: int = MAX_TRACE_CONTROL_BYTES


DEFAULT_TRACE_ENVELOPE_LIMITS = TraceEnvelopeLimits()


def _fail(
    code: str = "trace_resource_envelope_exceeded",
    *,
    reason: str | None = None,
    diagnostics: Mapping[str, str | int] | None = None,
) -> NoReturn:
    message = TRACE_ENVELOPE_MESSAGE
    if reason is not None and code == "trace_resource_envelope_exceeded":
        message = f"{message} ({reason})"
    error = ItemImportError(code, message)
    if diagnostics:
        error.diagnostics = dict(diagnostics)
    raise error


def _attach_diagnostics(error: ItemImportError, values: Mapping[str, str | int]) -> None:
    diagnostics = dict(error.diagnostics)
    for key, value in values.items():
        diagnostics.setdefault(key, value)
    error.diagnostics = diagnostics


class _ValueCounter:
    """Count an ASCII JSON value quoted twice more, snapshotting bounded DTOs.

    Each extra JSON string layer adds one byte per quote/backslash plus its
    two delimiters. Counting those characters explicitly covers historical
    JSON-looking strings' extra encoding and full-read JSON escaping. Unicode
    uses ASCII escapes (including surrogate pairs), an upper bound on UTF-8.
    Containers include the default encoder's spaces. Depth and every visited
    node are bounded before allocating their snapshots.
    """

    def __init__(
        self,
        limit: int,
        *,
        layers: int = 2,
        snapshot: bool = True,
        int64: bool = True,
        reason: str | None = None,
    ) -> None:
        self.limit = limit
        self.size = 0
        self.layers = layers
        self.snapshot = snapshot
        self.int64 = int64
        self.reason = reason

    def charge(self, size: int) -> None:
        self.size += size
        if self.size > self.limit:
            _fail(
                reason=self.reason,
                diagnostics={"value_bytes": self.size, "value_byte_limit": self.limit},
            )

    def token(self, text: str) -> None:
        n, q, b = len(text), text.count('"'), text.count("\\")
        # Two further string-encoding layers, derived from escaping rules.
        for _ in range(self.layers):
            n, q, b = n + q + b + 2, q + 2, 2 * b + q
        self.charge(n)

    def value(self, value: object, depth: int = 0) -> Any:
        if depth > 64:
            _fail("trace_resource_value_invalid")
        # AnyValue/array framing uses a field tag and at most a ten-byte
        # length at each of its enclosing protobuf levels. Charging 64 per
        # visited node also makes the accepted DTO object count finite.
        if self.layers:
            self.charge(64)
        if value is None or type(value) in (bool, int, float):
            if type(value) is int:
                if (self.int64 and not -(1 << 63) <= value < (1 << 63)) or value.bit_length() > 256:
                    _fail("trace_resource_value_invalid")
            if isinstance(value, float) and not math.isfinite(value):
                _fail("trace_resource_value_invalid")
            self.token(json.dumps(value, allow_nan=False))
            return value
        if type(value) is str:
            # Never ask json.dumps/encode to copy an unbounded source string.
            self.token('""')
            for offset in range(0, len(value), 4096):
                self.token(json.dumps(value[offset : offset + 4096], ensure_ascii=True)[1:-1])
            return value
        if isinstance(value, Mapping):
            if len(value) > self.limit // 8:
                _fail(
                    reason="value entry count",
                    diagnostics={"value_entries": len(value), "value_entry_limit": self.limit // 8},
                )
            self.token("{}")
            result: dict[str, Any] | None = {} if self.snapshot else None
            for key, item in value.items():
                if type(key) is not str:
                    _fail("trace_resource_value_invalid")
                self.value(key, depth + 1)
                self.token(": , ")
                item_copy = self.value(item, depth + 1)
                if result is not None:
                    result[key] = item_copy
            return result
        if type(value) in (list, tuple):
            sequence = cast(list[object] | tuple[object, ...], value)
            if len(sequence) > self.limit // 8:
                _fail(
                    reason="value entry count",
                    diagnostics={"value_entries": len(sequence), "value_entry_limit": self.limit // 8},
                )
            self.token("[]")
            result_list: list[Any] | None = [] if self.snapshot else None
            for item in sequence:
                self.token(", ")
                item_copy = self.value(item, depth + 1)
                if result_list is not None:
                    result_list.append(item_copy)
            return result_list
        _fail("trace_resource_value_invalid")


class SpanBudget:
    """Incremental producer hook: call account_spec before retaining each span.

    ``account_spec(child)`` accepts a TraceChildRequest-shaped DTO;
    ``account_spec(request, root=True)`` accepts a TraceCreateRequest-shaped
    DTO. It returns a detached bounded copy when retaining, or ``None`` in
    measurement mode. The root counts toward 4096.
    """

    def __init__(
        self,
        *,
        max_bytes: int = MAX_TRACE_SERIALIZED_SPAN_BYTES,
        max_spans: int = MAX_TRACE_SPANS,
        retain: bool = True,
    ) -> None:
        self.max_bytes = min(max_bytes, ABSOLUTE_MAX_TRACE_BYTES)
        self.max_spans = min(max_spans, ABSOLUTE_MAX_TRACE_SPANS)
        self.retain = retain
        self.serialized_span_upper_bound = 0
        self.span_count = 0

    def account_spec(self, spec: Any, *, root: bool = False) -> Any:
        if self.span_count >= self.max_spans:
            _fail(
                reason="span count",
                diagnostics={"span_count": self.span_count, "span_limit": self.max_spans},
            )
        counter = _ValueCounter(
            self.max_bytes - self.serialized_span_upper_bound,
            snapshot=self.retain,
            reason="serialized span bytes",
        )
        try:
            # Fixed span fields, IDs, timestamps, status, enclosing JSON/OTLP
            # messages, SDK-owned attributes and length-delimited field framing.
            # 4 KiB includes 10-byte varints at every enclosing protobuf level.
            counter.charge(4096)
            fields = (
                ("root_inputs", "root_outputs", "root_attributes")
                if root
                else ("inputs", "outputs", "attributes", "tags", "source_key", "parent_source_key", "span_type")
            )
            init_fields = {field.name for field in dataclass_fields(spec)}
            changes: dict[str, Any] = {}
            for name in ("name", *fields):
                value = counter.value(getattr(spec, name))
                if name in init_fields:
                    changes[name] = value
            for name in ("start_time_ns", "end_time_ns"):
                instant = getattr(spec, name)
                # Span timestamps are protobuf fixed64, whereas all JSON integers
                # inside AnyValue attributes/inputs/outputs are signed int64.
                if instant is not None and (type(instant) is not int or not 0 <= instant < (1 << 64)):
                    _fail("trace_resource_value_invalid")
                counter.charge(64)
                counter.token(json.dumps(instant))
            counter.value(spec.root_status if root else spec.status)
            # Each attribute's wire KeyValue/AnyValue has at most six tags and
            # six 10-byte lengths. Charge per entry, including SDK fields.
            attrs = spec.root_attributes if root else spec.attributes
            counter.charge(128 * (len(attrs) + 8))
        except ItemImportError as error:
            _attach_diagnostics(
                error,
                {
                    "admitted_span_count": self.span_count,
                    "admitted_span_bytes": self.serialized_span_upper_bound,
                    "candidate_span_bytes": counter.size,
                    "candidate_span_byte_limit": counter.limit,
                },
            )
            raise
        self.serialized_span_upper_bound += counter.size
        self.span_count += 1
        if not self.retain:
            return None
        return replace(spec, **changes)

    account_span = account_spec


@dataclass(frozen=True)
class TraceEnvelopeMeasurement:
    serialized_span_upper_bound: int
    span_count: int
    control_metadata_upper_bound: int


@dataclass(frozen=True)
class TraceEnvelope:
    request: TraceCreateRequest
    serialized_span_upper_bound: int
    span_count: int
    control_metadata_upper_bound: int

    @property
    def proof_tags(self) -> dict[str, str]:
        return {
            TRACE_RESOURCE_PROFILE_TAG: TRACE_RESOURCE_PROFILE,
            TRACE_RESOURCE_BYTES_TAG: str(self.serialized_span_upper_bound),
            TRACE_RESOURCE_SPANS_TAG: str(self.span_count),
        }


def _account_request(
    request: TraceCreateRequest,
    *,
    reserve_late_control: bool,
    retain: bool,
    max_bytes: int = MAX_TRACE_SERIALIZED_SPAN_BYTES,
    max_spans: int = MAX_TRACE_SPANS,
    max_control_bytes: int = MAX_TRACE_CONTROL_BYTES,
) -> tuple[TraceCreateRequest | None, TraceEnvelopeMeasurement]:
    root_tag_limit = max_control_bytes // 128
    if len(request.root_tags) > root_tag_limit:
        _fail(
            reason="root tag count",
            diagnostics={"root_tag_count": len(request.root_tags), "root_tag_limit": root_tag_limit},
        )
    request = replace(
        request, root_tags={key: value for key, value in request.root_tags.items() if key not in PROOF_TAG_KEYS}
    )
    budget = SpanBudget(max_bytes=max_bytes, max_spans=max_spans, retain=retain)
    root_spec = budget.account_spec(request, root=True)
    source = root_spec if retain else request
    controls = _ValueCounter(
        max_control_bytes - (LATE_CONTROL_RESERVATION_BYTES if reserve_late_control else 0),
        snapshot=retain,
        reason="control metadata",
    )
    changes: dict[str, Any] = {}
    accepted_children: list[Any] = []
    try:
        for name in (
            "metadata",
            "native_session_id",
            "client_request_id",
            "experiment_id",
            "expected_operation_key",
            "artifact_snapshot_fingerprint",
            "artifact_relative_path",
            "trace_name",
            "request_preview",
            "response_preview",
            "native_usage",
            "native_cost",
        ):
            changes[name] = controls.value(getattr(source, name))
        # Root tags are trace-info control data, not serialized span attributes.
        # Keeping their reservation separate prevents bounded late identity/map
        # decorations from consuming an already-admitted span-byte envelope.
        changes["root_tags"] = controls.value(source.root_tags)
        controls.charge(8192)  # generated analytics, native fields and proof tags
        for child in request.child_specs():
            if PROOF_TAG_KEYS.intersection(child.tags):
                _fail("trace_resource_value_invalid")
            controls.value(child.tags)
            accounted = budget.account_spec(child)
            if retain:
                accepted_children.append(accounted)
    except ItemImportError as error:
        _attach_diagnostics(
            error,
            {
                "control_bytes": controls.size,
                "control_byte_limit": controls.limit,
                "admitted_span_count": budget.span_count,
                "admitted_span_bytes": budget.serialized_span_upper_bound,
            },
        )
        raise
    measurement = TraceEnvelopeMeasurement(
        budget.serialized_span_upper_bound,
        budget.span_count,
        controls.size,
    )
    if not retain:
        return None, measurement
    assert root_spec is not None
    accepted = tuple(accepted_children)
    root = replace(root_spec, **changes, child_specs=lambda: iter(accepted))
    envelope = TraceEnvelope(
        root,
        measurement.serialized_span_upper_bound,
        measurement.span_count,
        measurement.control_metadata_upper_bound,
    )
    root = replace(
        root,
        root_tags={
            **{key: value for key, value in root.root_tags.items() if key not in PROOF_TAG_KEYS},
            **envelope.proof_tags,
        },
    )
    return root, measurement


def admit_trace(
    request: TraceCreateRequest,
    *,
    reserve_late_control: bool = False,
    limits: TraceEnvelopeLimits | None = None,
) -> TraceEnvelope:
    """Admit once, consuming the producer once and retaining only accepted DTOs.

    Global preflight reserves 1 MiB for late attempt/instance/map nonce and
    session tags. Creation repeats admission on the actual decorated request.
    Proof fields are writer-generated and never accepted from the caller.
    """
    effective = limits if limits is not None else DEFAULT_TRACE_ENVELOPE_LIMITS
    root, measurement = _account_request(
        request,
        reserve_late_control=reserve_late_control,
        retain=True,
        max_bytes=effective.max_bytes,
        max_spans=effective.max_spans,
        max_control_bytes=effective.max_control_bytes,
    )
    assert root is not None
    return TraceEnvelope(
        root,
        measurement.serialized_span_upper_bound,
        measurement.span_count,
        measurement.control_metadata_upper_bound,
    )


def measure_trace(
    request: TraceCreateRequest,
    *,
    max_bytes: int = DIAGNOSTIC_MAX_TRACE_BYTES,
    max_spans: int = DIAGNOSTIC_MAX_TRACE_SPANS,
    max_control_bytes: int = DIAGNOSTIC_MAX_TRACE_CONTROL_BYTES,
) -> TraceEnvelopeMeasurement:
    """Count one request without retaining its admitted child DTOs."""
    root, measurement = _account_request(
        request,
        reserve_late_control=True,
        retain=False,
        max_bytes=max_bytes,
        max_spans=max_spans,
        max_control_bytes=max_control_bytes,
    )
    assert root is None
    return measurement


def validate_control_metadata(
    tags: Mapping[str, object], metadata: Mapping[str, object], *, native_fields: object = None
) -> int:
    """Bound a proposed/returned control envelope without materializing a copy."""
    controls = _ValueCounter(MAX_TRACE_CONTROL_BYTES, snapshot=False)
    controls.value(tags)
    controls.value(metadata)
    controls.value(native_fields)
    return controls.size


def require_trace_proof(info: Any) -> tuple[int, int]:
    """Validate metadata-only proof before any full-payload request."""
    tags = info.tags
    try:
        if tags.get(TRACE_RESOURCE_PROFILE_TAG) != TRACE_RESOURCE_PROFILE:
            raise ValueError
        raw_bytes, raw_count = tags[TRACE_RESOURCE_BYTES_TAG], tags[TRACE_RESOURCE_SPANS_TAG]
        if len(raw_bytes) > 10 or len(raw_count) > 10:
            raise ValueError
        size, count = int(raw_bytes), int(raw_count)
        if not 0 < size <= ABSOLUTE_MAX_TRACE_BYTES or not 0 < count <= ABSOLUTE_MAX_TRACE_SPANS:
            raise ValueError
        if str(size) != raw_bytes or str(count) != raw_count:
            raise ValueError
    except (KeyError, ValueError, TypeError):
        _fail("trace_resource_profile_unverified")
    validate_control_metadata(
        tags,
        getattr(info, "trace_metadata", {}),
        native_fields=[
            getattr(info, name, None) for name in ("client_request_id", "request_preview", "response_preview")
        ],
    )
    return size, count


def assert_returned_trace(trace: Any, proof: tuple[int, int]) -> None:
    """Recheck actual spans after the trusted-writer metadata gate."""
    size, count = proof
    spans = trace.data.spans
    if len(spans) != count or len(spans) > ABSOLUTE_MAX_TRACE_SPANS:
        _fail("trace_resource_proof_mismatch")
    seen = 0
    json_seen = 0
    for span in spans:
        to_proto = getattr(span, "to_otel_proto", None)
        if callable(to_proto):
            try:
                seen += to_proto().ByteSize()
            except (ValueError, OverflowError, TypeError):
                _fail("trace_resource_proof_mismatch")
        # Count complete full-read JSON independently of the protobuf form.
        to_dict = getattr(span, "to_dict", None)
        value = to_dict() if callable(to_dict) else vars(span)
        counter = _ValueCounter(size, layers=0, snapshot=False, int64=False)
        try:
            counter.value(value)
        except ItemImportError:
            _fail("trace_resource_proof_mismatch")
        json_seen += counter.size
        if json_seen > size:
            _fail("trace_resource_proof_mismatch")
        if seen > size or seen > ABSOLUTE_MAX_TRACE_BYTES:
            _fail("trace_resource_proof_mismatch")
