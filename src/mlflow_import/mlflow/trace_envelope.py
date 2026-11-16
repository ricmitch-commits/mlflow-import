"""Pure, bounded admission for the pinned trace transport.

The bound covers both OTLP protobuf and JSON full reads, not just canonical
source JSON. No SDK, target, artifact, or journal operations occur here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
TRACE_RESOURCE_PROFILE = "mlflow-import.trace-envelope.v1"
TRACE_RESOURCE_PROFILE_TAG = "mlflow_import.trace_resource_profile"
TRACE_RESOURCE_BYTES_TAG = "mlflow_import.trace_resource_bound_bytes"
TRACE_RESOURCE_SPANS_TAG = "mlflow_import.trace_resource_span_count"
PROOF_TAG_KEYS = frozenset({TRACE_RESOURCE_PROFILE_TAG, TRACE_RESOURCE_BYTES_TAG, TRACE_RESOURCE_SPANS_TAG})


def _fail(code: str = "trace_resource_envelope_exceeded") -> NoReturn:
    raise ItemImportError(code, "the trace is outside the finite serialized resource envelope")


class _ValueCounter:
    """Count an ASCII JSON value quoted twice more, snapshotting bounded DTOs.

    Each extra JSON string layer adds one byte per quote/backslash plus its
    two delimiters. Counting those characters explicitly covers historical
    JSON-looking strings' extra encoding and full-read JSON escaping. Unicode
    uses ASCII escapes (including surrogate pairs), an upper bound on UTF-8.
    Containers include the default encoder's spaces. Depth and every visited
    node are bounded before allocating their snapshots.
    """

    def __init__(self, limit: int, *, layers: int = 2, snapshot: bool = True, int64: bool = True) -> None:
        self.limit = limit
        self.size = 0
        self.layers = layers
        self.snapshot = snapshot
        self.int64 = int64

    def charge(self, size: int) -> None:
        self.size += size
        if self.size > self.limit:
            _fail()

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
                _fail()
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
                _fail()
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
    DTO. It returns a detached bounded copy. The root counts toward 4096.
    """

    def __init__(self, *, max_bytes: int = MAX_TRACE_SERIALIZED_SPAN_BYTES, max_spans: int = MAX_TRACE_SPANS) -> None:
        self.max_bytes = min(max_bytes, MAX_TRACE_SERIALIZED_SPAN_BYTES)
        self.max_spans = min(max_spans, MAX_TRACE_SPANS)
        self.serialized_span_upper_bound = 0
        self.span_count = 0

    def account_spec(self, spec: Any, *, root: bool = False) -> Any:
        if self.span_count >= self.max_spans:
            _fail()
        counter = _ValueCounter(self.max_bytes - self.serialized_span_upper_bound)
        # Fixed span fields, IDs, timestamps, status, enclosing JSON/OTLP
        # messages, SDK-owned attributes and length-delimited field framing.
        # 4 KiB includes 10-byte varints at every enclosing protobuf level.
        counter.charge(4096)
        fields = (
            ("root_inputs", "root_outputs", "root_attributes")
            if root
            else ("inputs", "outputs", "attributes", "tags", "source_key", "parent_source_key", "span_type")
        )
        changes = {name: counter.value(getattr(spec, name)) for name in ("name", *fields)}
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
        self.serialized_span_upper_bound += counter.size
        self.span_count += 1
        return replace(spec, **changes)

    account_span = account_spec


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


def admit_trace(request: TraceCreateRequest, *, reserve_late_control: bool = False) -> TraceEnvelope:
    """Admit once, consuming the producer once and retaining only accepted DTOs.

    Global preflight reserves 1 MiB for late attempt/instance/map nonce and
    session tags. Creation repeats admission on the actual decorated request.
    Proof fields are writer-generated and never accepted from the caller.
    """
    if len(request.root_tags) > MAX_TRACE_CONTROL_BYTES // 128:
        _fail()
    request = replace(
        request, root_tags={key: value for key, value in request.root_tags.items() if key not in PROOF_TAG_KEYS}
    )
    budget = SpanBudget()
    root = budget.account_spec(request, root=True)
    controls = _ValueCounter(MAX_TRACE_CONTROL_BYTES - (LATE_CONTROL_RESERVATION_BYTES if reserve_late_control else 0))
    changes = {}
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
        changes[name] = controls.value(getattr(root, name))
    # Root tags are trace-info control data, not serialized span attributes.
    # Keeping their reservation separate prevents bounded late identity/map
    # decorations from consuming an already-admitted span-byte envelope.
    changes["root_tags"] = controls.value(root.root_tags)
    controls.charge(8192)  # generated analytics, native fields and proof tags
    children = []
    for child in request.child_specs():
        if PROOF_TAG_KEYS.intersection(child.tags):
            _fail("trace_resource_value_invalid")
        controls.value(child.tags)
        children.append(budget.account_spec(child))
    accepted = tuple(children)
    root = replace(root, **changes, child_specs=lambda: iter(accepted))
    envelope = TraceEnvelope(root, budget.serialized_span_upper_bound, budget.span_count, controls.size)
    root = replace(
        root,
        root_tags={
            **{key: value for key, value in root.root_tags.items() if key not in PROOF_TAG_KEYS},
            **envelope.proof_tags,
        },
    )
    return replace(envelope, request=root)


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
        if not 0 < size <= MAX_TRACE_SERIALIZED_SPAN_BYTES or not 0 < count <= MAX_TRACE_SPANS:
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
    if len(spans) != count or len(spans) > MAX_TRACE_SPANS:
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
        if seen > size or seen > MAX_TRACE_SERIALIZED_SPAN_BYTES:
            _fail("trace_resource_proof_mismatch")
