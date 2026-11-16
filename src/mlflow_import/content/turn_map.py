"""Pure, streaming v2 turn-map construction and pre-mutation size admission.

Admission consumes the already-admitted request's replayable child snapshot, not
the mapper. Upload uses the same builder with generated identities. No target,
artifact, journal, or temporary-file operations occur here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from mlflow_import.compatibility import request_ownership
from mlflow_import.content import accounting
from mlflow_import.content.canonical import StreamingArray, iter_canonical_json
from mlflow_import.content.input_context import INPUT_CONTEXT_ATTRIBUTE, TURN_MAP_VERSION
from mlflow_import.contracts import RedactionMode
from mlflow_import.errors import ItemImportError
from mlflow_import.source.replay import ReplayTurn

# Pinned generated IDs contain only JSON-safe ASCII. Reserve every occurrence
# at its maximum width, including aliases, context entries, and root targets.
MAX_GENERATED_TRACE_ID_CHARS = 35  # "tr-" + 32 hex digits
MAX_GENERATED_SPAN_ID_CHARS = 16


def build_turn_map(
    replay: ReplayTurn,
    *,
    children: Iterable[Any],
    accounting: Mapping[str, Any],
    ownership: Mapping[str, Any],
    trace_id: str,
    root_span_id: str,
    span_ids_by_source: Mapping[str, str],
    mode: RedactionMode,
) -> dict[str, Any]:
    """Build one-pass canonical wire values shared by admission and publication."""
    from mlflow_import.mlflow.mapping import SourceIdentity

    targets: dict[str, list[str]] = {}
    contexts: list[dict[str, Any]] = []
    for child in children:
        span_id = span_ids_by_source.get(child.source_key)
        if INPUT_CONTEXT_ATTRIBUTE in child.attributes and span_id is None:
            raise ItemImportError(
                "artifact_verification_failed", "a projected input context has no generated span identity"
            )
        if span_id is None:
            continue
        aliases = child.attributes.get("mlflow_import.source_keys", [child.source_key])
        if not isinstance(aliases, (list, tuple)) or any(not isinstance(key, str) for key in aliases):
            raise ItemImportError(
                "artifact_verification_failed", "projected source aliases are not complete identities"
            )
        for key in aliases:
            targets.setdefault(key, []).append(span_id)
        if INPUT_CONTEXT_ATTRIBUTE in child.attributes:
            contexts.append(
                {
                    "source_key": child.source_key,
                    "span_id": span_id,
                    "descriptor": child.attributes[INPUT_CONTEXT_ATTRIBUTE],
                }
            )
    first_record = next(replay.iter_records(), None)
    if first_record is not None:
        targets[SourceIdentity.of(first_record).key] = [root_span_id]

    def records() -> Iterator[dict[str, Any]]:
        for record in replay.iter_records():
            key = SourceIdentity.of(record).key
            yield {
                "source_table": record.source_table,
                "source_key": key,
                "scope": str(record.raw_columns.get("subagent") or ""),
                "source_id": record.source_id,
                "sequence": record.sequence_value,
                "occurrence_id": record.occurrence_id,
                "entity_id": record.entity_id or record.source_id,
                "invocation_id": record.invocation_id,
                "span_ids": targets.get(key, []),
            }

    return {
        "schema_version": TURN_MAP_VERSION,
        "turn_key": replay.summary.turn_key,
        "turn_fingerprint": ownership.get(
            "opencode.render_fingerprint", ownership.get("mlflow_import.turn_fingerprint")
        ),
        "mapping_version": ownership.get("opencode.mapping_version", ownership.get("mlflow_import.mapping_version")),
        "importer_version": ownership.get("opencode.importer_version", ownership.get("mlflow_import.importer_version")),
        "identity_map": {"trace_id": trace_id, **{"span:" + key: value for key, value in span_ids_by_source.items()}},
        "privacy": mode.value,
        "accounting": accounting,
        "input_contexts": StreamingArray(contexts),
        # No mutable session-generation or late artifact-placement pointer.
        "records": StreamingArray(records()),
    }


def iter_turn_map_json(value: Mapping[str, Any]) -> Iterator[bytes]:
    """Stream canonical bytes with the same finite bound before and after create."""
    limit = accounting.MAX_PRICING_MAP_BYTES
    emitted = 0
    for chunk in iter_canonical_json(value):
        emitted += len(chunk)
        if emitted > limit:
            raise ItemImportError(
                "turn_map_too_large", "the canonical turn map exceeds the qualified pricing/recovery envelope"
            )
        yield chunk


def validate_turn_map_admission(replay: ReplayTurn, request: Any, mode: RedactionMode) -> None:
    """Reject an oversized prospective v2 map before any target mutation.

    ``request`` must have passed trace-envelope admission and expose its frozen,
    replayable child factory. This function consumes that factory exactly once;
    it never reprojects activity or reselects accounting/pricing observations.
    Generated IDs use maximum-width placeholders; late artifact nonces and tags
    do not enter map content. The actual uploader enforces the bound again.
    """
    value = _prospective_turn_map(replay, request, mode)
    for _ in iter_turn_map_json(value):
        pass


def _prospective_turn_map(replay: ReplayTurn, request: Any, mode: RedactionMode) -> dict[str, Any]:
    children = tuple(request.child_specs())
    spans: dict[str, str] = {}
    for index, child in enumerate(children, 1):
        if child.source_key in spans:
            raise ItemImportError("artifact_verification_failed", "projected child source identities are duplicated")
        spans[child.source_key] = f"{index:0{MAX_GENERATED_SPAN_ID_CHARS}x}"
    accounting = request.root_attributes.get("mlflow_import.accounting")
    if not isinstance(accounting, Mapping):
        raise ItemImportError("artifact_verification_failed", "the admitted turn has no complete accounting provenance")
    return build_turn_map(
        replay,
        children=children,
        accounting=accounting,
        ownership=request_ownership(request),
        trace_id="t" * MAX_GENERATED_TRACE_ID_CHARS,
        root_span_id="0" * MAX_GENERATED_SPAN_ID_CHARS,
        span_ids_by_source=spans,
        mode=mode,
    )
