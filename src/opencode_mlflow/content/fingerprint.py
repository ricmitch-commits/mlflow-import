"""Streaming fingerprints over canonical-json-v1 material.

Every fingerprint is a 64-character lowercase-hex SHA-256 over a
versioned scheme/domain header followed by length-framed canonical
bytes, so different hash inputs can never collide by concatenation.

Boundaries fixed here (each is pinned by tests):

- :func:`session_context_fingerprint` covers only immutable session
  identity/selection context: session ID, schema (adapter) generation,
  project/workspace/parent/fork IDs, slug, title, directory/path,
  agent, configured model, OpenCode version, and the creation instant
  as epoch nanoseconds. Mutable session snapshot fields — update
  times, lifecycle/archival state, cost/token totals, summaries, turn
  counts, and the raw column map — are excluded.
- :func:`source_fingerprint` is turn-local: the canonical-json-v1 tag,
  the turn's schema (adapter) generation, the context fingerprint, the
  session-scoped records assigned to the turn, the selected ordered
  turn records, and the effective redaction mode. Unassigned records,
  other turns' records, and mutable session fields stay out, so a
  turn-local change replaces only that turn while a session identity
  change replaces every turn.
- :func:`render_fingerprint` combines a source fingerprint with the
  artifact schema, mapping, and importer versions plus the redaction
  mode; switching verbatim/redacted replaces the render.
- :func:`artifact_snapshot_fingerprint` streams the complete canonical
  session payload — every session field (mutable fields included,
  instants as epoch nanoseconds) plus all permitted records visible
  through the turn's replay cursors (the turn's own, session-scoped,
  and unassigned/other-turn records) — and terminates with the fixed
  :data:`IDENTITY_MAP_PENDING` placeholder instead of any generated
  MLflow identity. Replay values are canonicalized exactly as
  observed: the normalized snapshot persists post-redaction values, so
  no redactor runs here and the redaction-mode tag distinguishes modes.

:class:`EffectiveTurn` is the streaming view Tasks 5–6 consume; it
never holds a record list or generated MLflow IDs. The
:class:`~opencode_mlflow.source.replay.ReplayTurn` protocol it wraps is
owned exclusively by :mod:`opencode_mlflow.source.replay` and is
imported here, never redefined.

The shared hashing primitives also live here, so every fingerprint —
these and the turn-local fingerprints of
:mod:`opencode_mlflow.source.turns` — hashes the same record form the
same way: :func:`canonical_record_value` builds the one canonical form
of a :class:`~opencode_mlflow.source.model.SourceRecord`, and
:class:`FramedDigest` is the incremental length-framed digest (the
turn-local ``turn-records-v1`` hash is a ``FramedDigest`` over
:func:`canonical_record_value` chunks with no redactor, because
replayed snapshot values are already effective).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from itertools import chain
from typing import cast

from opencode_mlflow.content.canonical import CANONICAL_JSON_VERSION, canonical_json
from opencode_mlflow.contracts import IDENTITY_REDACTOR, RedactionMode, Redactor
from opencode_mlflow.source.model import (
    AdapterGeneration,
    SessionHeader,
    SourceRecord,
    TimestampValue,
    TurnSummary,
)
from opencode_mlflow.source.replay import ReplayTurn

__all__ = [
    "IDENTITY_MAP_PENDING",
    "EffectiveTurn",
    "FramedDigest",
    "artifact_snapshot_fingerprint",
    "build_effective_turn",
    "canonical_record_value",
    "render_fingerprint",
    "session_context_fingerprint",
    "source_fingerprint",
]

IDENTITY_MAP_PENDING = "identity_map_pending"
"""Fixed placeholder substituted for the generated identity map.

The generated MLflow trace/span identity map is deliberately excluded
from every fingerprint input; artifact writers hash with this
placeholder first and insert generated IDs only afterwards, so
generated IDs never change an artifact path or fingerprint."""

_FINGERPRINT_SCHEME = "opencode-mlflow-fingerprint-v1"
_DOMAIN_SESSION_CONTEXT = "session-context-v1"
_DOMAIN_TURN_SOURCE = "turn-source-v1"
_DOMAIN_RENDER = "render-v1"
_DOMAIN_ARTIFACT_SNAPSHOT = "artifact-snapshot-v1"

_SESSION_CONTEXT_FIELDS = (
    "project_id",
    "workspace_id",
    "parent_id",
    "fork_session_id",
    "slug",
    "title",
    "directory",
    "path",
    "version",
    "agent",
    "model",
)

_RECORD_OPTIONAL_FIELDS = (
    "source_id",
    "session_id",
    "sequence_domain",
    "sequence_value",
    "parent_source_id",
    "parent_source_table",
    "derived_child_ordinal",
    "source_identity",
    "join_projection_key",
    "classification",
)


@dataclass(frozen=True)
class EffectiveTurn:
    """Streaming canonical view of one logical turn.

    The view carries the immutable session-context fields
    (:attr:`session_header`), identity/outcome counts
    (:attr:`summary`), the redaction mode, and the wrapped
    :class:`~opencode_mlflow.source.replay.ReplayTurn` plus
    :class:`~opencode_mlflow.contracts.Redactor` that the fresh
    canonical-record cursors reopen. It never holds a list of records
    or generated MLflow IDs; :func:`build_effective_turn` is its
    producer.
    """

    session_header: SessionHeader
    summary: TurnSummary
    redaction_mode: RedactionMode
    replay: ReplayTurn = field(repr=False, compare=False)
    redactor: Redactor = field(repr=False, compare=False)

    @property
    def session_id(self) -> str:
        return self.session_header.session_id

    @property
    def turn_key(self) -> str:
        return self.summary.turn_key

    @property
    def schema_generation(self) -> AdapterGeneration:
        """The turn's schema generation (each session is exactly one generation)."""
        return self.session_header.adapter_generation

    def iter_canonical_records(self) -> Iterator[bytes]:
        """Fresh cursor of canonical bytes for the turn's selected ordered records."""
        return _iter_canonical_records(self.replay.iter_records(), self.redactor)

    def iter_canonical_session_records(self) -> Iterator[bytes]:
        """Fresh cursor of canonical bytes for the session-scoped records."""
        return _iter_canonical_records(self.replay.iter_session_records(), self.redactor)

    def iter_canonical_unassigned_records(self) -> Iterator[bytes]:
        """Fresh cursor of canonical bytes for the unassigned/other-turn records."""
        return _iter_canonical_records(self.replay.iter_unassigned_records(), self.redactor)


def build_effective_turn(
    replay_turn: ReplayTurn,
    redactor: Redactor,
    redaction_mode: RedactionMode = RedactionMode.VERBATIM,
) -> EffectiveTurn:
    """Build the streaming canonical view of ``replay_turn``.

    ``redactor`` runs on every record value before canonicalization, so
    the cursors expose exactly the effective values that feed turn
    fingerprints. ``redaction_mode`` records the content policy the view
    was built for (verbatim is the default policy); the mode parameter
    of :func:`source_fingerprint` stays the authoritative hash input.
    """
    return EffectiveTurn(
        session_header=replay_turn.session_header,
        summary=replay_turn.summary,
        redaction_mode=redaction_mode,
        replay=replay_turn,
        redactor=redactor,
    )


def session_context_fingerprint(turn: EffectiveTurn) -> str:
    """Fingerprint the immutable session identity/selection context.

    Covers session ID, schema (adapter) generation, project/workspace/
    parent/fork IDs, slug, title, directory/path, agent, configured
    model, OpenCode version, and the creation instant (epoch
    nanoseconds). Everything else on the session header — mutable
    snapshot fields, turn counts, warnings, and the raw column map — is
    excluded, so lifecycle updates never replace a turn.
    """
    header = turn.session_header
    context: dict[str, object] = {
        "session_id": header.session_id,
        "schema_generation": header.adapter_generation,
    }
    if header.catalog_version:
        context["catalog_version"] = header.catalog_version
    for name in _SESSION_CONTEXT_FIELDS:
        value = getattr(header, name)
        if value is not None:
            context[name] = value
    if header.time_created is not None:
        context["time_created_ns"] = header.time_created.nanoseconds
    return _hash_chunks(_DOMAIN_SESSION_CONTEXT, [canonical_json(context)])


def source_fingerprint(turn: EffectiveTurn, redaction_mode: RedactionMode) -> str:
    """Fingerprint the turn's effective source content (turn-local replacement).

    Material: the canonical-json-v1 tag, the turn's schema (adapter)
    generation, the session context fingerprint, the session-scoped
    records assigned to the turn, the selected ordered turn records
    (both as redacted-then-canonicalized bytes), and the effective
    redaction mode. Unassigned records, other turns' records, and
    mutable session snapshot fields are excluded.
    """
    material_value: dict[str, object] = {
        "canonical_json": CANONICAL_JSON_VERSION,
        "schema_generation": turn.session_header.adapter_generation,
        "session_context_fingerprint": session_context_fingerprint(turn),
        "redaction_mode": redaction_mode.value,
    }
    if turn.session_header.catalog_version:
        material_value["catalog_version"] = turn.session_header.catalog_version
    material = canonical_json(material_value)
    return _hash_chunks(
        _DOMAIN_TURN_SOURCE,
        chain((material,), turn.iter_canonical_session_records(), turn.iter_canonical_records()),
    )


def render_fingerprint(
    source_fp: str,
    artifact_version: str,
    mapping_version: str,
    importer_version: str,
    redaction_mode: RedactionMode,
) -> str:
    """Fingerprint the artifact/render contract for one turn.

    Combines the turn source fingerprint with the artifact schema,
    mapping, and importer versions and the redaction mode, so any
    contract-version or content-policy change replaces the rendered
    artifact rather than being treated as an unchanged skip.
    """
    material = canonical_json(
        {
            "source_fingerprint": source_fp,
            "artifact_schema_version": artifact_version,
            "mapping_version": mapping_version,
            "importer_version": importer_version,
            "redaction_mode": redaction_mode.value,
        }
    )
    return _hash_chunks(_DOMAIN_RENDER, [material])


def artifact_snapshot_fingerprint(turn: ReplayTurn, redaction_mode: RedactionMode) -> str:
    """Fingerprint the complete canonical session payload before identity mapping.

    Streams the complete session payload — every session field (mutable
    snapshot fields included, instants as epoch nanoseconds) plus all
    permitted records visible through the turn's replay cursors — and
    terminates with the fixed :data:`IDENTITY_MAP_PENDING` placeholder.
    It changes for exactly the mutable fields and other-turn records
    that :func:`source_fingerprint` excludes.

    Ruling 11 (pinned): this material — the complete session header
    plus every permitted record from the three replay cursors plus the
    fixed placeholder — is complete over independent content. Root
    input/output payloads and the logical-turn boundary/summary
    sections the artifact persists are deterministic DERIVATIONS of the
    hashed records, not independent content: the artifact may persist
    them as derived sections, but any change to a derivation rule bumps
    the importer/mapping/artifact versions and therefore the render
    fingerprint, so a derived artifact section can never drift from
    this hash silently.
    """
    return _hash_chunks(_DOMAIN_ARTIFACT_SNAPSHOT, _iter_artifact_snapshot_chunks(turn, redaction_mode))


def _iter_artifact_snapshot_chunks(turn: ReplayTurn, redaction_mode: RedactionMode) -> Iterator[bytes]:
    """Stream the pre-identity-map artifact-snapshot material, record by record."""
    preamble: dict[str, object] = {
        "canonical_json": CANONICAL_JSON_VERSION,
        "schema_generation": turn.session_header.adapter_generation,
        "redaction_mode": redaction_mode.value,
    }
    if turn.session_header.catalog_version:
        preamble["catalog_version"] = turn.session_header.catalog_version
    yield canonical_json(preamble)
    yield canonical_json(_complete_session_value(turn.session_header))
    yield from _iter_canonical_records(turn.iter_records(), IDENTITY_REDACTOR)
    yield from _iter_canonical_records(turn.iter_session_records(), IDENTITY_REDACTOR)
    yield from _iter_canonical_records(turn.iter_unassigned_records(), IDENTITY_REDACTOR)
    yield canonical_json({"identity_map": IDENTITY_MAP_PENDING})


def _hash_chunks(domain: str, chunks: Iterable[bytes]) -> str:
    """SHA-256 over a versioned scheme/domain header plus length-framed chunks."""
    digest = FramedDigest(domain)
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


class FramedDigest:
    """Incremental SHA-256 over the versioned scheme/domain header plus length-framed chunks.

    Every fingerprint — the four in this module and the turn-local
    ``turn-records-v1`` hash in :mod:`opencode_mlflow.source.turns` — is
    computed the same way: a canonical-json-v1 ``{"scheme", "domain"}``
    header first, then each chunk prefixed with its lowercase-hex length
    and a colon, so two different chunk sequences never concatenate to
    the same byte stream and distinct domains never share a hash input.
    :meth:`update` absorbs one chunk at a time (streaming, never
    materializing the chunk list); :meth:`hexdigest` finalizes.
    """

    def __init__(self, domain: str) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(canonical_json({"scheme": _FINGERPRINT_SCHEME, "domain": domain}))

    def update(self, chunk: bytes) -> None:
        """Frame and absorb one chunk."""
        self._digest.update(b"%x:" % len(chunk))
        self._digest.update(chunk)

    def hexdigest(self) -> str:
        """Return the finalized lowercase-hex SHA-256."""
        return self._digest.hexdigest()


def canonical_record_value(record: SourceRecord, redactor: Redactor | None = None) -> dict[str, object]:
    """Build one record's canonical form for hashing.

    Structural identity and ordering fields are included verbatim when
    present (absent stays absent — a present ``None`` is JSON null, an
    omitted key is an absent column). Timestamps contribute epoch
    nanoseconds so the same instant in different units hashes equally.
    This is the one record form every fingerprint hashes: the
    effective-turn cursors here and the turn-local fingerprints in
    :mod:`opencode_mlflow.source.turns` both canonicalize records
    through this function.

    ``redactor`` runs on the decoded value, raw text, and raw columns
    before they enter the material, so the effective values are what
    fingerprints observe. ``None`` uses the values exactly as recorded —
    replayed snapshot values are already effective (post-redaction), so
    turn-local hashes over replay cursors pass no redactor; callers
    hashing live pre-redaction values must supply theirs.
    """

    def effective(value: object) -> object:
        if redactor is None:
            return value
        # Structural identity/order fields are emitted separately below;
        # decoded/raw projections are content, including additive raw-column
        # names that happen to resemble structural fields.
        return redactor.redact(value, ())

    value: dict[str, object] = {
        "source_table": record.source_table,
        "source_type": record.source_type,
        "warnings": list(record.warnings),
    }
    for name in _RECORD_OPTIONAL_FIELDS:
        field_value = getattr(record, name)
        if field_value is not None:
            value[name] = field_value
    if record.timestamp is not None:
        value["timestamp_ns"] = record.timestamp.nanoseconds
    if record.completion_timestamp is not None:
        value["completion_timestamp_ns"] = record.completion_timestamp.nanoseconds
    if record.raw_json is not None or record.decoded_value is not None:
        value["decoded"] = effective(record.decoded_value)
    if record.raw_json is not None:
        value["raw_text"] = effective(record.raw_json)
    if record.raw_columns:
        value["raw_columns"] = effective(record.raw_columns)
    return value


def _iter_canonical_records(records: Iterator[SourceRecord], redactor: Redactor) -> Iterator[bytes]:
    """Canonicalize each record's effective values, one record at a time."""
    for record in records:
        yield canonical_json(canonical_record_value(record, redactor))


def _complete_session_value(header: SessionHeader) -> dict[str, object]:
    """Build the complete session payload material (every field, instants as ns)."""
    value: dict[str, object] = {}
    for item in dataclass_fields(SessionHeader):
        field_value = getattr(header, item.name)
        # The database identity is provenance metadata, not source/session
        # identity. Including it here would make copies of the same source
        # session acquire different artifact paths and violate the explicit
        # reconciliation isolation contract.
        if item.name == "source_fingerprint":
            continue
        if item.name == "catalog_version" and not field_value:
            continue
        if item.name == "raw_columns":
            raw_columns = cast(Mapping[str, object], field_value)
            if raw_columns:
                value["raw_columns"] = dict(raw_columns)
            continue
        if isinstance(field_value, TimestampValue):
            value[item.name] = field_value.nanoseconds
        elif field_value is not None:
            value[item.name] = field_value
    return value
