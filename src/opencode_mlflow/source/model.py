"""Canonical source model for OpenCode SQLite session data.

These types are the stable in-memory seam between live exporters (Task 4),
the normalized temporary snapshot, and replay/discovery. The model is the
lowest source layer: it imports nothing from the catalog, snapshot,
replay, or probe modules.

Source-safety invariants owned here:

- :class:`TimestampValue` normalizes a source timestamp to a timezone-aware
  UTC instant while retaining the original numeric value and unit, so
  fingerprints can use exact integer nanoseconds and artifacts can show the
  original representation.
- :func:`source_database_fingerprint` hashes only the schema generation and
  available non-secret filesystem identity/statistics. It never reads
  source content and never includes the absolute path; it is metadata only,
  not reconciliation identity.
- :class:`SourceRecord` keeps raw source values (IDs, sequence values, raw
  JSON/text) distinct from decoded values, so a malformed payload can never
  corrupt identity or ordering.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat as stat_module
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, cast, runtime_checkable

from opencode_mlflow.json_values import decode_lossless_json

SchemaGeneration = Literal["legacy", "current", "mixed"]
"""Schema generation of a source database: legacy, current, or both (mixed)."""

AdapterGeneration = Literal["legacy", "current"]
"""Generation of an individually emitted session; mixed databases still emit
each session as exactly one of ``legacy`` or ``current``."""

TimestampUnit = Literal["s", "ms", "us", "ns"]
"""Original unit of a source timestamp value."""

RecordClassificationName = Literal["user", "assistant", "system", "synthetic"]
"""Canonical classification of a source record."""

_TIMESTAMP_UNIT_NANOSECONDS: Mapping[str, int] = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

_FINGERPRINT_VERSION = "opencode-mlflow-source-fingerprint-v1"


@dataclass(frozen=True)
class TimestampValue:
    """A source timestamp with its original numeric value and unit.

    ``utc`` is the timezone-aware UTC instant (datetime microsecond
    resolution); ``nanoseconds`` is the exact integer instant used for
    fingerprint material. The original ``value``/``unit`` pair is retained
    so artifacts can show the source representation unchanged.
    """

    value: int | float
    unit: TimestampUnit

    @property
    def nanoseconds(self) -> int:
        factor = _TIMESTAMP_UNIT_NANOSECONDS[self.unit]
        if isinstance(self.value, int):
            return self.value * factor
        if not math.isfinite(self.value):
            raise ValueError("a timestamp value must be finite")
        return int(round(self.value * factor))

    @property
    def utc(self) -> datetime:
        total_nanoseconds = self.nanoseconds
        seconds, remainder = divmod(total_nanoseconds, 1_000_000_000)
        microseconds, _ = divmod(remainder, 1_000)
        return _EPOCH + timedelta(seconds=seconds, microseconds=microseconds)


def normalize_timestamp(value: int | float | None, unit: str) -> TimestampValue | None:
    """Normalize a source timestamp of a known unit to a :class:`TimestampValue`.

    ``None`` (a missing source value) stays absent. Unknown units are
    rejected with :class:`ValueError`; adapters map that rejection to a
    warning and treat the timestamp as unnormalizable, rather than guessing
    a unit.
    """
    if value is None:
        return None
    if unit not in _TIMESTAMP_UNIT_NANOSECONDS:
        raise ValueError("unknown timestamp unit")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("a timestamp value must be finite")
    return TimestampValue(value=value, unit=cast(TimestampUnit, unit))


@dataclass(frozen=True)
class SchemaDescription:
    """Result of probing a source database against the schema catalog.

    ``tables`` maps each matched catalogued table to its observed column
    names (metadata only, never row values). ``warnings`` carries safe
    warning codes such as ``session_share_omitted``.

    ``tie_ranks`` (the fixed source-type rank table), ``timestamp_units``
    (each catalogued timestamp column's unit, keyed ``table.column``), and
    ``classifications`` (stable descriptions of the catalog's record
    classification rules, including how user records are classified) are
    populated by the live adapters from their owning catalog; the bare
    probe leaves them empty because it reads metadata only.
    """

    generation: SchemaGeneration
    catalog_version: str
    tables: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    tie_ranks: Mapping[str, int] = field(default_factory=dict)
    timestamp_units: Mapping[str, str] = field(default_factory=dict)
    classifications: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionHeader:
    """Normalized header of one source session.

    Typed properties hold the fields required for selection and MLflow
    mapping; missing values remain absent (``None``). ``raw_columns``
    retains the complete permitted source session-row map, including
    additive or unknown columns, for the artifact envelope. ``catalog_version``
    is stable catalog provenance carried through normalized replay.
    """

    session_id: str
    adapter_generation: AdapterGeneration
    project_id: str | None = None
    workspace_id: str | None = None
    parent_id: str | None = None
    fork_session_id: str | None = None
    fork_boundary: TimestampValue | None = None
    slug: str | None = None
    title: str | None = None
    directory: str | None = None
    path: str | None = None
    version: str | None = None
    agent: str | None = None
    model: object | None = None
    metadata: object | None = None
    revert: object | None = None
    permission: object | None = None
    cost: float | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    tokens_reasoning: int | None = None
    tokens_cache_read: int | None = None
    tokens_cache_write: int | None = None
    summary_additions: int | None = None
    summary_deletions: int | None = None
    summary_files: int | None = None
    summary_diffs: object | None = None
    time_created: TimestampValue | None = None
    time_updated: TimestampValue | None = None
    time_compacting: TimestampValue | None = None
    time_archived: TimestampValue | None = None
    time_idle: TimestampValue | None = None
    time_viewed: TimestampValue | None = None
    time_suspended: TimestampValue | None = None
    idle_outcome: str | None = None
    resume_attempts: int | None = None
    source_fingerprint: str | None = None
    raw_columns: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    catalog_version: str = ""


@dataclass(frozen=True)
class SourceRecord:
    """One canonical record emitted for a source row.

    Raw provenance (``source_id``, ``sequence_value``, ``raw_json``,
    ``raw_columns``) is deliberately distinct from ``decoded_value``: a
    malformed payload sets ``decoded_value`` to ``None`` and records a
    warning while the raw text is retained. Identity and ordering fields
    (``sequence_domain``, ``parent_source_id``, ``parent_source_table``,
    ``derived_child_ordinal``,
    ``source_identity``, ``join_projection_key``) let adapter contract
    tests verify session ownership, cross-table order, and safe
    missing-ID identity; exporters populate them via direct construction
    after :meth:`from_row`.
    """

    source_table: str
    source_type: str
    source_id: str | None = None
    raw_json: str | None = None
    decoded_value: object | None = None
    raw_columns: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    session_id: str | None = None
    sequence_domain: str | None = None
    sequence_value: object | None = None
    parent_source_id: str | None = None
    parent_source_table: str | None = None
    derived_child_ordinal: int | None = None
    source_identity: str | None = None
    join_projection_key: str | None = None
    timestamp: TimestampValue | None = None
    completion_timestamp: TimestampValue | None = None
    classification: RecordClassificationName | None = None

    @classmethod
    def from_row(
        cls,
        source_table: str,
        source_type: str,
        source_id: str | None,
        raw_json: str | None,
        **raw_columns: object,
    ) -> SourceRecord:
        """Build a record from a live source row, decoding ``raw_json``.

        Malformed JSON keeps the raw text, clears the decoded value, and
        records ``malformed_json``. JSON objects with duplicate keys record
        ``duplicate_json_key`` and decode recursively to the shared typed
        ordered-pairs wrapper while retaining the original raw text. Objects
        whose keys are unique remain ordinary mappings.
        """
        decoded, warnings = _decode_raw_json(raw_json)
        return cls(
            source_table=source_table,
            source_type=source_type,
            source_id=source_id,
            raw_json=raw_json,
            decoded_value=decoded,
            raw_columns=raw_columns,
            warnings=warnings,
        )


def _decode_raw_json(raw_json: str | None) -> tuple[object | None, tuple[str, ...]]:
    if raw_json is None:
        return None, ()
    try:
        decoded = decode_lossless_json(raw_json)
    except ValueError:
        return None, ("malformed_json",)
    warnings: tuple[str, ...] = ("duplicate_json_key",) if decoded.has_duplicate_keys else ()
    return decoded.value, warnings


@dataclass(frozen=True)
class TurnSummary:
    """Identity and outcome summary of one logical turn.

    The summary never stores prompt/response content: only identities,
    counts, timestamps, the terminal outcome, and the turn-local source
    fingerprint. ``source_fingerprint`` here is the turn-local hash of
    the ordered turn records (the ``turn-records-v1`` domain computed by
    :func:`opencode_mlflow.source.turns.iter_turn_summaries`); it is
    deliberately narrower than the full turn source fingerprint from
    :func:`opencode_mlflow.content.fingerprint.source_fingerprint`,
    which additionally covers the session context fingerprint, the
    session-scoped records, and the effective redaction mode through
    :func:`~opencode_mlflow.content.fingerprint.build_effective_turn`.
    Discovery computes the full fingerprint; this turn-local hash only
    identifies the turn's own records.
    """

    session_id: str
    turn_index: int
    turn_key: str
    first_user_source_id: str | None = None
    first_user_source_type: str | None = None
    start_time: TimestampValue | None = None
    end_time: TimestampValue | None = None
    record_count: int = 0
    user_record_count: int = 0
    assistant_record_count: int = 0
    terminal_outcome: str | None = None
    root_result_source_id: str | None = None
    source_fingerprint: str | None = None
    warnings: tuple[str, ...] = ()
    source_outcome: str | None = None


def source_database_fingerprint(schema_generation: SchemaGeneration, path: Path) -> str:
    """Return the non-secret SHA-256 identity fingerprint of a source database.

    The hash covers only the schema generation and available non-secret
    filesystem identity/statistics (file type, device, inode, size, and
    modification time in nanoseconds). It excludes the absolute path and
    never reads source content; it is metadata only, not reconciliation
    identity, so two hard links to one file share a fingerprint while two
    copies of identical content do not.
    """
    info = os.stat(path)
    material = "\0".join(
        (
            _FINGERPRINT_VERSION,
            schema_generation,
            str(stat_module.S_IFMT(info.st_mode)),
            str(info.st_dev),
            str(info.st_ino),
            str(info.st_size),
            str(info.st_mtime_ns),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@runtime_checkable
class SourceAdapter(Protocol):
    """Read access to normalized sessions and records over a source snapshot.

    Implementations read only the normalized replay tables; they never
    reconnect to the live source database and never issue source-schema
    SQL. An implementation may additionally expose the optional
    ``selection_evidence(session_id)`` capability; discovery requires that
    capability only for directory/model-filtered runs.
    """

    schema_generation: SchemaGeneration

    def inspect(self) -> SchemaDescription: ...

    def iter_session_headers(self) -> Iterator[SessionHeader]: ...

    def iter_records(self, session_id: str) -> Iterator[SourceRecord]: ...

    def iter_session_scoped_records(self, session_id: str) -> Iterator[SourceRecord]: ...
