"""Replay over the normalized snapshot.

This module owns the :class:`ReplayTurn` protocol — the streaming replay
view of one logical turn — exclusively; no other module defines or
re-exports a second ``ReplayTurn``. Task 5's ``build_replay_turn``
returns a concrete implementation of this protocol with fresh
normalized-snapshot cursors and bounded root-value loaders, so no module
creates an in-memory turn/list as a cross-layer substitute.

:class:`NormalizedSnapshotAdapter` implements :class:`SourceAdapter`
over ``snapshot_sessions``/``snapshot_records``: it reconstructs headers
and records from the persisted canonical rows, yields them in the
adapter-provided written order with fresh cursors per call, and never
issues source-schema SQL or reconnects to the live database. Replayed
records carry the values exactly as persisted (post-redaction), so
discovery and replay observe the same effective values that feed
fingerprints and artifacts.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Any, Protocol, cast, runtime_checkable

from opencode_mlflow.source.catalog import OPENCODE_SCHEMA_CATALOG_V1
from opencode_mlflow.source.model import (
    RecordClassificationName,
    SchemaDescription,
    SchemaGeneration,
    SessionHeader,
    SourceRecord,
    TimestampValue,
    TurnSummary,
)
from opencode_mlflow.source.selection import SelectionEvidence
from opencode_mlflow.source.snapshot import (
    SourceSnapshot,
    loads_snapshot_json,
)


@runtime_checkable
class ReplayTurn(Protocol):
    """Streaming replay view of one logical turn.

    ``iter_records``/``iter_session_records``/``iter_unassigned_records``
    return fresh cursors over the normalized snapshot on every call.
    ``load_root_input``/``load_root_output`` load the bounded root values
    for the turn. Implementations never retain the turn's records as an
    in-memory list.
    """

    @property
    def session_header(self) -> SessionHeader: ...

    @property
    def summary(self) -> TurnSummary: ...

    def iter_records(self) -> Iterator[SourceRecord]: ...

    def iter_session_records(self) -> Iterator[SourceRecord]: ...

    def iter_unassigned_records(self) -> Iterator[SourceRecord]: ...

    def load_root_input(self, max_bytes: int) -> object | None: ...

    def load_root_output(self, max_bytes: int) -> object | None: ...


class NormalizedSnapshotAdapter:
    """:class:`SourceAdapter` over the normalized snapshot tables.

    Sessions and records replay in the order the exporter wrote them
    (row order), with fresh cursors per call. Session-scoped records are
    separated from turn records using the catalogued session-scoped table
    names for the snapshot's schema generation (the union for mixed
    snapshots).
    """

    def __init__(self, snapshot: SourceSnapshot) -> None:
        self._snapshot = snapshot
        self._scoped_tables = _session_scoped_tables(snapshot.schema_description.generation)
        # A plain attribute (not a property): the SourceAdapter protocol
        # declares a settable `schema_generation` member.
        self.schema_generation = snapshot.schema_description.generation

    def inspect(self) -> SchemaDescription:
        return self._snapshot.schema_description

    @property
    def snapshot(self) -> SourceSnapshot:
        """The normalized snapshot this adapter replays.

        Discovery exposes the replay adapter through the
        :class:`SourceAdapter` protocol but still needs the snapshot to
        build each turn through
        :func:`opencode_mlflow.source.turns.build_replay_turn`; this
        read-only view is the seam for that, never a second connection.
        """
        return self._snapshot

    def iter_session_headers(self) -> Iterator[SessionHeader]:
        cursor = self._snapshot.connection.execute("SELECT canonical_session_row FROM snapshot_sessions ORDER BY rowid")
        for (canonical_session_row,) in cursor:
            yield self._header_with_source_fingerprint(_restore_session_header(cast(str, canonical_session_row)))

    def session_header(self, session_id: str) -> SessionHeader | None:
        """Load only the requested header; do not retain every session payload."""
        row = self._snapshot.connection.execute(
            "SELECT canonical_session_row FROM snapshot_sessions WHERE session_id = ? LIMIT 1", (session_id,)
        ).fetchone()
        return self._header_with_source_fingerprint(_restore_session_header(row[0])) if row is not None else None

    def _header_with_source_fingerprint(self, header: SessionHeader) -> SessionHeader:
        """Attach the snapshot's non-secret database identity at replay time."""
        if header.source_fingerprint is not None:
            return header
        return replace(header, source_fingerprint=self._snapshot.fingerprint)

    def selection_evidence(self, session_id: str) -> SelectionEvidence | None:
        """Load invocation-specific pre-redaction match booleans only."""
        if not self._snapshot.selection_evidence_enabled:
            return None
        row = self._snapshot.connection.execute(
            "SELECT directory_filter_match, model_filter_match FROM snapshot_sessions WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            raise ValueError("selection evidence is missing for a snapshot session")
        return SelectionEvidence(directory_match=bool(row[0]), model_match=bool(row[1]))

    def iter_records(self, session_id: str) -> Iterator[SourceRecord]:
        placeholders = _placeholders(self._scoped_tables)
        sql = (
            "SELECT session_id, source_table, source_id, canonical_record, raw_permitted_fields"
            f" FROM snapshot_records WHERE session_id = ? AND source_table NOT IN ({placeholders})"
            " ORDER BY rowid"
        )
        yield from self._iter_stored_records(sql, (session_id, *sorted(self._scoped_tables)))

    def iter_session_scoped_records(self, session_id: str) -> Iterator[SourceRecord]:
        placeholders = _placeholders(self._scoped_tables)
        sql = (
            "SELECT session_id, source_table, source_id, canonical_record, raw_permitted_fields"
            f" FROM snapshot_records WHERE session_id = ? AND source_table IN ({placeholders})"
            " ORDER BY rowid"
        )
        yield from self._iter_stored_records(sql, (session_id, *sorted(self._scoped_tables)))

    def _iter_stored_records(self, sql: str, parameters: tuple[str, ...]) -> Iterator[SourceRecord]:
        cursor = self._snapshot.connection.execute(sql, parameters)
        for session_id, source_table, source_id, canonical_record, raw_permitted_fields in cursor:
            yield _restore_source_record(
                session_id=session_id,
                source_table=cast(str, source_table),
                source_id=cast("str | None", source_id),
                canonical_record=cast(str, canonical_record),
                raw_permitted_fields=cast(str, raw_permitted_fields),
            )


def _session_scoped_tables(generation: SchemaGeneration) -> frozenset[str]:
    """Catalogued session-scoped table names for a schema generation."""
    names: set[str] = set()
    for catalog in OPENCODE_SCHEMA_CATALOG_V1:
        if generation == "mixed" or catalog.generation == generation:
            names.update(catalog.session_scoped_tables)
    return frozenset(names)


def _placeholders(tables: frozenset[str]) -> str:
    return ",".join("?" for _ in tables)


def _restore_session_header(canonical_session_row: str) -> SessionHeader:
    """Reconstruct a :class:`SessionHeader` from its persisted canonical row."""
    stored = _stored_mapping(canonical_session_row)
    session_fields = _section(stored, "session")
    raw_columns = _as_mapping(stored.get("raw_columns"))
    kwargs: dict[str, Any] = {}
    for item in dataclass_fields(SessionHeader):
        value = session_fields.get(item.name)
        if item.name == "warnings":
            kwargs["warnings"] = _as_warnings(value)
        elif item.name == "raw_columns":
            kwargs["raw_columns"] = raw_columns
        elif item.name == "catalog_version":
            kwargs["catalog_version"] = value if isinstance(value, str) else ""
        elif item.name in _HEADER_TIMESTAMP_FIELDS:
            kwargs[item.name] = _restore_timestamp_value(value)
        else:
            kwargs[item.name] = value
    return SessionHeader(**kwargs)


def _restore_source_record(
    session_id: str,
    source_table: str,
    source_id: str | None,
    canonical_record: str,
    raw_permitted_fields: str,
) -> SourceRecord:
    """Reconstruct a :class:`SourceRecord` from its persisted canonical row."""
    stored = _stored_mapping(canonical_record)
    record_fields = _section(stored, "record")
    permitted = _as_mapping(loads_snapshot_json(raw_permitted_fields))
    raw_columns = _as_mapping(stored.get("raw_columns"))
    # The replayed raw column map is the union of the exporter's permitted
    # raw fields and the envelope's canonical raw columns (the envelope
    # wins on collision), so permitted raw row values survive the round
    # trip.
    merged: dict[str, object] = dict(permitted)
    merged.update(raw_columns)
    classification = record_fields.get("classification")
    record = SourceRecord(
        source_table=source_table,
        source_type=_required_str(record_fields, "source_type"),
        source_id=source_id,
        raw_json=_optional_str(stored.get("raw_text")),
        decoded_value=stored.get("decoded"),
        raw_columns=merged,
        warnings=_as_warnings(stored.get("warnings")),
        session_id=_optional_str(record_fields.get("session_id")),
        sequence_domain=_optional_str(record_fields.get("sequence_domain")),
        sequence_value=record_fields.get("sequence_value"),
        parent_source_id=_optional_str(record_fields.get("parent_source_id")),
        parent_source_table=_optional_str(record_fields.get("parent_source_table")),
        derived_child_ordinal=_optional_int(record_fields.get("derived_child_ordinal")),
        source_identity=_optional_str(record_fields.get("source_identity")),
        join_projection_key=_optional_str(record_fields.get("join_projection_key")),
        timestamp=_restore_timestamp_value(record_fields.get("timestamp")),
        completion_timestamp=_restore_timestamp_value(record_fields.get("completion_timestamp")),
        classification=cast("RecordClassificationName | None", classification),
    )
    # The row's session_id column is authoritative and NOT NULL; cross-check
    # the stored record section against it so a corrupted row fails loudly
    # rather than silently reassigning a record to another session.
    if record.session_id != session_id:
        raise ValueError("a stored record's session identity does not match its row")
    return record


def _stored_mapping(text: str) -> dict[str, object]:
    loaded = loads_snapshot_json(text)
    if not isinstance(loaded, dict):
        raise ValueError("a stored canonical row must decode to an object")
    return loaded


def _section(stored: Mapping[str, object], name: str) -> dict[str, object]:
    value = stored.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"a stored canonical row is missing its {name} section")
    return value


def _as_mapping(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("a stored raw column value must decode to an object")
    return value


def _as_warnings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("stored warnings must decode to a list")
    return tuple(str(item) for item in value)


def _required_str(fields: Mapping[str, object], name: str) -> str:
    value = fields.get(name)
    if not isinstance(value, str):
        raise ValueError(f"a stored record is missing its {name}")
    return value


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _restore_timestamp_value(value: object) -> TimestampValue | None:
    if value is None:
        return None
    if isinstance(value, dict) and "value" in value and "unit" in value:
        unit = value["unit"]
        if unit in ("s", "ms", "us", "ns"):
            return TimestampValue(value=cast("int | float", value["value"]), unit=unit)
    raise ValueError("a stored timestamp must decode to a value/unit pair")


_HEADER_TIMESTAMP_FIELDS = frozenset(
    {
        "fork_boundary",
        "time_created",
        "time_updated",
        "time_compacting",
        "time_archived",
        "time_idle",
        "time_viewed",
        "time_suspended",
    }
)
