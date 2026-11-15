"""Normalized temporary snapshots extracted from a live read-only source.

The snapshot driver owns source safety end to end: it opens the single
live read-only connection (URI ``mode=ro``, ``query_only``, bounded busy
timeout, exactly one deferred ``BEGIN``), injects that connection into
the exporter through :meth:`SourceExporter.bind_connection`, delegates
the generation-specific extraction SQL to the exporter, and closes the
live connection immediately after extraction. Discovery, capability
checks, and MLflow writes then use only the read-only replay connection
over the temporary copy — never the live database.

The temporary snapshot lives in a mode-``0o700`` directory under the requested
temp root; its files are mode ``0o600``. Complete permitted payloads and headers
live in a sequential ``snapshot.payloads`` spool. SQLite stores summaries,
occurrence/entity/call locators, model identities and byte offsets. Bounded
writer batches can defer finalization while required relatives are frozen
in the same live read transaction after direct content-filter evaluation.

Storage format (internal to the source layer; ``canonical-json-v1``
arrives in Task 4A and does not bind it): each canonical row column holds
a deterministic stdlib-JSON serialization of the
:class:`~mlflow_import.contracts.CanonicalRecordEnvelope` — the
envelope's four fields plus a reserved ``session``/``record`` section
carrying the typed header/record fields needed for lossless replay.
SQLite BLOBs and other non-JSON values are carried through a namespaced
base64 tag, and duplicate-key objects through a namespaced ordered-pairs
tag. ``snapshot_sessions`` also stores only two invocation-specific
pre-redaction selection booleans (directory/model matches); raw selector
values and secret-derived hashes never enter the snapshot. The injected
redaction callback is applied by the writer to every persisted envelope
value and permitted raw field *before* temporary persistence, with only
exact structural envelope fields protected from masking; whatever callback
is injected is the policy applied (the importer wires the mode in Task 12).

Extraction and population failures — including busy-timeout and
disk-space failures — map to :class:`~mlflow_import.errors.SnapshotError`
with code ``source_snapshot_failed`` and exit status ``2``. Path
validation failures map to ``source_unavailable``. Cleanup happens in an
outer ``finally``; cleanup failures are recorded on the snapshot as safe
messages without paths or content.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sqlite3
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from mlflow_import.contracts import PROTECTED_IDENTITY_FIELDS, CanonicalRecordEnvelope, DuplicateKeyObject
from mlflow_import.errors import ImporterError, SnapshotError
from mlflow_import.json_values import json_text_from_decoded
from mlflow_import.retry import check_budget
from mlflow_import.source.connection import open_live_connection
from mlflow_import.source.id_index import DiskIds
from mlflow_import.source.model import (
    SchemaDescription,
    SchemaGeneration,
    SessionHeader,
    SourceRecord,
    TimestampValue,
    source_database_fingerprint,
    source_namespace,
)
from mlflow_import.source.normalize import decode_json_column
from mlflow_import.source.selection import SelectionEvidence, SelectionEvidenceEvaluator
from mlflow_import.source.spool import PayloadSpool

RedactionCallback = Callable[[object, Collection[str]], object]
"""Redaction policy applied to values before they are persisted."""

SNAPSHOT_FILENAME = "snapshot.db"
"""File name of the normalized snapshot inside its temporary directory."""

_SNAPSHOT_SCHEMA = """
CREATE TABLE snapshot_sessions (
    session_id TEXT NOT NULL,
    schema TEXT NOT NULL,
    canonical_session_row TEXT NOT NULL,
    directory_filter_match INTEGER NOT NULL,
    model_filter_match INTEGER NOT NULL,
    creation_ns INTEGER
);
CREATE TABLE snapshot_records (
    session_id TEXT NOT NULL,
    order_key TEXT NOT NULL,
    source_table TEXT NOT NULL,
    source_id TEXT,
    canonical_record TEXT NOT NULL,
    raw_permitted_fields TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    path_ordinal INTEGER,
    anchor_ordinal INTEGER,
    entity_id TEXT,
    invocation_id TEXT,
    call_id TEXT,
    parent_id TEXT,
    occurrence_id TEXT,
    payload_offset INTEGER,
    payload_length INTEGER,
    raw_offset INTEGER,
    raw_length INTEGER,
    occurrence_kind TEXT NOT NULL DEFAULT 'original',
    record_scope TEXT NOT NULL DEFAULT '',
    parent_entity_id TEXT
);
CREATE INDEX snapshot_session_lookup ON snapshot_sessions(session_id);
CREATE TABLE source_identity_headers(id TEXT PRIMARY KEY);
CREATE INDEX snapshot_record_lookup ON snapshot_records(session_id);
CREATE INDEX snapshot_locator_lookup ON snapshot_records(canonical_record);
CREATE INDEX snapshot_effective_lookup ON snapshot_records(session_id,active,path_ordinal);
CREATE INDEX snapshot_entity_lookup ON snapshot_records(session_id,entity_id);
CREATE INDEX snapshot_parent_lookup ON snapshot_records(session_id,source_id);
CREATE INDEX snapshot_call_lookup ON snapshot_records(session_id,call_id);
CREATE INDEX snapshot_invocation_lookup ON snapshot_records(session_id,invocation_id);
CREATE INDEX snapshot_ancestry_lookup ON snapshot_records(session_id,parent_id);
CREATE TABLE model_identities(session_id TEXT,identity TEXT,PRIMARY KEY(session_id,identity));
CREATE TABLE source_summaries(session_id TEXT PRIMARY KEY,
    record_count INTEGER,message_count INTEGER,part_count INTEGER,
    physical_count INTEGER,derived_count INTEGER,effective_count INTEGER,entity_count INTEGER DEFAULT 0);
CREATE TABLE calls(session_id TEXT,call_id TEXT,invocation_id TEXT,locator TEXT,fragment INTEGER,
    PRIMARY KEY(locator,fragment));
CREATE INDEX call_fragments ON calls(session_id,call_id);
CREATE TABLE agent_links(session_id TEXT,agent_id TEXT,occurrence INTEGER,
    PRIMARY KEY(session_id,agent_id,occurrence));
CREATE TABLE entities(session_id TEXT, entity_id TEXT, parent_id TEXT, occurrence INTEGER,
    ordinal INTEGER, visited INTEGER NOT NULL DEFAULT 0, scope TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(session_id,entity_id));
CREATE INDEX entity_parents ON entities(session_id,parent_id);
"""

_BYTES_TAG = "__mlflow_import_bytes_b64__"
_DUPLICATE_KEYS_TAG = "__mlflow_import_duplicate_keys__"
_ESCAPED_MAPPING_TAG = "__mlflow_import_mapping__"

_SNAPSHOT_POPULATE_FAILED = "the normalized source snapshot could not be created or fully populated"
_SNAPSHOT_CREATE_FAILED = "the temporary snapshot could not be created"
_CLEANUP_FAILED = "temporary snapshot cleanup failed"

_SESSION_TIMESTAMP_FIELDS = frozenset(
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


@dataclass(frozen=True)
class ExportStats:
    """Outcome of one snapshot export."""

    session_count: int
    record_count: int
    warning_count: int
    complete: bool
    schema_generation: SchemaGeneration


@runtime_checkable
class SourceExporter(Protocol):
    """Live-source exporter owning the generation-specific extraction SQL.

    The snapshot driver opens the single live read-only connection itself
    — so ``mode=ro``, ``query_only``, the bounded busy timeout, the one
    deferred ``BEGIN``, and prompt closure are enforced in exactly one
    place — and injects it into the exporter through ``bind_connection``
    before ``inspect``/``export_to_snapshot``. Exporters read the live
    source only through the bound connection, select only permitted
    records/fields (including observed additive columns), and attest completion by returning
    ``writer.finalize()`` stats. An exporter is never reused for slow
    discovery/replay.
    """

    schema_generation: SchemaGeneration

    def bind_connection(self, connection: sqlite3.Connection) -> None: ...

    def inspect(self) -> SchemaDescription: ...

    def export_to_snapshot(
        self,
        writer: SnapshotWriter,
        redaction: RedactionCallback,
        session_id: str | None = None,
        session_ids: Collection[str] | None = None,
    ) -> ExportStats: ...


@dataclass
class SourceSnapshot:
    """An open, read-only normalized snapshot ready for discovery/replay.

    ``fingerprint`` is the non-secret source database identity,
    ``snapshot_path`` the normalized temporary snapshot (mode ``0o600``
    inside a mode ``0o700`` directory), ``schema_description`` the probed
    source schema, and ``connection`` the read-only replay connection
    holding one open replay transaction. ``export_warnings`` preserves the
    exporter's safe row-association warning codes; it is deliberately not
    reduced to ``stats.warning_count``. No live source connection remains
    after context setup. ``cleanup_warnings`` collects safe cleanup
    failure messages (never paths or content) as the context tears down.
    """

    fingerprint: str
    snapshot_path: Path
    schema_description: SchemaDescription
    connection: sqlite3.Connection
    stats: ExportStats
    cleanup_warnings: list[str] = field(default_factory=list)
    selection_evidence_enabled: bool = False
    export_warnings: tuple[str, ...] = ()
    turn_index: sqlite3.Connection | None = field(default=None, repr=False)
    source_namespace: str | None = None

    @property
    def warning_codes(self) -> tuple[str, ...]:
        """Safe exporter warning codes preserved without reducing them to a count."""
        return self.export_warnings


MAX_SESSION_HEADER_BYTES = 8 * 1024 * 1024


class SnapshotWriter:
    """Bounded batch writer for the temporary normalized snapshot.

    Its exposed surface is exactly ``write_session``, ``write_record``,
    and ``finalize``. The injected redaction callback runs on every
    persisted envelope value and permitted raw field before temporary
    persistence; typed structural identity fields are written verbatim
    while raw-column projections are redacted as content. ``finalize`` is idempotent and
    flushes any pending batch. Each flush uses one explicit SQLite
    transaction, including when its connection is configured for autocommit.
    In the quota profile private journals are disabled: any failed scratch
    snapshot is discarded, never published or reused as a partial transaction.
    An optional progress callback receives committed session and record counts.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        schema_generation: SchemaGeneration,
        redaction: RedactionCallback,
        write_batch: int = 256,
        selection_evaluator: SelectionEvidenceEvaluator | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        if write_batch < 1:
            raise ValueError("write_batch must be a positive integer")
        self._connection = connection
        self._schema_generation = schema_generation
        self._redaction_policy = redaction
        self._selection_evaluator = selection_evaluator
        self._progress_callback = progress_callback
        self._write_batch = min(write_batch, 256)
        self._session_buffer_bytes = 0
        self._session_rows: list[tuple[str, str, str, int, int, int | None]] = []
        self._record_rows: list[tuple[object, ...]] = []
        self._session_count = 0
        self._record_count = 0
        self._buffer_bytes = 0
        self._committed_session_count = 0
        self._committed_record_count = 0
        self._warning_count = 0
        self._stats: ExportStats | None = None
        self._current_session_id: str | None = None
        self._current_evidence = SelectionEvidence()
        database = connection.execute("PRAGMA database_list").fetchone()[2]
        self._spool = PayloadSpool(Path(database).with_suffix(".payloads")) if database else None
        self._models: list[tuple[str, str]] = []
        self._calls: list[tuple[object, ...]] = []
        self.defer_finalization = False

    def close(self) -> None:
        """Close a partially populated spool on every failure/interrupt path."""
        if self._spool is not None:
            self._spool.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Adapter-owned replay working index, never a live-source connection."""
        return self._connection

    def flush(self) -> None:
        """Make occurrence locators visible to adapter replay."""
        self._flush_sessions()
        self._flush_records()

    def _payload(self, value: object, *, header: bool = False) -> str:
        text = dumps_snapshot_json(value)
        if header and len(text.encode("utf-8")) > MAX_SESSION_HEADER_BYTES:
            from mlflow_import.errors import SourceCompatibilityError

            raise SourceCompatibilityError(
                "source_metadata_too_large", "the merged source header exceeds the qualified metadata envelope"
            )
        self._buffer_bytes += len(text.encode("utf-8"))
        return self._spool.append(text) if self._spool is not None else text

    def write_session(self, header: SessionHeader, canonical_session_row: CanonicalRecordEnvelope) -> None:
        """Write one normalized session row from its header and envelope."""
        self._ensure_open()
        check_budget()
        evidence = (
            self._selection_evaluator.session_evidence(header)
            if self._selection_evaluator is not None
            else SelectionEvidence()
        )
        self._current_session_id = header.session_id
        self._current_evidence = evidence
        self._begin_redaction_record()
        try:
            stored: dict[str, object] = {
                "decoded": self._redaction(canonical_session_row.decoded, ()),
                "raw_text": self._redaction(canonical_session_row.raw_text, ()),
                "raw_columns": self._redaction(canonical_session_row.raw_columns, ()),
                "warnings": list(canonical_session_row.warnings),
                "session": self._redaction(_session_storage_fields(header), PROTECTED_IDENTITY_FIELDS),
            }
        finally:
            self._end_redaction_record()
        self._session_rows.append(
            (
                header.session_id,
                header.adapter_generation,
                self._payload(stored, header=True),
                int(evidence.directory_match),
                int(evidence.model_match),
                header.time_created.nanoseconds if header.time_created else None,
            )
        )
        self._session_count += 1
        self._session_buffer_bytes += sum(
            len(item.encode("utf-8")) for item in self._session_rows[-1] if isinstance(item, str)
        )
        self._warning_count += len(canonical_session_row.warnings) + len(header.warnings)
        if len(self._session_rows) >= self._write_batch or self._session_buffer_bytes >= 4 * 1024 * 1024:
            self._flush_sessions()

    def write_record(
        self,
        record: SourceRecord,
        order_key: tuple[object, ...],
        canonical_record: CanonicalRecordEnvelope,
        raw_permitted_fields: Mapping[str, object],
    ) -> None:
        """Write one normalized record row from its record, order key, and envelope."""
        self._ensure_open()
        check_budget()
        if record.session_id is None:
            raise ValueError("a snapshot record must carry its session_id")
        if (
            self._selection_evaluator is not None
            and not self._current_evidence.model_match
            and self._selection_evaluator.record_model_matches(record)
        ):
            self._mark_model_match(record.session_id)
        self._begin_redaction_record()
        try:
            stored: dict[str, object] = {
                "decoded": self._redaction(canonical_record.decoded, ()),
                "raw_text": self._redaction(canonical_record.raw_text, ()),
                "raw_columns": self._redaction(canonical_record.raw_columns, ()),
                "warnings": list(canonical_record.warnings),
                "record": _record_storage_fields(record),
            }
            redacted_permitted_fields = self._redaction(raw_permitted_fields, ())
        finally:
            self._end_redaction_record()
        canonical_locator = self._payload(stored)
        raw_locator = self._payload(redacted_permitted_fields)
        native_key = record.source_id or record.source_identity
        file_record = record.source_table in {"jsonl", "jsonl_message"}
        entity = (
            record.entity_id
            if file_record
            else record.entity_id or (record.source_table + ":" + native_key if native_key is not None else None)
        )
        parent_entity = (
            record.parent_source_id
            if file_record
            else (
                (record.parent_source_table or "message") + ":" + record.parent_source_id
                if record.parent_source_id is not None
                else None
            )
        )
        invocation = record.invocation_id
        if (
            invocation is None
            and record.classification == "assistant"
            and record.source_table in {"message", "session_message", "part"}
        ):
            invocation = (
                parent_entity if record.derived_child_ordinal is not None or record.source_table == "part" else entity
            )
        self._record_rows.append(
            (
                record.session_id,
                dumps_json_value(list(order_key)),
                record.source_table,
                record.source_id,
                canonical_locator,
                raw_locator,
                entity,
                invocation,
                record.call_id,
                record.parent_source_id,
                record.occurrence_id or record.source_identity,
                *_spool_offsets(canonical_locator),
                *_spool_offsets(raw_locator),
                "effective"
                if "effective_from_occurrence" in record.raw_columns
                else "derived"
                if record.derived_child_ordinal is not None
                else "original",
                record.raw_columns.get("subagent") or "",
                parent_entity,
            )
        )
        self._record_count += 1
        from mlflow_import.source.model_identity import extract_model_identity

        identity = extract_model_identity(stored["decoded"])
        if identity is not None:
            self._models.append((record.session_id, identity))
        first_call: str | None = None
        multiple_calls = False
        for fragment, call in enumerate(_call_ids(record)):
            if first_call is None:
                first_call = call
            elif first_call != call:
                multiple_calls = True
            self._calls.append((record.session_id, call, invocation, self._record_rows[-1][4], fragment))
        if record.call_id is None and first_call is not None and not multiple_calls:
            row = self._record_rows[-1]
            self._record_rows[-1] = (*row[:8], first_call, *row[9:])
        self._buffer_bytes += sum(
            len(value.encode("utf-8")) for value in self._record_rows[-1] if isinstance(value, str)
        )
        self._warning_count += len(canonical_record.warnings)
        if len(self._record_rows) >= self._write_batch or self._buffer_bytes >= 4 * 1024 * 1024:
            self._flush_records()

    def finalize(self) -> ExportStats:
        """Flush pending batches and attest that the export completed."""
        if self._stats is not None:
            return self._stats
        self._flush_sessions()
        self._flush_records()
        if self.defer_finalization:
            return ExportStats(
                self._session_count, self._record_count, self._warning_count, True, self._schema_generation
            )
        self._connection.execute(
            "INSERT OR IGNORE INTO source_identity_headers SELECT session_id FROM snapshot_sessions"
        )
        self._connection.execute("""INSERT INTO source_summaries
            (session_id,record_count,message_count,part_count,physical_count,derived_count,effective_count)
            SELECT session_id,COUNT(*),SUM(source_table!='part'),SUM(source_table='part'),
                SUM(occurrence_kind='original'),SUM(occurrence_kind='derived'),SUM(occurrence_kind='effective')
            FROM snapshot_records GROUP BY session_id""")
        self._connection.execute("""INSERT OR IGNORE INTO entities
            (session_id,entity_id,parent_id,occurrence,ordinal,visited)
            SELECT session_id,source_table || ':' || COALESCE(source_id,occurrence_id,CAST(rowid AS TEXT)),
                parent_entity_id,rowid,rowid,0 FROM snapshot_records
            WHERE source_table NOT IN ('jsonl','jsonl_message')""")
        self._connection.execute("""UPDATE source_summaries SET entity_count=
            (SELECT COUNT(*) FROM entities WHERE entities.session_id=source_summaries.session_id)""")
        if self._spool is not None:
            self._spool.close()
        self._stats = ExportStats(
            session_count=self._session_count,
            record_count=self._record_count,
            warning_count=self._warning_count,
            complete=True,
            schema_generation=self._schema_generation,
        )
        return self._stats

    def _redaction(self, value: object, protected: Collection[str]) -> object:
        """Sanitize JSON text structurally as well as its decoded projection.

        Duplicate/malformed text must retain its structure and is scanned in
        place. Valid JSON is reserialized only when its decoded value changed,
        preserving byte fidelity in verbatim mode.
        """
        if isinstance(value, DuplicateKeyObject):
            # This is a lossless Mapping-compatible wrapper; send it intact
            # so the policy sees every ordered pair exactly once.
            return self._redaction_policy(value, protected)
        if isinstance(value, Mapping):
            # Apply the policy to one root field at a time. That preserves
            # field-name precedence while preventing a nested mapping that
            # was prepared by a child call from being traversed again by an
            # ancestor call.
            return {key: self._redact_mapping_field(key, item, protected) for key, item in value.items()}
        if isinstance(value, (list, tuple, DuplicateKeyObject)):
            prepared = self._prepare_embedded_json(value)
            return self._redaction_policy(prepared, protected)
        if isinstance(value, str):
            parsed = decode_json_column(value)
            if "malformed_json" not in parsed.warnings and "malformed_source_value" not in parsed.warnings:
                sanitized = self._redaction(parsed.decoded, protected)
                if sanitized != parsed.decoded:
                    return json_text_from_decoded(sanitized)
        return self._redaction_policy(value, protected)

    def _redact_mapping_field(self, key: object, value: object, protected: Collection[str]) -> object:
        """Redact one mapping field without re-walking sibling/ancestor values."""
        if isinstance(key, str) and key not in protected and isinstance(value, str):
            parsed = decode_json_column(value)
            if "malformed_json" not in parsed.warnings and "malformed_source_value" not in parsed.warnings:
                prepared = self._prepare_embedded_json(parsed.decoded)
                sanitized = self._redaction_policy({key: prepared}, protected)
                if isinstance(sanitized, Mapping) and key in sanitized:
                    redacted = sanitized[key]
                    if redacted != prepared or prepared != parsed.decoded:
                        return json_text_from_decoded(redacted)
                    return value
        sanitized = self._redaction_policy({key: value}, protected)
        if isinstance(sanitized, Mapping) and key in sanitized:
            return sanitized[key]
        return sanitized

    def _prepare_embedded_json(self, value: object) -> object:
        """Redact nested JSON strings before their containing tree is visited."""
        if isinstance(value, DuplicateKeyObject):
            return DuplicateKeyObject(
                pairs=tuple((key, self._prepare_embedded_json(item)) for key, item in value.pairs),
                raw_text=value.raw_text,
            )
        if isinstance(value, Mapping):
            return {key: self._prepare_embedded_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._prepare_embedded_json(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._prepare_embedded_json(item) for item in value)
        if not isinstance(value, str):
            return value
        parsed = decode_json_column(value)
        if "malformed_json" in parsed.warnings or "malformed_source_value" in parsed.warnings:
            return value
        prepared = self._prepare_embedded_json(parsed.decoded)
        sanitized = self._redaction(prepared, ())
        if sanitized != prepared:
            return json_text_from_decoded(sanitized)
        return value

    def _begin_redaction_record(self) -> None:
        begin = getattr(self._redaction_policy, "begin_source_record", None)
        if callable(begin):
            begin()

    def _end_redaction_record(self) -> None:
        end = getattr(self._redaction_policy, "end_source_record", None)
        if callable(end):
            end()

    def _ensure_open(self) -> None:
        if self._stats is not None:
            raise sqlite3.ProgrammingError("the snapshot writer was already finalized")

    def _mark_model_match(self, session_id: str) -> None:
        """Persist a pre-redaction model match without retaining selector data."""
        if self._current_session_id == session_id:
            self._current_evidence = replace(self._current_evidence, model_match=True)
        for index, row in enumerate(self._session_rows):
            if row[0] == session_id:
                self._session_rows[index] = (*row[:4], 1, row[5])
                return
        self._connection.execute(
            "UPDATE snapshot_sessions SET model_filter_match = 1 WHERE session_id = ?",
            (session_id,),
        )

    def _flush_sessions(self) -> None:
        if not self._session_rows:
            return
        self._execute_batch(
            "INSERT INTO snapshot_sessions"
            " (session_id, schema, canonical_session_row, directory_filter_match, model_filter_match,creation_ns)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            self._session_rows,
        )
        self._committed_session_count += len(self._session_rows)
        self._session_rows.clear()
        self._session_buffer_bytes = 0
        self._report_progress()

    def _flush_records(self) -> None:
        if not self._record_rows:
            return
        self._execute_batch(
            "INSERT INTO snapshot_records"
            " (session_id, order_key, source_table, source_id, canonical_record, raw_permitted_fields,"
            " entity_id,invocation_id,call_id,parent_id,occurrence_id,"
            " payload_offset,payload_length,raw_offset,raw_length,occurrence_kind,record_scope,parent_entity_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self._record_rows,
        )
        self._committed_record_count += len(self._record_rows)
        if self._models:
            self._execute_batch("INSERT OR IGNORE INTO model_identities VALUES(?,?)", self._models)
            self._models.clear()
        if self._calls:
            self._execute_batch("INSERT INTO calls VALUES(?,?,?,?,?)", self._calls)
            self._calls.clear()
        self._record_rows.clear()
        self._buffer_bytes = 0
        self._report_progress()

    def _execute_batch(self, statement: str, rows: Iterable[Sequence[object]]) -> None:
        """Commit one buffered batch atomically, even on autocommit connections."""
        transaction_started = False
        try:
            self._connection.execute("BEGIN")
            transaction_started = True
            self._connection.executemany(statement, rows)
            self._connection.execute("COMMIT")
            transaction_started = False
        except BaseException:
            if transaction_started and self._connection.in_transaction:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.Error:
                    # Preserve the original batch failure; the temporary
                    # snapshot is discarded by its owning context manager.
                    pass
            raise

    def _report_progress(self) -> None:
        """Report committed aggregate counts without making diagnostics fatal."""
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(self._committed_session_count, self._committed_record_count)
        except Exception:
            # Progress output is observational; a broken stderr must not
            # abort snapshot creation after a batch has already committed.
            self._progress_callback = None


@contextmanager
def create_normalized_snapshot(
    path: Path,
    temp_root: Path,
    exporter: SourceExporter,
    redaction: RedactionCallback,
    busy_timeout_ms: int = 5000,
    write_batch: int = 256,
    selection_evaluator: SelectionEvidenceEvaluator | None = None,
    session_id: str | None = None,
    session_ids: Collection[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Iterator[SourceSnapshot]:
    """Extract a normalized read-only snapshot of the source at ``path``.

    The live source is opened through :func:`open_live_connection`
    (validating the path, applying the busy timeout, and issuing the one
    deferred ``BEGIN``), the exporter is bound to that connection and
    performs the catalogued extraction into a fresh temporary snapshot,
    and the live connection is closed immediately after extraction. The
    yielded :class:`SourceSnapshot` keeps only the read-only replay
    connection open; its temporary files are removed when the context
    exits, and cleanup failures are recorded safely on
    ``snapshot.cleanup_warnings``.

    Extraction/population failures (including busy-timeout and disk
    failures) raise :class:`SnapshotError` with code
    ``source_snapshot_failed`` and exit status ``2``; an unusable source
    path raises with code ``source_unavailable``.
    """
    try:
        from mlflow_import import resources

        temp_dir = resources.make_temp_dir(prefix="mlflow-import-snapshot-", dir=temp_root)
    except OSError as error:
        raise SnapshotError("source_snapshot_failed", _SNAPSHOT_CREATE_FAILED) from error
    snapshot_path = temp_dir / SNAPSHOT_FILENAME
    cleanup_warnings: list[str] = []
    snapshot: SourceSnapshot | None = None
    provisional: SourceSnapshot | None = None
    returned: ExportStats | None = None
    stats: ExportStats | None = None
    snapshot_error: SnapshotError | None = None
    importer_error: ImporterError | None = None
    selection_warnings: set[str] = set()
    try:
        try:
            with open_live_connection(path, busy_timeout_ms) as live:
                fingerprint = source_database_fingerprint(exporter.schema_generation, path)
                exporter.bind_connection(live.connection)
                description = exporter.inspect()
                # Enumeration and extraction must observe the same committed
                # WAL view. The earlier probe is advisory, never the selector.
                if selection_evaluator is not None:
                    from mlflow_import.source.candidates import candidate_session_ids

                    content_filters = (
                        selection_evaluator.filters.model is not None
                        or selection_evaluator.filters.min_user_turns is not None
                    )
                    session_ids = candidate_session_ids(
                        live.connection,
                        description,
                        selection_evaluator.filters,
                        session_id,
                        include_relatives=not content_filters,
                        warnings=selection_warnings,
                    )
                writer_connection = resources.sqlite_connect(snapshot_path, isolation_level=None)
                try:
                    writer_connection.executescript(_SNAPSHOT_SCHEMA)
                    writer_connection.execute("PRAGMA cache_size=-2048")
                    writer_connection.execute("PRAGMA max_page_count=2097152")
                    os.chmod(snapshot_path, 0o600)
                    writer = SnapshotWriter(
                        writer_connection,
                        exporter.schema_generation,
                        redaction,
                        write_batch,
                        selection_evaluator,
                        progress_callback,
                    )
                    writer.defer_finalization = (
                        selection_evaluator is not None
                        and (
                            selection_evaluator.filters.model is not None
                            or selection_evaluator.filters.min_user_turns is not None
                        )
                        and session_ids is not None
                    )
                    try:
                        if shutil.disk_usage(temp_dir).free < 64 * 1024 * 1024:
                            raise SnapshotError("temporary_storage_exhausted", "insufficient temporary storage")
                        if session_id is None and session_ids is None:
                            returned = exporter.export_to_snapshot(writer, redaction)
                        else:
                            returned = exporter.export_to_snapshot(writer, redaction, session_id, session_ids)
                        if writer.defer_finalization:
                            from mlflow_import.source.planning import content_matches

                            assert selection_evaluator is not None and session_ids is not None
                            provisional = SourceSnapshot(
                                fingerprint=fingerprint,
                                snapshot_path=snapshot_path,
                                schema_description=description,
                                connection=writer_connection,
                                stats=returned,
                                selection_evidence_enabled=True,
                            )
                            matching = DiskIds(content_matches(provisional, selection_evaluator.filters))
                            required = candidate_session_ids(
                                live.connection, description, selection_evaluator.filters, seed_ids=matching
                            )
                            additional = DiskIds(native for native in (required or ()) if native not in session_ids)
                            if additional:
                                returned = exporter.export_to_snapshot(writer, redaction, session_ids=additional)
                            writer.defer_finalization = False
                    finally:
                        try:
                            import sys

                            active = sys.exception()
                            if active is None or isinstance(active, Exception):
                                stats = writer.finalize()
                        finally:
                            writer.close()
                    if isinstance(session_ids, DiskIds):
                        writer_connection.executemany(
                            "INSERT OR IGNORE INTO source_identity_headers VALUES(?)",
                            session_ids.db.execute("SELECT id FROM headers"),
                        )
                finally:
                    writer_connection.close()
            if returned is None or not returned.complete or stats is None:
                raise SnapshotError("source_snapshot_failed", _SNAPSHOT_POPULATE_FAILED)
            stats = _merge_exporter_warnings(stats, returned)
            replay_connection = _open_replay_connection(snapshot_path)
            snapshot = SourceSnapshot(
                fingerprint=fingerprint,
                source_namespace=source_namespace(path),
                snapshot_path=snapshot_path,
                schema_description=description,
                connection=replay_connection,
                stats=stats,
                export_warnings=(*_export_warning_codes(exporter), *sorted(selection_warnings)),
                selection_evidence_enabled=selection_evaluator is not None,
                cleanup_warnings=cleanup_warnings,
                turn_index=provisional.turn_index if provisional else None,
            )
        except SnapshotError as error:
            snapshot_error = error
            raise
        except ImporterError as error:
            importer_error = error
            raise
        except OSError as error:
            snapshot_error = SnapshotError("source_snapshot_failed", _SNAPSHOT_CREATE_FAILED)
            raise snapshot_error from error
        except Exception as error:
            snapshot_error = SnapshotError("source_snapshot_failed", _SNAPSHOT_POPULATE_FAILED)
            raise snapshot_error from error
        yield snapshot
    finally:
        if snapshot is None and provisional is not None and provisional.turn_index is not None:
            provisional.turn_index.close()
        _cleanup_snapshot(snapshot, snapshot_path, cleanup_warnings)
        failure = snapshot_error or importer_error
        if failure is not None:
            failure.attach_cleanup_warnings(tuple(cleanup_warnings))


def _open_replay_connection(snapshot_path: Path) -> sqlite3.Connection:
    """Open the temporary snapshot read-only in one deferred replay transaction."""
    uri = Path(os.path.abspath(snapshot_path)).as_uri() + "?mode=ro"
    from mlflow_import.resources import reader_connect

    connection = reader_connect(uri, uri=True, isolation_level=None)
    from mlflow_import.resources import configure_reader

    configure_reader(connection)
    connection.execute("PRAGMA query_only = ON")
    connection.execute("BEGIN")
    return connection


def _merge_exporter_warnings(stats: ExportStats, returned: ExportStats) -> ExportStats:
    """Fold the exporter's returned warning count into the kept stats.

    Exporters report warnings for rows that are never written —
    unassociated session-scoped records, skipped sessions, duplicate
    session IDs — by returning the writer's stats with those warnings
    added on top. The driver keeps the writer's own ``finalize()`` stats
    (both calls are idempotent and share one cached result), so the
    returned count is merged in exactly once: written-row warnings stay
    counted by the writer, exporter-level warnings by the return value.
    An exporter returning no extra warnings leaves the stats unchanged.
    """
    if returned.warning_count <= stats.warning_count:
        return stats
    return replace(stats, warning_count=returned.warning_count)


_SAFE_EXPORT_WARNING_CODE = re.compile(r"^[a-z0-9_]+$")


def _export_warning_codes(exporter: SourceExporter) -> tuple[str, ...]:
    """Read only exporter-supplied safe codes after extraction completes.

    Warning text is never accepted from an exporter: only the documented
    lower-case code vocabulary can cross into the snapshot/report surface.
    Repeated row warnings remain repeated events so callers do not lose
    row-level cardinality by retaining only ``warning_count``.
    """
    values = getattr(exporter, "export_warnings", ())
    if not isinstance(values, (tuple, list)):
        return ()
    return tuple(value for value in values if isinstance(value, str) and _SAFE_EXPORT_WARNING_CODE.fullmatch(value))


def _cleanup_snapshot(
    snapshot: SourceSnapshot | None,
    snapshot_path: Path,
    cleanup_warnings: list[str],
) -> None:
    """Close the replay connection and delete snapshot files and directory.

    Cleanup failures are recorded as one safe message without paths or
    content; cleanup itself never raises.
    """
    if snapshot is not None:
        if snapshot.turn_index is not None:
            snapshot.turn_index.close()
            snapshot.turn_index = None
        try:
            snapshot.connection.close()
        except sqlite3.Error:
            _record_cleanup_failure(cleanup_warnings)
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(snapshot_path) + suffix)
        try:
            from mlflow_import.resources import remove_file

            remove_file(candidate)
        except OSError:
            _record_cleanup_failure(cleanup_warnings)
    for name in ("turn-index.db", "turn-index.db-journal", "snapshot.payloads"):
        try:
            remove_file(snapshot_path.parent / name)
        except OSError:
            _record_cleanup_failure(cleanup_warnings)
    try:
        snapshot_path.parent.rmdir()
        from mlflow_import.resources import current_storage

        if budget := current_storage():
            budget.forget(snapshot_path.parent)
    except OSError:
        _record_cleanup_failure(cleanup_warnings)


def _record_cleanup_failure(cleanup_warnings: list[str]) -> None:
    if _CLEANUP_FAILED not in cleanup_warnings:
        cleanup_warnings.append(_CLEANUP_FAILED)


def _session_storage_fields(header: SessionHeader) -> dict[str, object]:
    """Serialize a header's typed fields (excluding its raw column map)."""
    stored: dict[str, object] = {}
    for item in fields(SessionHeader):
        if item.name == "raw_columns":
            continue
        value = getattr(header, item.name)
        if item.name in _SESSION_TIMESTAMP_FIELDS:
            stored[item.name] = _timestamp_storage_value(value)
        else:
            stored[item.name] = value
    return stored


def _record_storage_fields(record: SourceRecord) -> dict[str, object]:
    """Serialize a record's structural identity and ordering fields."""
    return {
        "source_table": record.source_table,
        "source_type": record.source_type,
        "source_id": record.source_id,
        "session_id": record.session_id,
        "sequence_domain": record.sequence_domain,
        "sequence_value": record.sequence_value,
        "parent_source_id": record.parent_source_id,
        "parent_source_table": record.parent_source_table,
        "derived_child_ordinal": record.derived_child_ordinal,
        "source_identity": record.source_identity,
        "join_projection_key": record.join_projection_key,
        "timestamp": _timestamp_storage_value(record.timestamp),
        "completion_timestamp": _timestamp_storage_value(record.completion_timestamp),
        "classification": record.classification,
        "occurrence_id": record.occurrence_id,
        "entity_id": record.entity_id,
        "invocation_id": record.invocation_id,
        "call_id": record.call_id,
    }


def _timestamp_storage_value(value: object) -> object:
    if value is None:
        return None
    timestamp = cast(TimestampValue, value)
    return {"value": timestamp.value, "unit": timestamp.unit}


def _call_ids(record: SourceRecord) -> Iterator[str]:
    """Index attributable call fragments without creating a second payload copy."""
    if record.call_id is not None:
        yield record.call_id
    value = record.decoded_value
    if not isinstance(value, Mapping):
        return
    for key in ("callID", "call_id", "toolCallId", "tool_use_id"):
        call = value.get(key)
        if isinstance(call, str) and call != record.call_id:
            yield call
    for field_name in ("content", "toolCalls"):
        items = value.get(field_name)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            kind = item.get("type")
            if field_name == "toolCalls" or kind in {"tool_use", "tool_call", "tool_result", "toolCall"}:
                call = item.get("tool_use_id", item.get("id"))
                if isinstance(call, str):
                    yield call


def _spool_offsets(locator: str) -> tuple[int | None, int | None]:
    from mlflow_import.source.spool import PREFIX

    if locator.startswith(PREFIX):
        offset, length = json.loads(locator[len(PREFIX) :])
        return int(offset), int(length)
    return None, None


def dumps_snapshot_json(value: object) -> str:
    """Serialize a value deterministically for snapshot storage.

    Keys are sorted, separators are fixed, output is ASCII-safe, and
    SQLite BLOBs are carried through a namespaced base64 tag. This is the
    storage format of the temporary snapshot only; the
    ``canonical-json-v1`` fingerprint format (Task 4A) is separate.
    """
    return json.dumps(_encode_snapshot_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def dumps_json_value(value: object) -> str:
    """Serialize a plain (already storage-safe) value deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def loads_snapshot_json(text: str) -> object:
    """Deserialize snapshot storage, restoring tagged BLOB values."""
    return _decode_snapshot_value(json.loads(text))


def _encode_snapshot_value(value: object) -> object:
    if isinstance(value, bytes):
        return {_BYTES_TAG: base64.b64encode(value).decode("ascii")}
    if isinstance(value, DuplicateKeyObject):
        return {
            _DUPLICATE_KEYS_TAG: {
                "pairs": [[key, _encode_snapshot_value(item)] for key, item in value.pairs],
                "raw_text": value.raw_text,
            }
        }
    if isinstance(value, Mapping):
        encoded = {str(key): _encode_snapshot_value(item) for key, item in value.items()}
        if _needs_escaped_mapping(value):
            return {_ESCAPED_MAPPING_TAG: [[str(key), _encode_snapshot_value(item)] for key, item in value.items()]}
        return encoded
    if isinstance(value, (list, tuple)):
        return [_encode_snapshot_value(item) for item in value]
    return value


def _decode_snapshot_value(value: object) -> object:
    if isinstance(value, dict):
        if len(value) == 1 and _BYTES_TAG in value:
            payload = value[_BYTES_TAG]
            if isinstance(payload, str):
                try:
                    return base64.b64decode(payload, validate=True)
                except ValueError:
                    pass
        if len(value) == 1 and _DUPLICATE_KEYS_TAG in value:
            payload = value[_DUPLICATE_KEYS_TAG]
            if _is_duplicate_snapshot_payload(payload):
                assert isinstance(payload, Mapping)
                pairs = cast(list[list[object]], payload["pairs"])
                return DuplicateKeyObject(
                    pairs=tuple((cast(str, pair[0]), _decode_snapshot_value(pair[1])) for pair in pairs),
                    raw_text=cast(str, payload["raw_text"]),
                )
        if len(value) == 1 and _ESCAPED_MAPPING_TAG in value:
            payload = value[_ESCAPED_MAPPING_TAG]
            if _is_mapping_entries(payload):
                assert isinstance(payload, list)
                return {cast(str, pair[0]): _decode_snapshot_value(pair[1]) for pair in payload}
        return {key: _decode_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_snapshot_value(item) for item in value]
    return value


def _needs_escaped_mapping(value: Mapping[object, object]) -> bool:
    """Escape ordinary singleton mappings that resemble internal tags."""
    if len(value) != 1:
        return False
    key = next(iter(value))
    return key in {_BYTES_TAG, _DUPLICATE_KEYS_TAG, _ESCAPED_MAPPING_TAG}


def _is_mapping_entries(value: object) -> bool:
    """Whether a value has the exact list-of-key/value-pairs shape."""
    if not isinstance(value, list):
        return False
    return all(isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str) for pair in value)


def _is_duplicate_snapshot_payload(value: object) -> bool:
    """Recognize only the internal duplicate wrapper payload shape."""
    if not isinstance(value, Mapping) or set(value) != {"pairs", "raw_text"}:
        return False
    return isinstance(value["raw_text"], str) and _is_mapping_entries(value["pairs"])
