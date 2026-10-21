"""Normalized temporary snapshots extracted from a live read-only source.

The snapshot driver owns source safety end to end: it opens the single
live read-only connection (URI ``mode=ro``, ``query_only``, bounded busy
timeout, exactly one deferred ``BEGIN``), injects that connection into
the exporter through :meth:`SourceExporter.bind_connection`, delegates
the generation-specific extraction SQL to the exporter, and closes the
live connection immediately after extraction. Discovery, capability
checks, and MLflow writes then use only the read-only replay connection
over the temporary copy — never the live database.

The temporary snapshot lives in a mode-``0o700`` directory under the
requested temp root, its file is mode ``0o600``, and it contains only the
``snapshot_sessions`` and ``snapshot_records`` tables. Rows are written
in bounded batches through :class:`SnapshotWriter`, whose surface is
exactly ``write_session``/``write_record``/``finalize``.

Storage format (internal to the source layer; ``canonical-json-v1``
arrives in Task 4A and does not bind it): each canonical row column holds
a deterministic stdlib-JSON serialization of the
:class:`~opencode_mlflow.contracts.CanonicalRecordEnvelope` — the
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
disk-space failures — map to :class:`~opencode_mlflow.errors.SnapshotError`
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
import sqlite3
import tempfile
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from opencode_mlflow.contracts import PROTECTED_IDENTITY_FIELDS, CanonicalRecordEnvelope, DuplicateKeyObject
from opencode_mlflow.errors import ImporterError, SnapshotError
from opencode_mlflow.json_values import json_text_from_decoded
from opencode_mlflow.source.connection import open_live_connection
from opencode_mlflow.source.model import (
    SchemaDescription,
    SchemaGeneration,
    SessionHeader,
    SourceRecord,
    TimestampValue,
    source_database_fingerprint,
)
from opencode_mlflow.source.normalize import decode_json_column
from opencode_mlflow.source.selection import SelectionEvidence, SelectionEvidenceEvaluator

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
    model_filter_match INTEGER NOT NULL
);
CREATE TABLE snapshot_records (
    session_id TEXT NOT NULL,
    order_key TEXT NOT NULL,
    source_table TEXT NOT NULL,
    source_id TEXT,
    canonical_record TEXT NOT NULL,
    raw_permitted_fields TEXT NOT NULL
);
"""

_BYTES_TAG = "__opencode_mlflow_bytes_b64__"
_DUPLICATE_KEYS_TAG = "__opencode_mlflow_duplicate_keys__"
_ESCAPED_MAPPING_TAG = "__opencode_mlflow_mapping__"

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

    def export_to_snapshot(self, writer: SnapshotWriter, redaction: RedactionCallback) -> ExportStats: ...


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

    @property
    def warning_codes(self) -> tuple[str, ...]:
        """Safe exporter warning codes preserved without reducing them to a count."""
        return self.export_warnings


class SnapshotWriter:
    """Bounded batch writer for the temporary normalized snapshot.

    Its exposed surface is exactly ``write_session``, ``write_record``,
    and ``finalize``. The injected redaction callback runs on every
    persisted envelope value and permitted raw field before temporary
    persistence; typed structural identity fields are written verbatim
    while raw-column projections are redacted as content. ``finalize`` is idempotent and
    flushes any pending batch.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        schema_generation: SchemaGeneration,
        redaction: RedactionCallback,
        write_batch: int = 256,
        selection_evaluator: SelectionEvidenceEvaluator | None = None,
    ) -> None:
        if write_batch < 1:
            raise ValueError("write_batch must be a positive integer")
        self._connection = connection
        self._schema_generation = schema_generation
        self._redaction_policy = redaction
        self._selection_evaluator = selection_evaluator
        self._write_batch = write_batch
        self._session_rows: list[tuple[str, str, str, int, int]] = []
        self._record_rows: list[tuple[str, str, str, str | None, str, str]] = []
        self._session_count = 0
        self._record_count = 0
        self._warning_count = 0
        self._stats: ExportStats | None = None
        self._current_session_id: str | None = None
        self._current_evidence = SelectionEvidence()

    def write_session(self, header: SessionHeader, canonical_session_row: CanonicalRecordEnvelope) -> None:
        """Write one normalized session row from its header and envelope."""
        self._ensure_open()
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
                dumps_snapshot_json(stored),
                int(evidence.directory_match),
                int(evidence.model_match),
            )
        )
        self._session_count += 1
        self._warning_count += len(canonical_session_row.warnings) + len(header.warnings)
        if len(self._session_rows) >= self._write_batch:
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
        self._record_rows.append(
            (
                record.session_id,
                dumps_json_value(list(order_key)),
                record.source_table,
                record.source_id,
                dumps_snapshot_json(stored),
                dumps_snapshot_json(redacted_permitted_fields),
            )
        )
        self._record_count += 1
        self._warning_count += len(canonical_record.warnings)
        if len(self._record_rows) >= self._write_batch:
            self._flush_records()

    def finalize(self) -> ExportStats:
        """Flush pending batches and attest that the export completed."""
        if self._stats is not None:
            return self._stats
        self._flush_sessions()
        self._flush_records()
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
                self._session_rows[index] = (*row[:4], 1)
                return
        self._connection.execute(
            "UPDATE snapshot_sessions SET model_filter_match = 1 WHERE session_id = ?",
            (session_id,),
        )

    def _flush_sessions(self) -> None:
        if not self._session_rows:
            return
        self._connection.executemany(
            "INSERT INTO snapshot_sessions"
            " (session_id, schema, canonical_session_row, directory_filter_match, model_filter_match)"
            " VALUES (?, ?, ?, ?, ?)",
            self._session_rows,
        )
        self._session_rows.clear()

    def _flush_records(self) -> None:
        if not self._record_rows:
            return
        self._connection.executemany(
            "INSERT INTO snapshot_records"
            " (session_id, order_key, source_table, source_id, canonical_record, raw_permitted_fields)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            self._record_rows,
        )
        self._record_rows.clear()


@contextmanager
def create_normalized_snapshot(
    path: Path,
    temp_root: Path,
    exporter: SourceExporter,
    redaction: RedactionCallback,
    busy_timeout_ms: int = 5000,
    write_batch: int = 256,
    selection_evaluator: SelectionEvidenceEvaluator | None = None,
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
        temp_dir = Path(tempfile.mkdtemp(prefix="opencode-mlflow-snapshot-", dir=temp_root))
    except OSError as error:
        raise SnapshotError("source_snapshot_failed", _SNAPSHOT_CREATE_FAILED) from error
    snapshot_path = temp_dir / SNAPSHOT_FILENAME
    cleanup_warnings: list[str] = []
    snapshot: SourceSnapshot | None = None
    returned: ExportStats | None = None
    stats: ExportStats | None = None
    snapshot_error: SnapshotError | None = None
    importer_error: ImporterError | None = None
    try:
        try:
            with open_live_connection(path, busy_timeout_ms) as live:
                fingerprint = source_database_fingerprint(exporter.schema_generation, path)
                exporter.bind_connection(live.connection)
                description = exporter.inspect()
                writer_connection = sqlite3.connect(snapshot_path, isolation_level=None)
                try:
                    writer_connection.executescript(_SNAPSHOT_SCHEMA)
                    os.chmod(snapshot_path, 0o600)
                    writer = SnapshotWriter(
                        writer_connection,
                        exporter.schema_generation,
                        redaction,
                        write_batch,
                        selection_evaluator,
                    )
                    try:
                        returned = exporter.export_to_snapshot(writer, redaction)
                    finally:
                        stats = writer.finalize()
                finally:
                    writer_connection.close()
            if returned is None or not returned.complete or stats is None:
                raise SnapshotError("source_snapshot_failed", _SNAPSHOT_POPULATE_FAILED)
            stats = _merge_exporter_warnings(stats, returned)
            replay_connection = _open_replay_connection(snapshot_path)
            snapshot = SourceSnapshot(
                fingerprint=fingerprint,
                snapshot_path=snapshot_path,
                schema_description=description,
                connection=replay_connection,
                stats=stats,
                export_warnings=_export_warning_codes(exporter),
                selection_evidence_enabled=selection_evaluator is not None,
                cleanup_warnings=cleanup_warnings,
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
        _cleanup_snapshot(snapshot, snapshot_path, cleanup_warnings)
        failure = snapshot_error or importer_error
        if failure is not None:
            failure.attach_cleanup_warnings(tuple(cleanup_warnings))


def _open_replay_connection(snapshot_path: Path) -> sqlite3.Connection:
    """Open the temporary snapshot read-only in one deferred replay transaction."""
    uri = Path(os.path.abspath(snapshot_path)).as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None)
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
        try:
            snapshot.connection.close()
        except sqlite3.Error:
            _record_cleanup_failure(cleanup_warnings)
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(snapshot_path) + suffix)
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            _record_cleanup_failure(cleanup_warnings)
    try:
        snapshot_path.parent.rmdir()
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
    }


def _timestamp_storage_value(value: object) -> object:
    if value is None:
        return None
    timestamp = cast(TimestampValue, value)
    return {"value": timestamp.value, "unit": timestamp.unit}


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
