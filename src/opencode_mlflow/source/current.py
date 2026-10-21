"""Live exporter for the catalogued current OpenCode schema generation.

:class:`CurrentAdapter` owns every SELECT against the current tables
(``session_v2``, ``session_message``, and the session-scoped
``session_pending``/``session_inbox``/``instruction_entry``/
``instruction_blob``/``instruction_state`` tables). All SQL is
parameterized and lives here, in the adapter owning the matching catalog;
values from the prohibited secret tables and the omitted
``session_share`` table are never selected.

Export shape (pinned by the adapter contract tests):

- ``session_message`` uses ``(session_id, seq)`` as the authoritative
  sequence domain; messages with a missing or malformed ``seq`` sort
  after the sequenced rows;
- nested ``data.content`` items become derived records with IDs
  ``<session_message_id>#content-<index>`` and a child ordinal
  immediately after their parent; nested user content is content, never
  another turn start (derived children carry no prompt classification);
- explicit-ID rows deduplicate on the
  ``(schema_generation, source_table, source_id)`` projection only;
  missing-ID rows use the shared fallback identity (canonical row digest
  plus occurrence ordinal), never ``None`` as a deduplication key;
- session-scoped rows are emitted only for sessions that exist;
  unassociated pending/inbox/instruction rows and unreferenced
  instruction blobs are reported (warnings), never assigned;
- ``instruction_blob`` rows are session-scoped only transitively through
  ``instruction_entry.value = instruction_blob.hash`` and deduplicate on
  their explicit hash identity;
- in a mixed database, current wins overlapping session IDs; the
  suppressed legacy row is retained as a ``legacy_session_suppressed``
  warning on the authoritative current header.

Child-table access is a streaming merge-join, not a query per session:
``session_message``, every session-scoped table, and the transitive
instruction blobs are each read through ONE whole-table cursor joined to
``session_v2`` (blobs through ``instruction_entry``) and ordered in
emission order, zipped with the session stream through
:class:`OrderedGroupReader` (one row of lookahead, bounded memory, a
constant number of statements regardless of row counts). SQLite executes
the unindexed joins through automatic transient indexes and the ORDER BY
through its disk-backed sorter, so the export cost is a near-linear
number of row visits instead of one full scan per session. The implicit
``rowid`` discriminates duplicate session rows and nullable-ID child rows
during the merge; tables without one (``WITHOUT ROWID``) fall back to the
per-session queries, declared primary-key ordering where available, and
local occurrence tracking. A schema exposing neither rowid nor a declared
key cannot distinguish otherwise identical ties beyond SQLite source order.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from typing import Any

from opencode_mlflow.contracts import CanonicalRecordEnvelope
from opencode_mlflow.source.catalog import SchemaCatalog
from opencode_mlflow.source.model import (
    SchemaDescription,
    SchemaGeneration,
    SessionHeader,
    SourceRecord,
    TimestampValue,
)
from opencode_mlflow.source.normalize import (
    MALFORMED_VALUE_WARNING,
    MAX_NORMALIZABLE_EPOCH_MS,
    UNASSOCIATED_RECORD_WARNING,
    UNKNOWN_FIELD_WARNING,
    IdentityTracker,
    OrderedGroupReader,
    canonical_row_digest,
    catalogued_columns,
    decode_json_column,
    describe_classifications,
    describe_timestamp_units,
    evaluate_classification,
    fallback_source_identity,
    has_rowid,
    normalize_completion_timestamp,
    normalize_epoch_ms,
    permitted_observed_columns,
    primary_key_columns,
    quote_identifier,
    session_order_key,
    session_row_sort_key,
    stable_key_order,
    unknown_json_field_warning,
    unknown_record_type_warning,
)
from opencode_mlflow.source.ordering import source_type_rank
from opencode_mlflow.source.probe import probe_schema
from opencode_mlflow.source.snapshot import (
    ExportStats,
    RedactionCallback,
    SnapshotWriter,
)

SESSION_TABLE = "session_v2"
MESSAGE_TABLE = "session_message"
PENDING_TABLE = "session_pending"
INBOX_TABLE = "session_inbox"
ENTRY_TABLE = "instruction_entry"
BLOB_TABLE = "instruction_blob"
STATE_TABLE = "instruction_state"

_ROWID_KEY = "__opencode_mlflow_source_rowid__"

SCOPED_TABLES = (PENDING_TABLE, INBOX_TABLE, ENTRY_TABLE, STATE_TABLE)
_ALL_CURRENT_TABLES = (SESSION_TABLE, MESSAGE_TABLE, *SCOPED_TABLES, BLOB_TABLE)

MESSAGE_SEQUENCE_DOMAIN = "session_message.seq"
LEGACY_SESSION_SUPPRESSED_WARNING = "legacy_session_suppressed"

SESSION_MISSING_ID_WARNING = "session_missing_id"
DUPLICATE_SESSION_ID_WARNING = "duplicate_session_id"

_TIMESTAMP_ORDER_COLUMNS = frozenset(
    {
        "time_created",
        "time_updated",
        "time_compacting",
        "time_archived",
        "time_idle",
        "time_viewed",
        "time_suspended",
        "fork_boundary",
        "epoch_start",
    }
)
_TEXT_ORDER_COLUMNS = frozenset({"id"})


def _text(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _numeric_order(
    column: str,
    *tiebreaks: str,
    qualifier: str | None = None,
    max_abs: int | None = None,
) -> str:
    """Ordering SQL that sorts a numeric column with missing/malformed last.

    The guard expression matches :func:`normalize_epoch_ms`: a value that
    is NULL or not numeric belongs to the missing group (value 0), so SQL
    order and the Python sort key agree. ``qualifier`` prefixes each
    column with a table alias for joined cursors.
    """

    def qualified(name: str) -> str:
        quoted = quote_identifier(name)
        return f"{qualifier}.{quoted}" if qualifier is not None else quoted

    def numeric_guard(name: str, max_value: int | None) -> str:
        value = qualified(name)
        malformed = f"(typeof({value}) = 'real' AND ({value} = 1e999 OR {value} != {value}))"
        if max_value is not None:
            malformed = (
                f"({malformed} OR (typeof({value}) IN ('integer', 'real') "
                f"AND ({value} > {max_value} OR {value} < -{max_value})))"
            )
        return f"({value} IS NULL OR typeof({value}) NOT IN ('integer', 'real') OR {malformed})"

    quoted = qualified(column)
    guard = numeric_guard(column, max_abs)
    parts = [guard, f"CASE WHEN {guard} THEN 0 ELSE {quoted} END"]
    for name in tiebreaks:
        tie = qualified(name)
        if name in _TEXT_ORDER_COLUMNS:
            parts.append(f"({tie} IS NULL)")
            parts.append(f"COALESCE(CAST({tie} AS TEXT), '')")
            continue
        tie_max_abs = MAX_NORMALIZABLE_EPOCH_MS if name in _TIMESTAMP_ORDER_COLUMNS else None
        tie_guard = numeric_guard(name, tie_max_abs)
        parts.append(tie_guard)
        parts.append(f"CASE WHEN {tie_guard} THEN 0 ELSE {tie} END")
    return ", ".join(parts)


def _present_ns(timestamp: TimestampValue | None) -> tuple[int, int]:
    if timestamp is None:
        return (1, 0)
    return (0, timestamp.nanoseconds)


_SESSION_ORDER = _numeric_order("time_created", "id", qualifier="s", max_abs=MAX_NORMALIZABLE_EPOCH_MS)


def _scoped_numeric_order(
    observed: tuple[str, ...] | tuple[()],
    order_columns: tuple[str, ...],
    qualifier: str | None = None,
) -> str:
    """Ordering SQL for an optional (session-scoped) table, observed columns only.

    Session-scoped tables are individually optional and their catalogued
    columns are not probe-validated, so ordering references only columns
    the table actually has. A table missing a tiebreak column exports
    cleanly (ordered by the remaining columns, or by the session stream)
    exactly as the pre-merge-join code did, instead of failing identifier
    resolution inside the joined cursor.
    """
    present = [name for name in order_columns if name in observed]
    if not present:
        return ""
    return _numeric_order(
        present[0],
        *present[1:],
        qualifier=qualifier,
        max_abs=MAX_NORMALIZABLE_EPOCH_MS if present[0] in _TIMESTAMP_ORDER_COLUMNS else None,
    )


def _numeric_value(value: object) -> tuple[object | None, tuple[str, ...]]:
    """A numeric sequence value, or ``None`` with a warning when malformed."""
    if value is None:
        return None, ()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, (MALFORMED_VALUE_WARNING,)
    if isinstance(value, float) and not math.isfinite(value):
        return None, (MALFORMED_VALUE_WARNING,)
    return value, ()


class CurrentAdapter:
    """:class:`SourceExporter` for the catalogued current generation.

    ``connection`` is the constructor-supplied live connection serving
    ``inspect()``; the snapshot driver rebinds the export connection
    through :meth:`bind_connection` (Ruling 7).
    """

    def __init__(self, connection: sqlite3.Connection, catalog: SchemaCatalog) -> None:
        if catalog.generation != "current":
            raise ValueError("CurrentAdapter requires a current-generation schema catalog")
        self.schema_generation: SchemaGeneration = catalog.generation
        self.catalog = catalog
        self.connection = connection
        self._export_warnings: list[str] = []
        self._tracker: IdentityTracker | None = None
        self._known_session_ids: set[str] = set()
        self._overlap_session_ids: frozenset[str] = frozenset()
        self._active_session_blob_references: set[str] = set()
        self._columns: dict[str, tuple[str, ...]] = {}
        self._rowid_tables: dict[str, bool] = {}
        self._primary_key_columns: dict[str, tuple[str, ...]] = {}
        self._joined_children = False
        self._message_reader: OrderedGroupReader | None = None
        self._scoped_readers: dict[str, OrderedGroupReader] = {}
        self._blob_reader: OrderedGroupReader | None = None

    # ------------------------------------------------------------------
    # SourceExporter protocol
    # ------------------------------------------------------------------

    def bind_connection(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def inspect(self) -> SchemaDescription:
        description = probe_schema(self.connection, [self.catalog])
        return replace(
            description,
            tie_ranks=dict(self.catalog.tie_ranks),
            timestamp_units=describe_timestamp_units(self.catalog),
            classifications=describe_classifications(self.catalog),
        )

    @property
    def export_warnings(self) -> tuple[str, ...]:
        """Safe warning codes collected by the most recent export."""
        return tuple(self._export_warnings)

    def export_to_snapshot(self, writer: SnapshotWriter, redaction: RedactionCallback) -> ExportStats:
        self.begin_export()
        for _key, row, child_key in self.iter_exportable_sessions():
            self.export_session_records(writer, row, child_key)
        self.report_unassociated_records()
        stats = writer.finalize()
        if not self._export_warnings:
            return stats
        return replace(stats, warning_count=stats.warning_count + len(self._export_warnings))

    # ------------------------------------------------------------------
    # Composition seams (used standalone and by MixedAdapter)
    # ------------------------------------------------------------------

    def session_ids(self) -> set[str]:
        """All non-NULL current session IDs (for mixed overlap detection)."""
        rows = self.connection.execute(f"SELECT id FROM {quote_identifier(SESSION_TABLE)}")
        return {str(row[0]) for row in rows if row[0] is not None and (not isinstance(row[0], str) or row[0].strip())}

    def begin_export(self, *, overlap_session_ids: frozenset[str] = frozenset()) -> None:
        """Reset per-export state; overlapping mixed IDs warn on the header."""
        self._export_warnings = []
        self._tracker = IdentityTracker(self.schema_generation)
        self._known_session_ids = set()
        self._overlap_session_ids = frozenset(overlap_session_ids)
        self._active_session_blob_references = set()
        self._columns = {table: permitted_observed_columns(self.connection, table) for table in _ALL_CURRENT_TABLES}
        self._rowid_tables = {
            table: bool(columns) and has_rowid(self.connection, table) for table, columns in self._columns.items()
        }
        self._primary_key_columns = {
            table: primary_key_columns(self.connection, table) for table, columns in self._columns.items() if columns
        }
        self._message_reader = None
        self._scoped_readers = {}
        self._blob_reader = None
        self._joined_children = bool(self._columns[SESSION_TABLE]) and self._rowid_tables.get(SESSION_TABLE, False)
        if self._joined_children:
            self._open_child_readers()

    def iter_exportable_sessions(
        self,
    ) -> Iterator[tuple[tuple[int, int, str], dict[str, object], tuple[Any, ...]]]:
        """Yield ``(sort key, raw session row, child key)`` in session order.

        The ``child key`` extends the sort key with the session row's
        ``rowid`` so the merge-joined child cursors can group by the exact
        parent row.
        """
        tracker = self._require_tracker()
        columns = self._columns[SESSION_TABLE]
        if not columns:
            return
        select = ", ".join(quote_identifier(name) for name in columns)
        order = _numeric_order("time_created", "id", max_abs=MAX_NORMALIZABLE_EPOCH_MS)
        if self._joined_children:
            select += ", rowid"
            order += ", rowid"
        sql = f"SELECT {select} FROM {quote_identifier(SESSION_TABLE)} ORDER BY {order}"
        for row in self.connection.execute(sql):
            values = dict(zip(columns, row[: len(columns)], strict=True))
            raw_id = values.get("id")
            if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
                self._export_warnings.append(SESSION_MISSING_ID_WARNING)
                continue
            session_id = str(raw_id)
            if not tracker.register_explicit(SESSION_TABLE, session_id):
                self._export_warnings.append(DUPLICATE_SESSION_ID_WARNING)
                continue
            self._known_session_ids.add(session_id)
            sort_key = session_row_sort_key(values)
            child_key = (*sort_key, row[len(columns)]) if self._joined_children else tuple(sort_key)
            yield sort_key, values, child_key

    def export_session_records(
        self,
        writer: SnapshotWriter,
        row: Mapping[str, object],
        child_key: tuple[Any, ...],
        *,
        overlap: bool = False,
    ) -> None:
        """Write one session's header, turn records, and scoped records."""
        session_id = str(row["id"])
        header, session_decoded = self._build_header(row, overlap=overlap)
        writer.write_session(
            header,
            CanonicalRecordEnvelope(
                decoded=session_decoded,
                raw_text=None,
                raw_columns=dict(row),
                warnings=(),
            ),
        )
        self._active_session_blob_references = set()
        try:
            self._emit_messages(writer, session_id, child_key)
            self._emit_scoped(writer, session_id, child_key)
        finally:
            # Blob references are needed only while this session's scoped
            # projection is emitted. Retaining them across the export makes
            # memory proportional to every session's instruction history.
            self._active_session_blob_references.clear()

    def report_unassociated_records(self) -> None:
        """Report session-scoped rows whose session was not exported."""
        for table in SCOPED_TABLES:
            columns = self._columns.get(table, ())
            if not columns or "session_id" not in columns:
                continue
            rows = self.connection.execute(
                f"SELECT session_id, count(*) FROM {quote_identifier(table)} GROUP BY session_id"
            )
            for session_id, count in rows:
                if session_id is None or str(session_id) not in self._known_session_ids:
                    self._export_warnings.extend([UNASSOCIATED_RECORD_WARNING] * int(count))
        blob_columns = self._columns.get(BLOB_TABLE, ())
        if not blob_columns or "hash" not in blob_columns:
            return
        entry_columns = self._columns.get(ENTRY_TABLE, ())
        if "session_id" in entry_columns and "value" in entry_columns and self._columns.get(SESSION_TABLE):
            sql = (
                f"SELECT b.hash FROM {quote_identifier(BLOB_TABLE)} b "
                f"WHERE b.hash IS NOT NULL AND NOT EXISTS ("
                f"SELECT 1 FROM {quote_identifier(ENTRY_TABLE)} e "
                f"JOIN {quote_identifier(SESSION_TABLE)} s ON e.session_id = s.id "
                "WHERE e.value = b.hash AND e.session_id IS NOT NULL "
                "AND trim(CAST(e.session_id AS TEXT)) <> ''"
                ")"
            )
        else:
            sql = f"SELECT hash FROM {quote_identifier(BLOB_TABLE)} WHERE hash IS NOT NULL"
        for _row in self.connection.execute(sql):
            self._export_warnings.append(UNASSOCIATED_RECORD_WARNING)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_tracker(self) -> IdentityTracker:
        if self._tracker is None:
            raise RuntimeError("begin_export must run before exporting records")
        return self._tracker

    def _open_child_readers(self) -> None:
        """Open the merge-joined child cursors for the whole database.

        ``session_message`` and every session-scoped table are each ONE
        whole-table pass joined to ``session_v2`` and ordered in emission
        order; the instruction-blob pass joins transitively through
        ``instruction_entry``. Requires the session table to expose a
        ``rowid``; without one the per-session queries are used. Child
        tables without a ``rowid`` use local occurrence tracking instead.
        """
        self._message_reader = self._open_session_child_reader(
            MESSAGE_TABLE,
            _numeric_order("seq", "id", qualifier="m"),
            alias="m",
            include_rowid=self._rowid_tables.get(MESSAGE_TABLE, False),
        )
        for table, order_columns in (
            (PENDING_TABLE, ("time_created", "admitted_seq", "id")),
            (INBOX_TABLE, ("time_created", "enqueued_seq", "id")),
            (ENTRY_TABLE, ("time_created", "time_updated", "removed")),
            (STATE_TABLE, ("epoch_start", "through_seq")),
        ):
            order_by = self._scoped_order(table, order_columns, qualifier="t")
            reader = self._open_session_child_reader(
                table,
                order_by,
                alias="t",
                include_rowid=self._rowid_tables.get(table, False),
            )
            if reader is not None:
                self._scoped_readers[table] = reader

        blob_columns = self._columns[BLOB_TABLE]
        # The joined blob pass resolves `e.value` (and the entry table
        # itself) as real identifiers, so it is only usable when the
        # optional instruction_entry table is present WITH its catalogued
        # join column; otherwise the per-hash fallback path runs, which
        # tolerates an absent or value-less entry table exactly as the
        # pre-merge-join code did.
        entry_columns = self._columns.get(ENTRY_TABLE, ())
        if (
            "hash" in blob_columns
            and "session_id" in entry_columns
            and "value" in entry_columns
            and self._columns.get(SESSION_TABLE)
        ):
            blob_width = len(blob_columns)
            blob_select = ", ".join(f"b.{quote_identifier(name)}" for name in blob_columns)
            blob_select += f", s.{quote_identifier('time_created')}, s.{quote_identifier('id')}, s.rowid"
            blob_order = ", ".join((_SESSION_ORDER, "s.rowid", f"b.{quote_identifier('hash')}"))
            blob_sql = (
                f"SELECT {blob_select} FROM {quote_identifier(BLOB_TABLE)} b"
                f" JOIN {quote_identifier(ENTRY_TABLE)} e ON b.hash = e.value"
                f" JOIN {quote_identifier(SESSION_TABLE)} s ON e.session_id = s.id"
                f" ORDER BY {blob_order}"
            )

            def blob_pairs() -> Iterator[tuple[tuple[Any, ...], dict[str, object]]]:
                cursor = self.connection.execute(blob_sql)
                for row in cursor:
                    session_key = session_order_key(row[blob_width], row[blob_width + 1])
                    yield ((*session_key, row[blob_width + 2]), dict(zip(blob_columns, row[:blob_width], strict=True)))

            self._blob_reader = OrderedGroupReader(blob_pairs())

    def _open_session_child_reader(
        self,
        table: str,
        order_by: str,
        *,
        alias: str,
        include_rowid: bool = False,
    ) -> OrderedGroupReader | None:
        """One joined child cursor over ``table``, grouped per session row."""
        columns = self._columns.get(table, ())
        if not columns or "session_id" not in columns:
            return None
        width = len(columns)
        select = ", ".join(f"{alias}.{quote_identifier(name)}" for name in columns)
        if include_rowid:
            select += f", {alias}.rowid"
        select += f", s.{quote_identifier('time_created')}, s.{quote_identifier('id')}, s.rowid"
        table_order = f", {order_by}" if order_by else ""
        rowid_order = f", {alias}.rowid" if include_rowid else ""
        sql = (
            f"SELECT {select} FROM {quote_identifier(table)} {alias}"
            f" JOIN {quote_identifier(SESSION_TABLE)} s ON {alias}.session_id = s.id"
            f" ORDER BY {_SESSION_ORDER}, s.rowid{table_order}{rowid_order}"
        )

        def pairs() -> Iterator[tuple[tuple[Any, ...], dict[str, object]]]:
            cursor = self.connection.execute(sql)
            for row in cursor:
                rowid_offset = 1 if include_rowid else 0
                session_key = session_order_key(row[width + rowid_offset], row[width + rowid_offset + 1])
                values = dict(zip(columns, row[:width], strict=True))
                if include_rowid:
                    values[_ROWID_KEY] = row[width]
                yield ((*session_key, row[width + rowid_offset + 2]), values)

        return OrderedGroupReader(pairs())

    def _scoped_order(
        self,
        table: str,
        order_columns: tuple[str, ...],
        *,
        qualifier: str | None = None,
        include_rowid: bool = False,
    ) -> str:
        """Return structural scoped ordering plus rowid or declared-key fallback.

        Content-bearing optional columns are intentionally absent from
        ``order_columns``.  The caller appends an ordinary table's physical
        rowid; a ``WITHOUT ROWID`` table gets its declared primary key, which
        is the strongest catalog-stable key available.
        Remaining ties on a malformed schema without either key are
        inherently dependent on SQLite's source order.
        """
        observed = self._columns.get(table, ())
        order = _scoped_numeric_order(observed, order_columns, qualifier=qualifier)
        if self._rowid_tables.get(table, False) and include_rowid:
            discriminator = f"{qualifier}.rowid" if qualifier is not None else "rowid"
        elif not self._rowid_tables.get(table, False):
            discriminator = stable_key_order(
                observed,
                self._primary_key_columns.get(table, ()),
                qualifier=qualifier,
            )
        else:
            discriminator = ""
        return ", ".join(part for part in (order, discriminator) if part)

    def _unknown_columns(self, table: str) -> tuple[str, ...]:
        observed = self._columns.get(table, ())
        catalogued = set(catalogued_columns(self.catalog, table))
        return tuple(name for name in observed if name not in catalogued)

    def _json_columns(self, table: str) -> tuple[str, ...]:
        columns = self.catalog.tables.get(table) or self.catalog.session_scoped_tables.get(table) or ()
        return tuple(column.name for column in columns if column.json_encoded)

    def _timestamp_columns(self, table: str) -> tuple[str, ...]:
        columns = self.catalog.tables.get(table) or self.catalog.session_scoped_tables.get(table) or ()
        return tuple(column.name for column in columns if column.timestamp_unit is not None)

    def _build_header(
        self, row: Mapping[str, object], *, overlap: bool = False
    ) -> tuple[SessionHeader, dict[str, object]]:
        warnings: list[str] = []
        session_decoded: dict[str, object] = {}
        for name in self._json_columns(SESSION_TABLE):
            decoded_column = decode_json_column(row.get(name))
            warnings.extend(decoded_column.warnings)
            if name == "model" and unknown_json_field_warning(decoded_column.decoded, "model") is not None:
                warnings.append(UNKNOWN_FIELD_WARNING)
            session_decoded[name] = decoded_column.decoded

        timestamps: dict[str, TimestampValue | None] = {}
        for name in self._timestamp_columns(SESSION_TABLE):
            normalized, warning = normalize_epoch_ms(row.get(name))
            warnings.extend(warning)
            timestamps[name] = normalized

        if any(row.get(name) is not None for name in self._unknown_columns(SESSION_TABLE)):
            warnings.append(UNKNOWN_FIELD_WARNING)
        if overlap:
            # Mixed precedence: this current session wins the overlapping
            # legacy row, and the suppression is retained as provenance.
            warnings.append(LEGACY_SESSION_SUPPRESSED_WARNING)

        session_id = str(row["id"])
        header = SessionHeader(
            session_id=session_id,
            adapter_generation="current",
            catalog_version=self.catalog.version,
            project_id=_text(row.get("project_id")),
            workspace_id=_text(row.get("workspace_id")),
            parent_id=_text(row.get("parent_id")),
            fork_session_id=_text(row.get("fork_session_id")),
            fork_boundary=timestamps.get("fork_boundary"),
            slug=_text(row.get("slug")),
            directory=_text(row.get("directory")),
            path=_text(row.get("path")),
            title=_text(row.get("title")),
            version=_text(row.get("version")),
            agent=_text(row.get("agent")),
            model=session_decoded.get("model"),
            metadata=session_decoded.get("metadata"),
            revert=row.get("revert"),
            permission=row.get("permission"),
            cost=_float(row.get("cost")),
            tokens_input=_int(row.get("tokens_input")),
            tokens_output=_int(row.get("tokens_output")),
            tokens_reasoning=_int(row.get("tokens_reasoning")),
            tokens_cache_read=_int(row.get("tokens_cache_read")),
            tokens_cache_write=_int(row.get("tokens_cache_write")),
            summary_additions=_int(row.get("summary_additions")),
            summary_deletions=_int(row.get("summary_deletions")),
            summary_files=_int(row.get("summary_files")),
            summary_diffs=session_decoded.get("summary_diffs"),
            time_created=timestamps.get("time_created"),
            time_updated=timestamps.get("time_updated"),
            time_compacting=timestamps.get("time_compacting"),
            time_archived=timestamps.get("time_archived"),
            time_idle=timestamps.get("time_idle"),
            time_viewed=timestamps.get("time_viewed"),
            time_suspended=timestamps.get("time_suspended"),
            idle_outcome=_text(row.get("idle_outcome")),
            resume_attempts=_int(row.get("resume_attempts")),
            raw_columns=dict(row),
            warnings=tuple(dict.fromkeys(warnings)),
        )
        return header, session_decoded

    def _message_rows(self, session_id: str, child_key: tuple[Any, ...]) -> Iterator[dict[str, object]]:
        """The session's message rows: merge-joined stream or per-session query."""
        if self._message_reader is not None:
            yield from self._message_reader.take(child_key)
            return
        columns = self._columns[MESSAGE_TABLE]
        if not columns:
            return
        select = ", ".join(quote_identifier(name) for name in columns)
        include_rowid = self._rowid_tables.get(MESSAGE_TABLE, False)
        if include_rowid:
            select += ", rowid"
        rowid_order = ", rowid" if include_rowid else ""
        sql = (
            f"SELECT {select} FROM {quote_identifier(MESSAGE_TABLE)}"
            f" WHERE session_id = ? ORDER BY {_numeric_order('seq', 'id')}{rowid_order}"
        )
        for row in self.connection.execute(sql, (session_id,)):
            values = dict(zip(columns, row[: len(columns)], strict=True))
            if include_rowid:
                values[_ROWID_KEY] = row[len(columns)]
            yield values

    def _emit_messages(self, writer: SnapshotWriter, session_id: str, child_key: tuple[Any, ...]) -> None:
        tracker = self._require_tracker()
        columns = self._columns[MESSAGE_TABLE]
        if not columns:
            return
        unknown = self._unknown_columns(MESSAGE_TABLE)
        for values in self._message_rows(session_id, child_key):
            message_rowid = values.pop(_ROWID_KEY, None)
            raw_id = values.get("id")
            message_id = _text(raw_id)
            if message_id is not None and not tracker.register_explicit(MESSAGE_TABLE, message_id):
                continue
            decoded_column = decode_json_column(values.get("data"))
            decoded = decoded_column.decoded
            message_type = _text(values.get("type"))
            source_type = message_type if message_type is not None else "unknown"
            sequence_value, sequence_warnings = _numeric_value(values.get("seq"))
            timestamp, timestamp_warnings = normalize_epoch_ms(values.get("time_created"))
            completion_timestamp, completion_warnings = normalize_completion_timestamp(
                decoded, values, self.catalog, MESSAGE_TABLE
            )
            warnings = [*decoded_column.warnings, *sequence_warnings, *timestamp_warnings, *completion_warnings]
            if message_type is None:
                warnings.append(UNKNOWN_FIELD_WARNING)
            elif unknown_record_type_warning(MESSAGE_TABLE, message_type) is not None:
                warnings.append(UNKNOWN_FIELD_WARNING)
            if unknown_json_field_warning(decoded, MESSAGE_TABLE) is not None:
                warnings.append(UNKNOWN_FIELD_WARNING)
            if any(values.get(name) is not None for name in unknown):
                warnings.append(UNKNOWN_FIELD_WARNING)
            warning_tuple = tuple(dict.fromkeys(warnings))

            classification = evaluate_classification(self.catalog.classifications, MESSAGE_TABLE, values, decoded, None)
            message_key = self._row_key(
                MESSAGE_TABLE,
                message_id,
                values,
                decoded_column.raw_text,
                session_id,
                sequence_value,
                timestamp,
                row_discriminator=message_rowid,
            )
            order_key = (
                0 if sequence_value is not None else 1,
                sequence_value if sequence_value is not None else 0,
                source_type_rank(MESSAGE_TABLE),
                message_key,
            )
            record = SourceRecord(
                source_table=MESSAGE_TABLE,
                source_type=source_type,
                source_id=message_id,
                raw_json=decoded_column.raw_text,
                decoded_value=decoded,
                raw_columns=values,
                warnings=warning_tuple,
                session_id=session_id,
                sequence_domain=MESSAGE_SEQUENCE_DOMAIN,
                sequence_value=sequence_value,
                timestamp=timestamp,
                completion_timestamp=completion_timestamp,
                classification=classification,
                source_identity=None if message_id is not None else message_key,
            )
            writer.write_record(
                record,
                order_key,
                CanonicalRecordEnvelope(
                    decoded=decoded,
                    raw_text=decoded_column.raw_text,
                    raw_columns=values,
                    warnings=warning_tuple,
                ),
                values,
            )
            self._emit_content_items(
                writer,
                session_id,
                decoded,
                message_key,
                order_key,
                sequence_value,
                timestamp,
                completion_timestamp,
            )

    def _emit_content_items(
        self,
        writer: SnapshotWriter,
        session_id: str,
        decoded: object | None,
        message_key: str,
        parent_order_key: tuple[object, ...],
        sequence_value: object | None,
        timestamp: TimestampValue | None,
        completion_timestamp: TimestampValue | None,
    ) -> None:
        """Emit derived nested content records immediately after the parent.

        Each item gets the derived ID ``<parent key>#content-<index>`` and
        a child ordinal; derived content never carries a prompt
        classification, so nested user content can never start a turn.
        """
        content = decoded.get("content") if isinstance(decoded, Mapping) else None
        if not isinstance(content, list):
            return
        for index, item in enumerate(content):
            item_type = item.get("type") if isinstance(item, Mapping) else None
            item_completion, item_completion_warnings = normalize_completion_timestamp(
                item, {}, self.catalog, MESSAGE_TABLE
            )
            warning_values = list(item_completion_warnings)
            if not isinstance(item_type, str) or unknown_record_type_warning("part", item_type) is not None:
                warning_values.append(UNKNOWN_FIELD_WARNING)
            if unknown_json_field_warning(item, "part") is not None:
                warning_values.append(UNKNOWN_FIELD_WARNING)
            warnings = tuple(dict.fromkeys(warning_values))
            record = SourceRecord(
                source_table=MESSAGE_TABLE,
                source_type=item_type if isinstance(item_type, str) else "unknown",
                source_id=f"{message_key}#content-{index}",
                raw_json=None,
                decoded_value=item,
                raw_columns={},
                warnings=warnings,
                session_id=session_id,
                sequence_domain=MESSAGE_SEQUENCE_DOMAIN,
                sequence_value=sequence_value,
                parent_source_id=message_key,
                parent_source_table=MESSAGE_TABLE,
                derived_child_ordinal=index,
                timestamp=timestamp,
                completion_timestamp=item_completion or completion_timestamp,
                classification=None,
            )
            writer.write_record(
                record,
                (*parent_order_key, 1, index),
                CanonicalRecordEnvelope(
                    decoded=item,
                    raw_text=None,
                    raw_columns={},
                    warnings=warnings,
                ),
                {},
            )

    def _emit_scoped(self, writer: SnapshotWriter, session_id: str, child_key: tuple[Any, ...]) -> None:
        for table, order_columns in (
            (PENDING_TABLE, ("time_created", "admitted_seq", "id")),
            (INBOX_TABLE, ("time_created", "enqueued_seq", "id")),
            (ENTRY_TABLE, ("time_created", "time_updated", "removed")),
            (STATE_TABLE, ("epoch_start", "through_seq")),
        ):
            order_by = self._scoped_order(table, order_columns)
            self._emit_scoped_table(
                writer,
                session_id,
                child_key,
                table,
                order_by=order_by,
            )
        self._emit_referenced_blobs(writer, session_id, child_key)

    def _scoped_rows(
        self, session_id: str, child_key: tuple[Any, ...], table: str, order_by: str
    ) -> Iterator[dict[str, object]]:
        """The session's scoped rows: merge-joined stream or per-session query."""
        reader = self._scoped_readers.get(table)
        if reader is not None:
            yield from reader.take(child_key)
            return
        columns = self._columns.get(table, ())
        if not columns:
            return
        select = ", ".join(quote_identifier(name) for name in columns)
        include_rowid = self._rowid_tables.get(table, False)
        if include_rowid:
            select += ", rowid"
        order_clause = f" ORDER BY {order_by}" if order_by else ""
        if include_rowid:
            order_clause += ", rowid"
        sql = f"SELECT {select} FROM {quote_identifier(table)} WHERE session_id = ?{order_clause}"
        for row in self.connection.execute(sql, (session_id,)):
            values = dict(zip(columns, row[: len(columns)], strict=True))
            if include_rowid:
                values[_ROWID_KEY] = row[len(columns)]
            yield values

    def _emit_scoped_table(
        self,
        writer: SnapshotWriter,
        session_id: str,
        child_key: tuple[Any, ...],
        table: str,
        *,
        order_by: str,
    ) -> None:
        tracker = self._require_tracker()
        columns = self._columns.get(table, ())
        if not columns or "session_id" not in columns:
            return
        unknown = self._unknown_columns(table)
        json_columns = self._json_columns(table)
        for values in self._scoped_rows(session_id, child_key, table, order_by):
            row_discriminator = values.pop(_ROWID_KEY, None)
            source_id = _text(values.get("id")) if "id" in values else None
            if source_id is not None and not tracker.register_explicit(table, source_id):
                continue
            warnings: list[str] = []
            decoded: object | None = None
            raw_json: str | None = None
            if len(json_columns) == 1:
                decoded_column = decode_json_column(values.get(json_columns[0]))
                decoded = decoded_column.decoded
                raw_json = decoded_column.raw_text
                warnings.extend(decoded_column.warnings)
            elif json_columns:
                combined: dict[str, object] = {}
                for name in json_columns:
                    decoded_column = decode_json_column(values.get(name))
                    combined[name] = decoded_column.decoded
                    warnings.extend(decoded_column.warnings)
                decoded = combined
            if table in {PENDING_TABLE, INBOX_TABLE} and unknown_json_field_warning(decoded, table) is not None:
                warnings.append(UNKNOWN_FIELD_WARNING)
            timestamp, timestamp_warnings = normalize_epoch_ms(values.get("time_created"))
            completion_timestamp, completion_warnings = normalize_completion_timestamp(
                decoded, values, self.catalog, table
            )
            warnings.extend((*timestamp_warnings, *completion_warnings))
            if any(values.get(name) is not None for name in unknown):
                warnings.append(UNKNOWN_FIELD_WARNING)
            warning_tuple = tuple(dict.fromkeys(warnings))

            row_type = _text(values.get("type"))
            source_type = row_type if row_type is not None else table
            digest = canonical_row_digest(
                table,
                values,
                raw_json,
                session_id=session_id,
                parent_key=session_id,
                sequence_or_timestamp=timestamp,
                row_discriminator=row_discriminator,
            )
            if source_id is None:
                occurrence = tracker.next_occurrence(
                    table,
                    session_id,
                    session_id,
                    timestamp,
                    digest,
                    row_discriminator=row_discriminator,
                )
                source_identity = fallback_source_identity(
                    self.schema_generation, table, session_id, session_id, timestamp, digest, occurrence
                )
            else:
                source_identity = None
            if table == ENTRY_TABLE:
                hash_value = _text(values.get("value"))
                if hash_value is not None:
                    self._active_session_blob_references.add(hash_value)

            record = SourceRecord(
                source_table=table,
                source_type=source_type,
                source_id=source_id,
                raw_json=raw_json,
                decoded_value=decoded,
                raw_columns=values,
                warnings=warning_tuple,
                session_id=session_id,
                timestamp=timestamp,
                completion_timestamp=completion_timestamp,
                classification=None,
                source_identity=source_identity,
            )
            writer.write_record(
                record,
                (
                    *_present_ns(timestamp),
                    source_type_rank(table),
                    source_id or source_identity or "",
                    row_discriminator if row_discriminator is not None else -1,
                ),
                CanonicalRecordEnvelope(
                    decoded=decoded,
                    raw_text=raw_json,
                    raw_columns=values,
                    warnings=warning_tuple,
                ),
                values,
            )

    def _emit_referenced_blobs(self, writer: SnapshotWriter, session_id: str, child_key: tuple[Any, ...]) -> None:
        """Emit instruction blobs referenced by this session's entries.

        The merge-joined blob pass fans a blob out to every session whose
        entries reference it, so rows are filtered to the hashes actually
        referenced by this session's emitted entries (deduplicated within
        each session while preserving the original source hash), and the
        per-session fallback queries each referenced hash directly.
        """
        tracker = self._require_tracker()
        columns = self._columns.get(BLOB_TABLE, ())
        if not columns or "hash" not in columns:
            return
        unknown = self._unknown_columns(BLOB_TABLE)
        emitted_references = self._active_session_blob_references
        if self._blob_reader is not None:
            blob_rows: Iterable[dict[str, object]] = (
                values
                for values in self._blob_reader.take(child_key)
                if _text(values.get("hash")) in emitted_references
            )
        else:
            referenced = sorted(emitted_references)

            def per_hash_rows() -> Iterator[dict[str, object]]:
                select = ", ".join(quote_identifier(name) for name in columns)
                for hash_value in referenced:
                    rows = self.connection.execute(
                        f"SELECT {select} FROM {quote_identifier(BLOB_TABLE)} WHERE hash = ?", (hash_value,)
                    )
                    for row in rows:
                        yield dict(zip(columns, row, strict=True))

            blob_rows = per_hash_rows()
        for values in blob_rows:
            hash_value = _text(values.get("hash"))
            if hash_value is None:
                continue
            if not tracker.register_explicit(BLOB_TABLE, repr((session_id, hash_value))):
                continue
            warnings: tuple[str, ...] = ()
            if any(values.get(name) is not None for name in unknown):
                warnings = (UNKNOWN_FIELD_WARNING,)
            record = SourceRecord(
                source_table=BLOB_TABLE,
                source_type=BLOB_TABLE,
                source_id=hash_value,
                raw_json=None,
                decoded_value=None,
                raw_columns=values,
                warnings=warnings,
                session_id=session_id,
                classification=None,
            )
            writer.write_record(
                record,
                (1, 0, source_type_rank(BLOB_TABLE), hash_value),
                CanonicalRecordEnvelope(
                    decoded=None,
                    raw_text=None,
                    raw_columns=values,
                    warnings=warnings,
                ),
                values,
            )

    def _row_key(
        self,
        table: str,
        source_id: str | None,
        values: Mapping[str, object],
        raw_json: str | None,
        session_id: str,
        sequence_value: object | None,
        timestamp: TimestampValue | None,
        *,
        row_discriminator: object | None = None,
    ) -> str:
        """The row's stable key: its explicit ID or its fallback identity."""
        if source_id is not None:
            return source_id
        tracker = self._require_tracker()
        explicit_or_timestamp = sequence_value if sequence_value is not None else timestamp
        digest = canonical_row_digest(
            table,
            values,
            raw_json,
            session_id=session_id,
            parent_key=session_id,
            sequence_or_timestamp=explicit_or_timestamp,
            row_discriminator=row_discriminator,
        )
        occurrence = tracker.next_occurrence(
            table,
            session_id,
            session_id,
            explicit_or_timestamp,
            digest,
            row_discriminator=row_discriminator,
        )
        return fallback_source_identity(
            self.schema_generation, table, session_id, session_id, explicit_or_timestamp, digest, occurrence
        )
