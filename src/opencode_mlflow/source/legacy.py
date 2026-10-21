"""Live exporter for the catalogued legacy OpenCode schema generation.

:class:`LegacyAdapter` owns every SELECT against the legacy tables
(``session``, ``message``, ``part``, and the session-scoped ``todo``
table). All SQL is parameterized and lives here, in the adapter owning
the matching catalog; values from the prohibited secret tables and the
omitted ``session_share`` table are never selected.

Export shape (pinned by the adapter contract tests):

- sessions stream in normalized ``(creation_time_present/value,
  session_id)`` order; a session without an ID is skipped and reported;
- each canonical ``message`` row is written exactly once, immediately
  followed by its ``part`` rows (part timestamp then ID) — a joined
  parent is never emitted once per child and a part never precedes its
  parent;
- rows with an explicit ID deduplicate on the
  ``(schema_generation, source_table, source_id)`` projection only;
  rows without one get the shared missing-ID identity (canonical row
  digest plus occurrence ordinal), and part projections carry a join
  projection key so repeated join projections collapse while distinct
  no-ID rows stay distinct;
- ``todo`` rows are session-scoped records with fallback identities;
  todos whose session does not exist are reported (warnings), never
  assigned.

Child-table access is a streaming merge-join, not a query per parent:
messages, parts, and todos are each read through ONE whole-table cursor
joined to its parent chain and ordered in emission order, and zipped
with the session stream through :class:`OrderedGroupReader` (one row of
lookahead, bounded memory, a constant number of statements regardless of
row counts). SQLite executes the unindexed joins through automatic
transient indexes and the ORDER BY through its disk-backed sorter, so
the export cost is a near-linear number of row visits instead of one
full scan per parent. The implicit ``rowid`` discriminates duplicate
parent rows during the merge; tables without one (``WITHOUT ROWID``)
fall back to per-session/per-parent queries, and parts of missing-ID
parents are always fetched with the NULL-safe per-parent query. Declared
primary keys are the best catalog-stable discriminator for such tables;
schemas without either a rowid or a declared key retain SQLite's source
order for otherwise identical ties.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from typing import Any

from opencode_mlflow.contracts import CanonicalRecordEnvelope
from opencode_mlflow.source.catalog import SchemaCatalog
from opencode_mlflow.source.model import (
    RecordClassificationName,
    SchemaDescription,
    SchemaGeneration,
    SessionHeader,
    SourceRecord,
    TimestampValue,
)
from opencode_mlflow.source.normalize import (
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
    join_projection_identity,
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

SESSION_TABLE = "session"
MESSAGE_TABLE = "message"
PART_TABLE = "part"
TODO_TABLE = "todo"

_ROWID_KEY = "__opencode_mlflow_source_rowid__"

MESSAGE_TO_PART_JOIN = "message->part"

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


_SESSION_ORDER = _numeric_order("time_created", "id", qualifier="s", max_abs=MAX_NORMALIZABLE_EPOCH_MS)


def _scoped_numeric_order(
    observed: tuple[str, ...],
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


def _present_ns(timestamp: TimestampValue | None) -> tuple[int, int]:
    if timestamp is None:
        return (1, 0)
    return (0, timestamp.nanoseconds)


def _message_parent_order_key(time_created: object, message_id: object, rowid: object) -> tuple[object, ...]:
    """Mirror the legacy message SQL order for parent/part merge keys."""
    timestamp, _warnings = normalize_epoch_ms(time_created)
    normalized_id = "" if message_id is None else str(message_id)
    return (*_present_ns(timestamp), int(message_id is None), normalized_id, rowid)


class LegacyAdapter:
    """:class:`SourceExporter` for the catalogued legacy generation.

    ``connection`` is the constructor-supplied live connection serving
    ``inspect()``; the snapshot driver rebinds the export connection
    through :meth:`bind_connection` (Ruling 7).
    """

    def __init__(self, connection: sqlite3.Connection, catalog: SchemaCatalog) -> None:
        if catalog.generation != "legacy":
            raise ValueError("LegacyAdapter requires a legacy-generation schema catalog")
        self.schema_generation: SchemaGeneration = catalog.generation
        self.catalog = catalog
        self.connection = connection
        self._export_warnings: list[str] = []
        self._tracker: IdentityTracker | None = None
        self._known_session_ids: set[str] = set()
        self._suppressed_session_ids: frozenset[str] = frozenset()
        self._columns: dict[str, tuple[str, ...]] = {}
        self._rowid_tables: dict[str, bool] = {}
        self._primary_key_columns: dict[str, tuple[str, ...]] = {}
        self._joined_children = False
        self._message_reader: OrderedGroupReader | None = None
        self._part_reader: OrderedGroupReader | None = None
        self._todo_reader: OrderedGroupReader | None = None

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
        for _key, row, _child_key in self.iter_exportable_sessions():
            self.export_session_records(writer, row, _child_key)
        self.report_unassociated_records()
        stats = writer.finalize()
        if not self._export_warnings:
            return stats
        # Export-level warnings (unassociated records, skipped sessions)
        # are not attached to any written row, so they are added to the
        # returned stats on top of the writer's own count.
        return replace(stats, warning_count=stats.warning_count + len(self._export_warnings))

    # ------------------------------------------------------------------
    # Composition seams (used standalone and by MixedAdapter)
    # ------------------------------------------------------------------

    def session_ids(self) -> set[str]:
        """All non-NULL legacy session IDs (for mixed overlap detection)."""
        rows = self.connection.execute(f"SELECT id FROM {quote_identifier(SESSION_TABLE)}")
        return {str(row[0]) for row in rows if row[0] is not None and (not isinstance(row[0], str) or row[0].strip())}

    def begin_export(
        self,
        *,
        suppressed_session_ids: frozenset[str] = frozenset(),
    ) -> None:
        """Reset per-export state; suppressed sessions are never emitted."""
        self._export_warnings = []
        self._tracker = IdentityTracker(self.schema_generation)
        self._known_session_ids = set()
        self._suppressed_session_ids = frozenset(suppressed_session_ids)
        self._columns = {
            table: permitted_observed_columns(self.connection, table)
            for table in (SESSION_TABLE, MESSAGE_TABLE, PART_TABLE, TODO_TABLE)
        }
        self._rowid_tables = {
            table: bool(columns) and has_rowid(self.connection, table) for table, columns in self._columns.items()
        }
        self._primary_key_columns = {
            table: primary_key_columns(self.connection, table) for table, columns in self._columns.items() if columns
        }
        self._message_reader = None
        self._part_reader = None
        self._ambiguous_part_reader: OrderedGroupReader | None = None
        self._todo_reader = None
        self._missing_message_counts = (
            {
                str(session_id): int(count)
                for session_id, count in self.connection.execute(
                    f"SELECT session_id, COUNT(*) FROM {quote_identifier(MESSAGE_TABLE)}"
                    " WHERE id IS NULL GROUP BY session_id"
                )
            }
            if self._columns[MESSAGE_TABLE]
            else {}
        )
        self._joined_children = bool(self._columns[SESSION_TABLE]) and self._rowid_tables.get(SESSION_TABLE, False)
        if self._joined_children:
            self._open_child_readers()

    def iter_exportable_sessions(
        self,
    ) -> Iterator[tuple[tuple[int, int, str], dict[str, object], tuple[Any, ...]]]:
        """Yield ``(sort key, raw session row, child key)`` in session order.

        Sessions without an ID are skipped and reported; duplicate session
        IDs collapse to the first projection; suppressed sessions (mixed
        overlap, current wins) are omitted entirely. The ``child key``
        extends the sort key with the session row's ``rowid`` so the
        merge-joined child cursors can group by the exact parent row.
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
            if session_id in self._suppressed_session_ids:
                continue
            if not tracker.register_explicit(SESSION_TABLE, session_id):
                self._export_warnings.append(DUPLICATE_SESSION_ID_WARNING)
                continue
            self._known_session_ids.add(session_id)
            sort_key = session_row_sort_key(values)
            child_key = (*sort_key, row[len(columns)]) if self._joined_children else tuple(sort_key)
            yield sort_key, values, child_key

    def export_session_records(
        self, writer: SnapshotWriter, row: Mapping[str, object], child_key: tuple[Any, ...]
    ) -> None:
        """Write one session's header, turn records, and scoped records."""
        session_id = str(row["id"])
        header, session_decoded = self._build_header(row)
        writer.write_session(
            header,
            CanonicalRecordEnvelope(
                decoded=session_decoded,
                raw_text=None,
                raw_columns=dict(row),
                warnings=(),
            ),
        )
        self._emit_messages(writer, session_id, child_key)
        self._emit_todos(writer, session_id, child_key)

    def report_unassociated_records(self) -> None:
        """Report session-scoped rows whose session was not exported."""
        columns = self._columns[TODO_TABLE]
        if not columns:
            return
        if "session_id" not in columns:
            count = self.connection.execute(f"SELECT count(*) FROM {quote_identifier(TODO_TABLE)}").fetchone()[0]
            self._export_warnings.extend([UNASSOCIATED_RECORD_WARNING] * int(count))
            return
        rows = self.connection.execute(
            f"SELECT session_id, count(*) FROM {quote_identifier(TODO_TABLE)} GROUP BY session_id"
        )
        for session_id, count in rows:
            if session_id is None or str(session_id) not in self._known_session_ids:
                self._export_warnings.extend([UNASSOCIATED_RECORD_WARNING] * int(count))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_tracker(self) -> IdentityTracker:
        if self._tracker is None:
            raise RuntimeError("begin_export must run before exporting records")
        return self._tracker

    def _open_child_readers(self) -> None:
        """Open the merge-joined child cursors (messages, parts, todos).

        Each cursor is ONE whole-table pass joined to its parent chain
        and ordered in emission order; ``rowid`` columns discriminate
        duplicate parent rows. Requires the session table to expose a
        ``rowid``; parts additionally require one on the message table
        (otherwise parts keep the per-parent query).
        """
        message_columns = self._columns[MESSAGE_TABLE]
        part_columns = self._columns[PART_TABLE]
        todo_columns = self._columns[TODO_TABLE]
        message_has_rowid = self._rowid_tables.get(MESSAGE_TABLE, False)
        part_has_rowid = self._rowid_tables.get(PART_TABLE, False)

        if message_columns:
            message_width = len(message_columns)
            message_select = ", ".join(f"m.{quote_identifier(name)}" for name in message_columns)
            message_select += f", s.{quote_identifier('time_created')}, s.{quote_identifier('id')}, s.rowid"
            message_order = (
                _SESSION_ORDER
                + ", s.rowid, "
                + _numeric_order("time_created", "id", qualifier="m", max_abs=MAX_NORMALIZABLE_EPOCH_MS)
            )
            if message_has_rowid:
                message_select += ", m.rowid"
                message_order += ", m.rowid"
            message_sql = (
                f"SELECT {message_select} FROM {quote_identifier(MESSAGE_TABLE)} m"
                f" JOIN {quote_identifier(SESSION_TABLE)} s ON m.session_id = s.id"
                f" ORDER BY {message_order}"
            )

            def message_pairs() -> Iterator[tuple[tuple[Any, ...], tuple[dict[str, object], Any]]]:
                cursor = self.connection.execute(message_sql)
                for row in cursor:
                    session_key = session_order_key(row[message_width], row[message_width + 1])
                    message_values = dict(zip(message_columns, row[:message_width], strict=True))
                    parent_rowid = row[message_width + 3] if message_has_rowid else None
                    yield ((*session_key, row[message_width + 2]), (message_values, parent_rowid))

            self._message_reader = OrderedGroupReader(message_pairs())

        if part_columns and message_has_rowid:
            part_width = len(part_columns)
            part_select = ", ".join(f"p.{quote_identifier(name)}" for name in part_columns)
            if part_has_rowid:
                part_select += ", p.rowid"
            part_select += (
                f", s.{quote_identifier('time_created')}, s.{quote_identifier('id')}, s.rowid, m.rowid"
                ", m.time_created, m.id"
            )
            part_order = ", ".join(
                (
                    _SESSION_ORDER,
                    "s.rowid",
                    _numeric_order("time_created", "id", qualifier="m", max_abs=MAX_NORMALIZABLE_EPOCH_MS),
                    "m.rowid",
                    _numeric_order("time_created", "id", qualifier="p", max_abs=MAX_NORMALIZABLE_EPOCH_MS),
                    "p.rowid"
                    if part_has_rowid
                    else stable_key_order(part_columns, self._primary_key_columns.get(PART_TABLE, ()), qualifier="p"),
                )
            )
            part_sql = (
                f"SELECT {part_select} FROM {quote_identifier(PART_TABLE)} p"
                f" JOIN {quote_identifier(MESSAGE_TABLE)} m ON p.message_id IS m.id AND p.session_id = m.session_id"
                f" JOIN {quote_identifier(SESSION_TABLE)} s ON m.session_id = s.id"
                f" ORDER BY {part_order}"
            )

            def part_pairs() -> Iterator[tuple[tuple[Any, ...], dict[str, object]]]:
                cursor = self.connection.execute(part_sql)
                for row in cursor:
                    rowid_offset = 1 if part_has_rowid else 0
                    session_key = session_order_key(row[part_width + rowid_offset], row[part_width + rowid_offset + 1])
                    part_values = dict(zip(part_columns, row[:part_width], strict=True))
                    if part_has_rowid:
                        part_values[_ROWID_KEY] = row[part_width]
                    yield (
                        (
                            *session_key,
                            row[part_width + rowid_offset + 2],
                            *_message_parent_order_key(
                                row[part_width + rowid_offset + 4],
                                row[part_width + rowid_offset + 5],
                                row[part_width + rowid_offset + 3],
                            ),
                        ),
                        part_values,
                    )

            self._part_reader = OrderedGroupReader(part_pairs())

            # Preserve ambiguous NULL-parent parts once, before any prompt,
            # as unassigned records rather than attributing them arbitrarily.
            ambiguous_sql = (
                "SELECT "
                + ", ".join(f"p.{quote_identifier(name)}" for name in part_columns)
                + (", p.rowid" if part_has_rowid else "")
                + f", s.time_created, s.id, s.rowid FROM {quote_identifier(PART_TABLE)} p"
                + f" JOIN {quote_identifier(SESSION_TABLE)} s ON p.session_id = s.id"
                + f" JOIN (SELECT session_id FROM {quote_identifier(MESSAGE_TABLE)} WHERE id IS NULL"
                + " GROUP BY session_id HAVING COUNT(*) > 1) a ON a.session_id = p.session_id"
                + " WHERE p.message_id IS NULL ORDER BY "
                + _SESSION_ORDER
                + ", s.rowid, "
                + _numeric_order("time_created", "id", qualifier="p", max_abs=MAX_NORMALIZABLE_EPOCH_MS)
                + (", p.rowid" if part_has_rowid else "")
            )

            def ambiguous_pairs() -> Iterator[tuple[tuple[Any, ...], dict[str, object]]]:
                for row in self.connection.execute(ambiguous_sql):
                    rowid_offset = 1 if part_has_rowid else 0
                    key = (
                        *session_order_key(row[part_width + rowid_offset], row[part_width + rowid_offset + 1]),
                        row[part_width + rowid_offset + 2],
                    )
                    values = dict(zip(part_columns, row[:part_width], strict=True))
                    if part_has_rowid:
                        values[_ROWID_KEY] = row[part_width]
                    yield key, values

            self._ambiguous_part_reader = OrderedGroupReader(ambiguous_pairs())

        if todo_columns and "session_id" in todo_columns:
            todo_width = len(todo_columns)
            todo_select = ", ".join(f"t.{quote_identifier(name)}" for name in todo_columns)
            todo_has_rowid = self._rowid_tables.get(TODO_TABLE, False)
            if todo_has_rowid:
                todo_select += ", t.rowid"
            todo_select += f", s.{quote_identifier('time_created')}, s.{quote_identifier('id')}, s.rowid"
            # The optional todo table's columns are not probe-validated, so
            # ordering references only the columns it actually has.
            todo_table_order = _scoped_numeric_order(
                todo_columns, ("time_created", "position", "time_updated"), qualifier="t"
            )
            if not todo_has_rowid:
                todo_table_order = ", ".join(
                    part
                    for part in (
                        todo_table_order,
                        stable_key_order(
                            todo_columns,
                            self._primary_key_columns.get(TODO_TABLE, ()),
                            qualifier="t",
                        ),
                    )
                    if part
                )
            todo_order = ", ".join(
                part
                for part in (
                    _SESSION_ORDER,
                    "s.rowid",
                    todo_table_order or None,
                    "t.rowid" if todo_has_rowid else None,
                )
                if part is not None
            )
            todo_sql = (
                f"SELECT {todo_select} FROM {quote_identifier(TODO_TABLE)} t"
                f" JOIN {quote_identifier(SESSION_TABLE)} s ON t.session_id = s.id"
                f" ORDER BY {todo_order}"
            )

            def todo_pairs() -> Iterator[tuple[tuple[Any, ...], dict[str, object]]]:
                cursor = self.connection.execute(todo_sql)
                for row in cursor:
                    rowid_offset = 1 if todo_has_rowid else 0
                    session_key = session_order_key(row[todo_width + rowid_offset], row[todo_width + rowid_offset + 1])
                    values = dict(zip(todo_columns, row[:todo_width], strict=True))
                    if todo_has_rowid:
                        values[_ROWID_KEY] = row[todo_width]
                    yield (
                        (*session_key, row[todo_width + rowid_offset + 2]),
                        values,
                    )

            self._todo_reader = OrderedGroupReader(todo_pairs())

    @staticmethod
    def _row_values_with_optional_rowid(
        columns: tuple[str, ...], row: tuple[object, ...], include_rowid: bool
    ) -> dict[str, object]:
        """Decode one SELECT row while retaining an optional physical rowid."""
        values = dict(zip(columns, row[: len(columns)], strict=True))
        if include_rowid:
            values[_ROWID_KEY] = row[len(columns)]
        return values

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

    def _build_header(self, row: Mapping[str, object]) -> tuple[SessionHeader, dict[str, object]]:
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

        session_id = str(row["id"])
        header = SessionHeader(
            session_id=session_id,
            adapter_generation="legacy",
            catalog_version=self.catalog.version,
            project_id=_text(row.get("project_id")),
            workspace_id=_text(row.get("workspace_id")),
            parent_id=_text(row.get("parent_id")),
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
            raw_columns=dict(row),
            warnings=tuple(dict.fromkeys(warnings)),
        )
        return header, session_decoded

    def _message_rows(self, session_id: str, child_key: tuple[Any, ...]) -> Iterator[tuple[dict[str, object], Any]]:
        """The session's message rows: merge-joined stream or per-session query."""
        if self._message_reader is not None:
            yield from self._message_reader.take(child_key)
            return
        columns = self._columns[MESSAGE_TABLE]
        if not columns:
            return
        select = ", ".join(quote_identifier(name) for name in columns)
        message_has_rowid = self._rowid_tables.get(MESSAGE_TABLE, False)
        if message_has_rowid:
            select += ", rowid"
        rowid_order = ", rowid" if message_has_rowid else ""
        sql = (
            f"SELECT {select} FROM {quote_identifier(MESSAGE_TABLE)}"
            " WHERE session_id = ? ORDER BY "
            f"{_numeric_order('time_created', 'id', max_abs=MAX_NORMALIZABLE_EPOCH_MS)}{rowid_order}"
        )
        for row in self.connection.execute(sql, (session_id,)):
            values = dict(zip(columns, row[: len(columns)], strict=True))
            message_rowid = row[len(columns)] if message_has_rowid else None
            yield values, message_rowid

    def _part_rows(
        self,
        session_id: str,
        child_key: tuple[Any, ...],
        message_rowid: Any,
        message_raw_id: object,
        message_created: object,
    ) -> Iterator[dict[str, object]]:
        """One parent message's part rows.

        The NULL-safe merge join serves explicit and unique missing-ID
        parents in session/message order. Tables without rowid retain the
        session-scoped parameterized fallback. Ambiguous missing-ID parents
        receive a warning rather than an arbitrary part assignment.
        """
        if message_raw_id is None and self._missing_message_counts.get(session_id, 0) > 1:
            self._export_warnings.append("ambiguous_missing_message_id")
            return
        if self._part_reader is not None:
            yield from self._part_reader.take(
                (*child_key, *_message_parent_order_key(message_created, message_raw_id, message_rowid))
            )
            return
        columns = self._columns[PART_TABLE]
        if not columns:
            return
        select = ", ".join(quote_identifier(name) for name in columns)
        part_has_rowid = self._rowid_tables.get(PART_TABLE, False)
        if part_has_rowid:
            select += ", rowid"
        rowid_order = ", rowid" if part_has_rowid else ""
        sql = (
            f"SELECT {select} FROM {quote_identifier(PART_TABLE)}"
            " WHERE message_id IS ? AND session_id = ? ORDER BY "
            f"{_numeric_order('time_created', 'id', max_abs=MAX_NORMALIZABLE_EPOCH_MS)}{rowid_order}"
        )
        for row in self.connection.execute(sql, (message_raw_id, session_id)):
            values = dict(zip(columns, row[: len(columns)], strict=True))
            if part_has_rowid:
                values[_ROWID_KEY] = row[len(columns)]
            yield values

    def _emit_messages(self, writer: SnapshotWriter, session_id: str, child_key: tuple[Any, ...]) -> None:
        tracker = self._require_tracker()
        if self._missing_message_counts.get(session_id, 0) > 1 and self._columns[PART_TABLE]:
            if self._ambiguous_part_reader is not None:
                ambiguous_rows = self._ambiguous_part_reader.take(child_key)
            else:
                columns = self._columns[PART_TABLE]
                select = ", ".join(quote_identifier(name) for name in columns)
                part_has_rowid = self._rowid_tables.get(PART_TABLE, False)
                if part_has_rowid:
                    select += ", rowid"
                rowid_order = ", rowid" if part_has_rowid else ""
                rows = self.connection.execute(
                    f"SELECT {select} FROM {quote_identifier(PART_TABLE)}"
                    f" WHERE message_id IS NULL AND session_id = ?"
                    " ORDER BY "
                    f"{_numeric_order('time_created', 'id', max_abs=MAX_NORMALIZABLE_EPOCH_MS)}{rowid_order}",
                    (session_id,),
                )
                ambiguous_rows = (self._row_values_with_optional_rowid(columns, row, part_has_rowid) for row in rows)
            self._emit_part_rows(writer, tracker, session_id, None, (), None, ambiguous_rows)
        unknown = self._unknown_columns(MESSAGE_TABLE)
        for values, message_rowid in self._message_rows(session_id, child_key):
            raw_id = values.get("id")
            message_id = _text(raw_id)
            if message_id is not None and not tracker.register_explicit(MESSAGE_TABLE, message_id):
                continue
            decoded_column = decode_json_column(values.get("data"))
            decoded = decoded_column.decoded
            role = decoded.get("role") if isinstance(decoded, Mapping) else None
            source_type = role if isinstance(role, str) else "unknown"
            timestamp, timestamp_warnings = normalize_epoch_ms(values.get("time_created"))
            completion_timestamp, completion_warnings = normalize_completion_timestamp(
                decoded, values, self.catalog, MESSAGE_TABLE
            )
            warnings = [*decoded_column.warnings, *timestamp_warnings, *completion_warnings]
            if raw_id is None and self._missing_message_counts.get(session_id, 0) > 1:
                warnings.append("ambiguous_missing_message_id")
            if not isinstance(role, str) or unknown_record_type_warning(MESSAGE_TABLE, role) is not None:
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
                session_id,
                timestamp,
                row_discriminator=message_rowid,
            )
            source_identity = None if message_id is not None else message_key
            order_key = (*_present_ns(timestamp), source_type_rank(MESSAGE_TABLE), message_key)

            record = SourceRecord(
                source_table=MESSAGE_TABLE,
                source_type=source_type,
                source_id=message_id,
                raw_json=decoded_column.raw_text,
                decoded_value=decoded,
                raw_columns=values,
                warnings=warning_tuple,
                session_id=session_id,
                timestamp=timestamp,
                completion_timestamp=completion_timestamp,
                classification=classification,
                source_identity=source_identity,
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
            part_rows = self._part_rows(session_id, child_key, message_rowid, raw_id, values.get("time_created"))
            self._emit_part_rows(
                writer,
                tracker,
                session_id,
                message_key,
                order_key,
                classification,
                part_rows,
            )

    def _emit_part_rows(
        self,
        writer: SnapshotWriter,
        tracker: IdentityTracker,
        session_id: str,
        message_key: str | None,
        parent_order_key: tuple[object, ...],
        parent_classification: RecordClassificationName | None,
        part_rows: Iterable[dict[str, object]],
    ) -> None:
        """Emit one parent message's part rows (shared by both sources)."""
        columns = self._columns[PART_TABLE]
        if not columns:
            return
        unknown = self._unknown_columns(PART_TABLE)
        child_ordinal = 0
        for values in part_rows:
            child_ordinal += 1
            row_discriminator = values.pop(_ROWID_KEY, None)
            raw_id = values.get("id")
            part_id = _text(raw_id)
            decoded_column = decode_json_column(values.get("data"))
            decoded = decoded_column.decoded
            part_type = decoded.get("type") if isinstance(decoded, Mapping) else None
            source_type = part_type if isinstance(part_type, str) else "unknown"
            timestamp, timestamp_warnings = normalize_epoch_ms(values.get("time_created"))
            digest = canonical_row_digest(
                PART_TABLE,
                values,
                decoded_column.raw_text,
                session_id=session_id,
                parent_key=message_key,
                sequence_or_timestamp=timestamp,
                row_discriminator=row_discriminator,
            )

            if part_id is not None:
                if not tracker.register_explicit(PART_TABLE, part_id):
                    continue
                source_identity = None
                join_projection_key = None
                part_key: str = part_id
            else:
                join_key = join_projection_identity(MESSAGE_TO_PART_JOIN, digest, child_ordinal)
                if not tracker.register_join_projection(join_key):
                    # The same physical row projected again through this
                    # join (for example under a duplicated no-ID parent):
                    # a repeated join projection collapses.
                    continue
                occurrence = tracker.next_occurrence(
                    PART_TABLE,
                    session_id,
                    message_key,
                    timestamp,
                    digest,
                    row_discriminator=row_discriminator,
                )
                source_identity = fallback_source_identity(
                    self.schema_generation, PART_TABLE, session_id, message_key, timestamp, digest, occurrence
                )
                join_projection_key = join_key
                part_key = source_identity

            completion_timestamp, completion_warnings = normalize_completion_timestamp(
                decoded, values, self.catalog, PART_TABLE
            )
            warnings = [*decoded_column.warnings, *timestamp_warnings, *completion_warnings]
            if message_key is None:
                warnings.append("ambiguous_missing_message_id")
            if not isinstance(part_type, str) or unknown_record_type_warning(PART_TABLE, part_type) is not None:
                warnings.append(UNKNOWN_FIELD_WARNING)
            if unknown_json_field_warning(decoded, PART_TABLE) is not None:
                warnings.append(UNKNOWN_FIELD_WARNING)
            if any(values.get(name) is not None for name in unknown):
                warnings.append(UNKNOWN_FIELD_WARNING)
            warning_tuple = tuple(dict.fromkeys(warnings))

            classification = evaluate_classification(
                self.catalog.classifications, PART_TABLE, values, decoded, parent_classification
            )
            order_key = (*parent_order_key, 1, *_present_ns(timestamp), part_key)

            record = SourceRecord(
                source_table=PART_TABLE,
                source_type=source_type,
                source_id=part_id,
                raw_json=decoded_column.raw_text,
                decoded_value=decoded,
                raw_columns=values,
                warnings=warning_tuple,
                session_id=session_id,
                parent_source_id=message_key,
                parent_source_table=MESSAGE_TABLE if message_key is not None else None,
                timestamp=timestamp,
                completion_timestamp=completion_timestamp,
                classification=classification,
                source_identity=source_identity,
                join_projection_key=join_projection_key,
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

    def _emit_todos(self, writer: SnapshotWriter, session_id: str, child_key: tuple[Any, ...]) -> None:
        tracker = self._require_tracker()
        columns = self._columns[TODO_TABLE]
        if not columns or "session_id" not in columns:
            return
        unknown = self._unknown_columns(TODO_TABLE)
        if self._todo_reader is not None:
            todo_rows: Iterable[dict[str, object]] = self._todo_reader.take(child_key)
        else:
            select = ", ".join(quote_identifier(name) for name in columns)
            todo_has_rowid = self._rowid_tables.get(TODO_TABLE, False)
            if todo_has_rowid:
                select += ", rowid"
            order_by = _scoped_numeric_order(columns, ("time_created", "position", "time_updated"))
            if not todo_has_rowid:
                order_by = ", ".join(
                    part
                    for part in (
                        order_by,
                        stable_key_order(columns, self._primary_key_columns.get(TODO_TABLE, ())),
                    )
                    if part
                )
            rowid_order = ", rowid" if todo_has_rowid else ""
            order_clause = f" ORDER BY {order_by}" if order_by else ""
            sql = f"SELECT {select} FROM {quote_identifier(TODO_TABLE)} WHERE session_id = ?{order_clause}{rowid_order}"
            todo_rows = (
                self._row_values_with_optional_rowid(columns, row, todo_has_rowid)
                for row in self.connection.execute(sql, (session_id,))
            )
        for values in todo_rows:
            row_discriminator = values.pop(_ROWID_KEY, None)
            timestamp, timestamp_warnings = normalize_epoch_ms(values.get("time_created"))
            completion_timestamp, completion_warnings = normalize_completion_timestamp(
                None, values, self.catalog, TODO_TABLE
            )
            digest = canonical_row_digest(
                TODO_TABLE,
                values,
                None,
                session_id=session_id,
                parent_key=session_id,
                sequence_or_timestamp=timestamp,
                row_discriminator=row_discriminator,
            )
            occurrence = tracker.next_occurrence(
                TODO_TABLE,
                session_id,
                session_id,
                timestamp,
                digest,
                row_discriminator=row_discriminator,
            )
            source_identity = fallback_source_identity(
                self.schema_generation, TODO_TABLE, session_id, session_id, timestamp, digest, occurrence
            )
            warnings = [*timestamp_warnings, *completion_warnings]
            if any(values.get(name) is not None for name in unknown):
                warnings.append(UNKNOWN_FIELD_WARNING)
            warning_tuple = tuple(dict.fromkeys(warnings))

            record = SourceRecord(
                source_table=TODO_TABLE,
                source_type="todo",
                source_id=None,
                raw_json=None,
                decoded_value=None,
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
                (*_present_ns(timestamp), source_type_rank(TODO_TABLE), source_identity),
                CanonicalRecordEnvelope(
                    decoded=None,
                    raw_text=None,
                    raw_columns=values,
                    warnings=warning_tuple,
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
        parent_key: str,
        timestamp: TimestampValue | None,
        *,
        row_discriminator: object | None = None,
    ) -> str:
        """The row's stable key: its explicit ID or its fallback identity."""
        if source_id is not None:
            return source_id
        tracker = self._require_tracker()
        digest = canonical_row_digest(
            table,
            values,
            raw_json,
            session_id=session_id,
            parent_key=parent_key,
            sequence_or_timestamp=timestamp,
            row_discriminator=row_discriminator,
        )
        occurrence = tracker.next_occurrence(
            table,
            session_id,
            parent_key,
            timestamp,
            digest,
            row_discriminator=row_discriminator,
        )
        return fallback_source_identity(
            self.schema_generation, table, session_id, parent_key, timestamp, digest, occurrence
        )
