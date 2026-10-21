"""Schema probe: classify a source database against the versioned catalog.

The probe reads only metadata: table names from ``sqlite_master`` and
column names through the parameterized ``pragma_table_info`` table-valued
function. It never reads row values or source content.

Classification follows the catalog matrix exactly:

- **current** when ``session_v2`` plus ``session_message`` match the
  current catalog entry;
- **legacy** when ``session``, ``message``, and ``part`` match the legacy
  entry;
- **mixed** when both entries match; and
- unsupported/malformed otherwise — a :class:`SourceCompatibilityError`
  with code ``unsupported_schema`` whose safe message names the missing
  catalog table or column, never raw SQL values or source content.

Session-scoped tables are described when present but never required for a
match. Unknown additive columns are tolerated (and observed); catalogued
optional columns are read when present. A ``session_share`` table that
exists with no catalogued safe-column entry is reported through the
description's warnings as ``session_share_omitted``.

Exporter and replay factories are deliberately absent: Task 4 registers
them in its adapter registry only after the concrete adapters exist, so
schema probing never calls a class that does not yet exist.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from opencode_mlflow.errors import SourceCompatibilityError
from opencode_mlflow.source.catalog import (
    OPENCODE_SCHEMA_CATALOG_V1,
    SESSION_SHARE_TABLE,
    SchemaCatalog,
)
from opencode_mlflow.source.model import SchemaDescription, SchemaGeneration

_SESSION_SHARE_OMITTED = "session_share_omitted"

_TABLE_NAMES_SQL = "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
_TABLE_COLUMNS_SQL = "SELECT name FROM pragma_table_info(?)"


def probe_schema(
    connection: sqlite3.Connection,
    catalogs: Sequence[SchemaCatalog] = OPENCODE_SCHEMA_CATALOG_V1,
) -> SchemaDescription:
    """Probe ``connection`` against ``catalogs`` and describe the match.

    Raises :class:`SourceCompatibilityError` (code ``unsupported_schema``,
    exit status ``2``) when no catalog entry matches; the safe message
    names each catalog's first missing table or column.
    """
    existing = _existing_tables(connection)
    warnings: list[str] = []
    if SESSION_SHARE_TABLE in existing:
        # Every session_share column is omitted until a reviewed
        # safe-column catalog entry exists.
        warnings.append(_SESSION_SHARE_OMITTED)

    matched: list[SchemaCatalog] = []
    problems: list[str] = []
    for catalog in catalogs:
        problem = _first_missing_structure(connection, existing, catalog)
        if problem is None:
            matched.append(catalog)
        else:
            problems.append(f"catalog '{catalog.name}' {problem}")

    if not matched:
        detail = "; ".join(problems) if problems else "no catalogs were supplied"
        raise SourceCompatibilityError(
            "unsupported_schema",
            f"the source database matches no catalogued schema: {detail}",
        )

    generations = {catalog.generation for catalog in matched}
    generation: SchemaGeneration = "mixed" if len(generations) > 1 else matched[0].generation
    tables: dict[str, tuple[str, ...]] = {}
    for catalog in matched:
        for table in (*catalog.tables, *catalog.session_scoped_tables):
            if table in existing and table not in tables:
                tables[table] = _observed_columns(connection, table)

    return SchemaDescription(
        generation=generation,
        catalog_version=matched[0].version,
        tables=tables,
        warnings=tuple(warnings),
    )


def _existing_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(_TABLE_NAMES_SQL).fetchall()
    return {str(row[0]) for row in rows}


def _observed_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = connection.execute(_TABLE_COLUMNS_SQL, (table,)).fetchall()
    return tuple(str(row[0]) for row in rows)


def _first_missing_structure(
    connection: sqlite3.Connection,
    existing: set[str],
    catalog: SchemaCatalog,
) -> str | None:
    """Return the first missing required table/column diagnostic, or ``None``."""
    for table, columns in catalog.tables.items():
        if table not in existing:
            return f"requires table '{table}'"
        observed = _observed_columns(connection, table)
        for column in columns:
            if column.required and column.name not in observed:
                return f"requires column '{column.name}' on table '{table}'"
    return None
