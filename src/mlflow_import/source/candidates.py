"""Header-only OpenCode candidate discovery.

This pass intentionally reads session metadata only. Transcript and scoped
record tables are consulted later, after the candidate IDs have been selected.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection
from pathlib import Path

from mlflow_import.errors import SourceCompatibilityError
from mlflow_import.retry import check_budget
from mlflow_import.source.id_index import DiskIds
from mlflow_import.source.model import SchemaDescription
from mlflow_import.source.normalize import normalize_epoch_ms
from mlflow_import.source.selection import SelectionFilters, matches_directory, time_bounds


def candidate_session_ids(
    connection: sqlite3.Connection,
    description: SchemaDescription,
    filters: SelectionFilters,
    session_id: str | None = None,
    *,
    include_relatives: bool = True,
    seed_ids: Collection[str] | None = None,
    warnings: set[str] | None = None,
) -> DiskIds | None:
    """Return header-filtered session IDs, or ``None`` when no pushdown applies."""
    if seed_ids is None and not any(
        (
            session_id,
            filters.from_time,
            filters.to_time,
            filters.directory,
            filters.model,
            filters.min_user_turns,
        )
    ):
        return None
    tables = {
        "legacy": ("session",),
        "current": ("session_v2",),
        "mixed": ("session_v2", "session"),
    }[description.generation]
    result = DiskIds()
    lower, upper = time_bounds(filters)
    index = result.db
    index.executescript("""CREATE TABLE headers(id TEXT PRIMARY KEY,owner TEXT,matched INTEGER DEFAULT 0);
        CREATE TABLE edges(child TEXT,parent TEXT,PRIMARY KEY(child,parent));
        CREATE INDEX edge_parent ON edges(parent);""")
    for table in tables:
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if "id" not in columns:
            continue
        selected = [
            name
            for name in ("id", "time_created", "directory", "path", "parent_id", "fork_session_id")
            if name in columns
        ]
        quoted = ", ".join(f'"{name}"' for name in selected)
        for row in connection.execute(f'SELECT {quoted} FROM "{table}"'):
            check_budget()
            values = dict(zip(selected, row, strict=True))
            native_id = values.get("id")
            if native_id is None or not str(native_id).strip():
                raise SourceCompatibilityError(
                    "invalid_session_identity", "an authoritative session header has no identity"
                )
            native_id = str(native_id)
            # V2 header precedence applies to pruning and relationship edges.
            if index.execute("INSERT OR IGNORE INTO headers(id,owner) VALUES(?,?)", (native_id, table)).rowcount == 0:
                if index.execute("SELECT owner FROM headers WHERE id=?", (native_id,)).fetchone()[0] == table:
                    raise SourceCompatibilityError(
                        "duplicate_session_identity", "authoritative session headers have duplicate identities"
                    )
                continue
            for parent_id in (values.get("parent_id"), values.get("fork_session_id")):
                if parent_id is None or not str(parent_id).strip():
                    continue
                parent = str(parent_id)
                index.execute("INSERT OR IGNORE INTO edges VALUES(?,?)", (native_id, parent))
            if seed_ids is not None:
                if native_id in seed_ids:
                    index.execute("UPDATE headers SET matched=1 WHERE id=?", (native_id,))
                continue
            if session_id is not None and native_id != session_id:
                continue
            created = _created_time(values.get("time_created"))
            if created is None and (lower is not None or upper is not None) and warnings is not None:
                warnings.add("missing_session_creation_time")
            if lower is not None and (created is None or created < lower):
                continue
            if upper is not None and (created is None or created > upper):
                continue
            directory = values.get("directory") or values.get("path")
            if filters.directory is not None and (
                not isinstance(directory, str) or not matches_directory(Path(directory), filters.directory)
            ):
                continue
            index.execute("UPDATE headers SET matched=1 WHERE id=?", (native_id,))
    if include_relatives:
        index.executescript("""WITH RECURSIVE ancestors(id) AS (
            SELECT id FROM headers WHERE matched=1 UNION
            SELECT e.parent FROM edges e JOIN ancestors a ON e.child=a.id),
            descendants(id) AS (SELECT id FROM headers WHERE matched=1 UNION
            SELECT e.child FROM edges e JOIN descendants d ON e.parent=d.id)
            INSERT INTO ids SELECT id FROM headers WHERE id IN
                (SELECT id FROM ancestors UNION SELECT id FROM descendants);""")
    else:
        index.execute("INSERT INTO ids SELECT id FROM headers WHERE matched=1")
    return result


def _ancestry_closure(
    seeds: set[str],
    session_ids: set[str],
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> set[str]:
    """Include each seed's ancestors and descendants without crossing siblings."""
    included = {seed for seed in seeds if seed in session_ids}
    pending = list(included)
    visited: set[str] = set()
    while pending:
        parent = pending.pop()
        if parent in visited:
            continue
        visited.add(parent)
        for ancestor in parents.get(parent, ()):
            if ancestor in session_ids and ancestor not in visited:
                included.add(ancestor)
                pending.append(ancestor)

    pending = list(seeds & session_ids)
    visited = set()
    while pending:
        child = pending.pop()
        if child in visited:
            continue
        visited.add(child)
        for descendant in children.get(child, ()):
            if descendant in session_ids and descendant not in visited:
                included.add(descendant)
                pending.append(descendant)
    return included


def _created_time(value: object) -> int | None:
    timestamp, _warnings = normalize_epoch_ms(value)
    return timestamp.nanoseconds if timestamp is not None else None
