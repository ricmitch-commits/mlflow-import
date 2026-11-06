"""Header-only OpenCode candidate discovery.

This pass intentionally reads session metadata only. Transcript and scoped
record tables are consulted later, after the candidate IDs have been selected.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from mlflow_import.source.model import SchemaDescription
from mlflow_import.source.normalize import normalize_epoch_ms
from mlflow_import.source.selection import SelectionFilters, matches_directory


def candidate_session_ids(
    connection: sqlite3.Connection,
    description: SchemaDescription,
    filters: SelectionFilters,
    session_id: str | None = None,
) -> frozenset[str] | None:
    """Return header-filtered session IDs, or ``None`` when no pushdown applies."""
    if not any(
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
    found: set[str] = set()
    all_session_ids: set[str] = set()
    parents: dict[str, set[str]] = {}
    children: dict[str, set[str]] = {}
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
            values = dict(zip(selected, row, strict=True))
            native_id = values.get("id")
            if native_id is None or not str(native_id).strip():
                continue
            native_id = str(native_id)
            all_session_ids.add(native_id)
            for parent_id in (values.get("parent_id"), values.get("fork_session_id")):
                if parent_id is None or not str(parent_id).strip():
                    continue
                parent = str(parent_id)
                parents.setdefault(native_id, set()).add(parent)
                children.setdefault(parent, set()).add(native_id)
            if session_id is not None and native_id != session_id:
                continue
            created = _created_time(values.get("time_created"))
            if filters.from_time is not None and (created is None or created < filters.from_time):
                continue
            if filters.to_time is not None and (created is None or created > filters.to_time):
                continue
            directory = values.get("directory") or values.get("path")
            if filters.directory is not None and (
                not isinstance(directory, str) or not matches_directory(Path(directory), filters.directory)
            ):
                continue
            found.add(native_id)
    return frozenset(_ancestry_closure(found, all_session_ids, parents, children))


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


def _created_time(value: object) -> datetime | None:
    timestamp, _warnings = normalize_epoch_ms(value)
    return timestamp.utc if timestamp is not None else None
