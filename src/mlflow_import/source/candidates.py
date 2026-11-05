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
    for table in tables:
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if "id" not in columns:
            continue
        selected = [name for name in ("id", "time_created", "directory", "path") if name in columns]
        quoted = ", ".join(f'"{name}"' for name in selected)
        for row in connection.execute(f'SELECT {quoted} FROM "{table}"'):
            values = dict(zip(selected, row, strict=True))
            native_id = values.get("id")
            if native_id is None or not str(native_id).strip():
                continue
            native_id = str(native_id)
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
    return frozenset(found)


def _created_time(value: object) -> datetime | None:
    timestamp, _warnings = normalize_epoch_ms(value)
    return timestamp.utc if timestamp is not None else None
