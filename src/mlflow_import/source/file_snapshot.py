"""Restricted normalized snapshots for fixture-qualified JSONL sources."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Collection, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, cast

from mlflow_import import resources
from mlflow_import.errors import ImporterError, SnapshotError
from mlflow_import.retry import retry_budget
from mlflow_import.source.model import source_database_fingerprint, source_namespace
from mlflow_import.source.selection import SelectionEvidenceEvaluator
from mlflow_import.source.snapshot import (
    _SNAPSHOT_SCHEMA,
    SNAPSHOT_FILENAME,
    RedactionCallback,
    SnapshotWriter,
    SourceExporter,
    SourceSnapshot,
    _open_replay_connection,
)


def _source_manifest(path: Path) -> tuple[tuple[str, int, int, int], ...]:
    """Return a stable stat manifest for a file or a JSONL source directory."""
    paths = (path,) if path.is_file() else tuple(sorted(path.rglob("*.jsonl")))
    manifest: list[tuple[str, int, int, int]] = []
    for source_path in paths:
        if not source_path.is_file():
            continue
        stat = source_path.stat()
        manifest.append((str(source_path), stat.st_size, stat.st_mtime_ns, stat.st_ino))
    return tuple(manifest)


@contextmanager
def _create_file_snapshot_once(
    path: Path,
    temp_root: Path,
    exporter: SourceExporter,
    redaction: RedactionCallback,
    selection_evaluator: SelectionEvidenceEvaluator | None = None,
    session_id: str | None = None,
    session_ids: Collection[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Iterator[SourceSnapshot]:
    """Freeze one JSONL source into the same replay schema as SQLite sources."""
    if not path.exists() or path.is_symlink():
        raise SnapshotError("source_unavailable", "the selected source file is unavailable")
    run_dir = resources.make_temp_dir(prefix="mlflow-import-file-", dir=temp_root)
    snapshot_path = run_dir / SNAPSHOT_FILENAME
    connection: sqlite3.Connection | None = None
    replay: sqlite3.Connection | None = None
    snapshot: SourceSnapshot | None = None
    writer: SnapshotWriter | None = None
    try:
        connection = resources.sqlite_connect(snapshot_path, isolation_level=None)
        connection.executescript(_SNAPSHOT_SCHEMA)
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA max_page_count=2097152")
        os.chmod(snapshot_path, 0o600)
        writer = SnapshotWriter(
            connection,
            exporter.schema_generation,
            redaction,
            selection_evaluator=selection_evaluator,
            progress_callback=progress_callback,
        )
        exporter.bind_connection(connection)
        if selection_evaluator is not None:
            cast(Any, exporter).filters = selection_evaluator.filters
        stats = exporter.export_to_snapshot(writer, redaction, session_id, session_ids)
        stats = writer.finalize()
        connection.close()
        connection = None
        replay = _open_replay_connection(snapshot_path)
        description = exporter.inspect()
        snapshot = SourceSnapshot(
            fingerprint=source_database_fingerprint(exporter.schema_generation, path),
            source_namespace=source_namespace(path, str(getattr(exporter, "harness", "opencode"))),
            snapshot_path=snapshot_path,
            schema_description=description,
            connection=replay,
            stats=stats,
            export_warnings=tuple(getattr(exporter, "export_warnings", ())),
            selection_evidence_enabled=selection_evaluator is not None,
            turn_index=getattr(exporter, "turn_index", None),
        )
    except ImporterError:
        raise
    except Exception as error:
        raise SnapshotError("source_snapshot_failed", "the file source snapshot could not be created") from error
    else:
        yield snapshot
    finally:
        if writer is not None:
            writer.close()
        if snapshot is not None and snapshot.turn_index is not None:
            snapshot.turn_index.close()
        elif getattr(exporter, "turn_index", None) is not None:
            cast(Any, exporter).turn_index.close()
        if hasattr(exporter, "turn_index"):
            cast(Any, exporter).turn_index = None
        if replay is not None:
            replay.close()
        if connection is not None:
            connection.close()
        resources.remove_tree(run_dir, ignore_errors=True)


@contextmanager
def create_file_snapshot(
    path: Path,
    temp_root: Path,
    exporter: SourceExporter,
    redaction: RedactionCallback,
    selection_evaluator: SelectionEvidenceEvaluator | None = None,
    session_id: str | None = None,
    session_ids: Collection[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Iterator[SourceSnapshot]:
    """Retry unstable selected views at most twice within the shared budget.

    Only construction is retried. Exceptions from consumers (including target
    writes) propagate unchanged and can never trigger a live-source reread.
    """
    with retry_budget() as budget, ExitStack() as stack:
        for attempt in range(3):
            budget.check()
            try:
                snapshot = stack.enter_context(
                    _create_file_snapshot_once(
                        path,
                        temp_root,
                        exporter,
                        redaction,
                        selection_evaluator,
                        session_id,
                        session_ids,
                        progress_callback,
                    )
                )
            except SnapshotError as error:
                if error.code != "source_changed" or attempt == 2:
                    raise
                budget.check(retry=True)
            else:
                break
        yield snapshot
