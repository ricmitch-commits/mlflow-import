"""Restricted normalized snapshots for fixture-qualified JSONL sources."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mlflow_import.errors import SnapshotError
from mlflow_import.source.model import source_database_fingerprint
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


@contextmanager
def create_file_snapshot(
    path: Path,
    temp_root: Path,
    exporter: SourceExporter,
    redaction: RedactionCallback,
    selection_evaluator: SelectionEvidenceEvaluator | None = None,
    session_id: str | None = None,
    session_ids: frozenset[str] | None = None,
) -> Iterator[SourceSnapshot]:
    """Freeze one JSONL source into the same replay schema as SQLite sources."""
    if not path.is_file() or path.is_symlink():
        raise SnapshotError("source_unavailable", "the selected source file is unavailable")
    run_dir = Path(tempfile.mkdtemp(prefix="mlflow-import-file-", dir=temp_root))
    snapshot_path = run_dir / SNAPSHOT_FILENAME
    connection: sqlite3.Connection | None = None
    replay: sqlite3.Connection | None = None
    try:
        before = path.stat()
        connection = sqlite3.connect(snapshot_path, isolation_level=None)
        connection.executescript(_SNAPSHOT_SCHEMA)
        os.chmod(snapshot_path, 0o600)
        writer = SnapshotWriter(
            connection, exporter.schema_generation, redaction, selection_evaluator=selection_evaluator
        )
        exporter.bind_connection(connection)
        stats = exporter.export_to_snapshot(writer, redaction, session_id, session_ids)
        stats = writer.finalize()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise SnapshotError("source_changed", "the source file changed while it was being frozen")
        connection.close()
        connection = None
        replay = _open_replay_connection(snapshot_path)
        description = exporter.inspect()
        yield SourceSnapshot(
            fingerprint=source_database_fingerprint(exporter.schema_generation, path),
            snapshot_path=snapshot_path,
            schema_description=description,
            connection=replay,
            stats=stats,
            export_warnings=tuple(getattr(exporter, "export_warnings", ())),
            selection_evidence_enabled=selection_evaluator is not None,
        )
    except SnapshotError:
        raise
    except Exception as error:
        raise SnapshotError("source_snapshot_failed", "the file source snapshot could not be created") from error
    finally:
        if replay is not None:
            replay.close()
        if connection is not None:
            connection.close()
        shutil.rmtree(run_dir, ignore_errors=True)
