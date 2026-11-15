"""Evaluate content filters against frozen effective records, before closure."""

from collections.abc import Iterator
from pathlib import Path

from mlflow_import.source.replay import NormalizedSnapshotAdapter
from mlflow_import.source.selection import SelectionFilters, SessionDiscoverySummary, select_session
from mlflow_import.source.snapshot import SourceSnapshot
from mlflow_import.source.turns import indexed_turn_summaries


def content_matches(snapshot: SourceSnapshot, filters: SelectionFilters) -> Iterator[str]:
    """Use cached turn summaries and all-observation pre-redaction evidence."""
    adapter = NormalizedSnapshotAdapter(snapshot)
    for header in adapter.iter_session_headers():
        turns = sum(1 for _ in indexed_turn_summaries(snapshot, header.session_id))
        summary = SessionDiscoverySummary(
            header.session_id,
            header.time_created.utc if header.time_created else None,
            Path(header.directory) if header.directory else None,
            frozenset(),
            turns,
            creation_time_ns=header.time_created.nanoseconds if header.time_created else None,
        )
        if select_session(summary, filters, adapter.selection_evidence(header.session_id)):
            yield header.session_id
