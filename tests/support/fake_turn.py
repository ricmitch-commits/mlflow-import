"""Fake ``ReplayTurn`` doubles for content-layer fingerprint tests.

Task 3's snapshot tests injected a fake ``SourceExporter`` because the
real adapters did not exist yet; these tests inject a fake
``ReplayTurn`` satisfying the protocol that
:mod:`opencode_mlflow.source.replay` owns, because the concrete
implementation (Task 5's ``build_replay_turn``) does not exist yet.

Cursor semantics mirror the contract the concrete implementation must
provide: :meth:`FakeReplayTurn.iter_records` yields this turn's selected
ordered records, :meth:`iter_session_records` the session-scoped
records, and :meth:`iter_unassigned_records` every other permitted
session record (other turns' records plus records no turn claimed), so
the union of the three cursors is the complete permitted session
payload that artifact snapshots hash.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import cast

from opencode_mlflow.source.model import (
    SessionHeader,
    SourceRecord,
    TimestampValue,
    TurnSummary,
    normalize_timestamp,
)

_UNSET: object = object()


def fake_header(session_id: str = "ses-1", generation: str = "current", **overrides: object) -> SessionHeader:
    """Deterministic session header with identity and mutable fields populated.

    ``overrides`` replace individual fields, mirroring
    :func:`dataclasses.replace` with defaults already in place.
    """
    values: dict[str, object] = {
        "session_id": session_id,
        "adapter_generation": generation,
        "project_id": "proj-1",
        "workspace_id": "ws-1",
        "parent_id": None,
        "fork_session_id": None,
        "fork_boundary": None,
        "slug": "synthetic-session",
        "title": "Synthetic session",
        "directory": "/repo/project",
        "path": "/repo/project/session.jsonl",
        "version": "1.2.3",
        "agent": "synthetic-agent",
        "model": {"id": "provider/model"},
        "metadata": {"kind": "synthetic"},
        "revert": None,
        "permission": {"mode": "default"},
        "cost": 0.25,
        "tokens_input": 120,
        "tokens_output": 80,
        "tokens_reasoning": 12,
        "tokens_cache_read": 40,
        "tokens_cache_write": 4,
        "summary_additions": 3,
        "summary_deletions": 1,
        "summary_files": 2,
        "summary_diffs": {"src/app.py": 11},
        "time_created": normalize_timestamp(1_772_000_000_000, "ms"),
        "time_updated": normalize_timestamp(1_772_000_500_000, "ms"),
        "time_compacting": None,
        "time_archived": None,
        "time_idle": None,
        "time_viewed": None,
        "time_suspended": None,
        "idle_outcome": None,
        "resume_attempts": None,
        "source_fingerprint": "synthetic-database-fingerprint",
        "raw_columns": {"id": session_id, "directory": "/repo/project"},
        "warnings": (),
    }
    values.update(overrides)
    return SessionHeader(**cast("dict[str, object]", values))


def fake_summary(
    session_id: str = "ses-1",
    turn_index: int = 1,
    turn_key: str | None = None,
    record_count: int = 1,
    **overrides: object,
) -> TurnSummary:
    """Deterministic turn summary (identities and counts only, never content)."""
    values: dict[str, object] = {
        "session_id": session_id,
        "turn_index": turn_index,
        "turn_key": turn_key if turn_key is not None else f"turn-synthetic-{turn_index}",
        "record_count": record_count,
    }
    values.update(overrides)
    return TurnSummary(**cast("dict[str, object]", values))


def fake_record(
    source_id: str,
    raw_json: str | None = '{"text": "synthetic prompt"}',
    decoded: object = _UNSET,
    session_id: str = "ses-1",
    source_table: str = "session_message",
    source_type: str = "user",
    seq: int | None = 1,
    classification: str | None = "user",
    timestamp: TimestampValue | None = None,
    raw_columns: Mapping[str, object] | None = None,
) -> SourceRecord:
    """Deterministic source record; ``decoded`` derives from ``raw_json`` unless given.

    Pass ``decoded=None`` with ``raw_json`` set to model a malformed
    payload, and ``raw_json=None`` to model an absent one. ``raw_columns``
    keeps the NULL-versus-absent distinction: ``{"data": None}`` is a
    present NULL column, ``None``/``{}`` an absent one.
    """
    if decoded is _UNSET:
        decoded_value = json.loads(raw_json) if raw_json is not None else None
    else:
        decoded_value = decoded
    return SourceRecord(
        source_table=source_table,
        source_type=source_type,
        source_id=source_id,
        raw_json=raw_json,
        decoded_value=decoded_value,
        raw_columns=dict(raw_columns) if raw_columns else {},
        session_id=session_id,
        sequence_domain="session_message.seq" if source_table == "session_message" else None,
        sequence_value=seq,
        timestamp=timestamp if timestamp is not None else normalize_timestamp(1_772_000_010, "s"),
        classification=classification,
    )


@dataclass
class FakeReplayTurn:
    """In-memory ``ReplayTurn`` double over deterministic records.

    Each cursor returns a fresh iterator per call, exactly like the
    normalized-snapshot cursors the concrete implementation reopens.
    """

    session_header: SessionHeader
    summary: TurnSummary
    turn_records: tuple[SourceRecord, ...] = ()
    session_records: tuple[SourceRecord, ...] = ()
    unassigned_records: tuple[SourceRecord, ...] = ()
    root_input: object | None = None
    root_output: object | None = None

    def iter_records(self) -> Iterator[SourceRecord]:
        """Fresh cursor over this turn's selected ordered records."""
        return iter(list(self.turn_records))

    def iter_session_records(self) -> Iterator[SourceRecord]:
        """Fresh cursor over the session-scoped records."""
        return iter(list(self.session_records))

    def iter_unassigned_records(self) -> Iterator[SourceRecord]:
        """Fresh cursor over every other permitted session record."""
        return iter(list(self.unassigned_records))

    def load_root_input(self, max_bytes: int) -> object | None:
        return self.root_input

    def load_root_output(self, max_bytes: int) -> object | None:
        return self.root_output
