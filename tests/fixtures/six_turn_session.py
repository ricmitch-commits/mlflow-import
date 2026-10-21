"""Sanitized synthetic six-turn session fixture for mapping tests.

One current-generation OpenCode session (``ses-synthetic-1``) with six
logical turns exercising every record shape the turn-to-trace mapping
handles: user prompts, assistant results with and without model-call
metadata, an earlier assistant result, tool calls, reasoning/text/error
parts, a no-result turn, a failed turn, an unassigned system record, and
session-scoped lifecycle rows. Every ID, prompt, path, tool output, and
model name is synthetic; no real session data appears here.

:func:`six_turn_session_fixture` is the factory: it materializes the
fixture database, runs the REAL public pipeline (current adapter →
normalized snapshot → replay adapter → logical-turn summaries →
``build_replay_turn``), and yields a :class:`SixTurnSession` holding the
open snapshot, the six replay turns, and the
:class:`~opencode_mlflow.mlflow.mapping.MappingContext` the mapping is
exercised under. The factory is a context manager because the replay
turns keep fresh cursors over the temporary snapshot; the pytest fixture
wrapper lives in ``tests/conftest.py``.

Turn layout (message ``seq`` values; derived content items ride their
parent message):

- turn 1 — success: user prompt, assistant model call whose parts are
  reasoning, a completed tool call, and the final text answer;
- turn 2 — success: user prompt, an earlier plain assistant result, then
  the final model-call result split into two ordered text parts;
- turn 3 — no result: a user prompt directly followed by the next one;
- turn 4 — explicit failure: a model-call result whose parts are a
  partial text answer and an error part;
- turn 5 — success: a plain assistant result without model metadata;
- turn 6 — success: a single-part model-call result.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from opencode_mlflow import __version__ as IMPORTER_VERSION
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.mlflow.mapping import MappingContext
from opencode_mlflow.source.catalog import CURRENT_CATALOG_V1
from opencode_mlflow.source.current import CurrentAdapter
from opencode_mlflow.source.discovery import ARTIFACT_SCHEMA_VERSION, MAPPING_VERSION
from opencode_mlflow.source.model import SourceRecord
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter, ReplayTurn
from opencode_mlflow.source.snapshot import SourceSnapshot, create_normalized_snapshot
from opencode_mlflow.source.turns import TurnSummary, build_replay_turn, iter_turn_summaries
from tests.support.fake_source import identity_redaction

__all__ = [
    "SESSION_ID",
    "SixTurnSession",
    "build_six_turn_source_db",
    "six_turn_session_fixture",
]

SESSION_ID = "ses-synthetic-1"
"""The fixture session's exact OpenCode session ID."""

_TURNS = 6

_CURRENT_DDL = """
CREATE TABLE session_v2 (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    workspace_id TEXT,
    parent_id TEXT,
    fork_session_id TEXT,
    fork_boundary INTEGER,
    slug TEXT,
    directory TEXT,
    path TEXT,
    title TEXT,
    version TEXT,
    share_url TEXT,
    summary_additions INTEGER,
    summary_deletions INTEGER,
    summary_files INTEGER,
    summary_diffs TEXT,
    metadata TEXT,
    cost REAL,
    tokens_input INTEGER,
    tokens_output INTEGER,
    tokens_reasoning INTEGER,
    tokens_cache_read INTEGER,
    tokens_cache_write INTEGER,
    revert TEXT,
    permission TEXT,
    agent TEXT,
    model TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    time_idle INTEGER,
    time_viewed INTEGER,
    idle_outcome TEXT,
    time_compacting INTEGER,
    time_archived INTEGER,
    time_suspended INTEGER,
    resume_attempts INTEGER
);
CREATE TABLE session_message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    type TEXT NOT NULL,
    seq INTEGER,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
CREATE TABLE session_pending (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    type TEXT,
    data TEXT,
    delivery TEXT,
    admitted_seq INTEGER,
    time_created INTEGER
);
"""

_SESSION_ROW = (
    "INSERT INTO session_v2 VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?, NULL,"
    " 5, 2, 3, ?, ?, ?, 310, 180, 17, 58, 7, NULL, NULL, ?, ?, ?, ?, NULL, NULL, NULL,"
    " NULL, NULL, NULL, NULL)",
    (
        SESSION_ID,
        "proj-synthetic-1",
        "ws-synthetic-1",
        "synthetic-six-turn-slug",
        "/repo/synthetic-project",
        "/repo/synthetic-project/session.jsonl",
        "Synthetic six turn session",
        "synthetic-opencode-version",
        '{"modified": ["synthetic-file-a"]}',
        '{"environment": "synthetic"}',
        0.0055,
        "synthetic-agent",
        '{"provider": "synthetic-provider", "model": "synthetic-model"}',
        1772000000000,
        1772000110000,
    ),
)

_MODEL = {"provider": "synthetic-provider", "model": "synthetic-model"}

_MESSAGES: tuple[tuple[str, str, int, int, object], ...] = (
    # An unassigned system record before the first user prompt.
    (
        "msg-synthetic-system-0",
        "system",
        0,
        1772000005000,
        {"content": [{"type": "text", "text": "synthetic session notice"}]},
    ),
    # Turn 1: success with reasoning, a completed tool call, and a text answer.
    ("msg-synthetic-u1", "user", 1, 1772000010000, {"content": [{"type": "text", "text": "synthetic prompt one"}]}),
    (
        "msg-synthetic-a1",
        "assistant",
        2,
        1772000020000,
        {
            "content": [
                {"type": "reasoning", "text": "synthetic reasoning one"},
                {
                    "type": "tool",
                    "tool": "bash",
                    "call_id": "synthetic-call-1",
                    "state": {
                        "input": {"command": "ls synthetic"},
                        "output": "synthetic file list",
                        "status": "completed",
                    },
                },
                {"type": "text", "text": "synthetic answer one"},
            ],
            "model": _MODEL,
            "providerID": "synthetic-provider",
            "tokens": {"input": 120, "output": 80, "reasoning": 12, "cache": {"read": 40, "write": 4}},
            "cost": 0.0025,
        },
    ),
    # Turn 2: an earlier plain assistant result, then a split final model result.
    ("msg-synthetic-u2", "user", 4, 1772000030000, {"content": [{"type": "text", "text": "synthetic prompt two"}]}),
    (
        "msg-synthetic-a2-early",
        "assistant",
        5,
        1772000035000,
        {"content": [{"type": "text", "text": "synthetic draft answer"}]},
    ),
    (
        "msg-synthetic-a2",
        "assistant",
        6,
        1772000040000,
        {
            "content": [
                {"type": "text", "text": "synthetic final part one"},
                {"type": "text", "text": "synthetic final part two"},
            ],
            "model": _MODEL,
            "tokens": {"input": 90, "output": 60, "reasoning": 0, "cache": {"read": 10, "write": 2}},
            "cost": 0.0015,
        },
    ),
    # Turn 3: no assistant result at all.
    ("msg-synthetic-u3", "user", 8, 1772000050000, {"content": [{"type": "text", "text": "synthetic prompt three"}]}),
    # Turn 4: an explicit failure — a partial text answer plus an error part.
    ("msg-synthetic-u4", "user", 9, 1772000060000, {"content": [{"type": "text", "text": "synthetic prompt four"}]}),
    (
        "msg-synthetic-a4",
        "assistant",
        10,
        1772000070000,
        {
            "content": [
                {"type": "text", "text": "synthetic partial answer four"},
                {"type": "error", "error": {"name": "synthetic-error", "message": "synthetic failure four"}},
            ],
            "model": _MODEL,
            "tokens": {"input": 40, "output": 10, "reasoning": 0, "cache": {"read": 0, "write": 0}},
            "cost": 0.0005,
        },
    ),
    # Turn 5: a plain assistant result without model-call metadata.
    ("msg-synthetic-u5", "user", 11, 1772000080000, {"content": [{"type": "text", "text": "synthetic prompt five"}]}),
    (
        "msg-synthetic-a5",
        "assistant",
        12,
        1772000090000,
        {"content": [{"type": "text", "text": "synthetic plain answer five"}]},
    ),
    # Turn 6: a single-part model-call result.
    ("msg-synthetic-u6", "user", 13, 1772000100000, {"content": [{"type": "text", "text": "synthetic prompt six"}]}),
    (
        "msg-synthetic-a6",
        "assistant",
        14,
        1772000110000,
        {
            "content": [{"type": "text", "text": "synthetic answer six"}],
            "model": _MODEL,
            "tokens": {"input": 60, "output": 30, "reasoning": 5, "cache": {"read": 8, "write": 1}},
            "cost": 0.001,
        },
    ),
)

_PENDING_ROWS = (
    (
        "INSERT INTO session_pending VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "pending-synthetic-1",
            SESSION_ID,
            "tool",
            '{"name": "synthetic-pending-tool", "outcome": "admitted"}',
            "polled",
            2,
            1772000025000,
        ),
    ),
    (
        "INSERT INTO session_pending VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "pending-synthetic-2",
            SESSION_ID,
            "event",
            '{"event": "synthetic-lifecycle-event", "outcome": "observed"}',
            "queued",
            9,
            1772000065000,
        ),
    ),
)


@dataclass
class SixTurnSession:
    """The real-pipeline mapping fixture: replay turns plus mapping context."""

    session_id: str
    turns: list[ReplayTurn]
    summaries: list[TurnSummary]
    context: MappingContext
    snapshot: SourceSnapshot
    source_path: Path

    @property
    def turn_count(self) -> int:
        """The number of logical turns the fixture session reconstructed."""
        return len(self.turns)

    def message_records(self) -> list[SourceRecord]:
        """Every message-container record of the session (all turns plus unassigned)."""
        return [
            record
            for record in self.snapshot_adapter().iter_records(self.session_id)
            if record.derived_child_ordinal is None
        ]

    def part_records(self) -> list[SourceRecord]:
        """Every derived content-part record of the session."""
        return [
            record
            for record in self.snapshot_adapter().iter_records(self.session_id)
            if record.derived_child_ordinal is not None
        ]

    def snapshot_adapter(self) -> NormalizedSnapshotAdapter:
        """A fresh replay adapter over the still-open snapshot."""
        return NormalizedSnapshotAdapter(self.snapshot)


def build_six_turn_source_db(temp_root: Path) -> Path:
    """Materialize the sanitized synthetic fixture database.

    Public builder for boundary-level tests (for example the BDD
    acceptance scenarios) that drive the full import pipeline over the
    fixture's exact rows; :func:`six_turn_session_fixture` builds on it.
    """
    source = temp_root / "six-turn-source.db"
    connection = sqlite3.connect(source)
    try:
        connection.executescript(_CURRENT_DDL)
        connection.execute(*_SESSION_ROW)
        for message_id, message_type, seq, created, payload in _MESSAGES:
            connection.execute(
                "INSERT INTO session_message VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    SESSION_ID,
                    message_type,
                    seq,
                    created,
                    created,
                    json.dumps(payload),
                ),
            )
        for sql, parameters in _PENDING_ROWS:
            connection.execute(sql, parameters)
        connection.commit()
    finally:
        connection.close()
    return source


@contextmanager
def six_turn_session_fixture(temp_root: Path, source: Path | None = None) -> Iterator[SixTurnSession]:
    """Build the fixture through the real pipeline and keep its snapshot open.

    Fixture database → current adapter → normalized snapshot → replay
    adapter → logical-turn summaries → replay turns, plus the mapping
    context the write phase would carry (versions, schema, redaction
    mode, experiment, source database fingerprint, session counts).
    ``source`` materializes the fixture's exact rows into a caller-chosen
    database path (the real-server integration tests reuse the very file
    an import consumed, so the independently mapped expectations —
    including the source database fingerprint, which covers file
    identity statistics — match the imported traces exactly); the
    default builds a fresh fixture database under ``temp_root``.
    """
    if source is None:
        source = build_six_turn_source_db(temp_root)
    connection = sqlite3.connect(source)
    try:
        exporter = CurrentAdapter(connection, CURRENT_CATALOG_V1)
    finally:
        connection.close()
    with create_normalized_snapshot(source, temp_root, exporter, identity_redaction) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = next(adapter.iter_session_headers())
        if header.session_id != SESSION_ID:
            raise AssertionError("the six-turn fixture reconstructed an unexpected session")
        records = list(adapter.iter_records(SESSION_ID))
        summaries = list(iter_turn_summaries(records))
        if len(summaries) != _TURNS:
            raise AssertionError(f"the six-turn fixture reconstructed {len(summaries)} turns, expected {_TURNS}")
        turns = [build_replay_turn(snapshot, header, summary) for summary in summaries]
        context = MappingContext(
            importer_version=IMPORTER_VERSION,
            mapping_version=MAPPING_VERSION,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            source_schema=header.adapter_generation,
            redaction_mode=RedactionMode.VERBATIM,
            experiment_id="exp-synthetic-six-turn",
            catalog_version=snapshot.schema_description.catalog_version,
            source_database_fingerprint=snapshot.fingerprint,
            inline_limit_bytes=None,
            logical_turn_count=len(summaries),
            source_message_count=sum(1 for record in records if record.derived_child_ordinal is None),
            source_part_count=sum(1 for record in records if record.derived_child_ordinal is not None),
        )
        yield SixTurnSession(
            session_id=header.session_id,
            turns=turns,
            summaries=summaries,
            context=context,
            snapshot=snapshot,
            source_path=source,
        )
