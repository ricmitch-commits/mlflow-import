"""Unit tests for the streaming, sanitized, disk-backed discovery plan.

Discovery is driven through the public source pipeline: a sanitized
synthetic fixture database, the real generation adapter through the
snapshot driver, the replay adapter over the normalized snapshot, and
:func:`discover_sessions` with :class:`SelectionFilters`. The plan lands
in a restrictive temporary directory; the reader's rejection contract is
pinned against hand-built plans written through
:class:`DiscoveryPlanWriter`.
"""

from __future__ import annotations

import gc
import json
import sqlite3
import stat
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, cast

import pytest

from opencode_mlflow import __version__ as importer_version
from opencode_mlflow.content.fingerprint import (
    build_effective_turn,
    render_fingerprint,
    session_context_fingerprint,
    source_fingerprint,
)
from opencode_mlflow.content.redaction import DeterministicRedactor
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.errors import ImporterError, SourceCompatibilityError
from opencode_mlflow.source import discovery as discovery_module
from opencode_mlflow.source.catalog import CURRENT_CATALOG_V1, LEGACY_CATALOG_V1
from opencode_mlflow.source.current import CurrentAdapter
from opencode_mlflow.source.discovery import (
    ARTIFACT_SCHEMA_VERSION,
    DISCOVERY_PLAN_SCHEMA_VERSION,
    MAPPING_VERSION,
    DiscoveryPlanReader,
    DiscoveryPlanWriter,
    DiscoveryTurnEntry,
    SessionScanEntry,
    discover_sessions,
    extract_model_identity,
)
from opencode_mlflow.source.legacy import LegacyAdapter
from opencode_mlflow.source.model import (
    AdapterGeneration,
    SchemaDescription,
    SessionHeader,
    SourceRecord,
    TurnSummary,
    normalize_timestamp,
)
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter
from opencode_mlflow.source.selection import SelectionEvidenceEvaluator, SelectionFilters
from opencode_mlflow.source.snapshot import SourceSnapshot, create_normalized_snapshot
from opencode_mlflow.source.turns import NO_ASSISTANT_RESULT_OUTCOME, build_replay_turn, iter_turn_summaries
from tests.support.fake_redactor import IdentityRedactor
from tests.support.fake_source import identity_redaction
from tests.support.sqlite_builder import build_fixture_db

pytestmark = pytest.mark.unit

_CREATED_MS = 1_772_000_000_000

_PROMPT_JSON = '{"content": [{"type": "text", "text": "synthetic discovery prompt sentinel"}]}'
_REPLY_JSON = '{"content": [{"type": "text", "text": "synthetic discovery reply sentinel"}]}'

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
"""

_LEGACY_DDL = """
CREATE TABLE session (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    workspace_id TEXT,
    parent_id TEXT,
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
    time_compacting INTEGER,
    time_archived INTEGER
);
CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
CREATE TABLE part (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
"""

Insert = tuple[str, tuple[object, ...]]


def _current_session(
    session_id: str,
    *,
    directory: str | None = None,
    created_ms: int | None = _CREATED_MS,
    model: str | None = None,
) -> Insert:
    columns = ["id", "time_created", "time_updated"]
    values: list[object] = [session_id, created_ms, created_ms]
    if directory is not None:
        columns.append("directory")
        values.append(directory)
    if model is not None:
        columns.append("model")
        values.append(model)
    placeholders = ", ".join("?" for _ in columns)
    return (f"INSERT INTO session_v2 ({', '.join(columns)}) VALUES ({placeholders})", tuple(values))


def _current_message(message_id: str, session_id: str, message_type: str, seq: int, data: str) -> Insert:
    return (
        "INSERT INTO session_message (id, session_id, type, seq, time_created, time_updated, data)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (message_id, session_id, message_type, seq, _CREATED_MS + seq * 1_000, _CREATED_MS + seq * 1_000, data),
    )


def _legacy_session(
    session_id: str,
    *,
    directory: str | None = None,
    created_ms: int | None = _CREATED_MS,
    model: str | None = None,
) -> Insert:
    columns = ["id", "time_created", "time_updated"]
    values: list[object] = [session_id, created_ms, created_ms]
    if directory is not None:
        columns.append("directory")
        values.append(directory)
    if model is not None:
        columns.append("model")
        values.append(model)
    placeholders = ", ".join("?" for _ in columns)
    return (f"INSERT INTO session ({', '.join(columns)}) VALUES ({placeholders})", tuple(values))


def _legacy_message(message_id: str, session_id: str, data: str, *, time_created: int = _CREATED_MS + 1_000) -> Insert:
    return (
        "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
        (message_id, session_id, time_created, time_created, data),
    )


def _legacy_part(part_id: str, message_id: str, session_id: str, data: str) -> Insert:
    return (
        "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?, ?)",
        (part_id, message_id, session_id, _CREATED_MS + 2_000, _CREATED_MS + 2_000, data),
    )


def _exporter_for(generation: str, source: Path) -> object:
    connection = sqlite3.connect(source)
    try:
        if generation == "current":
            return CurrentAdapter(connection, CURRENT_CATALOG_V1)
        return LegacyAdapter(connection, LEGACY_CATALOG_V1)
    finally:
        connection.close()


@contextmanager
def _replay_adapter(
    tmp_path: Path,
    name: str,
    generation: str,
    inserts: Sequence[Insert],
    *,
    selection_filters: SelectionFilters | None = None,
) -> Iterator[NormalizedSnapshotAdapter]:
    """Build fixture DB → normalized snapshot → replay adapter, kept open."""
    source = tmp_path / f"{name}.db"
    connection = sqlite3.connect(source)
    try:
        connection.executescript(_CURRENT_DDL if generation == "current" else _LEGACY_DDL)
        for sql, parameters in inserts:
            connection.execute(sql, parameters)
        connection.commit()
    finally:
        connection.close()
    exporter = _exporter_for(generation, source)
    with create_normalized_snapshot(
        source,
        tmp_path,
        exporter,
        identity_redaction,
        selection_evaluator=(SelectionEvidenceEvaluator(selection_filters) if selection_filters is not None else None),
    ) as snapshot:
        yield NormalizedSnapshotAdapter(snapshot)


@contextmanager
def _fixture_replay_adapter(tmp_path: Path, name: str) -> Iterator[NormalizedSnapshotAdapter]:
    """The same pipeline over a committed sanitized fixture database."""
    source = tmp_path / f"{name}.db"
    connection = build_fixture_db(source, name)
    connection.close()
    exporter = _exporter_for("legacy" if name == "legacy" else "current", source)
    with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
        yield NormalizedSnapshotAdapter(snapshot)


class _HeaderWrappingAdapter:
    """Test double wrapping a replay adapter's header stream.

    Without options it observes the adapter lazily (recording the order
    sessions, records, and session-scoped records are consumed).
    ``reverse`` materializes and reverses the header order; ``rewrite``
    replaces selected sessions' adapter generation.
    """

    def __init__(
        self,
        adapter: NormalizedSnapshotAdapter,
        *,
        reverse: bool = False,
        rewrite: Mapping[str, AdapterGeneration] | None = None,
    ) -> None:
        self._adapter = adapter
        self._reverse = reverse
        self._rewrite = dict(rewrite or {})
        self.events: list[str] = []

    @property
    def snapshot(self) -> SourceSnapshot:
        return self._adapter.snapshot

    @property
    def schema_generation(self) -> str:
        return self._adapter.schema_generation

    def inspect(self) -> SchemaDescription:
        return self._adapter.inspect()

    def iter_session_headers(self) -> Iterator[SessionHeader]:
        headers: Iterator[SessionHeader] = self._adapter.iter_session_headers()
        if self._reverse:
            headers = iter(reversed(list(headers)))
        for header in headers:
            self.events.append(f"header:{header.session_id}")
            generation = self._rewrite.get(header.session_id)
            yield header if generation is None else replace(header, adapter_generation=generation)

    def iter_records(self, session_id: str) -> Iterator[SourceRecord]:
        self.events.append(f"records:{session_id}")
        return self._adapter.iter_records(session_id)

    def iter_session_scoped_records(self, session_id: str) -> Iterator[SourceRecord]:
        self.events.append(f"scoped:{session_id}")
        return self._adapter.iter_session_scoped_records(session_id)

    def selection_evidence(self, session_id: str):
        return self._adapter.selection_evidence(session_id)


class _LegacyReplayAdapterWithoutSelectionEvidence:
    """Existing replay adapter shape from before optional selection evidence."""

    def __init__(self, adapter: NormalizedSnapshotAdapter) -> None:
        self._adapter = adapter
        self.schema_generation = adapter.schema_generation

    @property
    def snapshot(self) -> SourceSnapshot:
        return self._adapter.snapshot

    def inspect(self) -> SchemaDescription:
        return self._adapter.inspect()

    def iter_session_headers(self) -> Iterator[SessionHeader]:
        return self._adapter.iter_session_headers()

    def iter_records(self, session_id: str) -> Iterator[SourceRecord]:
        return self._adapter.iter_records(session_id)

    def iter_session_scoped_records(self, session_id: str) -> Iterator[SourceRecord]:
        return self._adapter.iter_session_scoped_records(session_id)


def _plan_path(restrictive_tmp_path: Path) -> Path:
    return restrictive_tmp_path / "discovery-plan.jsonl"


# ---------------------------------------------------------------------------
# Model identity extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("provider/model", "provider/model"),
        ("  provider/model ", "provider/model"),
        ("", None),
        ("   ", None),
        ({"provider": "p", "model": "m"}, "p/m"),
        ({"provider": "p"}, "p/"),
        ({"model": "m"}, "/m"),
        ({"provider": "p", "model": None}, "p/"),
        ({"provider": 7, "model": "m"}, "/m"),
        ({}, None),
        (None, None),
        (42, None),
        ([], None),
    ],
)
def test_extract_model_identity_builds_only_explicit_forms(value: object, expected: str | None) -> None:
    assert extract_model_identity(value) == expected


# ---------------------------------------------------------------------------
# Discovery over the public source pipeline
# ---------------------------------------------------------------------------


def test_discovery_writes_one_scan_entry_per_compatible_session(tmp_path: Path, restrictive_tmp_path: Path) -> None:
    inserts: list[Insert] = [
        _current_session("ses-a", directory="/repo/project"),
        _current_session("ses-b", directory="/repo/other"),
        _current_message("msg-a1", "ses-a", "user", 1, _PROMPT_JSON),
        _current_message("msg-a2", "ses-a", "assistant", 2, _REPLY_JSON),
        _current_message("msg-b1", "ses-b", "user", 1, _PROMPT_JSON),
    ]
    with _replay_adapter(tmp_path, "two-sessions", "current", inserts) as adapter:
        plan_path = _plan_path(restrictive_tmp_path)
        stats = discover_sessions(adapter, SelectionFilters(), plan_path)

        scans = list(DiscoveryPlanReader(plan_path).iter_session_scans())

    assert stats.session_scan_count == 2
    assert stats.selected_session_count == 2
    assert stats.incompatible_session_count == 0
    assert stats.plan_path == plan_path
    assert [scan.session_id for scan in scans] == ["ses-a", "ses-b"]
    assert all(scan.selected for scan in scans)
    assert all(scan.complete_scan for scan in scans)
    assert [scan.logical_turn_count for scan in scans] == [1, 1]


def test_discovery_without_directory_or_model_filter_supports_legacy_adapter_shape(
    tmp_path: Path, restrictive_tmp_path: Path
) -> None:
    inserts: list[Insert] = [
        _current_session("ses-legacy-adapter-shape"),
        _current_message("msg-legacy-adapter-shape", "ses-legacy-adapter-shape", "user", 1, _PROMPT_JSON),
    ]
    with _replay_adapter(tmp_path, "legacy-adapter-shape", "current", inserts) as replay:
        adapter = _LegacyReplayAdapterWithoutSelectionEvidence(replay)
        stats = discover_sessions(adapter, SelectionFilters(), _plan_path(restrictive_tmp_path))

    assert stats.selected_session_count == 1


def test_discovery_requires_selection_evidence_when_directory_or_model_filter_is_used(
    tmp_path: Path, restrictive_tmp_path: Path
) -> None:
    inserts: list[Insert] = [
        _current_session("ses-missing-evidence"),
        _current_message("msg-missing-evidence", "ses-missing-evidence", "user", 1, _PROMPT_JSON),
    ]
    with _replay_adapter(tmp_path, "missing-evidence", "current", inserts) as replay:
        adapter = _LegacyReplayAdapterWithoutSelectionEvidence(replay)
        with pytest.raises(SourceCompatibilityError) as excinfo:
            discover_sessions(
                adapter,
                SelectionFilters(model="provider/model"),
                _plan_path(restrictive_tmp_path),
            )

    assert excinfo.value.code == "selection_evidence_unsupported"


def test_discovery_selects_only_sessions_matching_every_supplied_filter(
    tmp_path: Path, restrictive_tmp_path: Path
) -> None:
    inserts: list[Insert] = [
        _current_session("ses-a", directory="/repo/project"),
        _current_session("ses-b", directory="/repo/projector"),
        _current_message("msg-a1", "ses-a", "user", 1, _PROMPT_JSON),
        _current_message("msg-a2", "ses-a", "assistant", 2, _REPLY_JSON),
        _current_message("msg-b1", "ses-b", "user", 1, _PROMPT_JSON),
    ]
    filters = SelectionFilters(directory=Path("/repo/project"))
    with _replay_adapter(tmp_path, "filtered", "current", inserts, selection_filters=filters) as adapter:
        plan_path = _plan_path(restrictive_tmp_path)
        stats = discover_sessions(adapter, filters, plan_path)

        reader = DiscoveryPlanReader(plan_path)
        scans = {scan.session_id: scan for scan in reader.iter_session_scans()}
        turns = list(reader.iter_turns())

    assert stats.session_scan_count == 2
    assert stats.selected_session_count == 1
    assert stats.turn_entry_count == 1
    assert scans["ses-a"].selected is True
    assert scans["ses-a"].retirement_scope is True
    # The unselected session still gets a scan entry (with its turn count),
    # but none of its turns enter the plan.
    assert scans["ses-b"].selected is False
    assert scans["ses-b"].retirement_scope is False
    assert scans["ses-b"].logical_turn_count == 1
    assert [turn.session_id for turn in turns] == ["ses-a"]


def test_discovery_uses_pre_redaction_filter_evidence_without_putting_it_in_the_plan(
    tmp_path: Path, restrictive_tmp_path: Path
) -> None:
    raw_directory = "/repo/password=directory-secret/project"
    requested_directory = Path("/repo/password=directory-secret")
    raw_model = "provider/sk-abcdefghijklmnop"
    reply = json.dumps(
        {
            "content": [{"type": "text", "text": "synthetic reply"}],
            "model": {"provider": "provider", "model": "sk-abcdefghijklmnop"},
        }
    )
    inserts: list[Insert] = [
        _current_session("ses-evidence", directory=raw_directory),
        _current_message("msg-evidence-user", "ses-evidence", "user", 1, _PROMPT_JSON),
        _current_message("msg-evidence-assistant", "ses-evidence", "assistant", 2, reply),
    ]
    source = tmp_path / "pre-redaction-evidence.db"
    connection = sqlite3.connect(source)
    try:
        connection.executescript(_CURRENT_DDL)
        for sql, parameters in inserts:
            connection.execute(sql, parameters)
        connection.commit()
    finally:
        connection.close()
    filters = SelectionFilters(directory=requested_directory, model=raw_model)
    exporter = _exporter_for("current", source)
    redactor = DeterministicRedactor(RedactionMode.REDACTED)

    with create_normalized_snapshot(
        source,
        tmp_path,
        exporter,
        redactor.redact,
        selection_evaluator=SelectionEvidenceEvaluator(filters),
    ) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = next(adapter.iter_session_headers())
        evidence = adapter.selection_evidence("ses-evidence")
        plan_path = _plan_path(restrictive_tmp_path)
        stats = discover_sessions(adapter, filters, plan_path)
        scan = next(DiscoveryPlanReader(plan_path).iter_session_scans())

    assert header.directory != raw_directory
    assert evidence.directory_match is True
    assert evidence.model_match is True
    assert stats.selected_session_count == 1
    assert scan.selected is True
    plan_text = plan_path.read_text(encoding="utf-8")
    assert raw_directory not in plan_text
    assert raw_model not in plan_text
    assert "directory_match" not in plan_text
    assert "model_match" not in plan_text


def test_discovery_writes_one_turn_entry_per_logical_turn(tmp_path: Path, restrictive_tmp_path: Path) -> None:
    inserts: list[Insert] = [
        _current_session("ses-a", directory="/repo/project"),
        _current_message("msg-a1", "ses-a", "user", 1, _PROMPT_JSON),
        _current_message("msg-a2", "ses-a", "assistant", 2, _REPLY_JSON),
        _current_message("msg-a3", "ses-a", "user", 3, _PROMPT_JSON),
    ]
    with _replay_adapter(tmp_path, "two-turns", "current", inserts) as adapter:
        plan_path = _plan_path(restrictive_tmp_path)
        stats = discover_sessions(adapter, SelectionFilters(), plan_path)

        turns = list(DiscoveryPlanReader(plan_path).iter_turns())
        summaries = list(iter_turn_summaries(adapter.iter_records("ses-a")))

    assert stats.turn_entry_count == 2
    assert len(turns) == len(summaries) == 2
    for entry, summary in zip(turns, summaries, strict=True):
        assert entry.session_id == "ses-a"
        assert entry.turn_key == summary.turn_key
        assert entry.turn_index == summary.turn_index
        assert entry.adapter_generation == "current"
        assert entry.planned_action == "import"
        assert entry.record_count == summary.record_count
        assert entry.user_record_count == summary.user_record_count
        assert entry.assistant_record_count == summary.assistant_record_count
        assert entry.terminal_outcome == summary.terminal_outcome
        assert entry.start_time_ns == summary.start_time.nanoseconds
        assert entry.end_time_ns == summary.end_time.nanoseconds
        assert entry.warnings == summary.warnings


def test_discovery_fingerprints_use_the_task4a_functions(tmp_path: Path, restrictive_tmp_path: Path) -> None:
    inserts: list[Insert] = [
        _current_session("ses-a", directory="/repo/project"),
        _current_message("msg-a1", "ses-a", "user", 1, _PROMPT_JSON),
        _current_message("msg-a2", "ses-a", "assistant", 2, _REPLY_JSON),
    ]
    with _replay_adapter(tmp_path, "fingerprints", "current", inserts) as adapter:
        plan_path = _plan_path(restrictive_tmp_path)
        discover_sessions(adapter, SelectionFilters(), plan_path)

        reader = DiscoveryPlanReader(plan_path)
        scan = next(reader.iter_session_scans())
        turns = {turn.turn_key: turn for turn in DiscoveryPlanReader(plan_path).iter_turns()}
        header = next(adapter.iter_session_headers())
        for summary in iter_turn_summaries(adapter.iter_records("ses-a")):
            entry = turns[summary.turn_key]
            replay = build_replay_turn(adapter.snapshot, header, summary)
            effective = build_effective_turn(replay, IdentityRedactor())

            assert entry.source_fingerprint == source_fingerprint(effective, RedactionMode.VERBATIM)
            assert entry.session_context_fingerprint == session_context_fingerprint(effective)
            assert entry.render_fingerprint == render_fingerprint(
                entry.source_fingerprint,
                ARTIFACT_SCHEMA_VERSION,
                MAPPING_VERSION,
                importer_version,
                RedactionMode.VERBATIM,
            )
            assert entry.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
            assert entry.mapping_version == MAPPING_VERSION
            assert entry.importer_version == importer_version
            assert entry.redaction_mode == RedactionMode.VERBATIM.value
            # The scan's session-identity fingerprint is the same Task 4A
            # session context fingerprint every turn of the session carries.
            assert scan.session_context_fingerprint == entry.session_context_fingerprint


def test_discovery_collects_model_identities_from_session_default_and_message_levels(
    tmp_path: Path, restrictive_tmp_path: Path
) -> None:
    default_model = '{"provider": "default-provider", "model": "default-model"}'
    message_model_data = (
        '{"content": [{"type": "text", "text": "synthetic discovery reply sentinel"}],'
        ' "model": {"provider": "message-provider", "model": "message-model"}}'
    )
    inserts: list[Insert] = [
        _current_session("ses-default", directory="/repo/one", model=default_model),
        _current_session("ses-message", directory="/repo/two"),
        _current_message("msg-d1", "ses-default", "user", 1, _PROMPT_JSON),
        _current_message("msg-m1", "ses-message", "user", 1, _PROMPT_JSON),
        _current_message("msg-m2", "ses-message", "assistant", 2, message_model_data),
    ]
    with _replay_adapter(
        tmp_path,
        "model-levels-default",
        "current",
        inserts,
        selection_filters=SelectionFilters(model="default-provider/default-model"),
    ) as adapter:
        plan_path = _plan_path(restrictive_tmp_path)
        discover_sessions(adapter, SelectionFilters(model="default-provider/default-model"), plan_path)
        default_scans = {scan.session_id: scan for scan in DiscoveryPlanReader(plan_path).iter_session_scans()}

    with _replay_adapter(
        tmp_path,
        "model-levels-message",
        "current",
        inserts,
        selection_filters=SelectionFilters(model="message-provider/message-model"),
    ) as adapter:
        discover_sessions(adapter, SelectionFilters(model="message-provider/message-model"), plan_path)
        message_scans = {scan.session_id: scan for scan in DiscoveryPlanReader(plan_path).iter_session_scans()}

        turns = list(DiscoveryPlanReader(plan_path).iter_turns())

    assert default_scans["ses-default"].selected is True
    assert default_scans["ses-message"].selected is False
    assert message_scans["ses-default"].selected is False
    assert message_scans["ses-message"].selected is True
    # Turn entries carry the identities found in the turn's own records;
    # the session default never leaks into a turn entry.
    assert all(turn.session_id == "ses-message" for turn in turns)
    assert {turn.model_identities for turn in turns} == {frozenset({"message-provider/message-model"})}


def test_discovery_collects_model_identities_from_part_level(tmp_path: Path, restrictive_tmp_path: Path) -> None:
    part_model_data = '{"type": "step-finish", "model": {"provider": "part-provider", "model": "part-model"}}'
    inserts: list[Insert] = [
        _legacy_session("ses-part", directory="/repo/three"),
        _legacy_message("msg-p1", "ses-part", '{"role": "user", "content": "synthetic discovery prompt sentinel"}'),
        _legacy_message("msg-p2", "ses-part", '{"role": "assistant"}'),
        _legacy_part("part-p1", "msg-p2", "ses-part", part_model_data),
    ]
    filters = SelectionFilters(model="part-provider/part-model")
    with _replay_adapter(tmp_path, "part-model", "legacy", inserts, selection_filters=filters) as adapter:
        plan_path = _plan_path(restrictive_tmp_path)
        discover_sessions(adapter, filters, plan_path)

        scans = {scan.session_id: scan for scan in DiscoveryPlanReader(plan_path).iter_session_scans()}
        turns = list(DiscoveryPlanReader(plan_path).iter_turns())

    assert scans["ses-part"].selected is True
    assert turns[0].model_identities == frozenset({"part-provider/part-model"})


def test_discovery_missing_creation_time_is_ineligible_under_time_filters(
    tmp_path: Path, restrictive_tmp_path: Path
) -> None:
    # The committed legacy fixture carries ses-legacy-003 without a
    # creation timestamp, sorting after every timestamped session.
    with _fixture_replay_adapter(tmp_path, "legacy") as adapter:
        plan_path = _plan_path(restrictive_tmp_path)
        discover_sessions(adapter, SelectionFilters(from_time=datetime(2026, 1, 1, tzinfo=UTC)), plan_path)
        filtered = {scan.session_id: scan.selected for scan in DiscoveryPlanReader(plan_path).iter_session_scans()}

        discover_sessions(adapter, SelectionFilters(), plan_path)
        unfiltered = {scan.session_id: scan.selected for scan in DiscoveryPlanReader(plan_path).iter_session_scans()}

    assert filtered == {"ses-legacy-001": True, "ses-legacy-002": True, "ses-legacy-003": False}
    assert unfiltered == {"ses-legacy-001": True, "ses-legacy-002": True, "ses-legacy-003": True}


def test_discovery_writes_a_tombstone_for_a_selected_complete_zero_turn_scan(
    tmp_path: Path, restrictive_tmp_path: Path
) -> None:
    inserts: list[Insert] = [
        _current_session("ses-empty", directory="/repo/project"),
        _current_message(
            "msg-e1",
            "ses-empty",
            "system",
            1,
            '{"content": [{"type": "text", "text": "synthetic discovery notice"}]}',
        ),
    ]
    with _replay_adapter(tmp_path, "zero-turns", "current", inserts) as adapter:
        plan_path = _plan_path(restrictive_tmp_path)
        stats = discover_sessions(adapter, SelectionFilters(), plan_path)

        reader = DiscoveryPlanReader(plan_path)
        scan = next(reader.iter_session_scans())
        turns = list(reader.iter_turns())

    assert scan.session_id == "ses-empty"
    assert scan.logical_turn_count == 0
    assert scan.record_count == 2  # the system message and its derived content item
    assert scan.selected is True
    assert scan.complete_scan is True
    assert scan.tombstone is True
    assert scan.retirement_scope is True
    assert turns == []
    assert stats.turn_entry_count == 0
    assert stats.tombstone_count == 1


def test_discovery_does_not_tombstone_unselected_zero_turn_scans(tmp_path: Path, restrictive_tmp_path: Path) -> None:
    inserts: list[Insert] = [
        _current_session("ses-empty", directory="/repo/project"),
        _current_message("msg-e1", "ses-empty", "user", 1, _PROMPT_JSON),
        _current_message("msg-e2", "ses-empty", "user", 2, _PROMPT_JSON),
    ]
    with _replay_adapter(tmp_path, "unselected-empty", "current", inserts) as adapter:
        plan_path = _plan_path(restrictive_tmp_path)
        stats = discover_sessions(adapter, SelectionFilters(min_user_turns=3), plan_path)

        scan = next(DiscoveryPlanReader(plan_path).iter_session_scans())

    assert scan.selected is False
    assert scan.logical_turn_count == 2
    assert scan.tombstone is False
    assert scan.retirement_scope is False
    assert stats.tombstone_count == 0
    assert stats.turn_entry_count == 0


def test_discovery_plan_contains_no_transcript_or_source_content(tmp_path: Path, restrictive_tmp_path: Path) -> None:
    secret_sentinels = ("credential-sentinel", "SECRET-ACCOUNT-TOKEN-SENTINEL", "SECRET-SHARE-SENTINEL")
    inserts: list[Insert] = [
        _current_session("ses-a", directory="/repo/project"),
        _current_message("msg-a1", "ses-a", "user", 1, _PROMPT_JSON),
        _current_message("msg-a2", "ses-a", "assistant", 2, _REPLY_JSON),
    ]
    with _replay_adapter(tmp_path, "sanitized", "current", inserts) as adapter:
        plan_path = _plan_path(restrictive_tmp_path)
        discover_sessions(adapter, SelectionFilters(), plan_path)

    text = plan_path.read_text(encoding="utf-8")
    assert "synthetic discovery prompt sentinel" not in text
    assert "synthetic discovery reply sentinel" not in text
    for sentinel in secret_sentinels:
        assert sentinel not in text
    # Every plan line stays inside the sanitized key whitelist.
    scan_keys = {
        "kind",
        "session_id",
        "adapter_generation",
        "selected",
        "complete_scan",
        "tombstone",
        "logical_turn_count",
        "record_count",
        "creation_time_ns",
        "session_context_fingerprint",
        "warnings",
        "catalog_version",
    }
    turn_keys = scan_keys | {
        "turn_index",
        "turn_key",
        "planned_action",
        "start_time_ns",
        "end_time_ns",
        "user_record_count",
        "assistant_record_count",
        "terminal_outcome",
        "model_identities",
        "source_fingerprint",
        "render_fingerprint",
        "artifact_schema_version",
        "mapping_version",
        "importer_version",
        "redaction_mode",
    }
    allowed = {"kind", "plan_schema_version"} | scan_keys | turn_keys
    for line in text.splitlines():
        payload = json.loads(line)
        assert isinstance(payload, dict)
        assert set(payload) <= allowed
        assert payload["kind"] in ("plan_header", "session_scan", "turn")


def test_discovery_plan_file_is_owner_only(tmp_path: Path, restrictive_tmp_path: Path) -> None:
    inserts: list[Insert] = [
        _current_session("ses-a", directory="/repo/project"),
        _current_message("msg-a1", "ses-a", "user", 1, _PROMPT_JSON),
    ]
    with _replay_adapter(tmp_path, "permissions", "current", inserts) as adapter:
        plan_path = _plan_path(restrictive_tmp_path)
        discover_sessions(adapter, SelectionFilters(), plan_path)

    assert stat.S_IMODE(plan_path.stat().st_mode) == 0o600


def test_discovery_consumes_headers_one_session_at_a_time(tmp_path: Path, restrictive_tmp_path: Path) -> None:
    inserts: list[Insert] = [
        _current_session("ses-a", directory="/repo/project", created_ms=_CREATED_MS),
        _current_session("ses-b", directory="/repo/other", created_ms=_CREATED_MS + 1_000),
        _current_message("msg-a1", "ses-a", "user", 1, _PROMPT_JSON),
        _current_message("msg-b1", "ses-b", "user", 1, _PROMPT_JSON),
    ]
    with _replay_adapter(tmp_path, "streaming", "current", inserts) as adapter:
        spy = _HeaderWrappingAdapter(adapter)
        discover_sessions(spy, SelectionFilters(), _plan_path(restrictive_tmp_path))

    # Each session is fully scanned before the next header is pulled, so
    # the adapter's header stream is consumed lazily, never materialized
    # merely to sort.
    assert spy.events == [
        "header:ses-a",
        "records:ses-a",
        "scoped:ses-a",
        "header:ses-b",
        "records:ses-b",
        "scoped:ses-b",
    ]


def test_discovery_spools_high_turn_cardinality_summaries_before_writing(
    restrictive_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn-summary objects are not retained while selected entries are written."""
    session_id = "ses-many-turns"
    header = SessionHeader(
        session_id=session_id,
        adapter_generation="current",
        time_created=normalize_timestamp(_CREATED_MS, "ms"),
    )
    references: list[weakref.ReferenceType[TurnSummary]] = []
    turn_count = 10_000

    def summary_stream(_records: Iterable[SourceRecord]) -> Iterator[TurnSummary]:
        for index in range(turn_count):
            summary = TurnSummary(
                session_id=session_id,
                turn_index=index + 1,
                turn_key=f"turn-{index:064x}",
                record_count=1,
                user_record_count=1,
                terminal_outcome=NO_ASSISTANT_RESULT_OUTCOME,
            )
            references.append(weakref.ref(summary))
            yield summary

    def turn_entry(
        _snapshot: object,
        scan_header: SessionHeader,
        summary: TurnSummary,
        session_fingerprint: str,
        _catalog_version: str,
    ) -> DiscoveryTurnEntry:
        if summary.turn_index == 1:
            gc.collect()
            # At most the loop's current object may still be alive. A
            # list-based discovery pass keeps all 10,000 source summaries
            # alive here.
            assert sum(reference() is not None for reference in references) <= 1
        return DiscoveryTurnEntry(
            session_id=scan_header.session_id,
            turn_index=summary.turn_index,
            turn_key=summary.turn_key,
            adapter_generation="current",
            planned_action="import",
            start_time_ns=None,
            end_time_ns=None,
            record_count=summary.record_count,
            user_record_count=summary.user_record_count,
            assistant_record_count=summary.assistant_record_count,
            terminal_outcome=summary.terminal_outcome,
            model_identities=frozenset(),
            warnings=(),
            session_context_fingerprint=session_fingerprint,
            source_fingerprint="a" * 64,
            render_fingerprint="b" * 64,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            mapping_version=MAPPING_VERSION,
            importer_version=importer_version,
            redaction_mode=RedactionMode.VERBATIM.value,
        )

    class _Adapter:
        def iter_records(self, _session_id: str) -> Iterator[SourceRecord]:
            return iter(())

        def iter_session_scoped_records(self, _session_id: str) -> Iterator[SourceRecord]:
            return iter(())

    monkeypatch.setattr(discovery_module, "iter_turn_summaries", summary_stream)
    monkeypatch.setattr(discovery_module, "_turn_entry_for", turn_entry)
    plan_path = restrictive_tmp_path / "many-turns-plan.jsonl"
    with DiscoveryPlanWriter(plan_path) as writer:
        outcome = discovery_module._scan_session(
            object(),
            _Adapter(),
            header,
            SelectionFilters(),
            writer,
            "catalog-v1",
        )

    assert outcome.turn_entry_count == turn_count
    assert len(list(DiscoveryPlanReader(plan_path).iter_turns())) == turn_count


def test_discovery_rejects_out_of_order_session_headers(tmp_path: Path, restrictive_tmp_path: Path) -> None:
    inserts: list[Insert] = [
        _current_session("ses-a", directory="/repo/project", created_ms=_CREATED_MS),
        _current_session("ses-b", directory="/repo/other", created_ms=_CREATED_MS + 1_000),
        _current_message("msg-a1", "ses-a", "user", 1, _PROMPT_JSON),
    ]
    with _replay_adapter(tmp_path, "out-of-order", "current", inserts) as adapter:
        reversed_adapter = _HeaderWrappingAdapter(adapter, reverse=True)
        with pytest.raises(SourceCompatibilityError) as exc_info:
            discover_sessions(reversed_adapter, SelectionFilters(), _plan_path(restrictive_tmp_path))

    assert exc_info.value.code == "adapter_session_order"


def test_discovery_reports_incompatible_sessions_through_stats_only(tmp_path: Path, restrictive_tmp_path: Path) -> None:
    inserts: list[Insert] = [
        _current_session("ses-a", directory="/repo/project"),
        _current_session("ses-b", directory="/repo/other"),
        _current_message("msg-a1", "ses-a", "user", 1, _PROMPT_JSON),
        _current_message("msg-b1", "ses-b", "user", 1, _PROMPT_JSON),
    ]
    with _replay_adapter(tmp_path, "incompatible", "current", inserts) as adapter:
        wrapped = _HeaderWrappingAdapter(adapter, rewrite={"ses-b": cast(AdapterGeneration, "future")})
        plan_path = _plan_path(restrictive_tmp_path)
        stats = discover_sessions(wrapped, SelectionFilters(), plan_path)

        scans = list(DiscoveryPlanReader(plan_path).iter_session_scans())

    assert stats.incompatible_session_count == 1
    assert stats.session_scan_count == 1
    assert stats.warning_count >= 1
    assert [scan.session_id for scan in scans] == ["ses-a"]


# ---------------------------------------------------------------------------
# Plan reader validation
# ---------------------------------------------------------------------------


def _scan_entry(**overrides: object) -> SessionScanEntry:
    values: dict[str, object] = {
        "session_id": "ses-1",
        "adapter_generation": "current",
        "selected": True,
        "complete_scan": True,
        "tombstone": False,
        "logical_turn_count": 1,
        "record_count": 2,
        "creation_time_ns": _CREATED_MS * 1_000_000,
        "session_context_fingerprint": "a" * 64,
        "warnings": (),
    }
    values.update(overrides)
    return SessionScanEntry(**values)


def _turn_entry(**overrides: object) -> DiscoveryTurnEntry:
    values: dict[str, object] = {
        "session_id": "ses-1",
        "turn_index": 1,
        "turn_key": "turn-abc",
        "adapter_generation": "current",
        "planned_action": "import",
        "start_time_ns": _CREATED_MS * 1_000_000,
        "end_time_ns": (_CREATED_MS + 1_000) * 1_000_000,
        "record_count": 2,
        "user_record_count": 1,
        "assistant_record_count": 1,
        "terminal_outcome": "assistant_result",
        "model_identities": frozenset(),
        "warnings": (),
        "session_context_fingerprint": "a" * 64,
        "source_fingerprint": "b" * 64,
        "render_fingerprint": "c" * 64,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "mapping_version": MAPPING_VERSION,
        "importer_version": importer_version,
        "redaction_mode": RedactionMode.VERBATIM.value,
    }
    values.update(overrides)
    return DiscoveryTurnEntry(**values)


def _write_plan(plan_path: Path, build: Callable[[DiscoveryPlanWriter], None]) -> None:
    with DiscoveryPlanWriter(plan_path) as writer:
        build(writer)


def _write_lines(plan_path: Path, lines: Iterable[str]) -> None:
    handle: IO[str] = plan_path.open("w", encoding="utf-8")
    try:
        for line in lines:
            handle.write(line + "\n")
    finally:
        handle.close()


def test_reader_round_trips_a_discovery_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.jsonl"
    _write_plan(
        plan_path,
        lambda writer: (
            writer.write_session_scan(_scan_entry()),
            writer.write_turn(_turn_entry()),
        ),
    )

    scans = list(DiscoveryPlanReader(plan_path).iter_session_scans())
    turns = list(DiscoveryPlanReader(plan_path).iter_turns())

    assert scans == [_scan_entry()]
    assert turns == [_turn_entry()]


def test_reader_rejects_unknown_plan_schema_versions(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.jsonl"
    unknown_header = json.dumps({"kind": "plan_header", "plan_schema_version": "opencode-mlflow-discovery-plan-v99"})
    _write_lines(plan_path, [unknown_header])

    with pytest.raises(ImporterError) as exc_info:
        list(DiscoveryPlanReader(plan_path).iter_entries())

    assert exc_info.value.code == "discovery_plan_version_unknown"


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param([], id="empty-file"),
        pytest.param(
            [json.dumps({"kind": "session_scan", "session_id": "ses-1"})],
            id="missing-header",
        ),
        pytest.param(
            [json.dumps({"kind": "session_header", "plan_schema_version": DISCOVERY_PLAN_SCHEMA_VERSION})],
            id="wrong-header-kind",
        ),
    ],
)
def test_reader_rejects_missing_or_malformed_headers(tmp_path: Path, lines: list[str]) -> None:
    plan_path = tmp_path / "plan.jsonl"
    _write_lines(plan_path, lines)

    with pytest.raises(ImporterError) as exc_info:
        list(DiscoveryPlanReader(plan_path).iter_entries())

    assert exc_info.value.code == "discovery_plan_invalid"


@pytest.mark.parametrize(
    "entry_line",
    [
        pytest.param("{not json", id="invalid-json"),
        pytest.param(json.dumps(["not", "an", "object"]), id="not-an-object"),
        pytest.param(json.dumps({"session_id": "ses-1"}), id="missing-kind"),
        pytest.param(json.dumps({"kind": "surprise"}), id="unknown-kind"),
        pytest.param(
            json.dumps(
                {
                    "kind": "session_scan",
                    "session_id": 7,
                    "adapter_generation": "current",
                    "selected": True,
                    "complete_scan": True,
                    "tombstone": False,
                    "logical_turn_count": 1,
                    "record_count": 1,
                    "creation_time_ns": None,
                    "session_context_fingerprint": "a" * 64,
                    "warnings": [],
                }
            ),
            id="wrong-field-type",
        ),
        pytest.param(
            json.dumps(
                {
                    "kind": "session_scan",
                    "adapter_generation": "current",
                    "selected": True,
                    "complete_scan": True,
                    "tombstone": False,
                    "logical_turn_count": 1,
                    "record_count": 1,
                    "creation_time_ns": None,
                    "session_context_fingerprint": "a" * 64,
                    "warnings": [],
                }
            ),
            id="missing-field",
        ),
    ],
)
def test_reader_rejects_malformed_entries(tmp_path: Path, entry_line: str) -> None:
    plan_path = tmp_path / "plan.jsonl"
    header = json.dumps({"kind": "plan_header", "plan_schema_version": DISCOVERY_PLAN_SCHEMA_VERSION})
    _write_lines(plan_path, [header, entry_line])

    with pytest.raises(ImporterError) as exc_info:
        list(DiscoveryPlanReader(plan_path).iter_entries())

    assert exc_info.value.code == "discovery_plan_invalid"


def test_reader_rejects_duplicate_turn_identities(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.jsonl"

    def build(writer: DiscoveryPlanWriter) -> None:
        writer.write_session_scan(_scan_entry())
        writer.write_turn(_turn_entry())
        writer.write_turn(_turn_entry(turn_index=2, source_fingerprint="d" * 64))

    _write_plan(plan_path, build)

    with pytest.raises(ImporterError) as exc_info:
        list(DiscoveryPlanReader(plan_path).iter_entries())

    assert exc_info.value.code == "discovery_plan_duplicate_turn"


@pytest.mark.parametrize(
    "entry_factory",
    [
        pytest.param(lambda: _scan_entry(session_context_fingerprint=""), id="scan-without-fingerprint"),
        pytest.param(lambda: _turn_entry(source_fingerprint=""), id="turn-without-source-fingerprint"),
        pytest.param(lambda: _turn_entry(render_fingerprint=""), id="turn-without-render-fingerprint"),
    ],
)
def test_reader_rejects_missing_fingerprints(tmp_path: Path, entry_factory: Callable[[], object]) -> None:
    plan_path = tmp_path / "plan.jsonl"

    def build(writer: DiscoveryPlanWriter) -> None:
        writer.write_session_scan(_scan_entry())
        entry = entry_factory()
        if isinstance(entry, SessionScanEntry):
            writer.write_session_scan(entry)
        else:
            writer.write_turn(cast(DiscoveryTurnEntry, entry))

    _write_plan(plan_path, build)

    with pytest.raises(ImporterError) as exc_info:
        list(DiscoveryPlanReader(plan_path).iter_entries())

    assert exc_info.value.code == "discovery_plan_invalid"


def test_reader_rejects_turn_entries_without_a_matching_session_scan(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.jsonl"

    def build(writer: DiscoveryPlanWriter) -> None:
        writer.write_turn(_turn_entry(session_id="ses-other"))

    _write_plan(plan_path, build)

    with pytest.raises(ImporterError) as exc_info:
        list(DiscoveryPlanReader(plan_path).iter_entries())

    assert exc_info.value.code == "discovery_plan_turn_without_scan"


def test_reader_rejects_a_selected_complete_scan_missing_its_zero_turn_tombstone(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.jsonl"

    def build(writer: DiscoveryPlanWriter) -> None:
        writer.write_session_scan(_scan_entry(logical_turn_count=0, record_count=0, tombstone=False))

    _write_plan(plan_path, build)

    with pytest.raises(ImporterError) as exc_info:
        list(DiscoveryPlanReader(plan_path).iter_entries())

    assert exc_info.value.code == "discovery_plan_invalid"


def test_reader_rejects_a_tombstone_outside_its_definition(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.jsonl"

    def build(writer: DiscoveryPlanWriter) -> None:
        writer.write_session_scan(_scan_entry(tombstone=True))

    _write_plan(plan_path, build)

    with pytest.raises(ImporterError) as exc_info:
        list(DiscoveryPlanReader(plan_path).iter_entries())

    assert exc_info.value.code == "discovery_plan_invalid"
