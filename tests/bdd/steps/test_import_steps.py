"""Step definitions for the importer's Gherkin feature files.

pytest-bdd refuses to bind feature files that contain no scenarios
(``NoScenariosFound``), so only feature files whose scenarios have landed
are bound here. Task 2 added the first scenarios to
``cli_and_selection.feature``; Task 5 added the ordering/turn-boundary
scenarios to ``source_safety.feature`` (driven through the public source
pipeline: fixture database, normalized snapshot, replay adapter, logical
turns); Task 7 added the redaction scenario to
``privacy_and_reporting.feature`` (driven through the public redaction
seam: redact_record over a turn's records with the deterministic
redactor); Task 10 added the mapping scenarios to
``mapping_and_fidelity.feature`` (driven through the public mapping
seam: the six-turn fixture, replay turns, and ``map_turn_to_trace``);
Task 11 added the reconciliation scenarios to ``reconciliation.feature``
(driven through the public reconciliation seam: ``FakeMlflowGateway``
plus plan/apply over the real source pipeline). Task 13 completed the
acceptance suite: every acceptance criterion now also has a boundary
scenario that drives the public ``main``/``run_import`` pipeline end to
end with fake dependencies — the synthetic source-database factory, the
per-scenario fake gateway, and the boundary runner live in
``tests/bdd/conftest.py``.
"""

from __future__ import annotations

import json
import shlex
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from opencode_mlflow.cli import main
from opencode_mlflow.config import normalize_path, parse_timestamp
from opencode_mlflow.content.canonical import canonical_json
from opencode_mlflow.content.fingerprint import artifact_snapshot_fingerprint
from opencode_mlflow.content.redaction import DeterministicRedactor, redact_record
from opencode_mlflow.contracts import RedactionMode, Redactor
from opencode_mlflow.mlflow.capabilities import CapabilityRequest
from opencode_mlflow.mlflow.gateway import TRACE_SESSION_METADATA_KEY
from opencode_mlflow.mlflow.mapping import (
    MappingContext,
    bind_mlflow_identity_map,
    map_turn_to_trace,
)
from opencode_mlflow.mlflow.reconciliation import (
    OwnedTraceCandidate,
    ReconciliationDependencies,
    apply_reconciliation,
    plan_reconciliation,
)
from opencode_mlflow.mlflow.reconciliation import upload_artifact as upload_reconciliation_artifact
from opencode_mlflow.retry import RetryPolicy
from opencode_mlflow.source.catalog import CURRENT_CATALOG_V1, LEGACY_CATALOG_V1
from opencode_mlflow.source.current import CurrentAdapter
from opencode_mlflow.source.discovery import (
    ARTIFACT_SCHEMA_VERSION,
    MAPPING_VERSION,
    DiscoveryPlanReader,
    discover_sessions,
)
from opencode_mlflow.source.legacy import LegacyAdapter
from opencode_mlflow.source.model import SourceRecord, TurnSummary
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter
from opencode_mlflow.source.selection import SelectionEvidenceEvaluator, SelectionFilters, matches_model
from opencode_mlflow.source.snapshot import (
    SourceExporter,
    SourceSnapshot,
    create_normalized_snapshot,
)
from opencode_mlflow.source.turns import build_replay_turn, iter_turn_records, iter_turn_summaries
from tests.fixtures.six_turn_session import build_six_turn_source_db
from tests.support.fake_gateway import FakeMlflowGateway, trace_spec
from tests.support.fake_source import identity_redaction
from tests.support.import_fakes import (
    BDD_EXPERIMENT_NAME,
    BDD_SESSION_ID,
    DEFAULT_SESSION_CREATED_MS,
)

pytestmark = pytest.mark.bdd

scenarios("../features/cli_and_selection.feature")
scenarios("../features/source_safety.feature")
scenarios("../features/privacy_and_reporting.feature")
scenarios("../features/mapping_and_fidelity.feature")
scenarios("../features/reconciliation.feature")


@given(parsers.parse('a session contains the model identity "{identity}"'))
def session_contains_model_identity(scenario_context: dict[str, object], identity: str) -> None:
    """Record the model identities a source session exposes."""
    scenario_context["available_model_identities"] = (identity,)


@when(parsers.parse('I invoke the importer with "{arguments}"'))
def invoke_importer(scenario_context: dict[str, object], capsys: pytest.CaptureFixture[str], arguments: str) -> None:
    """Invoke the CLI in-process and record its exit status and output.

    The quoted ``arguments`` are appended to an otherwise well-formed
    ``import`` subcommand invocation (subcommand plus a fixed, non-secret
    experiment name) so the scenario exercises the argument under test
    rather than unrelated required options.
    """
    argv = ["import", "--experiment-name", "bdd-import-experiment", *shlex.split(arguments)]
    scenario_context["exit_status"] = main(argv)
    captured = capsys.readouterr()
    scenario_context["result"] = captured.out + captured.err


@when(parsers.parse('I select model "{model}"'))
def select_model(scenario_context: dict[str, object], model: str) -> None:
    """Select sessions by the requested model identity."""
    scenario_context["requested_model"] = model


@then(parsers.parse("the process exits with status {status:d}"))
def process_exits_with_status(scenario_context: dict[str, object], status: int) -> None:
    """The CLI's return value is the process exit status."""
    assert scenario_context["exit_status"] == status


@then(parsers.parse('the safe error code is "{code}"'))
def safe_error_code_is(scenario_context: dict[str, object], code: str) -> None:
    """The failure surfaced the stable safe error code."""
    assert code in scenario_context["result"]


@then(parsers.parse('the result does not contain "{text}"'))
def result_does_not_contain(scenario_context: dict[str, object], text: str) -> None:
    """No user-supplied value leaked into the result output."""
    assert text not in scenario_context["result"]


@then("the session is selected")
def session_is_selected(scenario_context: dict[str, object]) -> None:
    """The scenario's session is selected.

    Model-identity scenarios assert the authoritative
    :func:`opencode_mlflow.source.selection.matches_model` seam;
    discovery scenarios assert the selection recorded in the discovery
    plan.
    """
    if "selected_session_ids" in scenario_context:
        assert scenario_context["session_id"] in scenario_context["selected_session_ids"]
        return
    available = scenario_context["available_model_identities"]
    requested = scenario_context["requested_model"]
    assert isinstance(available, tuple)
    assert isinstance(requested, str)
    assert matches_model(available, requested)


@then("the session is not selected")
def session_is_not_selected(scenario_context: dict[str, object]) -> None:
    """The scenario's session is not selected by the discovery filters."""
    if "selected_session_ids" in scenario_context:
        assert scenario_context["session_id"] not in scenario_context["selected_session_ids"]
        return
    available = scenario_context.get("available_model_identities")
    requested = scenario_context.get("requested_model")
    assert isinstance(available, tuple)
    assert isinstance(requested, str)
    assert not matches_model(available, requested)


@then(parsers.parse('a complete model request does not match "{identity}"'))
def complete_model_request_does_not_match(scenario_context: dict[str, object], identity: str) -> None:
    """A complete ``provider/model`` request never matches an incomplete source identity."""
    available = scenario_context["available_model_identities"]
    assert isinstance(available, tuple)
    complete_request = f"{identity.split('/', 1)[0]}/model"
    assert not matches_model(available, complete_request)


# ---------------------------------------------------------------------------
# Selection scenarios (cli_and_selection.feature)
#
# The steps drive the public source pipeline only: a sanitized synthetic
# fixture database, the real generation adapter through the snapshot
# driver, the replay adapter over the normalized snapshot, and discovery
# with the scenario's filters. Selection is asserted from the discovery
# plan and stats; the CLI/importer boundary arrives in a later task.
# ---------------------------------------------------------------------------

_BDD_REPLY_JSON = '{"content": [{"type": "text", "text": "synthetic bdd reply"}]}'
_SELECTION_CONTROL_DIRECTORY = "/repo/projector"
_SELECTION_TARGET_SESSION = "ses-bdd-selection-target"
_SELECTION_CONTROL_SESSION = "ses-bdd-selection-control"
_SELECTION_TURNS_SESSION = "ses-bdd-selection-turns"


def _epoch_ms(timestamp: str) -> int:
    """Epoch milliseconds for an ISO-8601 instant with an explicit offset."""
    return int(round(parse_timestamp(timestamp).timestamp() * 1000))


def _selection_session_insert(session_id: str, directory: str, created_ms: int) -> tuple[str, tuple[object, ...]]:
    return (
        "INSERT INTO session_v2 (id, directory, time_created, time_updated) VALUES (?, ?, ?, ?)",
        (session_id, directory, created_ms, created_ms),
    )


def _run_selection_discovery(
    scenario_context: dict[str, object],
    tmp_path: Path,
    restrictive_tmp_path: Path,
    filters: SelectionFilters,
) -> None:
    """Run fixture DB → normalized snapshot → replay adapter → discovery plan."""
    source = scenario_context["source_path"]
    assert isinstance(source, Path)
    connection = sqlite3.connect(source)
    try:
        exporter = _exporter_for(str(scenario_context["generation"]), connection)
    finally:
        connection.close()
    with create_normalized_snapshot(
        source,
        tmp_path,
        exporter,
        identity_redaction,
        selection_evaluator=SelectionEvidenceEvaluator(filters),
    ) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        plan_path = restrictive_tmp_path / "selection-plan.jsonl"
        stats = discover_sessions(adapter, filters, plan_path)
        reader = DiscoveryPlanReader(plan_path)
        scenario_context["discovery_stats"] = stats
        scenario_context["selected_session_ids"] = {
            scan.session_id for scan in reader.iter_session_scans() if scan.selected
        }
        scenario_context["logical_turn_counts"] = {
            scan.session_id: scan.logical_turn_count for scan in reader.iter_session_scans()
        }
        turn_entry_counts: dict[str, int] = {}
        for turn in reader.iter_turns():
            turn_entry_counts[turn.session_id] = turn_entry_counts.get(turn.session_id, 0) + 1
        scenario_context["turn_entry_counts"] = turn_entry_counts


@given(parsers.parse('a session was created at "{created_at}" in "{directory}"'))
def session_created_at_in_directory(
    scenario_context: dict[str, object], tmp_path: Path, created_at: str, directory: str
) -> None:
    """A current session created at an exact instant, plus the sibling-prefix control.

    The control session is created at the same instant in
    ``/repo/projector`` so a directory filter's rejection of it is
    attributable to the component boundary alone.
    """
    created_ms = _epoch_ms(created_at)
    source = _build_source_db(
        tmp_path,
        "current",
        [
            _selection_session_insert(_SELECTION_TARGET_SESSION, directory, created_ms),
            _selection_session_insert(_SELECTION_CONTROL_SESSION, _SELECTION_CONTROL_DIRECTORY, created_ms),
            _current_message_insert("bdd-sel-target-user", _SELECTION_TARGET_SESSION, "user", 1, _BDD_PROMPT_JSON),
            _current_message_insert(
                "bdd-sel-target-assistant", _SELECTION_TARGET_SESSION, "assistant", 2, _BDD_REPLY_JSON
            ),
            _current_message_insert("bdd-sel-control-user", _SELECTION_CONTROL_SESSION, "user", 1, _BDD_PROMPT_JSON),
            _current_message_insert(
                "bdd-sel-control-assistant", _SELECTION_CONTROL_SESSION, "assistant", 2, _BDD_REPLY_JSON
            ),
        ],
    )
    scenario_context["generation"] = "current"
    scenario_context["session_id"] = _SELECTION_TARGET_SESSION
    scenario_context["sessions_by_directory"] = {
        directory: _SELECTION_TARGET_SESSION,
        _SELECTION_CONTROL_DIRECTORY: _SELECTION_CONTROL_SESSION,
    }
    scenario_context["source_path"] = source


@given(parsers.parse("a session has {count:d} logical user turns"))
def session_has_logical_user_turns(scenario_context: dict[str, object], tmp_path: Path, count: int) -> None:
    """A current session whose consecutive user prompts form exactly ``count`` logical turns."""
    inserts = [
        _selection_session_insert(_SELECTION_TURNS_SESSION, "/repo/project", _epoch_ms("2026-10-21T12:00:00Z")),
        *(
            _current_message_insert(
                f"bdd-sel-turn-user-{index}", _SELECTION_TURNS_SESSION, "user", index, _BDD_PROMPT_JSON
            )
            for index in range(1, count + 1)
        ),
    ]
    source = _build_source_db(tmp_path, "current", inserts)
    scenario_context["generation"] = "current"
    scenario_context["session_id"] = _SELECTION_TURNS_SESSION
    scenario_context["source_path"] = source


@when(parsers.parse('I filter from "{from_time}" to "{to_time}" and directory "{directory}"'))
def filter_sessions_by_time_and_directory(
    scenario_context: dict[str, object],
    tmp_path: Path,
    restrictive_tmp_path: Path,
    from_time: str,
    to_time: str,
    directory: str,
) -> None:
    """Run discovery with inclusive time bounds and a directory filter."""
    _run_selection_discovery(
        scenario_context,
        tmp_path,
        restrictive_tmp_path,
        SelectionFilters(
            from_time=parse_timestamp(from_time),
            to_time=parse_timestamp(to_time),
            directory=normalize_path(directory),
        ),
    )


@when(parsers.parse("I select sessions with minimum user turns {minimum:d}"))
def select_sessions_with_minimum_user_turns(
    scenario_context: dict[str, object], tmp_path: Path, restrictive_tmp_path: Path, minimum: int
) -> None:
    """Run discovery with a strict minimum-turn threshold."""
    _run_selection_discovery(scenario_context, tmp_path, restrictive_tmp_path, SelectionFilters(min_user_turns=minimum))


@then(parsers.parse('a session in "{directory}" is not selected'))
def session_in_directory_is_not_selected(scenario_context: dict[str, object], directory: str) -> None:
    """No session rooted in the sibling-prefix directory was selected."""
    sessions_by_directory = scenario_context["sessions_by_directory"]
    selected = scenario_context["selected_session_ids"]
    assert isinstance(sessions_by_directory, dict)
    assert isinstance(selected, set)
    assert sessions_by_directory[directory] not in selected


@then(parsers.parse("the session is selected with all {count:d} turns"))
def session_is_selected_with_all_turns(scenario_context: dict[str, object], count: int) -> None:
    """The session is selected and the plan carries every one of its turns."""
    session_id = scenario_context["session_id"]
    selected = scenario_context["selected_session_ids"]
    turn_entry_counts = scenario_context["turn_entry_counts"]
    logical_turn_counts = scenario_context["logical_turn_counts"]
    assert isinstance(selected, set)
    assert isinstance(turn_entry_counts, dict)
    assert isinstance(logical_turn_counts, dict)
    assert session_id in selected
    assert turn_entry_counts.get(session_id) == count
    assert logical_turn_counts.get(session_id) == count


# ---------------------------------------------------------------------------
# Source-pipeline scenarios (source_safety.feature)
#
# The steps drive the public pipeline only: a sanitized synthetic fixture
# database, the real generation adapter through the snapshot driver, the
# replay adapter over the normalized snapshot, and the public turn
# reconstruction functions. The CLI/importer boundary arrives in a later
# task, so the scenarios assert at the source-pipeline level.
# ---------------------------------------------------------------------------

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

_BDD_PROMPT_JSON = '{"content": [{"type": "text", "text": "synthetic bdd prompt"}]}'
_BDD_PART_JSON = '{"type": "text", "text": "synthetic bdd part"}'


def _build_source_db(
    tmp_path: Path,
    generation: str,
    inserts: Sequence[tuple[str, tuple[object, ...]]],
    name: str = "bdd-source.db",
) -> Path:
    """Build a sanitized synthetic source database for one scenario."""
    source = tmp_path / name
    connection = sqlite3.connect(source)
    try:
        connection.executescript(_CURRENT_DDL if generation == "current" else _LEGACY_DDL)
        for sql, parameters in inserts:
            connection.execute(sql, parameters)
        connection.commit()
    finally:
        connection.close()
    return source


def _session_insert(session_id: str, generation: str) -> tuple[str, tuple[object, ...]]:
    created = 1_772_000_000_000 if generation == "current" else 1_771_000_000_000
    table = "session_v2" if generation == "current" else "session"
    return (f"INSERT INTO {table} (id, time_created, time_updated) VALUES (?, ?, ?)", (session_id, created, created))


def _current_message_insert(
    message_id: str | None, session_id: str, message_type: str, seq: int, data: str
) -> tuple[str, tuple[object, ...]]:
    return (
        "INSERT INTO session_message (id, session_id, type, seq, time_created, time_updated, data)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            message_id,
            session_id,
            message_type,
            seq,
            1_772_000_010_000 + seq * 1_000,
            1_772_000_010_000 + seq * 1_000,
            data,
        ),
    )


def _legacy_message_insert(
    message_id: str | None, session_id: str, data: str, *, time_created: int = 1_771_000_010_000
) -> tuple[str, tuple[object, ...]]:
    return (
        "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
        (message_id, session_id, time_created, time_created, data),
    )


def _legacy_part_insert(
    part_id: str, message_id: str | None, session_id: str, *, time_created: int = 1_771_000_011_000
) -> tuple[str, tuple[object, ...]]:
    return (
        "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?, ?)",
        (part_id, message_id, session_id, time_created, time_created, _BDD_PART_JSON),
    )


def _exporter_for(generation: str, connection: sqlite3.Connection) -> SourceExporter:
    if generation == "current":
        return CurrentAdapter(connection, CURRENT_CATALOG_V1)
    return LegacyAdapter(connection, LEGACY_CATALOG_V1)


def _record_identity(record: SourceRecord) -> str:
    """The record's stable identity (source ID or fallback identity)."""
    if record.source_id is not None:
        return record.source_id
    if record.source_identity is not None:
        return record.source_identity
    return ""


@given(
    parsers.parse(
        'a current source session "{session_id}" with consecutive user prompts "{first_id}" and "{second_id}"'
    )
)
def current_session_with_consecutive_prompts(
    scenario_context: dict[str, object], tmp_path: Path, session_id: str, first_id: str, second_id: str
) -> None:
    """A current session whose two user prompts have no assistant result between them."""
    source = _build_source_db(
        tmp_path,
        "current",
        [
            _session_insert(session_id, "current"),
            _current_message_insert(first_id, session_id, "user", 1, _BDD_PROMPT_JSON),
            _current_message_insert(second_id, session_id, "user", 2, _BDD_PROMPT_JSON),
        ],
    )
    scenario_context["generation"] = "current"
    scenario_context["session_id"] = session_id
    scenario_context["source_path"] = source


@given(
    parsers.parse(
        'a legacy source session "{session_id}" with a user message "{message_id}"'
        ' followed by parts "{first_part}" and "{second_part}"'
    )
)
def legacy_session_with_message_and_parts(
    scenario_context: dict[str, object],
    tmp_path: Path,
    session_id: str,
    message_id: str,
    first_part: str,
    second_part: str,
) -> None:
    """A legacy session with one user message and two attached text parts."""
    source = _build_source_db(
        tmp_path,
        "legacy",
        [
            _session_insert(session_id, "legacy"),
            _legacy_message_insert(message_id, session_id, '{"role": "user", "content": "synthetic bdd prompt"}'),
            _legacy_part_insert(first_part, message_id, session_id, time_created=1_771_000_011_000),
            _legacy_part_insert(second_part, message_id, session_id, time_created=1_771_000_012_000),
        ],
    )
    scenario_context["generation"] = "legacy"
    scenario_context["session_id"] = session_id
    scenario_context["source_path"] = source


@given(parsers.parse('a legacy source session "{session_id}" with {count:d} identical missing-ID user messages'))
def legacy_session_with_identical_missing_id_messages(
    scenario_context: dict[str, object], tmp_path: Path, session_id: str, count: int
) -> None:
    """A legacy session whose user message rows carry no source IDs at all."""
    inserts = [
        _session_insert(session_id, "legacy"),
        *(
            _legacy_message_insert(None, session_id, '{"role": "user", "content": "synthetic bdd no-id prompt"}')
            for _ in range(count)
        ),
    ]
    source = _build_source_db(tmp_path, "legacy", inserts)
    scenario_context["generation"] = "legacy"
    scenario_context["session_id"] = session_id
    scenario_context["source_path"] = source


@given(parsers.parse('a current source session "{session_id}" with a system record before user prompt "{user_id}"'))
def current_session_with_system_record_before_prompt(
    scenario_context: dict[str, object], tmp_path: Path, session_id: str, user_id: str
) -> None:
    """A current session whose first record is a system notice, not a prompt."""
    source = _build_source_db(
        tmp_path,
        "current",
        [
            _session_insert(session_id, "current"),
            _current_message_insert(
                "bdd-system-1",
                session_id,
                "system",
                1,
                '{"content": [{"type": "text", "text": "synthetic bdd system notice"}]}',
            ),
            _current_message_insert(user_id, session_id, "user", 2, _BDD_PROMPT_JSON),
        ],
    )
    scenario_context["generation"] = "current"
    scenario_context["session_id"] = session_id
    scenario_context["source_path"] = source


@when("the source pipeline reconstructs the session's logical turns")
def reconstruct_logical_turns(scenario_context: dict[str, object], tmp_path: Path) -> None:
    """Run fixture DB → normalized snapshot → adapter → logical turns."""
    generation = str(scenario_context["generation"])
    session_id = str(scenario_context["session_id"])
    source = scenario_context["source_path"]
    assert isinstance(source, Path)
    connection = sqlite3.connect(source)
    try:
        exporter = _exporter_for(generation, connection)
    finally:
        connection.close()
    with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        records = list(adapter.iter_records(session_id))
        summaries = list(iter_turn_summaries(records))
        scenario_context["summaries"] = summaries
        scenario_context["turn_records"] = {
            summary.turn_key: [_record_identity(record) for record in iter_turn_records(records, summary)]
            for summary in summaries
        }


@then(parsers.parse("the session has {count:d} logical turns"))
def session_has_logical_turns(scenario_context: dict[str, object], count: int) -> None:
    """The session reconstructed exactly ``count`` logical turns."""
    _assert_logical_turn_count(scenario_context, count)


@then(parsers.parse("the session has {count:d} logical turn"))
def session_has_one_logical_turn(scenario_context: dict[str, object], count: int) -> None:
    """The session reconstructed exactly ``count`` logical turn (singular phrasing)."""
    _assert_logical_turn_count(scenario_context, count)


def _assert_logical_turn_count(scenario_context: dict[str, object], count: int) -> None:
    summaries = scenario_context["summaries"]
    assert isinstance(summaries, list)
    assert len(summaries) == count


@then(parsers.parse('turn {index:d} has terminal outcome "{outcome}"'))
def turn_has_terminal_outcome(scenario_context: dict[str, object], index: int, outcome: str) -> None:
    """The 1-based turn ended with the given terminal outcome."""
    summaries = scenario_context["summaries"]
    assert isinstance(summaries, list)
    summary = summaries[index - 1]
    assert isinstance(summary, TurnSummary)
    assert summary.terminal_outcome == outcome


@then(parsers.parse('turn {index:d} starts at user prompt "{source_id}"'))
def turn_starts_at_user_prompt(scenario_context: dict[str, object], index: int, source_id: str) -> None:
    """The 1-based turn starts at the given user prompt identity."""
    summaries = scenario_context["summaries"]
    assert isinstance(summaries, list)
    summary = summaries[index - 1]
    assert isinstance(summary, TurnSummary)
    assert summary.first_user_source_id == source_id


@then(parsers.parse('turn {index:d} replays the records "{record_ids}"'))
def turn_replays_records(scenario_context: dict[str, object], index: int, record_ids: str) -> None:
    """The 1-based turn replays exactly the given record identities, in order."""
    summaries = scenario_context["summaries"]
    turn_records = scenario_context["turn_records"]
    assert isinstance(summaries, list)
    assert isinstance(turn_records, dict)
    summary = summaries[index - 1]
    assert isinstance(summary, TurnSummary)
    expected = [item.strip() for item in record_ids.split(",")]
    assert turn_records[summary.turn_key] == expected


@then("the turns have distinct turn keys")
def turns_have_distinct_keys(scenario_context: dict[str, object]) -> None:
    """No two reconstructed turns were merged into one key."""
    summaries = scenario_context["summaries"]
    assert isinstance(summaries, list)
    keys = [summary.turn_key for summary in summaries]
    assert len(keys) == len(set(keys))


@then(parsers.parse('turn {index:d} records the warning "{warning}"'))
def turn_records_warning(scenario_context: dict[str, object], index: int, warning: str) -> None:
    """The 1-based turn's summary carries the given warning code."""
    summaries = scenario_context["summaries"]
    assert isinstance(summaries, list)
    summary = summaries[index - 1]
    assert isinstance(summary, TurnSummary)
    assert warning in summary.warnings


# ---------------------------------------------------------------------------
# Privacy scenarios (privacy_and_reporting.feature)
#
# The steps drive the public redaction seam available now —
# :func:`opencode_mlflow.content.redaction.redact_record` over a turn's
# records, with the deterministic redactor the importer wires into the
# snapshot and effective-turn pipelines in Task 12 — so the same step
# names stay reusable once the full import pipeline exists.
# ---------------------------------------------------------------------------

_BDD_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.bdd-bearer-credentials"


@given(parsers.parse('a turn contains an authorization bearer token and source ID "{source_id}"'))
def turn_contains_authorization_bearer_token(scenario_context: dict[str, object], source_id: str) -> None:
    """A turn record whose content carries an Authorization bearer token."""
    record = SourceRecord.from_row(
        source_table="session_message",
        source_type="message",
        source_id=source_id,
        raw_json=json.dumps(
            {
                "type": "user",
                "text": f"authenticate with Authorization: Bearer {_BDD_BEARER_TOKEN}",
                "headers": {"Authorization": f"Bearer {_BDD_BEARER_TOKEN}"},
            }
        ),
    )
    scenario_context["turn_records"] = [record]
    scenario_context["bearer_token"] = _BDD_BEARER_TOKEN
    scenario_context["source_id"] = source_id


@when("I import with redaction enabled")
def import_with_redaction_enabled(scenario_context: dict[str, object]) -> None:
    """Redact the turn's records with the deterministic redactor in active mode."""
    redactor = DeterministicRedactor(RedactionMode.REDACTED)
    assert isinstance(redactor, Redactor)
    records = scenario_context["turn_records"]
    assert isinstance(records, list)
    results = [redact_record(record) for record in records]
    scenario_context["redacted_records"] = [result.value for result in results]
    scenario_context["redaction_reports"] = [result.report for result in results]


@then(parsers.parse('the trace and artifact contain "{marker}"'))
def trace_and_artifact_contain(scenario_context: dict[str, object], marker: str) -> None:
    """Every redacted record's trace payload and artifact text carry the marker."""
    for record in scenario_context["redacted_records"]:
        assert isinstance(record, SourceRecord)
        assert marker in canonical_json(record.decoded_value).decode("utf-8")
        assert marker in (record.raw_json or "")


@then("neither contains the original bearer token")
def neither_contains_the_original_bearer_token(scenario_context: dict[str, object]) -> None:
    """No redacted record content or raw column retains the bearer token."""
    token = scenario_context["bearer_token"]
    assert isinstance(token, str)
    for record in scenario_context["redacted_records"]:
        assert isinstance(record, SourceRecord)
        assert token not in canonical_json(record.decoded_value).decode("utf-8")
        assert token not in (record.raw_json or "")
        assert token not in repr(record.raw_columns)


@then(parsers.parse('the source ID remains "{source_id}"'))
def source_id_remains(scenario_context: dict[str, object], source_id: str) -> None:
    """Reconciliation identity survived redaction unchanged."""
    for record in scenario_context["redacted_records"]:
        assert isinstance(record, SourceRecord)
        assert record.source_id == source_id


@then("the report contains detector counts but not the original token")
def report_contains_detector_counts_but_not_the_original_token(scenario_context: dict[str, object]) -> None:
    """The redaction report carries counts only — never the matched token."""
    token = scenario_context["bearer_token"]
    assert isinstance(token, str)
    reports = scenario_context["redaction_reports"]
    assert isinstance(reports, list)
    assert reports
    for report in reports:
        assert report.detector_counts
        assert report.affected_record_count == 1
        assert token not in repr(report)


# ---------------------------------------------------------------------------
# Mapping scenarios (mapping_and_fidelity.feature)
#
# The steps drive the public mapping seam available now — the sanitized
# synthetic six-turn session fixture, its replay turns, and
# ``map_turn_to_trace`` — so the same step names stay reusable once the
# full import pipeline (Task 12) maps turns through the gateway. The
# CLI/importer boundary arrives in a later task.
# ---------------------------------------------------------------------------


def _mapped_specs(scenario_context: dict[str, object]) -> list:
    specs = scenario_context["mapping_specs"]
    assert isinstance(specs, list)
    return specs


def _children_by_source(scenario_context: dict[str, object], index: int) -> dict[str, object]:
    specs = _mapped_specs(scenario_context)
    return {
        str(child.attributes.get("opencode.source_id", child.source_key)): child
        for child in specs[index - 1].child_specs()
    }


@given("the synthetic six-turn session fixture")
def synthetic_six_turn_session_fixture(scenario_context: dict[str, object], six_turn_session) -> None:
    """Record the six-turn fixture for the scenario's mapping steps."""
    scenario_context["six_turn_session"] = six_turn_session


@when("I map every replayed turn to a trace spec")
def map_every_replayed_turn(scenario_context: dict[str, object]) -> None:
    """Map each replayed turn through the public mapping seam."""
    fixture = scenario_context["six_turn_session"]
    scenario_context["mapping_specs"] = [map_turn_to_trace(turn, fixture.context) for turn in fixture.turns]


@when("I map every replayed turn to a trace spec twice")
def map_every_replayed_turn_twice(scenario_context: dict[str, object]) -> None:
    """Map each replayed turn twice; both passes must agree exactly."""
    fixture = scenario_context["six_turn_session"]
    scenario_context["mapping_specs"] = [map_turn_to_trace(turn, fixture.context) for turn in fixture.turns]
    scenario_context["mapping_specs_again"] = [map_turn_to_trace(turn, fixture.context) for turn in fixture.turns]


@when(parsers.parse("I bind generated identities for turn {index:d}"))
def bind_generated_identities(scenario_context: dict[str, object], index: int) -> None:
    """Bind synthetic generated MLflow identities to the turn's envelope."""
    fixture = scenario_context["six_turn_session"]
    spec = _mapped_specs(scenario_context)[index - 1]
    children = {child.source_key: child for child in spec.child_specs()}
    scenario_context["identity_map"] = bind_mlflow_identity_map(
        spec.identity_envelope(),
        f"tr-bdd-{index}",
        {key: f"span-bdd-{position}" for position, key in enumerate(sorted(children), start=1)},
    )
    scenario_context["bound_fingerprint"] = artifact_snapshot_fingerprint(
        fixture.turns[index - 1], fixture.context.redaction_mode
    )


@then(parsers.parse("the mapping produces {count:d} trace specs"))
def mapping_produces_trace_specs(scenario_context: dict[str, object], count: int) -> None:
    """One spec per replayed logical turn."""
    assert len(_mapped_specs(scenario_context)) == count


@then(parsers.parse('every spec carries the native session id "{session_id}"'))
def every_spec_carries_native_session_id(scenario_context: dict[str, object], session_id: str) -> None:
    """The native MLflow session id is the exact OpenCode session id."""
    assert all(spec.native_session_id == session_id for spec in _mapped_specs(scenario_context))


@then(parsers.parse("the specs carry {count:d} distinct client request ids"))
def specs_carry_distinct_client_request_ids(scenario_context: dict[str, object], count: int) -> None:
    """Every turn maps under its own deterministic idempotency key."""
    specs = _mapped_specs(scenario_context)
    assert len({spec.client_request_id for spec in specs}) == count


@then(parsers.parse('every spec metadata records "{key}" as "{value}"'))
def every_spec_metadata_records(scenario_context: dict[str, object], key: str, value: str) -> None:
    """The namespaced ownership metadata round-trips on every spec."""
    assert all(spec.metadata[key] == value for spec in _mapped_specs(scenario_context))


@then(parsers.parse("the root input of turn {index:d} is the first user prompt"))
def root_input_is_the_first_user_prompt(scenario_context: dict[str, object], index: int) -> None:
    """The root input is the turn's first-user record's decoded value."""
    fixture = scenario_context["six_turn_session"]
    first_record = next(fixture.turns[index - 1].iter_records())
    assert _mapped_specs(scenario_context)[index - 1].root_inputs == first_record.decoded_value


@then(parsers.parse("the root output of turn {index:d} is the final assistant text part"))
def root_output_is_the_final_text_part(scenario_context: dict[str, object], index: int) -> None:
    """A single-part result combines to that user-visible part."""
    assert _mapped_specs(scenario_context)[index - 1].root_outputs == [{"type": "text", "text": "synthetic answer one"}]


@then(parsers.parse("the root output of turn {index:d} combines {count:d} ordered text parts"))
def root_output_combines_ordered_text_parts(scenario_context: dict[str, object], index: int, count: int) -> None:
    """A split result combines its ordered user-visible parts."""
    output = _mapped_specs(scenario_context)[index - 1].root_outputs
    assert isinstance(output, list)
    assert len(output) == count
    assert output == [
        {"type": "text", "text": "synthetic final part one"},
        {"type": "text", "text": "synthetic final part two"},
    ]


@then(parsers.parse("the root output of turn {index:d} is omitted"))
def root_output_is_omitted(scenario_context: dict[str, object], index: int) -> None:
    """A turn without an assistant result never manufactures output."""
    assert _mapped_specs(scenario_context)[index - 1].root_outputs is None


@then(parsers.parse('turn {index:d} maps to root status "{status}" with outcome "{outcome}"'))
def turn_maps_to_root_status_with_outcome(
    scenario_context: dict[str, object], index: int, status: str, outcome: str
) -> None:
    """The source terminal outcome drives the authoritative root status."""
    spec = _mapped_specs(scenario_context)[index - 1]
    assert spec.root_status == status
    assert spec.metadata["opencode.outcome"] == outcome


@then(parsers.parse("turn {index:d} has {count:d} children"))
def turn_has_children(scenario_context: dict[str, object], index: int, count: int) -> None:
    """Every turn record except the root-input prompt maps to a child."""
    assert len(_children_by_source(scenario_context, index)) == count


@then(parsers.parse('turn {index:d} has a child of span type "{span_type}" with source id "{source_id}"'))
def turn_has_a_child_of_span_type(
    scenario_context: dict[str, object], index: int, span_type: str, source_id: str
) -> None:
    """The typed child mapping preserves the record's source identity."""
    child = _children_by_source(scenario_context, index)[source_id]
    assert child.span_type == span_type


@then(parsers.parse('turn {index:d} has an event child with source id "{source_id}"'))
def turn_has_an_event_child(scenario_context: dict[str, object], index: int, source_id: str) -> None:
    """Annotation parts map to events (span type "event")."""
    child = _children_by_source(scenario_context, index)[source_id]
    assert child.span_type == "event"


@then(parsers.parse('the child of turn {index:d} with source id "{source_id}" has parent source id "{parent_id}"'))
def child_has_parent_source_id(scenario_context: dict[str, object], index: int, source_id: str, parent_id: str) -> None:
    """Source parent linkage survives into the span tree."""
    child = _children_by_source(scenario_context, index)[source_id]
    assert child.parent_source_key is not None
    assert child.parent_source_key.endswith(f":{parent_id}") or child.parent_source_key == parent_id


@then(parsers.parse('the child of turn {index:d} with source id "{source_id}" has the trace root as parent'))
def child_has_the_trace_root_as_parent(scenario_context: dict[str, object], index: int, source_id: str) -> None:
    """Parts of the root-input prompt attach to the trace root."""
    child = _children_by_source(scenario_context, index)[source_id]
    assert child.parent_source_key is None


@then(parsers.parse("iterating the children of turn {index:d} twice yields the same {count:d} children"))
def iterating_children_twice_yields_the_same(scenario_context: dict[str, object], index: int, count: int) -> None:
    """Child cursors are lazy and reopen fresh on every call."""
    spec = _mapped_specs(scenario_context)[index - 1]
    first = [child.source_key for child in spec.child_specs()]
    second = [child.source_key for child in spec.child_specs()]
    assert first == second
    assert len(first) == count


@then("both mappings produce identical specs")
def both_mappings_produce_identical_specs(scenario_context: dict[str, object]) -> None:
    """Mapping the same snapshot twice yields the identical plan."""
    specs = _mapped_specs(scenario_context)
    again = scenario_context["mapping_specs_again"]
    assert isinstance(again, list)
    assert len(specs) == len(again)
    for first, second in zip(specs, again, strict=True):
        assert first.client_request_id == second.client_request_id
        assert dict(first.metadata) == dict(second.metadata)
        assert dict(first.root_attributes) == dict(second.root_attributes)
        assert first.artifact_relative_path == second.artifact_relative_path
        assert [child.source_key for child in first.child_specs()] == [
            child.source_key for child in second.child_specs()
        ]


@then(parsers.parse("the artifact snapshot fingerprint of turn {index:d} is unchanged"))
def artifact_snapshot_fingerprint_is_unchanged(scenario_context: dict[str, object], index: int) -> None:
    """Generated ids bind after the fingerprints were fixed."""
    fixture = scenario_context["six_turn_session"]
    spec = _mapped_specs(scenario_context)[index - 1]
    bound = scenario_context["bound_fingerprint"]
    assert isinstance(bound, str)
    assert bound == spec.artifact_snapshot_fingerprint
    assert artifact_snapshot_fingerprint(fixture.turns[index - 1], fixture.context.redaction_mode) == bound


@then("the bound identity map records the trace and span ids")
def bound_identity_map_records_ids(scenario_context: dict[str, object]) -> None:
    """The artifact identity section carries the generated trace/span ids."""
    identity_map = scenario_context["identity_map"]
    section = identity_map.artifact_section()
    assert section["trace_id"] == identity_map.trace_id
    assert section.keys() >= {"trace_id", "span:session_message:msg-synthetic-a1"}
    assert all(key.startswith("span:") for key in section if key != "trace_id")


# ---------------------------------------------------------------------------
# Reconciliation scenarios (reconciliation.feature)
#
# The steps drive the public reconciliation seam available now —
# ``FakeMlflowGateway`` plus ``plan_reconciliation``/``apply_reconciliation``
# with replay/mapping dependencies over the real source pipeline (fixture
# database → normalized snapshot → discovery → replay → mapping) — so the
# same step names stay reusable once Task 12 wires the full import
# pipeline. The previous import is modeled by driving the fake gateway
# directly through the same public seams (create → upload → verify).
# ---------------------------------------------------------------------------

_RECON_EXPERIMENT_NAME = "bdd-recon-experiment"
_RECON_PROMPT_JSON = '{"content": [{"type": "text", "text": "synthetic reconciliation prompt"}]}'
_RECON_ORIGINAL_REPLY_JSON = '{"content": [{"type": "text", "text": "synthetic original reply"}]}'
_RECON_CHANGED_REPLY_JSON = '{"content": [{"type": "text", "text": "synthetic changed reply"}]}'
_RECON_IMMEDIATE_POLICY = RetryPolicy(max_attempts=1, initial_delay_seconds=0.0, max_delay_seconds=0.0)


def _recon_source(tmp_path: Path, session_id: str, reply_json: str, name: str) -> Path:
    """A current-generation source database with one synthetic user/assistant turn."""
    return _build_source_db(
        tmp_path,
        "current",
        [
            _selection_session_insert(session_id, "/repo/recon", _epoch_ms("2026-10-21T12:00:00Z")),
            _current_message_insert("bdd-recon-user", session_id, "user", 1, _RECON_PROMPT_JSON),
            _current_message_insert("bdd-recon-assistant", session_id, "assistant", 2, reply_json),
        ],
        name=name,
    )


def _recon_snapshot(scenario_context: dict[str, object], tmp_path: Path, source: Path) -> SourceSnapshot:
    """Open the normalized snapshot for a source database, tracking it for cleanup."""
    connection = sqlite3.connect(source)
    try:
        exporter = CurrentAdapter(connection, CURRENT_CATALOG_V1)
    finally:
        connection.close()
    managed = create_normalized_snapshot(source, tmp_path, exporter, identity_redaction)
    snapshot = managed.__enter__()
    scenario_context.setdefault("recon_snapshots", []).append(managed)
    return snapshot


def _recon_discover(snapshot: SourceSnapshot, plan_path: Path) -> tuple[list, list]:
    """Run discovery over the snapshot and return (session scans, turn entries)."""
    adapter = NormalizedSnapshotAdapter(snapshot)
    discover_sessions(adapter, SelectionFilters(), plan_path)
    reader = DiscoveryPlanReader(plan_path)
    return list(reader.iter_session_scans()), list(reader.iter_turns())


def _recon_context(experiment_id: str, snapshot: SourceSnapshot, source_schema: str) -> MappingContext:
    """The mapping context the scenario's turns are mapped under."""
    from opencode_mlflow import __version__ as importer_version

    return MappingContext(
        importer_version=importer_version,
        mapping_version=MAPPING_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        source_schema=source_schema,
        redaction_mode=RedactionMode.VERBATIM,
        experiment_id=experiment_id,
        catalog_version=snapshot.schema_description.catalog_version,
        source_database_fingerprint=snapshot.fingerprint,
    )


def _reconciliation_bootstrap(
    scenario_context: dict[str, object], tmp_path: Path, restrictive_tmp_path: Path, session_id: str
) -> None:
    """A prior successful import: one owned trace with its verified artifact.

    Also seeds an unowned trace sharing the same native session id, so the
    scenarios can pin that reconciliation never touches foreign traces.
    """
    gateway = FakeMlflowGateway()
    report = gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
    assert report.is_compatible, report.safe_error
    experiment = gateway.resolve_experiment(_RECON_EXPERIMENT_NAME, create=True)
    assert experiment.experiment_id is not None

    source = _recon_source(tmp_path, session_id, _RECON_ORIGINAL_REPLY_JSON, "bdd-recon-original.db")
    connection = sqlite3.connect(source)
    try:
        exporter = CurrentAdapter(connection, CURRENT_CATALOG_V1)
    finally:
        connection.close()
    with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
        scans, turns = _recon_discover(snapshot, restrictive_tmp_path / "recon-original-plan.jsonl")
        assert len(turns) == 1, "the reconciliation fixture must produce exactly one turn"
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = next(adapter.iter_session_headers())
        summary = next(iter_turn_summaries(adapter.iter_records(session_id)))
        replay = build_replay_turn(snapshot, header, summary)
        context = _recon_context(experiment.experiment_id, snapshot, header.adapter_generation)
        request = map_turn_to_trace(replay, context).to_create_request()
        created = gateway.create_trace(request)
        verification = upload_reconciliation_artifact(
            replay,
            created,
            request.artifact_relative_path,
            RedactionMode.VERBATIM,
            gateway=gateway,
            temporary_directory=restrictive_tmp_path,
        )
        gateway.verify_named_json(created.trace_id, verification.relative_path, verification.sha256, verification.size)
        gateway.record_artifact_verification(created.trace_id, verification.sha256, verification.size)
        unowned = gateway.create_trace(
            trace_spec(owned=False, native_session_id=session_id, experiment_id=experiment.experiment_id)
        )
        owned = next(
            trace for trace in gateway.iter_owned_traces(experiment.experiment_id) if trace.trace_id == created.trace_id
        )
        candidate = OwnedTraceCandidate.from_owned_trace(owned, experiment.experiment_id)
        scenario_context["recon_mapping_context"] = context

    scenario_context["recon_gateway"] = gateway
    scenario_context["recon_candidate"] = candidate
    scenario_context["recon_unowned"] = unowned
    scenario_context["recon_experiment"] = experiment
    scenario_context["recon_event_marker"] = len(gateway.events)


def _reconciliation_changed_source(
    scenario_context: dict[str, object], tmp_path: Path, restrictive_tmp_path: Path, session_id: str
) -> None:
    """Change the turn's assistant reply: same turn identity, new source fingerprint."""
    source = _recon_source(tmp_path, session_id, _RECON_CHANGED_REPLY_JSON, "bdd-recon-changed.db")
    snapshot = _recon_snapshot(scenario_context, tmp_path, source)
    scans, turns = _recon_discover(snapshot, restrictive_tmp_path / "recon-changed-plan.jsonl")
    assert len(turns) == 1, "the changed fixture must keep exactly one turn"
    assert turns[0].turn_key == scenario_context["recon_candidate"].turn_key
    scenario_context["recon_scans"] = scans
    scenario_context["recon_turns"] = turns
    scenario_context["recon_snapshot"] = snapshot


def _run_reconciliation(scenario_context: dict[str, object], restrictive_tmp_path: Path) -> None:
    """Plan and apply reconciliation over the changed source and the fake gateway."""
    gateway: FakeMlflowGateway = scenario_context["recon_gateway"]
    snapshot: SourceSnapshot = scenario_context["recon_snapshot"]
    scans = scenario_context["recon_scans"]
    turns = scenario_context["recon_turns"]
    candidate: OwnedTraceCandidate = scenario_context["recon_candidate"]
    context: MappingContext = scenario_context["recon_mapping_context"]

    adapter = NormalizedSnapshotAdapter(snapshot)
    headers = {header.session_id: header for header in adapter.iter_session_headers()}
    summaries: dict[str, TurnSummary] = {}
    for entry in turns:
        for summary in iter_turn_summaries(adapter.iter_records(entry.session_id)):
            if summary.turn_key == entry.turn_key:
                summaries[entry.turn_key] = summary

    def replay_turn(entry) -> object:
        return build_replay_turn(snapshot, headers[entry.session_id], summaries[entry.turn_key])

    def upload(turn, created, artifact_relative_path, redaction_mode):
        return upload_reconciliation_artifact(
            turn,
            created,
            artifact_relative_path,
            redaction_mode,
            gateway=gateway,
            temporary_directory=restrictive_tmp_path,
        )

    dependencies = ReconciliationDependencies(
        replay=replay_turn,
        map_trace=lambda turn: map_turn_to_trace(turn, context).to_create_request(),
        artifact_snapshot=lambda turn: artifact_snapshot_fingerprint(turn, RedactionMode.VERBATIM),
        upload_artifact=upload,
    )
    plan = plan_reconciliation(scans, turns, [candidate])
    apply_reconciliation(plan, gateway, dependencies, _RECON_IMMEDIATE_POLICY)
    scenario_context["recon_plan"] = plan

    for managed in scenario_context.pop("recon_snapshots", []):
        managed.__exit__(None, None, None)


@given(parsers.parse('an importer-owned trace exists for session "{session_id}" turn "{turn_label}"'))
def importer_owned_trace_exists_for_turn(
    scenario_context: dict[str, object], tmp_path: Path, restrictive_tmp_path: Path, session_id: str, turn_label: str
) -> None:
    """A previously imported turn: an owned trace with its verified artifact."""
    assert turn_label.startswith("turn-"), "the scenario's turn label names the first logical turn"
    _reconciliation_bootstrap(scenario_context, tmp_path, restrictive_tmp_path, session_id)


@given("the source fingerprint has changed")
def source_fingerprint_has_changed(
    scenario_context: dict[str, object], tmp_path: Path, restrictive_tmp_path: Path
) -> None:
    """The turn's content changed without changing its turn identity."""
    session_id = scenario_context["recon_candidate"].session_id
    _reconciliation_changed_source(scenario_context, tmp_path, restrictive_tmp_path, session_id)


@when("I import the session")
def import_the_session(scenario_context: dict[str, object], restrictive_tmp_path: Path) -> None:
    """Plan and apply reconciliation for every selected turn."""
    _run_reconciliation(scenario_context, restrictive_tmp_path)


@when("I import the changed turn")
def import_the_changed_turn(scenario_context: dict[str, object], restrictive_tmp_path: Path) -> None:
    """Plan and apply reconciliation for the changed turn."""
    _run_reconciliation(scenario_context, restrictive_tmp_path)


@then("the event order is create, verify, upload, verify, retire-old")
def event_order_is_create_verify_upload_verify_retire_old(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """The replacement mutated in the strict order, retiring the old trace last.

    Dual-mode: the reconciliation-seam scenarios read the bootstrap
    gateway's event history since its marker; the boundary scenarios
    read the scenario gateway's events since the latest run's marker.
    """
    if "recon_gateway" in scenario_context:
        gateway: FakeMlflowGateway = scenario_context["recon_gateway"]
        marker: int = scenario_context["recon_event_marker"]
    else:
        run = _latest_run(scenario_context)
        gateway = bdd_gateway
        marker = run.event_marker
    kinds = [event.kind for event in gateway.events[marker:]]
    relevant = [kind for kind in kinds if kind in _REPLACEMENT_EVENT_KINDS]
    assert relevant == _REPLACEMENT_EVENT_ORDER


@then("an unowned trace with the same native session remains unchanged")
def unowned_trace_with_same_native_session_remains_unchanged(scenario_context: dict[str, object]) -> None:
    """A foreign trace sharing the native session id was never touched."""
    gateway: FakeMlflowGateway = scenario_context["recon_gateway"]
    unowned = scenario_context["recon_unowned"]
    stored = gateway.stored_trace(unowned.trace_id)
    assert stored is not None, "the unowned trace must not be retired"
    assert stored.metadata.get("opencode.importer") is None
    assert stored.metadata.get(TRACE_SESSION_METADATA_KEY) is not None


@given("a changed turn whose replacement cannot verify its artifact")
def replacement_artifact_verification_fails(
    scenario_context: dict[str, object], tmp_path: Path, restrictive_tmp_path: Path
) -> None:
    """A changed turn whose replacement cannot verify its uploaded artifact."""
    _reconciliation_bootstrap(scenario_context, tmp_path, restrictive_tmp_path, "ses-1")
    _reconciliation_changed_source(scenario_context, tmp_path, restrictive_tmp_path, "ses-1")
    gateway: FakeMlflowGateway = scenario_context["recon_gateway"]
    old: OwnedTraceCandidate = scenario_context["recon_candidate"]
    gateway.fail_artifact_verification = True
    gateway.artifact_verification_exempt_trace_ids = {old.trace_id}


@then("the old trace remains active")
def old_trace_remains_active(scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway) -> None:
    """The previously imported trace was never retired on the failure path.

    Dual-mode: the reconciliation-seam scenarios read the bootstrap
    candidate's trace id; the boundary scenarios read the trace id the
    scenario's first run recorded for the session's turn.
    """
    if "recon_gateway" in scenario_context:
        gateway: FakeMlflowGateway = scenario_context["recon_gateway"]
        old: OwnedTraceCandidate = scenario_context["recon_candidate"]
        old_trace_id = old.trace_id
    else:
        runs = _cli_runs(scenario_context)
        gateway = bdd_gateway
        old_trace_id = _report_session(runs[-2].report, BDD_SESSION_ID)["turns"][0]["trace_id"]
    assert old_trace_id is not None, "the previous run recorded no trace id"
    stored = gateway.stored_trace(old_trace_id)
    assert stored is not None, "the old trace must remain"
    assert stored.state == "OK"


@then("the new trace is rolled back or reported as a recovery conflict")
def new_trace_rolled_back_or_reported_as_recovery_conflict(scenario_context: dict[str, object]) -> None:
    """Every trace created during the run was either rolled back or reported."""
    gateway: FakeMlflowGateway = scenario_context["recon_gateway"]
    marker: int = scenario_context["recon_event_marker"]
    plan = scenario_context["recon_plan"]
    created_ids = [event.details[0] for event in gateway.events[marker:] if event.kind == "trace_create"]
    assert created_ids, "the replacement must have been attempted"
    recovery_codes = ("duplicate_owned_identity", "replacement_cleanup_failed")
    for trace_id in created_ids:
        rolled_back = not gateway.has_trace(trace_id)
        reported = any(
            result.error_code in recovery_codes and trace_id in result.affected_trace_ids for result in plan.results
        )
        assert rolled_back or reported, f"trace {trace_id} was neither rolled back nor reported as a conflict"


# ---------------------------------------------------------------------------
# CLI boundary scenarios (every feature file)
#
# These scenarios drive the public ``main`` boundary end to end with fake
# dependencies: the real source pipeline over sanitized synthetic source
# databases (the shared builder fixture), the per-scenario in-memory
# ``FakeMlflowGateway``, the deterministic clock, and the recording lock
# factory. The runner fixture (``tests/bdd/conftest.py``) captures the
# parsed report JSON, the exit status, the gateway event marker and
# mutation count, and the source-file state before and after each run;
# the steps below assert only those observables.
# ---------------------------------------------------------------------------

_CLI_BEARER_TOKEN = "ghp_bddcli0123456789abcde"
_CLI_PROMPT_SENTINEL = "SENTINEL-BDD-PROMPT-4e29"
_WAL_SESSION_ID = "ses-bdd-wal"
_WAL_DIRECTORY = "/repo/bdd-wal"

_REPLACEMENT_EVENT_KINDS = ("trace_create", "trace_verify", "artifact_upload", "artifact_verify", "trace_retire")
_REPLACEMENT_EVENT_ORDER = ["trace_create", "trace_verify", "artifact_upload", "artifact_verify", "trace_retire"]


def _cli_runs(scenario_context: dict[str, object]) -> list:
    """Every boundary run this scenario performed, in order."""
    runs = scenario_context.get("cli_runs")
    assert isinstance(runs, list) and runs, "the scenario must have run an import through the CLI boundary"
    return runs


def _latest_run(scenario_context: dict[str, object]):
    """The scenario's most recent boundary run."""
    return _cli_runs(scenario_context)[-1]


def _report_session(report: dict, session_id: str) -> dict:
    """One report session entry by its exact session id."""
    for session in report["sessions"]:
        if session["session_id"] == session_id:
            return session
    raise AssertionError(f"the report carries no session {session_id!r}")


def _session_turns(scenario_context: dict[str, object], session_id: str) -> list:
    """The latest run's turn entries for one session."""
    return _report_session(_latest_run(scenario_context).report, session_id)["turns"]


def _latest_turn_entries(scenario_context: dict[str, object]) -> list[tuple[int, str]]:
    """The latest run's ``(turn_index, trace_id)`` entries, in report order."""
    entries = [
        (turn["turn_index"], turn["trace_id"])
        for session in _latest_run(scenario_context).report["sessions"]
        for turn in session["turns"]
        if turn["trace_id"]
    ]
    assert entries, "the latest run recorded no trace ids"
    return entries


def _stored_turn_trace(scenario_context: dict[str, object], gateway: FakeMlflowGateway, index: int):
    """The gateway-stored trace of the latest run's 1-based turn."""
    entries = dict(_latest_turn_entries(scenario_context))
    assert index in entries, f"the latest run recorded no turn {index}"
    stored = gateway.stored_trace(entries[index])
    assert stored is not None, f"the trace of turn {index} is no longer stored"
    return stored


# -- Given: synthetic sources -------------------------------------------------


@given("a source session with one synthetic turn")
def cli_source_session_with_one_synthetic_turn(bdd_source_builder) -> None:
    """A source database whose single session has one logical turn."""
    bdd_source_builder.add_session(BDD_SESSION_ID, created_ms=DEFAULT_SESSION_CREATED_MS)


@given("a source session with two synthetic turns")
def cli_source_session_with_two_synthetic_turns(bdd_source_builder) -> None:
    """A source database whose single session has two logical turns."""
    bdd_source_builder.add_session(BDD_SESSION_ID, created_ms=DEFAULT_SESSION_CREATED_MS, turns=2)


@given("a source session whose prompt contains a synthetic bearer token")
def cli_source_session_with_bearer_token(bdd_source_builder) -> None:
    """A source database whose prompt carries a synthetic bearer token."""
    prompt = json.dumps({"content": [{"type": "text", "text": f"{_CLI_PROMPT_SENTINEL} {_CLI_BEARER_TOKEN}"}]})
    bdd_source_builder.add_session(BDD_SESSION_ID, created_ms=DEFAULT_SESSION_CREATED_MS, prompt_json=prompt)


@given("a tracking target that fails the capability check")
def cli_tracking_target_fails_the_capability_check(scenario_context: dict[str, object]) -> None:
    """The scenario's fake gateway reports an incompatible target."""
    scenario_context["cli_incompatible_gateway"] = True


@given("a source session with one failing turn write")
def cli_source_session_with_one_failing_turn_write(bdd_source_builder, bdd_gateway: FakeMlflowGateway) -> None:
    """A single-turn source whose first trace verification fails."""
    bdd_source_builder.add_session(BDD_SESSION_ID, created_ms=DEFAULT_SESSION_CREATED_MS)
    bdd_gateway.fail_next_trace_verification = 1


@given("the first session's trace write fails")
def the_first_sessions_trace_write_fails(bdd_gateway: FakeMlflowGateway) -> None:
    """The earliest-created session's turn write fails; later ones continue.

    Sessions import in creation-time order, so failing the next trace
    verification isolates the failure to the scenario's first session
    while the remaining sessions still import.
    """
    bdd_gateway.fail_next_trace_verification = 1


@given("the gateway fails the next trace verification")
def gateway_fails_the_next_trace_verification(bdd_gateway: FakeMlflowGateway) -> None:
    """The next created trace's verification fails with a typed error."""
    bdd_gateway.fail_next_trace_verification = 1


# The synthetic-session Given patterns below are deliberately textually
# disjoint. pytest-bdd binds a step text to the matching ``parsers.parse``
# pattern whose generated fixture name sorts last, and an unconstrained
# ``{created_at}`` could otherwise swallow an ``in "{directory}"``
# segment — so the directory-less form says ``and no directory`` instead
# of trailing directly after the creation instant.


@given(
    parsers.parse('a synthetic session "{session_id}" created at "{created_at}" with {turns:d} turns and no directory')
)
@given(
    parsers.parse('a synthetic session "{session_id}" created at "{created_at}" with {turns:d} turn and no directory')
)
def synthetic_session_with_turns(bdd_source_builder, session_id: str, created_at: str, turns: int) -> None:
    """A current-generation session with no recorded directory."""
    bdd_source_builder.add_session(session_id, created_ms=_epoch_ms(created_at), turns=turns)


@given(
    parsers.parse('a synthetic session "{session_id}" created at "{created_at}" in "{directory}" with {turns:d} turns')
)
@given(
    parsers.parse('a synthetic session "{session_id}" created at "{created_at}" in "{directory}" with {turns:d} turn')
)
def synthetic_session_in_directory_with_turns(
    bdd_source_builder, session_id: str, created_at: str, directory: str, turns: int
) -> None:
    """A current-generation session with the given creation instant and turn count."""
    bdd_source_builder.add_session(session_id, created_ms=_epoch_ms(created_at), directory=directory, turns=turns)


@given(
    parsers.parse(
        'a synthetic session "{session_id}" created at "{created_at}" in "{directory}"'
        ' with {turns:d} turns and message model "{model}"'
    )
)
@given(
    parsers.parse(
        'a synthetic session "{session_id}" created at "{created_at}" in "{directory}"'
        ' with {turns:d} turn and message model "{model}"'
    )
)
def synthetic_session_with_message_model(
    bdd_source_builder, session_id: str, created_at: str, directory: str, turns: int, model: str
) -> None:
    """A session whose model identity rides its assistant session-message data."""
    bdd_source_builder.add_session(
        session_id, created_ms=_epoch_ms(created_at), directory=directory, turns=turns, message_model=model
    )


@given(
    parsers.parse(
        'a synthetic session "{session_id}" created at "{created_at}" in "{directory}"'
        ' with {turns:d} turns and part model "{model}"'
    )
)
@given(
    parsers.parse(
        'a synthetic session "{session_id}" created at "{created_at}" in "{directory}"'
        ' with {turns:d} turn and part model "{model}"'
    )
)
def synthetic_session_with_part_model(
    bdd_source_builder, session_id: str, created_at: str, directory: str, turns: int, model: str
) -> None:
    """A session whose model identity rides one content part of its reply."""
    bdd_source_builder.add_session(
        session_id, created_ms=_epoch_ms(created_at), directory=directory, turns=turns, part_model=model
    )


@given(
    parsers.parse(
        'a synthetic session "{session_id}" created at "{created_at}" in "{directory}"'
        ' with message activity at "{activity_at}" and {turns:d} turns'
    )
)
@given(
    parsers.parse(
        'a synthetic session "{session_id}" created at "{created_at}" in "{directory}"'
        ' with message activity at "{activity_at}" and {turns:d} turn'
    )
)
def synthetic_session_with_message_activity(
    bdd_source_builder, session_id: str, created_at: str, directory: str, activity_at: str, turns: int
) -> None:
    """A session created at one instant whose messages were written at another."""
    bdd_source_builder.add_session(
        session_id,
        created_ms=_epoch_ms(created_at),
        directory=directory,
        turns=turns,
        activity_ms=_epoch_ms(activity_at),
    )


@given(
    parsers.parse(
        'a synthetic legacy session "{session_id}" created at "{created_at}"'
        ' with {turns:d} turns and message model "{model}"'
    )
)
@given(
    parsers.parse(
        'a synthetic legacy session "{session_id}" created at "{created_at}"'
        ' with {turns:d} turn and message model "{model}"'
    )
)
def synthetic_legacy_session_with_message_model(
    bdd_source_builder, session_id: str, created_at: str, turns: int, model: str
) -> None:
    """A legacy-generation session whose model identity rides its message data."""
    bdd_source_builder.add_legacy_session(
        session_id, created_ms=_epoch_ms(created_at), turns=turns, message_model=model
    )


@given("the synthetic six-turn session source database")
def synthetic_six_turn_session_source_database(scenario_context: dict[str, object], tmp_path: Path) -> None:
    """The six-turn fixture's exact sanitized rows as a boundary source."""
    scenario_context["cli_source_path"] = build_six_turn_source_db(tmp_path)


@given("a live WAL-mode source database with one synthetic turn")
def live_wal_mode_source_database_with_one_synthetic_turn(
    scenario_context: dict[str, object], bdd_source_builder, request: pytest.FixtureRequest
) -> None:
    """A WAL-mode source whose committed rows live in the ``-wal`` file.

    The writer connection stays open for the scenario's lifetime so the
    committed-WAL rows (and the ``-wal``/``-shm`` companions) remain
    available to the importer's concurrent read-only snapshot.
    """
    bdd_source_builder.add_session(_WAL_SESSION_ID, created_ms=DEFAULT_SESSION_CREATED_MS, directory=_WAL_DIRECTORY)
    source, writer = bdd_source_builder.materialize_wal()
    request.addfinalizer(writer.close)
    scenario_context["cli_source_path"] = source


@given("a source database with an unsupported schema")
def source_database_with_an_unsupported_schema(scenario_context: dict[str, object], tmp_path: Path) -> None:
    """A readable SQLite database that matches no catalogued schema."""
    source = tmp_path / "bdd-unsupported-source.db"
    connection = sqlite3.connect(source)
    try:
        connection.execute("CREATE TABLE not_an_opencode_table (id TEXT PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()
    scenario_context["cli_source_path"] = source


# -- Given: source mutations and MLflow-side seeding ---------------------------


@given("the source turn's reply changed")
def source_turn_reply_changed(bdd_source_builder) -> None:
    """The single turn's assistant reply changed; its turn identity did not."""
    bdd_source_builder.change_reply(BDD_SESSION_ID)


@given("the source lost its second turn")
def source_lost_its_second_turn(bdd_source_builder) -> None:
    """The second logical turn was removed from the two-turn session."""
    bdd_source_builder.drop_turn(BDD_SESSION_ID, turn=2)


@given(parsers.parse('the source lost every turn of session "{session_id}"'))
def source_lost_every_turn_of_session(bdd_source_builder, session_id: str) -> None:
    """The session row remains but carries zero messages (a tombstone)."""
    bdd_source_builder.drop_all_turns(session_id)


@given("an unowned foreign trace sharing the native session id exists in the experiment")
def unowned_foreign_trace_sharing_native_session_id(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """Seed a foreign (non-importer) trace under the same native session id.

    Driven through the fake gateway's public seams (probe → resolve →
    create) so the trace is stored exactly the way a foreign writer's
    trace would be: no importer ownership metadata, sharing the native
    MLflow session id.
    """
    bdd_gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
    experiment = bdd_gateway.resolve_experiment(BDD_EXPERIMENT_NAME, create=True)
    assert experiment.experiment_id is not None
    foreign = bdd_gateway.create_trace(
        trace_spec(owned=False, native_session_id=BDD_SESSION_ID, experiment_id=experiment.experiment_id)
    )
    stored = bdd_gateway.stored_trace(foreign.trace_id)
    assert stored is not None
    scenario_context["cli_foreign_trace_id"] = foreign.trace_id
    scenario_context["cli_foreign_metadata_before"] = dict(stored.metadata)


@given("replacement artifact verification fails")
def replacement_artifact_verification_fails_at_boundary(bdd_gateway: FakeMlflowGateway) -> None:
    """Every artifact verification after a replacement upload fails.

    The changed turn's owned-candidate verification short-circuits on its
    fingerprint mismatch before any artifact read, so only the NEW
    replacement's post-upload verification observes the failure.
    """
    bdd_gateway.fail_artifact_verification = True


# -- When: boundary invocations ------------------------------------------------


@when("I run a dry-run import through the CLI with fake dependencies")
def run_dry_run_import_through_cli(cli_import_runner) -> None:
    """Run a dry-run import through the real CLI seam with fakes."""
    cli_import_runner(dry_run=True)


@when("I run an import through the CLI with fake dependencies")
def run_import_through_cli(cli_import_runner) -> None:
    """Run a normal import through the real CLI seam with fakes."""
    cli_import_runner()


@when("I rerun the import through the CLI with fake dependencies")
def rerun_import_through_cli(cli_import_runner) -> None:
    """Run the import again over the same source and tracking target."""
    cli_import_runner()


@when("I rerun the import through the CLI with fake dependencies without filters")
def rerun_import_through_cli_without_filters(cli_import_runner) -> None:
    """Run the import again with no optional selection filter."""
    cli_import_runner()


@when(parsers.parse('I run an import through the CLI with fake dependencies from "{from_time}" to "{to_time}"'))
def run_import_with_time_bounds(cli_import_runner, from_time: str, to_time: str) -> None:
    """Run the import bounded by inclusive creation-time bounds."""
    cli_import_runner(extra_args=("--from-time", from_time, "--to-time", to_time))


@when(
    parsers.parse(
        'I run an import through the CLI with fake dependencies from "{from_time}" to "{to_time}"'
        ' in directory "{directory}" for model "{model}" with minimum user turns {minimum:d}'
    )
)
def run_import_with_all_filters(
    cli_import_runner, from_time: str, to_time: str, directory: str, model: str, minimum: int
) -> None:
    """Run the import with every selection dimension supplied at once."""
    cli_import_runner(
        extra_args=(
            "--from-time",
            from_time,
            "--to-time",
            to_time,
            "--directory",
            directory,
            "--model",
            model,
            "--min-user-turns",
            str(minimum),
        )
    )


@when(parsers.parse('I run an import through the CLI with fake dependencies for model "{model}"'))
def run_import_for_model(cli_import_runner, model: str) -> None:
    """Run the import filtered by one exact model identity."""
    cli_import_runner(extra_args=("--model", model))


@when(parsers.parse('I run an import through the CLI with fake dependencies in directory "{directory}"'))
def run_import_in_directory(cli_import_runner, directory: str) -> None:
    """Run the import restricted to one directory and its descendants."""
    cli_import_runner(extra_args=("--directory", directory))


@when(parsers.parse('I rerun the import through the CLI with fake dependencies filtered to directory "{directory}"'))
def rerun_import_filtered_to_directory(cli_import_runner, directory: str) -> None:
    """Run the import again restricted to one directory and its descendants."""
    cli_import_runner(extra_args=("--directory", directory))


@when("I run a redacting import through the CLI with fake dependencies")
def run_redacting_import_through_cli(cli_import_runner) -> None:
    """Run the import with deterministic redaction enabled."""
    cli_import_runner(extra_args=("--redact",))


# -- Then: report shape, selection, and exit status -----------------------------


@then(parsers.parse('only the sessions "{session_ids}" were selected'))
def only_the_sessions_were_selected(scenario_context: dict[str, object], session_ids: str) -> None:
    """Exactly the named sessions passed every supplied filter."""
    run = _latest_run(scenario_context)
    expected = {item.strip() for item in session_ids.split(",")}
    selected = {session["session_id"] for session in run.report["sessions"] if session["selected"]}
    assert selected == expected
    assert run.report["source"]["sessions_selected"] == len(expected)
    for session in run.report["sessions"]:
        if session["selected"]:
            assert session["turns"], "a selected session must carry its planned turns"
            assert all(turn["action"] == "imported" for turn in session["turns"])
        else:
            assert session["action"] == "skipped"
            assert session["reason"] == "not_selected"
            assert session["turns"] == []


def _assert_summary_bucket(scenario_context: dict[str, object], bucket: str, count: int) -> None:
    assert _latest_run(scenario_context).report["summary"][bucket] == count


@then(parsers.parse("the report records {count:d} imported turns"))
def report_records_imported_turns(scenario_context: dict[str, object], count: int) -> None:
    """The summary counts exactly ``count`` imported turns."""
    _assert_summary_bucket(scenario_context, "turns_imported", count)


@then(parsers.parse("the report records {count:d} imported turn"))
def report_records_one_imported_turn(scenario_context: dict[str, object], count: int) -> None:
    """The summary counts exactly ``count`` imported turn (singular phrasing)."""
    _assert_summary_bucket(scenario_context, "turns_imported", count)


def _assert_session_imported_turns(scenario_context: dict[str, object], session_id: str, count: int) -> None:
    turns = _session_turns(scenario_context, session_id)
    assert len(turns) == count
    assert all(turn["action"] == "imported" and turn["trace_id"] for turn in turns)


@then(parsers.parse('the report records {count:d} imported turns for session "{session_id}"'))
def report_records_imported_turns_for_session(scenario_context: dict[str, object], count: int, session_id: str) -> None:
    """One session imported exactly ``count`` turns."""
    _assert_session_imported_turns(scenario_context, session_id, count)


@then(parsers.parse('the report records {count:d} imported turn for session "{session_id}"'))
def report_records_one_imported_turn_for_session(
    scenario_context: dict[str, object], count: int, session_id: str
) -> None:
    """One session imported exactly ``count`` turn (singular phrasing)."""
    _assert_session_imported_turns(scenario_context, session_id, count)


@then(parsers.parse('the report carries the safe error "{code}"'))
def report_carries_the_safe_error(scenario_context: dict[str, object], code: str) -> None:
    """The result document's first error entry carries the safe code."""
    errors = _latest_run(scenario_context).report["errors"]
    assert errors, "the report carries no error entry"
    assert errors[0]["code"] == code


@then("the report marks the turn as imported")
def report_marks_the_turn_as_imported(scenario_context: dict[str, object]) -> None:
    """The single session's turn was imported with a trace id."""
    turn = _session_turns(scenario_context, BDD_SESSION_ID)[0]
    assert turn["action"] == "imported"
    assert turn["trace_id"]


@then("the report marks the turn as failed")
def report_marks_the_turn_as_failed(scenario_context: dict[str, object]) -> None:
    """The single session's turn failed with a safe error code."""
    turn = _session_turns(scenario_context, BDD_SESSION_ID)[0]
    assert turn["action"] == "failed"
    assert turn["error_code"]


@then("the report records both turns as imported")
def report_records_both_turns_as_imported(scenario_context: dict[str, object]) -> None:
    """The two-turn session imported both of its turns."""
    turns = _session_turns(scenario_context, BDD_SESSION_ID)
    assert [turn["action"] for turn in turns] == ["imported", "imported"]
    assert all(turn["trace_id"] for turn in turns)


@then("the report marks every turn as skipped unchanged")
def report_marks_every_turn_as_skipped_unchanged(scenario_context: dict[str, object]) -> None:
    """Every planned turn was recognized as unchanged and skipped."""
    turns = _session_turns(scenario_context, BDD_SESSION_ID)
    assert turns, "the rerun planned no turns"
    assert all(turn["action"] == "skipped" and turn["reason"] == "unchanged" for turn in turns)


@then(parsers.parse('the report records session "{session_id}" as failed'))
def report_records_session_as_failed(scenario_context: dict[str, object], session_id: str) -> None:
    """One scanned session's action is failed with failed turn entries."""
    session = _report_session(_latest_run(scenario_context).report, session_id)
    assert session["selected"] is True
    assert session["action"] == "failed"
    assert session["turns"]
    assert all(turn["action"] == "failed" and turn["error_code"] for turn in session["turns"])


@then(parsers.parse('the report records session "{session_id}" as imported'))
def report_records_session_as_imported(scenario_context: dict[str, object], session_id: str) -> None:
    """One scanned session's action is imported with trace ids."""
    session = _report_session(_latest_run(scenario_context).report, session_id)
    assert session["selected"] is True
    assert session["action"] == "imported"
    assert all(turn["action"] == "imported" and turn["trace_id"] for turn in session["turns"])


# -- Then: experiment create/reuse ----------------------------------------------


@then("the report records the experiment as created")
def report_records_the_experiment_as_created(scenario_context: dict[str, object]) -> None:
    """The experiment name did not exist, so the run created it."""
    experiment = _latest_run(scenario_context).report["experiment"]
    assert experiment["name"] == BDD_EXPERIMENT_NAME
    assert experiment["status"] == "created"
    assert experiment["id"]


@then("the gateway recorded exactly one experiment creation")
def gateway_recorded_exactly_one_experiment_creation(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """Exactly one experiment_create mutation targeted the experiment name."""
    creations = [mutation for mutation in bdd_gateway.mutations if mutation[0] == "experiment_create"]
    assert len(creations) == 1
    assert creations[0][1] == BDD_EXPERIMENT_NAME
    experiment = bdd_gateway.stored_experiment(BDD_EXPERIMENT_NAME)
    assert experiment is not None
    scenario_context["cli_experiment_settings"] = (experiment.experiment_id, experiment.artifact_location)


@then("the report records the experiment as resolved with the same experiment id")
def report_records_experiment_resolved_with_same_id(scenario_context: dict[str, object]) -> None:
    """The rerun resolved (did not create) the experiment, keeping its id."""
    runs = _cli_runs(scenario_context)
    previous, latest = runs[-2], runs[-1]
    assert latest.report["experiment"]["status"] == "resolved"
    assert latest.report["experiment"]["id"] == previous.report["experiment"]["id"]


@then("the gateway recorded no new experiment creation")
def gateway_recorded_no_new_experiment_creation(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """The rerun reused the experiment instead of creating it again."""
    run = _latest_run(scenario_context)
    assert [m for m in bdd_gateway.mutations[run.mutation_count_before :] if m[0] == "experiment_create"] == []
    assert any(event.kind == "experiment_reuse" for event in bdd_gateway.events[run.event_marker :])


@then("the reused experiment kept its original settings")
def reused_experiment_kept_its_original_settings(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """Reuse changed neither the experiment id nor its artifact location."""
    settings = scenario_context["cli_experiment_settings"]
    experiment = bdd_gateway.stored_experiment(BDD_EXPERIMENT_NAME)
    assert experiment is not None
    assert (experiment.experiment_id, experiment.artifact_location) == settings


# -- Then: reruns, replacement, and rollback ------------------------------------


@then("the gateway recorded no new mutations")
def gateway_recorded_no_new_mutations(scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway) -> None:
    """The latest run mutated nothing: no experiment, trace, or artifact write."""
    run = _latest_run(scenario_context)
    assert bdd_gateway.mutations[run.mutation_count_before :] == []


@then("the owned traces were not duplicated")
def owned_traces_were_not_duplicated(scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway) -> None:
    """The rerun kept every owned trace id; no duplicate was created."""
    runs = _cli_runs(scenario_context)
    previous_ids = [turn["trace_id"] for turn in _report_session(runs[-2].report, BDD_SESSION_ID)["turns"]]
    latest_ids = [turn["trace_id"] for turn in _report_session(runs[-1].report, BDD_SESSION_ID)["turns"]]
    assert previous_ids and latest_ids == previous_ids
    assert len(bdd_gateway.trace_ids()) == len(previous_ids)


@then("the report marks the turn as updated")
def report_marks_the_turn_as_updated(scenario_context: dict[str, object]) -> None:
    """The changed turn was replaced, not skipped or failed."""
    turn = _session_turns(scenario_context, BDD_SESSION_ID)[0]
    assert turn["action"] == "updated"
    assert turn["trace_id"]


@then("the report records one retired trace")
def report_records_one_retired_trace(scenario_context: dict[str, object]) -> None:
    """The run retired exactly one previously owned trace."""
    assert _latest_run(scenario_context).report["summary"]["retired_traces"] == 1


@then("the old trace is retired and the replacement is active")
def old_trace_is_retired_and_replacement_is_active(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """The replaced trace is gone; its verified replacement remains active."""
    runs = _cli_runs(scenario_context)
    old_id = _report_session(runs[-2].report, BDD_SESSION_ID)["turns"][0]["trace_id"]
    new_id = _report_session(runs[-1].report, BDD_SESSION_ID)["turns"][0]["trace_id"]
    assert old_id is not None and new_id is not None and new_id != old_id
    assert not bdd_gateway.has_trace(old_id)
    stored = bdd_gateway.stored_trace(new_id)
    assert stored is not None
    assert stored.state == "OK"


@then("the failed replacement was rolled back")
def failed_replacement_was_rolled_back(scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway) -> None:
    """The new trace created for the failed replacement was retired."""
    turn = _session_turns(scenario_context, BDD_SESSION_ID)[0]
    assert turn["trace_id"] is not None
    assert not bdd_gateway.has_trace(turn["trace_id"])
    assert turn["error_code"] == "artifact_verification_failed"


# -- Then: stale retirement scope -------------------------------------------------


@then("the report marks the first turn as skipped unchanged")
def report_marks_the_first_turn_as_skipped_unchanged(scenario_context: dict[str, object]) -> None:
    """The surviving turn kept its trace and was skipped as unchanged."""
    turn = _session_turns(scenario_context, BDD_SESSION_ID)[0]
    assert turn["action"] == "skipped"
    assert turn["reason"] == "unchanged"


@then("the report marks the lost turn as a retired stale turn")
def report_marks_the_lost_turn_as_a_retired_stale_turn(scenario_context: dict[str, object]) -> None:
    """The removed turn's owned trace was retired with source_present=false."""
    turn = _session_turns(scenario_context, BDD_SESSION_ID)[1]
    assert turn["action"] == "retired"
    assert turn["reason"] == "stale_turn"
    assert turn["source_present"] is False
    assert turn["trace_id"] is not None


@then("the gateway no longer stores the retired trace")
def gateway_no_longer_stores_the_retired_trace(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """The stale turn's owned trace was actually retired."""
    turn = _session_turns(scenario_context, BDD_SESSION_ID)[1]
    assert turn["trace_id"] is not None
    assert not bdd_gateway.has_trace(turn["trace_id"])


@then("the report counts one unexamined owned trace")
def report_counts_one_unexamined_owned_trace(scenario_context: dict[str, object]) -> None:
    """The unselected session's owned trace stayed outside the run's scope."""
    assert _latest_run(scenario_context).report["summary"]["unexamined_owned_traces"] == 1


def _recorded_trace_ids(scenario_context: dict[str, object], session_id: str) -> set[str]:
    """Every trace id any run recorded for one session."""
    trace_ids = {
        turn["trace_id"]
        for run in _cli_runs(scenario_context)
        for session in run.report["sessions"]
        for turn in session["turns"]
        if turn["session_id"] == session_id and turn["trace_id"]
    }
    assert trace_ids, f"no run recorded a trace for session {session_id!r}"
    return trace_ids


@then(parsers.parse('the owned traces of session "{session_id}" remain stored'))
def owned_traces_of_session_remain_stored(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway, session_id: str
) -> None:
    """No run retired the unselected session's owned traces."""
    for trace_id in _recorded_trace_ids(scenario_context, session_id):
        assert bdd_gateway.has_trace(trace_id), f"trace {trace_id} was unexpectedly retired"


@then(parsers.parse('the report marks every trace of session "{session_id}" as retired tombstones'))
def report_marks_every_trace_as_retired_tombstones(scenario_context: dict[str, object], session_id: str) -> None:
    """The zero-turn session's owned traces retired as session tombstones."""
    turns = _session_turns(scenario_context, session_id)
    assert turns
    assert all(
        turn["action"] == "retired" and turn["reason"] == "session_tombstone" and turn["source_present"] is False
        for turn in turns
    )


@then(parsers.parse('the gateway no longer stores any trace of session "{session_id}"'))
def gateway_no_longer_stores_any_trace_of_session(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway, session_id: str
) -> None:
    """Every owned trace of the tombstoned session was retired."""
    for trace_id in _recorded_trace_ids(scenario_context, session_id):
        assert not bdd_gateway.has_trace(trace_id), f"trace {trace_id} was not retired"


# -- Then: foreign-trace protection -------------------------------------------------


@then("the foreign trace remains stored and unchanged")
def foreign_trace_remains_stored_and_unchanged(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """The foreign trace survived the run byte-for-byte unchanged."""
    trace_id = scenario_context["cli_foreign_trace_id"]
    stored = bdd_gateway.stored_trace(trace_id)
    assert stored is not None, "the foreign trace must not be retired"
    assert dict(stored.metadata) == scenario_context["cli_foreign_metadata_before"]
    assert stored.metadata.get("opencode.importer") is None


@then("no gateway event targeted the foreign trace")
def no_gateway_event_targeted_the_foreign_trace(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """No mutation or verification in the run concerned the foreign trace."""
    trace_id = scenario_context["cli_foreign_trace_id"]
    run = _latest_run(scenario_context)
    targeted = [event for event in bdd_gateway.events[run.event_marker :] if trace_id in event.details]
    assert targeted == []


# -- Then: six-turn representation through the gateway --------------------------------


@then(parsers.parse('the gateway stored {count:d} traces for the native session id "{session_id}"'))
def gateway_stored_traces_for_native_session_id(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway, count: int, session_id: str
) -> None:
    """One stored trace per logical turn, all under the native session id."""
    trace_ids = [trace_id for _index, trace_id in _latest_turn_entries(scenario_context)]
    assert len(trace_ids) == count
    for trace_id in trace_ids:
        stored = bdd_gateway.stored_trace(trace_id)
        assert stored is not None
        assert stored.native_session_id == session_id
        assert stored.metadata[TRACE_SESSION_METADATA_KEY] == session_id


@then(parsers.parse("the imported traces carry {count:d} distinct trace ids"))
def imported_traces_carry_distinct_trace_ids(scenario_context: dict[str, object], count: int) -> None:
    """Every logical turn kept its own distinct MLflow trace id."""
    trace_ids = [trace_id for _index, trace_id in _latest_turn_entries(scenario_context)]
    assert len(trace_ids) == count
    assert len(set(trace_ids)) == count


@then("the stored root statuses follow the source outcomes")
def stored_root_statuses_follow_the_source_outcomes(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """The six-turn outcomes round-tripped as authoritative root statuses."""
    entries = dict(_latest_turn_entries(scenario_context))
    expected = {1: "OK", 3: "UNSET", 4: "ERROR"}
    for turn_index, status in expected.items():
        stored = bdd_gateway.stored_trace(entries[turn_index])
        assert stored is not None
        assert stored.root_status == status


@then(parsers.parse("the root input of imported turn {index:d} is the first user prompt"))
def root_input_of_imported_turn_is_the_first_user_prompt(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway, index: int
) -> None:
    """The stored root input is the turn's first user prompt payload."""
    stored = _stored_turn_trace(scenario_context, bdd_gateway, index)
    assert stored.inputs == {"content": [{"type": "text", "text": "synthetic prompt one"}]}


@then(parsers.parse("the root output of imported turn {index:d} is the final assistant text part"))
def root_output_of_imported_turn_is_the_final_text_part(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway, index: int
) -> None:
    """The stored root output is the turn's final user-visible text part."""
    stored = _stored_turn_trace(scenario_context, bdd_gateway, index)
    assert stored.outputs == [{"type": "text", "text": "synthetic answer one"}]


@then(parsers.parse("the root output of imported turn {index:d} is omitted"))
def root_output_of_imported_turn_is_omitted(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway, index: int
) -> None:
    """A turn without an assistant result never manufactured output."""
    stored = _stored_turn_trace(scenario_context, bdd_gateway, index)
    assert stored.outputs is None


@then(parsers.parse('imported turn {index:d} has a child of span type "{span_type}" with source id "{source_id}"'))
def imported_turn_has_child_of_span_type(
    scenario_context: dict[str, object],
    bdd_gateway: FakeMlflowGateway,
    index: int,
    span_type: str,
    source_id: str,
) -> None:
    """The typed child mapping preserved the record's source identity."""
    stored = _stored_turn_trace(scenario_context, bdd_gateway, index)
    span = stored.spans_by_source.get(source_id)
    if span is None:
        span = next(
            (candidate for key, candidate in stored.spans_by_source.items() if key.endswith(f":{source_id}")),
            None,
        )
    assert span is not None, f"the stored trace carries no child with source id {source_id!r}"
    assert span.span_type == span_type


@then(parsers.parse('the child of imported turn {index:d} with source id "{source_id}" is attached to the trace root'))
def imported_turn_child_is_attached_to_the_trace_root(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway, index: int, source_id: str
) -> None:
    """A part of the root-input prompt attached to the trace root span."""
    stored = _stored_turn_trace(scenario_context, bdd_gateway, index)
    span = stored.spans_by_source.get(source_id)
    if span is None:
        span = next(
            (candidate for key, candidate in stored.spans_by_source.items() if key.endswith(f":{source_id}")),
            None,
        )
    assert span is not None
    assert span.parent_id == stored.root_span_id


# -- Then: source safety observables -----------------------------------------------------


@then("the source database bytes are unchanged")
def source_database_bytes_are_unchanged(scenario_context: dict[str, object]) -> None:
    """The importer never rewrote the live source database file."""
    run = _latest_run(scenario_context)
    assert run.source_state_before["db"] is not None
    assert run.source_state_after["db"] == run.source_state_before["db"]


@then("the source WAL companion is unchanged")
def source_wal_companion_is_unchanged(scenario_context: dict[str, object]) -> None:
    """The importer neither appended to nor checkpointed the WAL."""
    run = _latest_run(scenario_context)
    assert run.source_state_before["-wal"] is not None, "the WAL source must carry a -wal companion"
    assert run.source_state_after["-wal"] == run.source_state_before["-wal"]


@then("the source SHM companion keeps its size and inode")
def source_shm_companion_keeps_its_size_and_inode(scenario_context: dict[str, object]) -> None:
    """The wal-index companion was never deleted or recreated."""
    run = _latest_run(scenario_context)
    assert run.source_state_before["-shm"] is not None, "the WAL source must carry a -shm companion"
    assert run.source_state_after["-shm"] == run.source_state_before["-shm"]


# -- Then: redaction observables -----------------------------------------------------------


def _latest_stored_trace(scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway):
    """The gateway-stored trace of the latest run's first turn."""
    trace_id = _latest_turn_entries(scenario_context)[0][1]
    stored = bdd_gateway.stored_trace(trace_id)
    assert stored is not None
    return stored


@then("the stored trace content is masked without the bearer token")
def stored_trace_content_is_masked_without_the_bearer_token(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """The persisted root values carry the redaction marker, never the token."""
    stored = _latest_stored_trace(scenario_context, bdd_gateway)
    content = json.dumps({"inputs": stored.inputs, "outputs": stored.outputs})
    assert "[REDACTED]" in content
    assert _CLI_BEARER_TOKEN not in content


@then("the stored artifact is masked without the bearer token")
def stored_artifact_is_masked_without_the_bearer_token(
    scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway
) -> None:
    """The persisted artifact document carries the marker, never the token."""
    stored = _latest_stored_trace(scenario_context, bdd_gateway)
    relative_path = stored.tags["opencode.artifact_path"]
    artifact = bdd_gateway.stored_artifact(stored.trace_id, relative_path).decode("utf-8")
    assert "[REDACTED]" in artifact
    assert _CLI_BEARER_TOKEN not in artifact


@then("the report shows redaction counts without the bearer token")
def report_shows_redaction_counts_without_the_bearer_token(scenario_context: dict[str, object]) -> None:
    """The report carries detector counts; the token never reached the output."""
    run = _latest_run(scenario_context)
    redaction = run.report["redaction"]
    assert redaction["mode"] == "redacted"
    assert redaction["detector_counts"], "the report must carry redaction detector counts"
    assert redaction["affected_field_count"] >= 1
    assert redaction["affected_record_count"] >= 1
    assert _CLI_BEARER_TOKEN not in run.output


@then("the redacted identifiers survive")
def the_redacted_identifiers_survive(scenario_context: dict[str, object], bdd_gateway: FakeMlflowGateway) -> None:
    """Redaction masked content but preserved reconciliation identifiers."""
    stored = _latest_stored_trace(scenario_context, bdd_gateway)
    assert stored.metadata["opencode.session_id"] == BDD_SESSION_ID
    assert stored.metadata["opencode.importer"] == "opencode-mlflow"
    relative_path = stored.tags["opencode.artifact_path"]
    artifact = bdd_gateway.stored_artifact(stored.trace_id, relative_path).decode("utf-8")
    assert "ses-bdd-cli-user-1" in artifact, "the artifact must preserve the record source ids"
    assert _CLI_PROMPT_SENTINEL in artifact, "non-secret prompt text must survive redaction"


# -- Then: legacy report assertions (kept from the earlier tasks) ----------------------------


@then("the JSON report marks the turn as skipped for dry-run")
def json_report_marks_the_turn_as_skipped_for_dry_run(scenario_context: dict[str, object]) -> None:
    """The dry-run report carries the non-mutating action and its prediction."""
    report = scenario_context["cli_report"]
    turn = report["sessions"][0]["turns"][0]
    assert turn["action"] == "skipped"
    assert turn["reason"] == "dry_run"
    assert turn["planned_action"] in {"imported", "updated", "skipped", "failed", "retired"}


@then("the JSON report records the failed turn")
def json_report_records_the_failed_turn(scenario_context: dict[str, object]) -> None:
    """The report isolates the failed turn with a safe error code."""
    report = scenario_context["cli_report"]
    turn = report["sessions"][0]["turns"][0]
    assert turn["action"] == "failed"
    assert turn["error_code"]
    assert report["summary"]["turns_failed"] == 1


@then("the fake gateway recorded no mutations")
def fake_gateway_recorded_no_mutations(scenario_context: dict[str, object]) -> None:
    """The run never mutated the tracking target."""
    dependencies = scenario_context["cli_dependencies"]
    assert dependencies.gateway.mutations == []


@then("the fake lock factory recorded no acquisitions")
def fake_lock_factory_recorded_no_acquisitions(scenario_context: dict[str, object]) -> None:
    """The run never acquired the import lock."""
    dependencies = scenario_context["cli_dependencies"]
    assert dependencies.lock_factory.acquisitions == []


@then("the report output contains no prompt content")
def report_output_contains_no_prompt_content(scenario_context: dict[str, object]) -> None:
    """No prompt text reached the rendered report."""
    assert _CLI_PROMPT_SENTINEL not in scenario_context["cli_output"]


@then("the report output contains no bearer token")
def report_output_contains_no_bearer_token(scenario_context: dict[str, object]) -> None:
    """No matched secret reached the rendered report."""
    assert _CLI_BEARER_TOKEN not in scenario_context["cli_output"]


@then("the report output contains no absolute source path")
def report_output_contains_no_absolute_source_path(scenario_context: dict[str, object]) -> None:
    """The source database path never reached the rendered report."""
    source = scenario_context["cli_source_path"]
    assert isinstance(source, Path)
    assert str(source) not in scenario_context["cli_output"]
