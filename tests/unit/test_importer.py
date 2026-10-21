"""Unit tests for the two-pass import orchestration.

The tests drive :func:`opencode_mlflow.importer.run_import` through the
REAL source pipeline (fixture database → live read-only probe →
registered exporter → normalized snapshot → replay adapter → discovery)
with in-memory MLflow/lock fakes, asserting the binding phase order, the
dry-run no-mutation contract, per-item failure isolation, exit statuses,
the replay fingerprint check, redaction wiring, and temporary-file
cleanup.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import opencode_mlflow.importer as importer_module
from opencode_mlflow.config import ImportConfig
from opencode_mlflow.content.fingerprint import build_effective_turn, source_fingerprint
from opencode_mlflow.contracts import IDENTITY_REDACTOR, RedactionMode
from opencode_mlflow.errors import SourceCompatibilityError
from opencode_mlflow.importer import exit_status, replay_turn, run_import
from opencode_mlflow.mlflow.capabilities import CapabilityRequest
from opencode_mlflow.mlflow.reconciliation import OwnedTraceCandidate
from opencode_mlflow.reporting import render_human, render_json
from opencode_mlflow.source.discovery import DiscoveryPlanReader, DiscoveryTurnEntry, discover_sessions
from opencode_mlflow.source.probe import probe_schema
from opencode_mlflow.source.registry import build_live_exporter
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter, ReplayTurn
from opencode_mlflow.source.selection import SelectionEvidenceEvaluator, SelectionFilters
from opencode_mlflow.source.snapshot import (
    RedactionCallback,
    SourceExporter,
    SourceSnapshot,
    create_normalized_snapshot,
)
from tests.support.fake_gateway import FakeMlflowGateway
from tests.support.fake_source import ScriptedExporter
from tests.support.fake_turn import fake_record
from tests.support.import_fakes import (
    DEFAULT_SESSION_CREATED_MS,
    LEGACY_DDL,
    UNIT_PROMPT_JSON,
    UNIT_REPLY_JSON,
    UNIT_SESSION_ID,
    FakeLockFactory,
    SyntheticSourceBuilder,
    build_current_source_db,
    build_import_fakes,
)
from tests.support.report_assertions import assert_counts_are_disjoint, assert_no_sentinels, assert_result_document

pytestmark = pytest.mark.unit


def _artifact_uploads(gateway: FakeMlflowGateway) -> list[tuple[str, str]]:
    """Every recorded artifact upload as a (trace id, relative path) pair."""
    return [(event.details[0], event.details[1]) for event in gateway.events if event.kind == "artifact_upload"]


# ---------------------------------------------------------------------------
# The task's verbatim dry-run example
# ---------------------------------------------------------------------------


def test_dry_run_reports_planned_actions_without_mutation(import_dependencies, config):
    result = run_import(replace(config, dry_run=True), import_dependencies)

    assert result.dry_run is True
    assert result.report["sessions"][0]["turns"][0]["planned_action"] in {
        "imported",
        "updated",
        "skipped",
        "failed",
        "retired",
    }
    assert result.report["sessions"][0]["turns"][0]["action"] == "skipped"
    assert result.report["sessions"][0]["turns"][0]["reason"] == "dry_run"
    assert import_dependencies.gateway.mutations == []
    assert import_dependencies.lock_factory.acquisitions == []


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_normal_run_imports_every_selected_turn(import_dependencies, config):
    result = run_import(config, import_dependencies)

    assert exit_status(result) == 0
    assert result.dry_run is False
    assert_result_document(result.report)
    assert_counts_are_disjoint(result.report)
    turns = result.report["sessions"][0]["turns"]
    assert [turn["action"] for turn in turns] == ["imported", "imported"]
    assert all(turn["trace_id"] for turn in turns)
    assert all(import_dependencies.gateway.has_trace(turn["trace_id"]) for turn in turns)
    summary = result.report["summary"]
    assert summary["turns_imported"] == 2
    assert summary["retired_traces"] == 0
    assert summary["unexamined_owned_traces"] == 0
    assert import_dependencies.lock_factory.acquisitions == [
        (import_dependencies.gateway.tracking_uri, config.experiment_name)
    ]
    assert len(import_dependencies.lock_factory.releases) == len(import_dependencies.lock_factory.acquisitions)
    assert import_dependencies.gateway.mutations


def test_rerun_converges_with_unchanged_skips(import_dependencies, config):
    first = run_import(config, import_dependencies)
    assert exit_status(first) == 0
    mutations_after_first = len(import_dependencies.gateway.mutations)

    second = run_import(config, import_dependencies)

    assert exit_status(second) == 0
    turns = second.report["sessions"][0]["turns"]
    assert [turn["action"] for turn in turns] == ["skipped", "skipped"]
    assert all(turn["reason"] == "unchanged" for turn in turns)
    assert all(turn["trace_id"] for turn in turns)
    assert len(import_dependencies.gateway.mutations) == mutations_after_first
    assert second.report["summary"]["turns_skipped"] == 2


def test_dry_run_exits_zero_without_writes(import_dependencies, config):
    result = run_import(replace(config, dry_run=True), import_dependencies)

    assert exit_status(result) == 0
    assert result.report["summary"]["turns_skipped"] == 2
    assert result.report["summary"]["planned"] == {"imported": 2, "updated": 0, "retired": 0}
    assert import_dependencies.gateway.mutations == []
    assert import_dependencies.gateway.trace_ids() == []


def test_experiment_status_is_reported_per_mode(import_dependencies, config):
    dry = run_import(replace(config, dry_run=True), import_dependencies)
    assert dry.report["experiment"] == {"name": config.experiment_name, "id": None, "status": "planned"}

    normal = run_import(config, import_dependencies)
    experiment = normal.report["experiment"]
    assert experiment["name"] == config.experiment_name
    assert experiment["status"] == "created"
    assert experiment["id"]

    again = run_import(config, import_dependencies)
    assert again.report["experiment"]["status"] == "resolved"


# ---------------------------------------------------------------------------
# Phase order and source-safety invariants
# ---------------------------------------------------------------------------


def _wrap_method(target: object, name: str, order: list[str], label: str) -> None:
    """Wrap one method so each call appends ``label`` to the shared order."""
    original = getattr(target, name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        order.append(label)
        return original(*args, **kwargs)

    setattr(target, name, wrapped)


def test_phase_order_and_live_connection_safety(config, restrictive_tmp_path: Path) -> None:
    order: list[str] = []
    captured: dict[str, object] = {}
    created_dirs: list[Path] = []
    gateway = FakeMlflowGateway()
    for method, label in (
        ("probe", "mlflow_probe"),
        ("resolve_experiment", "mlflow_resolve_experiment"),
        ("create_trace", "mlflow_create_trace"),
        ("upload_named_json", "mlflow_upload_artifact"),
        ("retire_trace", "mlflow_retire_trace"),
    ):
        _wrap_method(gateway, method, order, label)

    def exporter_factory(connection: sqlite3.Connection, description: Any) -> SourceExporter:
        captured["live_connection"] = connection
        order.append("exporter_built")
        return build_live_exporter(connection, description)

    def snapshot_factory(
        path: Path,
        temp_root: Path,
        exporter: SourceExporter,
        redaction: RedactionCallback,
        selection_evaluator: SelectionEvidenceEvaluator,
    ) -> AbstractContextManager[SourceSnapshot]:
        order.append("snapshot_open")

        @contextmanager
        def wrapped() -> Iterator[SourceSnapshot]:
            with create_normalized_snapshot(
                path,
                temp_root,
                exporter,
                redaction,
                selection_evaluator=selection_evaluator,
            ) as snapshot:
                yield snapshot
            order.append("snapshot_exited")

        return wrapped()

    def replay_adapter_factory(snapshot: SourceSnapshot) -> NormalizedSnapshotAdapter:
        order.append("replay_adapter_built")
        return NormalizedSnapshotAdapter(snapshot)

    def gateway_factory(dry_run: bool) -> FakeMlflowGateway:
        order.append("gateway_created")
        run_root = captured["run_root"]
        assert isinstance(run_root, Path)
        # Discovery must have written its plan before any MLflow work begins.
        assert any(run_root.glob("*.jsonl")), "the discovery plan must exist before MLflow work"
        order.append("discovery_plan_present")
        # No live source connection may remain during MLflow work.
        live = captured["live_connection"]
        assert isinstance(live, sqlite3.Connection)
        try:
            live.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            order.append("live_connection_closed")
        else:
            order.append("live_connection_still_open")
        gateway.dry_run = dry_run
        return gateway

    def lock_factory(tracking_endpoint: str, experiment_name: str) -> AbstractContextManager[Any]:
        order.append(f"lock:{experiment_name}")
        return FakeLockFactory()(tracking_endpoint, experiment_name)

    def temp_directory_factory(prefix: str) -> Path:
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=restrictive_tmp_path))
        created_dirs.append(path)
        captured["run_root"] = path
        return path

    dependencies = replace(
        build_import_fakes(restrictive_tmp_path=restrictive_tmp_path, gateway=gateway),
        exporter_factory=exporter_factory,
        snapshot_factory=snapshot_factory,
        replay_adapter_factory=replay_adapter_factory,
        gateway_factory=gateway_factory,
        lock_factory=lock_factory,
        temp_directory_factory=temp_directory_factory,
    )

    result = run_import(config, dependencies)

    assert exit_status(result) == 0
    assert order[0] == "exporter_built"
    assert order.index("snapshot_open") > order.index("exporter_built")
    assert order.index("gateway_created") > order.index("replay_adapter_built")
    assert "discovery_plan_present" in order
    assert "live_connection_closed" in order
    assert "live_connection_still_open" not in order
    assert order.index("mlflow_probe") > order.index("gateway_created")
    assert order.index("lock:unit-import-experiment") > order.index("mlflow_probe")
    assert order.index("mlflow_resolve_experiment") > order.index("lock:unit-import-experiment")
    first_write = min(
        order.index(label)
        for label in ("mlflow_create_trace", "mlflow_upload_artifact", "mlflow_retire_trace")
        if label in order
    )
    assert first_write > order.index("mlflow_resolve_experiment")
    assert order[-1] == "snapshot_exited"
    assert gateway.mutations
    assert not any(path.exists() for path in created_dirs)


def test_four_argument_snapshot_factory_remains_compatible_without_evidence_filters(
    config, restrictive_tmp_path: Path
) -> None:
    calls: list[str] = []

    def snapshot_factory(
        path: Path, temp_root: Path, exporter: SourceExporter, redaction: RedactionCallback
    ) -> AbstractContextManager[SourceSnapshot]:
        calls.append("four")
        return create_normalized_snapshot(path, temp_root, exporter, redaction)

    dependencies = replace(
        build_import_fakes(restrictive_tmp_path=restrictive_tmp_path),
        snapshot_factory=snapshot_factory,
    )

    result = run_import(replace(config, dry_run=True), dependencies)

    assert exit_status(result) == 0
    assert calls == ["four"]


def test_four_argument_snapshot_factory_is_rejected_when_selector_evidence_is_required(
    config, restrictive_tmp_path: Path
) -> None:
    calls: list[str] = []

    def snapshot_factory(
        path: Path, temp_root: Path, exporter: SourceExporter, redaction: RedactionCallback
    ) -> AbstractContextManager[SourceSnapshot]:
        calls.append("four")
        return create_normalized_snapshot(path, temp_root, exporter, redaction)

    dependencies = replace(
        build_import_fakes(restrictive_tmp_path=restrictive_tmp_path),
        snapshot_factory=snapshot_factory,
    )

    result = run_import(
        replace(config, dry_run=True, directory=restrictive_tmp_path / "repo", model="provider/model"),
        dependencies,
    )

    assert exit_status(result) == 2
    assert result.error is not None
    assert result.error.code == "snapshot_factory_selection_evidence_unsupported"
    assert calls == []


@pytest.mark.parametrize(
    "selector",
    [
        {"directory": Path("/repo/project-alpha")},
        {"model": "synth-provider/synth-model"},
    ],
    ids=["directory-only", "model-only"],
)
def test_four_argument_snapshot_factory_is_rejected_for_each_selector_alone(
    config, restrictive_tmp_path: Path, selector: dict[str, object]
) -> None:
    """A legacy custom factory cannot silently lose one filter's evidence."""
    calls: list[str] = []

    def snapshot_factory(
        path: Path, temp_root: Path, exporter: SourceExporter, redaction: RedactionCallback
    ) -> AbstractContextManager[SourceSnapshot]:
        calls.append("four")
        return create_normalized_snapshot(path, temp_root, exporter, redaction)

    dependencies = replace(
        build_import_fakes(restrictive_tmp_path=restrictive_tmp_path),
        snapshot_factory=snapshot_factory,
    )

    result = run_import(replace(config, dry_run=True, **selector), dependencies)

    assert exit_status(result) == 2
    assert result.error is not None
    assert result.error.code == "snapshot_factory_selection_evidence_unsupported"
    assert calls == []


def test_type_error_inside_compatible_snapshot_factory_is_not_retried_with_four_arguments(
    config, restrictive_tmp_path: Path
) -> None:
    calls: list[int] = []

    def snapshot_factory(
        path: Path,
        temp_root: Path,
        exporter: SourceExporter,
        redaction: RedactionCallback,
        selection_evaluator: SelectionEvidenceEvaluator,
    ) -> AbstractContextManager[SourceSnapshot]:
        del path, temp_root, exporter, redaction, selection_evaluator
        calls.append(5)
        raise TypeError("factory body failure")

    dependencies = replace(
        build_import_fakes(restrictive_tmp_path=restrictive_tmp_path),
        snapshot_factory=snapshot_factory,
    )

    result = run_import(replace(config, dry_run=True), dependencies)

    assert exit_status(result) == 2
    assert result.error is not None
    assert result.error.code == "import_failed"
    assert calls == [5]


def test_capability_failure_blocks_every_mutation(config, restrictive_tmp_path: Path) -> None:
    gateway = FakeMlflowGateway(compatible=False)
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path, gateway=gateway)

    result = run_import(config, dependencies)

    assert exit_status(result) == 2
    assert result.report["errors"] == [
        {"code": "mlflow_capability_unsupported", "message": "the fake gateway was configured incompatible"}
    ]
    assert gateway.mutations == []
    assert gateway.trace_ids() == []
    assert dependencies.lock_factory.acquisitions == []
    assert_result_document(result.report)


def test_held_lock_fails_safely_before_experiment_resolution(config, restrictive_tmp_path: Path) -> None:
    gateway = FakeMlflowGateway()
    dependencies = build_import_fakes(
        restrictive_tmp_path=restrictive_tmp_path,
        gateway=gateway,
        lock_factory=FakeLockFactory(fail=True),
    )

    result = run_import(config, dependencies)

    assert exit_status(result) == 2
    assert result.report["errors"][0]["code"] == "import_lock_held"
    assert result.report["experiment"]["status"] == "unresolved"
    assert gateway.mutations == []
    assert gateway.trace_ids() == []


def test_missing_source_fails_with_exit_two(import_dependencies, config, tmp_path: Path) -> None:
    result = run_import(replace(config, db_path=tmp_path / "missing.db"), import_dependencies)

    assert exit_status(result) == 2
    assert result.report["errors"][0]["code"] == "source_unavailable"
    assert import_dependencies.gateway.mutations == []


# ---------------------------------------------------------------------------
# Replay fingerprint checks
# ---------------------------------------------------------------------------


class _TamperedReplayTurn:
    """ReplayTurn wrapper that appends one synthetic record to the turn cursor."""

    def __init__(self, inner: ReplayTurn) -> None:
        self._inner = inner

    @property
    def session_header(self) -> Any:
        return self._inner.session_header

    @property
    def summary(self) -> Any:
        return self._inner.summary

    def iter_records(self) -> Iterator[Any]:
        yield from self._inner.iter_records()
        yield fake_record("tampered-extra-record")

    def iter_session_records(self) -> Iterator[Any]:
        return self._inner.iter_session_records()

    def iter_unassigned_records(self) -> Iterator[Any]:
        return self._inner.iter_unassigned_records()

    def load_root_input(self, max_bytes: int) -> object | None:
        return self._inner.load_root_input(max_bytes)

    def load_root_output(self, max_bytes: int) -> object | None:
        return self._inner.load_root_output(max_bytes)


def test_replay_fingerprint_mismatch_fails_the_invocation(config, restrictive_tmp_path: Path) -> None:
    def tampering_builder(snapshot: SourceSnapshot) -> Callable[[DiscoveryTurnEntry], ReplayTurn]:
        def replay(entry: DiscoveryTurnEntry) -> ReplayTurn:
            return _TamperedReplayTurn(replay_turn(entry, snapshot))  # type: ignore[return-value]

        return replay

    gateway = FakeMlflowGateway()
    dependencies = build_import_fakes(
        restrictive_tmp_path=restrictive_tmp_path,
        gateway=gateway,
        replay_builder=tampering_builder,
    )

    result = run_import(config, dependencies)

    assert exit_status(result) == 2
    assert result.report["errors"][0]["code"] == "replay_fingerprint_mismatch"
    # The experiment creation is deferred past every pre-write check, so
    # the consistency failure aborted the invocation with ZERO mutations.
    assert gateway.mutations == []
    assert gateway.trace_ids() == []
    assert dependencies.lock_factory.acquisitions


def test_replay_turn_rebuilds_the_plan_entry_turn(source_db: Path, restrictive_tmp_path: Path) -> None:
    """The public single-turn replay matches its plan entry's fingerprint."""
    connection = sqlite3.connect(source_db)
    try:
        exporter = build_live_exporter(connection, probe_schema(connection))
    finally:
        connection.close()
    with create_normalized_snapshot(source_db, restrictive_tmp_path, exporter, lambda value, paths: value) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        plan_path = restrictive_tmp_path / "replay-check-plan.jsonl"
        discover_sessions(adapter, SelectionFilters(), plan_path)
        entry = next(DiscoveryPlanReader(plan_path).iter_turns())

        replayed = replay_turn(entry, snapshot)

        assert replayed.summary.turn_key == entry.turn_key
        assert replayed.session_header.session_id == entry.session_id
        effective = build_effective_turn(replayed, IDENTITY_REDACTOR, RedactionMode.VERBATIM)
        assert source_fingerprint(effective, RedactionMode.VERBATIM) == entry.source_fingerprint


# ---------------------------------------------------------------------------
# Per-item failure isolation and exit statuses
# ---------------------------------------------------------------------------


def test_item_failure_is_isolated_and_exits_one(import_dependencies, config) -> None:
    import_dependencies.gateway.fail_next_trace_verification = 1

    result = run_import(config, import_dependencies)

    assert exit_status(result) == 1
    turns = result.report["sessions"][0]["turns"]
    assert [turn["action"] for turn in turns] == ["failed", "imported"]
    assert turns[0]["error_code"] == "trace_verification_failed"
    assert turns[0]["trace_id"] is not None
    assert not import_dependencies.gateway.has_trace(turns[0]["trace_id"])
    assert turns[1]["trace_id"] is not None
    summary = result.report["summary"]
    assert summary["turns_failed"] == 1
    assert summary["turns_imported"] == 1
    assert_counts_are_disjoint(result.report)


def test_changed_turn_replaces_only_its_own_trace(import_dependencies, config, tmp_path: Path) -> None:
    """Editing ONE turn of a multi-turn session replaces only that turn's trace.

    The acceptance criterion: changing a source turn replaces only its
    importer-owned trace. The unchanged sibling keeps its trace id and its
    artifact at its recorded location (a historical record of the session
    as of that turn's creation), because candidate verification checks
    integrity at the candidate's OWN recorded path — the deterministic
    path embeds the session-complete artifact-snapshot fingerprint and
    moves for every sibling whenever any turn changes.
    """
    first = run_import(config, import_dependencies)
    assert [turn["action"] for turn in first.report["sessions"][0]["turns"]] == ["imported", "imported"]
    uploads_after_first = _artifact_uploads(import_dependencies.gateway)
    assert len(uploads_after_first) == 2
    sibling_trace_id = first.report["sessions"][0]["turns"][0]["trace_id"]
    sibling_uploads = [upload for upload in uploads_after_first if upload[0] == sibling_trace_id]
    assert len(sibling_uploads) == 1
    sibling_path = sibling_uploads[0][1]

    changed_source = build_current_source_db(
        tmp_path,
        name="edited-turn-source.db",
        sessions=[(UNIT_SESSION_ID, DEFAULT_SESSION_CREATED_MS)],
        messages=[
            ("unit-user-1", UNIT_SESSION_ID, "user", 1, UNIT_PROMPT_JSON),
            ("unit-assistant-1", UNIT_SESSION_ID, "assistant", 2, UNIT_REPLY_JSON),
            ("unit-user-2", UNIT_SESSION_ID, "user", 3, UNIT_PROMPT_JSON),
            (
                "unit-assistant-2",
                UNIT_SESSION_ID,
                "assistant",
                4,
                '{"content": [{"type": "text", "text": "synthetic edited reply"}]}',
            ),
        ],
    )

    result = run_import(replace(config, db_path=changed_source), import_dependencies)

    assert exit_status(result) == 0
    turns = result.report["sessions"][0]["turns"]
    assert [turn["action"] for turn in turns] == ["skipped", "updated"]
    assert turns[0]["reason"] == "unchanged"
    assert turns[0]["trace_id"] == sibling_trace_id
    assert turns[1]["trace_id"] is not None and turns[1]["trace_id"] != sibling_trace_id
    assert result.report["summary"]["retired_traces"] == 1

    # The sibling was not replaced, re-uploaded, or retired: exactly one
    # new artifact (the changed turn's replacement) was uploaded.
    uploads_after_second = _artifact_uploads(import_dependencies.gateway)
    assert len(uploads_after_second) == 3
    assert [upload for upload in uploads_after_second if upload[0] == sibling_trace_id] == sibling_uploads
    stored = import_dependencies.gateway.stored_trace(sibling_trace_id)
    assert stored is not None
    assert stored.tags["opencode.artifact_path"] == sibling_path
    assert import_dependencies.gateway.has_trace(sibling_trace_id)


def test_stale_turn_is_retired_with_source_present_false(import_dependencies, config, tmp_path: Path) -> None:
    run_import(config, import_dependencies)

    trimmed_source = build_current_source_db(
        tmp_path,
        name="trimmed-source.db",
        sessions=[(UNIT_SESSION_ID, DEFAULT_SESSION_CREATED_MS)],
        messages=[
            ("unit-user-1", UNIT_SESSION_ID, "user", 1, UNIT_PROMPT_JSON),
            ("unit-assistant-1", UNIT_SESSION_ID, "assistant", 2, UNIT_REPLY_JSON),
        ],
    )

    result = run_import(replace(config, db_path=trimmed_source), import_dependencies)

    assert exit_status(result) == 0
    turns = result.report["sessions"][0]["turns"]
    # The surviving sibling is skipped-unchanged (its artifact stays at its
    # recorded location); the removed turn's owned trace is stale and retired.
    assert [turn["action"] for turn in turns] == ["skipped", "retired"]
    assert turns[0]["reason"] == "unchanged"
    assert turns[1]["reason"] == "stale_turn"
    assert turns[1]["source_present"] is False
    assert turns[1]["trace_id"] is not None
    assert not import_dependencies.gateway.has_trace(turns[1]["trace_id"])
    assert result.report["summary"]["retired_traces"] == 1
    assert_counts_are_disjoint(result.report)


def test_failed_retirement_is_reported_as_a_failed_item(import_dependencies, config, tmp_path: Path) -> None:
    """A retirement that fails is reported as a failed item and exits one.

    Task 11's ``_apply_retirements`` appends an ``ACTION_FAILED`` result
    for a retirement that could not be performed; this pins that the
    importer surfaces it (failed action, safe error code, exit 1) instead
    of misreporting it as retired.
    """
    first = run_import(config, import_dependencies)
    stale_trace_id = first.report["sessions"][0]["turns"][1]["trace_id"]
    assert stale_trace_id is not None
    import_dependencies.gateway.fail_retire_trace_ids = {stale_trace_id}

    trimmed_source = build_current_source_db(
        tmp_path,
        name="failed-retirement-source.db",
        sessions=[(UNIT_SESSION_ID, DEFAULT_SESSION_CREATED_MS)],
        messages=[
            ("unit-user-1", UNIT_SESSION_ID, "user", 1, UNIT_PROMPT_JSON),
            ("unit-assistant-1", UNIT_SESSION_ID, "assistant", 2, UNIT_REPLY_JSON),
        ],
    )

    result = run_import(replace(config, db_path=trimmed_source), import_dependencies)

    assert exit_status(result) == 1
    turns = result.report["sessions"][0]["turns"]
    assert [turn["action"] for turn in turns] == ["skipped", "failed"]
    assert turns[1]["reason"] == "stale_turn"
    assert turns[1]["error_code"] == "retirement_failed"
    assert turns[1]["source_present"] is False
    assert turns[1]["trace_id"] == stale_trace_id
    assert result.report["summary"]["turns_failed"] == 1
    assert result.report["summary"]["retired_traces"] == 0
    assert import_dependencies.gateway.has_trace(stale_trace_id)
    assert_counts_are_disjoint(result.report)


def test_tombstone_session_retires_every_owned_trace(import_dependencies, config, tmp_path: Path) -> None:
    run_import(config, import_dependencies)

    empty_source = build_current_source_db(
        tmp_path,
        name="empty-source.db",
        sessions=[(UNIT_SESSION_ID, DEFAULT_SESSION_CREATED_MS)],
    )

    result = run_import(replace(config, db_path=empty_source), import_dependencies)

    assert exit_status(result) == 0
    turns = result.report["sessions"][0]["turns"]
    assert [turn["action"] for turn in turns] == ["retired", "retired"]
    assert all(turn["reason"] == "session_tombstone" for turn in turns)
    assert all(turn["source_present"] is False for turn in turns)
    assert result.report["summary"]["retired_traces"] == 2
    assert import_dependencies.gateway.trace_ids() == []


def test_unexamined_owned_traces_are_counted(config, restrictive_tmp_path: Path) -> None:
    gateway = FakeMlflowGateway()
    gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
    experiment = gateway.resolve_experiment(config.experiment_name, create=True)
    assert experiment.experiment_id is not None
    gateway.owned_traces(
        [
            OwnedTraceCandidate(
                trace_id="tr-foreign-owned",
                session_id="ses-not-in-source",
                turn_key="turn-foreign",
                request_time_ms=None,
                experiment_id=experiment.experiment_id,
            )
        ]
    )
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path, gateway=gateway)

    result = run_import(config, dependencies)

    assert exit_status(result) == 0
    assert result.report["summary"]["unexamined_owned_traces"] == 1
    assert gateway.has_trace("tr-foreign-owned")


def test_dry_run_reports_selected_malformed_candidate_as_ownership_conflict(
    config,
    restrictive_tmp_path: Path,
) -> None:
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)
    first = run_import(config, dependencies)
    assert exit_status(first) == 0
    initial_trace_ids = dependencies.gateway.trace_ids()
    trace_id = first.report["sessions"][0]["turns"][0]["trace_id"]
    trace = dependencies.gateway.stored_trace(trace_id)
    assert trace is not None
    trace.metadata["opencode.source_fingerprint"] = None

    result = run_import(replace(config, dry_run=True), dependencies)

    assert result.report["summary"]["failed_items"] == 1
    turn = result.report["sessions"][0]["turns"][0]
    assert turn["action"] == "failed"
    assert turn["error_code"] == "ownership_conflict"
    assert turn["affected_trace_ids"] == [trace_id]
    assert turn["conflicting_trace_ids"] == [trace_id]
    assert dependencies.gateway.trace_ids() == initial_trace_ids


# ---------------------------------------------------------------------------
# Selection warnings, redaction, and cleanup
# ---------------------------------------------------------------------------


def test_broad_selection_warning_without_filters(import_dependencies, config) -> None:
    unfiltered = run_import(replace(config, dry_run=True), import_dependencies)
    warnings = unfiltered.report["summary"]["warnings"]
    assert [warning["code"] for warning in warnings] == ["broad_selection"]
    assert "dry-run" in warnings[0]["message"]

    filtered = run_import(replace(config, dry_run=True, min_user_turns=0), import_dependencies)
    assert filtered.report["summary"]["warnings"] == []


def test_redaction_counts_are_reported_without_leaking_secrets(restrictive_tmp_path: Path, tmp_path: Path) -> None:
    token = "ghp_unit0123456789abcdefg"
    prompt = json.dumps({"content": [{"type": "text", "text": f"authenticate with {token} please"}]})
    source = build_current_source_db(
        tmp_path,
        sessions=[("ses-redaction", DEFAULT_SESSION_CREATED_MS)],
        messages=[
            ("redaction-user", "ses-redaction", "user", 1, prompt),
            (
                "redaction-assistant",
                "ses-redaction",
                "assistant",
                2,
                '{"content": [{"type": "text", "text": "synthetic reply"}]}',
            ),
        ],
    )
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)
    config = ImportConfig(experiment_name="unit-redaction-experiment", db_path=source, redact=True)

    result = run_import(config, dependencies)

    assert exit_status(result) == 0
    redaction = result.report["redaction"]
    assert redaction["mode"] == "redacted"
    assert redaction["detector_counts"].get("known_token_prefix", 0) >= 1
    assert redaction["affected_field_count"] >= 1
    assert redaction["affected_record_count"] >= 1
    assert_no_sentinels(render_json(result), (token,))
    assert_no_sentinels(render_human(result), (token,))


def test_redaction_counts_one_source_row_once_across_storage_projections(
    restrictive_tmp_path: Path, tmp_path: Path
) -> None:
    source = tmp_path / "single-redacted-row.db"
    connection = sqlite3.connect(source)
    try:
        connection.executescript(LEGACY_DDL)
        connection.execute(
            "INSERT INTO session (id, time_created, time_updated) VALUES (?, ?, ?)",
            ("ses-single-redacted-row", DEFAULT_SESSION_CREATED_MS, DEFAULT_SESSION_CREATED_MS),
        )
        connection.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
            (
                "msg-single-redacted-row",
                "ses-single-redacted-row",
                DEFAULT_SESSION_CREATED_MS + 1_000,
                DEFAULT_SESSION_CREATED_MS + 1_000,
                '{"role":"user","password":"one-source-secret"}',
            ),
        )
        connection.commit()
    finally:
        connection.close()
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)
    config = ImportConfig(
        experiment_name="unit-single-redacted-row",
        db_path=source,
        redact=True,
        dry_run=True,
    )

    result = run_import(config, dependencies)

    redaction = result.report["redaction"]
    assert redaction["affected_record_count"] == 1
    # The decoded value, raw text, raw-column map, and permitted-field map
    # are four semantic storage projections. Each is traversed once; the
    # record scope deduplicates their affected-record contribution.
    assert redaction["affected_field_count"] == 4
    assert redaction["detector_counts"] == {"secret_key_value": 4}


def test_nested_redaction_counts_do_not_scale_with_json_depth(restrictive_tmp_path: Path, tmp_path: Path) -> None:
    reports: list[dict[str, object]] = []
    for index, data in enumerate(
        (
            '{"role":"user","password":"depth-one-secret"}',
            '{"role":"user","outer":{"inner":{"password":"depth-three-secret"}}}',
        )
    ):
        source = tmp_path / f"nested-redaction-{index}.db"
        connection = sqlite3.connect(source)
        try:
            connection.executescript(LEGACY_DDL)
            connection.execute(
                "INSERT INTO session (id, time_created, time_updated) VALUES (?, ?, ?)",
                (f"ses-nested-redaction-{index}", DEFAULT_SESSION_CREATED_MS, DEFAULT_SESSION_CREATED_MS),
            )
            connection.execute(
                "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
                (
                    f"msg-nested-redaction-{index}",
                    f"ses-nested-redaction-{index}",
                    DEFAULT_SESSION_CREATED_MS + 1_000,
                    DEFAULT_SESSION_CREATED_MS + 1_000,
                    data,
                ),
            )
            connection.commit()
        finally:
            connection.close()

        dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)
        result = run_import(
            ImportConfig(f"unit-nested-redaction-{index}", source, redact=True, dry_run=True), dependencies
        )
        reports.append(result.report["redaction"])

    assert reports == [
        {
            "mode": "redacted",
            "detector_counts": {"secret_key_value": 4},
            "affected_field_count": 4,
            "affected_record_count": 1,
        },
        {
            "mode": "redacted",
            "detector_counts": {"secret_key_value": 4},
            "affected_field_count": 4,
            "affected_record_count": 1,
        },
    ]


def test_redacted_import_selects_with_pre_redaction_directory_and_model_evidence(
    restrictive_tmp_path: Path, tmp_path: Path
) -> None:
    directory_secret = "directory-secret"
    model_secret = "sk-abcdefghijklmnop"
    requested_directory = tmp_path / f"password={directory_secret}"
    source_builder = SyntheticSourceBuilder(tmp_path)
    source_builder.add_session(
        "ses-pre-redaction-selection",
        created_ms=DEFAULT_SESSION_CREATED_MS,
        directory=str(requested_directory / "project"),
        message_model=f"provider/{model_secret}",
    )
    source = source_builder.materialize()
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)
    config = ImportConfig(
        experiment_name="unit-pre-redaction-selection",
        db_path=source,
        directory=requested_directory,
        model=f"provider/{model_secret}",
        redact=True,
    )

    result = run_import(config, dependencies)

    assert exit_status(result) == 0
    assert result.report["summary"]["selected_sessions"] == 1
    assert result.report["summary"]["selected_turns"] == 1
    assert result.report["sessions"][0]["selected"] is True
    serialized = render_json(result)
    assert directory_secret not in serialized
    assert model_secret not in serialized
    turn = result.report["sessions"][0]["turns"][0]
    trace = dependencies.gateway.stored_trace(turn["trace_id"])
    artifact = dependencies.gateway.stored_artifact(trace.trace_id, trace.tags["opencode.artifact_path"])
    for serialized_payload in (repr(trace), artifact.decode("utf-8"), serialized):
        assert directory_secret not in serialized_payload
        assert model_secret not in serialized_payload
        assert "directory_filter_match" not in serialized_payload
        assert "model_filter_match" not in serialized_payload


def test_pre_redaction_selection_evidence_rejects_independent_directory_and_model_decoys(
    restrictive_tmp_path: Path, tmp_path: Path
) -> None:
    requested_directory = tmp_path / "repo" / "password=directory-secret" / "project"
    requested_model = "provider/sk-abcdefghijklmnop"
    builder = SyntheticSourceBuilder(tmp_path)
    builder.add_session(
        "ses-selection-match",
        created_ms=DEFAULT_SESSION_CREATED_MS,
        directory=str(requested_directory / "child"),
        message_model=requested_model,
    )
    builder.add_session(
        "ses-selection-directory-decoy",
        created_ms=DEFAULT_SESSION_CREATED_MS + 1,
        directory=str(tmp_path / "repo" / "other"),
        message_model=requested_model,
    )
    builder.add_session(
        "ses-selection-model-decoy",
        created_ms=DEFAULT_SESSION_CREATED_MS + 2,
        directory=str(requested_directory / "sibling"),
        message_model="provider/other-model",
    )
    source = builder.materialize()
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)

    result = run_import(
        ImportConfig(
            experiment_name="unit-selection-decoys",
            db_path=source,
            directory=requested_directory,
            model=requested_model,
            redact=True,
            dry_run=True,
        ),
        dependencies,
    )

    assert exit_status(result) == 0
    scans = {item["session_id"]: item for item in result.report["sessions"]}
    assert scans["ses-selection-match"]["selected"] is True
    assert scans["ses-selection-directory-decoy"]["selected"] is False
    assert scans["ses-selection-model-decoy"]["selected"] is False
    assert result.report["summary"]["selected_sessions"] == 1
    serialized = render_json(result)
    assert "directory_filter_match" not in serialized
    assert "model_filter_match" not in serialized


def test_directory_matching_is_lexical_while_artifact_keeps_original_provenance(
    restrictive_tmp_path: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    raw_directory = str(tmp_path) + "/repo/project/./nested/../child//"
    source_builder = SyntheticSourceBuilder(tmp_path)
    source_builder.add_session(
        "ses-directory-provenance",
        created_ms=DEFAULT_SESSION_CREATED_MS,
        directory=raw_directory,
    )
    source = source_builder.materialize()
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)
    config = ImportConfig(
        experiment_name="unit-directory-provenance",
        db_path=source,
        directory=tmp_path / "repo" / "project",
    )

    result = run_import(config, dependencies)

    assert exit_status(result) == 0
    assert result.report["summary"]["selected_sessions"] == 1
    turn = result.report["sessions"][0]["turns"][0]
    trace = dependencies.gateway.stored_trace(turn["trace_id"])
    artifact = json.loads(dependencies.gateway.stored_artifact(trace.trace_id, trace.tags["opencode.artifact_path"]))
    assert artifact["session"]["directory"] == raw_directory
    assert artifact["session"]["raw_columns"]["directory"] == raw_directory


def test_verbatim_mode_reports_zero_redaction_counts(import_dependencies, config) -> None:
    result = run_import(config, import_dependencies)

    assert result.report["redaction"]["mode"] == "verbatim"
    assert result.report["redaction"]["detector_counts"] == {}
    assert result.report["redaction"]["affected_field_count"] == 0
    assert result.report["redaction"]["affected_record_count"] == 0


def test_run_temporaries_are_cleaned_on_success_and_failure(config, restrictive_tmp_path: Path) -> None:
    created: list[Path] = []

    def temp_directory_factory(prefix: str) -> Path:
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=restrictive_tmp_path))
        created.append(path)
        return path

    dependencies = replace(
        build_import_fakes(restrictive_tmp_path=restrictive_tmp_path),
        temp_directory_factory=temp_directory_factory,
    )
    run_import(config, dependencies)
    assert created
    assert not any(path.exists() for path in created)

    created.clear()
    failing = replace(
        build_import_fakes(restrictive_tmp_path=restrictive_tmp_path, gateway=FakeMlflowGateway(compatible=False)),
        temp_directory_factory=temp_directory_factory,
    )
    result = run_import(config, failing)
    assert exit_status(result) == 2
    assert created
    assert not any(path.exists() for path in created)

    for failure in (OSError("secret path /tmp/private"), RuntimeError("secret token value")):

        def failing_temp_directory(_prefix: str, error: Exception = failure) -> Path:
            raise error

        dependencies = replace(
            build_import_fakes(restrictive_tmp_path=restrictive_tmp_path),
            temp_directory_factory=failing_temp_directory,
        )
        result = run_import(config, dependencies)
        assert exit_status(result) == 2
        assert result.error is not None
        assert result.error.code == "temporary_directory_unavailable"
        assert result.report["errors"] == [
            {
                "code": "temporary_directory_unavailable",
                "message": "the temporary run directory could not be prepared",
            }
        ]
        assert str(restrictive_tmp_path) not in render_json(result)
        assert "secret token value" not in render_json(result)


@pytest.mark.parametrize("compatible", [True, False], ids=["normal", "preflight"])
def test_run_root_cleanup_failure_is_reported_without_paths(
    config,
    restrictive_tmp_path: Path,
    compatible: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rmtree_kwargs: list[dict[str, object]] = []

    def failing_rmtree(_path: str | Path, *args: object, **kwargs: object) -> None:
        rmtree_kwargs.append(kwargs)
        raise OSError("secret temporary path")

    monkeypatch.setattr(importer_module.shutil, "rmtree", failing_rmtree)
    dependencies = build_import_fakes(
        restrictive_tmp_path=restrictive_tmp_path,
        gateway=FakeMlflowGateway(compatible=compatible),
    )

    result = run_import(config, dependencies)

    assert result.failed_items == 1
    assert result.report["summary"]["failed_items"] == 1
    assert result.report["summary"]["turns_failed"] == 0
    warning_codes = [warning["code"] for warning in result.report["summary"]["warnings"]]
    assert warning_codes.count("temporary_run_cleanup_failed") == 1
    assert rmtree_kwargs == [{}]
    assert "secret temporary path" not in render_json(result)
    assert str(restrictive_tmp_path) not in render_json(result)
    if compatible:
        assert result.error is None
        assert exit_status(result) == 1
    else:
        assert result.error is not None
        assert result.error.code == "mlflow_capability_unsupported"
        assert exit_status(result) == 2


def test_snapshot_cleanup_warning_reaches_report_human_and_json_with_item_failure(
    config,
    restrictive_tmp_path: Path,
) -> None:
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)
    original_factory = dependencies.snapshot_factory

    @contextmanager
    def injected_cleanup_failure(*args: object, **kwargs: object) -> Iterator[SourceSnapshot]:
        snapshot: SourceSnapshot | None = None
        with original_factory(*args, **kwargs) as opened:  # type: ignore[arg-type]
            snapshot = opened
            yield opened
        assert snapshot is not None
        snapshot.cleanup_warnings.extend(
            [
                "temporary snapshot cleanup failed",
                "temporary snapshot cleanup failed",
            ]
        )

    dependencies = replace(dependencies, snapshot_factory=injected_cleanup_failure)

    result = run_import(config, dependencies)

    assert exit_status(result) == 1
    assert result.failed_items == 1
    assert result.report["summary"]["failed_items"] == result.failed_items
    warning = next(
        warning
        for warning in result.report["summary"]["warnings"]
        if warning["code"] == "temporary_snapshot_cleanup_failed"
    )
    assert warning["code"] == "temporary_snapshot_cleanup_failed"
    assert "sensitive temporary data may remain" in warning["message"]
    assert "temporary snapshot cleanup failed" in render_json(result)
    assert "temporary_snapshot_cleanup_failed" in render_human(result)
    assert str(restrictive_tmp_path) not in render_json(result)
    assert str(restrictive_tmp_path) not in render_human(result)


def test_snapshot_cleanup_failure_plus_capability_error_counts_cleanup_once(
    config,
    restrictive_tmp_path: Path,
) -> None:
    dependencies = build_import_fakes(
        restrictive_tmp_path=restrictive_tmp_path,
        gateway=FakeMlflowGateway(compatible=False),
    )
    original_factory = dependencies.snapshot_factory

    @contextmanager
    def injected_cleanup_failure(*args: object, **kwargs: object) -> Iterator[SourceSnapshot]:
        snapshot: SourceSnapshot | None = None
        try:
            with original_factory(*args, **kwargs) as opened:  # type: ignore[arg-type]
                snapshot = opened
                yield opened
        finally:
            assert snapshot is not None
            snapshot.cleanup_warnings.append("temporary snapshot cleanup failed")

    dependencies = replace(dependencies, snapshot_factory=injected_cleanup_failure)

    result = run_import(config, dependencies)

    assert result.error is not None
    assert result.error.code == "mlflow_capability_unsupported"
    assert exit_status(result) == 2
    assert result.failed_items == 1
    assert result.report["summary"]["failed_items"] == 1
    assert [warning["code"] for warning in result.report["summary"]["warnings"]].count(
        "temporary_snapshot_cleanup_failed"
    ) == 1


def test_population_error_with_snapshot_cleanup_failure_reaches_report(
    config,
    restrictive_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)
    original_unlink = Path.unlink
    original_rmdir = Path.rmdir

    def failing_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name == "snapshot.db":
            raise OSError("synthetic unlink failure")
        original_unlink(self, *args, **kwargs)

    def failing_rmdir(self: Path) -> None:
        if self.name.startswith("opencode-mlflow-snapshot-"):
            raise OSError("synthetic rmdir failure")
        original_rmdir(self)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    monkeypatch.setattr(Path, "rmdir", failing_rmdir)

    original_exporter_factory = dependencies.exporter_factory

    def failing_exporter_factory(connection: sqlite3.Connection, description: Any) -> Any:
        exporter = original_exporter_factory(connection, description)

        def fail_export(*_args: object, **_kwargs: object) -> object:
            raise sqlite3.OperationalError("synthetic population failure")

        exporter.export_to_snapshot = fail_export  # type: ignore[method-assign]
        return exporter

    dependencies = replace(dependencies, exporter_factory=failing_exporter_factory)

    result = run_import(config, dependencies)

    assert result.error is not None
    assert result.error.code == "source_snapshot_failed"
    assert result.failed_items == 1
    assert result.report["summary"]["failed_items"] == 1
    warning_codes = [warning["code"] for warning in result.report["summary"]["warnings"]]
    assert warning_codes.count("temporary_snapshot_cleanup_failed") == 1

    class RaisingInspectExporter(ScriptedExporter):
        def inspect(self):  # type: ignore[no-untyped-def]
            raise SourceCompatibilityError("unsupported_schema", "the synthetic schema is unsupported")

    compatibility = replace(
        build_import_fakes(restrictive_tmp_path=restrictive_tmp_path),
        exporter_factory=lambda _connection, _description: RaisingInspectExporter(),
    )
    compatibility_result = run_import(config, compatibility)
    assert compatibility_result.error is not None
    assert compatibility_result.error.code == "unsupported_schema"
    assert exit_status(compatibility_result) == 2
    assert compatibility_result.failed_items == 1
    assert compatibility_result.report["summary"]["failed_items"] == 1
    compatibility_warnings = [warning["code"] for warning in compatibility_result.report["summary"]["warnings"]]
    assert compatibility_warnings.count("temporary_snapshot_cleanup_failed") == 1
    assert "synthetic rmdir failure" not in render_json(compatibility_result)


def test_staging_setup_failure_precedes_deferred_experiment_creation(
    config,
    restrictive_tmp_path: Path,
) -> None:
    gateway = FakeMlflowGateway()
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path, gateway=gateway)
    created_roots: list[Path] = []

    def temp_directory_factory(prefix: str) -> Path:
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=restrictive_tmp_path))
        (path / "artifact-staging").write_text("not a directory")
        created_roots.append(path)
        return path

    dependencies = replace(dependencies, temp_directory_factory=temp_directory_factory)

    result = run_import(config, dependencies)

    assert exit_status(result) == 2
    assert result.error is not None
    assert result.error.code == "staging_directory_unavailable"
    assert gateway.mutations == []
    assert gateway.stored_experiment(config.experiment_name) is None
    assert gateway.trace_ids() == []
    assert created_roots


def test_unexpected_failure_is_sanitized(import_dependencies, config) -> None:
    def exploding_gateway_factory(dry_run: bool) -> Any:
        raise RuntimeError("boom with /absolute/path and secret-content")

    dependencies = replace(import_dependencies, gateway_factory=exploding_gateway_factory)

    result = run_import(config, dependencies)

    assert exit_status(result) == 2
    assert result.report["errors"][0]["code"] == "import_failed"
    assert_no_sentinels(render_json(result), ("/absolute/path", "secret-content"))
