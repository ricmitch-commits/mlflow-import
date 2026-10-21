"""Wave 3 report/exit regressions."""

from __future__ import annotations

from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.importer import ImportResult, _record_normal_results, exit_status
from opencode_mlflow.mlflow.reconciliation import (
    ACTION_RETIRED,
    OwnedTraceCandidate,
    ReconciliationAction,
    ReconciliationPlan,
)
from opencode_mlflow.reporting import ReportBuilder
from opencode_mlflow.source.discovery import DiscoveryTurnEntry


def _entry() -> DiscoveryTurnEntry:
    return DiscoveryTurnEntry(
        session_id="ses-wave3",
        turn_index=3,
        turn_key="turn-wave3",
        adapter_generation="current",
        planned_action="import",
        start_time_ns=1,
        end_time_ns=2,
        record_count=1,
        user_record_count=1,
        assistant_record_count=0,
        terminal_outcome=None,
        model_identities=frozenset(),
        warnings=("stale_turn_index_warning",),
        session_context_fingerprint="a" * 64,
        source_fingerprint="b" * 64,
        render_fingerprint="c" * 64,
        artifact_schema_version="artifact-v1",
        mapping_version="mapping-v1",
        importer_version="importer-v1",
        redaction_mode=RedactionMode.VERBATIM.value,
    )


def test_missing_reconciliation_result_is_failed_and_exit_one() -> None:
    builder = ReportBuilder(dry_run=False, experiment_name="wave3", redaction_mode=RedactionMode.VERBATIM)
    entry = _entry()
    builder.add_session_scan(entry.session_id, selected=True, logical_turn_count=1)

    failed_items = _record_normal_results(builder, ReconciliationPlan(), [entry])
    report = builder.build()

    assert failed_items == 1
    assert exit_status(ImportResult(dry_run=False, report=report, failed_items=failed_items)) == 1
    turn = report["sessions"][0]["turns"][0]
    assert turn["action"] == "failed"
    assert turn["error_code"] == "reconciliation_result_missing"


def test_missing_stale_retirement_result_fails_closed_and_preserves_identity() -> None:
    builder = ReportBuilder(dry_run=False, experiment_name="wave3", redaction_mode=RedactionMode.VERBATIM)
    candidate = OwnedTraceCandidate(
        trace_id="trace-stale-wave3",
        session_id="ses-wave3",
        turn_key="turn-gone",
        request_time_ms=1,
        turn_index=7,
        warnings=("unknown_source_field",),
    )
    planned = ReconciliationAction(action=ACTION_RETIRED, trace=candidate, reason="stale_turn")

    failed_items = _record_normal_results(builder, ReconciliationPlan(retirements=[planned]), [])
    report = builder.build()
    turn = report["sessions"][0]["turns"][0]

    assert failed_items == 1
    assert turn["action"] == "failed"
    assert turn["error_code"] == "reconciliation_result_missing"
    assert turn["trace_id"] == candidate.trace_id
    assert turn["turn_index"] == candidate.turn_index
    assert turn["source_present"] is False
    assert turn["warnings"][0]["code"] == "unknown_source_field"
    assert turn["affected_trace_ids"] == [candidate.trace_id]
    assert turn["retired_trace_ids"] == []
