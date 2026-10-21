"""Unit tests for the versioned result document and its renderers.

The report layer is asserted two ways: directly through
:class:`opencode_mlflow.reporting.ReportBuilder` (shape, count units,
safe error entries) and end-to-end through ``run_import`` with the fake
dependencies (schema version, action enum, dry-run planned actions,
per-session/per-turn identities, sanitization sentinels, and the
human-rendered broad-selection warning).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from opencode_mlflow.config import ImportConfig
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.errors import ImporterError
from opencode_mlflow.importer import ImportResult, exit_status, run_import
from opencode_mlflow.reporting import (
    RESULT_SCHEMA,
    ReportBuilder,
    TurnResult,
    render_human,
    render_json,
)
from tests.support.fake_gateway import FakeMlflowGateway
from tests.support.import_fakes import (
    DEFAULT_SESSION_CREATED_MS,
    UNIT_PROMPT_JSON,
    UNIT_REPLY_JSON,
    UNIT_SESSION_ID,
    build_current_source_db,
    build_import_fakes,
)
from tests.support.report_assertions import (
    REQUIRED_TOP_LEVEL_KEYS,
    assert_counts_are_disjoint,
    assert_no_sentinels,
    assert_result_document,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Schema version and required sections
# ---------------------------------------------------------------------------


def test_result_schema_is_the_versioned_result_name() -> None:
    assert RESULT_SCHEMA == "opencode-mlflow.result.v1"


def test_render_json_carries_the_schema_version_and_required_sections(import_dependencies, config) -> None:
    result = run_import(config, import_dependencies)

    document = json.loads(render_json(result))

    assert document["schema_version"] == RESULT_SCHEMA
    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in document
    assert document == json.loads(json.dumps(result.report))


def test_partial_failure_reports_still_carry_every_section(restrictive_tmp_path: Path, config) -> None:
    from tests.support.fake_gateway import FakeMlflowGateway

    dependencies = build_import_fakes(
        restrictive_tmp_path=restrictive_tmp_path, gateway=FakeMlflowGateway(compatible=False)
    )

    result = run_import(config, dependencies)

    assert exit_status(result) == 2
    assert_result_document(result.report)
    # Discovery ran before the preflight, so the scanned session is
    # reported — with no turns, because nothing was written.
    assert len(result.report["sessions"]) == 1
    assert result.report["sessions"][0]["turns"] == []
    assert result.report["errors"] == [
        {"code": "mlflow_capability_unsupported", "message": "the fake gateway was configured incompatible"}
    ]


# ---------------------------------------------------------------------------
# Action enum, dry-run planned actions, identities, and trace ids
# ---------------------------------------------------------------------------


def test_actions_stay_inside_the_normative_enum_across_outcomes(
    import_dependencies, config, restrictive_tmp_path: Path, tmp_path: Path
) -> None:
    dry = run_import(replace(config, dry_run=True), import_dependencies)
    assert_result_document(dry.report)

    normal = run_import(config, import_dependencies)
    assert_result_document(normal.report)

    trimmed = build_current_source_db(
        tmp_path,
        name="report-trimmed.db",
        sessions=[(UNIT_SESSION_ID, DEFAULT_SESSION_CREATED_MS)],
        messages=[
            ("unit-user-1", UNIT_SESSION_ID, "user", 1, UNIT_PROMPT_JSON),
            ("unit-assistant-1", UNIT_SESSION_ID, "assistant", 2, UNIT_REPLY_JSON),
        ],
    )
    retired = run_import(replace(config, db_path=trimmed), import_dependencies)
    assert_result_document(retired.report)

    observed = {turn["action"] for turn in retired.report["sessions"][0]["turns"]}
    assert observed == {"skipped", "retired"}

    # A fresh gateway with an injected trace-verification failure covers
    # the "failed" action (an already-imported source takes the candidate
    # verification path, which never calls verify_trace).
    failing_gateway = FakeMlflowGateway()
    failing_gateway.fail_next_trace_verification = 1
    failing = run_import(config, build_import_fakes(restrictive_tmp_path=restrictive_tmp_path, gateway=failing_gateway))
    assert_result_document(failing.report)
    assert {turn["action"] for turn in failing.report["sessions"][0]["turns"]} == {"failed", "imported"}


def test_dry_run_uses_skipped_with_a_same_enum_planned_action(import_dependencies, config) -> None:
    result = run_import(replace(config, dry_run=True), import_dependencies)

    turn = result.report["sessions"][0]["turns"][0]
    assert turn["action"] == "skipped"
    assert turn["reason"] == "dry_run"
    assert turn["planned_action"] == "imported"
    assert turn["planned_action"] in {"imported", "updated", "skipped", "failed", "retired"}


def test_sessions_and_turns_carry_identities_and_trace_ids(import_dependencies, config) -> None:
    result = run_import(config, import_dependencies)

    session = result.report["sessions"][0]
    assert session["session_id"] == UNIT_SESSION_ID
    assert session["selected"] is True
    turns = session["turns"]
    assert [turn["turn_index"] for turn in turns] == [1, 2]
    assert len({turn["turn_key"] for turn in turns}) == 2
    assert all(turn["turn_key"].startswith("turn-") for turn in turns)
    assert all(turn["trace_id"] for turn in turns)


def test_unselected_sessions_are_reported_as_skipped(restrictive_tmp_path: Path, tmp_path: Path) -> None:
    source = build_current_source_db(
        tmp_path,
        sessions=[(UNIT_SESSION_ID, DEFAULT_SESSION_CREATED_MS), ("ses-other", DEFAULT_SESSION_CREATED_MS + 1)],
        messages=[
            ("unit-user-1", UNIT_SESSION_ID, "user", 1, UNIT_PROMPT_JSON),
            ("unit-assistant-1", UNIT_SESSION_ID, "assistant", 2, UNIT_REPLY_JSON),
            ("unit-user-2", UNIT_SESSION_ID, "user", 3, UNIT_PROMPT_JSON),
            ("unit-assistant-2", UNIT_SESSION_ID, "assistant", 4, UNIT_REPLY_JSON),
            ("other-user-1", "ses-other", "user", 1, UNIT_PROMPT_JSON),
            ("other-assistant-1", "ses-other", "assistant", 2, UNIT_REPLY_JSON),
        ],
    )
    config = ImportConfig(experiment_name="unit-report-selection", db_path=source, min_user_turns=1)
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)

    result = run_import(config, dependencies)

    assert exit_status(result) == 0
    sessions = {session["session_id"]: session for session in result.report["sessions"]}
    assert set(sessions) == {UNIT_SESSION_ID, "ses-other"}
    assert sessions[UNIT_SESSION_ID]["selected"] is True
    unselected = sessions["ses-other"]
    assert unselected["selected"] is False
    assert unselected["action"] == "skipped"
    assert unselected["reason"] == "not_selected"
    assert unselected["turns"] == []
    assert result.report["summary"]["turns_imported"] == 2


def test_safe_configuration_experiment_and_source_sections(import_dependencies, config) -> None:
    result = run_import(config, import_dependencies)

    report = result.report
    assert report["dry_run"] is False
    assert report["experiment"]["name"] == config.experiment_name
    assert report["experiment"]["id"]
    assert report["experiment"]["status"] == "created"
    assert report["source"]["schema_generation"] == "current"
    assert len(report["source"]["fingerprint"]) == 64
    assert report["source"]["sessions_scanned"] == 1
    assert report["source"]["sessions_selected"] == 1
    assert report["redaction"]["mode"] == "verbatim"


# ---------------------------------------------------------------------------
# Count units (ReportBuilder level)
# ---------------------------------------------------------------------------


def _builder() -> ReportBuilder:
    return ReportBuilder(dry_run=False, experiment_name="unit-report-experiment", redaction_mode=RedactionMode.VERBATIM)


def test_builder_counts_current_turns_disjointly_and_retirement_separately() -> None:
    builder = _builder()
    builder.add_session_scan(UNIT_SESSION_ID, selected=True)
    builder.add_turn(
        TurnResult(session_id=UNIT_SESSION_ID, turn_key="turn-a", turn_index=1, action="imported", trace_id="tr-1")
    )
    builder.add_turn(
        TurnResult(session_id=UNIT_SESSION_ID, turn_key="turn-b", turn_index=2, action="updated", trace_id="tr-2")
    )
    builder.add_turn(
        TurnResult(session_id=UNIT_SESSION_ID, turn_key="turn-c", turn_index=3, action="skipped", trace_id="tr-3")
    )
    builder.add_turn(
        TurnResult(
            session_id=UNIT_SESSION_ID,
            turn_key="turn-stale",
            turn_index=None,
            action="retired",
            reason="stale_turn",
            trace_id="tr-4",
            source_present=False,
        )
    )
    builder.set_retired_traces(2)
    builder.set_unexamined_owned_traces(5)

    report = builder.build()

    summary = report["summary"]
    assert summary["turns_imported"] == 1
    assert summary["turns_updated"] == 1
    assert summary["turns_skipped"] == 1
    assert summary["turns_failed"] == 0
    assert summary["retired_traces"] == 2
    assert summary["unexamined_owned_traces"] == 5
    assert_counts_are_disjoint(report)
    stale = report["sessions"][0]["turns"][-1]
    assert stale["source_present"] is False
    assert stale["reason"] == "stale_turn"


def test_builder_reports_all_affected_and_retired_trace_ids() -> None:
    builder = _builder()
    builder.add_session_scan(UNIT_SESSION_ID, selected=True)
    builder.add_turn(
        TurnResult(
            session_id=UNIT_SESSION_ID,
            turn_key="turn-conflict",
            turn_index=1,
            action="failed",
            trace_id="tr-new",
            error_code="duplicate_owned_identity",
            affected_trace_ids=("tr-old", "tr-new", "tr-duplicate"),
            retired_trace_ids=("tr-old",),
            conflicting_trace_ids=("tr-new", "tr-duplicate"),
        )
    )

    turn = builder.build()["sessions"][0]["turns"][0]

    assert turn["trace_id"] == "tr-new"
    assert turn["affected_trace_ids"] == ["tr-old", "tr-new", "tr-duplicate"]
    assert turn["retired_trace_ids"] == ["tr-old"]
    assert turn["conflicting_trace_ids"] == ["tr-new", "tr-duplicate"]


def test_builder_derives_the_session_action_from_its_turns() -> None:
    builder = _builder()
    builder.add_session_scan("ses-clean", selected=True)
    builder.add_session_scan("ses-failing", selected=True)
    builder.add_session_scan("ses-retired", selected=True)
    builder.add_session_scan("ses-quiet", selected=True)
    builder.add_session_scan("ses-out", selected=False)
    builder.add_turn(TurnResult(session_id="ses-clean", turn_key="turn-a", turn_index=1, action="imported"))
    builder.add_turn(TurnResult(session_id="ses-failing", turn_key="turn-b", turn_index=1, action="failed"))
    builder.add_turn(
        TurnResult(session_id="ses-retired", turn_key="turn-c", turn_index=1, action="retired", source_present=False)
    )

    report = builder.build()

    actions = {session["session_id"]: session["action"] for session in report["sessions"]}
    assert actions == {
        "ses-clean": "imported",
        "ses-failing": "failed",
        "ses-retired": "retired",
        "ses-quiet": "skipped",
        "ses-out": "skipped",
    }


def test_builder_records_safe_error_codes_and_messages() -> None:
    builder = _builder()
    builder.add_error("some_safe_code", "a safe message with no content")

    report = builder.build()

    assert report["errors"] == [{"code": "some_safe_code", "message": "a safe message with no content"}]


def test_builder_dry_run_reports_planned_counts() -> None:
    builder = ReportBuilder(
        dry_run=True, experiment_name="unit-report-experiment", redaction_mode=RedactionMode.VERBATIM
    )
    builder.add_session_scan(UNIT_SESSION_ID, selected=True)
    builder.add_turn(
        TurnResult(
            session_id=UNIT_SESSION_ID,
            turn_key="turn-a",
            turn_index=1,
            action="skipped",
            reason="dry_run",
            planned_action="imported",
        )
    )
    builder.add_turn(
        TurnResult(
            session_id=UNIT_SESSION_ID,
            turn_key="turn-stale",
            turn_index=None,
            action="skipped",
            reason="dry_run",
            planned_action="retired",
            source_present=False,
        )
    )
    builder.set_retired_traces(1)

    report = builder.build()

    assert report["dry_run"] is True
    assert report["summary"]["planned"] == {"imported": 1, "updated": 0, "retired": 1}
    assert report["summary"]["turns_skipped"] == 1
    assert report["summary"]["retired_traces"] == 0


# ---------------------------------------------------------------------------
# Human rendering
# ---------------------------------------------------------------------------


def test_human_output_is_concise_progress_oriented_and_warns_on_broad_selection(import_dependencies, config) -> None:
    result = run_import(replace(config, dry_run=True), import_dependencies)

    text = render_human(result)

    assert "opencode-mlflow import dry-run complete" in text
    assert "experiment:" in text
    assert "turns:" in text
    assert "broad_selection" in text
    assert "dry-run" in text
    assert len(text.splitlines()) <= 12


def test_human_output_reports_failures_and_errors(import_dependencies, config) -> None:
    import_dependencies.gateway.fail_next_trace_verification = 1
    result = run_import(config, import_dependencies)

    text = render_human(result)

    assert exit_status(result) == 1
    assert "failures" in text
    assert "1 failed" in text


def test_human_output_reports_dry_run_errors_as_failed() -> None:
    builder = ReportBuilder(dry_run=True, experiment_name="dry-run-error", redaction_mode=RedactionMode.VERBATIM)
    builder.add_error("prewrite_failed", "the pre-write checks failed")
    result = ImportResult(dry_run=True, report=builder.build())

    text = render_human(result)

    assert text.splitlines()[0] == "opencode-mlflow import failed"


def test_human_output_reports_result_error_as_failed_even_without_report_event() -> None:
    builder = ReportBuilder(dry_run=True, experiment_name="dry-run-error", redaction_mode=RedactionMode.VERBATIM)
    result = ImportResult(
        dry_run=True,
        report=builder.build(),
        error=ImporterError("prewrite_failed", "the pre-write checks failed"),
    )

    text = render_human(result)

    assert text.splitlines()[0] == "opencode-mlflow import failed"


# ---------------------------------------------------------------------------
# Sanitization sentinels (end-to-end)
# ---------------------------------------------------------------------------


def test_report_output_never_contains_source_content_or_secrets(restrictive_tmp_path: Path, tmp_path: Path) -> None:
    prompt_sentinel = "SENTINEL-PROMPT-CONTENT-9f1a"
    reply_sentinel = "SENTINEL-REPLY-CONTENT-2b7c"
    token = "ghp_report0123456789abcde"
    source = build_current_source_db(
        tmp_path,
        sessions=[("ses-sanitize", DEFAULT_SESSION_CREATED_MS)],
        messages=[
            (
                "sanitize-user",
                "ses-sanitize",
                "user",
                1,
                json.dumps({"content": [{"type": "text", "text": f"{prompt_sentinel} {token}"}]}),
            ),
            (
                "sanitize-assistant",
                "ses-sanitize",
                "assistant",
                2,
                json.dumps({"content": [{"type": "text", "text": reply_sentinel}]}),
            ),
        ],
    )
    config = ImportConfig(
        experiment_name="unit-report-sanitize",
        db_path=source,
        tracking_uri="https://user:report-secret@tracking.invalid",
    )
    dependencies = build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)

    result = run_import(config, dependencies)

    assert exit_status(result) == 0
    sentinels = (prompt_sentinel, reply_sentinel, token, str(source), "report-secret", "tracking.invalid")
    assert_no_sentinels(render_json(result), sentinels)
    assert_no_sentinels(render_human(result), sentinels)
    assert_result_document(result.report)
