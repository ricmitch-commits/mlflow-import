"""Independent expectations for the v1 result accounting contract."""

import pytest

from mlflow_import.contracts import RedactionMode
from mlflow_import.reporting import ReportBuilder, TurnResult

pytestmark = pytest.mark.unit


def test_unchanged_is_not_counted_twice_as_skipped():
    builder = ReportBuilder(dry_run=False, experiment_name="synthetic", redaction_mode=RedactionMode.VERBATIM)
    builder.add_session_scan("synthetic", selected=True, logical_turn_count=1)
    builder.add_turn(TurnResult("synthetic", "stable-turn", "skipped", reason="unchanged"))
    summary = builder.build()["summary"]
    assert summary["unchanged_turns"] == 1
    assert summary["skipped_turns"] == 0


def test_empty_session_has_explicit_no_turn_reason():
    builder = ReportBuilder(dry_run=False, experiment_name="synthetic", redaction_mode=RedactionMode.VERBATIM)
    builder.add_session_scan("synthetic", selected=True, logical_turn_count=0)
    assert builder.build()["sessions"][0]["reason"] == "no_importable_turns"
