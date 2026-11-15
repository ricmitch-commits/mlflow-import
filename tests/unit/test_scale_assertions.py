"""The reference benchmark must reject successful but incomplete imports."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.support.scale_assertions import assert_imported_workload

pytestmark = pytest.mark.unit


def _result():
    def turn(session, index):
        return dict(
            session_id=session,
            turn_key=str(index),
            trace_id=f"trace-{session}-{index}",
            source_present=True,
            action="imported",
            error_code=None,
            reason=None,
        )

    return SimpleNamespace(
        report={
            "errors": [],
            "source": {"sessions_scanned": 2, "sessions_selected": 2},
            "summary": dict(
                selected_sessions=2,
                selected_turns=3,
                imported_turns=3,
                updated_turns=0,
                skipped_turns=0,
                unchanged_turns=0,
                failed_turns=0,
                failed_sessions=0,
                failed_items=0,
                failed_retirements=0,
            ),
            "sessions": [
                dict(
                    session_id="one",
                    selected=True,
                    logical_turn_count=2,
                    error=None,
                    turns=[turn("one", 1), turn("one", 2)],
                ),
                dict(session_id="two", selected=True, logical_turn_count=1, error=None, turns=[turn("two", 1)]),
            ],
        }
    )


def test_complete_imported_workload_matches_actual_trace_ids():
    assert_imported_workload(_result(), 2, 3, {"trace-one-1", "trace-one-2", "trace-two-1"})


@pytest.mark.parametrize("omission", ["session", "turn", "trace", "accounting"])
def test_successful_exit_cannot_hide_incomplete_workload(omission):
    result = deepcopy(_result())
    ids = {"trace-one-1", "trace-one-2", "trace-two-1"}
    if omission == "session":
        result.report["sessions"].pop()
    elif omission == "turn":
        result.report["sessions"][0]["turns"].pop()
    elif omission == "trace":
        ids.remove("trace-two-1")
    else:
        result.report["summary"]["imported_turns"] = 2
    with pytest.raises(AssertionError):
        assert_imported_workload(result, 2, 3, ids)
