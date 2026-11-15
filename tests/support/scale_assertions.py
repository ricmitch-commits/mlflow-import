"""Independent imported-workload checks, beyond synthetic input generation counts."""

from collections import Counter


def assert_imported_workload(result, expected_sessions, expected_turns, actual_trace_ids):
    report, summary = result.report, result.report["summary"]
    assert report["errors"] == []
    assert report["source"]["sessions_scanned"] == expected_sessions
    assert report["source"]["sessions_selected"] == expected_sessions
    assert summary["selected_sessions"] == expected_sessions
    assert summary["selected_turns"] == expected_turns
    assert summary["failed_turns"] == summary["failed_sessions"] == summary["failed_items"] == 0
    assert summary["failed_retirements"] == 0
    assert len(report["sessions"]) == expected_sessions
    selected = [session for session in report["sessions"] if session["selected"]]
    assert len(selected) == expected_sessions
    assert len({session["session_id"] for session in selected}) == expected_sessions
    assert sum(session["logical_turn_count"] for session in selected) == expected_turns
    assert all(session["error"] is None for session in selected)
    current = [turn for session in selected for turn in session["turns"] if turn["source_present"]]
    assert len(current) == expected_turns
    assert all(turn["action"] in {"imported", "updated", "skipped"} and turn["error_code"] is None for turn in current)
    assert len({(turn["session_id"], turn["turn_key"]) for turn in current}) == expected_turns
    counts = Counter(
        "unchanged" if turn["action"] == "skipped" and turn["reason"] == "unchanged" else turn["action"]
        for turn in current
    )
    for action in ("imported", "updated", "skipped", "unchanged"):
        assert summary[action + "_turns"] == counts[action]
    assert sum(counts.values()) == expected_turns
    ids = {turn["trace_id"] for turn in current}
    assert None not in ids and len(ids) == expected_turns
    assert len(actual_trace_ids) == expected_turns and set(actual_trace_ids) == ids
