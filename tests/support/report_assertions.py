"""Shared assertion helpers for importer result documents.

These helpers are the one place the result-document contract is asserted,
so unit tests, CLI tests, and BDD scenarios check the same shape: the
schema version, the required top-level sections, the normative action
enum for sessions and turns, the disjoint current-turn count units, and
the absence of source content or secrets in rendered output.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from opencode_mlflow.reporting import ACTION_SET, RESULT_SCHEMA

RESULT_ACTIONS = ACTION_SET
"""The normative action enum for ``SessionResult.action`` and ``TurnResult.action``."""

REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "dry_run",
    "experiment",
    "source",
    "redaction",
    "summary",
    "sessions",
    "errors",
)
"""Top-level keys every result document always carries."""


def assert_result_document(report: Mapping[str, object]) -> None:
    """Assert the result document's schema version, sections, and action enum."""
    assert report["schema_version"] == RESULT_SCHEMA
    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in report, f"the result document is missing its {key} section"
    assert isinstance(report["sessions"], list)
    for session in report["sessions"]:
        assert isinstance(session, Mapping)
        assert session["action"] in RESULT_ACTIONS
        assert session["session_id"]
        assert isinstance(session["turns"], list)
        for turn in session["turns"]:
            assert isinstance(turn, Mapping)
            assert turn["action"] in RESULT_ACTIONS
            assert turn["turn_key"]
            planned = turn.get("planned_action")
            if planned is not None:
                assert planned in RESULT_ACTIONS
            for trace_id_field in ("affected_trace_ids", "retired_trace_ids", "conflicting_trace_ids"):
                assert isinstance(turn[trace_id_field], list)
                assert all(isinstance(trace_id, str) for trace_id in turn[trace_id_field])


def assert_counts_are_disjoint(report: Mapping[str, object]) -> None:
    """Assert the summary's count units partition the report's entries exactly.

    Every current turn (``source_present`` true) is counted in exactly one
    of the imported/updated/skipped/failed buckets; every failed entry —
    including a failed trace retirement, which is a stale entry — counts
    in ``turns_failed``; retired traces are reported separately as
    ``retired_traces`` and never count as current turns.
    """
    summary = report["summary"]
    assert isinstance(summary, Mapping)
    entries = [turn for session in report["sessions"] for turn in session["turns"]]
    current = [turn for turn in entries if turn["source_present"]]
    failed = [turn for turn in entries if turn["action"] == "failed"]

    assert summary["turns_failed"] == len(failed)
    for bucket, action in (
        ("turns_imported", "imported"),
        ("turns_updated", "updated"),
        ("turns_skipped", "skipped"),
    ):
        assert summary[bucket] == len([turn for turn in current if turn["action"] == action])
    # Every current turn is counted exactly once: the non-failed ones across
    # the imported/updated/skipped buckets, the failed ones (current failed
    # writes and stale failed retirements alike) in turns_failed.
    assert summary["turns_imported"] + summary["turns_updated"] + summary["turns_skipped"] == len(
        [turn for turn in current if turn["action"] != "failed"]
    )
    retired_entries = [turn for turn in entries if turn["action"] == "retired"]
    assert retired_entries == [turn for turn in retired_entries if not turn["source_present"]], (
        "a retired trace entry must record source_present=false"
    )
    assert isinstance(summary["retired_traces"], int)
    assert isinstance(summary["unexamined_owned_traces"], int)


def assert_no_sentinels(text: str, sentinels: Iterable[str]) -> None:
    """Assert none of the sentinel strings appears in rendered output."""
    for sentinel in sentinels:
        assert sentinel not in text, f"rendered output leaked the sentinel {sentinel!r}"
