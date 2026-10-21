"""The versioned result document and its human/JSON renderers.

The result document is the importer's ONLY output surface: every field
carries identifiers, counts, warning/error codes, hashes, timestamps, or
the experiment name — never prompt, response, tool, or source-code
content, absolute source paths, credentials, or tracking-URI secrets.

Shape (schema :data:`RESULT_SCHEMA`):

- the top level always carries ``schema_version``, ``dry_run``,
  ``experiment``, ``source``, ``redaction``, ``summary``, ``sessions``,
  and ``errors``;
- ``TurnResult.action`` and ``SessionResult.action`` are exactly the
  normative enum ``imported | updated | skipped | failed | retired``
  (imported from :mod:`opencode_mlflow.mlflow.reconciliation`, which
  documents the identical vocabulary);
- a dry run reports every turn as ``action="skipped"`` with
  ``reason="dry_run"`` and a predicted ``planned_action`` drawn from the
  same enum, so automation can distinguish preview from completion
  without an action outside the enum;
- the summary's current-turn counts are disjoint — every current turn
  (``source_present`` true) is counted in exactly one of the
  imported/updated/skipped/failed buckets — while ``retired_traces``
  and ``unexamined_owned_traces`` are separate trace-level counts; a
  FAILED trace retirement (a stale entry whose retirement could not be
  performed) also counts in ``turns_failed``, so an exit status of 1 is
  always explainable from the summary alone;
- a stale-trace retirement appears as a turn-shaped entry with
  ``source_present=false`` (the source no longer carries that turn) and
  is excluded from the current-turn counts;
- the broad-selection warning (no optional selection filter supplied) is
  a report-level event in ``summary.warnings`` — never a log line with
  content.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from opencode_mlflow.content.redaction import RedactionReport
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.mlflow.reconciliation import (
    ACTION_FAILED,
    ACTION_IMPORTED,
    ACTION_RETIRED,
    ACTION_SKIPPED,
    ACTION_UPDATED,
)

if TYPE_CHECKING:
    from opencode_mlflow.importer import ImportResult

__all__ = [
    "ACTION_SET",
    "RESULT_SCHEMA",
    "REASON_DRY_RUN",
    "REASON_NOT_SELECTED",
    "ReportBuilder",
    "SessionResult",
    "TurnResult",
    "render_human",
    "render_json",
]

RESULT_SCHEMA = "opencode-mlflow.result.v1"
"""Schema version of the importer's result document."""

ACTION_SET = frozenset({ACTION_IMPORTED, ACTION_UPDATED, ACTION_SKIPPED, ACTION_FAILED, ACTION_RETIRED})
"""The normative action enum for sessions and turns."""

REASON_DRY_RUN = "dry_run"
"""A dry-run turn's reason: the action was predicted, not performed."""

REASON_NOT_SELECTED = "not_selected"
"""An unselected session's reason: the filters excluded it."""

_EXPERIMENT_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class TurnResult:
    """One turn-shaped report entry.

    ``action`` is exactly the normative enum; ``planned_action`` (same
    enum) is set only by a dry run; ``trace_id`` is present wherever the
    run knows it; ``source_present`` is ``False`` only for a stale-trace
    retirement (the trace's turn is no longer in the source); and
    ``error_code`` carries a failed item's safe code.
    """

    session_id: str
    turn_key: str
    action: str
    turn_index: int | None = None
    reason: str | None = None
    planned_action: str | None = None
    trace_id: str | None = None
    source_present: bool = True
    error_code: str | None = None
    warnings: tuple[str, ...] = ()
    affected_trace_ids: tuple[str, ...] = ()
    retired_trace_ids: tuple[str, ...] = ()
    conflicting_trace_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionResult:
    """One session-shaped report entry (derived by the builder)."""

    session_id: str
    selected: bool
    action: str
    reason: str | None = None
    turns: tuple[TurnResult, ...] = ()


@dataclass
class _SessionDraft:
    """Mutable accumulation state for one session while a run reports."""

    session_id: str
    selected: bool
    turns: list[TurnResult] = field(default_factory=list)
    logical_turn_count: int = 0
    record_count: int = 0
    warnings: tuple[str, ...] = ()


class ReportBuilder:
    """Accumulates one run's safe report fields and builds the document.

    Sections stay partially-filled on a pre-write failure; :meth:`build`
    always emits every top-level section, defaulting the unset ones, so a
    failure report is as valid a result document as a success report.
    """

    def __init__(
        self,
        *,
        dry_run: bool,
        experiment_name: str,
        redaction_mode: RedactionMode,
        generated_ns: int | None = None,
    ) -> None:
        self._dry_run = dry_run
        self._experiment_name = experiment_name
        self._redaction_mode = redaction_mode
        self._generated_ns = generated_ns
        self._experiment_id: str | None = None
        self._experiment_status = _EXPERIMENT_UNRESOLVED
        self._source_generation: str | None = None
        self._source_fingerprint: str | None = None
        self._sessions_scanned = 0
        self._sessions_selected = 0
        self._redaction_report = RedactionReport()
        self._sessions: dict[str, _SessionDraft] = {}
        self._retired_traces = 0
        self._unexamined_owned_traces = 0
        self._cleanup_failures = 0
        self._errors: list[dict[str, str]] = []
        self._warnings: list[dict[str, str]] = []

    # -- section setters ------------------------------------------------------

    def set_experiment(self, experiment_id: str | None, status: str) -> None:
        """Record the resolved (or planned) experiment."""
        self._experiment_id = experiment_id
        self._experiment_status = status

    def set_source(self, *, schema_generation: str | None, fingerprint: str | None) -> None:
        """Record the source schema generation and database fingerprint (no path)."""
        self._source_generation = schema_generation
        self._source_fingerprint = fingerprint

    def set_source_counts(self, *, sessions_scanned: int, sessions_selected: int) -> None:
        """Record the discovery scan counts."""
        self._sessions_scanned = sessions_scanned
        self._sessions_selected = sessions_selected

    def set_redaction_report(self, report: RedactionReport) -> None:
        """Record the run's redaction counts (counts only, never content)."""
        self._redaction_report = report

    def set_retired_traces(self, count: int) -> None:
        """Record actual trace retirements; dry-run predictions stay planned-only."""
        self._retired_traces = 0 if self._dry_run else count

    def set_unexamined_owned_traces(self, count: int) -> None:
        """Record how many owned traces were outside the reconciliation scope."""
        self._unexamined_owned_traces = count

    def set_cleanup_failures(self, count: int) -> None:
        """Record sensitive temporary-data cleanup failures outside item turns."""
        self._cleanup_failures = count

    def add_cleanup_failure(self) -> None:
        """Count one distinct temporary-data cleanup failure outside item turns."""
        self._cleanup_failures += 1

    def add_error(self, code: str, message: str) -> None:
        """Append one safe invocation-level error entry."""
        self._errors.append({"code": code, "message": message})

    def add_warning(self, code: str, message: str) -> None:
        """Append one safe report-level warning event."""
        self._warnings.append({"code": code, "message": message})

    def add_session_scan(
        self,
        session_id: str,
        *,
        selected: bool,
        logical_turn_count: int = 0,
        record_count: int = 0,
        warnings: tuple[str, ...] = (),
    ) -> None:
        """Register one scanned session (selected or not), in scan order."""
        self._sessions.setdefault(
            session_id,
            _SessionDraft(
                session_id=session_id,
                selected=selected,
                logical_turn_count=logical_turn_count,
                record_count=record_count,
                warnings=warnings,
            ),
        )

    def add_turn(self, turn: TurnResult) -> None:
        """Append one turn-shaped entry to its session (registering unknown sessions)."""
        draft = self._sessions.get(turn.session_id)
        if draft is None:
            draft = _SessionDraft(session_id=turn.session_id, selected=True)
            self._sessions[turn.session_id] = draft
        draft.turns.append(turn)

    # -- document assembly ------------------------------------------------------

    def build(self) -> dict[str, object]:
        """Build the result document with every top-level section present."""
        sessions: list[dict[str, object]] = []
        action_counts: Counter[str] = Counter()
        planned_counts: Counter[str] = Counter()
        failed_retirements = 0
        for draft in self._sessions.values():
            for turn in draft.turns:
                if turn.source_present:
                    action_counts[turn.action] += 1
                elif turn.action == ACTION_FAILED:
                    # A failed trace retirement is a failed item: the
                    # summary's failed count covers every failed entry, so
                    # an exit status of 1 is always explainable from the
                    # summary alone.
                    failed_retirements += 1
                if turn.planned_action is not None:
                    planned_counts[turn.planned_action] += 1
            sessions.append(_session_payload(draft, dry_run=self._dry_run))
        summary: dict[str, object] = {
            "selected_sessions": self._sessions_selected,
            "selected_turns": sum(d.logical_turn_count for d in self._sessions.values() if d.selected),
            "unchanged_turns": sum(
                t.action == ACTION_SKIPPED and t.reason == "unchanged" and t.source_present
                for d in self._sessions.values()
                for t in d.turns
            ),
            "imported_turns": action_counts[ACTION_IMPORTED],
            "updated_turns": action_counts[ACTION_UPDATED],
            "skipped_turns": action_counts[ACTION_SKIPPED],
            "failed_turns": action_counts[ACTION_FAILED],
            "turns_imported": action_counts[ACTION_IMPORTED],
            "turns_updated": action_counts[ACTION_UPDATED],
            "turns_skipped": action_counts[ACTION_SKIPPED],
            "turns_failed": action_counts[ACTION_FAILED] + failed_retirements,
            "failed_items": action_counts[ACTION_FAILED] + failed_retirements + self._cleanup_failures,
            "failed_retirements": failed_retirements,
            "retired_traces": self._retired_traces,
            "unexamined_owned_traces": self._unexamined_owned_traces,
            "warnings": list(self._warnings),
        }
        if self._dry_run:
            summary["planned"] = {
                "imported": planned_counts[ACTION_IMPORTED],
                "updated": planned_counts[ACTION_UPDATED],
                "retired": planned_counts[ACTION_RETIRED],
            }
        if self._generated_ns is not None:
            summary["generated_ns"] = self._generated_ns
        return {
            "schema_version": RESULT_SCHEMA,
            "dry_run": self._dry_run,
            "experiment": {
                "name": self._experiment_name,
                "id": self._experiment_id,
                "status": self._experiment_status,
            },
            "source": {
                "schema_generation": self._source_generation,
                "fingerprint": self._source_fingerprint,
                "sessions_scanned": self._sessions_scanned,
                "sessions_selected": self._sessions_selected,
                "logical_turn_count": sum(d.logical_turn_count for d in self._sessions.values()),
                "record_count": sum(d.record_count for d in self._sessions.values()),
            },
            "redaction": {
                "mode": self._redaction_mode.value,
                "detector_counts": dict(self._redaction_report.detector_counts),
                "affected_field_count": self._redaction_report.affected_field_count,
                "affected_record_count": self._redaction_report.affected_record_count,
            },
            "summary": summary,
            "sessions": sessions,
            "errors": list(self._errors),
        }


def _turn_payload(turn: TurnResult) -> dict[str, object]:
    """The sanitized JSON form of one turn entry."""
    return {
        "session_id": turn.session_id,
        "turn_key": turn.turn_key,
        "turn_index": turn.turn_index,
        "action": turn.action,
        "reason": turn.reason,
        "planned_action": turn.planned_action,
        "trace_id": turn.trace_id,
        "source_present": turn.source_present,
        "error_code": turn.error_code,
        "error": _error_payload(turn.error_code),
        "affected_trace_ids": list(turn.affected_trace_ids),
        "retired_trace_ids": list(turn.retired_trace_ids),
        "conflicting_trace_ids": list(turn.conflicting_trace_ids),
        "warnings": [_warning_payload(code) for code in turn.warnings],
    }


def _session_payload(draft: _SessionDraft, *, dry_run: bool = False) -> dict[str, object]:
    """The sanitized JSON form of one session entry."""
    result = _session_result(draft)
    return {
        "session_id": result.session_id,
        "selected": result.selected,
        "action": result.action,
        "reason": REASON_DRY_RUN if dry_run and draft.selected else result.reason,
        "turns": [_turn_payload(turn) for turn in result.turns],
        "logical_turn_count": draft.logical_turn_count,
        "record_count": draft.record_count,
        "warnings": [_warning_payload(code) for code in draft.warnings],
        "error": next((_error_payload(t.error_code) for t in draft.turns if t.error_code), None),
    }


def _session_result(draft: _SessionDraft) -> SessionResult:
    """Derive one session's action from its turns (unselected → skipped)."""
    if not draft.selected:
        return SessionResult(
            session_id=draft.session_id,
            selected=False,
            action=ACTION_SKIPPED,
            reason=REASON_NOT_SELECTED,
            turns=tuple(draft.turns),
        )
    actions = {turn.action for turn in draft.turns}
    if ACTION_FAILED in actions:
        action = ACTION_FAILED
    elif ACTION_UPDATED in actions or (ACTION_RETIRED in actions and ACTION_IMPORTED in actions):
        action = ACTION_UPDATED
    elif ACTION_IMPORTED in actions:
        action = ACTION_IMPORTED
    elif ACTION_RETIRED in actions:
        action = ACTION_RETIRED
    else:
        action = ACTION_SKIPPED
    return SessionResult(session_id=draft.session_id, selected=True, action=action, turns=tuple(draft.turns))


def _error_payload(code: str | None) -> dict[str, str] | None:
    return {"code": code, "message": "the item could not be completed"} if code else None


def _warning_payload(code: str) -> dict[str, str]:
    return {"code": code, "message": "the source record carries a normalization warning"}


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_json(result: ImportResult) -> str:
    """Render the result document as deterministic, sorted, indented JSON."""
    return json.dumps(dict(result.report), sort_keys=True, indent=2)


def render_human(result: ImportResult) -> str:
    """Render the result document as concise, progress-oriented lines.

    Every line derives from the (already sanitized) result document, so
    the human output carries no source content or secrets either. The
    broad-selection warning is included as a warning line.
    """
    report = result.report
    experiment = _section(report, "experiment")
    source = _section(report, "source")
    redaction = _section(report, "redaction")
    summary = _section(report, "summary")
    has_errors = result.error is not None or bool(_event_entries(report, "errors"))
    lines: list[str] = []
    if has_errors:
        lines.append("opencode-mlflow import failed")
    elif result.dry_run:
        lines.append("opencode-mlflow import dry-run complete")
    elif summary.get("turns_failed") or summary.get("failed_items"):
        lines.append("opencode-mlflow import finished with item failures")
    else:
        lines.append("opencode-mlflow import complete")
    experiment_id = experiment.get("id")
    experiment_suffix = f", {experiment_id}" if experiment_id else ""
    lines.append(f"experiment: {experiment.get('name')} ({experiment.get('status')}{experiment_suffix})")
    generation = source.get("schema_generation") or "unknown"
    lines.append(
        f"source: {generation} schema, {source.get('sessions_scanned')} session(s) scanned, "
        f"{source.get('sessions_selected')} selected"
    )
    redaction_line = f"redaction: {redaction.get('mode')}"
    affected_fields = redaction.get("affected_field_count") or 0
    if affected_fields:
        redaction_line += (
            f" ({affected_fields} field(s) affected across {redaction.get('affected_record_count')} record(s))"
        )
    lines.append(redaction_line)
    lines.append(
        f"turns: {summary.get('turns_imported')} imported, {summary.get('turns_updated')} updated, "
        f"{summary.get('turns_skipped')} skipped, {summary.get('turns_failed')} failed"
    )
    planned = summary.get("planned")
    if isinstance(planned, Mapping):
        lines.append(
            f"planned: {planned.get('imported')} import(s), {planned.get('updated')} update(s), "
            f"{planned.get('retired')} retirement(s)"
        )
    lines.append(
        f"retired traces: {summary.get('retired_traces')}; "
        f"unexamined owned traces: {summary.get('unexamined_owned_traces')}"
    )
    for warning in _event_entries(summary, "warnings"):
        lines.append(f"warning ({warning.get('code')}): {warning.get('message')}")
    for error in _event_entries(report, "errors"):
        lines.append(f"error ({error.get('code')}): {error.get('message')}")
    return "\n".join(lines)


def _section(report: Mapping[str, object], name: str) -> Mapping[str, object]:
    """One top-level section as a mapping (defensive on partial documents)."""
    value = report.get(name)
    return value if isinstance(value, Mapping) else {}


def _event_entries(section: Mapping[str, object], name: str) -> list[Mapping[str, object]]:
    """One section's event entries as mappings (defensive on partial documents)."""
    value = section.get(name)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []
