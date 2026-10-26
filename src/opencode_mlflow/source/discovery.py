"""Streaming, sanitized, disk-backed session discovery plan.

:func:`discover_sessions` is one streaming pass over the replay
adapter's compatible sessions. It consumes :meth:`SourceAdapter.iter_session_headers`
in the adapter's normalized order — asserting monotonic
``(creation_time, session_id)`` keys while consuming, never
materializing the header list merely to sort it — and for every
compatible session:

- streams the session's records once through
  :func:`opencode_mlflow.source.turns.iter_turn_summaries`, collecting
  model identities (session default, message, and part level), source
  counts, and safe warning codes as a side effect;
- evaluates :func:`opencode_mlflow.source.selection.select_session`
  against the resulting :class:`SessionDiscoverySummary`;
- appends exactly one sanitized :class:`SessionScanEntry` — for every
  compatible session, selected or not — and one
  :class:`DiscoveryTurnEntry` per logical turn of every *selected*
  session (the write phase replays and imports only selected sessions,
  so unselected sessions' turns never enter the plan; their scan entries
  still record the logical turn count).

Every turn of a selected session is built through Task 5's
:func:`opencode_mlflow.source.turns.build_replay_turn` and Task 4A's
:func:`opencode_mlflow.content.fingerprint.build_effective_turn`, and
its fingerprints come exclusively from the Task 4A functions — the
session context fingerprint (also the scan entry's session-identity
fingerprint, computed through the same seam so zero-turn sessions carry
it too), the turn source fingerprint, and the render fingerprint with
the recorded render-contract version inputs. Discovery invents no
second canonicalization or hashing format.

Sanitization contract: the plan is JSONL and carries identifiers,
counts, warning codes, timestamps, model identities, and hashes only —
never transcript, prompt, response, tool, or source-code content. The
plan file is created with mode ``0o600`` inside the caller's
restrictive temporary directory.

Tombstones: a selected session whose scan completed with zero logical
turns is an explicit tombstone (``selected``, ``complete_scan``, and
``tombstone`` all true), which permits retiring all previously owned
turns of that session during reconciliation while a filtered run never
retires traces of unselected sessions.

Redaction boundary: discovery replays the normalized snapshot, whose
values are already effective (redaction ran before temporary
persistence). Directory/model selection uses only the snapshot's
invocation-specific pre-redaction match booleans; those booleans never
enter plan entries or fingerprint material. Effective turns are built
with an identity redactor and fingerprinted under the verbatim mode tag;
every turn entry records that mode and the render-contract versions it
was hashed with, so the write phase compares replayed fingerprints under
the recorded inputs.

:class:`DiscoveryPlanReader` streams the plan back and rejects
malformed entries, unknown plan schema versions, duplicate
``(session_id, turn_key)`` identities, missing fingerprints, turn
entries without a matching session scan, and a selected complete scan
missing its zero-turn tombstone.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, cast

from opencode_mlflow import __version__ as _IMPORTER_VERSION
from opencode_mlflow.content.fingerprint import (
    build_effective_turn,
    render_fingerprint,
    session_context_fingerprint,
    source_fingerprint,
)
from opencode_mlflow.contracts import IDENTITY_REDACTOR, RedactionMode
from opencode_mlflow.errors import ImporterError, SourceCompatibilityError
from opencode_mlflow.source.model import (
    AdapterGeneration,
    SessionHeader,
    SourceAdapter,
    SourceRecord,
    TimestampUnit,
    TimestampValue,
    TurnSummary,
)
from opencode_mlflow.source.model_identity import extract_model_identity
from opencode_mlflow.source.ordering import session_sort_key
from opencode_mlflow.source.outcomes import SESSION_EVENT_TABLES
from opencode_mlflow.source.selection import (
    SelectionEvidence,
    SelectionFilters,
    SessionDiscoverySummary,
    select_session,
)
from opencode_mlflow.source.snapshot import SourceSnapshot
from opencode_mlflow.source.turns import build_replay_turn, iter_turn_summaries

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DISCOVERY_PLAN_SCHEMA_VERSION",
    "DiscoveryPlanReader",
    "DiscoveryPlanWriter",
    "DiscoveryStats",
    "DiscoveryTurnEntry",
    "MAPPING_VERSION",
    "PLANNED_ACTION_IMPORT",
    "SessionScanEntry",
    "discover_sessions",
    "extract_model_identity",
]

DISCOVERY_PLAN_SCHEMA_VERSION = "opencode-mlflow-discovery-plan-v1"
"""Schema version of the discovery plan; the reader rejects any other value."""

ARTIFACT_SCHEMA_VERSION = "artifact-v2"
"""Artifact schema version recorded as a render-contract input in the plan."""

MAPPING_VERSION = "mapping-v4"
"""Turn-to-trace mapping version recorded as a render-contract input in the plan."""

PLANNED_ACTION_IMPORT = "import"
"""Discovery-level planned action for every turn of a selected session."""

SUPPORTED_ADAPTER_GENERATIONS: frozenset[str] = frozenset(("legacy", "current"))
"""Adapter generations the discovery pass knows how to scan."""

_DISCOVERY_REDACTION_MODE = RedactionMode.VERBATIM
"""The mode tag discovery hashes the effective (already-persisted) replay values under."""

_PLAN_HEADER_KIND = "plan_header"
_SESSION_SCAN_KIND = "session_scan"
_TURN_KIND = "turn"


# ---------------------------------------------------------------------------
# Plan entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionScanEntry:
    """One compatible session's sanitized scan record.

    Written for every compatible session, selected or not, including a
    selected session with zero logical turns. Carries the selection and
    reconciliation-scope flags (``selected``/``complete_scan``/
    ``tombstone``, with :attr:`retirement_scope` deriving the stale-turn
    retirement scope), the logical turn and record counts, the creation
    instant, the Task 4A session-identity fingerprint, and safe warning
    codes — never transcript or source-code content.
    """

    session_id: str
    adapter_generation: AdapterGeneration
    selected: bool
    complete_scan: bool
    tombstone: bool
    logical_turn_count: int
    record_count: int
    creation_time_ns: int | None
    session_context_fingerprint: str
    warnings: tuple[str, ...]
    catalog_version: str = ""

    @property
    def retirement_scope(self) -> bool:
        """Whether the session's previously owned turns are in retirement scope.

        Only a selected session whose scan completed may retire stale
        turns; a filtered run never retires traces of unselected
        sessions.
        """
        return self.selected and self.complete_scan


@dataclass(frozen=True)
class DiscoveryTurnEntry:
    """One logical turn of a selected session, sanitized for the plan.

    Carries the source/session/turn identities, normalized timestamps
    (epoch nanoseconds), record counts, the turn's own model
    identities, safe warning codes, the Task 4A fingerprints with the
    render-contract version inputs they were computed under, and the
    discovery-level planned action — never prompt, response, tool, or
    source-code content.
    """

    session_id: str
    turn_index: int
    turn_key: str
    adapter_generation: AdapterGeneration
    planned_action: str
    start_time_ns: int | None
    end_time_ns: int | None
    record_count: int
    user_record_count: int
    assistant_record_count: int
    terminal_outcome: str | None
    model_identities: frozenset[str]
    warnings: tuple[str, ...]
    session_context_fingerprint: str
    source_fingerprint: str
    render_fingerprint: str
    artifact_schema_version: str
    mapping_version: str
    importer_version: str
    redaction_mode: str
    catalog_version: str = ""


@dataclass(frozen=True)
class DiscoveryStats:
    """Outcome of one discovery pass."""

    session_scan_count: int
    selected_session_count: int
    turn_entry_count: int
    tombstone_count: int
    incompatible_session_count: int
    warning_count: int
    plan_path: Path


# ---------------------------------------------------------------------------
# Plan writer
# ---------------------------------------------------------------------------


class DiscoveryPlanWriter:
    """Append-only JSONL writer for the sanitized discovery plan.

    The first line is the plan header naming :data:`DISCOVERY_PLAN_SCHEMA_VERSION`;
    each following line is one sanitized entry serialized through an
    explicit field whitelist. The file is created with mode ``0o600``
    in the caller's (already restrictive) directory. The writer
    serializes exactly what it is given — plan validation is the
    reader's contract.
    """

    def __init__(self, plan_path: Path) -> None:
        self._plan_path = plan_path
        self._handle: IO[str] = plan_path.open("w", encoding="utf-8", newline="\n")
        try:
            os.chmod(plan_path, 0o600)
            self._handle.write(
                _dumps({"kind": _PLAN_HEADER_KIND, "plan_schema_version": DISCOVERY_PLAN_SCHEMA_VERSION}) + "\n"
            )
        except BaseException:
            self._handle.close()
            raise
        self._closed = False

    def write_session_scan(self, entry: SessionScanEntry) -> None:
        """Append one session scan entry."""
        self._write(_session_scan_payload(entry))

    def write_turn(self, entry: DiscoveryTurnEntry) -> None:
        """Append one turn entry."""
        self._write(_turn_payload(entry))

    def close(self) -> None:
        """Flush and close the plan file (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._handle.close()

    def __enter__(self) -> DiscoveryPlanWriter:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _write(self, payload: Mapping[str, object]) -> None:
        if self._closed:
            raise ValueError("the discovery plan writer was already closed")
        self._handle.write(_dumps(payload) + "\n")


class _TurnSummarySpool:
    """Restrictive disk-backed replay for one session's sanitized summaries."""

    def __init__(self, directory: Path) -> None:
        descriptor, path = tempfile.mkstemp(prefix=".turn-summaries-", suffix=".jsonl", dir=directory)
        os.chmod(path, 0o600)
        self._path = Path(path)
        self._handle = os.fdopen(descriptor, "w+", encoding="utf-8", newline="\n")
        self._closed = False

    def write(self, summary: TurnSummary) -> None:
        if self._closed:
            raise ValueError("the turn-summary spool was already closed")
        self._handle.write(_dumps(_turn_summary_payload(summary)) + "\n")

    def __iter__(self) -> Iterator[TurnSummary]:
        if self._closed:
            raise ValueError("the turn-summary spool was already closed")
        self._handle.flush()
        self._handle.seek(0)
        for line in self._handle:
            payload = json.loads(line)
            if not isinstance(payload, dict):  # pragma: no cover - only this module writes the spool
                raise ValueError("the turn-summary spool contains a malformed entry")
            yield _turn_summary_from_payload(payload)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._handle.close()
        finally:
            self._path.unlink(missing_ok=True)

    def __enter__(self) -> _TurnSummarySpool:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def _dumps(payload: Mapping[str, object]) -> str:
    """Serialize one plan line deterministically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _turn_summary_payload(summary: TurnSummary) -> dict[str, object]:
    """Serialize only the content-free fields needed to write a plan entry."""
    return {
        "session_id": summary.session_id,
        "turn_index": summary.turn_index,
        "turn_key": summary.turn_key,
        "first_user_source_id": summary.first_user_source_id,
        "first_user_source_type": summary.first_user_source_type,
        "start_time": _timestamp_payload(summary.start_time),
        "end_time": _timestamp_payload(summary.end_time),
        "record_count": summary.record_count,
        "user_record_count": summary.user_record_count,
        "assistant_record_count": summary.assistant_record_count,
        "terminal_outcome": summary.terminal_outcome,
        "root_result_source_id": summary.root_result_source_id,
        "source_fingerprint": summary.source_fingerprint,
        "warnings": list(summary.warnings),
        "source_outcome": summary.source_outcome,
    }


def _timestamp_payload(timestamp: TimestampValue | None) -> dict[str, object] | None:
    if timestamp is None:
        return None
    return {"value": timestamp.value, "unit": timestamp.unit}


def _turn_summary_from_payload(payload: Mapping[str, object]) -> TurnSummary:
    """Restore one internally generated summary spool entry."""
    start = _timestamp_from_payload(payload.get("start_time"))
    end = _timestamp_from_payload(payload.get("end_time"))
    warnings = payload.get("warnings")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise ValueError("the turn-summary spool contains malformed warnings")
    return TurnSummary(
        session_id=cast(str, payload["session_id"]),
        turn_index=cast(int, payload["turn_index"]),
        turn_key=cast(str, payload["turn_key"]),
        first_user_source_id=cast(str | None, payload.get("first_user_source_id")),
        first_user_source_type=cast(str | None, payload.get("first_user_source_type")),
        start_time=start,
        end_time=end,
        record_count=cast(int, payload["record_count"]),
        user_record_count=cast(int, payload["user_record_count"]),
        assistant_record_count=cast(int, payload["assistant_record_count"]),
        terminal_outcome=cast(str | None, payload.get("terminal_outcome")),
        root_result_source_id=cast(str | None, payload.get("root_result_source_id")),
        source_fingerprint=cast(str | None, payload.get("source_fingerprint")),
        warnings=tuple(warnings),
        source_outcome=cast(str | None, payload.get("source_outcome")),
    )


def _timestamp_from_payload(value: object) -> TimestampValue | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("the turn-summary spool contains a malformed timestamp")
    raw_value = value.get("value")
    unit = value.get("unit")
    if not isinstance(unit, str) or unit not in {"s", "ms", "us", "ns"}:
        raise ValueError("the turn-summary spool contains a malformed timestamp")
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError("the turn-summary spool contains a malformed timestamp")
    return TimestampValue(value=raw_value, unit=cast(TimestampUnit, unit))


def _session_scan_payload(entry: SessionScanEntry) -> dict[str, object]:
    """The sanitized JSON form of one session scan entry."""
    return {
        "kind": _SESSION_SCAN_KIND,
        "session_id": entry.session_id,
        "adapter_generation": entry.adapter_generation,
        "selected": entry.selected,
        "complete_scan": entry.complete_scan,
        "tombstone": entry.tombstone,
        "logical_turn_count": entry.logical_turn_count,
        "record_count": entry.record_count,
        "creation_time_ns": entry.creation_time_ns,
        "session_context_fingerprint": entry.session_context_fingerprint,
        "warnings": list(entry.warnings),
        "catalog_version": entry.catalog_version,
    }


def _turn_payload(entry: DiscoveryTurnEntry) -> dict[str, object]:
    """The sanitized JSON form of one turn entry."""
    return {
        "kind": _TURN_KIND,
        "session_id": entry.session_id,
        "turn_index": entry.turn_index,
        "turn_key": entry.turn_key,
        "adapter_generation": entry.adapter_generation,
        "planned_action": entry.planned_action,
        "start_time_ns": entry.start_time_ns,
        "end_time_ns": entry.end_time_ns,
        "record_count": entry.record_count,
        "user_record_count": entry.user_record_count,
        "assistant_record_count": entry.assistant_record_count,
        "terminal_outcome": entry.terminal_outcome,
        "model_identities": sorted(entry.model_identities),
        "warnings": list(entry.warnings),
        "session_context_fingerprint": entry.session_context_fingerprint,
        "source_fingerprint": entry.source_fingerprint,
        "render_fingerprint": entry.render_fingerprint,
        "artifact_schema_version": entry.artifact_schema_version,
        "mapping_version": entry.mapping_version,
        "importer_version": entry.importer_version,
        "redaction_mode": entry.redaction_mode,
        "catalog_version": entry.catalog_version,
    }


# ---------------------------------------------------------------------------
# Plan reader
# ---------------------------------------------------------------------------


class DiscoveryPlanReader:
    """Validating streaming reader for the discovery plan.

    :meth:`iter_entries` yields every entry in file order, rejecting
    malformed entries, unknown plan schema versions, duplicate
    ``(session_id, turn_key)`` identities, missing fingerprints, turn
    entries without a matching (earlier) session scan, and a selected
    complete scan missing its zero-turn tombstone.
    :meth:`iter_session_scans`/:meth:`iter_turns` are filtered views
    over the same validating stream.
    """

    def __init__(self, plan_path: Path) -> None:
        self._plan_path = plan_path

    def iter_entries(self) -> Iterator[SessionScanEntry | DiscoveryTurnEntry]:
        """Stream every plan entry, validating the plan as it is read."""
        seen_scans: set[str] = set()
        seen_turns: set[tuple[str, str]] = set()
        with self._plan_path.open("r", encoding="utf-8") as handle:
            header_seen = False
            for line in handle:
                if not line.strip():
                    continue
                payload = _load_line(line)
                if not header_seen:
                    header_seen = True
                    _validate_header(payload)
                    continue
                entry = self._build_entry(payload)
                if isinstance(entry, SessionScanEntry):
                    seen_scans.add(entry.session_id)
                    yield entry
                    continue
                if entry.session_id not in seen_scans:
                    raise _malformed_plan(
                        "discovery_plan_turn_without_scan",
                        "a discovery turn entry has no matching session scan",
                    )
                identity = (entry.session_id, entry.turn_key)
                if identity in seen_turns:
                    raise _malformed_plan(
                        "discovery_plan_duplicate_turn",
                        "the discovery plan contains a duplicate turn identity",
                    )
                seen_turns.add(identity)
                yield entry
        if not header_seen:
            raise _malformed_plan("discovery_plan_invalid", "the discovery plan is missing its header")

    def iter_session_scans(self) -> Iterator[SessionScanEntry]:
        """Stream every session scan entry (validating the whole plan)."""
        for entry in self.iter_entries():
            if isinstance(entry, SessionScanEntry):
                yield entry

    def iter_turns(self) -> Iterator[DiscoveryTurnEntry]:
        """Stream every turn entry (validating the whole plan)."""
        for entry in self.iter_entries():
            if not isinstance(entry, SessionScanEntry):
                yield entry

    def _build_entry(self, payload: Mapping[str, object]) -> SessionScanEntry | DiscoveryTurnEntry:
        kind = payload.get("kind")
        if kind == _SESSION_SCAN_KIND:
            return _session_scan_from_payload(payload)
        if kind == _TURN_KIND:
            return _turn_from_payload(payload)
        raise _malformed_plan("discovery_plan_invalid", "a discovery plan entry has an unknown kind")


def _load_line(line: str) -> dict[str, object]:
    try:
        payload = json.loads(line)
    except ValueError:
        raise _malformed_plan("discovery_plan_invalid", "a discovery plan entry is not valid JSON") from None
    if not isinstance(payload, dict):
        raise _malformed_plan("discovery_plan_invalid", "a discovery plan entry is not a JSON object")
    return payload


def _validate_header(payload: Mapping[str, object]) -> None:
    version = payload.get("plan_schema_version")
    if payload.get("kind") != _PLAN_HEADER_KIND or not isinstance(version, str):
        raise _malformed_plan("discovery_plan_invalid", "the discovery plan is missing its header")
    if version != DISCOVERY_PLAN_SCHEMA_VERSION:
        raise _malformed_plan(
            "discovery_plan_version_unknown",
            "the discovery plan schema version is not supported",
        )


def _session_scan_from_payload(payload: Mapping[str, object]) -> SessionScanEntry:
    entry = SessionScanEntry(
        session_id=_required_str(payload, "session_id"),
        adapter_generation=cast(AdapterGeneration, _required_str(payload, "adapter_generation")),
        selected=_required_bool(payload, "selected"),
        complete_scan=_required_bool(payload, "complete_scan"),
        tombstone=_required_bool(payload, "tombstone"),
        logical_turn_count=_required_int(payload, "logical_turn_count"),
        record_count=_required_int(payload, "record_count"),
        creation_time_ns=_optional_int(payload, "creation_time_ns"),
        session_context_fingerprint=_required_fingerprint(payload, "session_context_fingerprint"),
        warnings=_required_str_tuple(payload, "warnings"),
        catalog_version=_optional_str(payload, "catalog_version") or "",
    )
    tombstone_required = entry.selected and entry.complete_scan and entry.logical_turn_count == 0
    if entry.tombstone != tombstone_required:
        if tombstone_required:
            raise _malformed_plan(
                "discovery_plan_invalid",
                "a selected complete scan with zero turns is missing its tombstone",
            )
        raise _malformed_plan(
            "discovery_plan_invalid",
            "a discovery plan entry carries a tombstone outside its definition",
        )
    return entry


def _turn_from_payload(payload: Mapping[str, object]) -> DiscoveryTurnEntry:
    return DiscoveryTurnEntry(
        session_id=_required_str(payload, "session_id"),
        turn_index=_required_int(payload, "turn_index"),
        turn_key=_required_str(payload, "turn_key"),
        adapter_generation=cast(AdapterGeneration, _required_str(payload, "adapter_generation")),
        planned_action=_required_str(payload, "planned_action"),
        start_time_ns=_optional_int(payload, "start_time_ns"),
        end_time_ns=_optional_int(payload, "end_time_ns"),
        record_count=_required_int(payload, "record_count"),
        user_record_count=_required_int(payload, "user_record_count"),
        assistant_record_count=_required_int(payload, "assistant_record_count"),
        terminal_outcome=_optional_str(payload, "terminal_outcome"),
        model_identities=frozenset(_required_str_tuple(payload, "model_identities")),
        warnings=_required_str_tuple(payload, "warnings"),
        session_context_fingerprint=_required_fingerprint(payload, "session_context_fingerprint"),
        source_fingerprint=_required_fingerprint(payload, "source_fingerprint"),
        render_fingerprint=_required_fingerprint(payload, "render_fingerprint"),
        artifact_schema_version=_required_str(payload, "artifact_schema_version"),
        mapping_version=_required_str(payload, "mapping_version"),
        importer_version=_required_str(payload, "importer_version"),
        redaction_mode=_required_str(payload, "redaction_mode"),
        catalog_version=_optional_str(payload, "catalog_version") or "",
    )


def _malformed_plan(code: str, message: str) -> ImporterError:
    return ImporterError(code, message)


def _required_str(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise _malformed_plan("discovery_plan_invalid", f"a discovery plan entry is missing its {field}")
    return value


def _required_bool(payload: Mapping[str, object], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise _malformed_plan("discovery_plan_invalid", f"a discovery plan entry is missing its {field}")
    return value


def _required_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _malformed_plan("discovery_plan_invalid", f"a discovery plan entry is missing its {field}")
    return value


def _optional_int(payload: Mapping[str, object], field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    return _required_int(payload, field)


def _optional_str(payload: Mapping[str, object], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _malformed_plan("discovery_plan_invalid", f"a discovery plan entry is missing its {field}")
    return value


def _required_str_tuple(payload: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _malformed_plan("discovery_plan_invalid", f"a discovery plan entry is missing its {field}")
    return tuple(value)


def _required_fingerprint(payload: Mapping[str, object], field: str) -> str:
    value = _required_str(payload, field)
    if not value:
        raise _malformed_plan("discovery_plan_invalid", "a discovery plan entry is missing a fingerprint")
    return value


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_sessions(
    adapter: SourceAdapter,
    filters: SelectionFilters,
    plan_path: Path,
) -> DiscoveryStats:
    """Stream every compatible session into the sanitized plan at ``plan_path``.

    Sessions are consumed in the adapter's normalized order, asserting
    monotonic ``(creation_time, session_id)`` keys while streaming (the
    header list is never materialized merely to sort it). Every
    compatible session gets one :class:`SessionScanEntry`; every
    selected session additionally gets one :class:`DiscoveryTurnEntry`
    per logical turn, fingerprinted through the Task 4A seam.
    Incompatible/unsupported sessions are reported through the returned
    stats, never as plan entries.
    """
    snapshot = _replay_snapshot(adapter)
    description = adapter.inspect()
    catalog_version = description.catalog_version
    schema_warnings = tuple(dict.fromkeys(description.warnings))
    session_scan_count = 0
    selected_session_count = 0
    turn_entry_count = 0
    tombstone_count = 0
    incompatible_session_count = 0
    warning_count = 0
    previous_key: tuple[int, int, str] | None = None
    with DiscoveryPlanWriter(plan_path) as writer:
        for header in adapter.iter_session_headers():
            key = session_sort_key(header)
            if previous_key is not None and key < previous_key:
                raise SourceCompatibilityError(
                    "adapter_session_order",
                    "the replay adapter emitted sessions out of normalized order",
                )
            previous_key = key
            if header.adapter_generation not in SUPPORTED_ADAPTER_GENERATIONS:
                incompatible_session_count += 1
                warning_count += 1
                continue
            outcome = _scan_session(
                snapshot,
                adapter,
                header,
                filters,
                writer,
                catalog_version,
                schema_warnings,
            )
            session_scan_count += 1
            selected_session_count += 1 if outcome.selected else 0
            turn_entry_count += outcome.turn_entry_count
            tombstone_count += 1 if outcome.tombstone else 0
            warning_count += len(outcome.warnings)
    return DiscoveryStats(
        session_scan_count=session_scan_count,
        selected_session_count=selected_session_count,
        turn_entry_count=turn_entry_count,
        tombstone_count=tombstone_count,
        incompatible_session_count=incompatible_session_count,
        warning_count=warning_count,
        plan_path=plan_path,
    )


@dataclass
class _SessionScanOutcome:
    """Per-session bookkeeping returned by one scan."""

    selected: bool
    tombstone: bool
    turn_entry_count: int
    warnings: tuple[str, ...]


def _scan_session(
    snapshot: SourceSnapshot,
    adapter: SourceAdapter,
    header: SessionHeader,
    filters: SelectionFilters,
    writer: DiscoveryPlanWriter,
    catalog_version: str,
    schema_warnings: tuple[str, ...] = (),
) -> _SessionScanOutcome:
    """Scan one compatible session and append its sanitized plan entries.

    The session's records stream once through
    :func:`iter_turn_summaries` (collecting model identities, record
    counts, and warnings as a side effect) into a restrictive disk-backed
    summary spool, the session-scoped records stream once more for the same
    collection, selection is evaluated against the resulting counters, the
    scan entry is appended, and only then — for a selected session — are
    summaries replayed from disk and fingerprinted through the Task 4A seam.
    """
    collector = _SessionCollector(header)
    collector.record_warnings(schema_warnings)
    with _TurnSummarySpool(writer._plan_path.parent) as summaries:
        logical_turn_count = 0
        for turn_summary in iter_turn_summaries(collector.observe(adapter.iter_records(header.session_id))):
            summaries.write(turn_summary)
            logical_turn_count += 1
            collector.record_warnings(turn_summary.warnings)
        collector.drain(adapter.iter_session_scoped_records(header.session_id))

        complete_scan = True
        summary = SessionDiscoverySummary(
            session_id=header.session_id,
            creation_time=header.time_created.utc if header.time_created is not None else None,
            directory=Path(header.directory) if header.directory else None,
            model_identities=frozenset(collector.model_identities),
            logical_turn_count=logical_turn_count,
            warnings=collector.warnings,
            complete_scan=complete_scan,
        )
        selected = select_session(summary, filters, _selection_evidence(adapter, header.session_id, filters))
        tombstone = selected and complete_scan and logical_turn_count == 0

        # The session-identity fingerprint is the Task 4A session context
        # fingerprint; the header-only replay view lets every session —
        # including one with zero logical turns — carry it through the same
        # public build_effective_turn seam (its cursors are never opened).
        session_fingerprint = session_context_fingerprint(
            build_effective_turn(_SessionHeaderReplayTurn(header), IDENTITY_REDACTOR, _DISCOVERY_REDACTION_MODE)
        )
        writer.write_session_scan(
            SessionScanEntry(
                session_id=header.session_id,
                adapter_generation=header.adapter_generation,
                selected=selected,
                complete_scan=complete_scan,
                tombstone=tombstone,
                logical_turn_count=logical_turn_count,
                record_count=collector.record_count,
                creation_time_ns=header.time_created.nanoseconds if header.time_created is not None else None,
                session_context_fingerprint=session_fingerprint,
                warnings=collector.warnings,
                catalog_version=catalog_version,
            )
        )

        turn_entry_count = 0
        if not selected:
            return _SessionScanOutcome(selected, tombstone, turn_entry_count, collector.warnings)
        for turn_summary in summaries:
            writer.write_turn(
                _turn_entry_for(
                    snapshot,
                    header,
                    _summary_with_session_timing(turn_summary, collector),
                    session_fingerprint,
                    catalog_version,
                )
            )
            turn_entry_count += 1
        return _SessionScanOutcome(selected, tombstone, turn_entry_count, collector.warnings)


def _turn_entry_for(
    snapshot: SourceSnapshot,
    header: SessionHeader,
    turn_summary: TurnSummary,
    session_fingerprint: str,
    catalog_version: str,
) -> DiscoveryTurnEntry:
    """Build one sanitized turn entry through the Task 4A fingerprint seam."""
    replay_turn = build_replay_turn(snapshot, header, turn_summary)
    effective = build_effective_turn(replay_turn, IDENTITY_REDACTOR, _DISCOVERY_REDACTION_MODE)
    source_fp = source_fingerprint(effective, _DISCOVERY_REDACTION_MODE)
    identities: set[str] = set()
    warnings: list[str] = list(turn_summary.warnings)
    for record in replay_turn.iter_records():
        _collect_record_model_identity(record, identities)
        _collect_record_warnings(record, warnings)
    return DiscoveryTurnEntry(
        session_id=header.session_id,
        turn_index=turn_summary.turn_index,
        turn_key=turn_summary.turn_key,
        adapter_generation=header.adapter_generation,
        planned_action=PLANNED_ACTION_IMPORT,
        start_time_ns=turn_summary.start_time.nanoseconds if turn_summary.start_time is not None else None,
        end_time_ns=turn_summary.end_time.nanoseconds if turn_summary.end_time is not None else None,
        record_count=turn_summary.record_count,
        user_record_count=turn_summary.user_record_count,
        assistant_record_count=turn_summary.assistant_record_count,
        terminal_outcome=turn_summary.terminal_outcome,
        model_identities=frozenset(identities),
        warnings=tuple(dict.fromkeys(warnings)),
        session_context_fingerprint=session_fingerprint,
        source_fingerprint=source_fp,
        render_fingerprint=render_fingerprint(
            source_fp, ARTIFACT_SCHEMA_VERSION, MAPPING_VERSION, _IMPORTER_VERSION, _DISCOVERY_REDACTION_MODE
        ),
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        mapping_version=MAPPING_VERSION,
        importer_version=_IMPORTER_VERSION,
        redaction_mode=_DISCOVERY_REDACTION_MODE.value,
        catalog_version=catalog_version,
    )


def _collect_record_model_identity(record: SourceRecord, identities: set[str]) -> None:
    """Collect the model identity a record's decoded value carries, if any."""
    decoded = record.decoded_value
    if isinstance(decoded, Mapping):
        identity = extract_model_identity(decoded)
        if identity is not None:
            identities.add(identity)


def _collect_record_warnings(record: SourceRecord, warnings: list[str]) -> None:
    """Collect a record's safe warning codes."""
    warnings.extend(record.warnings)


class _SessionCollector:
    """Side-collecting observer for one session's streaming record pass.

    Wraps the session's record cursors in a generator so
    :func:`iter_turn_summaries` consumes each record exactly once while
    the collector accumulates the session's model identities (from the
    session default plus every record's decoded ``model`` field),
    record count, and deduplicated warning codes.
    """

    def __init__(self, header: SessionHeader) -> None:
        self.model_identities: set[str] = set()
        self.record_count = 0
        self._warnings: list[str] = []
        self._session_first_timestamp: TimestampValue | None = None
        self._session_last_timestamp: TimestampValue | None = None
        identity = extract_model_identity(header.model)
        if identity is not None:
            self.model_identities.add(identity)
        self._absorb_warnings(header.warnings)

    def observe(
        self,
        records: Iterable[SourceRecord],
        *,
        collect_model_identity: bool = True,
    ) -> Iterator[SourceRecord]:
        """Yield each record once, collecting its identity and warnings."""
        for record in records:
            self.record_count += 1
            if collect_model_identity:
                _collect_record_model_identity(record, self.model_identities)
            self._absorb_warnings(record.warnings)
            if record.source_table in SESSION_EVENT_TABLES:
                first = record.timestamp or record.completion_timestamp
                if self._session_first_timestamp is None and first is not None:
                    self._session_first_timestamp = first
                last = record.completion_timestamp or record.timestamp
                if last is not None:
                    self._session_last_timestamp = last
            yield record

    def drain(self, records: Iterable[SourceRecord]) -> None:
        """Observe records without forwarding them (session-scoped pass)."""
        for _record in self.observe(records, collect_model_identity=False):
            pass

    def record_warnings(self, warnings: Iterable[str]) -> None:
        """Absorb scan-time warning codes (for example turn-key fallbacks)."""
        self._absorb_warnings(warnings)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def _absorb_warnings(self, warnings: Iterable[str]) -> None:
        for warning in warnings:
            if warning not in self._warnings:
                self._warnings.append(warning)


def _summary_with_session_timing(summary: TurnSummary, collector: _SessionCollector) -> TurnSummary:
    """Include timed session events in the sanitized plan's root timing."""
    start = summary.start_time or collector._session_first_timestamp
    end = summary.end_time
    session_last = collector._session_last_timestamp
    if session_last is not None and (end is None or session_last.nanoseconds > end.nanoseconds):
        end = session_last
    if start == summary.start_time and end == summary.end_time:
        return summary
    return replace(summary, start_time=start, end_time=end)


def _selection_evidence(
    adapter: SourceAdapter,
    session_id: str,
    filters: SelectionFilters,
) -> SelectionEvidence | None:
    """Load optional pre-redaction evidence only for filters that need it."""
    if filters.directory is None and filters.model is None:
        return None
    capability = getattr(adapter, "selection_evidence", None)
    if not callable(capability):
        raise SourceCompatibilityError(
            "selection_evidence_unsupported",
            "the replay adapter cannot provide required pre-redaction selection evidence",
        )
    evidence = capability(session_id)
    if not isinstance(evidence, SelectionEvidence):
        raise SourceCompatibilityError(
            "selection_evidence_unavailable",
            "the replay adapter did not provide required pre-redaction selection evidence",
        )
    return evidence


class _SessionHeaderReplayTurn:
    """Header-only :class:`ReplayTurn` view for session-identity fingerprinting.

    Implements the protocol with the session's real header and a
    placeholder summary; every cursor and loader raises because this
    view exists only so :func:`session_context_fingerprint` — which
    reads the header alone — runs through the public
    :func:`build_effective_turn` seam for sessions with no logical
    turns (and uniformly for every session, since the session context
    fingerprint never touches records).
    """

    def __init__(self, header: SessionHeader) -> None:
        self._header = header

    @property
    def session_header(self) -> SessionHeader:
        return self._header

    @property
    def summary(self) -> TurnSummary:
        return TurnSummary(session_id=self._header.session_id, turn_index=0, turn_key="")

    def iter_records(self) -> Iterator[SourceRecord]:
        raise RuntimeError("the header-only replay view exposes no records")

    def iter_session_records(self) -> Iterator[SourceRecord]:
        raise RuntimeError("the header-only replay view exposes no records")

    def iter_unassigned_records(self) -> Iterator[SourceRecord]:
        raise RuntimeError("the header-only replay view exposes no records")

    def load_root_input(self, max_bytes: int) -> object | None:
        raise RuntimeError("the header-only replay view loads no root values")

    def load_root_output(self, max_bytes: int) -> object | None:
        raise RuntimeError("the header-only replay view loads no root values")


def _replay_snapshot(adapter: SourceAdapter) -> SourceSnapshot:
    """The normalized snapshot a replay adapter exposes for turn building.

    :func:`build_replay_turn` builds turns over the snapshot the replay
    adapter wraps, so discovery requires an adapter that exposes it
    (the registered :class:`~opencode_mlflow.source.replay.NormalizedSnapshotAdapter`
    does); anything else fails loudly with a safe compatibility error
    instead of guessing.
    """
    snapshot = getattr(adapter, "snapshot", None)
    if not isinstance(snapshot, SourceSnapshot):
        raise SourceCompatibilityError(
            "replay_snapshot_unavailable",
            "discovery requires a replay adapter over a normalized source snapshot",
        )
    return snapshot
