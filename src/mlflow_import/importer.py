"""Two-pass import orchestration: discovery, preflight, replay, writes, reporting.

:func:`run_import` wires every completed layer into one invocation. The
phase order is strict and binding:

1. configuration (already validated by :func:`mlflow_import.config.parse_config`);
2. live read-only source connection plus ``probe_schema``;
3. the registered :class:`~mlflow_import.source.snapshot.SourceExporter`
   for the probed generation;
4. normalized snapshot extraction, which closes its own live connection
   immediately after the catalogued extraction (the probing connection is
   closed before extraction, so no live SQLite connection remains during
   discovery or any MLflow work);
5. the :class:`~mlflow_import.source.replay.NormalizedSnapshotAdapter`
   replay snapshot, kept open across discovery and replay;
6. the sanitized discovery plan;
7. the global MLflow capability preflight (retirement plus named
   artifacts plus payload limits; ``needs_retirement=True`` because a
   normal write requires rollback after any partial create);
8. the non-blocking normal-run lock (skipped entirely for a dry run),
   acquired before experiment resolution and owned-trace lookup;
9. exact experiment resolution — read-only at lookup time in normal mode
   (a missing experiment has zero owned traces), with the
   create-on-missing deferred until after the replay fingerprint checks
   (still after the capability check and the lock, still before any
   write); a planned reference in dry-run;
10. importer-owned trace lookup;
11. replay fingerprint checks: every plan entry's source fingerprint is
    recomputed from the same immutable snapshot under the entry's
    recorded redaction mode, and a mismatch fails the whole invocation
    safely (exit ``2``, ``replay_fingerprint_mismatch``) before any
    mutation — discovery and replay read the same snapshot, so a mismatch
    is an internal inconsistency, never permission to import an
    unplanned payload;
12. reconciliation writes through Task 11's plan/apply, one selected
    session per disposable execution store, with per-item failure isolation;
13. the versioned report.

No MLflow mutation occurs before every pre-write check passes, and the
dry run reports predicted ``planned_action`` values without mutating
anything or acquiring the lock.

Redaction wiring: ``config.redact`` selects
``DeterministicRedactor(RedactionMode.REDACTED)`` as the snapshot's
redaction callback (the identity/verbatim redactor otherwise), applied
before temporary persistence, with a counting wrapper accumulating the
Task 7 report vocabulary (detector-category counts plus affected
field/record counts, per detection across decoded and raw positions;
one source row counts as at most one affected record across its storage
projections). The invocation's directory/model filters are also passed
to a pre-redaction evidence evaluator; only match booleans enter the
snapshot, never selector values or hashes.
The snapshot persists post-redaction values; discovery fingerprints run
under the fixed verbatim consistency tag (existing behavior), while the
ownership fingerprints the mapper computes carry the run's redaction
mode tag over the already-effective values.

Every seam is injected through :class:`ImportDependencies` so tests run
deterministically; :func:`build_dependencies` constructs the production
set. :class:`ImportDependencies` also constructs Task 11's
:class:`~mlflow_import.mlflow.reconciliation.ReconciliationDependencies`
for one run, binding the gateway, replay, per-session mapping contexts,
and the gateway-bound artifact uploader.
"""

from __future__ import annotations

import inspect
import sqlite3
import sys
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, MutableMapping
from contextlib import AbstractContextManager, ExitStack, closing, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from itertools import chain as iter_chain
from pathlib import Path
from typing import Any, Protocol, cast

from mlflow_import import __version__ as IMPORTER_VERSION
from mlflow_import import resources
from mlflow_import.config import ImportConfig
from mlflow_import.content.fingerprint import (
    artifact_snapshot_fingerprint,
    build_effective_turn,
    source_fingerprint,
)
from mlflow_import.content.redaction import DeterministicRedactor, RedactionReport
from mlflow_import.contracts import IDENTITY_REDACTOR, RedactionMode
from mlflow_import.errors import ConfigurationError, ImporterError, ItemImportError
from mlflow_import.locking import ImportLock, acquire_import_lock, resolve_lock_root
from mlflow_import.mlflow.capabilities import CapabilityReport, CapabilityRequest, require_capabilities
from mlflow_import.mlflow.gateway import (
    DEFAULT_UPLOAD_CHUNK_BYTES,
    ArtifactVerification,
    CreatedTrace,
    ExperimentRef,
    MlflowGateway,
    TraceCreateRequest,
    build_gateway,
)
from mlflow_import.mlflow.mapping import MappingContext, map_turn_to_trace
from mlflow_import.mlflow.reconciliation import (
    ACTION_FAILED,
    ACTION_IMPORTED,
    ACTION_RETIRED,
    ACTION_SKIPPED,
    ACTION_UPDATED,
    OwnedTraceCandidate,
    ReconciliationAction,
    ReconciliationDependencies,
    ReconciliationPlan,
    apply_reconciliation,
    plan_reconciliation,
    upload_artifact,
)
from mlflow_import.privacy import safe_client_diagnostics
from mlflow_import.reporting import REASON_DRY_RUN, ReportBuilder, TurnResult
from mlflow_import.retry import _BUDGET, RetryPolicy, remember_failure, retry_budget
from mlflow_import.source.candidates import candidate_session_ids
from mlflow_import.source.connection import open_live_connection
from mlflow_import.source.discovery import (
    ARTIFACT_SCHEMA_VERSION,
    MAPPING_VERSION,
    DiscoveryPlanReader,
    DiscoveryTurnEntry,
    SessionScanEntry,
    discover_sessions,
)
from mlflow_import.source.file_snapshot import create_file_snapshot
from mlflow_import.source.model import SchemaDescription, SessionHeader, SourceAdapter, TurnSummary
from mlflow_import.source.probe import probe_schema
from mlflow_import.source.registry import build_live_exporter
from mlflow_import.source.replay import NormalizedSnapshotAdapter, ReplayTurn
from mlflow_import.source.selection import SelectionEvidenceEvaluator, SelectionFilters
from mlflow_import.source.snapshot import (
    RedactionCallback,
    SourceExporter,
    SourceSnapshot,
    create_normalized_snapshot,
)
from mlflow_import.source.turns import build_replay_turn, indexed_turn_summary
from mlflow_import.storage import DiskList, DiskMap, Store, disk_collections

__all__ = [
    "BROAD_SELECTION_MESSAGE",
    "BROAD_SELECTION_WARNING",
    "DISCOVERY_PLAN_FILENAME",
    "ImportDependencies",
    "ImportResult",
    "REPLAY_FINGERPRINT_MISMATCH",
    "ArtifactUploader",
    "build_dependencies",
    "build_replay_index",
    "exit_status",
    "replay_turn",
    "run_import",
]

REPLAY_FINGERPRINT_MISMATCH = "replay_fingerprint_mismatch"
"""Safe code for a replayed turn whose source fingerprint diverged from its plan entry."""

REPLAY_FINGERPRINT_MISMATCH_MESSAGE = (
    "a replayed turn's source fingerprint does not match its discovery plan entry; "
    "the invocation is aborted before any write"
)

BROAD_SELECTION_WARNING = "broad_selection"
"""Warning-event code for an invocation with no optional selection filter."""

BROAD_SELECTION_MESSAGE = (
    "no selection filters were supplied, so every session in the source is in scope; "
    "run with --dry-run first to preview the planned actions"
)

DISCOVERY_PLAN_FILENAME = "discovery-plan.jsonl"
"""File name of the discovery plan inside the run's temporary directory."""

SNAPSHOT_PROGRESS_INTERVAL = 10_000
"""Record count between safe snapshot progress messages in DEBUG mode."""

_ARTIFACT_STAGING_DIRNAME = "artifact-staging"
"""Subdirectory of the run root where artifact uploads stage their streams."""

_IMPORT_FAILED = "import_failed"
_IMPORT_FAILED_MESSAGE = "the import run failed before completing"
_TEMP_DIRECTORY_FAILED = "temporary_directory_unavailable"
_TEMP_DIRECTORY_FAILED_MESSAGE = "the temporary run directory could not be prepared"
_SNAPSHOT_CLEANUP_WARNING = "temporary_snapshot_cleanup_failed"
_SNAPSHOT_CLEANUP_MESSAGE = "sensitive temporary data may remain because temporary snapshot cleanup failed"
_RUN_CLEANUP_WARNING = "temporary_run_cleanup_failed"
_RUN_CLEANUP_MESSAGE = "sensitive temporary data may remain because temporary run cleanup failed"

# ---------------------------------------------------------------------------
# Injectable seam types
# ---------------------------------------------------------------------------

SnapshotFactory4 = Callable[
    [Path, Path, SourceExporter, RedactionCallback],
    AbstractContextManager[SourceSnapshot],
]
SnapshotFactory5 = Callable[
    [Path, Path, SourceExporter, RedactionCallback, SelectionEvidenceEvaluator],
    AbstractContextManager[SourceSnapshot],
]
SnapshotFactory = SnapshotFactory4 | SnapshotFactory5
"""Builds the normalized snapshot for one source (see ``create_normalized_snapshot``)."""

ExporterFactory = Callable[[sqlite3.Connection, SchemaDescription], SourceExporter]
"""Builds the registered live exporter for a probed schema description."""

ReplayAdapterFactory = Callable[[SourceSnapshot], SourceAdapter]
"""Builds the replay adapter over one normalized snapshot."""

GatewayFactory = Callable[[bool], MlflowGateway]
"""Builds the MLflow gateway for one run mode (``dry_run`` flag)."""

LockFactory = Callable[[str, str], AbstractContextManager[ImportLock]]
"""Acquires the advisory import lock for (tracking endpoint, experiment name)."""

TempDirectoryFactory = Callable[[str], Path]
"""Creates one restrictive temporary directory for the run (caller removes it)."""

ReplayBuilder = Callable[[DiscoveryTurnEntry], ReplayTurn]
"""Replays one discovery turn entry over the run's snapshot."""

ReplayBuilderFactory = Callable[[SourceSnapshot], ReplayBuilder]
"""Builds the memoized entry-to-replay callable for one snapshot."""

ContextFactory = Callable[[str], MappingContext]
"""Builds the per-session mapping context for a session id."""

Mapper = Callable[[ReplayTurn, MappingContext], TraceCreateRequest]
"""Maps one replayed turn to its gateway trace-create request (pure)."""

ArtifactSnapshotFunction = Callable[[ReplayTurn, RedactionMode], str]
"""The replayed turn's pre-identity-map artifact-snapshot fingerprint."""

Clock = Callable[[], int]
"""The run's clock, in epoch nanoseconds."""


class ArtifactUploader(Protocol):
    """Streams and uploads one turn's artifact bound to a created trace.

    Matches :func:`mlflow_import.mlflow.reconciliation.upload_artifact`
    exactly (the gateway, staging directory, and chunk size are
    keyword-only), so the production default is the reconciliation
    uploader itself.
    """

    def __call__(
        self,
        replay: ReplayTurn,
        created: CreatedTrace,
        artifact_relative_path: str,
        redaction_mode: RedactionMode,
        *,
        gateway: MlflowGateway,
        temporary_directory: Path | None = ...,
        chunk_size: int = ...,
    ) -> ArtifactVerification: ...


@dataclass(frozen=True)
class ImportDependencies:
    """Every injectable seam of the two-pass import pipeline.

    The factories construct the source pipeline, gateway, and lock; the
    pure functions (replay builder, mapper, artifact-snapshot
    fingerprint, artifact uploader) are Task 11's reconciliation seams;
    the clock and retry policy keep runs deterministic under test.
    """

    snapshot_factory: SnapshotFactory
    exporter_factory: ExporterFactory
    replay_adapter_factory: ReplayAdapterFactory
    gateway_factory: GatewayFactory
    lock_factory: LockFactory
    temp_directory_factory: TempDirectoryFactory
    replay_builder: ReplayBuilderFactory
    upload_chunk_bytes: int
    mapper: Mapper
    artifact_snapshot: ArtifactSnapshotFunction
    artifact_uploader: ArtifactUploader
    clock: Clock
    retry_policy: RetryPolicy

    def reconciliation_dependencies(
        self,
        *,
        gateway: MlflowGateway,
        replay: ReplayBuilder,
        context_for: ContextFactory,
        redaction_mode: RedactionMode,
        staging_directory: Path | None,
    ) -> ReconciliationDependencies:
        """Construct Task 11's ``ReconciliationDependencies`` for one run.

        The mapper binds each turn's per-session mapping context; the
        artifact uploader binds this run's gateway, staging directory,
        and chunk size; the artifact-snapshot closure uses the RUN's
        redaction mode so it matches the mapped request's fingerprint
        and the artifact document itself. The entry's fixed VERBATIM
        mode is used only by the separate discovery-consistency guard.
        """

        def map_trace(turn: ReplayTurn) -> TraceCreateRequest:
            return self.mapper(turn, context_for(turn.session_header.session_id))

        def upload(
            replay_turn: ReplayTurn,
            created: CreatedTrace,
            artifact_relative_path: str,
            mode: RedactionMode,
        ) -> ArtifactVerification:
            return self.artifact_uploader(
                replay_turn,
                created,
                artifact_relative_path,
                mode,
                gateway=gateway,
                temporary_directory=staging_directory,
                chunk_size=self.upload_chunk_bytes,
            )

        def artifact_snapshot(turn: ReplayTurn) -> str:
            return self.artifact_snapshot(turn, redaction_mode)

        return ReconciliationDependencies(
            replay=replay,
            map_trace=map_trace,
            artifact_snapshot=artifact_snapshot,
            upload_artifact=upload,
            redaction_mode=redaction_mode,
        )


@dataclass(frozen=True)
class ImportResult:
    """One import invocation's outcome.

    ``report`` is the versioned result document (a mapping, schema
    ``mlflow-import.result.v1``); ``error`` is the safe pre-write
    failure when the invocation aborted (its ``exit_status`` decides the
    process status); ``failed_items`` counts failed session/turn items and
    distinct sensitive temporary-data cleanup failures.
    """

    dry_run: bool
    report: dict[str, object]
    error: ImporterError | None = None
    failed_items: int = 0
    _store: Store | None = field(default=None, repr=False, compare=False)

    def close(self) -> None:
        """Release lazy report storage after the caller finishes reading/rendering."""
        if self._store is not None:
            self._store.close()


def exit_status(result: ImportResult) -> int:
    """The process exit status for one result.

    ``0`` for a complete or no-op run, ``1`` when any item or temporary-data
    cleanup failed, ``2`` for an invalid or pre-write failure (configuration,
    schema, snapshot, capability, lock).
    """
    if result.error is not None:
        return result.error.exit_status
    if result.failed_items:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Replay over the normalized snapshot
# ---------------------------------------------------------------------------


def replay_turn(entry: DiscoveryTurnEntry, snapshot: SourceSnapshot) -> ReplayTurn:
    """Replay one discovery turn entry against its normalized snapshot.

    The single-turn convenience form of :func:`build_replay_index`: it
    rebuilds the snapshot's session-header and turn-summary index, so
    callers replaying many turns should prefer
    :func:`build_replay_index` (which memoizes the index once).
    """
    return _SessionReplayIndex(snapshot).replay(entry)


def build_replay_index(snapshot: SourceSnapshot) -> ReplayBuilder:
    """Build the bounded entry-to-replay callable the pipeline uses.

    Only the last requested session header is retained; turn summaries
    are searched as a stream. Every replayed turn opens fresh snapshot cursors
    through :func:`mlflow_import.source.turns.build_replay_turn`.
    """
    index = _SessionReplayIndex(snapshot)
    return index.replay


class _SessionReplayIndex:
    """Per-snapshot replay lookup without run-wide payload/summary caches.

    A plan entry whose session or turn is absent from the snapshot fails
    loudly with a safe error — the plan and the snapshot were produced
    from the same immutable extraction, so divergence is an internal
    inconsistency.
    """

    def __init__(self, snapshot: SourceSnapshot) -> None:
        self._snapshot = snapshot
        self._adapter = NormalizedSnapshotAdapter(snapshot)
        self._last_header: SessionHeader | None = None

    def replay(self, entry: DiscoveryTurnEntry) -> ReplayTurn:
        """Replay one plan entry's turn with fresh snapshot cursors."""
        header = self._header(entry.session_id)
        summary = self._summary(entry.session_id, entry.turn_key)
        return build_replay_turn(self._snapshot, header, summary)

    def _header(self, session_id: str) -> SessionHeader:
        header = self._last_header
        if header is None or header.session_id != session_id:
            header = self._adapter.session_header(session_id)
            self._last_header = header
        if header is None:
            raise ImporterError(
                "replay_session_unavailable",
                "a discovery plan session is absent from the normalized snapshot",
            )
        return header

    def _summary(self, session_id: str, turn_key: str) -> TurnSummary:
        summary = indexed_turn_summary(self._snapshot, session_id, turn_key)
        if summary is None:
            raise ImporterError(
                "replay_turn_unavailable",
                "a discovery plan turn is absent from the normalized snapshot",
            )
        return summary


def _verify_replayed_fingerprints(turns: Iterable[DiscoveryTurnEntry], replay: ReplayBuilder) -> None:
    """Reject any replayed source fingerprint that differs from its plan entry.

    The check recomputes each entry's source fingerprint from the same
    snapshot under the entry's RECORDED redaction mode (discovery hashes
    the already-effective replay values under the fixed verbatim tag and
    records that mode), so a mismatch means the snapshot changed mid-run
    — an internal inconsistency that fails the invocation safely before
    any write.
    """
    for entry in turns:
        replayed = replay(entry)
        try:
            mode = RedactionMode(entry.redaction_mode)
        except ValueError:
            mode = RedactionMode.VERBATIM
        if entry.catalog_version and replayed.session_header.catalog_version != entry.catalog_version:
            raise ImporterError(REPLAY_FINGERPRINT_MISMATCH, REPLAY_FINGERPRINT_MISMATCH_MESSAGE)
        effective = build_effective_turn(replayed, IDENTITY_REDACTOR, mode)
        if source_fingerprint(effective, mode) != entry.source_fingerprint:
            raise ImporterError(REPLAY_FINGERPRINT_MISMATCH, REPLAY_FINGERPRINT_MISMATCH_MESSAGE)


# ---------------------------------------------------------------------------
# Per-session mapping contexts
# ---------------------------------------------------------------------------


class _MappingContexts:
    """Lazily built per-session mapping contexts over one snapshot.

    Each context carries the render-contract versions, the run's
    redaction mode, the resolved experiment, the source database
    fingerprint, the negotiated inline payload limit, and the session's
    own adapter generation, logical-turn count, and message/part counts
    (streamed once per session on first access).
    """

    def __init__(
        self,
        *,
        adapter: SourceAdapter,
        scans: Iterable[SessionScanEntry],
        snapshot: SourceSnapshot,
        experiment_id: str,
        inline_limit_bytes: int | None,
        redaction_mode: RedactionMode,
        trace_max_bytes: int,
        trace_max_spans: int,
    ) -> None:
        self._adapter = adapter
        self._turn_counts = DiskMap((scan.session_id, scan.logical_turn_count) for scan in scans)
        self._snapshot = snapshot
        self._experiment_id = experiment_id
        self._inline_limit_bytes = inline_limit_bytes
        self._redaction_mode = redaction_mode
        self._trace_max_bytes = trace_max_bytes
        self._trace_max_spans = trace_max_spans
        self._cache: dict[str, MappingContext] = {}
        self._generations: dict[str, str] = {}

    def context_for(self, session_id: str) -> MappingContext:
        """The mapping context for one session (memoized per run)."""
        context = self._cache.get(session_id)
        if context is None:
            context = self._build(session_id)
            self._cache.clear()
            self._cache[session_id] = context
        return context

    def _build(self, session_id: str) -> MappingContext:
        message_count = 0
        part_count = 0
        for record in self._adapter.iter_records(session_id):
            if record.source_table == "part" or record.derived_child_ordinal is not None:
                part_count += 1
            else:
                message_count += 1
        return MappingContext(
            importer_version=IMPORTER_VERSION,
            mapping_version=MAPPING_VERSION,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            source_schema=self._generation(session_id),
            redaction_mode=self._redaction_mode,
            experiment_id=self._experiment_id,
            catalog_version=self._snapshot.schema_description.catalog_version,
            source_database_fingerprint=self._snapshot.fingerprint,
            inline_limit_bytes=self._inline_limit_bytes,
            trace_max_bytes=self._trace_max_bytes,
            trace_max_spans=self._trace_max_spans,
            logical_turn_count=self._turn_counts.get(session_id),
            source_message_count=message_count,
            source_part_count=part_count,
        )

    def _generation(self, session_id: str) -> str:
        if isinstance(self._adapter, NormalizedSnapshotAdapter):
            header = self._adapter.session_header(session_id)
            return header.adapter_generation if header is not None else ""
        return next(
            (
                header.adapter_generation
                for header in self._adapter.iter_session_headers()
                if header.session_id == session_id
            ),
            "",
        )


# ---------------------------------------------------------------------------
# Redaction counting
# ---------------------------------------------------------------------------


class _CountingRedaction:
    """Redaction callback that accumulates the deterministic redactor's counts.

    Wraps the mode-aware redactor's result-returning helper so the
    snapshot's redaction callback (applied to every persisted envelope
    value and permitted raw field) also produces the run-level
    :class:`RedactionReport` — detector-category counts plus affected
    field/record counts, exactly as recorded (per detection across
    decoded and raw positions). Verbatim mode reports zero counts.
    """

    def __init__(self, redactor: DeterministicRedactor) -> None:
        self._redactor = redactor
        self._report = RedactionReport()
        self._record_report: RedactionReport | None = None

    def __call__(self, value: object, protected_paths: Collection[str]) -> object:
        result = self._redactor.redact_value(value, protected_paths)
        if self._record_report is None:
            self._report = self._report.merge(result.report)
        else:
            self._record_report = self._record_report.merge(result.report)
        return result.value

    def begin_source_record(self) -> None:
        """Start counting one source row across all persisted projections."""
        if self._record_report is not None:
            raise RuntimeError("a redaction record scope is already active")
        self._record_report = RedactionReport()

    def end_source_record(self) -> None:
        """Merge one row's counts with an affected-record count of at most one."""
        if self._record_report is None:
            raise RuntimeError("no redaction record scope is active")
        report = self._record_report
        self._record_report = None
        normalized = RedactionReport(
            detector_counts=report.detector_counts,
            affected_field_count=report.affected_field_count,
            affected_record_count=int(report.affected_field_count > 0),
        )
        self._report = self._report.merge(normalized)

    @property
    def report(self) -> RedactionReport:
        return self._report


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def run_import(config: ImportConfig, dependencies: ImportDependencies) -> ImportResult:
    """Execute within one finite invocation retry/deadline budget."""
    from mlflow_import.compatibility import IdentityIndex, identity_index

    budget = None
    work_budget = None
    result = None
    try:
        with resources.temporary_storage(retain_result=True) as budget:
            with (
                retry_budget(deadline_seconds=config.timeout_seconds) as work_budget,
                safe_client_diagnostics(),
                disk_collections() as store,
                identity_index(IdentityIndex()),
            ):
                if config.__dict__.get("_discover_defaults", False):
                    result = _run_defaults(config, dependencies)
                else:
                    result = _run_import(config, dependencies)
                result.report["resources"] = {
                    "temporary_limit_bytes": budget.max_bytes,
                    "temporary_peak_bytes": budget.peak_bytes,
                    "sqlite_heap_limit_bytes": budget.sqlite_heap_bytes,
                    "temporary_sql_statements": budget.sql_statements,
                    "temporary_peak_entries": budget.peak_entries,
                }
                budget.retained = store.root
                return replace(result, _store=store)
    except resources.TemporaryStorageError as error:
        # A full result store cannot be used to report its own exhaustion. This
        # fixed-size emergency envelope allocates no more temporary storage.
        return _storage_failure_result(config, (work_budget.primary_error if work_budget else None) or error, budget)
    except ConfigurationError as error:
        if result is None and work_budget is not None and work_budget.primary_error is not None:
            return _storage_failure_result(config, work_budget.primary_error, budget)
        if error.code != "temporary_storage_cleanup_failed" or result is None:
            raise
        summary = cast(dict[str, Any], result.report["summary"])
        warnings = cast(list[dict[str, str]], summary["warnings"])
        failed = result.failed_items
        if not any(item["code"] in {_RUN_CLEANUP_WARNING, "temporary_storage_cleanup_failed"} for item in warnings):
            warnings.append({"code": error.code, "message": error.safe_message})
            summary["failed_items"] += 1
            failed += 1
        return replace(result, _store=store, failed_items=failed)
    except ImporterError as error:
        return _storage_failure_result(config, (work_budget.primary_error if work_budget else None) or error, budget)


def _storage_failure_result(config: ImportConfig, error: ImporterError, budget: Any) -> ImportResult:
    phase2 = bool(budget and budget.phase2)
    # Do not retain failed statement/source frames (and their decoded records)
    # through an exception traceback on the returned lazy API result.
    original = error
    error = ImporterError(error.code, error.safe_message, exit_status=1 if phase2 else 2)
    error.diagnostics = dict(original.diagnostics)
    summary: dict[str, Any] = {
        key: None
        for key in (
            "selected_sessions",
            "selected_turns",
            "imported_turns",
            "updated_turns",
            "skipped_turns",
            "unchanged_turns",
            "failed_turns",
            "turns_imported",
            "turns_updated",
            "turns_skipped",
            "turns_failed",
            "failed_sessions",
            "failed_retirements",
            "retired_traces",
            "unexamined_owned_traces",
        )
    }
    summary.update(
        failed_items=None,
        reporting_failures=1,
        incomplete=True,
        counts_complete=False,
        recovery_pending=phase2,
        warnings=[],
    )
    if config.dry_run:
        summary["planned"] = {"imported": None, "updated": None, "retired": None}
    return ImportResult(
        config.dry_run,
        {
            "schema_version": "mlflow-import.result.v1",
            "dry_run": config.dry_run,
            "experiment": {"name": config.experiment_name, "id": None, "status": "unresolved"},
            "source": {
                "schema_generation": None,
                "fingerprint": None,
                "sessions_scanned": None,
                "sessions_selected": None,
                "logical_turn_count": None,
                "record_count": None,
            },
            "redaction": {
                "mode": "redacted" if config.redact else "verbatim",
                "detector_counts": {},
                "affected_field_count": None,
                "affected_record_count": None,
            },
            "summary": summary,
            "sessions": [],
            "errors": [{"code": error.code, "message": error.safe_message, "diagnostics": error.diagnostics}],
            "resources": {
                "temporary_limit_bytes": budget.max_bytes if budget else 0,
                "temporary_peak_bytes": budget.peak_bytes if budget else 0,
            },
        },
        error=error,
        failed_items=int(phase2),
    )


def _register_identities(snapshot: SourceSnapshot, harness: str) -> None:
    from mlflow_import.compatibility import _IDENTITIES

    index = _IDENTITIES.get()
    if index is not None:
        for (native,) in snapshot.connection.execute("SELECT id FROM source_identity_headers"):
            index.add(harness, snapshot.source_namespace or snapshot.fingerprint, native)


def _run_defaults(config: ImportConfig, dependencies: ImportDependencies) -> ImportResult:
    """Freeze every available adapter-owned root before the first target write.

    Snapshots remain owned by the outer stack throughout sequential consumption.
    Executable ordering, identity resolution and result aggregation use private
    disk catalogs; source-root handles are bounded by the adapter registry.
    """
    from mlflow_import.sources import ADAPTERS

    reports: DiskList[dict[str, object]] = DiskList()
    first_error: ImporterError | None = None
    failed_items = 0
    cleanup_failed = False
    mode = RedactionMode.REDACTED if config.redact else RedactionMode.VERBATIM
    try:
        with ExitStack() as stack:
            root = dependencies.temp_directory_factory("mlflow-import-defaults-").resolve()
            if budget := resources.current_storage():
                budget.adopt(root)

            def cleanup_sources() -> None:
                nonlocal cleanup_failed
                cleanup_failed = _cleanup_run_root(root)

            stack.callback(cleanup_sources)
            frozen_sources: list[tuple[ImportConfig, SourceSnapshot, _CountingRedaction]] = []
            filters = SelectionFilters(
                config.from_time,
                config.to_time,
                config.directory,
                config.model,
                config.min_user_turns,
                from_time_ns=config.from_time_ns,
                to_time_ns=config.to_time_ns,
            )
            for name, adapter in ADAPTERS.items():
                if config.harness is not None and config.harness != name:
                    continue
                for candidate in adapter.default_sources():
                    path = candidate.path
                    selected = replace(config, db_path=path)
                    for key in ("_session", "_on_conflict", "_from_time_ns", "_to_time_ns"):
                        object.__setattr__(selected, key, config.__dict__.get(key))
                    object.__setattr__(selected, "_harness", name)
                    object.__setattr__(selected, "_source", None)
                    counting = _CountingRedaction(DeterministicRedactor(mode))
                    if name == "opencode":
                        with open_live_connection(path) as live:
                            description = probe_schema(live.connection)
                            exporter = dependencies.exporter_factory(live.connection, description)
                    else:
                        exporter = adapter.exporter(path)
                    frozen = stack.enter_context(
                        _default_snapshot_factory(
                            path,
                            root,
                            exporter,
                            counting,
                            SelectionEvidenceEvaluator(filters),
                            session_id=config.session,
                        )
                    )
                    frozen_sources.append((selected, frozen, counting))
            if not frozen_sources:
                raise ConfigurationError("source_unavailable", "no supported default source roots are available")
            # Complete the identity inventory before resolving any bare alias.
            # One root's unique native ID may collide with a later root.
            for selected, frozen, _ in frozen_sources:
                _register_identities(frozen, selected.harness or "opencode")
            store = reports.store
            schedule = "schedule_" + store.table()
            store.db.execute(
                f"CREATE TABLE {schedule}(root INTEGER,session TEXT,created INTEGER,harness TEXT,namespace TEXT)"
            )
            readers: list[DiscoveryPlanReader] = []
            for root_index, (selected, frozen, _) in enumerate(frozen_sources):
                plan_path = root / f"discovery-{root_index}.jsonl"
                discover_sessions(dependencies.replay_adapter_factory(frozen), filters, plan_path, config.session)
                reader = DiscoveryPlanReader(plan_path).freeze()
                readers.append(reader)
                store.db.executemany(
                    f"INSERT INTO {schedule} VALUES(?,?,?,?,?)",
                    (
                        (root_index, scan.session_id, scan.creation_time_ns, selected.harness, frozen.source_namespace)
                        for scan in reader.iter_session_scans()
                    ),
                )

            def append_report(result: ImportResult, root_index: int) -> None:
                nonlocal first_error, failed_items
                selected, frozen, _ = frozen_sources[root_index]
                sessions = cast(list[dict[str, Any]], result.report["sessions"])
                for index, session in enumerate(sessions):
                    session["harness"] = selected.harness
                    session["source_namespace"] = frozen.source_namespace
                    sessions[index] = session
                reports.append(result.report)
                first_error = first_error or result.error
                failed_items += result.failed_items

            # All roots pass the read-only conflict/recovery/projection phase
            # before the first target mutation. Hold one writer lock across it
            # and the globally ordered write phase.
            gateway = dependencies.gateway_factory(config.dry_run)
            with (
                nullcontext()
                if config.dry_run
                else dependencies.lock_factory(gateway.tracking_uri, config.experiment_name)
            ) as held_lock:
                for root_index, (selected, frozen, counting) in enumerate(frozen_sources):
                    preview = replace(selected, dry_run=True)
                    for key, value in selected.__dict__.items():
                        if key.startswith("_"):
                            object.__setattr__(preview, key, value)
                    append_report(_run_import(preview, dependencies, frozen, counting, readers[root_index]), root_index)
                if not config.dry_run and first_error is None and failed_items == 0:
                    reports.clear()
                    # Authorize this invocation once, under the writer lock,
                    # after every read-only preview and before any mutation.
                    normal_capability = gateway.probe(
                        CapabilityRequest(needs_retirement=True, needs_named_artifacts=True), dry_run=False
                    )
                    require_capabilities(normal_capability)
                    active_dependencies = replace(
                        dependencies,
                        gateway_factory=lambda dry_run: gateway,
                        lock_factory=lambda tracking_uri, experiment_name: nullcontext(cast(ImportLock, held_lock)),
                    )
                    reported_roots: set[int] = set()
                    for root_index, session_id in store.db.execute(
                        f"SELECT root,session FROM {schedule} "
                        "ORDER BY created IS NULL,created,session,harness,namespace"
                    ):
                        selected, frozen, counting = frozen_sources[root_index]
                        result = _run_import(
                            selected,
                            active_dependencies,
                            frozen,
                            counting,
                            readers[root_index].for_session(session_id),
                            normal_capability=normal_capability,
                        )
                        if root_index in reported_roots:
                            redaction = cast(dict[str, Any], result.report["redaction"])
                            for key, value in redaction.items():
                                if isinstance(value, int):
                                    redaction[key] = 0
                                elif isinstance(value, dict):
                                    redaction[key] = {name: 0 for name in value}
                        reported_roots.add(root_index)
                        append_report(result, root_index)
                        if result.error is not None:
                            break
                elif config.dry_run:
                    # The same schedule determines result order in preview mode.
                    sessions_by_key: DiskMap[tuple[str, str, str], dict[str, Any]] = DiskMap()
                    for report in reports:
                        for session in cast(list[dict[str, Any]], report["sessions"]):
                            sessions_by_key[
                                (session["harness"], session["source_namespace"], session["session_id"])
                            ] = session
                    ordered: DiskList[dict[str, Any]] = DiskList()
                    for harness, namespace, session_id in store.db.execute(
                        f"SELECT harness,namespace,session FROM {schedule} "
                        "ORDER BY created IS NULL,created,session,harness,namespace"
                    ):
                        ordered.append(sessions_by_key[(harness, namespace, session_id)])
                    # Attach after numeric aggregation below; keep root reports
                    # for their source/redaction and selection summaries.
                    ordered_sessions = ordered
    except Exception as failure:
        error = (
            failure
            if isinstance(failure, ImporterError)
            else ConfigurationError("source_discovery_failed", "default source discovery could not be completed")
        )
        error = remember_failure(first_error or error)
        if work := _BUDGET.get():
            work.phase = "error_reporting"
        builder = ReportBuilder(dry_run=config.dry_run, experiment_name=config.experiment_name, redaction_mode=mode)
        builder.add_error(error.code, error.safe_message)
        if cleanup_failed:
            _record_run_cleanup_failure(builder)
        return ImportResult(
            dry_run=config.dry_run, report=builder.build(), error=error, failed_items=int(cleanup_failed)
        )
    if not reports:
        builder = ReportBuilder(dry_run=config.dry_run, experiment_name=config.experiment_name, redaction_mode=mode)
        reports.append(builder.build())
    report = dict(reports[0])
    report["dry_run"] = config.dry_run
    for section in ("summary", "source", "redaction"):
        merged = dict(cast(dict[str, Any], report[section]))
        for following in reports[1:]:
            _merge_report_counts(merged, cast(dict[str, Any], following[section]))
        report[section] = merged
    sources: DiskMap[tuple[str, str], dict[str, Any]] = DiskMap()
    for following in reports:
        source = cast(dict[str, Any], following["source"])
        source_key = (str(source.get("schema_generation")), str(source.get("fingerprint")))
        if source_key not in sources:
            sources[source_key] = source
        else:
            merged_source = sources[source_key]
            _merge_report_counts(merged_source, source)
            sources[source_key] = merged_source
    report["sources"] = DiskList(sources.values())
    report["sessions"] = (
        ordered_sessions
        if config.dry_run
        else DiskList(session for following in reports for session in cast(list[Any], following["sessions"]))
    )
    report["errors"] = DiskList(error for following in reports for error in cast(list[Any], following["errors"]))
    if cleanup_failed:
        summary = cast(dict[str, Any], report["summary"])
        summary["failed_items"] += 1
        summary["warnings"].append({"code": _RUN_CLEANUP_WARNING, "message": _RUN_CLEANUP_MESSAGE})
    return ImportResult(
        dry_run=config.dry_run,
        report=report,
        error=first_error,
        failed_items=failed_items + int(cleanup_failed),
    )


def _merge_report_counts(target: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    """Merge nested counters without dropping earlier roots' detector/coverage keys."""
    for key, value in incoming.items():
        if isinstance(value, int) and not isinstance(value, bool) and key != "generated_ns":
            target[key] = target.get(key, 0) + value
        elif isinstance(value, list):
            target[key] = DiskList(iter_chain(target.get(key, []), value))
        elif isinstance(value, dict):
            nested = dict(target.get(key, {}))
            _merge_report_counts(nested, value)
            target[key] = nested
        else:
            target.setdefault(key, value)


def _run_import(
    config: ImportConfig,
    dependencies: ImportDependencies,
    frozen: SourceSnapshot | None = None,
    counting_override: _CountingRedaction | None = None,
    reader_override: DiscoveryPlanReader | None = None,
    *,
    normal_capability: CapabilityReport | None = None,
) -> ImportResult:
    """Run the two-pass import pipeline and return its versioned result.

    Pre-write failures (configuration, schema, snapshot, capability,
    lock, replay consistency) are caught and returned as a failure
    result carrying the safe error in the report — the caller renders it
    and exits with :func:`exit_status`. Session/turn write failures are
    isolated per item by the reconciliation layer and reported as failed
    items (exit ``1``) while later items continue.
    """
    redaction_mode = RedactionMode.REDACTED if config.redact else RedactionMode.VERBATIM
    counting = counting_override or _CountingRedaction(DeterministicRedactor(redaction_mode))
    builder = ReportBuilder(
        dry_run=config.dry_run,
        experiment_name=config.experiment_name,
        redaction_mode=redaction_mode,
        generated_ns=dependencies.clock(),
    )
    filters = SelectionFilters(
        from_time=config.from_time,
        to_time=config.to_time,
        directory=config.directory,
        model=config.model,
        min_user_turns=config.min_user_turns,
        from_time_ns=config.from_time_ns,
        to_time_ns=config.to_time_ns,
    )
    if not _any_optional_filter(filters, config.session):
        builder.add_warning(BROAD_SELECTION_WARNING, BROAD_SELECTION_MESSAGE)
    run_root: Path | None = None
    result: ImportResult | None = None
    try:
        try:
            run_root = dependencies.temp_directory_factory("mlflow-import-run-").resolve()
            if budget := resources.current_storage():
                budget.adopt(run_root)
        except resources.TemporaryStorageError:
            raise
        except Exception:
            temp_failure = ConfigurationError(_TEMP_DIRECTORY_FAILED, _TEMP_DIRECTORY_FAILED_MESSAGE)
            remember_failure(temp_failure)
            builder.add_error(temp_failure.code, temp_failure.safe_message)
            result = ImportResult(
                dry_run=config.dry_run,
                report=builder.build(),
                error=temp_failure,
                failed_items=0,
            )
        else:
            assert run_root is not None
            result = _run_pipeline(
                config,
                dependencies,
                builder,
                counting,
                filters,
                run_root,
                frozen,
                reader_override,
                normal_capability=normal_capability,
            )
    except ImporterError as error:
        remember_failure(error)
        if work := _BUDGET.get():
            work.phase = "error_reporting"
        _record_snapshot_cleanup_failure(builder, error)
        builder.add_error(error.code, error.safe_message)
        result = ImportResult(
            dry_run=config.dry_run,
            report=builder.build(),
            error=error,
            failed_items=0,
        )
    except Exception:
        # An unexpected failure is sanitized: no value, path, or content
        # from the underlying exception reaches the report.
        unexpected_failure = ImporterError(_IMPORT_FAILED, _IMPORT_FAILED_MESSAGE)
        remember_failure(unexpected_failure)
        builder.add_error(unexpected_failure.code, unexpected_failure.safe_message)
        result = ImportResult(
            dry_run=config.dry_run,
            report=builder.build(),
            error=unexpected_failure,
            failed_items=0,
        )
    finally:
        if run_root is not None and _cleanup_run_root(run_root):
            _record_run_cleanup_failure(builder)
    assert result is not None
    return _finalize_result(result, builder)


def _run_pipeline(
    config: ImportConfig,
    dependencies: ImportDependencies,
    builder: ReportBuilder,
    counting: _CountingRedaction,
    filters: SelectionFilters,
    run_root: Path,
    frozen: SourceSnapshot | None = None,
    reader_override: DiscoveryPlanReader | None = None,
    *,
    normal_capability: CapabilityReport | None = None,
) -> ImportResult:
    """Phases 1-7: source, snapshot, discovery, and the global preflight."""
    redaction_mode = RedactionMode.REDACTED if config.redact else RedactionMode.VERBATIM

    # Phases 2-3: one live read-only connection for probing, the
    # registered exporter bound to it, then the connection closes before
    # extraction (the snapshot driver opens and closes its own).
    exporter: SourceExporter
    if frozen is not None:
        description = frozen.schema_description
        exporter = cast(SourceExporter, None)
        candidate_ids = None
    elif config.harness in {"claude", "codex", "pi", "gemini"}:
        from mlflow_import.sources import get_adapter

        exporter = get_adapter(config.harness).exporter(config.db_path)
        description = exporter.inspect()
        candidate_ids = None
    else:
        with open_live_connection(config.db_path) as live:
            description = probe_schema(live.connection)
            exporter = dependencies.exporter_factory(live.connection, description)
            candidate_ids = candidate_session_ids(live.connection, description, filters, config.session)

    snapshot_started = _debug_phase_started(config, "snapshot")
    with (
        nullcontext(frozen)
        if frozen is not None
        else _snapshot_reporting_context(
            dependencies.snapshot_factory,
            config.db_path,
            run_root,
            exporter,
            counting,
            SelectionEvidenceEvaluator(filters),
            session_id=config.session,
            session_ids=candidate_ids,
            progress_callback=_snapshot_progress_callback(config),
            require_selection_evidence=filters.directory is not None or filters.model is not None,
            builder=builder,
        )
    ) as snapshot:
        if reader_override is None:
            _register_identities(snapshot, config.harness or "opencode")
        _debug_phase_completed(
            config,
            "snapshot",
            snapshot_started,
            sessions=snapshot.stats.session_count,
            records=snapshot.stats.record_count,
        )
        builder.set_source(schema_generation=description.generation, fingerprint=snapshot.fingerprint)
        builder.set_redaction_report(counting.report)
        for warning_code in snapshot.export_warnings:
            builder.add_warning(warning_code, "the source exporter reported a safe warning")
        for warning_code in snapshot.schema_description.warnings:
            builder.add_warning(warning_code, "the source schema probe reported a safe warning")

        # Phase 5-6: the replay snapshot and the sanitized discovery plan.
        adapter = dependencies.replay_adapter_factory(snapshot)
        plan_path = run_root / DISCOVERY_PLAN_FILENAME
        discovery_started = _debug_phase_started(config, "discovery")
        if reader_override is None:
            stats = discover_sessions(adapter, filters, plan_path, config.session)
            reader = DiscoveryPlanReader(plan_path).freeze()
        else:
            from mlflow_import.source.discovery import DiscoveryStats

            reader = reader_override
            scans = list(reader.iter_session_scans())  # exactly one session in an ordered view
            stats = DiscoveryStats(
                len(scans),
                sum(scan.selected for scan in scans),
                sum(1 for _ in reader.iter_turns()),
                sum(scan.tombstone for scan in scans),
                0,
                0,
                plan_path,
            )
        _debug_phase_completed(
            config,
            "discovery",
            discovery_started,
            sessions_scanned=stats.session_scan_count,
            sessions_selected=stats.selected_session_count,
            turns=stats.turn_entry_count,
        )
        builder.set_source_counts(
            sessions_scanned=stats.session_scan_count, sessions_selected=stats.selected_session_count
        )
        for scan in reader.iter_session_scans():
            builder.add_session_scan(
                scan.session_id,
                selected=scan.selected,
                logical_turn_count=scan.logical_turn_count,
                record_count=scan.record_count,
                warnings=scan.warnings,
            )

        # Phase 7: the global MLflow capability preflight, after discovery
        # and before any mutation. Retirement and named artifacts are
        # always required: a normal write needs rollback after any partial
        # create, and every imported turn uploads its artifact.
        gateway = dependencies.gateway_factory(config.dry_run)
        capability_started = _debug_phase_started(config, "capability preflight")
        # Only the default-source orchestrator passes a normal authorization,
        # scoped to its current invocation, gateway and held writer lock.
        # A preview report must never authorize normal writes.
        capability = normal_capability
        if config.dry_run or capability is None or capability.dry_run:
            capability = gateway.probe(
                CapabilityRequest(needs_retirement=True, needs_named_artifacts=True), dry_run=config.dry_run
            )
        require_capabilities(capability)
        _debug_phase_completed(config, "capability preflight", capability_started)

        pending_excluded = 0
        if isinstance(gateway, MlflowGateway) and (reader_override is None or reader_override._session_id is None):
            existing = _resolve_existing_experiment(gateway, config.experiment_name)
            if existing is not None and existing.experiment_id is not None:
                namespace = snapshot.source_namespace or snapshot.fingerprint
                harness = config.harness or "opencode"
                selected_ids = DiskMap((scan.session_id, True) for scan in reader.iter_session_scans() if scan.selected)
                for session_id in gateway.pending_sessions(
                    existing.experiment_id, f"mlflow-import/v2/{harness}/{namespace}"
                ):
                    if config.session is not None and config.session != session_id:
                        continue
                    if session_id not in selected_ids:
                        pending_excluded += 1
                        builder.add_session_error(session_id, "recovery_pending")
                if pending_excluded:
                    builder.add_error(
                        "recovery_pending",
                        "unfinished attempts require relaxing time, directory, "
                        "model or minimum-turn filters for the reported sessions",
                    )

        if config.dry_run:
            result = _run_dry(config, dependencies, builder, gateway, snapshot, reader)
        else:
            # Phase 8: the normal-run lock, acquired before experiment
            # resolution and owned-trace lookup (a dry run acquires nothing).
            with dependencies.lock_factory(gateway.tracking_uri, config.experiment_name):
                result = _run_normal(
                    config,
                    dependencies,
                    builder,
                    gateway,
                    snapshot,
                    reader,
                    capability=capability,
                    redaction_mode=redaction_mode,
                    run_root=run_root,
                )

    if pending_excluded:
        result = replace(result, failed_items=result.failed_items + pending_excluded, report=builder.build())
    return result


def _debug_phase_started(config: ImportConfig, phase: str) -> float | None:
    """Emit a safe DEBUG phase-start event and return its monotonic start time."""
    if work := _BUDGET.get():
        work.phase = phase
    if config.log_level != "DEBUG":
        return None
    _write_debug(f"mlflow-import: phase {phase} started")
    return time.perf_counter()


def _debug_phase_completed(
    config: ImportConfig,
    phase: str,
    started: float | None,
    **counts: int,
) -> None:
    """Emit elapsed phase time and optional safe aggregate counters."""
    if started is None:
        return
    elapsed = time.perf_counter() - started
    details = " ".join(f"{name}={value}" for name, value in counts.items())
    suffix = f" {details}" if details else ""
    _write_debug(f"mlflow-import: phase {phase} complete elapsed_s={elapsed:.3f}{suffix}")


def _write_debug(message: str) -> None:
    """Write best-effort diagnostics that cannot change the import outcome."""
    try:
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def _snapshot_progress_callback(config: ImportConfig) -> Callable[[int, int], None] | None:
    """Create a throttled, path-free callback for committed snapshot progress."""
    if config.log_level != "DEBUG":
        return None
    last_reported_records = 0

    def report(sessions: int, records: int) -> None:
        nonlocal last_reported_records
        if records - last_reported_records < SNAPSHOT_PROGRESS_INTERVAL:
            return
        _write_debug(f"mlflow-import: phase snapshot progress sessions={sessions} records={records}")
        last_reported_records = records

    return report


def _run_dry(
    config: ImportConfig,
    dependencies: ImportDependencies,
    builder: ReportBuilder,
    gateway: MlflowGateway,
    snapshot: SourceSnapshot,
    reader: DiscoveryPlanReader,
) -> ImportResult:
    """The dry-run tail: resolve, look up, verify fingerprints, and report.

    Every turn is reported as ``skipped`` with reason ``dry_run`` and a
    predicted ``planned_action`` from the pure reconciliation planner —
    imports predict ``imported``, updates predict ``updated`` (the final
    unchanged-versus-replace decision waits for verification, which a
    dry run does not perform), and stale-turn or tombstone retirements
    predict ``retired`` with ``source_present=false``. Pre-populated
    ownership conflicts remain failed and unexamined rather than becoming
    fresh import predictions.
    """
    planning_started = _debug_phase_started(config, "planning and owned-trace lookup")
    experiment = _resolve_existing_experiment(gateway, config.experiment_name)
    replay = dependencies.replay_builder(snapshot)
    pricing_basis_for = (
        _prepare_pricing_bases(config, gateway, experiment, snapshot, reader, snapshot.snapshot_path.parent)
        if isinstance(gateway, MlflowGateway)
        else None
    )
    _preflight_trace_envelopes(config, dependencies, snapshot, reader, replay, experiment, pricing_basis_for)
    if experiment is None:
        experiment = gateway.resolve_experiment(config.experiment_name, create=True)
    _record_experiment(builder, experiment)
    plan = plan_reconciliation(
        reader.iter_session_scans(), reader.iter_turns(), _owned_candidates(gateway, experiment, snapshot, reader)
    )
    builder.set_unexamined_owned_traces(len(plan.unexamined_owned_traces))
    _debug_phase_completed(config, "planning and owned-trace lookup", planning_started, turns=len(plan.results))

    # The replay fingerprint checks still run: a dry run validates.
    replay_started = _debug_phase_started(config, "replay fingerprint validation")
    _verify_replayed_fingerprints(reader.iter_turns(), dependencies.replay_builder(snapshot))
    _debug_phase_completed(config, "replay fingerprint validation", replay_started, turns=len(plan.results))

    failed_items = 0
    for result in plan.results:
        entry = result.turn
        if entry is None:
            continue
        if result.action == ACTION_FAILED:
            failed_items += 1
        builder.add_turn(
            TurnResult(
                session_id=entry.session_id,
                turn_key=entry.turn_key,
                turn_index=entry.turn_index,
                action=result.action,
                reason=result.reason,
                trace_id=result.trace_id,
                error_code=result.error_code,
                affected_trace_ids=result.affected_trace_ids,
                retired_trace_ids=result.retired_trace_ids,
                conflicting_trace_ids=result.conflicting_trace_ids,
                warnings=entry.warnings,
            )
        )

    for action in plan.imports:
        entry = action.turn
        assert entry is not None
        builder.add_turn(
            TurnResult(
                session_id=entry.session_id,
                turn_key=entry.turn_key,
                turn_index=entry.turn_index,
                action=ACTION_SKIPPED,
                reason=REASON_DRY_RUN,
                planned_action=ACTION_IMPORTED,
                warnings=entry.warnings,
            )
        )
    for action in plan.updates:
        entry = action.turn
        assert entry is not None
        builder.add_turn(
            TurnResult(
                session_id=entry.session_id,
                turn_key=entry.turn_key,
                turn_index=entry.turn_index,
                action=ACTION_SKIPPED,
                reason=REASON_DRY_RUN,
                planned_action=ACTION_UPDATED,
                warnings=entry.warnings,
            )
        )
    for action in plan.retirements:
        candidate = action.trace
        assert candidate is not None
        builder.add_turn(
            TurnResult(
                session_id=candidate.session_id,
                turn_key=candidate.turn_key,
                turn_index=candidate.turn_index,
                action=ACTION_SKIPPED,
                reason=REASON_DRY_RUN,
                planned_action=ACTION_RETIRED,
                trace_id=candidate.trace_id,
                source_present=False,
                warnings=candidate.warnings,
            )
        )
    # Dry-run counts are actual effects, not predictions. Planned stale
    # retirements remain in ``summary.planned`` and on each turn's
    # ``planned_action`` field.
    builder.set_retired_traces(0)
    return ImportResult(dry_run=True, report=builder.build(), failed_items=failed_items)


def _run_normal(
    config: ImportConfig,
    dependencies: ImportDependencies,
    builder: ReportBuilder,
    gateway: MlflowGateway,
    snapshot: SourceSnapshot,
    reader: DiscoveryPlanReader,
    *,
    capability: CapabilityReport,
    redaction_mode: RedactionMode,
    run_root: Path,
) -> ImportResult:
    """The normal-run tail: resolve read-only, look up, verify, create, write, report."""
    # Phase 9 (read-only): exact experiment resolution for the owned-trace
    # lookup. A missing experiment has zero owned traces, so its lookup is
    # trivially empty and the creation is deferred: no MLflow mutation may
    # happen before every pre-write check has passed.
    planning_started = _debug_phase_started(config, "planning and owned-trace lookup")
    experiment = _resolve_existing_experiment(gateway, config.experiment_name)
    plan = plan_reconciliation(
        reader.iter_session_scans(), reader.iter_turns(), _owned_candidates(gateway, experiment, snapshot, reader)
    )
    builder.set_unexamined_owned_traces(len(plan.unexamined_owned_traces))
    _debug_phase_completed(config, "planning and owned-trace lookup", planning_started, turns=len(plan.results))

    # Phase 11: every replay fingerprint check passes before any mutation.
    replay = dependencies.replay_builder(snapshot)
    pricing_basis_for = (
        _prepare_pricing_bases(config, gateway, experiment, snapshot, reader, run_root)
        if isinstance(gateway, MlflowGateway)
        else None
    )
    replay_started = _debug_phase_started(config, "replay fingerprint validation")
    _verify_replayed_fingerprints(reader.iter_turns(), replay)
    _debug_phase_completed(config, "replay fingerprint validation", replay_started, turns=len(plan.results))

    _preflight_trace_envelopes(config, dependencies, snapshot, reader, replay, experiment, pricing_basis_for)

    # Prepare every local write dependency before the deferred experiment
    # creation. A local staging failure is a pre-write failure and must not
    # leave a newly-created experiment behind.
    staging = _prepare_staging_directory(run_root)

    # Phase 9 (create-on-missing): the deferred creation — still after the
    # capability check and the lock, still before any write.
    if experiment is None:
        experiment = gateway.resolve_experiment(config.experiment_name, create=True)
    _record_experiment(builder, experiment)
    assert experiment.experiment_id is not None, "a normal-mode resolution always yields an experiment id"

    # Phase 12: the complete selection and read-only checks are now frozen.
    # Partition once on disk, then run each session in a disposable store.
    # Only sanitized report DTOs and retirement IDs escape a session's scope.
    sessions = _partition_reconciliation_plan(plan)
    retired_ids: DiskMap[str, bool] = DiskMap()
    failed_items = 0
    reconciliation_started = _debug_phase_started(config, "reconciliation")
    for session_id, session_plan in sessions.items():
        with closing(Store()) as scratch, disk_collections(scratch):
            failed_items += _import_session(
                config,
                dependencies,
                builder,
                gateway,
                snapshot,
                reader.for_session(session_id),
                session_plan,
                experiment_id=experiment.experiment_id,
                capability=capability,
                redaction_mode=redaction_mode,
                staging=staging,
                pricing_basis_for=pricing_basis_for,
                retired_ids=retired_ids,
            )
    _debug_phase_completed(
        config,
        "reconciliation",
        reconciliation_started,
        sessions=len(sessions),
    )
    builder.set_retired_traces(len(retired_ids))
    return ImportResult(dry_run=False, report=builder.build(), failed_items=failed_items)


def _partition_reconciliation_plan(plan: ReconciliationPlan) -> DiskMap[str, ReconciliationPlan]:
    """Index prevalidated actions by session without rescanning the whole plan per session."""
    sessions: DiskMap[str, ReconciliationPlan] = DiskMap()
    for scan in plan.session_scans:
        if scan.selected:
            sessions[scan.session_id] = ReconciliationPlan(session_scans=[scan])
    for name in ("imports", "updates", "retirements", "results"):
        for action in getattr(plan, name):
            identity = action.turn or action.trace
            assert identity is not None, "planned actions carry a source identity"
            session = sessions.get(identity.session_id)
            if session is None:
                session = ReconciliationPlan()
                sessions[identity.session_id] = session
            getattr(session, name).append(action)
    return sessions


def _import_session(
    config: ImportConfig,
    dependencies: ImportDependencies,
    builder: ReportBuilder,
    gateway: MlflowGateway,
    snapshot: SourceSnapshot,
    reader: DiscoveryPlanReader,
    planned: ReconciliationPlan,
    *,
    experiment_id: str,
    capability: CapabilityReport,
    redaction_mode: RedactionMode,
    staging: Path,
    pricing_basis_for: Callable[[str], Mapping[str, dict[str, Any]]] | None,
    retired_ids: MutableMapping[str, bool],
) -> int:
    """Import one session; no replay, request or verification state escapes this frame.

    The caller owns and closes the current scratch store after this returns,
    including on failure. The discovery snapshot and final report have separate
    invocation-wide owners.
    """
    plan = ReconciliationPlan(
        imports=DiskList(planned.imports),
        updates=DiskList(planned.updates),
        retirements=DiskList(planned.retirements),
        results=DiskList(planned.results),
        session_scans=DiskList(planned.session_scans),
    )
    replay = dependencies.replay_builder(snapshot)
    contexts = _MappingContexts(
        adapter=dependencies.replay_adapter_factory(snapshot),
        scans=reader.iter_session_scans(),
        snapshot=snapshot,
        experiment_id=experiment_id,
        inline_limit_bytes=capability.max_inline_payload_bytes,
        redaction_mode=redaction_mode,
        trace_max_bytes=config.trace_max_bytes,
        trace_max_spans=config.trace_max_spans,
    )
    reconciliation = dependencies.reconciliation_dependencies(
        gateway=gateway,
        replay=replay,
        context_for=contexts.context_for,
        redaction_mode=redaction_mode,
        staging_directory=staging,
    )

    def replay_session(session_id: str) -> ReplayTurn:
        header = NormalizedSnapshotAdapter(snapshot).session_header(session_id)
        if header is None:
            raise ImporterError("replay_session_unavailable", "a selected frozen session is unavailable")
        return build_replay_turn(snapshot, header, TurnSummary(session_id, 0, ""))

    reconciliation = replace(
        reconciliation,
        conflict_policy=config.on_conflict,
        replay_session=replay_session,
        experiment_id=experiment_id,
        pricing_basis_for=pricing_basis_for,
    )
    with gateway.session_resources() if isinstance(gateway, MlflowGateway) else nullcontext():
        apply_reconciliation(
            plan, gateway, reconciliation, dependencies.retry_policy, session_wide=isinstance(gateway, MlflowGateway)
        )

    failed_items = _record_normal_results(builder, plan, reader.iter_turns())
    for result in plan.results:
        for trace_id in result.retired_trace_ids:
            retired_ids[trace_id] = True
    for trace_id in plan.confirmed_retirements:
        retired_ids[trace_id] = True
    return failed_items


def _preflight_trace_envelopes(
    config: ImportConfig,
    dependencies: ImportDependencies,
    snapshot: SourceSnapshot,
    reader: DiscoveryPlanReader,
    replay: ReplayBuilder,
    experiment: ExperimentRef | None,
    pricing_basis_for: Callable[[str], Mapping[str, dict[str, Any]]] | None,
) -> None:
    """Admit every selected turn before experiment, staging or recovery writes.

    The unresolved experiment identity is used only for this disposable pure
    mapping. No request, placeholder fingerprint, or proof is persisted.
    Creation re-admits actual late-decorated requests at the gateway boundary.
    """
    from mlflow_import.content.accounting import pricing_bases
    from mlflow_import.content.turn_map import validate_turn_map_admission
    from mlflow_import.mlflow.capabilities import PINNED_MAX_INLINE_PAYLOAD_BYTES
    from mlflow_import.mlflow.trace_envelope import TraceEnvelopeLimits, admit_trace
    from mlflow_import.retry import check_budget

    limits = TraceEnvelopeLimits(
        max_bytes=config.trace_max_bytes,
        max_spans=config.trace_max_spans,
    )
    contexts = _MappingContexts(
        adapter=dependencies.replay_adapter_factory(snapshot),
        scans=reader.iter_session_scans(),
        snapshot=snapshot,
        experiment_id=(experiment.experiment_id or "") if experiment is not None else "",
        inline_limit_bytes=PINNED_MAX_INLINE_PAYLOAD_BYTES,
        redaction_mode=RedactionMode.REDACTED if config.redact else RedactionMode.VERBATIM,
        trace_max_bytes=config.trace_max_bytes,
        trace_max_spans=config.trace_max_spans,
    )
    token = pricing_bases.set({})
    try:
        for entry in reader.iter_turns():
            check_budget()
            pricing_bases.set(pricing_basis_for(entry.session_id) if pricing_basis_for is not None else {})
            request: TraceCreateRequest | None = None
            frozen_replay: ReplayTurn | None = None
            context: MappingContext | None = None
            try:
                frozen_replay = replay(entry)
                context = contexts.context_for(entry.session_id)
                request = dependencies.mapper(frozen_replay, context)
                admitted = admit_trace(request, reserve_late_control=True, limits=limits)
                validate_turn_map_admission(frozen_replay, admitted.request, context.redaction_mode)
            except ItemImportError as error:
                _attach_preflight_trace_measurement(
                    request,
                    frozen_replay,
                    context,
                    dependencies.mapper,
                    error,
                )
                _debug_preflight_turn_failure(config, entry, error)
                raise _preflight_turn_error(entry, error) from error
            else:
                _debug_preflight_turn(config, entry, admitted)
    finally:
        pricing_bases.reset(token)


def _preflight_turn_error(entry: DiscoveryTurnEntry, error: ItemImportError) -> ItemImportError:
    wrapped = ItemImportError(
        error.code,
        f"session {entry.session_id} turn {entry.turn_index} ({entry.turn_key}): {error.safe_message}",
        error.exit_status,
    )
    wrapped.diagnostics = dict(error.diagnostics)
    return wrapped


def _attach_preflight_trace_measurement(
    request: TraceCreateRequest | None,
    frozen_replay: ReplayTurn | None,
    context: MappingContext | None,
    mapper: Callable[[ReplayTurn, MappingContext], TraceCreateRequest],
    error: ItemImportError,
) -> None:
    from mlflow_import.mlflow.trace_envelope import (
        DIAGNOSTIC_MAX_TRACE_BYTES,
        DIAGNOSTIC_MAX_TRACE_CONTROL_BYTES,
        DIAGNOSTIC_MAX_TRACE_SPANS,
        measure_trace,
    )

    diagnostic_request = request
    if frozen_replay is not None and context is not None:
        try:
            diagnostic_context = replace(
                context,
                trace_max_bytes=DIAGNOSTIC_MAX_TRACE_BYTES,
                trace_max_spans=DIAGNOSTIC_MAX_TRACE_SPANS,
            )
            diagnostic_request = mapper(frozen_replay, diagnostic_context)
        except ItemImportError as measurement_error:
            diagnostics = dict(error.diagnostics)
            diagnostics.setdefault("measurement_error", measurement_error.code)
            error.diagnostics = diagnostics
            return
    if diagnostic_request is None:
        return
    try:
        measurement = measure_trace(
            diagnostic_request,
            max_bytes=DIAGNOSTIC_MAX_TRACE_BYTES,
            max_spans=DIAGNOSTIC_MAX_TRACE_SPANS,
            max_control_bytes=DIAGNOSTIC_MAX_TRACE_CONTROL_BYTES,
        )
    except ItemImportError as measurement_error:
        diagnostics = dict(error.diagnostics)
        diagnostics.setdefault("measurement_error", measurement_error.code)
        error.diagnostics = diagnostics
        return
    diagnostics = dict(error.diagnostics)
    diagnostics.setdefault("measured_span_count", measurement.span_count)
    diagnostics.setdefault("measured_span_bytes", measurement.serialized_span_upper_bound)
    diagnostics.setdefault("measured_control_bytes", measurement.control_metadata_upper_bound)
    error.diagnostics = diagnostics


def _debug_preflight_turn(config: ImportConfig, entry: DiscoveryTurnEntry, admitted: Any) -> None:
    if config.log_level != "DEBUG":
        return
    _write_debug(
        "mlflow-import: preflight turn "
        f"session={entry.session_id} turn={entry.turn_index} "
        f"bytes={admitted.serialized_span_upper_bound} spans={admitted.span_count} "
        f"control={admitted.control_metadata_upper_bound}"
    )


def _debug_preflight_turn_failure(config: ImportConfig, entry: DiscoveryTurnEntry, error: ItemImportError) -> None:
    if config.log_level != "DEBUG":
        return
    diagnostics = " ".join(f"{key}={value}" for key, value in sorted(error.diagnostics.items()))
    suffix = f" {diagnostics}" if diagnostics else ""
    _write_debug(
        "mlflow-import: preflight turn failed "
        f"session={entry.session_id} turn={entry.turn_index} code={error.code}{suffix}"
    )


def _prepare_pricing_bases(
    config: ImportConfig,
    gateway: MlflowGateway,
    experiment: ExperimentRef | None,
    snapshot: SourceSnapshot,
    reader: DiscoveryPlanReader,
    directory: Path,
) -> Callable[[str], Mapping[str, dict[str, Any]]]:
    """Freeze instance-scoped verified pricing donors during read-only planning."""
    from mlflow_import.content.accounting import freeze_pricing_bases
    from mlflow_import.content.bundles import source_prefix
    from mlflow_import.mlflow.attempts import AttemptJournal
    from mlflow_import.mlflow.reconciliation import _select_pricing_basis
    from mlflow_import.retry import check_budget
    from mlflow_import.source.turns import build_replay_turn, indexed_turn_summaries

    mode = RedactionMode.REDACTED if config.redact else RedactionMode.VERBATIM
    adapter = NormalizedSnapshotAdapter(snapshot)
    bases: DiskMap[str, Mapping[str, dict[str, Any]]] = DiskMap()
    for scan in reader.iter_session_scans():
        check_budget()
        if not scan.selected:
            continue
        selected: Mapping[str, dict[str, Any]] = {}
        header = adapter.session_header(scan.session_id)
        if header is None:
            raise ConfigurationError("replay_session_unavailable", "a selected source header is missing")
        if experiment is not None and experiment.experiment_id is not None:
            prefix = source_prefix(header, mode).rsplit("/", 1)[0]
            states = DiskList(AttemptJournal(gateway, experiment.experiment_id, prefix).attempts())
            if (
                sum("complete" not in state and "aborted" not in state and "discarded" not in state for state in states)
                > 1
            ):
                raise ConfigurationError("attempt_state_conflict", "multiple unfinished attempts require recovery")
            selected = _select_pricing_basis(states, gateway, config.on_conflict)
        selected = freeze_pricing_bases(
            (
                build_replay_turn(snapshot, header, summary)
                for summary in indexed_turn_summaries(snapshot, scan.session_id)
            ),
            selected,
        )
        bases[scan.session_id] = selected

    def load(session_id: str) -> Mapping[str, dict[str, Any]]:
        return bases.get(session_id, {})

    return load


def _prepare_staging_directory(run_root: Path) -> Path:
    """Create and validate the local artifact staging dependency."""
    staging = run_root / _ARTIFACT_STAGING_DIRNAME
    try:
        if not run_root.is_dir():
            raise OSError("run root is not a directory")
        if budget := resources.current_storage():
            budget.resize(staging, 0)
        staging.mkdir(mode=0o700, exist_ok=True)
        if not staging.is_dir():
            raise OSError("staging path is not a directory")
    except OSError:
        raise ConfigurationError(
            "staging_directory_unavailable",
            "the local artifact staging directory could not be prepared",
        ) from None
    return staging


def _resolve_existing_experiment(gateway: MlflowGateway, name: str) -> ExperimentRef | None:
    """Resolve the experiment READ-ONLY; ``None`` when it does not exist yet.

    A missing experiment has zero owned traces, so the importer defers its
    creation until every pre-write check has passed. The gateway's
    ``create=False`` resolution raises the typed ``experiment_not_found``
    error for a missing name; this helper converts exactly that error to
    ``None`` and lets every other failure propagate.
    """
    try:
        return gateway.resolve_experiment(name, create=False)
    except ConfigurationError as error:
        if error.code != "experiment_not_found":
            raise
        return None


def _record_experiment(builder: ReportBuilder, experiment: ExperimentRef) -> None:
    """Record the resolved experiment: planned, created, or resolved."""
    if experiment.planned_creation:
        status = "planned"
    elif experiment.created:
        status = "created"
    else:
        status = "resolved"
    builder.set_experiment(experiment.experiment_id, status)


def _owned_candidates(
    gateway: MlflowGateway,
    experiment: ExperimentRef | None,
    snapshot: SourceSnapshot | None = None,
    reader: DiscoveryPlanReader | None = None,
) -> list[OwnedTraceCandidate]:
    """Index the experiment's importer-owned traces (read-only search).

    A missing (or dry-run planned) experiment has no owned traces to look
    up, so its candidate list is trivially empty.
    """
    if experiment is None or experiment.experiment_id is None:
        return []
    if isinstance(gateway, MlflowGateway) and snapshot is not None and reader is not None:
        from mlflow_import.identity import qualified_session

        adapter = NormalizedSnapshotAdapter(snapshot)
        candidates: list[OwnedTraceCandidate] = DiskList()
        for scan in reader.iter_session_scans():
            if not scan.selected:
                continue
            header = adapter.session_header(scan.session_id)
            if header is None:
                raise ConfigurationError("replay_session_unavailable", "a selected session header is missing")
            for trace in gateway.iter_session_traces(
                experiment.experiment_id, scan.session_id, qualified_session(header)
            ):
                from mlflow_import.compatibility import decode_ownership

                namespace = decode_ownership(trace.tags, trace.metadata).get("mlflow_import.source_namespace")
                if namespace is not None and namespace != header.source_fingerprint:
                    continue
                candidate = OwnedTraceCandidate.from_owned_trace(trace, experiment.experiment_id)
                if not candidate.ownership_complete:
                    gateway.remember_foreign(qualified_session(header), scan.session_id, trace)
                    continue
                from mlflow_import.mlflow.trace_envelope import require_trace_proof

                require_trace_proof(trace)
                candidates.append(candidate)
        return candidates
    return DiskList(
        OwnedTraceCandidate.from_owned_trace(trace, experiment.experiment_id)
        for trace in gateway.iter_owned_traces(experiment.experiment_id)
    )


def _record_normal_results(
    builder: ReportBuilder,
    plan: ReconciliationPlan,
    turns: Iterable[DiscoveryTurnEntry],
) -> int:
    """Record the applied reconciliation outcomes in plan order."""
    failed_items = sum(1 for result in plan.results if result.action == ACTION_FAILED) + len(plan.session_errors)
    for session_id, code in plan.session_errors.items():
        builder.add_session_error(session_id, code)
    for session_id, values in plan.session_attempts.items():
        builder.set_session_attempt(session_id, values)
    results_by_key: MutableMapping[tuple[str, str], ReconciliationAction] = DiskMap()
    retirements_by_trace: MutableMapping[str, ReconciliationAction] = DiskMap()
    reported_retirements: DiskMap[str, bool] = DiskMap()
    for applied in plan.results:
        for trace_id in applied.retired_trace_ids:
            reported_retirements[trace_id] = True
        if applied.turn is None and applied.trace_id is not None:
            retirements_by_trace[applied.trace_id] = applied
            continue
        key = _result_key(applied)
        if key is not None:
            results_by_key[key] = applied
    unreported_by_session: DiskMap[str, list[str]] = DiskMap()
    for trace_id, session_id in plan.confirmed_retirements.items():
        if trace_id not in reported_retirements:
            unreported_by_session.setdefault(session_id, DiskList()).append(trace_id)
    for entry in turns:
        outcome = results_by_key.get((entry.session_id, entry.turn_key))
        if outcome is None:
            # Every planned turn receives a result; a missing one is
            # recorded as failed rather than silently dropped.
            failed_items += 1
            builder.add_turn(
                TurnResult(
                    session_id=entry.session_id,
                    turn_key=entry.turn_key,
                    turn_index=entry.turn_index,
                    action=ACTION_FAILED,
                    error_code="reconciliation_result_missing",
                )
            )
            continue
        recovered = tuple(unreported_by_session.pop(entry.session_id, ()))
        for trace_id in recovered:
            reported_retirements[trace_id] = True
        outcome = replace(outcome, retired_trace_ids=outcome.retired_trace_ids + recovered)
        builder.add_turn(
            TurnResult(
                session_id=entry.session_id,
                turn_key=entry.turn_key,
                turn_index=entry.turn_index,
                action=outcome.action,
                reason=outcome.reason,
                trace_id=outcome.trace_id,
                error_code=outcome.error_code,
                affected_trace_ids=outcome.affected_trace_ids,
                retired_trace_ids=outcome.retired_trace_ids,
                conflicting_trace_ids=outcome.conflicting_trace_ids,
                warnings=entry.warnings,
                attempt_id=outcome.attempt_id,
                import_instance_id=outcome.import_instance_id,
                attempt_state=outcome.attempt_state,
            )
        )
    for action in plan.retirements:
        candidate = action.trace
        if candidate is None:
            continue
        outcome = retirements_by_trace.get(candidate.trace_id)
        if outcome is None:
            failed_items += 1
        builder.add_turn(
            TurnResult(
                session_id=candidate.session_id,
                turn_key=candidate.turn_key,
                turn_index=candidate.turn_index,
                action=outcome.action if outcome is not None else ACTION_FAILED,
                reason=outcome.reason if outcome is not None else action.reason,
                trace_id=outcome.trace_id if outcome is not None else candidate.trace_id,
                source_present=False,
                error_code=outcome.error_code if outcome is not None else "reconciliation_result_missing",
                affected_trace_ids=outcome.affected_trace_ids if outcome is not None else (candidate.trace_id,),
                retired_trace_ids=outcome.retired_trace_ids if outcome is not None else (),
                conflicting_trace_ids=outcome.conflicting_trace_ids if outcome is not None else (),
                warnings=candidate.warnings,
                attempt_id=outcome.attempt_id if outcome else None,
                import_instance_id=outcome.import_instance_id if outcome else None,
                attempt_state=outcome.attempt_state if outcome else None,
            )
        )
    for session_id, trace_ids in unreported_by_session.items():
        for trace_id in trace_ids:
            builder.add_turn(
                TurnResult(
                    session_id=session_id,
                    turn_key="",
                    action=ACTION_RETIRED,
                    trace_id=trace_id,
                    source_present=False,
                    retired_trace_ids=(trace_id,),
                )
            )
    return failed_items


def _result_key(result: ReconciliationAction) -> tuple[str, str] | None:
    """The (session, turn) identity a reconciliation result concerns."""
    if result.turn is not None:
        return (result.turn.session_id, result.turn.turn_key)
    if result.trace is not None:
        return (result.trace.session_id, result.trace.turn_key)
    return None


def _any_optional_filter(filters: SelectionFilters, session_id: str | None = None) -> bool:
    """Whether any optional selection filter was supplied."""
    return any(
        (
            filters.from_time is not None,
            filters.to_time is not None,
            filters.directory is not None,
            filters.model is not None,
            filters.min_user_turns is not None,
            session_id is not None,
        )
    )


# ---------------------------------------------------------------------------
# Production dependencies
# ---------------------------------------------------------------------------


def _default_snapshot_factory(
    path: Path,
    temp_root: Path,
    exporter: SourceExporter,
    redaction: RedactionCallback,
    selection_evaluator: SelectionEvidenceEvaluator,
    session_id: str | None = None,
    session_ids: Collection[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AbstractContextManager[SourceSnapshot]:
    """The production normalized-snapshot factory."""
    snapshot_factory = create_file_snapshot if getattr(exporter, "file_source", False) else create_normalized_snapshot
    return snapshot_factory(
        path,
        temp_root,
        exporter,
        redaction,
        selection_evaluator=selection_evaluator,
        session_id=session_id,
        session_ids=session_ids,
        progress_callback=progress_callback,
    )


@contextmanager
def _snapshot_reporting_context(
    factory: SnapshotFactory,
    path: Path,
    temp_root: Path,
    exporter: SourceExporter,
    redaction: RedactionCallback,
    selection_evaluator: SelectionEvidenceEvaluator,
    session_id: str | None = None,
    session_ids: Collection[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    *,
    require_selection_evidence: bool,
    builder: ReportBuilder,
) -> Iterator[SourceSnapshot]:
    """Add safe snapshot-cleanup warnings even when the pipeline raises."""
    snapshot: SourceSnapshot | None = None
    try:
        with _open_snapshot(
            factory,
            path,
            temp_root,
            exporter,
            redaction,
            selection_evaluator,
            session_id,
            session_ids,
            progress_callback,
            require_selection_evidence=require_selection_evidence,
        ) as opened:
            snapshot = opened
            yield opened
    finally:
        if snapshot is not None and snapshot.cleanup_warnings:
            builder.add_warning(_SNAPSHOT_CLEANUP_WARNING, _SNAPSHOT_CLEANUP_MESSAGE)
            # A cleanup failure is one invocation-level failed item even if
            # several filesystem operations report the same safe condition.
            builder.add_cleanup_failure()


def _open_snapshot(
    factory: SnapshotFactory,
    path: Path,
    temp_root: Path,
    exporter: SourceExporter,
    redaction: RedactionCallback,
    selection_evaluator: SelectionEvidenceEvaluator,
    session_id: str | None = None,
    session_ids: Collection[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    *,
    require_selection_evidence: bool,
) -> AbstractContextManager[SourceSnapshot]:
    """Invoke old/new snapshot factories without catching factory-body errors."""
    arguments = (path, temp_root, exporter, redaction)
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        if require_selection_evidence:
            raise ConfigurationError(
                "snapshot_factory_selection_evidence_unsupported",
                "the configured snapshot factory cannot capture required selection evidence",
            ) from None
        return factory(*arguments)  # type: ignore[call-arg]

    progress_kwargs: dict[str, Callable[[int, int], None]] = {}
    if progress_callback is not None:
        candidate_keywords: list[dict[str, object]] = [
            {"selection_evaluator": selection_evaluator},
        ]
        if session_id is not None or session_ids is not None:
            candidate_keywords.insert(
                0,
                {
                    "selection_evaluator": selection_evaluator,
                    "session_id": session_id,
                    "session_ids": session_ids,
                },
            )
        for keywords in candidate_keywords:
            try:
                signature.bind(*arguments, **keywords, progress_callback=progress_callback)
            except TypeError:
                continue
            progress_kwargs["progress_callback"] = progress_callback
            break

    if session_id is None and session_ids is None:
        try:
            signature.bind(*arguments, selection_evaluator=selection_evaluator, **progress_kwargs)
        except TypeError:
            try:
                signature.bind(*arguments, selection_evaluator)
            except TypeError:
                if require_selection_evidence:
                    raise ConfigurationError(
                        "snapshot_factory_selection_evidence_unsupported",
                        "the configured snapshot factory cannot capture required selection evidence",
                    ) from None
                return factory(*arguments)  # type: ignore[call-arg]
            return factory(*arguments, selection_evaluator)  # type: ignore[call-arg]
        return factory(*arguments, selection_evaluator=selection_evaluator, **progress_kwargs)  # type: ignore[call-arg]

    try:
        signature.bind(
            *arguments,
            selection_evaluator=selection_evaluator,
            session_id=session_id,
            session_ids=session_ids,
            **progress_kwargs,
        )
    except TypeError:
        try:
            signature.bind(*arguments, selection_evaluator, session_id, session_ids)
        except TypeError:
            try:
                signature.bind(*arguments, selection_evaluator=selection_evaluator, **progress_kwargs)
            except TypeError:
                if require_selection_evidence:
                    raise ConfigurationError(
                        "snapshot_factory_selection_evidence_unsupported",
                        "the configured snapshot factory cannot capture required selection evidence",
                    ) from None
                return factory(*arguments)  # type: ignore[call-arg]
            if session_id is not None:
                raise ConfigurationError(
                    "snapshot_factory_session_filter_unsupported",
                    "the configured snapshot factory cannot apply a session filter safely",
                ) from None
            return factory(*arguments, selection_evaluator=selection_evaluator, **progress_kwargs)  # type: ignore[call-arg]
        return factory(*arguments, selection_evaluator, session_id, session_ids)  # type: ignore[call-arg]
    return factory(
        *arguments,
        selection_evaluator=selection_evaluator,
        session_id=session_id,
        session_ids=session_ids,
        **progress_kwargs,
    )  # type: ignore[call-arg]


def _report_failed_items(report: Mapping[str, object]) -> int:
    """Read the single authoritative failed-item count from a report."""
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return 0
    value = summary.get("failed_items")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _finalize_result(result: ImportResult, builder: ReportBuilder) -> ImportResult:
    """Build the final report after every cleanup boundary has completed."""
    if work := _BUDGET.get():
        work.phase = "reporting"
    report = builder.build()
    return ImportResult(
        dry_run=result.dry_run,
        report=report,
        error=result.error,
        failed_items=_report_failed_items(report),
    )


def _cleanup_run_root(run_root: Path) -> bool:
    """Remove the run root and return whether safe cleanup reporting is needed."""
    try:
        resources.remove_tree(run_root)
    except Exception:
        # The report must never expose the path or exception details. The
        # caller records one generic invocation-level cleanup warning.
        return True
    return False


def _record_run_cleanup_failure(builder: ReportBuilder) -> None:
    """Record one generic run-root cleanup warning and failed item."""
    builder.add_warning(_RUN_CLEANUP_WARNING, _RUN_CLEANUP_MESSAGE)
    builder.add_cleanup_failure()


def _record_snapshot_cleanup_failure(builder: ReportBuilder, error: ImporterError) -> None:
    """Preserve safe snapshot cleanup state when construction never yielded."""
    warnings = getattr(error, "cleanup_warnings", ())
    if not isinstance(warnings, tuple):
        return
    if "temporary snapshot cleanup failed" not in warnings:
        return
    builder.add_warning(_SNAPSHOT_CLEANUP_WARNING, _SNAPSHOT_CLEANUP_MESSAGE)
    builder.add_cleanup_failure()


def _default_mapper(turn: ReplayTurn, context: MappingContext) -> TraceCreateRequest:
    """The production pure mapper, through the lazy request boundary."""
    return map_turn_to_trace(turn, context).to_create_request()


def build_dependencies(config: ImportConfig) -> ImportDependencies:
    """The production dependency set for one invocation.

    The gateway factory builds a real MLflow client for the configured
    tracking URI (the ambient MLflow configuration when unset); the lock
    factory acquires the real advisory lock under the resolved lock
    root; the temporary-file factory creates restrictive ``mkdtemp``
    directories; everything else is the production default over the
    registered adapters, the pure mapper, and the reconciliation
    artifact uploader.
    """

    def gateway_factory(dry_run: bool) -> MlflowGateway:
        from mlflow_import.mlflow.trace_envelope import TraceEnvelopeLimits

        return build_gateway(
            config.tracking_uri,
            dry_run=dry_run,
            trace_limits=TraceEnvelopeLimits(
                max_bytes=config.trace_max_bytes,
                max_spans=config.trace_max_spans,
            ),
        )

    def lock_factory(tracking_endpoint: str, experiment_name: str) -> AbstractContextManager[ImportLock]:
        return acquire_import_lock(tracking_endpoint, experiment_name, resolve_lock_root())

    def temp_directory_factory(prefix: str) -> Path:
        return resources.make_temp_dir(prefix=prefix)

    return ImportDependencies(
        snapshot_factory=_default_snapshot_factory,
        exporter_factory=build_live_exporter,
        replay_adapter_factory=NormalizedSnapshotAdapter,
        gateway_factory=gateway_factory,
        lock_factory=lock_factory,
        temp_directory_factory=temp_directory_factory,
        replay_builder=build_replay_index,
        upload_chunk_bytes=DEFAULT_UPLOAD_CHUNK_BYTES,
        mapper=_default_mapper,
        artifact_snapshot=artifact_snapshot_fingerprint,
        artifact_uploader=upload_artifact,
        clock=time.time_ns,
        retry_policy=RetryPolicy(),
    )
