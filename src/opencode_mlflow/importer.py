"""Two-pass import orchestration: discovery, preflight, replay, writes, reporting.

:func:`run_import` wires every completed layer into one invocation. The
phase order is strict and binding:

1. configuration (already validated by :func:`opencode_mlflow.config.parse_config`);
2. live read-only source connection plus ``probe_schema``;
3. the registered :class:`~opencode_mlflow.source.snapshot.SourceExporter`
   for the probed generation;
4. normalized snapshot extraction, which closes its own live connection
   immediately after the catalogued extraction (the probing connection is
   closed before extraction, so no live SQLite connection remains during
   discovery or any MLflow work);
5. the :class:`~opencode_mlflow.source.replay.NormalizedSnapshotAdapter`
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
12. reconciliation writes through Task 11's plan/apply with per-item
    failure isolation;
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
:class:`~opencode_mlflow.mlflow.reconciliation.ReconciliationDependencies`
for one run, binding the gateway, replay, per-session mapping contexts,
and the gateway-bound artifact uploader.
"""

from __future__ import annotations

import inspect
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from opencode_mlflow import __version__ as IMPORTER_VERSION
from opencode_mlflow.config import ImportConfig
from opencode_mlflow.content.fingerprint import (
    artifact_snapshot_fingerprint,
    build_effective_turn,
    source_fingerprint,
)
from opencode_mlflow.content.redaction import DeterministicRedactor, RedactionReport
from opencode_mlflow.contracts import IDENTITY_REDACTOR, RedactionMode
from opencode_mlflow.errors import ConfigurationError, ImporterError
from opencode_mlflow.locking import ImportLock, acquire_import_lock, resolve_lock_root
from opencode_mlflow.mlflow.capabilities import CapabilityReport, CapabilityRequest, require_capabilities
from opencode_mlflow.mlflow.gateway import (
    DEFAULT_UPLOAD_CHUNK_BYTES,
    ArtifactVerification,
    CreatedTrace,
    ExperimentRef,
    MlflowGateway,
    TraceCreateRequest,
    build_gateway,
)
from opencode_mlflow.mlflow.mapping import MappingContext, map_turn_to_trace
from opencode_mlflow.mlflow.reconciliation import (
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
from opencode_mlflow.reporting import REASON_DRY_RUN, ReportBuilder, TurnResult
from opencode_mlflow.retry import RetryPolicy
from opencode_mlflow.source.connection import open_live_connection
from opencode_mlflow.source.discovery import (
    ARTIFACT_SCHEMA_VERSION,
    MAPPING_VERSION,
    DiscoveryPlanReader,
    DiscoveryTurnEntry,
    SessionScanEntry,
    discover_sessions,
)
from opencode_mlflow.source.model import SchemaDescription, SessionHeader, SourceAdapter, TurnSummary
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
from opencode_mlflow.source.turns import build_replay_turn, iter_turn_summaries

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

    Matches :func:`opencode_mlflow.mlflow.reconciliation.upload_artifact`
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
    ``opencode-mlflow.result.v1``); ``error`` is the safe pre-write
    failure when the invocation aborted (its ``exit_status`` decides the
    process status); ``failed_items`` counts failed session/turn items and
    distinct sensitive temporary-data cleanup failures.
    """

    dry_run: bool
    report: dict[str, object]
    error: ImporterError | None = None
    failed_items: int = 0


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
    through :func:`opencode_mlflow.source.turns.build_replay_turn`.
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
        summary = next(
            (
                summary
                for summary in iter_turn_summaries(self._adapter.iter_records(session_id))
                if summary.turn_key == turn_key
            ),
            None,
        )
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
    ) -> None:
        self._adapter = adapter
        self._turn_counts = {scan.session_id: scan.logical_turn_count for scan in scans}
        self._snapshot = snapshot
        self._experiment_id = experiment_id
        self._inline_limit_bytes = inline_limit_bytes
        self._redaction_mode = redaction_mode
        self._cache: dict[str, MappingContext] = {}
        self._generations: dict[str, str] = {}

    def context_for(self, session_id: str) -> MappingContext:
        """The mapping context for one session (memoized per run)."""
        context = self._cache.get(session_id)
        if context is None:
            context = self._build(session_id)
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
            logical_turn_count=self._turn_counts.get(session_id),
            source_message_count=message_count,
            source_part_count=part_count,
        )

    def _generation(self, session_id: str) -> str:
        if not self._generations:
            for header in self._adapter.iter_session_headers():
                self._generations.setdefault(header.session_id, header.adapter_generation)
        return self._generations.get(session_id, "")


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
    """Run the two-pass import pipeline and return its versioned result.

    Pre-write failures (configuration, schema, snapshot, capability,
    lock, replay consistency) are caught and returned as a failure
    result carrying the safe error in the report — the caller renders it
    and exits with :func:`exit_status`. Session/turn write failures are
    isolated per item by the reconciliation layer and reported as failed
    items (exit ``1``) while later items continue.
    """
    redaction_mode = RedactionMode.REDACTED if config.redact else RedactionMode.VERBATIM
    counting = _CountingRedaction(DeterministicRedactor(redaction_mode))
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
    )
    if not _any_optional_filter(filters):
        builder.add_warning(BROAD_SELECTION_WARNING, BROAD_SELECTION_MESSAGE)
    run_root: Path | None = None
    result: ImportResult | None = None
    try:
        try:
            run_root = dependencies.temp_directory_factory("opencode-mlflow-run-")
        except Exception:
            temp_failure = ConfigurationError(_TEMP_DIRECTORY_FAILED, _TEMP_DIRECTORY_FAILED_MESSAGE)
            builder.add_error(temp_failure.code, temp_failure.safe_message)
            result = ImportResult(
                dry_run=config.dry_run,
                report=builder.build(),
                error=temp_failure,
                failed_items=0,
            )
        else:
            assert run_root is not None
            result = _run_pipeline(config, dependencies, builder, counting, filters, run_root)
    except ImporterError as error:
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
) -> ImportResult:
    """Phases 1-7: source, snapshot, discovery, and the global preflight."""
    redaction_mode = RedactionMode.REDACTED if config.redact else RedactionMode.VERBATIM

    # Phases 2-3: one live read-only connection for probing, the
    # registered exporter bound to it, then the connection closes before
    # extraction (the snapshot driver opens and closes its own).
    with open_live_connection(config.db_path) as live:
        description = probe_schema(live.connection)
        exporter = dependencies.exporter_factory(live.connection, description)

    with _snapshot_reporting_context(
        dependencies.snapshot_factory,
        config.db_path,
        run_root,
        exporter,
        counting,
        SelectionEvidenceEvaluator(filters),
        require_selection_evidence=filters.directory is not None or filters.model is not None,
        builder=builder,
    ) as snapshot:
        builder.set_source(schema_generation=description.generation, fingerprint=snapshot.fingerprint)
        builder.set_redaction_report(counting.report)
        for warning_code in snapshot.export_warnings:
            builder.add_warning(warning_code, "the source exporter reported a safe warning")
        for warning_code in snapshot.schema_description.warnings:
            builder.add_warning(warning_code, "the source schema probe reported a safe warning")

        # Phase 5-6: the replay snapshot and the sanitized discovery plan.
        adapter = dependencies.replay_adapter_factory(snapshot)
        plan_path = run_root / DISCOVERY_PLAN_FILENAME
        stats = discover_sessions(adapter, filters, plan_path)
        builder.set_source_counts(
            sessions_scanned=stats.session_scan_count, sessions_selected=stats.selected_session_count
        )
        reader = DiscoveryPlanReader(plan_path)
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
        capability = gateway.probe(
            CapabilityRequest(needs_retirement=True, needs_named_artifacts=True), dry_run=config.dry_run
        )
        require_capabilities(capability)

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

    return result


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
    experiment = gateway.resolve_experiment(config.experiment_name, create=True)
    _record_experiment(builder, experiment)
    plan = plan_reconciliation(reader.iter_session_scans(), reader.iter_turns(), _owned_candidates(gateway, experiment))
    builder.set_unexamined_owned_traces(len(plan.unexamined_owned_traces))

    # The replay fingerprint checks still run: a dry run validates.
    _verify_replayed_fingerprints(reader.iter_turns(), dependencies.replay_builder(snapshot))

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
    experiment = _resolve_existing_experiment(gateway, config.experiment_name)
    plan = plan_reconciliation(reader.iter_session_scans(), reader.iter_turns(), _owned_candidates(gateway, experiment))
    builder.set_unexamined_owned_traces(len(plan.unexamined_owned_traces))

    # Phase 11: every replay fingerprint check passes before any mutation.
    replay = dependencies.replay_builder(snapshot)
    _verify_replayed_fingerprints(reader.iter_turns(), replay)

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

    # Phase 12: reconciliation writes with per-session mapping contexts.
    contexts = _MappingContexts(
        adapter=dependencies.replay_adapter_factory(snapshot),
        scans=reader.iter_session_scans(),
        snapshot=snapshot,
        experiment_id=experiment.experiment_id,
        inline_limit_bytes=capability.max_inline_payload_bytes,
        redaction_mode=redaction_mode,
    )
    reconciliation = dependencies.reconciliation_dependencies(
        gateway=gateway,
        replay=replay,
        context_for=contexts.context_for,
        redaction_mode=redaction_mode,
        staging_directory=staging,
    )
    apply_reconciliation(plan, gateway, reconciliation, dependencies.retry_policy)

    failed_items = _record_normal_results(builder, plan, reader.iter_turns())
    builder.set_retired_traces(sum(len(result.retired_trace_ids) for result in plan.results))
    return ImportResult(dry_run=False, report=builder.build(), failed_items=failed_items)


def _prepare_staging_directory(run_root: Path) -> Path:
    """Create and validate the local artifact staging dependency."""
    staging = run_root / _ARTIFACT_STAGING_DIRNAME
    try:
        if not run_root.is_dir():
            raise OSError("run root is not a directory")
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


def _owned_candidates(gateway: MlflowGateway, experiment: ExperimentRef | None) -> list[OwnedTraceCandidate]:
    """Index the experiment's importer-owned traces (read-only search).

    A missing (or dry-run planned) experiment has no owned traces to look
    up, so its candidate list is trivially empty.
    """
    if experiment is None or experiment.experiment_id is None:
        return []
    return [
        OwnedTraceCandidate.from_owned_trace(trace, experiment.experiment_id)
        for trace in gateway.iter_owned_traces(experiment.experiment_id)
    ]


def _record_normal_results(
    builder: ReportBuilder,
    plan: ReconciliationPlan,
    turns: Iterable[DiscoveryTurnEntry],
) -> int:
    """Record the applied reconciliation outcomes in plan order."""
    failed_items = sum(1 for result in plan.results if result.action == ACTION_FAILED)
    results_by_key: dict[tuple[str, str], ReconciliationAction] = {}
    retirements_by_trace: dict[str, ReconciliationAction] = {}
    for applied in plan.results:
        if applied.turn is None and applied.trace_id is not None:
            retirements_by_trace[applied.trace_id] = applied
            continue
        key = _result_key(applied)
        if key is not None:
            results_by_key[key] = applied
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


def _any_optional_filter(filters: SelectionFilters) -> bool:
    """Whether any optional selection filter was supplied."""
    return any(
        (
            filters.from_time is not None,
            filters.to_time is not None,
            filters.directory is not None,
            filters.model is not None,
            filters.min_user_turns is not None,
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
) -> AbstractContextManager[SourceSnapshot]:
    """The production normalized-snapshot factory."""
    return create_normalized_snapshot(
        path,
        temp_root,
        exporter,
        redaction,
        selection_evaluator=selection_evaluator,
    )


@contextmanager
def _snapshot_reporting_context(
    factory: SnapshotFactory,
    path: Path,
    temp_root: Path,
    exporter: SourceExporter,
    redaction: RedactionCallback,
    selection_evaluator: SelectionEvidenceEvaluator,
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

    try:
        signature.bind(*arguments, selection_evaluator=selection_evaluator)
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
        # The binding check is the only TypeError handling here. The actual
        # call is deliberately outside the try blocks so a factory's own
        # TypeError is never mistaken for a legacy four-argument signature.
        return factory(*arguments, selection_evaluator)  # type: ignore[call-arg]
    return factory(*arguments, selection_evaluator=selection_evaluator)  # type: ignore[call-arg]


def _report_failed_items(report: Mapping[str, object]) -> int:
    """Read the single authoritative failed-item count from a report."""
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return 0
    value = summary.get("failed_items")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _finalize_result(result: ImportResult, builder: ReportBuilder) -> ImportResult:
    """Build the final report after every cleanup boundary has completed."""
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
        shutil.rmtree(run_root)
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
        return build_gateway(config.tracking_uri, dry_run=dry_run)

    def lock_factory(tracking_endpoint: str, experiment_name: str) -> AbstractContextManager[ImportLock]:
        return acquire_import_lock(tracking_endpoint, experiment_name, resolve_lock_root())

    def temp_directory_factory(prefix: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=prefix))

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
