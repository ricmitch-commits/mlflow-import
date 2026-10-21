"""Real-pipeline import execution against a live MLflow server.

The Task 14 integration modules drive the COMPLETE production pipeline —
live read-only source connection, registered exporter, normalized
snapshot, replay adapter, discovery, capability preflight, lock,
experiment resolution, reconciliation writes, reporting — against the
real gateway over the session's live server. Only the clock and the
retry policy stay deterministic (a fixed nanosecond instant and a
no-retry policy), exactly like the fake-seam harness the BDD suite
uses; every other seam is the production default, including the real
advisory lock (acquired and released per run under the resolved lock
root).

:func:`run_real_import` is the one entry point; the gateway factory is
injectable so reconciliation scenarios can wrap the production gateway
in an event-recording or failure-injecting subclass at the boundary
(the controller-sanctioned way to observe order and inject failures
without breaking the server).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from opencode_mlflow.config import ImportConfig
from opencode_mlflow.content.fingerprint import artifact_snapshot_fingerprint
from opencode_mlflow.importer import (
    ImportDependencies,
    ImportResult,
    build_replay_index,
    exit_status,
    run_import,
)
from opencode_mlflow.locking import acquire_import_lock, resolve_lock_root
from opencode_mlflow.mlflow.gateway import (
    DEFAULT_UPLOAD_CHUNK_BYTES,
    MlflowGateway,
    TraceCreateRequest,
    build_gateway,
)
from opencode_mlflow.mlflow.mapping import MappingContext, map_turn_to_trace
from opencode_mlflow.mlflow.reconciliation import upload_artifact
from opencode_mlflow.retry import RetryPolicy
from opencode_mlflow.source.registry import build_live_exporter
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter, ReplayTurn
from opencode_mlflow.source.snapshot import create_normalized_snapshot

__all__ = [
    "DETERMINISTIC_CLOCK_NS",
    "IMMEDIATE_RETRY_POLICY",
    "RealImportRun",
    "production_mapper",
    "run_real_import",
]

DETERMINISTIC_CLOCK_NS = 1_772_100_000_000_000_000
"""The fixed report-generation instant every integration run uses."""

IMMEDIATE_RETRY_POLICY = RetryPolicy(max_attempts=1, initial_delay_seconds=0.0, max_delay_seconds=0.0)
"""A retry policy that never retries and never sleeps (deterministic runs)."""

GatewayFactory = Callable[[bool], MlflowGateway]
"""Builds the gateway for one run mode, mirroring the importer's seam."""


def production_mapper(turn: ReplayTurn, context: MappingContext) -> TraceCreateRequest:
    """The production pure mapper (identical to the importer's default seam)."""
    return map_turn_to_trace(turn, context).to_create_request()


@dataclass(frozen=True)
class RealImportRun:
    """One real-pipeline invocation's observable outcome."""

    exit_status: int
    result: ImportResult
    report: dict[str, object]
    experiment_name: str
    source: Path


def turn_entries(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    """The report's turn entries (the single selected session's turns)."""
    sessions = report["sessions"]
    assert isinstance(sessions, list) and len(sessions) == 1
    turns = sessions[0]["turns"]
    assert isinstance(turns, list)
    return [turn for turn in turns if isinstance(turn, Mapping)]


def trace_ids_by_key(report: Mapping[str, object]) -> dict[str, str]:
    """The report's turn_key -> trace_id mapping."""
    mapping: dict[str, str] = {}
    for turn in turn_entries(report):
        trace_id = turn.get("trace_id")
        assert isinstance(trace_id, str) and trace_id, turn
        mapping[str(turn["turn_key"])] = trace_id
    return mapping


def run_real_import(
    *,
    server_uri: str,
    source: Path,
    experiment_name: str,
    tmp_root: Path,
    dry_run: bool = False,
    gateway_factory: GatewayFactory | None = None,
) -> RealImportRun:
    """Run one import through the production pipeline against the live server.

    Every seam is the production default over ``source`` — real snapshot,
    registered exporter, replay adapter, replay index, mapper, artifact
    snapshot, artifact uploader, real advisory lock, and restrictive
    temporary directories under ``tmp_root`` — except the deterministic
    clock and no-retry policy. ``gateway_factory`` defaults to the
    production ``build_gateway`` bound to ``server_uri``; scenarios may
    inject a recording or failure-injecting gateway subclass.
    """

    def default_gateway_factory(dry_run_flag: bool) -> MlflowGateway:
        return build_gateway(server_uri, dry_run=dry_run_flag)

    def lock_factory(tracking_endpoint: str, experiment: str):
        return acquire_import_lock(tracking_endpoint, experiment, resolve_lock_root())

    def temp_directory_factory(prefix: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=prefix, dir=tmp_root))

    dependencies = ImportDependencies(
        snapshot_factory=lambda path, temp_root, exporter, redaction, selection_evaluator: create_normalized_snapshot(
            path,
            temp_root,
            exporter,
            redaction,
            selection_evaluator=selection_evaluator,
        ),
        exporter_factory=build_live_exporter,
        replay_adapter_factory=NormalizedSnapshotAdapter,
        gateway_factory=gateway_factory if gateway_factory is not None else default_gateway_factory,
        lock_factory=lock_factory,
        temp_directory_factory=temp_directory_factory,
        replay_builder=build_replay_index,
        upload_chunk_bytes=DEFAULT_UPLOAD_CHUNK_BYTES,
        mapper=production_mapper,
        artifact_snapshot=artifact_snapshot_fingerprint,
        artifact_uploader=upload_artifact,
        clock=lambda: DETERMINISTIC_CLOCK_NS,
        retry_policy=IMMEDIATE_RETRY_POLICY,
    )
    config = ImportConfig(
        experiment_name=experiment_name,
        db_path=source,
        tracking_uri=server_uri,
        dry_run=dry_run,
    )
    result = run_import(config, dependencies)
    return RealImportRun(
        exit_status=exit_status(result),
        result=result,
        report=result.report,
        experiment_name=experiment_name,
        source=source,
    )
