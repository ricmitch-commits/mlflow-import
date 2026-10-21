"""Unit tests for ownership reconciliation and safe replacement.

Plan tests pin the pure planner: candidates are grouped by the
``(session_id, turn_key)`` ownership key, turns without candidates plan
as imports, turns with candidates plan as updates whose final decision
waits for verification, stale turns of SELECTED complete scans (and
zero-turn tombstones) plan as retirements, and owned traces outside the
reconciliation scope are reported as unexamined — a filtered run never
retires an unselected session's traces.

Apply tests drive :meth:`FakeMlflowGateway` (no network, no real
MLflow): every candidate is verified before any winner selection,
create, or retirement; a verified winner is selected by newest creation
timestamp with ties broken by the lexicographically greatest trace ID;
changed turns are replaced in the strict create → verify → upload →
verify → retire-old order; failed replacements roll the new trace back
and retain the old one; ambiguous creates search for a surviving
candidate before retrying; and duplicate identities that cannot be
verified are reported as conflicts without mutation.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from opencode_mlflow.content.fingerprint import artifact_snapshot_fingerprint as calculate_artifact_snapshot_fingerprint
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.errors import ItemImportError
from opencode_mlflow.mlflow.capabilities import CapabilityRequest
from opencode_mlflow.mlflow.gateway import CreatedTrace, OwnedTrace, TraceChildRequest, TraceCreateRequest
from opencode_mlflow.mlflow.mapping import (
    ArtifactIdentityEnvelope,
    MappingContext,
    bind_mlflow_identity_map,
    map_turn_to_trace,
)
from opencode_mlflow.mlflow.reconciliation import (
    ACTION_FAILED,
    ACTION_IMPORTED,
    ACTION_RETIRED,
    ACTION_SKIPPED,
    ACTION_UPDATED,
    ARTIFACT_PATH_TAG,
    CandidateContract,
    OwnedTraceCandidate,
    OwnershipKey,
    ReconciliationDependencies,
    apply_reconciliation,
    created_artifact_identity_map,
    plan_reconciliation,
    upload_artifact,
)
from opencode_mlflow.retry import RetryPolicy
from opencode_mlflow.source.catalog import SCHEMA_CATALOG_VERSION
from opencode_mlflow.source.discovery import (
    ARTIFACT_SCHEMA_VERSION,
    MAPPING_VERSION,
    DiscoveryTurnEntry,
    SessionScanEntry,
)
from opencode_mlflow.source.replay import ReplayTurn
from tests.support.fake_gateway import FakeMlflowException, FakeMlflowGateway
from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_record, fake_summary

pytestmark = pytest.mark.unit

_IMMEDIATE_POLICY = RetryPolicy(max_attempts=1, initial_delay_seconds=0.0, max_delay_seconds=0.0)


def _turn_key(name: str) -> str:
    """A valid ``turn-<64 lowercase hex>`` key derived from a label."""
    return f"turn-{hashlib.sha256(name.encode('utf-8')).hexdigest()}"


def _contract_hex(value: str) -> str:
    """Keep test labels distinct while emitting production-shaped hex fields."""
    if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return value
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scan(
    session_id: str,
    *,
    selected: bool = True,
    complete_scan: bool = True,
    tombstone: bool = False,
    logical_turn_count: int = 1,
) -> SessionScanEntry:
    return SessionScanEntry(
        session_id=session_id,
        adapter_generation="current",
        selected=selected,
        complete_scan=complete_scan,
        tombstone=tombstone,
        logical_turn_count=logical_turn_count,
        record_count=logical_turn_count * 2,
        creation_time_ns=1_700_000_000_000_000_000,
        session_context_fingerprint="e" * 64,
        warnings=(),
    )


def _turn_entry(session_id: str, key_name: str, *, turn_index: int = 1) -> DiscoveryTurnEntry:
    return DiscoveryTurnEntry(
        session_id=session_id,
        turn_index=turn_index,
        turn_key=_turn_key(key_name),
        adapter_generation="current",
        planned_action="import",
        start_time_ns=1_700_000_000_000_000_000,
        end_time_ns=1_700_000_000_100_000_000,
        record_count=2,
        user_record_count=1,
        assistant_record_count=1,
        terminal_outcome="assistant_result",
        model_identities=frozenset({"provider/model"}),
        warnings=(),
        session_context_fingerprint="e" * 64,
        source_fingerprint="f" * 64,
        render_fingerprint="9" * 64,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        mapping_version=MAPPING_VERSION,
        importer_version="test",
        redaction_mode=RedactionMode.VERBATIM.value,
    )


def owned(
    session_id: str,
    *,
    turn_key: str | None = None,
    trace_id: str | None = None,
    request_time_ms: int | None = 1_000,
    artifact_sha256: str | None = None,
    artifact_size: int | None = None,
    experiment_id: str | None = "exp-1",
) -> OwnedTraceCandidate:
    """One importer-owned candidate with deterministic identity defaults."""
    return OwnedTraceCandidate(
        trace_id=trace_id if trace_id is not None else f"tr-{session_id}",
        session_id=session_id,
        turn_key=turn_key if turn_key is not None else _turn_key(f"stale-{session_id}"),
        request_time_ms=request_time_ms,
        artifact_sha256=artifact_sha256,
        artifact_size=artifact_size,
        experiment_id=experiment_id,
        source_fingerprint="f" * 64,
        source_schema="current",
        importer_version="test",
        mapping_version=MAPPING_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        redaction_mode=RedactionMode.VERBATIM.value,
        catalog_version=SCHEMA_CATALOG_VERSION,
        render_fingerprint="9" * 64,
        operation_key="c" * 64,
        turn_index=1,
        importer_marker="opencode-mlflow",
    )


class _SessionScans:
    """Scan collection with the filtered-run view the planner contract pins."""

    def __init__(self, scans: Sequence[SessionScanEntry]) -> None:
        self._scans = list(scans)

    def __iter__(self) -> Iterator[SessionScanEntry]:
        return iter(self._scans)

    def selected_only(self, session_id: str) -> list[SessionScanEntry]:
        """Every scan, with only the named session selected (a filtered run)."""
        return [
            replace(
                scan,
                selected=scan.session_id == session_id,
                tombstone=scan.tombstone and scan.session_id == session_id,
            )
            for scan in self._scans
        ]


@dataclass
class DiscoveryEntries:
    """The sanitized plan entries one discovery pass produced."""

    session_scans: _SessionScans
    turns: list[DiscoveryTurnEntry]


@pytest.fixture
def discovery_entries() -> DiscoveryEntries:
    """Two selected complete sessions: session-a with two turns, session-b with one."""
    return DiscoveryEntries(
        session_scans=_SessionScans(
            [
                _scan("session-a", logical_turn_count=2),
                _scan("session-b", logical_turn_count=1),
            ]
        ),
        turns=[
            _turn_entry("session-a", "session-a-turn-1", turn_index=1),
            _turn_entry("session-a", "session-a-turn-2", turn_index=2),
            _turn_entry("session-b", "session-b-turn-1", turn_index=1),
        ],
    )


@pytest.fixture
def fake_gateway() -> FakeMlflowGateway:
    """A preflight-passed fake gateway (no network, no real MLflow)."""
    gateway = FakeMlflowGateway()
    report = gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
    assert report.is_compatible, report.safe_error
    return gateway


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def test_plan_marks_turns_without_candidates_as_imports(discovery_entries: DiscoveryEntries) -> None:
    plan = plan_reconciliation(discovery_entries.session_scans, discovery_entries.turns, [])

    assert [action.action for action in plan.imports] == [ACTION_IMPORTED] * 3
    assert [action.turn for action in plan.imports] == discovery_entries.turns
    assert plan.updates == []
    assert plan.retirements == []
    assert plan.unexamined_owned_traces == ()
    assert plan.verifications == {}
    assert plan.results == []


def test_plan_marks_turns_with_candidates_as_pending_updates(discovery_entries: DiscoveryEntries) -> None:
    candidate = owned("session-a", turn_key=_turn_key("session-a-turn-1"), trace_id="tr-existing")

    plan = plan_reconciliation(discovery_entries.session_scans, discovery_entries.turns, [candidate])

    assert len(plan.updates) == 1
    action = plan.updates[0]
    assert action.action == ACTION_UPDATED
    assert action.turn is discovery_entries.turns[0]
    assert action.candidates == (candidate,)
    # A pure planner never selects a winner from unverified candidates.
    assert plan.verifications == {}
    assert plan.results == []
    assert len(plan.imports) == 2


def test_plan_groups_candidates_by_session_and_turn_key(discovery_entries: DiscoveryEntries) -> None:
    first = owned("session-a", turn_key=_turn_key("session-a-turn-1"), trace_id="tr-1")
    second = owned("session-a", turn_key=_turn_key("session-a-turn-1"), trace_id="tr-2")

    plan = plan_reconciliation(discovery_entries.session_scans, discovery_entries.turns, [first, second])

    assert len(plan.updates) == 1
    assert plan.updates[0].candidates == (first, second)
    assert OwnershipKey("session-a", _turn_key("session-a-turn-1")) == OwnershipKey(
        first.session_id,
        _turn_key("session-a-turn-1"),
    )


def test_plan_retires_stale_turns_of_selected_complete_sessions(discovery_entries: DiscoveryEntries) -> None:
    stale = owned("session-a", turn_key=_turn_key("removed-turn"), trace_id="tr-stale")

    plan = plan_reconciliation(discovery_entries.session_scans, discovery_entries.turns, [stale])

    assert len(plan.retirements) == 1
    action = plan.retirements[0]
    assert action.action == ACTION_RETIRED
    assert action.trace is stale
    assert action.trace.session_id == "session-a"
    assert action.reason == "stale_turn"


def test_zero_turn_tombstone_session_retires_all_previously_owned_turns() -> None:
    scans = _SessionScans([_scan("session-gone", logical_turn_count=0, tombstone=True)])
    first = owned("session-gone", turn_key=_turn_key("old-turn-1"), trace_id="tr-old-1")
    second = owned("session-gone", turn_key=_turn_key("old-turn-2"), trace_id="tr-old-2")

    plan = plan_reconciliation(scans, [], [first, second])

    assert [action.trace for action in plan.retirements] == [first, second]
    assert all(action.reason == "session_tombstone" for action in plan.retirements)


def test_incomplete_owned_candidate_is_unexamined_and_never_stale_retired() -> None:
    """An importer marker plus session/turn identity is not cleanup authority."""
    scan = _SessionScans([_scan("session-gone", logical_turn_count=0, tombstone=True)])
    owned_preview = OwnedTrace(
        trace_id="trace-incomplete-owner",
        request_time_ms=1_000,
        tags={"opencode.importer": "opencode-mlflow"},
        metadata={
            "opencode.importer": "opencode-mlflow",
            "opencode.session_id": "session-gone",
            "opencode.turn_index": "1",
            "opencode.turn_key": _turn_key("removed-turn"),
        },
    )
    candidate = OwnedTraceCandidate.from_owned_trace(owned_preview, "exp-1")

    plan = plan_reconciliation(scan, [], [candidate])

    assert plan.retirements == []
    assert plan.unexamined_owned_traces == (candidate,)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", ""),
        ("turn_key", "turn-not-a-lowercase-sha256"),
        ("source_fingerprint", "not-a-fingerprint"),
        ("render_fingerprint", "g" * 64),
        ("operation_key", "C" * 64),
        ("source_schema", "future"),
        ("importer_version", ""),
        ("mapping_version", None),
        ("artifact_schema_version", ""),
        ("redaction_mode", "masked"),
        ("catalog_version", None),
        ("importer_marker", "not-opencode-mlflow"),
    ],
)
def test_malformed_ownership_contract_is_unexamined_and_never_retired(field: str, value: object) -> None:
    scan = _SessionScans([_scan("session-gone", logical_turn_count=0, tombstone=True)])
    candidate = replace(owned("session-gone", turn_key=_turn_key("removed-turn")), **{field: value})

    plan = plan_reconciliation(scan, [], [candidate])

    assert plan.retirements == []
    assert plan.unexamined_owned_traces == (candidate,)


def test_invalid_candidate_warning_metadata_is_generic_and_blocks_cleanup() -> None:
    metadata = {
        "opencode.importer": "opencode-mlflow",
        "opencode.session_id": "session-gone",
        "opencode.turn_index": "1",
        "opencode.turn_key": _turn_key("removed-turn"),
        "opencode.source_fingerprint": "f" * 64,
        "opencode.source_schema": "current",
        "opencode.importer_version": "0.1.0",
        "opencode.mapping_version": MAPPING_VERSION,
        "opencode.artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "opencode.redaction_mode": RedactionMode.VERBATIM.value,
        "opencode.catalog_version": SCHEMA_CATALOG_VERSION,
        "opencode.render_fingerprint": "9" * 64,
        "opencode.operation_key": "c" * 64,
        "opencode.warnings": "safe_warning,SECRET-MUST-NOT-ECHO",
    }
    candidate = OwnedTraceCandidate.from_owned_trace(
        OwnedTrace(
            trace_id="trace-warning-metadata",
            request_time_ms=1,
            tags={"opencode.importer": "opencode-mlflow"},
            metadata=metadata,
        ),
        "exp-1",
    )
    plan = plan_reconciliation(
        [_scan("session-gone", logical_turn_count=0, tombstone=True)],
        [],
        [candidate],
    )

    assert candidate.warnings == ("invalid_warning_metadata",)
    assert "SECRET-MUST-NOT-ECHO" not in repr(candidate)
    assert plan.retirements == []
    assert plan.unexamined_owned_traces == (candidate,)


def test_malformed_candidate_for_selected_turn_is_a_prepopulated_conflict_not_an_import() -> None:
    turn = _turn_entry("session-selected", "selected-turn")
    candidate = replace(
        owned("session-selected", turn_key=turn.turn_key, trace_id="trace-malformed-selected"),
        source_fingerprint=None,
    )

    plan = plan_reconciliation(
        [_scan("session-selected")],
        [turn],
        [candidate],
    )

    assert plan.imports == []
    assert plan.updates == []
    assert plan.unexamined_owned_traces == (candidate,)
    assert len(plan.results) == 1
    result = plan.results[0]
    assert result.error_code == "ownership_conflict"
    assert result.affected_trace_ids == (candidate.trace_id,)
    assert result.conflicting_trace_ids == (candidate.trace_id,)


def test_overlong_warning_metadata_candidate_for_selected_turn_is_unexamined_conflict() -> None:
    turn = _turn_entry("session-warning", "warning-turn")
    candidate = OwnedTraceCandidate.from_owned_trace(
        OwnedTrace(
            trace_id="trace-overlong-warning",
            request_time_ms=1,
            tags={"opencode.importer": "opencode-mlflow"},
            metadata={
                "opencode.importer": "opencode-mlflow",
                "opencode.session_id": "session-warning",
                "opencode.turn_index": "1",
                "opencode.turn_key": turn.turn_key,
                "opencode.source_fingerprint": "f" * 64,
                "opencode.source_schema": "current",
                "opencode.importer_version": "test",
                "opencode.mapping_version": MAPPING_VERSION,
                "opencode.artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "opencode.redaction_mode": RedactionMode.VERBATIM.value,
                "opencode.catalog_version": SCHEMA_CATALOG_VERSION,
                "opencode.render_fingerprint": "9" * 64,
                "opencode.operation_key": "c" * 64,
                "opencode.warnings": "w" * 251,
            },
        ),
        "exp-1",
    )

    plan = plan_reconciliation([_scan("session-warning")], [turn], [candidate])

    assert plan.imports == []
    assert plan.updates == []
    assert plan.unexamined_owned_traces == (candidate,)
    assert plan.results[0].error_code == "ownership_conflict"
    assert plan.results[0].affected_trace_ids == (candidate.trace_id,)
    assert plan.results[0].conflicting_trace_ids == (candidate.trace_id,)


def test_non_string_candidate_warning_metadata_is_unexamined_conflict() -> None:
    turn = _turn_entry("session-warning-type", "warning-type-turn")
    candidate = replace(
        owned("session-warning-type", turn_key=turn.turn_key, trace_id="trace-non-string-warning"),
        warnings=("safe_warning", object()),
    )

    plan = plan_reconciliation([_scan("session-warning-type")], [turn], [candidate])

    assert candidate.warnings == ("invalid_warning_metadata",)
    assert not candidate.ownership_complete
    assert plan.imports == []
    assert plan.updates == []
    assert plan.unexamined_owned_traces == (candidate,)
    assert plan.results[0].error_code == "ownership_conflict"


def test_filtered_scan_cannot_retire_unselected_session(fake_gateway, discovery_entries):
    plan = plan_reconciliation(
        discovery_entries.session_scans.selected_only("session-a"),
        discovery_entries.turns,
        fake_gateway.owned_traces([owned("session-a"), owned("session-b")]),
    )

    assert not any(action.trace.session_id == "session-b" for action in plan.retirements)


def test_incomplete_scan_cannot_retire_stale_turns(discovery_entries: DiscoveryEntries) -> None:
    scans = _SessionScans([replace(_scan("session-a", logical_turn_count=2), complete_scan=False)])
    stale = owned("session-a", turn_key=_turn_key("removed-turn"), trace_id="tr-stale")

    plan = plan_reconciliation(scans, discovery_entries.turns, [stale])

    assert plan.retirements == []
    assert plan.unexamined_owned_traces == (stale,)


def test_owned_traces_outside_scope_are_reported_unexamined(discovery_entries: DiscoveryEntries) -> None:
    unselected = owned("session-b", trace_id="tr-unselected")
    unknown = owned("session-unknown", trace_id="tr-unknown")
    scans = discovery_entries.session_scans.selected_only("session-a")

    plan = plan_reconciliation(scans, discovery_entries.turns, [unselected, unknown])

    assert plan.unexamined_owned_traces == (unselected, unknown)
    assert plan.retirements == []


def test_plan_accepts_iterators_and_rejects_duplicate_turn_identities(discovery_entries: DiscoveryEntries) -> None:
    plan = plan_reconciliation(iter(discovery_entries.session_scans), iter(discovery_entries.turns), iter([]))

    assert len(plan.imports) == 3

    with pytest.raises(ItemImportError) as excinfo:
        plan_reconciliation(discovery_entries.session_scans, [*discovery_entries.turns, discovery_entries.turns[0]], [])

    assert excinfo.value.code == "reconciliation_plan_invalid"


# ---------------------------------------------------------------------------
# Apply harness
# ---------------------------------------------------------------------------


def _request(
    session_id: str = "ses-1",
    turn_key: str | None = None,
    *,
    experiment_id: str = "exp-1",
    source_fingerprint: str = "a" * 64,
    source_schema: str = "current",
    render_fingerprint: str = "b" * 64,
    operation_key: str = "c" * 64,
    artifact_snapshot_fingerprint: str | None = None,
    children: Sequence[TraceChildRequest] = (),
) -> TraceCreateRequest:
    """A complete mapped request carrying the importer-ownership metadata.

    Like the real mapper's output, the request records the artifact's
    exact relative path as the ``opencode.artifact_path`` root tag at
    creation, so candidates created through this helper verify at their
    own recorded location.
    """
    key = turn_key if turn_key is not None else _turn_key("turn-1")
    source_fingerprint = _contract_hex(source_fingerprint)
    render_fingerprint = _contract_hex(render_fingerprint)
    operation_key = _contract_hex(operation_key)
    snapshot_fingerprint = artifact_snapshot_fingerprint or calculate_artifact_snapshot_fingerprint(
        _replay_turn(session_id, key), RedactionMode.VERBATIM
    )
    return TraceCreateRequest(
        client_request_id=operation_key,
        experiment_id=experiment_id,
        name="opencode turn 1",
        native_session_id=session_id,
        root_inputs={"prompt": "synthetic prompt"},
        root_outputs={"reply": "synthetic reply"},
        root_attributes={"opencode.session_id": session_id},
        root_tags={
            ARTIFACT_PATH_TAG: (
                f"opencode-mlflow/v1/sessions/{session_id}/turns/{key}/{render_fingerprint}-{snapshot_fingerprint}.json"
            )
        },
        metadata={
            "opencode.importer": "opencode-mlflow",
            "opencode.session_id": session_id,
            "opencode.turn_index": "1",
            "opencode.turn_key": key,
            "opencode.source_fingerprint": source_fingerprint,
            "opencode.source_schema": source_schema,
            "opencode.importer_version": "test",
            "opencode.mapping_version": MAPPING_VERSION,
            "opencode.artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "opencode.redaction_mode": RedactionMode.VERBATIM.value,
            "opencode.catalog_version": SCHEMA_CATALOG_VERSION,
            "opencode.render_fingerprint": render_fingerprint,
            "opencode.operation_key": operation_key,
        },
        start_time_ns=1_700_000_000_000_000_000,
        end_time_ns=1_700_000_000_100_000_000,
        child_specs=lambda: iter(children),
        expected_operation_key=operation_key,
        artifact_snapshot_fingerprint=snapshot_fingerprint,
        artifact_relative_path=(
            f"opencode-mlflow/v1/sessions/{session_id}/turns/{key}/{render_fingerprint}-{snapshot_fingerprint}.json"
        ),
        root_status="OK",
    )


def _replay_turn(session_id: str = "ses-1", turn_key: str | None = None) -> FakeReplayTurn:
    """A deterministic replay turn for artifact streaming and mapping."""
    key = turn_key if turn_key is not None else _turn_key("turn-1")
    records = (
        fake_record("msg-user-1", raw_json='{"role": "user", "content": "synthetic prompt"}', session_id=session_id),
        fake_record(
            "msg-assistant-1",
            raw_json='{"role": "assistant", "content": "synthetic reply"}',
            session_id=session_id,
            source_type="assistant",
            classification="assistant",
            seq=2,
        ),
    )
    return FakeReplayTurn(
        session_header=fake_header(session_id),
        summary=fake_summary(session_id, turn_index=1, turn_key=key, record_count=2),
        turn_records=records,
        session_records=(),
        unassigned_records=(),
        root_input={"role": "user", "content": "synthetic prompt"},
        root_output={"role": "assistant", "content": "synthetic reply"},
    )


class _ScriptedRun:
    """Reconciliation dependencies over one fake replay turn and scripted mapping.

    ``map_trace`` produces a fresh request from ``request_factory`` on every
    call; ``artifact_snapshot`` reports the mapped request's snapshot
    fingerprint (or an injected inconsistent value); ``upload_artifact``
    uploads ``upload_payload`` through the fake gateway so verification is
    honest. A prior import is modeled by driving the gateway directly in the
    test (create + streamed-artifact upload).
    """

    def __init__(
        self,
        gateway: FakeMlflowGateway,
        turn: FakeReplayTurn,
        request_factory: Callable[[], TraceCreateRequest],
        *,
        upload_payload: bytes = b'{"synthetic": "artifact"}',
        snapshot_fingerprint: str | None = None,
    ) -> None:
        self.gateway = gateway
        self.turn = turn
        self.request_factory = request_factory
        self.upload_payload = upload_payload
        self.snapshot_fingerprint = snapshot_fingerprint
        self.requests: list[TraceCreateRequest] = []
        self.uploads: list[tuple[str, str]] = []

    def dependencies(self) -> ReconciliationDependencies:
        return ReconciliationDependencies(
            replay=self._replay,
            map_trace=self._map_trace,
            artifact_snapshot=self._artifact_snapshot,
            upload_artifact=self._upload_artifact,
        )

    def _replay(self, entry: DiscoveryTurnEntry) -> ReplayTurn:
        return self.turn

    def _map_trace(self, turn: ReplayTurn) -> TraceCreateRequest:
        request = self.request_factory()
        self.requests.append(request)
        return request

    def _artifact_snapshot(self, turn: ReplayTurn) -> str:
        if self.snapshot_fingerprint is not None:
            return self.snapshot_fingerprint
        assert self.requests, "artifact_snapshot is consulted after map_trace"
        return self.requests[-1].artifact_snapshot_fingerprint

    def _upload_artifact(
        self, turn: ReplayTurn, created: CreatedTrace, artifact_relative_path: str, redaction_mode: RedactionMode
    ):
        self.uploads.append((created.trace_id, artifact_relative_path))
        with tempfile.TemporaryDirectory(prefix="opencode-mlflow-test-upload-") as directory:
            staged = Path(directory) / "artifact.json"
            staged.write_bytes(self.upload_payload)
            return self.gateway.upload_named_json(created.trace_id, artifact_relative_path, staged)


def _previous_import(
    gateway: FakeMlflowGateway,
    turn: FakeReplayTurn,
    request: TraceCreateRequest,
    *,
    record: bool = True,
) -> tuple[OwnedTraceCandidate, CreatedTrace]:
    """Create one owned trace like a prior completed run.

    Drives the same public write path the importer uses — create, upload the
    streamed artifact, verify its bytes, then RECORD the verified digest on
    the trace — and indexes the candidate from the SEARCH PREVIEW
    (``iter_owned_traces``), exactly the way the write phase builds
    candidates. ``record=False`` models a crash between upload and recording:
    the artifact exists but the trace carries no recorded digest.
    """
    created = gateway.create_trace(request)
    verification = upload_artifact(
        turn,
        created,
        request.artifact_relative_path,
        RedactionMode.VERBATIM,
        gateway=gateway,
    )
    gateway.verify_named_json(created.trace_id, verification.relative_path, verification.sha256, verification.size)
    if record:
        gateway.record_artifact_verification(created.trace_id, verification.sha256, verification.size)
    owned = next(
        trace for trace in gateway.iter_owned_traces(request.experiment_id) if trace.trace_id == created.trace_id
    )
    candidate = OwnedTraceCandidate.from_owned_trace(owned, request.experiment_id)
    return candidate, created


_RELEVANT_EVENT_KINDS = frozenset(
    {
        "trace_create",
        "trace_verify",
        "artifact_upload",
        "artifact_verify",
        "trace_tag_set",
        "trace_retire",
        "candidate_verify",
    }
)


def _kinds(gateway: FakeMlflowGateway, start: int = 0) -> list[str]:
    """The reconciliation-relevant event kinds (setup noise filtered out)."""
    return [event.kind for event in gateway.events[start:] if event.kind in _RELEVANT_EVENT_KINDS]


def _apply(
    gateway: FakeMlflowGateway,
    run: _ScriptedRun,
    scans: Sequence[SessionScanEntry],
    turns: Sequence[DiscoveryTurnEntry],
    candidates: Sequence[OwnedTraceCandidate],
):
    plan = plan_reconciliation(scans, turns, candidates)
    apply_reconciliation(plan, gateway, run.dependencies(), _IMMEDIATE_POLICY)
    return plan


def _single_scan_turns(session_id: str = "ses-1") -> tuple[list[SessionScanEntry], list[DiscoveryTurnEntry]]:
    return [_scan(session_id)], [_turn_entry(session_id, "turn-1")]


def test_apply_preserves_selected_malformed_candidate_without_duplicate_create(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _turn_entry("ses-1", "turn-1")
    candidate = replace(
        owned("ses-1", turn_key=turn.turn_key, trace_id="trace-malformed-existing"),
        source_fingerprint=None,
    )
    run = _ScriptedRun(fake_gateway, _replay_turn(turn_key=turn.turn_key), lambda: _request(turn_key=turn.turn_key))

    plan = _apply(fake_gateway, run, [_scan("ses-1")], [turn], [candidate])

    assert len(plan.results) == 1
    assert plan.results[0].error_code == "ownership_conflict"
    assert not any(event.kind == "trace_create" for event in fake_gateway.events)
    assert plan.unexamined_owned_traces == (candidate,)

    first_key = _turn_key("contract-invalid-first")
    second_key = _turn_key("contract-valid-second")
    turn = _replay_turn()
    first_request = _request(turn_key=first_key)
    second_request = _request(turn_key=second_key)
    first_candidate, first_created = _previous_import(fake_gateway, turn, first_request)
    second_candidate, second_created = _previous_import(fake_gateway, turn, second_request)

    invalid_metadata = dict(first_request.metadata)
    del invalid_metadata["opencode.source_fingerprint"]
    invalid_request = replace(first_request, metadata=invalid_metadata)
    requests = iter((invalid_request, second_request, second_request))
    run = _ScriptedRun(fake_gateway, turn, lambda: next(requests))
    scans = [_scan("ses-1", logical_turn_count=2)]
    turns = [
        _turn_entry("ses-1", "contract-invalid-first"),
        _turn_entry("ses-1", "contract-valid-second", turn_index=2),
    ]

    plan = _apply(fake_gateway, run, scans, turns, [first_candidate, second_candidate])

    assert len(plan.results) == 2
    assert plan.results[0].action == ACTION_FAILED
    assert plan.results[0].error_code == "ownership_contract_invalid"
    assert plan.results[0].affected_trace_ids == (first_created.trace_id,)
    assert plan.results[1].action == ACTION_SKIPPED
    assert plan.results[1].reason == "unchanged"
    assert plan.results[1].trace_id == second_created.trace_id


# ---------------------------------------------------------------------------
# Apply: imports and strict ordering
# ---------------------------------------------------------------------------


def test_apply_imports_a_turn_in_create_verify_upload_verify_order(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [])

    assert [action.action for action in plan.results] == [ACTION_IMPORTED]
    assert plan.results[0].trace_id is not None
    assert _kinds(fake_gateway) == [
        "trace_create",
        "trace_verify",
        "artifact_upload",
        "artifact_verify",
        "trace_tag_set",
        "trace_tag_set",
    ]


def test_ambiguous_import_retires_redundant_verified_survivors_without_initial_candidates(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _replay_turn()
    request = _request()
    first, first_created = _previous_import(fake_gateway, turn, request)
    second, second_created = _previous_import(fake_gateway, turn, request)

    def ambiguous_without_survivor(_request: TraceCreateRequest):
        raise FakeMlflowException("synthetic timeout", "TEMPORARILY_UNAVAILABLE")

    fake_gateway.create_trace = ambiguous_without_survivor  # type: ignore[method-assign]
    run = _ScriptedRun(fake_gateway, turn, lambda: request)
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [])

    result = plan.results[0]
    assert result.action == ACTION_IMPORTED
    assert result.trace_id == second_created.trace_id
    assert result.retired_trace_ids == (first_created.trace_id,)
    assert result.affected_trace_ids == ()
    assert result.conflicting_trace_ids == ()
    assert fake_gateway.trace_ids() == [second_created.trace_id]
    assert first.trace_id != second.trace_id


def test_ambiguous_import_cleanup_failure_preserves_winner_and_reports_each_survivor(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _replay_turn()
    request = _request()
    first, first_created = _previous_import(fake_gateway, turn, request)
    second, second_created = _previous_import(fake_gateway, turn, request)
    fake_gateway.fail_retire_trace_ids = {first_created.trace_id}

    def ambiguous_without_survivor(_request: TraceCreateRequest):
        raise FakeMlflowException("synthetic timeout", "TEMPORARILY_UNAVAILABLE")

    fake_gateway.create_trace = ambiguous_without_survivor  # type: ignore[method-assign]
    run = _ScriptedRun(fake_gateway, turn, lambda: request)
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "replacement_cleanup_failed"
    assert result.trace_id == second_created.trace_id
    assert result.affected_trace_ids == (first_created.trace_id,)
    assert result.retired_trace_ids == ()
    assert result.conflicting_trace_ids == (first_created.trace_id,)
    assert fake_gateway.trace_ids() == [first_created.trace_id, second_created.trace_id]
    assert first.trace_id != second.trace_id


def test_transient_create_exhaustion_with_authoritative_empty_search_is_not_old_candidate_conflict(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _replay_turn()
    old_request = _request(source_fingerprint="old" + "0" * 61, operation_key="o" * 64)
    old_candidate, old_created = _previous_import(fake_gateway, turn, old_request)

    def transient_create(_request: TraceCreateRequest):
        raise FakeMlflowException("synthetic timeout", "TEMPORARILY_UNAVAILABLE")

    fake_gateway.create_trace = transient_create  # type: ignore[method-assign]
    new_request = _request(source_fingerprint="new" + "0" * 61, operation_key="n" * 64)
    run = _ScriptedRun(fake_gateway, turn, lambda: new_request)
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [old_candidate])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "trace_create_failed"
    assert result.conflicting_trace_ids == ()
    assert result.affected_trace_ids == (old_created.trace_id,)
    assert fake_gateway.trace_ids() == [old_created.trace_id]


def test_ambiguous_create_that_commits_before_timeout_recovers_verified_survivor(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _replay_turn()
    request = _request()
    original_create = fake_gateway.create_trace

    def commit_then_timeout(spec: TraceCreateRequest):
        created = original_create(spec)
        verification = upload_artifact(
            turn,
            created,
            spec.artifact_relative_path,
            RedactionMode.VERBATIM,
            gateway=fake_gateway,
        )
        fake_gateway.verify_named_json(
            created.trace_id, verification.relative_path, verification.sha256, verification.size
        )
        fake_gateway.record_artifact_verification(created.trace_id, verification.sha256, verification.size)
        raise FakeMlflowException("synthetic timeout after commit", "TEMPORARILY_UNAVAILABLE")

    fake_gateway.create_trace = commit_then_timeout  # type: ignore[method-assign]
    run = _ScriptedRun(fake_gateway, turn, lambda: request)
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [])

    result = plan.results[0]
    assert result.action == ACTION_IMPORTED
    assert result.trace_id == "fake-tr-1"
    assert fake_gateway.trace_ids() == ["fake-tr-1"]
    assert [event.kind for event in fake_gateway.events[marker:] if event.kind == "trace_create"] == ["trace_create"]
    assert [event.kind for event in fake_gateway.events[marker:] if event.kind == "candidate_verify"] == [
        "candidate_verify"
    ]


def test_apply_verifies_every_candidate_before_any_winner_or_create(fake_gateway: FakeMlflowGateway) -> None:
    first_turn = _replay_turn(turn_key=_turn_key("turn-1"))
    second_turn = _replay_turn(turn_key=_turn_key("turn-2"))
    first_candidate, _first = _previous_import(
        fake_gateway, first_turn, _request(turn_key=_turn_key("turn-1"), source_fingerprint="1" * 64)
    )
    second_candidate, _second = _previous_import(
        fake_gateway, second_turn, _request(turn_key=_turn_key("turn-2"), source_fingerprint="2" * 64)
    )

    def upload(turn: ReplayTurn, created: CreatedTrace, path: str, mode: RedactionMode):
        with tempfile.TemporaryDirectory(prefix="opencode-mlflow-test-upload-") as directory:
            staged = Path(directory) / "artifact.json"
            staged.write_bytes(b'{"synthetic": "artifact"}')
            return fake_gateway.upload_named_json(created.trace_id, path, staged)

    turns_by_key = {_turn_key("turn-1"): first_turn, _turn_key("turn-2"): second_turn}
    dependencies = ReconciliationDependencies(
        replay=lambda entry: turns_by_key[entry.turn_key],
        map_trace=lambda turn: _request(turn_key=turn.summary.turn_key, source_fingerprint="3" * 64),
        artifact_snapshot=lambda turn: calculate_artifact_snapshot_fingerprint(turn, RedactionMode.VERBATIM),
        upload_artifact=upload,
    )
    plan = plan_reconciliation(
        [_scan("ses-1", logical_turn_count=2)],
        [_turn_entry("ses-1", "turn-1"), _turn_entry("ses-1", "turn-2", turn_index=2)],
        [first_candidate, second_candidate],
    )
    marker = len(fake_gateway.events)

    apply_reconciliation(plan, fake_gateway, dependencies, _IMMEDIATE_POLICY)

    kinds = _kinds(fake_gateway, marker)
    first_create = kinds.index("trace_create")
    last_candidate_verify = len(kinds) - 1 - kinds[::-1].index("candidate_verify")
    first_retire = kinds.index("trace_retire")
    assert last_candidate_verify < first_create < first_retire
    assert kinds.count("candidate_verify") == 2


# ---------------------------------------------------------------------------
# Apply: unchanged skips and winner selection
# ---------------------------------------------------------------------------


def test_unchanged_rerun_with_search_indexed_candidates_converges_without_mutation(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _replay_turn()
    candidate, created = _previous_import(fake_gateway, turn, _request())
    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)
    mutations = len(fake_gateway.mutations)

    plan = _apply(fake_gateway, run, scans, turns, [candidate])

    assert [action.action for action in plan.results] == [ACTION_SKIPPED]
    assert plan.results[0].reason == "unchanged"
    assert plan.results[0].trace_id == created.trace_id
    assert plan.results[0].retired_trace_ids == ()
    # ZERO mutations: nothing was created, uploaded, re-recorded, or retired.
    # The only post-marker events are the candidate's own artifact integrity
    # verification (inside candidate verification) and the verification record.
    assert fake_gateway.mutations[mutations:] == []
    assert _kinds(fake_gateway, marker) == ["artifact_verify", "candidate_verify"]
    assert fake_gateway.has_trace(created.trace_id)
    verification = plan.verifications[created.trace_id]
    assert verification.matches is True
    assert verification.error_code is None


def test_winner_is_newest_then_lexicographically_greatest_trace_id(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    older, older_created = _previous_import(fake_gateway, turn, _request())
    newer, newer_created = _previous_import(fake_gateway, turn, _request())
    assert newer_created.trace_id > older_created.trace_id
    older = replace(older, request_time_ms=1_000)
    newer = replace(newer, request_time_ms=2_000)

    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [older, newer])

    result = plan.results[0]
    assert result.action == ACTION_SKIPPED
    assert result.reason == "unchanged"
    assert result.trace_id == newer_created.trace_id
    assert result.retired_trace_ids == (older_created.trace_id,)
    assert not fake_gateway.has_trace(older_created.trace_id)
    assert fake_gateway.has_trace(newer_created.trace_id)

    # Equal creation timestamps break the tie by the greatest trace ID.
    third, third_created = _previous_import(fake_gateway, turn, _request())
    tied_third = replace(third, request_time_ms=5_000)
    tied_newer = replace(newer, request_time_ms=5_000)
    plan = _apply(fake_gateway, run, scans, turns, [tied_newer, tied_third])
    assert plan.results[0].trace_id == max(third_created.trace_id, newer_created.trace_id)


def test_crash_leftover_without_artifact_is_replaced_and_retired(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    request = _request()
    created = fake_gateway.create_trace(request)  # a crash leftover: no artifact was ever uploaded
    leftover = OwnedTraceCandidate(
        trace_id=created.trace_id,
        session_id="ses-1",
        turn_key=request.metadata["opencode.turn_key"],
        request_time_ms=1_000,
        experiment_id=request.experiment_id,
        source_fingerprint=request.metadata["opencode.source_fingerprint"],
        source_schema=request.metadata["opencode.source_schema"],
        importer_version=request.metadata["opencode.importer_version"],
        mapping_version=request.metadata["opencode.mapping_version"],
        artifact_schema_version=request.metadata.get("opencode.artifact_schema_version"),
        catalog_version=request.metadata.get("opencode.catalog_version"),
        redaction_mode=request.metadata["opencode.redaction_mode"],
        render_fingerprint=request.metadata["opencode.render_fingerprint"],
        operation_key=request.metadata["opencode.operation_key"],
        turn_index=int(request.metadata["opencode.turn_index"]),
        importer_marker=request.metadata["opencode.importer"],
    )
    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [leftover])

    result = plan.results[0]
    assert result.action == ACTION_UPDATED
    assert result.trace_id is not None and result.trace_id != created.trace_id
    assert result.retired_trace_ids == (created.trace_id,)
    assert plan.verifications[created.trace_id].matches is False
    assert _kinds(fake_gateway, marker) == [
        "candidate_verify",
        "trace_create",
        "trace_verify",
        "artifact_upload",
        "artifact_verify",
        "trace_tag_set",
        "trace_tag_set",
        "trace_retire",
    ]


def test_crash_between_upload_and_recording_is_replaced_and_retired(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    # The artifact was uploaded and verified, but the digest was never recorded
    # on the trace: an incomplete candidate.
    candidate, created = _previous_import(fake_gateway, turn, _request(), record=False)
    assert candidate.artifact_sha256 is None
    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [candidate])

    result = plan.results[0]
    assert result.action == ACTION_UPDATED
    assert result.trace_id is not None and result.trace_id != created.trace_id
    assert result.retired_trace_ids == (created.trace_id,)
    verification = plan.verifications[created.trace_id]
    assert verification.matches is False
    assert verification.reason == "artifact_digest_missing"
    # The incomplete candidate was replaced and retired only after the
    # verified replacement (with its recorded digest) existed.
    assert _kinds(fake_gateway, marker) == [
        "candidate_verify",
        "trace_create",
        "trace_verify",
        "artifact_upload",
        "artifact_verify",
        "trace_tag_set",
        "trace_tag_set",
        "trace_retire",
    ]
    replacement = fake_gateway.stored_trace(result.trace_id or "")
    assert replacement is not None
    assert replacement.tags["opencode.artifact_sha256"] is not None
    assert not fake_gateway.has_trace(created.trace_id)


def test_tampered_artifact_fails_integrity_and_is_replaced(fake_gateway: FakeMlflowGateway, tmp_path: Path) -> None:
    turn = _replay_turn()
    request = _request()
    candidate, created = _previous_import(fake_gateway, turn, request)
    # The stored artifact bytes changed after the write: the digest recorded
    # on the trace no longer describes the stored bytes.
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(b'{"tampered": "artifact"}')
    fake_gateway.upload_named_json(created.trace_id, request.artifact_relative_path, tampered)
    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [candidate])

    result = plan.results[0]
    assert result.action == ACTION_UPDATED
    assert result.retired_trace_ids == (created.trace_id,)
    verification = plan.verifications[created.trace_id]
    assert verification.matches is False
    assert verification.reason == "artifact_mismatch"
    assert "candidate_verify" in _kinds(fake_gateway, marker)


def test_changed_turn_is_replaced_without_deleting_the_old_first(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    candidate, created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old" + "0" * 61))
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(source_fingerprint="new" + "0" * 61))
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [candidate])

    result = plan.results[0]
    assert result.action == ACTION_UPDATED
    assert result.trace_id is not None
    assert result.retired_trace_ids == (created.trace_id,)
    assert _kinds(fake_gateway, marker) == [
        "candidate_verify",
        "trace_create",
        "trace_verify",
        "artifact_upload",
        "artifact_verify",
        "trace_tag_set",
        "trace_tag_set",
        "trace_retire",
    ]
    # The replacement exists; the old trace was retired only after it verified.
    assert fake_gateway.has_trace(result.trace_id)
    assert not fake_gateway.has_trace(created.trace_id)


# ---------------------------------------------------------------------------
# Apply: rollback and cleanup failures
# ---------------------------------------------------------------------------


def test_replacement_rolls_back_new_trace_when_trace_verification_fails(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    candidate, created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old" + "0" * 61))
    fake_gateway.fail_next_trace_verification = 1
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(source_fingerprint="new" + "0" * 61))
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [candidate])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "trace_verification_failed"
    assert result.trace_id is not None and result.trace_id != created.trace_id
    assert result.retired_trace_ids == (result.trace_id,)
    assert created.trace_id not in result.affected_trace_ids
    # The old trace stays; the only retirement is the rolled-back replacement.
    assert fake_gateway.has_trace(created.trace_id)
    assert "trace_retire" in _kinds(fake_gateway, marker)
    assert set(fake_gateway.trace_ids()) == {created.trace_id}


def test_replacement_rolls_back_new_trace_when_artifact_verification_fails(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _replay_turn()
    candidate, created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old" + "0" * 61))
    fake_gateway.fail_artifact_verification = True
    fake_gateway.artifact_verification_exempt_trace_ids = {created.trace_id}
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(source_fingerprint="new" + "0" * 61))
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [candidate])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "artifact_verification_failed"
    assert result.retired_trace_ids == (result.trace_id,)
    # The old trace remains active; the new trace was rolled back.
    assert fake_gateway.has_trace(created.trace_id)
    assert set(fake_gateway.trace_ids()) == {created.trace_id}
    assert created.trace_id not in result.affected_trace_ids
    assert "trace_retire" in _kinds(fake_gateway, marker)  # the rollback retirement


def test_failed_rollback_reports_both_trace_ids_as_a_conflict(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    candidate, created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old" + "0" * 61))
    fake_gateway.fail_artifact_verification = True
    fake_gateway.artifact_verification_exempt_trace_ids = {created.trace_id}

    def retire_fails_new_traces(experiment_id: str, trace_id: str) -> None:
        if trace_id != created.trace_id:
            raise ItemImportError("trace_retirement_failed", "synthetic retirement failure")

    fake_gateway.retire_hook = retire_fails_new_traces
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(source_fingerprint="new" + "0" * 61))
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [candidate])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "duplicate_owned_identity"
    assert result.trace_id is not None
    assert set(result.affected_trace_ids) == {created.trace_id, result.trace_id}
    # The old valid trace is retained; both owned traces remain for recovery.
    assert fake_gateway.has_trace(created.trace_id)
    assert len(fake_gateway.trace_ids()) == 2


def test_failed_rollback_reports_all_old_candidates_and_new_trace(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    first, first_created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old-1" + "0" * 59))
    second, second_created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old-2" + "0" * 59))
    fake_gateway.fail_artifact_verification = True
    fake_gateway.artifact_verification_exempt_trace_ids = {first_created.trace_id, second_created.trace_id}

    def retire_only_old_traces(_experiment_id: str, trace_id: str) -> None:
        if trace_id not in {first_created.trace_id, second_created.trace_id}:
            raise ItemImportError("trace_retirement_failed", "synthetic retirement failure")

    fake_gateway.retire_hook = retire_only_old_traces
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(source_fingerprint="new" + "0" * 61))
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [first, second])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "duplicate_owned_identity"
    assert result.trace_id is not None
    expected_ids = {first_created.trace_id, second_created.trace_id, result.trace_id}
    assert set(result.affected_trace_ids) == expected_ids
    assert set(result.conflicting_trace_ids) == expected_ids
    assert len(result.affected_trace_ids) == len(expected_ids)
    assert len(result.conflicting_trace_ids) == len(expected_ids)
    assert fake_gateway.has_trace(first_created.trace_id)
    assert fake_gateway.has_trace(second_created.trace_id)
    assert fake_gateway.has_trace(result.trace_id)


def test_cleanup_conflict_with_multiple_old_candidates_reports_all_ids(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    first, first_created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old-1" + "0" * 59))
    second, second_created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old-2" + "0" * 59))
    # The newest old candidate is the primary cleanup target.
    first = replace(first, request_time_ms=2_000)
    second = replace(second, request_time_ms=1_000)

    def remove_primary_ownership(_experiment_id: str, trace_id: str) -> None:
        if trace_id == first_created.trace_id:
            stored = fake_gateway.stored_trace(trace_id)
            assert stored is not None
            stored.metadata.pop("opencode.importer", None)

    fake_gateway.retire_hook = remove_primary_ownership
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(source_fingerprint="new" + "0" * 61))
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [first, second])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert result.trace_id is not None
    expected_ids = {first_created.trace_id, second_created.trace_id, result.trace_id}
    assert set(result.affected_trace_ids) == expected_ids
    assert set(result.conflicting_trace_ids) == expected_ids
    assert len(result.affected_trace_ids) == len(expected_ids)
    assert len(result.conflicting_trace_ids) == len(expected_ids)
    assert fake_gateway.has_trace(first_created.trace_id)
    assert fake_gateway.has_trace(second_created.trace_id)
    assert not fake_gateway.has_trace(result.trace_id)


def test_redundant_cleanup_conflict_reports_all_old_and_new_ids(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    first, first_created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old-1" + "0" * 59))
    second, second_created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old-2" + "0" * 59))
    first = replace(first, request_time_ms=2_000)
    second = replace(second, request_time_ms=1_000)

    def remove_redundant_ownership(_experiment_id: str, trace_id: str) -> None:
        if trace_id == second_created.trace_id:
            stored = fake_gateway.stored_trace(trace_id)
            assert stored is not None
            stored.metadata.pop("opencode.importer", None)

    fake_gateway.retire_hook = remove_redundant_ownership
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(source_fingerprint="new" + "0" * 61))
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [first, second])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert result.trace_id is not None
    expected_ids = {first_created.trace_id, second_created.trace_id, result.trace_id}
    assert set(result.affected_trace_ids) == expected_ids
    assert set(result.conflicting_trace_ids) == expected_ids
    assert not fake_gateway.has_trace(first_created.trace_id)
    assert fake_gateway.has_trace(second_created.trace_id)
    assert fake_gateway.has_trace(result.trace_id)


def test_old_retirement_failure_compensates_by_retiring_the_new_trace(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    candidate, created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old" + "0" * 61))
    fake_gateway.fail_retire_trace_ids = {created.trace_id}
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(source_fingerprint="new" + "0" * 61))
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [candidate])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "retirement_failed"
    assert result.retired_trace_ids == (result.trace_id,)
    # The old trace is never deleted on uncertainty; the new one is compensated away.
    assert fake_gateway.has_trace(created.trace_id)
    assert set(fake_gateway.trace_ids()) == {created.trace_id}


def test_redundant_cleanup_failure_keeps_winner_and_reports_affected_ids(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    winner_candidate, winner_created = _previous_import(fake_gateway, turn, _request())
    leftover_candidate, leftover_created = _previous_import(fake_gateway, turn, _request())
    # The intended winner is the newest; the leftover is an older duplicate.
    winner_candidate = replace(winner_candidate, request_time_ms=2_000_000_000_000)
    leftover_candidate = replace(leftover_candidate, request_time_ms=1_000_000_000_000)
    fake_gateway.fail_retire_trace_ids = {leftover_created.trace_id}
    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [winner_candidate, leftover_candidate])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.reason == "unchanged"
    assert result.trace_id == winner_created.trace_id
    assert result.error_code == "replacement_cleanup_failed"
    assert result.affected_trace_ids == (leftover_created.trace_id,)
    # The valid winner is kept.
    assert fake_gateway.has_trace(winner_created.trace_id)


def test_verify_all_retains_contracts_not_replay_or_root_payloads(fake_gateway):
    from opencode_mlflow.mlflow.reconciliation import _verify_all_candidates

    turn = _replay_turn()
    candidate, _created = _previous_import(fake_gateway, turn, _request())
    scans, turns = _single_scan_turns()
    plan = plan_reconciliation(scans, turns, [candidate])
    run = _ScriptedRun(fake_gateway, turn, _request)
    works = _verify_all_candidates(plan, fake_gateway, run.dependencies(), _IMMEDIATE_POLICY)
    assert len(works) == 1
    assert all(
        work.replay is None and work.request is None and work.expected_contract is not None for work in works.values()
    )


def test_remapped_contract_mismatch_fails_before_mutation(fake_gateway):
    turn = _replay_turn()
    candidate, created = _previous_import(fake_gateway, turn, _request())
    scans, turns = _single_scan_turns()
    calls = []

    def request():
        calls.append(1)
        return _request(source_fingerprint="a" * 64 if len(calls) == 1 else "b" * 64)

    marker = len(fake_gateway.mutations)
    plan = _apply(fake_gateway, _ScriptedRun(fake_gateway, turn, request), scans, turns, [candidate])
    assert plan.results[0].action == ACTION_FAILED
    assert plan.results[0].error_code == "source_consistency_mismatch"
    assert fake_gateway.has_trace(created.trace_id)
    assert fake_gateway.mutations[marker:] == []


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_known_partial_create_is_rolled_back_without_touching_old(fake_gateway, rollback_fails):
    from opencode_mlflow.errors import PartialTraceCreateError

    turn = _replay_turn()
    old, old_created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old" + "0" * 61))
    original = fake_gateway.create_trace
    partial_ids = []

    def partial(request):
        created = original(request)
        partial_ids.append(created.trace_id)
        if rollback_fails:
            fake_gateway.fail_retire_trace_ids.add(created.trace_id)
        raise PartialTraceCreateError(created, RuntimeError("synthetic child failure"))

    fake_gateway.create_trace = partial
    scans, turns = _single_scan_turns()
    plan = _apply(fake_gateway, _ScriptedRun(fake_gateway, turn, _request), scans, turns, [old])
    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.trace_id == partial_ids[0]
    assert result.error_code == ("partial_trace_cleanup_failed" if rollback_fails else "trace_create_failed")
    assert result.retired_trace_ids == (() if rollback_fails else (partial_ids[0],))
    assert fake_gateway.has_trace(old_created.trace_id)
    assert fake_gateway.has_trace(partial_ids[0]) == rollback_fails
    if rollback_fails:
        assert set(result.affected_trace_ids) == {old_created.trace_id, partial_ids[0]}
        assert set(result.conflicting_trace_ids) == {old_created.trace_id, partial_ids[0]}


def test_unverifiable_duplicate_identity_is_a_conflict_without_mutation(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    candidate, created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old" + "0" * 61))
    missing = owned("ses-1", turn_key=_turn_key("turn-1"), trace_id="tr-does-not-exist", request_time_ms=2_000)
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(source_fingerprint="new" + "0" * 61))
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [candidate, missing])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "duplicate_owned_identity"
    assert set(result.affected_trace_ids) == {candidate.trace_id, missing.trace_id}
    assert _kinds(fake_gateway, marker) == ["candidate_verify", "candidate_verify"]
    assert fake_gateway.has_trace(created.trace_id)


def test_matching_candidate_plus_unverifiable_duplicate_is_a_conflict_without_cleanup(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _replay_turn()
    matching, created = _previous_import(fake_gateway, turn, _request())
    unverifiable = owned(
        "ses-1",
        turn_key=_turn_key("turn-1"),
        trace_id="tr-unverifiable-duplicate",
        request_time_ms=2_000,
    )
    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)
    mutations = len(fake_gateway.mutations)

    plan = _apply(fake_gateway, run, scans, turns, [matching, unverifiable])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "duplicate_owned_identity"
    assert set(result.affected_trace_ids) == {matching.trace_id, unverifiable.trace_id}
    assert result.retired_trace_ids == ()
    assert _kinds(fake_gateway, marker) == ["artifact_verify", "candidate_verify", "candidate_verify"]
    assert fake_gateway.mutations[mutations:] == []
    assert fake_gateway.has_trace(created.trace_id)


def test_ownership_removed_after_candidate_lookup_is_a_conflict_without_replacement(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _replay_turn()
    candidate, created = _previous_import(fake_gateway, turn, _request(source_fingerprint="old" + "0" * 61))
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(source_fingerprint="new" + "0" * 61))
    scans, turns = _single_scan_turns()
    plan = plan_reconciliation(scans, turns, [candidate])

    stored = fake_gateway.stored_trace(created.trace_id)
    assert stored is not None
    stored.metadata.pop("opencode.importer")
    marker = len(fake_gateway.mutations)

    apply_reconciliation(plan, fake_gateway, run.dependencies(), _IMMEDIATE_POLICY)

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert result.affected_trace_ids == (created.trace_id,)
    assert fake_gateway.mutations[marker:] == []
    assert fake_gateway.has_trace(created.trace_id)


def test_malformed_contract_on_final_candidate_fetch_is_a_conflict_without_replacement(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _replay_turn()
    candidate, created = _previous_import(fake_gateway, turn, _request(source_fingerprint="0" * 64))
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(source_fingerprint="1" * 64))
    scans, turns = _single_scan_turns()
    plan = plan_reconciliation(scans, turns, [candidate])
    stored = fake_gateway.stored_trace(created.trace_id)
    assert stored is not None
    stored.metadata["opencode.source_fingerprint"] = "malformed"
    marker = len(fake_gateway.mutations)

    apply_reconciliation(plan, fake_gateway, run.dependencies(), _IMMEDIATE_POLICY)

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert result.affected_trace_ids == (created.trace_id,)
    assert fake_gateway.mutations[marker:] == []
    assert fake_gateway.has_trace(created.trace_id)


@pytest.mark.parametrize(
    ("metadata_key", "mutated_value"),
    [
        ("opencode.turn_index", "2"),
        ("opencode.importer_version", "changed-version"),
        ("opencode.mapping_version", "changed-mapping"),
        ("opencode.artifact_schema_version", "changed-artifact-schema"),
        ("opencode.redaction_mode", RedactionMode.REDACTED.value),
    ],
)
def test_creation_contract_mutation_is_not_an_unchanged_match(
    fake_gateway: FakeMlflowGateway,
    metadata_key: str,
    mutated_value: str,
) -> None:
    turn = _replay_turn()
    request = _request()
    candidate, created = _previous_import(fake_gateway, turn, request)
    stored = fake_gateway.stored_trace(created.trace_id)
    assert stored is not None
    stored.metadata[metadata_key] = mutated_value
    expected = CandidateContract.from_create_request(request, "", 0)

    verification = fake_gateway.verify_owned_candidate(candidate, expected)

    assert verification.matches is False
    assert verification.error_code is None
    assert verification.reason == "metadata_mismatch"
    assert fake_gateway.has_trace(created.trace_id)


def test_stale_retirement_race_refuses_to_delete_after_ownership_marker_disappears(
    fake_gateway: FakeMlflowGateway,
) -> None:
    candidate = owned("session-a", turn_key=_turn_key("removed-turn"), trace_id="tr-stale")
    fake_gateway.owned_traces([candidate])
    scans = [_scan("session-a", logical_turn_count=1)]
    plan = plan_reconciliation(scans, [], [candidate])

    stored = fake_gateway.stored_trace(candidate.trace_id)
    assert stored is not None

    def remove_marker_before_delete(_experiment_id: str, _trace_id: str) -> None:
        stored.metadata.pop("opencode.importer", None)

    fake_gateway.retire_hook = remove_marker_before_delete
    marker = len(fake_gateway.mutations)
    apply_reconciliation(
        plan,
        fake_gateway,
        ReconciliationDependencies(
            replay=lambda _entry: _replay_turn(),
            map_trace=lambda _turn: _request(),
            artifact_snapshot=lambda _turn: "d" * 64,
            upload_artifact=lambda *_args: None,  # type: ignore[return-value]
        ),
        _IMMEDIATE_POLICY,
    )

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert fake_gateway.mutations[marker:] == []
    assert fake_gateway.has_trace(candidate.trace_id)


def test_single_unverifiable_candidate_fails_the_item_without_mutation(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    missing = owned("ses-1", turn_key=_turn_key("turn-1"), trace_id="tr-does-not-exist")
    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [missing])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code is not None
    assert _kinds(fake_gateway, marker) == ["candidate_verify"]


def test_replayed_snapshot_inconsistency_fails_the_item_before_mutation(fake_gateway: FakeMlflowGateway) -> None:
    turn = _replay_turn()
    candidate, _created = _previous_import(fake_gateway, turn, _request())
    run = _ScriptedRun(fake_gateway, turn, lambda: _request(), snapshot_fingerprint="0" * 64)
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [candidate])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "source_consistency_mismatch"
    assert _kinds(fake_gateway, marker) == []


# ---------------------------------------------------------------------------
# Apply: stale retirement and ambiguous creates
# ---------------------------------------------------------------------------


def test_stale_retirement_is_executed_and_reported(fake_gateway: FakeMlflowGateway) -> None:
    stale = fake_gateway.owned_traces([owned("ses-1", turn_key=_turn_key("removed-turn"), trace_id="tr-stale")])[0]
    run = _ScriptedRun(fake_gateway, _replay_turn(), lambda: _request())
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [stale])

    assert [action.action for action in plan.results] == [ACTION_IMPORTED, ACTION_RETIRED]
    retired = plan.results[1]
    assert retired.trace is stale
    assert retired.trace_id == "tr-stale"
    assert retired.retired_trace_ids == ("tr-stale",)
    assert not fake_gateway.has_trace("tr-stale")
    assert "trace_retire" in _kinds(fake_gateway, marker)


def test_ambiguous_unverifiable_survivor_blocks_retry_and_preserves_all_ids(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _replay_turn()
    old_request = _request(source_fingerprint="old" + "0" * 61, render_fingerprint="1" * 64, operation_key="2" * 64)
    candidate, created = _previous_import(fake_gateway, turn, old_request)
    fake_gateway.ambiguous_next_creates = 1  # the create "times out" after taking effect
    new_request = _request(source_fingerprint="new" + "0" * 61, render_fingerprint="3" * 64, operation_key="4" * 64)
    run = _ScriptedRun(fake_gateway, turn, lambda: new_request)
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = plan_reconciliation(scans, turns, [candidate])
    apply_reconciliation(
        plan,
        fake_gateway,
        run.dependencies(),
        RetryPolicy(max_attempts=2, initial_delay_seconds=0.0, max_delay_seconds=0.0),
    )

    kinds = _kinds(fake_gateway, marker)
    # The survivor (created by the ambiguous attempt) was searched and
    # verified, but its incomplete artifact is not permission to create again.
    assert kinds.count("trace_create") == 1
    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert set(result.affected_trace_ids) == {created.trace_id, "fake-tr-2"}
    assert set(result.conflicting_trace_ids) == {created.trace_id, "fake-tr-2"}
    assert set(fake_gateway.trace_ids()) == {created.trace_id, "fake-tr-2"}


def test_ambiguous_unverifiable_survivor_stops_without_a_second_create(
    fake_gateway: FakeMlflowGateway,
) -> None:
    """An ambiguous survivor is never followed by another non-idempotent create."""
    turn = _replay_turn()
    old_request = _request(source_fingerprint="old" + "0" * 61)
    old_candidate, old_created = _previous_import(fake_gateway, turn, old_request)
    fake_gateway.ambiguous_next_creates = 1
    # The old trace is fake-tr-1; the ambiguous create stores fake-tr-2 before
    # reporting its timeout. Make that survivor unverifiable during recovery.
    survivor_id = "fake-tr-2"
    fake_gateway.candidate_verification_error_ids.add(survivor_id)
    new_request = _request(source_fingerprint="new" + "0" * 61, operation_key="4" * 64)
    run = _ScriptedRun(fake_gateway, turn, lambda: new_request)
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = plan_reconciliation(scans, turns, [old_candidate])
    apply_reconciliation(
        plan,
        fake_gateway,
        run.dependencies(),
        RetryPolicy(max_attempts=2, initial_delay_seconds=0.0, max_delay_seconds=0.0),
    )

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert set(result.affected_trace_ids) == {old_created.trace_id, survivor_id}
    assert [event.kind for event in fake_gateway.events[marker:] if event.kind == "trace_create"] == ["trace_create"]
    assert set(fake_gateway.trace_ids()) == {old_created.trace_id, survivor_id}


def test_ambiguous_final_attempt_searches_and_preserves_unverifiable_survivor(
    fake_gateway: FakeMlflowGateway,
) -> None:
    """The last transient create failure still performs safe recovery.

    Exhausting the create budget is not permission to lose the trace id that
    may have been created before the timeout, nor is it permission to issue a
    further create. An unverifiable survivor is an ownership conflict.
    """
    turn = _replay_turn()
    fake_gateway.ambiguous_next_creates = 1
    survivor_id = "fake-tr-1"
    fake_gateway.candidate_verification_error_ids.add(survivor_id)
    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = plan_reconciliation(scans, turns, [])
    apply_reconciliation(
        plan,
        fake_gateway,
        run.dependencies(),
        RetryPolicy(max_attempts=1, initial_delay_seconds=0.0, max_delay_seconds=0.0),
    )

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert result.affected_trace_ids == (survivor_id,)
    assert [event.kind for event in fake_gateway.events[marker:] if event.kind == "trace_create"] == ["trace_create"]
    assert fake_gateway.has_trace(survivor_id)


def test_ambiguous_search_failure_reports_survivors_seen_before_failed_page(
    fake_gateway: FakeMlflowGateway,
) -> None:
    """A partial search is not an authoritative empty result."""
    turn = _replay_turn()
    fake_gateway.ambiguous_next_creates = 1
    survivor_id = "fake-tr-1"
    original_iter = fake_gateway.iter_owned_traces

    def search_fails_after_yield(experiment_id: str):
        yield from original_iter(experiment_id)
        raise RuntimeError("transport detail must not be reported")

    fake_gateway.iter_owned_traces = search_fails_after_yield  # type: ignore[method-assign]
    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(
        fake_gateway,
        run,
        scans,
        turns,
        [],
    )

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert result.affected_trace_ids == (survivor_id,)
    assert result.conflicting_trace_ids == (survivor_id,)
    assert [event.kind for event in fake_gateway.events[marker:] if event.kind == "trace_create"] == ["trace_create"]
    assert fake_gateway.has_trace(survivor_id)


def test_ambiguous_search_failure_reports_old_and_new_observed_ids(
    fake_gateway: FakeMlflowGateway,
) -> None:
    turn = _replay_turn()
    old_request = _request(source_fingerprint="old" + "0" * 61, operation_key="o" * 64)
    old_candidate, old_created = _previous_import(fake_gateway, turn, old_request)
    fake_gateway.ambiguous_next_creates = 1
    survivor_id = "fake-tr-2"
    original_iter = fake_gateway.iter_owned_traces

    def search_fails_after_yield(experiment_id: str):
        yield from original_iter(experiment_id)
        raise RuntimeError("synthetic search failure")

    fake_gateway.iter_owned_traces = search_fails_after_yield  # type: ignore[method-assign]
    request = _request(source_fingerprint="new" + "0" * 61, operation_key="n" * 64)
    run = _ScriptedRun(fake_gateway, turn, lambda: request)
    scans, turns = _single_scan_turns()

    plan = _apply(fake_gateway, run, scans, turns, [old_candidate])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert set(result.affected_trace_ids) == {old_created.trace_id, survivor_id}
    assert set(result.conflicting_trace_ids) == {old_created.trace_id, survivor_id}


def test_ambiguous_recovery_verifies_all_survivors_before_conflicting(
    fake_gateway: FakeMlflowGateway,
) -> None:
    """An unverifiable survivor or duplicate blocks another non-idempotent create."""
    turn = _replay_turn()
    request = _request()
    fake_gateway.ambiguous_next_creates = 1
    first_id = "fake-tr-1"
    second_id = "fake-tr-duplicate"
    fake_gateway.candidate_verification_error_ids.update({first_id, second_id})

    def all_survivors(_experiment_id: str):
        metadata = dict(request.metadata)
        yield OwnedTrace(first_id, 1, {}, metadata)
        yield OwnedTrace(second_id, 2, {}, metadata)

    fake_gateway.iter_owned_traces = all_survivors  # type: ignore[method-assign]
    run = _ScriptedRun(fake_gateway, turn, lambda: request)
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [])

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert set(result.affected_trace_ids) == {first_id, second_id}
    assert set(result.conflicting_trace_ids) == {first_id, second_id}
    assert [event.kind for event in fake_gateway.events[marker:] if event.kind == "trace_create"] == ["trace_create"]
    assert [event.kind for event in fake_gateway.events[marker:] if event.kind == "candidate_verify"] == [
        "candidate_verify",
        "candidate_verify",
    ]


def test_ambiguous_multiple_verified_survivors_are_deduplicated_before_cleanup(
    fake_gateway: FakeMlflowGateway,
) -> None:
    """Every exact survivor is verified, then only redundant ids are retired once."""
    turn = _replay_turn()
    old_request = _request(source_fingerprint="old" + "0" * 61, operation_key="o" * 64)
    request = _request(source_fingerprint="new" + "0" * 61)
    old, _old_created = _previous_import(fake_gateway, turn, old_request)
    _first, _first_created = _previous_import(fake_gateway, turn, request)
    second, second_created = _previous_import(fake_gateway, turn, request)

    def ambiguous_without_survivor(_request: TraceCreateRequest):
        raise FakeMlflowException("synthetic timeout", "TEMPORARILY_UNAVAILABLE")

    fake_gateway.create_trace = ambiguous_without_survivor  # type: ignore[method-assign]
    run = _ScriptedRun(fake_gateway, turn, lambda: request)
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = _apply(fake_gateway, run, scans, turns, [old])

    result = plan.results[0]
    assert result.action == ACTION_UPDATED
    assert len(result.retired_trace_ids) == 2
    assert result.trace_id == second_created.trace_id
    assert result.trace_id not in result.retired_trace_ids
    assert len(set(result.retired_trace_ids)) == len(result.retired_trace_ids)
    assert len(fake_gateway.trace_ids()) == 1
    assert [event.kind for event in fake_gateway.events[marker:] if event.kind == "trace_retire"].count(
        "trace_retire"
    ) == 2


def test_ambiguous_survivor_with_changed_turn_identity_stops_without_retrying_create(
    fake_gateway: FakeMlflowGateway,
) -> None:
    """A survivor with the idempotency key but a changed turn is a conflict."""
    turn = _replay_turn()
    fake_gateway.ambiguous_next_creates = 1
    original_iter = fake_gateway.iter_owned_traces

    def changed_survivor(experiment_id: str):
        for trace in original_iter(experiment_id):
            if trace.trace_id == "fake-tr-1":
                trace.metadata["opencode.turn_key"] = _turn_key("different-turn")  # type: ignore[index]
            yield trace

    fake_gateway.iter_owned_traces = changed_survivor  # type: ignore[method-assign]
    run = _ScriptedRun(fake_gateway, turn, lambda: _request())
    scans, turns = _single_scan_turns()
    marker = len(fake_gateway.events)

    plan = plan_reconciliation(scans, turns, [])
    apply_reconciliation(
        plan,
        fake_gateway,
        run.dependencies(),
        RetryPolicy(max_attempts=2, initial_delay_seconds=0.0, max_delay_seconds=0.0),
    )

    result = plan.results[0]
    assert result.action == ACTION_FAILED
    assert result.error_code == "ownership_conflict"
    assert result.affected_trace_ids == ("fake-tr-1",)
    assert [event.kind for event in fake_gateway.events[marker:] if event.kind == "trace_create"] == ["trace_create"]


# ---------------------------------------------------------------------------
# Candidate contracts and the default artifact upload
# ---------------------------------------------------------------------------


def test_candidate_contract_is_built_from_the_mapped_request() -> None:
    request = _request()
    contract = CandidateContract.from_create_request(request, "1" * 64, 42)

    assert contract.session_id == "ses-1"
    assert contract.turn_key == request.metadata["opencode.turn_key"]
    assert contract.source_fingerprint == "a" * 64
    assert contract.source_schema == "current"
    assert contract.render_fingerprint == "b" * 64
    assert contract.operation_key == "c" * 64
    assert contract.artifact_relative_path == request.artifact_relative_path
    assert contract.artifact_sha256 == "1" * 64
    assert contract.artifact_size == 42
    assert contract.ownership_key == OwnershipKey("ses-1", request.metadata["opencode.turn_key"])


def test_ownership_metadata_keys_match_the_real_mapper() -> None:
    turn = _replay_turn()
    context = MappingContext(
        importer_version="test-importer",
        mapping_version=MAPPING_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        source_schema="current",
        redaction_mode=RedactionMode.VERBATIM,
        experiment_id="exp-1",
        catalog_version=SCHEMA_CATALOG_VERSION,
    )
    request = map_turn_to_trace(turn, context).to_create_request()

    contract = CandidateContract.from_create_request(request, "2" * 64, 7)

    assert contract.session_id == turn.session_header.session_id
    assert contract.turn_key == turn.summary.turn_key
    assert contract.source_schema == "current"
    assert contract.source_fingerprint == request.metadata["opencode.source_fingerprint"]
    assert contract.render_fingerprint == request.metadata["opencode.render_fingerprint"]
    assert contract.operation_key == request.expected_operation_key


def test_created_identity_map_matches_the_mapper_artifact_section() -> None:
    created = CreatedTrace(
        trace_id="tr-1",
        root_span_id="span-1",
        span_ids_by_source={"msg-1": "span-2", "msg-2": "span-3"},
        verified_metadata={},
        root_status="OK",
    )
    envelope = ArtifactIdentityEnvelope(
        session_id="ses-1",
        turn_key=_turn_key("turn-1"),
        render_fingerprint="b" * 64,
        snapshot_fingerprint="d" * 64,
        artifact_relative_path="opencode-mlflow/v1/sessions/ses-1/turns/x/b-d.json",
    )
    bound = bind_mlflow_identity_map(envelope, "tr-1", {"msg-1": "span-2", "msg-2": "span-3"})

    assert created_artifact_identity_map(created) == bound.artifact_section()


def test_default_upload_artifact_streams_and_uploads_the_bound_document(
    fake_gateway: FakeMlflowGateway, tmp_path: Path
) -> None:
    turn = _replay_turn()
    request = _request()
    created = fake_gateway.create_trace(request)

    verification = upload_artifact(
        turn,
        created,
        request.artifact_relative_path,
        RedactionMode.VERBATIM,
        gateway=fake_gateway,
        temporary_directory=tmp_path,
    )

    assert verification.trace_id == created.trace_id
    assert verification.relative_path == request.artifact_relative_path
    stored = fake_gateway.stored_artifact(created.trace_id, request.artifact_relative_path)
    document = json.loads(stored.decode("utf-8"))
    assert document["identity_map"] == created_artifact_identity_map(created)
    assert verification.sha256 == hashlib.sha256(stored).hexdigest()
    assert verification.size == len(stored)
    fake_gateway.verify_named_json(
        created.trace_id, request.artifact_relative_path, verification.sha256, verification.size
    )


def test_default_upload_artifact_rejects_a_divergent_upload_result(
    fake_gateway: FakeMlflowGateway, tmp_path: Path
) -> None:
    turn = _replay_turn()
    request = _request()
    created = fake_gateway.create_trace(request)

    def lying_upload(trace_id: str, relative_path: str, local_file: Path, chunk_size: int = 8 * 1024 * 1024):
        from opencode_mlflow.mlflow.gateway import ArtifactVerification

        return ArtifactVerification(trace_id=trace_id, relative_path=relative_path, sha256="0" * 64, size=1)

    fake_gateway.upload_named_json = lying_upload  # type: ignore[method-assign]
    with pytest.raises(ItemImportError) as excinfo:
        upload_artifact(
            turn,
            created,
            request.artifact_relative_path,
            RedactionMode.VERBATIM,
            gateway=fake_gateway,
            temporary_directory=tmp_path,
        )

    assert excinfo.value.code == "artifact_upload_failed"
