"""Integration: the importer's real trace contract against a live MLflow server.

Drives the COMPLETE production pipeline over the sanitized synthetic
six-turn session — real source connection, registered current adapter,
normalized snapshot, replay, discovery, capability preflight, lock,
experiment resolution, reconciliation writes — against the real gateway
over the session's live MLflow server (``tests/integration/server.py``;
only the report clock and the retry policy are deterministic). The
module-scoped ``imported`` fixture performs ONE import under a unique
experiment name; every test then verifies the OBSERVABLE server state
through the public client API against expectations mapped independently
from the same source file through the same pure mapping
(:data:`tests.fixtures.six_turn_session.SESSION_ID` is the native
session id all six traces must share).

Verified contract (Task 14, Step 2):

- exact experiment resolution/creation/reuse (created once by exact
  name, resolved and reused by the unchanged rerun);
- one trace per logical turn — six traces for six turns, all owned;
- six distinct trace ids sharing ONE native session id
  (``mlflow.trace.session``);
- FULL root payload retrieval through ``get_trace`` (never search
  previews): complete root inputs/outputs round-trip byte-exact;
- nested model/tool/content spans and events with source ids, span
  types, parent linkage, payloads, and per-record attributes;
- status, timing, token, and cost metadata (root and LLM-span level);
- deterministic artifact paths and exact artifact bytes: every turn's
  recorded ``opencode.artifact_path`` equals the independently mapped
  deterministic path, the stored bytes hash to the recorded digest, and
  re-streaming the artifact document with the retrieved identity map
  reproduces the stored bytes exactly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from mlflow import MlflowClient
from mlflow.artifacts import get_artifact_repository

from opencode_mlflow.content.artifacts import ArtifactProvenance, stream_turn_artifact
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.mlflow.gateway import (
    RETRIEVED_SPAN_STATUS,
    TRACE_SESSION_METADATA_KEY,
    TRACE_STATE_FOR_ROOT_STATUS,
    TraceChildRequest,
    TraceCreateRequest,
)
from opencode_mlflow.source.replay import ReplayTurn
from tests.fixtures.six_turn_session import (
    SESSION_ID,
    build_six_turn_source_db,
    six_turn_session_fixture,
)
from tests.integration.import_runner import (
    RealImportRun,
    production_mapper,
    run_real_import,
    trace_ids_by_key,
    turn_entries,
)
from tests.integration.mlflow_profile import MAX_INLINE_PAYLOAD_BYTES
from tests.integration.server import LiveServer, unique_experiment_name

pytestmark = pytest.mark.integration

_TURNS = 6


@dataclass(frozen=True)
class ImportedSession:
    """The module's shared imported state: one run, its experiment, and traces."""

    run: RealImportRun
    client: MlflowClient
    experiment_id: str
    trace_ids_by_key: Mapping[str, str]
    source: Path
    tmp_root: Path


@dataclass(frozen=True)
class ExpectedMapping:
    """Expectations mapped independently from the same source through the pure mapping."""

    turns: tuple[ReplayTurn, ...]
    requests: tuple[TraceCreateRequest, ...]
    children: tuple[tuple[TraceChildRequest, ...], ...]


def _status_value(status: object) -> str | None:
    """The comparable status string of a retrieved span's status."""
    if status is None:
        return None
    code = getattr(status, "status_code", None)
    value = getattr(code, "value", None) if code is not None else getattr(status, "value", status)
    return None if value is None else str(value)


def _root_span(trace: Any) -> Any:
    """The trace's root span (the span without a parent)."""
    roots = [span for span in trace.data.spans if span.parent_id is None]
    assert len(roots) == 1, f"trace {trace.trace_id} has {len(roots)} root spans"
    return roots[0]


# ---------------------------------------------------------------------------
# Module fixtures: one import, one unchanged rerun, independent expectations
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def imported(live_server: LiveServer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[ImportedSession]:
    """One complete real-pipeline import of the six-turn session."""
    tmp_root = tmp_path_factory.mktemp("trace-contract-")
    source = build_six_turn_source_db(tmp_root)
    name = unique_experiment_name("trace-contract")
    run = run_real_import(
        server_uri=live_server.uri,
        source=source,
        experiment_name=name,
        tmp_root=tmp_root,
    )
    assert run.exit_status == 0, run.report["errors"]
    experiment = live_server.client.get_experiment_by_name(name)
    assert experiment is not None, "the import must have created the experiment"
    yield ImportedSession(
        run=run,
        client=live_server.client,
        experiment_id=experiment.experiment_id,
        trace_ids_by_key=trace_ids_by_key(run.report),
        source=source,
        tmp_root=tmp_root,
    )


@pytest.fixture(scope="module")
def rerun(imported: ImportedSession) -> RealImportRun:
    """One unchanged rerun over the same source and experiment."""
    run = run_real_import(
        server_uri=imported.client.tracking_uri,
        source=imported.source,
        experiment_name=imported.run.experiment_name,
        tmp_root=imported.tmp_root,
    )
    assert run.exit_status == 0, run.report["errors"]
    return run


@pytest.fixture(scope="module")
def expected(imported: ImportedSession) -> Iterator[ExpectedMapping]:
    """The independently mapped expectations over the exact imported source file."""
    with six_turn_session_fixture(imported.tmp_root, source=imported.source) as fixture:
        context = replace(
            fixture.context,
            experiment_id=imported.experiment_id,
            inline_limit_bytes=MAX_INLINE_PAYLOAD_BYTES,
        )
        requests = tuple(production_mapper(turn, context) for turn in fixture.turns)
        children = tuple(tuple(request.child_specs()) for request in requests)
        assert len(requests) == _TURNS
        yield ExpectedMapping(turns=tuple(fixture.turns), requests=requests, children=children)


# ---------------------------------------------------------------------------
# Experiment resolution, creation, and reuse
# ---------------------------------------------------------------------------


def test_import_creates_the_experiment_exactly_once_by_exact_name(imported: ImportedSession) -> None:
    report_experiment = imported.run.report["experiment"]
    assert isinstance(report_experiment, Mapping)
    assert report_experiment["status"] == "created"
    assert report_experiment["id"] == imported.experiment_id

    found = imported.client.get_experiment_by_name(imported.run.experiment_name)
    assert found is not None
    assert found.experiment_id == imported.experiment_id

    # Exact resolution, never name mangling or duplication: exactly one
    # experiment on the whole server carries this exact name.
    matches = [
        experiment
        for experiment in imported.client.search_experiments()
        if experiment.name == imported.run.experiment_name
    ]
    assert [experiment.experiment_id for experiment in matches] == [imported.experiment_id]


def test_unchanged_rerun_reuses_the_experiment_and_skips_every_turn(
    imported: ImportedSession, rerun: RealImportRun
) -> None:
    report_experiment = rerun.report["experiment"]
    assert isinstance(report_experiment, Mapping)
    assert report_experiment["status"] == "resolved"
    assert report_experiment["id"] == imported.experiment_id

    summary = rerun.report["summary"]
    assert isinstance(summary, Mapping)
    assert summary["turns_skipped"] == _TURNS
    assert summary["turns_imported"] == 0
    assert summary["turns_updated"] == 0
    assert summary["turns_failed"] == 0
    assert summary["retired_traces"] == 0

    # Every turn is skipped as unchanged with its ORIGINAL trace id: no
    # duplicate owned trace was created for any logical turn.
    for turn in turn_entries(rerun.report):
        assert turn["action"] == "skipped"
        assert turn["reason"] == "unchanged"
        assert turn["trace_id"] == imported.trace_ids_by_key[str(turn["turn_key"])]


# ---------------------------------------------------------------------------
# One trace per logical turn; six ids sharing one native session id
# ---------------------------------------------------------------------------


def _search_traces(imported: ImportedSession) -> list[Any]:
    page = imported.client.search_traces(
        locations=[imported.experiment_id],
        max_results=100,
        include_spans=False,
        flush=True,
    )
    return list(page)


def test_one_trace_per_logical_turn(imported: ImportedSession, expected: ExpectedMapping) -> None:
    traces = _search_traces(imported)

    assert len(traces) == _TURNS
    expected_keys = {request.metadata["opencode.turn_key"] for request in expected.requests}
    seen_keys: set[str] = set()
    for trace in traces:
        metadata = trace.info.trace_metadata
        assert metadata["opencode.importer"] == "opencode-mlflow"
        assert metadata["opencode.session_id"] == SESSION_ID
        turn_key = metadata["opencode.turn_key"]
        assert turn_key in expected_keys, turn_key
        assert turn_key not in seen_keys, f"a duplicate trace exists for turn {turn_key}"
        seen_keys.add(turn_key)
    assert seen_keys == expected_keys
    assert {trace.trace_id for trace in traces} == set(imported.trace_ids_by_key.values())


def test_six_distinct_trace_ids_share_one_native_session_id(imported: ImportedSession) -> None:
    traces = _search_traces(imported)
    trace_ids = [trace.trace_id for trace in traces]

    assert len(trace_ids) == _TURNS
    assert len(set(trace_ids)) == _TURNS, "every logical turn must own a distinct trace id"

    for trace_id in trace_ids:
        trace = imported.client.get_trace(trace_id, display=False, flush=True)
        assert trace.info.trace_metadata[TRACE_SESSION_METADATA_KEY] == SESSION_ID
        assert trace.info.trace_metadata["opencode.session_id"] == SESSION_ID


# ---------------------------------------------------------------------------
# Full root payload retrieval
# ---------------------------------------------------------------------------


def test_full_root_payload_round_trips_through_get_trace(imported: ImportedSession, expected: ExpectedMapping) -> None:
    for request in expected.requests:
        trace_id = imported.trace_ids_by_key[request.metadata["opencode.turn_key"]]
        trace = imported.client.get_trace(trace_id, display=False, flush=True)
        root = _root_span(trace)

        assert trace.trace_id == trace_id
        assert root.name == request.name
        assert root.span_type == "CHAIN"
        # The COMPLETE root values — never search previews — round-trip
        # byte-exact through full-payload retrieval.
        assert root.inputs == request.root_inputs
        assert root.outputs == request.root_outputs
        assert root.start_time_ns == request.start_time_ns
        assert root.end_time_ns == request.end_time_ns

        metadata = trace.info.trace_metadata
        assert metadata[TRACE_SESSION_METADATA_KEY] == request.native_session_id
        for key, value in request.metadata.items():
            # Every fixture metadata value is short enough to round-trip
            # exactly (the pinned server truncates only above 250 chars).
            assert len(value) <= 250
            assert metadata.get(key) == value, key
        for key, value in request.root_tags.items():
            assert len(value) <= 4096
            assert trace.info.tags.get(key) == value, key


# ---------------------------------------------------------------------------
# Nested spans/events with source ids
# ---------------------------------------------------------------------------


def test_nested_model_tool_content_spans_and_events(imported: ImportedSession, expected: ExpectedMapping) -> None:
    for turn_children, request in zip(expected.children, expected.requests, strict=True):
        trace_id = imported.trace_ids_by_key[request.metadata["opencode.turn_key"]]
        trace = imported.client.get_trace(trace_id, display=False, flush=True)
        root = _root_span(trace)
        spans_by_source = {
            span.attributes["opencode.source_key"]: span for span in trace.data.spans if span.parent_id is not None
        }
        span_ids = {span.span_id for span in trace.data.spans}
        assert set(spans_by_source) == {child.source_key for child in turn_children}

        for child in turn_children:
            span = spans_by_source[child.source_key]
            assert span.name == child.name
            assert span.span_type == child.span_type
            assert span.inputs == child.inputs
            assert span.outputs == child.outputs
            # The pinned server forces every non-ERROR span status to OK at
            # end time, so an ended UNSET child retrieves as OK.
            assert _status_value(span.status) == RETRIEVED_SPAN_STATUS.get(child.status, child.status)
            if child.start_time_ns is not None:
                assert span.start_time_ns == child.start_time_ns
            if child.end_time_ns is not None:
                assert span.end_time_ns == child.end_time_ns
            # Nested linkage: a child of the root points at the root span; a
            # child of another child points at that child's span.
            expected_parent = root if child.parent_source_key is None else spans_by_source[child.parent_source_key]
            assert span.parent_id == expected_parent.span_id
            # Source provenance attributes survive as a subset.
            for key, value in child.attributes.items():
                assert span.attributes.get(key) == value, (child.source_key, key)
            assert span.span_id in span_ids

    # Structural spot checks over the richest turn (turn 1): the model-call
    # result is an LLM span under the root, and its reasoning/tool/text
    # parts nest UNDER the model span — not flat under the root. (The
    # root-input prompt's own text part stays a child of the root.)
    first = expected.requests[0]
    trace_id = imported.trace_ids_by_key[first.metadata["opencode.turn_key"]]
    trace = imported.client.get_trace(trace_id, display=False, flush=True)
    root = _root_span(trace)
    spans_by_source = {
        span.attributes["opencode.source_key"]: span for span in trace.data.spans if span.parent_id is not None
    }
    model_span = spans_by_source["session_message:msg-synthetic-a1"]
    tool_span = spans_by_source["session_message:msg-synthetic-a1#content-1"]
    reasoning = spans_by_source["session_message:msg-synthetic-a1#content-0"]
    answer_text = spans_by_source["session_message:msg-synthetic-a1#content-2"]
    prompt_text = spans_by_source["session_message:msg-synthetic-u1#content-0"]
    assert model_span.name == "model synthetic-provider/synthetic-model"
    assert model_span.span_type == "LLM"
    assert model_span.parent_id == root.span_id
    assert tool_span.name == "tool bash"
    assert tool_span.span_type == "TOOL"
    assert tool_span.parent_id == model_span.span_id
    assert tool_span.inputs == {"command": "ls synthetic"}
    assert tool_span.outputs == "synthetic file list"
    assert reasoning.name == "reasoning part"
    assert reasoning.span_type == "event"
    assert reasoning.parent_id == model_span.span_id
    assert answer_text.name == "text part"
    assert answer_text.span_type == "UNKNOWN"
    assert answer_text.parent_id == model_span.span_id
    assert prompt_text.parent_id == root.span_id


# ---------------------------------------------------------------------------
# Status, timing, token, and cost metadata
# ---------------------------------------------------------------------------


def test_root_status_outcome_and_timing_metadata(imported: ImportedSession, expected: ExpectedMapping) -> None:
    expected_status_by_index = {1: "OK", 2: "OK", 3: "UNSET", 4: "ERROR", 5: "OK", 6: "OK"}
    expected_outcome_by_index = {
        1: "assistant_result",
        2: "assistant_result",
        3: "lifecycle_observed",
        4: "explicit_failure",
        5: "assistant_result",
        6: "assistant_result",
    }
    for request in expected.requests:
        turn_index = int(request.metadata["opencode.turn_index"])
        trace_id = imported.trace_ids_by_key[request.metadata["opencode.turn_key"]]
        trace = imported.client.get_trace(trace_id, display=False, flush=True)
        root = _root_span(trace)
        applied = request.root_status
        assert applied == expected_status_by_index[turn_index]

        # The pinned server's LiveSpan.end forces every non-ERROR status to
        # OK, so an incomplete turn (UNSET root) retrieves as span status OK
        # and trace state OK, while the source-known outcome stays recorded
        # in the opencode.outcome metadata.
        assert _status_value(root.status) == RETRIEVED_SPAN_STATUS.get(applied, applied)
        assert str(getattr(trace.info.state, "value", trace.info.state)) == TRACE_STATE_FOR_ROOT_STATUS.get(
            applied, applied
        )
        assert request.metadata["opencode.outcome"] == expected_outcome_by_index[turn_index]
        # Source-derived timing, never the local wall clock.
        assert root.start_time_ns == request.start_time_ns
        assert root.end_time_ns == request.end_time_ns
        assert "opencode.timing_estimated" not in trace.info.trace_metadata


def test_token_and_cost_metadata(imported: ImportedSession, expected: ExpectedMapping) -> None:
    # Session-level aggregates ride every root span (from the session row).
    for request in expected.requests:
        trace_id = imported.trace_ids_by_key[request.metadata["opencode.turn_key"]]
        trace = imported.client.get_trace(trace_id, display=False, flush=True)
        root = _root_span(trace)
        for key, value in request.root_attributes.items():
            assert root.attributes.get(key) == value, key
        metadata = trace.info.trace_metadata
        assert metadata["opencode.tokens_input"] == "310"
        assert metadata["opencode.tokens_output"] == "180"
        assert metadata["opencode.tokens_reasoning"] == "17"
        assert metadata["opencode.tokens_cache_read"] == "58"
        assert metadata["opencode.tokens_cache_write"] == "7"
        assert metadata["opencode.cost"] == "0.0055"

    # Per-message usage rides each model-call (LLM) span; the plain
    # assistant message without model metadata carries none.
    first = expected.requests[0]
    trace_id = imported.trace_ids_by_key[first.metadata["opencode.turn_key"]]
    trace = imported.client.get_trace(trace_id, display=False, flush=True)
    model_span = next(span for span in trace.data.spans if span.name == "model synthetic-provider/synthetic-model")
    assert model_span.attributes["opencode.tokens.input"] == 120
    assert model_span.attributes["opencode.tokens.output"] == 80
    assert model_span.attributes["opencode.tokens.reasoning"] == 12
    assert model_span.attributes["opencode.tokens.cache_read"] == 40
    assert model_span.attributes["opencode.tokens.cache_write"] == 4
    assert model_span.attributes["opencode.cost"] == 0.0025
    assert model_span.attributes["opencode.model"] == "synthetic-provider/synthetic-model"
    assert model_span.attributes["opencode.provider"] == "synthetic-provider"

    fifth = expected.requests[4]
    trace_id = imported.trace_ids_by_key[fifth.metadata["opencode.turn_key"]]
    trace = imported.client.get_trace(trace_id, display=False, flush=True)
    agent_span = next(span for span in trace.data.spans if span.name == "assistant message")
    assert agent_span.span_type == "AGENT"
    assert "opencode.tokens.input" not in agent_span.attributes
    assert "opencode.cost" not in agent_span.attributes


# ---------------------------------------------------------------------------
# Deterministic artifact paths and exact bytes
# ---------------------------------------------------------------------------


def _identity_map_from_trace(trace: Any) -> dict[str, str]:
    """Rebuild the artifact identity map from a retrieved trace's spans."""
    section: dict[str, str] = {"trace_id": trace.trace_id}
    for span in trace.data.spans:
        if span.parent_id is not None:
            section[f"span:{span.attributes['opencode.source_key']}"] = span.span_id
    return section


def test_deterministic_artifact_path_and_exact_bytes(imported: ImportedSession, expected: ExpectedMapping) -> None:
    for turn, request in zip(expected.turns, expected.requests, strict=True):
        trace_id = imported.trace_ids_by_key[request.metadata["opencode.turn_key"]]
        trace = imported.client.get_trace(trace_id, display=False, flush=True)

        # The recorded path equals the deterministic path mapped
        # independently from the same source: render and artifact-snapshot
        # fingerprints are pure functions of the turn and the contract
        # versions, so the path is reproducible offline.
        recorded_path = trace.info.tags["opencode.artifact_path"]
        assert recorded_path == request.artifact_relative_path
        recorded_sha = trace.info.tags["opencode.artifact_sha256"]
        recorded_size = int(trace.info.tags["opencode.artifact_size"])
        assert len(recorded_sha) == 64

        # Retrieve the stored artifact through the public artifact channel
        # (never the gateway) and byte-compare against the recorded digest.
        repository = get_artifact_repository(
            trace.info.tags["mlflow.artifactLocation"], tracking_uri=imported.client.tracking_uri
        )
        with TemporaryDirectory(prefix="opencode-mlflow-contract-") as dst:
            downloaded = Path(repository.download_artifacts(recorded_path, dst))
            stored_bytes = downloaded.read_bytes()
        assert len(stored_bytes) == recorded_size
        assert hashlib.sha256(stored_bytes).hexdigest() == recorded_sha

        # Re-stream the artifact document from the same replayed turn with
        # the identity map recovered from the retrieved trace: the document
        # is a deterministic function of (turn, identity map), so the
        # re-streamed bytes must equal the stored bytes exactly.
        with TemporaryDirectory(prefix="opencode-mlflow-restream-") as restream:
            restaged = Path(restream) / "artifact.json"
            with restaged.open("wb") as handle:
                manifest = stream_turn_artifact(
                    turn,
                    handle,
                    _identity_map_from_trace(trace),
                    RedactionMode(request.metadata["opencode.redaction_mode"]),
                    provenance=ArtifactProvenance.from_request(request),
                )
            restaged_bytes = restaged.read_bytes()
        assert manifest.sha256 == recorded_sha
        assert manifest.size_bytes == recorded_size
        assert restaged_bytes == stored_bytes

        document = json.loads(stored_bytes)
        assert document["identity_map"]["trace_id"] == trace_id
        assert document["session"]["session_id"] == SESSION_ID
        assert document["artifact"]["catalog_version"] == request.metadata["opencode.catalog_version"]
        assert document["turn_records"], "the artifact must persist the turn's records"
        assert document["identity_map"] == _identity_map_from_trace(trace)


def test_expected_requests_cover_every_reported_turn(imported: ImportedSession, expected: ExpectedMapping) -> None:
    """The independent expectations and the run's report agree on turn identity."""
    reported_keys = set(imported.trace_ids_by_key)
    expected_keys = {request.metadata["opencode.turn_key"] for request in expected.requests}
    assert reported_keys == expected_keys
    indexes = sorted(int(request.metadata["opencode.turn_index"]) for request in expected.requests)
    assert indexes == list(range(1, _TURNS + 1))
    assert len({request.client_request_id for request in expected.requests}) == _TURNS
    assert all(request.experiment_id == imported.experiment_id for request in expected.requests)
