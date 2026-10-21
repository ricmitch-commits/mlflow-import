"""Unit tests for the MLflow gateway boundary (real gateway, fake seams).

The production :class:`opencode_mlflow.mlflow.gateway.MlflowGateway` is
driven against the fake public-client/LiveSpan-shaped objects from
``tests/support/fake_gateway.py``: no network, no real MLflow server. These
tests pin the writer-boundary contract — experiment resolution, preflight
gating, lazy child consumption with a bounded parent map, complete root
payload verification, ownership-scoped pagination, public retirement, the
named-artifact composition, and dry-run non-mutation.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import mlflow
import pytest

from opencode_mlflow.errors import CapabilityError, ConfigurationError, ItemImportError, PartialTraceCreateError
from opencode_mlflow.mlflow.capabilities import CapabilityRequest
from opencode_mlflow.mlflow.gateway import (
    NO_OP_TRACE_ID,
    TRACE_SESSION_METADATA_KEY,
    CreatedTrace,
    MlflowGateway,
    TraceChildRequest,
    build_gateway,
)
from opencode_mlflow.mlflow.reconciliation import (
    ARTIFACT_PATH_TAG,
    CandidateContract,
    OwnedTraceCandidate,
)
from tests.support.fake_gateway import (
    FakeLiveSpan,
    FakeMlflowClient,
    FakeMlflowException,
    FakeMlflowGateway,
    FakeNoOpSpan,
    child_spec,
    trace_spec,
)

pytestmark = pytest.mark.unit

_TEST_TURN_KEY = "turn-" + "a" * 64
_RECORDED_TURN_KEY = "turn-" + "d" * 64


def _checked_gateway(client: FakeMlflowClient, *, dry_run: bool = False, search_page_size: int = 100) -> MlflowGateway:
    """A gateway whose capability preflight has already passed."""
    gateway = client.gateway(dry_run=dry_run, search_page_size=search_page_size)
    report = gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
    assert report.is_compatible, report.safe_error
    return gateway


def test_production_builder_does_not_expose_version_transport() -> None:
    assert "version_transport" not in inspect.signature(build_gateway).parameters


@pytest.mark.parametrize("field", ["start_time_ns", "end_time_ns", "child_start", "child_end"])
def test_missing_source_timing_rejected_before_any_span_mutation(fake_mlflow_client, field):
    gateway = _checked_gateway(fake_mlflow_client)
    child = child_spec("child")
    if field.startswith("child_"):
        child = replace(child, **{field.removeprefix("child_") + "_time_ns": None})
        spec = trace_spec(children=[child])
    else:
        spec = replace(trace_spec(children=[child]), **{field: None})
    with pytest.raises(ItemImportError, match="source-derived") as error:
        gateway.create_trace(spec)
    assert error.value.code == "source_timestamp_missing"
    assert fake_mlflow_client.mutations == []
    assert fake_mlflow_client.trace_api.calls == []


@pytest.mark.parametrize(
    "field",
    [
        "opencode.warnings",
        "opencode.session_id",
        "opencode.outcome",
        "opencode.turn_index",
        "opencode.importer_version",
        "opencode.mapping_version",
        "opencode.artifact_schema_version",
        "opencode.redaction_mode",
        "opencode.catalog_version",
    ],
)
def test_oversized_required_metadata_rejected_before_any_trace_is_created(fake_mlflow_client, field):
    gateway = _checked_gateway(fake_mlflow_client)
    oversized = "x" * 500
    metadata = {
        "opencode.importer": "opencode-mlflow",
        "opencode.session_id": oversized if field == "opencode.session_id" else "ses-sample",
        "opencode.turn_key": _TEST_TURN_KEY,
        field: oversized,
    }
    spec = trace_spec(
        native_session_id=oversized if field == "opencode.session_id" else "ses-sample",
        metadata=metadata,
    )

    with pytest.raises(ItemImportError) as error:
        gateway.create_trace(spec)

    assert error.value.code == "ownership_metadata_too_long"
    assert oversized not in error.value.safe_message
    assert fake_mlflow_client.mutations == []
    assert fake_mlflow_client.snapshot_state()[1] == ()
    assert fake_mlflow_client.trace_api.calls == []


def test_partial_create_ends_root_and_preserves_identity_after_successful_child(fake_mlflow_client):
    gateway = _checked_gateway(fake_mlflow_client)
    original = fake_mlflow_client.trace_api.start_span_no_context

    def failing(**kwargs):
        if kwargs["name"] == "broken":
            raise RuntimeError("synthetic failure")
        return original(**kwargs)

    fake_mlflow_client.trace_api.start_span_no_context = failing
    spec = trace_spec(children=[child_spec("first"), replace(child_spec("second"), name="broken")])
    with pytest.raises(PartialTraceCreateError) as error:
        gateway.create_trace(spec)
    created = error.value.created
    trace = fake_mlflow_client.get_trace(created.trace_id)
    assert len(trace.data.spans) == 2
    assert len(created.span_ids_by_source) == 1
    assert all(span.end_time_ns == spec.end_time_ns or span.parent_id is not None for span in trace.data.spans)
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert root.status == "ERROR"


def test_owned_trace_search_rejects_repeated_pagination_tokens(fake_mlflow_client):
    gateway = _checked_gateway(fake_mlflow_client)

    class Page(list):
        token = "repeated"

    calls = []

    def repeated(*args, **kwargs):
        calls.append(kwargs.get("page_token"))
        return Page()

    fake_mlflow_client.search_traces = repeated
    with pytest.raises(CapabilityError) as error:
        list(gateway.iter_owned_traces("exp"))
    assert error.value.code == "mlflow_pagination_cycle"
    assert len(calls) == 3  # initial filtered page, fallback, one continuation


def test_retirement_flushes_pending_exports_before_delete(fake_mlflow_client, monkeypatch):
    gateway = _checked_gateway(fake_mlflow_client)
    created = gateway.create_trace(trace_spec())
    events = []
    monkeypatch.setattr(mlflow, "flush_trace_async_logging", lambda: events.append("flush"))
    original = fake_mlflow_client.delete_traces

    def deleting(*args, **kwargs):
        events.append("delete")
        return original(*args, **kwargs)

    monkeypatch.setattr(fake_mlflow_client, "delete_traces", deleting)
    gateway.retire_trace("exp-1", created.trace_id, session_id="ses-sample", turn_key=_TEST_TURN_KEY)
    assert events == ["flush", "delete"]


def test_retirement_refuses_a_source_identity_race_after_the_first_gateway_check(
    fake_mlflow_client: FakeMlflowClient,
) -> None:
    """The final production gateway boundary must re-check before deleting.

    The fake returns an isolated first retrieval and mutates the backing trace
    immediately afterwards. A gateway that only checks once can therefore
    delete a trace whose source identity changed between checks.
    """
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("retire-race")
    created = gateway.create_trace(
        trace_spec(
            experiment_id=experiment.experiment_id,
            native_session_id="ses-race",
            metadata={
                "opencode.importer": "opencode-mlflow",
                "opencode.session_id": "ses-race",
                "opencode.turn_key": _TEST_TURN_KEY,
                "opencode.source_fingerprint": "a" * 64,
            },
        )
    )
    stored = fake_mlflow_client._traces[created.trace_id]

    def race(_trace_id: str) -> None:
        stored.info.trace_metadata["opencode.session_id"] = "ses-other"
        stored.info.trace_metadata["opencode.source_fingerprint"] = "b" * 64

    fake_mlflow_client.after_get_trace_hook = race
    mutation_count = len(fake_mlflow_client.mutations)

    with pytest.raises(ItemImportError) as excinfo:
        gateway.retire_trace(
            experiment.experiment_id,
            created.trace_id,
            session_id="ses-race",
            turn_key=_TEST_TURN_KEY,
            expected_metadata={"opencode.source_fingerprint": "a" * 64},
        )

    assert excinfo.value.code == "ownership_conflict"
    assert fake_mlflow_client.mutations[mutation_count:] == []
    assert fake_mlflow_client.get_trace(created.trace_id).trace_id == created.trace_id


def test_retirement_refuses_a_turn_identity_race_between_checks(
    fake_mlflow_client: FakeMlflowClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final retirement check must re-read the source turn identity.

    The first ownership check is deliberately made to race after it returns.
    A delete is safe only when the final check observes the same turn key.
    """
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("retire-turn-race")
    created = gateway.create_trace(
        trace_spec(
            experiment_id=experiment.experiment_id,
            native_session_id="ses-race",
            metadata={
                "opencode.importer": "opencode-mlflow",
                "opencode.session_id": "ses-race",
                "opencode.turn_key": _TEST_TURN_KEY,
                "opencode.source_fingerprint": "a" * 64,
            },
        )
    )
    stored = fake_mlflow_client._traces[created.trace_id]
    original_verify = gateway.verify_owned_trace_for_retirement

    def verify_then_race(*args: object, **kwargs: object) -> None:
        original_verify(*args, **kwargs)  # type: ignore[arg-type]
        stored.info.trace_metadata["opencode.turn_key"] = "turn-other"

    monkeypatch.setattr(gateway, "verify_owned_trace_for_retirement", verify_then_race)
    mutation_count = len(fake_mlflow_client.mutations)

    with pytest.raises(ItemImportError) as excinfo:
        gateway.retire_trace(
            experiment.experiment_id,
            created.trace_id,
            session_id="ses-race",
            turn_key=_TEST_TURN_KEY,
            expected_metadata={"opencode.source_fingerprint": "a" * 64},
        )

    assert excinfo.value.code == "ownership_conflict"
    assert fake_mlflow_client.mutations[mutation_count:] == []
    assert not any(call[0] == "delete_traces" for call in fake_mlflow_client.calls)


def test_retirement_refuses_a_malformed_ownership_contract_before_delete(
    fake_mlflow_client: FakeMlflowClient,
) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("retire-malformed-contract")
    created = gateway.create_trace(
        trace_spec(experiment_id=experiment.experiment_id, native_session_id="ses-malformed")
    )
    stored = fake_mlflow_client._traces[created.trace_id]
    stored.info.trace_metadata["opencode.source_fingerprint"] = "not-a-64-hex-fingerprint"
    mutation_count = len(fake_mlflow_client.mutations)

    with pytest.raises(ItemImportError) as excinfo:
        gateway.retire_trace(
            experiment.experiment_id,
            created.trace_id,
            session_id="ses-malformed",
            turn_key=_TEST_TURN_KEY,
        )

    assert excinfo.value.code == "ownership_conflict"
    assert fake_mlflow_client.mutations[mutation_count:] == []
    assert fake_mlflow_client.get_trace(created.trace_id).trace_id == created.trace_id


def test_retirement_refuses_overlong_warning_metadata_before_delete(
    fake_mlflow_client: FakeMlflowClient,
) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("retire-overlong-warning")
    created = gateway.create_trace(
        trace_spec(experiment_id=experiment.experiment_id, native_session_id="ses-overlong-warning")
    )
    fake_mlflow_client._traces[created.trace_id].info.trace_metadata["opencode.warnings"] = "w" * 251
    mutation_count = len(fake_mlflow_client.mutations)

    with pytest.raises(ItemImportError) as excinfo:
        gateway.retire_trace(
            experiment.experiment_id,
            created.trace_id,
            session_id="ses-overlong-warning",
            turn_key=_TEST_TURN_KEY,
        )

    assert excinfo.value.code == "ownership_conflict"
    assert fake_mlflow_client.mutations[mutation_count:] == []
    assert fake_mlflow_client.get_trace(created.trace_id).trace_id == created.trace_id


def test_create_trace_rejects_malformed_warning_metadata_before_any_span(
    fake_mlflow_client: FakeMlflowClient,
) -> None:
    gateway = _checked_gateway(fake_mlflow_client)

    with pytest.raises(ItemImportError) as excinfo:
        gateway.create_trace(trace_spec(metadata={"opencode.warnings": "NOT-LOWERCASE"}))

    assert excinfo.value.code == "ownership_metadata_invalid"
    assert fake_mlflow_client.mutations == []
    assert fake_mlflow_client.trace_api.calls == []


# ---------------------------------------------------------------------------
# Experiment resolution
# ---------------------------------------------------------------------------


def test_resolve_experiment_reuses_existing_experiment_exactly(fake_mlflow_client: FakeMlflowClient) -> None:
    existing = fake_mlflow_client.add_experiment("team/opencode", artifact_location="mlflow-artifacts://exp-7")
    gateway = _checked_gateway(fake_mlflow_client)

    reference = gateway.resolve_experiment("team/opencode", create=True)

    assert reference.experiment_id == existing.experiment_id
    assert reference.created is False
    assert reference.planned_creation is False
    assert fake_mlflow_client.mutations == []
    assert existing.artifact_location == "mlflow-artifacts://exp-7"


def test_resolve_experiment_creates_missing_only_after_preflight(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = fake_mlflow_client.gateway()

    with pytest.raises(CapabilityError) as excinfo:
        gateway.resolve_experiment("fresh-experiment", create=True)

    assert excinfo.value.code == "mlflow_preflight_required"
    assert fake_mlflow_client.mutations == []
    assert fake_mlflow_client.get_experiment_by_name("fresh-experiment") is None

    report = gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
    assert report.is_compatible
    reference = gateway.resolve_experiment("fresh-experiment", create=True)

    assert reference.experiment_id is not None
    assert reference.created is True
    assert fake_mlflow_client.mutations == [("create_experiment", "fresh-experiment")]


def test_resolve_experiment_missing_without_create_fails(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)

    with pytest.raises(ConfigurationError) as excinfo:
        gateway.resolve_experiment("does-not-exist", create=False)

    assert excinfo.value.code == "experiment_not_found"
    assert "does-not-exist" not in excinfo.value.safe_message
    assert fake_mlflow_client.mutations == []


def test_resolve_experiment_dry_run_plans_creation_without_mutating(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client, dry_run=True)

    reference = gateway.resolve_experiment("planned-experiment", create=True)

    assert reference.planned_creation is True
    assert reference.created is False
    assert reference.experiment_id is None
    assert fake_mlflow_client.mutations == []
    assert fake_mlflow_client.get_experiment_by_name("planned-experiment") is None


def test_gateway_refuses_mutations_in_dry_run_mode(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client, dry_run=True)

    with pytest.raises(CapabilityError) as create_error:
        gateway.create_trace(trace_spec())
    assert create_error.value.code == "dry_run_mutation"

    with pytest.raises(CapabilityError):
        gateway.retire_trace("exp-1", "tr-1")
    with pytest.raises(CapabilityError):
        gateway.upload_named_json("tr-1", "a/b.json", Path("ignored.json"))

    assert fake_mlflow_client.mutations == []


def test_gateway_refuses_mutation_after_incompatible_probe(
    fake_mlflow_client: FakeMlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway = fake_mlflow_client.gateway()
    monkeypatch.setattr(mlflow, "__version__", "0.0.0")
    report = gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
    assert not report.is_compatible

    with pytest.raises(CapabilityError) as excinfo:
        gateway.resolve_experiment("any", create=True)

    assert excinfo.value.code == "mlflow_capability_unsupported"
    assert fake_mlflow_client.mutations == []


def test_gateway_probe_defaults_to_gateway_mode(fake_mlflow_client: FakeMlflowClient) -> None:
    dry_gateway = fake_mlflow_client.gateway(dry_run=True)

    report = dry_gateway.probe(CapabilityRequest(needs_retirement=False))

    assert report.dry_run is True
    assert report.needs_retirement is False


def test_normal_mode_mutation_requires_normal_mode_probe(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = fake_mlflow_client.gateway()

    report = gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True), dry_run=True)
    assert report.is_compatible

    with pytest.raises(CapabilityError) as excinfo:
        gateway.resolve_experiment("any", create=True)

    assert excinfo.value.code == "mlflow_preflight_required"
    assert fake_mlflow_client.mutations == []


# ---------------------------------------------------------------------------
# Trace creation
# ---------------------------------------------------------------------------


def test_create_trace_starts_root_and_children_with_parent_map(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("e")
    spec = trace_spec(
        experiment_id=experiment.experiment_id,
        children=[
            child_spec("msg-1", name="user prompt", span_type="CHAIN", inputs={"text": "hi"}),
            child_spec(
                "msg-2", name="assistant reply", span_type="AGENT", parent_source_key="msg-1", outputs={"text": "yo"}
            ),
        ],
    )

    created = gateway.create_trace(spec)

    assert created.trace_id
    assert created.root_span_id
    assert created.span_ids_by_source.keys() == {"msg-1", "msg-2"}
    assert created.span_ids_by_source["msg-1"] != created.span_ids_by_source["msg-2"]

    trace = fake_mlflow_client.get_trace(created.trace_id)
    spans = {span.span_id: span for span in trace.data.spans}
    root = spans[created.root_span_id]
    child = spans[created.span_ids_by_source["msg-1"]]
    grandchild = spans[created.span_ids_by_source["msg-2"]]
    assert root.parent_id is None
    assert child.parent_id == created.root_span_id
    assert grandchild.parent_id == created.span_ids_by_source["msg-1"]
    assert root.name == spec.name
    assert root.inputs == spec.root_inputs
    assert root.outputs == spec.root_outputs


def test_create_trace_sets_native_session_metadata(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(native_session_id="ses-native-42", metadata={"opencode.session_id": "ses-native-42"})

    created = gateway.create_trace(spec)

    assert created.verified_metadata[TRACE_SESSION_METADATA_KEY] == "ses-native-42"
    trace = fake_mlflow_client.get_trace(created.trace_id)
    assert trace.info.trace_metadata[TRACE_SESSION_METADATA_KEY] == "ses-native-42"


def test_create_trace_consumes_children_one_at_a_time(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    ended_before_next_fetch: list[bool] = []

    def lazy_children() -> Iterator[TraceChildRequest]:
        for index in range(3):
            # Every previously yielded child must already be ended before the
            # next one is fetched: the gateway never buffers the child list.
            ended = [call for call in fake_mlflow_client.trace_api.calls if call[0] == "end_span"]
            started = [call for call in fake_mlflow_client.trace_api.calls if call[0] == "start_span_no_context"]
            ended_before_next_fetch.append(not started or len(ended) == len(started) - 1)
            yield child_spec(f"child-{index}")

    spec = trace_spec(child_specs=lazy_children)

    created = gateway.create_trace(spec)

    assert created.span_ids_by_source.keys() == {"child-0", "child-1", "child-2"}
    # One read-only timing preflight pass, then one streamed create pass.
    assert ended_before_next_fetch == [True] * 6
    # Root is ended last, after every child.
    end_calls = [call for call in fake_mlflow_client.trace_api.calls if call[0] == "end_span"]
    assert end_calls[-1][1]["span_id"] == created.root_span_id


def test_create_trace_rejects_unknown_parent_source_key(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(children=[child_spec("orphan", parent_source_key="never-created")])

    with pytest.raises(ItemImportError) as excinfo:
        gateway.create_trace(spec)

    assert excinfo.value.code == "unresolvable_parent_span"
    assert "never-created" not in excinfo.value.safe_message


def test_create_trace_rejects_noop_span_degradation(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)

    original_start = fake_mlflow_client.trace_api.start_span_no_context

    def degraded(**kwargs: object) -> FakeLiveSpan:
        return FakeLiveSpan(
            span_id=None,
            trace_id=NO_OP_TRACE_ID,
            parent_id=None,
            name="no-op",
            span_type="UNKNOWN",
            experiment_id=None,
            start_time_ns=None,
        )

    fake_mlflow_client.trace_api.start_span_no_context = degraded  # type: ignore[method-assign]
    try:
        with pytest.raises(CapabilityError) as excinfo:
            gateway.create_trace(trace_spec())
    finally:
        fake_mlflow_client.trace_api.start_span_no_context = original_start  # type: ignore[method-assign]

    assert excinfo.value.code == "mlflow_span_degraded"


def test_child_span_with_experiment_id_degrades_in_the_fake_and_fails_loudly(
    fake_mlflow_client: FakeMlflowClient,
) -> None:
    """The fakes model the pinned pair's child+``experiment_id`` degradation.

    Live-verified in Task 14 (see ``docs/compatibility.md`` §
    "Live-verified gateway constraints"): the pinned 3.17.2 provider
    degrades a NON-root span carrying ``experiment_id`` to a NoOp span.
    The fake mirrors that degradation, so a future regression of the
    gateway's call shape can no longer pass the fake-backed suites — the
    gateway's loud ``mlflow_span_degraded`` check fires here exactly as
    it does against the live server.
    """
    gateway = _checked_gateway(fake_mlflow_client)
    reference = gateway.resolve_experiment("noop-degradation", create=True)
    assert reference.experiment_id is not None

    # (a) The fake itself degrades the regressed shape: constructing a
    # child span with experiment_id through the fake warns like the real
    # provider and returns a NoOp-shaped span, with nothing registered
    # server-side.
    root = fake_mlflow_client.trace_api.start_span_no_context(
        name="regression-root", span_type="CHAIN", experiment_id=reference.experiment_id
    )
    assert root.trace_id != NO_OP_TRACE_ID
    with pytest.warns(UserWarning, match="can only be used for root spans"):
        degraded = fake_mlflow_client.trace_api.start_span_no_context(
            name="regressed-child",
            span_type="LLM",
            parent_span=root,
            experiment_id=reference.experiment_id,
        )
    assert isinstance(degraded, FakeNoOpSpan)
    assert degraded.trace_id == NO_OP_TRACE_ID
    assert degraded.span_id is None
    # Nothing reached the fake server: the root trace still carries
    # exactly its own span.
    assert len(fake_mlflow_client.get_trace(root.trace_id).data.spans) == 1

    # (b) A future regression of the gateway's call shape — children
    # carrying experiment_id again — trips the gateway's loud check
    # against the fake exactly as it does against the live server.
    original_start = fake_mlflow_client.trace_api.start_span_no_context
    root_experiment_id: dict[str, str] = {}

    def regressed(**kwargs: object) -> FakeLiveSpan | FakeNoOpSpan:
        if kwargs.get("parent_span") is None and kwargs.get("experiment_id") is not None:
            root_experiment_id["id"] = str(kwargs["experiment_id"])
        elif kwargs.get("parent_span") is not None and kwargs.get("experiment_id") is None:
            kwargs["experiment_id"] = root_experiment_id["id"]
        return original_start(**kwargs)

    fake_mlflow_client.trace_api.start_span_no_context = regressed  # type: ignore[method-assign]
    try:
        with pytest.warns(UserWarning, match="can only be used for root spans"):
            with pytest.raises(ItemImportError) as excinfo:
                gateway.create_trace(trace_spec(children=[child_spec("msg-1")]))
    finally:
        fake_mlflow_client.trace_api.start_span_no_context = original_start  # type: ignore[method-assign]

    assert excinfo.value.code == "mlflow_span_degraded"


def test_create_trace_pins_global_tracking_uri_around_span_work(
    fake_mlflow_client: FakeMlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    pinned: list[str] = []
    real_set = mlflow.set_tracking_uri

    def recording_set(uri: str) -> None:
        pinned.append(uri)
        real_set(uri)

    monkeypatch.setattr(mlflow, "set_tracking_uri", recording_set)
    before = mlflow.get_tracking_uri()

    gateway.create_trace(trace_spec())

    # Span export follows the *global* tracking URI, so the gateway pins it to
    # the client's target for the whole create and restores the saved value.
    assert pinned[0] == fake_mlflow_client.tracking_uri
    assert pinned[-1] == before
    assert mlflow.get_tracking_uri() == before


def test_create_trace_ends_root_with_derived_status(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)

    completed = gateway.create_trace(trace_spec(root_outputs={"answer": 1}))
    incomplete = gateway.create_trace(trace_spec(root_outputs=None))

    end_calls = [call for call in fake_mlflow_client.trace_api.calls if call[0] == "end_span"]
    statuses = [
        call[1]["status"]
        for call in end_calls
        if call[1]["span_id"] in {completed.root_span_id, incomplete.root_span_id}
    ]
    assert "OK" in statuses
    assert "UNSET" in statuses
    # The effective status is exposed on CreatedTrace so verification has a
    # single-sourced expectation (the explicit root_status is preferred in
    # _effective_root_status whenever the mapping set one).
    assert completed.root_status == "OK"
    assert incomplete.root_status == "UNSET"


def test_create_trace_root_status_follows_last_child(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    recovered = trace_spec(children=[child_spec("msg-1", status="ERROR"), child_spec("msg-2", status="OK")])
    failed_spec = trace_spec(children=[child_spec("msg-1", status="OK"), child_spec("msg-2", status="ERROR")])

    created = gateway.create_trace(recovered)
    assert created.root_status == "OK"
    gateway.verify_trace(created, recovered)

    failed = gateway.create_trace(failed_spec)
    assert failed.root_status == "ERROR"
    gateway.verify_trace(failed, failed_spec)


def test_create_trace_prefers_the_explicit_root_status_over_derivation(fake_mlflow_client: FakeMlflowClient) -> None:
    # The case the derivation could not express: the mapping's explicit
    # root_status is authoritative, so a root ERROR verifies as ERROR even
    # when its last child ended OK, and the None default keeps the
    # verified derivation unchanged.
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(
        root_outputs={"answer": 42},
        root_status="ERROR",
        children=[child_spec("msg-1", status="OK")],
    )

    created = gateway.create_trace(spec)

    assert created.root_status == "ERROR"
    gateway.verify_trace(created, spec)
    root = _root_span(fake_mlflow_client, created)
    assert root.status == "ERROR"

    # The None default still derives from the verbatim request fields.
    derived_spec = trace_spec(
        root_outputs={"answer": 42},
        children=[child_spec("msg-1", status="OK")],
    )
    derived = gateway.create_trace(derived_spec)
    assert derived.root_status == "OK"
    gateway.verify_trace(derived, derived_spec)

    # An explicit status that AGREES with the derivation verifies too.
    agreeing = gateway.create_trace(trace_spec(root_outputs=None, root_status="UNSET"))
    assert agreeing.root_status == "UNSET"
    gateway.verify_trace(agreeing, trace_spec(root_outputs=None, root_status="UNSET"))


def test_create_trace_passes_child_payloads_and_times(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(
        children=[
            child_spec(
                "msg-1",
                inputs={"text": "hello"},
                attributes={"opencode.source_id": "msg-1"},
                tags={"opencode.span_role": "user"},
                start_time_ns=1_700_000_000_005_000_000,
                outputs={"text": "world"},
                status="OK",
                end_time_ns=1_700_000_000_006_000_000,
            )
        ]
    )

    created = gateway.create_trace(spec)

    start_calls = [call for call in fake_mlflow_client.trace_api.calls if call[0] == "start_span_no_context"]
    child_start = next(call[1] for call in start_calls if call[1]["parent_span"] is not None)
    assert child_start["inputs"] == {"text": "hello"}
    assert child_start["attributes"] == {"opencode.source_id": "msg-1"}
    assert child_start["tags"] == {"opencode.span_role": "user"}
    assert child_start["start_time_ns"] == 1_700_000_000_005_000_000
    # ``experiment_id`` is a root-span parameter: a child joins its parent's
    # trace, and the pinned 3.17.2 provider degrades a non-root span carrying
    # it to a NoOp span (verified live by tests/integration/
    # test_mlflow_trace_contract.py), so the child call must omit it.
    assert child_start["experiment_id"] is None

    trace = fake_mlflow_client.get_trace(created.trace_id)
    span = next(span for span in trace.data.spans if span.span_id == created.span_ids_by_source["msg-1"])
    assert span.outputs == {"text": "world"}
    assert span.status == "OK"
    assert span.end_time_ns == 1_700_000_000_006_000_000


# ---------------------------------------------------------------------------
# Trace verification
# ---------------------------------------------------------------------------


def test_verify_trace_compares_complete_root_payload(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    payload = {"prompt": "x" * 100_000, "rows": list(range(1000))}
    spec = trace_spec(root_inputs=payload, root_outputs={"answer": 42})

    created = gateway.create_trace(spec)
    gateway.verify_trace(created, spec)


def test_verify_trace_uses_full_retrieval_not_search_previews(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec()
    created = gateway.create_trace(spec)

    gateway.verify_trace(created, spec)

    retrieved = [
        call for call in fake_mlflow_client.calls if call[0] == "get_trace" and call[1]["trace_id"] == created.trace_id
    ]
    assert retrieved, "verify_trace must retrieve the full trace payload"
    searches = [call for call in fake_mlflow_client.calls if call[0] == "search_traces"]
    assert not searches, "verify_trace must not rely on search previews"


def test_verify_trace_rejects_payload_mismatch(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(root_outputs={"answer": 42})
    created = gateway.create_trace(spec)

    tampered = trace_spec(root_outputs={"answer": 43})
    with pytest.raises(ItemImportError) as excinfo:
        gateway.verify_trace(created, tampered)

    assert excinfo.value.code == "trace_verification_failed"


def test_verify_trace_rejects_native_session_mismatch(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(native_session_id="ses-a")
    created = gateway.create_trace(spec)

    with pytest.raises(ItemImportError):
        gateway.verify_trace(created, trace_spec(native_session_id="ses-b"))


def test_verify_trace_rejects_missing_child_span(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(children=[child_spec("msg-1")])
    created = gateway.create_trace(spec)

    impostor = CreatedTrace(
        trace_id=created.trace_id,
        root_span_id=created.root_span_id,
        span_ids_by_source={**created.span_ids_by_source, "ghost": "span-missing"},
        verified_metadata=created.verified_metadata,
        root_status=created.root_status,
    )
    with pytest.raises(ItemImportError):
        gateway.verify_trace(impostor, spec)


def test_verify_trace_checks_timing_and_tags(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(
        root_tags={"opencode.turn_index": "1"},
        start_time_ns=1_700_000_000_000_000_000,
        end_time_ns=1_700_000_000_030_000_000,
    )
    created = gateway.create_trace(spec)
    gateway.verify_trace(created, spec)

    with pytest.raises(ItemImportError):
        gateway.verify_trace(created, trace_spec(start_time_ns=1, end_time_ns=2))
    with pytest.raises(ItemImportError):
        gateway.verify_trace(created, trace_spec(root_tags={"opencode.missing": "x"}))


def test_verify_trace_tolerates_server_tag_truncation(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    # The fake (like the pinned server) silently truncates tag values at 4096
    # characters; verification must not rely on full-length tag round-trips.
    long_tag = "t" * 5000
    spec = trace_spec(root_tags={"opencode.long": long_tag})

    created = gateway.create_trace(spec)
    gateway.verify_trace(created, spec)

    trace = fake_mlflow_client.get_trace(created.trace_id)
    assert trace.info.tags["opencode.long"] == long_tag[:4096]


def _root_span(client: FakeMlflowClient, created: CreatedTrace):
    trace = client.get_trace(created.trace_id)
    return next(span for span in trace.data.spans if span.span_id == created.root_span_id)


def test_verify_trace_rejects_root_status_mismatch(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(root_outputs={"answer": 42})
    created = gateway.create_trace(spec)
    assert created.root_status == "OK"

    for tampered in ("UNSET", "ERROR", None):
        _root_span(fake_mlflow_client, created).status = tampered
        with pytest.raises(ItemImportError) as excinfo:
            gateway.verify_trace(created, spec)
        assert excinfo.value.code == "trace_verification_failed"

    _root_span(fake_mlflow_client, created).status = "OK"
    gateway.verify_trace(created, spec)


def test_verify_trace_rejects_trace_state_mismatch(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(root_outputs={"answer": 42})
    created = gateway.create_trace(spec)

    trace = fake_mlflow_client.get_trace(created.trace_id)
    assert trace.info.state == "OK"
    trace.info.state = "ERROR"
    with pytest.raises(ItemImportError) as excinfo:
        gateway.verify_trace(created, spec)
    assert excinfo.value.code == "trace_verification_failed"

    trace.info.state = "OK"
    gateway.verify_trace(created, spec)


def test_verify_trace_applies_the_trace_state_mapping_for_unset_roots(fake_mlflow_client: FakeMlflowClient) -> None:
    # MLflow maps a root span ended UNSET to trace state OK (its
    # maybe_update_trace_state overrides STATE_UNSPECIFIED to OK), so an
    # incomplete turn verifies: span status stays UNSET while the trace
    # state surfaces as OK. The fake mirrors that mapping.
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(root_outputs=None)
    created = gateway.create_trace(spec)

    assert created.root_status == "UNSET"
    trace = fake_mlflow_client.get_trace(created.trace_id)
    assert trace.info.state == "OK"
    gateway.verify_trace(created, spec)


def test_verify_trace_accepts_extra_internal_span_attributes(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(root_attributes={"opencode.turn_index": "1", "opencode.source_schema": "current"})
    created = gateway.create_trace(spec)

    # MLflow adds its own internal attribute keys to server-side spans; the
    # importer's attributes are verified as a subset, not by equality.
    _root_span(fake_mlflow_client, created).attributes["mlflow.spanFunctionName"] = "opencode"
    _root_span(fake_mlflow_client, created).attributes["mlflow.sessionId"] = "ses-internal"

    gateway.verify_trace(created, spec)


def test_verify_trace_rejects_root_attribute_mangling(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    spec = trace_spec(root_attributes={"opencode.turn_index": "1"})
    created = gateway.create_trace(spec)
    gateway.verify_trace(created, spec)

    root = _root_span(fake_mlflow_client, created)
    root.attributes["opencode.turn_index"] = "2"
    with pytest.raises(ItemImportError) as excinfo:
        gateway.verify_trace(created, spec)
    assert excinfo.value.code == "trace_verification_failed"

    del root.attributes["opencode.turn_index"]
    with pytest.raises(ItemImportError):
        gateway.verify_trace(created, spec)


# ---------------------------------------------------------------------------
# Owned-trace search
# ---------------------------------------------------------------------------


def _seed_owned_and_unowned(gateway: MlflowGateway, experiment_id: str, count: int) -> None:
    for index in range(count):
        gateway.create_trace(
            trace_spec(experiment_id=experiment_id, native_session_id=f"ses-{index}", client_request_id=f"req-{index}")
        )
    gateway.create_trace(trace_spec(experiment_id=experiment_id, owned=False, client_request_id="unowned"))


def test_iter_owned_traces_walks_pagination_and_yields_only_owned(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client, search_page_size=2)
    experiment = fake_mlflow_client.add_experiment("owned")
    _seed_owned_and_unowned(gateway, experiment.experiment_id, count=5)

    owned = list(gateway.iter_owned_traces(experiment.experiment_id))

    assert len(owned) == 5
    assert all(trace.metadata.get("opencode.importer") == "opencode-mlflow" for trace in owned)
    assert all(
        trace.tags.get("opencode.importer") == "opencode-mlflow" or trace.metadata.get("opencode.importer")
        for trace in owned
    )

    search_calls = [call for call in fake_mlflow_client.calls if call[0] == "search_traces"]
    tokens = [call[1]["page_token"] for call in search_calls]
    assert len(search_calls) >= 3, "pagination must be walked page by page"
    assert tokens[0] is None
    assert all(token for token in tokens[1:])


def test_iter_owned_traces_never_yields_unowned_traces(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("mixed")
    unowned = gateway.create_trace(trace_spec(experiment_id=experiment.experiment_id, owned=False))
    owned = gateway.create_trace(trace_spec(experiment_id=experiment.experiment_id))

    found = list(gateway.iter_owned_traces(experiment.experiment_id))

    assert [trace.trace_id for trace in found] == [owned.trace_id]
    assert unowned.trace_id not in {trace.trace_id for trace in found}


def test_iter_owned_traces_scopes_to_the_requested_experiment(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    first = fake_mlflow_client.add_experiment("first")
    second = fake_mlflow_client.add_experiment("second")
    gateway.create_trace(trace_spec(experiment_id=first.experiment_id))
    gateway.create_trace(trace_spec(experiment_id=second.experiment_id))

    found = list(gateway.iter_owned_traces(first.experiment_id))

    assert len(found) == 1
    search_calls = [call for call in fake_mlflow_client.calls if call[0] == "search_traces"]
    assert all(call[1]["locations"] == [first.experiment_id] for call in search_calls)


def test_iter_owned_traces_falls_back_when_server_rejects_filter(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client, search_page_size=3)
    experiment = fake_mlflow_client.add_experiment("fallback")
    _seed_owned_and_unowned(gateway, experiment.experiment_id, count=4)
    fake_mlflow_client.reject_filters = True

    owned = list(gateway.iter_owned_traces(experiment.experiment_id))

    assert len(owned) == 4
    search_calls = [call for call in fake_mlflow_client.calls if call[0] == "search_traces"]
    assert any(call[1]["filter_string"] for call in search_calls), "the server-side filter must be attempted first"
    assert any(call[1]["filter_string"] is None for call in search_calls), (
        "a rejected filter falls back to local filtering"
    )


def test_iter_owned_traces_finds_tag_marked_owned_traces(fake_mlflow_client: FakeMlflowClient) -> None:
    # Ownership recorded in trace TAGS instead of metadata: the server-side
    # metadata filter returns an empty page, so the gateway falls back to
    # unfiltered pages and the local ownership check still finds the trace.
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("tag-owned")
    spec = trace_spec(
        experiment_id=experiment.experiment_id,
        owned=False,
        root_tags={"opencode.importer": "opencode-mlflow"},
    )
    created = gateway.create_trace(spec)

    found = list(gateway.iter_owned_traces(experiment.experiment_id))

    assert [trace.trace_id for trace in found] == [created.trace_id]
    search_calls = [call for call in fake_mlflow_client.calls if call[0] == "search_traces"]
    assert any(call[1]["filter_string"] for call in search_calls)
    assert any(call[1]["filter_string"] is None for call in search_calls)


# ---------------------------------------------------------------------------
# Retirement
# ---------------------------------------------------------------------------


def test_retire_trace_calls_public_delete_with_exact_ids(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("retire")
    created = gateway.create_trace(trace_spec(experiment_id=experiment.experiment_id))

    gateway.retire_trace(
        experiment.experiment_id,
        created.trace_id,
        session_id="ses-sample",
        turn_key=_TEST_TURN_KEY,
    )

    delete_calls = [call for call in fake_mlflow_client.calls if call[0] == "delete_traces"]
    assert delete_calls == [
        (
            "delete_traces",
            {"experiment_id": experiment.experiment_id, "trace_ids": [created.trace_id], "max_timestamp_millis": None},
        )
    ]
    assert fake_mlflow_client.mutations[-1] == ("delete_traces", experiment.experiment_id, (created.trace_id,))
    with pytest.raises(FakeMlflowException):
        fake_mlflow_client.get_trace(created.trace_id)


def test_retire_trace_fails_when_server_reports_no_deletion(fake_mlflow_client: FakeMlflowClient) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("retire-miss")

    with pytest.raises(ItemImportError) as excinfo:
        gateway.retire_trace(
            experiment.experiment_id,
            "tr-never-existed",
            session_id="ses-sample",
            turn_key=_TEST_TURN_KEY,
        )

    assert excinfo.value.code == "ownership_conflict"


# ---------------------------------------------------------------------------
# Named trace artifacts
# ---------------------------------------------------------------------------


def test_upload_named_json_preserves_relative_path_and_digest(
    fake_mlflow_client: FakeMlflowClient, tmp_path: Path
) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    created = gateway.create_trace(trace_spec())
    payload = b'{"rows": [1, 2, 3]}' * 100
    local = tmp_path / "payload.json"
    local.write_bytes(payload)

    verification = gateway.upload_named_json(
        created.trace_id, "opencode-mlflow/v1/sessions/s/nested/payload.json", local, chunk_size=512
    )

    assert verification.trace_id == created.trace_id
    assert verification.relative_path == "opencode-mlflow/v1/sessions/s/nested/payload.json"
    assert verification.sha256 == hashlib.sha256(payload).hexdigest()
    assert verification.size == len(payload)

    repository = fake_mlflow_client.artifacts.repositories[f"fake-artifact://traces/{created.trace_id}"]
    assert repository.files["opencode-mlflow/v1/sessions/s/nested/payload.json"] == payload
    assert repository.calls[0][0] == "log_artifact"
    assert repository.calls[0][2] == "opencode-mlflow/v1/sessions/s/nested"


def test_verify_named_json_round_trips_and_rejects_tampering(
    fake_mlflow_client: FakeMlflowClient, tmp_path: Path
) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    created = gateway.create_trace(trace_spec())
    local = tmp_path / "payload.json"
    local.write_bytes(b"exact bytes")

    verification = gateway.upload_named_json(created.trace_id, "a/b/payload.json", local)

    gateway.verify_named_json(created.trace_id, verification.relative_path, verification.sha256, verification.size)

    with pytest.raises(ItemImportError) as digest_error:
        gateway.verify_named_json(created.trace_id, verification.relative_path, "0" * 64, verification.size)
    assert digest_error.value.code == "artifact_verification_failed"

    with pytest.raises(ItemImportError):
        gateway.verify_named_json(
            created.trace_id, verification.relative_path, verification.sha256, verification.size + 1
        )


def test_upload_named_json_rejects_unsafe_relative_paths(fake_mlflow_client: FakeMlflowClient, tmp_path: Path) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    created = gateway.create_trace(trace_spec())
    local = tmp_path / "payload.json"
    local.write_bytes(b"{}")

    for unsafe in ("../escape.json", "/absolute.json", "", "."):
        with pytest.raises(ItemImportError) as excinfo:
            gateway.upload_named_json(created.trace_id, unsafe, local)
        assert excinfo.value.code == "unsafe_artifact_relative_path"

    assert fake_mlflow_client.artifacts.repositories[f"fake-artifact://traces/{created.trace_id}"].files == {}


def test_upload_named_json_rejects_non_positive_chunk_size(
    fake_mlflow_client: FakeMlflowClient, tmp_path: Path
) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    created = gateway.create_trace(trace_spec())
    local = tmp_path / "payload.json"
    local.write_bytes(b"{}")

    with pytest.raises(ValueError):
        gateway.upload_named_json(created.trace_id, "a.json", local, chunk_size=0)


# ---------------------------------------------------------------------------
# Owned-candidate verification
# ---------------------------------------------------------------------------


_OWNERSHIP_METADATA = {
    "opencode.importer": "opencode-mlflow",
    "opencode.session_id": "ses-recorded",
    "opencode.turn_key": _RECORDED_TURN_KEY,
    "opencode.source_fingerprint": "a" * 64,
    "opencode.source_schema": "current",
    "opencode.turn_index": "1",
    "opencode.importer_version": "0.1.0",
    "opencode.mapping_version": "mapping-v3",
    "opencode.artifact_schema_version": "artifact-v2",
    "opencode.redaction_mode": "verbatim",
    "opencode.catalog_version": "opencode-schema-catalog-v1",
    "opencode.render_fingerprint": "b" * 64,
    "opencode.operation_key": "c" * 64,
}


def _indexed_candidate(gateway: MlflowGateway, experiment_id: str) -> OwnedTraceCandidate:
    """Index the experiment's single owned trace as the write phase would."""
    owned = next(trace for trace in gateway.iter_owned_traces(experiment_id))
    return OwnedTraceCandidate.from_owned_trace(owned, experiment_id)


def test_verify_owned_candidate_checks_integrity_at_the_recorded_path(
    fake_mlflow_client: FakeMlflowClient, tmp_path: Path
) -> None:
    """Integrity is checked at the candidate's OWN recorded location.

    The current run's would-be path (the contract's
    ``artifact_relative_path``) is not an equivalence input: the
    deterministic path embeds the session-complete artifact-snapshot
    fingerprint and moves for every sibling turn whenever any turn of the
    session changes, so verifying there would falsely replace every
    unchanged sibling.
    """
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("recorded-path")
    recorded_path = f"opencode-mlflow/v1/sessions/ses-recorded/turns/{_RECORDED_TURN_KEY}/aaa-bbb.json"
    spec = trace_spec(
        experiment_id=experiment.experiment_id,
        native_session_id="ses-recorded",
        root_tags={ARTIFACT_PATH_TAG: recorded_path},
        metadata=dict(_OWNERSHIP_METADATA),
    )
    created = gateway.create_trace(spec)
    local = tmp_path / "artifact.json"
    local.write_bytes(b'{"artifact": "exact bytes"}')
    verification = gateway.upload_named_json(created.trace_id, recorded_path, local)
    gateway.record_artifact_verification(created.trace_id, verification.sha256, verification.size)
    candidate = _indexed_candidate(gateway, experiment.experiment_id)

    # The current run's contract points at a DIFFERENT would-be path (as if
    # a sibling turn of the same session had changed); the candidate still
    # matches because integrity runs at its own recorded location.
    contract = replace(
        CandidateContract.from_create_request(spec, verification.sha256, verification.size),
        artifact_relative_path=f"opencode-mlflow/v1/sessions/ses-recorded/turns/{_RECORDED_TURN_KEY}/zzz-www.json",
    )

    assert gateway.verify_owned_candidate(candidate, contract).matches is True

    # Tampered bytes at the recorded location still fail integrity.
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(b'{"tampered": "bytes"}')
    gateway.upload_named_json(created.trace_id, recorded_path, tampered)

    outcome = gateway.verify_owned_candidate(candidate, contract)

    assert outcome.matches is False
    assert outcome.reason == "artifact_mismatch"


def test_verify_owned_candidate_without_a_recorded_path_is_an_incomplete_non_match(
    fake_mlflow_client: FakeMlflowClient, tmp_path: Path
) -> None:
    """A trace written before the path tag exists is unverifiable: a safe replace."""
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("pre-tag-path")
    spec = trace_spec(
        experiment_id=experiment.experiment_id,
        native_session_id="ses-recorded",
        metadata=dict(_OWNERSHIP_METADATA),
    )
    created = gateway.create_trace(spec)
    local = tmp_path / "artifact.json"
    local.write_bytes(b'{"artifact": "bytes"}')
    legacy_path = f"opencode-mlflow/v1/sessions/ses-recorded/turns/{_RECORDED_TURN_KEY}/legacy.json"
    verification = gateway.upload_named_json(created.trace_id, legacy_path, local)
    gateway.record_artifact_verification(created.trace_id, verification.sha256, verification.size)
    candidate = _indexed_candidate(gateway, experiment.experiment_id)
    contract = CandidateContract.from_create_request(spec, verification.sha256, verification.size)

    outcome = gateway.verify_owned_candidate(candidate, contract)

    assert outcome.matches is False
    assert outcome.reason == "artifact_path_missing"


@pytest.mark.parametrize(
    ("field", "mutated"),
    [
        ("opencode.turn_index", "2"),
        ("opencode.importer_version", "changed-version"),
        ("opencode.mapping_version", "changed-mapping"),
        ("opencode.artifact_schema_version", "changed-artifact-schema"),
        ("opencode.redaction_mode", "redacted"),
    ],
)
def test_verify_owned_candidate_compares_every_creation_contract_field(
    fake_mlflow_client: FakeMlflowClient,
    tmp_path: Path,
    field: str,
    mutated: str,
) -> None:
    gateway = _checked_gateway(fake_mlflow_client)
    experiment = fake_mlflow_client.add_experiment("full-contract")
    recorded_path = f"opencode-mlflow/v1/sessions/ses-recorded/turns/{_RECORDED_TURN_KEY}/full.json"
    spec = trace_spec(
        experiment_id=experiment.experiment_id,
        native_session_id="ses-recorded",
        root_tags={ARTIFACT_PATH_TAG: recorded_path},
        metadata=dict(_OWNERSHIP_METADATA),
    )
    created = gateway.create_trace(spec)
    local = tmp_path / "artifact.json"
    local.write_bytes(b'{"artifact": "full contract"}')
    verification = gateway.upload_named_json(created.trace_id, recorded_path, local)
    gateway.record_artifact_verification(created.trace_id, verification.sha256, verification.size)
    candidate = _indexed_candidate(gateway, experiment.experiment_id)
    contract = CandidateContract.from_create_request(spec, verification.sha256, verification.size)
    fake_mlflow_client._traces[created.trace_id].info.trace_metadata[field] = mutated

    outcome = gateway.verify_owned_candidate(candidate, contract)

    assert outcome.matches is False
    assert outcome.reason == "metadata_mismatch"
    assert outcome.error_code is None
    assert fake_mlflow_client.get_trace(created.trace_id).trace_id == created.trace_id


# ---------------------------------------------------------------------------
# FakeMlflowGateway (the in-memory gateway used by later BDD tasks)
# ---------------------------------------------------------------------------


def test_fake_gateway_records_event_history_in_order(tmp_path: Path) -> None:
    fake = FakeMlflowGateway()
    assert fake.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True)).is_compatible

    reference = fake.resolve_experiment("bdd-experiment", create=True)
    assert reference.experiment_id is not None
    payload = tmp_path / "payload.json"
    payload.write_bytes(b'{"bdd": "artifact payload"}')

    spec = trace_spec(experiment_id=reference.experiment_id, children=[child_spec("msg-1")])
    created = fake.create_trace(spec)
    fake.verify_trace(created, spec)
    verification = fake.upload_named_json(created.trace_id, "opencode-mlflow/v1/x.json", payload)
    fake.verify_named_json(created.trace_id, verification.relative_path, verification.sha256, verification.size)
    owned = list(fake.iter_owned_traces(reference.experiment_id))
    fake.retire_trace(
        reference.experiment_id,
        created.trace_id,
        session_id=spec.native_session_id,
        turn_key=spec.metadata["opencode.turn_key"],
    )

    kinds = [event.kind for event in fake.events]
    assert kinds == [
        "probe",
        "experiment_create",
        "trace_create",
        "span_start",
        "span_end",
        "trace_end",
        "trace_verify",
        "artifact_upload",
        "artifact_verify",
        "trace_retire",
    ]
    assert [trace.trace_id for trace in owned] == [created.trace_id]
    assert fake.mutations == [
        ("experiment_create", "bdd-experiment", reference.experiment_id),
        ("trace_create", created.trace_id),
        ("artifact_upload", created.trace_id, "opencode-mlflow/v1/x.json"),
        ("trace_retire", reference.experiment_id, created.trace_id),
    ]


def test_fake_gateway_enforces_the_same_mutation_gates() -> None:
    unchecked = FakeMlflowGateway()
    with pytest.raises(CapabilityError):
        unchecked.resolve_experiment("x", create=True)

    dry = FakeMlflowGateway(dry_run=True)
    assert dry.probe(CapabilityRequest()).is_compatible
    planned = dry.resolve_experiment("x", create=True)
    assert planned.planned_creation is True
    with pytest.raises(CapabilityError):
        dry.create_trace(trace_spec())

    incompatible = FakeMlflowGateway(compatible=False)
    assert not incompatible.probe(CapabilityRequest()).is_compatible
    with pytest.raises(CapabilityError):
        incompatible.resolve_experiment("x", create=True)


def test_fake_gateway_keeps_payload_state_for_bdd_assertions() -> None:
    fake = FakeMlflowGateway()
    fake.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
    reference = fake.resolve_experiment("bdd", create=True)
    assert reference.experiment_id is not None
    spec = trace_spec(
        experiment_id=reference.experiment_id,
        root_inputs={"prompt": "hello"},
        root_outputs={"answer": 42},
        children=[child_spec("msg-1", inputs={"text": "hi"}, outputs={"text": "yo"})],
    )
    created = fake.create_trace(spec)

    assert created.verified_metadata[TRACE_SESSION_METADATA_KEY] == spec.native_session_id
    assert created.root_status == "OK"
    fake.verify_trace(created, spec)

    with pytest.raises(ItemImportError):
        fake.verify_trace(created, trace_spec(root_outputs={"answer": 43}))
