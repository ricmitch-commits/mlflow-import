"""Contract tests pinning the mapping's request shapes to the gateway boundary.

The mapping is pure, so its contract is the exactness of what it hands
the gateway: TraceSpec/SpanSpec/EventSpec must produce EXACTLY-valid
``TraceCreateRequest``/``TraceChildRequest`` objects that the (fake)
gateway accepts — round-tripped through ``FakeMlflowGateway``'s
``create_trace`` plus ``verify_trace`` — the explicit ``root_status``
must reach the applied/verified root status even when the last child
ended ``OK`` (the case the gateway's derivation could not express), an
oversized COMPLETE root value must raise ``payload_too_large`` BEFORE
any trace creation, and the ownership marker must equal the real
gateway's (so the two can never drift).

No network and no live server: everything runs against the in-memory
fakes and the sanitized synthetic six-turn session fixture.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path

import pytest

from opencode_mlflow.errors import ItemImportError
from opencode_mlflow.mlflow.capabilities import CapabilityRequest
from opencode_mlflow.mlflow.gateway import (
    OWNERSHIP_IMPORTER_VALUE,
    TraceChildRequest,
    TraceCreateRequest,
)
from opencode_mlflow.mlflow.mapping import (
    IMPORTER_IDENTITY,
    TraceSpec,
    bind_mlflow_identity_map,
    map_turn_to_trace,
)
from opencode_mlflow.source.turns import NO_ASSISTANT_RESULT_OUTCOME
from tests.support.fake_gateway import FakeMlflowGateway
from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_record, fake_summary

pytestmark = pytest.mark.contract

_TURN_KEY = "turn-" + "3" * 64
_SESSION_ID = "ses-mapping-contract"


def _fake_turn_with_huge_root() -> FakeReplayTurn:
    """A minimal turn whose COMPLETE root input exceeds any small limit."""
    huge = {"content": [{"type": "text", "text": "x" * 10_000}]}
    prompt = replace(
        fake_record(
            "msg-contract-u1",
            raw_json=None,
            decoded=huge,
            session_id=_SESSION_ID,
            source_type="user",
            classification="user",
            timestamp=None,
        ),
        timestamp=None,
    )
    return FakeReplayTurn(
        session_header=fake_header(_SESSION_ID),
        summary=fake_summary(
            _SESSION_ID,
            turn_key=_TURN_KEY,
            terminal_outcome=NO_ASSISTANT_RESULT_OUTCOME,
            root_result_source_id=None,
        ),
        turn_records=(prompt,),
        root_input=huge,
        root_output=None,
    )


def _checked_fake_gateway() -> FakeMlflowGateway:
    fake = FakeMlflowGateway()
    report = fake.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
    assert report.is_compatible, report.safe_error
    reference = fake.resolve_experiment("contract-mapping", create=True)
    assert reference.experiment_id is not None
    return fake


# ---------------------------------------------------------------------------
# Field parity
# ---------------------------------------------------------------------------


def test_trace_spec_fields_exactly_match_the_create_request() -> None:
    assert {field.name for field in dataclass_fields(TraceSpec)} == {
        field.name for field in dataclass_fields(TraceCreateRequest)
    }


def test_span_spec_fields_exactly_match_the_child_request() -> None:
    from opencode_mlflow.mlflow.mapping import SpanSpec

    assert {field.name for field in dataclass_fields(SpanSpec)} == {
        field.name for field in dataclass_fields(TraceChildRequest)
    }


def test_ownership_marker_matches_the_real_gateway_constant() -> None:
    assert IMPORTER_IDENTITY == OWNERSHIP_IMPORTER_VALUE


# ---------------------------------------------------------------------------
# Round-trip through the fake gateway
# ---------------------------------------------------------------------------


def test_six_turn_specs_round_trip_through_the_fake_gateway(six_turn_session) -> None:
    fake = _checked_fake_gateway()

    created_ids = []
    for turn in six_turn_session.turns:
        spec = map_turn_to_trace(turn, six_turn_session.context)
        request = spec.to_create_request()

        assert isinstance(request, TraceCreateRequest)
        assert request.client_request_id == spec.client_request_id
        assert request.root_status == spec.root_status
        assert request.artifact_relative_path == spec.artifact_relative_path
        assert request.artifact_snapshot_fingerprint == spec.artifact_snapshot_fingerprint

        created = fake.create_trace(request)
        fake.verify_trace(created, request)
        created_ids.append(created.trace_id)

        # The ownership metadata the fake stores is the exact submitted
        # set (native session key included by the gateway contract).
        assert created.verified_metadata["opencode.importer"] == IMPORTER_IDENTITY
        assert created.verified_metadata["opencode.session_id"] == six_turn_session.session_id
        assert created.verified_metadata["opencode.operation_key"] == spec.expected_operation_key

    assert len(set(created_ids)) == 6
    assert [event.kind for event in fake.events].count("trace_create") == 6
    assert [event.kind for event in fake.events].count("trace_verify") == 6


def test_explicit_error_root_status_verifies_even_when_the_last_child_ended_ok(
    six_turn_session,
) -> None:
    # The case the gateway's derivation could not express: the mapping
    # marks the turn an explicit failure while its LAST child (the text
    # part emitted after the error part... reordered here deliberately)
    # ends OK. The explicit root_status must be applied and verified.
    fake = _checked_fake_gateway()
    failed_turn = six_turn_session.turns[3]
    spec = map_turn_to_trace(failed_turn, six_turn_session.context)
    request = spec.to_create_request()

    # Reorder the fixture turn's children so the last one ends OK while
    # the root status stays the authoritative explicit ERROR: the error
    # part is moved before the final text part (parent order preserved,
    # so every parent still precedes its parts).
    assert spec.root_status == "ERROR"
    error_first = [
        "session_message:msg-synthetic-u4#content-0",
        "session_message:msg-synthetic-a4",
        "session_message:msg-synthetic-a4#content-1",
        "session_message:msg-synthetic-a4#content-0",
    ]
    by_key = {child.source_key: child for child in spec.child_specs()}
    reordered = [by_key[key] for key in error_first]
    request = replace(request, child_specs=lambda: (child.to_child_request() for child in reordered))

    created = fake.create_trace(request)

    assert created.root_status == "ERROR"
    assert reordered[-1].status == "OK"
    fake.verify_trace(created, request)


def test_round_tripped_children_are_real_child_requests(six_turn_session) -> None:
    fake = _checked_fake_gateway()
    spec = map_turn_to_trace(six_turn_session.turns[0], six_turn_session.context)
    request = spec.to_create_request()
    children = list(request.child_specs())

    created = fake.create_trace(request)

    assert children and all(isinstance(child, TraceChildRequest) for child in children)
    assert created.span_ids_by_source.keys() == {
        "session_message:msg-synthetic-u1#content-0",
        "session_message:msg-synthetic-a1",
        "session_message:msg-synthetic-a1#content-0",
        "session_message:msg-synthetic-a1#content-1",
        "session_message:msg-synthetic-a1#content-2",
        "session_pending:pending-synthetic-1",
        "session_pending:pending-synthetic-2",
    }


def test_identity_map_binds_created_ids_to_the_artifact_envelope(six_turn_session) -> None:
    fake = _checked_fake_gateway()
    turn = six_turn_session.turns[0]
    spec = map_turn_to_trace(turn, six_turn_session.context)
    request = spec.to_create_request()

    created = fake.create_trace(request)
    identity_map = bind_mlflow_identity_map(spec.identity_envelope(), created.trace_id, created.span_ids_by_source)

    assert identity_map.trace_id == created.trace_id
    assert identity_map.span_ids_by_source == dict(created.span_ids_by_source)
    assert identity_map.artifact_relative_path == spec.artifact_relative_path
    section = identity_map.artifact_section()
    assert section["trace_id"] == created.trace_id
    assert all(section[f"span:{key}"] == span_id for key, span_id in created.span_ids_by_source.items())


# ---------------------------------------------------------------------------
# Payload-too-large rejection before trace creation
# ---------------------------------------------------------------------------


def test_oversized_root_value_raises_payload_too_large_before_trace_creation(
    six_turn_session, restrictive_tmp_path: Path
) -> None:
    fake = _checked_fake_gateway()
    context = replace(six_turn_session.context, inline_limit_bytes=64)
    turn = _fake_turn_with_huge_root()

    with pytest.raises(ItemImportError) as excinfo:
        map_turn_to_trace(turn, context)

    assert excinfo.value.code == "payload_too_large"
    # The mapping failed before any gateway call was possible, so no
    # trace was created for the oversized turn.
    assert all(event.kind != "trace_create" for event in fake.events)
    assert not any(mutation[0] == "trace_create" for mutation in fake.mutations)
