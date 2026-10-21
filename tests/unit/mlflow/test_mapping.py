"""Unit tests for the pure turn-to-trace mapping.

The mapping consumes the ``ReplayTurn`` protocol and produces gateway
request specs without ever touching MLflow. These tests pin the pure
contract: root values, typed children, ownership metadata, timing
fallbacks, outcome-to-status mapping, lazy/bounded payloads, identity-map
binding, and module purity. The six-turn fixture exercises the real
pipeline (fixture database → snapshot → replay turns); the fake replay
turns isolate single-record and single-turn rules.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path

import pytest

from opencode_mlflow.content.canonical import canonical_json
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.errors import ItemImportError
from opencode_mlflow.mlflow import mapping
from opencode_mlflow.mlflow.mapping import (
    EVENT_SPAN_TYPE,
    IMPORTER_IDENTITY,
    OUTCOME_EXPLICIT_FAILURE,
    ArtifactIdentityEnvelope,
    EventSpec,
    MappingContext,
    MlflowIdentityMap,
    SourceIdentity,
    SpanSpec,
    TraceSpec,
    bind_mlflow_identity_map,
    map_source_record,
    map_turn_to_trace,
    operation_key,
)
from opencode_mlflow.source.model import SessionHeader, SourceRecord, TurnSummary, normalize_timestamp
from opencode_mlflow.source.turns import ASSISTANT_RESULT_OUTCOME, NO_ASSISTANT_RESULT_OUTCOME, iter_turn_summaries
from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_record, fake_summary

pytestmark = pytest.mark.unit


class _IdentityRedactor:
    """Redactor double that preserves every value (replayed values are effective)."""

    def redact(self, value: object, protected_paths: object) -> object:
        return value


_TURN_KEY = "turn-" + "1" * 64
_OTHER_TURN_KEY = "turn-" + "2" * 64
_SESSION_ID = "ses-mapping-unit"
_EXPERIMENT_ID = "exp-mapping-unit"


def _context(**overrides: object) -> MappingContext:
    values: dict[str, object] = {
        "importer_version": "0.1.0",
        "mapping_version": "mapping-v1",
        "artifact_schema_version": "artifact-v1",
        "source_schema": "current",
        "redaction_mode": RedactionMode.VERBATIM,
        "experiment_id": _EXPERIMENT_ID,
        "source_database_fingerprint": "synthetic-database-fingerprint",
        "logical_turn_count": 3,
        "source_message_count": 7,
        "source_part_count": 5,
    }
    values.update(overrides)
    return MappingContext(**values)  # type: ignore[arg-type]


def _message(
    source_id: str,
    message_type: str,
    seq: int,
    payload: object,
    *,
    session_id: str = _SESSION_ID,
    ts: int | None = 1_772_000_010_000,
) -> SourceRecord:
    record = fake_record(
        source_id,
        raw_json=None,
        decoded=payload,
        session_id=session_id,
        source_type=message_type,
        seq=seq,
        classification=message_type if message_type in ("user", "assistant", "system") else None,
        timestamp=None,
    )
    return replace(record, timestamp=normalize_timestamp(ts, "ms") if ts is not None else None)


def _part(
    source_id: str,
    parent_id: str,
    payload: object,
    *,
    ordinal: int = 0,
    seq: int = 1,
    ts: int | None = 1_772_000_011_000,
    source_table: str = "session_message",
) -> SourceRecord:
    record = fake_record(
        source_id,
        raw_json=None,
        decoded=payload,
        session_id=_SESSION_ID,
        source_table=source_table,
        source_type=payload.get("type") if isinstance(payload, dict) else "unknown",
        seq=seq,
        classification=None,
        timestamp=None,
    )
    return replace(
        record,
        parent_source_id=parent_id,
        derived_child_ordinal=ordinal,
        timestamp=normalize_timestamp(ts, "ms") if ts is not None else None,
    )


def _user_payload(text: str = "synthetic unit prompt") -> dict[str, object]:
    return {"content": [{"type": "text", "text": text}]}


def _summary(**overrides: object) -> TurnSummary:
    values: dict[str, object] = {
        "session_id": _SESSION_ID,
        "turn_index": 1,
        "turn_key": _TURN_KEY,
        "first_user_source_id": "msg-u1",
        "first_user_source_type": "user",
        "start_time": normalize_timestamp(1_772_000_010_000, "ms"),
        "end_time": normalize_timestamp(1_772_000_011_000, "ms"),
        "record_count": 3,
        "user_record_count": 1,
        "assistant_record_count": 1,
        "terminal_outcome": ASSISTANT_RESULT_OUTCOME,
        "root_result_source_id": "msg-a1",
    }
    values.update(overrides)
    return fake_summary(**values)  # type: ignore[arg-type]


def _replay_turn(
    records: list[SourceRecord],
    *,
    summary: TurnSummary | None = None,
    header: SessionHeader | None = None,
    root_input: object | None = None,
    root_output: object | None = None,
) -> FakeReplayTurn:
    return FakeReplayTurn(
        session_header=header if header is not None else fake_header(_SESSION_ID),
        summary=summary if summary is not None else _summary(),
        turn_records=tuple(records),
        root_input=root_input,
        root_output=root_output,
    )


# ---------------------------------------------------------------------------
# The verbatim six-turn shape test from the task brief
# ---------------------------------------------------------------------------


def test_mapping_preserves_six_turn_session_shape(six_turn_session) -> None:
    specs = [map_turn_to_trace(turn, six_turn_session.context) for turn in six_turn_session.turns]

    assert len(specs) == 6
    assert {spec.native_session_id for spec in specs} == {"ses-synthetic-1"}
    assert len({spec.client_request_id for spec in specs}) == 6
    assert all(spec.metadata["opencode.session_id"] == "ses-synthetic-1" for spec in specs)


# ---------------------------------------------------------------------------
# Module purity
# ---------------------------------------------------------------------------


def test_mapping_module_declares_no_mlflow_imports() -> None:
    tree = ast.parse(Path(mapping.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "mlflow" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module and node.module.split(".")[0] == "mlflow")


def test_importing_the_mapping_module_never_loads_mlflow() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, opencode_mlflow.mlflow.mapping; sys.exit(0 if 'mlflow' not in sys.modules else 1)",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Spec/request shape parity
# ---------------------------------------------------------------------------


def test_trace_spec_field_parity_with_the_create_request() -> None:
    from opencode_mlflow.mlflow.gateway import TraceCreateRequest

    assert {field.name for field in dataclass_fields(TraceSpec)} == {
        field.name for field in dataclass_fields(TraceCreateRequest)
    }


def test_span_spec_field_parity_with_the_child_request() -> None:
    from opencode_mlflow.mlflow.gateway import TraceChildRequest

    assert {field.name for field in dataclass_fields(SpanSpec)} == {
        field.name for field in dataclass_fields(TraceChildRequest)
    }


def test_event_spec_field_parity_with_the_child_request() -> None:
    from opencode_mlflow.mlflow.gateway import TraceChildRequest

    assert {field.name for field in dataclass_fields(EventSpec)} | {"span_type"} == {
        field.name for field in dataclass_fields(TraceChildRequest)
    }
    assert EventSpec.span_type is not None


def test_specs_convert_to_real_gateway_requests() -> None:
    from opencode_mlflow.mlflow.gateway import TraceChildRequest, TraceCreateRequest

    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload()),
            _message("msg-a1", "assistant", 2, {"content": [{"type": "text", "text": "synthetic answer"}]}),
        ],
        root_input=_user_payload(),
        root_output=[{"type": "text", "text": "synthetic answer"}],
    )
    spec = map_turn_to_trace(turn, _context())

    request = spec.to_create_request()
    assert isinstance(request, TraceCreateRequest)
    assert request.client_request_id == spec.client_request_id
    assert request.root_status == spec.root_status
    assert request.metadata == dict(spec.metadata)
    children = list(request.child_specs())
    assert children and all(isinstance(child, TraceChildRequest) for child in children)
    assert children[0].source_key == "session_message:msg-a1"
    assert children[0].span_type == "AGENT"


def test_event_spec_converts_with_the_event_span_type() -> None:
    from opencode_mlflow.mlflow.gateway import TraceChildRequest

    event = EventSpec(
        source_key="msg-a1#content-0",
        name="reasoning part",
        parent_source_key="msg-a1",
        inputs={"type": "reasoning", "text": "synthetic reasoning"},
        attributes={"opencode.part_type": "reasoning"},
        tags={},
        start_time_ns=1,
        outputs=None,
        status="OK",
        end_time_ns=1,
    )
    request = event.to_child_request()
    assert isinstance(request, TraceChildRequest)
    assert request.span_type == EVENT_SPAN_TYPE == "event"
    assert request.source_key == event.source_key


# ---------------------------------------------------------------------------
# Root values
# ---------------------------------------------------------------------------


def test_first_user_prompt_is_the_root_input() -> None:
    payload = _user_payload("synthetic root prompt")
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, payload), _message("msg-a1", "assistant", 2, {"content": []})],
        root_input=payload,
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_inputs == payload


def test_final_assistant_result_is_the_root_output() -> None:
    output = [{"type": "text", "text": "synthetic final answer"}]
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload()), _message("msg-a1", "assistant", 2, {"content": []})],
        root_input=_user_payload(),
        root_output=output,
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_outputs == output


def test_split_final_result_parts_combine_in_order() -> None:
    first = {"type": "text", "text": "synthetic final part one"}
    second = {"type": "text", "text": "synthetic final part two"}
    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload()),
            _message("msg-a1", "assistant", 2, {"content": []}),
            _part("msg-a1#content-0", "msg-a1", first, ordinal=0),
            _part("msg-a1#content-1", "msg-a1", second, ordinal=1),
        ],
        root_input=_user_payload(),
        root_output=[first, second],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_outputs == [first, second]


def test_no_assistant_result_omits_the_root_output() -> None:
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload())],
        summary=_summary(terminal_outcome=NO_ASSISTANT_RESULT_OUTCOME, root_result_source_id=None),
        root_input=_user_payload(),
        root_output=None,
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_outputs is None
    assert spec.root_status == "UNSET"
    assert spec.metadata["opencode.outcome"] == NO_ASSISTANT_RESULT_OUTCOME


def test_oversized_root_value_fails_with_payload_too_large() -> None:
    huge = {"content": [{"type": "text", "text": "x" * 10_000}]}
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, huge), _message("msg-a1", "assistant", 2, {"content": []})],
        root_input=huge,
        root_output=[],
    )

    with pytest.raises(ItemImportError) as excinfo:
        map_turn_to_trace(turn, _context(inline_limit_bytes=64))

    assert excinfo.value.code == "payload_too_large"


# ---------------------------------------------------------------------------
# Outcome-to-status mapping
# ---------------------------------------------------------------------------


def test_successful_result_maps_to_ok() -> None:
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload()), _message("msg-a1", "assistant", 2, {"content": []})],
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_status == "OK"
    assert spec.metadata["opencode.outcome"] == ASSISTANT_RESULT_OUTCOME


def test_error_part_on_the_root_result_maps_to_explicit_failure() -> None:
    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload()),
            _message("msg-a1", "assistant", 2, {"content": []}),
            _part("msg-a1#content-0", "msg-a1", {"type": "text", "text": "synthetic partial"}, ordinal=0),
            _part("msg-a1#content-1", "msg-a1", {"type": "error", "error": {"message": "boom"}}, ordinal=1),
        ],
        root_input=_user_payload(),
        root_output=[{"type": "text", "text": "synthetic partial"}],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_status == "ERROR"
    assert spec.metadata["opencode.outcome"] == OUTCOME_EXPLICIT_FAILURE


@pytest.mark.parametrize("collision_kind", ["part", "system"])
def test_non_root_record_with_colliding_source_id_cannot_change_root_outcome(collision_kind: str) -> None:
    """Root outcome matching is qualified by the actual root record shape."""
    if collision_kind == "part":
        colliding = _part(
            "shared-root-id",
            "different-parent",
            {"type": "error", "error": {"message": "not the root"}},
            source_table="part",
        )
    else:
        colliding = fake_record(
            "shared-root-id",
            raw_json=None,
            decoded={"status": "failed"},
            session_id=_SESSION_ID,
            source_table="system",
            source_type="system",
            seq=None,
            classification="system",
        )

    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload()),
            _message("shared-root-id", "assistant", 2, {"content": []}),
            colliding,
        ],
        summary=_summary(root_result_source_id="shared-root-id", terminal_outcome=ASSISTANT_RESULT_OUTCOME),
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_status == "OK"
    assert spec.metadata["opencode.outcome"] == ASSISTANT_RESULT_OUTCOME


@pytest.mark.parametrize("field", ["status", "finish"])
@pytest.mark.parametrize("outcome", ["error", "failed", "failure", "cancelled", "interrupted"])
def test_message_failure_outcomes_map_root_to_error_and_preserve_source_value(field: str, outcome: str) -> None:
    payload = {"content": [], field: outcome}
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload()), _message("msg-a1", "assistant", 2, payload)],
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_status == "ERROR"
    assert spec.metadata["opencode.outcome"] == outcome
    assert list(spec.child_specs())[-1].status == "ERROR"


def test_later_successful_assistant_result_supersedes_an_earlier_failed_result() -> None:
    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload()),
            _message("msg-a1", "assistant", 2, {"status": "failed", "content": []}),
            _message("msg-a2", "assistant", 3, {"content": []}),
        ],
        summary=_summary(root_result_source_id="msg-a2", terminal_outcome=ASSISTANT_RESULT_OUTCOME),
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_status == "OK"
    assert spec.metadata["opencode.outcome"] == ASSISTANT_RESULT_OUTCOME
    children = {child.source_key: child for child in spec.child_specs()}
    assert children["session_message:msg-a1"].status == "ERROR"
    assert children["session_message:msg-a2"].status == "OK"


def test_explicit_failure_wins_even_when_the_last_child_ended_ok() -> None:
    # The derivation the gateway could not express: the root result carries
    # an explicit error part, but the LAST child (a later text part) is OK.
    # The authoritative explicit root_status must stay ERROR.
    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload()),
            _message("msg-a1", "assistant", 2, {"content": []}),
            _part("msg-a1#content-0", "msg-a1", {"type": "error", "error": {"message": "boom"}}, ordinal=0),
            _part("msg-a1#content-1", "msg-a1", {"type": "text", "text": "synthetic recovery"}, ordinal=1),
        ],
        root_input=_user_payload(),
        root_output=[{"type": "text", "text": "synthetic recovery"}],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_status == "ERROR"
    children = list(spec.child_specs())
    assert children[-1].status == "OK"


# ---------------------------------------------------------------------------
# Typed child mapping
# ---------------------------------------------------------------------------


def test_model_call_message_maps_to_an_llm_span() -> None:
    payload = {
        "content": [{"type": "text", "text": "synthetic model answer"}],
        "model": {"provider": "synthetic-provider", "model": "synthetic-model"},
        "providerID": "synthetic-provider",
        "tokens": {"input": 120, "output": 80, "reasoning": 12, "cache": {"read": 40, "write": 4}},
        "cost": 0.0025,
        "finish": "stop",
    }
    record = _message("msg-a1", "assistant", 2, payload)

    spec = map_source_record(record, None)

    assert isinstance(spec, SpanSpec)
    assert spec.span_type == "LLM"
    assert spec.name == "model synthetic-provider/synthetic-model"
    assert spec.attributes["opencode.model"] == "synthetic-provider/synthetic-model"
    assert spec.attributes["opencode.provider"] == "synthetic-provider"
    assert spec.attributes["opencode.tokens.input"] == 120
    assert spec.attributes["opencode.tokens.output"] == 80
    assert spec.attributes["opencode.tokens.reasoning"] == 12
    assert spec.attributes["opencode.tokens.cache_read"] == 40
    assert spec.attributes["opencode.tokens.cache_write"] == 4
    assert spec.attributes["opencode.cost"] == 0.0025
    assert spec.attributes["opencode.finish"] == "stop"
    assert spec.inputs == payload


def test_model_call_envelope_inputs_and_outputs_extract_the_typed_keys() -> None:
    payload = {
        "model": "synthetic-provider/synthetic-model",
        "input": {"messages": ["synthetic"]},
        "output": {"text": "synthetic completion"},
        "tokens": {"input": 10, "output": 5},
    }
    record = _message("msg-a1", "assistant", 2, payload)

    spec = map_source_record(record, None)

    assert isinstance(spec, SpanSpec)
    assert spec.span_type == "LLM"
    assert spec.inputs == {"messages": ["synthetic"]}
    assert spec.outputs == {"text": "synthetic completion"}


def test_plain_assistant_message_maps_to_an_agent_span() -> None:
    payload = {"content": [{"type": "text", "text": "synthetic plain answer"}]}
    record = _message("msg-a1", "assistant", 2, payload)

    spec = map_source_record(record, "msg-u1")

    assert isinstance(spec, SpanSpec)
    assert spec.span_type == "AGENT"
    assert spec.name == "assistant message"
    assert spec.parent_source_key == "msg-u1"
    assert spec.inputs == payload
    assert spec.status == "OK"


def test_tool_part_maps_to_a_tool_span() -> None:
    payload = {
        "type": "tool",
        "tool": "bash",
        "call_id": "synthetic-call-1",
        "state": {"input": {"command": "ls synthetic"}, "output": "synthetic file list", "status": "completed"},
    }
    record = _part("msg-a1#content-1", "msg-a1", payload, ordinal=1)

    spec = map_source_record(record, "msg-a1")

    assert isinstance(spec, SpanSpec)
    assert spec.span_type == "TOOL"
    assert spec.name == "tool bash"
    assert spec.parent_source_key == "msg-a1"
    assert spec.inputs == {"command": "ls synthetic"}
    assert spec.outputs == "synthetic file list"
    assert spec.status == "OK"
    assert spec.attributes["opencode.tool"] == "bash"
    assert spec.attributes["opencode.tool_call_id"] == "synthetic-call-1"
    assert spec.attributes["opencode.tool_status"] == "completed"
    assert spec.start_time_ns == 1_772_000_011_000_000_000


def test_failed_tool_part_maps_to_an_error_status_span() -> None:
    payload = {
        "type": "tool",
        "tool": "bash",
        "state": {"input": {"command": "ls synthetic"}, "status": "error", "error": "synthetic tool failure"},
    }
    record = _part("msg-a1#content-1", "msg-a1", payload, ordinal=1)

    spec = map_source_record(record, "msg-a1")

    assert isinstance(spec, SpanSpec)
    assert spec.status == "ERROR"
    assert spec.attributes["opencode.error"] == "synthetic tool failure"
    assert spec.outputs is None


@pytest.mark.parametrize("status", ["failed", "failure", "cancelled", "interrupted"])
def test_all_explicit_failed_tool_statuses_map_to_error(status: str) -> None:
    payload = {
        "type": "tool",
        "tool": "bash",
        "state": {"input": {"command": "ls synthetic"}, "status": status},
    }

    spec = map_source_record(_part("msg-a1#content-1", "msg-a1", payload, ordinal=1), "msg-a1")

    assert spec.status == "ERROR"
    assert spec.attributes["opencode.source_outcome"] == status


def test_pending_tool_part_maps_to_an_unset_status_span() -> None:
    payload = {"type": "tool", "tool": "bash", "state": {"input": {"command": "ls synthetic"}, "status": "running"}}
    record = _part("msg-a1#content-1", "msg-a1", payload, ordinal=1)

    spec = map_source_record(record, "msg-a1")

    assert isinstance(spec, SpanSpec)
    assert spec.status == "UNSET"


def test_text_part_maps_to_a_content_span_and_annotations_to_events() -> None:
    text = {"type": "text", "text": "synthetic answer"}
    reasoning = {"type": "reasoning", "text": "synthetic reasoning"}
    error = {"type": "error", "error": {"message": "synthetic failure"}}

    text_spec = map_source_record(_part("p-text", "msg-a1", text, ordinal=0), "msg-a1")
    reasoning_spec = map_source_record(_part("p-reasoning", "msg-a1", reasoning, ordinal=1), "msg-a1")
    error_spec = map_source_record(_part("p-error", "msg-a1", error, ordinal=2), "msg-a1")

    assert isinstance(text_spec, SpanSpec)
    assert not isinstance(text_spec, EventSpec)
    assert text_spec.span_type != EVENT_SPAN_TYPE
    assert text_spec.name == "text part"
    assert text_spec.inputs == text

    assert isinstance(reasoning_spec, EventSpec)
    assert reasoning_spec.name == "reasoning part"
    assert reasoning_spec.inputs == reasoning

    assert isinstance(error_spec, EventSpec)
    assert error_spec.status == "ERROR"
    assert error_spec.attributes["opencode.error"] == {"message": "synthetic failure"}


def test_legacy_part_rows_map_as_content_parts() -> None:
    payload = {"type": "text", "text": "synthetic legacy part"}
    record = _part("part-legacy-1", "msg-legacy-1", payload, ordinal=0, source_table="part")

    spec = map_source_record(record, "msg-legacy-1")

    assert isinstance(spec, SpanSpec)
    assert spec.span_type != EVENT_SPAN_TYPE
    assert spec.inputs == payload
    assert spec.attributes["opencode.source_table"] == "part"


def test_lifecycle_record_maps_to_a_lifecycle_event() -> None:
    record = replace(
        fake_record(
            "pending-1",
            raw_json=None,
            decoded={"name": "synthetic-pending-tool", "outcome": "admitted"},
            source_table="session_pending",
            source_type="tool",
            seq=None,
            classification=None,
            timestamp=normalize_timestamp(1_772_000_012_000, "ms"),
        ),
        sequence_domain=None,
    )

    spec = map_source_record(record, None)

    assert isinstance(spec, EventSpec)
    assert spec.name == "lifecycle tool"
    assert spec.attributes["opencode.lifecycle_outcome"] == "admitted"
    assert spec.attributes["opencode.source_table"] == "session_pending"
    assert spec.start_time_ns == 1_772_000_012_000_000_000


def test_unknown_record_maps_to_an_opaque_event_with_the_normalized_record() -> None:
    record = replace(
        fake_record(
            "todo-1",
            raw_json=None,
            decoded={"content": "synthetic todo"},
            source_table="todo",
            source_type="todo",
            seq=None,
            classification=None,
            timestamp=normalize_timestamp(1_772_000_013_000, "ms"),
        ),
        sequence_domain=None,
    )

    spec = map_source_record(record, None)

    assert isinstance(spec, EventSpec)
    assert spec.name == "lifecycle todo"
    normalized = spec.attributes["opencode.record"]
    assert isinstance(normalized, dict)
    assert normalized["source_table"] == "todo"
    assert normalized["decoded"] == {"content": "synthetic todo"}


def test_todo_status_from_raw_columns_maps_to_a_lifecycle_error_event() -> None:
    record = replace(
        fake_record(
            "todo-failed",
            raw_json=None,
            decoded=None,
            raw_columns={"status": "failed"},
            source_table="todo",
            source_type="todo",
            seq=None,
            classification=None,
        ),
        sequence_domain=None,
    )

    spec = map_source_record(record, None)

    assert isinstance(spec, EventSpec)
    assert spec.name == "lifecycle todo"
    assert spec.attributes["opencode.lifecycle_outcome"] == "failed"
    assert spec.status == "ERROR"


@pytest.mark.parametrize("status, expected", [("failed", "ERROR"), ("pending", "UNSET")])
def test_session_todo_raw_outcome_controls_no_assistant_turn_status(status: str, expected: str) -> None:
    user = fake_record("user-lifecycle", classification="user")
    todo = replace(
        fake_record(
            "todo-lifecycle",
            raw_json=None,
            decoded=None,
            raw_columns={"status": status},
            source_table="todo",
            source_type="todo",
            classification=None,
        ),
        sequence_domain=None,
    )
    turn = FakeReplayTurn(
        session_header=fake_header(),
        summary=next(iter_turn_summaries([user])),
        turn_records=(user,),
        session_records=(todo,),
        root_input={"text": "synthetic prompt"},
        root_output=None,
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_status == expected
    assert spec.metadata["opencode.outcome"] == f"lifecycle_{status}"


def test_successful_assistant_result_supersedes_earlier_pending_lifecycle() -> None:
    user = fake_record("user-lifecycle-success", classification="user")
    pending = replace(
        fake_record(
            "pending-lifecycle",
            raw_json=None,
            decoded={"status": "pending"},
            source_table="session_pending",
            source_type="tool",
            classification=None,
        ),
        sequence_domain=None,
    )
    assistant = fake_record(
        "assistant-lifecycle-success",
        decoded={"role": "assistant", "status": "completed", "content": "done"},
        source_type="assistant",
        classification="assistant",
        seq=2,
    )
    summary = next(iter_turn_summaries([user, assistant]))
    turn = FakeReplayTurn(
        session_header=fake_header(),
        summary=summary,
        turn_records=(user, assistant),
        session_records=(pending,),
        root_input={"text": "synthetic prompt"},
        root_output={"text": "synthetic reply"},
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_status == "OK"
    assert spec.metadata["opencode.outcome"] == "completed"


def test_lifecycle_error_outcome_remains_terminal_after_later_assistant_success() -> None:
    user = fake_record("user-lifecycle-error", classification="user")
    lifecycle_failure = replace(
        fake_record(
            "failed-lifecycle",
            raw_json=None,
            decoded={"lifecycle_error_outcome": "failed"},
            source_table="session_pending",
            source_type="tool",
            classification=None,
        ),
        sequence_domain=None,
    )
    assistant = fake_record(
        "assistant-after-lifecycle-error",
        decoded={"role": "assistant", "status": "completed", "content": "done"},
        source_type="assistant",
        classification="assistant",
        seq=2,
    )
    summary = next(iter_turn_summaries([user, assistant]))
    turn = FakeReplayTurn(
        session_header=fake_header(),
        summary=summary,
        turn_records=(user, assistant),
        session_records=(lifecycle_failure,),
        root_input={"text": "synthetic prompt"},
        root_output={"text": "synthetic reply"},
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_status == "ERROR"
    assert spec.metadata["opencode.outcome"] == "lifecycle_failed"


def test_failed_opaque_record_maps_to_error_but_unknown_status_does_not() -> None:
    failed = replace(
        fake_record(
            "opaque-failed",
            raw_json=None,
            decoded={"status": "failed"},
            source_table="future_record",
            source_type="future",
            seq=None,
            classification=None,
        ),
        sequence_domain=None,
    )
    unknown = replace(failed, source_id="opaque-unknown", decoded_value={"status": "future-state"})

    failed_spec = map_source_record(failed, None)
    unknown_spec = map_source_record(unknown, None)

    assert failed_spec.status == "ERROR"
    assert failed_spec.attributes["opencode.source_outcome"] == "failed"
    assert unknown_spec.status == "OK"


def test_failed_opaque_row_status_maps_to_error_from_retained_raw_columns() -> None:
    record = replace(
        fake_record(
            "todo-failed",
            raw_json=None,
            decoded=None,
            source_table="todo",
            source_type="todo",
            seq=None,
            classification=None,
            raw_columns={"status": "failed"},
        ),
        sequence_domain=None,
    )

    spec = map_source_record(record, None)

    assert spec.status == "ERROR"
    assert spec.attributes["opencode.source_outcome"] == "failed"


def test_children_preserve_source_identity_sequence_and_times() -> None:
    payload = {"content": [{"type": "text", "text": "synthetic answer"}]}
    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload(), ts=1_772_000_010_000),
            _message("msg-a1", "assistant", 7, payload, ts=1_772_000_020_000),
        ],
        root_input=_user_payload(),
        root_output=[{"type": "text", "text": "synthetic answer"}],
    )

    spec = map_turn_to_trace(turn, _context())
    children = list(spec.child_specs())

    assert [child.source_key for child in children] == ["session_message:msg-a1"]
    child = children[0]
    assert child.attributes["opencode.source_id"] == "msg-a1"
    assert child.attributes["opencode.sequence"] == 7
    assert child.attributes["opencode.sequence_domain"] == "session_message.seq"
    assert child.attributes["opencode.timestamp_ns"] == 1_772_000_020_000_000_000
    assert child.start_time_ns == 1_772_000_020_000_000_000
    assert child.end_time_ns == 1_772_000_020_000_000_000


def test_child_parent_linkage_follows_source_structure() -> None:
    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload()),
            _part("msg-u1#content-0", "msg-u1", {"type": "text", "text": "synthetic prompt"}, ordinal=0),
            _message("msg-a1", "assistant", 2, {"content": []}),
            _part("msg-a1#content-0", "msg-a1", {"type": "reasoning", "text": "synthetic"}, ordinal=0),
            _part("msg-a1#content-1", "msg-a1", {"type": "text", "text": "synthetic"}, ordinal=1),
        ],
        root_input=_user_payload(),
        root_output=[{"type": "text", "text": "synthetic"}],
    )

    spec = map_turn_to_trace(turn, _context())
    children = {child.source_key: child for child in spec.child_specs()}

    # The first-user prompt is the root input, not a child; its part
    # therefore parents to the trace root.
    assert set(children) == {
        "session_message:msg-u1#content-0",
        "session_message:msg-a1",
        "session_message:msg-a1#content-0",
        "session_message:msg-a1#content-1",
    }
    assert children["session_message:msg-u1#content-0"].parent_source_key is None
    assert children["session_message:msg-a1"].parent_source_key is None
    assert children["session_message:msg-a1#content-0"].parent_source_key == "session_message:msg-a1"
    assert children["session_message:msg-a1#content-1"].parent_source_key == "session_message:msg-a1"


def test_earlier_assistant_results_stay_children() -> None:
    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload()),
            _message("msg-a2-early", "assistant", 2, {"content": [{"type": "text", "text": "draft"}]}),
            _message("msg-a2", "assistant", 3, {"content": []}),
            _part("msg-a2#content-0", "msg-a2", {"type": "text", "text": "final"}, ordinal=0),
        ],
        summary=_summary(root_result_source_id="msg-a2"),
        root_input=_user_payload(),
        root_output=[{"type": "text", "text": "final"}],
    )

    spec = map_turn_to_trace(turn, _context())
    children = {child.source_key: child for child in spec.child_specs()}

    assert "session_message:msg-a2-early" in children
    assert children["session_message:msg-a2-early"].span_type == "AGENT"
    assert children["session_message:msg-a2"].span_type == "AGENT"
    assert spec.root_outputs == [{"type": "text", "text": "final"}]


# ---------------------------------------------------------------------------
# Lazy and bounded payloads
# ---------------------------------------------------------------------------


def test_child_specs_are_lazy_with_fresh_cursors() -> None:
    payload = {"content": [{"type": "text", "text": "synthetic answer"}]}
    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload()),
            _message("msg-a1", "assistant", 2, payload),
            _part("msg-a1#content-0", "msg-a1", {"type": "text", "text": "synthetic"}, ordinal=0),
        ],
        root_input=_user_payload(),
        root_output=[{"type": "text", "text": "synthetic"}],
    )

    spec = map_turn_to_trace(turn, _context())
    first = spec.child_specs()
    assert not isinstance(first, list)
    first_children = list(first)
    second_children = list(spec.child_specs())

    assert [child.source_key for child in first_children] == [child.source_key for child in second_children]
    assert [child.source_key for child in first_children] == [
        "session_message:msg-a1",
        "session_message:msg-a1#content-0",
    ]


def test_oversized_child_payloads_become_artifact_references() -> None:
    huge = {"type": "reasoning", "text": "y" * 4_000}
    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload()),
            _message("msg-a1", "assistant", 2, {"content": []}),
            _part("msg-a1#content-0", "msg-a1", huge, ordinal=0),
        ],
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context(inline_limit_bytes=256))

    children = {item.source_key: item for item in spec.child_specs()}
    child = children["session_message:msg-a1#content-0"]
    reference = child.inputs["$artifact_reference"]
    assert child.inputs == {
        "$artifact_reference": {
            "sha256": hashlib.sha256(canonical_json(huge)).hexdigest(),
            "size_bytes": reference["size_bytes"],
        }
    }
    assert reference["size_bytes"] > 256


def test_oversized_opaque_attribute_becomes_an_artifact_reference() -> None:
    opaque = replace(
        fake_record(
            "todo-huge",
            raw_json=None,
            decoded={"content": "z" * 4_000},
            source_table="todo",
            source_type="todo",
            seq=None,
            classification=None,
            timestamp=None,
        ),
        sequence_domain=None,
    )
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload()), opaque],
        summary=_summary(terminal_outcome=NO_ASSISTANT_RESULT_OUTCOME, root_result_source_id=None),
        root_input=_user_payload(),
        root_output=None,
    )

    spec = map_turn_to_trace(turn, _context(inline_limit_bytes=256))

    child = next(iter(spec.child_specs()))
    assert isinstance(child, EventSpec)
    reference = child.attributes["opencode.record"]["$artifact_reference"]
    assert reference["size_bytes"] > 256


# ---------------------------------------------------------------------------
# Ownership and operation metadata
# ---------------------------------------------------------------------------


def test_ownership_metadata_carries_exactly_the_namespaced_keys() -> None:
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload()), _message("msg-a1", "assistant", 2, {"content": []})],
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    required = {
        "opencode.importer",
        "opencode.session_id",
        "opencode.turn_index",
        "opencode.turn_key",
        "opencode.source_fingerprint",
        "opencode.source_schema",
        "opencode.importer_version",
        "opencode.mapping_version",
        "opencode.redaction_mode",
        "opencode.render_fingerprint",
        "opencode.operation_key",
    }
    assert required <= set(spec.metadata)
    assert spec.metadata["opencode.importer"] == IMPORTER_IDENTITY == "opencode-mlflow"
    assert spec.metadata["opencode.session_id"] == _SESSION_ID
    assert spec.metadata["opencode.turn_index"] == "1"
    assert spec.metadata["opencode.turn_key"] == _TURN_KEY
    assert spec.metadata["opencode.source_schema"] == "current"
    assert spec.metadata["opencode.importer_version"] == "0.1.0"
    assert spec.metadata["opencode.mapping_version"] == "mapping-v1"
    assert spec.metadata["opencode.redaction_mode"] == "verbatim"
    assert spec.metadata["opencode.source_database_fingerprint"] == "synthetic-database-fingerprint"
    assert spec.metadata["opencode.logical_turn_count"] == "3"
    assert spec.metadata["opencode.source_message_count"] == "7"
    assert spec.metadata["opencode.source_part_count"] == "5"
    assert spec.metadata["opencode.first_user_source_id"] == "msg-u1"
    assert spec.metadata["opencode.root_result_source_id"] == "msg-a1"


def test_session_metadata_records_available_values_and_omits_absent_ones() -> None:
    header = fake_header(
        _SESSION_ID,
        title="Synthetic mapping session",
        slug="synthetic-slug",
        directory="/repo/synthetic",
        path="/repo/synthetic/session.jsonl",
        project_id="proj-synthetic",
        workspace_id="ws-synthetic",
        parent_id=None,
        fork_session_id=None,
        agent="synthetic-agent",
        version="synthetic-opencode-version",
        model={"provider": "synthetic-provider", "model": "synthetic-model"},
        cost=0.25,
        tokens_input=120,
        tokens_output=80,
        tokens_reasoning=12,
        tokens_cache_read=40,
        tokens_cache_write=4,
        time_created=normalize_timestamp(1_772_000_000_000, "ms"),
        time_updated=normalize_timestamp(1_772_000_500_000, "ms"),
    )
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload()), _message("msg-a1", "assistant", 2, {"content": []})],
        header=header,
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.metadata["opencode.title"] == "Synthetic mapping session"
    assert spec.metadata["opencode.slug"] == "synthetic-slug"
    assert spec.metadata["opencode.directory"] == "/repo/synthetic"
    assert spec.metadata["opencode.path"] == "/repo/synthetic/session.jsonl"
    assert spec.metadata["opencode.project_id"] == "proj-synthetic"
    assert spec.metadata["opencode.workspace_id"] == "ws-synthetic"
    assert spec.metadata["opencode.agent"] == "synthetic-agent"
    assert spec.metadata["opencode.version"] == "synthetic-opencode-version"
    assert spec.metadata["opencode.model"] == "synthetic-provider/synthetic-model"
    assert spec.metadata["opencode.cost"] == "0.25"
    assert spec.metadata["opencode.tokens_input"] == "120"
    assert spec.metadata["opencode.tokens_output"] == "80"
    assert spec.metadata["opencode.tokens_reasoning"] == "12"
    assert spec.metadata["opencode.tokens_cache_read"] == "40"
    assert spec.metadata["opencode.tokens_cache_write"] == "4"
    assert spec.metadata["opencode.time_created_ns"] == str(1_772_000_000_000_000_000)
    assert spec.metadata["opencode.time_updated_ns"] == str(1_772_000_500_000_000_000)
    # Absent native values are omitted, never invented.
    assert "opencode.parent_id" not in spec.metadata
    assert "opencode.fork_session_id" not in spec.metadata
    assert "opencode.time_archived_ns" not in spec.metadata


def test_oversized_metadata_value_becomes_an_artifact_reference() -> None:
    header = fake_header(_SESSION_ID, title="t" * 400)
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload())],
        header=header,
        summary=_summary(terminal_outcome=NO_ASSISTANT_RESULT_OUTCOME, root_result_source_id=None),
        root_input=_user_payload(),
        root_output=None,
    )

    spec = map_turn_to_trace(turn, _context())

    reference = json.loads(spec.metadata["opencode.title"])
    assert reference["$artifact_reference"]["size_bytes"] > 400


def test_root_attributes_carry_typed_aggregates() -> None:
    header = fake_header(
        _SESSION_ID,
        cost=0.25,
        tokens_input=120,
        tokens_output=80,
        tokens_reasoning=12,
        tokens_cache_read=40,
        tokens_cache_write=4,
    )
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload()), _message("msg-a1", "assistant", 2, {"content": []})],
        header=header,
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.root_attributes["opencode.session_id"] == _SESSION_ID
    assert spec.root_attributes["opencode.turn_index"] == 1
    assert spec.root_attributes["opencode.turn_key"] == _TURN_KEY
    assert spec.root_attributes["opencode.outcome"] == ASSISTANT_RESULT_OUTCOME
    assert spec.root_attributes["opencode.cost"] == 0.25
    assert spec.root_attributes["opencode.tokens.input"] == 120
    assert spec.root_attributes["opencode.tokens.output"] == 80
    assert spec.root_attributes["opencode.tokens.cache_read"] == 40
    assert spec.root_attributes["opencode.record_count"] == 3
    assert spec.root_attributes["opencode.logical_turn_count"] == 3
    assert spec.root_attributes["opencode.source_message_count"] == 7
    assert "opencode.timing_estimated" not in spec.root_attributes


# ---------------------------------------------------------------------------
# Fingerprints, operation key, determinism
# ---------------------------------------------------------------------------


def test_operation_key_is_the_versioned_sha256_of_the_exact_tuple() -> None:
    render_fp = "a" * 64
    expected = hashlib.sha256(
        "\0".join(("opencode-mlflow-operation-key-v1", _EXPERIMENT_ID, _SESSION_ID, _TURN_KEY, render_fp)).encode()
    ).hexdigest()

    assert operation_key(_EXPERIMENT_ID, _SESSION_ID, _TURN_KEY, render_fp) == expected
    assert operation_key(_EXPERIMENT_ID, _SESSION_ID, _OTHER_TURN_KEY, render_fp) != expected
    assert operation_key("exp-other", _SESSION_ID, _TURN_KEY, render_fp) != expected


def test_client_request_id_and_operation_key_metadata_are_the_operation_key() -> None:
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload()), _message("msg-a1", "assistant", 2, {"content": []})],
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.client_request_id == spec.expected_operation_key == spec.metadata["opencode.operation_key"]
    assert len(spec.client_request_id) == 64


def test_fingerprints_match_the_task_4a_seams() -> None:
    from opencode_mlflow.content.fingerprint import (
        artifact_snapshot_fingerprint,
        build_effective_turn,
        render_fingerprint,
        source_fingerprint,
    )

    records = [
        _message("msg-u1", "user", 1, _user_payload()),
        _message("msg-a1", "assistant", 2, {"content": []}),
    ]
    turn = _replay_turn(records, root_input=_user_payload(), root_output=[])
    context = _context()

    spec = map_turn_to_trace(turn, context)
    effective = build_effective_turn(turn, _IdentityRedactor(), context.redaction_mode)

    assert spec.metadata["opencode.source_fingerprint"] == source_fingerprint(effective, context.redaction_mode)
    render_fp = render_fingerprint(
        spec.metadata["opencode.source_fingerprint"],
        context.artifact_schema_version,
        context.mapping_version,
        context.importer_version,
        context.redaction_mode,
    )
    assert spec.metadata["opencode.render_fingerprint"] == render_fp
    assert spec.artifact_snapshot_fingerprint == artifact_snapshot_fingerprint(turn, context.redaction_mode)


def test_artifact_relative_path_is_deterministic_and_namespaced() -> None:
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload()), _message("msg-a1", "assistant", 2, {"content": []})],
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    expected = (
        f"opencode-mlflow/v1/sessions/{_SESSION_ID}/turns/{_TURN_KEY}/"
        f"{spec.metadata['opencode.render_fingerprint']}-{spec.artifact_snapshot_fingerprint}.json"
    )
    assert spec.artifact_relative_path == expected
    assert spec.artifact_relative_path.startswith("opencode-mlflow/v1/")


def test_mapping_is_deterministic_for_identical_turns() -> None:
    def build() -> TraceSpec:
        turn = _replay_turn(
            [
                _message("msg-u1", "user", 1, _user_payload()),
                _message("msg-a1", "assistant", 2, {"content": []}),
                _part("msg-a1#content-0", "msg-a1", {"type": "text", "text": "synthetic"}, ordinal=0),
            ],
            root_input=_user_payload(),
            root_output=[{"type": "text", "text": "synthetic"}],
        )
        return map_turn_to_trace(turn, _context())

    first, second = build(), build()

    assert first.client_request_id == second.client_request_id
    assert dict(first.metadata) == dict(second.metadata)
    assert dict(first.root_attributes) == dict(second.root_attributes)
    assert first.artifact_relative_path == second.artifact_relative_path
    assert [child.source_key for child in first.child_specs()] == [child.source_key for child in second.child_specs()]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def test_timing_uses_the_first_user_timestamp_and_last_record_timestamp() -> None:
    turn = _replay_turn(
        [
            _message("msg-u1", "user", 1, _user_payload(), ts=1_772_000_010_000),
            _message("msg-a1", "assistant", 2, {"content": []}, ts=1_772_000_030_000),
        ],
        summary=_summary(
            start_time=normalize_timestamp(1_772_000_010_000, "ms"),
            end_time=normalize_timestamp(1_772_000_030_000, "ms"),
        ),
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.start_time_ns == 1_772_000_010_000_000_000
    assert spec.end_time_ns == 1_772_000_030_000_000_000
    assert "opencode.timing_estimated" not in spec.metadata


def test_timing_falls_back_to_first_record_then_session_creation() -> None:
    header = fake_header(_SESSION_ID, time_created=normalize_timestamp(1_772_000_000_000, "ms"))
    records = [
        _message("msg-u1", "user", 1, _user_payload(), ts=None),
        _message("msg-a1", "assistant", 2, {"content": []}, ts=1_772_000_020_000),
    ]
    turn = _replay_turn(
        records,
        header=header,
        summary=_summary(
            start_time=None,
            end_time=normalize_timestamp(1_772_000_020_000, "ms"),
        ),
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    # First user timestamp absent → the first timestamped record anywhere
    # in the turn, marked estimated.
    assert spec.start_time_ns == 1_772_000_020_000_000_000
    assert spec.metadata["opencode.timing_estimated"] == "true"
    assert spec.root_attributes["opencode.timing_estimated"] is True

    event = fake_record(
        "pending-timing",
        raw_json=None,
        decoded={"outcome": "pending"},
        session_id=_SESSION_ID,
        source_table="session_pending",
        source_type="event",
        seq=None,
        classification=None,
        timestamp=normalize_timestamp(1_772_000_050_000, "ms"),
        raw_columns={"status": "pending"},
    )
    event_turn = FakeReplayTurn(
        session_header=fake_header(_SESSION_ID, time_created=None),
        summary=_summary(
            start_time=None,
            end_time=None,
            terminal_outcome=NO_ASSISTANT_RESULT_OUTCOME,
            root_result_source_id=None,
        ),
        turn_records=(_message("msg-u1", "user", 1, _user_payload(), ts=None),),
        session_records=(event,),
        root_input=_user_payload(),
        root_output=None,
    )
    event_spec = map_turn_to_trace(event_turn, _context())
    expected = 1_772_000_050_000_000_000
    assert event_spec.start_time_ns == expected
    assert event_spec.end_time_ns == expected
    child = next(child for child in event_spec.child_specs() if child.source_key == "session_pending:pending-timing")
    assert child.start_time_ns == expected
    assert child.end_time_ns == expected


def test_timing_uses_minimum_and_maximum_across_out_of_order_lifecycle_events() -> None:
    def lifecycle(source_id: str, timestamp_ms: int) -> SourceRecord:
        return replace(
            fake_record(
                source_id,
                raw_json=None,
                decoded={"status": "pending"},
                session_id=_SESSION_ID,
                source_table="session_pending",
                source_type="event",
                seq=None,
                classification=None,
                timestamp=normalize_timestamp(timestamp_ms, "ms"),
            ),
            sequence_domain=None,
        )

    turn = FakeReplayTurn(
        session_header=fake_header(_SESSION_ID, time_created=None),
        summary=_summary(
            start_time=None,
            end_time=None,
            terminal_outcome=ASSISTANT_RESULT_OUTCOME,
            root_result_source_id="msg-a1",
        ),
        turn_records=(
            _message("msg-u1", "user", 1, _user_payload(), ts=None),
            _message("msg-a1", "assistant", 2, {"content": []}, ts=1_772_000_030_000),
        ),
        # The newer lifecycle row is emitted first; the older row must not
        # overwrite the source-derived maximum end instant.
        session_records=(
            lifecycle("pending-newer", 1_772_000_040_000),
            lifecycle("pending-older", 1_772_000_020_000),
        ),
        root_input=_user_payload(),
        root_output=[],
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.start_time_ns == 1_772_000_020_000_000_000
    assert spec.end_time_ns == 1_772_000_040_000_000_000


def test_timing_falls_back_to_session_creation_when_no_record_has_timestamps() -> None:
    header = fake_header(_SESSION_ID, time_created=normalize_timestamp(1_772_000_000_000, "ms"))
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload(), ts=None)],
        header=header,
        summary=_summary(
            start_time=None,
            end_time=None,
            terminal_outcome=NO_ASSISTANT_RESULT_OUTCOME,
            root_result_source_id=None,
        ),
        root_input=_user_payload(),
        root_output=None,
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.start_time_ns == 1_772_000_000_000_000_000
    # End falls back to the source-derived start.
    assert spec.end_time_ns == 1_772_000_000_000_000_000
    assert spec.metadata["opencode.timing_estimated"] == "true"


def test_timing_is_omitted_when_all_source_timestamps_are_absent() -> None:
    header = fake_header(_SESSION_ID, time_created=None)
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload(), ts=None)],
        header=header,
        summary=_summary(
            start_time=None,
            end_time=None,
            terminal_outcome=NO_ASSISTANT_RESULT_OUTCOME,
            root_result_source_id=None,
        ),
        root_input=_user_payload(),
        root_output=None,
    )

    spec = map_turn_to_trace(turn, _context())

    assert spec.start_time_ns is None
    assert spec.end_time_ns is None
    assert "opencode.timing_estimated" not in spec.metadata


# ---------------------------------------------------------------------------
# Identity map binding
# ---------------------------------------------------------------------------


def test_source_identity_prefers_source_id_then_fallback_then_digest() -> None:
    with_id = SourceIdentity.of(_message("msg-1", "user", 1, _user_payload()))
    assert with_id.key == "session_message:msg-1"
    assert with_id.source_table == "session_message"

    fallback = SourceIdentity.of(
        replace(_message("msg-x", "user", 1, _user_payload()), source_id=None, source_identity="fallback-identity")
    )
    assert fallback.key == "session_message:fallback-identity"

    bare = SourceIdentity.of(
        replace(_message("msg-y", "user", 1, _user_payload()), source_id=None, source_identity=None)
    )
    assert bare.key.startswith("session_message:record-")
    assert len(bare.key) == len("session_message:record-") + 64


def test_bind_mlflow_identity_map_binds_generated_ids_purely() -> None:
    turn = _replay_turn(
        [_message("msg-u1", "user", 1, _user_payload()), _message("msg-a1", "assistant", 2, {"content": []})],
        root_input=_user_payload(),
        root_output=[],
    )
    spec = map_turn_to_trace(turn, _context())
    envelope = spec.identity_envelope()
    assert isinstance(envelope, ArtifactIdentityEnvelope)
    assert envelope == ArtifactIdentityEnvelope(
        session_id=_SESSION_ID,
        turn_key=_TURN_KEY,
        render_fingerprint=spec.metadata["opencode.render_fingerprint"],
        snapshot_fingerprint=spec.artifact_snapshot_fingerprint,
        artifact_relative_path=spec.artifact_relative_path,
    )

    identity_map = bind_mlflow_identity_map(envelope, "tr-generated", {"msg-a1": "span-generated"})

    assert isinstance(identity_map, MlflowIdentityMap)
    assert identity_map.trace_id == "tr-generated"
    assert identity_map.span_ids_by_source == {"msg-a1": "span-generated"}
    assert identity_map.session_id == _SESSION_ID
    assert identity_map.turn_key == _TURN_KEY
    assert identity_map.render_fingerprint == spec.metadata["opencode.render_fingerprint"]
    assert identity_map.snapshot_fingerprint == spec.artifact_snapshot_fingerprint
    assert identity_map.artifact_relative_path == spec.artifact_relative_path
    assert identity_map.artifact_section() == {
        "trace_id": "tr-generated",
        "span:msg-a1": "span-generated",
    }
    # The envelope is untouched (pure binding).
    assert envelope == spec.identity_envelope()


def test_artifact_identity_map_keeps_same_source_id_distinct_across_tables() -> None:
    turn = _replay_turn(
        [
            _message("shared-id", "user", 1, _user_payload()),
            _message("shared-id", "assistant", 2, {"content": []}),
        ],
        root_input=_user_payload(),
        root_output=[],
    )
    spec = map_turn_to_trace(turn, _context())
    identity_map = bind_mlflow_identity_map(
        spec.identity_envelope(),
        "tr-generated",
        {
            "session_message:shared-id": "span-message",
            "part:shared-id": "span-part",
        },
    )

    assert identity_map.artifact_section() == {
        "trace_id": "tr-generated",
        "span:part:shared-id": "span-part",
        "span:session_message:shared-id": "span-message",
    }


def test_binding_generated_ids_never_changes_the_fingerprints() -> None:
    from opencode_mlflow.content.fingerprint import (
        artifact_snapshot_fingerprint,
        build_effective_turn,
        source_fingerprint,
    )

    records = [
        _message("msg-u1", "user", 1, _user_payload()),
        _message("msg-a1", "assistant", 2, {"content": []}),
    ]
    turn = _replay_turn(records, root_input=_user_payload(), root_output=[])
    context = _context()
    spec = map_turn_to_trace(turn, context)
    effective = build_effective_turn(turn, _IdentityRedactor(), context.redaction_mode)
    before_snapshot = artifact_snapshot_fingerprint(turn, context.redaction_mode)
    before_source = source_fingerprint(effective, context.redaction_mode)

    bind_mlflow_identity_map(
        spec.identity_envelope(),
        "tr-generated",
        {"msg-a1": "span-generated", "msg-u1": "span-root-child"},
    )

    assert artifact_snapshot_fingerprint(turn, context.redaction_mode) == before_snapshot
    assert source_fingerprint(effective, context.redaction_mode) == before_source
    assert spec.artifact_snapshot_fingerprint == before_snapshot


# ---------------------------------------------------------------------------
# Six-turn fixture integration
# ---------------------------------------------------------------------------


def test_six_turn_root_values_follow_prompts_and_results(six_turn_session) -> None:
    specs = [map_turn_to_trace(turn, six_turn_session.context) for turn in six_turn_session.turns]

    assert specs[0].root_inputs == {"content": [{"type": "text", "text": "synthetic prompt one"}]}
    assert specs[1].root_outputs == [
        {"type": "text", "text": "synthetic final part one"},
        {"type": "text", "text": "synthetic final part two"},
    ]
    assert specs[2].root_outputs is None
    assert specs[3].root_outputs == [{"type": "text", "text": "synthetic partial answer four"}]


def test_six_turn_statuses_follow_source_outcomes(six_turn_session) -> None:
    specs = [map_turn_to_trace(turn, six_turn_session.context) for turn in six_turn_session.turns]

    assert [spec.root_status for spec in specs] == ["OK", "OK", "UNSET", "ERROR", "OK", "OK"]
    assert [spec.metadata["opencode.outcome"] for spec in specs] == [
        ASSISTANT_RESULT_OUTCOME,
        ASSISTANT_RESULT_OUTCOME,
        "lifecycle_observed",
        OUTCOME_EXPLICIT_FAILURE,
        ASSISTANT_RESULT_OUTCOME,
        ASSISTANT_RESULT_OUTCOME,
    ]


def test_six_turn_children_cover_the_typed_record_shapes(six_turn_session) -> None:
    specs = [map_turn_to_trace(turn, six_turn_session.context) for turn in six_turn_session.turns]
    children = {child.source_key: child for child in specs[0].child_specs()}

    assert set(children) == {
        "session_message:msg-synthetic-u1#content-0",
        "session_message:msg-synthetic-a1",
        "session_message:msg-synthetic-a1#content-0",
        "session_message:msg-synthetic-a1#content-1",
        "session_message:msg-synthetic-a1#content-2",
        "session_pending:pending-synthetic-1",
        "session_pending:pending-synthetic-2",
    }
    assert children["session_message:msg-synthetic-a1"].span_type == "LLM"
    assert children["session_message:msg-synthetic-a1"].name == "model synthetic-provider/synthetic-model"
    assert children["session_message:msg-synthetic-a1#content-1"].span_type == "TOOL"
    assert children["session_message:msg-synthetic-a1#content-2"].span_type != EVENT_SPAN_TYPE
    assert isinstance(children["session_message:msg-synthetic-a1#content-0"], EventSpec)
    assert (
        children["session_message:msg-synthetic-a1#content-1"].parent_source_key == "session_message:msg-synthetic-a1"
    )
    assert children["session_message:msg-synthetic-u1#content-0"].parent_source_key is None

    plain = {child.source_key: child for child in specs[4].child_specs()}
    assert plain["session_message:msg-synthetic-a5"].span_type == "AGENT"

    failed = {child.source_key: child for child in specs[3].child_specs()}
    assert failed["session_message:msg-synthetic-a4#content-1"].status == "ERROR"


def test_six_turn_metadata_carries_session_and_database_identity(six_turn_session) -> None:
    spec = map_turn_to_trace(six_turn_session.turns[0], six_turn_session.context)

    assert spec.metadata["opencode.title"] == "Synthetic six turn session"
    assert spec.metadata["opencode.slug"] == "synthetic-six-turn-slug"
    assert spec.metadata["opencode.directory"] == "/repo/synthetic-project"
    assert spec.metadata["opencode.agent"] == "synthetic-agent"
    assert spec.metadata["opencode.version"] == "synthetic-opencode-version"
    assert spec.metadata["opencode.model"] == "synthetic-provider/synthetic-model"
    assert spec.metadata["opencode.source_database_fingerprint"] == six_turn_session.snapshot.fingerprint
    assert spec.metadata["opencode.logical_turn_count"] == "6"
    assert spec.metadata["opencode.source_message_count"] == str(len(six_turn_session.message_records()))
    assert spec.metadata["opencode.source_part_count"] == str(len(six_turn_session.part_records()))


def test_six_turn_timing_comes_from_source_timestamps(six_turn_session) -> None:
    specs = [map_turn_to_trace(turn, six_turn_session.context) for turn in six_turn_session.turns]

    assert specs[0].start_time_ns == 1_772_000_010_000_000_000
    # The final timed session event is source-derived and extends the root;
    # no local wall-clock timestamp is invented.
    assert specs[0].end_time_ns == 1_772_000_065_000_000_000
    assert specs[2].start_time_ns == 1_772_000_050_000_000_000
    assert all("opencode.timing_estimated" in specs[index].metadata for index in (0, 1, 2))
    assert all("opencode.timing_estimated" not in specs[index].metadata for index in (3, 4, 5))
