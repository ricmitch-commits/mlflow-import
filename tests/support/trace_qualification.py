"""Deterministic synthetic trace workloads, independent of real source sessions."""

from dataclasses import replace

from mlflow_import.mlflow.gateway import TraceChildRequest, TraceCreateRequest

START = 1_794_580_115_000_000_000
END = START + 1_000_000_000


def assert_exact_value(actual: object, expected: object) -> None:
    """Check nested JSON types too: Python equality equates False with zero."""
    if type(actual) is not type(expected):
        raise AssertionError("round-trip JSON value type changed")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("round-trip JSON object keys changed")
        for key in expected:
            assert_exact_value(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError("round-trip JSON array length changed")
        for left, right in zip(actual, expected, strict=True):
            assert_exact_value(left, right)
    elif actual != expected:
        raise AssertionError("round-trip JSON scalar changed")


def qualification_request(experiment_id: str, children: int, value_bytes: int = 16) -> TraceCreateRequest:
    # Structured values deliberately include escaping, Unicode, and type traps.
    value = {"text": "x" * value_bytes, "typed": [None, False, 7, "7", '雪😀"\\\n']}

    def specs():
        for index in range(children):
            yield TraceChildRequest(
                source_key=f"child-{index}",
                name=f"synthetic-{index}",
                span_type="TOOL",
                parent_source_key=None,
                inputs=value,
                outputs={"index": index, "ok": True},
                attributes={"synthetic.index": index},
                tags={},
                start_time_ns=START + index + 1,
                end_time_ns=END - 1,
                status="OK",
            )

    return TraceCreateRequest(
        client_request_id=None,
        experiment_id=experiment_id,
        name="synthetic qualification",
        native_session_id="synthetic-qualification",
        root_inputs={"synthetic": True},
        root_outputs={"complete": True},
        root_attributes={"synthetic": True},
        root_tags={},
        metadata={},
        start_time_ns=START,
        end_time_ns=END,
        child_specs=specs,
        expected_operation_key="synthetic-qualification",
        artifact_snapshot_fingerprint="a" * 64,
        artifact_relative_path="synthetic-qualification.json",
        root_status="OK",
        historical=True,
        trace_name="synthetic qualification",
        request_preview="synthetic request",
        response_preview="synthetic response",
    )


def measured_shape_request(experiment_id: str, calls: int = 1000) -> TraceCreateRequest:
    """2105 children at 1000 calls, with compact, growing model history deltas.

    105 fixed activity spans plus a model and tool span per call. Each model
    carries only its new message and a predecessor key: linear serialized data.
    This hand-built public request qualifies transport, not mapping semantics.
    """
    base = qualification_request(experiment_id, 105)

    def specs():
        yield from base.child_specs()
        for index in range(calls):
            model = f"model-{index}"
            yield TraceChildRequest(
                source_key=model,
                name="synthetic model",
                span_type="LLM",
                parent_source_key=None,
                inputs={"messages": [{"role": "user", "content": f"synthetic step {index}"}]},
                outputs={"role": "assistant", "content": f"synthetic call {index}"},
                attributes={"synthetic.previous": f"model-{index - 1}" if index else None},
                tags={},
                start_time_ns=START + index + 200,
                end_time_ns=END - 1,
                status="OK",
            )
            yield TraceChildRequest(
                source_key=f"tool-{index}",
                name="synthetic tool",
                span_type="TOOL",
                parent_source_key=model,
                inputs={"query": f"synthetic query {index}"},
                outputs={"result": index},
                attributes={"synthetic.call": index},
                tags={},
                start_time_ns=START + index + 300,
                end_time_ns=END - 1,
                status="OK",
            )

    return replace(base, child_specs=specs)
