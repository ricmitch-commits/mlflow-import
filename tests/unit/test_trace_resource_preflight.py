"""All selected turns must admit before the first importer mutation."""

import json
import sqlite3
from dataclasses import replace

import pytest

from mlflow_import.importer import run_import
from mlflow_import.mlflow.attempts import AttemptJournal
from mlflow_import.mlflow.trace_envelope import (
    ABSOLUTE_MAX_TRACE_BYTES,
    ABSOLUTE_MAX_TRACE_SPANS,
    MAX_TRACE_SPANS,
    TraceEnvelopeLimits,
    admit_trace,
)
from tests.support.compact_trace_fixture import SESSION, mapped_source, synthetic_source
from tests.support.fake_gateway import child_spec
from tests.support.import_fakes import DEFAULT_SESSION_CREATED_MS, UNIT_SESSION_ID, build_import_fakes
from tests.support.trace_resource_gateway import LocalTraceClient, LocalTraceGateway

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dry_run", [False, True])
def test_later_trace_overflow_prevents_experiment_and_all_writes(import_dependencies, config, dry_run):
    original = import_dependencies.mapper
    projected = []

    def mapper(turn, context):
        request = original(turn, context)
        projected.append(turn.summary.turn_key)
        if len(projected) >= 2:
            return replace(request, child_specs=lambda: (child_spec(str(index)) for index in range(MAX_TRACE_SPANS)))
        return request

    dependencies = replace(import_dependencies, mapper=mapper)
    result = run_import(replace(config, dry_run=dry_run), dependencies)
    assert result.error is not None
    assert result.error.code == "trace_resource_envelope_exceeded"
    assert UNIT_SESSION_ID in result.error.safe_message
    assert projected[1] in result.error.safe_message
    assert UNIT_SESSION_ID in result.report["errors"][0]["message"]
    assert result.error.diagnostics["measured_span_count"] == MAX_TRACE_SPANS + 1
    assert len(projected) >= 2
    assert import_dependencies.gateway.mutations == []


def test_configured_trace_limits_flow_into_preflight(import_dependencies, config):
    result = run_import(replace(config, dry_run=True, trace_max_bytes=1000), import_dependencies)
    assert result.error is not None
    assert result.error.code == "trace_resource_envelope_exceeded"
    assert UNIT_SESSION_ID in result.error.safe_message


def test_preflight_debug_reports_each_admitted_turn(import_dependencies, config, capsys):
    result = run_import(replace(config, log_level="DEBUG", dry_run=True), import_dependencies)
    assert result.error is None, result.error
    stderr = capsys.readouterr().err
    assert stderr.count("mlflow-import: preflight turn session=") == 2
    assert "bytes=" in stderr
    assert "spans=" in stderr
    assert "control=" in stderr


def test_preflight_failure_debug_reports_measured_counts(import_dependencies, config, capsys):
    original = import_dependencies.mapper

    def mapper(turn, context):
        request = original(turn, context)
        return replace(request, child_specs=lambda: (child_spec(str(index)) for index in range(MAX_TRACE_SPANS)))

    dependencies = replace(import_dependencies, mapper=mapper)
    result = run_import(replace(config, log_level="DEBUG", dry_run=True), dependencies)
    assert result.error is not None
    stderr = capsys.readouterr().err
    assert "mlflow-import: preflight turn failed session=" in stderr
    assert "measured_span_count=" in stderr


@pytest.mark.parametrize("pending_state", ["prepared", "ready_to_retire"])
def test_oversize_map_blocks_existing_pending_recovery_before_any_target_mutation(
    tmp_path, restrictive_tmp_path, config, monkeypatch, pending_state
):
    from mlflow_import.errors import ItemImportError

    source_root = tmp_path / "pending-source"
    source_root.mkdir()
    source = synthetic_source(source_root, 2)
    config = replace(
        config,
        db_path=source,
        trace_max_bytes=ABSOLUTE_MAX_TRACE_BYTES,
        trace_max_spans=ABSOLUTE_MAX_TRACE_SPANS,
    )
    client = LocalTraceClient()
    artifact_root = tmp_path / "target-artifacts"
    artifact_root.mkdir()
    client.add_experiment(config.experiment_name, artifact_location=artifact_root.as_uri())
    gateway = LocalTraceGateway(client)
    dependencies = replace(
        build_import_fakes(restrictive_tmp_path=restrictive_tmp_path), gateway_factory=lambda _: gateway
    )
    publish = AttemptJournal.publish
    journals = []

    def interrupt_after_publication(journal, state, values):
        result = publish(journal, state, values)
        if state == pending_state:
            journals.append((journal.experiment_id, journal.prefix.removesuffix("/attempts")))
            raise ItemImportError("synthetic_interruption", "synthetic interruption after journal publication")
        return result

    with monkeypatch.context() as setup:
        setup.setattr(AttemptJournal, "publish", interrupt_after_publication)
        interrupted = run_import(config, dependencies)
    assert interrupted.error is None, interrupted.error
    assert interrupted.failed_items > 0
    assert journals, [
        (session.get("error_code"), [turn.get("error_code") for turn in session["turns"]])
        for session in interrupted.report["sessions"]
    ]
    experiment_id, prefix = journals[-1]
    states = list(AttemptJournal(gateway, experiment_id, prefix).attempts())
    assert len(states) == 1 and pending_state in states[0]
    assert "complete" not in states[0] and "aborted" not in states[0]

    # Source IDs repeat in descriptors, identity maps and provenance. This
    # source stays inside the admitted span profile while its v2 map exceeds
    # the independent 8 MiB pricing/recovery envelope.
    with sqlite3.connect(source) as connection:
        connection.execute("DELETE FROM session_message WHERE type = 'assistant'")
        connection.executemany(
            "INSERT INTO session_message (id, session_id, type, seq, time_created, time_updated, data)"
            " VALUES (?, ?, 'assistant', ?, ?, ?, ?)",
            [
                (
                    f"assistant-{index}-" + "x" * 2000,
                    SESSION,
                    2000 + index,
                    DEFAULT_SESSION_CREATED_MS + 2010000 + index * 1000,
                    DEFAULT_SESSION_CREATED_MS + 2010000 + index * 1000,
                    json.dumps(
                        {"role": "assistant", "model": "synthetic/model", "content": "answer", "finish": "stop"}
                    ),
                )
                for index in range(500)
            ],
        )
    with mapped_source(source, tmp_path) as mapped:
        assert len(mapped) == 1
        frozen_replay, request = mapped[0]
        assert frozen_replay.summary.record_count >= 500, frozen_replay.summary
        assert sum(1 for _ in frozen_replay.iter_records()) >= 500
        envelope = admit_trace(
            request,
            reserve_late_control=True,
            limits=TraceEnvelopeLimits(
                max_bytes=ABSOLUTE_MAX_TRACE_BYTES,
                max_spans=ABSOLUTE_MAX_TRACE_SPANS,
            ),
        )
        assert envelope.span_count <= ABSOLUTE_MAX_TRACE_SPANS
        assert envelope.serialized_span_upper_bound < ABSOLUTE_MAX_TRACE_BYTES
        from mlflow_import.content.turn_map import validate_turn_map_admission
        from mlflow_import.contracts import RedactionMode

        with pytest.raises(ItemImportError) as oversized:
            validate_turn_map_admission(frozen_replay, envelope.request, RedactionMode.VERBATIM)
        assert oversized.value.code == "turn_map_too_large"

    mutation_calls = []

    def forbid(name):
        def forbidden(*args, **kwargs):
            mutation_calls.append(name)
            raise AssertionError("oversize map reached a target mutation: " + name)

        return forbidden

    for name in (
        "create_trace",
        "retire_trace",
        "retire_conflict",
        "record_artifact_verification",
        "upload_named_json",
        "publish_experiment_file",
        "reconcile_tags",
        "reconcile_current_tags",
    ):
        monkeypatch.setattr(gateway, name, forbid(name))
    for name in ("create_experiment", "set_trace_tag", "delete_trace_tag", "delete_traces"):
        monkeypatch.setattr(client, name, forbid(name))
    monkeypatch.setattr(AttemptJournal, "publish", forbid("journal.publish"))
    before = list(client.mutations)
    result = run_import(config, dependencies)
    assert result.error is not None
    assert result.error.code == "turn_map_too_large", result.report
    assert mutation_calls == []
    assert client.mutations == before
    after = list(AttemptJournal(gateway, experiment_id, prefix).attempts())
    assert after == states
