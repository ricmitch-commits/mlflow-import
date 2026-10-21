"""Focused Wave 2 release-blocker regressions.

These tests deliberately cross the production seams instead of stopping at
pure mapper helpers: source SQL adapters feed the normalized snapshot,
replay reconstructs the turn, mapping creates the request, and the fake
gateway exercises the public gateway verification behavior.
"""

from __future__ import annotations

import io
import json
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from opencode_mlflow.config import ImportConfig
from opencode_mlflow.content.artifacts import ArtifactProvenance, stream_turn_artifact
from opencode_mlflow.content.canonical import canonical_json
from opencode_mlflow.contracts import DuplicateKeyObject, RedactionMode
from opencode_mlflow.errors import ItemImportError
from opencode_mlflow.importer import run_import
from opencode_mlflow.mlflow.mapping import MappingContext, map_turn_to_trace
from opencode_mlflow.source.catalog import CURRENT_CATALOG_V1, LEGACY_CATALOG_V1, SCHEMA_CATALOG_VERSION
from opencode_mlflow.source.current import CurrentAdapter
from opencode_mlflow.source.discovery import (
    ARTIFACT_SCHEMA_VERSION,
    MAPPING_VERSION,
    DiscoveryPlanReader,
    discover_sessions,
)
from opencode_mlflow.source.legacy import LegacyAdapter
from opencode_mlflow.source.model import normalize_timestamp
from opencode_mlflow.source.normalize import normalize_epoch_ms, unknown_json_field_warning
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter
from opencode_mlflow.source.selection import SelectionFilters
from opencode_mlflow.source.snapshot import create_normalized_snapshot
from opencode_mlflow.source.turns import build_replay_turn, iter_turn_summaries
from tests.support.fake_gateway import FakeMlflowClient, FakeMlflowGateway, child_spec, trace_spec
from tests.support.fake_source import identity_redaction
from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_record, fake_summary
from tests.support.import_fakes import CURRENT_DDL, LEGACY_DDL, build_import_fakes

pytestmark = pytest.mark.unit


def _current_db(path: Path, messages: list[tuple[str, str, int, int | None, int | None, object]]) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(CURRENT_DDL)
        connection.execute(
            "INSERT INTO session_v2 (id, time_created, time_updated) VALUES (?, ?, ?)",
            ("ses-wave2", 1_772_000_000_000, 1_772_000_004_000),
        )
        for message_id, message_type, seq, created, updated, payload in messages:
            connection.execute(
                "INSERT INTO session_message"
                " (id, session_id, type, seq, time_created, time_updated, data)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    "ses-wave2",
                    message_type,
                    seq,
                    created,
                    updated,
                    json.dumps(payload),
                ),
            )
        connection.commit()
    finally:
        connection.close()
    return path


def _legacy_collision_db(path: Path) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(LEGACY_DDL)
        connection.execute(
            "INSERT INTO session (id, time_created, time_updated) VALUES (?, ?, ?)",
            ("ses-collision", 1_772_000_000_000, 1_772_000_000_000),
        )
        connection.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
            ("user-id", "ses-collision", 1_772_000_001_000, 1_772_000_001_500, '{"role":"user"}'),
        )
        connection.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)",
            ("shared-id", "ses-collision", 1_772_000_003_000, 1_772_000_004_000, '{"role":"assistant"}'),
        )
        connection.execute(
            "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?, ?)",
            ("shared-id", "shared-id", "ses-collision", 1_772_000_001_100, 1_772_000_001_200, '{"type":"text"}'),
        )
        connection.commit()
    finally:
        connection.close()
    return path


def _real_current_turn(path: Path, temp_root: Path, catalog=CURRENT_CATALOG_V1):
    connection = sqlite3.connect(path)
    try:
        exporter = CurrentAdapter(connection, catalog)
    finally:
        connection.close()
    return create_normalized_snapshot(path, temp_root, exporter, identity_redaction)


def _context(snapshot_fingerprint: str, catalog_version: str = SCHEMA_CATALOG_VERSION) -> MappingContext:
    return MappingContext(
        importer_version="0.1.0",
        mapping_version=MAPPING_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        source_schema="current",
        redaction_mode=RedactionMode.VERBATIM,
        experiment_id="exp-wave2",
        catalog_version=catalog_version,
        source_database_fingerprint=snapshot_fingerprint,
        logical_turn_count=1,
        source_message_count=3,
        source_part_count=0,
    )


def test_completion_survives_adapter_snapshot_replay_and_mapping(tmp_path: Path) -> None:
    source = _current_db(
        tmp_path / "completion.db",
        [
            ("user-1", "user", 1, 1_772_000_001_000, 1_772_000_001_100, {"content": []}),
            (
                "assistant-1",
                "assistant",
                2,
                1_772_000_002_000,
                1_772_000_002_500,
                {"content": [], "time": {"completed": 1_772_000_003_000}},
            ),
            # This row is deliberately untimed and must not erase the known
            # completion of the preceding assistant result.
            ("system-1", "system", 3, None, None, {"content": []}),
        ],
    )
    with _real_current_turn(source, tmp_path) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        records = list(adapter.iter_records("ses-wave2"))
        assistant = next(record for record in records if record.source_id == "assistant-1")
        assert assistant.completion_timestamp == normalize_timestamp(1_772_000_003_000, "ms")
        header = adapter.session_header("ses-wave2")
        assert header is not None
        (summary,) = iter_turn_summaries(records)
        assert summary.end_time == assistant.completion_timestamp
        turn = build_replay_turn(snapshot, header, summary)
        request = map_turn_to_trace(turn, _context(snapshot.fingerprint)).to_create_request()
        children = list(request.child_specs())

    child = next(child for child in children if child.source_key.endswith(":assistant-1"))
    assert child.end_time_ns == 1_772_000_003_000_000_000
    assert child.attributes["opencode.completion_timestamp_ns"] == 1_772_000_003_000_000_000
    assert request.end_time_ns == 1_772_000_003_000_000_000


def test_catalog_version_survives_snapshot_replay_discovery_mapping_and_artifact(tmp_path: Path) -> None:
    catalog_version = "opencode-schema-catalog-v2-test"
    source = _current_db(
        tmp_path / "catalog-version.db",
        [("user-1", "user", 1, 1_772_000_001_000, 1_772_000_001_100, {"content": []})],
    )
    catalog = replace(CURRENT_CATALOG_V1, version=catalog_version)

    with _real_current_turn(source, tmp_path, catalog) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = adapter.session_header("ses-wave2")
        assert header is not None
        (summary,) = iter_turn_summaries(adapter.iter_records("ses-wave2"))
        assert snapshot.schema_description.catalog_version == catalog_version
        assert header.catalog_version == catalog_version
        turn = build_replay_turn(snapshot, header, summary)
        request = map_turn_to_trace(turn, _context(snapshot.fingerprint, catalog_version)).to_create_request()
        artifact = io.BytesIO()
        stream_turn_artifact(
            turn,
            artifact,
            {"trace_id": "tr-catalog"},
            RedactionMode.VERBATIM,
            provenance=ArtifactProvenance.from_request(request),
        )
        document = json.loads(artifact.getvalue())

    assert request.metadata["opencode.catalog_version"] == catalog_version
    assert document["artifact"]["catalog_version"] == catalog_version


def test_catalog_version_changes_artifact_snapshot_identity_but_not_database_provenance(tmp_path: Path) -> None:
    source = _current_db(
        tmp_path / "catalog-fingerprint.db",
        [("user-1", "user", 1, 1_772_000_001_000, 1_772_000_001_100, {"content": []})],
    )
    fingerprints: list[str] = []
    snapshots: list[str] = []
    database_fingerprints: list[str] = []
    for version in ("catalog-a", "catalog-b"):
        with _real_current_turn(source, tmp_path, replace(CURRENT_CATALOG_V1, version=version)) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            header = adapter.session_header("ses-wave2")
            assert header is not None
            (summary,) = iter_turn_summaries(adapter.iter_records("ses-wave2"))
            turn = build_replay_turn(snapshot, header, summary)
            request = map_turn_to_trace(turn, _context(snapshot.fingerprint, version)).to_create_request()
            fingerprints.append(request.metadata["opencode.source_fingerprint"])
            snapshots.append(request.artifact_snapshot_fingerprint)
            database_fingerprints.append(snapshot.fingerprint)

    assert fingerprints[0] != fingerprints[1]
    assert snapshots[0] != snapshots[1]
    assert database_fingerprints[0] == database_fingerprints[1]


def test_unknown_json_warning_survives_replay_mapping_and_artifact(tmp_path: Path) -> None:
    source = _current_db(
        tmp_path / "unknown-json-artifact.db",
        [("user-1", "user", 1, 1_772_000_001_000, 1_772_000_001_100, {"content": [], "future_key": "value"})],
    )

    with _real_current_turn(source, tmp_path) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = adapter.session_header("ses-wave2")
        assert header is not None
        (summary,) = iter_turn_summaries(adapter.iter_records("ses-wave2"))
        turn = build_replay_turn(snapshot, header, summary)
        request = map_turn_to_trace(turn, _context(snapshot.fingerprint)).to_create_request()
        destination = io.BytesIO()
        stream_turn_artifact(
            turn,
            destination,
            {"trace_id": "tr-warning"},
            RedactionMode.VERBATIM,
            provenance=ArtifactProvenance.from_request(request),
        )
        document = json.loads(destination.getvalue())

    assert "unknown_source_field" in document["warnings"]
    assert any(record["decoded"].get("future_key") == "value" for record in document["turn_records"])


def test_completion_priority_preserves_malformed_warning_and_falls_back(tmp_path: Path) -> None:
    source = _current_db(
        tmp_path / "completion-priority.db",
        [
            ("user-1", "user", 1, 1_772_000_001_000, 1_772_000_001_100, {"content": []}),
            (
                "assistant-1",
                "assistant",
                2,
                1_772_000_002_000,
                1_772_000_002_500,
                {"content": [], "time": {"completed": "not-a-timestamp", "end": 1_772_000_003_500}},
            ),
        ],
    )
    with _real_current_turn(source, tmp_path) as snapshot:
        record = next(
            record
            for record in NormalizedSnapshotAdapter(snapshot).iter_records("ses-wave2")
            if record.source_id == "assistant-1"
        )

    assert record.completion_timestamp == normalize_timestamp(1_772_000_003_500, "ms")
    assert "malformed_source_value" in record.warnings


def test_non_finite_sqlite_timestamp_is_a_warning_without_a_timestamp(tmp_path: Path) -> None:
    source = _current_db(
        tmp_path / "non-finite-timestamp.db",
        [("user-1", "user", 1, float("inf"), float("-inf"), {"content": []})],
    )

    with _real_current_turn(source, tmp_path) as snapshot:
        record = next(iter(NormalizedSnapshotAdapter(snapshot).iter_records("ses-wave2")))

    assert record.timestamp is None
    assert record.completion_timestamp is None
    assert "malformed_source_value" in record.warnings
    assert normalize_epoch_ms(float("nan")) == (None, ("malformed_source_value",))


def test_explicit_error_and_lifecycle_outcomes_are_not_lost_and_success_supersedes(tmp_path: Path) -> None:
    source = _current_db(
        tmp_path / "outcomes.db",
        [
            ("user-1", "user", 1, 1_772_000_001_000, 1_772_000_001_100, {"content": []}),
            (
                "assistant-error",
                "assistant",
                2,
                1_772_000_002_000,
                1_772_000_002_100,
                {"error": {"message": "synthetic failure"}, "content": []},
            ),
            ("assistant-success", "assistant", 3, 1_772_000_003_000, 1_772_000_003_100, {"content": []}),
        ],
    )
    with _real_current_turn(source, tmp_path) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = adapter.session_header("ses-wave2")
        assert header is not None
        records = list(adapter.iter_records("ses-wave2"))
        (summary,) = iter_turn_summaries(records)
        assert summary.root_result_source_id == "assistant-success"
        assert summary.terminal_outcome == "assistant_result"
        turn = build_replay_turn(snapshot, header, summary)
        request = map_turn_to_trace(turn, _context(snapshot.fingerprint)).to_create_request()
        children = {child.source_key: child for child in request.child_specs()}

    assert request.root_status == "OK"
    assert children["session_message:assistant-error"].status == "ERROR"


def test_all_failure_signals_survive_adapter_snapshot_replay_mapping_and_fake_gateway(tmp_path: Path) -> None:
    source = _current_db(
        tmp_path / "all-outcomes.db",
        [
            ("user-1", "user", 1, 1_772_000_001_000, 1_772_000_001_100, {"content": []}),
            (
                "assistant-error",
                "assistant",
                2,
                1_772_000_002_000,
                1_772_000_002_100,
                {
                    "status": "cancelled",
                    "finish": "interrupted",
                    "content": [
                        {
                            "type": "tool",
                            "tool": "bash",
                            "state": {"status": "failed", "error": "synthetic tool failure"},
                        }
                    ],
                },
            ),
            ("assistant-success", "assistant", 3, 1_772_000_003_000, 1_772_000_003_100, {"content": []}),
        ],
    )
    connection = sqlite3.connect(source)
    try:
        connection.execute(
            "CREATE TABLE session_pending (id TEXT PRIMARY KEY, session_id TEXT, type TEXT, data TEXT,"
            " delivery TEXT, admitted_seq INTEGER, time_created INTEGER)"
        )
        connection.execute(
            "INSERT INTO session_pending VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("pending-failed", "ses-wave2", "event", '{"outcome":"failed"}', "queued", 4, 1_772_000_004_000),
        )
        connection.commit()
    finally:
        connection.close()

    with _real_current_turn(source, tmp_path) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = adapter.session_header("ses-wave2")
        assert header is not None
        (summary,) = iter_turn_summaries(adapter.iter_records("ses-wave2"))
        turn = build_replay_turn(snapshot, header, summary)
        request = map_turn_to_trace(turn, _context(snapshot.fingerprint)).to_create_request()
        children = {child.source_key: child for child in request.child_specs()}
        gateway = FakeMlflowGateway()
        from opencode_mlflow.mlflow.capabilities import CapabilityRequest

        gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
        created = gateway.create_trace(request)
        gateway.verify_trace(created, request)

    assert request.root_status == "ERROR"
    assert request.metadata["opencode.outcome"] == "lifecycle_failed"
    assert children["session_message:assistant-error"].status == "ERROR"
    tool = next(child for child in children.values() if child.span_type == "TOOL")
    assert tool.status == "ERROR"
    assert children["session_pending:pending-failed"].status == "ERROR"
    assert created.root_status == "ERROR"


def test_lifecycle_records_map_as_children_once_in_the_artifact(tmp_path: Path) -> None:
    source = _current_db(
        tmp_path / "lifecycle.db",
        [("user-1", "user", 1, 1_772_000_001_000, 1_772_000_001_100, {"content": []})],
    )
    connection = sqlite3.connect(source)
    try:
        connection.execute(
            "CREATE TABLE session_pending (id TEXT PRIMARY KEY, session_id TEXT, type TEXT, data TEXT,"
            " delivery TEXT, admitted_seq INTEGER, time_created INTEGER)"
        )
        connection.execute(
            "INSERT INTO session_pending VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("pending-1", "ses-wave2", "event", '{"outcome":"failed"}', "queued", 2, 1_772_000_002_000),
        )
        connection.commit()
    finally:
        connection.close()

    with _real_current_turn(source, tmp_path) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = adapter.session_header("ses-wave2")
        assert header is not None
        (summary,) = iter_turn_summaries(adapter.iter_records("ses-wave2"))
        turn = build_replay_turn(snapshot, header, summary)
        request = map_turn_to_trace(turn, _context(snapshot.fingerprint)).to_create_request()
        children = list(request.child_specs())
        gateway = FakeMlflowGateway()
        from opencode_mlflow.mlflow.capabilities import CapabilityRequest

        gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
        created = gateway.create_trace(request)
        gateway.verify_trace(created, request)
        from io import BytesIO

        destination = BytesIO()
        stream_turn_artifact(
            turn,
            destination,
            {child.source_key: f"span-{index}" for index, child in enumerate(children)},
            RedactionMode.VERBATIM,
            provenance=ArtifactProvenance.from_request(request),
        )
        document = json.loads(destination.getvalue())

    pending_children = [child for child in children if child.source_key.endswith(":pending-1")]
    assert len(pending_children) == 1
    assert pending_children[0].parent_source_key is None
    locations = [
        record["source_id"]
        for section in ("session_records", "turn_records", "unassigned_records")
        for record in document[section]
        if record.get("source_id") == "pending-1"
    ]
    assert locations == ["pending-1"]
    assert document["artifact"]["catalog_version"] == request.metadata["opencode.catalog_version"]


def test_colliding_message_and_part_ids_keep_qualified_parent_lineage(tmp_path: Path) -> None:
    source = _legacy_collision_db(tmp_path / "collision.db")
    connection = sqlite3.connect(source)
    try:
        exporter = LegacyAdapter(connection, LEGACY_CATALOG_V1)
    finally:
        connection.close()
    with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = adapter.session_header("ses-collision")
        assert header is not None
        (summary,) = iter_turn_summaries(adapter.iter_records("ses-collision"))
        turn = build_replay_turn(snapshot, header, summary)
        request = map_turn_to_trace(turn, _context(snapshot.fingerprint)).to_create_request()
        children = list(request.child_specs())
        gateway = FakeMlflowGateway()
        from opencode_mlflow.mlflow.capabilities import CapabilityRequest

        gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
        created = gateway.create_trace(request)
        gateway.verify_trace(created, request)

    assert {child.source_key for child in children} >= {"message:shared-id", "part:shared-id"}
    part = next(child for child in children if child.source_key == "part:shared-id")
    assert part.parent_source_key == "message:shared-id"
    assert set(created.span_ids_by_source) >= {"message:shared-id", "part:shared-id"}


def test_legacy_session_scoped_todo_maps_as_a_root_event(tmp_path: Path) -> None:
    source = _legacy_collision_db(tmp_path / "legacy-scoped.db")
    connection = sqlite3.connect(source)
    try:
        connection.execute(
            "CREATE TABLE todo (session_id TEXT, content TEXT, status TEXT, priority INTEGER,"
            " position INTEGER, time_created INTEGER, time_updated INTEGER)"
        )
        connection.execute(
            "INSERT INTO todo VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ses-collision", "synthetic todo", "pending", 1, 1, 1_772_000_005_000, 1_772_000_005_500),
        )
        connection.commit()
    finally:
        connection.close()

    connection = sqlite3.connect(source)
    try:
        exporter = LegacyAdapter(connection, LEGACY_CATALOG_V1)
    finally:
        connection.close()
    with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = adapter.session_header("ses-collision")
        assert header is not None
        (summary,) = iter_turn_summaries(adapter.iter_records("ses-collision"))
        turn = build_replay_turn(snapshot, header, summary)
        request = map_turn_to_trace(turn, _context(snapshot.fingerprint)).to_create_request()
        children = list(request.child_specs())

    todo = next(child for child in children if child.source_key.startswith("todo:"))
    assert todo.parent_source_key is None


def test_uploaded_artifact_carries_exact_mapped_provenance(six_turn_session, tmp_path: Path) -> None:
    from opencode_mlflow.mlflow.reconciliation import upload_artifact

    turn = six_turn_session.turns[0]
    request = map_turn_to_trace(turn, six_turn_session.context).to_create_request()
    gateway = FakeMlflowGateway()
    from opencode_mlflow.mlflow.capabilities import CapabilityRequest

    gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
    created = gateway.create_trace(request)
    verification = upload_artifact(
        turn,
        created,
        request.artifact_relative_path,
        RedactionMode.VERBATIM,
        gateway=gateway,
        temporary_directory=tmp_path,
    )
    document = json.loads(gateway.stored_artifact(created.trace_id, verification.relative_path).decode("utf-8"))
    artifact = document["artifact"]

    assert artifact["source_database_fingerprint"] == request.metadata["opencode.source_database_fingerprint"]
    assert artifact["source_fingerprint"] == request.metadata["opencode.source_fingerprint"]
    assert artifact["render_fingerprint"] == request.metadata["opencode.render_fingerprint"]
    assert artifact["artifact_snapshot_fingerprint"] == request.artifact_snapshot_fingerprint
    assert artifact["artifact_path"] == request.artifact_relative_path
    assert artifact["source_schema"] == request.metadata["opencode.source_schema"]
    assert artifact["mapping_version"] == request.metadata["opencode.mapping_version"]
    assert artifact["redaction_mode"] == request.metadata["opencode.redaction_mode"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda span: setattr(span, "inputs", {"corrupt": "input"}),
        lambda span: setattr(span, "outputs", {"corrupt": "output"}),
        lambda span: setattr(span, "parent_id", "wrong-parent"),
        lambda span: setattr(span, "status", "ERROR"),
        lambda span: setattr(span, "start_time_ns", 1),
        lambda span: setattr(span, "end_time_ns", 2),
        lambda span: setattr(span, "name", "corrupt-name"),
        lambda span: setattr(span, "span_type", "CORRUPT"),
        lambda span: span.attributes.update({"opencode.source_key": "corrupt"}),
        lambda span: span.tags.update({"opencode.span_tag": "corrupt"}),
    ],
    ids=["inputs", "outputs", "parent", "status", "start", "end", "name", "type", "attributes", "tags"],
)
def test_gateway_verifies_every_retrieved_child_field(
    fake_mlflow_client: FakeMlflowClient, mutate: Callable[[object], None]
) -> None:
    gateway = fake_mlflow_client.gateway()
    from opencode_mlflow.mlflow.capabilities import CapabilityRequest

    assert gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True)).is_compatible
    child = child_spec(
        "session_message:child",
        inputs={"input": 1},
        outputs={"output": 2},
        attributes={"opencode.source_key": "session_message:child"},
        tags={"opencode.span_tag": "value"},
        start_time_ns=10,
        end_time_ns=20,
    )
    spec = trace_spec(children=[child])
    created = gateway.create_trace(spec)
    trace = fake_mlflow_client.get_trace(created.trace_id)
    retrieved = next(span for span in trace.data.spans if span.span_id == created.span_ids_by_source[child.source_key])
    mutate(retrieved)

    with pytest.raises(ItemImportError) as error:
        gateway.verify_trace(created, spec)
    assert error.value.code == "trace_verification_failed"


def test_mapping_gateway_boundary_uses_canonical_typed_payloads_losslessly() -> None:
    nested = DuplicateKeyObject(
        pairs=(("value", "first"), ("value", "second")),
        raw_text='{"value":"first","value":"second"}',
    )
    payload = {
        "duplicate": DuplicateKeyObject(
            pairs=(("duplicate", "one"), ("duplicate", "two"), ("nested", nested)),
            raw_text='{"duplicate":"one","duplicate":"two","nested":{}}',
        ),
        "bytes": b"\x00\xff",
        "not_a_number": float("nan"),
        # This value represents content after source redaction. Conversion must
        # not reintroduce the original value or otherwise alter the mask.
        "secret": "[REDACTED]",
    }
    turn = FakeReplayTurn(
        session_header=fake_header("ses-payloads"),
        summary=fake_summary(
            session_id="ses-payloads",
            turn_key="turn-" + "a" * 64,
            terminal_outcome="assistant_result",
            root_result_source_id="assistant-1",
        ),
        turn_records=(
            # The first record is the root input; the second maps to a child.
            replace(
                fake_record("user-1", decoded={"content": []}, session_id="ses-payloads"),
                classification="user",
            ),
            replace(
                fake_record(
                    "assistant-1",
                    decoded={"model": "provider/model", "input": payload, "output": payload},
                    session_id="ses-payloads",
                    source_type="assistant",
                    classification="assistant",
                ),
                sequence_value=2,
            ),
        ),
        root_input=payload,
        root_output=payload,
    )
    context = _context("payloads")
    request = map_turn_to_trace(turn, context).to_create_request()

    expected = json.loads(canonical_json(payload))
    assert request.root_inputs == expected
    assert request.root_outputs == expected
    child = next(request.child_specs())
    assert child.inputs == expected
    assert child.outputs == expected
    assert child.inputs["secret"] == "[REDACTED]"
    assert isinstance(child.inputs["duplicate"], dict)
    assert child.inputs["bytes"] == {"$bytes": "AP8="}
    assert child.inputs["not_a_number"] == {"$nonfinite": "NaN"}

    client = FakeMlflowClient()
    gateway = client.gateway()
    from opencode_mlflow.mlflow.capabilities import CapabilityRequest

    assert gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True)).is_compatible
    created = gateway.create_trace(request)
    gateway.verify_trace(created, request)
    root = client.get_trace(created.trace_id).data.spans[0]
    assert root.inputs == expected
    assert root.outputs == expected

    artifact = io.BytesIO()
    stream_turn_artifact(
        turn,
        artifact,
        {"session_message:assistant-1": "span-assistant-1"},
        RedactionMode.VERBATIM,
    )
    artifact_bytes = artifact.getvalue()
    assert b'"$duplicate_keys"' in artifact_bytes
    assert b'{"$bytes":"AP8="}' in artifact_bytes
    assert b'{"$nonfinite":"NaN"}' in artifact_bytes
    assert b"[REDACTED]" in artifact_bytes


@pytest.mark.parametrize("pending", ["pending", "running", "queued", "in_progress", "in-progress"])
def test_pending_assistant_outcome_is_incomplete_from_source_to_trace(tmp_path: Path, pending: str) -> None:
    source = _current_db(
        tmp_path / f"pending-{pending}.db",
        [
            ("user-1", "user", 1, 1_772_000_001_000, 1_772_000_001_100, {"content": []}),
            (
                "assistant-pending",
                "assistant",
                2,
                1_772_000_002_000,
                1_772_000_002_100,
                {"content": [], "status": pending},
            ),
        ],
    )
    with _real_current_turn(source, tmp_path) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = adapter.session_header("ses-wave2")
        assert header is not None
        (summary,) = iter_turn_summaries(adapter.iter_records("ses-wave2"))
        assert summary.terminal_outcome == pending
        turn = build_replay_turn(snapshot, header, summary)
        request = map_turn_to_trace(turn, _context(snapshot.fingerprint)).to_create_request()
        client = FakeMlflowClient()
        gateway = client.gateway()
        from opencode_mlflow.mlflow.capabilities import CapabilityRequest

        gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
        created = gateway.create_trace(request)
        gateway.verify_trace(created, request)

    assert request.root_status == "UNSET"
    assert request.root_outputs is None
    assert request.metadata["opencode.outcome"] == pending
    assert created.root_status == "UNSET"


def test_later_success_supersedes_pending_assistant_outcome(tmp_path: Path) -> None:
    source = _current_db(
        tmp_path / "pending-then-success.db",
        [
            ("user-1", "user", 1, 1_772_000_001_000, 1_772_000_001_100, {"content": []}),
            (
                "assistant-pending",
                "assistant",
                2,
                1_772_000_002_000,
                1_772_000_002_100,
                {"content": [], "status": "running"},
            ),
            ("assistant-success", "assistant", 3, 1_772_000_003_000, 1_772_000_003_100, {"content": []}),
        ],
    )
    with _real_current_turn(source, tmp_path) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = adapter.session_header("ses-wave2")
        assert header is not None
        (summary,) = iter_turn_summaries(adapter.iter_records("ses-wave2"))
        turn = build_replay_turn(snapshot, header, summary)
        request = map_turn_to_trace(turn, _context(snapshot.fingerprint)).to_create_request()
        children = {child.source_key: child for child in request.child_specs()}

    assert summary.root_result_source_id == "assistant-success"
    assert summary.terminal_outcome == "assistant_result"
    assert request.root_status == "OK"
    assert children["session_message:assistant-pending"].status == "UNSET"


@pytest.mark.parametrize("generation", ["current", "legacy"])
def test_nonfinite_and_huge_timestamp_ordering_is_missing_in_both_adapters(tmp_path: Path, generation: str) -> None:
    source = tmp_path / f"{generation}-timestamp-order.db"
    connection = sqlite3.connect(source)
    try:
        connection.executescript(CURRENT_DDL if generation == "current" else LEGACY_DDL)
        table = "session_v2" if generation == "current" else "session"
        connection.executemany(
            f"INSERT INTO {table} (id, time_created, time_updated) VALUES (?, ?, ?)",
            [
                ("finite", 1_000, 1_000),
                ("negative-inf", float("-inf"), float("-inf")),
                ("positive-inf", float("inf"), 1e308),
                ("huge-finite", 1e15, 1e15),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    connection = sqlite3.connect(source)
    try:
        exporter = (
            CurrentAdapter(connection, CURRENT_CATALOG_V1)
            if generation == "current"
            else LegacyAdapter(connection, LEGACY_CATALOG_V1)
        )
    finally:
        connection.close()

    with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        headers = list(adapter.iter_session_headers())

    assert [header.session_id for header in headers] == [
        "finite",
        "huge-finite",
        "negative-inf",
        "positive-inf",
    ]
    assert all(header.time_created is None for header in headers[1:])
    assert all("malformed_source_value" in header.warnings for header in headers[1:])


def test_provider_model_aliases_are_allowed_but_unknown_fields_still_warn(tmp_path: Path) -> None:
    source = _current_db(
        tmp_path / "model-aliases.db",
        [
            (
                "user-1",
                "user",
                1,
                1_772_000_001_000,
                1_772_000_001_100,
                {"content": [], "provider": "provider", "modelID": "model"},
            ),
            (
                "assistant-1",
                "assistant",
                2,
                1_772_000_002_000,
                1_772_000_002_100,
                {
                    "content": [],
                    "providerID": "provider",
                    "model": {"provider": "provider", "modelID": "model"},
                    "future_key": "warn-me",
                },
            ),
        ],
    )
    with _real_current_turn(source, tmp_path) as snapshot:
        records = list(NormalizedSnapshotAdapter(snapshot).iter_records("ses-wave2"))

    user, assistant = records
    assert "unknown_source_field" not in user.warnings
    assert "unknown_source_field" in assistant.warnings


@pytest.mark.parametrize(
    "part",
    [
        {"type": "tool", "provider_id": "provider"},
        {"type": "tool", "state": {"provider_id": "provider", "status": "completed"}},
    ],
)
def test_provider_id_is_allowed_in_every_part_json_allowlist(part: dict[str, object]) -> None:
    assert unknown_json_field_warning(part, "part") is None


@pytest.mark.parametrize(
    "model_value",
    [
        {"providerID": "provider", "modelID": "model"},
        {"provider_id": "provider", "modelID": "model"},
        {"provider": "provider", "model": {"provider_id": "provider", "modelID": "model"}},
    ],
)
def test_provider_model_aliases_do_not_emit_unknown_field_warning(model_value: dict[str, object]) -> None:
    assert unknown_json_field_warning(model_value, "model") is None


def test_probe_warning_reaches_discovery_session_and_import_report(tmp_path: Path) -> None:
    source = _current_db(
        tmp_path / "probe-warning.db",
        [("user-1", "user", 1, 1_772_000_001_000, 1_772_000_001_100, {"content": []})],
    )
    connection = sqlite3.connect(source)
    try:
        connection.execute("CREATE TABLE session_share (session_id TEXT, value TEXT)")
        connection.commit()
    finally:
        connection.close()

    with _real_current_turn(source, tmp_path) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        plan_path = tmp_path / "discovery-plan.jsonl"
        discover_sessions(adapter, SelectionFilters(), plan_path)
        scan = next(DiscoveryPlanReader(plan_path).iter_session_scans())
    assert "session_share_omitted" in scan.warnings

    result = run_import(
        ImportConfig("probe-warning-report", source),
        build_import_fakes(restrictive_tmp_path=tmp_path),
    )
    warning_codes = {warning["code"] for warning in result.report["summary"]["warnings"]}
    report_scan = result.report["sessions"][0]
    assert "session_share_omitted" in warning_codes
    assert "session_share_omitted" in {warning["code"] for warning in report_scan["warnings"]}
