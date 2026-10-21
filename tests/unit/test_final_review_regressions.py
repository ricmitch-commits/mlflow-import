"""Regressions for the scoped whole-branch final-review fix wave."""

from __future__ import annotations

import io
import json
import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from opencode_mlflow.config import ImportConfig
from opencode_mlflow.content.artifacts import stream_turn_artifact
from opencode_mlflow.content.redaction import DeterministicRedactor
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.importer import _record_normal_results, exit_status, run_import
from opencode_mlflow.mlflow.mapping import MappingContext, map_turn_to_trace
from opencode_mlflow.mlflow.reconciliation import (
    OwnedTraceCandidate,
    ReconciliationAction,
    ReconciliationPlan,
)
from opencode_mlflow.reporting import ReportBuilder, TurnResult, render_human, render_json
from opencode_mlflow.source.discovery import DiscoveryTurnEntry, extract_model_identity
from opencode_mlflow.source.probe import probe_schema
from opencode_mlflow.source.registry import build_live_exporter
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter
from opencode_mlflow.source.snapshot import create_normalized_snapshot
from opencode_mlflow.source.turns import build_replay_turn, iter_turn_summaries
from tests.support.import_fakes import LEGACY_DDL, build_current_source_db, build_import_fakes
from tests.support.sqlite_builder import build_fixture_db, fixture_script

pytestmark = pytest.mark.unit


@contextmanager
def exported(path, root, mode=RedactionMode.VERBATIM):
    with sqlite3.connect(path) as connection:
        exporter = build_live_exporter(connection, probe_schema(connection))
    with create_normalized_snapshot(path, root, exporter, DeterministicRedactor(mode).redact) as snapshot:
        yield snapshot, NormalizedSnapshotAdapter(snapshot)


def legacy_source(tmp_path, extra):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(LEGACY_DDL.replace("message_id TEXT NOT NULL", "message_id TEXT"))
        connection.executescript("INSERT INTO session (id, time_created) VALUES ('s1', 1000), ('s2', 2000);" + extra)
    return path


@pytest.mark.parametrize("generation", ["current", "legacy"])
def test_share_url_sentinel_stays_out_of_every_import_output_surface(tmp_path: Path, generation: str) -> None:
    sentinel = f"SHARE_URL_PIPELINE_SENTINEL_{generation}"
    path = tmp_path / f"{generation}-share-url.db"
    connection = build_fixture_db(path, generation)
    try:
        table = "session_v2" if generation == "current" else "session"
        connection.execute(f"UPDATE {table} SET share_url = ?", (sentinel,))
        connection.commit()
    finally:
        connection.close()

    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    result = run_import(ImportConfig(f"share-url-{generation}", path), dependencies)

    assert exit_status(result) == (1 if generation == "legacy" else 0)
    warning_codes = {warning["code"] for warning in result.report["summary"]["warnings"]}
    assert "session_share_omitted" in warning_codes
    assert sentinel not in render_json(result)
    assert sentinel not in render_human(result)
    for trace_id in dependencies.gateway.trace_ids():
        stored = dependencies.gateway.stored_trace(trace_id)
        assert stored is not None
        assert sentinel not in repr(stored)
        artifact_path = stored.tags["opencode.artifact_path"]
        assert sentinel not in dependencies.gateway.stored_artifact(trace_id, artifact_path).decode("utf-8")


def test_typed_headers_and_valid_raw_json_are_redacted_before_persistence_and_artifact(tmp_path):
    secret = "synthetic-header-secret"
    raw = '{"api\\u005fkey":"escaped-secret","Authorization":"Basic c3ludGhldGljOnNlY3JldA=="}'
    nested_basic = "bmVzdGVkOnNlY3JldA=="
    raw = raw[:-1] + ',"nested":' + json.dumps(json.dumps({"Authorization": f"Basic {nested_basic}"})) + "}"
    path = build_current_source_db(tmp_path, sessions=[("s1", 1000)], messages=[("u1", "s1", "user", 1, raw)])
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE session_v2 SET title=?, metadata=?, model=?, summary_diffs=?",
            (
                f"password={secret}",
                json.dumps({"api_key": secret}),
                json.dumps({"providerID": "p", "modelID": f"password={secret}"}),
                json.dumps({"token": secret}),
            ),
        )
    with exported(path, tmp_path, RedactionMode.REDACTED) as (snapshot, adapter):
        header = next(adapter.iter_session_headers())
        record = next(adapter.iter_records("s1"))
        assert header.session_id == "s1"
        assert header.time_created.value == 1000
        assert secret not in repr(header)
        assert json.loads(record.raw_json) == record.decoded_value
        assert json.loads(record.raw_columns["data"]) == record.decoded_value
        assert record.source_id == "u1"
        assert "c3ludGhldGljOnNlY3JldA==" not in repr(record)
        assert "escaped-secret" not in repr(record)
        assert nested_basic not in repr(record)
        summary = next(iter_turn_summaries(adapter.iter_records("s1")))
        output = io.BytesIO()
        manifest = stream_turn_artifact(
            build_replay_turn(snapshot, header, summary), output, {}, RedactionMode.REDACTED
        )
        assert secret.encode() not in output.getvalue()
    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    result = run_import(ImportConfig("redacted", path, redact=True), dependencies)
    assert exit_status(result) == 0
    trace_id = result.report["sessions"][0]["turns"][0]["trace_id"]
    trace = dependencies.gateway.stored_trace(trace_id)
    assert secret not in repr(trace)
    artifact = json.loads(dependencies.gateway.stored_artifact(trace_id, trace.tags["opencode.artifact_path"]))
    assert artifact["artifact"]["redaction_mode"] == "redacted"
    assert manifest.snapshot_fingerprint in trace.tags["opencode.artifact_path"]


def test_unknown_type_and_key_columns_are_redacted_without_changing_typed_identity(tmp_path):
    path = build_current_source_db(
        tmp_path,
        sessions=[("ses-unknown-raw-columns", 1000)],
        messages=[
            (
                "msg-unknown-raw-columns",
                "ses-unknown-raw-columns",
                "user",
                1,
                '{"content":[{"type":"text","text":"prompt"}]}',
            ),
            (
                "msg-unknown-raw-columns-assistant",
                "ses-unknown-raw-columns",
                "assistant",
                2,
                '{"content":[{"type":"text","text":"reply"}]}',
            ),
        ],
    )
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE session_v2 ADD COLUMN type TEXT")
        connection.execute("ALTER TABLE session_v2 ADD COLUMN key TEXT")
        connection.execute("ALTER TABLE session_message ADD COLUMN key TEXT")
        connection.execute(
            "UPDATE session_v2 SET type=?, key=? WHERE id=?",
            ("password=session-type-secret", "password=session-key-secret", "ses-unknown-raw-columns"),
        )
        connection.execute(
            "UPDATE session_message SET key=? WHERE id=?",
            ("password=message-key-secret", "msg-unknown-raw-columns"),
        )

    with exported(path, tmp_path, RedactionMode.REDACTED) as (snapshot, adapter):
        header = next(adapter.iter_session_headers())
        record = next(adapter.iter_records("ses-unknown-raw-columns"))
        persisted = "\n".join(
            str(row)
            for table in ("snapshot_sessions", "snapshot_records")
            for row in snapshot.connection.execute(f"SELECT * FROM {table}")
        )

    assert header.raw_columns["type"] == "password=[REDACTED]"
    assert header.raw_columns["key"] == "password=[REDACTED]"
    assert record.raw_columns["key"] == "password=[REDACTED]"
    assert record.source_id == "msg-unknown-raw-columns"
    assert record.classification == "user"
    for secret in ("session-type-secret", "session-key-secret", "message-key-secret"):
        assert secret not in persisted


def test_identity_like_transcript_keys_do_not_leak_but_structural_ids_and_turn_key_stay_unchanged(tmp_path):
    secrets = {
        "id": "nested-id-secret",
        "key": "nested-key-secret",
        "hash": "nested-hash-secret",
        "type": "nested-type-secret",
        "role": "nested-role-secret",
    }
    prompt = json.dumps(
        {
            "content": [{"type": "text", "text": "synthetic prompt"}],
            "id": {"password": secrets["id"]},
            "key": {"token": secrets["key"]},
            "hash": {"secret": secrets["hash"]},
            "type": {"credential": secrets["type"]},
            "role": {"api_key": secrets["role"]},
        }
    )
    path = build_current_source_db(
        tmp_path,
        sessions=[("ses-identity-boundary", 1000)],
        messages=[
            ("msg-identity-boundary-user", "ses-identity-boundary", "user", 1, prompt),
            (
                "msg-identity-boundary-assistant",
                "ses-identity-boundary",
                "assistant",
                2,
                '{"content":[{"type":"text","text":"synthetic reply"}]}',
            ),
        ],
    )
    verbatim_dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    verbatim = run_import(ImportConfig("identity-boundary-verbatim", path), verbatim_dependencies)
    redacted_dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    redacted = run_import(ImportConfig("identity-boundary-redacted", path, redact=True), redacted_dependencies)

    assert exit_status(verbatim) == exit_status(redacted) == 0
    verbatim_turn = verbatim.report["sessions"][0]["turns"][0]
    redacted_turn = redacted.report["sessions"][0]["turns"][0]
    assert redacted_turn["turn_key"] == verbatim_turn["turn_key"]
    trace = redacted_dependencies.gateway.stored_trace(redacted_turn["trace_id"])
    assert trace.metadata["opencode.session_id"] == "ses-identity-boundary"
    assert trace.metadata["opencode.turn_key"] == verbatim_turn["turn_key"]
    artifact_bytes = redacted_dependencies.gateway.stored_artifact(
        trace.trace_id,
        trace.tags["opencode.artifact_path"],
    )
    artifact = json.loads(artifact_bytes)
    assert any(record["source_id"] == "msg-identity-boundary-user" for record in artifact["turn_records"])
    for secret in secrets.values():
        assert secret not in repr(trace)
        assert secret.encode() not in artifact_bytes


def test_missing_id_turn_key_is_content_independent_across_redaction_modes(tmp_path):
    secret = "missing-id-turn-secret"
    prompt = json.dumps({"content": [{"type": "text", "text": "prompt"}], "password": secret})
    path = build_current_source_db(
        tmp_path,
        sessions=[("ses-missing-id-turn", 1000)],
        messages=[
            (None, "ses-missing-id-turn", "user", 1, prompt),
            (
                "msg-missing-id-turn-assistant",
                "ses-missing-id-turn",
                "assistant",
                2,
                '{"content":[{"type":"text","text":"reply"}]}',
            ),
        ],
    )
    verbatim_dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    redacted_dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)

    verbatim = run_import(ImportConfig("missing-id-verbatim", path), verbatim_dependencies)
    redacted = run_import(ImportConfig("missing-id-redacted", path, redact=True), redacted_dependencies)

    assert exit_status(verbatim) == exit_status(redacted) == 0
    verbatim_turn = verbatim.report["sessions"][0]["turns"][0]
    redacted_turn = redacted.report["sessions"][0]["turns"][0]
    assert [warning["code"] for warning in verbatim_turn["warnings"]] == [
        "turn_key_fallback_identity",
        "unknown_source_field",
    ]
    assert [warning["code"] for warning in redacted_turn["warnings"]] == [
        "turn_key_fallback_identity",
        "unknown_source_field",
    ]
    assert redacted_turn["turn_key"] == verbatim_turn["turn_key"]
    assert secret not in repr(redacted.report)


def test_missing_id_structural_identity_is_stable_when_only_secret_changes(tmp_path):
    def source(name: str, secret: str):
        return build_current_source_db(
            tmp_path,
            name=name,
            sessions=[("ses-missing-id-structural", 1000)],
            messages=[
                (
                    None,
                    "ses-missing-id-structural",
                    "user",
                    1,
                    json.dumps({"content": [{"type": "text", "text": "prompt"}], "password": secret}),
                ),
                (
                    "msg-missing-id-structural-assistant",
                    "ses-missing-id-structural",
                    "assistant",
                    2,
                    '{"content":[{"type":"text","text":"reply"}]}',
                ),
            ],
        )

    def import_one(path, *, redact: bool):
        dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
        result = run_import(
            ImportConfig(
                f"missing-id-structural-{redact}",
                path,
                redact=redact,
            ),
            dependencies,
        )
        assert exit_status(result) == 0
        turn = result.report["sessions"][0]["turns"][0]
        trace = dependencies.gateway.stored_trace(turn["trace_id"])
        return (
            turn["turn_key"],
            trace.metadata["opencode.source_fingerprint"],
            trace.tags["opencode.artifact_path"],
        )

    baseline = source("missing-id-structural-baseline.db", "password-alpha")
    changed = source("missing-id-structural-changed.db", "password-beta")

    baseline_redacted = import_one(baseline, redact=True)
    changed_redacted = import_one(changed, redact=True)
    assert changed_redacted == baseline_redacted

    baseline_verbatim = import_one(baseline, redact=False)
    changed_verbatim = import_one(changed, redact=False)
    assert changed_verbatim[0] == baseline_verbatim[0]


def test_model_filter_ignores_scoped_record_decoy(tmp_path):
    path = build_current_source_db(
        tmp_path,
        sessions=[("ses-scoped-model-decoy", 1000)],
        messages=[
            (
                "msg-scoped-model-decoy-user",
                "ses-scoped-model-decoy",
                "user",
                1,
                '{"content":[{"type":"text","text":"prompt"}]}',
            )
        ],
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE session_pending ("
            "id TEXT PRIMARY KEY, session_id TEXT NOT NULL, type TEXT, data TEXT, "
            "delivery TEXT, admitted_seq INTEGER, time_created INTEGER)"
        )
        connection.execute(
            "INSERT INTO session_pending VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "pending-scoped-model-decoy",
                "ses-scoped-model-decoy",
                "tool",
                '{"model":{"provider":"decoy-provider","model":"decoy-model"}}',
                "queued",
                1,
                1001,
            ),
        )

    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    result = run_import(
        ImportConfig(
            "scoped-model-decoy",
            path,
            model="decoy-provider/decoy-model",
            dry_run=True,
        ),
        dependencies,
    )

    assert exit_status(result) == 0
    assert result.report["sessions"][0]["selected"] is False
    assert result.report["summary"]["selected_sessions"] == 0


def test_exporter_orphan_and_missing_id_warnings_reach_end_to_end_report(tmp_path):
    path = build_current_source_db(
        tmp_path,
        name="export-warning-report.db",
        sessions=[("ses-export-warning", 1000)],
        messages=[("msg-export-warning", "ses-export-warning", "user", 1, "{}")],
    )
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO session_v2 (id, time_created) VALUES (NULL, 2000)")
        connection.execute(
            "CREATE TABLE session_pending ("
            "id TEXT PRIMARY KEY, session_id TEXT NOT NULL, type TEXT, data TEXT, "
            "delivery TEXT, admitted_seq INTEGER, time_created INTEGER)"
        )
        connection.execute(
            "INSERT INTO session_pending VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("pending-export-orphan", "ses-not-exported", "tool", "{}", "queued", 1, 1001),
        )

    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    result = run_import(ImportConfig("export-warning-report", path, dry_run=True), dependencies)

    warning_codes = {warning["code"] for warning in result.report["summary"]["warnings"]}
    assert {"session_missing_id", "unassociated_record"} <= warning_codes
    assert "ses-not-exported" not in repr(result.report)


@pytest.mark.parametrize("generation", ["legacy", "current"])
@pytest.mark.parametrize("status, expected_status", [("failed", "ERROR"), ("pending", "UNSET")])
def test_adapter_snapshot_turn_mapping_preserves_session_lifecycle_outcomes(
    tmp_path: Path, generation: str, status: str, expected_status: str
) -> None:
    if generation == "legacy":
        path = legacy_source(
            tmp_path,
            "INSERT INTO message VALUES ('lifecycle-user', 's1', 1000, 1000, '{\"role\": \"user\"}');",
        )
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE todo (session_id TEXT, content TEXT, status TEXT, priority TEXT, "
                "position INTEGER, time_created INTEGER, time_updated INTEGER)"
            )
            connection.execute(
                "INSERT INTO todo VALUES ('s1', 'synthetic todo', ?, 'normal', 1, 1001, 1001)",
                (status,),
            )
    else:
        path = build_current_source_db(
            tmp_path,
            name=f"lifecycle-{status}.db",
            sessions=[("s1", 1000)],
            messages=[("current-lifecycle-user", "s1", "user", 1, "{}")],
        )
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE session_pending ("
                "id TEXT PRIMARY KEY, session_id TEXT NOT NULL, type TEXT, data TEXT, "
                "delivery TEXT, admitted_seq INTEGER, time_created INTEGER)"
            )
            connection.execute(
                "INSERT INTO session_pending VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("pending-lifecycle", "s1", "tool", json.dumps({"status": status}), status, 1, 1001),
            )

    with exported(path, tmp_path) as (snapshot, adapter):
        header = next(adapter.iter_session_headers())
        summary = next(iter_turn_summaries(adapter.iter_records("s1")))
        replay = build_replay_turn(snapshot, header, summary)
        spec = map_turn_to_trace(
            replay,
            MappingContext(
                importer_version="test",
                mapping_version="mapping-v3",
                artifact_schema_version="artifact-v2",
                source_schema=generation,
                redaction_mode=RedactionMode.VERBATIM,
                experiment_id="exp-lifecycle",
            ),
        )

    assert spec.root_status == expected_status
    assert spec.metadata["opencode.outcome"] == f"lifecycle_{status}"

    if generation == "current" and status == "pending":
        path = build_current_source_db(
            tmp_path,
            name="lifecycle-timing.db",
            sessions=[("s1", 1000)],
            messages=[("timing-user", "s1", "user", 1, "{}")],
        )
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE session_v2 SET time_created=NULL, time_updated=NULL")
            connection.execute("UPDATE session_message SET time_created=NULL, time_updated=NULL")
            connection.execute(
                "CREATE TABLE session_pending ("
                "id TEXT PRIMARY KEY, session_id TEXT NOT NULL, type TEXT, data TEXT, "
                "delivery TEXT, admitted_seq INTEGER, time_created INTEGER)"
            )
            connection.execute(
                "INSERT INTO session_pending VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("pending-timing", "s1", "event", '{"outcome":"pending"}', "pending", 2, 2000),
            )

        dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
        result = run_import(ImportConfig("lifecycle-timing", path), dependencies)

        assert exit_status(result) == 0
        turn = result.report["sessions"][0]["turns"][0]
        trace = dependencies.gateway.stored_trace(turn["trace_id"])
        assert trace is not None
        assert trace.start_time_ns == 2_000_000_000
        assert trace.end_time_ns == 2_000_000_000


def test_escaped_duplicate_secret_keys_survive_losslessly_but_redacted_in_uploaded_artifact(tmp_path):
    first_secret = "first-duplicate-secret"
    escaped_secret = "escaped-duplicate-secret"
    raw = '{"password":"' + first_secret + '","\\u0070assword":"' + escaped_secret + '"}'
    path = build_current_source_db(
        tmp_path,
        sessions=[("ses-duplicate-artifact", 1000)],
        messages=[
            ("msg-duplicate-artifact-user", "ses-duplicate-artifact", "user", 1, raw),
            (
                "msg-duplicate-artifact-assistant",
                "ses-duplicate-artifact",
                "assistant",
                2,
                '{"content":[{"type":"text","text":"reply"}]}',
            ),
        ],
    )
    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)

    result = run_import(ImportConfig("duplicate-artifact", path, redact=True), dependencies)

    assert exit_status(result) == 0
    turn = result.report["sessions"][0]["turns"][0]
    trace = dependencies.gateway.stored_trace(turn["trace_id"])
    artifact_bytes = dependencies.gateway.stored_artifact(trace.trace_id, trace.tags["opencode.artifact_path"])
    artifact = json.loads(artifact_bytes)
    record = next(item for item in artifact["turn_records"] if item["source_id"] == "msg-duplicate-artifact-user")
    assert record["decoded"]["$duplicate_keys"] == [
        ["password", "[REDACTED]"],
        ["password", "[REDACTED]"],
    ]
    assert record["raw_text"] == '{"password":"[REDACTED]","password":"[REDACTED]"}'
    for secret in (first_secret, escaped_secret):
        assert secret not in repr(trace)
        assert secret.encode() not in artifact_bytes


def test_truncated_escaped_secret_json_is_masked_in_snapshot_and_artifact(tmp_path):
    secret = "truncated-escaped-secret"
    raw = '{"pass\\u0077ord":"' + secret + '"'
    path = build_current_source_db(
        tmp_path,
        sessions=[("ses-truncated-escaped", 1000)],
        messages=[
            ("msg-truncated-escaped-user", "ses-truncated-escaped", "user", 1, raw),
            (
                "msg-truncated-escaped-assistant",
                "ses-truncated-escaped",
                "assistant",
                2,
                '{"content":[{"type":"text","text":"reply"}]}',
            ),
        ],
    )
    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)

    result = run_import(ImportConfig("truncated-escaped", path, redact=True), dependencies)

    assert exit_status(result) == 0
    turn = result.report["sessions"][0]["turns"][0]
    trace = dependencies.gateway.stored_trace(turn["trace_id"])
    artifact_bytes = dependencies.gateway.stored_artifact(trace.trace_id, trace.tags["opencode.artifact_path"])
    artifact = json.loads(artifact_bytes)
    record = next(item for item in artifact["turn_records"] if item["source_id"] == "msg-truncated-escaped-user")
    assert record["raw_text"] == '{"pass\\u0077ord":"[REDACTED]"'
    assert secret not in artifact_bytes.decode("utf-8")


def test_legacy_merge_order_keeps_later_assistant_parts_and_complete_prompt(tmp_path):
    path = legacy_source(
        tmp_path,
        """
        INSERT INTO message VALUES ('a', 's1', 3000, 3000, '{"role":"assistant"}');
        INSERT INTO message VALUES ('empty', 's1', 1000, 1000, '{"role":"user"}');
        INSERT INTO message VALUES ('u', 's1', 2000, 2000, '{"role":"user"}');
        INSERT INTO part VALUES ('pa', 'a', 's1', 3000, 3000, '{"type":"text","text":"reply"}');
        INSERT INTO part VALUES ('p1', 'u', 's1', 2000, 2000, '{"type":"text","text":"prompt one"}');
        INSERT INTO part VALUES ('p2', 'u', 's1', 2001, 2001, '{"type":"text","text":"prompt two"}');
    """,
    )
    with exported(path, tmp_path) as (snapshot, adapter):
        assert {r.source_id for r in adapter.iter_records("s1")} == {"a", "empty", "u", "pa", "p1", "p2"}
        summaries = list(iter_turn_summaries(adapter.iter_records("s1")))
        replay = build_replay_turn(snapshot, next(adapter.iter_session_headers()), summaries[-1])
        prompt = replay.load_root_input(1000)
        assert [p["text"] for p in prompt.value] == ["prompt one", "prompt two"]
        assert replay.load_root_input(10).payload_too_large
        assert replay.load_root_output(1000).value == [{"type": "text", "text": "reply"}]
    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    result = run_import(ImportConfig("legacy", path), dependencies)
    assert exit_status(result) == 0
    trace = dependencies.gateway.stored_trace(result.report["sessions"][0]["turns"][-1]["trace_id"])
    assert trace.metadata["opencode.source_message_count"] == "3"
    assert trace.metadata["opencode.source_part_count"] == "3"


def test_missing_id_part_ownership_is_session_scoped(tmp_path):
    path = legacy_source(
        tmp_path,
        """
        INSERT INTO message VALUES (NULL, 's1', 1000, 1000, '{"role":"user"}');
        INSERT INTO message VALUES (NULL, 's2', 2000, 2000, '{"role":"user"}');
        INSERT INTO part VALUES ('p1', NULL, 's1', 1001, 1001, '{"type":"text","text":"one"}');
        INSERT INTO part VALUES ('p2', NULL, 's2', 2001, 2001, '{"type":"text","text":"two"}');
    """,
    )
    with exported(path, tmp_path) as (_snapshot, adapter):
        s1_records = list(adapter.iter_records("s1"))
        s2_records = list(adapter.iter_records("s2"))
        assert [r.source_id for r in s1_records] == [None, "p1"]
        assert [r.source_id for r in s2_records] == [None, "p2"]
        for records, part_id in ((s1_records, "p1"), (s2_records, "p2")):
            parent, part = records
            assert parent.source_id is None
            assert parent.source_identity is not None
            assert part.source_id == part_id
            assert part.parent_source_id == parent.source_identity


def test_ambiguous_missing_id_parts_are_preserved_unassigned_not_given_a_parent(tmp_path):
    path = legacy_source(
        tmp_path,
        """
        INSERT INTO message VALUES (NULL, 's1', 1000, 1000, '{"role":"user"}');
        INSERT INTO message VALUES (NULL, 's1', 2000, 2000, '{"role":"user"}');
        INSERT INTO part VALUES ('unassigned', NULL, 's1', 1500, 1500, '{"type":"text","text":"ambiguous"}');
    """,
    )
    with exported(path, tmp_path) as (snapshot, adapter):
        records = list(adapter.iter_records("s1"))
        assert records[0].source_id == "unassigned"
        assert records[0].parent_source_id is None
        assert "ambiguous_missing_message_id" in records[0].warnings
        summary = next(iter_turn_summaries(adapter.iter_records("s1")))
        replay = build_replay_turn(snapshot, next(adapter.iter_session_headers()), summary)
        assert "unassigned" not in [r.source_id for r in replay.iter_records()]
        assert "unassigned" in [r.source_id for r in replay.iter_unassigned_records()]


@pytest.mark.parametrize(
    "value",
    [
        {"providerID": "provider", "modelID": "model"},
        {"providerID": "provider", "model": "model"},
        {"model": {"providerID": "provider", "modelID": "model"}},
        {"providerID": "provider", "model": {"modelID": "model"}},
        {"provider": "provider", "model": "model"},
    ],
)
def test_shared_model_identity_aliases(value):
    assert extract_model_identity(value) == "provider/model"


@pytest.mark.parametrize("generation", ["legacy", "current"])
def test_duplicate_root_keeps_unique_typed_fields_visible_to_real_adapters(tmp_path, generation):
    if generation == "legacy":
        data = '{"role":"assistant","role":"user","providerID":"decoy","providerID":"provider","modelID":"model"}'
        path = legacy_source(tmp_path, "")
        with sqlite3.connect(path) as connection:
            connection.execute("INSERT INTO message VALUES ('u', 's1', 1000, 1000, ?)", (data,))
    else:
        data = '{"providerID":"decoy","providerID":"provider","modelID":"model"}'
        path = build_current_source_db(
            tmp_path,
            sessions=[("s1", 1000)],
            messages=[("u", "s1", "user", 1, data)],
        )

    with exported(path, tmp_path) as (_snapshot, adapter):
        record = next(record for record in adapter.iter_records("s1") if record.source_id == "u")
        summaries = list(iter_turn_summaries(adapter.iter_records("s1")))

    assert isinstance(record.decoded_value, Mapping)
    assert record.classification == "user"
    assert extract_model_identity(record.decoded_value) == "provider/model"
    assert len(summaries) == 1


@pytest.mark.parametrize("generation", ["legacy", "current"])
@pytest.mark.parametrize(
    "model",
    [
        {"model": {"providerID": "provider", "modelID": "model"}},
        {"providerID": "provider", "modelID": "model"},
        {"providerID": "provider", "model": "model"},
    ],
)
def test_realistic_model_aliases_drive_selection_and_llm_mapping(tmp_path, generation, model):
    data = json.dumps({"role": "assistant", **model})
    if generation == "legacy":
        path = legacy_source(
            tmp_path,
            """
            INSERT INTO message VALUES ('u', 's1', 1000, 1000, '{"role":"user"}');
        """,
        )
        with sqlite3.connect(path) as connection:
            connection.execute("INSERT INTO message VALUES ('a', 's1', 2000, 2000, ?)", (data,))
    else:
        path = build_current_source_db(
            tmp_path,
            sessions=[("s1", 1000)],
            messages=[
                ("u", "s1", "user", 1, "{}"),
                ("a", "s1", "assistant", 2, data),
            ],
        )
    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    result = run_import(ImportConfig("models", path, model="provider/model"), dependencies)
    assert exit_status(result) == 0
    assert result.report["source"]["sessions_selected"] == 1
    entry = result.report["sessions"][0]["turns"][0]
    stored = dependencies.gateway.stored_trace(entry["trace_id"])
    assert any(
        span.span_type == "LLM" and span.attributes["opencode.model"] == "provider/model"
        for span in stored.spans_by_source.values()
    )


def test_missing_timestamps_fail_item_without_trace_create(tmp_path):
    path = build_current_source_db(tmp_path, sessions=[("s1", 1000)], messages=[("u", "s1", "user", 1, "{}")])
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE session_message SET time_created=NULL, time_updated=NULL")
        connection.execute("UPDATE session_v2 SET time_created=NULL, time_updated=NULL")
    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    result = run_import(ImportConfig("missing-time", path), dependencies)
    assert exit_status(result) == 1
    turn = result.report["sessions"][0]["turns"][0]
    assert turn["error"]["code"] == "source_timestamp_missing"
    assert not any(event.kind == "trace_create" for event in dependencies.gateway.events)


def test_duplicate_stale_trace_results_do_not_collapse_and_report_contract():
    builder = ReportBuilder(dry_run=False, experiment_name="test", redaction_mode=RedactionMode.VERBATIM)
    builder.set_source_counts(sessions_scanned=1, sessions_selected=1)
    builder.add_session_scan(
        "s1", selected=True, logical_turn_count=2, record_count=5, warnings=("unknown_source_field",)
    )
    first = OwnedTraceCandidate("t1", "s1", "gone", 1)
    second = replace(first, trace_id="t2")
    retirements = [ReconciliationAction(action="retired", trace=c) for c in (first, second)]
    plan = ReconciliationPlan(retirements=retirements)
    plan.results = [
        replace(retirements[0], trace_id="t1"),
        replace(retirements[1], action="failed", trace_id="t2", error_code="retirement_failed"),
    ]
    _record_normal_results(builder, plan, [])
    builder.add_turn(TurnResult("s1", "current", "imported"))
    report = builder.build()
    session = report["sessions"][0]
    assert [(t["trace_id"], t["action"]) for t in session["turns"][:2]] == [("t1", "retired"), ("t2", "failed")]
    assert session["logical_turn_count"] == 2
    assert session["warnings"][0]["code"] == "unknown_source_field"
    assert session["error"]["code"] == "retirement_failed"
    assert session["turns"][0]["error"] is None
    assert report["summary"]["selected_turns"] == 2
    assert report["source"]["record_count"] == 5
    assert report["summary"]["failed_turns"] == 0
    assert report["summary"]["failed_items"] == 1


def test_reconciliation_trace_id_sets_reach_the_turn_report():
    builder = ReportBuilder(dry_run=False, experiment_name="test", redaction_mode=RedactionMode.VERBATIM)
    builder.add_session_scan("s1", selected=True, logical_turn_count=1)
    source_entry = DiscoveryTurnEntry(
        session_id="s1",
        turn_index=1,
        turn_key="turn-report-id",
        adapter_generation="current",
        planned_action="import",
        start_time_ns=1,
        end_time_ns=2,
        record_count=1,
        user_record_count=1,
        assistant_record_count=0,
        terminal_outcome=None,
        model_identities=frozenset(),
        warnings=(),
        session_context_fingerprint="s" * 64,
        source_fingerprint="a" * 64,
        render_fingerprint="b" * 64,
        artifact_schema_version="artifact-v1",
        mapping_version="mapping-v1",
        importer_version="importer-v1",
        redaction_mode=RedactionMode.VERBATIM.value,
    )
    outcome = ReconciliationAction(
        action="failed",
        turn=source_entry,
        trace_id="new-trace",
        affected_trace_ids=("old-trace", "new-trace"),
        retired_trace_ids=("old-trace",),
    )
    plan = ReconciliationPlan(results=[outcome])
    _record_normal_results(builder, plan, [source_entry])

    payload = builder.build()["sessions"][0]["turns"][0]
    assert payload["trace_id"] == "new-trace"
    assert payload["affected_trace_ids"] == ["old-trace", "new-trace"]
    assert payload["retired_trace_ids"] == ["old-trace"]


def test_session_mixed_import_retire_is_updated_and_unchanged_is_subcount():
    builder = ReportBuilder(dry_run=False, experiment_name="test", redaction_mode=RedactionMode.VERBATIM)
    builder.add_turn(TurnResult("s1", "new", "imported"))
    builder.add_turn(TurnResult("s1", "old", "retired", source_present=False))
    builder.add_turn(TurnResult("s1", "same", "skipped", reason="unchanged"))
    report = builder.build()
    assert report["sessions"][0]["action"] == "updated"
    assert report["summary"]["unchanged_turns"] == 1


def test_redundant_retirement_failure_is_exit_one_and_keeps_verified_winner(tmp_path):
    path = build_current_source_db(tmp_path, sessions=[("s1", 1000)], messages=[("u", "s1", "user", 1, "{}")])
    config = ImportConfig("cleanup", path)
    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    first = run_import(config, dependencies)
    old_id = first.report["sessions"][0]["turns"][0]["trace_id"]
    gateway = dependencies.gateway
    original = gateway.iter_owned_traces
    gateway.iter_owned_traces = lambda experiment_id: iter(())
    second = run_import(config, dependencies)
    gateway.iter_owned_traces = original
    winner_id = second.report["sessions"][0]["turns"][0]["trace_id"]
    gateway.fail_retire_trace_ids.add(old_id)
    result = run_import(config, dependencies)
    assert exit_status(result) == 1
    turn = result.report["sessions"][0]["turns"][0]
    assert turn["action"] == "failed"
    assert turn["trace_id"] == winner_id
    assert turn["error"]["code"] == "replacement_cleanup_failed"
    assert gateway.has_trace(winner_id)
    assert gateway.has_trace(old_id)


def test_manifest_fingerprint_must_match_mapped_request(tmp_path):
    path = build_current_source_db(tmp_path, sessions=[("s1", 1000)], messages=[("u", "s1", "user", 1, "{}")])
    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    original = dependencies.artifact_uploader

    def divergent(*args, **kwargs):
        return replace(original(*args, **kwargs), snapshot_fingerprint="0" * 64)

    result = run_import(ImportConfig("manifest", path, redact=True), replace(dependencies, artifact_uploader=divergent))
    assert exit_status(result) == 1
    assert result.report["sessions"][0]["turns"][0]["error"]["code"] == "source_consistency_mismatch"
    assert dependencies.gateway.trace_ids() == []


def test_shared_instruction_blob_survives_selecting_only_later_session(tmp_path):
    path = tmp_path / "shared.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(fixture_script("current"))
        connection.execute("UPDATE session_v2 SET directory='/selected' WHERE id='ses-current-002'")
    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    result = run_import(ImportConfig("shared", path, directory=Path("/selected")), dependencies)
    assert exit_status(result) == 0
    selected = next(s for s in result.report["sessions"] if s["selected"])
    assert selected["session_id"] == "ses-current-002"
    trace = dependencies.gateway.stored_trace(selected["turns"][0]["trace_id"])
    artifact = json.loads(dependencies.gateway.stored_artifact(trace.trace_id, trace.tags["opencode.artifact_path"]))
    blobs = [r for r in artifact["session_records"] if r["source_table"] == "instruction_blob"]
    assert len(blobs) == 1
    assert blobs[0]["source_id"] == "synth-blob-hash-001"
    assert blobs[0]["raw_columns"]["value"] == "synthetic instruction blob content"


def test_ci_job_gate_uses_supported_variable_context_not_secrets():
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    gated = [job for job in workflow["jobs"].values() if "if" in job]
    assert len(gated) == 1
    assert "secrets." not in gated[0]["if"]
    assert "vars.MLFLOW_INTEGRATION_ENABLED" in gated[0]["if"]
    step = next(step for step in gated[0]["steps"] if "env" in step)
    assert "secrets.MLFLOW_INTEGRATION_URI" in step["env"]["MLFLOW_TRACKING_URI"]


def test_corrected_mapping_contract_replaces_old_version_once(tmp_path):
    path = build_current_source_db(tmp_path, sessions=[("s1", 1000)], messages=[("u", "s1", "user", 1, "{}")])
    config = ImportConfig("version-migration", path)
    dependencies = build_import_fakes(restrictive_tmp_path=tmp_path)
    mapper = dependencies.mapper
    old = replace(
        dependencies, mapper=lambda turn, context: mapper(turn, replace(context, mapping_version="mapping-v1"))
    )
    first = run_import(config, old)
    assert exit_status(first) == 0
    second = run_import(config, dependencies)
    assert exit_status(second) == 0
    assert second.report["sessions"][0]["turns"][0]["action"] == "updated"
    trace = dependencies.gateway.stored_trace(second.report["sessions"][0]["turns"][0]["trace_id"])
    assert trace.metadata["opencode.mapping_version"] == "mapping-v3"
    marker = len(dependencies.gateway.mutations)
    third = run_import(config, dependencies)
    assert exit_status(third) == 0
    assert third.report["sessions"][0]["turns"][0]["reason"] == "unchanged"
    assert dependencies.gateway.mutations[marker:] == []
