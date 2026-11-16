"""Local-only six-source fixtures: source artifacts, deltas, map publication."""

import hashlib
import io
import json
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from mlflow_import.content.activity import activity_specs, project_activity
from mlflow_import.content.bundles import stream_session_object
from mlflow_import.content.canonical import canonical_json
from mlflow_import.content.input_context import INPUT_CONTEXT_ATTRIBUTE, validate_input_contexts
from mlflow_import.contracts import RedactionMode
from mlflow_import.errors import ItemImportError
from mlflow_import.identity import qualified_session
from mlflow_import.mlflow.gateway import ArtifactVerification, CreatedTrace
from mlflow_import.mlflow.mapping import MappingContext, map_turn_to_trace
from mlflow_import.mlflow.reconciliation import _upload_bundle
from mlflow_import.source.catalog import CURRENT_CATALOG_V1, LEGACY_CATALOG_V1
from mlflow_import.source.current import CurrentAdapter
from mlflow_import.source.discovery import ARTIFACT_SCHEMA_VERSION, MAPPING_VERSION
from mlflow_import.source.file_snapshot import create_file_snapshot
from mlflow_import.source.legacy import LegacyAdapter
from mlflow_import.source.replay import NormalizedSnapshotAdapter
from mlflow_import.source.snapshot import create_normalized_snapshot
from mlflow_import.source.turns import build_replay_turn, indexed_turn_summaries
from mlflow_import.sources.jsonl_common import JsonlExporter
from mlflow_import.storage import disk_collections
from tests.support.import_fakes import build_current_source_db
from tests.support.sqlite_builder import fixture_script

pytestmark = pytest.mark.unit


def values(harness):
    tool = {"id": "c", "name": "read", "input": {"path": "literal path"}}
    assistant = {"role": "assistant", "model": "synthetic/model", "content": "before"}
    final = {"role": "assistant", "model": "synthetic/model", "content": "final", "stop_reason": "stop"}
    if harness == "claude":
        return [
            {"sessionId": "s", "type": "user", "uuid": "u", "message": {"role": "user", "content": "prompt"}},
            {
                "sessionId": "s",
                "type": "assistant",
                "uuid": "a",
                "parentUuid": "u",
                "message": {**assistant, "id": "inv-a", "content": [{"type": "tool_use", **tool}]},
            },
            {
                "sessionId": "s",
                "type": "user",
                "uuid": "r",
                "parentUuid": "a",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "c", "content": "actual result"}],
                },
            },
            {
                "sessionId": "s",
                "type": "assistant",
                "uuid": "z",
                "parentUuid": "r",
                "message": {**final, "id": "inv-z"},
            },
        ]
    if harness == "pi":
        return [
            {"type": "session", "version": 3, "id": "s"},
            {"type": "message", "id": "u", "parentId": None, "message": {"role": "user", "content": "prompt"}},
            {
                "type": "message",
                "id": "a",
                "parentId": "u",
                "message": {**assistant, "content": [{"type": "toolCall", **tool, "arguments": tool["input"]}]},
            },
            {
                "type": "message",
                "id": "r",
                "parentId": "a",
                "message": {"role": "toolResult", "toolCallId": "c", "content": "actual result"},
            },
            {"type": "message", "id": "z", "parentId": "r", "message": final},
        ]
    if harness == "gemini":
        return [
            {"sessionId": "s", "projectHash": "synthetic"},
            {"type": "user", "id": "u", "content": "prompt"},
            {"type": "gemini", "id": "a", "model": "synthetic/model", "content": "obsolete"},
            {
                "type": "gemini",
                "id": "a",
                "model": "synthetic/model",
                "content": "before",
                "toolCalls": [{"id": "c", "name": "read", "args": tool["input"]}],
            },
            {"$patch": {"id": "a", "toolCalls": [{"id": "c", "result": "actual result"}]}},
            {"type": "gemini", "id": "z", "model": "synthetic/model", "content": "final"},
            {"type": "user", "id": "discarded", "content": "inactive private branch"},
            {"$rewindTo": "discarded"},
        ]
    return [
        {"type": "session_meta", "payload": {"id": "s"}},
        {"type": "response_item", "payload": {"id": "u", "type": "message", "role": "user", "content": "prompt"}},
        {"type": "turn_context", "payload": {"model": "synthetic/model"}},
        {"type": "response_item", "payload": {"id": "a", "type": "message", **assistant}},
        {
            "type": "response_item",
            "payload": {
                "id": "call",
                "type": "function_call",
                "call_id": "c",
                "name": "read",
                "arguments": '{"path":"literal path"}',
            },
        },
        {
            "type": "response_item",
            "payload": {"id": "r", "type": "function_call_output", "call_id": "c", "output": "actual result"},
        },
        {"type": "response_item", "payload": {"id": "z", "type": "message", **final}},
    ]


@contextmanager
def frozen(tmp_path, source, mode):
    def redact(value, _):
        if mode is RedactionMode.REDACTED:
            # Source values, including raw envelopes, cross privacy before projection.
            if isinstance(value, str):
                return value.replace("literal path", "[REDACTED]")
            if isinstance(value, dict):
                return {k: redact(v, ()) for k, v in value.items()}
            if isinstance(value, list):
                return [redact(v, ()) for v in value]
        return value

    if source.startswith("opencode"):
        user = {"role": "user", "content": "prompt"}
        assistant = {
            "role": "assistant",
            "model": "synthetic/model",
            "content": [
                {
                    "type": "tool",
                    "tool": "read",
                    "callID": "c",
                    "state": {"input": {"path": "literal path"}, "output": "actual result", "status": "completed"},
                }
            ],
        }
        final = {"role": "assistant", "model": "synthetic/model", "content": "final", "finish": "stop"}
        if source == "opencode-current":
            path = build_current_source_db(
                tmp_path,
                sessions=[("s", 1000)],
                messages=[
                    ("u", "s", "user", 1, json.dumps(user)),
                    ("a", "s", "assistant", 2, json.dumps(assistant)),
                    ("z", "s", "assistant", 3, json.dumps(final)),
                ],
            )
            db = sqlite3.connect(path)
            try:
                exporter = CurrentAdapter(db, CURRENT_CATALOG_V1)
            finally:
                db.close()
        else:
            path = tmp_path / "legacy.db"
            with sqlite3.connect(path) as db:
                db.executescript(fixture_script("legacy"))
                for table in ("part", "message", "todo", "session"):
                    db.execute(f"DELETE FROM {table}")
                db.execute("INSERT INTO session(id,time_created) VALUES('s',1000)")
                for i, (key, value) in enumerate((("u", user), ("a", assistant), ("z", final)), 1):
                    db.execute(
                        "INSERT INTO message(id,session_id,time_created,data) VALUES(?,'s',?,?)",
                        (key, i * 1000, json.dumps(value)),
                    )
                exporter = LegacyAdapter(db, LEGACY_CATALOG_V1)
        context = create_normalized_snapshot(path, tmp_path, exporter, redact)
    else:
        path = tmp_path / f"{source}.jsonl"
        path.write_text("".join(json.dumps(value) + "\n" for value in values(source)))
        context = create_file_snapshot(path, tmp_path, JsonlExporter(path, source), redact)
    with context as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = next(adapter.iter_session_headers())
        summary = next(indexed_turn_summaries(snapshot, header.session_id))
        yield build_replay_turn(snapshot, header, summary)


@pytest.mark.parametrize("source", ["opencode-current", "opencode-legacy", "claude", "codex", "pi", "gemini"])
@pytest.mark.parametrize("mode", [RedactionMode.VERBATIM, RedactionMode.REDACTED])
def test_six_source_deltas_map_and_full_artifact_remain_consistent(tmp_path, source, mode):
    with frozen(tmp_path, source, mode) as turn:
        activity = project_activity(turn)
        specs = activity_specs(activity, turn)
        calls = [s for s in specs if INPUT_CONTEXT_ATTRIBUTE in s.attributes]
        assert len(calls) == 2
        assert b"prompt" in canonical_json(calls[0].inputs)
        assert b"prompt" not in canonical_json(calls[1].inputs)
        messages = calls[1].inputs["messages"]
        assert {"role": "tool", "tool_call_id": "c", "content": "actual result"} in messages
        assert any(tool["id"] == "c" for m in messages for tool in m.get("tool_calls", []))
        assert calls[1].attributes[INPUT_CONTEXT_ATTRIBUTE]["provider_request"] == "unknown"
        assert specs == activity_specs(project_activity(turn), turn)
        artifact = io.BytesIO()
        stream_session_object(turn, artifact, mode)
        retained = json.loads(artifact.getvalue())
        assert retained["schema_version"] == "mlflow-import.session.v1"
        assert b"prompt" in artifact.getvalue() and b"actual result" in artifact.getvalue()
        assert (b"literal path" in artifact.getvalue()) is (mode is RedactionMode.VERBATIM)
        if source == "gemini":
            assert b"obsolete" in artifact.getvalue() and b"inactive private branch" in artifact.getvalue()
            assert b"obsolete" not in canonical_json([s.inputs for s in calls])
            origin = calls[0].attributes[INPUT_CONTEXT_ATTRIBUTE]["invocation_sources"][0]["occurrence_id"]
            assert "effective:" in origin
        context = MappingContext(
            "0.1.0",
            MAPPING_VERSION,
            ARTIFACT_SCHEMA_VERSION,
            turn.session_header.adapter_generation,
            mode,
            "local-only",
        )
        mapped = map_turn_to_trace(turn, context)
        assert b"prompt" in canonical_json(mapped.root_inputs)
        repeated = map_turn_to_trace(turn, context)
        assert mapped.root_inputs == repeated.root_inputs
        assert mapped.root_tags == repeated.root_tags
        assert list(mapped.child_specs()) == list(repeated.child_specs())


@pytest.mark.parametrize("missing_context_span", [False, True])
def test_upload_bundle_persists_validator_compatible_linear_index_without_target(tmp_path, missing_context_span):
    with frozen(tmp_path, "gemini", RedactionMode.VERBATIM) as turn:
        specs = activity_specs(project_activity(turn), turn)
        ownership = {
            "mlflow_import.qualified_session_id": qualified_session(turn.session_header),
            "mlflow_import.harness": "gemini",
            "mlflow_import.source_namespace": turn.session_header.source_fingerprint,
            "mlflow_import.session_id": turn.session_header.session_id,
            "mlflow_import.turn_key": turn.summary.turn_key,
            "opencode.turn_key": turn.summary.turn_key,
            "opencode.render_fingerprint": "f" * 64,
            "opencode.mapping_version": MAPPING_VERSION,
            "opencode.importer_version": "0.1.0",
        }
        span_ids = {s.source_key: f"span{i}" for i, s in enumerate(specs)}
        if missing_context_span:
            del span_ids[next(s.source_key for s in specs if INPUT_CONTEXT_ATTRIBUTE in s.attributes)]
        created = CreatedTrace("trace", "root", span_ids, ownership, "OK")
        captured = {}

        def publish(experiment, path, local, *, trace_id=""):
            payload = local.read_bytes()
            captured[path] = json.loads(payload)
            return ArtifactVerification(trace_id, path, hashlib.sha256(payload).hexdigest(), len(payload))

        gateway = SimpleNamespace(
            _session_objects={},
            trace_info=lambda _: SimpleNamespace(experiment_id="local"),
            publish_experiment_file=publish,
            reconcile_tags=lambda *args, **kwargs: None,
        )
        with disk_collections() as store:
            try:
                if missing_context_span:
                    with pytest.raises(ItemImportError, match="no generated span identity"):
                        _upload_bundle(turn, created, "map.json", RedactionMode.VERBATIM, gateway, tmp_path)
                    assert "map.json" not in captured
                    return
                verification = _upload_bundle(turn, created, "map.json", RedactionMode.VERBATIM, gateway, tmp_path)
            finally:
                store.close()
        assert verification.snapshot_fingerprint
        value = captured["map.json"]
        assert validate_input_contexts(value, ownership)
        spans = [SimpleNamespace(span_id="root", trace_id="trace", parent_id=None, attributes={})]
        spans.extend(
            SimpleNamespace(span_id=span_ids[s.source_key], trace_id="trace", parent_id="root", attributes=s.attributes)
            for s in specs
        )
        assert validate_input_contexts(value, ownership, spans=spans)
        assert len(value["input_contexts"]) == 2
        assert (
            value["input_contexts"][1]["descriptor"]["previous_context_id"]
            == (value["input_contexts"][0]["descriptor"]["context_id"])
        )


def test_sibling_append_changes_session_artifact_but_not_existing_turn_context_or_render(tmp_path):
    path = tmp_path / "stable.jsonl"
    original = values("gemini")
    path.write_text("".join(json.dumps(value) + "\n" for value in original))

    def capture():
        with create_file_snapshot(path, tmp_path, JsonlExporter(path, "gemini"), lambda value, _: value) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            header = next(adapter.iter_session_headers())
            summary = next(indexed_turn_summaries(snapshot, header.session_id))
            turn = build_replay_turn(snapshot, header, summary)
            context = MappingContext(
                "0.1.0",
                MAPPING_VERSION,
                ARTIFACT_SCHEMA_VERSION,
                header.adapter_generation,
                RedactionMode.VERBATIM,
                "local-only",
            )
            mapped = map_turn_to_trace(turn, context)
            full = io.BytesIO()
            stream_session_object(turn, full, RedactionMode.VERBATIM)
            return mapped, list(mapped.child_specs()), full.getvalue()

    before, before_children, before_artifact = capture()
    original.extend(
        [
            {"type": "user", "id": "sibling-user", "content": "later sibling prompt"},
            {"type": "gemini", "id": "sibling-answer", "model": "synthetic/model", "content": "later answer"},
        ]
    )
    path.write_text("".join(json.dumps(value) + "\n" for value in original))
    after, after_children, after_artifact = capture()
    assert before_artifact != after_artifact
    assert before.expected_operation_key == after.expected_operation_key
    assert before.root_tags["mlflow_import.turn_fingerprint"] == after.root_tags["mlflow_import.turn_fingerprint"]
    assert before_children == after_children


@pytest.mark.parametrize("operation", ["usage", "compaction", "branch_summary"])
@pytest.mark.parametrize("with_tool", [False, True])
def test_pi_persisted_accounting_entries_do_not_consume_conversation_inputs(
    tmp_path, monkeypatch, operation, with_tool
):
    from mlflow_import.compatibility import request_ownership
    from mlflow_import.content.canonical import iter_canonical_json
    from mlflow_import.content.turn_map import _prospective_turn_map, validate_turn_map_admission
    from mlflow_import.mlflow.trace_envelope import admit_trace

    original = values("pi")
    if with_tool:
        sequence = original[:-1]
        parent = "r"
    else:
        sequence = original[:2]
        parent = "u"
    sequence.append(
        {
            "type": operation,
            "id": "accounting",
            "parentId": parent,
            "usage": {"input": 1, "output": 1},
            "summary": "accounting-only summary",
        }
    )
    sequence.append({**original[-1], "parentId": "accounting"})
    monkeypatch.setitem(globals(), "values", lambda _: sequence)
    with frozen(tmp_path, "pi", RedactionMode.VERBATIM) as turn:
        context = MappingContext(
            "0.1.0",
            MAPPING_VERSION,
            ARTIFACT_SCHEMA_VERSION,
            turn.session_header.adapter_generation,
            RedactionMode.VERBATIM,
            "local-only",
        )
        request = admit_trace(map_turn_to_trace(turn, context).to_create_request()).request
        children = list(request.child_specs())
        unavailable = next(s for s in children if s.attributes.get("mlflow_import.source_id") == "accounting")
        final = next(s for s in children if s.attributes.get("mlflow_import.source_id") == "z")
        assert unavailable.inputs is None
        assert unavailable.attributes[INPUT_CONTEXT_ATTRIBUTE]["delta"] == []
        if with_tool:
            assert {"role": "tool", "tool_call_id": "c", "content": "actual result"} in final.inputs["messages"]
            assert final.inputs["messages"][0]["tool_calls"][0]["id"] == "c"
        else:
            assert final.inputs["messages"] == [{"role": "user", "content": "prompt"}]
        assert (
            final.attributes[INPUT_CONTEXT_ATTRIBUTE]["previous_context_id"]
            == (unavailable.attributes[INPUT_CONTEXT_ATTRIBUTE]["context_id"])
        )
        validate_turn_map_admission(turn, request, RedactionMode.VERBATIM)
        value = json.loads(b"".join(iter_canonical_json(_prospective_turn_map(turn, request, RedactionMode.VERBATIM))))
        assert validate_input_contexts(value, request_ownership(request))
