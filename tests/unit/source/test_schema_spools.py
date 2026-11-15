"""Independent synthetic source contract and disk-locator regressions."""

import gzip
import json

import pytest

from mlflow_import.config import parse_config
from mlflow_import.errors import SourceCompatibilityError
from mlflow_import.source.file_snapshot import create_file_snapshot
from mlflow_import.source.model import source_namespace
from mlflow_import.source.replay import NormalizedSnapshotAdapter
from mlflow_import.source.selection import SelectionEvidenceEvaluator, SelectionFilters
from mlflow_import.source.turns import build_replay_turn, indexed_turn_summaries
from mlflow_import.sources import detect_harness, get_adapter
from mlflow_import.sources.jsonl_common import JsonlExporter

pytestmark = pytest.mark.unit


def write(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(value) + "\n" for value in values)
    if path.name.endswith(".gz"):
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)
    return path


def freeze(path, tmp_path, harness, **kwargs):
    return create_file_snapshot(path, tmp_path, JsonlExporter(path, harness), lambda value, _: value, **kwargs)


def test_pi_persisted_last_entry_head_not_timestamp_or_last_user(tmp_path):
    path = write(
        tmp_path / "sessions" / "p.jsonl",
        [
            {"type": "session", "version": 3, "id": "p", "cwd": "/synthetic", "timestamp": "2026-01-01T00:00:00Z"},
            {"type": "message", "id": "u", "parentId": None, "message": {"role": "user", "content": "first"}},
            {
                "type": "message",
                "id": "old",
                "parentId": "u",
                "message": {"role": "assistant", "model": "old", "provider": "synthetic"},
            },
            {
                "type": "message",
                "id": "other-user",
                "parentId": "old",
                "message": {"role": "user", "content": "inactive"},
            },
            {"type": "branch_summary", "id": "branch", "parentId": "u", "fromId": "other-user", "summary": "preserved"},
            {
                "type": "message",
                "id": "answer",
                "parentId": "branch",
                "message": {"role": "assistant", "content": "chosen"},
            },
        ],
    )
    assert detect_harness(path) == "pi"
    with freeze(path, tmp_path, "pi") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        assert [r.source_id for r in adapter.iter_records("p")] == ["u", "branch", "answer"]
        (summary,) = indexed_turn_summaries(snapshot, "p")
        assert summary.first_user_source_id == "u"
        assert summary.root_result_source_id == "answer"
        assert adapter.session_header("p").version is None
        assert adapter.session_header("p").metadata["session_format_version"] == 3
        replay = build_replay_turn(snapshot, adapter.session_header("p"), summary)
        assert "other-user" in {r.source_id for r in replay.iter_all_records()}
        assert snapshot.snapshot_path.with_suffix(".payloads").stat().st_mode & 0o777 == 0o600
        assert all(
            row[0].startswith("@spool:")
            for row in snapshot.connection.execute("SELECT canonical_record FROM snapshot_records")
        )
        assert snapshot.turn_index.execute("SELECT count(*) FROM turn_records").fetchone()[0] == summary.record_count


@pytest.mark.parametrize("parent", ["absent", "u"])
def test_pi_rejects_missing_parent_and_cycles(tmp_path, parent):
    path = write(
        tmp_path / "pi.jsonl",
        [
            {"type": "session", "version": 3, "id": "p"},
            {"type": "message", "id": "u", "parentId": parent, "message": {"role": "user"}},
        ],
    )
    with pytest.raises(SourceCompatibilityError, match="ancestry"):
        with freeze(path, tmp_path, "pi"):
            pass


def test_gemini_replacement_patch_rewind_and_all_occurrences(tmp_path):
    path = write(
        tmp_path / "arbitrary-name.jsonl.gz",
        [
            {"sessionId": "g", "projectHash": "synthetic", "startTime": "2026-01-01T00:00:00Z"},
            {"id": "u", "type": "user", "content": "hello"},
            {"id": "a", "type": "gemini", "content": "draft", "model": "obsolete", "tokens": {"input": 0}},
            {
                "id": "a",
                "type": "gemini",
                "content": "updated",
                "model": "current",
                "toolCalls": [{"id": "c", "name": "read"}],
            },
            {"$patch": {"id": "a", "content": "patched", "toolCalls": [{"id": "c", "result": "complete"}]}},
            {"id": "discard", "type": "user", "content": "rewound"},
            {"$rewindTo": "discard"},
            {"$set": {"summary": "complete"}},
        ],
    )
    assert detect_harness(path) == "gemini"
    with freeze(path, tmp_path, "gemini") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        records = list(adapter.iter_records("g"))
        answer = next(r for r in records if r.source_id == "a")
        assert answer.decoded_value["content"] == "patched"
        assert answer.decoded_value["toolCalls"][0]["result"] == "complete"
        assert adapter.session_header("g").title == "complete"
        assert adapter.session_header("g").project_id == "synthetic"
        assert "discard" not in [r.source_id for r in records]
        (summary,) = indexed_turn_summaries(snapshot, "g")
        replay = build_replay_turn(snapshot, adapter.session_header("g"), summary)
        assert sum(r.source_id == "a" for r in replay.iter_all_records()) == 3


def test_codex_real_envelopes_and_duplicate_event_prompt(tmp_path):
    path = write(
        tmp_path / "not-a-rollout-name.jsonl",
        [
            {
                "type": "session_meta",
                "payload": {
                    "id": "c",
                    "cwd": "/synthetic",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "git": {"commit_hash": "123"},
                },
            },
            {"type": "event_msg", "payload": {"type": "user_message", "message": "duplicate"}},
            {
                "type": "response_item",
                "payload": {
                    "id": "u",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "id": "a",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "function_call", "call_id": "call", "name": "shell", "arguments": "{}"},
            },
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "call", "output": "done"}},
        ],
    )
    assert detect_harness(path) == "codex"
    with freeze(path, tmp_path, "codex") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        (summary,) = indexed_turn_summaries(snapshot, "c")
        assert summary.first_user_source_id == "u"
        assert summary.root_result_source_id == "a"
        assert adapter.session_header("c").raw_columns["git"] == {"commit_hash": "123"}


def test_header_filter_never_decodes_unrelated_body_or_sidecars(tmp_path):
    root = tmp_path / "projects"
    selected = write(
        root / "demo" / "selected.jsonl",
        [
            {"sessionId": "selected", "type": "user", "uuid": "u", "cwd": "/synthetic"},
        ],
    )
    excluded = write(root / "demo" / "other.jsonl", [{"sessionId": "other", "type": "user", "uuid": "x"}])
    with excluded.open("a") as handle:
        handle.write("malformed unrelated body MUST NOT BE DECODED\n")
    with freeze(root, tmp_path, "claude", session_id="selected") as snapshot:
        assert [h.session_id for h in NormalizedSnapshotAdapter(snapshot).iter_session_headers()] == ["selected"]
    assert source_namespace(selected, "claude") == source_namespace(root, "claude")


def test_claude_subagent_and_mandatory_exclusions_before_spooling(tmp_path):
    main = write(
        tmp_path / "projects" / "demo" / "main.jsonl",
        [
            {
                "sessionId": "main",
                "uuid": "u",
                "type": "user",
                "message": {"role": "user", "content": "prompt"},
                "credentials": {"api_key": "NEVER-STAGED"},
            },
            {
                "sessionId": "main",
                "uuid": "a",
                "parentUuid": "u",
                "type": "assistant",
                "message": {"id": "invocation", "role": "assistant", "model": "synthetic", "content": []},
            },
        ],
    )
    write(
        main.parent / "main" / "subagents" / "agent-child.jsonl",
        [
            {
                "sessionId": "main",
                "uuid": "su",
                "parentUuid": "a",
                "type": "user",
                "message": {"role": "user", "content": "delegated"},
            },
            {
                "sessionId": "main",
                "uuid": "sa",
                "parentUuid": "su",
                "type": "assistant",
                "message": {"role": "assistant", "content": "done"},
            },
        ],
    )
    with freeze(main, tmp_path, "claude") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        assert len(list(indexed_turn_summaries(snapshot, "main"))) == 1
        assert any(r.raw_columns.get("subagent") for r in adapter.iter_records("main"))
        assert b"NEVER-STAGED" not in snapshot.snapshot_path.with_suffix(".payloads").read_bytes()
        user = next(r for r in adapter.iter_records("main") if r.source_id == "u")
        assert user.raw_json is None


def test_model_selection_sees_replay_superseded_observation(tmp_path):
    path = write(
        tmp_path / "g.jsonl",
        [
            {"sessionId": "g", "projectHash": "synthetic"},
            {"id": "u", "type": "user", "content": "hello"},
            {"id": "a", "type": "gemini", "model": "old"},
            {"id": "a", "type": "gemini", "model": "new"},
        ],
    )
    evaluator = SelectionEvidenceEvaluator(SelectionFilters(model="/old"))
    with freeze(path, tmp_path, "gemini", selection_evaluator=evaluator) as snapshot:
        assert NormalizedSnapshotAdapter(snapshot).selection_evidence("g").model_match


def test_default_roots_do_not_require_single_session_file(monkeypatch, tmp_path):
    for name, env in [
        ("claude", "CLAUDE_CONFIG_DIR"),
        ("codex", "CODEX_HOME"),
        ("pi", "PI_CODING_AGENT_DIR"),
        ("gemini", "GEMINI_CLI_HOME"),
    ]:
        monkeypatch.setenv(env, str(tmp_path / name))
        config = parse_config(["--experiment-name", "synthetic", "--harness", name])
        assert config.db_path == get_adapter(name).default_locations()[0]


def test_all_default_roots_freeze_before_any_target_work(monkeypatch, tmp_path, import_dependencies):
    from mlflow_import.importer import run_import
    from mlflow_import.sources import Adapter

    claude = write(
        tmp_path / "claude" / "projects" / "main.jsonl",
        [
            {"sessionId": "same-native", "type": "user", "uuid": "u"},
        ],
    )
    pi = write(
        tmp_path / "pi" / "sessions" / "main.jsonl",
        [
            {"type": "session", "version": 3, "id": "same-native"},
            {"type": "message", "id": "u", "parentId": None, "message": {"role": "user"}},
        ],
    )
    roots = {"claude": claude.parent, "pi": pi.parent}
    monkeypatch.setattr(
        Adapter, "default_locations", lambda self, session_id=None: (roots.get(self.name, tmp_path / "absent"),)
    )
    result = run_import(parse_config(["--experiment-name", "synthetic", "--dry-run"]), import_dependencies)
    assert result.error is None, result.report
    assert result.report["source"]["sessions_scanned"] == 2
    assert {s["harness"] for s in result.report["sessions"]} == {"claude", "pi"}
    assert len({s["source_namespace"] for s in result.report["sessions"]}) == 2


def test_default_root_results_follow_global_session_creation_order(monkeypatch, tmp_path, import_dependencies):
    from mlflow_import.importer import run_import
    from mlflow_import.sources import Adapter

    claude = write(
        tmp_path / "claude" / "late.jsonl",
        [{"sessionId": "late", "type": "user", "uuid": "u", "timestamp": "2026-02-01T00:00:00Z"}],
    )
    pi = write(
        tmp_path / "pi" / "early.jsonl",
        [
            {"type": "session", "version": 3, "id": "early", "timestamp": "2026-01-01T00:00:00Z"},
            {"type": "message", "id": "u", "parentId": None, "message": {"role": "user"}},
        ],
    )
    roots = {"claude": claude.parent, "pi": pi.parent}
    monkeypatch.setattr(
        Adapter, "default_locations", lambda self, session_id=None: (roots.get(self.name, tmp_path / "absent"),)
    )
    result = run_import(parse_config(["--experiment-name", "synthetic", "--dry-run"]), import_dependencies)
    assert result.error is None, result.report
    assert [session["session_id"] for session in result.report["sessions"]] == ["early", "late"]


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("timestamp", ["2026-01-01T00:00:00Z", None], ids=["equal-time", "missing-time"])
def test_default_root_ties_order_native_id_before_harness_and_namespace(
    monkeypatch, tmp_path, import_dependencies, dry_run, timestamp
):
    from contextlib import closing
    from dataclasses import replace

    from mlflow_import.importer import run_import
    from mlflow_import.sources import Adapter
    from tests.support.fake_gateway import FakeMlflowGateway

    roots = {"claude": (tmp_path / "claude",), "pi": (tmp_path / "pi-b", tmp_path / "pi-a")}
    identities = []
    for harness, paths in roots.items():
        for root in paths:
            native_ids = ["shared", "z-native"] if harness == "claude" else ["shared", "a-native"]
            for native_id in native_ids:
                time = {"timestamp": timestamp} if timestamp else {}
                if harness == "claude":
                    records = [
                        {
                            "sessionId": native_id,
                            "type": "user",
                            "uuid": "u",
                            "message": {"role": "user", "content": "synthetic prompt"},
                            **time,
                        },
                        {
                            "sessionId": native_id,
                            "type": "assistant",
                            "uuid": "a",
                            "parentUuid": "u",
                            "message": {"role": "assistant", "content": "synthetic reply", "stop_reason": "end_turn"},
                        },
                    ]
                else:
                    records = [
                        {"type": "session", "version": 3, "id": native_id, **time},
                        {
                            "type": "message",
                            "id": "u",
                            "parentId": None,
                            "message": {"role": "user", "content": "synthetic prompt"},
                        },
                        {
                            "type": "message",
                            "id": "a",
                            "parentId": "u",
                            "message": {"role": "assistant", "content": "synthetic reply", "stopReason": "stop"},
                        },
                    ]
                write(root / f"{native_id}.jsonl", records)
                identities.append((harness, source_namespace(root, harness), native_id))
    monkeypatch.setattr(
        Adapter, "default_locations", lambda self, session_id=None: roots.get(self.name, (tmp_path / "absent",))
    )
    created = []
    original_create = import_dependencies.gateway.create_trace

    def create(spec):
        created.append(tuple(json.loads(spec.root_tags["mlflow_import.qualified_session_id"])[1:]))
        return original_create(spec)

    monkeypatch.setattr(import_dependencies.gateway, "create_trace", create)
    dependencies = replace(
        import_dependencies,
        gateway_factory=lambda preview: FakeMlflowGateway(dry_run=True) if preview else import_dependencies.gateway,
    )
    expected = sorted(identities, key=lambda identity: (identity[2], identity[0], identity[1]))
    config = parse_config(["--experiment-name", "synthetic", *(["--dry-run"] if dry_run else [])])
    with closing(run_import(config, dependencies)) as result:
        assert result.error is None, result.report
        assert result.failed_items == 0
        actual = [(s["harness"], s["source_namespace"], s["session_id"]) for s in result.report["sessions"]]
        assert actual == expected
        assert created == ([] if dry_run else expected)
        if dry_run:
            assert import_dependencies.gateway.mutations == []


def test_stability_digest_detects_same_stat_edit_and_bounded_retry(tmp_path, monkeypatch):
    from mlflow_import.sources import jsonl_common

    path = write(
        tmp_path / "stable.jsonl",
        [
            {"sessionId": "s", "uuid": "u", "type": "user", "content": "AAAA"},
        ],
    )
    original_lines = jsonl_common.lines
    calls = 0

    def edit_during_copy(source, digest=None):
        nonlocal calls
        calls += 1
        yield from original_lines(source, digest)
        if calls == 1:
            import os

            stat = path.stat()
            path.write_text(path.read_text().replace("AAAA", "BBBB"))
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    monkeypatch.setattr(jsonl_common, "lines", edit_during_copy)
    with freeze(path, tmp_path, "claude") as snapshot:
        record = next(NormalizedSnapshotAdapter(snapshot).iter_records("s"))
        assert record.decoded_value["content"] == "BBBB"
    assert calls == 2
    assert not list(tmp_path.glob("mlflow-import-file-*"))


def test_consumer_error_does_not_refreeze_or_mask_error(tmp_path):
    path = write(tmp_path / "main.jsonl", [{"sessionId": "s", "uuid": "u", "type": "user"}])
    with pytest.raises(RuntimeError, match="consumer"):
        with freeze(path, tmp_path, "claude"):
            raise RuntimeError("consumer")


def test_claude_external_tool_result_is_associated_and_primary_output_stays_primary(tmp_path):
    path = write(
        tmp_path / "main.jsonl",
        [
            {"sessionId": "s", "uuid": "u", "type": "user"},
            {
                "sessionId": "s",
                "uuid": "a",
                "parentUuid": "u",
                "type": "assistant",
                "message": {
                    "id": "inv",
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tool-call", "name": "read"}],
                },
            },
        ],
    )
    side = path.parent / "main" / "tool-results" / "tool-call.txt"
    side.parent.mkdir(parents=True)
    side.write_text("synthetic complete tool output")
    with freeze(path, tmp_path, "claude") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        (summary,) = indexed_turn_summaries(snapshot, "s")
        assert summary.root_result_source_id == "a"
        tool = next(r for r in adapter.iter_records("s") if r.source_type == "tool_result_sidecar")
        assert tool.call_id == "tool-call"
        assert tool.decoded_value["content"] == "synthetic complete tool output"


def test_content_filter_precedes_relative_freezing(tmp_path):
    root = tmp_path / "projects"
    parent = write(root / "parent.jsonl", [{"sessionId": "parent", "uuid": "pu", "type": "user", "cwd": "/other"}])
    with parent.open("a") as handle:
        handle.write("UNRELATED INVALID TRANSCRIPT\n")
    write(
        root / "child.jsonl",
        [
            {"sessionId": "child", "parentSessionId": "parent", "uuid": "u", "type": "user", "cwd": "/selected"},
            {
                "sessionId": "child",
                "uuid": "a",
                "parentUuid": "u",
                "type": "assistant",
                "message": {"role": "assistant", "model": "wrong"},
            },
        ],
    )
    from pathlib import Path

    filters = SelectionFilters(directory=Path("/selected"), model="/wanted")
    with freeze(root, tmp_path, "claude", selection_evaluator=SelectionEvidenceEvaluator(filters)) as snapshot:
        assert [h.session_id for h in NormalizedSnapshotAdapter(snapshot).iter_session_headers()] == ["child"]


def test_inline_sidechain_preserves_primary_head_and_invocation_fragments(tmp_path):
    path = write(
        tmp_path / "main.jsonl",
        [
            {"sessionId": "s", "uuid": "u", "type": "user"},
            {
                "sessionId": "s",
                "uuid": "a1",
                "parentUuid": "u",
                "type": "assistant",
                "message": {
                    "id": "inv",
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "call1"}, {"type": "tool_use", "id": "call2"}],
                },
            },
            {
                "sessionId": "s",
                "uuid": "a2",
                "parentUuid": "a1",
                "type": "assistant",
                "message": {"id": "inv", "role": "assistant", "content": "primary"},
            },
            {
                "sessionId": "s",
                "uuid": "su",
                "parentUuid": "a1",
                "isSidechain": True,
                "agentId": "child",
                "type": "user",
            },
            {
                "sessionId": "s",
                "uuid": "sa",
                "parentUuid": "su",
                "isSidechain": True,
                "agentId": "child",
                "type": "assistant",
            },
        ],
    )
    with freeze(path, tmp_path, "claude") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        (summary,) = indexed_turn_summaries(snapshot, "s")
        assert summary.root_result_source_id == "a2"
        assistants = [r for r in adapter.iter_records("s") if r.source_id in {"a1", "a2"}]
        assert [r.invocation_id for r in assistants] == ["inv", "inv"]
        assert {row[0] for row in snapshot.connection.execute("SELECT call_id FROM calls")} == {"call1", "call2"}


def test_gemini_subagent_rewind_cannot_remove_primary_entities(tmp_path):
    main = write(
        tmp_path / "chats" / "session-main.jsonl",
        [
            {"sessionId": "main", "projectHash": "synthetic"},
            {"id": "u", "type": "user", "content": "primary"},
            {"id": "a", "type": "gemini", "content": "primary answer"},
        ],
    )
    write(
        main.parent / "main" / "child.jsonl",
        [
            {"sessionId": "child", "projectHash": "synthetic", "kind": "subagent"},
            {"id": "su", "type": "user", "content": "delegated"},
            {"id": "sa", "type": "gemini", "content": "nested"},
            {"$rewindTo": "missing"},
        ],
    )
    with freeze(main.parent, tmp_path, "gemini") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        assert [h.session_id for h in adapter.iter_session_headers()] == ["main"]
        (summary,) = indexed_turn_summaries(snapshot, "main")
        assert summary.root_result_source_id == "a"
        assert {r.source_id for r in adapter.iter_records("main")} == {"u", "a"}


def test_unsupported_pi_and_codex_replay_variants_fail_explicitly(tmp_path):
    pi = write(tmp_path / "old-pi.jsonl", [{"type": "session", "version": 2, "id": "p"}])
    with pytest.raises(SourceCompatibilityError, match="version 3"):
        with freeze(pi, tmp_path, "pi"):
            pass
    codex = write(
        tmp_path / "codex.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "c"}},
            {"type": "event_msg", "payload": {"type": "thread_rolled_back", "num_turns": 1}},
        ],
    )
    with pytest.raises(SourceCompatibilityError, match="rollback"):
        with freeze(codex, tmp_path, "codex"):
            pass


def test_sqlite_content_filter_does_not_freeze_nonrequired_parent(tmp_path, import_dependencies):
    import sqlite3
    from pathlib import Path

    from mlflow_import.config import ImportConfig
    from mlflow_import.importer import run_import
    from tests.support.import_fakes import build_current_source_db

    path = build_current_source_db(
        tmp_path,
        sessions=[("parent", 1000), ("child", 2000)],
        messages=[
            ("pu", "parent", "user", 1, "{}"),
            ("u", "child", "user", 1, "{}"),
            ("a", "child", "assistant", 2, '{"model":"wrong"}'),
        ],
    )
    with sqlite3.connect(path) as db:
        db.execute("UPDATE session_v2 SET directory='/other'")
        db.execute("UPDATE session_v2 SET directory='/selected',parent_id='parent' WHERE id='child'")
    result = run_import(
        ImportConfig("synthetic", path, model="/wanted", directory=Path("/selected"), dry_run=True), import_dependencies
    )
    assert result.error is None, result.report
    assert result.report["source"]["sessions_scanned"] == 1
    assert [s["session_id"] for s in result.report["sessions"]] == ["child"]


def test_submicrosecond_creation_bound_is_not_rounded_into_range(tmp_path):
    config = parse_config(["--experiment-name", "synthetic", "--to-time", "2026-11-13"])
    path = write(
        tmp_path / "main.jsonl",
        [
            {"sessionId": "s", "uuid": "u", "type": "user", "timestamp": "2026-11-13T23:59:59.000000001Z"},
        ],
    )
    filters = SelectionFilters(to_time=config.to_time)
    with freeze(path, tmp_path, "claude", selection_evaluator=SelectionEvidenceEvaluator(filters)) as snapshot:
        assert list(NormalizedSnapshotAdapter(snapshot).iter_session_headers()) == []


def test_later_unsupported_default_source_prevents_all_writes(monkeypatch, tmp_path, import_dependencies):
    from mlflow_import.importer import run_import
    from mlflow_import.sources import Adapter

    claude = write(tmp_path / "claude" / "main.jsonl", [{"sessionId": "s", "uuid": "u", "type": "user"}])
    pi = write(tmp_path / "pi" / "old.jsonl", [{"type": "session", "version": 2, "id": "old"}])
    roots = {"claude": claude.parent, "pi": pi.parent}
    monkeypatch.setattr(
        Adapter, "default_locations", lambda self, session_id=None: (roots.get(self.name, tmp_path / "absent"),)
    )
    result = run_import(parse_config(["--experiment-name", "synthetic"]), import_dependencies)
    assert result.error.code == "unsupported_schema"
    assert import_dependencies.gateway.mutations == []


def test_numeric_turn_offsets_resolve_exact_spool_bytes(tmp_path):
    path = write(
        tmp_path / "p.jsonl",
        [
            {"type": "session", "version": 3, "id": "p"},
            {"type": "message", "id": "u", "parentId": None, "message": {"role": "user", "content": "synthetic"}},
            {"type": "message", "id": "a", "parentId": "u", "message": {"role": "assistant", "content": "complete"}},
        ],
    )
    with freeze(path, tmp_path, "pi") as snapshot:
        (summary,) = indexed_turn_summaries(snapshot, "p")
        rows = snapshot.turn_index.execute("SELECT payload_offset,payload_length FROM turn_records ORDER BY ordinal")
        with snapshot.snapshot_path.with_suffix(".payloads").open("rb") as spool:
            source_ids = []
            for offset, length in rows:
                spool.seek(offset)
                source_ids.append(json.loads(spool.read(length))["record"]["source_id"])
        assert source_ids == ["u", "a"]
        assert summary.record_count == 2
        assert (
            snapshot.connection.execute("SELECT MAX(length(canonical_record)) FROM snapshot_records").fetchone()[0] < 80
        )


def test_many_entry_tree_has_bounded_payload_heap(tmp_path):
    import tracemalloc

    peaks = []
    for count in (150, 300):
        path = tmp_path / f"tree-{count}.jsonl"
        with path.open("w") as handle:
            handle.write(json.dumps({"type": "session", "version": 3, "id": "p"}) + "\n")
            for index in range(count):
                handle.write(
                    json.dumps(
                        {
                            "type": "message",
                            "id": f"m{index}",
                            "parentId": f"m{index - 1}" if index else None,
                            "message": {"role": "user" if index % 2 == 0 else "assistant", "content": "s" * 16384},
                        }
                    )
                    + "\n"
                )
        tracemalloc.start()
        with freeze(path, tmp_path, "pi") as snapshot:
            assert sum(1 for _ in indexed_turn_summaries(snapshot, "p")) == count // 2
            peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    assert max(peaks) < 8 * 1024**2
    assert peaks[1] < peaks[0] + 2 * 1024**2


def test_sidecar_agent_id_joins_the_invoking_turn_not_the_last_turn(tmp_path):
    main = write(
        tmp_path / "main.jsonl",
        [
            {"sessionId": "s", "uuid": "u1", "type": "user"},
            {
                "sessionId": "s",
                "uuid": "a1",
                "parentUuid": "u1",
                "type": "assistant",
                "message": {
                    "id": "inv",
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "call", "name": "Task"}],
                },
            },
            {
                "sessionId": "s",
                "uuid": "result",
                "parentUuid": "a1",
                "type": "user",
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call"}]},
                "toolUseResult": {"agentId": "child"},
            },
            {"sessionId": "s", "uuid": "a2", "parentUuid": "result", "type": "assistant"},
            {"sessionId": "s", "uuid": "u2", "parentUuid": "a2", "type": "user"},
            {"sessionId": "s", "uuid": "a3", "parentUuid": "u2", "type": "assistant"},
        ],
    )
    write(
        main.parent / "main" / "subagents" / "agent-child.jsonl",
        [
            {"sessionId": "s", "uuid": "su", "type": "user"},
            {"sessionId": "s", "uuid": "sa", "parentUuid": "su", "type": "assistant"},
        ],
    )
    with freeze(main, tmp_path, "claude") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        first, second = indexed_turn_summaries(snapshot, "s")
        first_records = list(build_replay_turn(snapshot, adapter.session_header("s"), first).iter_records())
        second_records = list(build_replay_turn(snapshot, adapter.session_header("s"), second).iter_records())
        assert any(r.raw_columns.get("subagent") for r in first_records)
        assert not any(r.raw_columns.get("subagent") for r in second_records)
        assert first.root_result_source_id == "a2"
        assert second.root_result_source_id == "a3"


def test_unattributed_sidecar_remains_in_original_inventory_without_invented_turn(tmp_path):
    main = write(tmp_path / "main.jsonl", [{"sessionId": "s", "uuid": "u", "type": "user"}])
    write(
        main.parent / "main" / "subagents" / "agent-unknown.jsonl",
        [
            {"sessionId": "s", "uuid": "su", "type": "user"},
        ],
    )
    with freeze(main, tmp_path, "claude") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        (summary,) = indexed_turn_summaries(snapshot, "s")
        replay = build_replay_turn(snapshot, adapter.session_header("s"), summary)
        assert len(list(replay.iter_records())) == 1
        assert len(list(replay.iter_original_records())) == 2
        assert "unattributed_subagent" in snapshot.export_warnings


def test_mixed_header_skips_prompt_and_credential_values_in_bounded_memory(tmp_path, monkeypatch):
    import tracemalloc

    from mlflow_import.source import file_headers
    from mlflow_import.sources.jsonl_common import schema_header

    path = write(
        tmp_path / "main.jsonl",
        [
            {
                "message": {"role": "user", "content": "DO-NOT-DECODE-" + "x" * 2_000_000},
                "credentials": {"api_key": "DO-NOT-DECODE-CREDENTIAL"},
                "sessionId": "s",
                "type": "user",
                "cwd": "/synthetic",
            }
        ],
    )
    original = file_headers.json.loads

    def header_only(text, *args, **kwargs):
        assert "DO-NOT-DECODE" not in text
        return original(text, *args, **kwargs)

    monkeypatch.setattr(file_headers.json, "loads", header_only)
    tracemalloc.start()
    header = schema_header(path, "claude")
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert header["session_id"] == "s"
    assert "message" not in header and "credentials" not in header
    assert peak < 256 * 1024


def test_codex_root_discovers_active_and_gzip_archives_without_global_history(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    write(
        tmp_path / "sessions" / "2026" / "active.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "active"}},
            {"type": "response_item", "payload": {"id": "u", "type": "message", "role": "user"}},
        ],
    )
    archive = write(
        tmp_path / "archived_sessions" / "archive.jsonl.gz",
        [
            {"type": "session_meta", "payload": {"id": "archive"}},
        ],
    )
    (tmp_path / "history.jsonl").write_text("UNRELATED GLOBAL HISTORY MUST NOT BE DECODED\n")
    with freeze(tmp_path, tmp_path, "codex") as snapshot:
        assert {h.session_id for h in NormalizedSnapshotAdapter(snapshot).iter_session_headers()} == {
            "active",
            "archive",
        }
    assert source_namespace(archive, "codex") == source_namespace(tmp_path, "codex")


def test_gemini_default_roots_are_home_relative_and_skip_runtime_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_CLI_HOME", str(tmp_path))
    root = tmp_path / ".gemini" / "tmp"
    write(root / "project" / "chats" / "session.jsonl", [{"sessionId": "g", "projectHash": "synthetic"}])
    bad = root / "project" / "logs" / "private.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_text("UNRELATED RUNTIME LOG MUST NOT BE DECODED\n")
    assert get_adapter("gemini").default_locations() == (root, tmp_path / ".cache" / ".gemini" / "tmp")
    with freeze(root, tmp_path, "gemini") as snapshot:
        assert [h.session_id for h in NormalizedSnapshotAdapter(snapshot).iter_session_headers()] == ["g"]


def test_pi_custom_session_root_has_same_explicit_file_namespace(tmp_path, monkeypatch):
    root = tmp_path / "custom-session-store"
    path = write(root / "workspace" / "p.jsonl", [{"type": "session", "version": 3, "id": "p"}])
    monkeypatch.setenv("PI_CODING_AGENT_SESSION_DIR", str(root))
    assert get_adapter("pi").default_locations() == (root,)
    assert source_namespace(path, "pi") == source_namespace(root, "pi")


def test_gemini_patch_provenance_and_turn_hash_do_not_depend_on_other_sessions(tmp_path):
    root = tmp_path / "chats"
    write(root / "a.jsonl", [{"sessionId": "a", "projectHash": "synthetic", "startTime": "2026-01-01T00:00:00Z"}])
    path = write(
        root / "g.jsonl",
        [
            {"sessionId": "g", "projectHash": "synthetic", "startTime": "2026-01-02T00:00:00Z"},
            {"id": "u", "type": "user", "content": "prompt"},
            {"id": "a", "type": "gemini", "content": "before"},
            {"$patch": {"id": "a", "content": "after"}},
        ],
    )
    fingerprints = []
    for selected in (root, path):
        with freeze(selected, tmp_path, "gemini") as snapshot:
            (summary,) = indexed_turn_summaries(snapshot, "g")
            fingerprints.append(summary.source_fingerprint)
    assert fingerprints[0] == fingerprints[1]


def test_copy_digest_detects_transient_content_restored_before_final_read(tmp_path, monkeypatch):
    import os

    from mlflow_import.sources import jsonl_common

    path = write(tmp_path / "main.jsonl", [{"sessionId": "s", "uuid": "u", "type": "user", "content": "AAAA"}])
    original_text = path.read_text()
    original_stat = path.stat()
    original_lines = jsonl_common.lines
    copies = 0

    def transient_copy(source, digest=None):
        nonlocal copies
        copies += 1
        if copies == 1:
            path.write_text(original_text.replace("AAAA", "BBBB"))
            os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            try:
                yield from original_lines(source, digest)
            finally:
                path.write_text(original_text)
                os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        else:
            yield from original_lines(source, digest)

    monkeypatch.setattr(jsonl_common, "lines", transient_copy)
    with freeze(path, tmp_path, "claude") as snapshot:
        record = next(NormalizedSnapshotAdapter(snapshot).iter_records("s"))
        assert record.decoded_value["content"] == "AAAA"
    assert copies == 2


def test_source_sidecar_lookup_rejects_parent_directory_identity(tmp_path):
    path = write(tmp_path / "chats" / "main.jsonl", [{"sessionId": "..", "projectHash": "synthetic"}])
    with pytest.raises(SourceCompatibilityError, match="unsafe sidecar"):
        with freeze(path, tmp_path, "gemini"):
            pass


def test_nanosecond_cli_bounds_survive_pruning_and_final_selection(tmp_path):
    from mlflow_import.source.discovery import discover_sessions

    exact = "2026-11-13T23:59:59.000000001Z"
    config = parse_config(["--experiment-name", "synthetic", "--from-time", exact, "--to-time", exact])
    filters = SelectionFilters(
        from_time=config.from_time,
        to_time=config.to_time,
        from_time_ns=config.from_time_ns,
        to_time_ns=config.to_time_ns,
    )
    root = tmp_path / "projects"
    for native, fraction in [("before", "000000000"), ("exact", "000000001"), ("after", "000000002")]:
        write(
            root / f"{native}.jsonl",
            [{"sessionId": native, "type": "user", "uuid": "u", "timestamp": f"2026-11-13T23:59:59.{fraction}Z"}],
        )
    with freeze(root, tmp_path, "claude", selection_evaluator=SelectionEvidenceEvaluator(filters)) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        assert [h.session_id for h in adapter.iter_session_headers()] == ["exact"]
        stats = discover_sessions(adapter, filters, tmp_path / "plan.jsonl")
        assert stats.selected_session_count == 1


def test_nanosecond_offset_and_reversed_cli_bounds_are_exact():
    from mlflow_import.errors import ConfigurationError
    from mlflow_import.sources.jsonl_common import timestamp

    z = "2026-11-13T00:00:00.000000001Z"
    offset = "2026-11-13T00:00:00.000000002+00:00:00.000000001"
    assert timestamp(z).nanoseconds == timestamp(offset).nanoseconds
    first = parse_config(["--experiment-name", "synthetic", "--from-time", z])
    second = parse_config(["--experiment-name", "synthetic", "--from-time", offset])
    assert first.from_time_ns == second.from_time_ns
    with pytest.raises(ConfigurationError, match="lower time bound"):
        parse_config(
            ["--experiment-name", "synthetic", "--from-time", "2026-11-13T00:00:00.000000002Z", "--to-time", z]
        )


def test_physical_identity_distinguishes_join_replay_from_corrupt_sqlite_duplicates():
    from mlflow_import.source.normalize import IdentityTracker

    tracker = IdentityTracker("legacy")
    assert tracker.register_explicit("part", "same-logical-id", 101)
    assert not tracker.register_explicit("part", "same-logical-id", 101)
    with pytest.raises(SourceCompatibilityError) as caught:
        tracker.register_explicit("part", "same-logical-id", 102)
    assert caught.value.code == "duplicate_record_identity"


def test_default_source_cleanup_failure_is_sanitized_and_fails_the_invocation(
    tmp_path, monkeypatch, import_dependencies
):
    import shutil
    from pathlib import Path

    from mlflow_import.importer import run_import
    from mlflow_import.sources import Adapter

    source = write(tmp_path / "claude" / "s.jsonl", [{"sessionId": "s", "uuid": "u", "type": "user"}])
    monkeypatch.setattr(
        Adapter,
        "default_locations",
        lambda self, session_id=None: (source.parent if self.name == "claude" else tmp_path / "absent",),
    )
    original = shutil.rmtree

    def fail_defaults(path, *args, **kwargs):
        if Path(path).name.startswith("mlflow-import-defaults-"):
            raise OSError("PRIVATE SOURCE LOCATION MUST NOT APPEAR")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", fail_defaults)
    result = run_import(parse_config(["--experiment-name", "synthetic", "--dry-run"]), import_dependencies)
    assert result.failed_items == 1
    assert result.report["summary"]["failed_items"] == 1
    assert "PRIVATE SOURCE LOCATION" not in repr(result.report)


def test_gemini_recorded_tool_agent_id_attributes_the_nested_turn(tmp_path):
    main = write(
        tmp_path / "chats" / "main.jsonl",
        [
            {"sessionId": "main", "projectHash": "synthetic"},
            {"id": "u1", "type": "user", "content": "first"},
            {"id": "a1", "type": "gemini", "toolCalls": [{"id": "call", "name": "delegate", "agentId": "child"}]},
            {"id": "a2", "type": "gemini", "content": "first answer"},
            {"id": "u2", "type": "user", "content": "second"},
            {"id": "a3", "type": "gemini", "content": "second answer"},
        ],
    )
    write(
        main.parent / "main" / "child.jsonl",
        [
            {"sessionId": "child", "projectHash": "synthetic", "kind": "subagent"},
            {"id": "su", "type": "user", "content": "delegated"},
            {"id": "sa", "type": "gemini", "content": "nested result"},
        ],
    )
    with freeze(main, tmp_path, "gemini") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        first, second = indexed_turn_summaries(snapshot, "main")
        assert any(
            r.raw_columns.get("subagent")
            for r in build_replay_turn(snapshot, adapter.session_header("main"), first).iter_records()
        )
        assert not any(
            r.raw_columns.get("subagent")
            for r in build_replay_turn(snapshot, adapter.session_header("main"), second).iter_records()
        )
        assert first.root_result_source_id == "a2"


def test_opencode_occurrence_refs_expose_links_and_counts_without_decoding(source_db, tmp_path, monkeypatch):
    import sqlite3

    from mlflow_import.source import replay
    from mlflow_import.source.probe import probe_schema
    from mlflow_import.source.registry import build_live_exporter
    from mlflow_import.source.snapshot import create_normalized_snapshot
    from tests.support.import_fakes import UNIT_SESSION_ID

    with sqlite3.connect(source_db) as connection:
        exporter = build_live_exporter(connection, probe_schema(connection))
    with create_normalized_snapshot(source_db, tmp_path, exporter, lambda value, _: value) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)

        def forbidden(*args, **kwargs):
            raise AssertionError("occurrence metadata must not decode payloads")

        monkeypatch.setattr(replay, "_restore_source_record", forbidden)
        refs = list(adapter.occurrences(UNIT_SESSION_ID))
        assistant = next(ref for ref in refs if ref.source_id == "unit-assistant-1")
        assert assistant.invocation_id == assistant.entity_id
        assert any(ref.parent_entity_id == assistant.entity_id for ref in refs)
        assert all(ref.payload_offset is not None and ref.payload_length > 0 for ref in refs)
        physical, derived, effective = snapshot.connection.execute(
            "SELECT physical_count,derived_count,effective_count FROM source_summaries WHERE session_id=?",
            (UNIT_SESSION_ID,),
        ).fetchone()
        assert physical == sum(ref.kind == "original" for ref in refs)
        assert derived == sum(ref.kind == "derived" for ref in refs)
        assert effective == 0


def test_claude_metadata_prelude_discovers_session_without_decoding_snapshot_body(tmp_path):
    path = write(
        tmp_path / "main.jsonl",
        [
            {"type": "file-history-snapshot", "messageId": "u", "snapshot": {"unknown": "synthetic file history"}},
            {"type": "queue-operation", "operation": "dequeue"},
            {
                "type": "user",
                "sessionId": "s",
                "uuid": "u",
                "timestamp": "2026-01-01T00:00:00Z",
                "cwd": "/synthetic",
                "message": {"role": "user", "content": "prompt"},
            },
            {
                "type": "assistant",
                "sessionId": "s",
                "uuid": "a",
                "parentUuid": "u",
                "message": {"role": "assistant", "content": "response"},
            },
        ],
    )
    assert detect_harness(path) == "claude"
    with freeze(path, tmp_path, "claude") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        (summary,) = indexed_turn_summaries(snapshot, "s")
        assert summary.record_count == 2
        replay = build_replay_turn(snapshot, adapter.session_header("s"), summary)
        assert len(list(replay.iter_original_records())) == 4


def test_claude_metadata_only_session_uses_qualified_uuid_filename(tmp_path):
    native = "c2b905fb-d7ab-41c9-bffc-347a428906af"
    path = write(tmp_path / f"{native}.jsonl", [{"type": "file-history-snapshot", "messageId": "u", "snapshot": {}}])
    with freeze(path, tmp_path, "claude") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        assert adapter.session_header(native).raw_columns["identity_origin"] == "filename"
        assert list(indexed_turn_summaries(snapshot, native)) == []


def test_explicit_file_uses_known_store_headers_for_relatives_without_sibling_bodies(tmp_path):
    root = tmp_path / "projects"
    write(root / "workspace" / "parent.jsonl", [{"sessionId": "p", "uuid": "pu", "type": "user"}])
    child = write(
        root / "workspace" / "child.jsonl", [{"sessionId": "c", "parentSessionId": "p", "uuid": "cu", "type": "user"}]
    )
    sibling = write(
        root / "workspace" / "sibling.jsonl",
        [{"sessionId": "sibling", "parentSessionId": "p", "uuid": "su", "type": "user"}],
    )
    with sibling.open("a") as handle:
        handle.write("UNRELATED SIBLING TRANSCRIPT MUST NOT BE DECODED\n")
    with freeze(child, tmp_path, "claude") as snapshot:
        assert {h.session_id for h in NormalizedSnapshotAdapter(snapshot).iter_session_headers()} == {"p", "c"}


def test_generic_tmp_ancestor_does_not_merge_independent_gemini_namespaces(tmp_path):
    first = write(tmp_path / "tmp" / "profile-a" / "session.jsonl", [{"sessionId": "g", "projectHash": "synthetic"}])
    second = write(tmp_path / "tmp" / "profile-b" / "session.jsonl", [{"sessionId": "g", "projectHash": "synthetic"}])
    assert source_namespace(first, "gemini") != source_namespace(second, "gemini")


def test_gemini_duplicate_order_ids_keep_first_insertion_order(tmp_path):
    path = write(
        tmp_path / "g.jsonl",
        [
            {"sessionId": "g", "projectHash": "synthetic"},
            {"id": "u", "type": "user", "content": "prompt"},
            {"id": "a", "type": "gemini", "content": "a"},
            {"id": "b", "type": "gemini", "content": "b"},
            {"$patch": {"orderIds": ["a", "b", "a"]}},
        ],
    )
    with freeze(path, tmp_path, "gemini") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        assert [r.source_id for r in adapter.iter_records("g")] == ["u", "a", "b"]
        (summary,) = indexed_turn_summaries(snapshot, "g")
        assert summary.root_result_source_id == "b"
