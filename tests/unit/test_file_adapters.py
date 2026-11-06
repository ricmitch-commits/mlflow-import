import json

from mlflow_import.source.file_snapshot import create_file_snapshot
from mlflow_import.source.replay import NormalizedSnapshotAdapter
from mlflow_import.source.turns import iter_turn_summaries
from mlflow_import.sources.claude import detect as detect_claude
from mlflow_import.sources.claude import exporter as claude_exporter
from mlflow_import.sources.codex import detect as detect_codex
from mlflow_import.sources.codex import exporter as codex_exporter


def _identity(value: object, protected: object) -> object:
    del protected
    return value


def test_claude_jsonl_freezes_into_shared_replay_contract(tmp_path):
    source = tmp_path / "projects" / "demo" / "session.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {
                    "sessionId": "claude-1",
                    "parentSessionId": "claude-parent",
                    "uuid": "u1",
                    "type": "user",
                    "timestamp": "2025-01-01T00:00:00Z",
                },
                {"sessionId": "claude-1", "uuid": "a1", "parentUuid": "u1", "type": "assistant"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    assert detect_claude(source)
    with create_file_snapshot(source, tmp_path, claude_exporter(source), _identity) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = adapter.session_header("claude-1")
        assert header is not None
        assert header.parent_id == "claude-parent"
        records = list(adapter.iter_records("claude-1"))
        assert [record.source_id for record in records] == ["u1", "a1"]


def test_claude_message_envelopes_produce_logical_turns(tmp_path):
    source = tmp_path / "projects" / "demo" / "session.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"type": "mode", "sessionId": "claude-1", "mode": "normal"},
                {
                    "sessionId": "claude-1",
                    "uuid": "u1",
                    "type": "user",
                    "message": {"role": "user", "content": "hello"},
                    "timestamp": "2025-01-01T00:00:00Z",
                },
                {
                    "sessionId": "claude-1",
                    "uuid": "a1",
                    "parentUuid": "u1",
                    "type": "assistant",
                    "message": {"role": "assistant", "content": []},
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with create_file_snapshot(source, tmp_path, claude_exporter(source), _identity) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        (summary,) = iter_turn_summaries(adapter.iter_records("claude-1"))
        assert summary.first_user_source_id == "u1"
        assert summary.assistant_record_count == 1


def test_claude_tool_result_envelope_does_not_start_a_second_turn(tmp_path):
    source = tmp_path / "projects" / "demo" / "session.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"sessionId": "claude-1", "uuid": "u1", "type": "user", "message": {"role": "user", "content": "run"}},
                {
                    "sessionId": "claude-1",
                    "uuid": "a1",
                    "parentUuid": "u1",
                    "type": "assistant",
                    "message": {"role": "assistant", "content": []},
                },
                {
                    "sessionId": "claude-1",
                    "uuid": "tool-result",
                    "parentUuid": "a1",
                    "type": "user",
                    "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
                },
                {
                    "sessionId": "claude-1",
                    "uuid": "a2",
                    "parentUuid": "tool-result",
                    "type": "assistant",
                    "message": {"role": "assistant", "content": []},
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with create_file_snapshot(source, tmp_path, claude_exporter(source), _identity) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        summaries = list(iter_turn_summaries(adapter.iter_records("claude-1")))
        assert len(summaries) == 1
        assert summaries[0].record_count == 4


def test_codex_rollout_detection_and_session_filter(tmp_path):
    source = tmp_path / "rollout-2025-01-01.jsonl"
    source.write_text(json.dumps({"session_id": "codex-1", "id": "m1", "type": "response_item"}) + "\n")
    assert detect_codex(source)
    with create_file_snapshot(source, tmp_path, codex_exporter(source), _identity, session_id="other") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        assert list(adapter.iter_session_headers()) == []


def test_codex_jsonl_normalizes_parent_session_id(tmp_path):
    source = tmp_path / "rollout-2025-01-01.jsonl"
    source.write_text(
        json.dumps(
            {
                "session_id": "codex-1",
                "parent_session_id": "codex-parent",
                "id": "m1",
                "type": "response_item",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with create_file_snapshot(source, tmp_path, codex_exporter(source), _identity) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = adapter.session_header("codex-1")

    assert header is not None
    assert header.parent_id == "codex-parent"


def test_claude_directory_snapshot_includes_parent_and_child_sessions(tmp_path):
    source = tmp_path / "projects" / "demo"
    source.mkdir(parents=True)
    (source / "parent.jsonl").write_text(
        json.dumps({"sessionId": "claude-parent", "uuid": "p1", "type": "user"}) + "\n",
        encoding="utf-8",
    )
    (source / "child.jsonl").write_text(
        json.dumps({"sessionId": "claude-child", "parentSessionId": "claude-parent", "uuid": "c1", "type": "user"})
        + "\n",
        encoding="utf-8",
    )

    source_root = source.parent.parent
    with create_file_snapshot(source_root, tmp_path, claude_exporter(source_root), _identity) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        headers = {header.session_id: header for header in adapter.iter_session_headers()}

    assert set(headers) == {"claude-parent", "claude-child"}
    assert headers["claude-child"].parent_id == "claude-parent"
