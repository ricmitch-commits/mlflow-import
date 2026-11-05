import json

from mlflow_import.source.file_snapshot import create_file_snapshot
from mlflow_import.source.replay import NormalizedSnapshotAdapter
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
                {"sessionId": "claude-1", "uuid": "u1", "type": "user", "timestamp": "2025-01-01T00:00:00Z"},
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
        records = list(adapter.iter_records("claude-1"))
        assert [record.source_id for record in records] == ["u1", "a1"]


def test_codex_rollout_detection_and_session_filter(tmp_path):
    source = tmp_path / "rollout-2025-01-01.jsonl"
    source.write_text(json.dumps({"session_id": "codex-1", "id": "m1", "type": "response_item"}) + "\n")
    assert detect_codex(source)
    with create_file_snapshot(source, tmp_path, codex_exporter(source), _identity, session_id="other") as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        assert list(adapter.iter_session_headers()) == []
