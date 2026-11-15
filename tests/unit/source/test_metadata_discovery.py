"""Synthetic metadata-only discovery boundaries; no real session fixtures."""

import hashlib
import json
from pathlib import Path

import pytest

from mlflow_import.errors import SourceCompatibilityError
from mlflow_import.source.discovery import discover_sessions
from mlflow_import.source.replay import NormalizedSnapshotAdapter
from mlflow_import.source.selection import SelectionEvidenceEvaluator, SelectionFilters
from mlflow_import.sources.jsonl_common import schema_header
from tests.unit.source.test_schema_spools import freeze, write

pytestmark = pytest.mark.unit


def test_claude_mode_prelude_prunes_before_malformed_body(tmp_path):
    root = tmp_path / "projects"
    for native, directory in [("wanted", "/selected"), ("excluded", "/other")]:
        path = write(
            root / f"{native}.jsonl",
            [
                {"type": "mode", "sessionId": native},
                {"type": "permission-mode", "sessionId": native},
                {"type": "atis-latch", "sessionId": native},
                {"type": "worktree-state", "sessionId": native},
                {"type": "tools-used", "sessionId": native},
                {"type": "last-prompt", "sessionId": native, "prompt": "NEVER_DECODE"},
                {"type": "cost-state", "sessionId": native},
                {"type": "user", "sessionId": native, "message": {"content": "first without cwd"}},
                {"type": "user", "sessionId": native, "cwd": directory, "uuid": "u"},
            ],
        )
        if native == "excluded":
            with path.open("a") as handle:
                handle.write("MALFORMED EXCLUDED BODY\n")
    filters = SelectionFilters(directory=Path("/selected"))
    with freeze(root, tmp_path, "claude", selection_evaluator=SelectionEvidenceEvaluator(filters)) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        assert [h.session_id for h in adapter.iter_session_headers()] == ["wanted"]
        assert discover_sessions(adapter, filters, tmp_path / "plan.jsonl").selected_session_count == 1


def test_claude_fork_context_sidecar_uses_later_header_and_owner(tmp_path):
    main = write(
        tmp_path / "projects" / "main.jsonl",
        [
            {"type": "mode", "sessionId": "main"},
            {"type": "user", "sessionId": "main", "uuid": "u", "cwd": "/selected"},
        ],
    )
    side = write(
        main.parent / "main" / "subagents" / "agent-child.jsonl",
        [
            {"type": "fork-context-ref", "sourceSessionId": "main"},
            {"type": "user", "sessionId": "main", "uuid": "su", "parentUuid": "u", "cwd": "/selected"},
        ],
    )
    with freeze(main, tmp_path, "claude") as snapshot:
        assert snapshot.stats.session_count == 1
    write(side, [{"type": "fork-context-ref"}, {"type": "user", "sessionId": "wrong", "uuid": "su"}])
    with pytest.raises(SourceCompatibilityError) as caught:
        with freeze(main, tmp_path, "claude"):
            pass
    assert caught.value.code == "source_identity_changed"


def test_claude_prelude_does_not_decode_transcript_values(tmp_path, monkeypatch):
    from mlflow_import.source import file_headers

    path = write(
        tmp_path / "s.jsonl",
        [
            {"type": "mode", "sessionId": "s"},
            {"type": "user", "sessionId": "s", "cwd": "/selected", "message": {"content": "NEVER_DECODE"}},
        ],
    )
    original = json.loads

    def guarded(text, *args, **kwargs):
        assert "NEVER_DECODE" not in text
        return original(text, *args, **kwargs)

    monkeypatch.setattr(file_headers.json, "loads", guarded)
    assert schema_header(path, "claude")["cwd"] == "/selected"


@pytest.mark.parametrize("change", ["startTime", "projectHash", "kind"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_gemini_all_partial_metadata_validates_identity(tmp_path, change, wrapped):
    original = {"sessionId": "g", "projectHash": "p", "startTime": "2026-01-01T00:00:00Z"}
    update = {change: "2026-02-01T00:00:00Z" if change == "startTime" else "different"}
    path = write(tmp_path / "g.jsonl", [original, {"$set": update} if wrapped else update])
    with pytest.raises(SourceCompatibilityError) as caught:
        with freeze(path, tmp_path, "gemini"):
            pass
    assert caught.value.code in {"unsupported_header_update", "source_identity_changed"}


def gemini_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_CLI_HOME", str(tmp_path))
    root = tmp_path / ".gemini" / "tmp"
    root.mkdir(parents=True)
    (root.parent / "projects.json").write_text(json.dumps({"projects": {"/selected": "registered", "/other": "other"}}))
    return root


def test_gemini_registry_filters_before_duplicate_resolution(tmp_path, monkeypatch):
    root = gemini_registry(tmp_path, monkeypatch)
    for index, (directory, group, native) in enumerate(
        [("/selected", "registered", "wanted"), ("/other", "other", "collision"), ("/other", "other", "collision")]
    ):
        path = write(
            root / group / "chats" / f"{index}.jsonl",
            [
                {
                    "sessionId": native,
                    "projectHash": hashlib.sha256(directory.encode()).hexdigest(),
                    "startTime": f"2026-01-0{index + 1}T00:00:00Z",
                }
            ],
        )
        if directory == "/other":
            with path.open("a") as handle:
                handle.write("MALFORMED EXCLUDED BODY\n")
    filters = SelectionFilters(directory=Path("/selected"))
    with freeze(root, tmp_path, "gemini", selection_evaluator=SelectionEvidenceEvaluator(filters)) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        assert [h.session_id for h in adapter.iter_session_headers()] == ["wanted"]
        assert adapter.session_header("wanted").directory == "/selected"
        assert discover_sessions(adapter, filters, tmp_path / "plan.jsonl").selected_session_count == 1


@pytest.mark.parametrize("directory", [None, "/selected"])
def test_gemini_selected_or_unknown_duplicate_never_chooses_newest(tmp_path, directory):
    for index in range(2):
        write(
            tmp_path / "chats" / f"{index}.jsonl",
            [
                {
                    "sessionId": "collision",
                    "projectHash": "p",
                    "cwd": directory,
                    "startTime": f"2026-01-0{index + 1}T00:00:00Z",
                }
            ],
        )
    with pytest.raises(SourceCompatibilityError) as caught:
        with freeze(
            tmp_path / "chats",
            tmp_path,
            "gemini",
            selection_evaluator=SelectionEvidenceEvaluator(SelectionFilters(directory=Path("/selected"))),
        ):
            pass
    assert caught.value.code == "ambiguous_source"
    assert "gemini" in str(caught.value).lower()


def test_claude_required_relatives_bypass_directory_but_not_siblings(tmp_path):
    root = tmp_path / "projects"
    for native, parent, cwd in [
        ("parent", None, "/other"),
        ("child", "parent", "/selected"),
        ("grandchild", "child", "/other"),
        ("sibling", "parent", "/other"),
    ]:
        path = write(
            root / f"{native}.jsonl",
            [
                {"type": "mode", "sessionId": native},
                {"type": "user", "sessionId": native, "parentSessionId": parent, "cwd": cwd, "uuid": native},
            ],
        )
        if native == "sibling":
            with path.open("a") as handle:
                handle.write("MALFORMED SIBLING\n")
    filters = SelectionFilters(directory=Path("/selected"))
    with freeze(root, tmp_path, "claude", selection_evaluator=SelectionEvidenceEvaluator(filters)) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        assert {h.session_id for h in adapter.iter_session_headers()} == {"parent", "child", "grandchild"}
        assert discover_sessions(adapter, filters, tmp_path / "plan.jsonl").selected_session_count == 3


def test_path_fallback_matches_selection_evidence():
    from mlflow_import.source.model import SessionHeader

    evaluator = SelectionEvidenceEvaluator(SelectionFilters(directory=Path("/selected")))
    header = SessionHeader("s", "current", directory=None, path="/selected/child")
    assert evaluator.session_evidence(header).directory_match


def test_unsupported_selected_claude_header_is_not_silently_dropped(tmp_path):
    path = write(
        tmp_path / "projects" / "s.jsonl",
        [
            {"type": "mode", "sessionId": "s"},
            {"type": "unsupported-new-record", "sessionId": "s", "cwd": "/selected"},
        ],
    )
    with pytest.raises(SourceCompatibilityError) as caught:
        with freeze(
            path.parent,
            tmp_path,
            "claude",
            selection_evaluator=SelectionEvidenceEvaluator(SelectionFilters(directory=Path("/selected"))),
        ):
            pass
    assert caught.value.code == "unsupported_schema"


def test_gemini_registry_never_infers_from_basename_or_wrong_hash(tmp_path, monkeypatch):
    root = gemini_registry(tmp_path, monkeypatch)
    path = write(root / "registered" / "chats" / "s.jsonl", [{"sessionId": "g", "projectHash": "wrong"}])
    with freeze(path, tmp_path, "gemini") as snapshot:
        assert NormalizedSnapshotAdapter(snapshot).session_header("g").directory is None


def test_gemini_duplicate_required_ancestor_is_not_ignored(tmp_path):
    root = tmp_path / "chats"
    for index in range(2):
        write(root / f"parent-{index}.jsonl", [{"sessionId": "p", "projectHash": "h", "cwd": "/other"}])
    write(
        root / "child.jsonl",
        [
            {
                "sessionId": "c",
                "projectHash": "h",
                "cwd": "/selected",
                "parentSessionId": "p",
            }
        ],
    )
    with pytest.raises(SourceCompatibilityError) as caught:
        with freeze(
            root,
            tmp_path,
            "gemini",
            selection_evaluator=SelectionEvidenceEvaluator(SelectionFilters(directory=Path("/selected"))),
        ):
            pass
    assert caught.value.code == "ambiguous_source"


def test_gemini_equivalent_repeated_metadata_is_validated_and_applied(tmp_path):
    path = write(
        tmp_path / "g.jsonl",
        [
            {"sessionId": "g", "projectHash": "p", "startTime": "2026-01-01T00:00:00Z"},
            {"sessionId": "g", "projectHash": "p", "startTime": "2026-01-01T00:00:00+00:00", "summary": "updated"},
        ],
    )
    with freeze(path, tmp_path, "gemini") as snapshot:
        assert NormalizedSnapshotAdapter(snapshot).session_header("g").title == "updated"


def test_claude_prelude_identity_contradiction_rejected(tmp_path):
    path = write(
        tmp_path / "s.jsonl",
        [
            {"type": "mode", "sessionId": "s"},
            {"type": "user", "sessionId": "different", "cwd": "/selected"},
        ],
    )
    with pytest.raises(SourceCompatibilityError) as caught:
        schema_header(path, "claude")
    assert caught.value.code == "source_identity_changed"


def test_repeated_gemini_identity_validation_precedes_redaction(tmp_path):
    from mlflow_import.source.file_snapshot import create_file_snapshot
    from mlflow_import.sources.jsonl_common import JsonlExporter

    path = write(tmp_path / "g.jsonl", [{"sessionId": "g", "projectHash": "secret-project"}] * 2)

    def redact(value, protected):
        if isinstance(value, dict):
            return {k: redact(v, protected) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v, protected) for v in value]
        return "[MASKED]" if value == "secret-project" else value

    with create_file_snapshot(path, tmp_path, JsonlExporter(path, "gemini"), redact) as snapshot:
        assert NormalizedSnapshotAdapter(snapshot).session_header("g").project_id == "[MASKED]"
        assert b"secret-project" not in snapshot.snapshot_path.with_suffix(".payloads").read_bytes()


def test_entirely_incompatible_root_does_not_become_clean_noop(tmp_path):
    root = tmp_path / "chats"
    write(root / "unsupported.jsonl", [{"type": "unsupported-format"}])
    with pytest.raises(SourceCompatibilityError) as caught:
        with freeze(root, tmp_path, "gemini"):
            pass
    assert caught.value.code == "unsupported_schema"


def test_claude_identityless_prefix_timestamp_is_not_creation_authority(tmp_path):
    from datetime import UTC, datetime

    path = write(
        tmp_path / "s.jsonl",
        [
            {"type": "queue-operation", "timestamp": "2025-01-01T00:00:00Z"},
            {"type": "mode", "sessionId": "s", "timestamp": "2026-01-01T00:00:00Z"},
            {"type": "user", "sessionId": "s", "cwd": "/selected", "uuid": "u"},
        ],
    )
    filters = SelectionFilters(from_time=datetime(2026, 1, 1, tzinfo=UTC))
    with freeze(path, tmp_path, "claude", selection_evaluator=SelectionEvidenceEvaluator(filters)) as snapshot:
        assert [h.session_id for h in NormalizedSnapshotAdapter(snapshot).iter_session_headers()] == ["s"]


def test_explicit_claude_file_does_not_reject_unqualified_sibling(tmp_path):
    root = tmp_path / "projects"
    selected = write(root / "selected.jsonl", [{"type": "user", "sessionId": "s", "uuid": "u"}])
    write(root / "unqualified.jsonl", [{"type": "unknown-metadata"}])
    with freeze(selected, tmp_path, "claude") as snapshot:
        assert [h.session_id for h in NormalizedSnapshotAdapter(snapshot).iter_session_headers()] == ["s"]


@pytest.mark.parametrize("wrapped", [False, True])
def test_gemini_kind_validation_is_pre_redaction(tmp_path, wrapped):
    from mlflow_import.source.file_snapshot import create_file_snapshot
    from mlflow_import.sources.jsonl_common import JsonlExporter

    update = {"kind": "main"}
    path = write(
        tmp_path / "s.jsonl",
        [
            {"sessionId": "s", "projectHash": "p", "kind": "main"},
            {"$set": update} if wrapped else update,
        ],
    )

    def redact(value, protected):
        if isinstance(value, dict):
            return {k: redact(v, protected) for k, v in value.items()}
        return "[MASKED]" if value == "main" else value

    with create_file_snapshot(path, tmp_path, JsonlExporter(path, "gemini"), redact) as snapshot:
        assert NormalizedSnapshotAdapter(snapshot).session_header("s").raw_columns["kind"] == "[MASKED]"


@pytest.mark.parametrize("layout", ["subagents", "tool-results"])
def test_claude_rejects_symlinked_intermediate_sidecar_directory(tmp_path, layout):
    root = tmp_path / "projects"
    main = write(root / "s.jsonl", [{"type": "user", "sessionId": "s", "uuid": "u"}])
    outside = tmp_path / "outside"
    (outside / layout).mkdir(parents=True)
    if layout == "subagents":
        write(outside / layout / "agent-child.jsonl", [{"type": "user", "sessionId": "s", "uuid": "su"}])
    else:
        (outside / layout / "call.txt").write_text("EXTERNAL CONTENT")
    (root / "s").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SourceCompatibilityError) as caught:
        with freeze(main, tmp_path, "claude"):
            pass
    assert caught.value.code == "unsupported_source"


@pytest.mark.parametrize("duplicate", [False, True])
def test_selected_unsupported_gemini_in_mixed_root_fails(tmp_path, duplicate):
    root = tmp_path / "chats"
    write(root / "valid.jsonl", [{"sessionId": "s", "projectHash": "p", "cwd": "/selected"}])
    write(
        root / "unknown.jsonl",
        [
            {
                "sessionId": "s" if duplicate else "unknown",
                "projectHash": "p",
                "cwd": "/selected",
                "type": "new-schema",
            }
        ],
    )
    with pytest.raises(SourceCompatibilityError) as caught:
        with freeze(
            root,
            tmp_path,
            "gemini",
            selection_evaluator=SelectionEvidenceEvaluator(SelectionFilters(directory=Path("/selected"))),
        ):
            pass
    assert caught.value.code == ("ambiguous_source" if duplicate else "unsupported_schema")


@pytest.mark.parametrize("invalid", ["", None, 0, False, "wrong"])
@pytest.mark.parametrize("sidecar", [False, True])
def test_claude_later_identity_presence_is_validated(tmp_path, invalid, sidecar):
    main = write(tmp_path / "projects" / "s.jsonl", [{"type": "user", "sessionId": "s", "uuid": "u"}])
    path = main
    if sidecar:
        path = write(
            main.parent / "s" / "subagents" / "agent-child.jsonl",
            [
                {"type": "fork-context-ref"},
                {"type": "user", "sessionId": "s", "uuid": "su"},
            ],
        )
    with path.open("a") as handle:
        handle.write(json.dumps({"type": "mode", "sessionId": invalid}) + "\n")
    with pytest.raises(SourceCompatibilityError) as caught:
        with freeze(main, tmp_path, "claude"):
            pass
    assert caught.value.code == "source_identity_changed"


def test_codex_auxiliary_observations_are_redacted_before_persistence(tmp_path):
    from mlflow_import.source.file_snapshot import create_file_snapshot
    from mlflow_import.sources.jsonl_common import JsonlExporter

    marker = "PRIVATE-METADATA-MARKER"
    path = write(
        tmp_path / "c.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "c"}},
            {"type": "turn_context", "payload": {"model": marker, "model_provider": "provider"}},
            {"type": "event_msg", "payload": {"info": {"total_token_usage": {"input_tokens": 1, "extra": marker}}}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "id": "a"}},
        ],
    )

    def redact(value, protected):
        if isinstance(value, dict):
            return {k: redact(v, protected) for k, v in value.items()}
        return value.replace(marker, "[MASKED]") if isinstance(value, str) else value

    filters = SelectionFilters(model="provider/" + marker)
    with create_file_snapshot(
        path, tmp_path, JsonlExporter(path, "codex"), redact, selection_evaluator=SelectionEvidenceEvaluator(filters)
    ) as snapshot:
        assert NormalizedSnapshotAdapter(snapshot).selection_evidence("c").model_match
        assert marker.encode() not in snapshot.snapshot_path.read_bytes()
        assert marker.encode() not in snapshot.snapshot_path.with_suffix(".payloads").read_bytes()
