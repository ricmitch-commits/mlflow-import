"""Unit tests for the streamed canonical turn artifacts.

The artifact contract pinned here comes from the design spec §6.1 and the
Task 8 controller rulings:

- every turn artifact repeats the COMPLETE permitted session snapshot
  (the full session row including mutable fields plus every record
  visible through the turn's three replay cursors) as one
  canonical-json-v1 document — the same canonical bytes and golden
  vectors Task 4A pinned, with no second format or vector set;
- the document adds derived sections (normalized turn boundaries,
  warning summary), the importer/adapter/artifact/mapping versions, the
  source database fingerprint, the redaction mode, and the generated
  identity map — inserted only AFTER the pre-identity-map
  ``artifact_snapshot_fingerprint`` (the fixed ``identity_map_pending``
  placeholder) has fixed the hash and path;
- relative paths are namespaced under
  ``opencode-mlflow/v1/sessions/<encoded-session-id>/turns/<stable-turn-key>/``
  with session IDs percent-encoded so path traversal is impossible;
- writing streams through fixed-size upload chunks, temporary files are
  created atomically with mode ``0o600`` and deleted best-effort;
- payload limits: a conservative 16 MiB inline limit applies when the
  target reports none, complete root values above the limit fail with
  ``payload_too_large`` before trace creation, and oversized child
  values stay complete in the artifact with only a digest/reference
  inline.

``ReplayTurn`` doubles come from ``tests/support/fake_turn.py``; the
snapshot-backed tests drive the real export/snapshot/replay pipeline
over sanitized synthetic fixture databases exactly like the discovery
tests.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from opencode_mlflow import __version__ as importer_version
from opencode_mlflow.content.artifacts import (
    DEFAULT_INLINE_LIMIT_BYTES,
    DEFAULT_UPLOAD_CHUNK_BYTES,
    ArtifactManifest,
    ArtifactUploadChunk,
    artifact_relative_path,
    inline_value_or_reference,
    iter_artifact_chunks,
    load_root_values,
    resolve_inline_limit,
    stream_turn_artifact,
    temporary_artifact_file,
)
from opencode_mlflow.content.canonical import DuplicateKeyObject, canonical_json
from opencode_mlflow.content.fingerprint import (
    artifact_snapshot_fingerprint,
    build_effective_turn,
    canonical_record_value,
    render_fingerprint,
    source_fingerprint,
)
from opencode_mlflow.content.redaction import DeterministicRedactor
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.errors import ItemImportError
from opencode_mlflow.source.catalog import CURRENT_CATALOG_V1
from opencode_mlflow.source.current import CurrentAdapter
from opencode_mlflow.source.discovery import ARTIFACT_SCHEMA_VERSION, MAPPING_VERSION
from opencode_mlflow.source.model import SessionHeader
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter, ReplayTurn
from opencode_mlflow.source.snapshot import RedactionCallback, create_normalized_snapshot
from opencode_mlflow.source.turns import BoundedRootValue, build_replay_turn, iter_turn_summaries
from tests.support.fake_redactor import IdentityRedactor
from tests.support.fake_source import identity_redaction
from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_record, fake_summary
from tests.support.sqlite_builder import build_fixture_db

pytestmark = pytest.mark.unit

TURN_KEY = "turn-" + "1" * 64
OTHER_TURN_KEY = "turn-" + "2" * 64
RENDER_FP = "a" * 64
SNAPSHOT_FP = "b" * 64

SECRET_TABLE_SENTINELS = (
    "SECRET-ACCOUNT-TOKEN-SENTINEL",
    "SECRET-ACCOUNT-STATE-SENTINEL",
    "SECRET-CONTROL-ACCOUNT-SENTINEL",
    "credential-sentinel",
    "SECRET-SHARE-SENTINEL",
)

REDACTION_SENTINEL = "sk-abcdef0123456789abcdef"

_CURRENT_DDL = """
CREATE TABLE session_v2 (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    workspace_id TEXT,
    parent_id TEXT,
    fork_session_id TEXT,
    fork_boundary INTEGER,
    slug TEXT,
    directory TEXT,
    path TEXT,
    title TEXT,
    version TEXT,
    share_url TEXT,
    summary_additions INTEGER,
    summary_deletions INTEGER,
    summary_files INTEGER,
    summary_diffs TEXT,
    metadata TEXT,
    cost REAL,
    tokens_input INTEGER,
    tokens_output INTEGER,
    tokens_reasoning INTEGER,
    tokens_cache_read INTEGER,
    tokens_cache_write INTEGER,
    revert TEXT,
    permission TEXT,
    agent TEXT,
    model TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    time_idle INTEGER,
    time_viewed INTEGER,
    idle_outcome TEXT,
    time_compacting INTEGER,
    time_archived INTEGER,
    time_suspended INTEGER,
    resume_attempts INTEGER
);
CREATE TABLE session_message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    type TEXT NOT NULL,
    seq INTEGER,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
"""


def _scoped_record(source_id: str = "todo-1") -> object:
    return fake_record(
        source_id,
        raw_json='{"name": "synthetic-todo"}',
        source_table="todo",
        source_type="todo",
        seq=None,
        classification=None,
    )


def _minimal_header() -> SessionHeader:
    """A header reduced to its required fields plus one instant."""
    return fake_header(
        project_id=None,
        workspace_id=None,
        slug=None,
        title=None,
        directory=None,
        path=None,
        version=None,
        agent=None,
        model=None,
        metadata=None,
        permission=None,
        cost=None,
        tokens_input=None,
        tokens_output=None,
        tokens_reasoning=None,
        tokens_cache_read=None,
        tokens_cache_write=None,
        summary_additions=None,
        summary_deletions=None,
        summary_files=None,
        summary_diffs=None,
        time_updated=None,
        source_fingerprint=None,
        raw_columns={},
    )


def _turn_pair() -> tuple[FakeReplayTurn, FakeReplayTurn]:
    """Two turns of one session; each sees the other's records as unassigned."""
    header = fake_header()
    turn_one = FakeReplayTurn(
        session_header=header,
        summary=fake_summary(turn_index=1, turn_key=TURN_KEY, record_count=2),
        turn_records=(
            fake_record("msg-1", seq=1, source_type="user", classification="user"),
            fake_record("msg-2", seq=2, source_type="assistant", classification="assistant"),
        ),
        session_records=(_scoped_record(),),
        unassigned_records=(
            fake_record("msg-5", seq=5, raw_json='{"text": "second turn prompt"}'),
            fake_record("msg-9", seq=9, raw_json='{"text": "third turn prompt"}'),
        ),
    )
    turn_two = FakeReplayTurn(
        session_header=header,
        summary=fake_summary(turn_index=2, turn_key=OTHER_TURN_KEY, record_count=1),
        turn_records=(fake_record("msg-5", seq=5, raw_json='{"text": "second turn prompt"}'),),
        session_records=(_scoped_record(),),
        unassigned_records=(
            fake_record("msg-1", seq=1),
            fake_record("msg-2", seq=2, source_type="assistant", classification="assistant"),
            fake_record("msg-9", seq=9, raw_json='{"text": "third turn prompt"}'),
        ),
    )
    return turn_one, turn_two


def _stream(turn: ReplayTurn, identity_map: dict[str, str], mode: RedactionMode) -> tuple[bytes, ArtifactManifest]:
    """Stream one artifact and return its bytes plus the manifest."""
    buffer = io.BytesIO()
    manifest = stream_turn_artifact(turn, buffer, identity_map, mode)
    return buffer.getvalue(), manifest


@contextmanager
def _replay_turn_over_source(tmp_path: Path, name: str, redaction: RedactionCallback) -> Iterator[ReplayTurn]:
    """Build a real export → snapshot → replay → turn pipeline over a source DB."""
    source = tmp_path / f"{name}.db"
    if name == "current-fixture":
        connection = build_fixture_db(source, "current")
        connection.close()
    else:
        connection = sqlite3.connect(source)
        try:
            connection.executescript(_CURRENT_DDL)
            connection.execute(
                "INSERT INTO session_v2 (id, time_created, time_updated, directory, title)"
                " VALUES ('ses-redact-1', 1775000000000, 1775000000000, '/repo/synth', 'Synthetic session')"
            )
            connection.execute(
                "INSERT INTO session_message (id, session_id, type, seq, time_created, time_updated, data)"
                " VALUES ('msg-r1', 'ses-redact-1', 'user', 1, 1775000010000, 1775000010000, ?)",
                ('{"content": [{"type": "text", "text": "synthetic prompt with ' + REDACTION_SENTINEL + ' token"}]',),
            )
            connection.execute(
                "INSERT INTO session_message (id, session_id, type, seq, time_created, time_updated, data)"
                " VALUES ('msg-r2', 'ses-redact-1', 'assistant', 2, 1775000020000, 1775000020000, ?)",
                ('{"content": [{"type": "text", "text": "synthetic reply with ' + REDACTION_SENTINEL + ' token"}]',),
            )
            connection.commit()
        finally:
            connection.close()
    live = sqlite3.connect(source)
    try:
        exporter = CurrentAdapter(live, CURRENT_CATALOG_V1)
    finally:
        live.close()
    with create_normalized_snapshot(source, tmp_path, exporter, redaction) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = next(iter(adapter.iter_session_headers()))
        summary = next(iter(iter_turn_summaries(adapter.iter_records(header.session_id))))
        yield build_replay_turn(snapshot, header, summary)


class TestArtifactRelativePath:
    def test_exact_namespaced_shape(self) -> None:
        path = artifact_relative_path("ses-1", TURN_KEY, RENDER_FP, SNAPSHOT_FP)

        assert path == f"opencode-mlflow/v1/sessions/ses-1/turns/{TURN_KEY}/{RENDER_FP}-{SNAPSHOT_FP}.json"

    def test_is_deterministic(self) -> None:
        first = artifact_relative_path("ses-1", TURN_KEY, RENDER_FP, SNAPSHOT_FP)
        second = artifact_relative_path("ses-1", TURN_KEY, RENDER_FP, SNAPSHOT_FP)

        assert first == second

    @pytest.mark.parametrize(
        "session_id",
        [
            "..",
            ".",
            "../..",
            "../../etc/passwd",
            "a/b",
            "ses/../../other",
            "ses%2F..%2F..",
            "leading/slash",
            "trailing/slash/",
            "..ses",
            "ses..id",
            "ses\x00id",
            "ses\nid",
            "ses/id\\with\\backslashes",
            "ses-id-with-dashes_and_underscores",
        ],
    )
    def test_path_traversal_is_impossible(self, session_id: str) -> None:
        path = artifact_relative_path(session_id, TURN_KEY, RENDER_FP, SNAPSHOT_FP)

        segments = path.split("/")
        assert len(segments) == 7
        encoded = segments[3]
        assert encoded not in ("", ".", "..")
        assert "/" not in encoded
        # No segment of the path can be read as a traversal component by any
        # path resolver: normalization is the identity and no part is "..".
        assert os.path.normpath(path) == path
        assert ".." not in Path(path).parts

    def test_rejects_a_recorded_tag_path_over_mlflow_tag_limit(self) -> None:
        with pytest.raises(ItemImportError) as exc_info:
            artifact_relative_path("ses-" + "x" * 5000, TURN_KEY, RENDER_FP, SNAPSHOT_FP)

        assert exc_info.value.code == "artifact_path_too_long"

    def test_non_ascii_session_ids_encode_as_utf8_percent_escapes(self) -> None:
        path = artifact_relative_path("ses-é😀", TURN_KEY, RENDER_FP, SNAPSHOT_FP)

        assert path.split("/")[3] == "ses-%C3%A9%F0%9F%98%80"

    def test_empty_session_id_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="session ID"):
            artifact_relative_path("", TURN_KEY, RENDER_FP, SNAPSHOT_FP)

    @pytest.mark.parametrize(
        "turn_key",
        ["turn-synthetic-1", "turn-one", "../evil", "turn-XYZ", "turn-" + "1" * 63, "prefix-" + "1" * 64],
    )
    def test_turn_keys_must_be_turn_hex(self, turn_key: str) -> None:
        with pytest.raises(ValueError, match="turn key"):
            artifact_relative_path("ses-1", turn_key, RENDER_FP, SNAPSHOT_FP)

    @pytest.mark.parametrize("fingerprint", ["short", "z" * 64, "A" * 64, "../" + "a" * 61])
    def test_fingerprints_must_be_64_lowercase_hex(self, fingerprint: str) -> None:
        with pytest.raises(ValueError, match="fingerprint"):
            artifact_relative_path("ses-1", TURN_KEY, fingerprint, SNAPSHOT_FP)
        with pytest.raises(ValueError, match="fingerprint"):
            artifact_relative_path("ses-1", TURN_KEY, RENDER_FP, fingerprint)


class TestStreamTurnArtifact:
    def test_writes_one_valid_canonical_json_document(self) -> None:
        raw, _manifest = _stream(_turn_pair()[0], {"msg-1": "tr_1"}, RedactionMode.VERBATIM)

        assert raw.endswith(b"}")
        document = json.loads(raw)
        assert set(document) == {
            "artifact",
            "identity_map",
            "session",
            "session_records",
            "turn",
            "turn_records",
            "unassigned_records",
            "warnings",
        }

    def test_streamed_document_equals_the_materialized_canonical_serialization(self) -> None:
        turn = FakeReplayTurn(
            session_header=_minimal_header(),
            summary=fake_summary(
                turn_key=TURN_KEY,
                record_count=2,
                first_user_source_id="msg-1",
                first_user_source_type="user",
                start_time=fake_record("msg-1").timestamp,
                end_time=fake_record("msg-2").timestamp,
                user_record_count=1,
                assistant_record_count=1,
                terminal_outcome="assistant_result",
                root_result_source_id="msg-2",
            ),
            turn_records=(
                fake_record("msg-1", seq=1, source_type="user", classification="user"),
                fake_record("msg-2", seq=2, source_type="assistant", classification="assistant"),
            ),
            session_records=(_scoped_record(),),
            unassigned_records=(fake_record("msg-9", seq=9, raw_json='{"text": "other"}'),),
        )
        expected = {
            "artifact": {
                "adapter_generation": "current",
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "canonical_json": "canonical-json-v1",
                "importer_version": importer_version,
                "mapping_version": MAPPING_VERSION,
                "redaction_mode": "verbatim",
            },
            "identity_map": {"msg-1": "tr_1"},
            "session": {
                "adapter_generation": "current",
                "session_id": "ses-1",
                "time_created": {"unit": "ms", "value": 1772000000000},
                "warnings": [],
            },
            "session_records": [canonical_record_value(_scoped_record())],
            "turn": {
                "assistant_record_count": 1,
                "first_user_source_id": "msg-1",
                "first_user_source_type": "user",
                "record_count": 2,
                "root_result_source_id": "msg-2",
                "session_id": "ses-1",
                "start_time": {"unit": "s", "value": 1772000010},
                "end_time": {"unit": "s", "value": 1772000010},
                "terminal_outcome": "assistant_result",
                "turn_index": 1,
                "turn_key": TURN_KEY,
                "user_record_count": 1,
            },
            "turn_records": [
                canonical_record_value(fake_record("msg-1", seq=1, source_type="user", classification="user")),
                canonical_record_value(
                    fake_record("msg-2", seq=2, source_type="assistant", classification="assistant")
                ),
            ],
            "unassigned_records": [canonical_record_value(fake_record("msg-9", seq=9, raw_json='{"text": "other"}'))],
            "warnings": [],
        }

        raw, _manifest = _stream(turn, {"msg-1": "tr_1"}, RedactionMode.VERBATIM)

        assert raw == canonical_json(expected)

    def test_manifest_reports_identity_fingerprint_size_and_digest(self) -> None:
        turn = _turn_pair()[0]
        raw, manifest = _stream(turn, {"msg-1": "tr_1"}, RedactionMode.VERBATIM)

        assert manifest.session_id == "ses-1"
        assert manifest.turn_key == TURN_KEY
        assert manifest.snapshot_fingerprint == artifact_snapshot_fingerprint(turn, RedactionMode.VERBATIM)
        assert manifest.redaction_mode is RedactionMode.VERBATIM
        assert manifest.size_bytes == len(raw)
        assert manifest.sha256 == hashlib.sha256(raw).hexdigest()
        assert len(manifest.snapshot_fingerprint) == 64

    def test_streaming_is_deterministic_across_passes(self) -> None:
        turn = _turn_pair()[0]
        identity_map = {"msg-1": "tr_1"}

        first_raw, first_manifest = _stream(turn, identity_map, RedactionMode.VERBATIM)
        second_raw, second_manifest = _stream(turn, identity_map, RedactionMode.VERBATIM)

        assert first_raw == second_raw
        assert first_manifest == second_manifest

    def test_identity_map_is_inserted_only_after_the_placeholder_hash(self) -> None:
        turn = _turn_pair()[0]
        placeholder_raw, placeholder_manifest = _stream(turn, {}, RedactionMode.VERBATIM)
        mapped_raw, mapped_manifest = _stream(turn, {"msg-1": "tr_1", "msg-2": "sp_2"}, RedactionMode.VERBATIM)

        # The pre-identity-map snapshot fingerprint — and therefore the
        # artifact path — is fixed before generated IDs exist.
        assert placeholder_manifest.snapshot_fingerprint == mapped_manifest.snapshot_fingerprint
        assert artifact_relative_path(
            "ses-1", TURN_KEY, RENDER_FP, placeholder_manifest.snapshot_fingerprint
        ) == artifact_relative_path("ses-1", TURN_KEY, RENDER_FP, mapped_manifest.snapshot_fingerprint)
        assert mapped_manifest.snapshot_fingerprint == artifact_snapshot_fingerprint(turn, RedactionMode.VERBATIM)
        # The document differs only in the inserted identity map section.
        assert mapped_raw != placeholder_raw
        assert json.loads(mapped_raw)["identity_map"] == {"msg-1": "tr_1", "msg-2": "sp_2"}
        assert json.loads(placeholder_raw)["identity_map"] == {}

    def test_artifact_identity_map_preserves_qualified_collision_keys(self) -> None:
        turn = _turn_pair()[0]

        raw, _manifest = _stream(
            turn,
            {
                "part:shared-id": "span-part",
                "session_message:shared-id": "span-message",
            },
            RedactionMode.VERBATIM,
        )

        assert json.loads(raw)["identity_map"] == {
            "part:shared-id": "span-part",
            "session_message:shared-id": "span-message",
        }

    def test_repeats_the_complete_session_snapshot_in_each_turn_artifact(self) -> None:
        turn_one, turn_two = _turn_pair()

        first_raw, _first = _stream(turn_one, {}, RedactionMode.VERBATIM)
        second_raw, _second = _stream(turn_two, {}, RedactionMode.VERBATIM)
        first = json.loads(first_raw)
        second = json.loads(second_raw)

        # Session row (mutable fields included) and session-scoped records.
        assert first["session"] == second["session"]
        assert first["session_records"] == second["session_records"]
        assert first["session"]["cost"] == 0.25
        assert first["session"]["tokens_input"] == 120
        assert first["session"]["time_updated"] == {"unit": "ms", "value": 1772000500000}
        assert first["session"]["raw_columns"] == {"id": "ses-1", "directory": "/repo/project"}

        # Every permitted record appears in both artifacts; only the turn
        # boundaries (which records belong to this turn) differ.
        def record_ids(document: dict) -> set[str]:
            sections = ("turn_records", "session_records", "unassigned_records")
            return {record["source_id"] for name in sections for record in document[name]}

        assert record_ids(first) == record_ids(second) == {"msg-1", "msg-2", "msg-5", "msg-9", "todo-1"}
        assert {record["source_id"] for record in first["turn_records"]} == {"msg-1", "msg-2"}
        assert {record["source_id"] for record in second["turn_records"]} == {"msg-5"}
        # The normalized turn boundary sections are the derived views.
        assert first["turn"]["turn_key"] == TURN_KEY
        assert second["turn"]["turn_key"] == OTHER_TURN_KEY

    def test_session_timestamps_retain_the_original_value_and_unit(self) -> None:
        raw, _manifest = _stream(_turn_pair()[0], {}, RedactionMode.VERBATIM)
        session = json.loads(raw)["session"]

        assert session["time_created"] == {"unit": "ms", "value": 1772000000000}
        assert session["time_updated"] == {"unit": "ms", "value": 1772000500000}

    def test_records_keep_raw_malformed_and_duplicate_key_fields(self) -> None:
        malformed = replace(fake_record("msg-bad", raw_json="{not json", decoded=None), warnings=("malformed_json",))
        duplicate = fake_record(
            "msg-dup",
            raw_json='{"text": "first", "text": "second"}',
            decoded=DuplicateKeyObject(
                pairs=(("text", "first"), ("text", "second")),
                raw_text='{"text": "first", "text": "second"}',
            ),
        )
        blob_column = fake_record("msg-blob", raw_columns={"data": b"\xff\xfe"})
        turn = FakeReplayTurn(
            session_header=_minimal_header(),
            summary=fake_summary(turn_key=TURN_KEY),
            turn_records=(malformed, duplicate, blob_column),
        )

        raw, _manifest = _stream(turn, {}, RedactionMode.VERBATIM)
        records = json.loads(raw)["turn_records"]

        by_id = {record["source_id"]: record for record in records}
        assert by_id["msg-bad"]["decoded"] is None
        assert by_id["msg-bad"]["raw_text"] == "{not json"
        assert by_id["msg-bad"]["warnings"] == ["malformed_json"]
        assert by_id["msg-dup"]["decoded"]["$duplicate_keys"] == [["text", "first"], ["text", "second"]]
        assert by_id["msg-dup"]["raw_text"] == '{"text": "first", "text": "second"}'
        assert by_id["msg-blob"]["raw_columns"]["data"] == {"$bytes": "//4="}

    def test_warning_summary_collects_unique_sorted_codes(self) -> None:
        warned = replace(fake_record("msg-w1"), warnings=("malformed_json",))
        duplicate = replace(fake_record("msg-w2"), warnings=("duplicate_json_key", "malformed_json"))
        unassigned = replace(fake_record("msg-w3"), warnings=("synthetic_warning",))
        turn = FakeReplayTurn(
            session_header=fake_header(warnings=("header_warning",)),
            summary=fake_summary(turn_key=TURN_KEY, warnings=("summary_warning",)),
            turn_records=(warned, duplicate),
            unassigned_records=(unassigned,),
        )

        raw, _manifest = _stream(turn, {}, RedactionMode.VERBATIM)
        warnings = json.loads(raw)["warnings"]

        assert warnings == [
            "duplicate_json_key",
            "header_warning",
            "malformed_json",
            "summary_warning",
            "synthetic_warning",
        ]

    def test_artifact_section_records_the_contract_versions(self) -> None:
        raw, _manifest = _stream(_turn_pair()[0], {}, RedactionMode.VERBATIM)
        artifact = json.loads(raw)["artifact"]

        assert artifact["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
        assert artifact["mapping_version"] == MAPPING_VERSION
        assert artifact["importer_version"] == importer_version
        assert artifact["adapter_generation"] == "current"
        assert artifact["canonical_json"] == "canonical-json-v1"
        assert artifact["redaction_mode"] == "verbatim"
        assert artifact["source_database_fingerprint"] == "synthetic-database-fingerprint"


class TestArtifactRedactionPrivacy:
    def test_redacted_mode_artifact_carries_effective_values_only(self, tmp_path: Path) -> None:
        redactor = DeterministicRedactor(RedactionMode.REDACTED).redact
        with _replay_turn_over_source(tmp_path, "redacted-source", redactor) as turn:
            raw, manifest = _stream(turn, {"msg-r1": "tr_1"}, RedactionMode.REDACTED)

        assert manifest.redaction_mode is RedactionMode.REDACTED
        assert REDACTION_SENTINEL.encode() not in raw
        assert b"[REDACTED]" in raw

    def test_verbatim_mode_tags_the_document_and_differs_from_redacted(self, tmp_path: Path) -> None:
        redactor = DeterministicRedactor(RedactionMode.REDACTED).redact
        with _replay_turn_over_source(tmp_path, "verbatim-source", identity_redaction) as turn:
            verbatim_raw, verbatim_manifest = _stream(turn, {}, RedactionMode.VERBATIM)
        with _replay_turn_over_source(tmp_path, "redacted-source", redactor) as turn:
            redacted_raw, redacted_manifest = _stream(turn, {}, RedactionMode.REDACTED)

        assert verbatim_manifest.redaction_mode is RedactionMode.VERBATIM
        assert redacted_manifest.redaction_mode is RedactionMode.REDACTED
        assert json.loads(verbatim_raw)["artifact"]["redaction_mode"] == "verbatim"
        assert json.loads(redacted_raw)["artifact"]["redaction_mode"] == "redacted"
        assert verbatim_raw != redacted_raw

    def test_excluded_secret_table_values_never_occur(self, tmp_path: Path) -> None:
        with _replay_turn_over_source(tmp_path, "current-fixture", identity_redaction) as turn:
            raw, manifest = _stream(turn, {"msg-current-001": "tr_1"}, RedactionMode.VERBATIM)
            expected_snapshot_fp = artifact_snapshot_fingerprint(turn, RedactionMode.VERBATIM)

        for sentinel in SECRET_TABLE_SENTINELS:
            assert sentinel.encode() not in raw
        document = json.loads(raw)
        assert document["turn"]["session_id"] == "ses-current-001"
        assert manifest.snapshot_fingerprint == expected_snapshot_fp


class TestUploadChunks:
    def test_yields_fixed_size_chunks_with_digests(self) -> None:
        raw, _manifest = _stream(_turn_pair()[0], {}, RedactionMode.VERBATIM)
        chunk_size = 64

        chunks = list(iter_artifact_chunks(io.BytesIO(raw), chunk_size))

        assert all(chunk.size == chunk_size for chunk in chunks[:-1])
        assert 0 < chunks[-1].size <= chunk_size
        assert all(chunk.size == len(chunk.payload) for chunk in chunks)
        assert all(chunk.digest == hashlib.sha256(chunk.payload).hexdigest() for chunk in chunks)
        assert b"".join(chunk.payload for chunk in chunks) == raw

    def test_chunk_shape_is_pinned(self) -> None:
        chunk = ArtifactUploadChunk(size=3, digest="d" * 64, payload=b"abc")

        assert (chunk.size, chunk.digest, chunk.payload) == (3, "d" * 64, b"abc")
        with pytest.raises(AttributeError):
            chunk.size = 4  # type: ignore[misc]

    def test_default_chunk_size_is_8MiB(self) -> None:
        assert DEFAULT_UPLOAD_CHUNK_BYTES == 8 * 1024 * 1024

    def test_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            list(iter_artifact_chunks(io.BytesIO(b"abc"), 0))


class TestTemporaryArtifactFile:
    def test_creates_a_restrictive_file_and_cleans_up(self, restrictive_tmp_path: Path) -> None:
        with temporary_artifact_file(restrictive_tmp_path) as (path, handle):
            assert path.exists()
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            raw, manifest = _stream(_turn_pair()[0], {}, RedactionMode.VERBATIM)
            handle.write(raw)
            handle.flush()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest.sha256

        assert not path.exists()

    def test_cleans_up_after_an_exception(self, restrictive_tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            with temporary_artifact_file(restrictive_tmp_path) as (path, handle):
                handle.write(b"partial")
                raise RuntimeError("boom")

        assert not path.exists()

    def test_yields_a_binary_stream(self, restrictive_tmp_path: Path) -> None:
        with temporary_artifact_file(restrictive_tmp_path) as (_path, handle):
            assert isinstance(handle, io.BufferedIOBase)


class TestPayloadLimits:
    def test_default_limit_is_16MiB_when_none_reported(self) -> None:
        assert DEFAULT_INLINE_LIMIT_BYTES == 16 * 1024 * 1024
        assert resolve_inline_limit(None) == DEFAULT_INLINE_LIMIT_BYTES

    def test_reported_limit_passes_through(self) -> None:
        assert resolve_inline_limit(1024) == 1024

    @pytest.mark.parametrize("reported", [0, -1])
    def test_non_positive_reported_limit_fails_loudly(self, reported: int) -> None:
        with pytest.raises(ValueError, match="inline payload limit"):
            resolve_inline_limit(reported)

    def test_root_values_within_the_limit_load_complete(self) -> None:
        turn = FakeReplayTurn(
            session_header=_minimal_header(),
            summary=fake_summary(turn_key=TURN_KEY),
            turn_records=(fake_record("msg-1"),),
            root_input={"text": "small"},
            root_output=["part"],
        )

        values = load_root_values(turn, 4096)

        assert values.root_input == {"text": "small"}
        assert values.root_output == ["part"]

    def test_complete_root_input_above_the_limit_fails_payload_too_large(self) -> None:
        large = {"text": "x" * 8192}
        turn = FakeReplayTurn(
            session_header=_minimal_header(),
            summary=fake_summary(turn_key=TURN_KEY),
            turn_records=(fake_record("msg-1"),),
            root_input=large,
        )

        with pytest.raises(ItemImportError) as error:
            load_root_values(turn, 1024)

        assert error.value.code == "payload_too_large"
        assert error.value.exit_status == 1
        assert error.value.safe_message == "the turn's complete root input exceeds the inline payload limit"

    def test_complete_root_output_above_the_limit_fails_payload_too_large(self) -> None:
        turn = FakeReplayTurn(
            session_header=_minimal_header(),
            summary=fake_summary(turn_key=TURN_KEY),
            turn_records=(fake_record("msg-1"),),
            root_output={"text": "y" * 8192},
        )

        with pytest.raises(ItemImportError) as error:
            load_root_values(turn, 1024)

        assert error.value.code == "payload_too_large"

    def test_bounded_loader_rejections_fail_payload_too_large(self) -> None:
        oversized = BoundedRootValue(value=None, size_bytes=999, max_bytes=100, payload_too_large=True)
        turn = FakeReplayTurn(
            session_header=_minimal_header(),
            summary=fake_summary(turn_key=TURN_KEY),
            turn_records=(fake_record("msg-1"),),
            root_input=oversized,
        )

        with pytest.raises(ItemImportError) as error:
            load_root_values(turn, 100)

        assert error.value.code == "payload_too_large"

    def test_bounded_loader_values_within_the_limit_pass_through(self) -> None:
        bounded = BoundedRootValue(value={"text": "ok"}, size_bytes=12, max_bytes=100)
        turn = FakeReplayTurn(
            session_header=_minimal_header(),
            summary=fake_summary(turn_key=TURN_KEY),
            turn_records=(fake_record("msg-1"),),
            root_input=bounded,
        )

        values = load_root_values(turn, 100)

        assert values.root_input == {"text": "ok"}

    def test_missing_root_values_stay_absent(self) -> None:
        turn = FakeReplayTurn(
            session_header=_minimal_header(),
            summary=fake_summary(turn_key=TURN_KEY),
            turn_records=(fake_record("msg-1"),),
        )

        values = load_root_values(turn)

        assert values.root_input is None
        assert values.root_output is None

    def test_child_values_within_the_limit_stay_complete(self) -> None:
        placement = inline_value_or_reference({"attributes": {"seq": 1}}, 4096)

        assert placement.inline is True
        assert placement.value == {"attributes": {"seq": 1}}
        assert placement.reference is None

    def test_child_values_above_the_limit_become_a_digest_reference_inline(self) -> None:
        value = {"attributes": {"text": "z" * 8192}}
        canonical = canonical_json(value)

        placement = inline_value_or_reference(value, 1024)

        assert placement.inline is False
        assert placement.value == {
            "$artifact_reference": {
                "sha256": hashlib.sha256(canonical).hexdigest(),
                "size_bytes": len(canonical),
            }
        }
        assert placement.reference is not None
        assert placement.reference.sha256 == hashlib.sha256(canonical).hexdigest()
        assert placement.reference.size_bytes == len(canonical)

    def test_oversized_child_values_remain_complete_in_the_artifact(self) -> None:
        large_text = "z" * 8192
        value = {"attributes": {"text": large_text}}
        turn = FakeReplayTurn(
            session_header=_minimal_header(),
            summary=fake_summary(turn_key=TURN_KEY),
            turn_records=(fake_record("msg-1", decoded=value, raw_json=None),),
        )

        raw, _manifest = _stream(turn, {}, RedactionMode.VERBATIM)
        placement = inline_value_or_reference(value, 1024)

        assert placement.inline is False
        assert canonical_json(value) in raw
        assert large_text.encode() in raw


class TestArtifactIntegrationWithFingerprints:
    def test_artifacts_module_never_imports_mlflow(self) -> None:
        from opencode_mlflow.content import artifacts as artifacts_module

        source = Path(artifacts_module.__file__).read_text(encoding="utf-8")

        assert "import mlflow" not in source
        assert "from mlflow" not in source

    def test_snapshot_fingerprint_matches_the_public_artifact_function(self) -> None:
        turn = _turn_pair()[0]

        raw, manifest = _stream(turn, {"msg-1": "tr_1"}, RedactionMode.VERBATIM)

        assert manifest.snapshot_fingerprint == artifact_snapshot_fingerprint(turn, RedactionMode.VERBATIM)
        assert manifest.snapshot_fingerprint != artifact_snapshot_fingerprint(turn, RedactionMode.REDACTED)

    def test_render_fingerprint_composes_with_the_manifest_for_the_path(self) -> None:
        turn = _turn_pair()[0]
        effective = build_effective_turn(turn, IdentityRedactor(), RedactionMode.VERBATIM)
        source_fp = source_fingerprint(effective, RedactionMode.VERBATIM)
        _raw, manifest = _stream(turn, {"msg-1": "tr_1"}, RedactionMode.VERBATIM)
        render_fp = render_fingerprint(
            source_fp, ARTIFACT_SCHEMA_VERSION, MAPPING_VERSION, importer_version, RedactionMode.VERBATIM
        )

        path = artifact_relative_path("ses-1", TURN_KEY, render_fp, manifest.snapshot_fingerprint)

        assert path == (
            f"opencode-mlflow/v1/sessions/ses-1/turns/{TURN_KEY}/{render_fp}-{manifest.snapshot_fingerprint}.json"
        )
        assert Path(path).name.endswith(".json")
