"""Unit tests for the streaming canonical fingerprints.

The fingerprint boundaries pinned here come from the design spec:

- ``session_context_fingerprint`` covers only immutable session
  identity/selection context; mutable session snapshot fields are
  excluded.
- ``source_fingerprint`` is turn-local: a turn's own content change
  replaces only that turn, a session identity change replaces every
  turn, unassigned records and turn counts stay out, and the effective
  redaction mode participates.
- The redactor runs before canonicalization: changing only the original
  value of a detected secret leaves a redacted fingerprint unchanged,
  while verbatim fingerprints change.
- ``render_fingerprint`` is deterministic in every input; switching
  verbatim/redacted replaces the render.
- ``artifact_snapshot_fingerprint`` streams the complete canonical
  session payload — mutable fields, other turns' records, and
  unassigned records included — and terminates with the fixed
  ``identity_map_pending`` placeholder instead of generated MLflow IDs.

The ``ReplayTurn`` doubles come from ``tests/support/fake_turn.py``
(the concrete implementation lands in Task 5), exactly like Task 3's
fake ``SourceExporter`` pattern.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from dataclasses import replace

import pytest

from opencode_mlflow.content.canonical import CANONICAL_JSON_VERSION, canonical_json
from opencode_mlflow.content.fingerprint import (
    IDENTITY_MAP_PENDING,
    EffectiveTurn,
    artifact_snapshot_fingerprint,
    build_effective_turn,
    render_fingerprint,
    session_context_fingerprint,
    source_fingerprint,
)
from opencode_mlflow.contracts import RedactionMode, Redactor
from opencode_mlflow.source import replay as replay_module
from opencode_mlflow.source.model import SessionHeader, SourceRecord, TimestampValue, normalize_timestamp
from opencode_mlflow.source.replay import ReplayTurn
from tests.support.fake_redactor import IdentityRedactor, SecretMaskingRedactor
from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_record, fake_summary

pytestmark = pytest.mark.unit

IDENTITY = IdentityRedactor()

MUTABLE_SESSION_FIELDS: list[tuple[str, object]] = [
    ("time_updated", normalize_timestamp(1_772_999_999_000, "ms")),
    ("time_compacting", normalize_timestamp(1_772_999_999_000, "ms")),
    ("time_archived", normalize_timestamp(1_772_999_999_000, "ms")),
    ("time_idle", normalize_timestamp(1_772_999_999_000, "ms")),
    ("time_viewed", normalize_timestamp(1_772_999_999_000, "ms")),
    ("time_suspended", normalize_timestamp(1_772_999_999_000, "ms")),
    ("fork_boundary", normalize_timestamp(1_772_000_000_000, "ms")),
    ("idle_outcome", "completed"),
    ("resume_attempts", 2),
    ("cost", 99.5),
    ("tokens_input", 999),
    ("tokens_output", 999),
    ("tokens_reasoning", 999),
    ("tokens_cache_read", 999),
    ("tokens_cache_write", 999),
    ("summary_additions", 99),
    ("summary_deletions", 99),
    ("summary_files", 99),
    ("summary_diffs", {"src/other.py": 5}),
    ("metadata", {"kind": "other"}),
    ("revert", {"checkpoint": 7}),
    ("permission", {"mode": "restricted"}),
    ("source_fingerprint", "another-database-fingerprint"),
    ("raw_columns", {"id": "ses-1", "unknown_column": "additive"}),
]

CONTEXT_FIELDS: list[tuple[str, object]] = [
    ("session_id", "ses-other"),
    ("adapter_generation", "legacy"),
    ("project_id", "proj-other"),
    ("workspace_id", "ws-other"),
    ("parent_id", "ses-parent"),
    ("fork_session_id", "ses-fork"),
    ("slug", "other-slug"),
    ("title", "Other title"),
    ("directory", "/repo/other"),
    ("path", "/repo/other/session.jsonl"),
    ("version", "9.9.9"),
    ("agent", "other-agent"),
    ("model", {"id": "provider/other"}),
    ("time_created", normalize_timestamp(1_999_000_000_000, "ms")),
]


def _header() -> SessionHeader:
    return fake_header()


def _user_record(source_id: str, seq: int, raw_json: str = '{"text": "synthetic prompt"}') -> SourceRecord:
    return fake_record(source_id, raw_json=raw_json, seq=seq, source_type="user", classification="user")


def _assistant_record(source_id: str, seq: int, raw_json: str = '{"text": "synthetic response"}') -> SourceRecord:
    return fake_record(source_id, raw_json=raw_json, seq=seq, source_type="assistant", classification="assistant")


def _scoped_record(source_id: str = "todo-1") -> SourceRecord:
    return fake_record(
        source_id,
        raw_json='{"name": "synthetic-todo"}',
        source_table="todo",
        source_type="todo",
        seq=None,
        classification=None,
    )


def _turn_pair() -> tuple[FakeReplayTurn, FakeReplayTurn]:
    """Two turns of one session; each sees the other's records as unassigned."""
    header = _header()
    turn_one = FakeReplayTurn(
        session_header=header,
        summary=fake_summary(turn_index=1, turn_key="turn-one", record_count=2),
        turn_records=(_user_record("msg-1", 1), _assistant_record("msg-2", 2)),
        session_records=(_scoped_record(),),
        unassigned_records=(_user_record("msg-9", 9, raw_json='{"text": "other turn prompt"}'),),
    )
    turn_two = FakeReplayTurn(
        session_header=header,
        summary=fake_summary(turn_index=2, turn_key="turn-two", record_count=1),
        turn_records=(_user_record("msg-5", 5, raw_json='{"text": "second turn prompt"}'),),
        session_records=(_scoped_record(),),
        unassigned_records=(_user_record("msg-1", 1), _assistant_record("msg-2", 2)),
    )
    return turn_one, turn_two


def _source_fp(
    replay: FakeReplayTurn,
    redactor: Redactor = IDENTITY,
    mode: RedactionMode = RedactionMode.VERBATIM,
) -> str:
    return source_fingerprint(build_effective_turn(replay, redactor), mode)


def _context_fp(replay: FakeReplayTurn) -> str:
    return session_context_fingerprint(build_effective_turn(replay, IDENTITY))


def _artifact_fp(replay: FakeReplayTurn, mode: RedactionMode = RedactionMode.VERBATIM) -> str:
    return artifact_snapshot_fingerprint(replay, mode)


def _with_creation(replay: FakeReplayTurn, time_created: object) -> FakeReplayTurn:
    return replace(replay, session_header=replace(_header(), time_created=time_created))


class TestEffectiveTurn:
    def test_streams_without_record_lists_or_generated_ids(self) -> None:
        turn = build_effective_turn(_turn_pair()[0], IDENTITY)

        assert {item.name for item in dataclass_fields(EffectiveTurn)} == {
            "session_header",
            "summary",
            "redaction_mode",
            "replay",
            "redactor",
        }
        assert isinstance(turn.replay, ReplayTurn)
        assert turn.session_id == "ses-1"
        assert turn.turn_key == "turn-one"
        assert turn.schema_generation == "current"

    def test_default_mode_is_verbatim_and_explicit_modes_are_stored(self) -> None:
        replay = _turn_pair()[0]

        assert build_effective_turn(replay, IDENTITY).redaction_mode is RedactionMode.VERBATIM
        assert build_effective_turn(replay, IDENTITY, RedactionMode.REDACTED).redaction_mode is RedactionMode.REDACTED

    def test_fake_replay_turn_satisfies_the_owned_protocol(self) -> None:
        assert isinstance(_turn_pair()[0], ReplayTurn)

    def test_replay_turn_is_imported_never_redefined(self) -> None:
        from opencode_mlflow.content import fingerprint

        assert fingerprint.ReplayTurn is replay_module.ReplayTurn
        assert "ReplayTurn" not in fingerprint.__all__

    def test_cursors_yield_fresh_deterministic_canonical_bytes(self) -> None:
        turn = build_effective_turn(_turn_pair()[0], IDENTITY)

        first = list(turn.iter_canonical_records())
        second = list(turn.iter_canonical_records())

        assert first == second
        assert len(first) == 2
        assert all(isinstance(chunk, bytes) for chunk in first)
        assert list(turn.iter_canonical_session_records()) == list(turn.iter_canonical_session_records())
        assert list(turn.iter_canonical_unassigned_records()) == list(turn.iter_canonical_unassigned_records())

    def test_cursors_redact_before_canonicalizing(self) -> None:
        replay = FakeReplayTurn(
            session_header=_header(),
            summary=fake_summary(),
            turn_records=(fake_record("msg-1", raw_json='{"secret": "alpha-sentinel", "text": "hi"}'),),
        )
        turn = build_effective_turn(replay, SecretMaskingRedactor())
        material = b"".join(turn.iter_canonical_records())

        assert b"alpha-sentinel" not in material
        assert b"[REDACTED]" in material


class TestSessionContextFingerprint:
    def test_is_deterministic_64_char_lowercase_hex(self) -> None:
        fingerprint = _context_fp(_turn_pair()[0])

        assert fingerprint == _context_fp(_turn_pair()[0])
        assert len(fingerprint) == 64
        assert fingerprint == fingerprint.lower()
        int(fingerprint, 16)

    @pytest.mark.parametrize(("field", "value"), MUTABLE_SESSION_FIELDS)
    def test_mutable_session_snapshot_fields_are_excluded(self, field: str, value: object) -> None:
        replay = _turn_pair()[0]

        changed = replace(replay, session_header=replace(_header(), **{field: value}))

        assert _context_fp(changed) == _context_fp(replay)

    @pytest.mark.parametrize(("field", "value"), CONTEXT_FIELDS)
    def test_identity_and_selection_fields_are_included(self, field: str, value: object) -> None:
        replay = _turn_pair()[0]

        changed = replace(replay, session_header=replace(_header(), **{field: value}))

        assert _context_fp(changed) != _context_fp(replay)

    def test_creation_instant_normalizes_to_epoch_nanoseconds(self) -> None:
        replay = _turn_pair()[0]
        milliseconds = _with_creation(replay, normalize_timestamp(1_772_000_000_000, "ms"))
        seconds = _with_creation(replay, normalize_timestamp(1_772_000_000, "s"))
        different = _with_creation(replay, normalize_timestamp(1_773_000_000_000, "ms"))

        assert _context_fp(milliseconds) == _context_fp(seconds)
        assert _context_fp(different) != _context_fp(replay)


class TestSourceFingerprint:
    def test_is_deterministic_64_char_lowercase_hex(self) -> None:
        fingerprint = _source_fp(_turn_pair()[0])

        assert fingerprint == _source_fp(_turn_pair()[0])
        assert len(fingerprint) == 64
        assert fingerprint == fingerprint.lower()
        int(fingerprint, 16)

    def test_session_identity_change_replaces_all_turns(self) -> None:
        turn_one, turn_two = _turn_pair()
        renamed_header = replace(_header(), title="Renamed session")

        renamed_one = replace(turn_one, session_header=renamed_header)
        renamed_two = replace(turn_two, session_header=renamed_header)

        assert _source_fp(renamed_one) != _source_fp(turn_one)
        assert _source_fp(renamed_two) != _source_fp(turn_two)

    def test_turn_local_change_replaces_only_that_turn(self) -> None:
        turn_one, turn_two = _turn_pair()
        base_one = _source_fp(turn_one)
        base_two = _source_fp(turn_two)
        changed_record = _user_record("msg-5", 5, raw_json='{"text": "changed second turn prompt"}')
        changed_two = replace(turn_two, turn_records=(changed_record,))
        # Turn one observes turn two's changed record only through its unassigned cursor.
        changed_one = replace(turn_one, unassigned_records=(changed_record,))

        assert _source_fp(changed_two) != base_two
        assert _source_fp(changed_one) == base_one

    def test_session_scoped_record_change_replaces_the_turn(self) -> None:
        turn_one, _ = _turn_pair()
        changed = replace(turn_one, session_records=(_scoped_record("todo-2"),))

        assert _source_fp(changed) != _source_fp(turn_one)

    def test_unassigned_record_change_does_not_replace_the_turn(self) -> None:
        turn_one, _ = _turn_pair()
        changed = replace(turn_one, unassigned_records=(_user_record("msg-77", 77),))

        assert _source_fp(changed) == _source_fp(turn_one)

    def test_turn_counts_are_excluded(self) -> None:
        turn_one, _ = _turn_pair()
        recounted = replace(turn_one, summary=replace(turn_one.summary, record_count=99, user_record_count=99))

        assert _source_fp(recounted) == _source_fp(turn_one)

    def test_effective_redaction_mode_participates(self) -> None:
        replay = _turn_pair()[0]

        assert _source_fp(replay, mode=RedactionMode.VERBATIM) != _source_fp(replay, mode=RedactionMode.REDACTED)

    def test_detected_secret_original_value_change_keeps_a_redacted_fingerprint(self) -> None:
        redactor = SecretMaskingRedactor()
        turn_alpha = FakeReplayTurn(
            session_header=_header(),
            summary=fake_summary(),
            turn_records=(fake_record("msg-1", raw_json='{"secret": "alpha-sentinel", "text": "hello"}'),),
        )
        turn_beta = FakeReplayTurn(
            session_header=_header(),
            summary=fake_summary(),
            turn_records=(fake_record("msg-1", raw_json='{"secret": "beta-sentinel", "text": "hello"}'),),
        )

        assert _source_fp(turn_alpha, redactor, RedactionMode.REDACTED) == _source_fp(
            turn_beta, redactor, RedactionMode.REDACTED
        )

    def test_verbatim_fingerprint_changes_when_only_a_secret_changes(self) -> None:
        turn_alpha = FakeReplayTurn(
            session_header=_header(),
            summary=fake_summary(),
            turn_records=(fake_record("msg-1", raw_json='{"secret": "alpha-sentinel", "text": "hello"}'),),
        )
        turn_beta = FakeReplayTurn(
            session_header=_header(),
            summary=fake_summary(),
            turn_records=(fake_record("msg-1", raw_json='{"secret": "beta-sentinel", "text": "hello"}'),),
        )

        assert _source_fp(turn_alpha) != _source_fp(turn_beta)

    def test_null_and_absent_raw_columns_differ(self) -> None:
        turn_null = FakeReplayTurn(
            session_header=_header(),
            summary=fake_summary(),
            turn_records=(fake_record("msg-1", raw_columns={"data": None}),),
        )
        turn_absent = FakeReplayTurn(
            session_header=_header(),
            summary=fake_summary(),
            turn_records=(fake_record("msg-1", raw_columns={}),),
        )

        assert _source_fp(turn_null) != _source_fp(turn_absent)

    def test_record_timestamps_normalize_to_epoch_nanoseconds(self) -> None:
        seconds = FakeReplayTurn(
            session_header=_header(),
            summary=fake_summary(),
            turn_records=(replace(_user_record("msg-1", 1), timestamp=normalize_timestamp(1_772_000_010, "s")),),
        )
        milliseconds = FakeReplayTurn(
            session_header=_header(),
            summary=fake_summary(),
            turn_records=(replace(_user_record("msg-1", 1), timestamp=normalize_timestamp(1_772_000_010_000, "ms")),),
        )

        assert _source_fp(seconds) == _source_fp(milliseconds)


class TestRenderFingerprint:
    def test_is_deterministic_64_char_lowercase_hex(self) -> None:
        fingerprint = render_fingerprint("a" * 64, "artifact-v1", "mapping-v1", "0.1.0", RedactionMode.VERBATIM)

        assert fingerprint == render_fingerprint("a" * 64, "artifact-v1", "mapping-v1", "0.1.0", RedactionMode.VERBATIM)
        assert len(fingerprint) == 64
        assert fingerprint == fingerprint.lower()
        int(fingerprint, 16)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("source_fp", "b" * 64),
            ("artifact_version", "artifact-v2"),
            ("mapping_version", "mapping-v2"),
            ("importer_version", "0.2.0"),
            ("redaction_mode", RedactionMode.REDACTED),
        ],
    )
    def test_every_input_participates(self, field: str, value: object) -> None:
        base: dict[str, object] = {
            "source_fp": "a" * 64,
            "artifact_version": "artifact-v1",
            "mapping_version": "mapping-v1",
            "importer_version": "0.1.0",
            "redaction_mode": RedactionMode.VERBATIM,
        }

        assert render_fingerprint(**{**base, field: value}) != render_fingerprint(**base)

    def test_verbatim_versus_redacted_modes_differ(self) -> None:
        verbatim = render_fingerprint("a" * 64, "artifact-v1", "mapping-v1", "0.1.0", RedactionMode.VERBATIM)
        redacted = render_fingerprint("a" * 64, "artifact-v1", "mapping-v1", "0.1.0", RedactionMode.REDACTED)

        assert verbatim != redacted
        assert len(verbatim) == 64
        assert len(redacted) == 64


class TestArtifactSnapshotFingerprint:
    def test_is_deterministic_64_char_lowercase_hex(self) -> None:
        fingerprint = _artifact_fp(_turn_pair()[0])

        assert fingerprint == _artifact_fp(_turn_pair()[0])
        assert len(fingerprint) == 64
        assert fingerprint == fingerprint.lower()
        int(fingerprint, 16)

    def test_changes_for_mutable_fields_the_turn_fingerprint_excludes(self) -> None:
        replay = _turn_pair()[0]
        changed = replace(
            replay,
            session_header=replace(_header(), time_updated=normalize_timestamp(1_772_999_999_000, "ms")),
        )

        assert _artifact_fp(changed) != _artifact_fp(replay)
        assert _source_fp(changed) == _source_fp(replay)

    def test_changes_when_another_turns_records_change(self) -> None:
        turn_one, _ = _turn_pair()
        changed_other = replace(
            turn_one,
            unassigned_records=(_user_record("msg-9", 9, raw_json='{"text": "changed other turn prompt"}'),),
        )

        assert _artifact_fp(changed_other) != _artifact_fp(turn_one)
        assert _source_fp(changed_other) == _source_fp(turn_one)

    def test_changes_when_unassigned_records_change(self) -> None:
        turn_one, _ = _turn_pair()
        with_unassigned = replace(turn_one, unassigned_records=(*turn_one.unassigned_records, _scoped_record("todo-9")))

        assert _artifact_fp(with_unassigned) != _artifact_fp(turn_one)
        assert _source_fp(with_unassigned) == _source_fp(turn_one)

    def test_effective_redaction_mode_participates(self) -> None:
        replay = _turn_pair()[0]

        assert _artifact_fp(replay, RedactionMode.VERBATIM) != _artifact_fp(replay, RedactionMode.REDACTED)

    def test_identity_map_placeholder_is_fixed_and_final(self) -> None:
        # White-box pin: Task 8's artifact writer must reproduce this hash
        # before inserting generated MLflow IDs, so the placeholder and its
        # position are part of the contract.
        from opencode_mlflow.content.fingerprint import _iter_artifact_snapshot_chunks

        chunks = list(_iter_artifact_snapshot_chunks(_turn_pair()[0], RedactionMode.VERBATIM))

        assert IDENTITY_MAP_PENDING == "identity_map_pending"
        assert chunks[-1] == canonical_json({"identity_map": IDENTITY_MAP_PENDING})

    def test_hashes_exactly_the_public_chunk_stream(self) -> None:
        from opencode_mlflow.content.fingerprint import (
            _DOMAIN_ARTIFACT_SNAPSHOT,
            _hash_chunks,
            _iter_artifact_snapshot_chunks,
        )

        replay = _turn_pair()[0]
        chunks = _iter_artifact_snapshot_chunks(replay, RedactionMode.VERBATIM)
        expected = _hash_chunks(_DOMAIN_ARTIFACT_SNAPSHOT, chunks)

        assert artifact_snapshot_fingerprint(replay, RedactionMode.VERBATIM) == expected

    def test_artifact_document_sections_reproduce_the_fingerprint(self) -> None:
        """Task 8 integration: the artifact document IS the hashed payload.

        The streamed artifact's session and record sections (plus the
        fixed placeholder) reconstruct exactly the artifact-snapshot
        fingerprint material, proving the document persists the complete
        canonical session payload the fingerprint covers (Ruling 11) —
        the session section differs only by carrying the original
        ``{"unit", "value"}`` timestamp representation instead of epoch
        nanoseconds, and the turn/warning sections are derived views.
        """
        import io
        import json

        from opencode_mlflow.content.artifacts import stream_turn_artifact
        from opencode_mlflow.content.fingerprint import _DOMAIN_ARTIFACT_SNAPSHOT, _hash_chunks

        replay = _turn_pair()[0]
        buffer = io.BytesIO()
        manifest = stream_turn_artifact(replay, buffer, {"msg-1": "tr_1"}, RedactionMode.VERBATIM)
        document = json.loads(buffer.getvalue())

        session_value: dict[str, object] = {}
        for name, value in document["session"].items():
            if name == "source_fingerprint":
                continue
            if isinstance(value, dict) and set(value) == {"unit", "value"}:
                session_value[name] = TimestampValue(value=value["value"], unit=value["unit"]).nanoseconds
            else:
                session_value[name] = value
        chunks = [
            canonical_json(
                {
                    "canonical_json": CANONICAL_JSON_VERSION,
                    "schema_generation": replay.session_header.adapter_generation,
                    "redaction_mode": RedactionMode.VERBATIM.value,
                }
            ),
            canonical_json(session_value),
        ]
        for section in ("turn_records", "session_records", "unassigned_records"):
            chunks.extend(canonical_json(record) for record in document[section])
        chunks.append(canonical_json({"identity_map": IDENTITY_MAP_PENDING}))

        assert _hash_chunks(_DOMAIN_ARTIFACT_SNAPSHOT, chunks) == manifest.snapshot_fingerprint
        assert manifest.snapshot_fingerprint == artifact_snapshot_fingerprint(replay, RedactionMode.VERBATIM)
