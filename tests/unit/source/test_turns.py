"""Unit tests for logical-turn reconstruction.

These tests pin the Task 5 turn contract: only persisted non-synthetic
user prompts start turns; the last non-synthetic assistant message
container is the root result; turn keys are stable with documented
fallback and collision handling; summaries stream without retaining
record lists and never carry prompt/response content; and
``build_replay_turn`` returns the concrete ``ReplayTurn`` with fresh
normalized-snapshot cursors, complete cursor coverage, and bounded
root loaders.

Fine-grained cursor and loader behavior runs over a scripted snapshot
(written order equals the given order); boundary semantics are also
pinned end to end through the real current and legacy adapters over the
sanitized synthetic fixture databases.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from opencode_mlflow.content.canonical import canonical_json
from opencode_mlflow.content.fingerprint import (
    FramedDigest,
    build_effective_turn,
    canonical_record_value,
)
from opencode_mlflow.contracts import CanonicalRecordEnvelope
from opencode_mlflow.source.catalog import CURRENT_CATALOG_V1, LEGACY_CATALOG_V1
from opencode_mlflow.source.current import CurrentAdapter
from opencode_mlflow.source.legacy import LegacyAdapter
from opencode_mlflow.source.model import SessionHeader, SourceRecord, normalize_timestamp
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter, ReplayTurn
from opencode_mlflow.source.snapshot import (
    SourceExporter,
    SourceSnapshot,
    create_normalized_snapshot,
)
from opencode_mlflow.source.turns import (
    ASSISTANT_ERROR_OUTCOME,
    ASSISTANT_RESULT_OUTCOME,
    NO_ASSISTANT_RESULT_OUTCOME,
    TURN_KEY_COLLISION_WARNING,
    TURN_KEY_FALLBACK_WARNING,
    BoundedRootValue,
    build_replay_turn,
    iter_turn_records,
    iter_turn_summaries,
    turn_key,
)
from tests.support.fake_redactor import IdentityRedactor
from tests.support.fake_source import (
    ScriptedExporter,
    ScriptedRecord,
    ScriptedSession,
    identity_redaction,
)
from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_summary
from tests.support.sqlite_builder import build_fixture_db, fixture_script

pytestmark = pytest.mark.unit

_SOURCE_COUNTER = itertools.count()

_USER_JSON = '{"content": [{"type": "text", "text": "synthetic prompt"}]}'
_REPLY_JSON = '{"content": [{"type": "text", "text": "synthetic reply"}]}'


def _record(
    source_id: str | None,
    *,
    source_type: str = "user",
    classification: str | None = "user",
    table: str = "session_message",
    seq: int | None = 1,
    timestamp_ms: int | None = 1_772_000_010_000,
    session_id: str = "ses-1",
    raw_json: str | None = _USER_JSON,
    parent_source_id: str | None = None,
    derived_child_ordinal: int | None = None,
    source_identity: str | None = None,
) -> SourceRecord:
    """One normalized record carrying the identity/ordering fields adapters set."""
    return SourceRecord(
        source_table=table,
        source_type=source_type,
        source_id=source_id,
        raw_json=raw_json,
        decoded_value=json.loads(raw_json) if raw_json is not None else None,
        session_id=session_id,
        sequence_domain="session_message.seq" if table == "session_message" else None,
        sequence_value=seq,
        parent_source_id=parent_source_id,
        derived_child_ordinal=derived_child_ordinal,
        source_identity=source_identity,
        timestamp=normalize_timestamp(timestamp_ms, "ms") if timestamp_ms is not None else None,
        classification=classification,
    )


def _user_prompt(
    source_id: str,
    *,
    seq: int = 1,
    raw_json: str | None = _USER_JSON,
    timestamp_ms: int | None = 1_772_000_010_000,
) -> SourceRecord:
    return _record(source_id, seq=seq, raw_json=raw_json, timestamp_ms=timestamp_ms)


def _assistant_result(
    source_id: str,
    *,
    seq: int = 2,
    raw_json: str | None = _REPLY_JSON,
    timestamp_ms: int | None = 1_772_000_010_000,
) -> SourceRecord:
    return _record(
        source_id,
        source_type="assistant",
        classification="assistant",
        seq=seq,
        raw_json=raw_json,
        timestamp_ms=timestamp_ms,
    )


def _identities(records: Iterable[SourceRecord]) -> list[str]:
    """The stable identities (source ID or fallback identity) of records."""
    return [record.source_id or record.source_identity or "" for record in records]


@pytest.fixture
def records() -> list[SourceRecord]:
    """Two consecutive user prompts with no assistant result between them."""
    return [_user_prompt("user-1", seq=1), _user_prompt("user-2", seq=2)]


def test_consecutive_user_prompts_close_previous_turn_without_synthetic_output(records):
    summaries = list(iter_turn_summaries(records))

    assert [summary.turn_index for summary in summaries] == [1, 2]
    assert summaries[0].root_result_source_id is None
    assert summaries[0].terminal_outcome == "no_assistant_result"
    assert summaries[1].first_user_source_id == "user-2"


class TestTurnBoundaries:
    @pytest.mark.parametrize(
        "statuses",
        [
            ("failed", "pending"),
            ("pending", "failed"),
        ],
    )
    def test_explicit_assistant_failure_dominates_pending_statuses(self, statuses: tuple[str, str]) -> None:
        records = [
            _user_prompt("user-1", seq=1),
            _assistant_result("asst-1", seq=2, raw_json=json.dumps({"status": statuses[0]})),
            _assistant_result("asst-2", seq=3, raw_json=json.dumps({"status": statuses[1]})),
        ]

        (summary,) = iter_turn_summaries(records)

        assert summary.terminal_outcome == ASSISTANT_ERROR_OUTCOME

    def test_later_success_supersedes_pending_lifecycle_but_not_lifecycle_failure(self) -> None:
        pending_then_success = [
            _user_prompt("user-pending", seq=1),
            _record(
                "pending-1",
                table="session_pending",
                source_type="event",
                classification=None,
                seq=2,
                raw_json='{"status": "pending"}',
            ),
            _assistant_result("asst-success", seq=3),
        ]
        failure_then_success = [
            _user_prompt("user-failure", seq=1),
            _record(
                "failed-1",
                table="session_pending",
                source_type="event",
                classification=None,
                seq=2,
                raw_json='{"status": "failed"}',
            ),
            _assistant_result("asst-success-2", seq=3),
        ]

        (pending_summary,) = iter_turn_summaries(pending_then_success)
        (failure_summary,) = iter_turn_summaries(failure_then_success)

        assert pending_summary.terminal_outcome == ASSISTANT_RESULT_OUTCOME
        assert failure_summary.terminal_outcome == "lifecycle_failed"

    def test_turn_contains_records_until_the_next_user_prompt(self) -> None:
        records = [
            _user_prompt("user-1", seq=1),
            _assistant_result("asst-1", seq=2),
            _record("sys-1", source_type="system", classification="system", seq=3),
            _user_prompt("user-2", seq=4),
            _assistant_result("asst-2", seq=5),
        ]

        summaries = list(iter_turn_summaries(records))

        assert [summary.record_count for summary in summaries] == [3, 2]
        assert [summary.first_user_source_id for summary in summaries] == ["user-1", "user-2"]

    def test_records_before_the_first_prompt_belong_to_no_turn(self) -> None:
        records = [
            _record("sys-1", source_type="system", classification="system", seq=1),
            _record("tool-1", source_type="tool", classification=None, seq=2),
            _user_prompt("user-1", seq=3),
            _assistant_result("asst-1", seq=4),
        ]

        (summary,) = iter_turn_summaries(records)

        # The pre-prompt records neither start a turn nor join one.
        assert summary.record_count == 2
        assert summary.first_user_source_id == "user-1"

    def test_a_session_without_user_prompts_has_no_turns(self) -> None:
        records = [
            _record("sys-1", source_type="system", classification="system", seq=1),
            _assistant_result("asst-1", seq=2),
            _record("tool-1", source_type="tool", classification=None, seq=3),
        ]

        assert list(iter_turn_summaries(records)) == []

    @pytest.mark.parametrize(
        ("source_type", "classification"),
        [
            ("system", "system"),
            ("tool", None),
            ("lifecycle", None),
            ("todo", None),
        ],
    )
    def test_non_user_records_never_start_turns(self, source_type: str, classification: str | None) -> None:
        records = [
            _record("other-1", source_type=source_type, classification=classification, seq=1),
            _user_prompt("user-1", seq=2),
        ]

        (summary,) = iter_turn_summaries(records)

        assert summary.turn_index == 1
        assert summary.first_user_source_id == "user-1"
        assert summary.record_count == 1

    def test_synthetic_user_rows_never_start_turns(self) -> None:
        records = [
            _record(
                "synthetic-user-1",
                classification="synthetic",
                seq=1,
                raw_json='{"synthetic": true, "content": [{"type": "text", "text": "synthetic marker"}]}',
            ),
            # A synthetic row the current classification rules would label
            # "user": the catalogued synthetic marker still excludes it.
            _record(
                "synthetic-user-2",
                classification="user",
                seq=2,
                raw_json='{"synthetic": true, "content": [{"type": "text", "text": "synthetic marker"}]}',
            ),
            _user_prompt("user-1", seq=3),
        ]

        (summary,) = iter_turn_summaries(records)

        assert summary.first_user_source_id == "user-1"
        assert summary.record_count == 1

    def test_legacy_parts_classified_user_do_not_start_turns(self) -> None:
        records = [
            _record("msg-1", table="message", seq=None),
            _record(
                "part-1",
                table="part",
                source_type="text",
                classification="user",
                seq=None,
                raw_json='{"type": "text", "text": "synthetic part text"}',
                parent_source_id="msg-1",
            ),
            _record(
                "part-2",
                table="part",
                source_type="text",
                classification="user",
                seq=None,
                raw_json='{"type": "text", "text": "another synthetic part"}',
                parent_source_id="msg-1",
            ),
        ]

        (summary,) = iter_turn_summaries(records)

        # The user message starts exactly one turn; its user-classified
        # parts are content inside that turn, not new prompts.
        assert summary.turn_index == 1
        assert summary.record_count == 3
        assert summary.user_record_count == 3

    def test_root_result_is_the_last_non_synthetic_assistant_message(self) -> None:
        records = [
            _user_prompt("user-1", seq=1),
            _assistant_result("asst-1", seq=2),
            _assistant_result("asst-2", seq=3),
        ]

        (summary,) = iter_turn_summaries(records)

        assert summary.root_result_source_id == "asst-2"
        assert summary.terminal_outcome == ASSISTANT_RESULT_OUTCOME
        assert summary.assistant_record_count == 2
        # The earlier assistant result stays a turn record (a child candidate).
        assert _identities(iter_turn_records(records, summary)) == ["user-1", "asst-1", "asst-2"]

    def test_synthetic_assistant_results_are_not_root_results(self) -> None:
        records = [
            _user_prompt("user-1", seq=1),
            _record(
                "asst-synthetic",
                source_type="assistant",
                classification="assistant",
                seq=2,
                raw_json='{"synthetic": true, "content": [{"type": "text", "text": "synthetic reply"}]}',
            ),
        ]

        (summary,) = iter_turn_summaries(records)

        assert summary.root_result_source_id is None
        assert summary.terminal_outcome == NO_ASSISTANT_RESULT_OUTCOME

    def test_assistant_parts_are_not_root_results(self) -> None:
        records = [
            _user_prompt("user-1", seq=1),
            _record("msg-1", table="message", source_type="assistant", classification="assistant", seq=None),
            _record(
                "part-1",
                table="part",
                source_type="text",
                classification="assistant",
                seq=None,
                raw_json='{"type": "text", "text": "synthetic assistant text"}',
                parent_source_id="msg-1",
            ),
        ]

        (summary,) = iter_turn_summaries(records)

        # Only the assistant message container is a result candidate; the
        # assistant-classified part is its split content.
        assert summary.root_result_source_id == "msg-1"

    def test_summary_counts_and_timestamps(self) -> None:
        records = [
            _user_prompt("user-1", seq=1, timestamp_ms=1_772_000_010_000),
            _record(
                "user-1#content-0",
                source_type="text",
                classification=None,
                seq=1,
                timestamp_ms=1_772_000_010_000,
                raw_json='{"type": "text", "text": "synthetic prompt"}',
                parent_source_id="user-1",
                derived_child_ordinal=0,
            ),
            _assistant_result("asst-1", seq=2, timestamp_ms=1_772_000_020_000),
            _record("sys-1", source_type="system", classification="system", seq=3, timestamp_ms=1_772_000_030_000),
        ]

        (summary,) = iter_turn_summaries(records)

        assert summary.record_count == 4
        assert summary.user_record_count == 1
        assert summary.assistant_record_count == 1
        assert summary.start_time is not None
        assert summary.start_time.nanoseconds == 1_772_000_010_000_000_000
        assert summary.end_time is not None
        assert summary.end_time.nanoseconds == 1_772_000_030_000_000_000


class TestTurnKeys:
    def test_normal_key_hashes_session_type_and_source_id(self) -> None:
        record = _user_prompt("user-1")

        expected = "turn-" + hashlib.sha256(b"ses-1\0user\0user-1").hexdigest()

        assert turn_key("ses-1", record, 1) == expected
        assert turn_key("ses-1", record, 7) == expected

    def test_keys_are_stable_across_reiteration(self) -> None:
        records = [_user_prompt("user-1", seq=1), _user_prompt("user-2", seq=2)]

        first = [summary.turn_key for summary in iter_turn_summaries(records)]
        second = [summary.turn_key for summary in iter_turn_summaries(records)]

        assert first == second
        assert len(set(first)) == 2

    def test_missing_source_id_hashes_the_fallback_tuple(self) -> None:
        raw_json = '{"content": [{"type": "text", "text": "synthetic no-id prompt"}]}'
        record = _record(None, source_identity=None, seq=7, raw_json=raw_json)
        turn_index = 3

        expected = (
            "turn-"
            + hashlib.sha256(
                b"\0".join(
                    (
                        canonical_json("ses-1"),
                        canonical_json("user"),
                        canonical_json(7),
                        canonical_json(turn_index),
                        canonical_json(turn_index),
                    )
                )
            ).hexdigest()
        )

        assert turn_key("ses-1", record, turn_index) == expected
        changed = _record(None, source_identity=None, seq=7, raw_json='{"password":"changed-content"}')
        assert turn_key("ses-1", record, turn_index) == turn_key("ses-1", changed, turn_index)
        assert turn_key("ses-1", record, turn_index) != turn_key("ses-1", _user_prompt("user-1"), turn_index)

    def test_fallback_omits_absent_sequence_and_timestamp(self) -> None:
        sequenced = _record(None, source_identity=None, seq=7)
        unsequenced = _record(None, source_identity=None, seq=None, timestamp_ms=None)

        assert turn_key("ses-1", sequenced, 1) != turn_key("ses-1", unsequenced, 1)
        # The omission is deterministic for repeated scans.
        assert turn_key("ses-1", unsequenced, 1) == turn_key("ses-1", unsequenced, 1)

    def test_identical_missing_id_records_stay_distinct_by_turn_index(self) -> None:
        records = [
            _record(None, source_identity=None, seq=None, timestamp_ms=None),
            _record(None, source_identity=None, seq=None, timestamp_ms=None),
        ]

        summaries = list(iter_turn_summaries(records))

        assert len(summaries) == 2
        assert len({summary.turn_key for summary in summaries}) == 2
        assert all(summary.warnings == (TURN_KEY_FALLBACK_WARNING,) for summary in summaries)

    def test_colliding_keys_add_the_occurrence_ordinal_and_warn(self) -> None:
        records = [_user_prompt("dup-1", seq=1), _user_prompt("dup-1", seq=2)]

        summaries = list(iter_turn_summaries(records))

        assert len(summaries) == 2
        assert summaries[0].turn_key != summaries[1].turn_key
        assert summaries[0].warnings == ()
        assert summaries[1].warnings == (TURN_KEY_COLLISION_WARNING,)
        expected = "turn-" + hashlib.sha256(b"ses-1\0user\0dup-1\0" + canonical_json(2)).hexdigest()
        assert summaries[1].turn_key == expected


class TestTurnSummariesStream:
    def test_summaries_arrive_after_consuming_only_the_completed_turn(self) -> None:
        records = [
            _user_prompt("user-1", seq=1),
            _assistant_result("asst-1", seq=2),
            _user_prompt("user-2", seq=3),
            _user_prompt("user-3", seq=4),
        ]
        consumed = 0

        def stream() -> Iterator[SourceRecord]:
            nonlocal consumed
            for record in records:
                consumed += 1
                yield record

        summaries = iter_turn_summaries(stream())
        assert next(summaries).first_user_source_id == "user-1"
        # Turn one closed exactly when the second prompt arrived: one
        # record of lookahead, never the whole stream.
        assert consumed == 3
        assert next(summaries).first_user_source_id == "user-2"
        assert next(summaries).first_user_source_id == "user-3"
        assert consumed == 4

    def test_lookahead_stays_bounded_across_many_turns(self) -> None:
        records = [_user_prompt(f"user-{index}", seq=index) for index in range(1, 11)]
        consumed = 0

        def stream() -> Iterator[SourceRecord]:
            nonlocal consumed
            for record in records:
                consumed += 1
                yield record

        seen = 0
        for _summary in iter_turn_summaries(stream()):
            seen += 1
            assert consumed <= seen + 1

        assert seen == 10
        assert consumed == 10

    def test_summary_fingerprint_is_deterministic_and_record_sensitive(self) -> None:
        base = [_user_prompt("user-1", seq=1), _assistant_result("asst-1", seq=2)]
        changed = [
            _user_prompt("user-1", seq=1),
            _record(
                "asst-1",
                source_type="assistant",
                classification="assistant",
                seq=2,
                raw_json='{"content": [{"type": "text", "text": "changed reply"}]}',
            ),
        ]

        first = next(iter(iter_turn_summaries(base)))
        repeated = next(iter(iter_turn_summaries(base)))
        different = next(iter(iter_turn_summaries(changed)))

        assert first.source_fingerprint == repeated.source_fingerprint
        assert first.source_fingerprint != different.source_fingerprint
        assert len(first.source_fingerprint) == 64

    def test_summary_fingerprint_hashes_the_shared_task_4a_record_form(self) -> None:
        record = _user_prompt("user-1")

        # Task 4A's machinery canonicalizes the same record through the
        # effective-turn cursor (identity redaction: the verbatim form).
        replay = FakeReplayTurn(
            session_header=fake_header("ses-1"),
            summary=fake_summary(session_id="ses-1"),
            turn_records=(record,),
        )
        chunks = list(build_effective_turn(replay, IdentityRedactor()).iter_canonical_records())

        # Cross-check: the no-redactor record form the turn-local
        # fingerprint hashes and the identity-redactor form Task 4A
        # hashes are the same canonical record.
        assert chunks == [canonical_json(canonical_record_value(record))]

        digest = FramedDigest("turn-records-v1")
        for chunk in chunks:
            digest.update(chunk)

        (summary,) = iter_turn_summaries([record])

        # The turn-local fingerprint is exactly the shared framed digest
        # over Task 4A's own canonical record chunks — the two modules
        # cannot drift apart.
        assert summary.source_fingerprint == digest.hexdigest()

    def test_summary_carries_no_prompt_response_or_tool_content(self) -> None:
        records = [
            _user_prompt("user-1", seq=1, raw_json='{"content": [{"type": "text", "text": "SENTINEL-PROMPT"}]}'),
            _record(
                "asst-1",
                source_type="assistant",
                classification="assistant",
                seq=2,
                raw_json='{"content": [{"type": "text", "text": "SENTINEL-REPLY"},'
                ' {"type": "tool", "name": "SENTINEL-TOOL", "code": "SENTINEL-SOURCE-CODE"}]}',
            ),
        ]

        summaries = list(iter_turn_summaries(records))

        assert summaries
        for summary in summaries:
            payload = repr(summary) + repr(asdict(summary))
            for sentinel in ("SENTINEL-PROMPT", "SENTINEL-REPLY", "SENTINEL-TOOL", "SENTINEL-SOURCE-CODE"):
                assert sentinel not in payload
            # The summary still carries identifiers, counts, and hashes.
            assert summary.turn_key.startswith("turn-")
            assert summary.source_fingerprint is not None


class TestIterTurnRecords:
    def test_yields_exactly_the_turns_records_in_order(self) -> None:
        records = [
            _user_prompt("user-1", seq=1),
            _assistant_result("asst-1", seq=2),
            _user_prompt("user-2", seq=3),
            _assistant_result("asst-2", seq=4),
        ]
        summaries = list(iter_turn_summaries(records))

        assert _identities(iter_turn_records(records, summaries[1])) == ["user-2", "asst-2"]
        assert _identities(iter_turn_records(records, summaries[0])) == ["user-1", "asst-1"]

    def test_unknown_turn_key_fails_loudly(self) -> None:
        records = [_user_prompt("user-1", seq=1)]
        (summary,) = iter_turn_summaries(records)
        stranger = replace(summary, turn_key="turn-does-not-exist")

        with pytest.raises(ValueError, match="turn key"):
            list(iter_turn_records(records, stranger))

    def test_turn_index_mismatch_fails_loudly(self) -> None:
        records = [_user_prompt("user-1", seq=1)]
        (summary,) = iter_turn_summaries(records)
        renumbered = replace(summary, turn_index=9)

        with pytest.raises(ValueError, match="turn index"):
            list(iter_turn_records(records, renumbered))


# ---------------------------------------------------------------------------
# Scripted-snapshot pipeline: fresh cursors and bounded root loaders
# ---------------------------------------------------------------------------


def _scripted_session(session_id: str = "ses-1") -> ScriptedSession:
    header = SessionHeader(
        session_id=session_id,
        adapter_generation="current",
        time_created=normalize_timestamp(1_772_000_000_000, "ms"),
    )
    return ScriptedSession(
        header=header,
        canonical_session_row=CanonicalRecordEnvelope(
            decoded={"title": "Synthetic session"},
            raw_text=None,
            raw_columns={"id": session_id},
            warnings=(),
        ),
    )


def _scripted(record: SourceRecord) -> ScriptedRecord:
    order = record.sequence_value if record.sequence_value is not None else 0
    return ScriptedRecord(
        record=record,
        order_key=(order, record.source_id or record.source_identity or ""),
        canonical_record=CanonicalRecordEnvelope(
            decoded=record.decoded_value,
            raw_text=record.raw_json,
            raw_columns={},
            warnings=record.warnings,
        ),
        raw_permitted_fields={},
    )


@contextmanager
def _scripted_snapshot(
    tmp_path: Path, records: Sequence[SourceRecord]
) -> Iterator[tuple[SourceSnapshot, NormalizedSnapshotAdapter]]:
    """Snapshot a scripted record stream; written order equals the given order."""
    source = tmp_path / f"source-{next(_SOURCE_COUNTER)}.db"
    connection = build_fixture_db(source, "current")
    connection.close()
    exporter = ScriptedExporter(
        generation="current",
        sessions=[_scripted_session()],
        records=[_scripted(record) for record in records],
    )
    with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
        yield snapshot, NormalizedSnapshotAdapter(snapshot)


def _scripted_turn_records() -> list[SourceRecord]:
    """One session: a pre-prompt system record, two user turns, one scoped record."""
    return [
        _record(
            "sys-1",
            source_type="system",
            classification="system",
            seq=1,
            raw_json='{"content": [{"type": "text", "text": "SENTINEL-SYSTEM-NOTICE"}]}',
        ),
        _record("user-1", seq=2, raw_json='{"content": [{"type": "text", "text": "SENTINEL-PROMPT"}]}'),
        _record(
            "user-1#content-0",
            source_type="text",
            classification=None,
            seq=2,
            raw_json='{"type": "text", "text": "SENTINEL-PROMPT"}',
            parent_source_id="user-1",
            derived_child_ordinal=0,
        ),
        _record(
            "asst-1",
            source_type="assistant",
            classification="assistant",
            seq=3,
            raw_json='{"content": [{"type": "text", "text": "SENTINEL-REPLY"},'
            ' {"type": "tool", "name": "SENTINEL-TOOL"}]}',
        ),
        _record(
            "asst-1#content-0",
            source_type="text",
            classification=None,
            seq=3,
            raw_json='{"type": "text", "text": "SENTINEL-REPLY"}',
            parent_source_id="asst-1",
            derived_child_ordinal=0,
        ),
        _record(
            "asst-1#content-1",
            source_type="tool",
            classification=None,
            seq=3,
            raw_json='{"type": "tool", "name": "SENTINEL-TOOL"}',
            parent_source_id="asst-1",
            derived_child_ordinal=1,
        ),
        _record("user-2", seq=4, raw_json='{"content": [{"type": "text", "text": "SENTINEL-PROMPT-2"}]}'),
        _record(
            "pending-1",
            source_type="tool",
            classification=None,
            table="session_pending",
            seq=None,
            raw_json='{"name": "SENTINEL-PENDING-TOOL"}',
        ),
    ]


class TestReplayTurnOverScriptedSnapshot:
    def test_build_replay_turn_satisfies_the_owned_protocol(self, tmp_path: Path) -> None:
        records = _scripted_turn_records()
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-1")))
            summary = summaries[0]

            turn = build_replay_turn(snapshot, header, summary)

        assert isinstance(turn, ReplayTurn)
        assert turn.session_header is header
        assert turn.summary is summary

    def test_iter_records_yields_the_turns_records_with_fresh_cursors(self, tmp_path: Path) -> None:
        records = _scripted_turn_records()
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-1")))
            summary = summaries[0]
            turn = build_replay_turn(snapshot, header, summary)

            first = _identities(turn.iter_records())
            second = _identities(turn.iter_records())

        assert first == second == ["user-1", "user-1#content-0", "asst-1", "asst-1#content-0", "asst-1#content-1"]

    def test_iter_session_records_yields_scoped_records(self, tmp_path: Path) -> None:
        records = _scripted_turn_records()
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-1")))
            summary = summaries[0]
            turn = build_replay_turn(snapshot, header, summary)

            assert _identities(turn.iter_session_records()) == ["pending-1"]

    def test_iter_unassigned_records_yields_other_turns_and_unclaimed_records(self, tmp_path: Path) -> None:
        records = _scripted_turn_records()
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-1")))
            summary = summaries[0]
            turn = build_replay_turn(snapshot, header, summary)

            assert _identities(turn.iter_unassigned_records()) == ["sys-1", "user-2"]

    def test_cursor_union_is_the_complete_session_payload(self, tmp_path: Path) -> None:
        records = _scripted_turn_records()
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-1")))
            summary = summaries[0]
            turn = build_replay_turn(snapshot, header, summary)

            union = [
                *_identities(turn.iter_records()),
                *_identities(turn.iter_session_records()),
                *_identities(turn.iter_unassigned_records()),
            ]

        assert sorted(union) == sorted(_identities(records))
        assert len(union) == len(records)

    def test_load_root_input_returns_the_first_user_decoded_value(self, tmp_path: Path) -> None:
        records = _scripted_turn_records()
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-1")))
            summary = summaries[0]
            turn = build_replay_turn(snapshot, header, summary)

            loaded = turn.load_root_input(10_000)

        expected = {"content": [{"type": "text", "text": "SENTINEL-PROMPT"}]}
        assert isinstance(loaded, BoundedRootValue)
        assert loaded.value == expected
        assert loaded.payload_too_large is False
        assert loaded.size_bytes == len(canonical_json(expected))

    def test_load_root_input_rejects_values_over_the_bound(self, tmp_path: Path) -> None:
        records = _scripted_turn_records()
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-1")))
            summary = summaries[0]
            turn = build_replay_turn(snapshot, header, summary)

            loaded = turn.load_root_input(8)

        expected_size = len(canonical_json({"content": [{"type": "text", "text": "SENTINEL-PROMPT"}]}))
        assert isinstance(loaded, BoundedRootValue)
        assert loaded.value is None
        assert loaded.payload_too_large is True
        assert loaded.size_bytes == expected_size
        assert loaded.max_bytes == 8

    def test_load_root_output_combines_user_visible_parts_of_the_final_result(self, tmp_path: Path) -> None:
        records = _scripted_turn_records()
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-1")))
            summary = summaries[0]
            turn = build_replay_turn(snapshot, header, summary)

            loaded = turn.load_root_output(10_000)

        expected = [{"type": "text", "text": "SENTINEL-REPLY"}]
        assert isinstance(loaded, BoundedRootValue)
        assert loaded.value == expected
        assert loaded.payload_too_large is False
        assert loaded.size_bytes == len(canonical_json(expected))

    def test_load_root_output_rejects_combined_values_over_the_bound(self, tmp_path: Path) -> None:
        records = _scripted_turn_records()
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-1")))
            summary = summaries[0]
            turn = build_replay_turn(snapshot, header, summary)

            loaded = turn.load_root_output(8)

        expected_size = len(canonical_json([{"type": "text", "text": "SENTINEL-REPLY"}]))
        assert isinstance(loaded, BoundedRootValue)
        assert loaded.value is None
        assert loaded.payload_too_large is True
        assert loaded.size_bytes == expected_size

    def test_load_root_output_uses_the_result_value_without_parts(self, tmp_path: Path) -> None:
        records = [
            _user_prompt("user-1", seq=1),
            _record(
                "asst-1",
                source_type="assistant",
                classification="assistant",
                seq=2,
                raw_json='{"content": "SENTINEL-WHOLE-REPLY"}',
            ),
        ]
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            (summary,) = iter_turn_summaries(adapter.iter_records("ses-1"))
            turn = build_replay_turn(snapshot, header, summary)

            loaded = turn.load_root_output(10_000)

        assert isinstance(loaded, BoundedRootValue)
        assert loaded.value == {"content": "SENTINEL-WHOLE-REPLY"}
        assert loaded.payload_too_large is False

    def test_load_root_output_returns_none_without_a_result(self, tmp_path: Path) -> None:
        records = [_user_prompt("user-1", seq=1), _user_prompt("user-2", seq=2)]
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-1")))
            turn = build_replay_turn(snapshot, header, summaries[0])

            assert turn.load_root_output(10_000) is None

    def test_mismatched_summary_session_fails_loudly(self, tmp_path: Path) -> None:
        records = [_user_prompt("user-1", seq=1)]
        with _scripted_snapshot(tmp_path, records) as (snapshot, adapter):
            (header,) = adapter.iter_session_headers()
            (summary,) = iter_turn_summaries(adapter.iter_records("ses-1"))
            stranger = replace(summary, session_id="ses-other")

            with pytest.raises(ValueError, match="session"):
                build_replay_turn(snapshot, header, stranger)


# ---------------------------------------------------------------------------
# Real-adapter pipelines over the sanitized synthetic fixture databases
# ---------------------------------------------------------------------------


def _header_of(adapter: NormalizedSnapshotAdapter, session_id: str) -> SessionHeader:
    """The session header of ``session_id`` from a multi-session fixture."""
    return next(header for header in adapter.iter_session_headers() if header.session_id == session_id)


def _current_exporter(connection: sqlite3.Connection) -> CurrentAdapter:
    return CurrentAdapter(connection, CURRENT_CATALOG_V1)


def _legacy_exporter(connection: sqlite3.Connection) -> LegacyAdapter:
    return LegacyAdapter(connection, LEGACY_CATALOG_V1)


@contextmanager
def _exported(
    tmp_path: Path,
    script: str,
    make_exporter: Callable[[sqlite3.Connection], SourceExporter],
) -> Iterator[tuple[SourceSnapshot, NormalizedSnapshotAdapter]]:
    """Export a script-built source database and yield (snapshot, adapter)."""
    source = tmp_path / f"source-{next(_SOURCE_COUNTER)}.db"
    connection = sqlite3.connect(source)
    try:
        connection.executescript(script)
        exporter = make_exporter(connection)
    finally:
        connection.close()
    with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
        yield snapshot, NormalizedSnapshotAdapter(snapshot)


class TestCurrentPipelineTurns:
    def test_current_fixture_session_reconstructs_expected_turns(self, tmp_path: Path) -> None:
        with _exported(tmp_path, fixture_script("current"), _current_exporter) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-current-001"))
            summaries = list(iter_turn_summaries(records))

        assert [summary.turn_index for summary in summaries] == [1, 2]
        assert [summary.first_user_source_id for summary in summaries] == ["msg-current-001", "msg-current-003"]
        # Turn one ends at the assistant reply; the system/synthetic row of
        # turn two is not an assistant result.
        assert summaries[0].root_result_source_id == "msg-current-002"
        assert summaries[0].terminal_outcome == ASSISTANT_RESULT_OUTCOME
        assert summaries[1].root_result_source_id is None
        assert summaries[1].terminal_outcome == NO_ASSISTANT_RESULT_OUTCOME
        # Each turn: the prompt message, its derived content child, the
        # next message, and that message's child.
        assert [summary.record_count for summary in summaries] == [4, 4]
        assert [summary.user_record_count for summary in summaries] == [1, 1]
        assert [summary.assistant_record_count for summary in summaries] == [1, 0]

    def test_current_turn_records_replay_parent_then_children(self, tmp_path: Path) -> None:
        with _exported(tmp_path, fixture_script("current"), _current_exporter) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-current-001"))
            summaries = list(iter_turn_summaries(records))

        assert _identities(iter_turn_records(records, summaries[0])) == [
            "msg-current-001",
            "msg-current-001#content-0",
            "msg-current-002",
            "msg-current-002#content-0",
        ]

    def test_current_turn_root_output_combines_text_parts(self, tmp_path: Path) -> None:
        with _exported(tmp_path, fixture_script("current"), _current_exporter) as (snapshot, adapter):
            header = _header_of(adapter, "ses-current-001")
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-current-001")))
            summary = summaries[0]
            turn = build_replay_turn(snapshot, header, summary)

            loaded = turn.load_root_output(10_000)

        assert isinstance(loaded, BoundedRootValue)
        assert loaded.value == [{"type": "text", "text": "synthetic assistant reply one"}]

    def test_current_scoped_records_stay_out_of_turns(self, tmp_path: Path) -> None:
        with _exported(tmp_path, fixture_script("current"), _current_exporter) as (snapshot, adapter):
            header = _header_of(adapter, "ses-current-001")
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-current-001")))
            summary = summaries[0]
            turn = build_replay_turn(snapshot, header, summary)

            scoped = _identities(turn.iter_session_records())

        # pending, inbox, the referenced instruction blob (explicit hash
        # identity), and the id-less instruction entry and state rows
        # (fallback identities) — five session-scoped records in all.
        assert len(scoped) == 5
        assert {"pending-001", "inbox-001", "synth-blob-hash-001"} <= set(scoped)

    def test_current_unassigned_cursor_covers_other_turn_records(self, tmp_path: Path) -> None:
        with _exported(tmp_path, fixture_script("current"), _current_exporter) as (snapshot, adapter):
            header = _header_of(adapter, "ses-current-001")
            summaries = list(iter_turn_summaries(adapter.iter_records("ses-current-001")))
            summary = summaries[0]
            turn = build_replay_turn(snapshot, header, summary)

            unassigned = _identities(turn.iter_unassigned_records())

        assert sorted(unassigned) == [
            "msg-current-003",
            "msg-current-003#content-0",
            "msg-current-004",
            "msg-current-004#content-0",
        ]

    def test_fork_session_turns_contain_only_fork_local_rows(self, tmp_path: Path) -> None:
        with _exported(tmp_path, fixture_script("current"), _current_exporter) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-current-002"))
            summaries = list(iter_turn_summaries(records))

        assert [summary.first_user_source_id for summary in summaries] == ["msg-current-101"]
        assert _identities(iter_turn_records(records, summaries[0])) == [
            "msg-current-101",
            "msg-current-101#content-0",
        ]


class TestLegacyPipelineTurns:
    def test_legacy_turn_contains_parent_message_then_its_parts(self, tmp_path: Path) -> None:
        with _exported(tmp_path, fixture_script("legacy"), _legacy_exporter) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-001"))
            summaries = list(iter_turn_summaries(records))

        (summary,) = summaries
        assert _identities(iter_turn_records(records, summary)) == [
            "msg-legacy-001",
            "part-legacy-001",
            "part-legacy-002",
            "part-legacy-006",
            "msg-legacy-002",
            "part-legacy-003",
            "part-legacy-004",
            "part-legacy-005",
        ]
        assert summary.record_count == 8
        # The user message plus its three parts classify user; the assistant
        # message and its text part classify assistant (the step parts are
        # synthetic).
        assert summary.user_record_count == 4
        assert summary.assistant_record_count == 2

    def test_legacy_root_result_and_combined_output(self, tmp_path: Path) -> None:
        with _exported(tmp_path, fixture_script("legacy"), _legacy_exporter) as (snapshot, adapter):
            header = _header_of(adapter, "ses-legacy-001")
            (summary,) = iter_turn_summaries(adapter.iter_records("ses-legacy-001"))
            turn = build_replay_turn(snapshot, header, summary)

            loaded = turn.load_root_output(10_000)

        assert summary.root_result_source_id == "msg-legacy-002"
        assert summary.terminal_outcome == ASSISTANT_RESULT_OUTCOME
        assert isinstance(loaded, BoundedRootValue)
        # Only the text part is user-visible; the step-start/step-finish
        # parts stay separately represented but out of the combined value.
        assert loaded.value == [{"type": "text", "text": "synthetic legacy assistant text"}]

    def test_interleaved_part_timestamps_stay_attached_in_replay_order(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO message VALUES ('msg-parent', 'ses-legacy-001', 1771000050000, 1771000050000,"
            ' \'{"role": "user", "content": "parent"}\');'
            "INSERT INTO message VALUES ('msg-next-user', 'ses-legacy-001', 1771000060000, 1771000060000,"
            ' \'{"role": "user", "content": "next prompt"}\');'
            "INSERT INTO part VALUES ('part-late', 'msg-parent', 'ses-legacy-001', 1771000070000, 1771000070000,"
            ' \'{"type": "text", "text": "late part of the parent"}\');'
        )
        script = fixture_script("legacy") + extra
        with _exported(tmp_path, script, _legacy_exporter) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-001"))
            summaries = list(iter_turn_summaries(records))

        # The pure record_sort_key would order the late part
        # (t=...070000) after msg-next-user (t=...060000), detaching it
        # from its parent. The adapter emission order — persisted as the
        # snapshot's written order and replayed by the cursors — is the
        # authoritative order for parent-attached streams: the part
        # stays immediately after its parent, and that order drives the
        # turn boundaries (turn two owns the late part, not turn three).
        assert len(summaries) == 3
        assert _identities(iter_turn_records(records, summaries[1])) == ["msg-parent", "part-late"]
        assert _identities(iter_turn_records(records, summaries[2])) == ["msg-next-user"]

    def test_fork_and_compacted_session_reconstructs_its_own_turn(self, tmp_path: Path) -> None:
        with _exported(tmp_path, fixture_script("legacy"), _legacy_exporter) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-002"))
            summaries = list(iter_turn_summaries(records))

        # The forked, compacted session still reconstructs its own single
        # turn from its persisted child rows only.
        assert [summary.first_user_source_id for summary in summaries] == ["msg-legacy-101"]
        assert _identities(iter_turn_records(records, summaries[0])) == ["msg-legacy-101"]
        assert summaries[0].terminal_outcome == NO_ASSISTANT_RESULT_OUTCOME

    def test_session_without_creation_time_still_reconstructs(self, tmp_path: Path) -> None:
        with _exported(tmp_path, fixture_script("legacy"), _legacy_exporter) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-003"))
            summaries = list(iter_turn_summaries(records))

        assert [summary.first_user_source_id for summary in summaries] == ["msg-legacy-301"]

    def test_identical_missing_id_messages_reconstruct_distinct_turns(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "identical no-id prompt"}\');'
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "identical no-id prompt"}\');'
        )
        script = fixture_script("legacy") + extra
        with _exported(tmp_path, script, _legacy_exporter) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-001"))
            summaries = list(iter_turn_summaries(records))

        # The fixture prompt plus the two identical no-ID prompts.
        assert len(summaries) == 3
        fallback_turns = [summary for summary in summaries if TURN_KEY_FALLBACK_WARNING in summary.warnings]
        assert len(fallback_turns) == 2
        assert len({summary.turn_key for summary in fallback_turns}) == 2
        # The no-ID turns carry their adapter fallback identities.
        assert all(len(summary.first_user_source_id or "") == 64 for summary in fallback_turns)

    def test_missing_id_turn_keys_are_stable_across_repeated_snapshots(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "identical no-id prompt"}\');'
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "identical no-id prompt"}\');'
        )
        script = fixture_script("legacy") + extra
        with _exported(tmp_path, script, _legacy_exporter) as (_snapshot, adapter):
            first = [summary.turn_key for summary in iter_turn_summaries(adapter.iter_records("ses-legacy-001"))]
        with _exported(tmp_path, script, _legacy_exporter) as (_snapshot, adapter):
            second = [summary.turn_key for summary in iter_turn_summaries(adapter.iter_records("ses-legacy-001"))]

        assert first == second

    def test_ambiguous_missing_parent_parts_stay_unassigned(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "identical"}\');'
            "INSERT INTO message VALUES (NULL, 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "identical"}\');'
            "INSERT INTO part VALUES (NULL, NULL, 'ses-legacy-001', 1771000016000, 1771000016000,"
            ' \'{"type": "text", "text": "shared orphan part"}\');'
        )
        script = fixture_script("legacy").replace("message_id TEXT NOT NULL", "message_id TEXT") + extra
        with _exported(tmp_path, script, _legacy_exporter) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-001"))
            summaries = list(iter_turn_summaries(records))

        # Ambiguous missing-ID parents remain distinct; their part cannot
        # safely be attributed to either parent.
        assert len(summaries) == 3
        (fallback_turn, _other) = [summary for summary in summaries if TURN_KEY_FALLBACK_WARNING in summary.warnings]
        turn_records = list(iter_turn_records(records, fallback_turn))
        assert len(turn_records) == 1
        assert "ambiguous_missing_message_id" in turn_records[0].warnings
        assert turn_records[0].source_id is None

    def test_distinct_missing_id_parts_stay_separate_turn_records(self, tmp_path: Path) -> None:
        extra = (
            "INSERT INTO message VALUES ('msg-no-id-parent', 'ses-legacy-001', 1771000015000, 1771000015000,"
            ' \'{"role": "user", "content": "parent"}\');'
            "INSERT INTO part VALUES (NULL, 'msg-no-id-parent', 'ses-legacy-001', 1771000016000, 1771000016000,"
            ' \'{"type": "text", "text": "duplicate part"}\');'
            "INSERT INTO part VALUES (NULL, 'msg-no-id-parent', 'ses-legacy-001', 1771000016000, 1771000016000,"
            ' \'{"type": "text", "text": "duplicate part"}\');'
        )
        script = fixture_script("legacy") + extra
        with _exported(tmp_path, script, _legacy_exporter) as (_snapshot, adapter):
            records = list(adapter.iter_records("ses-legacy-001"))
            summaries = list(iter_turn_summaries(records))

        parent_turn = next(summary for summary in summaries if summary.first_user_source_id == "msg-no-id-parent")
        turn_records = list(iter_turn_records(records, parent_turn))
        parts = [record for record in turn_records if record.parent_source_id == "msg-no-id-parent"]
        # Distinct join projection keys keep both identical no-ID parts.
        assert len(parts) == 2
        assert len({record.join_projection_key for record in parts}) == 2
        assert turn_records[0].source_id == "msg-no-id-parent"
