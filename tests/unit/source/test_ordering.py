"""Unit tests for the deterministic ordering keys.

These tests pin the ordering contract the live adapters already express
in SQL: within one authoritative sequence domain, records order by
``(sequence, fixed tie rank, stable source ID)``; across streams
without a shared domain, they order by ``(normalized timestamp
present/value, fixed tie rank, stable source ID)``; sessions order by
``(creation time present/value, session_id)``. Sequence values from
unrelated streams are never compared, and records without a normalized
timestamp sort after every timestamped record of the same shape.
"""

from __future__ import annotations

import pytest

from opencode_mlflow.source.catalog import TIE_RANKS
from opencode_mlflow.source.model import SessionHeader, SourceRecord, normalize_timestamp
from opencode_mlflow.source.ordering import (
    SOURCE_TYPE_RANKS,
    RecordOrderContext,
    record_sort_key,
    session_sort_key,
    source_type_rank,
)

pytestmark = pytest.mark.unit

CURRENT_MESSAGE_DOMAIN = RecordOrderContext("session_message.seq", authoritative=True)
LEGACY_NO_DOMAIN = RecordOrderContext(None, authoritative=False)


def _record(
    source_id: str | None = "rec-1",
    *,
    table: str = "session_message",
    source_type: str = "user",
    seq: int | None = 1,
    sequence_domain: str | None = "session_message.seq",
    timestamp_ms: int | None = 1_772_000_000,
    derived_child_ordinal: int | None = None,
    parent_source_id: str | None = None,
    source_identity: str | None = None,
) -> SourceRecord:
    """One normalized record carrying the identity/ordering fields adapters set."""
    return SourceRecord(
        source_table=table,
        source_type=source_type,
        source_id=source_id,
        session_id="ses-1",
        sequence_domain=sequence_domain,
        sequence_value=seq,
        parent_source_id=parent_source_id,
        derived_child_ordinal=derived_child_ordinal,
        source_identity=source_identity,
        timestamp=normalize_timestamp(timestamp_ms, "ms") if timestamp_ms is not None else None,
        classification=None,
    )


def _header(session_id: str, created_ms: int | None) -> SessionHeader:
    return SessionHeader(
        session_id=session_id,
        adapter_generation="current",
        time_created=normalize_timestamp(created_ms, "ms") if created_ms is not None else None,
    )


def _ordered(records: list[SourceRecord], context: RecordOrderContext) -> list[str]:
    """Sort records under ``context`` and return their stable identities."""
    ranked = sorted(records, key=lambda record: record_sort_key(record, context))
    return [record.source_id or record.source_identity or "" for record in ranked]


class TestSourceTypeRanks:
    def test_rank_table_is_the_fixed_catalog_table(self) -> None:
        assert dict(SOURCE_TYPE_RANKS) == {
            "message_container": 10,
            "content_part": 20,
            "lifecycle_event": 30,
            "scoped_record": 40,
            "unknown_record": 90,
        }
        assert dict(SOURCE_TYPE_RANKS) == TIE_RANKS

    @pytest.mark.parametrize(
        ("table", "derived_child", "expected"),
        [
            ("session_message", False, 10),
            ("message", False, 10),
            ("part", False, 20),
            ("session_message", True, 20),
            ("session_pending", False, 30),
            ("session_inbox", False, 30),
            ("todo", False, 40),
            ("instruction_entry", False, 40),
            ("not_a_catalogued_table", False, 90),
        ],
    )
    def test_source_type_rank_maps_tables_to_fixed_ranks(self, table: str, derived_child: bool, expected: int) -> None:
        assert source_type_rank(table, derived_child=derived_child) == expected


class TestAuthoritativeSequenceDomainOrdering:
    def test_records_order_by_sequence_then_rank_then_id(self) -> None:
        records = [
            _record("msg-b", seq=2),
            _record("msg-a", seq=1),
            _record("msg-c", seq=10),
            _record("msg-b#content-0", seq=2, derived_child_ordinal=0),
        ]

        # Sequence is primary; the fixed rank breaks the seq-2 tie (the
        # message container before its derived content item).
        assert _ordered(records, CURRENT_MESSAGE_DOMAIN) == ["msg-a", "msg-b", "msg-b#content-0", "msg-c"]

    def test_equal_sequence_ties_on_rank_then_stable_id(self) -> None:
        # Same sequence: the message container outranks content parts, and
        # the stable source ID breaks remaining ties.
        records = [
            _record("part-z", table="part", seq=5, sequence_domain=None),
            _record("msg-b", seq=5),
            _record("msg-a", seq=5),
            _record("part-a", table="part", seq=5, sequence_domain=None),
        ]

        assert _ordered(records, CURRENT_MESSAGE_DOMAIN) == ["msg-a", "msg-b", "part-a", "part-z"]

    def test_parent_precedes_derived_children_ordered_by_child_ordinal(self) -> None:
        # Derived children share the parent's sequence; the fixed ranks put
        # the container first and the child ordinal (numeric, not the
        # lexicographic derived ID) orders the children.
        records = [
            _record("msg-1#content-10", seq=1, derived_child_ordinal=10),
            _record("msg-1#content-2", seq=1, derived_child_ordinal=2),
            _record("msg-1", seq=1),
        ]

        assert _ordered(records, CURRENT_MESSAGE_DOMAIN) == [
            "msg-1",
            "msg-1#content-2",
            "msg-1#content-10",
        ]

    def test_missing_sequence_sorts_after_sequenced_records(self) -> None:
        records = [
            _record("msg-late", seq=99),
            _record("msg-unsequenced", seq=None),
        ]

        assert _ordered(records, CURRENT_MESSAGE_DOMAIN) == ["msg-late", "msg-unsequenced"]

    def test_records_outside_the_domain_never_contribute_their_sequence(self) -> None:
        # A record from an unrelated stream (a legacy message domain, or no
        # domain at all) sorts in the absent-sequence group even when its
        # own sequence value would order it among the domain's records.
        records = [
            _record("foreign-seq", seq=999_999, sequence_domain="message.time_created"),
            _record("domain-seq", seq=1),
            _record("no-domain", seq=888_888, sequence_domain=None),
        ]

        assert _ordered(records, CURRENT_MESSAGE_DOMAIN) == ["domain-seq", "foreign-seq", "no-domain"]

    def test_non_authoritative_domain_context_falls_back_to_timestamps(self) -> None:
        records = [
            _record("late", seq=1, timestamp_ms=1_772_000_020_000),
            _record("early", seq=999, timestamp_ms=1_772_000_010_000),
        ]
        context = RecordOrderContext("session_message.seq", authoritative=False)

        assert _ordered(records, context) == ["early", "late"]


class TestCrossStreamTimestampOrdering:
    def test_records_without_a_shared_domain_order_by_timestamp(self) -> None:
        records = [
            _record("late", table="message", seq=None, sequence_domain=None, timestamp_ms=1_772_000_030_000),
            _record("early", table="part", seq=None, sequence_domain=None, timestamp_ms=1_772_000_010_000),
            _record("middle", table="message", seq=None, sequence_domain=None, timestamp_ms=1_772_000_020_000),
        ]

        assert _ordered(records, LEGACY_NO_DOMAIN) == ["early", "middle", "late"]

    def test_missing_timestamp_sorts_after_timestamped_records(self) -> None:
        records = [
            _record("untimed", table="message", seq=None, sequence_domain=None, timestamp_ms=None),
            _record("timed", table="message", seq=None, sequence_domain=None, timestamp_ms=1_772_000_010_000),
        ]

        assert _ordered(records, LEGACY_NO_DOMAIN) == ["timed", "untimed"]

    def test_equal_timestamps_tie_on_rank_then_stable_id(self) -> None:
        records = [
            _record("part-b", table="part", seq=None, sequence_domain=None, timestamp_ms=1_772_000_000),
            _record("msg-a", table="message", seq=None, sequence_domain=None, timestamp_ms=1_772_000_000),
            _record("part-a", table="part", seq=None, sequence_domain=None, timestamp_ms=1_772_000_000),
        ]

        # The message container (rank 10) precedes the content parts
        # (rank 20) at the same instant — the ONLY case where the pure
        # key preserves parent/child attachment.
        assert _ordered(records, LEGACY_NO_DOMAIN) == ["msg-a", "part-a", "part-b"]

    def test_interleaved_part_timestamps_detach_in_the_pure_key(self) -> None:
        # Documented limitation: a per-record key cannot express
        # parent-key inheritance. When a part's own timestamp falls
        # after another message's, the pure key orders the part after
        # that message — detached from its parent. The adapter emission
        # order (persisted as the snapshot's written order and replayed
        # by the cursors) is the authoritative order for parent-attached
        # streams; test_turns.py pins that replay side.
        records = [
            _record("msg-parent", table="message", seq=None, sequence_domain=None, timestamp_ms=1_772_000_100),
            _record("msg-user", table="message", seq=None, sequence_domain=None, timestamp_ms=1_772_000_200),
            _record(
                "part-late",
                table="part",
                seq=None,
                sequence_domain=None,
                timestamp_ms=1_772_000_300,
                parent_source_id="msg-parent",
            ),
        ]

        assert _ordered(records, LEGACY_NO_DOMAIN) == ["msg-parent", "msg-user", "part-late"]

    def test_fallback_identity_breaks_id_ties_for_missing_id_records(self) -> None:
        records = [
            _record(None, table="message", seq=None, sequence_domain=None, source_identity="bbb"),
            _record(None, table="message", seq=None, sequence_domain=None, source_identity="aaa"),
        ]

        assert _ordered(records, LEGACY_NO_DOMAIN) == ["aaa", "bbb"]


class TestSessionSortKey:
    def test_timestamped_sessions_sort_before_missing_creation_time(self) -> None:
        records = [
            _header("ses-late", 1_772_000_020_000),
            _header("ses-missing", None),
            _header("ses-early", 1_772_000_010_000),
        ]

        ordered = sorted(records, key=session_sort_key)

        assert [header.session_id for header in ordered] == ["ses-early", "ses-late", "ses-missing"]

    def test_sessions_tie_on_creation_time_by_session_id(self) -> None:
        records = [
            _header("ses-b", 1_772_000_000_000),
            _header("ses-a", 1_772_000_000_000),
        ]

        ordered = sorted(records, key=session_sort_key)

        assert [header.session_id for header in ordered] == ["ses-a", "ses-b"]

    def test_missing_creation_time_sessions_tie_by_session_id(self) -> None:
        records = [
            _header("ses-b", None),
            _header("ses-a", None),
        ]

        ordered = sorted(records, key=session_sort_key)

        assert [header.session_id for header in ordered] == ["ses-a", "ses-b"]
