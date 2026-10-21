"""Deterministic ordering keys for normalized source sessions and records.

Task 4 shipped the two pieces the live exporters need for export
ordering: the session ordering key and the fixed source-type rank table.
Task 5 completes this module with ``record_sort_key``/
``RecordOrderContext`` and turn reconstruction on top of the same
constants — the ranks are catalog data, never an incidental database
query order.

The record ordering contract (the design spec's ordering rules, pinned
by ``tests/unit/source/test_ordering.py``):

- within ONE authoritative sequence domain, records order by
  ``(sequence, fixed tie rank, stable source ID)`` — sequence values
  from unrelated streams are never compared, so a record outside the
  context's domain contributes no sequence value and sorts with the
  absent group;
- across streams without a shared domain, records order by
  ``(normalized timestamp present/value, fixed tie rank, stable source
  ID)`` — records without a normalized timestamp sort after every
  timestamped record (their warnings already carry the adapters'
  malformed/absent markers);
- parent/child attachment (legacy parts, derived content items) is NOT
  expressible in a per-record key: the exporters emit each part
  immediately after its parent (parent order plus part
  sequence/timestamp/ID), and that emission order — persisted as the
  snapshot's written order and replayed by the cursors — is the
  authoritative order for parent-attached streams. ``record_sort_key``
  preserves attachment only through the fixed ranks (a part sorts
  after its same-timestamp parent) and the derived child ordinal; when
  a part's own timestamp falls after another message's, the pure key
  orders the part after that message, detached from its parent.

The write and replay phases consume records in this adapter-provided
source order (the normalized snapshot preserves the written order);
``record_sort_key`` expresses the flat cross-stream rules above and
never re-sorts replay streams. It is a partial contract, not a complete
re-derivation of source order: for parent-attached streams with
interleaved timestamps, the snapshot's written order — not this key —
is authoritative (the divergence is pinned by
``tests/unit/source/test_ordering.py`` and the replay-order tests in
``tests/unit/source/test_turns.py``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from opencode_mlflow.source.catalog import TIE_RANKS
from opencode_mlflow.source.model import SessionHeader, SourceRecord

SOURCE_TYPE_RANKS: Mapping[str, int] = TIE_RANKS
"""The fixed source-type rank table (10/20/30/40/90).

- ``message_container`` (10): authoritative message/session-message rows;
- ``content_part`` (20): persisted parts and derived nested content items;
- ``lifecycle_event`` (30): session pending/inbox lifecycle rows;
- ``scoped_record`` (40): todo/instruction rows;
- ``unknown_record`` (90): unrecognized permitted rows.
"""

_MESSAGE_CONTAINER_TABLES = frozenset({"message", "session_message"})
_LIFECYCLE_TABLES = frozenset({"session_pending", "session_inbox"})
_SCOPED_TABLES = frozenset({"todo", "instruction_entry", "instruction_blob", "instruction_state"})


def session_sort_key(header: SessionHeader) -> tuple[int, int, str]:
    """Normalized ``(creation_time_present/value, session_id)`` session key.

    Sessions whose creation timestamp is missing sort after every
    timestamped session and then by session ID.
    """
    if header.time_created is None:
        return (1, 0, header.session_id)
    return (0, header.time_created.nanoseconds, header.session_id)


def source_type_rank(source_table: str, *, derived_child: bool = False) -> int:
    """Fixed rank for a record's source-type role.

    Derived nested content items rank as content parts even though they
    share their parent's ``session_message`` table; a table outside every
    catalogued role ranks as an unknown record.
    """
    if derived_child:
        return SOURCE_TYPE_RANKS["content_part"]
    if source_table in _MESSAGE_CONTAINER_TABLES:
        return SOURCE_TYPE_RANKS["message_container"]
    if source_table == "part":
        return SOURCE_TYPE_RANKS["content_part"]
    if source_table in _LIFECYCLE_TABLES:
        return SOURCE_TYPE_RANKS["lifecycle_event"]
    if source_table in _SCOPED_TABLES:
        return SOURCE_TYPE_RANKS["scoped_record"]
    return SOURCE_TYPE_RANKS["unknown_record"]


@dataclass(frozen=True)
class RecordOrderContext:
    """The ordering context one record sort pass runs under.

    An authoritative context (``sequence_domain`` set and
    ``authoritative`` true) orders the records of that one sequence
    domain by explicit sequence. Every other context — no shared domain,
    or a domain that is not authoritative for ordering — orders by
    normalized timestamp instead. A record outside the context's domain
    never exposes its own (unrelated) sequence value.
    """

    sequence_domain: str | None
    authoritative: bool


def record_sort_key(record: SourceRecord, context: RecordOrderContext) -> tuple[int, int | float, int, int, str]:
    """Deterministic sort key for one record under ``context``.

    Authoritative sequence-domain context:
    ``(sequence_absent, sequence, tie_rank, child_position,
    stable_source_id)`` for records of that domain — a domain record
    whose sequence value is missing or not numeric sorts with the absent
    group, after every sequenced record. Records outside the domain also
    fall into the absent group, so sequence values from unrelated
    streams are never compared.

    Every other context: ``(timestamp_absent, timestamp_ns, tie_rank,
    child_position, stable_source_id)``; a record without a normalized
    timestamp sorts after every timestamped record.

    ``child_position`` is the derived child ordinal for nested content
    items (so ``#content-10`` orders after ``#content-2`` numerically)
    and ``0`` for every other record; the tie ranks already order
    message containers before their content parts.

    This key covers the flat rules only. Same-parent attachment is
    preserved exactly when the child's timestamp does not interleave
    with other messages' timestamps (the fixed ranks order a
    same-timestamp parent before its parts); a part whose own timestamp
    falls after another message sorts after that message, detached from
    its parent. The authoritative order for parent-attached streams is
    the adapter emission order persisted in the snapshot and replayed
    by the cursors, not this key.
    """
    rank = source_type_rank(record.source_table, derived_child=record.derived_child_ordinal is not None)
    child_position = record.derived_child_ordinal if record.derived_child_ordinal is not None else 0
    stable_id = _stable_sort_id(record)
    if context.sequence_domain is not None and context.authoritative:
        if (
            record.sequence_domain == context.sequence_domain
            and isinstance(record.sequence_value, (int, float))
            and not isinstance(record.sequence_value, bool)
            and (not isinstance(record.sequence_value, float) or math.isfinite(record.sequence_value))
        ):
            return (0, record.sequence_value, rank, child_position, stable_id)
        return (1, 0, rank, child_position, stable_id)
    if record.timestamp is None:
        return (1, 0, rank, child_position, stable_id)
    return (0, record.timestamp.nanoseconds, rank, child_position, stable_id)


def _stable_sort_id(record: SourceRecord) -> str:
    """The record's stable identity for tie-breaks (never ``None``)."""
    if record.source_id is not None:
        return record.source_id
    if record.source_identity is not None:
        return record.source_identity
    return ""
