"""Logical-turn reconstruction over ordered normalized records.

A logical turn starts at a persisted, non-synthetic user prompt — a
message-container record the adapters classified ``user`` — and contains
every following record until the next user prompt. System, synthetic,
tool-only, lifecycle, and compaction metadata never start a turn;
records before the first prompt stay session-level unassigned records.
Turns consume records in the adapter-provided source order (the
normalized snapshot preserves the exporter's written order); this
module never re-sorts replay streams.

Turn keys follow the design contract:

- the normal key is
  ``turn-<sha256(session_id + "\\0" + source_type + "\\0" + source_id)>``;
- a first-user record without a source ID hashes the ordered tuple
  ``(session_id, source_type, explicit sequence or normalized timestamp,
  stable source occurrence identity, turn_index)`` — every component
  rendered as canonical-json-v1 bytes and joined with single NUL
  separators, absent values omitted. The occurrence identity is the
  adapter-provided opaque fallback identity when present, otherwise the
  source-order turn index; decoded/raw content is deliberately excluded
  so redaction cannot rename the turn. The fallback is recorded as a
  ``turn_key_fallback_identity`` warning on the summary;
- two turns that would still collide add the stable occurrence ordinal
  to the hash input and emit a ``turn_key_collision`` warning; turns
  are never merged.

:func:`iter_turn_summaries` streams: it consumes records one at a time,
keeps only the current turn's first-user record plus constant counters
and an incremental hash, and releases each completed turn's records
before advancing. :func:`iter_turn_records` re-iterates the records of
one turn; :func:`build_replay_turn` returns the concrete
:class:`~opencode_mlflow.source.replay.ReplayTurn` over the same
temporary SQLite snapshot, with fresh cursors per call and bounded
root-value loaders, so the write phase never retains session content in
memory.
"""

from __future__ import annotations

import hashlib
from collections.abc import Generator, Iterable, Iterator, Mapping
from dataclasses import dataclass

from opencode_mlflow.content.canonical import canonical_json
from opencode_mlflow.content.fingerprint import FramedDigest, canonical_record_value
from opencode_mlflow.source.model import (
    SessionHeader,
    SourceRecord,
    TimestampValue,
    TurnSummary,
)
from opencode_mlflow.source.ordering import SOURCE_TYPE_RANKS, source_type_rank
from opencode_mlflow.source.outcomes import (
    SESSION_EVENT_TABLES,
    classify_outcome,
    declared_source_outcome,
    is_explicit_failure,
    record_source_outcome,
)
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter, ReplayTurn
from opencode_mlflow.source.snapshot import SourceSnapshot

__all__ = [
    "ASSISTANT_RESULT_OUTCOME",
    "ASSISTANT_ERROR_OUTCOME",
    "BoundedRootValue",
    "NO_ASSISTANT_RESULT_OUTCOME",
    "PART_ERROR_OUTCOME",
    "TURN_KEY_COLLISION_WARNING",
    "TURN_KEY_FALLBACK_WARNING",
    "build_replay_turn",
    "iter_turn_records",
    "iter_turn_summaries",
    "turn_key",
]

ASSISTANT_RESULT_OUTCOME = "assistant_result"
"""Terminal outcome of a turn that ended with a non-synthetic assistant result.

A turn with a root result is the successful case (it maps to an ``OK``
trace status); see the design spec's status mapping."""

NO_ASSISTANT_RESULT_OUTCOME = "no_assistant_result"
"""Terminal outcome of a turn without any non-synthetic assistant result.

Such a turn keeps a null/omitted root output and a distinguishable
non-``OK`` status; no synthetic output is ever manufactured."""

TURN_KEY_COLLISION_WARNING = "turn_key_collision"
"""Summary warning: two turns produced one key, so the occurrence ordinal was added."""

TURN_KEY_FALLBACK_WARNING = "turn_key_fallback_identity"
"""Summary warning: the first-user record has no source ID, so the documented fallback tuple was hashed."""

_TURN_KEY_PREFIX = "turn-"
_TURN_RECORDS_DOMAIN = "turn-records-v1"
_SESSION_EVENT_TABLES = SESSION_EVENT_TABLES

ASSISTANT_ERROR_OUTCOME = "assistant_error"
PART_ERROR_OUTCOME = "part_error"


def _is_error_outcome(outcome: str | None) -> bool:
    """Whether a normalized source outcome is an explicit failure."""
    if outcome is None:
        return False
    return outcome in {ASSISTANT_ERROR_OUTCOME, PART_ERROR_OUTCOME, "explicit_failure"} or (
        outcome.startswith("lifecycle_") and classify_outcome(outcome.removeprefix("lifecycle_")) == "ERROR"
    )


def turn_key(session_id: str, first_user: SourceRecord, turn_index: int) -> str:
    """The stable key of the turn starting at ``first_user``.

    Normal form (the record has a source ID):
    ``turn-<sha256(session_id + "\\0" + source_type + "\\0" + source_id)>``.

    A first-user record without a source ID hashes the ordered tuple
    ``(session_id, source_type, explicit sequence or normalized timestamp,
    stable source occurrence identity, turn_index)``: each component is
    canonical-json-v1 bytes joined with single NUL separators, absent
    values omitted (canonical-json-v1 output never contains a literal NUL
    byte, so the framing is unambiguous). Raw/decoded content is not part
    of the fallback identity. The fallback is deterministic for a repeated
    snapshot — exact duplicate fallback records stay distinct through their
    ordered turn index — while a source that provides no distinguishing
    identity may legitimately renumber later duplicates when its order
    changes.

    This function computes the first occurrence's key; the collision
    ordinal :func:`iter_turn_summaries` adds for repeated keys is
    internal to a scan.
    """
    return _TURN_KEY_PREFIX + hashlib.sha256(_turn_key_material(session_id, first_user, turn_index)).hexdigest()


def iter_turn_summaries(records: Iterable[SourceRecord]) -> Iterator[TurnSummary]:
    """Stream one :class:`TurnSummary` per logical turn, in source order.

    ``records`` are one session's records in adapter-provided source
    order. The scan is streaming: each record is consumed once, the
    completed turn's records are released before the next turn
    advances (only the current turn's first-user record, constant
    counters, and an incremental hash are retained), and a summary is
    yielded as soon as the next user prompt closes its turn. Records
    before the first user prompt belong to no turn and produce no
    summary. The turn-local ``source_fingerprint`` on each summary is
    the ``turn-records-v1`` :class:`~opencode_mlflow.content.fingerprint.FramedDigest`
    over the ordered turn records, canonicalized through the shared
    :func:`opencode_mlflow.content.fingerprint.canonical_record_value`
    form with no redactor (replayed values are already effective) — the
    same record form Task 4A's effective-turn cursors hash. The full
    turn source fingerprint (session context, session-scoped records,
    and redaction mode included) is computed later through
    :func:`opencode_mlflow.content.fingerprint.source_fingerprint`.
    """
    used_keys: set[str] = set()
    state: _TurnState | None = None
    for header, record in _scan_turns(records, used_keys):
        if header is None:
            # A record no turn claimed (before the first user prompt).
            continue
        if state is None or header.ordinal != state.header.ordinal:
            if state is not None:
                yield state.summary()
            state = _TurnState(header)
        state.add(record)
    if state is not None:
        yield state.summary()


def iter_turn_records(records: Iterable[SourceRecord], summary: TurnSummary) -> Generator[SourceRecord, None, None]:
    """Re-iterate the records of the turn identified by ``summary``.

    The scan is the same deterministic boundary pass
    :func:`iter_turn_summaries` runs, so re-iterating the same records
    reproduces the same turn. Records of other turns and records no turn
    claimed are consumed and released without being retained. A summary
    whose turn key matches no turn — or whose turn index disagrees with
    the matched turn — fails loudly rather than silently yielding
    nothing.
    """
    used_keys: set[str] = set()
    target_ordinal: int | None = None
    for header, record in _scan_turns(records, used_keys):
        if header is None:
            continue
        if target_ordinal is None:
            if header.key != summary.turn_key:
                continue
            if header.ordinal != summary.turn_index:
                raise ValueError("the matched turn index does not match its summary")
            target_ordinal = header.ordinal
        elif header.ordinal != target_ordinal:
            return
        yield record
    if target_ordinal is None:
        raise ValueError("the summary's turn key matches no turn in the supplied records")


def build_replay_turn(snapshot: SourceSnapshot, header: SessionHeader, summary: TurnSummary) -> ReplayTurn:
    """Build the concrete :class:`ReplayTurn` over the normalized snapshot.

    Every cursor call opens a fresh query over the snapshot tables and
    filters to the turn's records by their recorded written order and
    identity: :meth:`ReplayTurn.iter_records` re-scans the session's
    turn-stream records for this turn, ``iter_session_records`` yields
    the session-scoped records, and ``iter_unassigned_records`` yields
    every other permitted session record (other turns' records plus
    records no turn claimed), so the union of the three cursors is the
    complete permitted session payload that artifact snapshots hash.
    The root loaders stream bounded values and never return an unbounded
    prompt or response. The turn is never materialized as an in-memory
    record list.
    """
    if header.session_id != summary.session_id:
        raise ValueError("the turn summary does not belong to the supplied session header")
    return _SnapshotReplayTurn(snapshot, header, summary)


@dataclass(frozen=True)
class BoundedRootValue:
    """One root value loaded within a byte bound.

    ``size_bytes`` is the canonical-json-v1 size of the value. A value
    that does not fit within ``max_bytes`` is never returned unbounded
    and never truncated: ``payload_too_large`` is true and ``value``
    stays ``None``, so the caller rejects the item (``payload_too_large``)
    before trace creation instead of importing a partial prompt or
    response.
    """

    value: object | None
    size_bytes: int
    max_bytes: int
    payload_too_large: bool = False


# ---------------------------------------------------------------------------
# Turn scanning
# ---------------------------------------------------------------------------


@dataclass
class _TurnHeader:
    """Identity of one turn being scanned (created at its first user prompt)."""

    ordinal: int
    session_id: str
    key: str
    first_user: SourceRecord
    warnings: tuple[str, ...]


def _scan_turns(
    records: Iterable[SourceRecord], used_keys: set[str]
) -> Iterator[tuple[_TurnHeader | None, SourceRecord]]:
    """Yield ``(turn header, record)`` pairs in source order, one pass.

    The header is ``None`` for records no turn claimed (before the first
    user prompt); every record of a turn yields that turn's header. Turn
    keys (including fallback and collision handling) are resolved at the
    turn's first record, so a scan reproduces the same keys as
    :func:`iter_turn_summaries` produced for the same records.
    """
    current: _TurnHeader | None = None
    for record in records:
        if _starts_turn(record):
            ordinal = current.ordinal + 1 if current is not None else 1
            session_id = record.session_id if record.session_id is not None else ""
            key, warnings = _resolve_turn_key(session_id, record, ordinal, used_keys)
            current = _TurnHeader(ordinal, session_id, key, record, warnings)
        yield current, record


class _TurnState:
    """Incremental accumulator for one in-flight turn (bounded state)."""

    def __init__(self, header: _TurnHeader) -> None:
        self.header = header
        self._record_count = 0
        self._user_count = 0
        self._assistant_count = 0
        self._end_time: TimestampValue | None = None
        self._root_result_identity: str | None = None
        self._root_result_error = False
        self._terminal_outcome: str | None = None
        self._source_outcome: str | None = None
        self._lifecycle_failure_outcome: str | None = None
        self._assistant_failure_active = False
        self._digest = FramedDigest(_TURN_RECORDS_DOMAIN)

    def add(self, record: SourceRecord) -> None:
        self._record_count += 1
        if record.classification == "user":
            self._user_count += 1
        elif record.classification == "assistant":
            self._assistant_count += 1
        # Completion is the most faithful end instant for this record. An
        # untimed trailing record must never erase the last known end.
        observed_end = record.completion_timestamp or record.timestamp
        if observed_end is not None:
            self._end_time = observed_end
        lifecycle_outcome = _lifecycle_outcome(record)
        if lifecycle_outcome is not None:
            if _is_error_outcome(lifecycle_outcome):
                self._lifecycle_failure_outcome = lifecycle_outcome
                self._terminal_outcome = lifecycle_outcome
                self._source_outcome = record_source_outcome(
                    record.decoded_value,
                    record.source_type,
                    record.raw_columns,
                )
            elif (
                self._lifecycle_failure_outcome is None
                and not self._assistant_failure_active
                and self._root_result_identity is None
            ):
                self._terminal_outcome = lifecycle_outcome
                self._source_outcome = record_source_outcome(
                    record.decoded_value,
                    record.source_type,
                    record.raw_columns,
                )
        elif _is_root_result(record):
            # The LAST non-synthetic assistant result wins; earlier results
            # stay turn records (child candidates).
            self._root_result_identity = _record_identity(record)
            self._root_result_error = _record_has_explicit_error(record)
            record_outcome = record_source_outcome(record.decoded_value, record.source_type, record.raw_columns)
            if self._root_result_error and self._lifecycle_failure_outcome is None:
                self._assistant_failure_active = True
                self._terminal_outcome = ASSISTANT_ERROR_OUTCOME
                self._source_outcome = declared_source_outcome(record.decoded_value) or record_outcome
            elif self._lifecycle_failure_outcome is not None:
                pass
            elif classify_outcome(record_outcome) == "UNSET":
                # A persisted assistant envelope can describe work that has
                # not completed yet. It is still the latest result candidate,
                # but it must not erase an earlier explicit assistant error.
                if not self._assistant_failure_active:
                    self._terminal_outcome = record_outcome or "assistant_pending"
                    self._source_outcome = record_outcome
            else:
                self._assistant_failure_active = False
                self._terminal_outcome = ASSISTANT_RESULT_OUTCOME
                self._source_outcome = record_outcome or ASSISTANT_RESULT_OUTCOME
        elif _is_error_record(record):
            if self._root_result_identity is None:
                self._assistant_failure_active = True
                self._terminal_outcome = PART_ERROR_OUTCOME
                self._source_outcome = declared_source_outcome(record.decoded_value)
            elif record.parent_source_id == self._root_result_identity:
                self._root_result_error = True
                self._assistant_failure_active = True
                self._terminal_outcome = "explicit_failure"
                self._source_outcome = declared_source_outcome(record.decoded_value)
        # The shared Task 4A record form (no redactor: replayed values
        # are already the effective values).
        self._digest.update(canonical_json(canonical_record_value(record)))

    def summary(self) -> TurnSummary:
        return TurnSummary(
            session_id=self.header.session_id,
            turn_index=self.header.ordinal,
            turn_key=self.header.key,
            first_user_source_id=_record_identity(self.header.first_user),
            first_user_source_type=self.header.first_user.source_type,
            start_time=self.header.first_user.timestamp,
            end_time=self._end_time,
            record_count=self._record_count,
            user_record_count=self._user_count,
            assistant_record_count=self._assistant_count,
            terminal_outcome=(
                self._terminal_outcome if self._terminal_outcome is not None else NO_ASSISTANT_RESULT_OUTCOME
            ),
            source_outcome=self._source_outcome,
            root_result_source_id=self._root_result_identity,
            source_fingerprint=self._digest.hexdigest(),
            warnings=self.header.warnings,
        )


def _starts_turn(record: SourceRecord) -> bool:
    """Only a persisted, non-synthetic user prompt starts a turn.

    The classification comes from the adapters/catalog; the fixed rank
    table keeps content parts and derived children (which can inherit a
    ``user`` classification) from starting turns — only message
    containers are prompts.
    """
    return record.classification == "user" and _is_message_container(record) and not _is_synthetic(record)


def _is_root_result(record: SourceRecord) -> bool:
    """Whether ``record`` is a root-result candidate.

    A result is a non-synthetic assistant message container with a
    stable identity (the adapters always assign explicit or fallback
    identities to message rows, so a real result is always replayable).
    Assistant-classified parts are split content, not results.
    """
    return (
        record.classification == "assistant"
        and _is_message_container(record)
        and not _is_synthetic(record)
        and _record_identity(record) is not None
    )


def _record_has_explicit_error(record: SourceRecord) -> bool:
    """Whether a message/container carries an explicit error signal."""
    return (
        is_explicit_failure(record.decoded_value)
        or classify_outcome(record_source_outcome(record.decoded_value, record.source_type, record.raw_columns))
        == "ERROR"
    )


def _is_error_record(record: SourceRecord) -> bool:
    """Whether a part or opaque record is an explicit source error."""
    if classify_outcome(record.source_type) == "ERROR":
        return True
    decoded = record.decoded_value
    if not isinstance(decoded, Mapping):
        return False
    if decoded.get("type") == "error":
        return True
    return (
        is_explicit_failure(decoded)
        or classify_outcome(record_source_outcome(decoded, record.source_type, record.raw_columns)) == "ERROR"
    ) and not _is_root_result(record)


def _lifecycle_outcome(record: SourceRecord) -> str | None:
    """Normalize a session lifecycle result without making it a turn start."""
    if record.source_table not in _SESSION_EVENT_TABLES:
        return None
    raw = record_source_outcome(record.decoded_value, record.source_type, record.raw_columns)
    if raw is None:
        return None
    return f"lifecycle_{raw.strip().lower()}"


def _is_message_container(record: SourceRecord) -> bool:
    """Whether the record is a message container (not a part or derived child)."""
    rank = source_type_rank(record.source_table, derived_child=record.derived_child_ordinal is not None)
    return rank == SOURCE_TYPE_RANKS["message_container"]


def _is_synthetic(record: SourceRecord) -> bool:
    """Whether the record is synthetic.

    The adapter classification marks synthetic rows (``synthetic``
    class, legacy step parts). The current v1 catalog additionally marks
    synthetic ``session_message`` rows through ``data.synthetic`` while
    its assistant/system classification rules do not exclude them, so
    the catalogued marker is checked directly as well.
    """
    if record.classification == "synthetic":
        return True
    decoded = record.decoded_value
    if isinstance(decoded, Mapping):
        marker = decoded.get("synthetic")
        if marker is True or marker == "true":
            return True
    return False


def _record_identity(record: SourceRecord) -> str | None:
    """The record's stable identity: its source ID or fallback identity."""
    if record.source_id is not None:
        return record.source_id
    return record.source_identity


def _resolve_turn_key(
    session_id: str, first_user: SourceRecord, ordinal: int, used_keys: set[str]
) -> tuple[str, tuple[str, ...]]:
    """Compute the turn's key, disambiguating collisions by occurrence ordinal.

    A key already used by an earlier turn of the same scan gets the
    stable occurrence ordinal (2, 3, …) appended to the hash input and a
    ``turn_key_collision`` warning; turns are never merged.
    """
    material = _turn_key_material(session_id, first_user, ordinal)
    key = _TURN_KEY_PREFIX + hashlib.sha256(material).hexdigest()
    occurrence = 1
    warnings: list[str] = []
    if first_user.source_id is None:
        warnings.append(TURN_KEY_FALLBACK_WARNING)
    while key in used_keys:
        occurrence += 1
        key = _TURN_KEY_PREFIX + hashlib.sha256(material + b"\0" + canonical_json(occurrence)).hexdigest()
    if occurrence > 1:
        warnings.append(TURN_KEY_COLLISION_WARNING)
    used_keys.add(key)
    return key, tuple(warnings)


def _turn_key_material(session_id: str, first_user: SourceRecord, turn_index: int) -> bytes:
    """The hash input for the turn's first occurrence."""
    if first_user.source_id is not None:
        return "\0".join((session_id, first_user.source_type, first_user.source_id)).encode("utf-8")
    components = [canonical_json(session_id), canonical_json(first_user.source_type)]
    sequence_or_timestamp = _sequence_or_timestamp_material(first_user)
    if sequence_or_timestamp is not None:
        components.append(sequence_or_timestamp)
    # Adapters provide a stable opaque occurrence identity for missing source
    # IDs. It may be derived from source-row provenance, but it is never raw
    # content; when an adapter has none, source order is the only stable
    # occurrence discriminator available to this scan.
    components.append(
        canonical_json(first_user.source_identity if first_user.source_identity is not None else turn_index)
    )
    components.append(canonical_json(turn_index))
    return b"\0".join(components)


def _sequence_or_timestamp_material(record: SourceRecord) -> bytes | None:
    """The record's explicit sequence or normalized timestamp, canonically."""
    if isinstance(record.sequence_value, (int, float)) and not isinstance(record.sequence_value, bool):
        return canonical_json(record.sequence_value)
    if record.timestamp is not None:
        return canonical_json(record.timestamp.nanoseconds)
    return None


# ---------------------------------------------------------------------------
# The concrete ReplayTurn over the normalized snapshot
# ---------------------------------------------------------------------------


class _SnapshotReplayTurn:
    """Concrete ``ReplayTurn``: fresh snapshot cursors and bounded root loaders."""

    def __init__(self, snapshot: SourceSnapshot, header: SessionHeader, summary: TurnSummary) -> None:
        self._header = header
        self._summary = summary
        self._adapter = NormalizedSnapshotAdapter(snapshot)

    @property
    def session_header(self) -> SessionHeader:
        return self._header

    @property
    def summary(self) -> TurnSummary:
        return self._summary

    def iter_records(self) -> Generator[SourceRecord, None, None]:
        """Fresh cursor over this turn's selected ordered records."""
        return iter_turn_records(self._adapter.iter_records(self._summary.session_id), self._summary)

    def iter_session_records(self) -> Iterator[SourceRecord]:
        """Fresh cursor over the session-scoped records."""
        return self._adapter.iter_session_scoped_records(self._summary.session_id)

    def iter_unassigned_records(self) -> Iterator[SourceRecord]:
        """Fresh cursor over every other permitted session record.

        Other turns' records plus records no turn claimed; the union
        with :meth:`iter_records` and :meth:`iter_session_records` is
        the complete permitted session payload.
        """
        used_keys: set[str] = set()
        target_ordinal: int | None = None
        for header, record in _scan_turns(self._adapter.iter_records(self._summary.session_id), used_keys):
            if header is None:
                yield record
                continue
            if target_ordinal is None:
                if header.key != self._summary.turn_key:
                    yield record
                    continue
                target_ordinal = header.ordinal
                continue
            if header.ordinal != target_ordinal:
                yield record
        if target_ordinal is None:
            raise ValueError("the summary's turn key matches no turn in the supplied records")

    def load_root_input(self, max_bytes: int) -> BoundedRootValue | None:
        """Stream the turn's first-user record's decoded value, bounded.

        The turn's first record is its first-user prompt. Legacy split
        prompts reconstruct the complete ordered part list; a container
        without parts (and current message containers) uses its decoded
        value. The value is returned when its canonical-json-v1 serialization fits within
        ``max_bytes`` and rejected — never truncated — otherwise.
        """
        records = self.iter_records()
        try:
            prompt: SourceRecord | None = None
            collected: list[object] = []
            size = 2
            for record in records:
                if prompt is None:
                    prompt = record
                    if record.source_table != "message":
                        return _bounded_single_value(record.decoded_value, max_bytes)
                    continue
                if record.parent_source_id != _record_identity(prompt):
                    continue
                size += len(canonical_json(record.decoded_value)) + (1 if collected else 0)
                if size > max_bytes:
                    return BoundedRootValue(value=None, size_bytes=size, max_bytes=max_bytes, payload_too_large=True)
                collected.append(record.decoded_value)
            if prompt is not None:
                return _bounded_single_value(collected if collected else prompt.decoded_value, max_bytes)
            return None
        finally:
            records.close()

    def load_root_output(self, max_bytes: int) -> BoundedRootValue | None:
        """Stream the final assistant result's combined value, bounded.

        Combined-value rule: the root result is the turn's last
        non-synthetic assistant message container. Its split content is
        the ordered child parts (records whose ``parent_source_id`` is
        the result's stable identity — legacy part rows and current
        derived content items alike). The user-visible parts — child
        parts whose decoded value is a ``text`` content item — combine,
        in source order, into the root output list. A result with child
        parts but no user-visible ones combines to an empty list; a
        result with no child parts uses its own decoded value. Every
        part, user-visible or not, remains separately represented in
        :meth:`iter_records`. The canonical size is measured part by
        part and the value is rejected — never truncated — once it
        exceeds ``max_bytes``.
        """
        root_identity = self._summary.root_result_source_id
        if root_identity is None:
            return None
        if classify_outcome(self._summary.terminal_outcome) == "UNSET":
            # A pending/running/queued assistant envelope has no completed
            # output to publish. Its complete source value remains in the
            # artifact, while the mapped root stays explicitly incomplete.
            return None
        root_decoded: object | None = None
        found_root = False
        has_parts = False
        size = 2  # the JSON array brackets of the combined list
        collected: list[object] = []
        rejected = False
        records = self.iter_records()
        try:
            for record in records:
                if not found_root:
                    if _record_identity(record) == root_identity and _is_root_result(record):
                        found_root = True
                        root_decoded = record.decoded_value
                    continue
                if record.parent_source_id != root_identity:
                    continue
                has_parts = True
                if not _is_user_visible_content(record):
                    continue
                chunk = canonical_json(record.decoded_value)
                if collected:
                    size += 1  # the array member separator
                size += len(chunk)
                if size > max_bytes:
                    rejected = True
                    break
                collected.append(record.decoded_value)
        finally:
            records.close()
        if not found_root:
            raise ValueError("the summary's root result matches no record in the turn")
        if not has_parts:
            return _bounded_single_value(root_decoded, max_bytes)
        if rejected:
            return BoundedRootValue(value=None, size_bytes=size, max_bytes=max_bytes, payload_too_large=True)
        return BoundedRootValue(value=collected, size_bytes=size, max_bytes=max_bytes)


def _is_user_visible_content(record: SourceRecord) -> bool:
    """Whether a child part's decoded value is a user-visible text content item."""
    decoded = record.decoded_value
    return isinstance(decoded, Mapping) and decoded.get("type") == "text"


def _bounded_single_value(value: object | None, max_bytes: int) -> BoundedRootValue:
    """Bound one decoded value by its canonical-json-v1 size."""
    size = len(canonical_json(value))
    if size > max_bytes:
        return BoundedRootValue(value=None, size_bytes=size, max_bytes=max_bytes, payload_too_large=True)
    return BoundedRootValue(value=value, size_bytes=size, max_bytes=max_bytes)
