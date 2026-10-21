"""Generation-agnostic normalization helpers shared by the live exporters.

The legacy and current adapters own different SQL, but they share the
canonical normalization rules that Task 4 pins for both generations:

- safe SQL construction (identifier quoting, metadata-only column
  discovery through ``pragma_table_info``);
- JSON/text column decoding that retains raw values and warns
  ``malformed_json``/``duplicate_json_key``/``malformed_source_value``
  instead of guessing;
- epoch-millisecond timestamp normalization that treats non-numeric
  values as malformed (warned, retained, never normalized);
- catalog-driven record classification, evaluated rule-by-rule in catalog
  order with ``inherits_parent`` and ``exclude_synthetic`` semantics;
- missing-ID identity: a canonical row digest plus an occurrence ordinal
  build the fallback ``source_identity``, and the join path plus the
  parent/child occurrence build the ``join_projection_key`` that collapses
  repeated join projections while keeping distinct no-ID rows distinct.

This module issues no SQL of its own beyond the parameterized
``pragma_table_info`` metadata query; the generation-specific SELECTs live
in the adapter owning the matching catalog.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from opencode_mlflow.json_values import decode_lossless_json
from opencode_mlflow.source.catalog import SECURITY_EXCLUDED_COLUMNS, RecordClassification, SchemaCatalog
from opencode_mlflow.source.model import (
    RecordClassificationName,
    SchemaGeneration,
    TimestampValue,
    normalize_timestamp,
)

_IDENTITY_VERSION = "opencode-mlflow-source-identity-v1"
_JOIN_PROJECTION_VERSION = "opencode-mlflow-join-projection-v1"

_STRUCTURAL_COLUMN_NAMES = frozenset(
    {
        "session_id",
        "message_id",
        "seq",
        "admitted_seq",
        "enqueued_seq",
        "epoch_start",
        "through_seq",
        "position",
        "time_created",
        "time_updated",
    }
)

MALFORMED_VALUE_WARNING = "malformed_source_value"
UNKNOWN_FIELD_WARNING = "unknown_source_field"
UNASSOCIATED_RECORD_WARNING = "unassociated_record"

MAX_NORMALIZABLE_EPOCH_MS = 62_135_596_800_000
"""Conservative epoch-millisecond magnitude safely representable by ``utc``."""

_MESSAGE_JSON_FIELDS = frozenset(
    {
        "role",
        "content",
        "model",
        "provider",
        "providerID",
        "provider_id",
        "modelID",
        "tokens",
        "cost",
        "finish",
        "status",
        "outcome",
        "error",
        "input",
        "output",
        "time",
        "state",
        "synthetic",
        "usage",
    }
)
_PART_JSON_FIELDS = frozenset(
    {
        "type",
        "text",
        "tool",
        "callID",
        "call_id",
        "state",
        "error",
        "model",
        "provider",
        "providerID",
        "provider_id",
        "modelID",
        "tokens",
        "usage",
        "cost",
        "finish",
        "status",
        "outcome",
        "input",
        "output",
        "time",
    }
)
_MODEL_JSON_FIELDS = frozenset({"provider", "model", "providerID", "provider_id", "modelID"})
_LIFECYCLE_JSON_FIELDS = frozenset({"type", "name", "event", "outcome", "status", "error", "payload", "data"})
_KNOWN_MESSAGE_TYPES = frozenset({"user", "assistant", "system"})
_KNOWN_PART_TYPES = frozenset({"text", "reasoning", "tool", "error", "step-start", "step-finish"})


def quote_identifier(name: str) -> str:
    """Quote a schema-derived identifier for a SELECT list or ORDER BY."""
    return '"' + name.replace('"', '""') + '"'


def observed_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    """Return the observed column names of ``table`` (metadata only).

    A table that does not exist has no columns; callers treat an empty
    result as "table absent" rather than querying it.
    """
    rows = connection.execute("SELECT name FROM pragma_table_info(?)", (table,)).fetchall()
    return tuple(str(row[0]) for row in rows)


def permitted_observed_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    """Return observed columns safe for source-value projection.

    Schema probing may observe every column name as metadata, but exporters
    must not turn additive columns into a raw-value capture bypass.  The
    security exclusion is applied before every generation-specific SELECT and
    raw-column map is built.
    """
    excluded = {name.casefold() for name in SECURITY_EXCLUDED_COLUMNS}
    return tuple(name for name in observed_columns(connection, table) if name.casefold() not in excluded)


class DecodedColumn(NamedTuple):
    """Decoding result for one JSON-encoded source column."""

    raw_text: str | None
    decoded: object | None
    warnings: tuple[str, ...]
    retained: object | None


def decode_json_column(value: object) -> DecodedColumn:
    """Decode a JSON-encoded column value, retaining raw provenance.

    TEXT passes through and decodes (with duplicate-key detection); a
    BLOB decodes as UTF-8 when possible; an invalid-UTF-8 BLOB yields no
    raw JSON, a ``malformed_source_value`` warning, and the original
    bytes for permitted-field retention; any other scalar warns
    ``malformed_source_value`` and is retained.
    """
    if value is None:
        return DecodedColumn(None, None, (), None)
    text: str | None = None
    warnings: tuple[str, ...] = ()
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            return DecodedColumn(None, None, (MALFORMED_VALUE_WARNING,), value)
    elif isinstance(value, str):
        text = value
    else:
        return DecodedColumn(None, None, (MALFORMED_VALUE_WARNING,), value)
    try:
        decoded = decode_lossless_json(text)
    except ValueError:
        return DecodedColumn(text, None, ("malformed_json",), None)
    if decoded.has_duplicate_keys:
        warnings = ("duplicate_json_key",)
    return DecodedColumn(text, decoded.value, warnings, None)


def unknown_json_field_warning(decoded: object, table: str) -> str | None:
    """Return the safe warning code when a catalogued JSON object adds keys.

    Content-bearing metadata fields such as session ``metadata`` and
    instruction state are intentionally free-form and are not passed here.
    Unknown keys are never dropped: adapters retain both the raw text and the
    decoded value alongside this warning.
    """
    if not isinstance(decoded, Mapping):
        return None
    if table == "model":
        allowed = _MODEL_JSON_FIELDS
    else:
        allowed = _MESSAGE_JSON_FIELDS if table in {"message", "session_message"} else _PART_JSON_FIELDS
    if table in {"session_pending", "session_inbox"}:
        allowed = _LIFECYCLE_JSON_FIELDS
    if any(not isinstance(key, str) or key not in allowed for key in decoded):
        return UNKNOWN_FIELD_WARNING
    if "model" in decoded and _unknown_model_value(decoded.get("model")):
        return UNKNOWN_FIELD_WARNING
    if table in {"message", "session_message"}:
        content = decoded.get("content")
        if isinstance(content, list) and any(_unknown_part_item(item) for item in content):
            return UNKNOWN_FIELD_WARNING
    if table in {"part"} and _unknown_part_item(decoded):
        return UNKNOWN_FIELD_WARNING
    return None


def _unknown_part_item(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if any(not isinstance(key, str) or key not in _PART_JSON_FIELDS for key in value):
        return True
    state = value.get("state")
    if isinstance(state, Mapping):
        allowed_state = {"input", "output", "status", "error", "time", "outcome"} | _MODEL_JSON_FIELDS
        if any(key not in allowed_state for key in state):
            return True
        if "model" in state and _unknown_model_value(state.get("model")):
            return True
    return False


def _unknown_model_value(value: object) -> bool:
    """Validate provider/model aliases, including nested model envelopes."""
    if not isinstance(value, Mapping):
        return False
    if any(not isinstance(key, str) or key not in _MODEL_JSON_FIELDS for key in value):
        return True
    nested = value.get("model")
    return isinstance(nested, Mapping) and _unknown_model_value(nested)


def unknown_record_type_warning(table: str, value: object) -> str | None:
    """Warn for an unrecognized catalogued record type without rejecting it."""
    if not isinstance(value, str) or not value:
        return UNKNOWN_FIELD_WARNING
    if table in {"message", "session_message"}:
        return None if value in _KNOWN_MESSAGE_TYPES else UNKNOWN_FIELD_WARNING
    if table == "part":
        return None if value in _KNOWN_PART_TYPES else UNKNOWN_FIELD_WARNING
    return None


def normalize_epoch_ms(value: object) -> tuple[TimestampValue | None, tuple[str, ...]]:
    """Normalize a catalogued epoch-millisecond value.

    A missing value stays absent without warning. A present value that is
    not numeric (for example a TEXT or BLOB timestamp) warns
    ``malformed_source_value`` and normalizes to ``None`` — the adapter
    never guesses a unit or a value.
    """
    if value is None:
        return None, ()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, (MALFORMED_VALUE_WARNING,)
    if isinstance(value, float) and not math.isfinite(value):
        return None, (MALFORMED_VALUE_WARNING,)
    if abs(value) > MAX_NORMALIZABLE_EPOCH_MS:
        return None, (MALFORMED_VALUE_WARNING,)
    try:
        timestamp = normalize_timestamp(value, "ms")
        assert timestamp is not None
        _nanoseconds = timestamp.nanoseconds
        _utc = timestamp.utc
    except (OverflowError, ValueError):
        return None, (MALFORMED_VALUE_WARNING,)
    return timestamp, ()


_COMPLETION_PATHS: tuple[tuple[str, ...], ...] = (
    ("time", "completed"),
    ("time", "end"),
    ("state", "time", "end"),
)


def normalize_completion_timestamp(
    decoded: object,
    raw_columns: Mapping[str, object],
    catalog: SchemaCatalog,
    source_table: str,
) -> tuple[TimestampValue | None, tuple[str, ...]]:
    """Normalize the catalogued completion/end instant of one source row.

    OpenCode payloads have used several equivalent locations over time. The
    priority is intentionally data-catalogued and deterministic: decoded
    ``time.completed``, decoded ``time.end``, decoded ``state.time.end``, and
    finally the catalogued row ``time_updated`` column. A malformed higher
    priority value is warned about but does not hide a valid lower-priority
    value. ``time_updated`` is only considered when the owning catalog entry
    actually exposes that column, so unrelated session-scoped rows cannot
    acquire an invented completion time.
    """
    warnings: list[str] = []
    catalog_columns = catalogued_columns(catalog, source_table)
    if not catalog_columns:
        return None, ()

    for path in _COMPLETION_PATHS:
        present, value = _lookup_present(decoded, path)
        if not present or value is None:
            continue
        timestamp, timestamp_warnings = normalize_epoch_ms(value)
        warnings.extend(timestamp_warnings)
        if timestamp is not None:
            return timestamp, tuple(dict.fromkeys(warnings))

    if "time_updated" in catalog_columns:
        timestamp, timestamp_warnings = normalize_epoch_ms(raw_columns.get("time_updated"))
        warnings.extend(timestamp_warnings)
        if timestamp is not None:
            return timestamp, tuple(dict.fromkeys(warnings))
    return None, tuple(dict.fromkeys(warnings))


def _digest_component(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool:{value}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        return f"float:{value!r}"
    if isinstance(value, str):
        return f"str:{value}"
    if isinstance(value, (bytes, bytearray)):
        return "bytes:" + hashlib.sha256(bytes(value)).hexdigest()
    return f"value:{value!r}"


def canonical_row_digest(
    source_table: str,
    raw_columns: Mapping[str, object],
    raw_json: str | None,
    *,
    session_id: str | None = None,
    parent_key: str | None = None,
    sequence_or_timestamp: object | None = None,
    row_discriminator: object | None = None,
) -> str:
    """Digest structural row provenance without hashing source content.

    ``raw_columns`` and ``raw_json`` remain positional for compatibility with
    the original helper, but content values are deliberately ignored. Only
    structural columns are retained when explicit provenance is not supplied;
    adapters pass the normalized session/parent/sequence/timestamp and the
    physical ``rowid`` where SQLite exposes one. This digest is therefore
    stable when a secret-bearing JSON value changes while still separating
    rows whose structural provenance differs.
    """
    del raw_json
    structural: dict[str, object] = {}
    if session_id is None and parent_key is None and sequence_or_timestamp is None and row_discriminator is None:
        structural = {name: raw_columns[name] for name in sorted(_STRUCTURAL_COLUMN_NAMES) if name in raw_columns}
    if session_id is not None:
        structural["session_id"] = session_id
    if parent_key is not None:
        structural["parent_key"] = parent_key
    if sequence_or_timestamp is not None:
        structural["sequence_or_timestamp"] = _sequence_material(sequence_or_timestamp)
    if row_discriminator is not None:
        structural["row_discriminator"] = row_discriminator
    pairs = sorted((str(name), _digest_component(value)) for name, value in structural.items())
    material = json.dumps([_IDENTITY_VERSION, source_table, [[name, component] for name, component in pairs]])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _sequence_material(sequence_or_timestamp: object) -> str:
    if sequence_or_timestamp is None:
        return "none"
    if isinstance(sequence_or_timestamp, TimestampValue):
        return f"ns:{sequence_or_timestamp.nanoseconds}"
    if isinstance(sequence_or_timestamp, bool):
        return f"bool:{sequence_or_timestamp}"
    if isinstance(sequence_or_timestamp, (int, float)):
        return f"value:{sequence_or_timestamp!r}"
    return f"value:{sequence_or_timestamp!r}"


def fallback_source_identity(
    schema_generation: SchemaGeneration,
    source_table: str,
    session_id: str | None,
    parent_key: str | None,
    sequence_or_timestamp: object | None,
    row_digest: str,
    occurrence_ordinal: int,
) -> str:
    """Fallback identity for a source row without an explicit ID.

    The identity hashes the schema generation, table, session, parent
    key, the row's explicit sequence or normalized timestamp, the
    canonical row digest, and the occurrence ordinal — so identical
    no-ID rows in different occurrences remain distinct, and ``None`` is
    never used alone as a deduplication key.
    """
    material = json.dumps(
        [
            _IDENTITY_VERSION,
            str(schema_generation),
            source_table,
            session_id if session_id is not None else "",
            parent_key if parent_key is not None else "",
            _sequence_material(sequence_or_timestamp),
            row_digest,
            int(occurrence_ordinal),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def join_projection_identity(join_path: str, row_digest: str, child_ordinal: int) -> str:
    """Key identifying one parent/child projection of a no-ID row.

    Two emissions share this key exactly when they project the same row
    digest through the same join path at the same child ordinal within
    the parent's child enumeration; repeated join projections then
    collapse, while distinct no-ID rows (identical values in different
    occurrences) keep different ordinals and stay distinct.
    """
    material = json.dumps([_JOIN_PROJECTION_VERSION, join_path, row_digest, int(child_ordinal)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def session_order_key(time_created: object, session_id: object) -> tuple[int, int, str]:
    """Sort key for a session's creation timestamp and ID.

    Sessions whose creation timestamp is missing or malformed sort after
    every timestamped session and then by session ID.
    """
    timestamp, _warnings = normalize_epoch_ms(time_created)
    key = session_id if isinstance(session_id, str) else "" if session_id is None else str(session_id)
    if timestamp is None:
        return (1, 0, key)
    return (0, timestamp.nanoseconds, key)


def session_row_sort_key(row: Mapping[str, object]) -> tuple[int, int, str]:
    """Sort key for a raw session row, mirroring ``session_sort_key``."""
    return session_order_key(row.get("time_created"), row.get("id"))


def has_rowid(connection: sqlite3.Connection, table: str) -> bool:
    """Whether ``table`` exposes the implicit ``rowid`` column.

    Ordinary tables do; ``WITHOUT ROWID`` tables do not. The adapter uses
    the rowid as a per-row discriminator when streaming merge-joined
    child cursors, and falls back to per-parent queries for tables that
    lack it.
    """
    try:
        connection.execute(f"SELECT rowid FROM {quote_identifier(table)} LIMIT 0").fetchone()
    except sqlite3.OperationalError:
        return False
    return True


def primary_key_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    """Return a table's declared primary-key columns in key order.

    A ``WITHOUT ROWID`` table has no physical insertion discriminator.  Its
    declared primary key is the strongest catalog-stable ordering key SQLite
    exposes, so adapters use it as the final fallback.  If the schema has no
    such key (which is not valid for a ``WITHOUT ROWID`` table), tied rows
    necessarily remain dependent on SQLite's source order.
    """
    rows = connection.execute(
        "SELECT name FROM pragma_table_info(?) WHERE pk > 0 ORDER BY pk",
        (table,),
    )
    return tuple(str(row[0]) for row in rows)


def stable_key_order(
    observed: tuple[str, ...],
    key_columns: tuple[str, ...],
    qualifier: str | None = None,
) -> str:
    """Build a null-safe SQL order for observed declared-key columns."""
    present = [name for name in key_columns if name in observed]
    if not present:
        return ""

    def qualified(name: str) -> str:
        quoted = quote_identifier(name)
        return f"{qualifier}.{quoted}" if qualifier is not None else quoted

    return ", ".join(f"({qualified(name)} IS NULL), COALESCE(CAST({qualified(name)} AS TEXT), '')" for name in present)


class OrderedGroupReader:
    """Consecutive-group reader over an ordered ``(key, value)`` stream.

    The stream must be sorted by its keys. :meth:`take` yields the rows
    of exactly one group, holding a single pending pair as lookahead, so
    memory stays bounded by one row regardless of table size. Groups
    whose key sorts before a requested key belong to sessions that were
    never emitted (suppressed or duplicate) and are dropped.
    """

    def __init__(self, pairs: Iterator[tuple[tuple[Any, ...], Any]]) -> None:
        self._pairs = iter(pairs)
        self._pending: tuple[tuple[Any, ...], Any] | None = next(self._pairs, None)

    def take(self, key: tuple[Any, ...]) -> Iterator[Any]:
        """Yield the values of group ``key``, dropping stale earlier groups."""
        while self._pending is not None and self._pending[0] < key:
            self._pending = next(self._pairs, None)
        while self._pending is not None and self._pending[0] == key:
            yield self._pending[1]
            self._pending = next(self._pairs, None)


def _value_matches(actual: object, expected: str) -> bool:
    """Compare a decoded/raw value with a rule's expected string.

    Booleans compare via their lowercase JSON spelling; only strings and
    booleans can match.
    """
    if isinstance(actual, bool):
        return str(actual).lower() == expected
    return isinstance(actual, str) and actual == expected


def _navigate(decoded: object, path: Sequence[str]) -> object:
    value = decoded
    for segment in path:
        if not isinstance(value, Mapping) or segment not in value:
            return None
        value = value[segment]
    return value


def _lookup_present(decoded: object, path: Sequence[str]) -> tuple[bool, object | None]:
    """Return whether a decoded path exists, preserving present ``null``."""
    value = decoded
    for segment in path:
        if not isinstance(value, Mapping) or segment not in value:
            return False, None
        value = value[segment]
    return True, value


def _rule_matches(rule: RecordClassification, row_values: Mapping[str, object], decoded: object | None) -> bool:
    if rule.match_column is None or rule.match_value is None:
        return True
    if rule.match_json_path:
        actual = _navigate(decoded, rule.match_json_path)
    else:
        actual = row_values.get(rule.match_column)
    return _value_matches(actual, rule.match_value)


def _row_is_synthetic(
    rules: Sequence[RecordClassification], table: str, row_values: Mapping[str, object], decoded: object | None
) -> bool:
    return any(
        rule.name == "synthetic" and rule.table == table and _rule_matches(rule, row_values, decoded) for rule in rules
    )


def evaluate_classification(
    rules: Sequence[RecordClassification],
    table: str,
    row_values: Mapping[str, object],
    decoded: object | None,
    parent_classification: RecordClassificationName | None,
) -> RecordClassificationName | None:
    """Evaluate the catalog's classification rules for one row.

    Rules match in catalog order; the first match wins. An
    ``inherits_parent`` rule assigns the parent's classification (used for
    legacy parts) unless ``exclude_synthetic`` drops a synthetic row; a
    rule with ``exclude_synthetic`` never assigns its class to a row that
    matches any synthetic rule. Rows matching no rule classify as
    ``None`` (unclassified, never a user prompt).
    """
    synthetic = _row_is_synthetic(rules, table, row_values, decoded)
    for rule in rules:
        if rule.table != table:
            continue
        if rule.inherits_parent:
            if rule.exclude_synthetic and synthetic:
                continue
            if parent_classification is not None:
                return parent_classification
            continue
        if _rule_matches(rule, row_values, decoded):
            if rule.exclude_synthetic and synthetic:
                continue
            return rule.name
    return None


@dataclass
class IdentityTracker:
    """Per-export deduplication and occurrence-ordinal bookkeeping.

    Explicit-ID rows deduplicate on the
    ``(schema_generation, source_table, source_id)`` projection only.
    No-ID rows count occurrences per identity context, so identical rows
    keep distinct ordinals, and repeated join projections collapse on
    their join projection identity.
    """

    schema_generation: SchemaGeneration

    def __post_init__(self) -> None:
        self._seen_explicit: set[tuple[str, str, str]] = set()
        self._occurrences: dict[tuple[str, str | None, str | None, str, str, str], int] = {}
        self._seen_join_keys: set[str] = set()

    def register_explicit(self, source_table: str, source_id: str) -> bool:
        """Mark an explicit ID seen; ``False`` means a duplicate projection."""
        key = (self.schema_generation, source_table, source_id)
        if key in self._seen_explicit:
            return False
        self._seen_explicit.add(key)
        return True

    def next_occurrence(
        self,
        source_table: str,
        session_id: str | None,
        parent_key: str | None,
        sequence_or_timestamp: object | None,
        row_digest: str,
        row_discriminator: object | None = None,
    ) -> int:
        """Return the 1-based occurrence ordinal for an identical no-ID row.

        A physical row discriminator starts its own local occurrence stream;
        sources without one fall back to the order in which the adapter sees
        the structurally equivalent rows.
        """
        key = (
            source_table,
            session_id,
            parent_key,
            _sequence_material(sequence_or_timestamp),
            row_digest,
            _digest_component(row_discriminator),
        )
        count = self._occurrences.get(key, 0) + 1
        self._occurrences[key] = count
        return count

    def register_join_projection(self, join_key: str) -> bool:
        """Mark a join projection seen; ``False`` means a repeat to collapse."""
        if join_key in self._seen_join_keys:
            return False
        self._seen_join_keys.add(join_key)
        return True


def catalogued_columns(catalog: SchemaCatalog, table: str) -> tuple[str, ...]:
    """Catalogued column names for ``table`` (required plus optional)."""
    if table in catalog.tables:
        return tuple(column.name for column in catalog.tables[table])
    if table in catalog.session_scoped_tables:
        return tuple(column.name for column in catalog.session_scoped_tables[table])
    return ()


def describe_timestamp_units(catalog: SchemaCatalog) -> dict[str, str]:
    """Map every catalogued timestamp column (``table.column``) to its unit."""
    units: dict[str, str] = {}
    for table, columns in (*catalog.tables.items(), *catalog.session_scoped_tables.items()):
        for column in columns:
            if column.timestamp_unit is not None:
                units[f"{table}.{column.name}"] = column.timestamp_unit
    return units


def describe_classifications(catalog: SchemaCatalog) -> tuple[str, ...]:
    """Stable string descriptions of the catalog's classification rules."""
    descriptions: list[str] = []
    for rule in catalog.classifications:
        if rule.inherits_parent:
            target = f"{rule.table}:inherit-parent"
            if rule.exclude_synthetic:
                target += "(exclude-synthetic)"
        else:
            target = f"{rule.table}.{rule.match_column}"
            if rule.match_json_path:
                target += "." + ".".join(rule.match_json_path)
            target += f"={rule.match_value}"
            if rule.exclude_synthetic:
                target += "(exclude-synthetic)"
        descriptions.append(f"{target} -> {rule.name}")
    return tuple(descriptions)
