"""Versioned schema catalog for OpenCode SQLite source databases.

The catalog is pure data: it names the required tables/columns, optional
known columns, join predicates, timestamp columns and units, JSON
locations, record classification rules, permitted session-scoped tables,
authoritative sequence domains, fixed tie ranks, and raw-field handling
for each catalogued generation. ``docs/schema-catalog-v1.md`` mirrors this
matrix in prose so fixture DDL, exporter SQL, and adapter tests have one
reviewable source of truth.

Secret exclusion is encoded here: every catalog entry explicitly excludes
all values from the ``account``, ``account_state``, ``control_account``,
and ``credential`` tables, and omits every ``session_share`` column (no
safe-column entry exists in v1). A database that contains those tables
still matches a generation; their values are simply never selected, so
their presence cannot affect selection or turn counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from opencode_mlflow.source.model import (
    RecordClassificationName,
    SchemaGeneration,
    TimestampUnit,
)

SCHEMA_CATALOG_VERSION = "opencode-schema-catalog-v1"
"""Version of the catalog matrix; reported by schema descriptions."""

SECRET_TABLES: frozenset[str] = frozenset({"account", "account_state", "control_account", "credential"})
"""Tables whose values are never selected by any catalogued generation."""

SECURITY_EXCLUDED_COLUMNS: frozenset[str] = frozenset(
    {
        "share_url",
        "share_token",
        "share_secret",
        "access_token",
        "access_key",
        "session_token",
        "session_access_token",
    }
)
"""Column names that are never projected from any permitted source table.

These names are excluded independently of the catalog allowlist because
observed additive columns are otherwise selected for fidelity.  The
``session_share`` table remains omitted wholesale; this set also protects a
future or additive access/share column on a permitted table.
"""

SESSION_SHARE_TABLE = "session_share"
"""Table omitted entirely until a reviewed safe-column catalog entry exists."""

OMITTED_TABLE_REASONS: dict[str, str] = {
    SESSION_SHARE_TABLE: "no_safe_column_catalog_entry",
}
"""Safe reason codes for tables that are present but deliberately omitted."""


@dataclass(frozen=True)
class CatalogColumn:
    """One catalogued column of a source table.

    ``required`` columns must exist for the owning table to match a
    catalog entry; optional known columns are read when present. A
    ``timestamp_unit`` marks an OpenCode ``Timestamp`` column (epoch
    milliseconds in catalog v1). ``json_encoded`` marks a JSON location.
    """

    name: str
    required: bool = True
    timestamp_unit: TimestampUnit | None = None
    json_encoded: bool = False


@dataclass(frozen=True)
class JoinRule:
    """A catalogued join predicate between two source tables."""

    parent_table: str
    parent_column: str
    child_table: str
    child_column: str


@dataclass(frozen=True)
class RecordClassification:
    """Rule assigning a canonical classification to source records.

    A rule matches a row when the value at ``match_column`` (optionally at
    ``match_json_path`` inside that column's decoded JSON) equals
    ``match_value`` (string comparison of the decoded value; booleans
    compare via their lowercase JSON spelling). ``inherits_parent`` rules
    classify a child record from its parent instead; ``exclude_synthetic``
    keeps synthetic records out of the class.
    """

    name: RecordClassificationName
    table: str
    match_column: str | None = None
    match_json_path: tuple[str, ...] = ()
    match_value: str | None = None
    exclude_synthetic: bool = False
    inherits_parent: bool = False


@dataclass(frozen=True)
class SequenceDomain:
    """An explicit source sequence domain used for authoritative ordering."""

    table: str
    columns: tuple[str, ...]
    authoritative: bool
    unique: bool


@dataclass(frozen=True)
class JsonLocation:
    """A JSON value location: a whole column or a nested path within it."""

    table: str
    column: str
    path: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawFieldHandling:
    """Raw-field policy shared by the catalogued generations."""

    retain_raw_json: bool
    retain_raw_columns: bool
    unknown_field_warning: str
    malformed_value_warning: str


@dataclass(frozen=True)
class SchemaCatalog:
    """One catalogued source schema generation."""

    name: str
    generation: SchemaGeneration
    version: str
    tables: dict[str, tuple[CatalogColumn, ...]]
    session_scoped_tables: dict[str, tuple[CatalogColumn, ...]]
    joins: tuple[JoinRule, ...]
    session_scoped_joins: tuple[JoinRule, ...]
    classifications: tuple[RecordClassification, ...]
    sequence_domains: tuple[SequenceDomain, ...]
    tie_ranks: dict[str, int]
    json_locations: tuple[JsonLocation, ...]
    raw_field_handling: RawFieldHandling
    excluded_tables: frozenset[str] = SECRET_TABLES
    omitted_tables: dict[str, str] = field(default_factory=lambda: dict(OMITTED_TABLE_REASONS))


TIE_RANKS: dict[str, int] = {
    "message_container": 10,
    "content_part": 20,
    "lifecycle_event": 30,
    "scoped_record": 40,
    "unknown_record": 90,
}
"""Fixed source-type tie ranks; catalog data, never query order."""

RAW_FIELD_HANDLING = RawFieldHandling(
    retain_raw_json=True,
    retain_raw_columns=True,
    unknown_field_warning="unknown_source_field",
    malformed_value_warning="malformed_source_value",
)


def _plain(name: str) -> CatalogColumn:
    return CatalogColumn(name=name)


def _json(name: str) -> CatalogColumn:
    return CatalogColumn(name=name, json_encoded=True)


def _timestamp(name: str) -> CatalogColumn:
    return CatalogColumn(name=name, timestamp_unit="ms")


def _optional(column: CatalogColumn) -> CatalogColumn:
    return replace(column, required=False)


_LEGACY_SESSION_COLUMNS: tuple[CatalogColumn, ...] = (
    _plain("id"),
    _plain("project_id"),
    _plain("workspace_id"),
    _plain("parent_id"),
    _plain("slug"),
    _plain("directory"),
    _plain("path"),
    _plain("title"),
    _plain("version"),
    _plain("summary_additions"),
    _plain("summary_deletions"),
    _plain("summary_files"),
    _json("summary_diffs"),
    _json("metadata"),
    _plain("cost"),
    _plain("tokens_input"),
    _plain("tokens_output"),
    _plain("tokens_reasoning"),
    _plain("tokens_cache_read"),
    _plain("tokens_cache_write"),
    _plain("revert"),
    _plain("permission"),
    _plain("agent"),
    _json("model"),
    _timestamp("time_created"),
    _timestamp("time_updated"),
    _optional(_timestamp("time_compacting")),
    _optional(_timestamp("time_archived")),
)

_LEGACY_MESSAGE_COLUMNS: tuple[CatalogColumn, ...] = (
    _plain("id"),
    _plain("session_id"),
    _timestamp("time_created"),
    _timestamp("time_updated"),
    _json("data"),
)

_LEGACY_PART_COLUMNS: tuple[CatalogColumn, ...] = (
    _plain("id"),
    _plain("message_id"),
    _plain("session_id"),
    _timestamp("time_created"),
    _timestamp("time_updated"),
    _json("data"),
)

_LEGACY_TODO_COLUMNS: tuple[CatalogColumn, ...] = (
    _plain("session_id"),
    _plain("content"),
    _plain("status"),
    _plain("priority"),
    _plain("position"),
    _timestamp("time_created"),
    _timestamp("time_updated"),
)

_LEGACY_CLASSIFICATIONS: tuple[RecordClassification, ...] = (
    RecordClassification(
        name="user",
        table="message",
        match_column="data",
        match_json_path=("role",),
        match_value="user",
    ),
    RecordClassification(
        name="assistant",
        table="message",
        match_column="data",
        match_json_path=("role",),
        match_value="assistant",
    ),
    RecordClassification(
        name="system",
        table="message",
        match_column="data",
        match_json_path=("role",),
        match_value="system",
    ),
    # Parts inherit their parent message's classification, except
    # synthetic/ignored marker parts.
    RecordClassification(name="user", table="part", inherits_parent=True, exclude_synthetic=True),
    RecordClassification(
        name="synthetic", table="part", match_column="data", match_json_path=("type",), match_value="step-start"
    ),
    RecordClassification(
        name="synthetic", table="part", match_column="data", match_json_path=("type",), match_value="step-finish"
    ),
)

LEGACY_CATALOG_V1 = SchemaCatalog(
    name="legacy",
    generation="legacy",
    version=SCHEMA_CATALOG_VERSION,
    tables={
        "session": _LEGACY_SESSION_COLUMNS,
        "message": _LEGACY_MESSAGE_COLUMNS,
        "part": _LEGACY_PART_COLUMNS,
    },
    session_scoped_tables={"todo": _LEGACY_TODO_COLUMNS},
    joins=(
        JoinRule(parent_table="session", parent_column="id", child_table="message", child_column="session_id"),
        JoinRule(parent_table="message", parent_column="id", child_table="part", child_column="message_id"),
        JoinRule(parent_table="session", parent_column="id", child_table="part", child_column="session_id"),
    ),
    session_scoped_joins=(
        JoinRule(parent_table="session", parent_column="id", child_table="todo", child_column="session_id"),
    ),
    classifications=_LEGACY_CLASSIFICATIONS,
    sequence_domains=(),
    tie_ranks=dict(TIE_RANKS),
    json_locations=(
        JsonLocation(table="message", column="data"),
        JsonLocation(table="part", column="data"),
        JsonLocation(table="session", column="model"),
        JsonLocation(table="session", column="metadata"),
        JsonLocation(table="session", column="summary_diffs"),
    ),
    raw_field_handling=RAW_FIELD_HANDLING,
)

_CURRENT_SESSION_V2_COLUMNS: tuple[CatalogColumn, ...] = (
    _plain("id"),
    _plain("project_id"),
    _plain("workspace_id"),
    _plain("parent_id"),
    _plain("fork_session_id"),
    _timestamp("fork_boundary"),
    _plain("slug"),
    _plain("directory"),
    _plain("path"),
    _plain("title"),
    _plain("version"),
    _plain("summary_additions"),
    _plain("summary_deletions"),
    _plain("summary_files"),
    _json("summary_diffs"),
    _json("metadata"),
    _plain("cost"),
    _plain("tokens_input"),
    _plain("tokens_output"),
    _plain("tokens_reasoning"),
    _plain("tokens_cache_read"),
    _plain("tokens_cache_write"),
    _plain("revert"),
    _plain("permission"),
    _plain("agent"),
    _json("model"),
    _timestamp("time_created"),
    _timestamp("time_updated"),
    _timestamp("time_idle"),
    _timestamp("time_viewed"),
    _plain("idle_outcome"),
    _timestamp("time_compacting"),
    _timestamp("time_archived"),
    _optional(_timestamp("time_suspended")),
    _optional(_plain("resume_attempts")),
)

_CURRENT_SESSION_MESSAGE_COLUMNS: tuple[CatalogColumn, ...] = (
    _plain("id"),
    _plain("session_id"),
    _plain("type"),
    _plain("seq"),
    _timestamp("time_created"),
    _timestamp("time_updated"),
    _json("data"),
)

_CURRENT_SESSION_PENDING_COLUMNS: tuple[CatalogColumn, ...] = (
    _plain("id"),
    _plain("session_id"),
    _plain("type"),
    _json("data"),
    _plain("delivery"),
    _plain("admitted_seq"),
    _timestamp("time_created"),
)

_CURRENT_SESSION_INBOX_COLUMNS: tuple[CatalogColumn, ...] = (
    _plain("id"),
    _plain("session_id"),
    _plain("type"),
    _json("payload"),
    _plain("delivery"),
    _plain("enqueued_seq"),
    _timestamp("time_created"),
)

_CURRENT_INSTRUCTION_ENTRY_COLUMNS: tuple[CatalogColumn, ...] = (
    _plain("session_id"),
    _plain("key"),
    _plain("value"),
    _plain("removed"),
    _timestamp("time_created"),
    _timestamp("time_updated"),
)

_CURRENT_INSTRUCTION_BLOB_COLUMNS: tuple[CatalogColumn, ...] = (
    _plain("hash"),
    _plain("value"),
)

_CURRENT_INSTRUCTION_STATE_COLUMNS: tuple[CatalogColumn, ...] = (
    _plain("session_id"),
    _plain("epoch_start"),
    _plain("through_seq"),
    _json("initial_values"),
    _json("current_values"),
)

_CURRENT_CLASSIFICATIONS: tuple[RecordClassification, ...] = (
    RecordClassification(
        name="user",
        table="session_message",
        match_column="type",
        match_value="user",
        exclude_synthetic=True,
    ),
    RecordClassification(
        name="assistant",
        table="session_message",
        match_column="type",
        match_value="assistant",
    ),
    RecordClassification(
        name="system",
        table="session_message",
        match_column="type",
        match_value="system",
    ),
    RecordClassification(
        name="synthetic",
        table="session_message",
        match_column="data",
        match_json_path=("synthetic",),
        match_value="true",
    ),
)

CURRENT_CATALOG_V1 = SchemaCatalog(
    name="current",
    generation="current",
    version=SCHEMA_CATALOG_VERSION,
    tables={
        "session_v2": _CURRENT_SESSION_V2_COLUMNS,
        "session_message": _CURRENT_SESSION_MESSAGE_COLUMNS,
    },
    session_scoped_tables={
        "session_pending": _CURRENT_SESSION_PENDING_COLUMNS,
        "session_inbox": _CURRENT_SESSION_INBOX_COLUMNS,
        "instruction_entry": _CURRENT_INSTRUCTION_ENTRY_COLUMNS,
        "instruction_blob": _CURRENT_INSTRUCTION_BLOB_COLUMNS,
        "instruction_state": _CURRENT_INSTRUCTION_STATE_COLUMNS,
    },
    joins=(
        JoinRule(
            parent_table="session_v2", parent_column="id", child_table="session_message", child_column="session_id"
        ),
    ),
    session_scoped_joins=(
        JoinRule(
            parent_table="session_v2", parent_column="id", child_table="session_pending", child_column="session_id"
        ),
        JoinRule(parent_table="session_v2", parent_column="id", child_table="session_inbox", child_column="session_id"),
        JoinRule(
            parent_table="session_v2", parent_column="id", child_table="instruction_entry", child_column="session_id"
        ),
        JoinRule(
            parent_table="session_v2", parent_column="id", child_table="instruction_state", child_column="session_id"
        ),
        # instruction_blob carries no session column; it is session-scoped
        # only transitively, through instruction_entry values referencing
        # blob hashes.
        JoinRule(
            parent_table="instruction_entry", parent_column="value", child_table="instruction_blob", child_column="hash"
        ),
    ),
    classifications=_CURRENT_CLASSIFICATIONS,
    sequence_domains=(
        SequenceDomain(table="session_message", columns=("session_id", "seq"), authoritative=True, unique=True),
    ),
    tie_ranks=dict(TIE_RANKS),
    json_locations=(
        JsonLocation(table="session_message", column="data"),
        JsonLocation(table="session_message", column="data", path=("content",)),
        JsonLocation(table="session_message", column="data", path=("model",)),
        JsonLocation(table="session_v2", column="model"),
        JsonLocation(table="session_v2", column="metadata"),
        JsonLocation(table="session_v2", column="summary_diffs"),
    ),
    raw_field_handling=RAW_FIELD_HANDLING,
)

OPENCODE_SCHEMA_CATALOG_V1: tuple[SchemaCatalog, ...] = (LEGACY_CATALOG_V1, CURRENT_CATALOG_V1)
"""The versioned v1 catalog: the legacy and current schema generations."""

__all__ = [
    "CURRENT_CATALOG_V1",
    "LEGACY_CATALOG_V1",
    "OMITTED_TABLE_REASONS",
    "OPENCODE_SCHEMA_CATALOG_V1",
    "SCHEMA_CATALOG_VERSION",
    "SECRET_TABLES",
    "SESSION_SHARE_TABLE",
    "CatalogColumn",
    "JsonLocation",
    "JoinRule",
    "RawFieldHandling",
    "RecordClassification",
    "SchemaCatalog",
    "SequenceDomain",
    "TIE_RANKS",
]
