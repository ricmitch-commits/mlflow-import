"""Unit tests for the versioned OpenCode schema catalog (v1)."""

from __future__ import annotations

import pytest

from opencode_mlflow.source.catalog import (
    CURRENT_CATALOG_V1,
    LEGACY_CATALOG_V1,
    OPENCODE_SCHEMA_CATALOG_V1,
    SCHEMA_CATALOG_VERSION,
    SECRET_TABLES,
    SESSION_SHARE_TABLE,
    SchemaCatalog,
)

pytestmark = pytest.mark.unit


def _columns(catalog: SchemaCatalog, table: str) -> list[str]:
    return [column.name for column in catalog.tables[table]]


def _optional_columns(catalog: SchemaCatalog, table: str) -> set[str]:
    return {column.name for column in catalog.tables[table] if not column.required}


def _timestamp_columns(catalog: SchemaCatalog, table: str) -> dict[str, str]:
    return {
        column.name: str(column.timestamp_unit) for column in catalog.tables[table] if column.timestamp_unit is not None
    }


def _scoped_columns(catalog: SchemaCatalog, table: str) -> list[str]:
    return [column.name for column in catalog.session_scoped_tables[table]]


class TestCatalogConstants:
    def test_versioned_catalog_contains_legacy_and_current_entries(self) -> None:
        assert isinstance(OPENCODE_SCHEMA_CATALOG_V1, tuple)
        assert [catalog.name for catalog in OPENCODE_SCHEMA_CATALOG_V1] == ["legacy", "current"]
        assert [catalog.generation for catalog in OPENCODE_SCHEMA_CATALOG_V1] == ["legacy", "current"]

    def test_every_entry_shares_the_catalog_version(self) -> None:
        assert SCHEMA_CATALOG_VERSION == "opencode-schema-catalog-v1"
        for catalog in OPENCODE_SCHEMA_CATALOG_V1:
            assert catalog.version == SCHEMA_CATALOG_VERSION

    def test_tie_ranks_are_catalog_data_with_fixed_values(self) -> None:
        expected = {
            "message_container": 10,
            "content_part": 20,
            "lifecycle_event": 30,
            "scoped_record": 40,
            "unknown_record": 90,
        }
        for catalog in OPENCODE_SCHEMA_CATALOG_V1:
            assert catalog.tie_ranks == expected
            assert all(isinstance(rank, int) for rank in catalog.tie_ranks.values())

    def test_raw_field_handling_retains_raw_values_with_warning_codes(self) -> None:
        for catalog in OPENCODE_SCHEMA_CATALOG_V1:
            handling = catalog.raw_field_handling
            assert handling.retain_raw_json is True
            assert handling.retain_raw_columns is True
            assert handling.unknown_field_warning == "unknown_source_field"
            assert handling.malformed_value_warning == "malformed_source_value"

    def test_secret_tables_are_excluded_from_every_catalog(self) -> None:
        assert SECRET_TABLES == frozenset({"account", "account_state", "control_account", "credential"})
        for catalog in OPENCODE_SCHEMA_CATALOG_V1:
            assert SECRET_TABLES <= catalog.excluded_tables
            assert SESSION_SHARE_TABLE in catalog.omitted_tables
            assert catalog.omitted_tables[SESSION_SHARE_TABLE] == "no_safe_column_catalog_entry"
            known_tables = set(catalog.tables) | set(catalog.session_scoped_tables)
            assert known_tables.isdisjoint(catalog.excluded_tables)
            assert known_tables.isdisjoint(catalog.omitted_tables)


class TestLegacyCatalog:
    def test_session_columns(self) -> None:
        assert _columns(LEGACY_CATALOG_V1, "session") == [
            "id",
            "project_id",
            "workspace_id",
            "parent_id",
            "slug",
            "directory",
            "path",
            "title",
            "version",
            "summary_additions",
            "summary_deletions",
            "summary_files",
            "summary_diffs",
            "metadata",
            "cost",
            "tokens_input",
            "tokens_output",
            "tokens_reasoning",
            "tokens_cache_read",
            "tokens_cache_write",
            "revert",
            "permission",
            "agent",
            "model",
            "time_created",
            "time_updated",
            "time_compacting",
            "time_archived",
        ]

    def test_message_columns(self) -> None:
        assert _columns(LEGACY_CATALOG_V1, "message") == [
            "id",
            "session_id",
            "time_created",
            "time_updated",
            "data",
        ]

    def test_part_columns(self) -> None:
        assert _columns(LEGACY_CATALOG_V1, "part") == [
            "id",
            "message_id",
            "session_id",
            "time_created",
            "time_updated",
            "data",
        ]

    def test_optional_known_columns(self) -> None:
        assert _optional_columns(LEGACY_CATALOG_V1, "session") == {"time_compacting", "time_archived"}
        for table in ("message", "part"):
            assert _optional_columns(LEGACY_CATALOG_V1, table) == set()

    def test_timestamp_columns_use_epoch_milliseconds(self) -> None:
        assert _timestamp_columns(LEGACY_CATALOG_V1, "session") == {
            "time_created": "ms",
            "time_updated": "ms",
            "time_compacting": "ms",
            "time_archived": "ms",
        }
        assert _timestamp_columns(LEGACY_CATALOG_V1, "message") == {
            "time_created": "ms",
            "time_updated": "ms",
        }
        assert _timestamp_columns(LEGACY_CATALOG_V1, "part") == {
            "time_created": "ms",
            "time_updated": "ms",
        }

    def test_json_locations(self) -> None:
        locations = {(location.table, location.column, location.path) for location in LEGACY_CATALOG_V1.json_locations}
        assert ("message", "data", ()) in locations
        assert ("part", "data", ()) in locations
        assert ("session", "model", ()) in locations
        assert ("session", "metadata", ()) in locations
        assert ("session", "summary_diffs", ()) in locations

    def test_join_rules(self) -> None:
        joins = {
            (join.parent_table, join.parent_column, join.child_table, join.child_column)
            for join in LEGACY_CATALOG_V1.joins
        }
        assert ("session", "id", "message", "session_id") in joins
        assert ("message", "id", "part", "message_id") in joins
        assert ("session", "id", "part", "session_id") in joins

    def test_legacy_has_no_authoritative_sequence_domain(self) -> None:
        assert LEGACY_CATALOG_V1.sequence_domains == ()

    def test_user_classification_uses_message_role(self) -> None:
        rule = _classification(LEGACY_CATALOG_V1, "user", "message")
        assert rule is not None
        assert rule.match_column == "data"
        assert rule.match_json_path == ("role",)
        assert rule.match_value == "user"
        assert rule.exclude_synthetic is False
        assert rule.inherits_parent is False

    def test_assistant_and_system_classifications_use_message_role(self) -> None:
        assistant = _classification(LEGACY_CATALOG_V1, "assistant", "message")
        system = _classification(LEGACY_CATALOG_V1, "system", "message")
        assert assistant is not None and assistant.match_value == "assistant"
        assert system is not None and system.match_value == "system"

    def test_part_classification_inherits_parent_and_excludes_synthetic(self) -> None:
        rule = _classification(LEGACY_CATALOG_V1, "user", "part")
        assert rule is not None
        assert rule.inherits_parent is True
        assert rule.exclude_synthetic is True

    def test_synthetic_part_types_are_catalogued(self) -> None:
        synthetic_rules = [
            rule for rule in LEGACY_CATALOG_V1.classifications if rule.table == "part" and rule.name == "synthetic"
        ]
        assert {rule.match_value for rule in synthetic_rules} == {"step-start", "step-finish"}
        assert all(rule.match_json_path == ("type",) for rule in synthetic_rules)

    def test_todo_session_scoped_columns(self) -> None:
        assert _scoped_columns(LEGACY_CATALOG_V1, "todo") == [
            "session_id",
            "content",
            "status",
            "priority",
            "position",
            "time_created",
            "time_updated",
        ]

    def test_todo_join_is_unambiguous(self) -> None:
        joins = {
            (join.parent_table, join.parent_column, join.child_table, join.child_column)
            for join in LEGACY_CATALOG_V1.session_scoped_joins
        }
        assert ("session", "id", "todo", "session_id") in joins


class TestCurrentCatalog:
    def test_session_v2_columns(self) -> None:
        assert _columns(CURRENT_CATALOG_V1, "session_v2") == [
            "id",
            "project_id",
            "workspace_id",
            "parent_id",
            "fork_session_id",
            "fork_boundary",
            "slug",
            "directory",
            "path",
            "title",
            "version",
            "summary_additions",
            "summary_deletions",
            "summary_files",
            "summary_diffs",
            "metadata",
            "cost",
            "tokens_input",
            "tokens_output",
            "tokens_reasoning",
            "tokens_cache_read",
            "tokens_cache_write",
            "revert",
            "permission",
            "agent",
            "model",
            "time_created",
            "time_updated",
            "time_idle",
            "time_viewed",
            "idle_outcome",
            "time_compacting",
            "time_archived",
            "time_suspended",
            "resume_attempts",
        ]

    def test_session_message_columns(self) -> None:
        assert _columns(CURRENT_CATALOG_V1, "session_message") == [
            "id",
            "session_id",
            "type",
            "seq",
            "time_created",
            "time_updated",
            "data",
        ]

    def test_optional_known_columns(self) -> None:
        assert _optional_columns(CURRENT_CATALOG_V1, "session_v2") == {"time_suspended", "resume_attempts"}
        assert _optional_columns(CURRENT_CATALOG_V1, "session_message") == set()

    def test_timestamp_columns_use_epoch_milliseconds(self) -> None:
        assert _timestamp_columns(CURRENT_CATALOG_V1, "session_v2") == {
            "time_created": "ms",
            "time_updated": "ms",
            "fork_boundary": "ms",
            "time_idle": "ms",
            "time_viewed": "ms",
            "time_compacting": "ms",
            "time_archived": "ms",
            "time_suspended": "ms",
        }
        assert _timestamp_columns(CURRENT_CATALOG_V1, "session_message") == {
            "time_created": "ms",
            "time_updated": "ms",
        }

    def test_json_locations_include_nested_session_message_fields(self) -> None:
        locations = {(location.table, location.column, location.path) for location in CURRENT_CATALOG_V1.json_locations}
        assert ("session_message", "data", ()) in locations
        assert ("session_message", "data", ("content",)) in locations
        assert ("session_message", "data", ("model",)) in locations
        assert ("session_v2", "model", ()) in locations
        assert ("session_v2", "metadata", ()) in locations
        assert ("session_v2", "summary_diffs", ()) in locations

    def test_join_rules(self) -> None:
        joins = {
            (join.parent_table, join.parent_column, join.child_table, join.child_column)
            for join in CURRENT_CATALOG_V1.joins
        }
        assert ("session_v2", "id", "session_message", "session_id") in joins

    def test_authoritative_sequence_domain(self) -> None:
        (domain,) = CURRENT_CATALOG_V1.sequence_domains
        assert domain.table == "session_message"
        assert domain.columns == ("session_id", "seq")
        assert domain.authoritative is True
        assert domain.unique is True

    def test_user_classification_uses_message_type_excluding_synthetic(self) -> None:
        rule = _classification(CURRENT_CATALOG_V1, "user", "session_message")
        assert rule is not None
        assert rule.match_column == "type"
        assert rule.match_json_path == ()
        assert rule.match_value == "user"
        assert rule.exclude_synthetic is True

    def test_assistant_and_system_classifications_use_message_type(self) -> None:
        assistant = _classification(CURRENT_CATALOG_V1, "assistant", "session_message")
        system = _classification(CURRENT_CATALOG_V1, "system", "session_message")
        assert assistant is not None and assistant.match_value == "assistant"
        assert system is not None and system.match_value == "system"

    def test_synthetic_session_message_rule_uses_data_flag(self) -> None:
        rule = _classification(CURRENT_CATALOG_V1, "synthetic", "session_message")
        assert rule is not None
        assert rule.match_column == "data"
        assert rule.match_json_path == ("synthetic",)
        assert rule.match_value == "true"

    def test_session_scoped_tables_and_columns(self) -> None:
        assert _scoped_columns(CURRENT_CATALOG_V1, "session_pending") == [
            "id",
            "session_id",
            "type",
            "data",
            "delivery",
            "admitted_seq",
            "time_created",
        ]
        assert _scoped_columns(CURRENT_CATALOG_V1, "session_inbox") == [
            "id",
            "session_id",
            "type",
            "payload",
            "delivery",
            "enqueued_seq",
            "time_created",
        ]
        assert _scoped_columns(CURRENT_CATALOG_V1, "instruction_entry") == [
            "session_id",
            "key",
            "value",
            "removed",
            "time_created",
            "time_updated",
        ]
        assert _scoped_columns(CURRENT_CATALOG_V1, "instruction_blob") == ["hash", "value"]
        assert _scoped_columns(CURRENT_CATALOG_V1, "instruction_state") == [
            "session_id",
            "epoch_start",
            "through_seq",
            "initial_values",
            "current_values",
        ]

    def test_session_scoped_joins_are_unambiguous(self) -> None:
        joins = {
            (join.parent_table, join.parent_column, join.child_table, join.child_column)
            for join in CURRENT_CATALOG_V1.session_scoped_joins
        }
        assert ("session_v2", "id", "session_pending", "session_id") in joins
        assert ("session_v2", "id", "session_inbox", "session_id") in joins
        assert ("session_v2", "id", "instruction_entry", "session_id") in joins
        assert ("session_v2", "id", "instruction_state", "session_id") in joins
        # instruction_blob is session-scoped only transitively, through
        # instruction_entry's value referencing the blob hash.
        assert ("instruction_entry", "value", "instruction_blob", "hash") in joins

    def test_json_flags_on_columns(self) -> None:
        legacy_message = {column.name: column for column in LEGACY_CATALOG_V1.tables["message"]}
        assert legacy_message["data"].json_encoded is True
        current_session = {column.name: column for column in CURRENT_CATALOG_V1.tables["session_v2"]}
        assert current_session["model"].json_encoded is True
        assert current_session["metadata"].json_encoded is True
        assert current_session["summary_diffs"].json_encoded is True
        assert current_session["time_created"].json_encoded is False


def _classification(catalog: SchemaCatalog, name: str, table: str):
    for rule in catalog.classifications:
        if rule.name == name and rule.table == table:
            return rule
    return None
