"""Contract tests for the source adapters over the full live pipeline.

Every parametrized test drives the complete Task-4 pipeline for each
catalogued generation: the sanitized synthetic SQL fixture database is
materialized, opened through the live read-only connection, probed, an
exporter is built through the adapter registry, a normalized snapshot is
extracted, and the replay adapter is built from the same registry. The
same assertions apply to the legacy, current, and mixed fixtures so the
three exporters cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pytest

from opencode_mlflow.errors import SourceCompatibilityError
from opencode_mlflow.source.catalog import SCHEMA_CATALOG_VERSION, TIE_RANKS
from opencode_mlflow.source.connection import open_live_connection
from opencode_mlflow.source.current import CurrentAdapter
from opencode_mlflow.source.legacy import LegacyAdapter
from opencode_mlflow.source.model import SchemaDescription, SourceAdapter
from opencode_mlflow.source.probe import probe_schema
from opencode_mlflow.source.registry import (
    AdapterRegistry,
    MixedAdapter,
    build_live_exporter,
    build_replay_adapter,
    default_adapter_registry,
)
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter
from opencode_mlflow.source.snapshot import SourceExporter, SourceSnapshot, create_normalized_snapshot
from tests.support.fake_source import identity_redaction
from tests.support.sqlite_builder import build_fixture_db

pytestmark = pytest.mark.contract

SECRET_SENTINELS = (
    "credential-sentinel",
    "ACCOUNT-TOKEN-SENTINEL",
    "ACCOUNT-STATE-SENTINEL",
    "CONTROL-ACCOUNT-SENTINEL",
    "SHARE-SENTINEL",
)

EXPECTED_SESSIONS: dict[str, list[str]] = {
    "legacy": ["ses-legacy-001", "ses-legacy-002", "ses-legacy-003"],
    "current": ["ses-current-001", "ses-current-002"],
    "mixed": [
        "ses-mixed-legacy-001",
        "ses-mixed-current-002",
        "ses-mixed-shared-001",
        "ses-mixed-current-001",
    ],
}

EXPECTED_HEADER_MODELS: dict[str, str | None] = {
    "legacy": "ses-legacy-001",
    "current": "ses-current-001",
    "mixed": None,
}

EXPECTED_MODELS: dict[str, dict[str, str]] = {
    "legacy": {"provider": "synth-provider", "model": "synth-legacy-model"},
    "current": {"provider": "synth-provider", "model": "synth-model"},
    "mixed": {"provider": "synth-provider", "model": "synth-mixed-model"},
}

KNOWN_RAW_JSON: dict[str, str] = {
    "legacy": '{"type": "text", "text": "synthetic legacy user text"}',
    "current": (
        '{"content": [{"type": "text", "text": "synthetic user prompt one"}],'
        ' "model": {"provider": "synth-provider", "model": "synth-model"}}'
    ),
    "mixed": '{"broken',
}


@dataclass
class AdapterContractDatabase:
    """One exported fixture database with its replay adapter still open."""

    schema_name: str
    path: Path
    description: SchemaDescription
    snapshot: SourceSnapshot
    adapter: SourceAdapter


@pytest.fixture
def fixture_database(tmp_path: Path) -> Callable[[str], AdapterContractDatabase]:
    """Build the full live pipeline for a named fixture and keep it open."""
    stack = ExitStack()

    def create(schema_name: str) -> AdapterContractDatabase:
        source = tmp_path / f"{schema_name}-source.db"
        connection = build_fixture_db(source, schema_name)
        connection.close()
        registry = default_adapter_registry()
        with ExitStack() as setup:
            live = setup.enter_context(_open_live(source))
            description = probe_schema(live.connection)
            exporter = build_live_exporter(live.connection, description, registry)
        snapshot = stack.enter_context(create_normalized_snapshot(source, tmp_path, exporter, identity_redaction))
        adapter = build_replay_adapter(snapshot, registry)
        return AdapterContractDatabase(
            schema_name=schema_name,
            path=source,
            description=description,
            snapshot=snapshot,
            adapter=adapter,
        )

    yield create
    stack.close()


def _open_live(source: Path):
    return open_live_connection(source)


def _all_records(database: AdapterContractDatabase) -> list:
    records: list = []
    for header in database.adapter.iter_session_headers():
        records.extend(database.adapter.iter_records(header.session_id))
        records.extend(database.adapter.iter_session_scoped_records(header.session_id))
    return records


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_emits_canonical_records(schema_name, fixture_database):
    from opencode_mlflow.source.ordering import session_sort_key

    adapter = fixture_database(schema_name).adapter
    headers = list(adapter.iter_session_headers())

    assert headers == sorted(headers, key=session_sort_key)
    for header in headers:
        for record in adapter.iter_records(header.session_id):
            assert record.session_id == header.session_id
            assert "credential-sentinel" not in repr(record)


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_probes_and_selects_the_generation(schema_name, fixture_database):
    database = fixture_database(schema_name)

    assert database.description.generation == schema_name
    assert database.adapter.schema_generation == schema_name
    assert database.snapshot.schema_description.generation == schema_name
    assert database.snapshot.stats.complete is True
    assert database.snapshot.stats.schema_generation == schema_name
    assert database.snapshot.stats.session_count == len(EXPECTED_SESSIONS[schema_name])
    assert database.snapshot.stats.record_count > 0


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_orders_sessions_by_creation_time(schema_name, fixture_database):
    adapter = fixture_database(schema_name).adapter

    assert [header.session_id for header in adapter.iter_session_headers()] == EXPECTED_SESSIONS[schema_name]


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_emits_normalized_headers(schema_name, fixture_database):
    adapter = fixture_database(schema_name).adapter
    headers = list(adapter.iter_session_headers())

    assert headers
    for header in headers:
        assert header.session_id
        # The database description may be mixed, but every emitted session
        # belongs to exactly one generation.
        assert header.adapter_generation in ("legacy", "current")
        assert isinstance(header.raw_columns, dict)
        assert header.raw_columns.get("id") == header.session_id


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_never_leaks_secret_values(schema_name, fixture_database):
    database = fixture_database(schema_name)
    headers = list(database.adapter.iter_session_headers())
    records = _all_records(database)

    for sentinel in SECRET_SENTINELS:
        assert all(sentinel not in repr(header) for header in headers)
        assert all(sentinel not in repr(record) for record in records)


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_emits_one_record_per_source_identity(schema_name, fixture_database):
    database = fixture_database(schema_name)
    records = _all_records(database)

    explicit = [
        (
            record.session_id if record.source_table == "instruction_blob" else None,
            record.source_table,
            record.source_id,
        )
        for record in records
        if record.source_id is not None
    ]
    assert len(explicit) == len(set(explicit))
    identities = [record.source_identity for record in records if record.source_identity is not None]
    assert len(identities) == len(set(identities))
    # A missing ID never yields a None dedup key.
    assert all(record.source_id is not None or record.source_identity is not None for record in records)


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_parents_precede_children(schema_name, fixture_database):
    adapter = fixture_database(schema_name).adapter

    for header in adapter.iter_session_headers():
        seen: set[str] = set()
        for record in adapter.iter_records(header.session_id):
            if record.parent_source_id is not None:
                assert record.parent_source_id in seen
            if record.source_id is not None:
                seen.add(record.source_id)
            if record.source_identity is not None:
                seen.add(record.source_identity)


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_preserves_raw_json(schema_name, fixture_database):
    records = _all_records(fixture_database(schema_name))

    raw_jsons = [record.raw_json for record in records if record.raw_json is not None]
    assert KNOWN_RAW_JSON[schema_name] in raw_jsons


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_exposes_model_discovery_inputs(schema_name, fixture_database):
    database = fixture_database(schema_name)
    headers = list(database.adapter.iter_session_headers())
    records = _all_records(database)

    # Model identities are collectible from session defaults and from
    # records (message and part level, nested current locations).
    expected_model = EXPECTED_MODELS[schema_name]
    if EXPECTED_HEADER_MODELS[schema_name] is not None:
        header = next(h for h in headers if h.session_id == EXPECTED_HEADER_MODELS[schema_name])
        assert header.model == expected_model
    decoded_models = [
        record.decoded_value.get("model")
        for record in records
        if isinstance(record.decoded_value, dict) and isinstance(record.decoded_value.get("model"), dict)
    ]
    assert decoded_models
    assert all(model == expected_model for model in decoded_models)


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_persists_inspect_enrichment(schema_name, fixture_database):
    """The exporter's enriched description is persisted on the snapshot.

    ``create_normalized_snapshot`` keeps ``exporter.inspect()`` as the
    snapshot's schema description, so the rank table, timestamp-unit
    rules, and classification descriptions must survive the pipeline for
    every generation — including the mixed composer, which merges both
    catalogs' enrichment.
    """
    description = fixture_database(schema_name).snapshot.schema_description

    assert description.tie_ranks == dict(TIE_RANKS)
    assert description.timestamp_units
    assert description.classifications
    if schema_name == "mixed":
        # The mixed description carries BOTH catalogs' enrichment: the
        # union of timestamp-unit rules (disjoint table names) and both
        # generations' classification rules.
        assert "session.time_created" in description.timestamp_units
        assert "session_v2.time_created" in description.timestamp_units
        assert any(rule.startswith("message.data.role=user") for rule in description.classifications)
        assert any(rule.startswith("session_message.type=user") for rule in description.classifications)
    elif schema_name == "legacy":
        assert "session.time_created" in description.timestamp_units
        assert "session_v2.time_created" not in description.timestamp_units
        assert any(rule.startswith("message.data.role=user") for rule in description.classifications)
        assert not any(rule.startswith("session_message.") for rule in description.classifications)
    else:
        assert "session_v2.time_created" in description.timestamp_units
        assert "session.time_created" not in description.timestamp_units
        assert any(rule.startswith("session_message.type=user") for rule in description.classifications)
        assert not any(rule.startswith("message.") for rule in description.classifications)


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_filters_records_by_session_id(schema_name, fixture_database):
    adapter = fixture_database(schema_name).adapter
    headers = list(adapter.iter_session_headers())

    for header in headers:
        records = list(adapter.iter_records(header.session_id))
        assert records
        assert all(record.session_id == header.session_id for record in records)

    assert list(adapter.iter_records("ses-does-not-exist")) == []
    assert list(adapter.iter_session_scoped_records("ses-does-not-exist")) == []


@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_separates_session_scoped_records(schema_name, fixture_database):
    adapter = fixture_database(schema_name).adapter

    for header in adapter.iter_session_headers():
        turn_tables = {record.source_table for record in adapter.iter_records(header.session_id)}
        scoped_tables = {record.source_table for record in adapter.iter_session_scoped_records(header.session_id)}
        assert not (turn_tables & scoped_tables)


class TestMixedPrecedence:
    def test_mixed_current_wins_overlapping_sessions(self, fixture_database) -> None:
        database = fixture_database("mixed")
        headers = {header.session_id: header for header in database.adapter.iter_session_headers()}

        shared = headers["ses-mixed-shared-001"]
        assert shared.adapter_generation == "current"
        # Only the current generation's records are emitted for the
        # shared ID; the legacy row is suppressed, never merged.
        records = list(database.adapter.iter_records("ses-mixed-shared-001"))
        assert [record.source_id for record in records] == [
            "msg-mixed-shared-current-001",
            "msg-mixed-shared-current-001#content-0",
        ]

    def test_mixed_suppressed_overlap_is_retained_as_provenance(self, fixture_database) -> None:
        database = fixture_database("mixed")
        headers = {header.session_id: header for header in database.adapter.iter_session_headers()}

        assert "legacy_session_suppressed" in headers["ses-mixed-shared-001"].warnings
        # Non-overlapping current sessions carry no suppression warning.
        assert "legacy_session_suppressed" not in headers["ses-mixed-current-001"].warnings

    def test_mixed_exports_legacy_only_sessions(self, fixture_database) -> None:
        database = fixture_database("mixed")
        headers = {header.session_id: header for header in database.adapter.iter_session_headers()}

        legacy = headers["ses-mixed-legacy-001"]
        assert legacy.adapter_generation == "legacy"
        assert "legacy_session_suppressed" not in legacy.warnings
        records = list(database.adapter.iter_records("ses-mixed-legacy-001"))
        assert [record.source_id for record in records] == [
            "msg-mixed-legacy-001",
            "part-mixed-legacy-001",
            "part-mixed-legacy-002",
            "part-mixed-legacy-003",
        ]

    def test_mixed_never_emits_a_mixed_generation_session(self, fixture_database) -> None:
        database = fixture_database("mixed")

        for header in database.adapter.iter_session_headers():
            assert header.adapter_generation != "mixed"


class TestAdapterRegistry:
    def test_default_registry_builds_the_registered_exporters(self, tmp_path: Path) -> None:
        source = tmp_path / "mixed.db"
        connection = build_fixture_db(source, "mixed")
        connection.close()
        registry = default_adapter_registry()

        with ExitStack() as setup:
            live = setup.enter_context(_open_live(source))
            legacy_exporter = build_live_exporter(live.connection, _description("legacy"), registry)
            current_exporter = build_live_exporter(live.connection, _description("current"), registry)
            mixed_exporter = build_live_exporter(live.connection, _description("mixed"), registry)

        assert isinstance(legacy_exporter, LegacyAdapter)
        assert isinstance(current_exporter, CurrentAdapter)
        assert isinstance(mixed_exporter, MixedAdapter)

    def test_default_registry_builds_the_replay_adapter(self, fixture_database) -> None:
        database = fixture_database("current")

        assert isinstance(
            build_replay_adapter(database.snapshot, default_adapter_registry()), NormalizedSnapshotAdapter
        )

    def test_absent_live_exporter_registration_is_rejected(self, tmp_path: Path) -> None:
        source = tmp_path / "current.db"
        connection = build_fixture_db(source, "current")
        connection.close()
        registry = AdapterRegistry()

        with ExitStack() as setup:
            live = setup.enter_context(_open_live(source))
            with pytest.raises(SourceCompatibilityError) as excinfo:
                build_live_exporter(live.connection, _description("current"), registry)

        assert excinfo.value.code == "adapter_not_registered"
        assert excinfo.value.exit_status == 2

    def test_absent_replay_adapter_registration_is_rejected(self, fixture_database) -> None:
        database = fixture_database("legacy")

        with pytest.raises(SourceCompatibilityError) as excinfo:
            build_replay_adapter(database.snapshot, AdapterRegistry())

        assert excinfo.value.code == "adapter_not_registered"

    def test_registered_factories_satisfy_the_exporter_protocol(self, tmp_path: Path) -> None:
        source = tmp_path / "mixed.db"
        connection = build_fixture_db(source, "mixed")
        connection.close()
        registry = default_adapter_registry()

        with ExitStack() as setup:
            live = setup.enter_context(_open_live(source))
            exporters = [
                build_live_exporter(live.connection, _description(generation), registry)
                for generation in ("legacy", "current", "mixed")
            ]

        assert all(isinstance(exporter, SourceExporter) for exporter in exporters)


def _description(generation: str) -> SchemaDescription:
    return SchemaDescription(generation=generation, catalog_version=SCHEMA_CATALOG_VERSION)
