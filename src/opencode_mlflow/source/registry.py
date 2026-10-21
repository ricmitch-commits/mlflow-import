"""Adapter registry and the mixed-precedence composition exporter.

Ruling 9 places the plan's Task-4 registry surface here: the plan
requires ``AdapterRegistry``, ``build_live_exporter`` and
``build_replay_adapter`` but names no file, and the plan revision removed
them from ``probe.py`` so schema probing never references an exporter
class before it exists. This module imports the concrete adapters, so its
registrations are only constructible after those classes exist.

:class:`MixedAdapter` — the composition exporter for ``mixed``
databases — also lives here: the plan's file list names no ``mixed.py``,
and it owns no SQL of its own (every SELECT stays inside the adapter
owning the matching catalog). For a mixed database it compares only
session IDs between the two session tables: every current session is
exported (the current row wins an overlapping ID, and the suppressed
legacy row is retained as a ``legacy_session_suppressed`` warning on the
authoritative current header), legacy-only IDs are exported by the legacy
adapter, and duplicate IDs are never merged. Sessions stream in globally
normalized creation-time order, interleaving the two generations.
"""

from __future__ import annotations

import heapq
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from operator import itemgetter

from opencode_mlflow.errors import SourceCompatibilityError
from opencode_mlflow.source.catalog import CURRENT_CATALOG_V1, LEGACY_CATALOG_V1
from opencode_mlflow.source.current import CurrentAdapter
from opencode_mlflow.source.legacy import LegacyAdapter
from opencode_mlflow.source.model import (
    SchemaDescription,
    SchemaGeneration,
    SourceAdapter,
)
from opencode_mlflow.source.normalize import describe_classifications, describe_timestamp_units
from opencode_mlflow.source.probe import probe_schema
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter
from opencode_mlflow.source.snapshot import (
    ExportStats,
    RedactionCallback,
    SnapshotWriter,
    SourceExporter,
    SourceSnapshot,
)

LiveExporterFactory = Callable[[sqlite3.Connection], SourceExporter]
"""Builds the live exporter for one schema generation from a connection."""

ReplayAdapterFactory = Callable[[SourceSnapshot], SourceAdapter]
"""Builds the replay adapter over one normalized snapshot."""


class MixedAdapter:
    """``SourceExporter`` composing the current and legacy exporters.

    Current wins overlapping session IDs; legacy-only IDs export through
    the legacy adapter. Each emitted session belongs to exactly one
    generation — ``SessionHeader.adapter_generation`` is always
    ``current`` or ``legacy``, never ``mixed``.
    """

    def __init__(self, current: CurrentAdapter, legacy: LegacyAdapter) -> None:
        if current.catalog.generation != "current" or legacy.catalog.generation != "legacy":
            raise ValueError("MixedAdapter requires a current adapter and a legacy adapter")
        self.schema_generation: SchemaGeneration = "mixed"
        self._current = current
        self._legacy = legacy

    @property
    def current(self) -> CurrentAdapter:
        return self._current

    @property
    def legacy(self) -> LegacyAdapter:
        return self._legacy

    def bind_connection(self, connection: sqlite3.Connection) -> None:
        self._current.bind_connection(connection)
        self._legacy.bind_connection(connection)

    def inspect(self) -> SchemaDescription:
        # Both catalogs are probed through the bound (or constructor)
        # current connection; a mixed database matches both entries. The
        # description carries BOTH catalogs' enrichment — the fixed rank
        # table (identical in both), the union of the timestamp-unit
        # rules (disjoint table names), and both generations'
        # classification rules — so a mixed snapshot exposes the same
        # inspect() surface as a single-generation one.
        description = probe_schema(self._current.connection, [self._legacy.catalog, self._current.catalog])
        return replace(
            description,
            tie_ranks=dict(self._current.catalog.tie_ranks),
            timestamp_units={
                **describe_timestamp_units(self._legacy.catalog),
                **describe_timestamp_units(self._current.catalog),
            },
            classifications=(
                *describe_classifications(self._legacy.catalog),
                *describe_classifications(self._current.catalog),
            ),
        )

    @property
    def export_warnings(self) -> tuple[str, ...]:
        """Safe warning codes collected by the most recent export."""
        return (*self._current.export_warnings, *self._legacy.export_warnings)

    def export_to_snapshot(self, writer: SnapshotWriter, redaction: RedactionCallback) -> ExportStats:
        overlaps = self._current.session_ids() & self._legacy.session_ids()
        self._current.begin_export(overlap_session_ids=frozenset(overlaps))
        self._legacy.begin_export(suppressed_session_ids=frozenset(overlaps))

        merged = heapq.merge(
            ((key, True, row, child_key) for key, row, child_key in self._current.iter_exportable_sessions()),
            ((key, False, row, child_key) for key, row, child_key in self._legacy.iter_exportable_sessions()),
            key=itemgetter(0),
        )
        for _key, is_current, row, child_key in merged:
            if is_current:
                self._current.export_session_records(writer, row, child_key, overlap=str(row["id"]) in overlaps)
            else:
                self._legacy.export_session_records(writer, row, child_key)
        self._current.report_unassociated_records()
        self._legacy.report_unassociated_records()

        stats = writer.finalize()
        if not self.export_warnings:
            return stats
        return replace(stats, warning_count=stats.warning_count + len(self.export_warnings))


class AdapterRegistry:
    """Maps schema generations to live exporter factories and snapshots to the replay adapter."""

    def __init__(self) -> None:
        self._live_exporter_factories: dict[SchemaGeneration, LiveExporterFactory] = {}
        self._replay_adapter_factory: ReplayAdapterFactory | None = None

    def register_live_exporter(self, generation: SchemaGeneration, factory: LiveExporterFactory) -> None:
        """Register the live exporter factory for one schema generation."""
        self._live_exporter_factories[generation] = factory

    def register_replay_adapter(self, factory: ReplayAdapterFactory) -> None:
        """Register the replay adapter factory for normalized snapshots."""
        self._replay_adapter_factory = factory

    def live_exporter_factory(self, generation: SchemaGeneration) -> LiveExporterFactory:
        """Return the registered factory, or fail with a safe compatibility error.

        An absent registration is never worked around with heuristic SQL.
        """
        factory = self._live_exporter_factories.get(generation)
        if factory is None:
            raise SourceCompatibilityError(
                "adapter_not_registered",
                f"no live exporter is registered for schema generation '{generation}'",
            )
        return factory

    def replay_adapter_factory(self) -> ReplayAdapterFactory:
        """Return the registered replay adapter factory."""
        if self._replay_adapter_factory is None:
            raise SourceCompatibilityError(
                "adapter_not_registered",
                "no replay adapter is registered for normalized snapshots",
            )
        return self._replay_adapter_factory


def default_adapter_registry() -> AdapterRegistry:
    """Registry with both concrete exporters, the mixed composer, and the replay adapter.

    The factories are registered only here, after the adapter classes
    exist, so schema probing itself never instantiates an exporter.
    """
    registry = AdapterRegistry()
    registry.register_live_exporter("legacy", lambda connection: LegacyAdapter(connection, LEGACY_CATALOG_V1))
    registry.register_live_exporter("current", lambda connection: CurrentAdapter(connection, CURRENT_CATALOG_V1))
    registry.register_live_exporter(
        "mixed",
        lambda connection: MixedAdapter(
            CurrentAdapter(connection, CURRENT_CATALOG_V1),
            LegacyAdapter(connection, LEGACY_CATALOG_V1),
        ),
    )
    registry.register_replay_adapter(NormalizedSnapshotAdapter)
    return registry


def build_live_exporter(
    connection: sqlite3.Connection,
    description: SchemaDescription,
    registry: AdapterRegistry | None = None,
) -> SourceExporter:
    """Select the live exporter for a probed schema description.

    ``registry`` defaults to the fully registered default registry; an
    explicitly supplied registry with no registration for the probed
    generation fails with a safe internal compatibility error and never
    falls back to heuristic SQL.
    """
    active = registry if registry is not None else default_adapter_registry()
    return active.live_exporter_factory(description.generation)(connection)


def build_replay_adapter(
    snapshot: SourceSnapshot,
    registry: AdapterRegistry | None = None,
) -> SourceAdapter:
    """Select the replay adapter over a normalized snapshot."""
    active = registry if registry is not None else default_adapter_registry()
    return active.replay_adapter_factory()(snapshot)
