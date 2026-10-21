"""Fake ``SourceExporter`` implementations for snapshot tests.

Task 3's snapshot tests inject a fake exporter satisfying the
``SourceExporter`` protocol instead of instantiating the real legacy or
current adapters (those land in Task 4). Two shapes are provided:

- :class:`ScriptedExporter` replays programmable canonical rows without
  touching a live database, for writer/snapshot unit tests.
- :class:`FixtureExporter` reads permitted current-generation columns
  from the live read-only connection the snapshot driver binds to it,
  for replay round-trip and WAL integration tests. Its SELECT lists are
  fixed allowlists that never touch the prohibited secret tables, so
  tests can assert secret-table rows never reach a snapshot.

Both honor the driver's ``bind_connection`` injection seam: the snapshot
machinery opens the single live read-only connection itself and hands it
to the exporter before extraction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from opencode_mlflow.contracts import CanonicalRecordEnvelope
from opencode_mlflow.source.catalog import SCHEMA_CATALOG_VERSION
from opencode_mlflow.source.model import (
    SchemaDescription,
    SchemaGeneration,
    SessionHeader,
    SourceRecord,
    normalize_timestamp,
)
from opencode_mlflow.source.probe import probe_schema
from opencode_mlflow.source.snapshot import ExportStats, SnapshotWriter


def identity_redaction(value: object, protected_paths: Collection[str]) -> object:
    """Redaction callback that preserves every value (verbatim mode stand-in)."""
    return value


def masking_redaction(value: object, protected_paths: Collection[str]) -> object:
    """Deterministic masking callback: non-protected strings become ``[REDACTED]``.

    Protected keys (identity fields) pass through unchanged, mirroring the
    protected-identity contract the real redactor (Task 7) must honor.
    """
    if isinstance(value, Mapping):
        return {
            key: (item if key in protected_paths else masking_redaction(item, protected_paths))
            for key, item in value.items()
        }
    if isinstance(value, str):
        return "[REDACTED]" if value else value
    return value


@dataclass
class ScriptedSession:
    """One programmable ``write_session`` call."""

    header: SessionHeader
    canonical_session_row: CanonicalRecordEnvelope


@dataclass
class ScriptedRecord:
    """One programmable ``write_record`` call."""

    record: SourceRecord
    order_key: tuple
    canonical_record: CanonicalRecordEnvelope
    raw_permitted_fields: Mapping[str, object] = field(default_factory=dict)


class ScriptedExporter:
    """``SourceExporter`` that replays programmable rows without live SQL.

    ``failure`` is raised after all rows are written (so temporary files
    exist when the failure path is exercised). ``attest_completion``
    controls whether ``export_to_snapshot`` returns ``writer.finalize()``
    stats (the honest contract) or ``None`` (a silent partial export).
    """

    def __init__(
        self,
        generation: SchemaGeneration = "current",
        sessions: Sequence[ScriptedSession] = (),
        records: Sequence[ScriptedRecord] = (),
        failure: Exception | None = None,
        attest_completion: bool = True,
    ) -> None:
        self.schema_generation = generation
        self._sessions = list(sessions)
        self._records = list(records)
        self._failure = failure
        self._attest_completion = attest_completion
        self.connection: sqlite3.Connection | None = None
        self.bound: list[sqlite3.Connection] = []

    def bind_connection(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.bound.append(connection)

    def inspect(self) -> SchemaDescription:
        return SchemaDescription(
            generation=self.schema_generation,
            catalog_version=SCHEMA_CATALOG_VERSION,
            tables={},
        )

    def export_to_snapshot(
        self,
        writer: SnapshotWriter,
        redaction: object,  # the writer applies redaction at persistence
    ) -> ExportStats | None:
        for session in self._sessions:
            writer.write_session(session.header, session.canonical_session_row)
        for item in self._records:
            writer.write_record(
                item.record,
                item.order_key,
                item.canonical_record,
                item.raw_permitted_fields,
            )
        if self._failure is not None:
            raise self._failure
        if not self._attest_completion:
            return None
        return writer.finalize()


class FixtureExporter:
    """``SourceExporter`` reading permitted current-schema rows from the bound connection.

    The SELECT column lists below are fixed allowlists mirroring the
    current catalog's permitted columns; the prohibited secret tables are
    never queried for values.
    """

    schema_generation: SchemaGeneration = "current"

    _SESSION_COLUMNS = ("id", "directory", "title", "time_created", "time_updated")
    _MESSAGE_COLUMNS = ("id", "session_id", "type", "seq", "time_created", "data")

    def __init__(self, statements: list[str] | None = None) -> None:
        self._statements = statements
        self.connection: sqlite3.Connection | None = None

    def bind_connection(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        if self._statements is not None:
            connection.set_trace_callback(self._statements.append)

    def inspect(self) -> SchemaDescription:
        assert self.connection is not None, "the snapshot driver binds the live connection first"
        return probe_schema(self.connection)

    def export_to_snapshot(
        self,
        writer: SnapshotWriter,
        redaction: object,  # the writer applies redaction at persistence
    ) -> ExportStats:
        assert self.connection is not None, "the snapshot driver binds the live connection first"
        connection = self.connection
        session_select = ", ".join(self._SESSION_COLUMNS)
        for row in connection.execute(f"SELECT {session_select} FROM session_v2 ORDER BY time_created, id"):
            values = dict(zip(self._SESSION_COLUMNS, row, strict=True))
            header = SessionHeader(
                session_id=str(values["id"]),
                adapter_generation="current",
                directory=values["directory"],
                title=values["title"],
                time_created=normalize_timestamp(values["time_created"], "ms"),
                time_updated=normalize_timestamp(values["time_updated"], "ms"),
            )
            envelope = CanonicalRecordEnvelope(
                decoded={"directory": values["directory"], "title": values["title"]},
                raw_text=None,
                raw_columns=values,
                warnings=(),
            )
            writer.write_session(header, envelope)

        message_select = ", ".join(self._MESSAGE_COLUMNS)
        for row in connection.execute(f"SELECT {message_select} FROM session_message ORDER BY session_id, seq"):
            values = dict(zip(self._MESSAGE_COLUMNS, row, strict=True))
            raw_json, extra_warnings, retained_data = self._decode_data(values["data"])
            record = SourceRecord.from_row(
                source_table="session_message",
                source_type=str(values["type"]),
                source_id=values["id"],
                raw_json=raw_json,
                type=values["type"],
                seq=values["seq"],
            )
            record = replace(
                record,
                session_id=str(values["session_id"]),
                sequence_domain="session_message.seq",
                sequence_value=values["seq"],
                timestamp=normalize_timestamp(values["time_created"], "ms"),
                classification=self._classify(str(values["type"])),
            )
            envelope = CanonicalRecordEnvelope(
                decoded=record.decoded_value,
                raw_text=record.raw_json,
                raw_columns={"type": values["type"], "seq": values["seq"]},
                warnings=tuple(record.warnings) + tuple(extra_warnings),
            )
            raw_fields: dict[str, Any] = {
                "type": values["type"],
                "seq": values["seq"],
                "time_created": values["time_created"],
            }
            if retained_data is not None:
                raw_fields["data"] = retained_data
            order_key = (
                values["seq"] if values["seq"] is not None else -1,
                values["id"] if values["id"] is not None else "",
            )
            writer.write_record(record, order_key, envelope, raw_fields)
        return writer.finalize()

    @staticmethod
    def _decode_data(data: object) -> tuple[str | None, list[str], object | None]:
        """Decode a `data` column value, retaining undecodable bytes.

        Returns ``(raw_json, extra_warnings, retained_value)``: TEXT passes
        through, a BLOB is decoded as UTF-8 when possible, and an
        invalid-UTF-8 BLOB yields no raw JSON, a ``malformed_source_value``
        warning, and the original bytes for permitted-field retention.
        """
        if data is None:
            return None, [], None
        if isinstance(data, bytes):
            try:
                return data.decode("utf-8"), [], None
            except UnicodeDecodeError:
                return None, ["malformed_source_value"], data
        if isinstance(data, str):
            return data, [], None
        return None, ["malformed_source_value"], data

    @staticmethod
    def _classify(message_type: str) -> str | None:
        if message_type in ("user", "assistant", "system"):
            return message_type
        return None
