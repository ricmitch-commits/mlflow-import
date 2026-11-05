"""Small, fixture-qualified JSONL source exporter shared by Claude and Codex."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from mlflow_import.contracts import CanonicalRecordEnvelope
from mlflow_import.source.model import (
    RecordClassificationName,
    SchemaDescription,
    SchemaGeneration,
    SessionHeader,
    SourceRecord,
    TimestampValue,
    normalize_timestamp,
)
from mlflow_import.source.snapshot import ExportStats, RedactionCallback, SnapshotWriter


class JsonlExporter:
    """Read one qualified JSONL session without importing MLflow."""

    file_source = True
    schema_generation: SchemaGeneration = "current"

    def __init__(self, path: Path, harness: str) -> None:
        self.path = path
        self.harness = harness
        self._warnings: list[str] = []

    @property
    def export_warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def bind_connection(self, connection: sqlite3.Connection) -> None:
        del connection

    def inspect(self) -> SchemaDescription:
        return SchemaDescription(
            generation="current",
            catalog_version=f"{self.harness}-jsonl-v1",
            tables={"jsonl": ("path",)},
        )

    def export_to_snapshot(
        self,
        writer: SnapshotWriter,
        redaction: RedactionCallback,
        session_id: str | None = None,
        session_ids: frozenset[str] | None = None,
    ) -> ExportStats:
        del redaction, session_ids
        self._warnings = []
        records = self._read_records()
        native_id = self._session_id_from_records(records)
        if session_id is not None and native_id != session_id:
            return writer.finalize()
        header = self._header(native_id, records)
        writer.write_session(
            header,
            CanonicalRecordEnvelope(decoded={"harness": self.harness}, raw_text=None, raw_columns={}, warnings=()),
        )
        for record in records:
            normalized = replace(record, session_id=native_id)
            order_key = (record.sequence_value if record.sequence_value is not None else 0, str(record.source_id))
            writer.write_record(
                normalized,
                order_key,
                CanonicalRecordEnvelope(
                    decoded=normalized.decoded_value,
                    raw_text=normalized.raw_json,
                    raw_columns=normalized.raw_columns,
                    warnings=normalized.warnings,
                ),
                normalized.raw_columns,
            )
        return writer.finalize()

    def session_id(self) -> str:
        return self._session_id_from_records(self._read_records())

    @staticmethod
    def _session_id_from_records(records: list[SourceRecord]) -> str:
        for record in records:
            value = record.raw_columns.get("session_id") or record.raw_columns.get("sessionId")
            if value:
                return str(value)
            if isinstance(record.decoded_value, dict):
                value = record.decoded_value.get("session_id") or record.decoded_value.get("sessionId")
                if value:
                    return str(value)
        return hashlib.sha256(b"empty-jsonl-session").hexdigest()[:24]

    def _read_records(self) -> list[SourceRecord]:
        records: list[SourceRecord] = []
        with self.path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                text = line.rstrip("\n")
                if not text.strip():
                    continue
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    records.append(
                        SourceRecord(
                            source_table="jsonl",
                            source_type="malformed",
                            source_id=f"line-{line_number}",
                            raw_json=text,
                            warnings=("malformed_json",),
                            sequence_domain="line",
                            sequence_value=line_number,
                            classification="synthetic",
                        )
                    )
                    continue
                if not isinstance(value, dict):
                    value = {"value": value}
                source_id = value.get("uuid") or value.get("id") or f"line-{line_number}"
                record_type = str(value.get("type") or value.get("kind") or "record")
                classification = self._classification(value, record_type)
                timestamp = self._timestamp(value)
                parent = value.get("parentUuid") or value.get("parent_id") or value.get("parentId")
                records.append(
                    SourceRecord(
                        source_table="jsonl",
                        source_type=record_type,
                        source_id=str(source_id),
                        raw_json=text,
                        decoded_value=value,
                        raw_columns={
                            "session_id": value.get("sessionId") or value.get("session_id"),
                            "timestamp": value.get("timestamp") or value.get("created_at"),
                        },
                        sequence_domain="line",
                        sequence_value=line_number,
                        parent_source_id=str(parent) if parent is not None else None,
                        timestamp=timestamp,
                        classification=classification,
                    )
                )
        return records

    def _header(self, session_id: str, records: list[SourceRecord]) -> SessionHeader:
        first = records[0].decoded_value if records and isinstance(records[0].decoded_value, dict) else {}
        directory = first.get("cwd") or first.get("directory") or first.get("project")
        created = next((record.timestamp for record in records if record.timestamp is not None), None)
        return SessionHeader(
            session_id=session_id,
            adapter_generation="current",
            directory=str(directory) if directory is not None else None,
            path=str(self.path),
            title=str(first.get("title")) if first.get("title") is not None else None,
            agent=self.harness,
            model=first.get("model") or first.get("model_name"),
            time_created=created,
            time_updated=records[-1].timestamp if records else created,
            raw_columns={"source_path": str(self.path), "harness": self.harness},
            catalog_version=f"{self.harness}-jsonl-v1",
        )

    @staticmethod
    def _timestamp(value: dict[str, object]) -> TimestampValue | None:
        raw = value.get("timestamp") or value.get("created_at") or value.get("createdAt")
        if isinstance(raw, str):
            try:
                from datetime import datetime

                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return normalize_timestamp(parsed.timestamp() * 1000, "ms")
            except ValueError:
                return None
        if isinstance(raw, (int, float)):
            return normalize_timestamp(raw, "ms")
        return None

    @staticmethod
    def _classification(value: dict[str, object], record_type: str) -> RecordClassificationName:
        role = value.get("role")
        message = value.get("message")
        if isinstance(message, dict):
            role = role or message.get("role")
        if role in {"user", "human"} or record_type in {"user", "human"}:
            return "user"
        if role in {"assistant", "model"} or record_type in {"assistant", "response", "message"}:
            return "assistant"
        if record_type in {"system", "tool", "tool_result", "event"}:
            return "system"
        return "synthetic"
