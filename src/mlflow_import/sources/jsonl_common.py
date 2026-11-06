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
from mlflow_import.source.ordering import session_sort_key
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
        del redaction
        self._warnings = []
        records_by_session: dict[str, list[SourceRecord]] = {}
        for record in self._read_records():
            native_id = self._record_session_id(record)
            records_by_session.setdefault(native_id, []).append(record)
        selected_ids = set(records_by_session)
        if session_ids is not None:
            selected_ids &= set(session_ids)
        elif session_id is not None:
            selected_ids &= self._session_closure(session_id, records_by_session)
        sessions = [(self._header(native_id, records), records) for native_id, records in records_by_session.items()]
        for header, records in sorted(sessions, key=lambda item: session_sort_key(item[0])):
            native_id = header.session_id
            if native_id not in selected_ids:
                continue
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
        for index, source_path in enumerate(self._source_paths()):
            source_records = self._read_path(source_path)
            owner = next(
                (self._record_session_id(record) for record in source_records if self._has_session_id(record)),
                self._empty_session_id(source_path, index),
            )
            records.extend(
                replace(record, session_id=owner) if not self._has_session_id(record) else record
                for record in source_records
            )
        return records

    def _read_path(self, source_path: Path) -> list[SourceRecord]:
        records: list[SourceRecord] = []
        with source_path.open(encoding="utf-8") as source:
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
                source_table = "jsonl_message" if classification in {"user", "assistant"} else "jsonl"
                timestamp = self._timestamp(value)
                parent = value.get("parentUuid") or value.get("parent_id") or value.get("parentId")
                records.append(
                    SourceRecord(
                        source_table=source_table,
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

    def _source_paths(self) -> tuple[Path, ...]:
        if self.path.is_dir():
            return tuple(sorted(self.path.rglob("*.jsonl")))
        return (self.path,)

    def _empty_session_id(self, source_path: Path, index: int) -> str:
        if self.path.is_dir():
            locator = str(source_path.relative_to(self.path))
        else:
            locator = source_path.name
        digest = hashlib.sha256(locator.encode("utf-8")).hexdigest()[:24]
        return f"empty-jsonl-session-{digest or index}"

    @staticmethod
    def _record_session_id(record: SourceRecord) -> str:
        value = record.session_id or record.raw_columns.get("session_id") or record.raw_columns.get("sessionId")
        if value is None and isinstance(record.decoded_value, dict):
            value = record.decoded_value.get("session_id") or record.decoded_value.get("sessionId")
        return str(value) if value is not None else ""

    @classmethod
    def _has_session_id(cls, record: SourceRecord) -> bool:
        return bool(cls._record_session_id(record))

    def _session_closure(self, session_id: str, records_by_session: dict[str, list[SourceRecord]]) -> set[str]:
        parents = {
            native_id: {
                parent
                for record in records
                for parent in (self._record_parent_session_id(record),)
                if parent is not None
            }
            for native_id, records in records_by_session.items()
        }
        children: dict[str, set[str]] = {}
        for child, parent_ids in parents.items():
            for parent in parent_ids:
                children.setdefault(parent, set()).add(child)
        known = set(records_by_session)
        seeds = {session_id} & known
        included = set(seeds)
        pending = list(seeds)
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for ancestor in parents.get(current, ()):
                if ancestor in known and ancestor not in visited:
                    included.add(ancestor)
                    pending.append(ancestor)

        pending = list(seeds)
        visited = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for descendant in children.get(current, ()):
                if descendant in known and descendant not in visited:
                    included.add(descendant)
                    pending.append(descendant)
        return included

    def _header(self, session_id: str, records: list[SourceRecord]) -> SessionHeader:
        first = records[0].decoded_value if records and isinstance(records[0].decoded_value, dict) else {}
        directory = first.get("cwd") or first.get("directory") or first.get("project")
        created = next((record.timestamp for record in records if record.timestamp is not None), None)
        return SessionHeader(
            session_id=session_id,
            adapter_generation="current",
            parent_id=self._parent_session_id(records),
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
    def _parent_session_id(records: list[SourceRecord]) -> str | None:
        for record in records:
            parent = JsonlExporter._record_parent_session_id(record)
            if parent is not None:
                return parent
        return None

    @staticmethod
    def _record_parent_session_id(record: SourceRecord) -> str | None:
        if not isinstance(record.decoded_value, dict):
            return None
        payload = record.decoded_value.get("payload")
        for value in (record.decoded_value, payload):
            if isinstance(value, dict):
                parent = value.get("parentSessionId") or value.get("parent_session_id")
                if parent is not None:
                    return str(parent)
        return None

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
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(part, dict) and part.get("type") == "tool_result" for part in content
            ):
                return "system"
        if role in {"user", "human"} or record_type in {"user", "human"}:
            return "user"
        if role in {"assistant", "model"} or record_type in {"assistant", "response", "message"}:
            return "assistant"
        if record_type in {"system", "tool", "tool_result", "event"}:
            return "system"
        return "synthetic"
