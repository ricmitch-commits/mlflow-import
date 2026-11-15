"""Schema-aware streaming file adapters and disk-backed replay.

Only header records are inspected during inventory. Bodies are read for selected
sessions and their required relatives, twice to establish a stable view. No
transcript collection is retained in Python. Original occurrences and effective
entities have separate identities and byte locators in the common snapshot.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
from collections.abc import Collection, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast

from mlflow_import.contracts import CanonicalRecordEnvelope
from mlflow_import.errors import SnapshotError, SourceCompatibilityError
from mlflow_import.retry import check_budget
from mlflow_import.source.file_headers import read_headers
from mlflow_import.source.model import (
    RecordClassificationName,
    SchemaDescription,
    SchemaGeneration,
    SessionHeader,
    SourceRecord,
    TimestampValue,
    iso_fractional_ns,
    owning_source_root,
    source_namespace,
)
from mlflow_import.source.replay import _restore_source_record
from mlflow_import.source.selection import SelectionFilters, matches_directory, time_bounds
from mlflow_import.source.snapshot import (
    ExportStats,
    RedactionCallback,
    SnapshotWriter,
    SourceSnapshot,
    loads_snapshot_json,
)
from mlflow_import.source.spool import read_payload
from mlflow_import.sources.schemas import FORMATS

# Mandatory schema exclusions, not optional text redaction. Token usage is data.
_EXCLUDED = frozenset(
    {
        "apikey",
        "accesstoken",
        "refreshtoken",
        "authtoken",
        "authorization",
        "credentials",
        "oauth",
        "account",
        "accounts",
        "sessionshare",
        "share",
        "shareurl",
    }
)
_EXCLUDED_TYPES = frozenset({"credentials", "account", "oauth", "auth", "session_share"})
MAX_RECORD_BYTES = 32 * 1024**2
_CLAUDE_METADATA = frozenset(
    {
        "file-history-snapshot",
        "queue-operation",
        "summary",
        "progress",
        "system",
        "mode",
        "permission-mode",
        "atis-latch",
        "worktree-state",
        "tools-used",
        "last-prompt",
        "cost-state",
        "fork-context-ref",
    }
)


def permitted(value: Any) -> Any:
    """Preserve unknown data while removing known mixed-record secret fields."""
    if isinstance(value, dict):
        return {
            key: permitted(item)
            for key, item in value.items()
            if str(key).replace("_", "").replace("-", "").lower() not in _EXCLUDED
        }
    if isinstance(value, list):
        return [permitted(item) for item in value]
    return value


def open_text(path: Path) -> TextIO:
    if path.is_symlink():
        raise SnapshotError("source_unavailable", "source symlinks are not supported")
    handle = (
        gzip.open(path, "rt", encoding="utf-8", newline="")
        if path.name.endswith(".gz")
        else path.open(encoding="utf-8", newline="")
    )
    from mlflow_import.resources import account_reader

    return cast(TextIO, account_reader(handle, path))


def lines(path: Path, digest: Any = None) -> Iterator[tuple[int, str, dict[str, Any]]]:
    """Bound a physical record, including decompressed input, without truncation."""
    with open_text(path) as handle:
        number = 0
        while text := handle.readline(MAX_RECORD_BYTES + 1):
            check_budget()
            number += 1
            encoded = text.encode("utf-8")
            if len(encoded) > MAX_RECORD_BYTES:
                raise SourceCompatibilityError(
                    "source_record_too_large", "a source record exceeds the supported envelope"
                )
            if digest is not None:
                digest.update(encoded)
            if not text.strip():
                continue
            try:
                value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            except ValueError:
                # A malformed mixed record cannot be safely separated from secrets.
                raise SourceCompatibilityError(
                    "malformed_source_record", "a file record cannot be safely decoded"
                ) from None
            if not isinstance(value, dict):
                raise SourceCompatibilityError("unsupported_schema", "file records must be JSON objects")
            yield number, text.rstrip("\r\n"), value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("ambiguous file field")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError("non-JSON file constant")


def _claude_prelude(
    headers: Iterator[dict[str, Any]], first: dict[str, Any], *, need_directory: bool
) -> dict[str, Any]:
    """Merge qualified metadata prefixes and selection-bearing envelopes.

    A session-bearing mode is not a complete header. Only allowlisted fields
    are decoded by the lexer. With a directory filter, continue through known
    conversation envelopes until cwd is established (or EOF leaves it unknown).
    No transcript values are decoded while finding those metadata facts.
    Fork context references are markers, never permission to open a locator.
    """
    merged: dict[str, Any] = {}
    value: dict[str, Any] | None = first
    while value is not None:
        for key, item in value.items():
            if key != "type" and "sessionId" not in merged and "sessionId" not in value:
                continue  # Identity-less prefixes cannot establish session facts.
            if key == "sessionId" and key in merged and merged[key] != item:
                raise SourceCompatibilityError("source_identity_changed", "Claude prelude identity is contradictory")
            # Retain earliest timestamp/identity; later cwd fills mode-only headers.
            if key not in merged or merged[key] is None or key == "type":
                merged[key] = item
        if value.get("type") not in _CLAUDE_METADATA and (
            not need_directory
            or merged.get("cwd")
            or merged.get("directory")
            or value.get("type") not in {"user", "assistant"}
        ):
            break
        value = next(headers, None)
    return merged


def schema_header(path: Path, harness: str, *, need_directory: bool = True) -> dict[str, Any] | None:
    """Read a qualified metadata prelude without decoding transcript values."""
    try:
        with open_text(path) as handle:
            headers = read_headers(handle, MAX_RECORD_BYTES)
            value = next(headers, None)
            if harness == "claude" and value is not None:
                value = _claude_prelude(headers, value, need_directory=need_directory)
                if "sessionId" not in value and value.get("type") in _CLAUDE_METADATA | {"user", "assistant"}:
                    from uuid import UUID

                    native = path.name.removesuffix(".gz").removesuffix(".jsonl")
                    try:
                        UUID(native)
                    except ValueError:
                        return None
                    value = {**value, "sessionId": native, "identity_origin": "filename"}
    except ValueError:
        raise SourceCompatibilityError("unsupported_schema", "a source header cannot be safely decoded") from None
    if value is None:
        return None
    if value.get("type") in _EXCLUDED_TYPES:
        return None
    if harness == "pi":
        if value.get("type") != "session":
            return None
        if value.get("version") != 3 or not isinstance(value.get("id"), str) or not value["id"]:
            raise SourceCompatibilityError("unsupported_schema", "only Pi version 3 entry trees are qualified")
        return {**value, "session_id": _native_id(value["id"])}
    if harness == "gemini":
        if (
            "sessionId" not in value
            or not value.get("projectHash")
            or "messages" in value
            or value.get("type") not in {None, "metadata"}
        ):
            if "sessionId" in value:
                return {**value, "session_id": _native_id(value["sessionId"]), "unsupported_header": True}
            return None
        _native_id(value["projectHash"])
        return {**value, "session_id": _native_id(value["sessionId"])}
    if harness == "codex":
        if value.get("type") == "session_meta" and isinstance(value.get("payload"), dict):
            payload = value["payload"]
            if "id" in payload:
                return {
                    **payload,
                    "session_id": _native_id(payload["id"]),
                    "timestamp": payload.get("timestamp", value.get("timestamp")),
                }
            raise SourceCompatibilityError("invalid_source_identity", "a Codex session metadata header has no identity")
        # Legacy fixture/export variant: explicit session identity on each record.
        if value.get("session_id") and value.get("type") == "response_item":
            _native_id(value["session_id"])
            return value
        return None
    if harness == "claude" and "sessionId" in value and value.get("type") in _CLAUDE_METADATA | {"user", "assistant"}:
        return {**value, "session_id": _native_id(value["sessionId"])}
    if harness == "claude" and "sessionId" in value:
        # Preserve identity/selection facts so a selected unknown schema fails;
        # unrelated sessions can still be excluded without decoding their body.
        return {**value, "session_id": _native_id(value["sessionId"]), "unsupported_header": True}
    return None


def _native_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise SourceCompatibilityError(
            "invalid_source_identity", "a source header has no valid native session identity"
        )
    return value


def _header_digest(header: dict[str, Any], directory: str | None) -> str:
    """Content-free binding of allowlisted facts and registry-derived directory."""
    encoded = json.dumps([header, directory], sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def detect(path: Path, harness: str) -> bool:
    def supported(item: Path) -> bool:
        header = schema_header(item, harness, need_directory=False)
        return header is not None and not header.get("unsupported_header", False)

    try:
        if path.is_file():
            return supported(path)
        if path.is_dir():
            return any(supported(item) for item in candidate_paths(path, harness))
    except (OSError, UnicodeError, SourceCompatibilityError):
        return False
    return False


def source_paths(root: Path, _depth: int = 0) -> Iterator[Path]:
    if _depth > 32:
        raise SourceCompatibilityError(
            "source_path_too_deep", "the source directory tree exceeds the supported envelope"
        )
    if root.is_symlink():
        return
    if root.is_file():
        yield root
    elif root.is_dir():
        from mlflow_import.resources import current_storage, ordered_directory
        from mlflow_import.storage import DiskList

        children: list[Path] = DiskList() if current_storage() else []
        for name, is_dir, is_file in ordered_directory(root):
            if is_file and name.endswith((".jsonl", ".jsonl.gz")):
                yield root / name
            elif is_dir and name != "tool-results":
                children.append(root / name)
        for child in children:
            yield from source_paths(child, _depth + 1)


def candidate_paths(root: Path, harness: str) -> Iterator[Path]:
    """Restrict installed-store discovery to transcript-owned subtrees."""
    from mlflow_import.sources import get_adapter

    defaults = get_adapter(harness).default_locations()
    is_default = any(root.absolute() == item.expanduser().absolute() for item in defaults)
    if (
        root.is_dir()
        and harness == "codex"
        and (is_default or (root / "sessions").is_dir() or (root / "archived_sessions").is_dir())
    ):
        for name in ("sessions", "archived_sessions"):
            yield from source_paths(root / name)
    elif root.is_dir() and harness == "gemini" and is_default:
        import os

        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    yield from source_paths(Path(entry.path) / "chats")
    elif root.is_dir() and harness == "gemini" and (root / "chats").is_dir():
        yield from source_paths(root / "chats")
    else:
        yield from source_paths(root)


def session_base(path: Path) -> str:
    return path.name.removesuffix(".gz").removesuffix(".jsonl")


def timestamp(raw: Any) -> TimestampValue | None:
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return None
            delta = parsed.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
            tail = iso_fractional_ns(raw)
            return TimestampValue(
                ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000 + tail, "ns"
            )
        except ValueError:
            return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return TimestampValue(raw, "ms")
    return None


def identity_stat(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open_text(path) as handle:
        while chunk := handle.read(1024 * 1024):
            check_budget()
            digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


class JsonlExporter:
    """Four schema rules using one occurrence/entity working store."""

    file_source = True
    schema_generation: SchemaGeneration = "current"

    def __init__(self, path: Path, harness: str) -> None:
        self.path, self.harness = Path(os.path.abspath(path.expanduser())), harness
        self.filters = SelectionFilters()
        self.export_warnings: tuple[str, ...] = ()

    def bind_connection(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def inspect(self) -> SchemaDescription:
        return SchemaDescription(
            generation="current",
            catalog_version=FORMATS[self.harness].identifier,
            tables={"jsonl": ("header", "occurrence", "entity")},
        )

    def _inventory(self) -> None:
        db = self.connection
        db.executescript("""
            CREATE TABLE file_candidates(id TEXT,path TEXT PRIMARY KEY,created INTEGER,parent TEXT,
                directory_match INTEGER,header TEXT,selected INTEGER DEFAULT 0,
                matched INTEGER DEFAULT 0,frozen INTEGER DEFAULT 0);
            CREATE INDEX file_parent ON file_candidates(parent);
            CREATE INDEX file_native ON file_candidates(id);
            CREATE TABLE file_sides(owner TEXT,path TEXT PRIMARY KEY);
            CREATE TABLE usage_observations(session_id TEXT,scope TEXT,value TEXT,PRIMARY KEY(session_id,scope));
            CREATE TABLE usage_turns(session_id TEXT,scope TEXT,count INTEGER,PRIMARY KEY(session_id,scope));
            CREATE TABLE model_observations(session_id TEXT,scope TEXT,value TEXT,source_id TEXT,
                PRIMARY KEY(session_id,scope));
        """)
        inventory_root = self.path
        self._explicit_session_id: str | None = None
        if self.path.is_file():
            header = schema_header(self.path, self.harness, need_directory=self.filters.directory is not None)
            if header is None:
                raise SourceCompatibilityError("unsupported_schema", "the selected file schema is not supported")
            self._explicit_session_id = header["session_id"]
            if (self.harness == "claude" and "subagents" in self.path.parts) or header.get("kind") == "subagent":
                raise SourceCompatibilityError(
                    "unsupported_source", "select a primary session or owning source root for subagent input"
                )
            inventory_root = owning_source_root(self.path, self.harness, known_only=True) or self.path
        primary_files = 0
        for path in candidate_paths(inventory_root, self.harness):
            if self.harness == "claude" and "subagents" in path.parts:
                # Canonical Claude sidecar layout: <session>/subagents/agent-*.jsonl.
                db.execute("INSERT INTO file_sides VALUES(?,?)", (path.parent.parent.name, str(path)))
                continue
            primary_files += 1
            value = schema_header(path, self.harness, need_directory=self.filters.directory is not None)
            if value is None:
                if path == self.path and self.path.is_file():
                    raise SourceCompatibilityError("unsupported_schema", "the selected file schema is not supported")
                self.export_warnings = ("unassociated_source_file",)
                continue
            native = str(value["session_id"])
            created = timestamp(value.get("startTime", value.get("timestamp")))
            parent = value.get("parentSessionId", value.get("parent_session_id", value.get("forked_from_id")))
            if self.harness == "pi" and value.get("parentSession"):
                # Never open arbitrary parent locators. Resolve against inventoried files below.
                locator = Path(value["parentSession"])
                parent = "@path:" + (os.path.abspath(locator) if locator.is_absolute() else str(locator))
            directory = self._directory(value, path)
            # Persist only technical validation facts. Header values for excluded
            # sessions never enter a spool; selected headers are reread below and
            # pass through write_session's pre-persistence redaction boundary.
            header = {
                "digest": _header_digest(value, directory),
                "project_identity_digest": _header_digest({"projectHash": value.get("projectHash")}, None),
                "kind_digest": _header_digest({"kind": value.get("kind", "main")}, None),
                "subagent": value.get("kind") == "subagent",
            }
            try:
                db.execute(
                    "INSERT INTO file_candidates(id,path,created,parent,directory_match,header) VALUES(?,?,?,?,?,?)",
                    (
                        native,
                        str(path),
                        created.nanoseconds if created else None,
                        parent,
                        int(
                            self.filters.directory is None or matches_directory(Path(directory), self.filters.directory)
                        )
                        if directory is not None
                        else None,
                        self.writer._payload(header, header=True),
                    ),
                )
            except sqlite3.IntegrityError:
                raise SourceCompatibilityError(
                    "ambiguous_source", f"{self.harness}: multiple authoritative files claim the same session"
                ) from None
        if primary_files and not db.execute("SELECT 1 FROM file_candidates LIMIT 1").fetchone():
            raise SourceCompatibilityError(
                "unsupported_schema", f"{self.harness}: source root contains no qualified session headers"
            )
        if db.execute("""SELECT 1 FROM file_candidates c WHERE c.parent LIKE '@path:%'
            AND NOT EXISTS (SELECT 1 FROM file_candidates p WHERE '@path:' || p.path=c.parent) LIMIT 1""").fetchone():
            self._warn("unassociated_parent_session")
        db.execute("""UPDATE file_candidates SET parent=(SELECT p.id FROM file_candidates p
            WHERE '@path:' || p.path=file_candidates.parent) WHERE parent LIKE '@path:%'""")
        if self._explicit_session_id is not None:
            row = db.execute("SELECT id FROM file_candidates WHERE path=?", (str(self.path),)).fetchone()
            if row is None or row[0] != self._explicit_session_id:
                raise SnapshotError("source_changed", "the explicit file identity changed during discovery")
        if self.harness == "gemini":
            for path_text, header_text in db.execute("SELECT path,header FROM file_candidates"):
                if not self._header_payload(header_text)["subagent"]:
                    continue
                owner = Path(path_text).parent.name
                if db.execute("SELECT 1 FROM file_candidates WHERE id=?", (owner,)).fetchone():
                    db.execute("INSERT INTO file_sides VALUES(?,?)", (owner, path_text))
                else:
                    self.export_warnings = (*self.export_warnings, "unassociated_subagent")
                db.execute("DELETE FROM file_candidates WHERE path=?", (path_text,))
        # An explicit main file must still capture its associated subagent set.
        if self.harness == "claude" and self.path.is_file():
            owner = db.execute("SELECT id FROM file_candidates").fetchone()
            for side in source_paths(self._claude_side_root(self.path) / "subagents"):
                db.execute("INSERT OR IGNORE INTO file_sides VALUES(?,?)", (owner[0], str(side)))

    def _select(self, session_id: str | None, session_ids: Collection[str] | None) -> None:
        db, filters = self.connection, self.filters
        db.execute("INSERT OR IGNORE INTO source_identity_headers SELECT id FROM file_candidates")
        lower, upper = time_bounds(filters)
        for native, path, created, directory_match in db.execute(
            "SELECT id,path,created,directory_match FROM file_candidates"
        ):
            if self._explicit_session_id is not None and native != self._explicit_session_id:
                continue
            if session_ids is not None and native not in session_ids:
                continue
            if session_id is not None and native != session_id:
                continue
            if (
                created is None
                and (lower is not None or upper is not None)
                and "missing_session_creation_time" not in self.export_warnings
            ):
                self.export_warnings = (*self.export_warnings, "missing_session_creation_time")
            if lower is not None and (created is None or created < lower):
                continue
            if upper is not None and (created is None or created > upper):
                continue
            # Missing/non-authoritative directory does not justify pruning.
            if filters.directory and directory_match == 0:
                continue
            db.execute("UPDATE file_candidates SET selected=1 WHERE path=?", (path,))

    def _directory(self, header: dict[str, Any], path: Path) -> str | None:
        directory = header.get("cwd") or header.get("directory")
        if directory is not None and not isinstance(directory, str):
            raise SourceCompatibilityError("unsupported_schema", f"{self.harness}: invalid header directory")
        if self.harness != "gemini" or directory:
            return directory
        from mlflow_import.sources.gemini_registry import project_directory

        return project_directory(path, header.get("projectHash"))

    def _qualify_selected(self) -> None:
        """Reject ambiguity only after safe direct filters / relative closure.

        A provably excluded representation cannot override a unique eligible
        one. Excluded-only collisions remain inventoried for relative closure;
        if required later, all their claims are selected and rejected here.
        """
        db = self.connection
        if db.execute(
            "SELECT 1 FROM file_candidates WHERE selected=1 GROUP BY id HAVING count(*)>1 LIMIT 1"
        ).fetchone():
            raise SourceCompatibilityError(
                "ambiguous_source",
                f"{self.harness}: multiple eligible files claim one native session; "
                "creation timestamps and filenames do not establish authority; narrow the source or filters",
            )
        db.execute(
            "DELETE FROM file_candidates WHERE selected=0 AND id IN (SELECT id FROM file_candidates WHERE selected=1)"
        )

    def _expand(self) -> None:
        self.connection.execute("""WITH RECURSIVE ancestors(id) AS (
            SELECT id FROM file_candidates WHERE matched=1 UNION
            SELECT c.parent FROM file_candidates c JOIN ancestors a ON c.id=a.id WHERE c.parent IS NOT NULL),
            descendants(id) AS (SELECT id FROM file_candidates WHERE matched=1 UNION
            SELECT c.id FROM file_candidates c JOIN descendants d ON c.parent=d.id)
            UPDATE file_candidates SET selected=1
            WHERE id IN (SELECT id FROM ancestors UNION SELECT id FROM descendants)""")

    def export_to_snapshot(
        self,
        writer: SnapshotWriter,
        redaction: RedactionCallback,
        session_id: str | None = None,
        session_ids: Collection[str] | None = None,
    ) -> ExportStats:
        del redaction
        self.writer = writer
        self._inventory()
        self._select(session_id, session_ids)
        writer.defer_finalization = True
        self._freeze_candidates()
        writer.flush()
        if self.filters.model is not None or self.filters.min_user_turns is not None:
            from mlflow_import.source.planning import content_matches

            provisional = SourceSnapshot(
                fingerprint="working-source-index",
                snapshot_path=Path(self.connection.execute("PRAGMA database_list").fetchone()[2]),
                schema_description=self.inspect(),
                connection=self.connection,
                stats=ExportStats(0, 0, 0, False, "current"),
                selection_evidence_enabled=True,
            )
            try:
                for native in content_matches(provisional, self.filters):
                    self.connection.execute("UPDATE file_candidates SET matched=1 WHERE id=?", (native,))
            finally:
                self.turn_index = provisional.turn_index
        else:
            self.connection.execute("UPDATE file_candidates SET matched=selected")
        self._expand()
        self._freeze_candidates()
        writer.defer_finalization = False
        return writer.finalize()

    def _freeze_candidates(self) -> None:
        self._qualify_selected()
        writer = self.writer
        for native, path_text, created, parent in self.connection.execute(
            "SELECT id,path,created,parent FROM file_candidates"
            " WHERE selected=1 AND frozen=0 ORDER BY created IS NULL,created,id"
        ):
            path = Path(path_text)
            value = self._candidate_header(native, path)
            directory = self._directory(value, path)
            extra: dict[str, Any] = {
                key: permitted(value[key])
                for key in (
                    "title",
                    "summary",
                    "directories",
                    "projectHash",
                    "version",
                    "cli_version",
                    "model",
                    "git",
                    "kind",
                    "lastUpdated",
                    "identity_origin",
                    "client_request_id",
                    "user",
                    "user_id",
                    "username",
                    "mlflow.trace.user",
                    "mlflow.user",
                )
                if key in value
            }
            header = SessionHeader(
                session_id=native,
                adapter_generation="current",
                parent_id=parent,
                directory=directory,
                path=None,
                title=extra.get("title", extra.get("summary") if self.harness == "gemini" else None),
                project_id=extra.get("projectHash"),
                metadata={"session_format_version": extra.get("version")}
                if self.harness == "pi"
                else {"directories": extra["directories"]}
                if "directories" in extra
                else None,
                agent=self.harness,
                model=extra.get("model"),
                version=(
                    str(extra.get("cli_version", extra.get("version")))
                    if self.harness != "pi" and ("cli_version" in extra or "version" in extra)
                    else None
                ),
                time_created=TimestampValue(created, "ns") if created is not None else None,
                source_fingerprint=source_namespace(self.path, self.harness),
                raw_columns={"harness": self.harness, **extra},
                catalog_version=self.inspect().catalog_version,
            )
            writer.write_session(
                header, CanonicalRecordEnvelope(decoded=extra, raw_text=None, raw_columns=extra, warnings=())
            )
            associated = self._associated(native, path)
            for side, stream in associated:
                if side.suffix == ".txt":
                    self._freeze_tool_result(native, side)
                else:
                    self._freeze_path(native, side, stream)
            if associated != self._associated(native, path):
                raise SnapshotError("source_changed", "the selected session sidecar inventory changed")
            self.connection.execute("UPDATE file_candidates SET frozen=1 WHERE id=?", (native,))

    def _associated(self, native: str, path: Path) -> list[tuple[Path, str]]:
        # Sidecar metadata only; bodies are streamed individually.
        from mlflow_import.storage import DiskList

        result = DiskList([(path, "")])
        if self.harness == "claude":
            side_root = self._claude_side_root(path)
            result.extend((item, "agent:" + item.name + ":") for item in source_paths(side_root / "subagents"))
            tool_root = side_root / "tool-results"
            if tool_root.is_dir() and not tool_root.is_symlink():
                from mlflow_import.resources import ordered_directory

                result.extend(
                    (tool_root / name, "tool:")
                    for name, _, is_file in ordered_directory(tool_root)
                    if is_file and name.endswith(".txt")
                )
        if self.harness == "gemini":
            if Path(native).name != native or native in {".", ".."}:
                raise SourceCompatibilityError("unsupported_schema", "the session has an unsafe sidecar locator")
            result.extend((item, "agent:" + item.name + ":") for item in source_paths(path.parent / native))
        return result

    @staticmethod
    def _claude_side_root(path: Path) -> Path:
        root = path.parent / session_base(path)
        if root.is_symlink():
            raise SourceCompatibilityError(
                "unsupported_source", "Claude sidecar session directories cannot be symlinks"
            )
        return root

    def _header_payload(self, locator: str) -> dict[str, Any]:
        path = Path(self.connection.execute("PRAGMA database_list").fetchone()[2]).with_suffix(".payloads")
        return cast(dict[str, Any], loads_snapshot_json(read_payload(path, locator)))

    def _freeze_tool_result(self, native: str, path: Path) -> None:
        before = identity_stat(path)
        with open_text(path) as handle:
            content = handle.read(MAX_RECORD_BYTES + 1)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if len(content.encode("utf-8")) > MAX_RECORD_BYTES:
            raise SourceCompatibilityError(
                "source_record_too_large", "a tool-result sidecar exceeds the supported envelope"
            )
        occurrence = self._write(
            SourceRecord(
                source_table="jsonl",
                source_type="tool_result_sidecar",
                source_id="tool-result:" + path.stem,
                session_id=native,
                call_id=path.stem,
                decoded_value={"type": "tool_result", "tool_use_id": path.stem, "content": content},
                classification="system",
                occurrence_id="tool-result:" + path.name,
            )
        )
        anchor = self._call_anchor(native, path.stem)
        self.connection.execute(
            "UPDATE snapshot_records SET anchor_ordinal=?,active=? WHERE rowid=?",
            (anchor, int(anchor is not None), occurrence),
        )
        if anchor is None:
            self._warn("unattributed_tool_result")
        verified_digest = digest_file(path)
        if before != identity_stat(path) or digest != verified_digest:
            raise SnapshotError("source_changed", "a selected tool-result sidecar changed")

    def _candidate_header(self, native: str, path: Path, *, stream: str = "") -> dict[str, Any]:
        """Revalidate source facts without retaining unredacted inventory values."""
        observed_header = schema_header(path, self.harness, need_directory=self.filters.directory is not None)
        if observed_header is None:
            raise SourceCompatibilityError(
                "unsupported_schema", f"{self.harness}: a selected file has no supported session header"
            )
        if observed_header.get("unsupported_header"):
            raise SourceCompatibilityError("unsupported_schema", f"{self.harness}: selected header variant unsupported")
        if self.harness == "claude" and stream and observed_header["session_id"] != native:
            raise SourceCompatibilityError("source_identity_changed", "Claude sidecar identity contradicts its owner")
        if not stream:
            stored_header = self.connection.execute(
                "SELECT header FROM file_candidates WHERE id=? AND selected=1", (native,)
            ).fetchone()
            directory = self._directory(observed_header, path)
            if (
                observed_header["session_id"] != native
                or _header_digest(observed_header, directory) != self._header_payload(stored_header[0])["digest"]
            ):
                raise SnapshotError("source_changed", "a selected identity or selection header changed")
        return observed_header

    def _freeze_path(self, native: str, path: Path, stream: str) -> None:
        before = identity_stat(path)
        copied_digest = hashlib.sha256()
        observed_header = self._candidate_header(native, path, stream=stream)
        head: str | None = None
        for number, raw, original in lines(path, copied_digest):
            if original.get("type") in _EXCLUDED_TYPES:
                continue
            value = permitted(original)
            if self.harness == "codex":
                payload = value.get("payload")
                if isinstance(payload, dict) and payload.get("type") in {"thread_rolled_back", "rollback", "rewind"}:
                    raise SourceCompatibilityError("unsupported_replay", "this Codex rollback variant is not qualified")
            if (
                self.harness == "pi"
                and number > 1
                and (
                    not isinstance(value.get("id"), str)
                    or "parentId" not in value
                    or value["parentId"] is not None
                    and not isinstance(value["parentId"], str)
                )
            ):
                raise SourceCompatibilityError(
                    "invalid_ancestry", "a Pi entry is missing required identity or parent fields"
                )
            if any(key in value and value[key] != observed_header["session_id"] for key in ("sessionId", "session_id")):
                raise SourceCompatibilityError(
                    "source_identity_changed", "a selected session contains contradictory identity"
                )
            record_stream = stream
            if self.harness == "claude" and value.get("isSidechain") and not record_stream:
                record_stream = "agent:" + str(value.get("agentId", "sidechain")) + ":"
            record = self._record(native, value, number, record_stream)
            # Original bytes survive only when the complete record is permitted.
            record = replace(record, raw_json=raw if value == original else None)
            occurrence = self._write(record)
            if (self.harness == "pi" and number == 1) or (self.harness == "gemini" and "id" not in value):
                self.connection.execute("UPDATE snapshot_records SET active=0 WHERE rowid=?", (occurrence,))
            if self.harness == "gemini":
                if number > 1 and not stream and "id" not in value and not any(key.startswith("$") for key in value):
                    self._metadata(native, value)
                self._gemini(native, value, occurrence, stream, record.occurrence_id or "")
            elif record.entity_id is not None:
                assert record.source_id is not None
                key = record.source_id
                self._entity(
                    native,
                    key,
                    record.parent_source_id,
                    occurrence,
                    number,
                    unique=self.harness == "pi",
                    scope=record_stream,
                )
                if record_stream == stream and (self.harness == "pi" or record.source_table == "jsonl_message"):
                    head = key
            if self.harness == "claude" and value.get("type") in {"system", "progress"}:
                self.connection.execute("UPDATE snapshot_records SET active=1 WHERE rowid=?", (occurrence,))
        if self.harness == "pi":
            self._tree(native, head, stream)
        elif self.harness == "claude":
            # Claude records without ancestry are a qualified linear export;
            # with parentUuid, the last persisted conversational record is head.
            linked = self.connection.execute(
                "SELECT 1 FROM entities WHERE session_id=? AND parent_id IS NOT NULL LIMIT 1", (native,)
            ).fetchone()
            if linked:
                self._tree(native, head, stream)
            if not stream:
                for (scope,) in self.connection.execute(
                    "SELECT DISTINCT scope FROM entities WHERE session_id=? AND scope!=''", (native,)
                ):
                    scope_head = self.connection.execute(
                        "SELECT e.entity_id FROM entities e JOIN snapshot_records r ON r.rowid=e.occurrence"
                        " WHERE e.session_id=? AND e.scope=? AND r.source_table='jsonl_message'"
                        " ORDER BY e.ordinal DESC LIMIT 1",
                        (native, scope),
                    ).fetchone()
                    if scope_head:
                        self._tree(native, scope_head[0], scope)
                        self._anchor_scope(native, scope, {})
        if stream:
            self._anchor_scope(native, stream, observed_header)
        self.writer.flush()
        verified_digest = digest_file(path)
        if before != identity_stat(path) or copied_digest.hexdigest() != verified_digest:
            raise SnapshotError("source_changed", "a selected source file changed during its stable read")

    def _write(self, record: SourceRecord) -> int:
        self.writer.write_record(
            record,
            (record.sequence_value,),
            CanonicalRecordEnvelope(
                decoded=record.decoded_value,
                raw_text=record.raw_json,
                raw_columns=record.raw_columns,
                warnings=record.warnings,
            ),
            {},
        )
        self.writer.flush()
        occurrence = int(self.connection.execute("SELECT MAX(rowid) FROM snapshot_records").fetchone()[0])
        self.connection.execute(
            "UPDATE snapshot_records SET path_ordinal=? WHERE rowid=?",
            (record.sequence_value if isinstance(record.sequence_value, int) else occurrence, occurrence),
        )
        value = record.decoded_value
        if not record.raw_columns.get("subagent") and isinstance(value, dict):
            result = value.get("toolUseResult")
            if isinstance(result, dict) and isinstance(result.get("agentId"), str):
                self.connection.execute(
                    "INSERT OR IGNORE INTO agent_links VALUES(?,?,?)",
                    (record.session_id, result["agentId"], occurrence),
                )
            tools = value.get("toolCalls")
            if isinstance(tools, list):
                for tool in tools:
                    if isinstance(tool, dict) and isinstance(tool.get("agentId"), str):
                        self.connection.execute(
                            "INSERT OR IGNORE INTO agent_links VALUES(?,?,?)",
                            (record.session_id, tool["agentId"], occurrence),
                        )
        return occurrence

    def _warn(self, code: str) -> None:
        if code not in self.export_warnings:
            self.export_warnings = (*self.export_warnings, code)

    def _call_anchor(self, native: str, call_id: str) -> int | None:
        rows = self.connection.execute(
            "SELECT MIN(COALESCE(r.anchor_ordinal,r.path_ordinal,r.rowid))"
            " FROM calls c JOIN snapshot_records r ON r.canonical_record=c.locator"
            " WHERE c.session_id=? AND c.call_id=? AND c.invocation_id IS NOT NULL AND r.active=1"
            " GROUP BY c.invocation_id",
            (native, call_id),
        )
        first = rows.fetchone()
        if first is None or rows.fetchone() is not None:
            return None
        return int(first[0])

    def _anchor_scope(self, native: str, scope: str, header: dict[str, Any]) -> None:
        anchor: int | None = None
        anchor_source: str | None = None
        root = self.connection.execute(
            "SELECT parent_id FROM entities WHERE session_id=? AND scope=? ORDER BY ordinal LIMIT 1", (native, scope)
        ).fetchone()
        if root and root[0] is not None:
            row = self.connection.execute(
                "SELECT COALESCE(anchor_ordinal,path_ordinal,rowid),source_id FROM snapshot_records"
                " WHERE session_id=? AND source_id=? AND record_scope!=? AND active=1 LIMIT 1",
                (native, root[0], scope),
            ).fetchone()
            anchor = int(row[0]) if row else None
            anchor_source = row[1] if row else None
        call = header.get("parentToolUseID", header.get("parent_tool_use_id"))
        if anchor is None and isinstance(call, str):
            anchor = self._call_anchor(native, call)
            if anchor is not None:
                row = self.connection.execute(
                    "SELECT r.source_id FROM calls c JOIN snapshot_records r ON r.canonical_record=c.locator"
                    " WHERE c.session_id=? AND c.call_id=? AND c.invocation_id IS NOT NULL AND r.active=1"
                    " ORDER BY r.source_id LIMIT 1",
                    (native, call),
                ).fetchone()
                anchor_source = row[0] if row else None
        if anchor is None:
            agent = scope.removeprefix("agent:").removesuffix(":").removesuffix(".gz")
            if agent.endswith(".jsonl"):
                agent = agent.removesuffix(".jsonl")
                if self.harness == "claude":
                    agent = agent.removeprefix("agent-")
            rows = self.connection.execute(
                "SELECT COALESCE(r.anchor_ordinal,r.path_ordinal,r.rowid),MIN(r.source_id)"
                " FROM agent_links a JOIN snapshot_records r ON r.rowid=a.occurrence"
                " WHERE a.session_id=? AND a.agent_id=? AND r.active=1"
                " GROUP BY COALESCE(r.anchor_ordinal,r.path_ordinal,r.rowid)",
                (native, agent),
            )
            first = rows.fetchone()
            if first and rows.fetchone() is None:
                anchor = int(first[0])
                anchor_source = first[1]
        if anchor is None:
            self.connection.execute(
                "UPDATE snapshot_records SET active=0 WHERE session_id=? AND record_scope=?", (native, scope)
            )
            self._warn("unattributed_subagent")
        else:
            self.connection.execute(
                "UPDATE snapshot_records SET anchor_ordinal=? WHERE session_id=? AND record_scope=?",
                (anchor, native, scope),
            )
            # Retain the already-proven execution anchor for the shared target
            # projection, rather than reconstructing it from sidecar filenames.
            parent_row = self.connection.execute(
                "SELECT source_id,canonical_record FROM snapshot_records"
                " WHERE source_id=? AND session_id=? AND active=1",
                (anchor_source, native),
            ).fetchone()
            if parent_row is not None:
                if not isinstance(call, str):
                    call_rows = self.connection.execute(
                        "SELECT DISTINCT call_id FROM calls WHERE session_id=? AND locator=?",
                        (native, parent_row[1]),
                    )
                    only = call_rows.fetchone()
                    call = only[0] if only and call_rows.fetchone() is None else None
                if not isinstance(call, str) and self.harness == "gemini":
                    parent_record = self._load_entity(native, str(parent_row[0]))
                    parent_value = parent_record.decoded_value if parent_record is not None else None
                    agent_id = scope.removeprefix("agent:").removesuffix(":").removesuffix(".gz").removesuffix(".jsonl")
                    if isinstance(parent_value, dict):
                        matches = [
                            tool.get("id")
                            for tool in parent_value.get("toolCalls", [])
                            if isinstance(tool, dict)
                            and tool.get("agentId") == agent_id
                            and isinstance(tool.get("id"), str)
                        ]
                        if len(matches) == 1:
                            call = matches[0]
                stored = self._header_payload(
                    self.connection.execute(
                        "SELECT canonical_session_row FROM snapshot_sessions WHERE session_id=?", (native,)
                    ).fetchone()[0]
                )
                links = dict((stored["session"].get("metadata") or {}).get("execution_links", {}))
                links[scope] = {"source_id": parent_row[0], "call_id": call}
                self._metadata(native, {"execution_links": links})

    def _record(self, native: str, value: dict[str, Any], number: int, stream: str) -> SourceRecord:
        kind = str(value.get("type", "record"))
        payload: dict[str, Any] = (
            cast(dict[str, Any], value["payload"]) if isinstance(value.get("payload"), dict) else value
        )
        message: dict[str, Any] = (
            cast(dict[str, Any], value["message"]) if isinstance(value.get("message"), dict) else payload
        )
        role = message.get("role", kind)
        classification: RecordClassificationName = "system"
        content = message.get("content")
        tool_result = isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") in {"tool_result", "functionResponse"} for p in content
        )
        if (
            role == "user"
            and not tool_result
            and not value.get("isMeta")
            and not value.get("isCompactSummary")
            and not stream
        ):
            classification = "user"
        elif role in {"assistant", "gemini"}:
            classification = "assistant"
        if self.harness == "codex" and kind != "response_item":
            classification = "system"  # event_msg.user_message duplicates response_item prompts.
        source_id = value.get("uuid", value.get("id", payload.get("id")))
        fallback_id = source_id is None
        if source_id is not None and not isinstance(source_id, str):
            raise SourceCompatibilityError("invalid_record_identity", "a persisted record identity must be a string")
        source_id = stream + str(source_id) if source_id is not None else stream + f"line-{number}"
        parent = value.get("parentUuid", value.get("parentId", value.get("parent_id")))
        if parent is not None:
            primary = (
                self.connection.execute(
                    "SELECT 1 FROM entities WHERE session_id=? AND entity_id=? AND scope=''", (native, str(parent))
                ).fetchone()
                if stream
                else None
            )
            parent = str(parent) if primary else stream + str(parent)
        # Schema envelopes remain complete. Facts are additionally normalized for
        # the existing shared model/tool projection; adapters never invent usage.
        decoded = dict(value)
        if self.harness == "codex" and payload is not value:
            decoded.update(payload)
            if "model_provider" in payload:
                decoded["providerID"] = payload["model_provider"]
        model_context: dict[str, Any] = {}
        if self.harness == "codex":
            if kind == "turn_context" and isinstance(decoded.get("model"), str):
                context = {key: decoded[key] for key in ("model", "providerID") if key in decoded}
                self.connection.execute(
                    "INSERT OR REPLACE INTO model_observations VALUES(?,?,?,?)",
                    (native, stream, json.dumps(self.writer._redaction(context, ())), source_id),
                )
            elif classification == "assistant" and "model" not in decoded:
                observed = self.connection.execute(
                    "SELECT value,source_id FROM model_observations WHERE session_id=? AND scope=?", (native, stream)
                ).fetchone()
                if observed is not None:
                    decoded.update(json.loads(observed[0]))
                    model_context["model_observation_source_id"] = observed[1]
        if "modelId" in decoded:
            decoded["modelID"] = decoded["modelId"]
        if message is not value:
            for key in (
                "role",
                "content",
                "model",
                "provider",
                "usage",
                "stopReason",
                "stop_reason",
                "toolCallId",
                "toolName",
                "isError",
                "is_error",
                "error",
                "cost",
                "thinking",
                "reasoning",
                "tool_calls",
            ):
                if key in message:
                    decoded[key] = message[key]
        if self.harness == "gemini" and kind == "gemini":
            classification = "assistant"
        container = classification in {"user", "assistant"} or role == "user" or kind == "message"
        if stream:
            classification = "system"
        has_entity = (
            container
            or (self.harness == "pi" and kind != "session")
            or (self.harness == "gemini" and "id" in value)
            or (self.harness == "claude" and "uuid" in value and "parentUuid" in value)
        )
        raw_time = message.get("timestamp", value.get("timestamp"))
        observed_time = timestamp(raw_time)
        warnings: tuple[str, ...] = ("source_id_line_fallback",) if fallback_id else ()
        if raw_time is not None and observed_time is None:
            warnings += ("unavailable_source_timestamp",)
        usage_context: dict[str, Any] = {}
        if self.harness == "codex" and classification == "user":
            self.connection.execute(
                "INSERT INTO usage_turns VALUES(?,?,1) ON CONFLICT(session_id,scope) DO UPDATE SET count=count+1",
                (native, stream),
            )
        if self.harness == "codex" and isinstance(decoded.get("info"), dict):
            cumulative = decoded["info"].get("total_token_usage")
            if isinstance(cumulative, dict):
                previous = self.connection.execute(
                    "SELECT value FROM usage_observations WHERE session_id=? AND scope=?", (native, stream)
                ).fetchone()
                usage_context["usage_previous_cumulative"] = json.loads(previous[0]) if previous else {}
                started = self.connection.execute(
                    "SELECT count FROM usage_turns WHERE session_id=? AND scope=?", (native, stream)
                ).fetchone()
                usage_context["usage_initial_observation"] = previous is None and (started is None or started[0] <= 1)
                self.connection.execute(
                    "INSERT OR REPLACE INTO usage_observations VALUES(?,?,?)",
                    (native, stream, json.dumps(self.writer._redaction(cumulative, ()))),
                )
        return SourceRecord(
            source_table="jsonl_message" if container else "jsonl",
            source_type=kind,
            source_id=source_id,
            session_id=native,
            decoded_value=decoded,
            sequence_domain="line",
            sequence_value=number,
            parent_source_id=parent,
            source_identity=f"{stream}occurrence-{number}",
            occurrence_id=f"{stream}occurrence-{number}",
            entity_id=source_id if has_entity else None,
            invocation_id=(
                str(message["id"])
                if self.harness in {"claude", "codex"} and "id" in message
                else source_id
                if self.harness in {"pi", "gemini"}
                else None
            )
            if role in {"assistant", "gemini"}
            else None,
            call_id=message.get("call_id", message.get("toolCallId")),
            raw_columns={
                "harness": self.harness,
                "subagent": stream or None,
                "physical_line": number,
                **usage_context,
                **model_context,
            },
            timestamp=observed_time,
            classification=classification,
            warnings=warnings,
        )

    def _entity(
        self,
        native: str,
        key: str,
        parent: str | None,
        occurrence: int,
        ordinal: int,
        *,
        unique: bool = False,
        scope: str = "",
    ) -> None:
        previous = self.connection.execute(
            "SELECT occurrence,ordinal,scope FROM entities WHERE session_id=? AND entity_id=?", (native, key)
        ).fetchone()
        if previous:
            if previous[2] != scope:
                raise SourceCompatibilityError("ambiguous_entity_scope", "logical IDs collide across subagent scopes")
            if unique:
                raise SourceCompatibilityError("duplicate_entity", "the schema requires unique logical entry IDs")
            self.connection.execute("UPDATE snapshot_records SET active=0 WHERE rowid=?", (previous[0],))
            ordinal = previous[1]
        self.connection.execute(
            "INSERT OR REPLACE INTO entities VALUES(?,?,?,?,?,0,?)", (native, key, parent, occurrence, ordinal, scope)
        )
        self.connection.execute(
            "UPDATE snapshot_records SET entity_id=?,path_ordinal=?,active=1 WHERE rowid=?", (key, ordinal, occurrence)
        )

    def _tree(self, native: str, head: str | None, stream: str) -> None:
        db = self.connection
        # stream keys use a colon prefix; no LIKE wildcards from untrusted IDs.
        db.execute(
            "UPDATE snapshot_records SET active=0 WHERE rowid IN (SELECT occurrence FROM entities"
            " WHERE session_id=? AND scope=?)",
            (native, stream),
        )
        previous_order: int | None = None
        while head is not None:
            row = db.execute(
                "SELECT parent_id,occurrence,visited,scope,ordinal FROM entities WHERE session_id=? AND entity_id=?",
                (native, head),
            ).fetchone()
            if row is not None and stream and row[3] != stream:
                break  # Recorded invoking parent is outside this subagent's path.
            if row is None or row[2]:
                raise SourceCompatibilityError(
                    "invalid_ancestry", "persisted ancestry contains a cycle or unresolved parent"
                )
            if previous_order is not None and row[4] >= previous_order:
                raise SourceCompatibilityError("invalid_ancestry", "a persisted parent must precede its child")
            previous_order = row[4]
            db.execute("UPDATE entities SET visited=1 WHERE session_id=? AND entity_id=?", (native, head))
            db.execute("UPDATE snapshot_records SET active=1,path_ordinal=? WHERE rowid=?", (row[4], row[1]))
            head = row[0]

    def _load_entity(self, native: str, key: str) -> SourceRecord | None:
        row = self.connection.execute(
            "SELECT r.session_id,r.source_table,r.source_id,r.canonical_record,r.raw_permitted_fields"
            " FROM snapshot_records r JOIN entities e ON r.rowid=e.occurrence"
            " WHERE e.session_id=? AND e.entity_id=?",
            (native, key),
        ).fetchone()
        if row is None:
            return None
        path = Path(self.connection.execute("PRAGMA database_list").fetchone()[2]).with_suffix(".payloads")
        return _restore_source_record(row[0], row[1], row[2], read_payload(path, row[3]), read_payload(path, row[4]))

    def _gemini(self, native: str, value: dict[str, Any], occurrence: int, stream: str, provenance: str) -> None:
        db = self.connection
        if "$rewindTo" in value:
            row = db.execute(
                "SELECT ordinal FROM entities WHERE session_id=? AND entity_id=?", (native, stream + value["$rewindTo"])
            ).fetchone()
            predicate, params = ("ordinal>=?", (row[0],)) if row else ("1", ())
            predicate = "scope=? AND " + predicate
            params = (stream, *params)
            db.execute(
                "UPDATE snapshot_records SET active=0 WHERE rowid IN (SELECT occurrence FROM entities"
                f" WHERE session_id=? AND {predicate})",
                (native, *params),
            )
            db.execute(f"DELETE FROM entities WHERE session_id=? AND {predicate}", (native, *params))
        elif "$patch" in value:
            patch = value["$patch"]
            if not isinstance(patch, dict):
                raise SourceCompatibilityError("unsupported_schema", "invalid Gemini patch")
            for field_name in ("updates", "removeIds", "orderIds", "toolCalls"):
                if field_name in patch and not isinstance(patch[field_name], list):
                    raise SourceCompatibilityError("unsupported_replay", "Gemini replay lists have an invalid shape")
            for update in ([patch] if "id" in patch else []) + patch.get("updates", []):
                old = self._load_entity(native, stream + str(update.get("id")))
                if old is None:
                    continue  # Upstream ignores patches of removed/unknown IDs.
                decoded = dict(cast(dict[str, Any], old.decoded_value))
                if "content" in update:
                    decoded["content"] = update["content"]
                for tool_patch in update.get("toolCalls", []):
                    for tool in decoded.get("toolCalls", []):
                        if tool.get("id") == tool_patch.get("id") and "result" in tool_patch:
                            tool["result"] = tool_patch["result"]
                derived = replace(
                    old,
                    decoded_value=decoded,
                    raw_json=None,
                    raw_columns={**old.raw_columns, "effective_from_occurrence": provenance},
                    occurrence_id=f"effective:{provenance}:{old.source_id}",
                    source_identity=f"effective:{provenance}:{old.source_id}",
                )
                new = self._write(derived)
                self._entity(native, str(old.source_id), None, new, occurrence, scope=stream)
            for key in patch.get("removeIds", []):
                key = stream + key
                db.execute(
                    "UPDATE snapshot_records SET active=0 WHERE rowid=(SELECT occurrence FROM entities"
                    " WHERE session_id=? AND entity_id=?)",
                    (native, key),
                )
                db.execute("DELETE FROM entities WHERE session_id=? AND entity_id=?", (native, key))
            maximum = db.execute(
                "SELECT COALESCE(MAX(ordinal),0) FROM entities WHERE session_id=?", (native,)
            ).fetchone()[0]
            db.execute("CREATE TEMP TABLE IF NOT EXISTS replay_ordered(id TEXT PRIMARY KEY)")
            db.execute("DELETE FROM replay_ordered")
            for index, key in enumerate(patch.get("orderIds", []), 1):
                key = stream + key
                if db.execute("INSERT OR IGNORE INTO replay_ordered VALUES(?)", (key,)).rowcount == 0:
                    continue
                db.execute(
                    "UPDATE entities SET ordinal=? WHERE session_id=? AND entity_id=?", (maximum + index, native, key)
                )
            db.execute(
                "UPDATE snapshot_records SET path_ordinal=(SELECT ordinal FROM entities"
                " WHERE occurrence=snapshot_records.rowid) WHERE session_id=? AND entity_id IS NOT NULL",
                (native,),
            )
        elif "$set" in value:
            update = value["$set"]
            if not isinstance(update, dict) or not stream and "sessionId" in update and update["sessionId"] != native:
                raise SourceCompatibilityError("source_identity_changed", "invalid Gemini metadata identity")
            if not stream:
                self._metadata(native, update)
            if "messages" in update:
                if not isinstance(update["messages"], list):
                    raise SourceCompatibilityError("unsupported_replay", "Gemini checkpoints require a message array")
                db.execute(
                    "UPDATE snapshot_records SET active=0 WHERE rowid IN"
                    " (SELECT occurrence FROM entities WHERE session_id=? AND scope=?)",
                    (native, stream),
                )
                db.execute("DELETE FROM entities WHERE session_id=? AND scope=?", (native, stream))
                for index, message in enumerate(update["messages"], 1):
                    record = self._record(native, message, index, stream)
                    record = replace(
                        record,
                        raw_columns={**record.raw_columns, "effective_from_occurrence": provenance},
                        occurrence_id=f"checkpoint:{provenance}:{index}",
                        source_identity=f"checkpoint:{provenance}:{index}",
                    )
                    new = self._write(record)
                    self._entity(native, str(record.source_id), None, new, index, scope=stream)
        elif "id" in value:
            # Repeated IDs replace content without changing insertion order.
            maximum = db.execute(
                "SELECT COALESCE(MAX(ordinal),0)+1 FROM entities WHERE session_id=?", (native,)
            ).fetchone()[0]
            self._entity(native, stream + str(value["id"]), None, occurrence, maximum, scope=stream)

    def _metadata(self, native: str, updates: dict[str, Any]) -> None:
        row = self.connection.execute(
            "SELECT canonical_session_row FROM snapshot_sessions WHERE session_id=?", (native,)
        ).fetchone()
        stored = self._header_payload(row[0])
        safe = cast(
            dict[str, Any],
            self.writer._redaction({key: item for key, item in updates.items() if key != "messages"}, ()),
        )
        session = stored["session"]
        if "sessionId" in updates and updates["sessionId"] != native:
            raise SourceCompatibilityError("source_identity_changed", "Gemini metadata session identity changed")
        if "projectHash" in updates:
            original_header = self.connection.execute(
                "SELECT header FROM file_candidates WHERE id=? AND selected=1", (native,)
            ).fetchone()
            if (
                _header_digest({"projectHash": updates["projectHash"]}, None)
                != self._header_payload(original_header[0])["project_identity_digest"]
            ):
                raise SourceCompatibilityError("source_identity_changed", "Gemini metadata project identity changed")
        if "kind" in updates:
            original_header = self.connection.execute(
                "SELECT header FROM file_candidates WHERE id=? AND selected=1", (native,)
            ).fetchone()
            if (
                _header_digest({"kind": updates["kind"]}, None)
                != self._header_payload(original_header[0])["kind_digest"]
            ):
                raise SourceCompatibilityError(
                    "unsupported_header_update", "Gemini session-kind updates are not a qualified variant"
                )
        if "startTime" in updates:
            original = session.get("time_created")
            created = timestamp(updates["startTime"])
            if (
                created is None
                or original is None
                or TimestampValue(original["value"], original["unit"]).nanoseconds != created.nanoseconds
            ):
                raise SourceCompatibilityError(
                    "unsupported_header_update", "Gemini creation-time updates are not a qualified variant"
                )
        if "lastUpdated" in safe:
            observed = timestamp(safe["lastUpdated"])
            session["time_updated"] = {"value": observed.value, "unit": observed.unit} if observed else None
        if "summary" in safe:
            session["title"] = safe["summary"]
        if "projectHash" in safe:
            session["project_id"] = _native_id(safe["projectHash"])
        session["metadata"] = {**(session.get("metadata") or {}), **safe}
        stored["decoded"] = {**stored["decoded"], **safe}
        self.connection.execute(
            "UPDATE snapshot_sessions SET canonical_session_row=? WHERE session_id=?",
            (self.writer._payload(stored, header=True), native),
        )
