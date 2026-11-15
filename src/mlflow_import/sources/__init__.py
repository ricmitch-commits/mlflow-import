"""Explicit harness adapter registry."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from mlflow_import.errors import SourceCompatibilityError
from mlflow_import.source.model import source_namespace
from mlflow_import.sources.schemas import FORMATS, SourceFormat

if TYPE_CHECKING:
    from mlflow_import.source.snapshot import SourceExporter


@dataclass(frozen=True)
class SourceCandidate:
    harness: str
    path: Path
    namespace: str


class HarnessAdapter(Protocol):
    name: str
    qualified: bool

    def detect(self, source: Path) -> bool: ...

    def default_locations(self, session_id: str | None = None) -> tuple[Path, ...]: ...

    def default_sources(self) -> Iterator[SourceCandidate]: ...

    @property
    def source_format(self) -> SourceFormat: ...


@dataclass(frozen=True)
class Adapter:
    name: str
    qualified: bool
    locations: tuple[Path, ...] = ()

    @property
    def source_format(self) -> SourceFormat:
        return FORMATS[self.name]

    def default_sources(self) -> Iterator[SourceCandidate]:
        for location in self.default_locations():
            path = Path(os.path.abspath(location.expanduser()))
            if path.exists():
                yield SourceCandidate(self.name, path, source_namespace(path, self.name))

    def detect(self, source: Path) -> bool:
        if self.name == "opencode":
            from mlflow_import.source.connection import open_live_connection
            from mlflow_import.source.probe import probe_schema

            try:
                with open_live_connection(source) as live:
                    probe_schema(live.connection)
                return True
            except Exception:
                return False
        if self.name == "claude":
            from .claude import detect

            return detect(source)
        if self.name == "codex":
            from .codex import detect

            return detect(source)
        from .jsonl_common import detect as detect_file

        return detect_file(source, self.name)

    def default_locations(self, session_id: str | None = None) -> tuple[Path, ...]:
        if self.name == "claude":
            from .claude import default_sources as claude_default_sources

            return claude_default_sources(session_id)
        if self.name == "codex":
            from .codex import default_sources as codex_default_sources

            return codex_default_sources()
        if self.name == "pi":
            if override := os.environ.get("PI_CODING_AGENT_SESSION_DIR"):
                return (Path(override).expanduser(),)
            return (Path(os.environ.get("PI_CODING_AGENT_DIR", "~/.pi/agent")).expanduser() / "sessions",)
        if self.name == "gemini":
            home = Path(os.environ.get("GEMINI_CLI_HOME", "~")).expanduser()
            return (home / ".gemini" / "tmp", home / ".cache" / ".gemini" / "tmp")
        if self.name == "opencode":
            root = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
            return (root / "opencode" / "opencode.db",)
        return self.locations

    def exporter(self, source: Path) -> SourceExporter:
        """Construct the schema-backed file exporter (SQLite uses its catalog)."""
        if self.name == "opencode":
            raise SourceCompatibilityError(
                "sqlite_exporter_requires_transaction", "SQLite exporters require a bound read transaction"
            )
        from .jsonl_common import JsonlExporter

        return JsonlExporter(source, self.name)

    def require_qualified(self) -> None:
        if not self.qualified:
            raise SourceCompatibilityError(
                "adapter_not_qualified", "the selected harness has no qualified source schema"
            )


ADAPTERS: dict[str, Adapter] = {
    "opencode": Adapter("opencode", True, (Path("~/.local/share/opencode/opencode.db"),)),
    "claude": Adapter("claude", True),
    "codex": Adapter("codex", True),
    "pi": Adapter("pi", True),
    "gemini": Adapter("gemini", True),
}


def get_adapter(name: str) -> Adapter:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise SourceCompatibilityError("unknown_harness", "the requested harness is not supported") from None


def detect_harness(source: Path) -> str:
    matches = [name for name, adapter in ADAPTERS.items() if adapter.detect(source)]
    if len(matches) != 1:
        raise SourceCompatibilityError("ambiguous_source", "the source does not match exactly one harness schema")
    return matches[0]


__all__ = ["ADAPTERS", "Adapter", "HarnessAdapter", "SourceCandidate", "SourceFormat", "detect_harness", "get_adapter"]
