"""Explicit harness adapter registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mlflow_import.errors import SourceCompatibilityError


class HarnessAdapter(Protocol):
    name: str
    qualified: bool

    def detect(self, source: Path) -> bool: ...

    def default_locations(self, session_id: str | None = None) -> tuple[Path, ...]: ...


@dataclass(frozen=True)
class Adapter:
    name: str
    qualified: bool
    locations: tuple[Path, ...] = ()

    def detect(self, source: Path) -> bool:
        if self.name == "opencode":
            return source.suffix in {".db", ".sqlite", ".sqlite3"}
        if self.name == "claude":
            from .claude import detect

            return detect(source)
        if self.name == "codex":
            from .codex import detect

            return detect(source)
        return False

    def default_locations(self, session_id: str | None = None) -> tuple[Path, ...]:
        if self.name == "claude":
            from .claude import default_sources as claude_default_sources

            return claude_default_sources(session_id)
        if self.name == "codex":
            from .codex import default_sources as codex_default_sources

            return codex_default_sources()
        return self.locations

    def require_qualified(self) -> None:
        if not self.qualified:
            raise SourceCompatibilityError(
                "adapter_not_qualified", "the selected harness has no qualified source schema"
            )


ADAPTERS: dict[str, Adapter] = {
    "opencode": Adapter("opencode", True, (Path("~/.local/share/opencode/opencode.db"),)),
    "claude": Adapter("claude", True),
    "codex": Adapter("codex", True),
    "pi": Adapter("pi", False),
    "gemini": Adapter("gemini", False),
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


__all__ = ["ADAPTERS", "Adapter", "HarnessAdapter", "detect_harness", "get_adapter"]
