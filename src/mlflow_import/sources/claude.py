"""Fixture-qualified Claude Code JSONL adapter."""

from __future__ import annotations

import os
from pathlib import Path

from .jsonl_common import JsonlExporter

NAME = "claude"


def default_sources(session_id: str | None = None) -> tuple[Path, ...]:
    root = Path(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser()
    if session_id is not None:
        return tuple(sorted(root.glob(f"projects/*/{session_id}.jsonl")))
    return tuple(sorted(root.glob("projects/*/*.jsonl")))


def detect(path: Path) -> bool:
    return path.suffix == ".jsonl" and ("claude" in path.parts or "projects" in path.parts)


def exporter(path: Path) -> JsonlExporter:
    return JsonlExporter(path, NAME)


__all__ = ["NAME", "default_sources", "detect", "exporter"]
