"""Fixture-qualified Claude Code JSONL adapter."""

from __future__ import annotations

import os
from pathlib import Path

from .jsonl_common import JsonlExporter

NAME = "claude"


def default_sources(session_id: str | None = None) -> tuple[Path, ...]:
    root = Path(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser()
    return (root / "projects",)


def detect(path: Path) -> bool:
    from .jsonl_common import detect as detect_schema

    return detect_schema(path, NAME)


def exporter(path: Path) -> JsonlExporter:
    return JsonlExporter(path, NAME)


__all__ = ["NAME", "default_sources", "detect", "exporter"]
