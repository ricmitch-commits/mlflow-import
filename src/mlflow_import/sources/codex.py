"""Fixture-qualified Codex rollout JSONL adapter."""

from __future__ import annotations

import os
from pathlib import Path

from .jsonl_common import JsonlExporter

NAME = "codex"


def default_sources() -> tuple[Path, ...]:
    root = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    return tuple(sorted(root.glob("sessions/*/*/*/rollout-*.jsonl"))) + tuple(
        sorted(root.glob("archived_sessions/*.jsonl"))
    )


def detect(path: Path) -> bool:
    return (
        path.is_dir() and ("codex" in path.parts or "sessions" in path.parts or "archived_sessions" in path.parts)
    ) or (path.suffix == ".jsonl" and ("rollout-" in path.name or "archived_sessions" in path.parts))


def exporter(path: Path) -> JsonlExporter:
    return JsonlExporter(path, NAME)


__all__ = ["NAME", "default_sources", "detect", "exporter"]
