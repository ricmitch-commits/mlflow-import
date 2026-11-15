"""Fixture-qualified Codex rollout JSONL adapter."""

from __future__ import annotations

import os
from pathlib import Path

from .jsonl_common import JsonlExporter

NAME = "codex"


def default_sources() -> tuple[Path, ...]:
    root = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    return (root,)


def detect(path: Path) -> bool:
    from .jsonl_common import detect as detect_schema

    return detect_schema(path, NAME)


def exporter(path: Path) -> JsonlExporter:
    return JsonlExporter(path, NAME)


__all__ = ["NAME", "default_sources", "detect", "exporter"]
