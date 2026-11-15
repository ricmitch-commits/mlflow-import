"""Qualified Gemini projects.json path bindings, never basename inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mlflow_import.errors import SourceCompatibilityError
from mlflow_import.resources import open_reader


def project_directory(source: Path, project_hash: object) -> str | None:
    """Bind .gemini/tmp/<registered-group>/chats to a registered absolute cwd.

    The persisted registry maps exact absolute paths to allocated group names.
    Require both that mapping and SHA-256 of the registered path to agree with
    projectHash. Missing/unqualified mappings yield unknown, never a guessed cwd.
    """
    if source.parent.name != "chats":
        return None
    group, root = source.parent.parent.name, source.parent.parent.parent
    if root.name != "tmp" or root.parent.name != ".gemini":
        return None
    registry = root.parent / "projects.json"
    if not registry.exists():
        return None
    if registry.is_symlink():
        raise SourceCompatibilityError("unsupported_project_registry", "Gemini project registry symlinks unsupported")
    # Registry is metadata, but still has a separate pre-decode bound.
    with open_reader(registry, "r", encoding="utf-8") as handle:
        text = handle.read(1024 * 1024 + 1)
    if len(text.encode("utf-8")) > 1024 * 1024:
        raise SourceCompatibilityError("unsupported_project_registry", "Gemini project registry exceeds metadata bound")
    try:
        from mlflow_import.sources.jsonl_common import _unique_object

        value = json.loads(text, object_pairs_hook=_unique_object)
        projects = value["projects"]
        if not isinstance(projects, dict):
            raise ValueError
        matches = []
        for directory, allocated in projects.items():
            if not isinstance(allocated, str) or not Path(directory).is_absolute():
                raise ValueError
            if allocated == group and hashlib.sha256(directory.encode("utf-8")).hexdigest() == project_hash:
                matches.append(directory)
        if len(matches) > 1:
            raise ValueError
    except (ValueError, KeyError, TypeError):
        raise SourceCompatibilityError(
            "unsupported_project_registry", "Gemini project registry has an unsupported or ambiguous mapping"
        ) from None
    return matches[0] if matches else None
