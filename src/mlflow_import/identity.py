"""Versioned source identities shared by mapping, lookup, artifacts and recovery."""

import json
from collections.abc import Mapping

from mlflow_import.source.model import SessionHeader


def harness_name(header: SessionHeader) -> str:
    for name in ("claude", "codex", "pi", "gemini"):
        if header.catalog_version.startswith(name + "-"):
            return name
    return "opencode"


def qualified_session(header: SessionHeader, native_id: str | None = None) -> str:
    """Compact JSON tuple; no delimiter ambiguity or absolute locator exposure."""
    return json.dumps(
        [
            "v1",
            harness_name(header),
            header.source_fingerprint or "unqualified",
            native_id if native_id is not None else header.session_id,
        ],
        separators=(",", ":"),
    )


def native_metadata(header: SessionHeader) -> dict[str, str]:
    """Historical facts only; never use the importing process's environment."""
    # Public historical object ingestion does not merge the SDK process
    # environment. Missing facts can therefore remain truly absent.
    values = {"mlflow.source.name": harness_name(header)}
    source = dict(header.raw_columns)
    if isinstance(header.metadata, Mapping):
        source.update(header.metadata)
    # Header facts have already crossed the common source redaction boundary.
    nested = source.get("metadata")
    if isinstance(nested, Mapping):
        source.update(nested)
    user = source.get("user", source.get("user_id", source.get("username")))
    if isinstance(user, Mapping):
        user = user.get("id", user.get("username", user.get("email", user.get("name"))))
    if isinstance(user, str):
        values["mlflow.trace.user"] = user
    git = source.get("git")
    if isinstance(git, Mapping):
        for source_key, target in (
            ("repository_url", "repoURL"),
            ("repoURL", "repoURL"),
            ("commit_hash", "commit"),
            ("commit", "commit"),
            ("branch", "branch"),
        ):
            value = git.get(source_key)
            if isinstance(value, str):
                values["mlflow.source.git." + target] = value
    for key in (
        "mlflow.trace.user",
        "mlflow.user",
        "mlflow.source.git.repoURL",
        "mlflow.source.git.commit",
        "mlflow.source.git.branch",
        "mlflow_import.client_request_id",
    ):
        value = source.get(key)
        if value is not None:
            values[key] = str(value)
    request_id = source.get("client_request_id")
    if isinstance(request_id, str):
        values["mlflow_import.client_request_id"] = request_id
    parent = header.parent_id or header.fork_session_id
    if parent is not None:
        values["parent_session_id"] = parent
    from mlflow_import.errors import SourceCompatibilityError

    if len(header.session_id) > 250 or len(values.get("parent_session_id", "")) > 250:
        raise SourceCompatibilityError(
            "native_metadata_too_long", "the native session or parent identity exceeds the pinned metadata limit"
        )
    if len(values.get("mlflow_import.client_request_id", "")) > 50:
        raise SourceCompatibilityError(
            "native_request_id_too_long", "the historical request identity exceeds the pinned field limit"
        )
    if len(qualified_session(header)) > 4096 or any(
        len(value) > 4096 for key, value in values.items() if key != "parent_session_id"
    ):
        raise SourceCompatibilityError(
            "native_metadata_too_long", "the historical creation tags exceed the pinned tag limit"
        )
    return values
