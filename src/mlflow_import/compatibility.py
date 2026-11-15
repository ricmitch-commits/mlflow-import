"""Fixture-backed v1 ownership/result decoders and conflict identity rules.

Decoding is a read view, not authority to reuse/delete a trace. Qualified source
identity and the complete importer contract are checked separately.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from mlflow_import.errors import ConfigurationError
from mlflow_import.storage import DiskList, DiskMap

DECODER_VERSION = "compatibility-v1"
OWNERSHIP_ALIASES = {
    "importer": "importer",
    "session_id": "session_id",
    "turn_key": "turn_key",
    "turn_index": "turn_index",
    "importer_version": "importer_version",
    "mapping_version": "mapping_version",
    "artifact_schema_version": "artifact_schema_version",
    "redaction_mode": "redaction_mode",
    "catalog_version": "source_format",
    "render_fingerprint": "turn_fingerprint",
    "operation_key": "operation_key",
    "warnings": "warnings",
    "source_fingerprint": "source_fingerprint",
    "source_schema": "source_schema",
}


def importer_marker(tags: Mapping[str, str], metadata: Mapping[str, str]) -> str | None:
    values = [
        "mlflow-import" if key == "opencode.importer" and value == "opencode-mlflow" else value
        for mapping in (metadata, tags)
        for key, value in mapping.items()
        if key in {"opencode.importer", "mlflow_import.importer"}
    ]
    return "mlflow-import" if values and all(value == "mlflow-import" for value in values) else None


def decode_ownership(tags: Mapping[str, str], metadata: Mapping[str, str]) -> dict[str, str]:
    """Normalize neutral-v1 and opencode-v1 fields without rewriting the target."""
    result = dict(metadata)
    # Creation tags are immutable ownership evidence, not just display fields.
    # Normalize *all* namespaced facts so recovery and pricing share one view.
    for key, tag_value in tags.items():
        if key.startswith(("mlflow_import.attempt_", "mlflow_import.session_artifact", "opencode.artifact_")):
            continue
        if key.startswith("mlflow_import.current.") or key in {
            "mlflow_import.turn_index", "mlflow_import.source_revision", "mlflow_import.metadata_fingerprint"
        }:
            continue
        if key.startswith(("mlflow_import.", "opencode.", "mlflow.source.")) or key in {
            "parent_session_id", "mlflow.trace.session", "mlflow.trace.user", "mlflow.user"
        }:
            if key in metadata and metadata[key] != tag_value:
                raise ConfigurationError("ownership_conflict", "ownership locations contain contradictory values")
            result[key] = tag_value
    for legacy, neutral in OWNERSHIP_ALIASES.items():
        old, new = "opencode." + legacy, "mlflow_import." + neutral
        supplied = {
            (
                "mlflow-import"
                if legacy == "importer" and key == old and mapping[key] == "opencode-mlflow"
                else mapping[key]
            )
            for mapping in (metadata, tags)
            for key in (old, new)
            if key in mapping and not (legacy == "turn_index" and mapping is tags and key == new)
        }
        if len(supplied) > 1:
            raise ConfigurationError("ownership_conflict", "ownership aliases contain contradictory values")
        value = (
            metadata.get(new, metadata.get(old, tags.get(old)))
            if legacy == "turn_index"
            else metadata.get(new, tags.get(new, metadata.get(old, tags.get(old))))
        )
        if value is not None:
            if legacy == "importer" and value == "opencode-mlflow" and importer_marker(tags, metadata) is not None:
                result["mlflow_import.original_importer"] = value
                value = "mlflow-import"
            result[old] = value
            result[new] = value
    qualified = result.get("mlflow_import.qualified_session_id")
    if qualified is not None:
        expected = [
            "v1", result.get("mlflow_import.harness"),
            result.get("mlflow_import.source_namespace"), result.get("mlflow_import.session_id"),
        ]
        try:
            valid = json.loads(qualified) == expected and all(isinstance(item, str) and item for item in expected)
        except (ValueError, TypeError):
            valid = False
        if not valid or result.get("mlflow.trace.session", expected[-1]) not in {qualified, expected[-1]}:
            raise ConfigurationError("ownership_conflict", "qualified and native source identity disagree")
    return result


def request_ownership(request: Any) -> dict[str, str]:
    """Logical request facts; never serialize this view as physical metadata."""
    return decode_ownership(request.root_tags, request.metadata)


@dataclass(frozen=True)
class DecodedResult:
    original_schema_version: str
    document: Mapping[str, Any]
    decoder_version: str = DECODER_VERSION


def decode_result(value: Mapping[str, Any]) -> DecodedResult:
    """Read the two declared v1 result envelopes; retain the original wire version."""
    if not isinstance(value, Mapping):
        raise ConfigurationError("result_schema_invalid", "the result document must be a typed v1 object")
    version = value.get("schema_version")
    if not isinstance(version, str) or version not in {"mlflow-import.result.v1", "opencode-mlflow.result.v1"}:
        raise ConfigurationError(
            "result_schema_unsupported", "the result schema has no qualified compatibility decoder"
        )

    def invalid() -> None:
        raise ConfigurationError("result_schema_invalid", "the result document violates its typed v1 schema")

    def count(item: Any) -> bool:
        return type(item) is int and item >= 0

    def events(items: Any) -> None:
        if not isinstance(items, list):
            invalid()
        for event in items:
            if (
                not isinstance(event, dict)
                or not isinstance(event.get("code"), str)
                or not re.fullmatch(r"[a-z][a-z0-9_]*", event["code"])
                or not isinstance(event.get("message"), str)
            ):
                invalid()

    if (
        type(value.get("dry_run")) is not bool
        or any(not isinstance(value.get(key), dict) for key in ("experiment", "source", "summary", "redaction"))
        or not isinstance(value.get("sessions"), list)
    ):
        invalid()
    experiment, source, summary, redaction = (value[key] for key in ("experiment", "source", "summary", "redaction"))
    if (
        not isinstance(experiment.get("name"), str)
        or experiment.get("id") is not None
        and not isinstance(experiment["id"], str)
        or not isinstance(experiment.get("status"), str)
        or experiment["status"] not in {"planned", "created", "resolved", "unresolved"}
        or not isinstance(redaction.get("mode"), str)
        or redaction["mode"] not in {"redacted", "verbatim"}
        or not isinstance(redaction.get("detector_counts"), dict)
        or any(not count(item) for item in redaction["detector_counts"].values())
    ):
        invalid()
    for key in ("affected_field_count", "affected_record_count"):
        if not count(redaction.get(key)):
            invalid()
    for key in ("sessions_scanned", "sessions_selected", "logical_turn_count", "record_count"):
        if not count(source.get(key)):
            invalid()
    for key in (
        "selected_sessions",
        "selected_turns",
        "turns_imported",
        "turns_updated",
        "turns_skipped",
        "turns_failed",
        "retired_traces",
        "failed_items",
    ):
        if not count(summary.get(key)):
            invalid()
    events(value.get("errors"))
    events(summary.get("warnings"))
    actions = {"imported", "updated", "skipped", "failed", "retired"}

    def optional_details(item: Mapping[str, Any]) -> None:
        for key in (
            "trace_id",
            "error_code",
            "reason",
            "planned_action",
            "attempt_id",
            "import_instance_id",
            "attempt_state",
        ):
            if item.get(key) is not None and not isinstance(item[key], str):
                invalid()
        if item.get("planned_action") is not None and item["planned_action"] not in actions:
            invalid()
        if item.get("attempt_state") is not None and item["attempt_state"] not in {
            "prepared",
            "ready_to_retire",
            "complete",
            "aborted",
        }:
            invalid()
        if item.get("error_code") is not None and not re.fullmatch(r"[a-z][a-z0-9_]*", item["error_code"]):
            invalid()
        if item.get("error") is not None:
            events([item["error"]])

    for session in value["sessions"]:
        if (
            not isinstance(session, dict)
            or not isinstance(session.get("session_id"), str)
            or type(session.get("selected")) is not bool
            or not isinstance(session.get("action"), str)
            or session["action"] not in actions
            or not isinstance(session.get("turns"), list)
            or any(not count(session.get(key)) for key in ("record_count", "logical_turn_count"))
        ):
            invalid()
        events(session.get("warnings"))
        optional_details(session)
        for turn in session["turns"]:
            if (
                not isinstance(turn, dict)
                or turn.get("session_id") != session["session_id"]
                or not isinstance(turn.get("turn_key"), str)
                or not isinstance(turn.get("action"), str)
                or turn["action"] not in actions
                or type(turn.get("source_present")) is not bool
                or turn.get("turn_index") is not None
                and (not count(turn["turn_index"]) or turn["turn_index"] == 0)
            ):
                invalid()
            optional_details(turn)
            for key in ("affected_trace_ids", "retired_trace_ids", "conflicting_trace_ids"):
                if not isinstance(turn.get(key), list) or not all(isinstance(item, str) for item in turn[key]):
                    invalid()
            events(turn.get("warnings"))
    return DecodedResult(str(version), value)


class IdentityIndex:
    def __init__(self) -> None:
        self.identities: DiskMap[str, list[tuple[str, str]]] = DiskMap()

    def add(self, harness: str, namespace: str, native_id: str) -> None:
        bucket = self.identities.get(native_id)
        if bucket is None:
            bucket = DiskList(store=self.identities.store)
            self.identities[native_id] = bucket
        identity = (harness, namespace)
        if identity not in bucket:
            bucket.append(identity)

    def unique(self, native_id: str, harness: str, namespace: str) -> bool:
        identities = self.identities.get(native_id, ())
        return len(identities) == 1 and identities[0] == (harness, namespace)


_IDENTITIES: ContextVar[IdentityIndex | None] = ContextVar("source_identity_index", default=None)


@contextmanager
def identity_index(index: IdentityIndex) -> Iterator[None]:
    token = _IDENTITIES.set(index)
    try:
        yield
    finally:
        _IDENTITIES.reset(token)


def identity_projection(metadata: Mapping[str, str], tags: Mapping[str, str]) -> dict[str, Any]:
    """Identity-only proof: never persist foreign payloads or arbitrary metadata."""
    keys = (
        "mlflow.trace.session",
        "mlflow_import.harness",
        "mlflow_import.source_namespace",
        "mlflow_import.session_id",
        "opencode.session_id",
        "mlflow.source.name",
        "mlflow_import.import_instance_id",
        "mlflow_import.qualified_session_id",
    )
    return {key: metadata.get(key, tags.get(key)) for key in keys}


def identity_digest(metadata: Mapping[str, str], tags: Mapping[str, str]) -> str:
    encoded = json.dumps(identity_projection(metadata, tags), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def resolves_conflict(metadata: Mapping[str, str], tags: Mapping[str, str], qualified_id: str, native_id: str) -> bool:
    """Resolve canonical/native/legacy aliases; contradictory evidence disqualifies."""
    _, harness, namespace, session_id = json.loads(qualified_id)
    try:
        normalized = decode_ownership(
            {key: value for key, value in tags.items() if value is not None},
            {key: value for key, value in metadata.items() if value is not None},
        )
    except ConfigurationError:
        raise ConfigurationError("identity_conflict", "source identity contains contradictory locations") from None
    projection = identity_projection(normalized, {})
    native = projection["mlflow.trace.session"]
    if native not in {qualified_id, native_id, None}:
        return False
    legacy = projection["opencode.session_id"]
    if native is None and legacy != native_id and projection["mlflow_import.session_id"] != native_id:
        return False
    qualified = projection["mlflow_import.qualified_session_id"]
    if qualified is not None and qualified != qualified_id:
        raise ConfigurationError("identity_conflict", "qualified source identity contradicts the selected source")
    canonical = native == qualified_id or qualified == qualified_id
    for key in projection:
        if key in metadata and key in tags and metadata[key] != tags[key]:
            if canonical:
                raise ConfigurationError("identity_conflict", "canonical identity has contradictory source metadata")
            return False
    expected = {
        "mlflow_import.harness": harness,
        "mlflow.source.name": harness,
        "mlflow_import.source_namespace": namespace,
        "mlflow_import.session_id": session_id,
        "opencode.session_id": session_id,
    }
    aliases = {"claude_code": "claude", "gemini-cli": "gemini", "codex-cli": "codex", "OpenCode": "opencode"}
    evidence = False
    for key, wanted in expected.items():
        actual = projection[key]
        if actual is None:
            continue
        if key in {"mlflow_import.harness", "mlflow.source.name"}:
            actual = aliases.get(actual, actual)
            if key == "mlflow.source.name" and actual not in {"opencode", "claude", "codex", "pi", "gemini"}:
                # SDK source paths and plugin producer names are not harness
                # identity evidence. Only recognized harness values qualify.
                continue
        if actual != wanted:
            if canonical:
                raise ConfigurationError("identity_conflict", "canonical identity has contradictory source metadata")
            return False
        if key == "mlflow_import.source_namespace":
            evidence = True
    if canonical or evidence:
        return True
    index = _IDENTITIES.get()
    if index is None or not index.unique(native_id, harness, namespace):
        raise ConfigurationError("alias_ambiguous", "a native or legacy alias has no unambiguous source association")
    return True
