"""Shared source-outcome classification for turns and mapped children.

OpenCode persists outcome signals in several envelopes.  This module keeps
the vocabulary and precedence in one place so a status cannot be interpreted
as successful by one layer and failed by another.  The returned outcome value
is the source spelling (including case/whitespace); callers use
:func:`classify_outcome` only for the MLflow status.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "FAILURE_OUTCOMES",
    "PENDING_OUTCOMES",
    "SESSION_EVENT_TABLES",
    "classify_decoded",
    "classify_outcome",
    "declared_source_outcome",
    "decoded_source_outcome",
    "is_explicit_failure",
    "record_source_outcome",
    "tool_source_outcome",
]

FAILURE_OUTCOMES = frozenset(
    {
        "error",
        "failed",
        "failure",
        "cancel",
        "cancelled",
        "canceled",
        "interrupted",
        "aborted",
        "abort",
        "timeout",
        "timed_out",
        "terminated",
    }
)
"""Normalized source values that are explicit failures."""

PENDING_OUTCOMES = frozenset({"pending", "queued", "running", "in_progress", "in-progress"})
"""Normalized source values that are incomplete but not failed."""

SESSION_EVENT_TABLES = frozenset({"session_pending", "session_inbox", "todo"})
"""Catalogued session-scoped lifecycle/event tables with source outcomes."""


def classify_outcome(outcome: str | None) -> str:
    """Map one source outcome to the MLflow-compatible status vocabulary."""
    if outcome is None:
        return "OK"
    normalized = outcome.strip().lower()
    if normalized in FAILURE_OUTCOMES:
        return "ERROR"
    if normalized in PENDING_OUTCOMES:
        return "UNSET"
    return "OK"


def decoded_source_outcome(decoded: object) -> str | None:
    """Return the exact explicit outcome spelling carried by a JSON object.

    Failure-valued ``outcome``/``status``/``finish`` fields take precedence
    over successful fields, and an error payload is represented by the stable
    source outcome ``error`` when no more specific failure value exists.
    Unknown fields and unknown values do not become failures.
    """
    if not isinstance(decoded, Mapping):
        return None
    lifecycle_error = decoded.get("lifecycle_error_outcome")
    if isinstance(lifecycle_error, str) and lifecycle_error:
        return lifecycle_error
    values = [decoded.get(name) for name in ("outcome", "status", "finish")]
    strings = [value for value in values if isinstance(value, str) and value]
    for value in strings:
        if classify_outcome(value) == "ERROR":
            return value
    if decoded.get("error") is not None:
        return "error"
    return strings[0] if strings else None


def declared_source_outcome(decoded: object) -> str | None:
    """Return only an explicitly declared outcome/status/finish value.

    An ``error`` payload is a failure signal, but it has no source outcome
    spelling of its own.  Callers that need the existing semantic
    ``explicit_failure`` outcome use this distinction while child mapping can
    still expose ``error`` as a status signal.
    """
    if not isinstance(decoded, Mapping):
        return None
    lifecycle_error = decoded.get("lifecycle_error_outcome")
    if isinstance(lifecycle_error, str) and lifecycle_error:
        return lifecycle_error
    values = [decoded.get(name) for name in ("outcome", "status", "finish")]
    strings = [value for value in values if isinstance(value, str) and value]
    for value in strings:
        if classify_outcome(value) == "ERROR":
            return value
    return strings[0] if strings else None


def is_explicit_failure(decoded: object) -> bool:
    """Whether a decoded object carries an explicit failure signal."""
    if not isinstance(decoded, Mapping):
        return False
    return decoded.get("error") is not None or classify_outcome(decoded_source_outcome(decoded)) == "ERROR"


def classify_decoded(decoded: object) -> str:
    """Classify a decoded message/opaque payload without guessing unknown values."""
    return classify_outcome(decoded_source_outcome(decoded))


def tool_source_outcome(decoded: object) -> str | None:
    """Return the exact outcome from a tool state, falling back to its envelope."""
    if isinstance(decoded, Mapping):
        state = decoded.get("state")
        state_outcome = decoded_source_outcome(state)
        if state_outcome is not None:
            return state_outcome
    return decoded_source_outcome(decoded)


def record_source_outcome(
    decoded: object,
    source_type: str | None = None,
    raw_columns: Mapping[str, object] | None = None,
) -> str | None:
    """Return a decoded/row outcome or an explicit failure record type."""
    outcome = decoded_source_outcome(decoded)
    if outcome is not None:
        return outcome
    if raw_columns is not None:
        lifecycle_error = raw_columns.get("lifecycle_error_outcome")
        if isinstance(lifecycle_error, str) and lifecycle_error:
            return lifecycle_error
        for name in ("outcome", "status", "finish"):
            value = raw_columns.get(name)
            if isinstance(value, str) and value:
                return value
    if isinstance(source_type, str) and classify_outcome(source_type) == "ERROR":
        return source_type
    return None
