"""Normalized session selection filters.

:func:`select_session` evaluates one :class:`SessionDiscoverySummary`
against one :class:`SelectionFilters` instance. The semantics pinned here
come from the design spec's session filter table:

- **Time**: inclusive bounds on the session's ``time_created`` only —
  never a message timestamp and never ``time_updated``. A session whose
  creation time is missing is ineligible whenever any time filter is
  supplied; without a time filter it stays selectable.
- **Directory**: path-component equality/descendant matching. A session
  in ``/repo/project/app`` matches a requested ``/repo/project``; a
  session in ``/repo/projector`` never does, because matching compares
  whole path components, never string prefixes.
- **Model**: exact identity comparison after whitespace normalization.
  Incomplete forms keep their explicit shape (``provider/`` matches only
  a provider-only request, ``/model`` only a model-only request), and a
  complete ``provider/model`` request never matches an incomplete source
  identity. The importer never invents a missing provider or model
  component.
- **Minimum turns**: a session eligibility threshold — strictly more
  than ``N`` logical user turns. ``None`` (the option omitted) imposes
  no restriction, so zero-turn sessions stay selectable for stale-trace
  retirement; an explicit ``0`` means "more than zero turns" and
  excludes them. An eligible session always imports every logical turn.

Every supplied filter combines with logical AND; an omitted filter
imposes no restriction. Filters are pure functions over the
selection-facing summary — they never touch source records.
"""

from __future__ import annotations

import os
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from opencode_mlflow.config import normalize_model_identity
from opencode_mlflow.source.model import SessionHeader, SourceRecord
from opencode_mlflow.source.model_identity import extract_model_identity

_MODEL_EVIDENCE_TABLES = frozenset({"message", "part", "session_message"})

__all__ = [
    "SelectionEvidence",
    "SelectionEvidenceEvaluator",
    "SelectionFilters",
    "SessionDiscoverySummary",
    "matches_directory",
    "matches_model",
    "select_session",
]


@dataclass(frozen=True)
class SelectionEvidence:
    """Pre-redaction directory/model match results stored in the snapshot."""

    directory_match: bool = True
    model_match: bool = True


@dataclass(frozen=True)
class SelectionEvidenceEvaluator:
    """Evaluate invocation-specific directory/model filters before redaction."""

    filters: SelectionFilters

    def session_evidence(self, header: SessionHeader) -> SelectionEvidence:
        directory_match = self.filters.directory is None or (
            header.directory is not None and matches_directory(Path(header.directory), self.filters.directory)
        )
        identity = extract_model_identity(header.model)
        model_match = self.filters.model is None or (
            identity is not None and matches_model((identity,), self.filters.model)
        )
        return SelectionEvidence(directory_match=directory_match, model_match=model_match)

    def record_model_matches(self, record: SourceRecord) -> bool:
        """Whether one pre-redaction record satisfies the requested model."""
        if self.filters.model is None:
            return True
        if record.source_table not in _MODEL_EVIDENCE_TABLES:
            return False
        identity = extract_model_identity(record.decoded_value)
        return identity is not None and matches_model((identity,), self.filters.model)


@dataclass(frozen=True)
class SelectionFilters:
    """Immutable, normalized session selection filters.

    ``from_time``/``to_time`` are timezone-aware UTC instants applied as
    inclusive bounds on the session creation time, ``directory`` is an
    absolute path matched by component equality/descendance, ``model``
    is a whitespace-trimmed identity compared exactly (including
    incomplete ``provider/`` and ``/model`` forms), and
    ``min_user_turns`` is either ``None`` (no restriction) or a
    non-negative threshold applied as a strict greater-than. Every
    supplied filter must pass.
    """

    from_time: datetime | None = None
    to_time: datetime | None = None
    directory: Path | None = None
    model: str | None = None
    min_user_turns: int | None = None


@dataclass(frozen=True)
class SessionDiscoverySummary:
    """The selection-facing summary of one scanned session.

    The summary carries exactly what filtering needs — the session
    identity, the creation instant (the only time field selection ever
    consults), the normalized directory, the model identities found
    throughout the session, the logical user-turn count, safe warning
    codes, and whether the session's scan completed — and never carries
    prompt, response, tool, or source-code content.
    """

    session_id: str
    creation_time: datetime | None
    directory: Path | None
    model_identities: frozenset[str]
    logical_turn_count: int
    warnings: tuple[str, ...] = ()
    complete_scan: bool = True


def matches_directory(candidate: Path, requested: Path) -> bool:
    """Whether ``candidate`` is ``requested`` or a component descendant.

    Matching compares whole path components: the candidate matches when
    its leading components equal the requested path's components, so
    ``/repo/project/app`` matches ``/repo/project`` while the sibling
    ``/repo/projector`` does not. A string-prefix match is never used.
    An empty requested path matches nothing.
    """
    normalized_requested = _lexical_path(requested)
    normalized_candidate = _lexical_path(candidate)
    if normalized_requested.is_absolute() != normalized_candidate.is_absolute():
        return False
    requested_parts = normalized_requested.parts
    if not requested_parts:
        return False
    return normalized_candidate.parts[: len(requested_parts)] == requested_parts


def _lexical_path(value: Path) -> Path:
    """Normalize path syntax without rebasing relative source values."""
    return Path(os.path.normpath(os.path.expanduser(str(value))))


def matches_model(available: Collection[str], requested: str | None) -> bool:
    """Whether a requested model identity matches any available identity.

    Both sides are normalized by trimming surrounding whitespace (case
    and completeness preserved) and compared for exact equality, so an
    incomplete source identity such as ``provider/`` matches only a
    request in the same available form and a complete
    ``provider/model`` request never matches it. ``requested=None``
    imposes no restriction; a blank requested identity matches nothing.
    """
    if requested is None:
        return True
    wanted = normalize_model_identity(requested)
    if not wanted:
        return False
    return any(normalize_model_identity(identity) == wanted for identity in available)


def select_session(
    summary: SessionDiscoverySummary,
    filters: SelectionFilters,
    evidence: SelectionEvidence | None = None,
) -> bool:
    """Whether ``summary`` passes every supplied filter (logical AND).

    An omitted filter imposes no restriction. A session with a missing
    creation time is ineligible whenever any time filter is supplied.
    The minimum-turn threshold is a session eligibility rule: an
    eligible session imports every logical turn, never only the turns
    beyond the threshold.
    """
    if not _matches_creation_time(summary.creation_time, filters):
        return False
    if filters.directory is not None:
        directory_match = (
            evidence.directory_match
            if evidence is not None
            else summary.directory is not None and matches_directory(summary.directory, filters.directory)
        )
        if not directory_match:
            return False
    model_match = (
        evidence.model_match if evidence is not None else matches_model(summary.model_identities, filters.model)
    )
    if filters.model is not None and not model_match:
        return False
    if filters.min_user_turns is not None and summary.logical_turn_count <= filters.min_user_turns:
        return False
    return True


def _matches_creation_time(creation_time: datetime | None, filters: SelectionFilters) -> bool:
    """Apply inclusive creation-time bounds; missing times stay eligible without them."""
    if filters.from_time is None and filters.to_time is None:
        return True
    if creation_time is None:
        return False
    if filters.from_time is not None and creation_time < filters.from_time:
        return False
    if filters.to_time is not None and creation_time > filters.to_time:
        return False
    return True
