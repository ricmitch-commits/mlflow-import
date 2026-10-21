"""Ownership reconciliation: plan importer-owned traces, then apply safely.

Two phases with a strict boundary:

- :func:`plan_reconciliation` is a PURE planner. It groups the owned
  candidates a caller indexed (through
  :meth:`~opencode_mlflow.mlflow.gateway.MlflowGateway.iter_owned_traces`)
  by the ``(session_id, turn_key)`` ownership key — never the native
  MLflow trace id — and computes provisional actions: ``imported``
  (no owned candidate), ``updated`` (candidates exist; the final
  unchanged-versus-replace decision waits for verification), and
  ``retired`` (stale turns of SELECTED complete scans, including every
  previously owned turn of a selected zero-turn tombstone session). It
  issues no gateway calls, verifies nothing, and never selects a winner
  from unverified candidates. Owned traces outside the reconciliation
  scope (unselected sessions, unknown sessions, incomplete scans) are
  reported as ``unexamined_owned_traces`` — a filtered run never
  retires an unselected session's traces, and unowned traces sharing
  native session ids are never candidates at all.

- :func:`apply_reconciliation` is the effectful phase. FIRST it obtains
  EVERY :class:`CandidateVerification` — through
  :meth:`~opencode_mlflow.mlflow.gateway.MlflowGateway.verify_owned_candidate`
  with a :class:`CandidateContract` built from the mapped spec (public
  full-payload retrieval plus named-artifact verification, never search
  previews) — and only THEN calls the internal
  :func:`select_verified_winner` step: no create, retire, or winner
  decision occurs before the verification phase completes. The winner
  among verified exact matches is the newest creation timestamp, ties
  broken by the lexicographically greatest MLflow trace id. If none
  match, one replacement is created and verified through
  :class:`ReconciliationDependencies` in the strict order fresh
  ``ReplayTurn`` → ``map_trace`` → create → verify → upload → verify →
  record the verified artifact digest on the trace,
  and only then are the old and redundant owned candidates retired. A
  failed verification/upload rolls the NEW trace back and retains the
  old valid trace; a failed rollback reports both trace ids as a
  reconciliation conflict; an old-trace retirement failure after a
  verified replacement attempts bounded compensating retirement of the
  new trace and never deletes the old one on uncertainty. Only
  duplicate identities that cannot be verified or deterministically
  resolved become ``duplicate_owned_identity`` conflicts.

Fingerprint-mode context (Ruling 12): the ownership fingerprints the
contract compares are RUN-MODE-TAGGED and derive from the mapped
:class:`~opencode_mlflow.mlflow.gateway.TraceCreateRequest` — what the
new trace would carry — never from the discovery plan's turn-entry
fingerprint values (those are Task 12's replay-consistency values,
hashed under the fixed verbatim tag).

Recorded artifact digests (Ruling 16): the artifact document contains
the created trace's generated identity map, so its bytes are not
deterministic across runs and cannot be compared against a re-derived
"would-be" digest. Instead, after an artifact upload verifies, the
write path RECORDS the uploaded digest and size on the trace itself
(``opencode.artifact_sha256``/``opencode.artifact_size`` tags through
the gateway's public
:meth:`~opencode_mlflow.mlflow.gateway.MlflowGateway.record_artifact_verification`),
and the mapped request RECORDS the artifact's exact relative path at
trace creation (``opencode.artifact_path`` trace tag). Candidate
verification reads the candidate's OWN recorded path and digest from
full trace retrieval and checks the stored artifact's INTEGRITY at that
location — the path is deliberately NOT an equivalence field, because
it embeds the session-complete artifact-snapshot fingerprint and would
move for every sibling turn whenever any turn of the session changed;
equivalence to the current run stays in the complete creation-time
ownership contract compared against the current mapped spec. A trace
without a usable recorded path (written before the tag existed) or
digest (a crash between upload and recording) is an unverifiable
incomplete candidate that never verifies as a match — it is replaced
and retired after a verified winner exists, so unchanged reruns over
search-indexed candidates converge with zero mutations.

Retries: every gateway call runs under the caller's
:class:`~opencode_mlflow.retry.RetryPolicy`; creation uses the
deterministic ``opencode.operation_key`` idempotency key carried by the
mapped request and, after an ambiguous timeout, searches for operation-key
survivors BEFORE retrying the create, verifies the complete
owner/session/turn/render tuple, and records every surviving importer-owned
candidate for deterministic winner selection and explicit operator recovery.
"""

from __future__ import annotations

import random
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from opencode_mlflow.content.artifacts import (
    DEFAULT_UPLOAD_CHUNK_BYTES,
    stream_turn_artifact,
    temporary_artifact_file,
)
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.errors import ImporterError, ItemImportError, PartialTraceCreateError
from opencode_mlflow.retry import RetryPolicy, is_transient_error, retry_call
from opencode_mlflow.source.discovery import DiscoveryTurnEntry, SessionScanEntry
from opencode_mlflow.source.replay import ReplayTurn

if TYPE_CHECKING:
    from opencode_mlflow.mlflow.gateway import (
        ArtifactVerification,
        CreatedTrace,
        MlflowGateway,
        OwnedTrace,
        TraceCreateRequest,
    )

__all__ = [
    "ACTION_FAILED",
    "ACTION_IMPORTED",
    "ACTION_RETIRED",
    "ACTION_SKIPPED",
    "ACTION_UPDATED",
    "ARTIFACT_PATH_TAG",
    "ARTIFACT_SHA256_TAG",
    "ARTIFACT_SIZE_TAG",
    "CandidateContract",
    "CandidateVerification",
    "INVALID_WARNING_METADATA",
    "OwnedTraceCandidate",
    "OwnershipKey",
    "REASON_SESSION_TOMBSTONE",
    "REASON_STALE_TURN",
    "REASON_UNCHANGED",
    "ReconciliationAction",
    "ReconciliationDependencies",
    "ReconciliationPlan",
    "apply_reconciliation",
    "created_artifact_identity_map",
    "plan_reconciliation",
    "select_verified_winner",
    "upload_artifact",
]

# ---------------------------------------------------------------------------
# Ownership metadata keys (the namespaced fields the mapper writes at trace
# creation; a contract test pins them against the real mapper's output)
# ---------------------------------------------------------------------------

OWNERSHIP_SESSION_KEY = "opencode.session_id"
OWNERSHIP_IMPORTER_KEY = "opencode.importer"
OWNERSHIP_TURN_KEY = "opencode.turn_key"
OWNERSHIP_TURN_INDEX_KEY = "opencode.turn_index"
OWNERSHIP_SOURCE_FINGERPRINT_KEY = "opencode.source_fingerprint"
OWNERSHIP_SOURCE_SCHEMA_KEY = "opencode.source_schema"
OWNERSHIP_IMPORTER_VERSION_KEY = "opencode.importer_version"
OWNERSHIP_MAPPING_VERSION_KEY = "opencode.mapping_version"
OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY = "opencode.artifact_schema_version"
OWNERSHIP_REDACTION_MODE_KEY = "opencode.redaction_mode"
OWNERSHIP_CATALOG_VERSION_KEY = "opencode.catalog_version"
OWNERSHIP_RENDER_FINGERPRINT_KEY = "opencode.render_fingerprint"
OWNERSHIP_OPERATION_KEY = "opencode.operation_key"
OWNERSHIP_IMPORTER_VALUE = "opencode-mlflow"
OWNERSHIP_METADATA_VALUE_LIMIT = 250

INVALID_WARNING_METADATA = "invalid_warning_metadata"
"""Safe replacement code for malformed candidate warning metadata."""

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_TURN_KEY_RE = re.compile(r"^turn-[0-9a-f]{64}$")
_WARNING_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SUPPORTED_SOURCE_SCHEMAS = frozenset({"legacy", "current"})

#: Trace tags the write path records after an artifact upload verifies
#: (Ruling 16): the digest of the bytes that were actually uploaded, read
#: back by candidate verification for the artifact INTEGRITY check.
ARTIFACT_SHA256_TAG = "opencode.artifact_sha256"
ARTIFACT_SIZE_TAG = "opencode.artifact_size"

#: Trace tag the MAPPED request carries at root-span creation: the exact
#: relative path the turn's artifact was written to. Candidate verification
#: reads this RECORDED path from full trace retrieval and checks the stored
#: artifact's integrity at the candidate's own location — the path is never
#: an equivalence field, because it embeds the session-complete
#: artifact-snapshot fingerprint and would move for every sibling whenever
#: any turn of the session changed. It rides a trace TAG (not trace
#: metadata) because the pinned server silently truncates metadata values
#: at 250 characters while the deterministic path is typically longer;
#: trace tag values round-trip exactly up to 4096 characters.
ARTIFACT_PATH_TAG = "opencode.artifact_path"

#: The normative action enum (identical to the report schema's).
ACTION_IMPORTED = "imported"
ACTION_UPDATED = "updated"
ACTION_SKIPPED = "skipped"
ACTION_FAILED = "failed"
ACTION_RETIRED = "retired"

#: Reasons a final action can carry.
REASON_UNCHANGED = "unchanged"
REASON_STALE_TURN = "stale_turn"
REASON_SESSION_TOMBSTONE = "session_tombstone"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnershipKey:
    """The lookup key of one source identity: session id plus turn key.

    Never the native MLflow trace id: a replacement gets a new trace id
    while keeping the same ownership key.
    """

    session_id: str
    turn_key: str


@dataclass(frozen=True)
class CandidateContract:
    """The expected ownership state of one identity at write time.

    The metadata fields derive from the mapped
    :class:`~opencode_mlflow.mlflow.gateway.TraceCreateRequest` — what
    the new trace would carry — under the RUN's redaction mode; every
    creation-time ownership field is part of the equivalence contract.
    The artifact fields are
    informational compatibility fields, NOT verification inputs: when
    verifying a candidate they mirror the digest recorded on that
    candidate's trace (parsed at index time), and when writing they are
    the just-uploaded artifact's digest/size.
    ``artifact_relative_path`` is the CURRENT run's path (where a NEW
    trace's artifact will be written); it is not an equivalence field —
    candidate verification reads the candidate's own recorded
    ``opencode.artifact_path`` tag from full trace retrieval and checks
    the stored artifact's INTEGRITY at that location, because the
    deterministic path embeds the session-complete artifact-snapshot
    fingerprint and moves for every sibling turn whenever any turn of
    the session changes.
    """

    session_id: str
    turn_key: str
    source_fingerprint: str
    source_schema: str
    render_fingerprint: str
    operation_key: str
    artifact_relative_path: str
    artifact_sha256: str
    artifact_size: int
    catalog_version: str = ""
    importer_marker: str | None = OWNERSHIP_IMPORTER_VALUE
    turn_index: int | None = None
    importer_version: str | None = None
    mapping_version: str | None = None
    artifact_schema_version: str | None = None
    redaction_mode: str | None = None
    experiment_id: str | None = None
    warnings: tuple[str, ...] = ()
    warning_metadata_valid: bool = True

    @classmethod
    def from_create_request(
        cls,
        request: TraceCreateRequest,
        artifact_sha256: str,
        artifact_size: int,
    ) -> CandidateContract:
        """Build the contract from a mapped request and its artifact digest."""
        metadata = request.metadata
        warnings, warning_metadata_valid = _parse_warning_metadata(metadata.get("opencode.warnings"))
        contract = cls(
            session_id=metadata.get(OWNERSHIP_SESSION_KEY, ""),
            turn_key=metadata.get(OWNERSHIP_TURN_KEY, ""),
            source_fingerprint=metadata.get(OWNERSHIP_SOURCE_FINGERPRINT_KEY, ""),
            source_schema=metadata.get(OWNERSHIP_SOURCE_SCHEMA_KEY, ""),
            catalog_version=metadata.get(OWNERSHIP_CATALOG_VERSION_KEY, ""),
            render_fingerprint=metadata.get(OWNERSHIP_RENDER_FINGERPRINT_KEY, ""),
            operation_key=metadata.get(OWNERSHIP_OPERATION_KEY, ""),
            artifact_relative_path=request.artifact_relative_path,
            artifact_sha256=artifact_sha256,
            artifact_size=artifact_size,
            importer_marker=metadata.get(OWNERSHIP_IMPORTER_KEY),
            turn_index=_parse_turn_index(metadata.get(OWNERSHIP_TURN_INDEX_KEY)),
            importer_version=metadata.get(OWNERSHIP_IMPORTER_VERSION_KEY),
            mapping_version=metadata.get(OWNERSHIP_MAPPING_VERSION_KEY),
            artifact_schema_version=metadata.get(OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY),
            redaction_mode=metadata.get(OWNERSHIP_REDACTION_MODE_KEY),
            experiment_id=request.experiment_id,
            warnings=warnings,
            warning_metadata_valid=warning_metadata_valid,
        )
        if contract.ownership_errors:
            raise ItemImportError(
                "ownership_contract_invalid",
                "the mapped trace request does not carry a complete ownership contract",
            )
        return contract

    @property
    def ownership_key(self) -> OwnershipKey:
        """The contract's source identity."""
        return OwnershipKey(self.session_id, self.turn_key)

    @property
    def ownership_errors(self) -> tuple[str, ...]:
        """Validate every creation-time field before it authorizes cleanup."""
        return _ownership_contract_errors(
            trace_id="contract",
            experiment_id=self.experiment_id,
            importer_marker=self.importer_marker,
            session_id=self.session_id,
            turn_key=self.turn_key,
            turn_index=self.turn_index,
            source_fingerprint=self.source_fingerprint,
            source_schema=self.source_schema,
            importer_version=self.importer_version,
            mapping_version=self.mapping_version,
            artifact_schema_version=self.artifact_schema_version,
            redaction_mode=self.redaction_mode,
            catalog_version=self.catalog_version,
            render_fingerprint=self.render_fingerprint,
            operation_key=self.operation_key,
            warning_metadata_valid=self.warning_metadata_valid,
        )

    @property
    def creation_metadata(self) -> dict[str, str]:
        """The exact ownership metadata used for a final retirement check."""
        values = {
            OWNERSHIP_IMPORTER_KEY: self.importer_marker or "",
            OWNERSHIP_SESSION_KEY: self.session_id,
            OWNERSHIP_TURN_INDEX_KEY: str(self.turn_index) if self.turn_index is not None else "",
            OWNERSHIP_TURN_KEY: self.turn_key,
            OWNERSHIP_SOURCE_FINGERPRINT_KEY: self.source_fingerprint,
            OWNERSHIP_SOURCE_SCHEMA_KEY: self.source_schema,
            OWNERSHIP_IMPORTER_VERSION_KEY: self.importer_version or "",
            OWNERSHIP_MAPPING_VERSION_KEY: self.mapping_version or "",
            OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY: self.artifact_schema_version or "",
            OWNERSHIP_REDACTION_MODE_KEY: self.redaction_mode or "",
            OWNERSHIP_CATALOG_VERSION_KEY: self.catalog_version,
            OWNERSHIP_RENDER_FINGERPRINT_KEY: self.render_fingerprint,
            OWNERSHIP_OPERATION_KEY: self.operation_key,
            "opencode.warnings": ",".join(self.warnings),
        }
        return values


@dataclass(frozen=True)
class OwnedTraceCandidate:
    """One importer-owned trace indexed for reconciliation.

    ``artifact_sha256``/``artifact_size`` are the artifact digest RECORDED on
    the trace (the ``opencode.artifact_sha256``/``opencode.artifact_size``
    tags the write path sets after an upload verifies), parsed at index time
    from the search preview; ``None`` when the trace carries none — an
    incomplete candidate left by a crash between upload and recording.
    ``request_time_ms`` is the creation timestamp winner selection compares.
    """

    trace_id: str
    session_id: str
    turn_key: str
    request_time_ms: int | None
    experiment_id: str | None = None
    artifact_sha256: str | None = None
    artifact_size: int | None = None
    source_fingerprint: str | None = None
    source_schema: str | None = None
    catalog_version: str | None = None
    render_fingerprint: str | None = None
    operation_key: str | None = None
    turn_index: int | None = None
    warnings: tuple[str, ...] = ()
    warning_metadata_valid: bool = True
    importer_version: str | None = None
    mapping_version: str | None = None
    artifact_schema_version: str | None = None
    redaction_mode: str | None = None
    importer_marker: str | None = None

    def __post_init__(self) -> None:
        """Keep warning values safe even for manually constructed candidates."""
        parsed: tuple[str, ...]
        valid: bool
        if any(not isinstance(value, str) for value in self.warnings):
            parsed, valid = (INVALID_WARNING_METADATA,), False
        else:
            raw = ",".join(self.warnings)
            parsed, valid = _parse_warning_metadata(raw)
        if not self.warning_metadata_valid or not valid:
            parsed = (INVALID_WARNING_METADATA,)
            valid = False
        object.__setattr__(self, "warnings", parsed)
        object.__setattr__(self, "warning_metadata_valid", valid)

    @classmethod
    def from_owned_trace(
        cls,
        owned: OwnedTrace,
        experiment_id: str | None = None,
    ) -> OwnedTraceCandidate:
        """Build a candidate from one ``iter_owned_traces`` search preview.

        The identity comes from the preview's ownership metadata; the
        recorded artifact digest comes from the preview's tags. A preview
        without a parseable identity yields ``None`` session/turn keys, which
        the planner reports as unexamined.
        """
        metadata = owned.metadata
        tags = owned.tags
        size_text = tags.get(ARTIFACT_SIZE_TAG)
        turn_index_text = metadata.get(OWNERSHIP_TURN_INDEX_KEY)
        turn_index = _parse_turn_index(turn_index_text)
        warnings, warning_metadata_valid = _parse_warning_metadata(metadata.get("opencode.warnings"))
        marker = _candidate_importer_marker(tags, metadata)
        return cls(
            trace_id=owned.trace_id,
            session_id=metadata.get(OWNERSHIP_SESSION_KEY, ""),
            turn_key=metadata.get(OWNERSHIP_TURN_KEY, ""),
            request_time_ms=owned.request_time_ms,
            turn_index=turn_index,
            experiment_id=experiment_id,
            artifact_sha256=tags.get(ARTIFACT_SHA256_TAG) or None,
            artifact_size=int(size_text) if isinstance(size_text, str) and size_text.isdigit() else None,
            source_fingerprint=metadata.get(OWNERSHIP_SOURCE_FINGERPRINT_KEY),
            source_schema=metadata.get(OWNERSHIP_SOURCE_SCHEMA_KEY),
            importer_version=metadata.get(OWNERSHIP_IMPORTER_VERSION_KEY),
            mapping_version=metadata.get(OWNERSHIP_MAPPING_VERSION_KEY),
            artifact_schema_version=metadata.get(OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY),
            redaction_mode=metadata.get(OWNERSHIP_REDACTION_MODE_KEY),
            catalog_version=metadata.get(OWNERSHIP_CATALOG_VERSION_KEY),
            render_fingerprint=metadata.get(OWNERSHIP_RENDER_FINGERPRINT_KEY),
            operation_key=metadata.get(OWNERSHIP_OPERATION_KEY),
            warnings=warnings,
            warning_metadata_valid=warning_metadata_valid,
            importer_marker=marker,
        )

    @property
    def missing_ownership_fields(self) -> tuple[str, ...]:
        """Return required ownership/source fields absent or malformed on this candidate.

        Search previews are untrusted indexes.  A candidate is cleanup-safe
        only when the complete creation-time contract and the exact target
        identity are present; a marker plus a session/turn pair is not enough.
        """
        return _ownership_contract_errors(
            trace_id=self.trace_id,
            experiment_id=self.experiment_id,
            importer_marker=self.importer_marker,
            session_id=self.session_id,
            turn_key=self.turn_key,
            turn_index=self.turn_index,
            source_fingerprint=self.source_fingerprint,
            source_schema=self.source_schema,
            importer_version=self.importer_version,
            mapping_version=self.mapping_version,
            artifact_schema_version=self.artifact_schema_version,
            redaction_mode=self.redaction_mode,
            catalog_version=self.catalog_version,
            render_fingerprint=self.render_fingerprint,
            operation_key=self.operation_key,
            warning_metadata_valid=self.warning_metadata_valid,
        )

    @property
    def ownership_complete(self) -> bool:
        """Whether this candidate may participate in destructive cleanup."""
        return not self.missing_ownership_fields

    @property
    def ownership_key(self) -> OwnershipKey:
        """The candidate's source identity."""
        return OwnershipKey(self.session_id, self.turn_key)


def _candidate_importer_marker(tags: Mapping[str, str], metadata: Mapping[str, str]) -> str | None:
    """Read one exact marker, rejecting conflicting tag/metadata copies."""
    values = [mapping[OWNERSHIP_IMPORTER_KEY] for mapping in (tags, metadata) if OWNERSHIP_IMPORTER_KEY in mapping]
    if not values or any(value != values[0] for value in values):
        return None
    return values[0]


def _parse_warning_metadata(value: object) -> tuple[tuple[str, ...], bool]:
    """Parse only the documented lowercase ASCII warning-code syntax."""
    if value is None or value == "":
        return (), True
    if not isinstance(value, str):
        return (INVALID_WARNING_METADATA,), False
    if len(value) > OWNERSHIP_METADATA_VALUE_LIMIT:
        return (INVALID_WARNING_METADATA,), False
    codes = tuple(dict.fromkeys(value.split(",")))
    if not codes or any(_WARNING_CODE_RE.fullmatch(code) is None for code in codes):
        return (INVALID_WARNING_METADATA,), False
    return codes, True


def _parse_turn_index(value: object) -> int | None:
    """Parse the canonical positive decimal turn-index representation."""
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        return None
    return int(value)


def _parseable_candidate_identity(candidate: OwnedTraceCandidate) -> OwnershipKey | None:
    """Return a candidate identity even when its remaining contract is malformed."""
    if not isinstance(candidate.session_id, str) or not candidate.session_id.strip():
        return None
    if not isinstance(candidate.turn_key, str) or _TURN_KEY_RE.fullmatch(candidate.turn_key) is None:
        return None
    return OwnershipKey(candidate.session_id, candidate.turn_key)


def _ownership_contract_errors(
    *,
    trace_id: object,
    experiment_id: object,
    importer_marker: object,
    session_id: object,
    turn_key: object,
    turn_index: object,
    source_fingerprint: object,
    source_schema: object,
    importer_version: object,
    mapping_version: object,
    artifact_schema_version: object,
    redaction_mode: object,
    catalog_version: object,
    render_fingerprint: object,
    operation_key: object,
    warning_metadata_valid: bool = True,
) -> tuple[str, ...]:
    """Validate the complete production ownership contract.

    This validator is shared by search-preview candidates and the gateway's
    final pre-delete fetch.  A preview is never cleanup authority merely
    because it carries the importer marker and a source identity.
    """
    missing: list[str] = []
    if not isinstance(trace_id, str) or trace_id.strip() == "":
        missing.append("trace_id")
    if not isinstance(experiment_id, str) or experiment_id.strip() == "":
        missing.append("experiment_id")
    if importer_marker != OWNERSHIP_IMPORTER_VALUE:
        missing.append(OWNERSHIP_IMPORTER_KEY)
    if not isinstance(session_id, str) or session_id.strip() == "":
        missing.append(OWNERSHIP_SESSION_KEY)
    elif len(session_id) > OWNERSHIP_METADATA_VALUE_LIMIT:
        missing.append(OWNERSHIP_SESSION_KEY)
    if not isinstance(turn_key, str) or _TURN_KEY_RE.fullmatch(turn_key) is None:
        missing.append(OWNERSHIP_TURN_KEY)
    elif len(turn_key) > OWNERSHIP_METADATA_VALUE_LIMIT:
        missing.append(OWNERSHIP_TURN_KEY)
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 1:
        missing.append(OWNERSHIP_TURN_INDEX_KEY)
    elif len(str(turn_index)) > OWNERSHIP_METADATA_VALUE_LIMIT:
        missing.append(OWNERSHIP_TURN_INDEX_KEY)
    for name, value in (
        (OWNERSHIP_SOURCE_FINGERPRINT_KEY, source_fingerprint),
        (OWNERSHIP_RENDER_FINGERPRINT_KEY, render_fingerprint),
        (OWNERSHIP_OPERATION_KEY, operation_key),
    ):
        if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
            missing.append(name)
        elif len(value) > OWNERSHIP_METADATA_VALUE_LIMIT:
            missing.append(name)
    if not isinstance(source_schema, str) or source_schema not in _SUPPORTED_SOURCE_SCHEMAS:
        missing.append(OWNERSHIP_SOURCE_SCHEMA_KEY)
    for name, value in (
        (OWNERSHIP_IMPORTER_VERSION_KEY, importer_version),
        (OWNERSHIP_MAPPING_VERSION_KEY, mapping_version),
        (OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY, artifact_schema_version),
        (OWNERSHIP_CATALOG_VERSION_KEY, catalog_version),
    ):
        if not isinstance(value, str) or value.strip() == "":
            missing.append(name)
        elif len(value) > OWNERSHIP_METADATA_VALUE_LIMIT:
            missing.append(name)
    if redaction_mode not in {"verbatim", "redacted"}:
        missing.append(OWNERSHIP_REDACTION_MODE_KEY)
    if not warning_metadata_valid:
        missing.append("opencode.warnings")
    return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class CandidateVerification:
    """The outcome of verifying one candidate against the contract.

    ``matches`` is true only when the ownership metadata, every
    fingerprint, and the named artifact's exact bytes matched. A
    completed-but-non-matching verification carries a ``reason`` (the
    candidate is stale or a crash leftover and may be replaced). A
    verification that could not COMPLETE carries ``error_code`` (the
    candidate could not be retrieved or compared; the identity is not
    deterministically resolvable). An ownership marker that is absent or
    changed is an ``ownership_conflict`` error, never an ordinary mismatch.
    """

    trace_id: str
    matches: bool = False
    reason: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class ReconciliationAction:
    """One planned or applied reconciliation outcome.

    ``trace`` is the owned candidate the action concerns (the winner
    for a skip, the retired trace for a retirement); ``turn`` is the
     discovery turn entry for turn-level actions; ``candidates`` carries
     the identity's candidates for an update. ``affected_trace_ids``
     lists the traces an error concerns (cleanup failures, conflicts);
     ``retired_trace_ids`` lists the traces this action retired
     successfully; ``conflicting_trace_ids`` lists importer-owned traces
     that remain unresolved after an ownership-safety conflict.
    """

    action: str
    turn: DiscoveryTurnEntry | None = None
    trace: OwnedTraceCandidate | None = None
    candidates: tuple[OwnedTraceCandidate, ...] = ()
    trace_id: str | None = None
    reason: str | None = None
    error_code: str | None = None
    affected_trace_ids: tuple[str, ...] = ()
    retired_trace_ids: tuple[str, ...] = ()
    conflicting_trace_ids: tuple[str, ...] = ()


@dataclass
class ReconciliationPlan:
    """The pure planner's output plus the effectful phase's records.

    ``imports``/``updates``/``retirements`` are the provisional actions
    (an ``updated`` action's final unchanged-versus-replace decision
    waits for verification). ``verifications`` (keyed by trace id) and most
    ``results`` are filled by :func:`apply_reconciliation`, which stores every
    candidate verification in the plan before any winner selection. The
    planner may pre-populate a failed ownership-conflict result when a
    malformed candidate collides with a selected turn; that identity then has
    no provisional write action.
    """

    imports: list[ReconciliationAction] = field(default_factory=list)
    updates: list[ReconciliationAction] = field(default_factory=list)
    retirements: list[ReconciliationAction] = field(default_factory=list)
    unexamined_owned_traces: tuple[OwnedTraceCandidate, ...] = ()
    verifications: dict[str, CandidateVerification] = field(default_factory=dict)
    results: list[ReconciliationAction] = field(default_factory=list)


@dataclass(frozen=True)
class ReconciliationDependencies:
    """The write-phase seams reconciliation consumes.

    ``replay`` builds a fresh :class:`~opencode_mlflow.source.replay.ReplayTurn`
    for one plan entry; ``map_trace`` maps it to the exact
    :class:`~opencode_mlflow.mlflow.gateway.TraceCreateRequest`;
    ``artifact_snapshot`` returns the replayed turn's pre-identity-map
    artifact-snapshot fingerprint (the replay-consistency seam);
    ``upload_artifact`` streams and uploads the turn's artifact for a
    created trace (see :func:`upload_artifact` for the default
    implementation). The default artifact uploader is gateway-bound by
    the caller (Task 12) through ``functools.partial``.
    """

    replay: Callable[[DiscoveryTurnEntry], ReplayTurn]
    map_trace: Callable[[ReplayTurn], TraceCreateRequest]
    artifact_snapshot: Callable[[ReplayTurn], str]
    upload_artifact: Callable[[ReplayTurn, CreatedTrace, str, RedactionMode], ArtifactVerification]
    redaction_mode: RedactionMode = RedactionMode.VERBATIM


# ---------------------------------------------------------------------------
# The pure planner
# ---------------------------------------------------------------------------


def plan_reconciliation(
    session_scans: Iterable[SessionScanEntry],
    turn_entries: Iterable[DiscoveryTurnEntry],
    owned_traces: Iterable[OwnedTraceCandidate],
) -> ReconciliationPlan:
    """Group owned candidates by source identity and compute provisional actions.

    Pure: consumes each iterable once, issues no gateway calls, and
    never selects a winner from unverified candidates. Stale-turn
    retirement is scoped to SELECTED sessions whose scan completed (a
    filtered run never retires an unselected session's traces); a
    selected zero-turn tombstone session retires every previously owned
    turn. Owned traces for unselected, unknown, or incompletely scanned
    sessions — and candidates without a parseable identity — are
    reported as ``unexamined_owned_traces``.
    """
    scans: dict[str, SessionScanEntry] = {}
    for scan in session_scans:
        if scan.session_id in scans:
            raise ItemImportError(
                "reconciliation_plan_invalid",
                "the discovery plan contains a duplicate session scan",
            )
        scans[scan.session_id] = scan

    turns: dict[OwnershipKey, DiscoveryTurnEntry] = {}
    for entry in turn_entries:
        key = OwnershipKey(entry.session_id, entry.turn_key)
        if key in turns:
            raise ItemImportError(
                "reconciliation_plan_invalid",
                "the discovery plan contains a duplicate turn identity",
            )
        turns[key] = entry

    candidates: dict[OwnershipKey, list[OwnedTraceCandidate]] = {}
    malformed_candidates: dict[OwnershipKey, list[OwnedTraceCandidate]] = {}
    unexamined: list[OwnedTraceCandidate] = []
    for candidate in owned_traces:
        if not candidate.ownership_complete:
            # A partially populated search preview is not an owned/retirable
            # candidate. Keep it visible for operator review, but never let it
            # enter update or stale-retirement cleanup. If its source identity
            # is still parseable and collides with a selected turn, record that
            # collision below so the missing candidate cannot turn into a fresh
            # duplicate import.
            unexamined.append(candidate)
            candidate_key = _parseable_candidate_identity(candidate)
            candidate_scan = scans.get(candidate_key.session_id) if candidate_key is not None else None
            if (
                candidate_key is not None
                and candidate_scan is not None
                and candidate_scan.selected
                and candidate_key in turns
            ):
                malformed_candidates.setdefault(candidate_key, []).append(candidate)
            continue
        if not candidate.session_id or not candidate.turn_key:
            unexamined.append(candidate)
            continue
        found = scans.get(candidate.session_id)
        if found is None or not found.selected:
            unexamined.append(candidate)
            continue
        candidates.setdefault(OwnershipKey(candidate.session_id, candidate.turn_key), []).append(candidate)

    plan = ReconciliationPlan()
    for key, entry in turns.items():
        malformed = tuple(malformed_candidates.get(key, ()))
        if malformed:
            # A malformed preview with a parseable source identity is an
            # ownership conflict, not evidence that the turn is unowned. Do
            # not plan an update/import for this identity: the trace remains
            # unexamined and is preserved for operator recovery.
            complete = tuple(candidates.get(key, ()))
            conflicting = tuple(dict.fromkeys((*complete, *malformed)))
            trace_ids = tuple(candidate.trace_id for candidate in conflicting)
            plan.results.append(
                ReconciliationAction(
                    action=ACTION_FAILED,
                    turn=entry,
                    candidates=conflicting,
                    trace_id=trace_ids[0] if len(trace_ids) == 1 else None,
                    error_code="ownership_conflict",
                    affected_trace_ids=trace_ids,
                    conflicting_trace_ids=trace_ids,
                )
            )
            continue
        identity_candidates = tuple(candidates.get(key, ()))
        if identity_candidates:
            plan.updates.append(ReconciliationAction(action=ACTION_UPDATED, turn=entry, candidates=identity_candidates))
        else:
            plan.imports.append(ReconciliationAction(action=ACTION_IMPORTED, turn=entry))

    for key, key_candidates in candidates.items():
        if key in turns:
            continue
        scan = scans[key.session_id]
        if not scan.retirement_scope:
            # A selected but incomplete scan cannot prove a turn stale: its
            # turn list may be partial, so the candidates stay unexamined.
            unexamined.extend(key_candidates)
            continue
        reason = REASON_SESSION_TOMBSTONE if scan.tombstone else REASON_STALE_TURN
        for candidate in key_candidates:
            plan.retirements.append(
                ReconciliationAction(
                    action=ACTION_RETIRED,
                    trace=candidate,
                    candidates=tuple(key_candidates),
                    reason=reason,
                )
            )

    plan.unexamined_owned_traces = tuple(unexamined)
    return plan


# ---------------------------------------------------------------------------
# The default artifact upload
# ---------------------------------------------------------------------------


def created_artifact_identity_map(created: CreatedTrace) -> dict[str, str]:
    """The artifact identity-map section bound to one created trace.

    Mirrors
    :meth:`~opencode_mlflow.mlflow.mapping.MlflowIdentityMap.artifact_section`
    exactly (the generated trace id plus each child source key's span id
    under ``span:``-prefixed keys) so the artifact a created trace
    carries matches what the mapper's identity binding would persist.
    """
    section: dict[str, str] = {"trace_id": created.trace_id}
    for source_key in sorted(created.span_ids_by_source):
        section[f"span:{source_key}"] = created.span_ids_by_source[source_key]
    return section


def upload_artifact(
    replay: ReplayTurn,
    created: CreatedTrace,
    artifact_relative_path: str,
    redaction_mode: RedactionMode,
    *,
    gateway: MlflowGateway,
    temporary_directory: Path | None = None,
    chunk_size: int = DEFAULT_UPLOAD_CHUNK_BYTES,
) -> ArtifactVerification:
    """Stream one turn's artifact bound to the created identity map and upload it.

    The default ``upload_artifact`` dependency: the document is streamed
    section by section into a restrictive (``0o600``) temporary file
    with the created trace's identity map bound AFTER every fingerprint
    was fixed, uploaded through the gateway's public
    ``upload_named_json`` at exactly ``artifact_relative_path``, and
    cross-checked against the streamed manifest so the uploaded bytes
    are provably the document that was streamed. The gateway (and
    optionally the staging directory) are injected by the caller, which
    lets this function satisfy the dependency signature
    ``(replay, created, artifact_relative_path, redaction_mode)`` once
    bound through ``functools.partial``.
    """
    stage_root = temporary_directory
    if stage_root is None:
        stage_root = Path(tempfile.mkdtemp(prefix="opencode-mlflow-upload-"))
        remove_root = True
    else:
        remove_root = False
    try:
        with temporary_artifact_file(stage_root) as (staged, handle):
            provenance = created.artifact_provenance
            if provenance is not None:
                if provenance.artifact_path != artifact_relative_path:
                    raise ItemImportError(
                        "source_consistency_mismatch",
                        "the artifact path does not match the mapped trace request",
                    )
                if created.verified_tags.get(ARTIFACT_PATH_TAG) != artifact_relative_path:
                    raise ItemImportError(
                        "source_consistency_mismatch",
                        "the trace artifact path does not match the mapped request",
                    )
                expected_metadata = {
                    "opencode.source_fingerprint": provenance.source_fingerprint,
                    "opencode.render_fingerprint": provenance.render_fingerprint,
                    "opencode.source_schema": provenance.source_schema,
                    "opencode.catalog_version": provenance.catalog_version,
                    "opencode.mapping_version": provenance.mapping_version,
                    "opencode.artifact_schema_version": provenance.artifact_schema_version,
                    "opencode.redaction_mode": provenance.redaction_mode,
                }
                for key, value in expected_metadata.items():
                    if value is not None and created.verified_metadata.get(key) != value:
                        raise ItemImportError(
                            "source_consistency_mismatch",
                            "the trace provenance metadata does not match the mapped request",
                        )
            manifest = stream_turn_artifact(
                replay,
                handle,
                created_artifact_identity_map(created),
                redaction_mode,
                provenance=provenance,
            )
            handle.flush()
            verification = gateway.upload_named_json(created.trace_id, artifact_relative_path, staged, chunk_size)
    finally:
        if remove_root:
            shutil.rmtree(stage_root, ignore_errors=True)
    if (
        verification.trace_id != created.trace_id
        or verification.relative_path != artifact_relative_path
        or verification.sha256 != manifest.sha256
        or verification.size != manifest.size_bytes
    ):
        raise ItemImportError(
            "artifact_upload_failed",
            "the uploaded artifact did not match the streamed document at the requested path",
        )
    return replace(verification, snapshot_fingerprint=manifest.snapshot_fingerprint)


# ---------------------------------------------------------------------------
# The effectful phase
# ---------------------------------------------------------------------------


@dataclass
class _IdentityWork:
    """One update identity's write-phase context, built during verification."""

    key: OwnershipKey
    entry: DiscoveryTurnEntry
    redaction_mode: RedactionMode
    replay: ReplayTurn | None = None
    request: TraceCreateRequest | None = None
    error_code: str | None = None
    expected_contract: CandidateContract | None = None


@dataclass(frozen=True)
class _RetirementResult:
    """The bounded cleanup outcome, including an ownership safety conflict."""

    retired: bool
    error_code: str | None = None


class _AmbiguousCreateConflict(ItemImportError):
    """An ambiguous create left an owner that cannot be safely resolved."""

    def __init__(self, trace_ids: Sequence[str]) -> None:
        self.trace_ids = tuple(dict.fromkeys(trace_ids))
        super().__init__(
            "ownership_conflict",
            "an ambiguous trace create left an owner that could not be safely verified",
        )


def apply_reconciliation(
    plan: ReconciliationPlan,
    gateway: MlflowGateway,
    dependencies: ReconciliationDependencies,
    retry_policy: RetryPolicy,
) -> ReconciliationPlan:
    """Execute the plan: verify everything, then decide, create, and retire.

    Phase 1 obtains EVERY candidate verification (stored in
    ``plan.verifications``) before any winner selection, create, or
    retirement. Phase 2 decides each update identity (verified winner →
    skipped-unchanged; none matching → one verified replacement;
    unverifiable duplicates → conflict) and creates each import. Phase 3
    executes the stale retirements. The mutated plan is returned.
    """
    works = _verify_all_candidates(plan, gateway, dependencies, retry_policy)
    _decide_updates(plan, gateway, dependencies, retry_policy, works)
    _decide_imports(plan, gateway, dependencies, retry_policy)
    _apply_retirements(plan, gateway, retry_policy)
    return plan


def _verify_all_candidates(
    plan: ReconciliationPlan,
    gateway: MlflowGateway,
    dependencies: ReconciliationDependencies,
    retry_policy: RetryPolicy,
) -> dict[OwnershipKey, _IdentityWork]:
    """Phase 1: map every update identity and verify every one of its candidates."""
    works: dict[OwnershipKey, _IdentityWork] = {}
    for action in plan.updates:
        entry = action.turn
        if entry is None:  # pragma: no cover - the planner always attaches the turn
            continue
        key = OwnershipKey(entry.session_id, entry.turn_key)
        work = _IdentityWork(key=key, entry=entry, redaction_mode=dependencies.redaction_mode)
        try:
            work.replay = dependencies.replay(entry)
            work.request = dependencies.map_trace(work.replay)
        except ImporterError as error:
            works[key] = replace(work, replay=None, request=None, error_code=error.code)
            continue
        except Exception:
            works[key] = replace(work, replay=None, request=None, error_code="turn_replay_failed")
            continue
        assert work.replay is not None and work.request is not None
        try:
            replayed_snapshot = dependencies.artifact_snapshot(work.replay)
        except Exception:
            works[key] = replace(work, replay=None, request=None, error_code="source_consistency_mismatch")
            continue
        if replayed_snapshot != work.request.artifact_snapshot_fingerprint:
            # The replayed source no longer matches what was mapped: a
            # consistency failure, never permission to import an unplanned
            # payload.
            works[key] = replace(work, replay=None, request=None, error_code="source_consistency_mismatch")
            continue
        for candidate in action.candidates:
            plan.verifications[candidate.trace_id] = _verify_one_candidate(gateway, work, candidate, retry_policy)
        # Verify-all-before-winner does not require retaining every replay
        # header/root payload. Keep only the small contract, and remap one
        # identity at a time in the decision phase over the immutable snapshot.
        try:
            expected_contract = CandidateContract.from_create_request(work.request, "", 0)
        except ImporterError as error:
            works[key] = replace(work, replay=None, request=None, error_code=error.code)
            continue
        except Exception:
            works[key] = replace(work, replay=None, request=None, error_code="candidate_contract_failed")
            continue
        works[key] = replace(work, replay=None, request=None, expected_contract=expected_contract)
    return works


def _verify_one_candidate(
    gateway: MlflowGateway,
    work: _IdentityWork,
    candidate: OwnedTraceCandidate,
    retry_policy: RetryPolicy,
) -> CandidateVerification:
    """Verify one candidate against the contract built for it, with retries."""
    if not candidate.ownership_complete:
        return CandidateVerification(
            trace_id=candidate.trace_id,
            matches=False,
            reason="ownership_incomplete",
            error_code="ownership_conflict",
        )
    assert work.replay is not None and work.request is not None
    try:
        contract = _candidate_contract(work, candidate)
    except ImporterError as error:
        return CandidateVerification(trace_id=candidate.trace_id, matches=False, error_code=error.code)
    except Exception:
        return CandidateVerification(trace_id=candidate.trace_id, matches=False, error_code="candidate_contract_failed")
    try:
        return retry_call(
            lambda: gateway.verify_owned_candidate(candidate, contract),
            retry_policy,
        )
    except ImporterError as error:
        return CandidateVerification(trace_id=candidate.trace_id, matches=False, error_code=error.code)
    except Exception:
        return CandidateVerification(
            trace_id=candidate.trace_id, matches=False, error_code="candidate_verification_failed"
        )


def _candidate_contract(work: _IdentityWork, candidate: OwnedTraceCandidate) -> CandidateContract:
    """The expected contract for one candidate: mapped metadata plus its RECORDED digest.

    The metadata and the deterministic artifact path come from the mapped
    spec; the artifact fields mirror the digest RECORDED on the candidate's
    trace (parsed at index time — empty when the candidate is incomplete).
    The authoritative integrity check reads the recorded digest from full
    trace retrieval inside ``verify_owned_candidate``.
    """
    assert work.request is not None
    return CandidateContract.from_create_request(
        work.request,
        candidate.artifact_sha256 or "",
        candidate.artifact_size or 0,
    )


def select_verified_winner(
    verified_candidates: Sequence[tuple[OwnedTraceCandidate, CandidateVerification]],
) -> tuple[OwnedTraceCandidate, CandidateVerification]:
    """Select the winner among verified exact matches (internal step).

    The newest creation timestamp wins; ties break by the
    lexicographically greatest MLflow trace id. A missing creation
    timestamp counts as oldest.
    """
    if not verified_candidates:
        raise ValueError("a winner cannot be selected from no verified candidates")
    return max(
        verified_candidates,
        key=lambda pair: (
            pair[0].request_time_ms if pair[0].request_time_ms is not None else -1,
            pair[0].trace_id,
        ),
    )


def _decide_updates(
    plan: ReconciliationPlan,
    gateway: MlflowGateway,
    dependencies: ReconciliationDependencies,
    retry_policy: RetryPolicy,
    works: Mapping[OwnershipKey, _IdentityWork],
) -> None:
    """Phase 2a: decide every update identity after all verifications exist."""
    for action in plan.updates:
        entry = action.turn
        if entry is None:  # pragma: no cover - the planner always attaches the turn
            continue
        key = OwnershipKey(entry.session_id, entry.turn_key)
        work = works[key]
        if work.error_code is not None:
            plan.results.append(
                ReconciliationAction(
                    action=ACTION_FAILED,
                    turn=entry,
                    candidates=action.candidates,
                    trace_id=action.candidates[0].trace_id if len(action.candidates) == 1 else None,
                    error_code=work.error_code,
                    affected_trace_ids=tuple(candidate.trace_id for candidate in action.candidates),
                    conflicting_trace_ids=(
                        tuple(candidate.trace_id for candidate in action.candidates)
                        if work.error_code == "ownership_conflict"
                        else ()
                    ),
                )
            )
            continue
        try:
            work = replace(work, replay=dependencies.replay(entry))
            assert work.replay is not None
            work.request = dependencies.map_trace(work.replay)
            if CandidateContract.from_create_request(work.request, "", 0) != work.expected_contract:
                raise ItemImportError("source_consistency_mismatch", "the remapped contract differs from verification")
        except Exception as error:
            plan.results.append(
                ReconciliationAction(
                    action=ACTION_FAILED,
                    turn=entry,
                    candidates=action.candidates,
                    trace_id=action.candidates[0].trace_id if len(action.candidates) == 1 else None,
                    error_code=error.code if isinstance(error, ImporterError) else "turn_replay_failed",
                    affected_trace_ids=tuple(candidate.trace_id for candidate in action.candidates),
                    conflicting_trace_ids=(
                        tuple(candidate.trace_id for candidate in action.candidates)
                        if isinstance(error, ImporterError) and error.code == "ownership_conflict"
                        else ()
                    ),
                )
            )
            continue
        matching: list[tuple[OwnedTraceCandidate, CandidateVerification]] = []
        errored: list[tuple[OwnedTraceCandidate, CandidateVerification]] = []
        missing_verification: list[OwnedTraceCandidate] = []
        for candidate in action.candidates:
            verification = plan.verifications.get(candidate.trace_id)
            if verification is None:
                missing_verification.append(candidate)
                continue
            if verification.matches:
                matching.append((candidate, verification))
            elif verification.error_code is not None:
                errored.append((candidate, verification))
        if errored or missing_verification:
            # A matching candidate is not enough to establish ownership when
            # another candidate could not be fully retrieved or compared.
            # Treat the identity as a conflict and leave every candidate in
            # place; only completed non-matches (including known crash
            # leftovers) are eligible for cleanup.
            ownership_conflict = any(
                verification.error_code == "ownership_conflict" for _candidate, verification in errored
            )
            if len(action.candidates) > 1 and not ownership_conflict:
                plan.results.append(
                    ReconciliationAction(
                        action=ACTION_FAILED,
                        turn=entry,
                        candidates=action.candidates,
                        error_code="duplicate_owned_identity",
                        affected_trace_ids=tuple(candidate.trace_id for candidate in action.candidates),
                        conflicting_trace_ids=tuple(candidate.trace_id for candidate in action.candidates),
                    )
                )
            elif ownership_conflict:
                plan.results.append(
                    ReconciliationAction(
                        action=ACTION_FAILED,
                        turn=entry,
                        candidates=action.candidates,
                        trace_id=action.candidates[0].trace_id,
                        error_code="ownership_conflict",
                        affected_trace_ids=tuple(candidate.trace_id for candidate in action.candidates),
                        conflicting_trace_ids=tuple(candidate.trace_id for candidate in action.candidates),
                    )
                )
            else:
                error_code = (
                    "ownership_conflict"
                    if ownership_conflict
                    else errored[0][1].error_code
                    if errored
                    else "candidate_verification_failed"
                )
                candidate = action.candidates[0]
                plan.results.append(
                    ReconciliationAction(
                        action=ACTION_FAILED,
                        turn=entry,
                        candidates=action.candidates,
                        trace_id=candidate.trace_id,
                        error_code=error_code,
                        affected_trace_ids=(candidate.trace_id,),
                        conflicting_trace_ids=(candidate.trace_id,) if error_code == "ownership_conflict" else (),
                    )
                )
            continue
        if matching:
            winner = select_verified_winner(matching)
            winner_candidate, _verification = winner
            others = [candidate for candidate in action.candidates if candidate.trace_id != winner_candidate.trace_id]
            assert work.request is not None
            experiment_id = winner_candidate.experiment_id or work.request.experiment_id
            cleanup_failures, cleanup_conflicts = _retire_candidates(gateway, others, experiment_id, retry_policy)
            cleanup_error = (
                "ownership_conflict"
                if cleanup_conflicts
                else "replacement_cleanup_failed"
                if cleanup_failures
                else None
            )
            plan.results.append(
                ReconciliationAction(
                    action=ACTION_FAILED if cleanup_failures else ACTION_SKIPPED,
                    turn=entry,
                    trace=winner_candidate,
                    candidates=action.candidates,
                    trace_id=winner_candidate.trace_id,
                    reason=REASON_UNCHANGED,
                    error_code=cleanup_error,
                    affected_trace_ids=tuple(cleanup_failures),
                    conflicting_trace_ids=tuple(cleanup_conflicts or cleanup_failures),
                    retired_trace_ids=tuple(
                        dict.fromkeys(
                            candidate.trace_id for candidate in others if candidate.trace_id not in cleanup_failures
                        )
                    ),
                )
            )
            continue
        plan.results.append(_replace_identity(plan, gateway, dependencies, retry_policy, work, action))


def _decide_imports(
    plan: ReconciliationPlan,
    gateway: MlflowGateway,
    dependencies: ReconciliationDependencies,
    retry_policy: RetryPolicy,
) -> None:
    """Phase 2b: create and verify each turn with no owned candidate."""
    for action in plan.imports:
        entry = action.turn
        if entry is None:  # pragma: no cover - the planner always attaches the turn
            continue
        work = _IdentityWork(
            key=OwnershipKey(entry.session_id, entry.turn_key),
            entry=entry,
            redaction_mode=dependencies.redaction_mode,
        )
        try:
            work.replay = dependencies.replay(entry)
            work.request = dependencies.map_trace(work.replay)
            if dependencies.artifact_snapshot(work.replay) != work.request.artifact_snapshot_fingerprint:
                raise ItemImportError("source_consistency_mismatch", "the replay differs from its mapped fingerprint")
        except ImporterError as error:
            plan.results.append(ReconciliationAction(action=ACTION_FAILED, turn=entry, error_code=error.code))
            continue
        except Exception:
            plan.results.append(ReconciliationAction(action=ACTION_FAILED, turn=entry, error_code="turn_replay_failed"))
            continue
        try:
            assert work.request is not None
            CandidateContract.from_create_request(work.request, "", 0)
        except ImporterError as error:
            plan.results.append(ReconciliationAction(action=ACTION_FAILED, turn=entry, error_code=error.code))
            continue
        except Exception:
            plan.results.append(
                ReconciliationAction(action=ACTION_FAILED, turn=entry, error_code="candidate_contract_failed")
            )
            continue
        try:
            survivors: list[OwnedTraceCandidate] = []
            created, survivor = _create_with_ambiguous_recovery(plan, gateway, work, retry_policy, survivors)
        except _AmbiguousCreateConflict as error:
            plan.results.append(
                ReconciliationAction(
                    action=ACTION_FAILED,
                    turn=entry,
                    error_code=error.code,
                    trace_id=error.trace_ids[-1] if error.trace_ids else None,
                    affected_trace_ids=error.trace_ids,
                    conflicting_trace_ids=error.trace_ids,
                )
            )
            continue
        except ImporterError as error:
            affected_trace_ids = tuple(candidate.trace_id for candidate in survivors)
            if isinstance(error, PartialTraceCreateError):
                affected_trace_ids += (error.created.trace_id,)
            plan.results.append(
                ReconciliationAction(
                    action=ACTION_FAILED,
                    turn=entry,
                    error_code=error.code,
                    trace_id=error.created.trace_id if isinstance(error, PartialTraceCreateError) else None,
                    affected_trace_ids=tuple(dict.fromkeys(affected_trace_ids)),
                    retired_trace_ids=(error.created.trace_id,)
                    if isinstance(error, PartialTraceCreateError) and error.rollback_retired
                    else (),
                    conflicting_trace_ids=(
                        tuple(dict.fromkeys(affected_trace_ids))
                        if error.code == "ownership_conflict"
                        or (isinstance(error, PartialTraceCreateError) and not error.rollback_retired)
                        else ()
                    ),
                )
            )
            continue
        except Exception:
            plan.results.append(
                ReconciliationAction(
                    action=ACTION_FAILED,
                    turn=entry,
                    error_code="trace_create_failed",
                    affected_trace_ids=tuple(candidate.trace_id for candidate in survivors),
                    conflicting_trace_ids=(),
                )
            )
            continue
        if survivor is not None:
            # The ambiguous create evidently took effect and the survivor
            # verifies against the current contract. Any other fully verified
            # survivor is redundant and must be retired before this import is
            # reported as complete. The winner is deliberately not passed to
            # the bounded cleanup helper, so a cleanup failure can never
            # delete the only verified owner.
            assert work.request is not None
            cleanup_failures, cleanup_conflicts = _retire_candidates(
                gateway,
                survivors,
                work.request.experiment_id,
                retry_policy,
            )
            cleanup_error = (
                "ownership_conflict"
                if cleanup_conflicts
                else "replacement_cleanup_failed"
                if cleanup_failures
                else None
            )
            plan.results.append(
                ReconciliationAction(
                    action=ACTION_FAILED if cleanup_failures else ACTION_IMPORTED,
                    turn=entry,
                    trace=survivor,
                    trace_id=survivor.trace_id,
                    error_code=cleanup_error,
                    affected_trace_ids=tuple(cleanup_failures),
                    retired_trace_ids=tuple(
                        dict.fromkeys(
                            candidate.trace_id for candidate in survivors if candidate.trace_id not in cleanup_failures
                        )
                    ),
                    conflicting_trace_ids=tuple(cleanup_conflicts or cleanup_failures),
                )
            )
            continue
        assert created is not None
        result = _write_identity(gateway, dependencies, retry_policy, work, action, created)
        if result.action == ACTION_IMPORTED and survivors:
            # Retire any incomplete survivor an ambiguous create left behind
            # once the verified replacement exists.
            assert work.request is not None
            cleanup_failures, cleanup_conflicts = _retire_candidates(
                gateway,
                survivors,
                work.request.experiment_id,
                retry_policy,
            )
            cleanup_error = (
                "ownership_conflict"
                if cleanup_conflicts
                else "replacement_cleanup_failed"
                if cleanup_failures
                else None
            )
            result = replace(
                result,
                action=ACTION_FAILED if cleanup_failures else result.action,
                error_code=cleanup_error,
                affected_trace_ids=tuple(cleanup_failures),
                retired_trace_ids=tuple(
                    dict.fromkeys(
                        candidate.trace_id for candidate in survivors if candidate.trace_id not in cleanup_failures
                    )
                ),
                conflicting_trace_ids=tuple(cleanup_conflicts or cleanup_failures),
            )
        plan.results.append(result)


def _apply_retirements(plan: ReconciliationPlan, gateway: MlflowGateway, retry_policy: RetryPolicy) -> None:
    """Phase 3: retire stale owned traces of selected complete scans."""
    for action in plan.retirements:
        candidate = action.trace
        if candidate is None:  # pragma: no cover - the planner always attaches the trace
            continue
        if candidate.experiment_id is None:
            plan.results.append(
                replace(
                    action,
                    action=ACTION_FAILED,
                    trace_id=candidate.trace_id,
                    error_code="ownership_conflict",
                    affected_trace_ids=(candidate.trace_id,),
                    conflicting_trace_ids=(candidate.trace_id,),
                )
            )
            continue
        retirement = _bounded_retire(
            gateway,
            candidate.experiment_id,
            candidate.trace_id,
            retry_policy,
            candidate=candidate,
        )
        if retirement.retired:
            plan.results.append(
                replace(
                    action,
                    action=ACTION_RETIRED,
                    trace_id=candidate.trace_id,
                    retired_trace_ids=(candidate.trace_id,),
                )
            )
        else:
            plan.results.append(
                replace(
                    action,
                    action=ACTION_FAILED,
                    trace_id=candidate.trace_id,
                    error_code=retirement.error_code or "retirement_failed",
                    affected_trace_ids=(candidate.trace_id,),
                    conflicting_trace_ids=(candidate.trace_id,)
                    if retirement.error_code == "ownership_conflict"
                    else (),
                )
            )


# ---------------------------------------------------------------------------
# Replacement and rollback
# ---------------------------------------------------------------------------


def _replace_identity(
    plan: ReconciliationPlan,
    gateway: MlflowGateway,
    dependencies: ReconciliationDependencies,
    retry_policy: RetryPolicy,
    work: _IdentityWork,
    action: ReconciliationAction,
) -> ReconciliationAction:
    """Replace one identity: create/verify/upload/verify/record, then retire the old."""
    survivors: list[OwnedTraceCandidate] = []
    try:
        created, survivor_winner = _create_with_ambiguous_recovery(
            plan,
            gateway,
            work,
            retry_policy,
            survivors,
        )
    except _AmbiguousCreateConflict as error:
        trace_ids = tuple(
            dict.fromkeys([candidate.trace_id for candidate in action.candidates] + list(error.trace_ids))
        )
        return ReconciliationAction(
            action=ACTION_FAILED,
            turn=work.entry,
            candidates=action.candidates,
            error_code=error.code,
            trace_id=trace_ids[-1] if trace_ids else None,
            affected_trace_ids=trace_ids,
            conflicting_trace_ids=trace_ids,
        )
    except ImporterError as error:
        affected_trace_ids = (
            tuple(candidate.trace_id for candidate in action.candidates) + (error.created.trace_id,)
            if isinstance(error, PartialTraceCreateError)
            else tuple(candidate.trace_id for candidate in action.candidates)
        )
        return ReconciliationAction(
            action=ACTION_FAILED,
            turn=work.entry,
            candidates=action.candidates,
            error_code=error.code,
            trace_id=error.created.trace_id if isinstance(error, PartialTraceCreateError) else None,
            affected_trace_ids=tuple(dict.fromkeys(affected_trace_ids)),
            retired_trace_ids=(error.created.trace_id,)
            if isinstance(error, PartialTraceCreateError) and error.rollback_retired
            else (),
            conflicting_trace_ids=(
                tuple(dict.fromkeys(affected_trace_ids))
                if error.code == "ownership_conflict"
                or (isinstance(error, PartialTraceCreateError) and not error.rollback_retired)
                else ()
            ),
        )
    except Exception:
        return ReconciliationAction(
            action=ACTION_FAILED,
            turn=work.entry,
            candidates=action.candidates,
            error_code="trace_create_failed",
            affected_trace_ids=tuple(
                dict.fromkeys(
                    [candidate.trace_id for candidate in action.candidates]
                    + [candidate.trace_id for candidate in survivors]
                )
            ),
        )
    if survivor_winner is not None:
        # The ambiguous create evidently took effect and the survivor verifies
        # against the current contract: it is the winner; retire the others.
        others = [candidate for candidate in action.candidates if candidate.trace_id != survivor_winner.trace_id]
        others.extend(survivors)
        assert work.request is not None
        experiment_id = survivor_winner.experiment_id or work.request.experiment_id
        cleanup_failures, cleanup_conflicts = _retire_candidates(gateway, others, experiment_id, retry_policy)
        survivor_cleanup_error: str | None = None
        if cleanup_conflicts:
            survivor_cleanup_error = "ownership_conflict"
        elif cleanup_failures:
            survivor_cleanup_error = "replacement_cleanup_failed"
        return ReconciliationAction(
            action=ACTION_FAILED if cleanup_failures else ACTION_UPDATED,
            turn=work.entry,
            trace=survivor_winner,
            candidates=action.candidates,
            trace_id=survivor_winner.trace_id,
            error_code=survivor_cleanup_error,
            affected_trace_ids=tuple(cleanup_failures),
            retired_trace_ids=tuple(
                dict.fromkeys(candidate.trace_id for candidate in others if candidate.trace_id not in cleanup_failures)
            ),
            conflicting_trace_ids=tuple(cleanup_conflicts or cleanup_failures),
        )
    assert created is not None
    result = _write_identity(gateway, dependencies, retry_policy, work, action, created)
    if result.action != ACTION_UPDATED:
        return result
    # A verified replacement exists: retire the old primary, then every other
    # redundant owned candidate for this identity (including any survivor an
    # ambiguous create left behind).
    old_candidates = [*action.candidates, *survivors]
    primary = _primary_old(old_candidates)
    assert work.request is not None
    experiment_id = work.request.experiment_id
    all_identity_ids = tuple(dict.fromkeys([candidate.trace_id for candidate in old_candidates] + [created.trace_id]))
    retired: list[str] = []
    if primary is not None:
        primary_retirement = _bounded_retire(
            gateway,
            experiment_id,
            primary.trace_id,
            retry_policy,
            candidate=primary,
        )
        if not primary_retirement.retired:
            # Old-retirement failure after a verified replacement: bounded
            # compensating retirement of the NEW trace; the old trace is
            # never deleted on uncertainty.
            created_retirement = _bounded_retire(
                gateway,
                experiment_id,
                created.trace_id,
                retry_policy,
                session_id=work.request.metadata.get(OWNERSHIP_SESSION_KEY),
                turn_key=work.request.metadata.get(OWNERSHIP_TURN_KEY),
                expected_metadata=_request_creation_metadata(work.request),
            )
            if created_retirement.retired:
                return ReconciliationAction(
                    action=ACTION_FAILED,
                    turn=work.entry,
                    candidates=action.candidates,
                    trace_id=created.trace_id,
                    error_code=(
                        "ownership_conflict"
                        if primary_retirement.error_code == "ownership_conflict"
                        else "retirement_failed"
                    ),
                    affected_trace_ids=all_identity_ids,
                    retired_trace_ids=(created.trace_id,),
                    conflicting_trace_ids=all_identity_ids
                    if primary_retirement.error_code == "ownership_conflict"
                    else (),
                )
            return ReconciliationAction(
                action=ACTION_FAILED,
                turn=work.entry,
                candidates=action.candidates,
                trace_id=created.trace_id,
                error_code=(
                    "ownership_conflict"
                    if primary_retirement.error_code == "ownership_conflict"
                    or created_retirement.error_code == "ownership_conflict"
                    else "duplicate_owned_identity"
                ),
                affected_trace_ids=all_identity_ids,
                conflicting_trace_ids=all_identity_ids
                if primary_retirement.error_code == "ownership_conflict"
                or created_retirement.error_code == "ownership_conflict"
                else (),
            )
        retired.append(primary.trace_id)
    others = [candidate for candidate in old_candidates if primary is None or candidate.trace_id != primary.trace_id]
    cleanup_failures, cleanup_conflicts = _retire_candidates(gateway, others, experiment_id, retry_policy)
    retired.extend(candidate.trace_id for candidate in others if candidate.trace_id not in cleanup_failures)
    cleanup_error: str | None = None
    if cleanup_conflicts:
        cleanup_error = "ownership_conflict"
    elif cleanup_failures:
        cleanup_error = "replacement_cleanup_failed"
    cleanup_affected = all_identity_ids if cleanup_conflicts else tuple(cleanup_failures)
    cleanup_conflicting = all_identity_ids if cleanup_conflicts else tuple(cleanup_conflicts or cleanup_failures)
    return replace(
        result,
        action=ACTION_FAILED if cleanup_failures else result.action,
        error_code=cleanup_error,
        affected_trace_ids=cleanup_affected,
        retired_trace_ids=tuple(dict.fromkeys(retired)),
        conflicting_trace_ids=cleanup_conflicting,
    )


def _write_identity(
    gateway: MlflowGateway,
    dependencies: ReconciliationDependencies,
    retry_policy: RetryPolicy,
    work: _IdentityWork,
    action: ReconciliationAction,
    created: CreatedTrace,
) -> ReconciliationAction:
    """Verify, upload, verify, and record one created trace; roll back on failure.

    The strict order for both imports and replacements: create → verify
    metadata/status/timing/ids → upload the deterministic artifact →
    verify its exact bytes → record the verified digest on the trace.
    Any failure retires the NEW trace as rollback and retains every
    existing valid trace; a failed rollback reports both the old
    primary's and the new trace's ids as a reconciliation conflict.
    """
    assert work.replay is not None and work.request is not None
    replay = work.replay
    request = work.request
    try:
        retry_call(lambda: gateway.verify_trace(created, request), retry_policy)
    except ImporterError as error:
        return _rollback_after_replacement(gateway, work, action, created, error.code, retry_policy)
    except Exception:
        return _rollback_after_replacement(gateway, work, action, created, "trace_verify_failed", retry_policy)
    try:
        verification = retry_call(
            lambda: dependencies.upload_artifact(replay, created, request.artifact_relative_path, work.redaction_mode),
            retry_policy,
        )
        if verification.snapshot_fingerprint is not None and (
            verification.snapshot_fingerprint != request.artifact_snapshot_fingerprint
        ):
            raise ItemImportError("source_consistency_mismatch", "the artifact differs from its mapped fingerprint")
    except ImporterError as error:
        return _rollback_after_replacement(gateway, work, action, created, error.code, retry_policy)
    except Exception:
        return _rollback_after_replacement(gateway, work, action, created, "artifact_upload_failed", retry_policy)
    try:
        retry_call(
            lambda: gateway.verify_named_json(
                created.trace_id,
                verification.relative_path,
                verification.sha256,
                verification.size,
            ),
            retry_policy,
        )
    except ImporterError as error:
        return _rollback_after_replacement(gateway, work, action, created, error.code, retry_policy)
    except Exception:
        return _rollback_after_replacement(gateway, work, action, created, "artifact_verification_failed", retry_policy)
    # The artifact upload verified: RECORD the verified digest on the trace
    # (Ruling 16) so later runs can check the stored artifact's integrity
    # against the digest recorded at write time. A failure here rolls the new
    # trace back like any other write failure; a crash between the upload and
    # this recording leaves an incomplete candidate that the next run
    # replaces.
    try:
        retry_call(
            lambda: gateway.record_artifact_verification(
                created.trace_id,
                verification.sha256,
                verification.size,
            ),
            retry_policy,
        )
    except ImporterError as error:
        return _rollback_after_replacement(gateway, work, action, created, error.code, retry_policy)
    except Exception:
        return _rollback_after_replacement(gateway, work, action, created, "artifact_record_failed", retry_policy)
    action_kind = ACTION_IMPORTED if not action.candidates else ACTION_UPDATED
    return ReconciliationAction(
        action=action_kind,
        turn=work.entry,
        candidates=action.candidates,
        trace_id=created.trace_id,
    )


def _rollback_after_replacement(
    gateway: MlflowGateway,
    work: _IdentityWork,
    action: ReconciliationAction,
    created: CreatedTrace,
    error_code: str,
    retry_policy: RetryPolicy,
) -> ReconciliationAction:
    """Retire the new trace as rollback; retain every existing valid trace."""
    assert work.request is not None
    rollback = _bounded_retire(
        gateway,
        work.request.experiment_id,
        created.trace_id,
        retry_policy,
        session_id=work.request.metadata.get(OWNERSHIP_SESSION_KEY),
        turn_key=work.request.metadata.get(OWNERSHIP_TURN_KEY),
        expected_metadata=_request_creation_metadata(work.request),
    )
    if rollback.retired:
        return ReconciliationAction(
            action=ACTION_FAILED,
            turn=work.entry,
            candidates=action.candidates,
            trace_id=created.trace_id,
            error_code=error_code,
            affected_trace_ids=(created.trace_id,),
            retired_trace_ids=(created.trace_id,),
            conflicting_trace_ids=(),
        )
    # Rollback failed within the bounded policy: the old primary and the new
    # trace both remain importer-owned; report both ids as a reconciliation
    # conflict for explicit operator recovery.
    affected = tuple(dict.fromkeys([candidate.trace_id for candidate in action.candidates] + [created.trace_id]))
    return ReconciliationAction(
        action=ACTION_FAILED,
        turn=work.entry,
        candidates=action.candidates,
        trace_id=created.trace_id,
        error_code=(
            "ownership_conflict" if rollback.error_code == "ownership_conflict" else "duplicate_owned_identity"
        ),
        affected_trace_ids=affected,
        conflicting_trace_ids=affected,
    )


# ---------------------------------------------------------------------------
# Ambiguous-create handling
# ---------------------------------------------------------------------------


def _create_with_ambiguous_recovery(
    plan: ReconciliationPlan,
    gateway: MlflowGateway,
    work: _IdentityWork,
    retry_policy: RetryPolicy,
    survivors: list[OwnedTraceCandidate] | None = None,
) -> tuple[CreatedTrace | None, OwnedTraceCandidate | None]:
    """Create the trace under the retry policy, searching before each retry.

    The mapped request already carries the deterministic
    ``opencode.operation_key`` as its client request/idempotency key.
    After an ambiguous (transient) create failure, importer-owned traces with
    the deterministic operation key are searched BEFORE the create is retried;
    the full owner/session/turn/render tuple is then verified so a changed
    identity is a conflict, never another create. Every surviving candidate is
    verified and recorded in the plan for deterministic winner
    selection and explicit operator recovery, and appended to
    ``survivors`` so the caller retires it once a verified winner exists.
    Pre-existing candidates are deliberately not seeded into the observed
    survivor set: an authoritative empty search after a transient create
    exhaustion is still ``trace_create_failed``, not an ownership conflict.
    Returns ``(created, survivor_winner)`` — exactly one is non-``None`` on
    success.
    """
    assert work.request is not None
    attempts = 0
    discovered_ids: list[str] = []

    def remember(candidates: Iterable[OwnedTraceCandidate]) -> None:
        """Retain every operation-key id observed during recovery."""
        for candidate in candidates:
            if candidate.trace_id not in discovered_ids:
                discovered_ids.append(candidate.trace_id)

    def conflict() -> None:
        """Stop recovery before another create and preserve safe trace ids."""
        raise _AmbiguousCreateConflict(discovered_ids) from None

    while True:
        attempts += 1
        try:
            return gateway.create_trace(work.request), None
        except PartialTraceCreateError as error:
            cleanup = _bounded_retire(
                gateway,
                work.request.experiment_id,
                error.created.trace_id,
                retry_policy,
                session_id=work.request.metadata.get(OWNERSHIP_SESSION_KEY),
                turn_key=work.request.metadata.get(OWNERSHIP_TURN_KEY),
                expected_metadata=_request_creation_metadata(work.request),
            )
            if not cleanup.retired:
                error.code = (
                    "ownership_conflict"
                    if cleanup.error_code == "ownership_conflict"
                    else "partial_trace_cleanup_failed"
                )
            else:
                error.rollback_retired = True
            raise
        except Exception as exc:
            if not is_transient_error(exc):
                raise
            surviving_candidates, search_failed = _search_surviving_candidates(gateway, work, retry_policy)
            remember(surviving_candidates)
            if search_failed:
                conflict()
            matching: list[tuple[OwnedTraceCandidate, CandidateVerification]] = []
            verification_failed = False
            nonmatching_survivor = False
            for survivor in surviving_candidates:
                expected_metadata = work.request.metadata
                if (
                    survivor.session_id != expected_metadata.get(OWNERSHIP_SESSION_KEY)
                    or survivor.turn_key != expected_metadata.get(OWNERSHIP_TURN_KEY)
                    or survivor.render_fingerprint != expected_metadata.get(OWNERSHIP_RENDER_FINGERPRINT_KEY)
                    or survivor.operation_key != expected_metadata.get(OWNERSHIP_OPERATION_KEY)
                ):
                    verification = CandidateVerification(
                        trace_id=survivor.trace_id,
                        reason="ownership_identity_changed",
                        error_code="ownership_conflict",
                    )
                    plan.verifications[survivor.trace_id] = verification
                elif survivor.trace_id in plan.verifications:
                    verification = plan.verifications[survivor.trace_id]
                else:
                    verification = _verify_one_candidate(gateway, work, survivor, retry_policy)
                    plan.verifications[survivor.trace_id] = verification
                if verification.error_code is not None:
                    # Continue through every survivor so a later-page id is
                    # never hidden by the first unverifiable candidate.
                    verification_failed = True
                if verification.matches:
                    matching.append((survivor, verification))
                else:
                    nonmatching_survivor = True
            if verification_failed or nonmatching_survivor:
                conflict()
            if attempts >= retry_policy.max_attempts and not matching:
                if discovered_ids:
                    conflict()
                raise
            if matching:
                winner, _verification = select_verified_winner(matching)
                if survivors is not None:
                    for candidate, _ in matching:
                        if candidate.trace_id != winner.trace_id and all(
                            existing.trace_id != candidate.trace_id for existing in survivors
                        ):
                            survivors.append(candidate)
                return None, winner
            # A completed search that found an operation-key survivor is not
            # an authoritative empty result. Another non-idempotent create
            # would turn uncertainty into duplication.
            if surviving_candidates:
                conflict()
            time.sleep(retry_policy.delay_seconds(attempts, random.random))


def _search_surviving_candidates(
    gateway: MlflowGateway,
    work: _IdentityWork,
    retry_policy: RetryPolicy,
) -> tuple[list[OwnedTraceCandidate], bool]:
    """Search operation-key survivors, reporting whether the search was unverifiable."""
    assert work.request is not None
    request = work.request
    expected_operation = request.metadata[OWNERSHIP_OPERATION_KEY]
    observed: dict[str, OwnedTraceCandidate] = {}

    def collect_page_results() -> None:
        """Collect pages while retaining ids if a later page fails."""
        for trace in gateway.iter_owned_traces(request.experiment_id):
            metadata = trace.metadata
            if metadata.get(OWNERSHIP_OPERATION_KEY) == expected_operation:
                candidate = OwnedTraceCandidate.from_owned_trace(trace, request.experiment_id)
                observed.setdefault(candidate.trace_id, candidate)

    try:
        retry_call(collect_page_results, retry_policy)
    except Exception:
        # A partial page walk is never an authoritative empty result. Return
        # every survivor seen before the failure for safe conflict reporting.
        return list(observed.values()), True
    return list(observed.values()), False


# ---------------------------------------------------------------------------
# Retirement helpers
# ---------------------------------------------------------------------------


def _bounded_retire(
    gateway: MlflowGateway,
    experiment_id: str,
    trace_id: str,
    retry_policy: RetryPolicy,
    *,
    candidate: OwnedTraceCandidate | None = None,
    session_id: str | None = None,
    turn_key: str | None = None,
    expected_metadata: Mapping[str, str] | None = None,
) -> _RetirementResult:
    """Retire one trace through the gateway's atomic verification boundary."""
    if candidate is not None:
        if not candidate.ownership_complete:
            return _RetirementResult(False, "ownership_conflict")
        assert candidate.importer_marker is not None
        session_id = candidate.session_id
        turn_key = candidate.turn_key
        expected_metadata = candidate_creation_metadata(candidate)
    if session_id is None or turn_key is None:
        return _RetirementResult(False, "ownership_conflict")
    try:
        retry_call(
            lambda: gateway.retire_trace(
                experiment_id,
                trace_id,
                session_id=session_id,
                turn_key=turn_key,
                expected_metadata=expected_metadata,
            ),
            retry_policy,
        )
    except ImporterError as error:
        return _RetirementResult(
            False,
            "ownership_conflict" if error.code == "ownership_conflict" else "retirement_failed",
        )
    except Exception:
        return _RetirementResult(False, "retirement_failed")
    return _RetirementResult(True)


def _request_creation_metadata(request: TraceCreateRequest) -> dict[str, str]:
    """Return only exact ownership metadata for a final delete check."""
    return CandidateContract.from_create_request(request, "", 0).creation_metadata


def candidate_creation_metadata(candidate: OwnedTraceCandidate) -> dict[str, str]:
    """Return the indexed candidate's complete exact ownership contract."""
    return {
        OWNERSHIP_IMPORTER_KEY: candidate.importer_marker or "",
        OWNERSHIP_SESSION_KEY: candidate.session_id,
        OWNERSHIP_TURN_INDEX_KEY: str(candidate.turn_index) if candidate.turn_index is not None else "",
        OWNERSHIP_TURN_KEY: candidate.turn_key,
        OWNERSHIP_SOURCE_FINGERPRINT_KEY: candidate.source_fingerprint or "",
        OWNERSHIP_SOURCE_SCHEMA_KEY: candidate.source_schema or "",
        OWNERSHIP_IMPORTER_VERSION_KEY: candidate.importer_version or "",
        OWNERSHIP_MAPPING_VERSION_KEY: candidate.mapping_version or "",
        OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY: candidate.artifact_schema_version or "",
        OWNERSHIP_REDACTION_MODE_KEY: candidate.redaction_mode or "",
        OWNERSHIP_CATALOG_VERSION_KEY: candidate.catalog_version or "",
        OWNERSHIP_RENDER_FINGERPRINT_KEY: candidate.render_fingerprint or "",
        OWNERSHIP_OPERATION_KEY: candidate.operation_key or "",
        "opencode.warnings": ",".join(candidate.warnings),
    }


def _retire_candidates(
    gateway: MlflowGateway,
    candidates: Sequence[OwnedTraceCandidate],
    experiment_id: str | None,
    retry_policy: RetryPolicy,
) -> tuple[list[str], list[str]]:
    """Retire owned candidates, returning the trace ids that could not be retired.

    Each candidate retires in its own experiment when it carries one,
    falling back to the identity's resolved experiment.
    """
    failures: list[str] = []
    conflicts: list[str] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        if candidate.trace_id in seen_ids:
            continue
        seen_ids.add(candidate.trace_id)
        resolved = candidate.experiment_id if candidate.experiment_id is not None else experiment_id
        if resolved is None:
            failures.append(candidate.trace_id)
            conflicts.append(candidate.trace_id)
            continue
        retirement = _bounded_retire(
            gateway,
            resolved,
            candidate.trace_id,
            retry_policy,
            candidate=candidate,
        )
        if not retirement.retired:
            failures.append(candidate.trace_id)
            if retirement.error_code == "ownership_conflict":
                conflicts.append(candidate.trace_id)
    return failures, conflicts


def _primary_old(candidates: Sequence[OwnedTraceCandidate]) -> OwnedTraceCandidate | None:
    """The deterministic old primary: newest created, ties by greatest trace id."""
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            candidate.request_time_ms if candidate.request_time_ms is not None else -1,
            candidate.trace_id,
        ),
    )
