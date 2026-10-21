"""The public MLflow gateway boundary for the importer's writer.

This is the ONLY module that calls MLflow experiment/search/delete APIs or
starts/ends trace spans; mapping, source, content, and orchestration
modules never import MLflow. Every MLflow call flows through one of three
injectable seams so unit and contract tests run against fake
public-client/LiveSpan-shaped objects with no network and no server:

- the ``MlflowClient`` (experiment lookup/creation, trace retrieval/search,
  public trace tag update, retirement);
- :class:`PublicTraceApi`, the adapter over the module-level public trace
  API (``mlflow.start_span_no_context`` and ``LiveSpan.end``);
- the artifact-repository factory, defaulting to the public
  ``mlflow.artifacts.get_artifact_repository`` (Ruling 5: never the
  semi-private ``mlflow.store.artifact.artifact_repo`` import Task 0's
  availability check used).

MLflow imports are confined to the public ``mlflow``/``mlflow.tracking``/
``mlflow.artifacts`` entry points (pinned by the gateway contract test).

Behavioral facts adapted from the pinned 3.17.2 profile
(``tests/integration/mlflow_profile.py`` / ``docs/compatibility.md``):

- ``mlflow.start_span_no_context(name, span_type, parent_span, inputs,
  attributes, tags, metadata, experiment_id, start_time_ns)`` returns a
  ``LiveSpan`` whose ``trace_id``/``span_id`` are properties; ``tags`` and
  ``metadata`` update the TRACE; a failure inside the call degrades to a
  NoOp span, so :meth:`MlflowGateway.create_trace` fails loudly whenever a
  span's trace id equals the NoOp sentinel instead of silently losing the
  trace. ``experiment_id`` is a ROOT-span parameter only: children join
  their parent's trace (and inherit its experiment), and the pinned
  3.17.2 provider degrades a non-root span carrying ``experiment_id`` to
  a NoOp span, so :meth:`MlflowGateway.create_trace` passes it only for
  the root.
- ``LiveSpan.end(outputs, attributes, status, end_time_ns)`` completes a
  span; the trace state follows the root span's status, and the pinned
  3.17.2 ``end`` unconditionally forces every non-``ERROR`` span status to
  ``OK`` at end time — so a root ended ``UNSET`` (an incomplete turn)
  retrieves as span status ``OK`` and state ``OK`` while the source-known
  outcome stays in the ``opencode.outcome`` trace metadata.
- Span export follows the GLOBAL tracking URI, not the client's, so
  ``create_trace`` pins the global URI to the client's target for the whole
  create and restores the saved value afterwards.
- ``MlflowClient.search_traces`` paginates through an opaque page token;
  the deprecated ``experiment_ids=`` form is never used (only
  ``locations=``).
- Trace tag values silently truncate at 4096 characters and metadata values
  at 250, so :meth:`MlflowGateway.verify_trace` never relies on
  full-length tag/metadata round-trips: values at or below the documented
  limit are compared exactly, longer values only require the key's
  presence. Verification also compares the root span's status (against the
  single-sourced effective status on ``CreatedTrace``) and the trace state
  through the documented root-status→state mapping (an ``UNSET`` root
  surfaces as state ``OK``), and compares the root attributes as a subset
  because MLflow adds its own internal attribute keys to server-side spans.
- Named trace artifacts ride the public composition the profile proved:
  the trace's ``mlflow.artifactLocation`` tag resolved through
  ``get_artifact_repository`` to an ``ArtifactRepository`` whose
  ``log_artifact(local_file, artifact_path)`` preserves the requested
  relative path. :class:`TraceArtifactSink` reimplements the profile's
  ``upload_named_json``/``verify_named_json`` behavior identically (streamed
  staging in fixed chunks, sha256+size verification, traversal guard,
  tamper rejection); UUID-only attachments and run-artifact channels are
  never used.

Mutation gates: the capability check must pass before any mutation in
normal mode (a normal-mode mutation requires a normal-mode probe), and a
dry-run gateway never mutates — ``resolve_experiment`` represents a planned
creation instead.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import get_artifact_repository

from opencode_mlflow.content.artifacts import ArtifactProvenance
from opencode_mlflow.errors import CapabilityError, ConfigurationError, ItemImportError, PartialTraceCreateError
from opencode_mlflow.mlflow.capabilities import (
    CapabilityReport,
    CapabilityRequest,
    VersionTransport,
    probe_capabilities,
)

if TYPE_CHECKING:
    from opencode_mlflow.mlflow.reconciliation import CandidateContract, CandidateVerification, OwnedTraceCandidate

__all__ = [
    "DEFAULT_UPLOAD_CHUNK_BYTES",
    "NO_OP_TRACE_ID",
    "OWNERSHIP_IMPORTER_VALUE",
    "OWNERSHIP_TAG_KEY",
    "PublicTraceApi",
    "RETRIEVED_SPAN_STATUS",
    "TRACE_ARTIFACT_LOCATION_TAG",
    "TRACE_METADATA_VALUE_LIMIT",
    "TRACE_SESSION_METADATA_KEY",
    "TRACE_STATE_FOR_ROOT_STATUS",
    "TRACE_TAG_VALUE_LIMIT",
    "VERIFY_CHUNK_BYTES",
    "ArtifactVerification",
    "CreatedTrace",
    "ExperimentRef",
    "LiveSpanLike",
    "MlflowGateway",
    "OwnedTrace",
    "TraceArtifactSink",
    "TraceChildRequest",
    "TraceCreateRequest",
    "build_gateway",
    "validate_required_metadata",
]

# ---------------------------------------------------------------------------
# Constants pinned to the installed mlflow 3.17.2 (contract-tested against
# the real values; see tests/contract/test_gateway_contract.py)
# ---------------------------------------------------------------------------

#: The trace id every degraded (NoOp) span carries. ``start_span_no_context``
#: returns a NoOp span instead of raising when span creation fails, so this
#: sentinel is the only loud-failure signal for silent span loss.
NO_OP_TRACE_ID = "MLFLOW_NO_OP_SPAN_TRACE_ID"

#: The public tag every server-backed trace carries, pointing at the trace's
#: artifact directory (``mlflow.utils.mlflow_tags.MLFLOW_ARTIFACT_LOCATION``).
TRACE_ARTIFACT_LOCATION_TAG = "mlflow.artifactLocation"

#: The native session metadata key MLflow itself uses
#: (``mlflow.tracing.constant.TraceMetadataKey.TRACE_SESSION``); the gateway
#: sets it from ``TraceCreateRequest.native_session_id`` at root creation.
TRACE_SESSION_METADATA_KEY = "mlflow.trace.session"

#: Documented silent-truncation limits (``mlflow.tracing.constant``): trace
#: tag values at 4096 characters, trace metadata values at 250. Verification
#: never relies on full-length round-trips above these limits.
TRACE_TAG_VALUE_LIMIT = 4096
TRACE_METADATA_VALUE_LIMIT = 250

# These values participate in candidate ownership/equivalence or carry the
# source outcome/warning contract.  Unlike descriptive metadata, they cannot
# be replaced with a display-only preview: the pinned server silently
# truncates metadata values above this limit.
_EXACT_METADATA_KEYS = frozenset(
    {
        "opencode.importer",
        "opencode.session_id",
        "opencode.turn_key",
        "opencode.turn_index",
        "opencode.source_fingerprint",
        "opencode.source_schema",
        "opencode.importer_version",
        "opencode.mapping_version",
        "opencode.artifact_schema_version",
        "opencode.redaction_mode",
        "opencode.catalog_version",
        "opencode.render_fingerprint",
        "opencode.operation_key",
        "opencode.outcome",
        "opencode.warnings",
    }
)

#: Importer ownership marker: only traces carrying exactly this pair are
#: ever yielded by :meth:`MlflowGateway.iter_owned_traces` or retired.
OWNERSHIP_TAG_KEY = "opencode.importer"
OWNERSHIP_IMPORTER_VALUE = "opencode-mlflow"

#: Server-side ownership filter (the design's searchable trace metadata
#: filter). Applied only when the target accepts it; every page is always
#: re-verified locally so an unowned trace is never yielded.
_OWNERSHIP_FILTER = f"metadata.`{OWNERSHIP_TAG_KEY}` = '{OWNERSHIP_IMPORTER_VALUE}'"

#: Default fixed upload chunk for the artifact sink (8 MiB), matching the
#: content layer's upload chunk default.
DEFAULT_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024

#: Fixed read block (64 KiB) for verification streams, so no whole file is
#: ever held in memory.
VERIFY_CHUNK_BYTES = 65_536

#: The root span type for every created trace (the profile's root spans).
_ROOT_SPAN_TYPE = "CHAIN"

#: How a root span status the gateway APPLIED surfaces as the trace's
#: ``state``: ``OK``/``ERROR`` map directly, and a root ended ``UNSET``
#: surfaces as state ``OK`` because the pinned 3.17.2 ``LiveSpan.end``
#: coerces its span status to ``OK`` at end time (the live-verified
#: mechanism; see :data:`RETRIEVED_SPAN_STATUS`) — so an incomplete turn's
#: trace state reads ``OK`` on the pinned server.
TRACE_STATE_FOR_ROOT_STATUS: Mapping[str, str] = {"OK": "OK", "ERROR": "ERROR", "UNSET": "OK"}

#: How a span status the gateway APPLIED surfaces on retrieval: the pinned
#: 3.17.2 ``LiveSpan.end`` unconditionally forces every non-``ERROR`` span
#: status to ``OK`` at end time (its own documented workaround for spans that
#: never set a status), so a span ended ``UNSET`` — the root of an incomplete
#: turn, or a pending/running tool child — retrieves as span status ``OK``.
#: Verification compares retrieved span statuses through this mapping; the
#: source-known ``UNSET`` stays recorded on ``CreatedTrace.root_status`` and
#: in the ``opencode.outcome`` trace metadata.
RETRIEVED_SPAN_STATUS: Mapping[str, str] = {"OK": "OK", "ERROR": "ERROR", "UNSET": "OK"}


# ---------------------------------------------------------------------------
# Injectable protocol shapes
# ---------------------------------------------------------------------------


class LiveSpanLike(Protocol):
    """The LiveSpan-shaped surface the gateway consumes.

    Real ``LiveSpan`` objects satisfy this protocol (ids are properties on
    3.17.2); tests supply fake LiveSpan-shaped objects through
    :class:`PublicTraceApi`.
    """

    @property
    def trace_id(self) -> str: ...

    @property
    def span_id(self) -> str: ...

    @property
    def parent_id(self) -> str | None: ...

    def end(
        self,
        outputs: Any = None,
        attributes: Any = None,
        status: Any = None,
        end_time_ns: int | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class PublicTraceApi:
    """Adapter over the module-level public trace API.

    Wraps ``mlflow.start_span_no_context`` (span creation) and
    ``LiveSpan.end`` (span completion) so every module-level MLflow trace
    call the gateway makes flows through one injectable seam. Tests inject
    a fake with the same shape that returns fake LiveSpan-shaped objects.
    """

    def start_span_no_context(
        self,
        *,
        name: str,
        span_type: str,
        parent_span: LiveSpanLike | None = None,
        inputs: object | None = None,
        attributes: Mapping[str, object] | None = None,
        tags: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        experiment_id: str | None = None,
        start_time_ns: int | None = None,
    ) -> LiveSpanLike:
        # 3.17.2 annotates ``attributes`` as dict[str, str] but accepts
        # arbitrary JSON-serializable values (LiveSpan.set_attributes is
        # typed dict[str, Any]); the cast keeps this API-version difference
        # inside the adapter.
        return mlflow.start_span_no_context(
            name=name,
            span_type=span_type,
            parent_span=cast(Any, parent_span),
            inputs=inputs,
            attributes=cast(Any, dict(attributes)) if attributes is not None else None,
            tags=dict(tags) if tags is not None else None,
            metadata=dict(metadata) if metadata is not None else None,
            experiment_id=experiment_id,
            start_time_ns=start_time_ns,
        )

    def end_span(
        self,
        span: LiveSpanLike,
        *,
        outputs: object | None = None,
        attributes: Mapping[str, object] | None = None,
        status: str | None = None,
        end_time_ns: int | None = None,
    ) -> None:
        span.end(outputs=outputs, attributes=attributes, status=status, end_time_ns=end_time_ns)


# ---------------------------------------------------------------------------
# Request and result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceChildRequest:
    """One child span to create inside a trace, mapped from one source record.

    ``source_key`` is the child's stable source identity (used by later
    children's ``parent_source_key`` and the identity map);
    ``parent_source_key`` is ``None`` for a child of the root. Mappings are
    treated as immutable by the gateway.
    """

    source_key: str
    name: str
    span_type: str
    parent_source_key: str | None
    inputs: object | None
    attributes: Mapping[str, object]
    tags: Mapping[str, str]
    start_time_ns: int | None
    outputs: object | None
    status: str
    end_time_ns: int | None


@dataclass(frozen=True, slots=True)
class _ExpectedChildContract:
    """Compact full-field contract retained during trace verification.

    The gateway must compare every retrieved child field, but it must not
    retain a second graph of full request objects or source records. Mapping
    has already replaced oversized values with bounded artifact references;
    this slot-based value stores only those mapped fields and never truncates
    them. One contract exists per created span for the duration of one
    verification call, bounded by the target trace's child count.
    """

    name: str
    span_type: str
    parent_source_key: str | None
    inputs: object | None
    attributes: Mapping[str, object]
    tags: Mapping[str, str]
    start_time_ns: int | None
    outputs: object | None
    status: str
    end_time_ns: int | None

    @classmethod
    def from_request(cls, child: TraceChildRequest) -> _ExpectedChildContract:
        return cls(
            name=child.name,
            span_type=child.span_type,
            parent_source_key=child.parent_source_key,
            inputs=child.inputs,
            attributes=MappingProxyType(dict(child.attributes)),
            tags=MappingProxyType(dict(child.tags)),
            start_time_ns=child.start_time_ns,
            outputs=child.outputs,
            status=child.status,
            end_time_ns=child.end_time_ns,
        )


@dataclass(frozen=True)
class TraceCreateRequest:
    """One complete trace to create: the root span plus lazy child specs.

    ``child_specs`` returns a fresh lazy iterator of child requests; the
    gateway consumes it one child at a time and never collects the full
    child list. ``metadata`` and ``root_tags`` are supplied at root-span
    creation (with the native session metadata key added from
    ``native_session_id``), so all ownership/operation fields survive a
    process crash before trace completion. The artifact fields record the
    pre-identity-map snapshot fingerprint and the deterministic artifact
    relative path reconciliation will verify against. ``root_status`` is
    the mapping's authoritative root span status (Task 10): when set it
    wins over the gateway's derivation, so a source-known outcome (an
    explicit failure whose last child ended OK) is never re-derived;
    ``None`` keeps the verbatim-field derivation.
    """

    client_request_id: str
    experiment_id: str
    name: str
    native_session_id: str
    root_inputs: object
    root_outputs: object | None
    root_attributes: Mapping[str, object]
    root_tags: Mapping[str, str]
    metadata: Mapping[str, str]
    start_time_ns: int | None
    end_time_ns: int | None
    child_specs: Callable[[], Iterator[TraceChildRequest]]
    expected_operation_key: str
    artifact_snapshot_fingerprint: str
    artifact_relative_path: str
    root_status: str | None = None


@dataclass(frozen=True)
class ExperimentRef:
    """A resolved MLflow experiment.

    ``experiment_id`` is ``None`` only for a dry-run planned creation
    (``planned_creation=True``): the experiment does not exist yet and was
    not created. ``created`` marks a creation performed by this resolution.
    """

    name: str
    experiment_id: str | None
    artifact_location: str | None = None
    created: bool = False
    planned_creation: bool = False


@dataclass(frozen=True)
class CreatedTrace:
    """The result of one successful trace creation.

    ``span_ids_by_source`` maps every child's source key to its created
    span id (the root is ``root_span_id``); ``verified_metadata`` is the
    exact metadata the gateway submitted at root-span creation — the native
    session key included — which ``verify_trace`` and reconciliation treat
    as the expected metadata contract; ``root_status`` is the effective
    status the gateway applied to the root span (see
    :func:`_effective_root_status`), single-sourced so verification and
    reconciliation expect the same value. All mappings are read-only views.
    """

    trace_id: str
    root_span_id: str
    span_ids_by_source: Mapping[str, str]
    verified_metadata: Mapping[str, str]
    root_status: str
    verified_tags: Mapping[str, str] = field(default_factory=dict)
    artifact_provenance: ArtifactProvenance | None = None


@dataclass(frozen=True)
class OwnedTrace:
    """One importer-owned trace found by ``iter_owned_traces``."""

    trace_id: str
    request_time_ms: int | None
    tags: Mapping[str, str]
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class ArtifactVerification:
    """Uploaded-byte digest; the streaming uploader also binds its manifest fingerprint.

    A bare gateway upload has no replay/manifest knowledge and leaves
    ``snapshot_fingerprint`` unset. The production reconciliation uploader
    supplies it, and the write phase compares it with the mapped request.
    """

    trace_id: str
    relative_path: str
    sha256: str
    size: int
    snapshot_fingerprint: str | None = None


def validate_source_timing(spec: TraceCreateRequest) -> None:
    """MLflow cannot represent absent instants without inventing wall time."""
    if spec.start_time_ns is None or spec.end_time_ns is None:
        raise ItemImportError("source_timestamp_missing", "the target requires source-derived span timestamps")
    for child in spec.child_specs():
        if child.start_time_ns is None or child.end_time_ns is None:
            raise ItemImportError("source_timestamp_missing", "the target requires source-derived span timestamps")


def validate_required_metadata(spec: TraceCreateRequest) -> None:
    """Reject exact source/ownership metadata before the root span starts.

    MLflow 3.17.2 silently truncates trace metadata values at
    :data:`TRACE_METADATA_VALUE_LIMIT`.  Truncation is unsafe for source
    identity, candidate equivalence, and outcome/warning values, so these
    fields fail as an item error while no trace identity exists.  Descriptive
    fields may use the mapper's bounded artifact-reference representation.
    """
    values = dict(spec.metadata)
    values[TRACE_SESSION_METADATA_KEY] = spec.native_session_id
    for key in _EXACT_METADATA_KEYS | {TRACE_SESSION_METADATA_KEY}:
        value = values.get(key)
        if isinstance(value, str) and len(value) > TRACE_METADATA_VALUE_LIMIT:
            raise ItemImportError(
                "ownership_metadata_too_long",
                "a required ownership metadata value exceeds the tracking server limit",
            )
    if "opencode.warnings" in values:
        warning_metadata = values["opencode.warnings"]
        from opencode_mlflow.mlflow.reconciliation import _parse_warning_metadata

        _, valid = _parse_warning_metadata(warning_metadata) if isinstance(warning_metadata, str) else ((), False)
        if not valid:
            raise ItemImportError(
                "ownership_metadata_invalid",
                "a required ownership warning metadata value has an invalid format",
            )


# ---------------------------------------------------------------------------
# Root span status
# ---------------------------------------------------------------------------


def _effective_root_status(spec: TraceCreateRequest, last_child_status: str | None) -> str:
    """The root span status the gateway applies and verifies — the single source.

    The mapping's explicit ``root_status`` (Task 10) is read HERE and
    preferred whenever it is set: the mapping is authoritative over the
    source terminal outcome, so the gateway never re-derives what the
    source already knows (an explicit failure whose last child ended
    ``OK`` stays ``ERROR``). ``None`` keeps the verbatim-field
    derivation: the last child's status when children exist, otherwise
    ``OK`` when a root output is present and ``UNSET`` when it is not.
    """
    if spec.root_status is not None:
        return spec.root_status
    if last_child_status is not None:
        return last_child_status
    return "OK" if spec.root_outputs is not None else "UNSET"


def _span_status_value(status: object) -> str | None:
    """The comparable status string of a retrieved span's status.

    A real retrieved span exposes a ``SpanStatus`` entity (``.status_code``
    enum with a string value); LiveSpan-shaped fakes expose the string
    directly. Returns ``None`` when no status was recorded at all.
    """
    if status is None:
        return None
    code = getattr(status, "status_code", None)
    value = getattr(code, "value", None) if code is not None else getattr(status, "value", status)
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# The behavioral artifact sink
# ---------------------------------------------------------------------------


def _safe_relative_path(relative_path: str) -> PurePosixPath:
    """Validate a caller-chosen artifact relative path (no absolute, no traversal)."""
    rel = PurePosixPath(relative_path)
    if not relative_path or rel.is_absolute() or ".." in rel.parts or str(rel) == ".":
        raise ItemImportError(
            "unsafe_artifact_relative_path",
            "the artifact relative path must be a non-empty relative path without traversal",
        )
    return rel


class TraceArtifactSink:
    """Named trace artifacts through the public composition the profile proved.

    Reimplements the profile's ``upload_named_json``/``verify_named_json``
    behavior identically: the source is read and staged in fixed
    ``chunk_size`` blocks (never loaded whole), the public
    ``ArtifactRepository.log_artifact`` streams the staged file under the
    requested relative path's parent, and verification downloads through
    ``ArtifactRepository.download_artifacts`` and compares the streamed
    sha256 and size. The trace's artifact directory is resolved from the
    public ``mlflow.artifactLocation`` trace tag via the injectable
    repository factory (the public ``mlflow.artifacts`` factory by default).
    UUID-only attachment channels and run-artifact fallbacks are never used.
    """

    def __init__(
        self,
        client: MlflowClient,
        repository_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._client = client
        self._repository_factory: Callable[..., Any] = (
            repository_factory if repository_factory is not None else get_artifact_repository
        )

    def upload_named_json(
        self,
        trace_id: str,
        relative_path: str,
        local_file: Path,
        chunk_size: int = DEFAULT_UPLOAD_CHUNK_BYTES,
    ) -> ArtifactVerification:
        """Upload ``local_file`` as a named artifact at exactly ``relative_path``.

        Returns the digest to feed back into :meth:`verify_named_json`.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        rel = _safe_relative_path(relative_path)
        source = Path(local_file)
        digest = hashlib.sha256()
        size = 0
        with tempfile.TemporaryDirectory(prefix="opencode-mlflow-upload-") as tmp:
            staged = Path(tmp) / rel.name
            with source.open("rb") as src, staged.open("wb") as dst:
                while chunk := src.read(chunk_size):
                    digest.update(chunk)
                    size += len(chunk)
                    dst.write(chunk)
            repository = self._trace_artifact_repository(trace_id)
            parent = str(rel.parent)
            repository.log_artifact(str(staged), artifact_path=parent if parent != "." else None)
        return ArtifactVerification(
            trace_id=trace_id,
            relative_path=str(rel),
            sha256=digest.hexdigest(),
            size=size,
        )

    def verify_named_json(self, trace_id: str, relative_path: str, sha256: str, size: int) -> None:
        """Retrieve the named artifact and verify its exact streamed bytes.

        Returns ``None`` on an exact sha256 and size match; raises a safe
        :class:`~opencode_mlflow.errors.ItemImportError` on any mismatch.
        """
        rel = _safe_relative_path(relative_path)
        repository = self._trace_artifact_repository(trace_id)
        digest = hashlib.sha256()
        seen = 0
        with tempfile.TemporaryDirectory(prefix="opencode-mlflow-verify-") as tmp:
            downloaded = Path(repository.download_artifacts(str(rel), tmp))
            with downloaded.open("rb") as f:
                while chunk := f.read(VERIFY_CHUNK_BYTES):
                    digest.update(chunk)
                    seen += len(chunk)
        if seen != size:
            raise ItemImportError(
                "artifact_verification_failed",
                f"artifact size mismatch: expected {size} bytes, got {seen}",
            )
        if digest.hexdigest() != sha256:
            raise ItemImportError("artifact_verification_failed", "artifact sha256 mismatch")

    def _trace_artifact_repository(self, trace_id: str) -> Any:
        """Resolve the trace's public artifact repository from its location tag."""
        trace = self._client.get_trace(trace_id, display=False, flush=True)
        uri = trace.info.tags.get(TRACE_ARTIFACT_LOCATION_TAG)
        if not uri:
            raise ItemImportError(
                "artifact_location_missing",
                "the trace carries no artifact location tag, so named artifacts cannot be stored for it",
            )
        return self._repository_factory(uri, tracking_uri=self._client.tracking_uri)


def _retirement_metadata_value(
    tags: Mapping[str, str],
    metadata: Mapping[str, str],
    key: str,
) -> str:
    """Read one exact retirement field without treating absent warnings as data."""
    if key == "opencode.warnings":
        return metadata.get(key, "")
    if key in metadata:
        return metadata[key]
    return tags.get(key, "")


# ---------------------------------------------------------------------------
# The gateway
# ---------------------------------------------------------------------------


def build_gateway(
    tracking_uri: str | None = None,
    *,
    dry_run: bool = False,
) -> MlflowGateway:
    """Build the production gateway over a real client for one target.

    The single sanctioned construction seam for callers above the
    gateway (the importer's dependency builder): the client is built
    here, inside the only module allowed to import MLflow, so
    orchestration modules never import MLflow themselves.
    ``tracking_uri=None`` uses the ambient MLflow configuration.
    Production construction deliberately exposes no version-transport
    override: its capability probe always performs the real HTTP attestation.
    """
    return MlflowGateway(MlflowClient(tracking_uri=tracking_uri), dry_run=dry_run)


class MlflowGateway:
    """The only writer boundary: every MLflow mutation flows through here.

    The gateway accepts an ``MlflowClient`` plus the injected
    :class:`PublicTraceApi` (and artifact-repository factory) through
    dependency injection. Mutation gates: the capability check must pass
    before any mutation in normal mode, and a dry-run gateway never
    mutates.
    """

    def __init__(
        self,
        client: MlflowClient,
        trace_api: PublicTraceApi | None = None,
        artifact_repository_factory: Callable[..., Any] | None = None,
        dry_run: bool = False,
        search_page_size: int = 100,
        version_transport: VersionTransport | None = None,
    ) -> None:
        self._client = client
        self._trace_api = trace_api if trace_api is not None else PublicTraceApi()
        self._sink = TraceArtifactSink(client, artifact_repository_factory)
        self._dry_run = dry_run
        self._search_page_size = search_page_size
        self._version_transport = version_transport
        self._preflight: CapabilityReport | None = None
        self._preflight_dry_run: bool | None = None

    # -- capability probe ---------------------------------------------------

    @property
    def tracking_uri(self) -> str:
        """The wrapped client's tracking URI.

        The importer reads it as the lock-identity endpoint input (the
        lock normalizes it, stripping any credentials); it is never
        written into a report.
        """
        return str(self._client.tracking_uri)

    def probe(self, request: CapabilityRequest, dry_run: bool | None = None) -> CapabilityReport:
        """Run the read-only capability probe and store it as the preflight.

        ``dry_run`` defaults to the gateway's own mode. The stored report
        gates later mutations: a normal-mode mutation requires a
        compatible normal-mode probe.
        """
        effective_dry_run = self._dry_run if dry_run is None else dry_run
        report = probe_capabilities(
            self._client,
            request,
            effective_dry_run,
            version_transport=self._version_transport,
        )
        self._preflight = report
        self._preflight_dry_run = effective_dry_run
        return report

    # -- experiment resolution -------------------------------------------------

    def resolve_experiment(self, name: str, create: bool) -> ExperimentRef:
        """Resolve an experiment by exact name; never changes unrelated settings.

        A missing experiment is created only in normal mode and only after
        the complete global capability check; dry-run represents the
        planned creation without mutating.
        """
        found = self._client.get_experiment_by_name(name)
        if found is not None:
            return ExperimentRef(
                name=name,
                experiment_id=found.experiment_id,
                artifact_location=found.artifact_location,
            )
        if not create:
            raise ConfigurationError(
                "experiment_not_found",
                "the requested MLflow experiment does not exist and creation was not requested",
            )
        if self._dry_run:
            return ExperimentRef(name=name, experiment_id=None, planned_creation=True)
        self._require_mutation_allowed()
        experiment_id = self._client.create_experiment(name)
        return ExperimentRef(name=name, experiment_id=experiment_id, created=True)

    # -- trace lifecycle ---------------------------------------------------------

    def create_trace(self, spec: TraceCreateRequest) -> CreatedTrace:
        """Create one trace: a root span plus lazily consumed child spans.

        The root is started through ``mlflow.start_span_no_context`` with
        the request's root payload, tags, and metadata (plus the native
        session metadata key). The lazy child-spec iterator is consumed ONE
        child at a time — the full child list and no source payload is ever
        collected — each child's ``parent_source_key`` is resolved against
        a bounded ``source_key -> LiveSpan`` parent map (one handle per
        child source key in this trace; holds only span handles; created
        per call and released on return), and every span is completed
        through ``LiveSpan.end``. The root ends after all children, with the
        effective status from :func:`_effective_root_status` (the single
        source: the request's explicit ``root_status`` when the mapping set
        one, otherwise the last child's status when children exist, and
        ``OK``/``UNSET`` by root-output presence otherwise), and the applied
        status is exposed as ``CreatedTrace.root_status``.

        The global tracking URI is pinned to the client's target for the
        whole create (module-level span export follows the global URI, not
        the client's) and restored afterwards. A span that degrades to the
        NoOp sentinel fails loudly instead of silently losing the trace.
        """
        self._require_mutation_allowed()
        validate_required_metadata(spec)
        validate_source_timing(spec)
        root_metadata = dict(spec.metadata)
        root_metadata[TRACE_SESSION_METADATA_KEY] = spec.native_session_id
        artifact_provenance = ArtifactProvenance.from_request(spec)
        span_ids_by_source: dict[str, str] = {}
        parents: dict[str, LiveSpanLike] = {}
        last_child_status: str | None = None
        with self._pinned_tracking_uri():
            root = self._trace_api.start_span_no_context(
                name=spec.name,
                span_type=_ROOT_SPAN_TYPE,
                parent_span=None,
                inputs=spec.root_inputs,
                attributes=dict(spec.root_attributes),
                tags=dict(spec.root_tags),
                metadata=root_metadata,
                experiment_id=spec.experiment_id,
                start_time_ns=spec.start_time_ns,
            )
            self._require_real_span(root)
            trace_id = root.trace_id
            root_span_id = root.span_id
            active: LiveSpanLike | None = None
            try:
                for child in spec.child_specs():
                    parent = (
                        root
                        if child.parent_source_key is None
                        else self._resolve_parent(parents, child.parent_source_key)
                    )
                    # Children inherit the experiment; passing experiment_id
                    # here degrades the pinned MLflow provider to NoOp.
                    active = self._trace_api.start_span_no_context(
                        name=child.name,
                        span_type=child.span_type,
                        parent_span=parent,
                        inputs=child.inputs,
                        attributes=dict(child.attributes),
                        tags=dict(child.tags),
                        metadata=None,
                        experiment_id=None,
                        start_time_ns=child.start_time_ns,
                    )
                    self._require_real_span(active)
                    if active.parent_id != parent.span_id:
                        raise ItemImportError("span_parent_mismatch", "a child did not link to its resolved parent")
                    parents[child.source_key] = active
                    span_ids_by_source[child.source_key] = active.span_id
                    self._trace_api.end_span(
                        active,
                        outputs=child.outputs,
                        status=child.status,
                        end_time_ns=child.end_time_ns,
                    )
                    active = None
                    last_child_status = child.status
                root_status = _effective_root_status(spec, last_child_status)
                self._trace_api.end_span(
                    root,
                    outputs=spec.root_outputs,
                    status=root_status,
                    end_time_ns=spec.end_time_ns,
                )
            except Exception as error:
                # Preserve identity even if cleanup/end itself fails. Never
                # substitute a local clock for the source-derived end instant.
                for unfinished in (active, root):
                    if unfinished is not None:
                        try:
                            self._trace_api.end_span(
                                unfinished,
                                outputs=None,
                                status="ERROR",
                                end_time_ns=spec.end_time_ns,
                            )
                        except Exception:
                            pass
                raise PartialTraceCreateError(
                    CreatedTrace(
                        trace_id=trace_id,
                        root_span_id=root_span_id,
                        span_ids_by_source=MappingProxyType(span_ids_by_source),
                        verified_metadata=MappingProxyType(root_metadata),
                        root_status="ERROR",
                        verified_tags=MappingProxyType(dict(spec.root_tags)),
                        artifact_provenance=artifact_provenance,
                    ),
                    error,
                ) from error
        return CreatedTrace(
            trace_id=trace_id,
            root_span_id=root_span_id,
            span_ids_by_source=MappingProxyType(span_ids_by_source),
            verified_metadata=MappingProxyType(root_metadata),
            root_status=root_status,
            verified_tags=MappingProxyType(dict(spec.root_tags)),
            artifact_provenance=artifact_provenance,
        )

    def verify_trace(self, created: CreatedTrace, expected: TraceCreateRequest) -> None:
        """Verify a created trace against its request through FULL retrieval.

        Retrieves the complete trace payload with ``get_trace`` (never a
        search preview) and compares the complete root inputs/outputs, the
        root span name, the root span STATUS (against
        ``created.root_status``, the single-sourced effective status,
        translated through the pinned server's end-time coercion — a root
        ended ``UNSET`` retrieves as ``OK``) and — when the retrieved trace
        info carries one — the trace ``state`` (through the documented
        root-status→state mapping, so an ``UNSET`` root that MLflow
        surfaces as state ``OK`` still verifies), the created span ids,
        the root timing, the native session metadata, and the expected
        tags/metadata. The root ATTRIBUTES are compared as a subset
        (``expected ⊆ actual``): MLflow adds its own internal attribute
        keys to server-side spans, so only the importer's attributes are
        asserted to have survived, each with an equal value. Tag and
        metadata values at or below the documented silent-truncation
        limits are compared exactly; longer values only require the key's
        presence, because the pinned server silently truncates them.
        """
        trace = self._client.get_trace(created.trace_id, display=False, flush=True)
        if trace.trace_id != created.trace_id:
            raise ItemImportError(
                "trace_verification_failed",
                "the retrieved trace id does not match the created trace",
            )
        expected_children: dict[str, _ExpectedChildContract] = {}
        expected_source_keys: set[str] = set()
        for child in expected.child_specs():
            if child.source_key in expected_source_keys:
                raise ItemImportError("trace_verification_failed", "duplicate expected child source keys")
            expected_source_keys.add(child.source_key)
            span_id = created.span_ids_by_source.get(child.source_key)
            if span_id is None or span_id in expected_children:
                raise ItemImportError("trace_verification_failed", "the expected child identity map is inconsistent")
            expected_children[span_id] = _ExpectedChildContract.from_request(child)
        if set(created.span_ids_by_source) != expected_source_keys:
            raise ItemImportError("trace_verification_failed", "the expected child identity set did not round-trip")

        root: Any | None = None
        seen_child_ids: set[str] = set()
        for span in trace.data.spans:
            if span.span_id == created.root_span_id:
                if root is not None:
                    raise ItemImportError(
                        "trace_verification_failed", "the retrieved trace contains duplicate root spans"
                    )
                root = span
                continue
            expected_child = expected_children.get(span.span_id)
            if expected_child is None:
                raise ItemImportError(
                    "trace_verification_failed", "the retrieved trace contains an unexpected child span"
                )
            if span.span_id in seen_child_ids:
                raise ItemImportError(
                    "trace_verification_failed", "the retrieved trace contains a duplicate child span"
                )
            seen_child_ids.add(span.span_id)
            expected_parent_id = created.root_span_id
            if expected_child.parent_source_key is not None:
                resolved_parent_id = created.span_ids_by_source.get(expected_child.parent_source_key)
                if resolved_parent_id is None:
                    raise ItemImportError("trace_verification_failed", "an expected child parent identity is missing")
                expected_parent_id = resolved_parent_id
            if span.name != expected_child.name:
                raise ItemImportError("trace_verification_failed", "a child span name did not round-trip")
            if span.span_type != expected_child.span_type:
                raise ItemImportError("trace_verification_failed", "a child span type did not round-trip")
            if span.inputs != expected_child.inputs:
                raise ItemImportError("trace_verification_failed", "a child span input did not round-trip")
            if span.outputs != expected_child.outputs:
                raise ItemImportError("trace_verification_failed", "a child span output did not round-trip")
            if span.parent_id != expected_parent_id:
                raise ItemImportError("trace_verification_failed", "a child span parent did not round-trip")
            if _span_status_value(span.status) != RETRIEVED_SPAN_STATUS.get(
                expected_child.status, expected_child.status
            ):
                raise ItemImportError("trace_verification_failed", "a child span status did not round-trip")
            if span.start_time_ns != expected_child.start_time_ns:
                raise ItemImportError("trace_verification_failed", "a child span start time did not round-trip")
            if span.end_time_ns != expected_child.end_time_ns:
                raise ItemImportError("trace_verification_failed", "a child span end time did not round-trip")
            child_attributes = span.attributes or {}
            for key, value in expected_child.attributes.items():
                if key not in child_attributes or child_attributes[key] != value:
                    raise ItemImportError("trace_verification_failed", "a child span attribute did not round-trip")
            self._verify_labeled_values(
                expected_child.tags,
                getattr(span, "tags", {}) or {},
                TRACE_TAG_VALUE_LIMIT,
                "child tag",
            )
        if root is None:
            raise ItemImportError("trace_verification_failed", "the root span is missing from the retrieved trace")
        if seen_child_ids != set(expected_children):
            raise ItemImportError(
                "trace_verification_failed", "a created child span is missing from the retrieved trace"
            )
        if root.name != expected.name:
            raise ItemImportError("trace_verification_failed", "the root span name did not round-trip")
        if root.inputs != expected.root_inputs:
            raise ItemImportError(
                "trace_verification_failed",
                "the complete root inputs did not round-trip through full-payload retrieval",
            )
        if root.outputs != expected.root_outputs:
            raise ItemImportError(
                "trace_verification_failed",
                "the complete root outputs did not round-trip through full-payload retrieval",
            )
        # The pinned server's LiveSpan.end forces every non-ERROR status to
        # OK at end time, so a root ended UNSET retrieves as OK (the
        # source-known status stays on CreatedTrace and in the outcome
        # metadata); the comparison translates through that coercion.
        expected_status = RETRIEVED_SPAN_STATUS.get(created.root_status, created.root_status)
        if _span_status_value(root.status) != expected_status:
            raise ItemImportError("trace_verification_failed", "the root span status did not round-trip")
        state = getattr(trace.info, "state", None)
        if state is not None:
            expected_state = TRACE_STATE_FOR_ROOT_STATUS.get(created.root_status, created.root_status)
            if str(getattr(state, "value", state)) != expected_state:
                raise ItemImportError(
                    "trace_verification_failed",
                    "the trace state did not match the root span status",
                )
        attributes = root.attributes or {}
        for key, value in expected.root_attributes.items():
            if key not in attributes or attributes[key] != value:
                raise ItemImportError(
                    "trace_verification_failed",
                    "an expected root span attribute did not round-trip",
                )
        if expected.start_time_ns is not None and root.start_time_ns != expected.start_time_ns:
            raise ItemImportError("trace_verification_failed", "the trace start time did not round-trip")
        if expected.end_time_ns is not None and root.end_time_ns != expected.end_time_ns:
            raise ItemImportError("trace_verification_failed", "the trace end time did not round-trip")
        metadata = trace.info.trace_metadata
        if metadata.get(TRACE_SESSION_METADATA_KEY) != expected.native_session_id:
            raise ItemImportError("trace_verification_failed", "the native session metadata did not round-trip")
        self._verify_labeled_values(expected.root_tags, trace.info.tags, TRACE_TAG_VALUE_LIMIT, "tag")
        self._verify_labeled_values(expected.metadata, metadata, TRACE_METADATA_VALUE_LIMIT, "metadata")

    def iter_owned_traces(self, experiment_id: str) -> Iterator[OwnedTrace]:
        """Yield every importer-owned trace in the experiment, page by page.

        Searches with the exact server-side ownership metadata filter when
        the target accepts it (falling back to unfiltered pages otherwise),
        walks the pagination tokens to exhaustion, and re-verifies the
        exact ownership marker on every candidate locally — an unowned
        trace is never yielded.
        """
        filter_string: str | None = _OWNERSHIP_FILTER
        try:
            page = self._search_page(experiment_id, filter_string, page_token=None)
        except Exception:  # noqa: BLE001 - a rejected filter falls back to local filtering
            filter_string = None
            page = self._search_page(experiment_id, None, page_token=None)
        if filter_string is not None and len(page) == 0:
            # The target may accept the filter syntactically but interpret it
            # differently, or the ownership marker may live in tags rather
            # than metadata. A filter that silently matches nothing is not
            # meaningfully available: fall back to unfiltered pages with the
            # local ownership check so no owned trace is missed.
            filter_string = None
            page = self._search_page(experiment_id, None, page_token=None)
        seen_tokens: set[str] = set()
        while True:
            for trace in page:
                if self._is_owned(trace):
                    yield OwnedTrace(
                        trace_id=trace.trace_id,
                        request_time_ms=trace.info.request_time,
                        tags=MappingProxyType(dict(trace.info.tags)),
                        metadata=MappingProxyType(dict(trace.info.trace_metadata)),
                    )
            token = getattr(page, "token", None)
            if not token:
                return
            if token in seen_tokens:
                raise CapabilityError("mlflow_pagination_cycle", "the trace search repeated a pagination token")
            seen_tokens.add(token)
            page = self._search_page(experiment_id, filter_string, page_token=token)

    def verify_owned_candidate(
        self,
        candidate: OwnedTraceCandidate,
        expected: CandidateContract,
    ) -> CandidateVerification:
        """Verify one importer-owned candidate against the expected contract.

        Retrieves the FULL trace payload through public ``get_trace``
        (never a search preview), re-verifies the exact ownership marker
        on the retrieved trace, and compares the complete creation-time
        ownership contract — importer/session/turn identity, turn index,
        source/schema, importer/mapping/artifact versions, redaction and
        catalog versions, render/operation fingerprints, warning metadata,
        and experiment identity — against the contract. The artifact check
        is an INTEGRITY check (Ruling
        16) at the CANDIDATE'S OWN RECORDED LOCATION: the digest, size,
        and exact relative path RECORDED on the trace
        (``opencode.artifact_sha256``/``opencode.artifact_size``/
        ``opencode.artifact_path``) are read from the full retrieval and
        the stored artifact's exact bytes are verified against them
        through the public :meth:`verify_named_json` — the current run's
        would-be path (the contract's ``artifact_relative_path``) is
        deliberately NOT used here, because the deterministic path
        embeds the session-complete artifact-snapshot fingerprint and
        would move for every sibling turn whenever any turn of the
        session changed. A trace without a usable recorded path or
        digest is an incomplete candidate (a pre-tag trace, or a crash
        between upload and recording) and is returned as a completed
        non-match. A missing or changed importer marker is returned as an
        ``ownership_conflict`` error, so reconciliation never replaces or
        retires that trace. A completed-but-non-matching verification is
        otherwise RETURNED (``matches=False`` with a reason); only a
        retrieval or transport failure raises. Reconciliation calls this for
        every candidate before any winner selection.
        """
        from opencode_mlflow.mlflow.reconciliation import (
            ARTIFACT_PATH_TAG,
            ARTIFACT_SHA256_TAG,
            ARTIFACT_SIZE_TAG,
            OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY,
            OWNERSHIP_CATALOG_VERSION_KEY,
            OWNERSHIP_IMPORTER_KEY,
            OWNERSHIP_IMPORTER_VERSION_KEY,
            OWNERSHIP_MAPPING_VERSION_KEY,
            OWNERSHIP_OPERATION_KEY,
            OWNERSHIP_REDACTION_MODE_KEY,
            OWNERSHIP_RENDER_FINGERPRINT_KEY,
            OWNERSHIP_SESSION_KEY,
            OWNERSHIP_SOURCE_FINGERPRINT_KEY,
            OWNERSHIP_SOURCE_SCHEMA_KEY,
            OWNERSHIP_TURN_INDEX_KEY,
            OWNERSHIP_TURN_KEY,
            CandidateVerification,
            _candidate_importer_marker,
            _ownership_contract_errors,
            _parse_turn_index,
            _parse_warning_metadata,
        )

        if not candidate.ownership_complete or expected.ownership_errors:
            return CandidateVerification(
                trace_id=candidate.trace_id,
                matches=False,
                reason="ownership_incomplete",
                error_code="ownership_conflict",
            )
        trace = self._client.get_trace(candidate.trace_id, display=False, flush=True)
        if trace.trace_id != candidate.trace_id:
            raise ItemImportError(
                "trace_verification_failed",
                "the retrieved candidate trace id does not match the indexed candidate",
            )
        tags = trace.info.tags
        metadata = trace.info.trace_metadata
        if not self._has_exact_ownership_marker(tags, metadata):
            return CandidateVerification(
                trace_id=candidate.trace_id,
                matches=False,
                reason="ownership_missing",
                error_code="ownership_conflict",
            )
        _, warning_metadata_valid = _parse_warning_metadata(metadata.get("opencode.warnings"))
        contract_errors = _ownership_contract_errors(
            trace_id=trace.trace_id,
            experiment_id=getattr(trace.info, "experiment_id", None),
            importer_marker=_candidate_importer_marker(tags, metadata),
            session_id=metadata.get(OWNERSHIP_SESSION_KEY),
            turn_key=metadata.get(OWNERSHIP_TURN_KEY),
            turn_index=_parse_turn_index(metadata.get("opencode.turn_index")),
            source_fingerprint=metadata.get(OWNERSHIP_SOURCE_FINGERPRINT_KEY),
            source_schema=metadata.get(OWNERSHIP_SOURCE_SCHEMA_KEY),
            importer_version=metadata.get(OWNERSHIP_IMPORTER_VERSION_KEY),
            mapping_version=metadata.get(OWNERSHIP_MAPPING_VERSION_KEY),
            artifact_schema_version=metadata.get(OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY),
            redaction_mode=metadata.get(OWNERSHIP_REDACTION_MODE_KEY),
            catalog_version=metadata.get(OWNERSHIP_CATALOG_VERSION_KEY),
            render_fingerprint=metadata.get(OWNERSHIP_RENDER_FINGERPRINT_KEY),
            operation_key=metadata.get(OWNERSHIP_OPERATION_KEY),
            warning_metadata_valid=warning_metadata_valid,
        )
        if contract_errors:
            return CandidateVerification(
                trace_id=candidate.trace_id,
                matches=False,
                reason="ownership_incomplete",
                error_code="ownership_conflict",
            )
        for key, expected_identity in (
            (OWNERSHIP_SESSION_KEY, candidate.session_id),
            (OWNERSHIP_TURN_KEY, candidate.turn_key),
        ):
            if metadata.get(key) != expected_identity:
                return CandidateVerification(
                    trace_id=candidate.trace_id,
                    matches=False,
                    reason="ownership_identity_changed",
                    error_code="ownership_conflict",
                )
        expected_fields: tuple[tuple[str, str | None, str | None], ...] = (
            (OWNERSHIP_IMPORTER_KEY, _candidate_importer_marker(tags, metadata), expected.importer_marker),
            (OWNERSHIP_SESSION_KEY, metadata.get(OWNERSHIP_SESSION_KEY), expected.session_id),
            (OWNERSHIP_TURN_INDEX_KEY, metadata.get(OWNERSHIP_TURN_INDEX_KEY), str(expected.turn_index)),
            (OWNERSHIP_TURN_KEY, metadata.get(OWNERSHIP_TURN_KEY), expected.turn_key),
            (
                OWNERSHIP_SOURCE_FINGERPRINT_KEY,
                metadata.get(OWNERSHIP_SOURCE_FINGERPRINT_KEY),
                expected.source_fingerprint,
            ),
            (OWNERSHIP_SOURCE_SCHEMA_KEY, metadata.get(OWNERSHIP_SOURCE_SCHEMA_KEY), expected.source_schema),
            (OWNERSHIP_IMPORTER_VERSION_KEY, metadata.get(OWNERSHIP_IMPORTER_VERSION_KEY), expected.importer_version),
            (OWNERSHIP_MAPPING_VERSION_KEY, metadata.get(OWNERSHIP_MAPPING_VERSION_KEY), expected.mapping_version),
            (
                OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY,
                metadata.get(OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY),
                expected.artifact_schema_version,
            ),
            (OWNERSHIP_REDACTION_MODE_KEY, metadata.get(OWNERSHIP_REDACTION_MODE_KEY), expected.redaction_mode),
            (OWNERSHIP_CATALOG_VERSION_KEY, metadata.get(OWNERSHIP_CATALOG_VERSION_KEY), expected.catalog_version),
            (
                OWNERSHIP_RENDER_FINGERPRINT_KEY,
                metadata.get(OWNERSHIP_RENDER_FINGERPRINT_KEY),
                expected.render_fingerprint,
            ),
            (OWNERSHIP_OPERATION_KEY, metadata.get(OWNERSHIP_OPERATION_KEY), expected.operation_key),
        )
        if getattr(trace.info, "experiment_id", None) != expected.experiment_id:
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="metadata_mismatch")
        for _key, actual, value in expected_fields:
            if actual != value:
                return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="metadata_mismatch")
        actual_warnings, actual_warning_metadata_valid = _parse_warning_metadata(metadata.get("opencode.warnings"))
        if not actual_warning_metadata_valid:
            return CandidateVerification(
                trace_id=candidate.trace_id,
                matches=False,
                reason="ownership_incomplete",
                error_code="ownership_conflict",
            )
        if actual_warnings != expected.warnings:
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="metadata_mismatch")
        recorded_path = tags.get(ARTIFACT_PATH_TAG) or metadata.get(ARTIFACT_PATH_TAG)
        if not isinstance(recorded_path, str) or not recorded_path:
            # An incomplete candidate: the artifact location was never
            # recorded (a trace written before the path tag existed), so its
            # integrity cannot be verified at its own location. This is a
            # completed non-match, never an error.
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="artifact_path_missing")
        recorded_sha256 = tags.get(ARTIFACT_SHA256_TAG) or metadata.get(ARTIFACT_SHA256_TAG)
        recorded_size_text = tags.get(ARTIFACT_SIZE_TAG) or metadata.get(ARTIFACT_SIZE_TAG)
        if not recorded_sha256 or recorded_size_text is None:
            # An incomplete candidate: the artifact was never recorded (a
            # crash between upload and recording), so its integrity cannot be
            # verified. This is a completed non-match, never an error.
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="artifact_digest_missing")
        try:
            recorded_size = int(recorded_size_text)
        except ValueError:
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="artifact_digest_missing")
        try:
            self.verify_named_json(
                candidate.trace_id,
                recorded_path,
                recorded_sha256,
                recorded_size,
            )
        except ItemImportError:
            # A stored artifact that diverges from its recorded digest is a
            # completed non-match (tampered or replaced content).
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="artifact_mismatch")
        return CandidateVerification(trace_id=candidate.trace_id, matches=True)

    def verify_owned_trace_for_retirement(
        self,
        experiment_id: str,
        trace_id: str,
        *,
        session_id: str,
        turn_key: str,
        expected_metadata: Mapping[str, str] | None = None,
    ) -> Mapping[str, str]:
        """Re-fetch and validate one candidate immediately before retirement.

        Search results are only an index. The trace may lose its importer
        marker, change experiment, or be replaced under the same native
        identity before cleanup runs. A failed or ambiguous read is a safe
        conflict, never permission to delete by id.
        """
        try:
            trace = self._client.get_trace(trace_id, display=False, flush=True)
        except Exception as error:  # noqa: BLE001 - an unverifiable target is unsafe to delete
            raise ItemImportError(
                "ownership_conflict",
                "the trace ownership could not be re-verified before retirement",
            ) from error
        complete_expected_metadata = self._complete_retirement_metadata(trace, expected_metadata)
        self._validate_retirement_target(
            trace,
            experiment_id,
            trace_id,
            session_id=session_id,
            turn_key=turn_key,
            expected_metadata=complete_expected_metadata,
        )
        return complete_expected_metadata

    def record_artifact_verification(self, trace_id: str, sha256: str, size: int) -> None:
        """Record one verified artifact digest on its trace (public tag update).

        Sets the ``opencode.artifact_sha256``/``opencode.artifact_size``
        tags through the public ``set_trace_tag`` operation (an operation the
        capability probe checks for availability), so a later run can verify
        the stored artifact's INTEGRITY against the digest recorded when it
        was written (Ruling 16). The two tag updates are not atomic: a crash
        between them leaves an incomplete candidate with no usable recorded
        digest, which the next run treats as unverifiable and replaces.
        Mutation-gated like every other gateway mutation.
        """
        from opencode_mlflow.mlflow.reconciliation import ARTIFACT_SHA256_TAG, ARTIFACT_SIZE_TAG

        self._require_mutation_allowed()
        self._client.set_trace_tag(trace_id, ARTIFACT_SHA256_TAG, sha256)
        self._client.set_trace_tag(trace_id, ARTIFACT_SIZE_TAG, str(size))

    # -- artifacts ---------------------------------------------------------

    def upload_named_json(
        self,
        trace_id: str,
        relative_path: str,
        local_file: Path,
        chunk_size: int = DEFAULT_UPLOAD_CHUNK_BYTES,
    ) -> ArtifactVerification:
        """Upload a named artifact at exactly ``relative_path`` (see the sink)."""
        self._require_mutation_allowed()
        return self._sink.upload_named_json(trace_id, relative_path, local_file, chunk_size)

    def verify_named_json(self, trace_id: str, relative_path: str, sha256: str, size: int) -> None:
        """Verify a named artifact's exact bytes (see the sink)."""
        self._sink.verify_named_json(trace_id, relative_path, sha256, size)

    # -- retirement --------------------------------------------------------

    def retire_trace(
        self,
        experiment_id: str,
        trace_id: str,
        *,
        session_id: str | None = None,
        turn_key: str | None = None,
        expected_metadata: Mapping[str, str] | None = None,
    ) -> None:
        """Retire one trace through the public ``delete_traces`` API."""
        self._require_mutation_allowed()
        if session_id is None or turn_key is None:
            raise ItemImportError(
                "ownership_conflict",
                "the trace source identity was not supplied for retirement",
            )
        try:
            # The first check detects an already-invalid search candidate.
            # It is intentionally not the delete authorization: flushing
            # queued exports can change the target, so the final check below
            # repeats every identity and contract comparison.
            verified_metadata = self.verify_owned_trace_for_retirement(
                experiment_id,
                trace_id,
                session_id=session_id,
                turn_key=turn_key,
                expected_metadata=expected_metadata,
            )
            if verified_metadata is not None:
                expected_metadata = verified_metadata

            # Drain queued exports before the FINAL ownership check. The
            # final retrieval and validation are kept directly adjacent to the
            # public delete call; MLflow exposes no conditional-delete or
            # transaction API, so this is the strongest atomicity available at
            # the gateway boundary.
            with self._pinned_tracking_uri():
                mlflow.flush_trace_async_logging()
            final_trace = self._client.get_trace(trace_id, display=False, flush=True)
            self._validate_retirement_target(
                final_trace,
                experiment_id,
                trace_id,
                session_id=session_id,
                turn_key=turn_key,
                expected_metadata=expected_metadata,
            )
        except Exception as error:  # noqa: BLE001 - preserve the legacy missing-id error contract
            if isinstance(error, ItemImportError):
                raise
            raise ItemImportError(
                "ownership_conflict",
                "the trace ownership could not be re-verified before retirement",
            ) from error
        deleted = self._client.delete_traces(experiment_id=experiment_id, trace_ids=[trace_id])
        if deleted != 1:
            raise ItemImportError(
                "trace_retirement_failed",
                "the tracking server did not report the trace as retired",
            )

    @classmethod
    def _validate_retirement_target(
        cls,
        trace: Any,
        experiment_id: str,
        trace_id: str,
        *,
        session_id: str,
        turn_key: str,
        expected_metadata: Mapping[str, str] | None,
    ) -> None:
        """Validate every identity component immediately before retirement."""
        if trace.trace_id != trace_id or getattr(trace.info, "experiment_id", None) != experiment_id:
            raise ItemImportError("ownership_conflict", "the trace identity changed before retirement")
        tags = trace.info.tags
        metadata = trace.info.trace_metadata
        if not cls._has_exact_ownership_marker(tags, metadata):
            raise ItemImportError("ownership_conflict", "the importer ownership marker changed before retirement")
        from opencode_mlflow.mlflow.reconciliation import (
            OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY,
            OWNERSHIP_IMPORTER_VERSION_KEY,
            OWNERSHIP_MAPPING_VERSION_KEY,
            OWNERSHIP_REDACTION_MODE_KEY,
            _candidate_importer_marker,
            _ownership_contract_errors,
            _parse_turn_index,
            _parse_warning_metadata,
        )

        _, warning_metadata_valid = _parse_warning_metadata(metadata.get("opencode.warnings"))
        contract_errors = _ownership_contract_errors(
            trace_id=trace.trace_id,
            experiment_id=getattr(trace.info, "experiment_id", None),
            importer_marker=_candidate_importer_marker(tags, metadata),
            session_id=metadata.get("opencode.session_id"),
            turn_key=metadata.get("opencode.turn_key"),
            turn_index=_parse_turn_index(metadata.get("opencode.turn_index")),
            source_fingerprint=metadata.get("opencode.source_fingerprint"),
            source_schema=metadata.get("opencode.source_schema"),
            importer_version=metadata.get(OWNERSHIP_IMPORTER_VERSION_KEY),
            mapping_version=metadata.get(OWNERSHIP_MAPPING_VERSION_KEY),
            artifact_schema_version=metadata.get(OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY),
            redaction_mode=metadata.get(OWNERSHIP_REDACTION_MODE_KEY),
            catalog_version=metadata.get("opencode.catalog_version"),
            render_fingerprint=metadata.get("opencode.render_fingerprint"),
            operation_key=metadata.get("opencode.operation_key"),
            warning_metadata_valid=warning_metadata_valid,
        )
        if contract_errors:
            raise ItemImportError("ownership_conflict", "the trace ownership contract could not be validated")
        for key, expected in (("opencode.session_id", session_id), ("opencode.turn_key", turn_key)):
            values = [mapping[key] for mapping in (metadata, tags) if key in mapping]
            if not values or any(value != expected for value in values):
                raise ItemImportError("ownership_conflict", "the trace source identity changed before retirement")
        if expected_metadata is not None:
            for key, expected in expected_metadata.items():
                actual = _retirement_metadata_value(tags, metadata, key)
                if actual != expected:
                    raise ItemImportError(
                        "ownership_conflict", "the trace candidate contract changed before retirement"
                    )

    @staticmethod
    def _complete_retirement_metadata(
        trace: Any,
        expected_metadata: Mapping[str, str] | None,
    ) -> dict[str, str]:
        """Freeze every exact ownership field from the first safe fetch."""
        from opencode_mlflow.mlflow.reconciliation import (
            OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY,
            OWNERSHIP_CATALOG_VERSION_KEY,
            OWNERSHIP_IMPORTER_KEY,
            OWNERSHIP_IMPORTER_VERSION_KEY,
            OWNERSHIP_MAPPING_VERSION_KEY,
            OWNERSHIP_OPERATION_KEY,
            OWNERSHIP_REDACTION_MODE_KEY,
            OWNERSHIP_RENDER_FINGERPRINT_KEY,
            OWNERSHIP_SESSION_KEY,
            OWNERSHIP_SOURCE_FINGERPRINT_KEY,
            OWNERSHIP_SOURCE_SCHEMA_KEY,
            OWNERSHIP_TURN_INDEX_KEY,
            OWNERSHIP_TURN_KEY,
        )

        tags = trace.info.tags
        metadata = trace.info.trace_metadata
        keys = (
            OWNERSHIP_IMPORTER_KEY,
            OWNERSHIP_SESSION_KEY,
            OWNERSHIP_TURN_INDEX_KEY,
            OWNERSHIP_TURN_KEY,
            OWNERSHIP_SOURCE_FINGERPRINT_KEY,
            OWNERSHIP_SOURCE_SCHEMA_KEY,
            OWNERSHIP_IMPORTER_VERSION_KEY,
            OWNERSHIP_MAPPING_VERSION_KEY,
            OWNERSHIP_ARTIFACT_SCHEMA_VERSION_KEY,
            OWNERSHIP_REDACTION_MODE_KEY,
            OWNERSHIP_CATALOG_VERSION_KEY,
            OWNERSHIP_RENDER_FINGERPRINT_KEY,
            OWNERSHIP_OPERATION_KEY,
            "opencode.warnings",
        )
        supplied = expected_metadata or {}
        return {
            key: supplied[key] if key in supplied else _retirement_metadata_value(tags, metadata, key) for key in keys
        }

    # -- internals -----------------------------------------------------------

    def _search_page(self, experiment_id: str, filter_string: str | None, page_token: str | None) -> Any:
        """One search page in the experiment (the non-deprecated ``locations`` form)."""
        return self._client.search_traces(
            locations=[experiment_id],
            filter_string=filter_string,
            max_results=self._search_page_size,
            page_token=page_token,
            include_spans=False,
        )

    @staticmethod
    def _is_owned(trace: Any) -> bool:
        """The exact ownership marker in trace tags or metadata — never a guess."""
        tags = trace.info.tags
        metadata = trace.info.trace_metadata
        return bool(
            tags.get(OWNERSHIP_TAG_KEY) == OWNERSHIP_IMPORTER_VALUE
            or metadata.get(OWNERSHIP_TAG_KEY) == OWNERSHIP_IMPORTER_VALUE
        )

    @staticmethod
    def _has_exact_ownership_marker(tags: Mapping[str, str], metadata: Mapping[str, str]) -> bool:
        """Require every present ownership marker to carry the exact value."""
        values = [mapping[OWNERSHIP_TAG_KEY] for mapping in (tags, metadata) if OWNERSHIP_TAG_KEY in mapping]
        return bool(values) and all(value == OWNERSHIP_IMPORTER_VALUE for value in values)

    @staticmethod
    def _verify_labeled_values(
        expected: Mapping[str, str],
        actual: Mapping[str, str],
        limit: int,
        label: str,
    ) -> None:
        """Compare labeled values, tolerating the server's silent truncation."""
        for key, value in expected.items():
            if key not in actual:
                raise ItemImportError(
                    "trace_verification_failed",
                    f"an expected trace {label} key is missing from the retrieved trace",
                )
            if len(value) <= limit and actual[key] != value:
                raise ItemImportError(
                    "trace_verification_failed",
                    f"an expected trace {label} value did not round-trip",
                )

    @staticmethod
    def _resolve_parent(parents: Mapping[str, LiveSpanLike], parent_source_key: str) -> LiveSpanLike:
        parent = parents.get(parent_source_key)
        if parent is None:
            raise ItemImportError(
                "unresolvable_parent_span",
                "a child span referenced a parent source key that was not created",
            )
        return parent

    @staticmethod
    def _require_real_span(span: LiveSpanLike) -> None:
        if span.trace_id == NO_OP_TRACE_ID:
            raise CapabilityError(
                "mlflow_span_degraded",
                "span creation degraded to a no-op span; ambient OTEL exporter configuration "
                "may have rerouted span export away from the tracking server",
            )

    @contextmanager
    def _pinned_tracking_uri(self) -> Iterator[None]:
        """Pin the global tracking URI to the client's target for span work.

        Module-level span export follows ``mlflow.get_tracking_uri()``, not
        any client's URI; without pinning, spans created for a client's
        target could silently export elsewhere. The saved value is always
        restored.
        """
        saved = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(self._client.tracking_uri)
        try:
            yield
        finally:
            mlflow.set_tracking_uri(saved)

    def _require_mutation_allowed(self) -> None:
        if self._dry_run:
            raise CapabilityError(
                "dry_run_mutation",
                "the gateway is in dry-run mode and must not mutate the tracking target",
            )
        if self._preflight is None:
            raise CapabilityError(
                "mlflow_preflight_required",
                "the capability check must pass before any MLflow mutation",
            )
        if not self._preflight.is_compatible:
            raise CapabilityError(
                "mlflow_capability_unsupported",
                self._preflight.safe_error or "the tracking target failed the capability check",
            )
        if self._preflight_dry_run:
            raise CapabilityError(
                "mlflow_preflight_required",
                "a normal-mode mutation requires a normal-mode capability check",
            )
