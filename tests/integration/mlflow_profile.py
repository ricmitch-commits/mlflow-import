"""Pinned MLflow compatibility profile for the opencode-mlflow importer.

This is a library module (not a test). It records the exact public MLflow
surface the importer's writer will rely on and verifies — behaviorally, against
a live tracking server — that the pinned client/server pair provides it:

* experiment lookup and creation (``MlflowClient.get_experiment_by_name``,
  ``MlflowClient.create_experiment``);
* context-free span creation (module-level ``mlflow.start_span_no_context``)
  and completion (``LiveSpan.end``);
* full-payload trace retrieval (``MlflowClient.get_trace``) and paginated
  search (``MlflowClient.search_traces``);
* public trace tag update (``MlflowClient.set_trace_tag`` /
  ``MlflowClient.delete_trace_tag``);
* retirement (``MlflowClient.delete_traces(experiment_id, trace_ids=[...])``);
* named trace artifacts stored at a caller-chosen relative path with
  exact-byte verification, adapted to the ``TraceArtifactSink`` contract as
  ``upload_named_json`` / ``verify_named_json``.

``verify_profile`` comes in two flavors. The default (live) flavor exercises
every operation for real, including mutation, and therefore requires a
disposable server. The ``dry_run=True`` flavor probes method availability,
server reachability, protocol responses, full-payload retrieval, the artifact
channel, and retirement *availability* strictly through read-only calls: it
never creates an experiment, trace, or artifact.

Every resolved name, signature, and observed behavior is documented in
``docs/compatibility.md``; the reproducible server recipe lives in
``tests/integration/README.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import get_artifact_repository
from mlflow.entities import LiveSpan
from mlflow.entities.span import NO_OP_SPAN_TRACE_ID
from mlflow.exceptions import MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION

from opencode_mlflow.mlflow.capabilities import attest_server_version

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "MAX_INLINE_PAYLOAD_BYTES",
    "PINNED_MLFLOW_VERSION",
    "REQUIRED_METHODS",
    "ArtifactVerification",
    "CapabilityReport",
    "MlflowProfile",
    "upload_named_json",
    "verify_named_json",
    "verify_profile",
]

#: The exact stable MLflow release the profile pins (same API line as the
#: plan's unpublished ``3.17.2.dev0``; see the controller ruling of 2026-10-21).
PINNED_MLFLOW_VERSION = "3.17.2"

#: Largest inline payload (bytes) the profile verifies to round-trip byte-exact
#: through span inputs/outputs. Larger session payloads must be written as
#: named trace artifacts instead of inline span data.
MAX_INLINE_PAYLOAD_BYTES = 1_048_576

#: Fixed read block (bytes) used when hashing, staging, and re-reading artifact
#: payloads so no whole file is ever held in memory.
DEFAULT_CHUNK_SIZE = 65_536

#: Gateway-facing operations the writer depends on. ``verify_profile`` only
#: reports an operation in ``CapabilityReport.required_methods`` after its
#: behavioral proof succeeded (availability alone is not enough in live mode).
REQUIRED_METHODS: frozenset[str] = frozenset(
    {
        "server_version_attestation",
        "get_experiment_by_name",
        "create_experiment",
        "mlflow.start_span_no_context",
        "LiveSpan.end",
        "get_trace",
        "search_traces",
        "delete_traces",
        "upload_named_json",
        "verify_named_json",
    }
)

#: Ambient OTEL exporter configuration that silently reroutes MLflow span
#: export to an external OTLP collector. When the OTLP HTTP exporter package
#: is not installed, span creation degrades to NoOp spans and traces never
#: reach the tracking server. Scrubbed before the tracing provider initializes.
OTEL_HIJACK_ENV_VARS: tuple[str, ...] = (
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
)

#: Canonical command shape for the disposable server. ``<run-dir>`` is a fresh
#: directory per run and ``<free-port>`` a programmatically chosen port; the
#: 3.17.2 server serves proxied artifacts (``--serve-artifacts`` is on by
#: default) rooted at ``--artifacts-destination``.
INITIAL_SERVER_COMMAND: tuple[str, ...] = (
    "mlflow",
    "server",
    "--backend-store-uri",
    "sqlite:///<run-dir>/mlflow.db",
    "--artifacts-destination",
    "file://<run-dir>/artifacts",
    "--host",
    "127.0.0.1",
    "--port",
    "<free-port>",
)

#: Resolved public API surface of the pinned pair, keyed by the gateway-facing
#: operation name. The values are the exact resolved callables/parameter lists
#: observed against mlflow 3.17.2 (see docs/compatibility.md for behavior).
INITIAL_API_TABLE: Mapping[str, str] = {
    "server_version_attestation": (
        "GET <tracking-base>/version -> plain text '3.17.2' (JSON {version: '3.17.2'} also accepted)"
    ),
    "get_experiment_by_name": "MlflowClient.get_experiment_by_name(name: str) -> Experiment | None",
    "create_experiment": (
        "MlflowClient.create_experiment(name: str, artifact_location: str | None = None, "
        "tags: dict | None = None) -> str"
    ),
    "mlflow.start_span_no_context": (
        "mlflow.start_span_no_context(name: str, span_type: str = 'UNKNOWN', parent_span: LiveSpan | None"
        " = None, inputs=None, attributes=None, tags=None, metadata=None, experiment_id: str | None = None,"
        " start_time_ns: int | None = None, log_level=None, links=None, description=None) -> LiveSpan"
    ),
    "LiveSpan.end": (
        "LiveSpan.end(outputs=None, attributes: dict | None = None, status: SpanStatus | str | None = None,"
        " end_time_ns: int | None = None) -> None"
    ),
    "get_trace": ("MlflowClient.get_trace(trace_id: str, display: bool = True, flush: bool = False) -> Trace"),
    "search_traces": (
        "MlflowClient.search_traces(experiment_ids: list[str] | None = None, filter_string: str | None ="
        " None, max_results: int = 100, order_by=None, page_token: str | None = None, run_id=None,"
        " include_spans: bool = True, model_id=None, locations: list[str] | None = None, flush: bool ="
        " False) -> PagedList[Trace]  # 'experiment_ids' is deprecated in favor of 'locations'"
    ),
    "set_trace_tag": "MlflowClient.set_trace_tag(trace_id: str, key: str, value: str) -> None",
    "delete_trace_tag": "MlflowClient.delete_trace_tag(trace_id: str, key: str) -> None",
    "delete_traces": (
        "MlflowClient.delete_traces(experiment_id: str, max_timestamp_millis: int | None = None,"
        " max_traces: int | None = None, trace_ids: list[str] | None = None) -> int"
    ),
    "upload_named_json": (
        "tests.integration.mlflow_profile.upload_named_json(client, trace_id, relative_path, local_file,"
        " chunk_size) -> ArtifactVerification — adapted onto the public composition"
        " trace.info.tags['mlflow.artifactLocation'] -> mlflow.artifacts.get_artifact_repository(uri,"
        " tracking_uri) -> ArtifactRepository.log_artifact(local_file, artifact_path)"
    ),
    "verify_named_json": (
        "tests.integration.mlflow_profile.verify_named_json(client, trace_id, relative_path, sha256,"
        " size) -> None — adapted onto ArtifactRepository.download_artifacts(relative_path, dst_path)"
        " with streamed sha256/size comparison"
    ),
}

_PROBE_START_NS = 1_700_000_000_000_000_000


@dataclass(frozen=True)
class MlflowProfile:
    """The pinned client/server pair and the public API table the writer may use."""

    version: str
    server_command: Sequence[str]
    api_table: Mapping[str, str]

    @classmethod
    def initial(cls) -> MlflowProfile:
        """The initial pinned profile: mlflow 3.17.2 client and server."""
        return cls(
            version=PINNED_MLFLOW_VERSION,
            server_command=INITIAL_SERVER_COMMAND,
            api_table=INITIAL_API_TABLE,
        )


@dataclass(frozen=True)
class ArtifactVerification:
    """Digest of a named trace artifact as uploaded, for later exact verification."""

    trace_id: str
    relative_path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class CapabilityReport:
    """What ``verify_profile`` proved about the target against the pinned profile.

    ``required_methods`` only contains operations whose proof succeeded (in
    dry-run mode: operations whose availability and read-only protocol checks
    succeeded). ``safe_error`` is safe to display to a CLI user and carries the
    first causes of any incompatibility.
    """

    is_compatible: bool
    required_methods: frozenset[str]
    max_inline_payload_bytes: int
    named_artifact_supported: bool
    retirement_supported: bool
    safe_error: str | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MlflowException(message)


def _safe_relative_path(relative_path: str) -> PurePosixPath:
    """Validate a caller-chosen artifact relative path (no absolute, no traversal)."""
    rel = PurePosixPath(relative_path)
    _require(
        bool(relative_path) and not rel.is_absolute() and ".." not in rel.parts and str(rel) != ".",
        f"unsafe artifact relative path: {relative_path!r}",
    )
    return rel


def _trace_artifact_repo(client: MlflowClient, trace_id: str):
    """Resolve the public artifact repository for a trace's artifact directory.

    MLflow 3.17.2 records the trace artifact directory in the public
    ``mlflow.artifactLocation`` trace tag and serves it through the same
    public ``mlflow.artifacts.get_artifact_repository`` channel used for span
    payloads; this is the documented public route to trace-scoped artifacts.
    The lookup flushes the client-side async export queue first so the
    server-side trace (which carries the tag) is resolved instead of the
    not-yet-exported in-memory copy.
    """
    trace = client.get_trace(trace_id, display=False, flush=True)
    uri = trace.info.tags.get(MLFLOW_ARTIFACT_LOCATION)
    _require(bool(uri), f"trace {trace_id} carries no {MLFLOW_ARTIFACT_LOCATION} tag")
    return get_artifact_repository(uri, tracking_uri=client.tracking_uri)


def upload_named_json(
    client: MlflowClient,
    trace_id: str,
    relative_path: str,
    local_file: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> ArtifactVerification:
    """Upload ``local_file`` as a named artifact at exactly ``relative_path``.

    Adapted to the ``TraceArtifactSink.upload_named_json(trace_id, relative_path,
    local_file, chunk_size)`` contract. The source is read and staged in fixed
    ``chunk_size`` blocks (never loaded whole), the underlying public
    ``ArtifactRepository.log_artifact`` streams the staged file, and the stored
    path preserves the requested relative path exactly. Returns the digest to
    feed back into :func:`verify_named_json`.
    """
    _require(chunk_size > 0, f"chunk_size must be positive, got {chunk_size}")
    rel = _safe_relative_path(relative_path)
    source = Path(local_file)
    digest = hashlib.sha256()
    size = 0
    with TemporaryDirectory(prefix="opencode-mlflow-upload-") as tmp:
        staged = Path(tmp) / rel.name
        with source.open("rb") as src, staged.open("wb") as dst:
            while chunk := src.read(chunk_size):
                digest.update(chunk)
                size += len(chunk)
                dst.write(chunk)
        repo = _trace_artifact_repo(client, trace_id)
        parent = str(rel.parent)
        repo.log_artifact(str(staged), artifact_path=parent if parent != "." else None)
    return ArtifactVerification(trace_id=trace_id, relative_path=str(rel), sha256=digest.hexdigest(), size=size)


def verify_named_json(
    client: MlflowClient,
    trace_id: str,
    relative_path: str,
    sha256: str,
    size: int,
) -> None:
    """Retrieve the named artifact at ``relative_path`` and verify exact bytes.

    Adapted to the ``TraceArtifactSink.verify_named_json(trace_id, relative_path,
    sha256, size)`` contract: returns ``None`` when the artifact's streamed
    sha256 and size match, and raises :class:`MlflowException` on any mismatch.
    """
    rel = _safe_relative_path(relative_path)
    repo = _trace_artifact_repo(client, trace_id)
    digest = hashlib.sha256()
    seen = 0
    with TemporaryDirectory(prefix="opencode-mlflow-verify-") as tmp:
        downloaded = Path(repo.download_artifacts(str(rel), tmp))
        with downloaded.open("rb") as f:
            while chunk := f.read(DEFAULT_CHUNK_SIZE):
                digest.update(chunk)
                seen += len(chunk)
    _require(seen == size, f"artifact {rel} size mismatch: expected {size}, got {seen}")
    _require(
        digest.hexdigest() == sha256,
        f"artifact {rel} sha256 mismatch: expected {sha256}, got {digest.hexdigest()}",
    )


@contextmanager
def _isolated_tracing_env(client: MlflowClient) -> Iterator[None]:
    """Scrub ambient OTEL exporter config and pin the global tracking URI.

    ``mlflow.start_span_no_context`` exports spans through the *global*
    tracking URI (not the client's) and through an OTLP exporter when ambient
    ``OTEL_EXPORTER_OTLP_*`` variables are set, so both must be controlled
    before the tracing provider initializes for the first time.
    """
    saved = {name: os.environ.pop(name) for name in OTEL_HIJACK_ENV_VARS if name in os.environ}
    saved_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(client.tracking_uri)
    try:
        yield
    finally:
        mlflow.set_tracking_uri(saved_uri)
        os.environ.update(saved)


def _static_available_methods(client: MlflowClient) -> set[str]:
    """Availability of the pinned public surface without any server interaction."""
    available: set[str] = {
        name
        for name in (
            "get_experiment_by_name",
            "create_experiment",
            "get_trace",
            "search_traces",
            "delete_traces",
            "set_trace_tag",
            "delete_trace_tag",
        )
        if callable(getattr(client, name, None))
    }
    available.add("server_version_attestation")
    if callable(getattr(mlflow, "start_span_no_context", None)):
        available.add("mlflow.start_span_no_context")
    if callable(getattr(LiveSpan, "end", None)):
        available.add("LiveSpan.end")
    # Do not import MLflow's private repository classes here. The live named
    # artifact proof below exercises only the public factory/repository
    # composition and is authoritative for behavior.
    surface = callable(get_artifact_repository)
    if surface and callable(upload_named_json) and callable(verify_named_json):
        available.update({"upload_named_json", "verify_named_json"})
    return available


@contextmanager
def _proof(errors: list[str], operation: str) -> Iterator[None]:
    """Record a behavioral proof: success adds nothing, failure records the cause."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - a failed proof is a report input, not a crash
        errors.append(f"{operation}: {type(exc).__name__}: {exc}")


@dataclass
class _ProbeState:
    """Mutable state carried across the live proof steps."""

    client: MlflowClient
    probe_experiment_name: str
    inline_blob: str
    experiment_id: str | None = None
    root: LiveSpan | None = None
    child_span: LiveSpan | None = None
    trace_ids: list[str] = field(default_factory=list)
    upload: ArtifactVerification | None = None
    full_payload_verified: bool = False


def _step_experiments(state: _ProbeState, verified: set[str], errors: list[str]) -> None:
    if state.experiment_id is not None:
        return
    with _proof(errors, "get_experiment_by_name"):
        found = state.client.get_experiment_by_name(state.probe_experiment_name)
        _require(found is None, f"probe experiment {state.probe_experiment_name!r} already exists")
        verified.add("get_experiment_by_name")
    with _proof(errors, "create_experiment"):
        state.experiment_id = state.client.create_experiment(state.probe_experiment_name)
        verified.add("create_experiment")


def _step_spans(state: _ProbeState, verified: set[str], errors: list[str]) -> None:
    if state.experiment_id is None or state.root is not None:
        return
    with _proof(errors, "mlflow.start_span_no_context"):
        state.root = mlflow.start_span_no_context(
            name="opencode-mlflow-profile-root",
            span_type="CHAIN",
            inputs={"session_id": "opencode-mlflow-profile"},
            attributes={"probe": "profile"},
            tags={"opencode-mlflow.profile": "probe"},
            metadata={"probe": "1"},
            experiment_id=state.experiment_id,
            start_time_ns=_PROBE_START_NS,
        )
        _require(
            state.root.trace_id != NO_OP_SPAN_TRACE_ID,
            "span creation degraded to NoOp; ambient OTEL exporter configuration "
            "likely rerouted span export away from the tracking server",
        )
        child = mlflow.start_span_no_context(
            name="opencode-mlflow-profile-child",
            span_type="LLM",
            parent_span=state.root,
            start_time_ns=_PROBE_START_NS + 10_000_000,
        )
        _require(child.parent_id == state.root.span_id, "child span did not link to the root span")
        state.trace_ids.append(state.root.trace_id)
        state.child_span = child
        verified.add("mlflow.start_span_no_context")
    with _proof(errors, "LiveSpan.end"):
        _require(state.child_span is not None, "child span was not created")
        state.child_span.end(outputs={"token": "probe"}, status="OK", end_time_ns=_PROBE_START_NS + 20_000_000)
        _require(state.root is not None, "root span was not created")
        state.root.end(
            outputs={"answer": 42, "blob": state.inline_blob},
            attributes={"done": "yes"},
            status="OK",
            end_time_ns=_PROBE_START_NS + 30_000_000,
        )
        verified.add("LiveSpan.end")


def _step_retrieval(state: _ProbeState, verified: set[str], errors: list[str]) -> None:
    if state.root is None:
        return
    with _proof(errors, "get_trace"):
        trace = state.client.get_trace(state.root.trace_id, display=False, flush=True)
        spans = {span.name: span for span in trace.data.spans}
        root_span = spans["opencode-mlflow-profile-root"]
        _require(trace.info.trace_id == state.root.trace_id, "get_trace returned a different trace")
        _require(trace.info.state.value == "OK", f"trace state is {trace.info.state!r}, expected OK")
        _require(
            root_span.inputs == {"session_id": "opencode-mlflow-profile"},
            "span inputs did not round-trip exactly",
        )
        _require(
            root_span.outputs == {"answer": 42, "blob": state.inline_blob},
            f"inline payload of {MAX_INLINE_PAYLOAD_BYTES} bytes did not round-trip exactly",
        )
        _require(root_span.start_time_ns == _PROBE_START_NS, "start_time_ns was not preserved")
        _require(
            root_span.end_time_ns == _PROBE_START_NS + 30_000_000,
            "end_time_ns was not preserved",
        )
        _require(root_span.attributes.get("done") == "yes", "span attributes were not preserved")
        state.full_payload_verified = True
        verified.add("get_trace")


def _step_search(state: _ProbeState, verified: set[str], errors: list[str]) -> None:
    if state.experiment_id is None or state.root is None:
        return
    with _proof(errors, "search_traces"):
        for index in range(2):
            span = mlflow.start_span_no_context(
                name=f"opencode-mlflow-profile-page-{index}",
                span_type="CHAIN",
                experiment_id=state.experiment_id,
            )
            span.end(outputs={"page": index}, status="OK")
            state.trace_ids.append(span.trace_id)
        first = state.client.search_traces(
            locations=[state.experiment_id],
            filter_string="",
            max_results=2,
            page_token=None,
            flush=True,
            include_spans=False,
        )
        _require(len(first) == 2, f"first search page returned {len(first)} traces, expected 2")
        _require(bool(first.token), "search_traces did not return a pagination token")
        # The deprecated ``experiment_ids=`` form is exercised once, on
        # purpose, to prove it still paginates on the pinned pair; its
        # deprecation FutureWarning is expected noise, so it is silenced
        # here rather than polluting every profile run's output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            rest = state.client.search_traces(
                experiment_ids=[state.experiment_id],
                filter_string="",
                max_results=2,
                page_token=first.token,
                include_spans=False,
            )
        seen = {trace.trace_id for trace in first} | {trace.trace_id for trace in rest}
        _require(set(state.trace_ids) <= seen, "pagination did not cover every probe trace")
        filtered = state.client.search_traces(
            locations=[state.experiment_id],
            filter_string="tags.`mlflow.traceName` = 'opencode-mlflow-profile-root'",
            include_spans=False,
        )
        _require(
            any(trace.trace_id == state.root.trace_id for trace in filtered),
            "filter_string did not match the probe root trace",
        )
        verified.add("search_traces")


def _step_tag_update(state: _ProbeState, verified: set[str], errors: list[str]) -> None:
    if state.root is None:
        return
    with _proof(errors, "set_trace_tag"):
        state.client.set_trace_tag(state.root.trace_id, "opencode_mlflow_probe", "set")
        tags = state.client.get_trace(state.root.trace_id, display=False).info.tags
        _require(tags.get("opencode_mlflow_probe") == "set", "set_trace_tag did not persist")
        verified.add("set_trace_tag")
    with _proof(errors, "delete_trace_tag"):
        state.client.delete_trace_tag(state.root.trace_id, "opencode_mlflow_probe")
        tags = state.client.get_trace(state.root.trace_id, display=False).info.tags
        _require("opencode_mlflow_probe" not in tags, "delete_trace_tag did not remove the tag")
        verified.add("delete_trace_tag")


def _step_named_artifacts(state: _ProbeState, verified: set[str], errors: list[str]) -> None:
    if state.root is None:
        return
    with _proof(errors, "upload_named_json"):
        payload = {"probe": "named-artifact", "rows": list(range(256))}
        with TemporaryDirectory(prefix="opencode-mlflow-profile-upload-") as tmp:
            local = Path(tmp) / "profile-probe.json"
            local.write_bytes(json.dumps(payload, separators=(",", ":")).encode())
            state.upload = upload_named_json(
                state.client,
                state.root.trace_id,
                "probe/nested/profile-probe.json",
                local,
                chunk_size=512,
            )
        repo = _trace_artifact_repo(state.client, state.root.trace_id)
        listings = repo.list_artifacts("probe/nested")
        _require(
            any(
                info.path == "probe/nested/profile-probe.json" and info.file_size == state.upload.size
                for info in listings
            ),
            "named artifact is not listed at the requested relative path",
        )
        verified.add("upload_named_json")
    with _proof(errors, "verify_named_json"):
        _require(state.upload is not None, "upload proof did not produce a verification")
        verify_named_json(
            state.client,
            state.upload.trace_id,
            state.upload.relative_path,
            state.upload.sha256,
            state.upload.size,
        )
        try:
            verify_named_json(
                state.client,
                state.upload.trace_id,
                state.upload.relative_path,
                "0" * 64,
                state.upload.size,
            )
        except MlflowException:
            pass
        else:
            raise MlflowException("verify_named_json accepted a tampered sha256")
        verified.add("verify_named_json")


def _step_retirement(state: _ProbeState, verified: set[str], errors: list[str]) -> None:
    if state.experiment_id is None or not state.trace_ids:
        return
    with _proof(errors, "delete_traces"):
        unique_ids = list(dict.fromkeys(state.trace_ids))
        deleted = state.client.delete_traces(state.experiment_id, trace_ids=unique_ids)
        _require(
            deleted == len(unique_ids),
            f"delete_traces reported {deleted} deletions, expected {len(unique_ids)}",
        )
        if state.root is not None:
            try:
                state.client.get_trace(state.root.trace_id, display=False)
            except MlflowException:
                pass
            else:
                raise MlflowException("trace is still retrievable after delete_traces")
        verified.add("delete_traces")


def _version_errors(profile: MlflowProfile) -> list[str]:
    errors: list[str] = []
    installed = getattr(mlflow, "__version__", None)
    if installed != profile.version:
        errors.append(
            f"installed mlflow client version {installed!r} does not match the pinned "
            f"profile version {profile.version!r}"
        )
    return errors


def _verify_live(client: MlflowClient, profile: MlflowProfile) -> CapabilityReport:
    """Full behavioral verification; mutates the target (use a disposable server)."""
    errors = _version_errors(profile)
    available = _static_available_methods(client)
    missing = REQUIRED_METHODS - available
    if missing:
        errors.append("public methods not available: " + ", ".join(sorted(missing)))

    verified: set[str] = set()
    try:
        if attest_server_version(client.tracking_uri) != profile.version:
            errors.append("tracking server version does not match the pinned profile")
        else:
            verified.add("server_version_attestation")
    except Exception:  # noqa: BLE001 - keep profile diagnostics safe
        errors.append("tracking server version attestation failed")
    if errors:
        # The live profile's behavioral proof is mutation-based. Never use it
        # to probe a mismatched/unattested pair; the runtime gate must fail
        # before any experiment, trace, artifact, or retirement mutation.
        return CapabilityReport(
            is_compatible=False,
            required_methods=frozenset(verified),
            max_inline_payload_bytes=0,
            named_artifact_supported=False,
            retirement_supported=False,
            safe_error="; ".join(errors),
        )
    state = _ProbeState(
        client=client,
        probe_experiment_name=f"opencode-mlflow-profile-probe-{int(time.time() * 1000)}",
        inline_blob="x" * MAX_INLINE_PAYLOAD_BYTES,
    )
    with _isolated_tracing_env(client):
        _step_experiments(state, verified, errors)
        _step_spans(state, verified, errors)
        _step_retrieval(state, verified, errors)
        _step_search(state, verified, errors)
        _step_tag_update(state, verified, errors)
        _step_named_artifacts(state, verified, errors)
        _step_retirement(state, verified, errors)

    named_artifact_supported = {"upload_named_json", "verify_named_json"} <= verified
    retirement_supported = "delete_traces" in verified
    max_inline = MAX_INLINE_PAYLOAD_BYTES if state.full_payload_verified else 0
    is_compatible = (
        not errors
        and REQUIRED_METHODS <= verified
        and named_artifact_supported
        and retirement_supported
        and max_inline == MAX_INLINE_PAYLOAD_BYTES
    )
    return CapabilityReport(
        is_compatible=is_compatible,
        required_methods=frozenset(verified),
        max_inline_payload_bytes=max_inline,
        named_artifact_supported=named_artifact_supported,
        retirement_supported=retirement_supported,
        safe_error="; ".join(errors) if errors else None,
    )


def _discover_sample_trace_id(client: MlflowClient) -> str | None:
    """Read-only discovery of any existing trace across the reachable experiments."""
    for experiment in client.search_experiments():
        page = client.search_traces(locations=[experiment.experiment_id], max_results=5, include_spans=False)
        if page:
            return page[0].trace_id
    return None


def _verify_dry_run(client: MlflowClient, profile: MlflowProfile) -> CapabilityReport:
    """Non-mutating verification: read-only probes only, never a create/upload/delete."""
    errors = _version_errors(profile)
    available = _static_available_methods(client)
    missing = REQUIRED_METHODS - available
    if missing:
        errors.append("public methods not available: " + ", ".join(sorted(missing)))
    verified = available & REQUIRED_METHODS
    try:
        if attest_server_version(client.tracking_uri) != profile.version:
            errors.append("tracking server version does not match the pinned profile")
        else:
            verified.add("server_version_attestation")
    except Exception:  # noqa: BLE001 - keep profile diagnostics safe
        errors.append("tracking server version attestation failed")

    artifact_channel_ok = False
    with _isolated_tracing_env(client):
        # Drain the client-side async export queue once (never a server
        # mutation) so read-only discovery observes everything already written.
        mlflow.flush_trace_async_logging()
        try:
            found = client.get_experiment_by_name("opencode-mlflow-dry-run-probe")
            _require(
                found is None,
                "experiment 'opencode-mlflow-dry-run-probe' unexpectedly exists on the target",
            )
        except Exception as exc:  # noqa: BLE001 - a failed probe is a report input
            errors.append(f"server reachability probe failed: {type(exc).__name__}: {exc}")

        try:
            client.get_trace("tr-opencode-mlflow-dry-run-probe", display=False)
            errors.append("get_trace on an unknown trace id did not raise a protocol error")
        except MlflowException:
            pass  # a proper MLflow error response proves the trace protocol
        except Exception as exc:  # noqa: BLE001
            errors.append(f"trace read did not produce an MLflow protocol error: {type(exc).__name__}: {exc}")

        try:
            sample_id = _discover_sample_trace_id(client)
            if sample_id is None:
                errors.append("dry-run cannot prove full-payload retrieval: no existing trace is readable")
            else:
                trace = client.get_trace(sample_id, display=False)
                _require(bool(trace.data.spans), "retrieved trace carries no span payload")
                uri = trace.info.tags.get(MLFLOW_ARTIFACT_LOCATION)
                _require(
                    bool(uri),
                    "retrieved trace carries no artifactLocation tag; artifact channel cannot be probed",
                )
                repo = get_artifact_repository(uri, tracking_uri=client.tracking_uri)
                repo.list_artifacts(None)
                artifact_channel_ok = True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"read-only trace/artifact probing failed: {type(exc).__name__}: {exc}")

    named_artifact_supported = (
        artifact_channel_ok
        and {
            "upload_named_json",
            "verify_named_json",
        }
        <= available
    )
    retirement_supported = "delete_traces" in available
    is_compatible = not errors and REQUIRED_METHODS <= verified and named_artifact_supported and retirement_supported
    return CapabilityReport(
        is_compatible=is_compatible,
        required_methods=frozenset(verified),
        max_inline_payload_bytes=0,
        named_artifact_supported=named_artifact_supported,
        retirement_supported=retirement_supported,
        safe_error="; ".join(errors) if errors else None,
    )


def verify_profile(client: MlflowClient, profile: MlflowProfile, *, dry_run: bool = False) -> CapabilityReport:
    """Verify ``client``'s target satisfies the pinned ``profile``.

    The default flavor exercises every pinned operation for real against the
    server (including creating and retiring probe traces and uploading a probe
    artifact), so point it at a disposable server. With ``dry_run=True`` the
    verification is strictly read-only: it probes availability, reachability,
    protocol responses, full-payload retrieval of an existing trace, the
    artifact channel, and retirement *availability* — never creating an
    experiment, trace, or artifact. A target that cannot prove the contract
    without mutation is reported incompatible for dry-run.
    """
    if dry_run:
        return _verify_dry_run(client, profile)
    return _verify_live(client, profile)
