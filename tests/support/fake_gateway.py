"""Fake MLflow gateway seams for unit, contract, and BDD tests.

Two groups of fakes live here:

- **Fake public-client/LiveSpan-shaped objects** (`FakeMlflowClient`,
  `FakeTraceApi`, `FakeLiveSpan`, `FakeArtifactStore`): an in-memory stand-in
  for the real ``MlflowClient`` + module-level trace API + artifact
  repository factory. The *real* :class:`opencode_mlflow.mlflow.gateway.MlflowGateway`
  is driven against these in unit and contract tests, so the production
  gateway's public call shapes (keyword names, pagination tokens, ownership
  filtering, the named-artifact composition) are exercised without a server.
  The fake mirrors the behavioral facts pinned by Task 0's compatibility
  profile (``tests/integration/mlflow_profile.py``): tag values silently
  truncate at 4096 characters and metadata values at 250, ``get_trace`` on an
  unknown id raises an MLflow-shaped error, ``search_traces`` paginates with
  an opaque ``token`` and applies ``tags.`k` = 'v'`` /
  ``metadata.`k` = 'v'`` filter strings, and every server-backed trace
  carries the ``mlflow.artifactLocation`` tag.

- **`FakeMlflowGateway`**: a complete in-memory
  :class:`opencode_mlflow.mlflow.gateway.MlflowGateway` stand-in for later
  BDD tasks. It records an event/mutation history (every experiment create,
  trace create/end, tag set, artifact upload/verify, retirement) with enough
  state to verify payloads and ordering, and enforces the same
  mutation gates as the real gateway (preflight before mutation, dry-run
  never mutates).

This module is test support: it is importable from tests only, never from
``src``.
"""

from __future__ import annotations

import copy
import hashlib
import re
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opencode_mlflow.content.artifacts import ArtifactProvenance
from opencode_mlflow.content.canonical import canonical_json
from opencode_mlflow.errors import CapabilityError, ConfigurationError, ItemImportError
from opencode_mlflow.mlflow.capabilities import (
    PINNED_MAX_INLINE_PAYLOAD_BYTES,
    PINNED_MLFLOW_VERSION,
    REQUIRED_METHODS,
    CapabilityReport,
    CapabilityRequest,
    attest_server_version,
)
from opencode_mlflow.mlflow.gateway import (
    DEFAULT_UPLOAD_CHUNK_BYTES,
    NO_OP_TRACE_ID,
    RETRIEVED_SPAN_STATUS,
    TRACE_ARTIFACT_LOCATION_TAG,
    TRACE_METADATA_VALUE_LIMIT,
    TRACE_SESSION_METADATA_KEY,
    TRACE_STATE_FOR_ROOT_STATUS,
    TRACE_TAG_VALUE_LIMIT,
    ArtifactVerification,
    CreatedTrace,
    ExperimentRef,
    MlflowGateway,
    OwnedTrace,
    TraceChildRequest,
    TraceCreateRequest,
    # The fake mirrors the gateway's single-sourced effective-status
    # derivation (the mapping's explicit root_status request field is
    # preferred there), so the fake's stored/returned status can never
    # drift from production.
    _effective_root_status,
)
from opencode_mlflow.mlflow.reconciliation import (
    ARTIFACT_PATH_TAG,
    ARTIFACT_SHA256_TAG,
    ARTIFACT_SIZE_TAG,
    OWNERSHIP_CATALOG_VERSION_KEY,
    OWNERSHIP_OPERATION_KEY,
    OWNERSHIP_RENDER_FINGERPRINT_KEY,
    OWNERSHIP_SESSION_KEY,
    OWNERSHIP_SOURCE_FINGERPRINT_KEY,
    OWNERSHIP_SOURCE_SCHEMA_KEY,
    OWNERSHIP_TURN_KEY,
    CandidateContract,
    CandidateVerification,
    OwnedTraceCandidate,
    _candidate_importer_marker,
    _ownership_contract_errors,
    _parse_turn_index,
    _parse_warning_metadata,
)
from opencode_mlflow.source.catalog import SCHEMA_CATALOG_VERSION
from opencode_mlflow.source.discovery import ARTIFACT_SCHEMA_VERSION, MAPPING_VERSION

__all__ = [
    "FakeArtifactRepository",
    "FakeArtifactStore",
    "FakeLiveSpan",
    "FakeMlflowClient",
    "FakeMlflowException",
    "FakeMlflowGateway",
    "FakeNoOpSpan",
    "FakeTraceApi",
    "GatewayEvent",
    "fake_version_transport",
    "child_spec",
    "trace_spec",
]


class FakeMlflowException(Exception):
    """MLflow-protocol-shaped error: carries a machine-readable ``error_code``.

    The runtime capability probe classifies raised errors by their
    ``error_code`` attribute (real ``MlflowException`` exposes the same
    property), which distinguishes a proper MLflow protocol response from a
    transport failure.
    """

    def __init__(self, message: str, error_code: str = "RESOURCE_DOES_NOT_EXIST") -> None:
        super().__init__(message)
        self.error_code = error_code


def fake_version_transport(_tracking_uri: str) -> object:
    """Explicit test-only attestation used by the in-memory gateway fake."""
    return {"version": PINNED_MLFLOW_VERSION}


@dataclass(frozen=True)
class FakeExperiment:
    """Experiment-shaped record exposed by ``get_experiment_by_name``."""

    experiment_id: str
    name: str
    artifact_location: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass
class FakeLiveSpan:
    """LiveSpan-shaped object created by :class:`FakeTraceApi`.

    ``trace_id``/``span_id``/``parent_id`` are plain attributes standing in
    for the real ``LiveSpan`` properties; ``end`` mirrors the public
    ``LiveSpan.end(outputs, attributes, status, end_time_ns)`` contract,
    including the pinned 3.17.2 end-time coercion: every non-``ERROR``
    span status is forced to ``OK`` once the span ends (a root ended
    ``UNSET`` retrieves as ``OK``, exactly like the live server).
    """

    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    span_type: str
    experiment_id: str | None
    start_time_ns: int | None
    inputs: object | None = None
    outputs: object | None = None
    attributes: dict[str, object] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    status: str | None = None
    end_time_ns: int | None = None
    ended: bool = False

    def end(
        self,
        outputs: object | None = None,
        attributes: Mapping[str, object] | None = None,
        status: str | None = None,
        end_time_ns: int | None = None,
    ) -> None:
        if outputs is not None:
            self.outputs = outputs
        if attributes:
            self.attributes.update(attributes)
        if status is not None:
            self.status = status
        # The pinned server's LiveSpan.end forces every non-ERROR status to
        # OK at end time, so an UNSET (or unset) status never survives an
        # explicit end.
        if self.status != "ERROR":
            self.status = "OK"
        self.end_time_ns = end_time_ns
        self.ended = True


class FakeNoOpSpan:
    """``NoOpSpan``-shaped object mirroring mlflow's degraded span surface.

    The pinned 3.17.2 ``start_span_no_context`` swallows a failed span
    creation into a ``NoOpSpan`` whose ``trace_id`` is the NoOp sentinel
    (and whose ``span_id``/``name`` are ``None``). The gateway's loud
    ``mlflow_span_degraded`` check detects exactly that sentinel, so a
    degraded span produced by the fake is detectable by the same check —
    fake-based tests can no longer hide the child+``experiment_id``
    degradation the live server exhibits (live-verified in Task 14; see
    ``docs/compatibility.md`` § "Live-verified gateway constraints").
    """

    @property
    def trace_id(self) -> str:
        return NO_OP_TRACE_ID

    @property
    def span_id(self) -> str | None:
        return None

    @property
    def parent_id(self) -> str | None:
        return None

    @property
    def name(self) -> str | None:
        return None

    def end(
        self,
        outputs: object | None = None,
        attributes: Mapping[str, object] | None = None,
        status: object | None = None,
        end_time_ns: int | None = None,
    ) -> None:
        """No-op completion, like the real ``NoOpSpan.end``."""
        return None


@dataclass
class FakeTraceInfo:
    """TraceInfo-shaped record: tags, request metadata, creation time, state.

    ``state`` mirrors the real ``TraceInfo.state``: traces start
    ``IN_PROGRESS`` and the root span's end updates it through the same
    root-status→state mapping the pinned server applies (an ``UNSET`` root
    surfaces as state ``OK``).
    """

    trace_id: str
    experiment_id: str | None
    request_time: int
    state: str = "IN_PROGRESS"
    tags: dict[str, str] = field(default_factory=dict)
    trace_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class FakeTraceData:
    """TraceData-shaped record holding the trace's spans."""

    spans: list[FakeLiveSpan] = field(default_factory=list)


@dataclass
class FakeTrace:
    """Trace-shaped object: ``trace_id`` plus ``info``/``data`` records."""

    trace_id: str
    info: FakeTraceInfo
    data: FakeTraceData = field(default_factory=FakeTraceData)


class FakePagedList(list[FakeTrace]):
    """PagedList-shaped page with the opaque continuation ``token``."""

    def __init__(self, items: Sequence[FakeTrace], token: str | None) -> None:
        super().__init__(items)
        self.token = token


class FakeArtifactRepository:
    """ArtifactRepository-shaped in-memory store for one trace's artifacts.

    ``log_artifact`` stores ``<artifact_path>/<basename>`` exactly like the
    pinned public repository interface, and ``download_artifacts`` writes the
    stored bytes into ``dst_path`` and returns the local file path.
    """

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[str, ...]] = []

    def log_artifact(self, local_file: str, artifact_path: str | None = None) -> None:
        self.calls.append(("log_artifact", local_file, artifact_path))
        data = Path(local_file).read_bytes()
        name = Path(local_file).name
        relative = f"{artifact_path}/{name}" if artifact_path else name
        self.files[relative] = data

    def download_artifacts(self, artifact_path: str, dst_path: str) -> str:
        self.calls.append(("download_artifacts", artifact_path, dst_path))
        if artifact_path not in self.files:
            raise FakeMlflowException(f"artifact {artifact_path} not found")
        target = Path(dst_path) / Path(artifact_path).name
        target.write_bytes(self.files[artifact_path])
        return str(target)

    def list_artifacts(self, path: str | None = None) -> list[str]:
        prefix = f"{path}/" if path else ""
        return sorted(key for key in self.files if key.startswith(prefix))


class FakeArtifactStore:
    """Registry of per-trace fake repositories behind the factory seam.

    ``get_artifact_repository`` mirrors the public
    ``mlflow.artifacts.get_artifact_repository(artifact_uri, tracking_uri=...)``
    factory signature so it can be injected wherever the production gateway
    expects the real one.
    """

    def __init__(self) -> None:
        self.repositories: dict[str, FakeArtifactRepository] = {}

    def register(self, uri: str) -> FakeArtifactRepository:
        if uri not in self.repositories:
            self.repositories[uri] = FakeArtifactRepository()
        return self.repositories[uri]

    def get_artifact_repository(
        self,
        artifact_uri: str,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> FakeArtifactRepository:
        repository = self.repositories.get(artifact_uri)
        if repository is None:
            raise FakeMlflowException(f"no artifact repository registered for {artifact_uri}")
        return repository


class FakeTraceApi:
    """PublicTraceApi-shaped fake creating spans inside a :class:`FakeMlflowClient`.

    Every call is recorded (with the exact keyword names the production
    gateway used) so tests can assert the public call shapes the
    compatibility profile pins.
    """

    def __init__(self, client: FakeMlflowClient) -> None:
        self._client = client
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def start_span_no_context(
        self,
        *,
        name: str,
        span_type: str = "UNKNOWN",
        parent_span: FakeLiveSpan | None = None,
        inputs: object | None = None,
        attributes: Mapping[str, object] | None = None,
        tags: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        experiment_id: str | None = None,
        start_time_ns: int | None = None,
    ) -> FakeLiveSpan | FakeNoOpSpan:
        self.calls.append(
            (
                "start_span_no_context",
                {
                    "name": name,
                    "span_type": span_type,
                    "parent_span": parent_span,
                    "inputs": inputs,
                    "attributes": dict(attributes) if attributes is not None else None,
                    "tags": dict(tags) if tags is not None else None,
                    "metadata": dict(metadata) if metadata is not None else None,
                    "experiment_id": experiment_id,
                    "start_time_ns": start_time_ns,
                },
            )
        )
        return self._client.create_span(
            name=name,
            span_type=span_type,
            parent_span=parent_span,
            inputs=inputs,
            attributes=dict(attributes) if attributes is not None else None,
            tags=dict(tags) if tags is not None else None,
            metadata=dict(metadata) if metadata is not None else None,
            experiment_id=experiment_id,
            start_time_ns=start_time_ns,
        )

    def end_span(
        self,
        span: FakeLiveSpan,
        *,
        outputs: object | None = None,
        attributes: Mapping[str, object] | None = None,
        status: str | None = None,
        end_time_ns: int | None = None,
    ) -> None:
        self.calls.append(
            (
                "end_span",
                {
                    "span_id": span.span_id,
                    "outputs": outputs,
                    "attributes": dict(attributes) if attributes is not None else None,
                    "status": status,
                    "end_time_ns": end_time_ns,
                },
            )
        )
        span.end(outputs=outputs, attributes=attributes, status=status, end_time_ns=end_time_ns)
        if span.parent_id is None:
            # Mirror the pinned server: ending the root updates the trace
            # state through the root-status→state mapping (UNSET → OK).
            trace = self._client._traces.get(span.trace_id)
            if trace is not None:
                trace.info.state = TRACE_STATE_FOR_ROOT_STATUS.get(span.status, span.status)


def _truncate(value: str, limit: int) -> str:
    return value[:limit]


class FakeMlflowClient:
    """In-memory ``MlflowClient`` stand-in mirroring the pinned 3.17.2 surface.

    Only the client APIs the gateway adapts are provided: experiment
    lookup/creation, paginated trace search (with ``tags.``/``metadata.``
    filter-string support and opaque page tokens), full-payload trace
    retrieval, public trace tag update, and trace retirement. ``calls``
    records every call with its keyword arguments and ``mutations`` records
    the mutating subset (the read-only capability probes leave it empty);
    ``snapshot_state`` captures the experiment, trace, and artifact axes for
    before/after equality assertions.
    """

    tracking_uri = "fake://tracking"

    def __init__(self) -> None:
        self._experiments_by_name: dict[str, FakeExperiment] = {}
        self._traces: dict[str, FakeTrace] = {}
        self._next_experiment = 0
        self._next_trace = 0
        self._next_span = 0
        self.artifacts = FakeArtifactStore()
        self._trace_api = FakeTraceApi(self)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.mutations: list[tuple[str, ...]] = []
        self.reject_filters = False
        """When True, any non-empty ``filter_string`` raises (server rejects it)."""
        self.after_get_trace_hook: Callable[[str], None] | None = None
        """Optional race hook run after a one-shot retrieval snapshot is made."""

    # -- wiring helpers ----------------------------------------------------

    @property
    def trace_api(self) -> FakeTraceApi:
        """The PublicTraceApi-shaped fake bound to this client (one instance)."""
        return self._trace_api

    @property
    def artifact_repository_factory(self) -> Callable[..., FakeArtifactRepository]:
        """The artifact-repository factory seam bound to this client's store."""
        return self.artifacts.get_artifact_repository

    def gateway(
        self,
        *,
        dry_run: bool = False,
        search_page_size: int = 100,
        version_transport: Callable[[str], object] | None = None,
    ) -> MlflowGateway:
        """The real production gateway wired to this client's fakes."""
        # This explicit injected attestation keeps the fake transport
        # compatible without weakening the production probe's HTTP contract.
        if version_transport is None:

            def version_transport(_uri: str) -> object:
                return {"version": PINNED_MLFLOW_VERSION}

        return MlflowGateway(
            self,
            trace_api=self.trace_api,  # type: ignore[arg-type]
            artifact_repository_factory=self.artifact_repository_factory,
            dry_run=dry_run,
            search_page_size=search_page_size,
            version_transport=version_transport,
        )

    # -- experiment APIs ---------------------------------------------------

    def add_experiment(self, name: str, *, artifact_location: str | None = None) -> FakeExperiment:
        """Seed an existing experiment (not recorded as a mutation)."""
        self._next_experiment += 1
        experiment = FakeExperiment(
            experiment_id=f"exp-{self._next_experiment}",
            name=name,
            artifact_location=artifact_location
            if artifact_location is not None
            else f"fake-artifact://exp-{self._next_experiment}",
        )
        self._experiments_by_name[name] = experiment
        return experiment

    def get_experiment_by_name(self, name: str) -> FakeExperiment | None:
        self.calls.append(("get_experiment_by_name", {"name": name}))
        return self._experiments_by_name.get(name)

    def create_experiment(
        self,
        name: str,
        artifact_location: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        self.calls.append(("create_experiment", {"name": name, "artifact_location": artifact_location, "tags": tags}))
        if name in self._experiments_by_name:
            raise FakeMlflowException(f"experiment {name} already exists", "RESOURCE_ALREADY_EXISTS")
        self.mutations.append(("create_experiment", name))
        experiment = self.add_experiment(name, artifact_location=artifact_location)
        if tags:
            experiment.tags = dict(tags)
        return experiment.experiment_id

    # -- trace APIs ----------------------------------------------------------

    def create_span(
        self,
        *,
        name: str,
        span_type: str,
        parent_span: FakeLiveSpan | None,
        inputs: object | None,
        attributes: Mapping[str, object] | None,
        tags: Mapping[str, str] | None,
        metadata: Mapping[str, str] | None,
        experiment_id: str | None,
        start_time_ns: int | None,
    ) -> FakeLiveSpan | FakeNoOpSpan:
        """Create a span (and its trace for roots) like the real module-level API.

        Trace tags and metadata update the TRACE (mirroring 3.17.2's
        ``start_span_no_context``), values are silently truncated at the
        pinned limits, and every server-backed trace carries the public
        ``mlflow.artifactLocation`` tag backed by a registered repository.
        A NON-root span started with ``experiment_id`` degrades to a
        NoOp-shaped span with a warning, mirroring the pinned 3.17.2
        provider (live-verified in Task 14): nothing reaches the server.
        """
        if parent_span is not None and experiment_id is not None:
            # The real provider logs this warning on its non-root path and
            # then fails inside ``start_span_no_context`` (an AttributeError
            # on the raw OTel span), which the fluent wrapper swallows into
            # a NoOp span. The warning text mirrors the real wording.
            warnings.warn(
                "The `experiment_id` parameter can only be used for root spans, "
                f"but the span `{name}` is not a root span. The specified value "
                f"`{experiment_id}` will be ignored.",
                stacklevel=2,
            )
            return FakeNoOpSpan()
        self._next_span += 1
        span_id = f"span-{self._next_span}"
        if parent_span is None:
            self._next_trace += 1
            trace_id = f"tr-{self._next_trace}"
            info = FakeTraceInfo(
                trace_id=trace_id,
                experiment_id=experiment_id,
                request_time=(start_time_ns or 0) // 1_000_000,
            )
            trace = FakeTrace(trace_id=trace_id, info=info)
            uri = f"fake-artifact://traces/{trace_id}"
            info.tags[TRACE_ARTIFACT_LOCATION_TAG] = uri
            self.artifacts.register(uri)
            self._traces[trace_id] = trace
            parent_id = None
        else:
            trace = self._traces[parent_span.trace_id]
            trace_id = trace.trace_id
            parent_id = parent_span.span_id
        span = FakeLiveSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_id=parent_id,
            name=name,
            span_type=span_type,
            experiment_id=experiment_id,
            start_time_ns=start_time_ns,
            inputs=inputs,
            attributes=dict(attributes) if attributes is not None else {},
            tags=dict(tags) if tags is not None else {},
        )
        trace.data.spans.append(span)
        if tags:
            for key, value in tags.items():
                trace.info.tags[key] = _truncate(value, TRACE_TAG_VALUE_LIMIT)
        if metadata:
            for key, value in metadata.items():
                trace.info.trace_metadata[key] = _truncate(value, TRACE_METADATA_VALUE_LIMIT)
        return span

    def get_trace(self, trace_id: str, display: bool = True, flush: bool = False) -> FakeTrace:
        self.calls.append(("get_trace", {"trace_id": trace_id, "display": display, "flush": flush}))
        trace = self._traces.get(trace_id)
        if trace is None:
            raise FakeMlflowException(f"trace {trace_id} not found")
        if self.after_get_trace_hook is not None:
            # Return an isolated retrieval payload, then mutate the backing
            # target. This models a server-side race after a gateway check but
            # before the next client operation without changing normal fake
            # retrieval semantics used by existing tamper tests.
            snapshot = copy.deepcopy(trace)
            hook = self.after_get_trace_hook
            self.after_get_trace_hook = None
            hook(trace_id)
            return snapshot
        return trace

    def search_traces(
        self,
        *,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = 100,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        run_id: str | None = None,
        include_spans: bool = True,
        model_id: str | None = None,
        locations: list[str] | None = None,
        flush: bool = False,
    ) -> FakePagedList:
        self.calls.append(
            (
                "search_traces",
                {
                    "experiment_ids": experiment_ids,
                    "filter_string": filter_string,
                    "max_results": max_results,
                    "page_token": page_token,
                    "include_spans": include_spans,
                    "locations": locations,
                },
            )
        )
        if self.reject_filters and filter_string:
            raise FakeMlflowException("filter not supported by this target", "INVALID_PARAMETER")
        scoped = locations if locations is not None else experiment_ids
        matches = [
            trace
            for trace in self._traces.values()
            if (scoped is None or trace.info.experiment_id in scoped) and _matches_filter(trace, filter_string)
        ]
        offset = int(page_token) if page_token else 0
        page = matches[offset : offset + max_results]
        token = str(offset + max_results) if offset + max_results < len(matches) else None
        return FakePagedList(page, token)

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        self.calls.append(("set_trace_tag", {"trace_id": trace_id, "key": key, "value": value}))
        trace = self._traces.get(trace_id)
        if trace is None:
            raise FakeMlflowException(f"trace {trace_id} not found")
        self.mutations.append(("set_trace_tag", trace_id, key))
        trace.info.tags[key] = _truncate(value, TRACE_TAG_VALUE_LIMIT)

    def delete_trace_tag(self, trace_id: str, key: str) -> None:
        self.calls.append(("delete_trace_tag", {"trace_id": trace_id, "key": key}))
        trace = self._traces.get(trace_id)
        if trace is None:
            raise FakeMlflowException(f"trace {trace_id} not found")
        self.mutations.append(("delete_trace_tag", trace_id, key))
        trace.info.tags.pop(key, None)

    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        self.calls.append(
            (
                "delete_traces",
                {"experiment_id": experiment_id, "trace_ids": trace_ids, "max_timestamp_millis": max_timestamp_millis},
            )
        )
        self.mutations.append(("delete_traces", experiment_id, tuple(trace_ids or ())))
        targets = [
            trace_id
            for trace_id in (trace_ids or [])
            if trace_id in self._traces and self._traces[trace_id].info.experiment_id == experiment_id
        ]
        for trace_id in targets:
            del self._traces[trace_id]
        return len(targets)

    # -- state assertions ----------------------------------------------------

    def snapshot_state(self) -> tuple[object, object, object]:
        """Read-only three-axis snapshot: experiments, traces, artifact trees."""
        experiments = tuple(
            sorted(
                (e.name, e.experiment_id, e.artifact_location, tuple(sorted(e.tags.items())))
                for e in self._experiments_by_name.values()
            )
        )
        traces = tuple(
            sorted(
                (
                    t.trace_id,
                    t.info.experiment_id,
                    t.info.request_time,
                    t.info.state,
                    tuple(sorted(t.info.tags.items())),
                    tuple(sorted(t.info.trace_metadata.items())),
                    tuple(
                        sorted(
                            (
                                s.span_id,
                                s.parent_id,
                                s.name,
                                s.span_type,
                                s.status,
                                s.start_time_ns,
                                s.end_time_ns,
                                canonical_json(s.inputs),
                                canonical_json(s.outputs),
                                canonical_json(s.attributes),
                            )
                            for s in t.data.spans
                        )
                    ),
                )
                for t in self._traces.values()
            )
        )
        artifacts = tuple(
            sorted((uri, tuple(sorted(repo.files.items()))) for uri, repo in self.artifacts.repositories.items())
        )
        return experiments, traces, artifacts


_FILTER_PATTERN = re.compile(r"^(metadata|tags)\.`([^`]+)`\s*=\s*'([^']*)'$")


def _numeric_suffix(value: str) -> int:
    """The trailing integer of an id like ``fake-tr-7`` (0 when absent)."""
    match = re.search(r"(\d+)$", value)
    return int(match.group(1)) if match else 0


def _matches_filter(trace: FakeTrace, filter_string: str | None) -> bool:
    if not filter_string:
        return True
    match = _FILTER_PATTERN.match(filter_string.strip())
    if match is None:
        raise FakeMlflowException(f"unsupported filter string: {filter_string}", "INVALID_PARAMETER")
    scope, key, value = match.group(1), match.group(2), match.group(3)
    source = trace.info.trace_metadata if scope == "metadata" else trace.info.tags
    return source.get(key) == value


# ---------------------------------------------------------------------------
# Sample request builders
# ---------------------------------------------------------------------------


def child_spec(
    source_key: str,
    *,
    name: str = "child-span",
    span_type: str = "LLM",
    parent_source_key: str | None = None,
    inputs: object | None = None,
    attributes: Mapping[str, object] | None = None,
    tags: Mapping[str, str] | None = None,
    start_time_ns: int | None = 1_700_000_000_010_000_000,
    outputs: object | None = None,
    status: str = "OK",
    end_time_ns: int | None = 1_700_000_000_020_000_000,
) -> TraceChildRequest:
    """One child-span request with test-friendly defaults."""
    return TraceChildRequest(
        source_key=source_key,
        name=name,
        span_type=span_type,
        parent_source_key=parent_source_key,
        inputs=inputs,
        attributes=dict(attributes) if attributes is not None else {},
        tags=dict(tags) if tags is not None else {},
        start_time_ns=start_time_ns,
        outputs=outputs,
        status=status,
        end_time_ns=end_time_ns,
    )


def trace_spec(
    *,
    experiment_id: str = "exp-1",
    name: str = "opencode turn",
    native_session_id: str = "ses-sample",
    root_inputs: object | None = None,
    root_outputs: object | None = None,
    root_attributes: Mapping[str, object] | None = None,
    root_tags: Mapping[str, str] | None = None,
    metadata: Mapping[str, str] | None = None,
    start_time_ns: int | None = 1_700_000_000_000_000_000,
    end_time_ns: int | None = 1_700_000_000_030_000_000,
    children: Sequence[TraceChildRequest] = (),
    child_specs: Callable[[], Iterator[TraceChildRequest]] | None = None,
    client_request_id: str = "req-sample",
    expected_operation_key: str = "0" * 64,
    artifact_snapshot_fingerprint: str = "1" * 64,
    artifact_relative_path: str = (
        "opencode-mlflow/v1/sessions/ses-sample/turns/turn-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
        "111-222.json"
    ),
    root_status: str | None = None,
    owned: bool = True,
) -> TraceCreateRequest:
    """A complete trace-create request with importer-ownership defaults.

    ``owned=True`` seeds the exact ownership metadata the importer writes
    (``opencode.importer = opencode-mlflow`` plus the session identity), so
    ``iter_owned_traces`` finds the created trace; pass ``owned=False`` to
    build an unowned trace for exclusion tests.
    """
    if metadata is not None:
        resolved_metadata = dict(metadata)
    elif owned:
        resolved_metadata = {
            "opencode.importer": "opencode-mlflow",
            "opencode.session_id": native_session_id,
            "opencode.turn_index": "1",
            "opencode.turn_key": "turn-" + "a" * 64,
            "opencode.source_fingerprint": "a" * 64,
            "opencode.source_schema": "current",
            "opencode.importer_version": "0.1.0",
            "opencode.mapping_version": MAPPING_VERSION,
            "opencode.artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "opencode.redaction_mode": "verbatim",
            "opencode.catalog_version": SCHEMA_CATALOG_VERSION,
            "opencode.render_fingerprint": "b" * 64,
            "opencode.operation_key": expected_operation_key,
        }
    else:
        resolved_metadata = {}
    if owned:
        resolved_metadata.setdefault("opencode.turn_index", "1")
        resolved_metadata.setdefault("opencode.turn_key", "turn-" + "a" * 64)
        resolved_metadata.setdefault("opencode.source_fingerprint", "a" * 64)
        resolved_metadata.setdefault("opencode.source_schema", "current")
        resolved_metadata.setdefault("opencode.importer_version", "0.1.0")
        resolved_metadata.setdefault("opencode.mapping_version", MAPPING_VERSION)
        resolved_metadata.setdefault("opencode.artifact_schema_version", ARTIFACT_SCHEMA_VERSION)
        resolved_metadata.setdefault("opencode.redaction_mode", "verbatim")
        resolved_metadata.setdefault("opencode.catalog_version", SCHEMA_CATALOG_VERSION)
        resolved_metadata.setdefault("opencode.render_fingerprint", "b" * 64)
        resolved_metadata.setdefault("opencode.operation_key", expected_operation_key)
    if child_specs is None:
        child_specs = lambda: iter(children)  # noqa: E731 - local shim keeps the default lazy
    return TraceCreateRequest(
        client_request_id=client_request_id,
        experiment_id=experiment_id,
        name=name,
        native_session_id=native_session_id,
        root_inputs=root_inputs if root_inputs is not None else {"prompt": "synthetic prompt"},
        root_outputs=root_outputs,
        root_attributes=dict(root_attributes) if root_attributes is not None else {},
        root_tags=dict(root_tags) if root_tags is not None else {},
        metadata=resolved_metadata,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        child_specs=child_specs,
        expected_operation_key=expected_operation_key,
        artifact_snapshot_fingerprint=artifact_snapshot_fingerprint,
        artifact_relative_path=artifact_relative_path,
        root_status=root_status,
    )


# ---------------------------------------------------------------------------
# FakeMlflowGateway: the in-memory gateway for BDD tasks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatewayEvent:
    """One recorded gateway operation (kind plus stable details)."""

    kind: str
    details: tuple[object, ...]


@dataclass
class _StoredSpan:
    source_key: str
    span_id: str
    parent_id: str | None
    name: str
    span_type: str
    inputs: object | None
    outputs: object | None
    attributes: dict[str, object]
    tags: dict[str, str]
    status: str | None
    start_time_ns: int | None
    end_time_ns: int | None


@dataclass
class _StoredTrace:
    trace_id: str
    experiment_id: str
    name: str
    native_session_id: str
    root_span_id: str
    client_request_id: str
    inputs: object | None
    outputs: object | None
    root_attributes: dict[str, object]
    start_time_ns: int | None
    end_time_ns: int | None
    tags: dict[str, str]
    metadata: dict[str, str]
    spans_by_source: dict[str, _StoredSpan]
    root_status: str | None = None
    state: str = "IN_PROGRESS"
    ended: bool = False


class FakeMlflowGateway:
    """In-memory :class:`MlflowGateway` stand-in for BDD scenarios.

    Implements the same public surface and the same mutation gates (the
    capability check must pass before any mutation; dry-run never mutates)
    without touching MLflow, and records an event/mutation history with
    enough state to verify payloads and ordering: every experiment create,
    trace create/end, span start/end, artifact upload/verify, and
    retirement.
    """

    tracking_uri = "fake://tracking"

    def __init__(
        self,
        *,
        dry_run: bool = False,
        compatible: bool = True,
        version_transport: Callable[[str], object] = fake_version_transport,
    ) -> None:
        self.dry_run = dry_run
        self.compatible = compatible
        self.version_transport = version_transport
        self.events: list[GatewayEvent] = []
        self.mutations: list[tuple[str, ...]] = []
        self._experiments: dict[str, ExperimentRef] = {}
        self._traces: dict[str, _StoredTrace] = {}
        self._artifacts: dict[str, dict[str, bytes]] = {}
        self._next_experiment = 0
        self._next_trace = 0
        self._next_span = 0
        self._preflight: CapabilityReport | None = None
        self._preflight_dry_run: bool | None = None
        # -- failure-injection seams (test setup only, never production) ------
        self.fail_next_trace_verification = 0
        """Make the next N ``verify_trace`` calls fail with a typed error."""
        self.fail_artifact_verification = False
        """Make ``verify_named_json`` fail for traces not in the exempt set."""
        self.artifact_verification_exempt_trace_ids: set[str] = set()
        self.fail_retire_trace_ids: set[str] = set()
        """Trace ids whose ``retire_trace`` fails with a typed error."""
        self.retire_hook: Callable[[str, str], None] | None = None
        """Optional callable invoked before each retirement; may raise."""
        self.ambiguous_next_creates = 0
        """Make the next N ``create_trace`` calls store the trace and then raise
        a transient (timeout-shaped) error, modeling an ambiguous create."""
        self.candidate_verification_error_ids: set[str] = set()
        """Trace ids whose candidate verification raises an unverifiable error."""

    # -- capability probe ----------------------------------------------------

    def probe(self, request: CapabilityRequest, dry_run: bool | None = None) -> CapabilityReport:
        """Record and return the configured capability outcome."""
        effective_dry_run = self.dry_run if dry_run is None else dry_run
        attestation_error: str | None = None
        if self.compatible:
            try:
                if attest_server_version(self.tracking_uri, transport=self.version_transport) != PINNED_MLFLOW_VERSION:
                    attestation_error = "the fake tracking server version did not match the pinned profile"
            except Exception:
                attestation_error = "the fake tracking server version attestation failed"
        if self.compatible and attestation_error is None:
            report = CapabilityReport(
                is_compatible=True,
                required_methods=REQUIRED_METHODS,
                max_inline_payload_bytes=PINNED_MAX_INLINE_PAYLOAD_BYTES,
                named_artifact_supported=True,
                retirement_supported=True,
                needs_retirement=request.needs_retirement or not effective_dry_run,
                needs_named_artifacts=request.needs_named_artifacts,
                dry_run=effective_dry_run,
                safe_error=None,
            )
        else:
            report = CapabilityReport(
                is_compatible=False,
                required_methods=frozenset(),
                max_inline_payload_bytes=0,
                named_artifact_supported=False,
                retirement_supported=False,
                needs_retirement=request.needs_retirement or not effective_dry_run,
                needs_named_artifacts=request.needs_named_artifacts,
                dry_run=effective_dry_run,
                safe_error=attestation_error or "the fake gateway was configured incompatible",
            )
        self._preflight = report
        self._preflight_dry_run = effective_dry_run
        self.events.append(GatewayEvent("probe", (effective_dry_run, report.is_compatible)))
        return report

    # -- experiment resolution -------------------------------------------------

    def resolve_experiment(self, name: str, create: bool) -> ExperimentRef:
        existing = self._experiments.get(name)
        if existing is not None:
            # Mirror the real gateway: a found experiment resolves to a
            # fresh reference that was not created by this resolution.
            reused = ExperimentRef(
                name=name,
                experiment_id=existing.experiment_id,
                artifact_location=existing.artifact_location,
                created=False,
                planned_creation=False,
            )
            self.events.append(GatewayEvent("experiment_reuse", (name, reused.experiment_id)))
            return reused
        if not create:
            raise ConfigurationError(
                "experiment_not_found", "the requested experiment does not exist and creation was not requested"
            )
        if self.dry_run:
            planned = ExperimentRef(
                name=name,
                experiment_id=None,
                artifact_location=None,
                created=False,
                planned_creation=True,
            )
            self.events.append(GatewayEvent("experiment_plan_create", (name,)))
            return planned
        self._require_mutation_allowed()
        self._next_experiment += 1
        experiment_id = f"fake-exp-{self._next_experiment}"
        reference = ExperimentRef(
            name=name,
            experiment_id=experiment_id,
            artifact_location=f"fake-artifact://experiments/{experiment_id}",
            created=True,
            planned_creation=False,
        )
        self._experiments[name] = reference
        self.mutations.append(("experiment_create", name, experiment_id))
        self.events.append(GatewayEvent("experiment_create", (name, experiment_id)))
        return reference

    # -- trace lifecycle ---------------------------------------------------------

    def create_trace(self, spec: TraceCreateRequest) -> CreatedTrace:
        from opencode_mlflow.mlflow.gateway import validate_required_metadata, validate_source_timing

        self._require_mutation_allowed()
        validate_required_metadata(spec)
        validate_source_timing(spec)
        self._next_trace += 1
        trace_id = f"fake-tr-{self._next_trace}"
        ambiguous = self.ambiguous_next_creates > 0
        if ambiguous:
            self.ambiguous_next_creates -= 1
        self._next_span += 1
        root_span_id = f"fake-span-{self._next_span}"
        metadata = dict(spec.metadata)
        metadata[TRACE_SESSION_METADATA_KEY] = spec.native_session_id
        trace = _StoredTrace(
            trace_id=trace_id,
            experiment_id=spec.experiment_id,
            name=spec.name,
            native_session_id=spec.native_session_id,
            root_span_id=root_span_id,
            client_request_id=spec.client_request_id,
            inputs=spec.root_inputs,
            outputs=spec.root_outputs,
            root_attributes=dict(spec.root_attributes),
            start_time_ns=spec.start_time_ns,
            end_time_ns=spec.end_time_ns,
            tags=dict(spec.root_tags),
            metadata=metadata,
            spans_by_source={},
        )
        self._traces[trace_id] = trace
        self.mutations.append(("trace_create", trace_id))
        self.events.append(
            GatewayEvent(
                "trace_create",
                (trace_id, spec.client_request_id, spec.experiment_id, spec.native_session_id),
            )
        )
        if ambiguous:
            # The trace exists but the caller observes a timeout: the create is
            # ambiguous, exactly the situation the recovery search handles.
            raise FakeMlflowException(
                f"synthetic ambiguous create timeout for {trace_id}",
                "TEMPORARILY_UNAVAILABLE",
            )
        parents: dict[str, _StoredSpan] = {}
        last_child_status: str | None = None
        for child in spec.child_specs():
            parent = parents.get(child.parent_source_key) if child.parent_source_key is not None else None
            if child.parent_source_key is not None and parent is None:
                raise ItemImportError(
                    "unresolvable_parent_span",
                    "a child span referenced a parent source key that was not created",
                )
            self._next_span += 1
            span = _StoredSpan(
                source_key=child.source_key,
                span_id=f"fake-span-{self._next_span}",
                parent_id=parent.span_id if parent is not None else root_span_id,
                name=child.name,
                span_type=child.span_type,
                inputs=child.inputs,
                outputs=child.outputs,
                attributes=dict(child.attributes),
                tags=dict(child.tags),
                status=RETRIEVED_SPAN_STATUS.get(child.status, child.status),
                start_time_ns=child.start_time_ns,
                end_time_ns=child.end_time_ns,
            )
            trace.spans_by_source[child.source_key] = span
            # Every started child is resolvable as a later child's parent,
            # exactly like the real gateway's source_key -> LiveSpan map.
            parents[child.source_key] = span
            self.events.append(
                GatewayEvent(
                    "span_start",
                    (trace_id, span.span_id, child.source_key, span.parent_id, child.name, child.span_type),
                )
            )
            self.events.append(
                GatewayEvent(
                    "span_end",
                    (trace_id, span.span_id, child.source_key, child.status, child.end_time_ns),
                )
            )
            last_child_status = child.status
        root_status = _effective_root_status(spec, last_child_status)
        trace.root_status = root_status
        trace.state = TRACE_STATE_FOR_ROOT_STATUS.get(root_status, root_status)
        trace.ended = True
        self.events.append(GatewayEvent("trace_end", (trace_id, root_span_id, root_status)))
        return CreatedTrace(
            trace_id=trace_id,
            root_span_id=root_span_id,
            span_ids_by_source={key: span.span_id for key, span in trace.spans_by_source.items()},
            verified_metadata=metadata,
            root_status=root_status,
            verified_tags=dict(spec.root_tags),
            artifact_provenance=ArtifactProvenance.from_request(spec),
        )

    def verify_trace(self, created: CreatedTrace, expected: TraceCreateRequest) -> None:
        trace = self._traces.get(created.trace_id)
        if trace is None:
            raise ItemImportError("trace_verification_failed", "the created trace is missing")
        if self.fail_next_trace_verification > 0:
            self.fail_next_trace_verification -= 1
            raise ItemImportError("trace_verification_failed", "synthetic trace verification failure")
        if trace.name != expected.name:
            raise ItemImportError("trace_verification_failed", "the root span name did not round-trip")
        if trace.inputs != expected.root_inputs:
            raise ItemImportError("trace_verification_failed", "the complete root inputs did not round-trip")
        if trace.outputs != expected.root_outputs:
            raise ItemImportError("trace_verification_failed", "the complete root outputs did not round-trip")
        if trace.root_status != created.root_status:
            raise ItemImportError("trace_verification_failed", "the root span status did not round-trip")
        if trace.state != TRACE_STATE_FOR_ROOT_STATUS.get(created.root_status, created.root_status):
            raise ItemImportError("trace_verification_failed", "the trace state did not match the root span status")
        for key, value in expected.root_attributes.items():
            if trace.root_attributes.get(key) != value:
                raise ItemImportError("trace_verification_failed", "an expected root span attribute did not round-trip")
        if expected.start_time_ns is not None and trace.start_time_ns != expected.start_time_ns:
            raise ItemImportError("trace_verification_failed", "the trace start time did not round-trip")
        if expected.end_time_ns is not None and trace.end_time_ns != expected.end_time_ns:
            raise ItemImportError("trace_verification_failed", "the trace end time did not round-trip")
        if trace.metadata.get(TRACE_SESSION_METADATA_KEY) != expected.native_session_id:
            raise ItemImportError("trace_verification_failed", "the native session metadata did not round-trip")
        for key, value in expected.root_tags.items():
            if trace.tags.get(key) != value:
                raise ItemImportError("trace_verification_failed", "an expected trace tag did not round-trip")
        for key, value in expected.metadata.items():
            if trace.metadata.get(key) != value:
                raise ItemImportError(
                    "trace_verification_failed", "an expected trace metadata value did not round-trip"
                )
        expected_children: dict[str, TraceChildRequest] = {}
        for child in expected.child_specs():
            if child.source_key in expected_children:
                raise ItemImportError("trace_verification_failed", "duplicate expected child source keys")
            expected_children[child.source_key] = child
        if set(expected_children) != set(created.span_ids_by_source):
            raise ItemImportError("trace_verification_failed", "the expected child identity set did not round-trip")
        seen: set[str] = set()
        for source_key, span in trace.spans_by_source.items():
            child = expected_children.get(source_key)
            if child is None or span.span_id in seen:
                raise ItemImportError("trace_verification_failed", "the retrieved trace contains an unexpected child")
            seen.add(span.span_id)
            expected_parent = trace.root_span_id
            if child.parent_source_key is not None:
                parent = trace.spans_by_source.get(child.parent_source_key)
                if parent is None:
                    raise ItemImportError("trace_verification_failed", "an expected child parent is missing")
                expected_parent = parent.span_id
            if span.span_id != created.span_ids_by_source[source_key]:
                raise ItemImportError("trace_verification_failed", "a child span identity did not round-trip")
            if (
                span.name != child.name
                or span.span_type != child.span_type
                or span.inputs != child.inputs
                or span.outputs != child.outputs
                or span.parent_id != expected_parent
                or span.status != RETRIEVED_SPAN_STATUS.get(child.status, child.status)
                or span.start_time_ns != child.start_time_ns
                or span.end_time_ns != child.end_time_ns
            ):
                raise ItemImportError("trace_verification_failed", "a child span field did not round-trip")
            for key, value in child.attributes.items():
                if span.attributes.get(key) != value:
                    raise ItemImportError("trace_verification_failed", "a child span attribute did not round-trip")
            for key, value in child.tags.items():
                if span.tags.get(key) != value:
                    raise ItemImportError("trace_verification_failed", "a child span tag did not round-trip")
        if seen != set(created.span_ids_by_source.values()):
            raise ItemImportError("trace_verification_failed", "a created child span is missing from the trace")
        self.events.append(GatewayEvent("trace_verify", (created.trace_id, True)))

    def iter_owned_traces(self, experiment_id: str) -> Iterator[OwnedTrace]:
        for trace in list(self._traces.values()):
            if trace.experiment_id != experiment_id:
                continue
            if self._is_owned(trace):
                yield OwnedTrace(
                    trace_id=trace.trace_id,
                    request_time_ms=trace.start_time_ns // 1_000_000 if trace.start_time_ns is not None else None,
                    tags=trace.tags,
                    metadata=trace.metadata,
                )

    def owned_traces(self, candidates: Iterable[OwnedTraceCandidate]) -> list[OwnedTraceCandidate]:
        """Seed and return owned candidates for planning.

        Each candidate not already stored is seeded as a minimal owned
        trace carrying its identity metadata, so later retrieval and
        retirement work like a real indexed candidate; already-stored
        traces (created through :meth:`create_trace`) keep their full
        state. Returns the candidates in the given order — the pure
        planner consumes them without gateway calls.
        """
        seeded: list[OwnedTraceCandidate] = []
        for candidate in candidates:
            if candidate.trace_id not in self._traces:
                self._next_trace = max(self._next_trace, _numeric_suffix(candidate.trace_id))
                self._next_span += 1
                tags: dict[str, str] = {}
                if candidate.artifact_sha256 is not None:
                    tags[ARTIFACT_SHA256_TAG] = candidate.artifact_sha256
                if candidate.artifact_size is not None:
                    tags[ARTIFACT_SIZE_TAG] = str(candidate.artifact_size)
                self._traces[candidate.trace_id] = _StoredTrace(
                    trace_id=candidate.trace_id,
                    experiment_id=candidate.experiment_id if candidate.experiment_id is not None else "fake-exp-seeded",
                    name="seeded owned candidate",
                    native_session_id=candidate.session_id,
                    root_span_id=f"fake-span-seeded-{candidate.trace_id}",
                    client_request_id="seeded",
                    inputs=None,
                    outputs=None,
                    root_attributes={},
                    start_time_ns=(
                        candidate.request_time_ms * 1_000_000 if candidate.request_time_ms is not None else None
                    ),
                    end_time_ns=None,
                    tags=tags,
                    metadata={
                        "opencode.importer": candidate.importer_marker or "opencode-mlflow",
                        "opencode.session_id": candidate.session_id,
                        "opencode.turn_key": candidate.turn_key,
                        **(
                            {"opencode.turn_index": str(candidate.turn_index)}
                            if candidate.turn_index is not None
                            else {}
                        ),
                        **(
                            {"opencode.source_fingerprint": candidate.source_fingerprint}
                            if candidate.source_fingerprint is not None
                            else {}
                        ),
                        **(
                            {"opencode.source_schema": candidate.source_schema}
                            if candidate.source_schema is not None
                            else {}
                        ),
                        **(
                            {"opencode.importer_version": candidate.importer_version}
                            if candidate.importer_version is not None
                            else {}
                        ),
                        **(
                            {"opencode.mapping_version": candidate.mapping_version}
                            if candidate.mapping_version is not None
                            else {}
                        ),
                        **(
                            {"opencode.artifact_schema_version": candidate.artifact_schema_version}
                            if candidate.artifact_schema_version is not None
                            else {}
                        ),
                        **(
                            {"opencode.catalog_version": candidate.catalog_version}
                            if candidate.catalog_version is not None
                            else {}
                        ),
                        **(
                            {"opencode.redaction_mode": candidate.redaction_mode}
                            if candidate.redaction_mode is not None
                            else {}
                        ),
                        **(
                            {"opencode.render_fingerprint": candidate.render_fingerprint}
                            if candidate.render_fingerprint is not None
                            else {}
                        ),
                        **(
                            {"opencode.operation_key": candidate.operation_key}
                            if candidate.operation_key is not None
                            else {}
                        ),
                        **({"opencode.warnings": ",".join(candidate.warnings)} if candidate.warnings else {}),
                    },
                    spans_by_source={},
                    root_status="OK",
                    state="OK",
                    ended=True,
                )
            seeded.append(candidate)
        return seeded

    def verify_owned_candidate(
        self,
        candidate: OwnedTraceCandidate,
        expected: CandidateContract,
    ) -> CandidateVerification:
        """Parity with the real gateway's candidate verification.

        Retrieves the stored trace (full-payload retrieval stand-in),
        re-verifies the exact ownership marker, compares the contract's six
        fingerprint fields, reads the RECORDED artifact path and digest from
        the stored trace's tags, and verifies the named artifact's exact
        bytes at the candidate's OWN recorded location through the public
        :meth:`verify_named_json` (the current run's would-be path is
        deliberately not used — it moves whenever the session payload
        changes). A completed non-match is returned with a reason; a
        missing trace raises.
        """
        if candidate.trace_id in self.candidate_verification_error_ids:
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            raise ItemImportError("candidate_verification_failed", "the owned candidate could not be verified")
        trace = self._traces.get(candidate.trace_id)
        if trace is None:
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            raise ItemImportError(
                "trace_verification_failed",
                "the owned candidate trace could not be retrieved",
            )
        if expected.ownership_errors:
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            return CandidateVerification(
                trace_id=candidate.trace_id,
                matches=False,
                reason="ownership_incomplete",
                error_code="ownership_conflict",
            )
        if not self._has_exact_ownership_marker(trace):
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            return CandidateVerification(
                trace_id=candidate.trace_id,
                matches=False,
                reason="ownership_missing",
                error_code="ownership_conflict",
            )
        _, warning_metadata_valid = _parse_warning_metadata(trace.metadata.get("opencode.warnings"))
        contract_errors = _ownership_contract_errors(
            trace_id=trace.trace_id,
            experiment_id=trace.experiment_id,
            importer_marker=_candidate_importer_marker(trace.tags, trace.metadata),
            session_id=trace.metadata.get(OWNERSHIP_SESSION_KEY),
            turn_key=trace.metadata.get(OWNERSHIP_TURN_KEY),
            turn_index=_parse_turn_index(trace.metadata.get("opencode.turn_index")),
            source_fingerprint=trace.metadata.get(OWNERSHIP_SOURCE_FINGERPRINT_KEY),
            source_schema=trace.metadata.get(OWNERSHIP_SOURCE_SCHEMA_KEY),
            importer_version=trace.metadata.get("opencode.importer_version"),
            mapping_version=trace.metadata.get("opencode.mapping_version"),
            artifact_schema_version=trace.metadata.get("opencode.artifact_schema_version"),
            redaction_mode=trace.metadata.get("opencode.redaction_mode"),
            catalog_version=trace.metadata.get(OWNERSHIP_CATALOG_VERSION_KEY),
            render_fingerprint=trace.metadata.get(OWNERSHIP_RENDER_FINGERPRINT_KEY),
            operation_key=trace.metadata.get(OWNERSHIP_OPERATION_KEY),
            warning_metadata_valid=warning_metadata_valid,
        )
        if contract_errors:
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            return CandidateVerification(
                trace_id=candidate.trace_id,
                matches=False,
                reason="ownership_incomplete",
                error_code="ownership_conflict",
            )
        for key, value in ((OWNERSHIP_SESSION_KEY, candidate.session_id), (OWNERSHIP_TURN_KEY, candidate.turn_key)):
            if trace.metadata.get(key) != value:
                self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
                return CandidateVerification(
                    trace_id=candidate.trace_id,
                    matches=False,
                    reason="ownership_identity_changed",
                    error_code="ownership_conflict",
                )
        expected_fields = (
            ("opencode.importer", trace.metadata.get("opencode.importer"), expected.importer_marker),
            (OWNERSHIP_SESSION_KEY, trace.metadata.get(OWNERSHIP_SESSION_KEY), expected.session_id),
            ("opencode.turn_index", trace.metadata.get("opencode.turn_index"), str(expected.turn_index)),
            (OWNERSHIP_TURN_KEY, trace.metadata.get(OWNERSHIP_TURN_KEY), expected.turn_key),
            (
                OWNERSHIP_SOURCE_FINGERPRINT_KEY,
                trace.metadata.get(OWNERSHIP_SOURCE_FINGERPRINT_KEY),
                expected.source_fingerprint,
            ),
            (OWNERSHIP_SOURCE_SCHEMA_KEY, trace.metadata.get(OWNERSHIP_SOURCE_SCHEMA_KEY), expected.source_schema),
            ("opencode.importer_version", trace.metadata.get("opencode.importer_version"), expected.importer_version),
            ("opencode.mapping_version", trace.metadata.get("opencode.mapping_version"), expected.mapping_version),
            (
                "opencode.artifact_schema_version",
                trace.metadata.get("opencode.artifact_schema_version"),
                expected.artifact_schema_version,
            ),
            ("opencode.redaction_mode", trace.metadata.get("opencode.redaction_mode"), expected.redaction_mode),
            (
                OWNERSHIP_CATALOG_VERSION_KEY,
                trace.metadata.get(OWNERSHIP_CATALOG_VERSION_KEY),
                expected.catalog_version,
            ),
            (
                OWNERSHIP_RENDER_FINGERPRINT_KEY,
                trace.metadata.get(OWNERSHIP_RENDER_FINGERPRINT_KEY),
                expected.render_fingerprint,
            ),
            (OWNERSHIP_OPERATION_KEY, trace.metadata.get(OWNERSHIP_OPERATION_KEY), expected.operation_key),
        )
        if trace.experiment_id != expected.experiment_id:
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="metadata_mismatch")
        for _key, actual, value in expected_fields:
            if actual != value:
                self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
                return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="metadata_mismatch")
        actual_warnings, actual_warning_metadata_valid = _parse_warning_metadata(
            trace.metadata.get("opencode.warnings")
        )
        if not actual_warning_metadata_valid:
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            return CandidateVerification(
                trace_id=candidate.trace_id,
                matches=False,
                reason="ownership_incomplete",
                error_code="ownership_conflict",
            )
        if actual_warnings != expected.warnings:
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="metadata_mismatch")
        recorded_path = trace.tags.get(ARTIFACT_PATH_TAG) or trace.metadata.get(ARTIFACT_PATH_TAG)
        if not isinstance(recorded_path, str) or not recorded_path:
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="artifact_path_missing")
        recorded_sha256 = trace.tags.get(ARTIFACT_SHA256_TAG) or trace.metadata.get(ARTIFACT_SHA256_TAG)
        recorded_size_text = trace.tags.get(ARTIFACT_SIZE_TAG) or trace.metadata.get(ARTIFACT_SIZE_TAG)
        if not recorded_sha256 or recorded_size_text is None:
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="artifact_digest_missing")
        try:
            self.verify_named_json(
                candidate.trace_id,
                recorded_path,
                recorded_sha256,
                int(recorded_size_text),
            )
        except ItemImportError:
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="artifact_mismatch")
        except ValueError:
            self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, False)))
            return CandidateVerification(trace_id=candidate.trace_id, matches=False, reason="artifact_digest_missing")
        self.events.append(GatewayEvent("candidate_verify", (candidate.trace_id, True)))
        return CandidateVerification(trace_id=candidate.trace_id, matches=True)

    def verify_owned_trace_for_retirement(
        self,
        experiment_id: str,
        trace_id: str,
        *,
        session_id: str,
        turn_key: str,
        expected_metadata: Mapping[str, str] | None = None,
    ) -> None:
        """Re-fetch and validate the fake target before a retirement."""
        trace = self._traces.get(trace_id)
        if trace is None or trace.experiment_id != experiment_id:
            raise ItemImportError("ownership_conflict", "the trace identity changed before retirement")
        if not self._has_exact_ownership_marker(trace):
            raise ItemImportError("ownership_conflict", "the importer ownership marker changed before retirement")
        _, warning_metadata_valid = _parse_warning_metadata(trace.metadata.get("opencode.warnings"))
        contract_errors = _ownership_contract_errors(
            trace_id=trace.trace_id,
            experiment_id=trace.experiment_id,
            importer_marker=_candidate_importer_marker(trace.tags, trace.metadata),
            session_id=trace.metadata.get(OWNERSHIP_SESSION_KEY),
            turn_key=trace.metadata.get(OWNERSHIP_TURN_KEY),
            turn_index=_parse_turn_index(trace.metadata.get("opencode.turn_index")),
            source_fingerprint=trace.metadata.get(OWNERSHIP_SOURCE_FINGERPRINT_KEY),
            source_schema=trace.metadata.get(OWNERSHIP_SOURCE_SCHEMA_KEY),
            importer_version=trace.metadata.get("opencode.importer_version"),
            mapping_version=trace.metadata.get("opencode.mapping_version"),
            artifact_schema_version=trace.metadata.get("opencode.artifact_schema_version"),
            redaction_mode=trace.metadata.get("opencode.redaction_mode"),
            catalog_version=trace.metadata.get(OWNERSHIP_CATALOG_VERSION_KEY),
            render_fingerprint=trace.metadata.get(OWNERSHIP_RENDER_FINGERPRINT_KEY),
            operation_key=trace.metadata.get(OWNERSHIP_OPERATION_KEY),
            warning_metadata_valid=warning_metadata_valid,
        )
        if contract_errors:
            raise ItemImportError("ownership_conflict", "the trace ownership contract could not be validated")
        for key, expected in ((OWNERSHIP_SESSION_KEY, session_id), (OWNERSHIP_TURN_KEY, turn_key)):
            values = [mapping[key] for mapping in (trace.metadata, trace.tags) if key in mapping]
            if not values or any(value != expected for value in values):
                raise ItemImportError("ownership_conflict", "the trace source identity changed before retirement")
        complete_expected = self._complete_retirement_metadata(trace, expected_metadata)
        for key, expected in complete_expected.items():
            actual = (
                trace.metadata.get(key, "")
                if key == "opencode.warnings"
                else trace.metadata.get(key, trace.tags.get(key, ""))
            )
            if actual != expected:
                raise ItemImportError("ownership_conflict", "the trace candidate contract changed before retirement")

    def record_artifact_verification(self, trace_id: str, sha256: str, size: int) -> None:
        """Parity with the real gateway's recorded-digest tag update."""
        self._require_mutation_allowed()
        trace = self._traces.get(trace_id)
        if trace is None:
            raise ItemImportError("artifact_record_failed", "the trace could not be updated")
        for key, value in ((ARTIFACT_SHA256_TAG, sha256), (ARTIFACT_SIZE_TAG, str(size))):
            trace.tags[key] = value
            self.mutations.append(("trace_tag_set", trace_id, key))
            self.events.append(GatewayEvent("trace_tag_set", (trace_id, key, value)))

    # -- state inspection ------------------------------------------------------

    def has_trace(self, trace_id: str) -> bool:
        """Whether the trace is still stored (not retired)."""
        return trace_id in self._traces

    def trace_ids(self) -> list[str]:
        """Every stored trace id, sorted."""
        return sorted(self._traces)

    def stored_trace(self, trace_id: str) -> _StoredTrace | None:
        """The stored trace record, or ``None`` when retired/unknown."""
        return self._traces.get(trace_id)

    def stored_experiment(self, name: str) -> ExperimentRef | None:
        """The stored experiment reference for ``name`` (read-only observable).

        Scenarios use this to pin that an existing experiment was reused
        without changing its settings: the returned reference carries the
        exact experiment id and artifact location the target stored.
        """
        return self._experiments.get(name)

    def stored_artifact(self, trace_id: str, relative_path: str) -> bytes:
        """The bytes stored for one named artifact."""
        return self._artifacts[trace_id][relative_path]

    @staticmethod
    def _is_owned(trace: _StoredTrace) -> bool:
        return (
            trace.metadata.get("opencode.importer") == "opencode-mlflow"
            or trace.tags.get("opencode.importer") == "opencode-mlflow"
        )

    @staticmethod
    def _has_exact_ownership_marker(trace: _StoredTrace) -> bool:
        values = [
            mapping["opencode.importer"] for mapping in (trace.metadata, trace.tags) if "opencode.importer" in mapping
        ]
        return bool(values) and all(value == "opencode-mlflow" for value in values)

    @staticmethod
    def _complete_retirement_metadata(
        trace: _StoredTrace,
        expected_metadata: Mapping[str, str] | None,
    ) -> dict[str, str]:
        keys = (
            "opencode.importer",
            "opencode.session_id",
            "opencode.turn_index",
            "opencode.turn_key",
            "opencode.source_fingerprint",
            "opencode.source_schema",
            "opencode.importer_version",
            "opencode.mapping_version",
            "opencode.artifact_schema_version",
            "opencode.redaction_mode",
            "opencode.catalog_version",
            "opencode.render_fingerprint",
            "opencode.operation_key",
            "opencode.warnings",
        )
        supplied = expected_metadata or {}
        return {
            key: supplied[key] if key in supplied else trace.metadata.get(key, trace.tags.get(key, "")) for key in keys
        }

    # -- artifacts ------------------------------------------------------------

    def upload_named_json(
        self,
        trace_id: str,
        relative_path: str,
        local_file: Path,
        chunk_size: int = DEFAULT_UPLOAD_CHUNK_BYTES,
    ) -> ArtifactVerification:
        self._require_mutation_allowed()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        digest = hashlib.sha256()
        size = 0
        chunks: list[bytes] = []
        with Path(local_file).open("rb") as source:
            while chunk := source.read(chunk_size):
                digest.update(chunk)
                size += len(chunk)
                chunks.append(chunk)
        stored = self._artifacts.setdefault(trace_id, {})
        stored[relative_path] = b"".join(chunks)
        self.mutations.append(("artifact_upload", trace_id, relative_path))
        self.events.append(GatewayEvent("artifact_upload", (trace_id, relative_path, digest.hexdigest(), size)))
        return ArtifactVerification(
            trace_id=trace_id,
            relative_path=relative_path,
            sha256=digest.hexdigest(),
            size=size,
        )

    def verify_named_json(self, trace_id: str, relative_path: str, sha256: str, size: int) -> None:
        if self.fail_artifact_verification and trace_id not in self.artifact_verification_exempt_trace_ids:
            raise ItemImportError("artifact_verification_failed", "synthetic artifact verification failure")
        stored = self._artifacts.get(trace_id, {}).get(relative_path)
        if stored is None:
            raise ItemImportError("artifact_verification_failed", "the named artifact is missing")
        digest = hashlib.sha256(stored).hexdigest()
        if len(stored) != size:
            raise ItemImportError("artifact_verification_failed", "artifact size mismatch")
        if digest != sha256:
            raise ItemImportError("artifact_verification_failed", "artifact sha256 mismatch")
        self.events.append(GatewayEvent("artifact_verify", (trace_id, relative_path, True)))

    # -- retirement ---------------------------------------------------------------

    def retire_trace(
        self,
        experiment_id: str,
        trace_id: str,
        *,
        session_id: str | None = None,
        turn_key: str | None = None,
        expected_metadata: Mapping[str, str] | None = None,
    ) -> None:
        self._require_mutation_allowed()
        if self.retire_hook is not None:
            self.retire_hook(experiment_id, trace_id)
        if trace_id in self.fail_retire_trace_ids:
            raise ItemImportError("trace_retirement_failed", "synthetic retirement failure")
        trace = self._traces.get(trace_id)
        if trace is None or trace.experiment_id != experiment_id:
            raise ItemImportError("trace_retirement_failed", "the tracking target did not report the trace as retired")
        if not self._has_exact_ownership_marker(trace):
            raise ItemImportError("ownership_conflict", "the importer ownership marker changed before retirement")
        if session_id is None or turn_key is None:
            raise ItemImportError("ownership_conflict", "the trace source identity was not supplied for retirement")
        for key, expected in ((OWNERSHIP_SESSION_KEY, session_id), (OWNERSHIP_TURN_KEY, turn_key)):
            values = [mapping[key] for mapping in (trace.metadata, trace.tags) if key in mapping]
            if not values or any(value != expected for value in values):
                raise ItemImportError("ownership_conflict", "the trace source identity changed before retirement")
        complete_expected = self._complete_retirement_metadata(trace, expected_metadata)
        for key, expected in complete_expected.items():
            actual = (
                trace.metadata.get(key, "")
                if key == "opencode.warnings"
                else trace.metadata.get(key, trace.tags.get(key, ""))
            )
            if actual != expected:
                raise ItemImportError("ownership_conflict", "the trace candidate contract changed before retirement")
        del self._traces[trace_id]
        self._artifacts.pop(trace_id, None)
        self.mutations.append(("trace_retire", experiment_id, trace_id))
        self.events.append(GatewayEvent("trace_retire", (experiment_id, trace_id)))

    # -- gates ----------------------------------------------------------------------

    def _require_mutation_allowed(self) -> None:
        if self.dry_run:
            raise CapabilityError(
                "dry_run_mutation", "the gateway is in dry-run mode and must not mutate the tracking target"
            )
        if self._preflight is None:
            raise CapabilityError(
                "mlflow_preflight_required", "the capability check must pass before any MLflow mutation"
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
