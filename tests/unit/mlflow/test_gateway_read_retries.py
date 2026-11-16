"""Read-only gateway fault injection using the observed SDK exception shape."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import replace
from http.client import RemoteDisconnected
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode
from requests.exceptions import ConnectionError as RequestsConnectionError
from urllib3 import HTTPConnectionPool
from urllib3.exceptions import MaxRetryError, ReadTimeoutError

from mlflow_import import resources
from mlflow_import.errors import ConfigurationError, ItemImportError
from mlflow_import.mlflow.capabilities import CapabilityRequest
from mlflow_import.mlflow.gateway import MlflowGateway, TraceArtifactSink
from mlflow_import.mlflow.reconciliation import CandidateContract, OwnedTraceCandidate
from mlflow_import.retry import RetryPolicy, is_transient_error, retry_budget, retry_call
from tests.support.fake_gateway import FakeMlflowClient, resource_qualified_payload, trace_spec

pytestmark = pytest.mark.unit

READ_POLICY = RetryPolicy(max_attempts=3, initial_delay_seconds=0, max_delay_seconds=0)


class _Page(list[Any]):
    def __init__(self, items: list[Any], token: str | None = None) -> None:
        super().__init__(items)
        self.token = token


class _ArtifactResponse(io.BytesIO):
    status = 200

    def __init__(self, body: bytes, *, disconnect: bool = False) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": "2"}
        self.disconnect = disconnect

    def read1(self, size: int | None = -1, /) -> bytes:
        if self.disconnect:
            raise RemoteDisconnected("synthetic artifact disconnect")
        return super().read1(size)


def _transport_failure(*, timeout: bool = False) -> MlflowException:
    """Real SDK/library classes and implicit contexts, with synthetic text only."""
    pool = HTTPConnectionPool("synthetic.invalid")
    try:
        if timeout:
            try:
                raise TimeoutError("synthetic timeout")
            except TimeoutError:
                raise ReadTimeoutError(pool, "/synthetic", "synthetic timeout") from None
        raise RemoteDisconnected("synthetic disconnect")
    except (RemoteDisconnected, ReadTimeoutError) as disconnect:
        try:
            raise MaxRetryError(pool, "/synthetic", disconnect) from None
        except MaxRetryError as exhausted:
            try:
                raise RequestsConnectionError(exhausted) from None
            except RequestsConnectionError:
                try:
                    raise MlflowException("synthetic target transport failure") from None
                except MlflowException as wrapped:
                    return wrapped


@pytest.fixture
def target(tmp_path: Path) -> SimpleNamespace:
    client = Mock(spec=MlflowClient)
    client.tracking_uri = "https://synthetic.invalid"
    payload = json.dumps(
        {
            "schema_version": "mlflow-import.turn-map.v1",
            "identity_map": {"trace_id": "trace-1"},
            "turn_key": None,
            "turn_fingerprint": None,
        }
    ).encode()
    checksum = hashlib.sha256(payload).hexdigest()
    proof, data = resource_qualified_payload()
    info = SimpleNamespace(
        trace_id="trace-1",
        experiment_id="exp-1",
        tags={
            **proof,
            "opencode.importer": "mlflow-import",
            "opencode.artifact_path": "mlflow-import/v2/map.json",
            "opencode.artifact_sha256": checksum,
            "opencode.artifact_size": str(len(payload)),
            "mlflow_import.session_artifact": "mlflow-import/v2/session.json",
            "mlflow_import.session_artifact_sha256": checksum,
            "mlflow_import.session_artifact_size": str(len(payload)),
            "mlflow.artifactLocation": tmp_path.as_uri(),
        },
        trace_metadata={"opencode.session_id": "session-1"},
        state=SimpleNamespace(value="OK"),
        request_time=1,
        execution_duration=2,
        client_request_id=None,
        request_preview=None,
        response_preview=None,
        token_usage=None,
        cost=None,
    )
    trace = SimpleNamespace(trace_id="trace-1", info=info, data=data)
    client.get_trace_info = Mock(return_value=info)
    client.get_trace.return_value = trace
    client.get_experiment.return_value = SimpleNamespace(artifact_location=tmp_path.as_uri())
    client.get_experiment_by_name.return_value = SimpleNamespace(
        name="experiment", experiment_id="exp-1", artifact_location=tmp_path.as_uri()
    )
    repository = Mock()
    repository.artifact_uri = tmp_path.as_uri()

    def download(path: str, root: str) -> str:
        local = Path(root) / Path(path).name
        local.write_bytes(payload)
        return str(local)

    repository.download_artifacts.side_effect = download
    factory = Mock(return_value=repository)
    gateway = MlflowGateway(client, artifact_repository_factory=factory, read_retry_policy=READ_POLICY)
    return SimpleNamespace(
        gateway=gateway, client=client, trace=trace, repository=repository, factory=factory, payload=payload
    )


@pytest.mark.parametrize("timeout", [False, True])
def test_observed_wrapped_transport_shape_is_transient(timeout: bool) -> None:
    error = _transport_failure(timeout=timeout)
    assert error.error_code == "INTERNAL_ERROR"
    connection = error.__context__
    assert isinstance(connection, RequestsConnectionError)
    exhausted = connection.__context__
    assert isinstance(exhausted, MaxRetryError)
    assert isinstance(exhausted.__context__, ReadTimeoutError if timeout else RemoteDisconnected)
    assert is_transient_error(error)


@pytest.mark.parametrize("boundary", ["trace_info", "trace_exists", "replacement_reference", "verify_reference"])
@pytest.mark.parametrize("timeout", [False, True])
def test_full_trace_reads_retry_identical_request_without_writes(
    target: SimpleNamespace, boundary: str, timeout: bool
) -> None:
    gateway, client = target.gateway, target.client
    reference = gateway.replacement_reference("trace-1")
    client.reset_mock()
    target.factory.reset_mock()
    metadata_only = boundary in {"trace_info", "trace_exists"}
    reader = client.get_trace_info if metadata_only else client.get_trace
    reader.side_effect = [_transport_failure(timeout=timeout), target.trace.info if metadata_only else target.trace]

    if boundary == "verify_reference":
        gateway.verify_replacement_reference(reference)
    else:
        result = getattr(gateway, boundary)("trace-1")
        if boundary == "trace_exists":
            assert result is True
        elif boundary == "trace_info":
            assert result is target.trace.info
        else:
            assert result == reference

    assert reader.call_args_list == [call("trace-1") if metadata_only else call("trace-1", display=False)] * 2
    if metadata_only:
        client.get_trace.assert_not_called()
    assert {item[0] for item in client.mock_calls} <= {"get_trace", "get_trace_info", "get_experiment"}
    assert {item[0] for item in target.repository.mock_calls} <= {"download_artifacts"}


def test_full_created_trace_verification_retries_only_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeMlflowClient()
    setup = client.gateway()
    # Setup is entirely in-memory; capture every mutation before fault injection.
    setup.probe(CapabilityRequest())
    request = trace_spec()
    created = setup.create_trace(request)
    trace = client.get_trace(created.trace_id)
    state, mutations, span_calls = client.snapshot_state(), list(client.mutations), list(client.trace_api.calls)
    read = Mock(side_effect=[_transport_failure(), trace])
    monkeypatch.setattr(client, "get_trace", read)
    gateway = MlflowGateway(client, read_retry_policy=READ_POLICY)  # type: ignore[arg-type]

    gateway.verify_trace(created, request)

    assert read.call_args_list == [call(created.trace_id, display=False)] * 2
    assert client.snapshot_state() == state
    assert client.mutations == mutations
    assert client.trace_api.calls == span_calls


def test_scoped_search_retries_current_page_without_duplicate_yields(target: SimpleNamespace) -> None:
    target.trace.info.trace_metadata["mlflow_import.source_namespace"] = "namespace"
    first, second = target.trace, SimpleNamespace(trace_id="trace-2", info=target.trace.info)
    requests: list[dict[str, Any]] = []
    failed_tokens: set[str | None] = set()

    def search(**kwargs: Any) -> _Page:
        requests.append(kwargs)
        token = kwargs["page_token"]
        if token not in failed_tokens:
            failed_tokens.add(token)
            raise _transport_failure()
        return _Page([first], "next") if token is None else _Page([second])

    target.client.search_traces.side_effect = search
    with retry_budget(max_retries=2) as budget:
        qualified_id = json.dumps(["v1", "opencode", "namespace", "session-1"])
        traces = list(target.gateway.iter_session_traces("exp-1", "session-1", qualified_id))
        assert budget.retries_left == 0

    assert [trace.trace_id for trace in traces] == ["trace-1", "trace-2"]
    assert requests[0] == requests[1]
    assert requests[2] == requests[3]
    assert [request["page_token"] for request in requests[:4]] == [None, None, "next", "next"]
    assert all(request["locations"] == ["exp-1"] and request["include_spans"] is False for request in requests)
    assert {item[0] for item in target.client.mock_calls} == {"search_traces"}


def test_retry_budget_is_shared_between_reference_search_and_experiment_reads(target: SimpleNamespace) -> None:
    client, gateway = target.client, target.gateway
    client.get_trace.side_effect = [_transport_failure(), target.trace]
    client.get_trace_info.side_effect = [_transport_failure(), target.trace.info, target.trace.info]
    client.get_experiment.side_effect = _transport_failure()

    with retry_budget(max_retries=2) as budget:
        gateway.replacement_reference("trace-1")
        assert gateway.trace_info("trace-1", "exp-1") is target.trace.info
        with pytest.raises(ItemImportError) as error:
            gateway.experiment_repository("exp-1")
        assert error.value.code == "retry_budget_exhausted"
        assert budget.retries_left == 0

    assert client.get_trace.call_count == 2
    assert client.get_trace_info.call_count == 3
    client.search_traces.assert_not_called()
    assert client.get_experiment.call_count == 1
    target.factory.assert_not_called()


def test_exhausted_read_does_not_multiply_outer_retry_allowance(target: SimpleNamespace) -> None:
    failure = _transport_failure()
    target.client.get_trace.side_effect = failure
    with retry_budget(max_retries=20) as budget:
        with pytest.raises(ItemImportError) as error:
            retry_call(lambda: target.gateway.replacement_reference("trace-1"), READ_POLICY)
        assert budget.retries_left == 18
    assert error.value.code == "read_retry_exhausted"
    assert error.value.__cause__ is failure
    assert "synthetic" not in error.value.safe_message
    assert target.client.mock_calls == [call.get_trace_info("trace-1")] + [call.get_trace("trace-1", display=False)] * 3


@pytest.mark.parametrize("code", ["PERMISSION_DENIED", "RESOURCE_DOES_NOT_EXIST", "INTERNAL_ERROR"])
def test_failed_owned_search_never_restarts_with_an_unfiltered_read(target: SimpleNamespace, code: str) -> None:
    failure = MlflowException("synthetic search failure", error_code=ErrorCode.Value(code))
    target.client.search_traces.side_effect = failure
    expected_attempts = 3 if code == "INTERNAL_ERROR" else 1
    with pytest.raises(ItemImportError if code == "INTERNAL_ERROR" else MlflowException):
        list(target.gateway.iter_owned_traces("exp-1"))
    calls = target.client.search_traces.call_args_list
    assert len(calls) == expected_attempts
    assert all(item == calls[0] for item in calls)
    assert calls[0].kwargs["filter_string"] is not None


def test_candidate_payload_retrieval_retries_before_semantic_validation(target: SimpleNamespace) -> None:
    target.trace.info.tags["opencode.artifact_path"] = "legacy/map.json"
    target.client.get_trace.side_effect = [_transport_failure(), target.trace]
    with pytest.raises(ItemImportError) as error:
        target.gateway.verify_candidate_payload(SimpleNamespace(trace_id="trace-1"), trace_spec())
    assert error.value.code == "legacy_payload_requires_replacement"
    assert target.client.mock_calls == [call.get_trace_info("trace-1")] + [call.get_trace("trace-1", display=False)] * 2
    target.factory.assert_not_called()


def test_unverifiable_retirement_remains_fail_closed(target: SimpleNamespace) -> None:
    target.client.get_trace_info.side_effect = _transport_failure()
    with pytest.raises(ItemImportError) as error:
        target.gateway.verify_owned_trace_for_retirement("exp-1", "trace-1", session_id="session-1", turn_key="turn-1")
    assert error.value.code == "ownership_conflict"
    assert isinstance(error.value.__cause__, ItemImportError)
    assert error.value.__cause__.code == "read_retry_exhausted"
    assert target.client.mock_calls == [call.get_trace_info("trace-1")] * 3


@pytest.mark.parametrize("retries", [0, 8])
def test_artifact_read_exhaustion_is_not_a_candidate_nonmatch(retries: int, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeMlflowClient()
    gateway = client.gateway()
    gateway.probe(CapabilityRequest())
    digest = hashlib.sha256(b"{}").hexdigest()
    request = trace_spec(
        root_tags={
            "opencode.artifact_path": "map.json",
            "opencode.artifact_sha256": digest,
            "opencode.artifact_size": "2",
        }
    )
    created = gateway.create_trace(request)
    candidate = OwnedTraceCandidate.from_owned_trace(next(gateway.iter_owned_traces("exp-1")), "exp-1")
    contract = CandidateContract.from_create_request(request, digest, 2)
    repository = client.artifacts.repositories[client.get_trace(created.trace_id).info.tags["mlflow.artifactLocation"]]
    # The standalone sink also accepts the same optional read policy.
    gateway._sink = TraceArtifactSink(client, client.artifact_repository_factory, READ_POLICY)  # type: ignore[arg-type]
    download = Mock(side_effect=_transport_failure())
    monkeypatch.setattr(repository, "download_artifacts", download)
    before = client.snapshot_state(), list(client.mutations), list(client.trace_api.calls)

    with retry_budget(max_retries=retries):
        with pytest.raises(ItemImportError) as error:
            gateway.verify_owned_candidate(candidate, contract)

    assert error.value.code == ("retry_budget_exhausted" if retries == 0 else "read_retry_exhausted")
    assert download.call_count == (1 if retries == 0 else 3)
    assert (client.snapshot_state(), client.mutations, client.trace_api.calls) == before


@pytest.mark.parametrize("failure", ["disconnect", "truncated", "checksum", "none"])
def test_bounded_http_candidate_read_preserves_transport_vs_integrity_failure(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeMlflowClient()
    gateway = client.gateway()
    gateway.probe(CapabilityRequest())
    digest = hashlib.sha256(b"{}").hexdigest()
    request = trace_spec(
        root_tags={
            "opencode.artifact_path": "map.json",
            "opencode.artifact_sha256": digest,
            "opencode.artifact_size": "2",
        }
    )
    created = gateway.create_trace(request)
    candidate = OwnedTraceCandidate.from_owned_trace(next(gateway.iter_owned_traces("exp-1")), "exp-1")
    contract = CandidateContract.from_create_request(request, digest, 2)
    uri = "https://synthetic.invalid" + resources.ARTIFACT_PROXY_PATH + "/trace-artifacts"
    trace = client.get_trace(created.trace_id)
    trace.info.tags["mlflow.artifactLocation"] = uri
    repository = client.artifacts.register(uri)
    monkeypatch.setattr(repository, "artifact_uri", uri, raising=False)
    sdk_download = Mock(side_effect=AssertionError("bounded HTTP must not use the SDK download"))
    monkeypatch.setattr(repository, "download_artifacts", sdk_download)
    body = b"{" if failure == "truncated" else b"[]" if failure == "checksum" else b"{}"
    transient = failure in {"disconnect", "truncated"}
    responses = [_ArtifactResponse(body, disconnect=failure == "disconnect") for _ in range(4 if transient else 1)]
    opener = Mock()
    opener.open.side_effect = responses
    monkeypatch.setattr(resources, "artifact_http_configuration", lambda: (opener, {}))
    monkeypatch.setattr(
        "mlflow_import.retry.retry_call", lambda operation, policy: retry_call(operation, policy, sleep=lambda _: None)
    )
    before = client.snapshot_state(), list(client.mutations), list(client.trace_api.calls)

    with retry_budget(max_retries=8) as budget:
        if transient:
            with pytest.raises(ItemImportError) as error:
                # Outer verification retries must not replay exhausted HTTP reads.
                retry_call(lambda: gateway.verify_owned_candidate(candidate, contract), READ_POLICY)
            assert error.value.code == "artifact_transport_failed"
            assert "synthetic" not in error.value.safe_message
        else:
            verification = gateway.verify_owned_candidate(candidate, contract)
            assert verification.matches is (failure == "none")
            if failure == "checksum":
                assert verification.reason == "artifact_mismatch"
        assert budget.retries_left == (5 if transient else 8)

    assert opener.open.call_count == len(responses)
    requests = [item.args[0] for item in opener.open.call_args_list]
    assert all(request.get_method() == "GET" and request.data is None for request in requests)
    assert {request.full_url for request in requests} == {uri + "/map.json"}
    assert all(response.closed for response in responses)
    sdk_download.assert_not_called()
    assert (client.snapshot_state(), client.mutations, client.trace_api.calls) == before


@pytest.mark.parametrize(
    "changed_key, changed_value",
    [("mlflow_import.attempt_state", "complete"), ("mlflow_import.attempt_id", "b" * 32)],
)
@pytest.mark.parametrize("changed_location", ["tags", "metadata"])
def test_retirement_rechecks_both_physical_staging_locations_after_flush(
    changed_key: str, changed_value: str, changed_location: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeMlflowClient()
    gateway = client.gateway()
    gateway.probe(CapabilityRequest(needs_retirement=True))
    scope = {
        "mlflow_import.attempt_id": "a" * 32,
        "mlflow_import.attempt_state": "staged",
        "mlflow_import.harness": "opencode",
        "mlflow_import.source_namespace": "namespace",
        "mlflow_import.session_id": "ses-sample",
        "mlflow_import.qualified_session_id": json.dumps(["v1", "opencode", "namespace", "ses-sample"]),
        "mlflow_import.import_instance_id": "",  # An absent optional instance remains compatible.
    }
    physical = {key: value for key, value in scope.items() if key != "mlflow_import.import_instance_id"}
    request = trace_spec(root_tags=physical)
    request = replace(request, metadata={**request.metadata, **physical})
    created = gateway.create_trace(request)
    trace = client.get_trace(created.trace_id)
    initial_reads = gateway.verify_owned_trace_for_retirement(
        "exp-1",
        created.trace_id,
        session_id="ses-sample",
        turn_key=request.metadata["opencode.turn_key"],
        expected_metadata=scope,
    )
    assert initial_reads["mlflow_import.attempt_state"] == "staged"
    before_mutations = list(client.mutations)
    deleted = Mock(side_effect=AssertionError("changed staging evidence must prevent deletion"))
    monkeypatch.setattr(client, "delete_traces", deleted)

    def flush() -> None:
        values = trace.info.tags if changed_location == "tags" else trace.info.trace_metadata
        values[changed_key] = changed_value

    flushed = Mock(side_effect=flush)
    monkeypatch.setattr(mlflow, "flush_trace_async_logging", flushed)
    with pytest.raises(ItemImportError) as error:
        gateway.retire_trace(
            "exp-1",
            created.trace_id,
            session_id="ses-sample",
            turn_key=request.metadata["opencode.turn_key"],
            expected_metadata=scope,
        )
    assert error.value.code == "ownership_conflict"
    flushed.assert_called_once_with()
    deleted.assert_not_called()
    assert client.mutations == before_mutations
    assert client.get_trace(created.trace_id) is trace
    assert trace.info.tags[changed_key] != trace.info.trace_metadata[changed_key]


@pytest.mark.parametrize("location", ["tags", "metadata", "both"])
def test_retirement_expected_staging_supports_single_location_and_logical_aliases(location: str) -> None:
    client = FakeMlflowClient()
    gateway = client.gateway()
    gateway.probe(CapabilityRequest())
    staging = {"mlflow_import.attempt_id": "a" * 32, "mlflow_import.attempt_state": "staged"}
    request = trace_spec(root_tags=staging if location in {"tags", "both"} else {})
    request = replace(
        request,
        metadata={
            **request.metadata,
            "opencode.importer": "opencode-mlflow",  # The logical compatibility view normalizes this marker.
            **(staging if location in {"metadata", "both"} else {}),
        },
    )
    created = gateway.create_trace(request)
    expected = {**staging, "opencode.importer": "mlflow-import", "mlflow_import.import_instance_id": ""}
    verified = gateway.verify_owned_trace_for_retirement(
        "exp-1",
        created.trace_id,
        session_id="ses-sample",
        turn_key=request.metadata["opencode.turn_key"],
        expected_metadata=expected,
    )
    assert all(verified[key] == value for key, value in expected.items())


def test_deadline_exhaustion_prevents_any_read(target: SimpleNamespace) -> None:
    with retry_budget(deadline_seconds=0):
        with pytest.raises(ItemImportError) as error:
            target.gateway.trace_exists("trace-1")
    assert error.value.code == "retry_budget_exhausted"
    assert target.client.mock_calls == []


@pytest.mark.parametrize("code", ["PERMISSION_DENIED", "RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"])
@pytest.mark.parametrize("boundary", ["reference", "search", "experiment", "repository", "download", "listing"])
def test_permanent_errors_are_not_retried(target: SimpleNamespace, code: str, boundary: str) -> None:
    gateway, client = target.gateway, target.client
    repository = gateway.experiment_repository("exp-1")
    client.reset_mock()
    failure = MlflowException("synthetic permanent error", error_code=ErrorCode.Value(code))
    # Even a transport context cannot turn an explicit permanent protocol code transient.
    failure.__context__ = _transport_failure()
    operations: dict[str, tuple[Mock, Callable[[], Any]]] = {
        "reference": (client.get_trace, lambda: gateway.replacement_reference("trace-1")),
        "search": (client.search_traces, lambda: gateway._search_page("exp-1", "filter", None)),
        "experiment": (client.get_experiment_by_name, lambda: gateway.resolve_experiment("experiment", False)),
        "repository": (client.get_experiment, lambda: gateway.experiment_repository("exp-1")),
        "download": (
            target.repository.download_artifacts,
            lambda: resources.copy_artifact(repository, "map.json", "/synthetic", 100),
        ),
        "listing": (target.repository.list_artifacts, lambda: list(resources.iter_artifacts(repository, "prefix"))),
    }
    failing, operation = operations[boundary]
    failing.side_effect = failure
    with pytest.raises(MlflowException) as error:
        operation()
    assert error.value is failure
    assert failing.call_count == 1
    assert not client.create_experiment.called


def test_only_not_found_proves_absence(target: SimpleNamespace) -> None:
    client, gateway = target.client, target.gateway
    client.get_trace_info.side_effect = MlflowException(
        "synthetic missing", error_code=ErrorCode.Value("RESOURCE_DOES_NOT_EXIST")
    )
    assert gateway.trace_exists("trace-1") is False
    assert client.get_trace_info.call_count == 1
    client.get_trace_info.reset_mock(side_effect=True)
    client.get_trace_info.side_effect = _transport_failure()
    with pytest.raises(ItemImportError) as error:
        gateway.trace_exists("trace-1")
    assert error.value.code == "read_retry_exhausted"
    assert client.get_trace_info.call_count == 3
    client.get_trace.assert_not_called()


def test_semantic_ownership_conflict_is_not_retried(target: SimpleNamespace) -> None:
    target.trace.info.trace_metadata["opencode.importer"] = "foreign"
    with pytest.raises(ConfigurationError) as error:
        target.gateway.replacement_reference("trace-1")
    assert error.value.code == "ownership_conflict"
    assert target.client.mock_calls == [call.get_trace_info("trace-1"), call.get_trace("trace-1", display=False)]


def test_repository_lookup_factory_and_sdk_reads_share_budget(target: SimpleNamespace, tmp_path: Path) -> None:
    client, gateway, repository = target.client, target.gateway, target.repository
    experiment = client.get_experiment.return_value
    client.get_experiment.side_effect = [_transport_failure(), experiment]
    target.factory.side_effect = [_transport_failure(), repository]
    repository.list_artifacts.side_effect = [_transport_failure(), ["map.json"]]
    download = repository.download_artifacts.side_effect
    repository.download_artifacts.side_effect = [_transport_failure(), download("map.json", str(tmp_path))]

    with retry_budget(max_retries=4) as budget:
        bound = gateway.experiment_repository("exp-1")
        assert list(resources.iter_artifacts(bound, "prefix")) == ["map.json"]
        downloaded = resources.copy_artifact(bound, "map.json", str(tmp_path), len(target.payload))
        assert downloaded.read_bytes() == target.payload
        assert budget.retries_left == 0

    assert client.mock_calls == [call.get_experiment("exp-1")] * 2
    assert target.factory.call_args_list == [call(experiment.artifact_location, tracking_uri=client.tracking_uri)] * 2
    assert repository.list_artifacts.call_args_list == [call("prefix")] * 2
    assert repository.download_artifacts.call_args_list == [call("map.json", str(tmp_path))] * 2
    repository.log_artifact.assert_not_called()


def test_experiment_name_lookup_retries_without_creation(target: SimpleNamespace) -> None:
    experiment = target.client.get_experiment_by_name.return_value
    target.client.get_experiment_by_name.side_effect = [_transport_failure(), experiment]
    assert target.gateway.resolve_experiment("experiment", create=True).experiment_id == "exp-1"
    assert target.client.mock_calls == [call.get_experiment_by_name("experiment")] * 2


def test_legacy_artifact_sink_retries_trace_and_download_reads(target: SimpleNamespace) -> None:
    target.client.get_trace_info.side_effect = [_transport_failure(), target.trace.info]
    download = target.repository.download_artifacts.side_effect
    attempts = 0

    def flaky_download(path: str, root: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _transport_failure()
        return str(download(path, root))

    target.repository.download_artifacts.side_effect = flaky_download
    sink = TraceArtifactSink(target.client, target.factory, READ_POLICY)
    sink.verify_named_json("trace-1", "map.json", hashlib.sha256(target.payload).hexdigest(), len(target.payload))
    assert target.client.mock_calls == [call.get_trace_info("trace-1")] * 2
    assert attempts == 2
    calls = target.repository.download_artifacts.call_args_list
    assert calls[0] == calls[1]
    target.repository.log_artifact.assert_not_called()


def test_repository_mutations_are_delegated_once(target: SimpleNamespace) -> None:
    repository = target.gateway.experiment_repository("exp-1")
    failure = _transport_failure()
    target.repository.log_artifact.side_effect = failure
    with pytest.raises(MlflowException) as error:
        repository.log_artifact("/synthetic/local", artifact_path="prefix")
    assert error.value is failure
    target.repository.log_artifact.assert_called_once_with("/synthetic/local", artifact_path="prefix")
