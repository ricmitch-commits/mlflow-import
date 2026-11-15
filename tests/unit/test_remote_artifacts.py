"""Finite wire envelopes and standard configuration for the pinned artifact proxy."""

import base64
import io
import json
import ssl
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlflow_import import resources
from mlflow_import.errors import ConfigurationError, ItemImportError
from mlflow_import.mlflow.gateway import MlflowGateway
from mlflow_import.retry import retry_budget, retry_call

pytestmark = pytest.mark.unit


class Response(io.BytesIO):
    status = 200

    def __init__(self, body, headers=None):
        super().__init__(body)
        self.headers = headers or {}
        self.read_sizes = []

    def read1(self, size):
        self.read_sizes.append(size)
        return super().read1(size)


@pytest.fixture
def wire(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in (
        "MLFLOW_TRACKING_AUTH",
        "MLFLOW_TRACKING_AWS_SIGV4",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_TRACKING_TOKEN",
        "MLFLOW_TRACKING_INSECURE_TLS",
        "MLFLOW_TRACKING_SERVER_CERT_PATH",
        "MLFLOW_TRACKING_CLIENT_CERT_PATH",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        monkeypatch.delenv(key, raising=False)
    opener = Mock()
    monkeypatch.setattr(resources.urllib.request, "build_opener", lambda *args: opener)
    return opener


def repository(path="experiment"):
    return SimpleNamespace(artifact_uri=f"https://example.test/prefix{resources.ARTIFACT_PROXY_PATH}/{path}")


def test_streamed_download_quota_and_unknown_length(wire):
    body = b"x" * (resources.COPY_BATCH * 2 + 17)
    response = Response(body)
    wire.open.return_value = response
    with resources.temporary_storage(1024 * 1024) as budget:
        with resources.temporary_directory(prefix="download-") as root:
            path = resources.copy_artifact(repository(), "objects/body.json", root, len(body))
            assert path.read_bytes() == body
            assert budget.peak_bytes <= budget.max_bytes
            assert max(response.read_sizes) <= resources.COPY_BATCH
        assert len(budget.allocations) == 1
    assert response.closed
    request = wire.open.call_args.args[0]
    assert request.full_url.endswith("/experiment/objects/body.json")


@pytest.mark.parametrize("declared", [True, False])
def test_size_limit_rejects_before_overwrite_and_cleans(wire, declared):
    response = Response(b"x" * 100, {"Content-Length": "100"} if declared else {})
    wire.open.return_value = response
    with resources.temporary_storage(1024 * 1024) as budget:
        with resources.temporary_directory(prefix="download-") as root:
            with pytest.raises(ItemImportError, match="envelope"):
                resources.copy_artifact(repository(), "object", root, 90)
            assert not list(Path(root).iterdir())
        assert len(budget.allocations) == 1
    assert response.closed


@pytest.mark.parametrize("known_size", [True, False])
def test_download_shared_quota_failure_never_writes_extra_chunk(wire, known_size):
    size = resources.COPY_BATCH * 3
    response = Response(b"x" * size, {"Content-Length": str(size)} if known_size else {})
    wire.open.return_value = response
    with resources.temporary_storage(64 * 1024) as budget:
        with resources.temporary_directory(prefix="download-") as root:
            with pytest.raises(resources.TemporaryStorageError):
                resources.copy_artifact(repository(), "object", root, 1024 * 1024)
            assert budget.peak_bytes <= budget.max_bytes
            assert not list(Path(root).iterdir())
    assert response.closed
    if known_size:
        assert response.read_sizes == []


def test_listing_pagination_is_relative_and_existence_uses_parent(wire):
    wire.open.side_effect = [
        Response(
            json.dumps(
                {"files": [{"path": "one", "is_dir": False, "file_size": "3"}], "next_page_token": "opaque"}
            ).encode()
        ),
        Response(b'{"files":[{"path":"two","is_dir":true}]}'),
    ]
    assert list(resources.iter_artifacts(repository(), "attempts")) == [
        resources.ArtifactEntry("attempts/one", False, 3),
        resources.ArtifactEntry("attempts/two", True, 0),
    ]
    assert "path=experiment%2Fattempts&page_token=opaque" in wire.open.call_args.args[0].full_url
    wire.open.side_effect = None
    wire.open.return_value = Response(b'{"files":[{"path":"one","is_dir":false,"file_size":3}]}')
    assert resources.artifact_exists(repository(), "attempts/one")
    assert "path=experiment%2Fattempts" in wire.open.call_args.args[0].full_url


@pytest.mark.parametrize("case", ["bytes", "entries", "token", "pages", "traversal", "malformed"])
def test_listing_exhaustion_and_untrusted_paths(wire, monkeypatch, case):
    monkeypatch.setattr(resources, "MAX_ARTIFACT_LIST_PAGES", 2)
    monkeypatch.setattr(resources, "MAX_ARTIFACT_LIST_ENTRIES", 2)
    monkeypatch.setattr(resources, "MAX_ARTIFACT_LIST_BYTES", 256)
    payload = {"files": []}
    if case == "bytes":
        responses = [Response(b" " * 257)]
    elif case == "entries":
        payload["files"] = [{"path": str(i), "is_dir": True} for i in range(3)]
        responses = [Response(json.dumps(payload).encode())]
    elif case in {"token", "pages"}:
        responses = [
            Response(json.dumps({"next_page_token": t}).encode())
            for t in ("first", "first" if case == "token" else "second")
        ]
    elif case == "traversal":
        responses = [Response(b'{"files":[{"path":"%2e%2e","is_dir":false}]}')]
    else:
        responses = [Response(b'{"files":[null]}')]
    wire.open.side_effect = responses
    with pytest.raises(ItemImportError):
        list(resources.iter_artifacts(repository(), "attempts"))
    assert all(response.closed for response in responses)


def test_deadline_and_operation_allowance_checked_per_chunk(wire):
    response = Response(b"x" * resources.COPY_BATCH * 3)
    wire.open.return_value = response
    with resources.temporary_storage(1024 * 1024):
        with resources.temporary_directory(prefix="download-") as root:
            with retry_budget(max_operations=3):
                with pytest.raises(ItemImportError, match="allowance"):
                    resources.copy_artifact(repository(), "object", root, 1024 * 1024)
            with retry_budget() as budget:
                budget.deadline = time.monotonic() - 1
                with pytest.raises(ItemImportError, match="deadline"):
                    resources.copy_artifact(repository(), "object", root, 1024 * 1024)


@pytest.mark.parametrize("status, calls", [(401, 1), (403, 1), (404, 1), (302, 1), (503, 4), (429, 4)])
def test_finite_retries_and_safe_errors(wire, monkeypatch, status, calls):
    monkeypatch.setattr(
        "mlflow_import.retry.retry_call", lambda op, policy: retry_call(op, policy, sleep=lambda _: None)
    )
    wire.open.side_effect = lambda *a, **kw: (_ for _ in ()).throw(
        urllib.error.HTTPError("https://secret:password@example.test", status, "secret", {}, None)
    )
    if calls == 4:
        with retry_budget(max_retries=0):
            with pytest.raises(ItemImportError, match="exhausted"):
                list(resources.iter_artifacts(repository(), ""))
        assert wire.open.call_count == 1
    wire.open.reset_mock()
    with pytest.raises(ItemImportError) as caught:
        list(resources.iter_artifacts(repository(), ""))
    assert "secret" not in str(caught.value)
    assert wire.open.call_count == calls


def test_standard_basic_token_and_credentials_file(wire, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "token")
    _, headers = resources.artifact_http_configuration()
    assert headers["Authorization"] == "Bearer token"
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass")
    _, headers = resources.artifact_http_configuration()
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"user:pass").decode()
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME")
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD")
    (tmp_path / ".mlflow").mkdir()
    (tmp_path / ".mlflow" / "credentials").write_text(
        "[mlflow]\nmlflow_tracking_username=file-user\nmlflow_tracking_password=file-pass\n"
    )
    _, headers = resources.artifact_http_configuration()
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"file-user:file-pass").decode()


def test_tls_configuration_and_no_redirect(wire, monkeypatch):
    context = Mock()
    create_context = Mock(return_value=context)
    monkeypatch.setattr(resources.ssl, "create_default_context", create_context)
    monkeypatch.setenv("MLFLOW_TRACKING_SERVER_CERT_PATH", "/synthetic/ca.pem")
    monkeypatch.setenv("MLFLOW_TRACKING_CLIENT_CERT_PATH", "/synthetic/client.pem")
    resources.artifact_http_configuration()
    create_context.assert_called_once_with(cafile="/synthetic/ca.pem")
    context.load_verify_locations.assert_not_called()
    context.load_cert_chain.assert_called_once_with(certfile="/synthetic/client.pem")
    monkeypatch.setenv("MLFLOW_TRACKING_INSECURE_TLS", "true")
    with pytest.raises(ConfigurationError):
        resources.artifact_http_configuration()
    monkeypatch.delenv("MLFLOW_TRACKING_SERVER_CERT_PATH")
    resources.artifact_http_configuration()
    assert context.verify_mode == ssl.CERT_NONE and context.check_hostname is False
    assert resources._ArtifactNoRedirect().redirect_request(None, None, None, None, None, None) is None


@pytest.mark.parametrize("location", ["s3://bucket/root", "https://example.test/not-a-proxy"])
@pytest.mark.parametrize("missing", [True, False])
def test_profile_validation_before_any_experiment_write(wire, location, missing):
    experiment = SimpleNamespace(artifact_location=location, experiment_id="0", lifecycle_stage="active")
    client = Mock(tracking_uri="https://example.test")
    client.get_experiment_by_name.return_value = None if missing else experiment
    client.get_experiment.return_value = experiment
    gateway = MlflowGateway(client, artifact_repository_factory=lambda uri, **kw: SimpleNamespace(artifact_uri=uri))
    with resources.temporary_storage(1024 * 1024):
        with pytest.raises(ConfigurationError):
            gateway.resolve_experiment("synthetic", create=True)
    client.create_experiment.assert_not_called()


@pytest.mark.parametrize("matches", [True, False])
def test_ambiguous_upload_only_succeeds_after_bounded_exact_readback(wire, tmp_path, monkeypatch, matches):
    payload = b"synthetic artifact"
    source = tmp_path / "source.json"
    source.write_bytes(payload)
    remote = repository()
    remote.log_artifact = Mock(side_effect=TimeoutError("secret transport detail"))
    remote.download_artifacts = Mock(side_effect=AssertionError("SDK whole-file download forbidden"))
    client = Mock(tracking_uri="https://example.test")
    client.get_experiment.return_value = SimpleNamespace(artifact_location=remote.artifact_uri)
    wire.open.side_effect = [Response(b"{}"), Response(payload if matches else b"corrupted")]
    with resources.temporary_storage(1024 * 1024):
        gateway = MlflowGateway(client, artifact_repository_factory=lambda *args, **kw: remote)
        monkeypatch.setattr(gateway, "_require_mutation_allowed", lambda: None)
        if matches:
            result = gateway.publish_experiment_file("1", "test/body.json", source)
            assert result.size == len(payload)
        else:
            with pytest.raises(ItemImportError, match="checksum"):
                gateway.publish_experiment_file("1", "test/body.json", source)
        remote.log_artifact.assert_called_once()
        remote.download_artifacts.assert_not_called()


def test_partial_download_retry_restarts_same_path_and_releases_reservation(wire, monkeypatch):
    monkeypatch.setattr(
        "mlflow_import.retry.retry_call", lambda op, policy: retry_call(op, policy, sleep=lambda _: None)
    )
    wire.open.side_effect = [Response(b"partial", {"Content-Length": "14"}), Response(b"complete bytes")]
    with resources.temporary_storage(1024 * 1024) as budget:
        with resources.temporary_directory(prefix="retry-") as root:
            path = resources.copy_artifact(repository(), "object", root, 14)
            assert path.read_bytes() == b"complete bytes"
            assert wire.open.call_count == 2
        assert len(budget.allocations) == 1


@pytest.mark.parametrize("location", ["file:///synthetic/artifacts/0", repository("0").artifact_uri])
def test_new_experiment_pins_qualified_location_instead_of_unknown_server_default(wire, monkeypatch, location):
    wire.open.side_effect = [Response(b"{}"), Response(b"{}")]
    client = Mock(tracking_uri="https://example.test")
    client.get_experiment_by_name.return_value = None
    client.get_experiment.return_value = SimpleNamespace(artifact_location=location)
    client.create_experiment.return_value = "new-id"
    with resources.temporary_storage(1024 * 1024):
        gateway = MlflowGateway(client, artifact_repository_factory=lambda uri, **kw: SimpleNamespace(artifact_uri=uri))
        monkeypatch.setattr(gateway, "_require_mutation_allowed", lambda: None)
        result = gateway.resolve_experiment("synthetic", create=True)
        assert result.created and result.experiment_id == "new-id"
        assert result.artifact_location.startswith(location + "/mlflow-import-experiments/")
        client.create_experiment.assert_called_once_with("synthetic", artifact_location=result.artifact_location)
