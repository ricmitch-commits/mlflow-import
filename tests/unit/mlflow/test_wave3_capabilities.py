"""Wave 3 capability-attestation regression tests."""

from __future__ import annotations

import json

import mlflow
import pytest

import opencode_mlflow.mlflow.capabilities as capabilities_module
from opencode_mlflow.mlflow.capabilities import (
    MAX_VERSION_RESPONSE_BYTES,
    PINNED_MLFLOW_VERSION,
    CapabilityRequest,
    attest_server_version,
    probe_capabilities,
    version_endpoint,
)
from tests.support.fake_gateway import FakeMlflowClient

pytestmark = pytest.mark.unit


def _request() -> CapabilityRequest:
    return CapabilityRequest(needs_retirement=True, needs_named_artifacts=True)


def _version_transport(version: object):
    calls: list[str] = []

    def transport(tracking_uri: str) -> object:
        calls.append(tracking_uri)
        return {"version": version}

    return transport, calls


def test_profile_probe_requires_an_explicit_exact_server_attestation() -> None:
    client = FakeMlflowClient()
    transport, calls = _version_transport(PINNED_MLFLOW_VERSION)

    report = probe_capabilities(client, _request(), dry_run=True, version_transport=transport)

    assert report.is_compatible, report.safe_error
    assert calls == [client.tracking_uri]


def test_profile_probe_accepts_the_documented_plain_text_version_response() -> None:
    client = FakeMlflowClient()

    report = probe_capabilities(
        client,
        _request(),
        dry_run=True,
        version_transport=lambda _uri: "3.17.2\n",
    )

    assert report.is_compatible, report.safe_error


def test_server_version_mismatch_fails_closed_without_mutation() -> None:
    client = FakeMlflowClient()
    before = client.snapshot_state()
    transport, _calls = _version_transport("3.17.1")

    report = probe_capabilities(client, _request(), dry_run=False, version_transport=transport)

    assert not report.is_compatible
    assert report.safe_error is not None
    assert "server version" in report.safe_error
    assert report.max_inline_payload_bytes == 0
    assert not report.named_artifact_supported
    assert not report.retirement_supported
    assert client.mutations == []
    assert client.snapshot_state() == before


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"version": None},
        {"version": [PINNED_MLFLOW_VERSION]},
    ],
)
def test_malformed_server_version_response_fails_closed(response: object) -> None:
    client = FakeMlflowClient()

    report = probe_capabilities(client, _request(), dry_run=True, version_transport=lambda _uri: response)

    assert not report.is_compatible
    assert report.safe_error is not None
    assert "version attestation" in report.safe_error
    assert client.mutations == []


def test_missing_server_version_endpoint_does_not_leak_transport_detail() -> None:
    client = FakeMlflowClient()

    def missing_endpoint(_uri: str) -> object:
        raise RuntimeError("Authorization Bearer do-not-leak-this-token")

    report = probe_capabilities(client, _request(), dry_run=True, version_transport=missing_endpoint)

    assert not report.is_compatible
    assert report.safe_error is not None
    assert "version attestation" in report.safe_error
    assert "do-not-leak-this-token" not in report.safe_error
    assert "Authorization" not in report.safe_error
    assert client.mutations == []


def test_production_probe_rejects_unsupported_tracking_uri_scheme() -> None:
    client = FakeMlflowClient()

    report = probe_capabilities(client, _request(), dry_run=True)

    assert not report.is_compatible
    assert report.safe_error is not None
    assert "version attestation" in report.safe_error
    assert client.mutations == []


def test_client_version_mismatch_cannot_enable_profile_backed_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeMlflowClient()
    transport, _calls = _version_transport(PINNED_MLFLOW_VERSION)
    monkeypatch.setattr(mlflow, "__version__", "9.9.9")

    report = probe_capabilities(client, _request(), dry_run=True, version_transport=transport)

    assert not report.is_compatible
    assert report.max_inline_payload_bytes == 0
    assert not report.named_artifact_supported
    assert not report.retirement_supported


def test_version_endpoint_strips_query_and_fragment_credentials() -> None:
    endpoint = version_endpoint("https://user:password@example.test/static?token=sentinel#fragment")

    assert endpoint == "https://example.test/static/version"
    assert "sentinel" not in endpoint
    assert "fragment" not in endpoint
    assert "password" not in endpoint


def test_http_version_attestation_does_not_forward_tracking_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        status = 200

        def getcode(self) -> int:
            raise AssertionError("getcode must not be evaluated when status exists")

        def read(self, limit: int = -1) -> bytes:
            captured["read_limit"] = limit
            return b"3.17.2"

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Opener:
        def open(self, request: object, *, timeout: float) -> Response:
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

    monkeypatch.setattr(capabilities_module.urllib.request, "build_opener", lambda *_handlers: Opener())

    assert attest_server_version("https://user:password@example.test/static?token=sentinel#fragment") == "3.17.2"

    request = captured["request"]
    assert request.full_url == "https://example.test/static/version"  # type: ignore[union-attr]
    assert request.data is None  # type: ignore[union-attr]
    serialized = repr(request) + repr(request.header_items())  # type: ignore[union-attr]
    assert "sentinel" not in serialized
    assert captured["read_limit"] == MAX_VERSION_RESPONSE_BYTES + 1


def test_http_version_attestation_does_not_forward_tracking_uri_userinfo_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        status = 200

        def read(self, _limit: int = -1) -> bytes:
            return b"3.17.2"

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Opener:
        def open(self, request: object, *, timeout: float) -> Response:
            captured["request"] = request
            return Response()

    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    monkeypatch.setattr(capabilities_module.urllib.request, "build_opener", lambda *_handlers: Opener())

    assert attest_server_version("https://uri-user:uri-password@example.test") == "3.17.2"

    request = captured["request"]
    assert request.get_header("Authorization") is None  # type: ignore[union-attr]
    assert "uri-user" not in repr(request)  # type: ignore[arg-type]
    assert "uri-password" not in repr(request)  # type: ignore[arg-type]


def test_injected_version_response_size_is_bounded() -> None:
    response = {
        "version": PINNED_MLFLOW_VERSION,
        "padding": "x" * MAX_VERSION_RESPONSE_BYTES,
    }
    assert len(json.dumps(response).encode("utf-8")) > MAX_VERSION_RESPONSE_BYTES

    with pytest.raises(Exception, match="malformed response"):
        attest_server_version("https://example.test", transport=lambda _uri: response)


def test_http_version_attestation_rejects_non_success_status_without_reading_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status = 503

        def read(self, _limit: int = -1) -> bytes:
            raise AssertionError("a non-success response body must not be trusted")

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Opener:
        def open(self, _request: object, *, timeout: float) -> Response:
            assert timeout > 0
            return Response()

    monkeypatch.setattr(capabilities_module.urllib.request, "build_opener", lambda *_handlers: Opener())

    with pytest.raises(Exception, match="unexpected response"):
        attest_server_version("https://example.test")


def test_http_version_attestation_uses_getcode_when_status_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status = None

        def getcode(self) -> int:
            return 200

        def read(self, limit: int = -1) -> bytes:
            assert limit == MAX_VERSION_RESPONSE_BYTES + 1
            return b"3.17.2"

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Opener:
        def open(self, _request: object, *, timeout: float) -> Response:
            assert timeout > 0
            return Response()

    monkeypatch.setattr(capabilities_module.urllib.request, "build_opener", lambda *_handlers: Opener())

    assert attest_server_version("https://example.test") == PINNED_MLFLOW_VERSION


def test_http_version_attestation_rejects_oversized_body_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        status = 200

        def read(self, limit: int = -1) -> bytes:
            assert limit == MAX_VERSION_RESPONSE_BYTES + 1
            return b"x" * (MAX_VERSION_RESPONSE_BYTES + 1)

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Opener:
        def open(self, _request: object, *, timeout: float) -> Response:
            assert timeout > 0
            return Response()

    monkeypatch.setattr(capabilities_module.urllib.request, "build_opener", lambda *_handlers: Opener())

    with pytest.raises(Exception, match="malformed response"):
        attest_server_version("https://example.test")
