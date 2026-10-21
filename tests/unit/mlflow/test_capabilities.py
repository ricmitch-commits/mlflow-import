"""Unit tests for the runtime MLflow capability probe.

The runtime :func:`opencode_mlflow.mlflow.capabilities.probe_capabilities`
is strictly read-only in BOTH modes: it inspects client methods, server
reachability/protocol responses, the recorded compatibility profile, the
artifact composition, and reported payload limits — never creating a probe
experiment/trace/artifact, uploading, or deleting. Mutation-based proof
belongs only to Task 0's disposable-server profile environment; the runtime
probe leans on the pinned profile (version-gated) for the facts that cannot
be re-proven without mutation.

All tests run against the in-memory fake client; no network, no server.
"""

from __future__ import annotations

import mlflow
import pytest

from opencode_mlflow.errors import CapabilityError
from opencode_mlflow.mlflow.capabilities import (
    PINNED_MAX_INLINE_PAYLOAD_BYTES,
    PINNED_MLFLOW_VERSION,
    PROBE_EXPERIMENT_NAME,
    CapabilityRequest,
    require_capabilities,
)
from opencode_mlflow.mlflow.capabilities import probe_capabilities as _production_probe_capabilities
from tests.support.fake_gateway import FakeMlflowClient, FakeMlflowException

pytestmark = pytest.mark.unit


def probe_capabilities(client: FakeMlflowClient, request: CapabilityRequest, *, dry_run: bool):
    """Use the explicit fake attestation seam for legacy unit cases."""
    return _production_probe_capabilities(
        client,
        request,
        dry_run,
        version_transport=lambda _uri: {"version": PINNED_MLFLOW_VERSION},
    )


def _compatible_request() -> CapabilityRequest:
    return CapabilityRequest(needs_retirement=True, needs_named_artifacts=True)


def test_probe_reports_compatible_target_in_dry_run_mode(fake_mlflow_client: FakeMlflowClient) -> None:
    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=True)

    assert report.is_compatible, report.safe_error
    assert report.safe_error is None
    assert report.dry_run is True
    assert report.named_artifact_supported
    assert report.retirement_supported
    assert report.max_inline_payload_bytes == PINNED_MAX_INLINE_PAYLOAD_BYTES
    assert report.needs_retirement
    assert report.needs_named_artifacts


def test_probe_reports_compatible_target_in_normal_mode(fake_mlflow_client: FakeMlflowClient) -> None:
    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=False)

    assert report.is_compatible, report.safe_error
    assert report.dry_run is False


def test_probe_is_read_only_in_dry_run_mode(fake_mlflow_client: FakeMlflowClient) -> None:
    before = fake_mlflow_client.snapshot_state()

    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=True)

    assert report.is_compatible, report.safe_error
    assert fake_mlflow_client.mutations == []
    assert fake_mlflow_client.snapshot_state() == before


def test_probe_is_read_only_in_normal_mode(fake_mlflow_client: FakeMlflowClient) -> None:
    before = fake_mlflow_client.snapshot_state()

    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=False)

    assert report.is_compatible, report.safe_error
    assert fake_mlflow_client.mutations == []
    assert fake_mlflow_client.snapshot_state() == before


def test_probe_never_creates_probe_experiments_or_traces(fake_mlflow_client: FakeMlflowClient) -> None:
    probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=False)
    probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=True)

    called = {call[0] for call in fake_mlflow_client.calls}
    assert called <= {"get_experiment_by_name", "get_trace"}


def test_normal_write_always_requires_retirement(fake_mlflow_client: FakeMlflowClient) -> None:
    report = probe_capabilities(
        fake_mlflow_client,
        CapabilityRequest(needs_retirement=False, needs_named_artifacts=True),
        dry_run=False,
    )

    assert report.is_compatible
    assert report.needs_retirement is True


def test_dry_run_honors_requested_retirement(fake_mlflow_client: FakeMlflowClient) -> None:
    report = probe_capabilities(
        fake_mlflow_client,
        CapabilityRequest(needs_retirement=False, needs_named_artifacts=False),
        dry_run=True,
    )

    assert report.is_compatible
    assert report.needs_retirement is False
    assert report.needs_named_artifacts is False


def test_version_mismatch_fails_the_probe(
    fake_mlflow_client: FakeMlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mlflow, "__version__", "9.9.9")

    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=True)

    assert not report.is_compatible
    assert report.safe_error is not None
    assert "pinned compatibility profile" in report.safe_error
    assert PINNED_MLFLOW_VERSION in report.safe_error
    assert report.max_inline_payload_bytes == 0
    assert not report.named_artifact_supported


def test_missing_client_method_fails_the_probe(fake_mlflow_client: FakeMlflowClient) -> None:
    fake_mlflow_client.delete_traces = None  # type: ignore[assignment]

    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=False)

    assert not report.is_compatible
    assert report.safe_error is not None
    assert "delete_traces" in report.safe_error
    assert not report.retirement_supported


def test_missing_module_span_api_fails_the_probe(
    fake_mlflow_client: FakeMlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mlflow, "start_span_no_context", None, raising=False)

    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=True)

    assert not report.is_compatible
    assert report.safe_error is not None
    assert "start_span_no_context" in report.safe_error


def test_unreachable_server_fails_the_probe(fake_mlflow_client: FakeMlflowClient) -> None:
    def unreachable(name: str) -> object:
        raise RuntimeError("connection refused")

    fake_mlflow_client.get_experiment_by_name = unreachable  # type: ignore[method-assign]

    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=True)

    assert not report.is_compatible
    assert report.safe_error is not None
    assert "reachability" in report.safe_error


def test_reserved_probe_experiment_existing_fails_the_probe(fake_mlflow_client: FakeMlflowClient) -> None:
    fake_mlflow_client.add_experiment(PROBE_EXPERIMENT_NAME)

    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=True)

    assert not report.is_compatible
    assert report.safe_error is not None
    assert "already exists" in report.safe_error


def test_unknown_trace_must_raise_a_protocol_error(fake_mlflow_client: FakeMlflowClient) -> None:
    def no_error(trace_id: str, display: bool = True, flush: bool = False) -> object:
        return object()

    fake_mlflow_client.get_trace = no_error  # type: ignore[method-assign]

    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=True)

    assert not report.is_compatible
    assert report.safe_error is not None
    assert "protocol" in report.safe_error


def test_transport_error_on_trace_read_fails_the_probe(fake_mlflow_client: FakeMlflowClient) -> None:
    def transport_error(trace_id: str, display: bool = True, flush: bool = False) -> object:
        raise RuntimeError("socket hang up")

    fake_mlflow_client.get_trace = transport_error  # type: ignore[method-assign]

    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=True)

    assert not report.is_compatible
    assert report.safe_error is not None
    assert "protocol" in report.safe_error


def test_protocol_error_classification_uses_error_code(fake_mlflow_client: FakeMlflowClient) -> None:
    # The fake raises FakeMlflowException carrying error_code, which is how a
    # real MlflowException is distinguished from a transport failure.
    with pytest.raises(FakeMlflowException):
        fake_mlflow_client.get_trace("tr-unknown")


def test_require_capabilities_raises_safe_capability_error(fake_mlflow_client: FakeMlflowClient) -> None:
    report = probe_capabilities(fake_mlflow_client, CapabilityRequest(), dry_run=True)

    assert report.is_compatible
    require_capabilities(report)  # compatible: no raise


def test_require_capabilities_aborts_with_exit_status_2(
    fake_mlflow_client: FakeMlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mlflow, "__version__", "0.0.0")
    report = probe_capabilities(fake_mlflow_client, _compatible_request(), dry_run=False)

    with pytest.raises(CapabilityError) as excinfo:
        require_capabilities(report)

    assert excinfo.value.code == "mlflow_capability_unsupported"
    assert excinfo.value.exit_status == 2
    assert excinfo.value.safe_message
    assert str(excinfo.value) == excinfo.value.safe_message
