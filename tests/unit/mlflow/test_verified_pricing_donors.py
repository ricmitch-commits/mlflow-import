"""Single-read donor verification, using mutable local artifact/trace fakes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode
from requests.exceptions import ConnectionError as RequestsConnectionError

from mlflow_import.content.accounting import MAX_PRICING_MAP_BYTES
from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.gateway import MlflowGateway
from mlflow_import.mlflow.reconciliation import _select_pricing_basis
from mlflow_import.retry import RetryPolicy
from tests.support.fake_gateway import resource_qualified_payload

pytestmark = pytest.mark.unit


class DonorTarget:
    def __init__(self, root: Path) -> None:
        self.traces: dict[str, Any] = {}
        self.artifacts: dict[str, bytes] = {}
        self.downloads: Counter[str] = Counter()
        self.client = Mock(spec=MlflowClient)
        self.client.tracking_uri = "https://synthetic.invalid"
        self.client.get_trace.side_effect = self.get_trace
        self.client.get_trace_info = Mock(side_effect=self.get_trace_info)
        self.client.get_experiment.return_value = SimpleNamespace(artifact_location=root.as_uri())
        self.repository = SimpleNamespace(artifact_uri=root.as_uri(), download_artifacts=self.download)
        self.gateway = MlflowGateway(
            self.client,
            artifact_repository_factory=lambda *args, **kwargs: self.repository,
            read_retry_policy=RetryPolicy(max_attempts=1),
        )

    def get_trace(self, trace_id: str, *, display: bool) -> Any:
        assert display is False
        if trace_id not in self.traces:
            raise MlflowException("synthetic missing trace", error_code=ErrorCode.Value("RESOURCE_DOES_NOT_EXIST"))
        return self.traces[trace_id]

    def get_trace_info(self, trace_id: str) -> Any:
        if trace_id not in self.traces:
            raise MlflowException("synthetic missing trace", error_code=ErrorCode.Value("RESOURCE_DOES_NOT_EXIST"))
        return self.traces[trace_id].info

    def download(self, path: str, root: str) -> str:
        self.downloads[path] += 1
        local = Path(root) / Path(path).name
        local.write_bytes(self.artifacts[path])
        return str(local)

    def donor(self, trace_id: str, turn: str = "turn") -> dict[str, Any]:
        payload = json.dumps(
            {
                "schema_version": "mlflow-import.turn-map.v1",
                "identity_map": {"trace_id": trace_id},
                "turn_key": turn,
                "turn_fingerprint": "fingerprint",
                "accounting": {"invocations": {"call": {"cost": {"basis": {"donor": trace_id}}}}},
            }
        ).encode()
        map_path = f"mlflow-import/v2/{trace_id}.json"
        session_path = "mlflow-import/v2/session.json"
        self.artifacts[map_path] = payload
        self.artifacts[session_path] = b"session"
        metadata = {
            "mlflow_import.importer": "mlflow-import",
            "mlflow_import.harness": "opencode",
            "mlflow_import.source_namespace": "namespace",
            "mlflow_import.session_id": "session",
            "mlflow_import.turn_key": turn,
            "mlflow_import.qualified_session_id": json.dumps(["v1", "opencode", "namespace", "session"]),
            "opencode.render_fingerprint": "fingerprint",
        }
        tags = {
            "opencode.artifact_path": map_path,
            "opencode.artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "opencode.artifact_size": str(len(payload)),
            "mlflow_import.session_artifact": session_path,
            "mlflow_import.session_artifact_sha256": hashlib.sha256(b"session").hexdigest(),
            "mlflow_import.session_artifact_size": "7",
        }
        proof, data = resource_qualified_payload()
        tags.update(proof)
        info = SimpleNamespace(
            trace_id=trace_id,
            experiment_id="experiment",
            tags=tags,
            trace_metadata=metadata,
            state=SimpleNamespace(value="OK"),
            request_time=1,
            execution_duration=2,
            client_request_id=None,
            request_preview=None,
            response_preview=None,
            token_usage=None,
            cost=None,
        )
        self.traces[trace_id] = SimpleNamespace(
            trace_id=trace_id, info=info, data=data
        )
        return self.gateway.replacement_reference(trace_id)

    def reset_counts(self) -> None:
        self.client.reset_mock()
        self.downloads.clear()


@pytest.fixture
def target(tmp_path: Path) -> DonorTarget:
    return DonorTarget(tmp_path)


def completed(instance: str, *references: dict[str, Any]) -> dict[str, Any]:
    return {"complete": {"instance_id": instance, "replacements": list(references)}}


def test_selection_reads_each_eligible_donor_once_including_unselected_instances(target: DonorTarget) -> None:
    refs = [
        target.donor(trace_id, turn)
        for trace_id, turn in [("a", "turn"), ("z", "turn"), ("b", "turn"), ("c", "other"), ("x", "turn")]
    ]
    attempts = [completed("duplicate-z", refs[0]), completed("", *refs[1:4]), completed("duplicate-a", refs[4])]
    target.reset_counts()

    assert _select_pricing_basis(attempts, target.gateway, "overwrite") == {
        "turn": {"call": {"donor": "b"}}, "other": {"call": {"donor": "c"}}
    }
    assert Counter(call.args[0] for call in target.client.get_trace.call_args_list) == Counter("azbcx")
    assert target.downloads == Counter({**{ref["map"][0]: 1 for ref in refs}, refs[0]["session"][0]: 1})
    # No preview/global cache: another selection obtains fresh traces and maps.
    _select_pricing_basis(attempts, target.gateway, "overwrite")
    assert target.client.get_trace.call_count == 10
    assert all(target.downloads[ref["map"][0]] == 2 for ref in refs)
    assert target.downloads[refs[0]["session"][0]] == 1


@pytest.mark.parametrize("method", ["read_pricing_basis", "read_pricing_donor", "verify_replacement_reference"])
def test_public_read_and_verify_share_one_verified_map_read(target: DonorTarget, method: str) -> None:
    ref = target.donor("trace")
    target.reset_counts()
    result = getattr(target.gateway, method)(ref)
    assert result == (None if method == "verify_replacement_reference" else {"call": {"donor": "trace"}})
    assert target.client.get_trace.call_count == 1
    assert target.downloads == Counter({ref["map"][0]: 1, ref["session"][0]: 1})


@pytest.mark.parametrize("mutation", ["checksum", "short", "long"])
@pytest.mark.parametrize("method", ["read_pricing_basis", "verify_replacement_reference"])
def test_cached_map_verification_never_authorizes_different_pricing_bytes(
    target: DonorTarget, mutation: str, method: str
) -> None:
    ref = target.donor("trace")
    target.gateway.verify_experiment_file("experiment", *ref["map"])
    path = ref["map"][0]
    payload = target.artifacts[path]
    target.artifacts[path] = {
        "checksum": payload.replace(b"fingerprint", b"tamperprint"),
        "short": payload[:-1],
        "long": payload + b" ",
    }[mutation]
    target.reset_counts()
    with pytest.raises(ItemImportError, match="checksum differs") as raised:
        getattr(target.gateway, method)(ref)
    assert raised.value.code == "artifact_verification_failed"
    assert target.downloads[path] == 1
    assert _select_pricing_basis([completed("", ref)], target.gateway, "overwrite") == {}


def test_checksum_is_checked_against_the_bytes_actually_parsed(target: DonorTarget) -> None:
    ref = target.donor("trace")
    original = target.download

    def changing_download(path: str, root: str) -> str:
        local = original(path, root)
        if path == ref["map"][0]:
            # A former verify-then-refetch implementation parsed this changed
            # accounting payload on its second map download.
            target.artifacts[path] = target.artifacts[path].replace(b'"trace"', b'"rogue"')
        return local

    target.repository.download_artifacts = changing_download
    assert target.gateway.read_pricing_basis(ref) == {"call": {"donor": "trace"}}
    assert target.downloads[ref["map"][0]] == 1
    with pytest.raises(ItemImportError, match="checksum differs"):
        target.gateway.read_pricing_basis(ref)


@pytest.mark.parametrize("field", ["payload", "ownership", "native", "trace_id", "experiment"])
def test_changed_trace_is_not_a_donor(target: DonorTarget, field: str) -> None:
    ref = target.donor("trace")
    trace = target.traces["trace"]
    if field == "payload":
        trace.data.to_dict = lambda: {"spans": ["changed"]}
    elif field == "ownership":
        trace.info.trace_metadata["mlflow_import.redaction_mode"] = "changed"
    elif field == "native":
        trace.info.request_time += 1
    elif field == "trace_id":
        trace.trace_id = "other"
    else:
        trace.info.experiment_id = "other"
    assert _select_pricing_basis([completed("", ref)], target.gateway, "overwrite") == {}


@pytest.mark.parametrize("field", ["trace_id", "turn_key", "turn_fingerprint", "schema_version", "identity_map"])
def test_checksummed_map_must_still_match_source_identity(target: DonorTarget, field: str) -> None:
    ref = target.donor("trace")
    path = ref["map"][0]
    value = json.loads(target.artifacts[path])
    if field == "trace_id":
        value["identity_map"]["trace_id"] = "other"
    else:
        value[field] = "other"
    payload = json.dumps(value).encode()
    target.artifacts[path] = payload
    ref["map"] = [path, hashlib.sha256(payload).hexdigest(), len(payload)]
    assert _select_pricing_basis([completed("", ref)], target.gateway, "overwrite") == {}


def test_oversize_map_is_rejected_before_download(target: DonorTarget) -> None:
    ref = target.donor("trace")
    ref["map"][2] = MAX_PRICING_MAP_BYTES + 1
    with pytest.raises(ItemImportError) as raised:
        target.gateway.read_pricing_basis(ref)
    assert raised.value.code == "replacement_verification_failed"
    assert not target.downloads


def test_session_artifact_is_still_verified(target: DonorTarget) -> None:
    ref = target.donor("trace")
    target.artifacts[ref["session"][0]] = b"tampered"
    assert _select_pricing_basis([completed("", ref)], target.gateway, "overwrite") == {}
    assert target.downloads[ref["session"][0]] == 1


def test_missing_trace_invalidates_whole_completed_attempt(target: DonorTarget) -> None:
    good = target.donor("good")
    expired = target.donor("expired", "other")
    fallback = target.donor("fallback")
    del target.traces["expired"]
    target.reset_counts()
    assert _select_pricing_basis(
        [completed("", good, expired), completed("duplicate-a", fallback)], target.gateway, "overwrite"
    ) == {"turn": {"call": {"donor": "fallback"}}}
    assert Counter(call.args[0] for call in target.client.get_trace_info.call_args_list) == Counter(
        {"good": 1, "expired": 1, "fallback": 1}
    )
    assert Counter(call.args[0] for call in target.client.get_trace.call_args_list) == Counter(
        {"good": 1, "fallback": 1}
    )
    assert target.gateway.read_pricing_donor(expired) is None
    with pytest.raises(MlflowException):
        target.gateway.read_pricing_basis(expired)


@pytest.mark.parametrize("boundary", ["trace", "artifact"])
@pytest.mark.parametrize(
    "code", ["PERMISSION_DENIED", "RESOURCE_DOES_NOT_EXIST", "read_retry_exhausted", "retry_budget_exhausted"]
)
def test_only_missing_trace_is_skipped_other_read_failures_propagate(
    target: DonorTarget, boundary: str, code: str
) -> None:
    ref = target.donor("trace")
    error = (
        MlflowException("synthetic", error_code=ErrorCode.Value(code))
        if code.isupper()
        else ItemImportError(code, "synthetic")
    )
    if boundary == "trace":
        target.client.get_trace.side_effect = error
    else:
        target.repository.download_artifacts = Mock(side_effect=error)
    if boundary == "trace" and code == "RESOURCE_DOES_NOT_EXIST":
        assert _select_pricing_basis([completed("", ref)], target.gateway, "overwrite") == {}
    else:
        with pytest.raises(type(error)) as raised:
            _select_pricing_basis([completed("", ref)], target.gateway, "overwrite")
        assert raised.value is error


def test_fresh_pre_retirement_verification_does_not_reuse_donor_result(target: DonorTarget) -> None:
    ref = target.donor("trace")
    assert target.gateway.read_pricing_donor(ref)
    target.traces["trace"].info.execution_duration += 1
    target.reset_counts()
    with pytest.raises(ItemImportError) as raised:
        target.gateway.verify_replacement_reference(ref)
    assert raised.value.code == "replacement_verification_failed"
    assert target.client.get_trace.call_count == 1


def test_transient_donor_read_retries_then_verifies_one_map(target: DonorTarget) -> None:
    ref = target.donor("trace")
    target.gateway._read_retry_policy = RetryPolicy(
        max_attempts=2, initial_delay_seconds=0, max_delay_seconds=0
    )
    target.reset_counts()
    target.client.get_trace.side_effect = [RequestsConnectionError("synthetic disconnect"), target.traces["trace"]]
    assert _select_pricing_basis([completed("", ref)], target.gateway, "overwrite") == {
        "turn": {"call": {"donor": "trace"}}
    }
    assert target.client.get_trace.call_count == 2
    assert target.downloads[ref["map"][0]] == 1


@pytest.mark.parametrize("field", ["importer", "harness", "source_namespace", "qualified_session_id"])
def test_ineligible_completed_instance_is_not_read(target: DonorTarget, field: str) -> None:
    ref = target.donor("trace")
    del ref["metadata"]["mlflow_import." + field]
    target.reset_counts()
    assert _select_pricing_basis([completed("", ref)], target.gateway, "overwrite") == {}
    assert target.client.get_trace.call_count == 0
    assert not target.downloads


@pytest.mark.parametrize("policy", ["duplicate", "ignore"])
def test_non_donor_policies_and_pending_basis_do_not_read(target: DonorTarget, policy: str) -> None:
    ref = target.donor("trace")
    target.reset_counts()
    assert _select_pricing_basis([completed("", ref)], target.gateway, policy) == {}
    saved = {"turn": {"call": {"donor": "frozen"}}}
    pending = {"prepared": {"pricing_bases": saved}}
    assert _select_pricing_basis([pending, completed("", ref)], target.gateway, policy) == saved
    with pytest.raises(ItemImportError) as raised:
        _select_pricing_basis([pending, pending], target.gateway, policy)
    assert raised.value.code == "attempt_state_conflict"
    assert target.client.get_trace.call_count == 0
