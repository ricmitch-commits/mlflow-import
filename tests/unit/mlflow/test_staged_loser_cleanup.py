"""Exact scoped staging cleanup, including races at the real gateway delete guard."""

import json
from dataclasses import replace

import mlflow
import pytest

from mlflow_import.errors import ImporterError, ItemImportError
from mlflow_import.mlflow.capabilities import CapabilityRequest
from mlflow_import.mlflow.reconciliation import OwnedTraceCandidate, _cleanup_staged_losers
from tests.support.fake_gateway import trace_spec

pytestmark = pytest.mark.unit

_SESSION = "ses-sample"
_TURN = "turn-" + "a" * 64
_NAMESPACE = "c" * 64
_QUALIFIED = json.dumps(["v1", "opencode", _NAMESPACE, _SESSION], separators=(",", ":"))


@pytest.fixture
def cleanup_scope(fake_mlflow_client, monkeypatch):
    client = fake_mlflow_client
    gateway = client.gateway()
    assert gateway.probe(CapabilityRequest(needs_retirement=True)).is_compatible
    ownership = {
        "mlflow_import.harness": "opencode",
        "mlflow_import.source_namespace": _NAMESPACE,
        "mlflow_import.qualified_session_id": _QUALIFIED,
    }
    tags = {
        **ownership,
        "mlflow_import.attempt_id": "d" * 32,
        "mlflow_import.attempt_state": "staged",
    }
    created = gateway.create_trace(trace_spec(root_tags=tags))
    candidate = OwnedTraceCandidate(created.trace_id, _SESSION, _TURN, 1, experiment_id="exp-1")
    attempt = {
        "attempt_id": tags["mlflow_import.attempt_id"],
        "session_id": _SESSION,
        "qualified_id": _QUALIFIED,
        "instance_id": "",
        "prior_ids": ["original-prior"],
    }
    verified = []
    monkeypatch.setattr(
        gateway, "verify_replacement_reference", lambda reference: verified.append(reference["trace_id"])
    )
    monkeypatch.setattr(mlflow, "flush_trace_async_logging", lambda: None)
    confirmed = {}
    references = [{"trace_id": "winner"}, {"trace_id": "second-winner"}]

    def cleanup(losers=None):
        _cleanup_staged_losers(gateway, "exp-1", attempt, ownership, losers or [candidate], references, confirmed)

    return client, gateway, candidate, attempt, ownership, verified, confirmed, cleanup


_MISMATCHES = [
    ("opencode.importer", "foreign"),
    ("opencode.session_id", "different-session"),
    ("opencode.turn_key", "turn-" + "e" * 64),
    ("mlflow_import.harness", "claude"),
    ("mlflow_import.source_namespace", "e" * 64),
    ("mlflow_import.qualified_session_id", '["v1","opencode","other","ses-sample"]'),
    ("mlflow_import.import_instance_id", "f" * 32),
    ("mlflow_import.attempt_id", "f" * 32),
    ("mlflow_import.attempt_state", "complete"),
    ("experiment_id", "different-experiment"),
]


@pytest.mark.parametrize("key,value", _MISMATCHES)
@pytest.mark.parametrize("when", ["inventory", "immediately_before_delete"])
def test_identity_mismatch_never_deletes(cleanup_scope, monkeypatch, key, value, when):
    client, gateway, candidate, attempt, ownership, verified, confirmed, cleanup = cleanup_scope
    stored = client._traces[candidate.trace_id]

    def change():
        if key == "experiment_id":
            stored.info.experiment_id = value
        elif key.startswith("opencode."):
            stored.info.trace_metadata[key] = value
        else:
            stored.info.tags[key] = value

    if when == "inventory":
        change()
    else:
        monkeypatch.setattr(mlflow, "flush_trace_async_logging", change)
    mutations = len(client.mutations)
    with pytest.raises(ImporterError) as caught:
        cleanup()
    assert caught.value.code == "ownership_conflict"
    assert client.mutations[mutations:] == []
    assert candidate.trace_id in client._traces
    assert confirmed == {}
    assert verified == ["winner", "second-winner"]
    assert attempt["prior_ids"] == ["original-prior"]


def test_all_winners_reverified_without_cached_artifacts_before_cleanup(cleanup_scope, monkeypatch):
    client, gateway, candidate, attempt, ownership, verified, confirmed, cleanup = cleanup_scope
    gateway._artifact_checks["stale"] = True

    def verify(reference):
        assert not gateway._artifact_checks
        assert candidate.trace_id in client._traces
        verified.append(reference["trace_id"])
        if reference["trace_id"] == "second-winner":
            raise ItemImportError("replacement_verification_failed", "synthetic failed second winner")

    monkeypatch.setattr(gateway, "verify_replacement_reference", verify)
    mutations = len(client.mutations)
    with pytest.raises(ItemImportError, match="synthetic failed second winner"):
        cleanup()
    assert client.mutations[mutations:] == []
    assert confirmed == {}
    assert verified == ["winner", "second-winner"]


@pytest.mark.parametrize("protected", ["original-prior", "winner", "second-winner"])
def test_cleanup_never_deletes_a_prior_or_replacement(cleanup_scope, protected):
    client, gateway, candidate, attempt, ownership, verified, confirmed, cleanup = cleanup_scope
    mutations = len(client.mutations)
    with pytest.raises(ItemImportError) as caught:
        cleanup([replace(candidate, trace_id=protected)])
    assert caught.value.code == "ownership_conflict"
    assert client.mutations[mutations:] == []
    assert confirmed == {}


def test_cleanup_checks_every_loser_scope_before_first_delete(cleanup_scope):
    client, gateway, candidate, attempt, ownership, verified, confirmed, cleanup = cleanup_scope
    created = gateway.create_trace(
        trace_spec(
            root_tags={
                **ownership,
                "mlflow_import.attempt_id": "e" * 32,
                "mlflow_import.attempt_state": "staged",
            }
        )
    )
    mutations = len(client.mutations)
    with pytest.raises(ItemImportError) as caught:
        cleanup([candidate, replace(candidate, trace_id=created.trace_id)])
    assert caught.value.code == "ownership_conflict"
    assert client.mutations[mutations:] == []
    assert confirmed == {}


def test_confirmed_partial_cleanup_survives_a_later_cleanup_failure(cleanup_scope, monkeypatch):
    client, gateway, candidate, attempt, ownership, verified, confirmed, cleanup = cleanup_scope
    created = gateway.create_trace(
        trace_spec(
            root_tags={
                **ownership,
                "mlflow_import.attempt_id": attempt["attempt_id"],
                "mlflow_import.attempt_state": "staged",
            }
        )
    )
    retire = gateway.retire_trace

    def fail_second(experiment_id, trace_id, **kwargs):
        if trace_id == created.trace_id:
            raise ItemImportError("trace_retirement_failed", "synthetic second cleanup failure")
        retire(experiment_id, trace_id, **kwargs)

    monkeypatch.setattr(gateway, "retire_trace", fail_second)
    with pytest.raises(ItemImportError, match="synthetic second cleanup failure"):
        cleanup([candidate, replace(candidate, trace_id=created.trace_id)])
    assert confirmed == {candidate.trace_id: _SESSION}
    assert not gateway.trace_exists(candidate.trace_id)
    assert gateway.trace_exists(created.trace_id)
    assert attempt["prior_ids"] == ["original-prior"]


@pytest.mark.parametrize("instance_id", ["", "e" * 32])
def test_only_exact_attempt_instance_is_cleaned_and_confirmed(cleanup_scope, monkeypatch, instance_id):
    client, gateway, candidate, attempt, ownership, verified, confirmed, cleanup = cleanup_scope
    attempt["instance_id"] = instance_id
    client._traces[candidate.trace_id].info.tags["mlflow_import.import_instance_id"] = instance_id
    retire = gateway.retire_trace
    checks = []

    def guarded_retire(experiment_id, trace_id, **kwargs):
        assert verified == ["winner", "second-winner"]
        assert trace_id == candidate.trace_id
        assert kwargs["expected_metadata"]["mlflow_import.import_instance_id"] == instance_id
        assert kwargs["expected_metadata"]["mlflow_import.qualified_session_id"] == _QUALIFIED
        retire(experiment_id, trace_id, **kwargs)
        checks.append(trace_id)

    monkeypatch.setattr(gateway, "retire_trace", guarded_retire)
    cleanup()
    assert checks == [candidate.trace_id]
    assert not gateway.trace_exists(candidate.trace_id)
    assert confirmed == {candidate.trace_id: _SESSION}
    assert attempt["prior_ids"] == ["original-prior"]


@pytest.mark.parametrize("key", ["mlflow_import.attempt_id", "mlflow_import.attempt_state"])
def test_conflicting_staging_metadata_and_tags_are_rejected(cleanup_scope, key):
    client, gateway, candidate, attempt, ownership, verified, confirmed, cleanup = cleanup_scope
    client._traces[candidate.trace_id].info.trace_metadata[key] = "contradictory"
    mutations = len(client.mutations)
    with pytest.raises(ItemImportError) as caught:
        cleanup()
    assert caught.value.code == "ownership_conflict"
    assert client.mutations[mutations:] == []
    assert confirmed == {}


def test_prior_ids_are_indexed_once_instead_of_scanned_for_each_loser(cleanup_scope):
    client, gateway, candidate, attempt, ownership, verified, confirmed, cleanup = cleanup_scope

    class IndexOnlyPriors(list):
        def __contains__(self, value):
            pytest.fail("cleanup must use its disk index for prior membership")

    attempt["prior_ids"] = IndexOnlyPriors(f"prior-{index}" for index in range(1000))
    cleanup()
    assert confirmed == {candidate.trace_id: _SESSION}
