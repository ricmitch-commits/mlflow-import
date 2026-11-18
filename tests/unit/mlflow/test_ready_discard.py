"""Discarding an unverifiable ready batch never touches unrelated work."""

from __future__ import annotations

import mlflow
import pytest

from mlflow_import.compatibility import decode_ownership
from mlflow_import.mlflow.capabilities import CapabilityRequest
from mlflow_import.mlflow.reconciliation import _discard_invalid_ready
from tests.support.fake_gateway import trace_spec

pytestmark = pytest.mark.unit

_EXPERIMENT = "exp-1"
_SESSION = "ses-discard"
_ATTEMPT = "a" * 32


class _Journal:
    """Record the terminal publication without touching an artifact repository."""

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        self.published: list[tuple[str, dict[str, object]]] = []

    def publish(self, state: str, values):
        self.published.append((state, dict(values)))
        return {**values, "state": state}


def _staged(fake_mlflow_client, monkeypatch, *, attempt_id: str = _ATTEMPT):
    client = fake_mlflow_client
    gateway = client.gateway()
    assert gateway.probe(CapabilityRequest(needs_retirement=True)).is_compatible
    monkeypatch.setattr(mlflow, "flush_trace_async_logging", lambda: None)
    created = gateway.create_trace(
        trace_spec(
            experiment_id=_EXPERIMENT,
            native_session_id=_SESSION,
            root_tags={
                "mlflow_import.attempt_id": attempt_id,
                "mlflow_import.attempt_state": "staged",
            },
        )
    )
    info = client.get_trace_info(created.trace_id)
    reference = {
        "trace_id": created.trace_id,
        "metadata": decode_ownership(info.tags, info.trace_metadata),
    }
    prepared = {
        "attempt_id": attempt_id,
        "session_id": _SESSION,
        "qualified_id": '["v1","opencode","c","ses-discard"]',
    }
    return client, gateway, reference, prepared


def _states(prepared, reference, retirements=()):
    return {
        "prepared": prepared,
        "ready_to_retire": {"replacements": [reference], "retirements": list(retirements)},
    }


def test_unverifiable_ready_batch_is_discarded_and_its_staged_trace_removed(fake_mlflow_client, monkeypatch) -> None:
    client, gateway, reference, prepared = _staged(fake_mlflow_client, monkeypatch)
    journal = _Journal(_EXPERIMENT)

    assert _discard_invalid_ready(journal, _states(prepared, reference), gateway)

    assert journal.published == [("discarded", prepared)]
    assert reference["trace_id"] not in client._traces


def test_discard_refuses_when_a_recorded_retirement_is_already_gone(fake_mlflow_client, monkeypatch) -> None:
    client, gateway, reference, prepared = _staged(fake_mlflow_client, monkeypatch)
    journal = _Journal(_EXPERIMENT)
    retirements = [{"trace_id": "trace-already-retired"}]

    assert not _discard_invalid_ready(journal, _states(prepared, reference, retirements), gateway)

    assert journal.published == []
    assert reference["trace_id"] in client._traces


def test_discard_refuses_a_replacement_owned_by_another_attempt(fake_mlflow_client, monkeypatch) -> None:
    client, gateway, reference, prepared = _staged(fake_mlflow_client, monkeypatch, attempt_id="b" * 32)
    journal = _Journal(_EXPERIMENT)
    prepared = {**prepared, "attempt_id": _ATTEMPT}

    assert not _discard_invalid_ready(journal, _states(prepared, reference), gateway)

    assert journal.published == []
    assert reference["trace_id"] in client._traces


def test_discard_refuses_a_changed_replacement_metadata(fake_mlflow_client, monkeypatch) -> None:
    client, gateway, reference, prepared = _staged(fake_mlflow_client, monkeypatch)
    journal = _Journal(_EXPERIMENT)
    forged = {**reference, "metadata": {**reference["metadata"], "mlflow_import.session_id": "ses-other"}}

    assert not _discard_invalid_ready(journal, _states(prepared, forged), gateway)

    assert journal.published == []
    assert reference["trace_id"] in client._traces
