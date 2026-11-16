"""Native preview verification rejects NULL and generated text as absence."""

from dataclasses import replace
from typing import Any

import pytest

from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.capabilities import CapabilityRequest
from mlflow_import.mlflow.mapping import map_turn_to_trace

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("source_preview", [None, ""])
@pytest.mark.parametrize("field", ["request_preview", "response_preview"])
@pytest.mark.parametrize("corrupt_preview", [None, "generated text", " "])
def test_only_explicit_empty_native_preview_verifies_as_absent(
    fake_mlflow_client: Any,
    six_turn_session: Any,
    source_preview: str | None,
    field: str,
    corrupt_preview: str | None,
) -> None:
    gateway = fake_mlflow_client.gateway()
    assert gateway.probe(CapabilityRequest()).is_compatible
    experiment = fake_mlflow_client.add_experiment("native-absence")
    request = replace(
        map_turn_to_trace(six_turn_session.turns[0], six_turn_session.context).to_create_request(),
        experiment_id=experiment.experiment_id,
    )
    request = (
        replace(request, request_preview=source_preview)
        if field == "request_preview" else replace(request, response_preview=source_preview)
    )
    created = gateway.create_trace(request)
    info = fake_mlflow_client.get_trace(created.trace_id).info
    assert getattr(info, field) == ""
    gateway.verify_trace(created, request)
    assert getattr(request, field) is source_preview
    setattr(info, field, corrupt_preview)
    with pytest.raises(ItemImportError, match="the native text previews differ"):
        gateway.verify_trace(created, request)
