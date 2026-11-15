"""Compatibility never turns a conflicting wire identity into ownership."""

import json

import pytest

from mlflow_import.compatibility import (
    IdentityIndex,
    decode_ownership,
    decode_result,
    identity_index,
    resolves_conflict,
)
from mlflow_import.errors import ConfigurationError

pytestmark = pytest.mark.unit


def test_conflicting_ownership_aliases_are_rejected():
    with pytest.raises(ConfigurationError, match="ownership"):
        decode_ownership({}, {"opencode.importer": "another-tool", "mlflow_import.importer": "mlflow-import"})


def test_neutral_native_alias_uses_complete_identity_index():
    index = IdentityIndex()
    index.add("pi", "store", "native")
    qualified = json.dumps(["session-v1", "pi", "store", "native"])
    with identity_index(index):
        assert resolves_conflict({"mlflow_import.session_id": "native"}, {}, qualified, "native")
        index.add("pi", "other-store", "native")
        with pytest.raises(ConfigurationError) as caught:
            resolves_conflict({"mlflow_import.session_id": "native"}, {}, qualified, "native")
        assert caught.value.code == "alias_ambiguous"


def test_contradictory_harness_alias_is_disqualified():
    qualified = json.dumps(["session-v1", "pi", "store", "native"])
    assert not resolves_conflict(
        {"mlflow.trace.session": "native", "mlflow_import.harness": "codex"}, {}, qualified, "native"
    )


def test_canonical_identity_cannot_hide_contradictory_tag():
    qualified = json.dumps(["session-v1", "pi", "store", "native"])
    with pytest.raises(ConfigurationError) as caught:
        resolves_conflict(
            {"mlflow.trace.session": qualified, "mlflow_import.harness": "pi"},
            {"mlflow_import.harness": "codex"},
            qualified,
            "native",
        )
    assert caught.value.code == "identity_conflict"


@pytest.mark.parametrize("version", ["mlflow-import.result.v1", "opencode-mlflow.result.v1"])
def test_result_decoder_preserves_original_schema(version):
    import json
    from pathlib import Path

    document = json.loads((Path(__file__).parents[1] / "fixtures/payloads/historical-36ed44f.json").read_text())[
        "result"
    ]
    document["schema_version"] = version
    decoded = decode_result(document)
    assert decoded.original_schema_version == version
    assert decoded.document == document


def test_result_decoder_rejects_unknown_schema():
    with pytest.raises(ConfigurationError) as caught:
        decode_result({"schema_version": "future"})
    assert caught.value.code == "result_schema_unsupported"


@pytest.mark.parametrize(
    "key,value", [("sessions", "invalid"), ("summary", []), ("dry_run", 1), ("errors", ["invalid"])]
)
def test_result_decoder_rejects_invalid_types(key, value):
    import json
    from pathlib import Path

    document = json.loads((Path(__file__).parents[1] / "fixtures/payloads/historical-36ed44f.json").read_text())[
        "result"
    ]
    document[key] = value
    with pytest.raises(ConfigurationError) as caught:
        decode_result(document)
    assert caught.value.code == "result_schema_invalid"


def test_historical_marker_decodes_but_incomplete_identity_is_not_owned():
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from mlflow_import.mlflow.gateway import MlflowGateway, OwnedTrace
    from mlflow_import.mlflow.reconciliation import OwnedTraceCandidate

    fixture = json.loads((Path(__file__).parents[1] / "fixtures/payloads/historical-36ed44f.json").read_text())
    trace = fixture["trace"]
    assert fixture["artifact"]["artifact"]["mapping_version"] == "mapping-v4"
    assert MlflowGateway._is_owned(SimpleNamespace(info=SimpleNamespace(tags={}, trace_metadata=trace["metadata"])))
    candidate = OwnedTraceCandidate.from_owned_trace(
        OwnedTrace(trace["trace_id"], 0, {}, trace["metadata"]), trace["experiment_id"]
    )
    assert candidate.ownership_complete
    incomplete = dict(trace["metadata"])
    incomplete.pop("opencode.operation_key")
    assert not OwnedTraceCandidate.from_owned_trace(
        OwnedTrace(trace["trace_id"], 0, {}, incomplete), trace["experiment_id"]
    ).ownership_complete


@pytest.mark.parametrize("version", ["mlflow-import.result.v1", "opencode-mlflow.result.v1"])
@pytest.mark.parametrize(
    "location,key,invalid",
    [
        ("session", "attempt_id", False),
        ("session", "import_instance_id", ["invalid"]),
        ("session", "attempt_state", {}),
        ("session", "attempt_state", "invented"),
        ("session", "reason", []),
        ("session", "error", "invalid"),
        ("session", "error", {"code": False, "message": "safe"}),
        ("turn", "attempt_state", "invented"),
        ("turn", "error", []),
        ("turn", "error", {"code": "safe", "message": []}),
        ("turn", "planned_action", "invented"),
    ],
)
def test_known_nested_optional_fields_are_typed(version, location, key, invalid):
    import json
    from pathlib import Path

    value = json.loads((Path(__file__).parents[1] / "fixtures/payloads/historical-36ed44f.json").read_text())["result"]
    value["schema_version"] = version
    target = value["sessions"][0]
    if location == "turn":
        target = target["turns"][0]
    target[key] = invalid
    with pytest.raises(ConfigurationError) as caught:
        decode_result(value)
    assert caught.value.code == "result_schema_invalid"


@pytest.mark.parametrize("version", ["mlflow-import.result.v1", "opencode-mlflow.result.v1"])
@pytest.mark.parametrize("state", ["prepared", "ready_to_retire", "complete", "aborted"])
def test_known_optional_lifecycle_enums_roundtrip(version, state):
    import json
    from pathlib import Path

    value = json.loads((Path(__file__).parents[1] / "fixtures/payloads/historical-36ed44f.json").read_text())["result"]
    value["schema_version"] = version
    for item in (value["sessions"][0], value["sessions"][0]["turns"][0]):
        item.update(
            attempt_id="a" * 32,
            import_instance_id="",
            attempt_state=state,
            error={"code": "fixture_error", "message": "synthetic safe message"},
        )
    decoded = decode_result(value)
    assert decoded.original_schema_version == version
    assert decoded.document == value
