"""The real emergency producer qualifies for a narrow, typed v1 read path."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mlflow_import import cli, importer, resources
from mlflow_import.compatibility import decode_result
from mlflow_import.errors import ConfigurationError, ItemImportError
from mlflow_import.reporting import render_json
from mlflow_import.retry import _BUDGET, check_budget

pytestmark = pytest.mark.unit


@pytest.fixture
def emergency_document(config):
    result = importer._storage_failure_result(
        replace(config, dry_run=True),
        ItemImportError("retry_budget_exhausted", "the shared invocation deadline was exhausted"),
        None,
    )
    return json.loads(render_json(result))


@pytest.mark.parametrize("failure", ["timeout", "quota", "rendering"])
def test_cli_emergency_output_roundtrips_through_v1_decoder(config, import_dependencies, monkeypatch, capsys, failure):
    if failure == "rendering":

        def fail_render(*args, **kwargs):
            raise RuntimeError("PRIVATE rendering detail")

        monkeypatch.setattr(cli, "_stage_output", fail_render)
    else:

        def fail_pipeline(*args, **kwargs):
            if failure == "timeout":
                _BUDGET.get().deadline = -1
                check_budget()
            else:
                budget = resources.current_storage()
                budget.resize(budget.root / "synthetic-over-quota", budget.max_bytes + 1)

        monkeypatch.setattr(importer, "_run_pipeline", fail_pipeline)
    status = cli.main(
        ["--experiment-name", "synthetic", "--db", str(config.db_path), "--dry-run", "--output", "json"],
        import_dependencies,
    )
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    decoded = decode_result(document)
    assert decoded.document is document
    assert decoded.original_schema_version == "mlflow-import.result.v1"
    assert document["summary"]["counts_complete"] is False
    assert document["summary"]["turns_imported"] is None
    assert (
        document["errors"][0]["code"]
        == {
            "timeout": "retry_budget_exhausted",
            "quota": "temporary_storage_exhausted",
            "rendering": "result_render_failed",
        }[failure]
    )
    assert status == 2
    assert "PRIVATE" not in captured.out + captured.err
    assert captured.err == ""
    assert import_dependencies.gateway.mutations == []


@pytest.mark.parametrize("pending", [False, True])
def test_emergency_decoder_preserves_recovery_state_and_available_counts(emergency_document, pending):
    emergency_document["summary"]["recovery_pending"] = pending
    emergency_document["source"]["sessions_scanned"] = 3
    decoded = decode_result(emergency_document)
    assert decoded.document["summary"]["recovery_pending"] is pending
    assert decoded.document["source"]["sessions_scanned"] == 3


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("summary", "incomplete", False),
        ("summary", "incomplete", 1),
        ("summary", "counts_complete", True),
        ("summary", "counts_complete", 0),
        ("summary", "reporting_failures", 0),
        ("summary", "reporting_failures", True),
        ("summary", "reporting_failures", -1),
        ("summary", "reporting_failures", None),
        ("summary", "recovery_pending", None),
        ("summary", "recovery_pending", 1),
        ("summary", "warnings", [{"code": False, "message": "safe"}]),
        ("summary", "planned", {"imported": None, "updated": None, "retired": False}),
        ("source", "schema_generation", "current"),
        ("source", "fingerprint", {"invalid_identity": True}),
        ("redaction", "mode", "invented"),
        ("redaction", "detector_counts", {"invalid": -1}),
        ("experiment", "id", []),
        ("experiment", "status", "resolved"),
    ],
)
def test_malformed_incomplete_fields_are_rejected(emergency_document, section, key, value):
    emergency_document[section][key] = value
    with pytest.raises(ConfigurationError) as caught:
        decode_result(emergency_document)
    assert caught.value.code == "result_schema_invalid"


@pytest.mark.parametrize("key", ["incomplete", "counts_complete", "reporting_failures", "recovery_pending", "planned"])
def test_emergency_qualification_fields_cannot_be_missing(emergency_document, key):
    del emergency_document["summary"][key]
    with pytest.raises(ConfigurationError, match="typed v1 schema"):
        decode_result(emergency_document)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", "opencode-mlflow.result.v1"),
        ("dry_run", "true"),
        ("errors", []),
        ("errors", [{"code": "invalid code", "message": "safe"}]),
        ("errors", [{"code": "safe_code", "message": None}]),
        ("sessions", [{"session_id": "unvalidated", "action": "invented"}]),
        ("sessions", {"invalid": "identity"}),
        ("sources", [{"fingerprint": {"invalid": "identity"}}]),
    ],
)
def test_emergency_flags_do_not_bypass_envelope_or_identity_validation(emergency_document, key, value):
    emergency_document[key] = value
    with pytest.raises(ConfigurationError) as caught:
        decode_result(emergency_document)
    assert caught.value.code == "result_schema_invalid"


def test_emergency_envelope_cannot_carry_even_otherwise_valid_session_claims(emergency_document):
    fixture_path = Path(__file__).parents[1] / "fixtures/payloads/historical-36ed44f.json"
    historical = json.loads(fixture_path.read_text())["result"]
    assert decode_result(historical)
    emergency_document["sessions"] = historical["sessions"]
    with pytest.raises(ConfigurationError, match="typed v1 schema"):
        decode_result(emergency_document)


@pytest.mark.parametrize(
    "section,key",
    [
        ("source", "sessions_selected"),
        ("redaction", "affected_record_count"),
        ("summary", "turns_failed"),
        ("summary", "updated_turns"),
    ],
)
def test_unavailable_emergency_counters_must_still_be_present(emergency_document, section, key):
    del emergency_document[section][key]
    with pytest.raises(ConfigurationError, match="typed v1 schema"):
        decode_result(emergency_document)


@pytest.mark.parametrize(
    "section,key",
    [
        ("source", "sessions_scanned"),
        ("redaction", "affected_field_count"),
        ("summary", "turns_imported"),
        ("summary", "unchanged_turns"),
        ("summary", "failed_sessions"),
    ],
)
@pytest.mark.parametrize("invalid", [-1, True, 0.5, "unavailable", []])
def test_emergency_count_exception_allows_only_null_or_nonnegative_integer(emergency_document, section, key, invalid):
    emergency_document[section][key] = invalid
    with pytest.raises(ConfigurationError, match="typed v1 schema"):
        decode_result(emergency_document)


@pytest.mark.parametrize("version", ["mlflow-import.result.v1", "opencode-mlflow.result.v1"])
@pytest.mark.parametrize(
    "section,key",
    [
        ("source", "sessions_scanned"),
        ("source", "sessions_selected"),
        ("source", "logical_turn_count"),
        ("source", "record_count"),
        ("redaction", "affected_field_count"),
        ("redaction", "affected_record_count"),
        ("summary", "selected_sessions"),
        ("summary", "selected_turns"),
        ("summary", "turns_imported"),
        ("summary", "turns_updated"),
        ("summary", "turns_skipped"),
        ("summary", "turns_failed"),
        ("summary", "retired_traces"),
        ("summary", "failed_items"),
        ("summary", "imported_turns"),
        ("summary", "updated_turns"),
        ("summary", "skipped_turns"),
        ("summary", "unchanged_turns"),
        ("summary", "failed_turns"),
        ("summary", "failed_sessions"),
        ("summary", "failed_retirements"),
        ("summary", "unexamined_owned_traces"),
    ],
)
def test_normal_v1_null_counts_are_still_rejected(version, section, key):
    fixture_path = Path(__file__).parents[1] / "fixtures/payloads/historical-36ed44f.json"
    document = json.loads(fixture_path.read_text())["result"]
    document["schema_version"] = version
    assert decode_result(document).original_schema_version == version
    document[section][key] = None
    with pytest.raises(ConfigurationError, match="typed v1 schema"):
        decode_result(document)


@pytest.mark.parametrize("version", ["mlflow-import.result.v1", "opencode-mlflow.result.v1"])
def test_normal_v1_planned_counts_do_not_inherit_emergency_null_exception(version):
    fixture_path = Path(__file__).parents[1] / "fixtures/payloads/historical-36ed44f.json"
    document = json.loads(fixture_path.read_text())["result"]
    document["schema_version"] = version
    document["summary"]["planned"] = {"imported": 1, "updated": 0, "retired": None}
    with pytest.raises(ConfigurationError, match="typed v1 schema"):
        decode_result(document)
