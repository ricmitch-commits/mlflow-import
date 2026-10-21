"""Contract tests pinning the gateway boundary to the MLflow compatibility profile.

The gateway is the only module that touches MLflow. These contract tests pin
that boundary to Task 0's recorded profile
(``tests/integration/mlflow_profile.py`` / ``docs/compatibility.md``):

- the MLflow operations the gateway actually calls (observed through the
  fake client, fake public trace API, and fake artifact repositories during
  a full workout) are covered by ``MlflowProfile.initial().api_table`` and
  include every operation in the profile's ``REQUIRED_METHODS``;
- the call shapes match the recorded signatures (the non-deprecated
  ``locations`` search form, keyword ``parent_span``/``trace_ids``, no
  ``FutureWarning``);
- the pinned constants (version, inline payload floor, required method set,
  artifact-location tag, no-op sentinel, session metadata key, truncation
  limits) equal the profile's and the installed mlflow's values;
- only the gateway/capability modules import MLflow, and only through the
  public ``mlflow``/``mlflow.tracking``/``mlflow.artifacts`` entry points
  (never the semi-private ``mlflow.store.*`` import Task 0 used for its
  availability check);
- runtime capability probes are read-only in both modes.

No network and no live server: everything runs against the in-memory fakes.
"""

from __future__ import annotations

import ast
import warnings
from dataclasses import fields
from pathlib import Path

import mlflow
import pytest

from opencode_mlflow.mlflow import capabilities as capabilities_module
from opencode_mlflow.mlflow import gateway as gateway_module
from opencode_mlflow.mlflow.capabilities import (
    PINNED_MLFLOW_VERSION,
    CapabilityRequest,
)
from opencode_mlflow.mlflow.capabilities import probe_capabilities as _production_probe_capabilities
from tests.integration.mlflow_profile import (
    MAX_INLINE_PAYLOAD_BYTES,
    MlflowProfile,
)
from tests.integration.mlflow_profile import (
    REQUIRED_METHODS as PROFILE_REQUIRED_METHODS,
)
from tests.integration.mlflow_profile import CapabilityReport as ProfileCapabilityReport
from tests.support.fake_gateway import FakeMlflowClient, child_spec, trace_spec

pytestmark = pytest.mark.contract

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "opencode_mlflow"
_ALLOWED_MLFLOW_IMPORT_ROOTS = frozenset({"mlflow", "mlflow.tracking", "mlflow.artifacts"})
_GATEWAY_MODULES = ("gateway.py", "capabilities.py")


def probe_capabilities(client: FakeMlflowClient, request: CapabilityRequest, *, dry_run: bool):
    """Use an explicit fake version attestation in non-network contracts."""
    return _production_probe_capabilities(
        client,
        request,
        dry_run,
        version_transport=lambda _uri: {"version": PINNED_MLFLOW_VERSION},
    )


def test_dry_run_capability_probe_does_not_mutate(fake_mlflow_client):
    report = probe_capabilities(
        fake_mlflow_client,
        CapabilityRequest(needs_retirement=True, needs_named_artifacts=True),
        dry_run=True,
    )

    assert report.is_compatible
    assert fake_mlflow_client.mutations == []


def _workout_gateway(client: FakeMlflowClient, tmp_path: Path):
    """Drive every gateway operation once; returns the created trace ids."""
    gateway = client.gateway()
    report = gateway.probe(CapabilityRequest(needs_retirement=True, needs_named_artifacts=True))
    assert report.is_compatible, report.safe_error

    reference = gateway.resolve_experiment("contract-workout", create=True)
    assert reference.experiment_id is not None

    spec = trace_spec(
        experiment_id=reference.experiment_id,
        children=[child_spec("msg-1"), child_spec("part-1", parent_source_key="msg-1")],
    )
    created = gateway.create_trace(spec)
    gateway.verify_trace(created, spec)

    payload = tmp_path / "payload.json"
    payload.write_bytes(b'{"contract": "named-artifact"}')
    verification = gateway.upload_named_json(
        created.trace_id, "opencode-mlflow/v1/sessions/s/turns/turn-x/aaa-bbb.json", payload, chunk_size=64
    )
    gateway.verify_named_json(created.trace_id, verification.relative_path, verification.sha256, verification.size)
    gateway.record_artifact_verification(created.trace_id, verification.sha256, verification.size)

    owned = list(gateway.iter_owned_traces(reference.experiment_id))
    assert [trace.trace_id for trace in owned] == [created.trace_id]
    assert owned[0].tags["opencode.artifact_sha256"] == verification.sha256

    gateway.retire_trace(
        reference.experiment_id,
        created.trace_id,
        session_id=spec.native_session_id,
        turn_key=spec.metadata["opencode.turn_key"],
    )
    return created


def test_gateway_calls_only_profile_api_table_operations(tmp_path: Path) -> None:
    client = FakeMlflowClient()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _workout_gateway(client, tmp_path)

    observed = {call[0] for call in client.calls}
    # The trace-API adapter's calls map onto the module-level public API
    # operation names recorded in the profile's api table.
    trace_api_calls = {call[0] for call in client.trace_api.calls}
    observed |= {
        "mlflow.start_span_no_context" if name == "start_span_no_context" else "LiveSpan.end"
        for name in trace_api_calls
    }
    artifact_calls = set()
    for repository in client.artifacts.repositories.values():
        artifact_calls |= {call[0] for call in repository.calls}
    # The named-artifact operations are the public composition recorded in
    # the profile's api table: staging + log_artifact is upload_named_json,
    # download_artifacts + streamed digest is verify_named_json.
    observed |= {"upload_named_json" if name == "log_artifact" else "verify_named_json" for name in artifact_calls}
    # The fake gateway injects the read-only server-version attestation
    # explicitly; it has no MLflow client mutation call to observe.
    observed.add("server_version_attestation")

    api_table = MlflowProfile.initial().api_table
    assert observed <= set(api_table), observed - set(api_table)
    assert PROFILE_REQUIRED_METHODS <= observed, PROFILE_REQUIRED_METHODS - observed


def test_gateway_uses_the_non_deprecated_search_form(tmp_path: Path) -> None:
    client = FakeMlflowClient()
    _workout_gateway(client, tmp_path)

    searches = [call for call in client.calls if call[0] == "search_traces"]
    assert searches
    assert all(call[1]["locations"] is not None for call in searches)
    assert all(call[1]["experiment_ids"] is None for call in searches)


def test_gateway_call_shapes_match_the_recorded_signatures(tmp_path: Path) -> None:
    client = FakeMlflowClient()
    created = _workout_gateway(client, tmp_path)

    span_starts = [call[1] for call in client.trace_api.calls if call[0] == "start_span_no_context"]
    assert all(
        set(kwargs)
        == {
            "name",
            "span_type",
            "parent_span",
            "inputs",
            "attributes",
            "tags",
            "metadata",
            "experiment_id",
            "start_time_ns",
        }
        for kwargs in span_starts
    )
    child_starts = [kwargs for kwargs in span_starts if kwargs["parent_span"] is not None]
    assert all(kwargs["parent_span"].span_id for kwargs in child_starts)

    retrievals = [call[1] for call in client.calls if call[0] == "get_trace"]
    assert all(kwargs["display"] is False for kwargs in retrievals)
    assert any(kwargs["trace_id"] == created.trace_id and kwargs["flush"] is True for kwargs in retrievals), (
        "verify_trace and the artifact sink must retrieve the full payload deterministically"
    )

    deletions = [call[1] for call in client.calls if call[0] == "delete_traces"]
    assert len(deletions) == 1
    assert deletions[0]["trace_ids"] == [created.trace_id]
    assert deletions[0]["experiment_id"] is not None
    assert set(deletions[0]) == {"experiment_id", "trace_ids", "max_timestamp_millis"}


def test_pinned_constants_match_the_profile_and_installed_mlflow() -> None:
    profile = MlflowProfile.initial()

    assert capabilities_module.PINNED_MLFLOW_VERSION == profile.version == mlflow.__version__
    assert capabilities_module.PINNED_MAX_INLINE_PAYLOAD_BYTES == MAX_INLINE_PAYLOAD_BYTES
    assert capabilities_module.REQUIRED_METHODS == PROFILE_REQUIRED_METHODS
    assert set(profile.api_table) >= PROFILE_REQUIRED_METHODS

    # Field parity with the profile's report: the runtime report carries the
    # profile report's fields (plus the effective request and mode).
    profile_report_fields = {
        "is_compatible",
        "required_methods",
        "max_inline_payload_bytes",
        "named_artifact_supported",
        "retirement_supported",
        "safe_error",
    }
    assert {field.name for field in fields(ProfileCapabilityReport)} == profile_report_fields
    assert {field.name for field in fields(capabilities_module.CapabilityReport)} >= profile_report_fields


def test_gateway_constants_match_installed_mlflow() -> None:
    import mlflow.tracing.constant as tracing_constants
    from mlflow.entities.span import NO_OP_SPAN_TRACE_ID
    from mlflow.tracing.constant import TraceMetadataKey
    from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION

    assert gateway_module.TRACE_ARTIFACT_LOCATION_TAG == MLFLOW_ARTIFACT_LOCATION
    assert gateway_module.NO_OP_TRACE_ID == NO_OP_SPAN_TRACE_ID
    assert gateway_module.TRACE_SESSION_METADATA_KEY == TraceMetadataKey.TRACE_SESSION
    assert gateway_module.TRACE_TAG_VALUE_LIMIT == tracing_constants.MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE
    assert gateway_module.TRACE_METADATA_VALUE_LIMIT == tracing_constants.MAX_CHARS_IN_TRACE_INFO_METADATA


def _mlflow_import_roots(module_path: Path) -> set[str]:
    """The mlflow module roots a source file imports (AST, no execution)."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names if alias.name.split(".")[0] == "mlflow")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "mlflow":
            roots.add(node.module)
    return roots


def test_only_the_gateway_modules_import_mlflow() -> None:
    importers = sorted(
        str(path.relative_to(_SRC_ROOT)) for path in _SRC_ROOT.rglob("*.py") if _mlflow_import_roots(path)
    )
    assert importers == ["mlflow/capabilities.py", "mlflow/gateway.py"]


def test_gateway_imports_only_public_mlflow_entry_points() -> None:
    for name in _GATEWAY_MODULES:
        roots = _mlflow_import_roots(_SRC_ROOT / "mlflow" / name)
        assert roots <= _ALLOWED_MLFLOW_IMPORT_ROOTS, roots - _ALLOWED_MLFLOW_IMPORT_ROOTS
        assert all(not root.startswith("mlflow.store") for root in roots)


def test_runtime_probes_are_read_only_in_both_modes(fake_mlflow_client: FakeMlflowClient) -> None:
    before = fake_mlflow_client.snapshot_state()

    normal = probe_capabilities(
        fake_mlflow_client, CapabilityRequest(needs_retirement=True, needs_named_artifacts=True), dry_run=False
    )
    dry = probe_capabilities(
        fake_mlflow_client, CapabilityRequest(needs_retirement=True, needs_named_artifacts=True), dry_run=True
    )

    assert normal.is_compatible, normal.safe_error
    assert dry.is_compatible, dry.safe_error
    assert fake_mlflow_client.mutations == []
    assert fake_mlflow_client.snapshot_state() == before
