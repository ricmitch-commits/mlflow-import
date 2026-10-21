"""Contract test pinning the MLflow compatibility profile for the importer.

The test module runs only against a live MLflow tracking server (markers
``profile`` and ``integration``). The session-scoped ``profile_client``
fixture yields the client of the shared ``live_server`` fixture (see
``tests/integration/server.py``): it either connects to the server named
by ``MLFLOW_TRACKING_URI`` (when it is an HTTP URI) or launches a
disposable server from the installed ``mlflow`` distribution with a
temporary SQLite backend and local artifact root, health-checks it, and
always reaps it on exit. A requested server that cannot satisfy the
behavioral contract fails these tests loudly; nothing in this module
skips silently.

See ``tests/integration/README.md`` for the reproducible command sequence and
``docs/compatibility.md`` for the resolved 3.17.2 public API surface.
"""

from __future__ import annotations

import json

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION

from tests.integration.mlflow_profile import (
    MAX_INLINE_PAYLOAD_BYTES,
    MlflowProfile,
    upload_named_json,
    verify_named_json,
    verify_profile,
)
from tests.integration.server import LiveServer, unique_experiment_name

pytestmark = [pytest.mark.profile, pytest.mark.integration]


@pytest.fixture(scope="session")
def profile_client(live_server: LiveServer):
    """A client for the pinned live MLflow server (the shared session server)."""
    return live_server.client


_ArtifactTree = dict[str, tuple[tuple[str, bool, int], ...]]
_ServerState = tuple[frozenset[str], dict[str, str], _ArtifactTree]


def _list_repo_tree(repo, path: str | None, depth: int) -> list[tuple[str, bool, int]]:
    infos = repo.list_artifacts(path)
    entries = [(info.path, bool(info.is_dir), int(info.file_size or 0)) for info in infos]
    nested: list[tuple[str, bool, int]] = []
    if depth > 0:
        for info in infos:
            if info.is_dir:
                nested.extend(_list_repo_tree(repo, info.path, depth - 1))
    return entries + nested


def _capture_server_state(client: MlflowClient) -> _ServerState:
    """Read-only snapshot of every experiment, trace, and trace artifact tree.

    Flushes the client-side async export queue once up front (a client-side
    drain, never a server mutation) so locally pending traces are visible, then
    collects the state with span-free searches and artifact listings.
    """
    from mlflow.artifacts import get_artifact_repository

    mlflow.flush_trace_async_logging()
    experiments = frozenset(e.experiment_id for e in client.search_experiments())
    traces: dict[str, str] = {}
    for experiment in client.search_experiments():
        page = client.search_traces(locations=[experiment.experiment_id], max_results=100, include_spans=False)
        for trace in page:
            traces[trace.trace_id] = trace.info.tags.get(MLFLOW_ARTIFACT_LOCATION, "")

    artifacts: _ArtifactTree = {}
    for trace_id, uri in sorted(traces.items()):
        if not uri:
            continue
        repo = get_artifact_repository(uri, tracking_uri=client.tracking_uri)
        artifacts[trace_id] = tuple(_list_repo_tree(repo, None, 3))
    return experiments, traces, artifacts


def test_profile_requires_retirement_for_normal_writes(profile_client):
    report = verify_profile(profile_client, MlflowProfile.initial())

    assert report.required_methods >= {
        "get_experiment_by_name",
        "create_experiment",
        "mlflow.start_span_no_context",
        "LiveSpan.end",
        "get_trace",
        "search_traces",
        "delete_traces",
        "upload_named_json",
        "verify_named_json",
    }


def test_profile_report_is_compatible(profile_client):
    report = verify_profile(profile_client, MlflowProfile.initial())

    assert report.is_compatible, report.safe_error
    assert report.safe_error is None
    assert report.named_artifact_supported
    assert report.retirement_supported
    assert report.max_inline_payload_bytes == MAX_INLINE_PAYLOAD_BYTES


def test_full_payload_trace_retrieval(profile_client):
    experiment_id = profile_client.create_experiment(unique_experiment_name("profile-payload"))
    start_ns = 1_700_000_000_000_000_000
    root = mlflow.start_span_no_context(
        name="payload-root",
        span_type="CHAIN",
        inputs={"session_id": "s-1"},
        attributes={"scope": "profile"},
        tags={"opencode-mlflow": "payload-probe"},
        metadata={"probe": "1"},
        experiment_id=experiment_id,
        start_time_ns=start_ns,
    )
    child = mlflow.start_span_no_context(
        name="payload-child",
        span_type="LLM",
        parent_span=root,
        start_time_ns=start_ns + 10_000_000,
    )
    child.end(outputs={"token": "abc"}, status="OK", end_time_ns=start_ns + 20_000_000)
    root.end(
        outputs={"answer": 42},
        attributes={"done": "yes"},
        status="OK",
        end_time_ns=start_ns + 30_000_000,
    )

    trace = profile_client.get_trace(root.trace_id, display=False, flush=True)
    spans = {span.name: span for span in trace.data.spans}

    assert trace.info.trace_id == root.trace_id
    assert trace.info.state.value == "OK"
    assert spans["payload-root"].inputs == {"session_id": "s-1"}
    assert spans["payload-root"].outputs == {"answer": 42}
    assert spans["payload-root"].attributes.get("done") == "yes"
    assert spans["payload-root"].start_time_ns == start_ns
    assert spans["payload-root"].end_time_ns == start_ns + 30_000_000
    assert spans["payload-child"].parent_id == root.span_id
    assert spans["payload-child"].outputs == {"token": "abc"}


def test_named_artifact_preserves_relative_path_and_exact_bytes(profile_client, tmp_path):
    experiment_id = profile_client.create_experiment(unique_experiment_name("profile-artifact"))
    span = mlflow.start_span_no_context(name="artifact-probe", span_type="CHAIN", experiment_id=experiment_id)
    span.end(outputs={"done": True}, status="OK")

    payload = {"kind": "named-artifact-probe", "rows": list(range(512))}
    data = json.dumps(payload, separators=(",", ":")).encode()
    local_file = tmp_path / "session-payload.json"
    local_file.write_bytes(data)

    verification = upload_named_json(
        profile_client,
        span.trace_id,
        "sessions/2026/10/21/session-payload.json",
        local_file,
        chunk_size=1024,
    )

    assert isinstance(verification.sha256, str) and len(verification.sha256) == 64
    assert verification.size == len(data)
    assert verification.trace_id == span.trace_id
    assert verification.relative_path == "sessions/2026/10/21/session-payload.json"

    # Exact-byte retrieval and verification at the requested relative path.
    verify_named_json(profile_client, span.trace_id, verification.relative_path, verification.sha256, verification.size)

    # Verification must reject a tampered digest.
    tampered = "0" * 64
    with pytest.raises(MlflowException, match="sha256"):
        verify_named_json(profile_client, span.trace_id, verification.relative_path, tampered, verification.size)


def test_dry_run_probing_is_non_mutating(profile_client, tmp_path):
    experiment_id = profile_client.create_experiment(unique_experiment_name("profile-dryrun"))
    span = mlflow.start_span_no_context(name="dry-run-seed", span_type="CHAIN", experiment_id=experiment_id)
    span.end(outputs={"seed": True}, status="OK")
    seed_payload = tmp_path / "seed.json"
    seed_payload.write_bytes(b'{"seed": "payload"}')
    upload_named_json(profile_client, span.trace_id, "seed/payload.json", seed_payload, chunk_size=64)

    before = _capture_server_state(profile_client)

    report = verify_profile(profile_client, MlflowProfile.initial(), dry_run=True)

    assert report.is_compatible, report.safe_error
    after = _capture_server_state(profile_client)
    assert after[0] == before[0], "dry-run probe changed the experiment set"
    assert after[1] == before[1], "dry-run probe changed the trace set"
    assert after[2] == before[2], "dry-run probe changed the artifact trees"
