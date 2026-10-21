"""Integration: dry-run and reconciliation contracts against a live MLflow server.

Drives the COMPLETE production pipeline over the sanitized synthetic
six-turn session against the real gateway over the session's live MLflow
server (only the report clock and the retry policy are deterministic).
The module-scoped ``baseline`` fixture performs ONE import under a unique
experiment name; the scenarios below then rerun the pipeline against
carefully mutated copies of the same source — each scenario owns a
DIFFERENT logical turn, so every scenario's assertions are relative to
state it captures itself and no scenario disturbs another's turns.

Verified contract (Task 14, Step 3):

- a dry run creates NO experiment, trace, or artifact: a three-axis
  server-state snapshot (experiment set, trace set, artifact trees) is
  byte-equal before and after, and the report plans the creation;
- an unchanged rerun creates no duplicate owned trace (every turn skips
  as ``unchanged`` with its original trace id), and the recorded
  artifact digests (Ruling 16) are really checked against the stored
  bytes: tampering one trace's recorded digest forces exactly that
  turn's replacement while every sibling stays untouched;
- a changed turn is replaced in the strict order
  create → verify → upload → verify → record digest → retire the old
  (asserted through gateway-boundary event capture plus observable
  server state: the old trace is gone, the new trace carries the
  changed content and a new deterministic artifact path);
- a retirement failure after a verified replacement PRESERVES the old
  trace: the failure is injected at the gateway boundary (a wrapper
  dependency), never by breaking the server, and the new trace is
  rolled back while the old trace remains fully retrievable;
- stale-turn retirement is scoped: dropping one turn's rows retires
  exactly that turn's owned trace, every sibling keeps its trace id,
  and a foreign (non-importer) trace seeded through the public client
  is never touched.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.artifacts import get_artifact_repository
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION

from opencode_mlflow.importer import ImportResult
from opencode_mlflow.mlflow.gateway import DEFAULT_UPLOAD_CHUNK_BYTES, MlflowGateway, TraceCreateRequest
from opencode_mlflow.mlflow.reconciliation import (
    ARTIFACT_PATH_TAG,
    ARTIFACT_SHA256_TAG,
    ARTIFACT_SIZE_TAG,
)
from tests.fixtures.six_turn_session import (
    build_six_turn_source_db,
)
from tests.integration.import_runner import (
    RealImportRun,
    run_real_import,
    trace_ids_by_key,
    turn_entries,
)
from tests.integration.server import LiveServer, unique_experiment_name

pytestmark = pytest.mark.integration

_TURNS = 6

# ---------------------------------------------------------------------------
# Gateway-boundary event capture and failure injection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatewayEvent:
    """One observable call at the gateway boundary."""

    method: str
    trace_id: str | None
    detail: str | None


class RecordingGateway(MlflowGateway):
    """Production gateway wrapper that records every boundary call.

    Delegates everything to the production gateway; appends one
    :class:`GatewayEvent` per create/verify/upload/verify/record/retire
    call so scenarios can assert the observable ORDER of the write
    protocol. ``fail_retire_trace_ids`` injects a retirement failure at
    the boundary for exactly the listed trace ids (the server never sees
    the delete, so the old trace provably survives).
    """

    def __init__(
        self,
        client: MlflowClient,
        *,
        events: list[GatewayEvent],
        fail_retire_trace_ids: frozenset[str] = frozenset(),
        dry_run: bool = False,
    ) -> None:
        super().__init__(client, dry_run=dry_run)
        self._recorded_events = events
        self._fail_retire_trace_ids = fail_retire_trace_ids

    def create_trace(self, spec: TraceCreateRequest):
        created = super().create_trace(spec)
        self._recorded_events.append(
            GatewayEvent("create_trace", created.trace_id, spec.metadata.get("opencode.turn_key"))
        )
        return created

    def verify_trace(self, created, expected: TraceCreateRequest) -> None:
        self._recorded_events.append(
            GatewayEvent("verify_trace", created.trace_id, expected.metadata.get("opencode.turn_key"))
        )
        super().verify_trace(created, expected)

    def upload_named_json(
        self,
        trace_id: str,
        relative_path: str,
        local_file: Path,
        chunk_size: int = DEFAULT_UPLOAD_CHUNK_BYTES,
    ):
        verification = super().upload_named_json(trace_id, relative_path, local_file, chunk_size)
        self._recorded_events.append(GatewayEvent("upload_named_json", trace_id, relative_path))
        return verification

    def verify_named_json(self, trace_id: str, relative_path: str, sha256: str, size: int) -> None:
        self._recorded_events.append(GatewayEvent("verify_named_json", trace_id, relative_path))
        super().verify_named_json(trace_id, relative_path, sha256, size)

    def record_artifact_verification(self, trace_id: str, sha256: str, size: int) -> None:
        self._recorded_events.append(GatewayEvent("record_artifact_verification", trace_id, None))
        super().record_artifact_verification(trace_id, sha256, size)

    def retire_trace(self, experiment_id: str, trace_id: str, **kwargs: object) -> None:
        if trace_id in self._fail_retire_trace_ids:
            # Injected at the gateway boundary: the server never receives
            # the delete, so the old trace provably survives the run.
            self._recorded_events.append(GatewayEvent("retire_trace_failed", trace_id, experiment_id))
            raise RuntimeError("synthetic retirement failure injected at the gateway boundary")
        super().retire_trace(experiment_id, trace_id, **kwargs)
        self._recorded_events.append(GatewayEvent("retire_trace", trace_id, experiment_id))


def _recording_gateway_factory(
    client: MlflowClient,
    events: list[GatewayEvent],
    fail_retire_trace_ids: frozenset[str] = frozenset(),
):
    """A gateway factory building one recording gateway per run mode."""

    def factory(dry_run: bool) -> MlflowGateway:
        return RecordingGateway(
            client,
            events=events,
            fail_retire_trace_ids=fail_retire_trace_ids,
            dry_run=dry_run,
        )

    return factory


# ---------------------------------------------------------------------------
# Read-only three-axis server state
# ---------------------------------------------------------------------------

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


def capture_server_state(client: MlflowClient) -> _ServerState:
    """Read-only snapshot of every experiment, trace, and trace artifact tree.

    Flushes the client-side async export queue once up front (a
    client-side drain, never a server mutation) so locally pending traces
    are visible, then collects the state with span-free searches and
    artifact listings.
    """
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
        artifacts[trace_id] = tuple(_list_repo_tree(repo, None, 4))
    return experiments, traces, artifacts


# ---------------------------------------------------------------------------
# Source variants and shared state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Baseline:
    """The module's shared imported state and its rerun helper."""

    run: RealImportRun
    client: MlflowClient
    experiment_id: str
    source: Path
    tmp_root: Path
    trace_ids_by_key: dict[str, str]
    turn_keys_by_index: dict[int, str] = field(default_factory=dict)

    def rerun(
        self,
        *,
        source: Path | None = None,
        events: list[GatewayEvent] | None = None,
        fail_retire_trace_ids: frozenset[str] = frozenset(),
    ) -> RealImportRun:
        """Rerun the pipeline over a (possibly mutated) copy of the source."""
        gateway_factory = None
        if events is not None or fail_retire_trace_ids:
            gateway_factory = _recording_gateway_factory(
                self.client, events if events is not None else [], fail_retire_trace_ids
            )
        return run_real_import(
            server_uri=self.client.tracking_uri,
            source=source if source is not None else self.source,
            experiment_name=self.run.experiment_name,
            tmp_root=self.tmp_root,
            gateway_factory=gateway_factory,
        )

    def owned_trace_ids(self) -> set[str]:
        """Every importer-owned trace id currently in the experiment."""
        page = self.client.search_traces(
            locations=[self.experiment_id], max_results=100, include_spans=False, flush=True
        )
        return {
            trace.trace_id for trace in page if trace.info.trace_metadata.get("opencode.importer") == "opencode-mlflow"
        }


def _variant_source(baseline: Baseline, name: str) -> Path:
    """A fresh copy of the baseline source database to mutate."""
    variant = baseline.tmp_root / name
    shutil.copyfile(baseline.source, variant)
    return variant


def _changed_reply_source(baseline: Baseline, message_id: str, old_text: str, new_text: str) -> Path:
    """A source copy whose assistant reply ``message_id`` carries new text."""
    variant = _variant_source(baseline, f"changed-{message_id}.db")
    connection = sqlite3.connect(variant)
    try:
        row = connection.execute("SELECT data FROM session_message WHERE id = ?", (message_id,)).fetchone()
        assert row is not None, message_id
        assert old_text in row[0], f"the fixture reply {message_id} must contain {old_text!r}"
        payload = row[0].replace(old_text, new_text)
        connection.execute("UPDATE session_message SET data = ? WHERE id = ?", (payload, message_id))
        connection.commit()
    finally:
        connection.close()
    return variant


def _dropped_turn_source(source: Path, tmp_root: Path, message_ids: tuple[str, ...]) -> Path:
    """A source copy whose turn rows ``message_ids`` were removed."""
    variant = tmp_root / "dropped-turn.db"
    shutil.copyfile(source, variant)
    connection = sqlite3.connect(variant)
    try:
        for message_id in message_ids:
            deleted = connection.execute("DELETE FROM session_message WHERE id = ?", (message_id,)).rowcount
            assert deleted == 1, message_id
        connection.commit()
    finally:
        connection.close()
    return variant


def _seed_foreign_trace(client: MlflowClient, experiment_id: str) -> str:
    """Seed one non-importer trace in the experiment through the public client."""
    span = mlflow.start_span_no_context(
        name="foreign-non-importer-trace",
        span_type="CHAIN",
        inputs={"origin": "foreign-seed"},
        experiment_id=experiment_id,
    )
    span.end(outputs={"foreign": True}, status="OK")
    return span.trace_id


def _trace_exists(client: MlflowClient, trace_id: str) -> bool:
    try:
        client.get_trace(trace_id, display=False, flush=True)
    except Exception:
        return False
    return True


def _assert_result_complete(run: RealImportRun) -> ImportResult:
    assert run.exit_status in (0, 1), run.report["errors"]
    assert run.report["errors"] == [], run.report["errors"]
    return run.result


@pytest.fixture(scope="module")
def baseline(live_server: LiveServer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Baseline]:
    """One complete real-pipeline import of the six-turn session."""
    tmp_root = tmp_path_factory.mktemp("reconciliation-")
    source = build_six_turn_source_db(tmp_root)
    name = unique_experiment_name("reconciliation")
    run = run_real_import(
        server_uri=live_server.uri,
        source=source,
        experiment_name=name,
        tmp_root=tmp_root,
    )
    assert run.exit_status == 0, run.report["errors"]
    experiment = live_server.client.get_experiment_by_name(name)
    assert experiment is not None
    ids = trace_ids_by_key(run.report)
    keys_by_index = {int(turn["turn_index"]): str(turn["turn_key"]) for turn in turn_entries(run.report)}
    yield Baseline(
        run=run,
        client=live_server.client,
        experiment_id=experiment.experiment_id,
        source=source,
        tmp_root=tmp_root,
        trace_ids_by_key=ids,
        turn_keys_by_index=keys_by_index,
    )


# ---------------------------------------------------------------------------
# Dry run: no experiment, no trace, no artifact
# ---------------------------------------------------------------------------


def test_dry_run_creates_no_experiment_traces_or_artifacts(live_server: LiveServer, tmp_path: Path) -> None:
    tmp_root = tmp_path / "dry-run"
    tmp_root.mkdir()
    source = build_six_turn_source_db(tmp_root)
    name = unique_experiment_name("reconciliation-dry-run")

    before = capture_server_state(live_server.client)
    run = run_real_import(
        server_uri=live_server.uri,
        source=source,
        experiment_name=name,
        tmp_root=tmp_root,
        dry_run=True,
    )
    _assert_result_complete(run)

    # The report plans (never performs) the creation and every import.
    assert run.report["dry_run"] is True
    experiment = run.report["experiment"]
    assert isinstance(experiment, Mapping)
    assert experiment["status"] == "planned"
    assert experiment["id"] is None
    summary = run.report["summary"]
    assert isinstance(summary, Mapping)
    assert summary["planned"] == {"imported": _TURNS, "updated": 0, "retired": 0}
    for turn in turn_entries(run.report):
        assert turn["action"] == "skipped"
        assert turn["reason"] == "dry_run"
        assert turn["planned_action"] == "imported"
        assert turn["trace_id"] is None
    assert live_server.client.get_experiment_by_name(name) is None

    # Three-axis state equality: no experiment, trace, or artifact changed.
    after = capture_server_state(live_server.client)
    assert after[0] == before[0], "the dry run changed the experiment set"
    assert after[1] == before[1], "the dry run changed the trace set"
    assert after[2] == before[2], "the dry run changed the artifact trees"


# ---------------------------------------------------------------------------
# Unchanged rerun and recorded-digest verification (Ruling 16)
# ---------------------------------------------------------------------------


def test_unchanged_rerun_creates_no_duplicates_and_tampered_digest_replaces(baseline: Baseline) -> None:
    before_ids = baseline.owned_trace_ids()
    assert len(before_ids) == _TURNS

    rerun = baseline.rerun()
    _assert_result_complete(rerun)
    assert rerun.exit_status == 0

    summary = rerun.report["summary"]
    assert isinstance(summary, Mapping)
    assert summary["turns_skipped"] == _TURNS
    assert summary["turns_imported"] == 0
    assert summary["turns_updated"] == 0
    assert summary["turns_failed"] == 0
    for turn in turn_entries(rerun.report):
        assert turn["action"] == "skipped"
        assert turn["reason"] == "unchanged"
        assert turn["trace_id"] == baseline.trace_ids_by_key[str(turn["turn_key"])]
    assert baseline.owned_trace_ids() == before_ids, "the unchanged rerun must not create or retire anything"

    # Ruling 16: the recorded digest is really checked against the stored
    # bytes on the live server. Tamper ONE trace's recorded sha256 tag
    # through the public client; the next run must replace exactly that
    # turn (its stored artifact no longer matches its recorded digest)
    # while every sibling verifies unchanged.
    tampered_key = baseline.turn_keys_by_index[6]
    tampered_id = baseline.trace_ids_by_key[tampered_key]
    baseline.client.set_trace_tag(tampered_id, ARTIFACT_SHA256_TAG, "0" * 64)

    replaced = baseline.rerun()
    _assert_result_complete(replaced)
    assert replaced.exit_status == 0

    summary = replaced.report["summary"]
    assert isinstance(summary, Mapping)
    assert summary["turns_updated"] == 1
    assert summary["turns_skipped"] == _TURNS - 1
    assert summary["turns_failed"] == 0
    turns = {str(turn["turn_key"]): turn for turn in turn_entries(replaced.report)}
    tampered_turn = turns[tampered_key]
    assert tampered_turn["action"] == "updated"
    new_id = tampered_turn["trace_id"]
    assert isinstance(new_id, str) and new_id != tampered_id
    for key, turn in turns.items():
        if key == tampered_key:
            continue
        assert turn["action"] == "skipped", (key, turn)
        assert turn["reason"] == "unchanged"
        assert turn["trace_id"] == baseline.trace_ids_by_key[key]

    # Observable server state: the tampered trace is retired, the
    # replacement exists with a re-recorded consistent digest, and the
    # owned population is still exactly six traces.
    assert not _trace_exists(baseline.client, tampered_id)
    replacement = baseline.client.get_trace(new_id, display=False, flush=True)
    recorded_sha = replacement.info.tags[ARTIFACT_SHA256_TAG]
    recorded_size = int(replacement.info.tags[ARTIFACT_SIZE_TAG])
    recorded_path = replacement.info.tags[ARTIFACT_PATH_TAG]
    assert len(recorded_sha) == 64 and recorded_sha != "0" * 64
    with TemporaryDirectory(prefix="opencode-mlflow-recon-") as dst:
        repo = get_artifact_repository(
            replacement.info.tags[MLFLOW_ARTIFACT_LOCATION], tracking_uri=baseline.client.tracking_uri
        )
        downloaded = Path(repo.download_artifacts(recorded_path, dst))
        stored = downloaded.read_bytes()
    assert len(stored) == recorded_size
    assert hashlib.sha256(stored).hexdigest() == recorded_sha
    assert len(baseline.owned_trace_ids()) == _TURNS

    # Keep the module state consistent for later scenarios.
    baseline.trace_ids_by_key[tampered_key] = new_id


# ---------------------------------------------------------------------------
# Changed-turn replacement order
# ---------------------------------------------------------------------------


def test_changed_turn_replacement_follows_create_verify_upload_verify_retire_order(
    baseline: Baseline,
) -> None:
    changed_key = baseline.turn_keys_by_index[2]
    old_id = baseline.trace_ids_by_key[changed_key]
    old_trace = baseline.client.get_trace(old_id, display=False, flush=True)
    old_path = old_trace.info.tags[ARTIFACT_PATH_TAG]
    old_output = [span for span in old_trace.data.spans if span.parent_id is None][0].outputs

    variant = _changed_reply_source(
        baseline, "msg-synthetic-a2", "synthetic final part one", "synthetic changed part one"
    )
    events: list[GatewayEvent] = []
    run = baseline.rerun(source=variant, events=events)
    _assert_result_complete(run)
    assert run.exit_status == 0

    turns = {str(turn["turn_key"]): turn for turn in turn_entries(run.report)}
    changed_turn = turns[changed_key]
    assert changed_turn["action"] == "updated"
    new_id = changed_turn["trace_id"]
    assert isinstance(new_id, str) and new_id != old_id
    for key, turn in turns.items():
        if key == changed_key:
            continue
        assert turn["action"] == "skipped", (key, turn)
        assert turn["trace_id"] == baseline.trace_ids_by_key[key]

    # The write protocol's observable order at the gateway boundary, scoped
    # to the replacement's new trace id: create → verify → upload → verify
    # → record the digest, and only THEN retire the old trace.
    new_trace_events = [event for event in events if event.trace_id == new_id]
    methods = [event.method for event in new_trace_events]
    assert methods[:5] == [
        "create_trace",
        "verify_trace",
        "upload_named_json",
        "verify_named_json",
        "record_artifact_verification",
    ]
    old_retire_index = next(
        index for index, event in enumerate(events) if event.method == "retire_trace" and event.trace_id == old_id
    )
    record_index = next(index for index, event in enumerate(events) if event.method == "record_artifact_verification")
    assert old_retire_index > record_index, "the old trace must be retired only after the replacement is verified"
    assert not any(event.method == "retire_trace" and event.trace_id == new_id for event in events)

    # Observable server state: the old trace is gone, the replacement
    # carries the changed content and a NEW deterministic artifact path.
    assert not _trace_exists(baseline.client, old_id)
    replacement = baseline.client.get_trace(new_id, display=False, flush=True)
    root = [span for span in replacement.data.spans if span.parent_id is None][0]
    assert root.outputs != old_output
    assert "synthetic changed part one" in str(root.outputs)
    new_path = replacement.info.tags[ARTIFACT_PATH_TAG]
    assert new_path != old_path
    assert new_path.startswith("opencode-mlflow/v1/sessions/ses-synthetic-1/turns/")

    baseline.trace_ids_by_key[changed_key] = new_id


# ---------------------------------------------------------------------------
# Retirement failure preserves the old trace
# ---------------------------------------------------------------------------


def test_retirement_failure_after_verified_replacement_preserves_the_old_trace(
    baseline: Baseline,
) -> None:
    changed_key = baseline.turn_keys_by_index[4]
    old_id = baseline.trace_ids_by_key[changed_key]
    old_trace = baseline.client.get_trace(old_id, display=False, flush=True)
    old_root = [span for span in old_trace.data.spans if span.parent_id is None][0]

    variant = _changed_reply_source(
        baseline, "msg-synthetic-a4", "synthetic partial answer four", "synthetic changed partial four"
    )
    events: list[GatewayEvent] = []
    run = baseline.rerun(
        source=variant,
        events=events,
        fail_retire_trace_ids=frozenset({old_id}),
    )
    _assert_result_complete(run)

    # The item fails loudly with the retirement-failure code and the run
    # exits 1; the compensating rollback retired the NEW trace.
    turns = {str(turn["turn_key"]): turn for turn in turn_entries(run.report)}
    failed_turn = turns[changed_key]
    assert failed_turn["action"] == "failed"
    assert failed_turn["error_code"] == "retirement_failed"
    new_id = failed_turn["trace_id"]
    assert isinstance(new_id, str) and new_id != old_id
    assert run.exit_status == 1
    assert any(event.method == "retire_trace_failed" and event.trace_id == old_id for event in events), (
        "the retirement failure must have been injected at the boundary"
    )

    # The OLD trace is preserved and fully retrievable with its original
    # content and recorded artifact; the new trace was rolled back.
    assert _trace_exists(baseline.client, old_id)
    preserved = baseline.client.get_trace(old_id, display=False, flush=True)
    preserved_root = [span for span in preserved.data.spans if span.parent_id is None][0]
    assert preserved_root.outputs == old_root.outputs
    assert preserved.info.tags[ARTIFACT_PATH_TAG] == old_trace.info.tags[ARTIFACT_PATH_TAG]
    assert not _trace_exists(baseline.client, new_id)
    # ... and the run left the owned population at exactly six traces.
    assert len(baseline.owned_trace_ids()) == _TURNS


# ---------------------------------------------------------------------------
# Stale-turn retirement is scoped; foreign traces stay untouched
# ---------------------------------------------------------------------------


def test_stale_retirement_is_scoped_and_foreign_traces_remain_untouched(
    live_server: LiveServer, tmp_path: Path
) -> None:
    # Self-contained state: this scenario retires a turn permanently, so it
    # owns its private baseline instead of sharing the module's, keeping
    # every scenario in this module order-independent.
    tmp_root = tmp_path / "stale-retirement"
    tmp_root.mkdir()
    source = build_six_turn_source_db(tmp_root)
    name = unique_experiment_name("reconciliation-stale")
    first = run_real_import(
        server_uri=live_server.uri,
        source=source,
        experiment_name=name,
        tmp_root=tmp_root,
    )
    _assert_result_complete(first)
    assert first.exit_status == 0
    client = live_server.client
    experiment = client.get_experiment_by_name(name)
    assert experiment is not None
    ids_by_key = trace_ids_by_key(first.report)
    keys_by_index = {int(turn["turn_index"]): str(turn["turn_key"]) for turn in turn_entries(first.report)}

    foreign_id = _seed_foreign_trace(client, experiment.experiment_id)
    foreign_before = client.get_trace(foreign_id, display=False, flush=True)

    dropped_key = keys_by_index[5]
    dropped_id = ids_by_key[dropped_key]
    surviving_before = {key: trace_id for key, trace_id in ids_by_key.items() if key != dropped_key}

    variant = _dropped_turn_source(source, tmp_root, ("msg-synthetic-u5", "msg-synthetic-a5"))
    run = run_real_import(
        server_uri=live_server.uri,
        source=variant,
        experiment_name=name,
        tmp_root=tmp_root,
    )
    _assert_result_complete(run)
    assert run.exit_status == 0

    # The dropped turn's owned trace is retired and reported with
    # source_present=False; every surviving sibling keeps its trace id.
    turns = {str(turn["turn_key"]): turn for turn in turn_entries(run.report) if turn["source_present"] is True}
    assert dropped_key not in turns
    retirements = [
        turn for turn in turn_entries(run.report) if turn["source_present"] is False and turn["action"] == "retired"
    ]
    assert [turn["trace_id"] for turn in retirements] == [dropped_id]
    assert [turn["reason"] for turn in retirements] == ["stale_turn"]
    for key, turn in turns.items():
        assert turn["action"] == "skipped", (key, turn)
        assert turn["reason"] == "unchanged"
        assert turn["trace_id"] == surviving_before[key]

    # Observable server state: exactly the dropped trace is gone; the
    # foreign, non-importer trace is untouched (same content and tags).
    assert not _trace_exists(client, dropped_id)
    foreign_after = client.get_trace(foreign_id, display=False, flush=True)
    assert foreign_after.info.tags == foreign_before.info.tags
    assert foreign_after.info.trace_metadata == foreign_before.info.trace_metadata
    page = client.search_traces(locations=[experiment.experiment_id], max_results=100, include_spans=False, flush=True)
    owned = {
        trace.trace_id for trace in page if trace.info.trace_metadata.get("opencode.importer") == "opencode-mlflow"
    }
    assert dropped_id not in owned
    assert foreign_id not in owned
    assert len(owned) == _TURNS - 1
