"""A checksum-valid recovery artifact is not deletion authority."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from mlflow_import.compatibility import identity_projection
from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.attempts import AttemptJournal
from mlflow_import.mlflow.gateway import MlflowGateway
from mlflow_import.mlflow.reconciliation import _finish_ready

pytestmark = pytest.mark.unit


class Target:
    def __init__(self):
        self.files = {}
        self.deleted = []
        self._artifact_checks = {}
        self.info = SimpleNamespace(experiment_id="exp", trace_metadata={"mlflow.trace.session": "unrelated"}, tags={})
        self._client = SimpleNamespace(get_trace=lambda *args, **kwargs: SimpleNamespace(info=self.info))

    def experiment_repository(self, experiment_id):
        return self

    def publish_experiment_file(self, experiment_id, path, local):
        self.files[path] = Path(local).read_bytes()

    def download_artifacts(self, path, root):
        local = Path(root) / PurePosixPath(path).name
        local.write_bytes(self.files[path])
        return str(local)

    def list_artifacts(self, prefix):
        entries = {}
        for path in self.files:
            if path.startswith(prefix + "/"):
                rest = path[len(prefix) + 1 :]
                first = rest.split("/")[0]
                entries[first] = SimpleNamespace(path=prefix + "/" + first, is_dir="/" in rest)
        return list(entries.values())

    def verify_experiment_file(self, experiment, path, digest, size):
        payload = self.files[path]
        assert len(payload) == size and hashlib.sha256(payload).hexdigest() == digest

    def trace_exists(self, trace_id):
        return True

    def authorize_retirement(self, experiment, retirement, ready):
        MlflowGateway.authorize_retirement(self, experiment, retirement, ready)

    def retire_conflict(self, experiment, retirement):
        self.deleted.append(retirement["trace_id"])


def _ready(tmp_path, *, session_turns=(), proof=None):
    target = Target()
    native = "selected"
    base = f"mlflow-import/v2/opencode/{'a' * 64}/{hashlib.sha256(native.encode()).hexdigest()}"
    journal = AttemptJournal(target, "exp", base)
    qualified = journal.qualified_id(native)
    if proof == "in-scope":
        proof = identity_projection({"mlflow.trace.session": qualified}, {})

    def publish(kind, value):
        payload = json.dumps(value, separators=(",", ":")).encode()
        digest = hashlib.sha256(payload).hexdigest()
        path = f"{base}/verbatim/{kind}/{digest}.json"
        target.files[path] = payload
        return [path, digest, len(payload)]

    session = publish(
        "sessions",
        {
            "schema_version": "mlflow-import.session.v1",
            "session": {"session_id": native, "source": "opencode"},
            "privacy": "verbatim",
            "turns": [{"turn_key": turn} for turn in session_turns],
        },
    )
    prepared = journal.publish(
        "prepared",
        {
            "attempt_id": "b" * 32,
            "session_id": native,
            "qualified_id": qualified,
            "desired": "c" * 64,
            "privacy": "verbatim",
            "policy": "overwrite",
            "instance_id": "",
            "expected_turns": [],
            "prior_ids": ["unrelated-trace"],
            "session_object": session,
        },
    )
    manifest = publish(
        "manifests",
        {
            "schema_version": "mlflow-import.session-manifest.v1",
            "session_id": qualified,
            "session": session,
            "privacy": "verbatim",
            "instance_id": "",
            "turns": [],
        },
    )
    ready = journal.publish(
        "ready_to_retire",
        {
            **prepared,
            "manifest": manifest,
            "replacements": [],
            "retirements": [
                {
                    "trace_id": "unrelated-trace",
                    "session_id": native,
                    "turn_key": "",
                    "foreign": True,
                    "identity": proof if proof is not None else identity_projection(target.info.trace_metadata, {}),
                }
            ],
        },
    )
    return journal, target, ready


@pytest.mark.parametrize("forged_proof", [False, True])
def test_checksum_valid_empty_ready_cannot_delete_unrelated_trace(tmp_path, forged_proof):
    journal, target, ready = _ready(tmp_path, proof="in-scope" if forged_proof else None)
    with pytest.raises(ItemImportError):
        _finish_ready(journal, ready, target)
    assert target.deleted == []


def test_empty_ready_cannot_hide_nonempty_source_turn_inventory(tmp_path):
    journal, target, ready = _ready(tmp_path, session_turns=("turn-existing",))
    with pytest.raises(ItemImportError) as caught:
        _finish_ready(journal, ready, target)
    assert caught.value.code == "attempt_state_invalid"
    assert target.deleted == []


@pytest.mark.parametrize("field,value", [("session_id", "other"), ("instance_id", "d" * 32), ("desired", "e" * 64)])
def test_transition_chain_cannot_change_prepared_scope(tmp_path, field, value):
    journal, target, ready = _ready(tmp_path)
    original = next(journal.attempts())["prepared"]
    changed = {**ready, field: value}
    with pytest.raises(ItemImportError):
        journal.validate_chain({"prepared": original, "ready_to_retire": changed})
    assert target.deleted == []


def test_state_reference_envelope_is_checked_before_materializing_id_sets(tmp_path, monkeypatch):
    from mlflow_import.mlflow import attempts

    journal, _, _ = _ready(tmp_path)
    prepared = next(journal.attempts())["prepared"]
    monkeypatch.setattr(attempts, "MAX_STATE_REFERENCES", 2)
    with pytest.raises(ItemImportError) as caught:
        journal.validate({**prepared, "expected_turns": ["one", "two", "three"]})
    assert caught.value.code == "attempt_state_too_large"


@pytest.mark.parametrize("area", ["session", "summary"])
def test_session_verifier_rejects_oversized_metadata_before_decoding(tmp_path, area):
    journal, _, ready = _ready(tmp_path)
    value = {
        "schema_version": "mlflow-import.session.v1",
        "privacy": "verbatim",
        "session": {"session_id": "selected", "source": "opencode"},
        "turns": [],
    }
    if area == "session":
        value["session"]["padding"] = "x" * (8_388_608 + 1)
    else:
        value["turns"] = [{"turn_key": "turn-existing", "padding": "x" * (8_388_608 + 1)}]
        ready["expected_turns"] = ["turn-existing"]
    path = tmp_path / "session.json"
    path.write_text(json.dumps(value, separators=(",", ":")))
    ready["session_object"][2] = path.stat().st_size
    with pytest.raises(ItemImportError) as caught:
        journal._verify_session(path, ready)
    assert caught.value.code == "attempt_state_invalid"


def test_large_records_stream_past_independent_metadata_limits(tmp_path, monkeypatch):
    journal, _, ready = _ready(tmp_path)
    value = {
        "schema_version": "mlflow-import.session.v1",
        "privacy": "verbatim",
        "records": [{"text": "x" * (16 * 1024 * 1024)}],
        "session": {"session_id": "selected", "source": "opencode"},
        "turns": [],
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(value, separators=(",", ":")))
    ready["session_object"][2] = path.stat().st_size
    loads = json.loads

    def bounded_loads(raw, *args, **kwargs):
        assert len(raw) <= 8 * 1024 * 1024
        return loads(raw, *args, **kwargs)

    monkeypatch.setattr(json, "loads", bounded_loads)
    journal._verify_session(path, ready)
