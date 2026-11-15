"""Checksummed immutable attempt states in the experiment artifact repository."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from mlflow_import import resources
from mlflow_import.content.canonical import iter_canonical_json
from mlflow_import.errors import ItemImportError

if TYPE_CHECKING:
    from mlflow_import.mlflow.gateway import MlflowGateway

STATES = frozenset({"prepared", "ready_to_retire", "complete", "aborted"})
MAX_STATE_BYTES = 8 * 1024 * 1024
MAX_STATE_REFERENCES = 65_536


def _invalid() -> ItemImportError:
    return ItemImportError("attempt_state_invalid", "the recovery state is invalid or outside its authorized scope")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _invalid()
        result[key] = value
    return result


def write_state_json(path: Path, value: object) -> tuple[str, int]:
    """Finite metadata envelope; never assemble a session-sized byte blob."""
    digest, size = hashlib.sha256(), 0
    with resources.open_file(path, "wb") as handle:
        for chunk in iter_canonical_json(value):
            size += len(chunk)
            if size > MAX_STATE_BYTES:
                raise ItemImportError("attempt_state_too_large", "the attempt exceeds the qualified state envelope")
            handle.write(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


class AttemptJournal:
    """One source/instance scope. Listing and reads never mutate the target."""

    def __init__(self, gateway: MlflowGateway, experiment_id: str, prefix: str) -> None:
        self.gateway = gateway
        self.experiment_id = experiment_id
        self.prefix = prefix + "/attempts"

    def qualified_id(self, session_id: str) -> str:
        parts = self.prefix.split("/")
        if (
            len(parts) != 6
            or parts[:2] != ["mlflow-import", "v2"]
            or parts[2] not in {"opencode", "claude", "codex", "pi", "gemini"}
            or not re.fullmatch(r"[a-f0-9]{64}", parts[3])
            or hashlib.sha256(session_id.encode()).hexdigest() != parts[4]
        ):
            raise _invalid()
        return json.dumps(["v1", parts[2], parts[3], session_id], separators=(",", ":"))

    def validate(self, value: Mapping[str, Any], path: str | None = None) -> None:
        """A checksum is integrity only; bind every typed field to this journal."""
        if (
            value.get("schema_version") != "mlflow-import.attempt.v1"
            or not isinstance(value.get("state"), str)
            or value["state"] not in STATES
        ):
            raise _invalid()
        for key in ("attempt_id", "session_id", "desired", "instance_id", "qualified_id", "privacy", "policy"):
            if not isinstance(value.get(key), str):
                raise _invalid()
        if (
            not re.fullmatch(r"[a-f0-9]{32}", value["attempt_id"])
            or not re.fullmatch(r"[a-f0-9]{64}", value["desired"])
            or not value["session_id"]
            or value["qualified_id"] != self.qualified_id(value["session_id"])
            or value["privacy"] not in {"redacted", "verbatim"}
            or value["policy"] not in {"overwrite", "ignore", "duplicate"}
            or (value["instance_id"] and not re.fullmatch(r"[a-f0-9]{32}", value["instance_id"]))
            or (value["policy"] == "duplicate" and not value["instance_id"])
        ):
            raise _invalid()
        if path is not None:
            expected = f"{self.prefix}/{value['attempt_id']}/{value['state']}-"
            if (
                not path.startswith(expected)
                or PurePosixPath(path).parent.as_posix() != f"{self.prefix}/{value['attempt_id']}"
            ):
                raise _invalid()
        base = self.prefix.removesuffix("/attempts") + "/" + value["privacy"]

        def reference(item: Any, kind: str) -> None:
            if (
                not isinstance(item, list)
                or len(item) != 3
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)
                or not re.fullmatch(r"[a-f0-9]{64}", item[1])
                or type(item[2]) is not int
                or item[2] < 0
                or not item[0].startswith(base + "/" + kind + "/")
                or ".." in PurePosixPath(item[0]).parts
            ):
                raise _invalid()
            if kind != "turns" and item[0] != f"{base}/{kind}/{item[1]}.json":
                raise _invalid()

        prior = value.get("prior_ids")
        turns = value.get("expected_turns")
        if (isinstance(prior, list) and len(prior) > MAX_STATE_REFERENCES) or (
            isinstance(turns, list) and len(turns) > MAX_STATE_REFERENCES
        ):
            raise ItemImportError("attempt_state_too_large", "the attempt exceeds the supported metadata envelope")
        if (
            not isinstance(prior, list)
            or not all(isinstance(item, str) and item for item in prior)
            or len(set(prior)) != len(prior)
            or not isinstance(turns, list)
            or not all(isinstance(item, str) and item for item in turns)
            or len(set(turns)) != len(turns)
            or not isinstance(value.get("pricing_bases", {}), Mapping)
        ):
            raise _invalid()
        reference(value.get("session_object"), "sessions")
        prior_set, turn_set = set(prior), set(turns)
        bases = value.get("pricing_bases", {})
        if any(
            not isinstance(key, str) or key not in turn_set or not isinstance(item, Mapping)
            for key, item in bases.items()
        ):
            raise _invalid()
        if value["state"] not in {"ready_to_retire", "complete"}:
            return
        reference(value.get("manifest"), "manifests")
        replacements, retirements = value.get("replacements"), value.get("retirements")
        if not isinstance(replacements, list) or not isinstance(retirements, list):
            raise _invalid()
        found: set[str] = set()
        ids: set[str] = set()
        for replacement in replacements:
            if not isinstance(replacement, dict) or not isinstance(replacement.get("metadata"), dict):
                raise _invalid()
            metadata = replacement["metadata"]
            if not all(isinstance(key, str) and isinstance(item, str) for key, item in metadata.items()):
                raise _invalid()
            from mlflow_import.mlflow.gateway import OwnedTrace
            from mlflow_import.mlflow.reconciliation import OwnedTraceCandidate

            candidate = OwnedTraceCandidate.from_owned_trace(
                OwnedTrace(str(replacement.get("trace_id", "")), 0, {}, metadata), self.experiment_id
            )
            if (
                not candidate.ownership_complete
                or metadata.get("mlflow_import.harness") != self.prefix.split("/")[2]
                or metadata.get("mlflow_import.source_namespace") != self.prefix.split("/")[3]
                or replacement.get("experiment_id") != self.experiment_id
                or metadata.get("mlflow_import.qualified_session_id", metadata.get("mlflow.trace.session"))
                != value["qualified_id"]
                or metadata.get("mlflow.trace.session") not in {value["qualified_id"], value["session_id"]}
                or metadata.get("mlflow_import.session_id") != value["session_id"]
                or metadata.get("mlflow_import.import_instance_id", "") != value["instance_id"]
                or not isinstance(replacement.get("trace_id"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", str(replacement.get("payload_sha256", "")))
                or not isinstance(replacement.get("native"), dict)
                or replacement.get("schema_version", "mlflow-import.replacement.v1")
                not in {"mlflow-import.replacement.v1", "mlflow-import.replacement.v2"}
                or not all(isinstance(key, str) and isinstance(item, str) for key, item in metadata.items())
            ):
                raise _invalid()
            turn = metadata.get("mlflow_import.turn_key")
            if turn not in turn_set or turn in found or replacement["trace_id"] in ids:
                raise _invalid()
            found.add(turn)
            ids.add(replacement["trace_id"])
            reference(replacement.get("map"), "turns")
            reference(replacement.get("session"), "sessions")
            if replacement["session"] != value["session_object"]:
                raise _invalid()
        if found != turn_set:
            raise _invalid()
        deleted: set[str] = set()
        for retirement in retirements:
            if (
                not isinstance(retirement, dict)
                or not isinstance(retirement.get("trace_id"), str)
                or retirement["trace_id"] not in prior_set
                or retirement["trace_id"] in ids
                or retirement["trace_id"] in deleted
                or retirement.get("session_id") != value["session_id"]
                or not isinstance(retirement.get("turn_key"), str)
                or not isinstance(retirement.get("identity"), dict)
                or type(retirement.get("foreign", False)) is not bool
                or any(item is not None and not isinstance(item, str) for item in retirement["identity"].values())
                or retirement.get("qualified_id", value["qualified_id"]) != value["qualified_id"]
            ):
                raise _invalid()
            deleted.add(retirement["trace_id"])

    def validate_chain(self, states: Mapping[str, dict[str, Any]]) -> None:
        if "prepared" not in states or ("aborted" in states and ("complete" in states or "ready_to_retire" in states)):
            raise _invalid()
        prepared = states["prepared"]
        for name, state in states.items():
            self.validate(state)
            if state["state"] != name:
                raise _invalid()
            if any(state.get(key) != item for key, item in prepared.items() if key != "state"):
                raise _invalid()
            if name == "complete":
                ready = states.get("ready_to_retire")
                if ready is None or any(state.get(key) != item for key, item in ready.items() if key != "state"):
                    raise _invalid()

    def verify_ready(self, ready: Mapping[str, Any]) -> None:
        """Verify the persisted chain, join manifest and streamed source turn inventory."""
        self.validate(ready)
        chain = next(
            (states for states in self.attempts() if states["prepared"]["attempt_id"] == ready["attempt_id"]), None
        )
        if chain is None or chain.get("ready_to_retire") != ready:
            raise _invalid()
        repository = self.gateway.experiment_repository(self.experiment_id)
        for name in ("manifest", "session_object"):
            path, checksum, size = ready[name]
            self.gateway.verify_experiment_file(self.experiment_id, path, checksum, size)
        with resources.temporary_directory(prefix="mlflow-import-ready-") as root:
            path = self._checked_download(repository, ready["manifest"], root)
            if path.stat().st_size > MAX_STATE_BYTES:
                raise _invalid()
            with resources.open_reader(path) as handle:
                manifest = json.load(handle, object_pairs_hook=_unique_object)
            expected = [
                {"turn_key": ref["metadata"]["mlflow_import.turn_key"], "trace_id": ref["trace_id"], "map": ref["map"]}
                for ref in ready["replacements"]
            ]
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema_version") != "mlflow-import.session-manifest.v1"
                or manifest.get("session_id") != ready["qualified_id"]
                or manifest.get("session") != ready["session_object"]
                or manifest.get("privacy") != ready["privacy"]
                or manifest.get("instance_id", "") != ready["instance_id"]
                or manifest.get("turns") != expected
            ):
                raise _invalid()
            path = self._checked_download(repository, ready["session_object"], root)
            self._verify_session(path, ready)

    @staticmethod
    def _checked_download(repository: Any, reference: list[Any], root: str) -> Path:
        path = resources.copy_artifact(repository, reference[0], root, reference[2])
        if not path.resolve().is_relative_to(Path(root).resolve()):
            raise _invalid()
        digest, size = hashlib.sha256(), 0
        with resources.open_reader(path) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                if size > reference[2]:
                    raise _invalid()
        if size != reference[2] or digest.hexdigest() != reference[1]:
            raise _invalid()
        return path

    def _verify_session(self, path: Path, ready: Mapping[str, Any]) -> None:
        from mlflow_import.source.file_headers import _HeaderReader
        from mlflow_import.storage import DiskMap

        turns: DiskMap[str, bool] = DiskMap((key, False) for key in ready["expected_turns"])
        facts: dict[str, Any] = {}
        seen: set[str] = set()
        with resources.open_reader(path, "r", encoding="utf-8") as handle:
            reader = _HeaderReader(handle, ready["session_object"][2])

            def metadata(limit: int = MAX_STATE_BYTES) -> Any:
                try:
                    return reader.bounded_value(limit)
                except (ValueError, RecursionError):
                    raise _invalid() from None

            if reader.take() != "{":
                raise _invalid()
            while True:
                key = metadata(1024)
                if not isinstance(key, str) or reader.take() != ":":
                    raise _invalid()
                if key in {
                    "schema_version",
                    "session",
                    "privacy",
                    "turns",
                    "records",
                    "source_fingerprint",
                    "projections",
                }:
                    if key in seen:
                        raise _invalid()
                    seen.add(key)
                if key == "turns":
                    if reader.take() != "[":
                        raise _invalid()
                    while reader.peek() != "]":
                        item = metadata()
                        turn = item.get("turn_key") if isinstance(item, dict) else None
                        if not isinstance(turn, str) or turn not in turns or turns[turn]:
                            raise _invalid()
                        turns[turn] = True
                        if reader.peek() != ",":
                            break
                        reader.take()
                    if reader.take() != "]":
                        raise _invalid()
                elif key in {"schema_version", "session", "privacy"}:
                    facts[key] = metadata()
                else:
                    reader.value(False)
                delimiter = reader.take()
                if delimiter == "}":
                    break
                if delimiter != ",":
                    raise _invalid()
        if (
            facts.get("schema_version") != "mlflow-import.session.v1"
            or facts.get("privacy") != ready["privacy"]
            or not isinstance(facts.get("session"), dict)
            or facts["session"].get("session_id") != ready["session_id"]
            or facts["session"].get("source") != self.prefix.split("/")[2]
            or "turns" not in seen
            or not all(turns.values())
        ):
            raise _invalid()

    def read(self, path: str) -> dict[str, Any]:
        if (
            re.fullmatch(
                re.escape(self.prefix)
                + r"/[a-f0-9]{32}/(?:prepared|ready_to_retire|complete|aborted)-[a-f0-9]{64}\.json",
                path,
            )
            is None
        ):
            raise _invalid()
        repository = self.gateway.experiment_repository(self.experiment_id)
        with resources.temporary_directory(prefix="mlflow-import-state-") as root:
            downloaded = resources.copy_artifact(repository, path, root, MAX_STATE_BYTES)
            with resources.open_reader(downloaded) as handle:
                payload = handle.read(MAX_STATE_BYTES + 1)
        if len(payload) > MAX_STATE_BYTES or hashlib.sha256(payload).hexdigest() != Path(path).stem.split("-")[-1]:
            raise ItemImportError("attempt_state_invalid", "an attempt state failed checksum or size validation")
        try:
            value = json.loads(payload, object_pairs_hook=_unique_object)
        except (ValueError, RecursionError):
            raise _invalid() from None
        if not isinstance(value, dict) or value.get("schema_version") != "mlflow-import.attempt.v1":
            raise ItemImportError("attempt_state_invalid", "an attempt state has an unsupported schema")
        if value.get("state") not in STATES:
            raise ItemImportError("attempt_state_invalid", "an attempt state has an unknown transition")
        self.validate(value, path)
        return value

    def attempts(self) -> Iterator[dict[str, dict[str, Any]]]:
        repository = self.gateway.experiment_repository(self.experiment_id)
        from mlflow_import.storage import DiskList

        for directory in DiskList(resources.iter_artifacts(repository, self.prefix)).sorted_by(lambda item: item.path):
            if not directory.is_dir or re.fullmatch(re.escape(self.prefix) + r"/[a-f0-9]{32}", directory.path) is None:
                raise ItemImportError("attempt_state_invalid", "an unexpected attempt object requires recovery")
            states: dict[str, dict[str, Any]] = {}
            for item in resources.iter_artifacts(repository, directory.path):
                if item.is_dir:
                    raise ItemImportError("attempt_state_invalid", "an unexpected nested attempt object exists")
                value = self.read(item.path)
                state = value["state"]
                if state in states and states[state] != value:
                    raise ItemImportError("attempt_state_conflict", "an attempt contains conflicting state records")
                states[state] = value
            if "prepared" not in states or ("complete" in states and "aborted" in states):
                raise ItemImportError("attempt_state_conflict", "an attempt has an invalid transition chain")
            if "complete" in states and "ready_to_retire" not in states:
                raise ItemImportError("attempt_state_conflict", "a completed attempt has no verified ready state")
            self.validate_chain(states)
            yield states

    def publish(self, state: str, values: Mapping[str, object]) -> dict[str, Any]:
        if state not in STATES:
            raise ValueError("unknown attempt state")
        value = {**values, "schema_version": "mlflow-import.attempt.v1", "state": state}
        self.validate(value)
        self.last_attempt_id = str(value["attempt_id"])
        with resources.temporary_directory(prefix="mlflow-import-state-") as root:
            source = Path(root) / "state.json"
            checksum, _ = write_state_json(source, value)
            path = f"{self.prefix}/{value['attempt_id']}/{state}-{checksum}.json"
            self.gateway.publish_experiment_file(self.experiment_id, path, source)
        return value
