"""Fake import-orchestration seams for importer, CLI, and BDD tests.

The importer's unit, CLI, and BDD tests run the REAL source pipeline —
fixture database → live read-only probe → registered exporter →
normalized snapshot → replay adapter → discovery — with FAKE MLflow and
lock seams:

- :class:`FakeLockFactory` is the ``ImportDependencies`` lock seam: it
  records acquisitions and releases and never touches disk (its
  ``acquisitions`` list is the accessor the task's verbatim dry-run
  assertion reads).
- One :class:`~tests.support.fake_gateway.FakeMlflowGateway` instance
  serves both run modes: the injected gateway factory switches its
  ``dry_run`` flag per call and returns the same object, so a test's
  handle on the gateway observes exactly the gateway the run used
  (``gateway.mutations`` is the accessor the verbatim assertion reads).
- :class:`FakeImportDependencies` subclasses the production
  :class:`~opencode_mlflow.importer.ImportDependencies` to carry the fake
  gateway as a direct test handle while every injected seam stays
  constructible through the production type.

:func:`build_current_source_db` materializes a sanitized synthetic
current-generation source database; :func:`build_import_fakes` wires the
full dependency set. This module is test support: importable from tests
only, never from ``src``.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from opencode_mlflow.content.fingerprint import artifact_snapshot_fingerprint
from opencode_mlflow.errors import ConfigurationError
from opencode_mlflow.importer import ImportDependencies, build_replay_index
from opencode_mlflow.locking import ImportLock
from opencode_mlflow.mlflow.gateway import TraceCreateRequest
from opencode_mlflow.mlflow.mapping import MappingContext, map_turn_to_trace
from opencode_mlflow.mlflow.reconciliation import upload_artifact
from opencode_mlflow.retry import RetryPolicy
from opencode_mlflow.source.discovery import DiscoveryTurnEntry
from opencode_mlflow.source.model import SchemaDescription
from opencode_mlflow.source.registry import build_live_exporter
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter, ReplayTurn
from opencode_mlflow.source.selection import SelectionEvidenceEvaluator
from opencode_mlflow.source.snapshot import (
    RedactionCallback,
    SourceExporter,
    SourceSnapshot,
    create_normalized_snapshot,
)
from tests.support.fake_gateway import FakeMlflowGateway

__all__ = [
    "BDD_CHANGED_REPLY_JSON",
    "BDD_EXPERIMENT_NAME",
    "BDD_PROMPT_JSON",
    "BDD_REPLY_JSON",
    "BDD_SESSION_ID",
    "CURRENT_DDL",
    "DEFAULT_SESSION_CREATED_MS",
    "FakeImportDependencies",
    "FakeLockFactory",
    "IMMEDIATE_RETRY_POLICY",
    "LEGACY_DDL",
    "SyntheticSourceBuilder",
    "UNIT_PROMPT_JSON",
    "UNIT_REPLY_JSON",
    "UNIT_SESSION_ID",
    "build_current_source_db",
    "build_import_fakes",
]

IMMEDIATE_RETRY_POLICY = RetryPolicy(max_attempts=1, initial_delay_seconds=0.0, max_delay_seconds=0.0)
"""A retry policy that never retries and never sleeps (deterministic tests)."""

CURRENT_DDL = """
CREATE TABLE session_v2 (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    workspace_id TEXT,
    parent_id TEXT,
    fork_session_id TEXT,
    fork_boundary INTEGER,
    slug TEXT,
    directory TEXT,
    path TEXT,
    title TEXT,
    version TEXT,
    share_url TEXT,
    summary_additions INTEGER,
    summary_deletions INTEGER,
    summary_files INTEGER,
    summary_diffs TEXT,
    metadata TEXT,
    cost REAL,
    tokens_input INTEGER,
    tokens_output INTEGER,
    tokens_reasoning INTEGER,
    tokens_cache_read INTEGER,
    tokens_cache_write INTEGER,
    revert TEXT,
    permission TEXT,
    agent TEXT,
    model TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    time_idle INTEGER,
    time_viewed INTEGER,
    idle_outcome TEXT,
    time_compacting INTEGER,
    time_archived INTEGER,
    time_suspended INTEGER,
    resume_attempts INTEGER
);
CREATE TABLE session_message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    type TEXT NOT NULL,
    seq INTEGER,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
"""

DEFAULT_SESSION_CREATED_MS = 1_772_000_000_000
"""Creation instant of every synthetic session (mirrors the other fixtures)."""

UNIT_SESSION_ID = "ses-import-unit-1"
"""The default fixture source's exact session ID."""

UNIT_PROMPT_JSON = '{"content": [{"type": "text", "text": "synthetic unit prompt"}]}'
"""The default fixture source's synthetic user-prompt payload."""

UNIT_REPLY_JSON = '{"content": [{"type": "text", "text": "synthetic unit reply"}]}'
"""The default fixture source's synthetic assistant-reply payload."""

BDD_EXPERIMENT_NAME = "bdd-cli-import-experiment"
"""The experiment name the BDD acceptance scenarios import under."""

BDD_SESSION_ID = "ses-bdd-cli"
"""The default synthetic session ID of the BDD boundary scenarios."""

BDD_PROMPT_JSON = '{"content": [{"type": "text", "text": "synthetic cli prompt"}]}'
"""The default synthetic user prompt of the BDD boundary scenarios."""

BDD_REPLY_JSON = '{"content": [{"type": "text", "text": "synthetic cli reply"}]}'
"""The default synthetic assistant reply of the BDD boundary scenarios."""

BDD_CHANGED_REPLY_JSON = '{"content": [{"type": "text", "text": "synthetic changed reply"}]}'
"""The changed assistant reply for replacement scenarios (same turn identity)."""

LEGACY_DDL = """
CREATE TABLE session (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    workspace_id TEXT,
    parent_id TEXT,
    slug TEXT,
    directory TEXT,
    path TEXT,
    title TEXT,
    version TEXT,
    share_url TEXT,
    summary_additions INTEGER,
    summary_deletions INTEGER,
    summary_files INTEGER,
    summary_diffs TEXT,
    metadata TEXT,
    cost REAL,
    tokens_input INTEGER,
    tokens_output INTEGER,
    tokens_reasoning INTEGER,
    tokens_cache_read INTEGER,
    tokens_cache_write INTEGER,
    revert TEXT,
    permission TEXT,
    agent TEXT,
    model TEXT,
    time_created INTEGER,
    time_updated INTEGER,
    time_compacting INTEGER,
    time_archived INTEGER
);
CREATE TABLE message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
CREATE TABLE part (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    time_created INTEGER,
    time_updated INTEGER,
    data TEXT
);
"""


def build_current_source_db(
    directory: Path,
    *,
    name: str = "import-source.db",
    sessions: Sequence[tuple[str, int]] = (),
    messages: Sequence[tuple[str, str, str, int, str]] = (),
) -> Path:
    """Materialize a sanitized synthetic current-generation source database.

    ``sessions`` are ``(session_id, created_ms)`` rows; ``messages`` are
    ``(message_id, session_id, type, seq, data_json)`` rows. Every value
    is synthetic; no real session data appears here.
    """
    path = directory / name
    connection = sqlite3.connect(path)
    try:
        connection.executescript(CURRENT_DDL)
        for session_id, created_ms in sessions:
            connection.execute(
                "INSERT INTO session_v2 (id, time_created, time_updated) VALUES (?, ?, ?)",
                (session_id, created_ms, created_ms),
            )
        for message_id, session_id, message_type, seq, data in messages:
            created_ms = DEFAULT_SESSION_CREATED_MS + 10_000 + seq * 1_000
            connection.execute(
                "INSERT INTO session_message (id, session_id, type, seq, time_created, time_updated, data)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, session_id, message_type, seq, created_ms, created_ms, data),
            )
        connection.commit()
    finally:
        connection.close()
    return path


@dataclass
class _SyntheticMessage:
    """One synthetic message row of a builder session."""

    message_id: str
    message_type: str
    seq: int
    time_created: int
    data_json: str


@dataclass
class _SyntheticSession:
    """One synthetic builder session: header fields plus ordered messages."""

    session_id: str
    created_ms: int
    directory: str | None
    messages: list[_SyntheticMessage] = field(default_factory=list)


def _model_object(identity: str) -> dict[str, str]:
    """Split a ``provider/model`` identity into its JSON object form."""
    provider, _, model = identity.partition("/")
    assert provider and model, f"the synthetic model identity must be provider/model: {identity!r}"
    return {"provider": provider, "model": model}


class SyntheticSourceBuilder:
    """Deterministic synthetic source databases for boundary-level scenarios.

    Scenarios accumulate sanitized synthetic sessions (current or legacy
    generation) through :meth:`add_session`/:meth:`add_legacy_session`,
    then mutate them (:meth:`change_reply`, :meth:`drop_turn`,
    :meth:`drop_all_turns`) and let the harness materialize the source
    database. Every row is synthetic; no real session data appears.

    Model identities are placed exactly where a scenario asks:
    ``message_model`` rides the assistant ``session_message``/``message``
    data, ``part_model`` rides one content part of the assistant reply.
    ``activity_ms`` overrides the message timestamps (for example to give
    a session message activity inside a selection window its creation
    time is outside).
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._sessions: dict[str, _SyntheticSession] = {}
        self._generation: str | None = None
        self._materialized_count = 0
        self._dirty = False

    @property
    def session_count(self) -> int:
        """How many synthetic sessions the builder holds."""
        return len(self._sessions)

    @property
    def dirty(self) -> bool:
        """Whether the accumulated rows changed since the last materialization."""
        return self._dirty

    def add_session(
        self,
        session_id: str,
        *,
        created_ms: int,
        directory: str | None = None,
        turns: int = 1,
        message_model: str | None = None,
        part_model: str | None = None,
        activity_ms: int | None = None,
        prompt_json: str | None = None,
        reply_json: str | None = None,
    ) -> None:
        """Add one current-generation session with ``turns`` user/assistant turns."""
        self._require_generation("current")
        session = _SyntheticSession(session_id=session_id, created_ms=created_ms, directory=directory)
        base_ms = activity_ms if activity_ms is not None else created_ms
        prompt = prompt_json if prompt_json is not None else BDD_PROMPT_JSON
        reply = self._assistant_data(reply_json, message_model, part_model)
        for turn in range(1, turns + 1):
            for role, data in (("user", prompt), ("assistant", reply)):
                seq = turn * 2 - (1 if role == "user" else 0)
                session.messages.append(
                    _SyntheticMessage(
                        message_id=f"{session_id}-{role}-{turn}",
                        message_type=role,
                        seq=seq,
                        time_created=base_ms + 10_000 + seq * 1_000,
                        data_json=data,
                    )
                )
        self._sessions[session_id] = session
        self._dirty = True

    def add_legacy_session(
        self,
        session_id: str,
        *,
        created_ms: int,
        turns: int = 1,
        message_model: str | None = None,
    ) -> None:
        """Add one legacy-generation session with ``turns`` user/assistant turns."""
        self._require_generation("legacy")
        session = _SyntheticSession(session_id=session_id, created_ms=created_ms, directory=None)
        for turn in range(1, turns + 1):
            for role, data in (
                ("user", '{"role": "user", "content": "synthetic legacy prompt"}'),
                ("assistant", self._legacy_assistant_data(message_model)),
            ):
                position = turn * 2 - (1 if role == "user" else 0)
                session.messages.append(
                    _SyntheticMessage(
                        message_id=f"{session_id}-{role}-{turn}",
                        message_type=role,
                        seq=position,
                        time_created=created_ms + 10_000 + position * 1_000,
                        data_json=data,
                    )
                )
        self._sessions[session_id] = session
        self._dirty = True

    def change_reply(self, session_id: str, *, turn: int = 1, reply_json: str = BDD_CHANGED_REPLY_JSON) -> None:
        """Change one turn's assistant reply: same turn identity, new content."""
        self._message(session_id, "assistant", turn).data_json = reply_json
        self._dirty = True

    def drop_turn(self, session_id: str, *, turn: int) -> None:
        """Remove one turn's user and assistant messages from the source."""
        session = self._session(session_id)
        dropped = {f"{session_id}-user-{turn}", f"{session_id}-assistant-{turn}"}
        session.messages = [message for message in session.messages if message.message_id not in dropped]
        self._dirty = True

    def drop_all_turns(self, session_id: str) -> None:
        """Remove every message of one session (a zero-turn tombstone)."""
        self._session(session_id).messages.clear()
        self._dirty = True

    def materialize(self) -> Path:
        """Write the accumulated sessions to a fresh source database.

        Called by the boundary harness whenever the rows are dirty or no
        source exists yet; each call writes a new numbered file under the
        builder's root so earlier materializations stay byte-stable.
        """
        assert self._generation is not None, "the builder holds no synthetic sessions"
        self._materialized_count += 1
        path = self._root / f"bdd-source-{self._materialized_count}.db"
        connection = sqlite3.connect(path)
        try:
            connection.executescript(CURRENT_DDL if self._generation == "current" else LEGACY_DDL)
            self._insert_sessions(connection)
            connection.commit()
        finally:
            connection.close()
        self._dirty = False
        return path

    def materialize_wal(self, *, name: str = "bdd-wal-source.db") -> tuple[Path, sqlite3.Connection]:
        """Write the current-generation sessions into a live WAL-mode database.

        WAL is enabled before any row is inserted, so every committed row
        lives in the ``-wal`` file. The returned writer connection stays
        OPEN (keeping the ``-wal``/``-shm`` companions alive for a
        concurrent read-only snapshot); the caller must close it.
        """
        assert self._generation == "current", "WAL materialization builds current-generation rows"
        path = self._root / name
        writer = sqlite3.connect(path, isolation_level=None)
        try:
            writer.execute("PRAGMA journal_mode = WAL")
            writer.executescript(CURRENT_DDL)
            self._insert_sessions(writer)
        except BaseException:
            writer.close()
            raise
        self._dirty = False
        return path, writer

    def _insert_sessions(self, connection: sqlite3.Connection) -> None:
        """Insert every accumulated session and message row."""
        current = self._generation == "current"
        session_sql = (
            "INSERT INTO session_v2 (id, directory, time_created, time_updated) VALUES (?, ?, ?, ?)"
            if current
            else "INSERT INTO session (id, directory, time_created, time_updated) VALUES (?, ?, ?, ?)"
        )
        message_sql = (
            "INSERT INTO session_message (id, session_id, type, seq, time_created, time_updated, data)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            if current
            else "INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, ?, ?, ?, ?)"
        )
        for session in self._sessions.values():
            connection.execute(
                session_sql,
                (session.session_id, session.directory, session.created_ms, session.created_ms),
            )
            for message in session.messages:
                if current:
                    connection.execute(
                        message_sql,
                        (
                            message.message_id,
                            session.session_id,
                            message.message_type,
                            message.seq,
                            message.time_created,
                            message.time_created,
                            message.data_json,
                        ),
                    )
                else:
                    connection.execute(
                        message_sql,
                        (
                            message.message_id,
                            session.session_id,
                            message.time_created,
                            message.time_created,
                            message.data_json,
                        ),
                    )

    def _assistant_data(self, reply_json: str | None, message_model: str | None, part_model: str | None) -> str:
        """The assistant reply payload with the requested model placement."""
        if part_model is not None:
            return json.dumps(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": "synthetic cli reply",
                            "model": _model_object(part_model),
                        }
                    ]
                }
            )
        data = json.loads(reply_json if reply_json is not None else BDD_REPLY_JSON)
        if message_model is not None:
            data["model"] = _model_object(message_model)
        return json.dumps(data)

    @staticmethod
    def _legacy_assistant_data(message_model: str | None) -> str:
        """The legacy assistant payload, optionally carrying a message-level model."""
        data: dict[str, object] = {"role": "assistant", "content": "synthetic legacy reply"}
        if message_model is not None:
            data["model"] = _model_object(message_model)
        return json.dumps(data)

    def _session(self, session_id: str) -> _SyntheticSession:
        session = self._sessions.get(session_id)
        assert session is not None, f"the builder holds no session {session_id!r}"
        return session

    def _message(self, session_id: str, role: str, turn: int) -> _SyntheticMessage:
        message_id = f"{session_id}-{role}-{turn}"
        for message in self._session(session_id).messages:
            if message.message_id == message_id:
                return message
        raise AssertionError(f"the builder holds no message {message_id!r}")

    def _require_generation(self, generation: str) -> None:
        """One builder serves one source generation per scenario."""
        if self._generation is None:
            self._generation = generation
        assert self._generation == generation, "a scenario's builder cannot mix source generations"


class FakeLockFactory:
    """``ImportDependencies`` lock seam that records acquisitions without disk.

    ``acquisitions``/``releases`` record ``(tracking_endpoint,
    experiment_name)`` pairs in order. ``fail=True`` models a held lock:
    acquisition raises the same safe ``import_lock_held`` error the real
    factory raises, without creating anything.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.acquisitions: list[tuple[str, str]] = []
        self.releases: list[tuple[str, str]] = []

    def __call__(self, tracking_endpoint: str, experiment_name: str) -> AbstractContextManager[ImportLock]:
        return self._acquire(tracking_endpoint, experiment_name)

    @contextmanager
    def _acquire(self, tracking_endpoint: str, experiment_name: str) -> Iterator[ImportLock]:
        if self.fail:
            raise ConfigurationError(
                "import_lock_held",
                "another importer already holds the lock for this tracking endpoint and experiment",
            )
        self.acquisitions.append((tracking_endpoint, experiment_name))
        try:
            yield ImportLock(
                path=Path(f"fake-import-lock-{len(self.acquisitions)}.lock"),
                experiment_name=experiment_name,
                endpoint_digest="fake-endpoint-digest",
                acquired=True,
            )
        finally:
            self.releases.append((tracking_endpoint, experiment_name))


@dataclass(frozen=True)
class FakeImportDependencies(ImportDependencies):
    """ImportDependencies over the real source pipeline and fake MLflow/lock seams.

    Adds the fake gateway as a direct handle (``import_dependencies.gateway``)
    so tests can assert on the exact gateway instance the run used; every
    other seam keeps its production field.
    """

    gateway: FakeMlflowGateway


def build_import_fakes(
    *,
    restrictive_tmp_path: Path,
    gateway: FakeMlflowGateway | None = None,
    lock_factory: FakeLockFactory | None = None,
    replay_builder: Callable[[SourceSnapshot], Callable[[DiscoveryTurnEntry], ReplayTurn]] | None = None,
    retry_policy: RetryPolicy = IMMEDIATE_RETRY_POLICY,
    clock: Callable[[], int] | None = None,
) -> FakeImportDependencies:
    """Build the importer dependencies: real source pipeline, fake MLflow/lock.

    The snapshot, exporter, replay-adapter, mapper, artifact-snapshot, and
    artifact-uploader seams are the production defaults over a sanitized
    synthetic source; the gateway and lock seams are the in-memory fakes,
    and one fake gateway instance serves both run modes.
    """
    fake_gateway = gateway if gateway is not None else FakeMlflowGateway()
    lock = lock_factory if lock_factory is not None else FakeLockFactory()
    effective_replay_builder = replay_builder if replay_builder is not None else build_replay_index

    def temp_directory_factory(prefix: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=prefix, dir=restrictive_tmp_path))

    def exporter_factory(connection: sqlite3.Connection, description: SchemaDescription) -> SourceExporter:
        return build_live_exporter(connection, description)

    def gateway_factory(dry_run: bool) -> FakeMlflowGateway:
        fake_gateway.dry_run = dry_run
        return fake_gateway

    def mapper(turn: ReplayTurn, context: MappingContext) -> TraceCreateRequest:
        return map_turn_to_trace(turn, context).to_create_request()

    def snapshot_factory(
        path: Path,
        temp_root: Path,
        exporter: SourceExporter,
        redaction: RedactionCallback,
        selection_evaluator: SelectionEvidenceEvaluator,
    ) -> AbstractContextManager[SourceSnapshot]:
        return create_normalized_snapshot(
            path,
            temp_root,
            exporter,
            redaction,
            selection_evaluator=selection_evaluator,
        )

    return FakeImportDependencies(
        snapshot_factory=snapshot_factory,
        exporter_factory=exporter_factory,
        replay_adapter_factory=NormalizedSnapshotAdapter,
        gateway_factory=gateway_factory,
        lock_factory=lock,
        temp_directory_factory=temp_directory_factory,
        replay_builder=effective_replay_builder,
        upload_chunk_bytes=8 * 1024 * 1024,
        mapper=mapper,
        artifact_snapshot=artifact_snapshot_fingerprint,
        artifact_uploader=upload_artifact,
        clock=clock if clock is not None else (lambda: 0),
        retry_policy=retry_policy,
        gateway=fake_gateway,
    )
