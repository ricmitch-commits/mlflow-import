"""Integration: bounded-memory stress test for streamed artifacts.

This module is marked ``integration`` and ``slow`` and runs only when
explicitly selected (``pytest tests/integration/test_scale_memory.py -m
integration -q``). It generates a synthetic session of at least 100,000
records whose streamed artifact is multi-hundred-megabyte — "multi-
hundred-megabyte artifact" here means the TOTAL artifact bytes streamed
through :func:`~opencode_mlflow.content.artifacts.stream_turn_artifact`
and uploaded through the chunk sink exceed 200 MiB — and proves the
artifact layer's bounded-memory contract:

- discovery (the effective-turn/source-fingerprint seam), canonicalization,
  and upload all stream: the synthetic ``ReplayTurn`` generates every
  record lazily, so no record list and no single in-memory object ever
  scales with the complete session;
- the source fingerprint is stable across independent full passes;
- every artifact byte is preserved through the fake chunk sink (running
  digest and size equal the manifest's sha256 and size);
- upload chunks are fixed-size (every chunk except the last is exactly
  the negotiated chunk size);
- Python heap growth from an empty-process baseline stays below 256 MiB;
- a COMPLETE root value above the REAL inline limit the pinned MLflow
  profile negotiates (1 MiB, ``MAX_INLINE_PAYLOAD_BYTES``) is rejected
  with a safe ``payload_too_large`` error at MAPPING time — the write
  phase's first per-turn step, before any trace could be created — while
  the complete value stays in the turn's streamed artifact, never
  truncated. (The fake-limit variants of this rejection — a 64-byte
  limit — are pinned in ``tests/unit/mlflow/test_mapping.py`` and
  ``tests/contract/test_mapping_contract.py``; this test pins the real
  negotiated limit, sized between it and the 16 MiB conservative
  default so only the negotiated limit can reject it.)

Memory-measurement judgment (documented): growth is measured with
``tracemalloc`` from the baseline taken immediately after tracing
starts — allocations made before that point are not traced, so the
baseline is an empty-process baseline for tracing purposes. tracemalloc
counts Python-level allocations only; C-level buffers (file-system page
cache, the writing handle's fixed buffer) are deliberately outside the
measurement because the contract bounds the *Python heap*. The peak
traced size is asserted in addition to the final size, so transient
accumulations fail the test too.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencode_mlflow.content.artifacts import (
    DEFAULT_UPLOAD_CHUNK_BYTES,
    ArtifactUploadChunk,
    iter_artifact_chunks,
    stream_turn_artifact,
    temporary_artifact_file,
)
from opencode_mlflow.content.fingerprint import build_effective_turn, source_fingerprint
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.errors import ItemImportError
from opencode_mlflow.importer import build_replay_index
from opencode_mlflow.mlflow.capabilities import CapabilityRequest
from opencode_mlflow.mlflow.gateway import MlflowGateway
from opencode_mlflow.mlflow.mapping import MappingContext, map_turn_to_trace
from opencode_mlflow.mlflow.reconciliation import created_artifact_identity_map
from opencode_mlflow.source.discovery import (
    ARTIFACT_SCHEMA_VERSION,
    MAPPING_VERSION,
    DiscoveryPlanReader,
    discover_sessions,
)
from opencode_mlflow.source.model import SessionHeader, SourceRecord, TimestampValue, TurnSummary
from opencode_mlflow.source.probe import probe_schema
from opencode_mlflow.source.registry import build_live_exporter
from opencode_mlflow.source.replay import NormalizedSnapshotAdapter, ReplayTurn
from opencode_mlflow.source.selection import SelectionFilters
from opencode_mlflow.source.snapshot import create_normalized_snapshot
from tests.integration.mlflow_profile import MAX_INLINE_PAYLOAD_BYTES
from tests.support.fake_gateway import FakeMlflowClient
from tests.support.fake_redactor import IdentityRedactor
from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_record, fake_summary
from tests.support.import_fakes import CURRENT_DDL

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SESSION_ID = "ses-stress-1"
TURN_KEY = "turn-" + "3" * 64
TURN_RECORD_COUNT = 100_000
SESSION_RECORD_COUNT = 2_000
UNASSIGNED_RECORD_COUNT = 3_000
TOTAL_RECORD_COUNT = TURN_RECORD_COUNT + SESSION_RECORD_COUNT + UNASSIGNED_RECORD_COUNT
TEXT_LENGTH = 2_100
"""Per-record synthetic text length; sized so the artifact exceeds 200 MiB."""

MINIMUM_ARTIFACT_BYTES = 200 * 1024 * 1024
"""'Multi-hundred-megabyte artifact': total streamed artifact bytes."""

MAXIMUM_HEAP_GROWTH_BYTES = 256 * 1024 * 1024
"""The stress budget: Python heap growth from the empty-process baseline."""

_BASE_TIMESTAMP_MS = 1_772_000_000_000
_TEXT_FILLER = "synthetic stress payload "


class _DiscardingSpanApi:
    """Span-handle sink: measures importer structures, NOT MLflow payload storage.

    The real SDK retains span payloads until root export. Discarding payloads
    here deliberately isolates the accumulating importer parent/identity maps;
    this is not evidence for a 100k-span live-MLflow heap bound.
    """

    def __init__(self):
        self.started = 0
        self.ended = 0

    def start_span_no_context(self, **kwargs):
        self.started += 1
        parent = kwargs.get("parent_span")
        return SimpleNamespace(
            trace_id="tr-scale",
            span_id=f"span-{self.started}",
            parent_id=parent.span_id if parent else None,
        )

    def end_span(self, span, **kwargs):
        self.ended += 1


def test_production_source_discovery_mapping_and_gateway_structure_budget(restrictive_tmp_path):
    """100k real SQLite records through production code, with SDK retention isolated."""
    source = restrictive_tmp_path / "production-scale.db"
    with sqlite3.connect(source) as connection:
        connection.executescript(CURRENT_DDL)
        connection.execute("INSERT INTO session_v2 (id,time_created,time_updated) VALUES ('scale',1000,2000)")
        connection.executemany(
            "INSERT INTO session_message VALUES (?, 'scale', ?, ?, ?, ?, ?)",
            (
                (
                    f"m-{i:06}",
                    "user" if i == 0 else "assistant",
                    i,
                    1000 + i,
                    1000 + i,
                    json.dumps({"text": "synthetic " + "x" * 128}),
                )
                for i in range(100_000)
            ),
        )
        exporter = build_live_exporter(connection, probe_schema(connection))
    sink = _DiscardingSpanApi()
    gateway = MlflowGateway(
        FakeMlflowClient(),
        trace_api=sink,
        version_transport=lambda _uri: {"version": "3.17.2"},
    )
    assert gateway.probe(CapabilityRequest()).is_compatible
    tracemalloc.start()
    try:
        with create_normalized_snapshot(
            source, restrictive_tmp_path, exporter, lambda value, protected: value
        ) as snapshot:
            adapter = NormalizedSnapshotAdapter(snapshot)
            plan = restrictive_tmp_path / "production-scale-plan.jsonl"
            stats = discover_sessions(adapter, SelectionFilters(), plan)
            entry = next(DiscoveryPlanReader(plan).iter_turns())
            replay = build_replay_index(snapshot)(entry)
            context = MappingContext(
                importer_version="0.1.0",
                mapping_version=MAPPING_VERSION,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                source_schema="current",
                redaction_mode=RedactionMode.VERBATIM,
                experiment_id="scale-exp",
                source_database_fingerprint=snapshot.fingerprint,
                inline_limit_bytes=MAX_INLINE_PAYLOAD_BYTES,
            )
            spec = map_turn_to_trace(replay, context).to_create_request()
            created = gateway.create_trace(spec)
            assert len(created.span_ids_by_source) == 99_999
            assert snapshot.stats.record_count == 100_000
            assert stats.turn_entry_count == 1
            with temporary_artifact_file(restrictive_tmp_path) as (path, handle):
                manifest = stream_turn_artifact(
                    replay, handle, created_artifact_identity_map(created), RedactionMode.VERBATIM
                )
                handle.flush()
                upload_sink = _FakeChunkSink()
                with path.open("rb") as stream:
                    for chunk in iter_artifact_chunks(stream, DEFAULT_UPLOAD_CHUNK_BYTES):
                        upload_sink.upload(chunk)
                assert upload_sink.sha256 == manifest.sha256
                assert upload_sink.size == manifest.size_bytes
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert sink.started == sink.ended == 100_000
    assert peak < MAXIMUM_HEAP_GROWTH_BYTES
    print(
        f"production importer structures: 100000 records, artifact={manifest.size_bytes / 1024**2:.1f} MiB, "
        f"peak={peak / 1024**2:.1f} MiB; SDK payloads excluded"
    )


class _StreamingSyntheticTurn:
    """A ``ReplayTurn`` whose cursors synthesize records on demand.

    No record list is ever built: each cursor regenerates the same
    deterministic record stream per pass (keyed by section and index),
    so the only in-memory state is the turn configuration. The complete
    session exists solely as the bytes streamed through the artifact.
    """

    def __init__(
        self,
        header: SessionHeader,
        summary: TurnSummary,
        turn_count: int,
        session_count: int,
        unassigned_count: int,
        text_length: int,
    ) -> None:
        self._header = header
        self._summary = summary
        self._turn_count = turn_count
        self._session_count = session_count
        self._unassigned_count = unassigned_count
        self._text_length = text_length

    @property
    def session_header(self) -> SessionHeader:
        return self._header

    @property
    def summary(self) -> TurnSummary:
        return self._summary

    def iter_records(self) -> Iterator[SourceRecord]:
        return self._synthesize("turn", self._turn_count)

    def iter_session_records(self) -> Iterator[SourceRecord]:
        return self._synthesize("session", self._session_count)

    def iter_unassigned_records(self) -> Iterator[SourceRecord]:
        return self._synthesize("unassigned", self._unassigned_count)

    def load_root_input(self, max_bytes: int) -> object | None:
        return None

    def load_root_output(self, max_bytes: int) -> object | None:
        return None

    def _synthesize(self, kind: str, count: int) -> Iterator[SourceRecord]:
        for index in range(count):
            yield SourceRecord(
                source_table="session_message",
                source_type="assistant",
                source_id=f"msg-{kind}-{index:07d}",
                raw_json=None,
                decoded_value={"content": [{"type": "text", "text": self._text(kind, index)}]},
                raw_columns={},
                warnings=("synthetic_stress_warning",) if index % 50_000 == 0 else (),
                session_id=SESSION_ID,
                sequence_domain="session_message.seq",
                sequence_value=index,
                timestamp=TimestampValue(value=_BASE_TIMESTAMP_MS + index, unit="ms"),
                classification="assistant",
            )

    def _text(self, kind: str, index: int) -> str:
        prefix = f"{kind}-{index:07d}-"
        filler = _TEXT_FILLER * (self._text_length // len(_TEXT_FILLER) + 1)
        return (prefix + filler)[: self._text_length]


class _FakeChunkSink:
    """Upload double: verifies and digests each chunk, never accumulating.

    Mirrors the verifying upload seam the gateway will wrap: each
    chunk's digest is checked against its payload, a running SHA-256
    and size accumulate across the whole upload, and the payload is
    released immediately — the sink never holds the artifact.
    """

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.size = 0
        self.chunk_sizes: list[int] = []

    def upload(self, chunk: ArtifactUploadChunk) -> None:
        assert chunk.size == len(chunk.payload)
        assert chunk.digest == hashlib.sha256(chunk.payload).hexdigest()
        self.chunk_sizes.append(chunk.size)
        self._digest.update(chunk.payload)
        self.size += chunk.size

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def _stress_turn() -> _StreamingSyntheticTurn:
    return _StreamingSyntheticTurn(
        header=fake_header(session_id=SESSION_ID),
        summary=fake_summary(session_id=SESSION_ID, turn_key=TURN_KEY, record_count=TURN_RECORD_COUNT),
        turn_count=TURN_RECORD_COUNT,
        session_count=SESSION_RECORD_COUNT,
        unassigned_count=UNASSIGNED_RECORD_COUNT,
        text_length=TEXT_LENGTH,
    )


def test_streamed_artifact_upload_stays_bounded(restrictive_tmp_path: Path) -> None:
    turn = _stress_turn()
    identity_map = {"msg-turn-0000000": "tr_stress_root"}
    assert isinstance(turn, ReplayTurn)

    tracemalloc.start()
    try:
        baseline_current, _baseline_peak = tracemalloc.get_traced_memory()

        # Discovery seam: the effective-turn source fingerprint over two
        # independent full passes must be stable.
        effective = build_effective_turn(turn, IdentityRedactor(), RedactionMode.VERBATIM)
        first_source_fp = source_fingerprint(effective, RedactionMode.VERBATIM)
        second_source_fp = source_fingerprint(effective, RedactionMode.VERBATIM)

        # Canonicalization and upload: stream the artifact into a
        # restrictive temporary file, then upload it through the fake
        # chunk sink in fixed-size pieces.
        with temporary_artifact_file(restrictive_tmp_path) as (path, handle):
            manifest = stream_turn_artifact(turn, handle, identity_map, RedactionMode.VERBATIM)
            handle.flush()
            sink = _FakeChunkSink()
            with path.open("rb") as source:
                for chunk in iter_artifact_chunks(source, DEFAULT_UPLOAD_CHUNK_BYTES):
                    sink.upload(chunk)

        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Scale: at least 100,000 synthetic records and a multi-hundred-
    # megabyte artifact (total streamed bytes).
    assert TOTAL_RECORD_COUNT >= 100_000
    assert manifest.size_bytes >= MINIMUM_ARTIFACT_BYTES

    # The source fingerprint is stable across independent full passes.
    assert first_source_fp == second_source_fp

    # Every artifact byte is preserved through the fake chunk sink.
    assert sink.size == manifest.size_bytes
    assert sink.sha256 == manifest.sha256

    # Fixed upload chunk sizes: every chunk except the last is exactly
    # the chunk size, and no payload ever exceeds it.
    assert sink.chunk_sizes
    assert all(size == DEFAULT_UPLOAD_CHUNK_BYTES for size in sink.chunk_sizes[:-1])
    assert 0 < sink.chunk_sizes[-1] <= DEFAULT_UPLOAD_CHUNK_BYTES
    assert max(sink.chunk_sizes) <= DEFAULT_UPLOAD_CHUNK_BYTES

    # Bounded memory: growth (final and peak) from the empty-process
    # baseline stays below 256 MiB while the artifact exceeds 200 MiB —
    # no single in-memory object (or accumulation) can hold, or scale
    # with, the complete session.
    growth = current - baseline_current
    peak_growth = peak - baseline_current
    assert growth < MAXIMUM_HEAP_GROWTH_BYTES
    assert peak_growth < MAXIMUM_HEAP_GROWTH_BYTES
    assert peak_growth < manifest.size_bytes
    print(
        f"artifact stream: {TOTAL_RECORD_COUNT} records, {manifest.size_bytes / 1024**2:.1f} MiB, "
        f"peak={peak_growth / 1024**2:.1f} MiB"
    )


# ---------------------------------------------------------------------------
# payload_too_large rejection at the REAL inline limit
# ---------------------------------------------------------------------------

_OVERSIZED_INPUT_CHARS = MAX_INLINE_PAYLOAD_BYTES + 65_536
"""A complete root value above the pinned 1 MiB inline limit but well
below the 16 MiB conservative default, so only the NEGOTIATED limit —
the one the capability probe reports and the importer injects — can
reject it."""


def test_oversized_root_value_rejected_at_the_real_inline_limit_before_trace_creation(
    restrictive_tmp_path: Path,
) -> None:
    """A COMPLETE root value above the real 1 MiB limit is rejected before any trace.

    The mapping is the write phase's first per-turn step (the importer
    replays and maps a turn before it ever calls the gateway's
    ``create_trace``), so the safe ``payload_too_large`` rejection fires
    before a trace could exist. The complete value is never truncated:
    it stays in the turn's streamed artifact, byte-for-byte.
    """
    huge_text = "synthetic oversized prompt " * (_OVERSIZED_INPUT_CHARS // 26 + 1)
    huge_text = huge_text[:_OVERSIZED_INPUT_CHARS]
    huge_input = {"content": [{"type": "text", "text": huge_text}]}
    reply = {"content": [{"type": "text", "text": "synthetic oversized reply"}]}
    turn = FakeReplayTurn(
        session_header=fake_header(session_id=SESSION_ID),
        summary=fake_summary(session_id=SESSION_ID, turn_key=TURN_KEY, record_count=2),
        turn_records=(
            fake_record(
                "msg-oversized-u1",
                decoded=huge_input,
                session_id=SESSION_ID,
                source_type="user",
                seq=1,
                classification="user",
            ),
            fake_record(
                "msg-oversized-a1",
                decoded=reply,
                session_id=SESSION_ID,
                source_type="assistant",
                seq=2,
                classification="assistant",
            ),
        ),
        root_input=huge_input,
        root_output=reply,
    )
    context = MappingContext(
        importer_version="0.1.0",
        mapping_version="mapping-v1",
        artifact_schema_version="artifact-v1",
        source_schema="current",
        redaction_mode=RedactionMode.VERBATIM,
        experiment_id="exp-scale-oversized",
        source_database_fingerprint="synthetic-database-fingerprint",
        # The real negotiated limit the capability probe reports for the
        # pinned pair (1 MiB) — exactly what the importer injects.
        inline_limit_bytes=MAX_INLINE_PAYLOAD_BYTES,
        logical_turn_count=1,
        source_message_count=2,
        source_part_count=0,
    )

    # The mapping raises the safe rejection at the REAL limit, before any
    # fingerprint, request, or gateway call exists.
    with pytest.raises(ItemImportError) as excinfo:
        map_turn_to_trace(turn, context)
    assert excinfo.value.code == "payload_too_large"

    # The complete value is never truncated: the turn's streamed artifact
    # carries the full oversized prompt byte-for-byte.
    with temporary_artifact_file(restrictive_tmp_path) as (path, handle):
        manifest = stream_turn_artifact(turn, handle, {"trace_id": "tr-scale-oversized"}, RedactionMode.VERBATIM)
        handle.flush()
        document = json.loads(path.read_text())
    assert manifest.size_bytes > _OVERSIZED_INPUT_CHARS
    stored_prompt = next(record for record in document["turn_records"] if record.get("source_id") == "msg-oversized-u1")
    assert stored_prompt["decoded"]["content"][0]["text"] == huge_text
