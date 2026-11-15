"""Stream canonical shared session objects, independent of generated target IDs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict
from typing import BinaryIO

from mlflow_import.content.canonical import canonical_json
from mlflow_import.content.fingerprint import (
    _complete_session_value,
    artifact_snapshot_fingerprint,
    canonical_record_value,
)
from mlflow_import.contracts import RedactionMode
from mlflow_import.source.model import SessionHeader
from mlflow_import.source.replay import ReplayTurn


def session_prefix(turn: ReplayTurn, mode: RedactionMode) -> str:
    """Bound untrusted identities to short path components without revealing locators."""
    return source_prefix(turn.session_header, mode)


def source_prefix(header: SessionHeader, mode: RedactionMode) -> str:
    """Stable artifact/recovery namespace from an authoritative header alone."""
    from mlflow_import.identity import harness_name

    harness = harness_name(header)
    namespace = header.source_fingerprint or "unqualified"
    identity = hashlib.sha256(header.session_id.encode("utf-8")).hexdigest()
    return f"mlflow-import/v2/{harness}/{namespace}/{identity}/{mode.value}"


def stream_session_object(turn: ReplayTurn, output: BinaryIO, mode: RedactionMode) -> tuple[str, int]:
    """Write one session with all permitted occurrences and indexed turn boundaries.

    A production replay offers a full-session cursor in persisted order. Legacy
    replay implementations retain their explicit three-cursor compatibility view.
    """
    digest = hashlib.sha256()
    size = 0

    def write(chunk: bytes) -> None:
        nonlocal size
        output.write(chunk)
        digest.update(chunk)
        size += len(chunk)

    def array(values: Iterable[object]) -> None:
        write(b"[")
        separator = b""
        for value in values:
            write(separator)
            write(canonical_json(value))
            separator = b","
        write(b"]")

    records = getattr(turn, "iter_all_records", None)
    if not callable(records):
        raise ValueError("a shared session object requires the frozen full-session cursor")
    write(b'{"privacy":')
    write(canonical_json(mode.value))
    write(b',"records":')
    array(canonical_record_value(record) for record in records())
    write(b',"schema_version":"mlflow-import.session.v1","session":')
    from mlflow_import.identity import harness_name, native_metadata

    write(
        canonical_json(
            {
                **_complete_session_value(turn.session_header),
                "source": harness_name(turn.session_header),
                "native_metadata": native_metadata(turn.session_header),
            }
        )
    )
    write(b',"source_fingerprint":')
    write(canonical_json(artifact_snapshot_fingerprint(turn, mode)))
    write(b',"turns":')
    summaries = getattr(turn, "iter_summaries", None)
    if not callable(summaries):
        raise ValueError("a shared session object requires indexed summaries")
    array(asdict(summary) for summary in summaries())
    write(b',"projections":')

    def projections() -> Iterable[object]:
        from mlflow_import.content.activity import project_activity
        from mlflow_import.content.coverage import coverage
        from mlflow_import.source.turns import build_replay_turn

        snapshot = getattr(turn, "snapshot", None)
        if snapshot is None:
            # Replay's public constructor context is kept on its adapter view.
            snapshot = getattr(turn, "_snapshot", None)
        if snapshot is None:
            raise ValueError("shared projections require a frozen snapshot")
        for summary in summaries():
            replay = build_replay_turn(snapshot, turn.session_header, summary)
            activity = project_activity(replay)
            yield {
                "turn_key": summary.turn_key,
                "accounting": activity.provenance,
                "coverage": coverage(turn.session_header, activity),
            }

    array(projections())
    write(b"}")
    return digest.hexdigest(), size
