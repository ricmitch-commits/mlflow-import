"""Operation-count gates over synthetic, frozen sessions."""

from dataclasses import replace

import pytest

from mlflow_import.source.replay import NormalizedSnapshotAdapter
from mlflow_import.source.turns import build_replay_turn, indexed_turn_summaries
from tests.unit.source.test_turns import _record, _scripted_snapshot

pytestmark = pytest.mark.unit


def test_direct_turn_reads_do_not_decode_other_turns(tmp_path, monkeypatch):
    records = [_record(f"u-{i}", seq=i) for i in range(80)]
    with _scripted_snapshot(tmp_path, records) as (snapshot, _):
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = next(adapter.iter_session_headers())
        summaries = list(indexed_turn_summaries(snapshot, header.session_id))
        import mlflow_import.source.replay as replay

        original = replay._restore_source_record
        decoded = 0

        def observe(*args, **kwargs):
            nonlocal decoded
            decoded += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(replay, "_restore_source_record", observe)
        for summary in summaries:
            turn = build_replay_turn(snapshot, header, summary)
            assert [record.source_id for record in turn.iter_records()] == [f"u-{summary.turn_index - 1}"]
        assert decoded == 80
        # Persisted summaries are read without reopening transcript cursors.
        assert list(indexed_turn_summaries(snapshot, header.session_id)) == summaries
        assert decoded == 80


def test_index_keeps_collision_keys_and_preamble(tmp_path):
    records = [replace(_record("preamble"), classification="system"), _record("same"), _record("same")]
    with _scripted_snapshot(tmp_path, records) as (snapshot, _):
        summaries = list(indexed_turn_summaries(snapshot, "ses-1"))
        assert len({summary.turn_key for summary in summaries}) == 2
        assert summaries[1].warnings == ("turn_key_collision",)


def test_shared_session_hash_and_artifact_are_turn_independent(tmp_path):
    import io
    import json

    from mlflow_import.content.bundles import stream_session_object
    from mlflow_import.content.fingerprint import artifact_snapshot_fingerprint
    from mlflow_import.contracts import RedactionMode

    with _scripted_snapshot(tmp_path, [_record("first"), _record("second")]) as (snapshot, adapter):
        header = next(adapter.iter_session_headers())
        turns = [build_replay_turn(snapshot, header, summary) for summary in indexed_turn_summaries(snapshot, "ses-1")]
        hashes = [artifact_snapshot_fingerprint(turn, RedactionMode.VERBATIM) for turn in turns]
        assert hashes[0] == hashes[1]
        output = io.BytesIO()
        stream_session_object(turns[0], output, RedactionMode.VERBATIM)
        document = json.loads(output.getvalue())
        assert [record["source_id"] for record in document["records"]] == ["first", "second"]
        assert len(document["turns"]) == 2
