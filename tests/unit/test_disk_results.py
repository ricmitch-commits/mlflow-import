"""Disk-backed DTOs preserve result semantics without materializing the corpus."""

import tracemalloc

import pytest

from mlflow_import.importer import ImportResult
from mlflow_import.reporting import iter_json
from mlflow_import.source.discovery import DiscoveryPlanReader, DiscoveryPlanWriter, SessionScanEntry
from mlflow_import.storage import DiskList, DiskMap, disk_collections

pytestmark = pytest.mark.unit


def test_disk_list_negative_indices_and_in_place_extension():
    with disk_collections():
        values = DiskList([1, 2])
        with pytest.raises(IndexError):
            _ = values[-3]
        values += [3]
        assert list(values) == [1, 2, 3]
        assert list(values[::-1]) == [3, 2, 1]


def test_nested_collections_survive_parent_dto_reload():
    with disk_collections():
        parents = DiskMap()
        parents["session"] = {"turns": DiskList()}
        parents["session"]["turns"].append({"action": "imported"})
        assert list(parents["session"]["turns"]) == [{"action": "imported"}]


def test_equal_map_keys_do_not_depend_on_python_object_sharing():
    with disk_collections():
        shared = "a synthetic identity " * 3
        key = (shared, shared)
        equivalent = (shared, "".join(["a synthetic identity " for _ in range(3)]))
        assert key == equivalent and key[1] is not equivalent[1]
        values = DiskMap()
        values[key] = "found"
        assert values[equivalent] == "found"
        values[equivalent] = "updated"
        assert len(values) == 1


def test_result_owner_can_be_closed_explicitly():
    with disk_collections() as store:
        result = ImportResult(False, {"sessions": DiskList()}, _store=store)
    assert store.path.exists()
    result.close()
    result.close()
    assert not store.root.exists()


def test_json_stream_does_not_materialize_large_result():
    with disk_collections() as store:
        sessions = DiskList({"session_id": str(index), "safe_code": "x" * 4096} for index in range(3000))
        result = ImportResult(False, {"sessions": sessions})
        tracemalloc.start()
        try:
            size = sum(len(chunk) for chunk in iter_json(result))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert size > 12_000_000
        assert peak < 2_000_000
    store.close()


def test_frozen_session_views_do_not_reopen_the_jsonl_plan(tmp_path):
    scan = SessionScanEntry(
        session_id="empty",
        adapter_generation="current",
        selected=True,
        complete_scan=True,
        tombstone=True,
        logical_turn_count=0,
        record_count=0,
        creation_time_ns=None,
        session_context_fingerprint="a" * 64,
        warnings=(),
    )
    path = tmp_path / "discovery.jsonl"
    with DiscoveryPlanWriter(path) as writer:
        writer.write_session_scan(scan)
    with disk_collections() as store:
        reader = DiscoveryPlanReader(path).freeze()
        path.unlink()
        assert list(reader.for_session("empty").iter_session_scans()) == [scan]
        assert list(reader.for_session("absent").iter_entries()) == []
    store.close()


def test_streamed_state_many_small_turns_has_bounded_memory_and_linear_work(tmp_path, monkeypatch):
    from mlflow_import.mlflow import attempts

    original = attempts.iter_canonical_json
    operations = 0

    def counted(value):
        nonlocal operations
        for chunk in original(value):
            operations += 1
            yield chunk

    monkeypatch.setattr(attempts, "iter_canonical_json", counted)
    measurements = []
    with disk_collections() as store:
        for count in (0, 1000, 2000):
            turns = DiskList(
                {"turn_key": f"turn-{index:08}", "metadata": "synthetic-" + "x" * 1024} for index in range(count)
            )
            operations = 0
            tracemalloc.start()
            try:
                _, size = attempts.write_state_json(tmp_path / f"state-{count}.json", {"turns": turns})
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            measurements.append((operations, size))
            assert peak < 2_000_000
    base, small, large = measurements
    assert large[0] - base[0] <= 2 * (small[0] - base[0]) + 1
    assert large[1] - base[1] <= 2 * (small[1] - base[1]) + 1
    store.close()


def test_streamed_state_envelope_fails_before_writing_oversized_chunk(tmp_path, monkeypatch):
    from mlflow_import.errors import ItemImportError
    from mlflow_import.mlflow import attempts

    monkeypatch.setattr(attempts, "MAX_STATE_BYTES", 2048)
    path = tmp_path / "bounded.json"
    with disk_collections() as store:
        with pytest.raises(ItemImportError) as caught:
            attempts.write_state_json(
                path, DiskList({"turn_key": str(index), "metadata": "x" * 1024} for index in range(100))
            )
        assert caught.value.code == "attempt_state_too_large"
        assert path.stat().st_size <= 2048
    store.close()


def test_streamed_canonical_state_matches_existing_wire_bytes():
    from mlflow_import.content.canonical import canonical_json, iter_canonical_json

    with disk_collections() as store:
        value = {
            "turns": DiskList([{"unicode": "é", "time": 42}, {"cost": 1.5}]),
            "bases": DiskMap({"z": {}, "a": {"absent": None}}),
        }
        assert b"".join(iter_canonical_json(value)) == canonical_json(value)
    store.close()
