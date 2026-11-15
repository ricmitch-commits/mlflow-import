"""Deterministic aggregate quota tests; no source or target user data."""

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlflow_import.resources import (
    TemporaryStorageError,
    copy_artifact,
    open_file,
    sqlite_connect,
    temporary_directory,
    temporary_storage,
)

pytestmark = pytest.mark.unit


def test_two_sqlite_stores_share_one_limit_and_cleanup(tmp_path):
    with temporary_storage(160 * 1024, directory=tmp_path) as budget:
        first = sqlite_connect(budget.root / "one.db", isolation_level=None)
        second = sqlite_connect(budget.root / "two.db", isolation_level=None)
        for db in (first, second):
            db.execute("CREATE TABLE payload(value BLOB)")
        first.execute("INSERT INTO payload VALUES(?)", (b"x" * 80_000,))
        with pytest.raises(TemporaryStorageError) as caught:
            second.execute("INSERT INTO payload VALUES(?)", (b"x" * 80_000,))
        assert caught.value.code == "temporary_storage_exhausted"
        assert budget.peak_bytes <= budget.max_bytes
        assert budget.used_bytes > 80_000
        root = budget.root
    assert not root.exists()
    assert not budget.connections and not budget.handles
    assert list(tmp_path.iterdir()) == []


def test_streaming_files_and_sqlite_aggregate_without_directory_walks(tmp_path, monkeypatch):
    with temporary_storage(192 * 1024, directory=tmp_path) as budget:
        db = sqlite_connect(budget.root / "index.db", isolation_level=None)
        db.execute("CREATE TABLE payload(value BLOB)")
        db.execute("INSERT INTO payload VALUES(?)", (b"x" * 64_000,))
        # Accounting is incremental. Directory enumeration is permitted only at cleanup.
        from pathlib import Path

        with monkeypatch.context() as patch:
            patch.setattr(Path, "rglob", lambda *args, **kwargs: pytest.fail("cumulative scan"))
            patch.setattr("os.scandir", lambda *args, **kwargs: pytest.fail("cumulative scan"))
            with open_file(budget.root / "payloads", "wb") as handle:
                with pytest.raises(TemporaryStorageError):
                    for _ in range(2000):
                        handle.write(b"x" * 1024)
        assert budget.peak_bytes <= budget.max_bytes
    assert list(tmp_path.iterdir()) == []


def test_active_metadata_entries_are_finite_and_released(tmp_path, monkeypatch):
    from mlflow_import import resources

    monkeypatch.setattr(resources, "MAX_TEMP_ENTRIES", 4)
    with temporary_storage(1024 * 1024, directory=tmp_path) as budget:
        for index in range(3):
            with open_file(budget.root / str(index), "wb") as handle:
                handle.write(b"x")
        with pytest.raises(TemporaryStorageError):
            open_file(budget.root / "fourth", "wb")
        assert budget.peak_entries == 4
    assert budget.used_bytes == 0 and not budget.allocations
    assert list(tmp_path.iterdir()) == []


def test_download_reserves_before_copy_and_never_uses_unbounded_sdk_transfer(tmp_path):
    source = tmp_path / "repository"
    source.mkdir()
    (source / "large.json").write_bytes(b"x" * 80_000)
    repository = SimpleNamespace(artifact_uri=source.as_uri(), download_artifacts=lambda *args: pytest.fail("SDK copy"))
    with temporary_storage(64 * 1024, directory=tmp_path) as budget:
        with temporary_directory(prefix="download-") as root:
            with pytest.raises(TemporaryStorageError):
                copy_artifact(repository, "large.json", root, 80_000)
        assert not list(budget.root.iterdir())
    assert (source / "large.json").stat().st_size == 80_000
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_interruption_closes_handles_and_removes_every_owned_temp(tmp_path, failure):
    with pytest.raises(failure):
        with temporary_storage(512 * 1024, directory=tmp_path) as budget:
            db = sqlite_connect(budget.root / "index.db")
            db.execute("CREATE TABLE payload(value)")
            handle = open_file(budget.root / "spool", "wb")
            handle.write(b"synthetic")
            raise failure()
    assert handle.closed
    with pytest.raises(sqlite3.ProgrammingError):
        db.execute("SELECT 1")
    assert list(tmp_path.iterdir()) == []


def test_sqlite_transients_are_memory_only_under_qualified_heap_ceiling(tmp_path):
    before = sqlite3.connect(":memory:")
    with temporary_storage(512 * 1024, directory=tmp_path) as budget:
        db = sqlite_connect(budget.root / "index.db")
        assert db.execute("PRAGMA temp_store").fetchone()[0] == 2
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "off"
        assert 0 < db.execute("PRAGMA hard_heap_limit").fetchone()[0] <= 64 * 1024 * 1024
        with pytest.raises(TemporaryStorageError):
            db.execute("SELECT randomblob(128 * 1024 * 1024)").fetchone()
    assert 0 < before.execute("PRAGMA hard_heap_limit").fetchone()[0] <= 64 * 1024 * 1024
    before.close()


@pytest.mark.parametrize("dry_run", [False, True])
def test_invocation_small_quota_fails_before_target_writes_and_releases_report(
    tmp_path, monkeypatch, config, import_dependencies, dry_run
):
    from dataclasses import replace

    from mlflow_import.importer import exit_status, run_import
    from tests.support.temporary_resources import isolate_workspaces

    home = isolate_workspaces(monkeypatch, tmp_path / "workspaces", 64 * 1024)
    result = run_import(replace(config, dry_run=dry_run), import_dependencies)
    assert exit_status(result) == 2, result.report
    assert result.report["errors"][0]["code"] == "temporary_storage_exhausted"
    assert not import_dependencies.gateway.mutations
    result.close()
    assert list(home.iterdir()) == []


def test_dry_run_success_and_real_pipeline_interruption_leave_no_workspace(
    tmp_path, monkeypatch, config, import_dependencies
):
    from dataclasses import replace

    from mlflow_import.importer import exit_status, run_import
    from mlflow_import.source.snapshot import SnapshotWriter
    from tests.support.temporary_resources import isolate_workspaces

    home = isolate_workspaces(monkeypatch, tmp_path / "workspaces", 8 * 1024 * 1024)
    result = run_import(replace(config, dry_run=True), import_dependencies)
    assert exit_status(result) == 0, result.report
    assert result.report["resources"]["temporary_peak_bytes"] <= 8 * 1024 * 1024
    result.close()
    assert list(home.iterdir()) == []

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(SnapshotWriter, "write_record", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_import(config, import_dependencies)
    assert list(home.iterdir()) == []


def test_reader_keeps_original_wal_and_database_outside_temp_accounting(tmp_path):
    from mlflow_import.resources import reader_connect

    source = tmp_path / "source.db"
    writer = sqlite3.connect(source)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE payload(value)")
    writer.execute("INSERT INTO payload VALUES('synthetic')")
    writer.commit()
    wal = source.with_name(source.name + "-wal")
    before = (source.read_bytes(), wal.read_bytes())
    try:
        with temporary_storage(256 * 1024, directory=tmp_path) as budget:
            reader = reader_connect(source.as_uri() + "?mode=ro", uri=True)
            assert reader.execute("SELECT value FROM payload").fetchone()[0] == "synthetic"
            assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert source not in budget.allocations and wal not in budget.allocations
        assert (source.read_bytes(), wal.read_bytes()) == before
    finally:
        writer.close()


def test_private_sqlite_configuration_cannot_enable_hidden_disk_growth(tmp_path):
    with temporary_storage(256 * 1024, directory=tmp_path) as budget:
        db = sqlite_connect(budget.root / "private.db")
        db.executescript("""-- preserve the admission profile even behind comments
            PRAGMA page_size=65536;
            PRAGMA temp_store = FILE;
            PRAGMA journal_mode=WAL;
            PRAGMA mmap_size=100000000;
            PRAGMA cache_size=-100000;
            CREATE TABLE payload(value);
        """)
        assert db.execute("PRAGMA page_size").fetchone()[0] == 4096
        assert db.execute("PRAGMA temp_store").fetchone()[0] == 2
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "off"
        assert db.execute("PRAGMA mmap_size").fetchone()[0] == 0
        assert db.execute("PRAGMA cache_size").fetchone()[0] == -2048
        assert [path.name for path in budget.root.iterdir()] == ["private.db"]


def test_closed_wal_without_companions_is_refused_without_creating_source_files(tmp_path):
    from mlflow_import.errors import ConfigurationError
    from mlflow_import.resources import reader_connect

    source = tmp_path / "closed.db"
    writer = sqlite3.connect(source)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE payload(value)")
    writer.commit()
    writer.close()
    assert not Path(str(source) + "-wal").exists()
    with temporary_storage(256 * 1024, directory=tmp_path):
        with pytest.raises(ConfigurationError) as caught:
            reader_connect(source.as_uri() + "?mode=ro", uri=True)
        assert caught.value.code == "source_wal_unavailable"
    assert list(tmp_path.iterdir()) == [source]


def test_cli_output_interruption_releases_retained_result_storage(tmp_path, monkeypatch, config, import_dependencies):
    from mlflow_import import cli
    from tests.support.temporary_resources import isolate_workspaces

    home = isolate_workspaces(monkeypatch, tmp_path / "workspaces", 8 * 1024 * 1024)

    def interrupt(result):
        yield "{"
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "iter_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli.main(
            ["import", "--experiment-name", "synthetic", "--db", str(config.db_path), "--output", "json"],
            import_dependencies,
        )
    assert list(home.iterdir()) == []


def test_nested_budget_does_not_reset_limit_and_exhaustion_is_terminal(tmp_path):
    with temporary_storage(64 * 1024, directory=tmp_path) as first:
        db = sqlite_connect(first.root / "db")
        with temporary_storage(1024 * 1024) as nested:
            assert nested is first and nested.max_bytes == 64 * 1024
            with pytest.raises(TemporaryStorageError):
                with open_file(first.root / "large", "wb") as handle:
                    handle.write(b"x" * 64_000)
        # Do not continue with a scratch database that may have had a failed
        # journal-free statement, even if a later close frees capacity.
        with pytest.raises(TemporaryStorageError):
            db.execute("CREATE TABLE after_failure(value)")
    assert not first.allocations and list(tmp_path.iterdir()) == []


def test_cleanup_finalizes_strongly_referenced_sql_cursors(tmp_path):
    with temporary_storage(512 * 1024, directory=tmp_path) as budget:
        db = sqlite_connect(budget.root / "db")
        db.executescript("CREATE TABLE values_(value); INSERT INTO values_ VALUES(1),(2);")
        cursor = db.execute("SELECT value FROM values_")
        assert cursor.fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError):
        cursor.fetchone()
    assert not budget.connections and not budget.handles
    assert list(tmp_path.iterdir()) == []


def test_attached_id_index_stays_charged_until_parent_closes(tmp_path):
    from mlflow_import.resources import reader_connect
    from mlflow_import.source.id_index import DiskIds

    source = tmp_path / "source.db"
    writer = sqlite3.connect(source)
    writer.execute("CREATE TABLE ids(id TEXT)")
    writer.execute("INSERT INTO ids VALUES('selected')")
    writer.commit()
    writer.close()
    with temporary_storage(512 * 1024, directory=tmp_path) as budget:
        ids = DiskIds(["selected"])
        parent = reader_connect(source.as_uri() + "?mode=ro", uri=True)
        clause, params = ids.predicate(parent, "id")
        charged = budget.used_bytes
        path, directory = ids.path, ids.root
        ids.close()
        assert path.exists() and directory.exists()
        assert budget.used_bytes == charged
        assert parent.execute("SELECT id FROM ids" + clause, params).fetchone()[0] == "selected"
        parent.close()
        assert not path.exists() and not directory.exists()
        assert budget.used_bytes < charged
    assert not budget.allocations and not budget.pins
    assert list(tmp_path.iterdir()) == [source]


def test_merged_source_header_has_a_finite_metadata_envelope(tmp_path, monkeypatch):
    import json

    from mlflow_import.errors import SourceCompatibilityError
    from mlflow_import.source import snapshot
    from mlflow_import.source.file_snapshot import create_file_snapshot
    from mlflow_import.sources.jsonl_common import JsonlExporter

    source = tmp_path / "gemini.jsonl"
    header = {"sessionId": "synthetic", "projectHash": "synthetic"}
    source.write_text(json.dumps(header) + "\n")
    monkeypatch.setattr(snapshot, "MAX_SESSION_HEADER_BYTES", 4096)
    with temporary_storage(8 * 1024 * 1024, directory=tmp_path) as budget:
        with create_file_snapshot(
            source, budget.root, JsonlExporter(source, "gemini"), lambda value, _: value
        ) as valid:
            assert valid.stats.session_count == 1
        source.write_text(json.dumps(header) + "\n" + json.dumps({"$set": {"summary": "x" * 8192}}) + "\n")
        with pytest.raises(SourceCompatibilityError) as caught:
            with create_file_snapshot(source, budget.root, JsonlExporter(source, "gemini"), lambda value, _: value):
                pass
        assert caught.value.code == "source_metadata_too_large"
    assert list(tmp_path.iterdir()) == [source]


def test_read_spool_is_not_unlinked_or_credited_while_reader_is_open(tmp_path):
    from mlflow_import import resources

    with temporary_storage(256 * 1024, directory=tmp_path) as budget:
        path = budget.root / "payloads"
        with open_file(path, "wb") as handle:
            handle.write(b"x" * 64_000)
        reader = resources.open_reader(path)
        charged = budget.used_bytes
        resources.remove_file(path)
        assert path.exists() and budget.used_bytes == charged
        assert reader.read(1) == b"x"
        reader.close()
        assert not path.exists() and budget.used_bytes < charged
    assert not budget.handles and not budget.pins and list(tmp_path.iterdir()) == []


def test_quota_failure_closes_listing_producer_before_exception_is_released(tmp_path, monkeypatch):
    from mlflow_import import resources
    from mlflow_import.storage import DiskList

    repository = tmp_path / "repository"
    repository.mkdir()
    for index in range(1000):
        (repository / f"entry-{index:04}").touch()
    original = resources.os.scandir
    opened = []

    class Listing:
        def __init__(self, *args, **kwargs):
            self.handle = original(*args, **kwargs)
            self.closed = False
            opened.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()
            self.closed = True

        def __iter__(self):
            return iter(self.handle)

    monkeypatch.setattr(resources.os, "scandir", Listing)
    with temporary_storage(64 * 1024, directory=tmp_path):
        with pytest.raises(TemporaryStorageError) as caught:
            DiskList(resources.iter_artifacts(SimpleNamespace(artifact_uri=repository.as_uri()), ""))
        assert caught.value.code == "temporary_storage_exhausted"
        assert opened[0].closed


def test_private_creation_permissions_and_umask_restore_on_exhaustion(tmp_path, monkeypatch):
    import errno
    import os

    from mlflow_import import resources

    previous = os.umask(0o022)
    try:
        with temporary_storage(512 * 1024, directory=tmp_path) as budget:
            handle, path = resources.open_temp_file(prefix="private-", dir=budget.root)
            assert path.stat().st_mode & 0o777 == 0o600
            handle.close()
            db = sqlite_connect(budget.root / "db")
            assert db.execute("PRAGMA journal_mode").fetchone()[0] == "off"
            assert (budget.root / "db").stat().st_mode & 0o777 == 0o600
            with monkeypatch.context() as patch:

                def fail(*args, **kwargs):
                    raise OSError(errno.ENOSPC, "synthetic failure")

                patch.setattr(Path, "open", fail)
                with pytest.raises(TemporaryStorageError):
                    open_file(budget.root / "failed", "wb")
        after = tmp_path / "after"
        after.touch()
        assert after.stat().st_mode & 0o777 == 0o644
    finally:
        os.umask(previous)


def test_sqlite_extended_io_error_is_sanitized_and_terminal(tmp_path):
    with temporary_storage(512 * 1024, directory=tmp_path) as budget:
        db = sqlite_connect(budget.root / "db")
        error = sqlite3.OperationalError("SYNTHETIC-PRIVATE-PATH")
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR_FSYNC

        def fail(statement):
            raise error

        with pytest.raises(TemporaryStorageError) as caught:
            db.guarded("SELECT 1", fail)
        assert caught.value.code == "temporary_storage_unavailable"
        assert "SYNTHETIC-PRIVATE-PATH" not in str(caught.value)
        with pytest.raises(TemporaryStorageError):
            db.execute("SELECT 1")
    assert list(tmp_path.iterdir()) == []


def test_stream_admission_counts_buffer_bytes_not_element_count(tmp_path):
    with temporary_storage(32 * 1024, directory=tmp_path) as budget:
        path = budget.root / "payload"
        with open_file(path, "wb", buffering=0) as handle:
            view = memoryview(bytearray(32 * 1024)).cast("I")
            assert len(view) < view.nbytes
            with pytest.raises(TemporaryStorageError):
                handle.write(view)
            assert path.stat().st_size == 0
    assert list(tmp_path.iterdir()) == []
