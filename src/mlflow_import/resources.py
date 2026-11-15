"""Invocation-owned temporary storage with admission before allocation.

The ledger charges rounded file capacity plus a metadata page per file/directory.
SQLite main databases are limited to the remaining shared capacity before every
statement. Journals/WAL/mmap are disabled on disposable stores; transient SQL
work stays in memory under the qualified process-wide SQLite hard heap ceiling.
No accounting operation enumerates a cumulative directory tree.
"""

from __future__ import annotations

import base64
import configparser
import errno
import http.client
import io
import json
import os
import re
import shutil
import sqlite3
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import weakref
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar, cast
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse
from urllib.request import url2pathname

from mlflow_import.errors import ConfigurationError, ImporterError, ItemImportError

DEFAULT_TEMP_BYTES = 16 * 1024**3
SQLITE_HEAP_BYTES = 64 * 1024**2
PAGE = 4096
COPY_BATCH = 64 * 1024
MAX_TEMP_ENTRIES = 8192
MAX_TEMP_DEPTH = 16
MAX_SQL_CONNECTIONS = 64
MAX_IO_HANDLES = 128
MAX_ARTIFACT_LIST_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_LIST_PAGES = 128
MAX_ARTIFACT_LIST_ENTRIES = 8192
ARTIFACT_HTTP_TIMEOUT = 5.0
ARTIFACT_PROXY_PATH = "/api/2.0/mlflow-artifacts/artifacts"
_T = TypeVar("_T")
_CURRENT: ContextVar[TemporaryStorage | None] = ContextVar("temporary_storage", default=None)
_SQL_PREFIX = re.compile(r"\A(?:[\s;]+|--[^\n]*(?:\n|\Z)|/\*[\s\S]*?\*/)*")
_CREATE_LOCK = threading.RLock()


def _statement(sql: str) -> str:
    return _SQL_PREFIX.sub("", sql, count=1)


def _allocate(operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    except OSError as error:
        if error.errno in {errno.ENOSPC, errno.EDQUOT}:
            budget = current_storage()
            raise TemporaryStorageError(bool(budget and budget.phase2)) from None
        raise


def _private_create(operation: Callable[[], _T]) -> _T:
    """Create private files atomically as 0600; restore umask before SDK I/O."""
    with _CREATE_LOCK:
        previous = os.umask(0o077)
        try:
            return operation()
        finally:
            os.umask(previous)


class TemporaryStorageError(ImporterError):
    def __init__(
        self, phase2: bool = False, *, owner: TemporaryStorage | None = None, unavailable: bool = False
    ) -> None:
        if budget := owner or current_storage():
            budget.failed = True
        super().__init__(
            "temporary_storage_unavailable" if unavailable else "temporary_storage_exhausted",
            "private temporary storage could not be accessed"
            if unavailable
            else "the shared temporary storage or SQLite workspace limit was exhausted",
            1 if phase2 else 2,
        )


def current_storage() -> TemporaryStorage | None:
    return _CURRENT.get()


def _sqlite_failure(error: BaseException, budget: TemporaryStorage) -> TemporaryStorageError | None:
    code = (getattr(error, "sqlite_errorcode", 0) or 0) & 255
    if isinstance(error, MemoryError) or code in {sqlite3.SQLITE_FULL, sqlite3.SQLITE_NOMEM}:
        return TemporaryStorageError(budget.phase2, owner=budget)
    if code == sqlite3.SQLITE_IOERR:
        return TemporaryStorageError(budget.phase2, owner=budget, unavailable=True)
    return None


def _capacity(size: int) -> int:
    return PAGE + ((size + PAGE - 1) // PAGE) * PAGE


class TemporaryStorage:
    def __init__(self, max_bytes: int, directory: Path | None) -> None:
        if max_bytes < PAGE:
            raise TemporaryStorageError()
        self.max_bytes = max_bytes
        self.used_bytes = self.peak_bytes = 0
        self.sql_statements = 0
        self.peak_entries = 0
        self.phase2 = False
        self.failed = False
        self.closed = False
        self.sqlite_heap_bytes = SQLITE_HEAP_BYTES
        self.allocations: dict[Path, int] = {}
        self.by_directory: dict[Path, set[Path]] = {}
        self.roots: set[Path] = set()
        self.handles: weakref.WeakSet[Any] = weakref.WeakSet()
        self.connections: weakref.WeakSet[Any] = weakref.WeakSet()
        self.pins: dict[Path, int] = {}
        self.pins_by_directory: dict[Path, set[Path]] = {}
        self.pending_files: set[Path] = set()
        self.pending_trees: set[Path] = set()
        self.lock = threading.RLock()
        self.retained: Path | None = None
        self.serial = 0
        self.space_until_check = 0
        try:
            self.root = Path(
                _private_create(lambda: tempfile.mkdtemp(prefix="mlflow-import-workspace-", dir=directory))
            ).resolve()
            self.roots.add(self.root)
            self.resize(self.root, 0)
        except BaseException as error:
            if hasattr(self, "root"):
                shutil.rmtree(self.root, ignore_errors=True)
            if not isinstance(error, OSError):
                raise
            raise ConfigurationError(
                "temporary_storage_unavailable", "private planning storage could not be initialized"
            ) from None

    def resize(self, path: Path, size: int, *, admitted: bool = False) -> None:
        path = path.absolute()
        with self.lock:
            if self.failed and not admitted:
                raise TemporaryStorageError(self.phase2, owner=self)
            old = self.allocations.get(path, 0)
            new = _capacity(size)
            growth = new - old
            if growth > 0:
                if self.used_bytes + growth > self.max_bytes:
                    raise TemporaryStorageError(self.phase2, owner=self)
                if growth >= self.space_until_check and not admitted:
                    if shutil.disk_usage(self.root).free < growth:
                        raise TemporaryStorageError(self.phase2, owner=self)
                    self.space_until_check = 1024 * 1024
                self.space_until_check -= growth
            if path not in self.allocations:
                ancestors = [] if path in self.roots else list(path.parents)
                for index, parent in enumerate(ancestors):
                    if parent == self.root or parent in self.roots:
                        ancestors = ancestors[: index + 1]
                        break
                if len(self.allocations) >= MAX_TEMP_ENTRIES or len(ancestors) > MAX_TEMP_DEPTH:
                    raise TemporaryStorageError(self.phase2, owner=self)
                for parent in ancestors:
                    self.by_directory.setdefault(parent, set()).add(path)
                    if parent == self.root or parent in self.roots:
                        break
            self.allocations[path] = new
            self.used_bytes += growth
            self.peak_bytes = max(self.peak_bytes, self.used_bytes)
            self.peak_entries = max(self.peak_entries, len(self.allocations))

    def forget(self, path: Path) -> None:
        path = path.absolute()
        with self.lock:
            for item in (path, *tuple(self.by_directory.get(path, ()))):
                self.used_bytes -= self.allocations.pop(item, 0)
                self.by_directory.pop(item, None)
                for parent in item.parents:
                    entries = self.by_directory.get(parent)
                    if entries is not None:
                        entries.discard(item)
                        if not entries:
                            self.by_directory.pop(parent, None)
                    if parent == self.root or parent in self.roots:
                        break
            self.roots.discard(path)

    def adopt(self, path: Path) -> None:
        path = path.resolve()
        if not any(path.is_relative_to(root) for root in self.roots):
            self.roots.add(path)
        try:
            self.resize(path, 0)
        except BaseException:
            shutil.rmtree(path, ignore_errors=True)
            self.roots.discard(path)
            raise

    def pin(self, path: Path) -> bool:
        if path not in self.allocations:
            return False  # Original source databases are never owned temporary files.
        count = self.pins.get(path, 0)
        self.pins[path] = count + 1
        if count == 0:
            for parent in path.parents:
                self.pins_by_directory.setdefault(parent, set()).add(path)
                if parent in self.roots:
                    break
        return True

    def unpin(self, path: Path) -> None:
        count = self.pins.get(path, 0)
        if count > 1:
            self.pins[path] = count - 1
            return
        if not count:
            return
        self.pins.pop(path)
        ancestors = []
        for parent in path.parents:
            ancestors.append(parent)
            entries = self.pins_by_directory.get(parent)
            if entries is not None:
                entries.discard(path)
                if not entries:
                    self.pins_by_directory.pop(parent, None)
            if parent in self.roots:
                break
        if path in self.pending_files:
            self.pending_files.discard(path)
            remove_file(path, owner=self)
        for parent in ancestors:
            if parent in self.pending_trees and not self.pins_by_directory.get(parent):
                self.pending_trees.discard(parent)
                remove_tree(parent, owner=self)

    @contextmanager
    def new_entry(self) -> Iterator[Path]:
        """Admit a directory/inode before asking the OS to create it."""
        with self.lock:
            self.serial += 1
            reservation = self.root / f".allocation-{self.serial}"
            self.resize(reservation, 0)
            try:
                yield reservation
            finally:
                self.forget(reservation)

    def cleanup(self, keep: Path | None = None) -> None:
        keep = keep.absolute() if keep is not None else None
        failed = False
        for handle in list(self.handles):
            if keep is None or not handle.resource_path.is_relative_to(keep):
                try:
                    handle.close()
                except (OSError, sqlite3.Error, ImporterError):
                    failed = True
        for root in tuple(self.roots):
            try:
                if keep is not None and keep.is_relative_to(root):
                    # One final cleanup walk, never in record/statement accounting.
                    for child in root.iterdir():
                        if child != keep:
                            remove_tree(child, owner=self) if child.is_dir() else remove_file(child, owner=self)
                else:
                    shutil.rmtree(root)
                    self.forget(root)
            except FileNotFoundError:
                self.forget(root)
            except OSError:
                failed = True
        # Directly adopted scratch files (unit/public helper consumers).
        for path in tuple(self.allocations):
            if path.exists() and path.is_file() and not any(path.is_relative_to(root) for root in self.roots):
                remove_file(path, owner=self)
        if keep is None and not self.allocations:
            self.closed = True
            self.pins.clear()
            self.pins_by_directory.clear()
            self.pending_files.clear()
            self.pending_trees.clear()
        if failed:
            raise ConfigurationError(
                "temporary_storage_cleanup_failed",
                "private temporary storage could not be fully removed",
                exit_status=1,
            )


@contextmanager
def temporary_storage(
    max_bytes: int | None = None, *, directory: Path | None = None, retain_result: bool = False
) -> Iterator[TemporaryStorage]:
    existing = current_storage()
    if existing is not None:
        yield existing
        return
    if max_bytes is None:
        try:
            max_bytes = int(os.environ.get("MLFLOW_IMPORT_TEMP_BYTES", DEFAULT_TEMP_BYTES))
        except ValueError:
            raise ConfigurationError(
                "invalid_temporary_storage_limit", "the temporary storage limit must be a positive integer"
            ) from None
    if max_bytes <= 0:
        raise ConfigurationError(
            "invalid_temporary_storage_limit", "the temporary storage limit must be a positive integer"
        )
    budget: TemporaryStorage | None = None
    token = None
    completed = False
    try:
        budget = TemporaryStorage(max_bytes, directory)
        token = _CURRENT.set(budget)
        # SQLite's public pragma can lower, but cannot restore/raise, a hard
        # limit. This finite process envelope is part of the one-shot profile,
        # not an attempt to control another dynamically loaded SQLite library.
        check = None
        try:
            check = sqlite3.connect(":memory:")
            limit = check.execute(f"PRAGMA hard_heap_limit={SQLITE_HEAP_BYTES}").fetchone()
            if limit is None or not 0 < limit[0] <= SQLITE_HEAP_BYTES:
                raise ConfigurationError(
                    "sqlite_resource_profile_unsupported", "the SQLite heap envelope could not be enforced"
                )
            budget.sqlite_heap_bytes = limit[0]
        except (sqlite3.Error, MemoryError):
            raise ConfigurationError(
                "temporary_storage_unavailable", "private planning storage could not be initialized"
            ) from None
        finally:
            if check is not None:
                check.close()
        yield budget
        completed = True
    finally:
        try:
            if budget is not None:
                budget.cleanup(budget.retained if completed and retain_result else None)
        finally:
            if token is not None:
                _CURRENT.reset(token)


def make_temp_dir(*, prefix: str, dir: Path | str | None = None) -> Path:
    budget = current_storage()
    if budget is None:
        return Path(_allocate(lambda: _private_create(lambda: tempfile.mkdtemp(prefix=prefix, dir=dir)))).resolve()
    with budget.new_entry() as reservation:
        path = Path(
            _allocate(
                lambda: _private_create(
                    lambda: tempfile.mkdtemp(prefix=prefix, dir=dir if dir is not None else budget.root)
                )
            )
        ).resolve()
        budget.forget(reservation)
        budget.adopt(path)
        return path


def open_temp_file(
    *, prefix: str, suffix: str = "", dir: Path | str | None = None, mode: str = "wb", **kwargs: Any
) -> tuple[Any, Path]:
    budget = current_storage()
    # Know the name and admit it before C-level open. A file object, not a bare
    # integer descriptor, owns the OS resource at every interrupt boundary.
    directory = Path(dir) if dir is not None else budget.root if budget else Path(tempfile.gettempdir())
    for _ in range(4):
        path = directory / (prefix + uuid.uuid4().hex + suffix)
        if path.exists() or (budget is not None and path.absolute() in budget.allocations):
            continue
        return open_file(path, mode.replace("w", "x"), **kwargs), path
    raise TemporaryStorageError(bool(budget and budget.phase2), owner=budget)


@contextmanager
def temporary_directory(*, prefix: str, dir: Path | str | None = None) -> Iterator[str]:
    path = make_temp_dir(prefix=prefix, dir=dir)
    try:
        yield str(path)
    finally:
        remove_tree(path)


def remove_file(path: Path, *, owner: TemporaryStorage | None = None) -> None:
    path = path.resolve()
    budget = owner or current_storage()
    if budget and budget.pins.get(path):
        budget.pending_files.add(path)
        return
    path.unlink(missing_ok=True)
    if budget:
        budget.forget(path)


def remove_tree(path: Path, *, ignore_errors: bool = False, owner: TemporaryStorage | None = None) -> None:
    path = path.resolve()
    budget = owner or current_storage()
    if budget and (budget.pins.get(path) or budget.pins_by_directory.get(path)):
        budget.pending_trees.add(path)
        return
    if path.is_file():
        remove_file(path, owner=budget)
        return
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError:
        if not ignore_errors:
            raise
        return
    if budget:
        budget.forget(path)


class AccountedFile:
    def __init__(self, handle: Any, path: Path, budget: TemporaryStorage, *, writable: bool = True) -> None:
        self.handle, self.resource_path, self.budget = handle, path.resolve(), budget
        self.writable = writable
        self.pinned = False
        if len(budget.handles) >= MAX_IO_HANDLES:
            handle.close()
            raise TemporaryStorageError(budget.phase2, owner=budget)
        self.pinned = budget.pin(self.resource_path)
        budget.handles.add(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.handle, name)

    def write(self, value: Any) -> int:
        self._check()
        if not self.writable:
            raise io.UnsupportedOperation("the source stream is read-only")
        length = (
            len(value.encode(self.handle.encoding or "utf-8"))
            if isinstance(value, str)
            else len(value)
            if isinstance(value, (bytes, bytearray))
            else memoryview(value).nbytes
        )
        end = self.handle.tell() + length
        current = self.budget.allocations[self.resource_path] - PAGE
        self.budget.resize(self.resource_path, max(current, end))
        try:
            return int(self.handle.write(value))
        except OSError as error:
            if error.errno in {errno.ENOSPC, errno.EDQUOT}:
                raise TemporaryStorageError(self.budget.phase2, owner=self.budget) from None
            raise

    def _check(self) -> None:
        if self.budget.closed:
            raise ValueError("the temporary storage context is closed")
        if self.budget.failed:
            raise TemporaryStorageError(self.budget.phase2, owner=self.budget)
        from mlflow_import.retry import _BUDGET

        if deadline := _BUDGET.get():
            deadline.check()

    def read(self, size: int = -1) -> Any:
        self._check()
        value = self.handle.read(8 * 1024 * 1024 + 1 if size < 0 else size)
        if size < 0 and len(value.encode("utf-8") if isinstance(value, str) else value) > 8 * 1024 * 1024:
            raise TemporaryStorageError(self.budget.phase2, owner=self.budget)
        return value

    def readline(self, size: int = -1) -> Any:
        self._check()
        return self.handle.readline(size)

    def __iter__(self) -> AccountedFile:
        return self

    def __next__(self) -> Any:
        self._check()
        return next(self.handle)

    def __enter__(self) -> AccountedFile:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.handle.close()
        except OSError as error:
            if error.errno in {errno.ENOSPC, errno.EDQUOT}:
                raise TemporaryStorageError(self.budget.phase2, owner=self.budget) from None
            raise
        finally:
            self.budget.handles.discard(self)
            if self.pinned:
                self.pinned = False
                self.budget.unpin(self.resource_path)

    def __del__(self) -> None:
        try:
            self.close()
        except (OSError, ImporterError, AttributeError):
            pass

    def flush(self) -> None:
        try:
            self.handle.flush()
        except OSError as error:
            if error.errno in {errno.ENOSPC, errno.EDQUOT}:
                raise TemporaryStorageError(self.budget.phase2, owner=self.budget) from None
            raise

    def writelines(self, values: Iterable[Any]) -> None:
        for value in values:
            self.write(value)

    def truncate(self, size: int | None = None) -> int:
        self._check()
        if not self.writable:
            raise io.UnsupportedOperation("the source stream is read-only")
        target = self.handle.tell() if size is None else size
        if target > self.budget.allocations[self.resource_path] - PAGE:
            self.budget.resize(self.resource_path, target)
        result = int(self.handle.truncate(target))
        self.budget.resize(self.resource_path, result)
        return result


def account_file(handle: Any, path: Path) -> Any:
    path = path.resolve()
    if budget := current_storage():
        try:
            budget.resize(path, max(budget.allocations.get(path.absolute(), PAGE) - PAGE, path.stat().st_size))
            return AccountedFile(handle, path, budget)
        except BaseException:
            handle.close()
            raise
    return handle


def open_file(path: Path, mode: str, **kwargs: Any) -> Any:
    path = path.resolve()
    budget = current_storage()
    if budget:
        if len(budget.handles) >= MAX_IO_HANDLES:
            raise TemporaryStorageError(budget.phase2, owner=budget)
        if path.exists() and path not in budget.allocations:
            raise ConfigurationError(
                "temporary_file_unsupported", "temporary writers cannot adopt an existing external file"
            )
        budget.resize(
            path, max(budget.allocations.get(path.absolute(), PAGE) - PAGE, path.stat().st_size if path.exists() else 0)
        )
    try:
        return account_file(_allocate(lambda: _private_create(lambda: path.open(mode, **kwargs))), path)
    except BaseException:
        if budget and not path.exists():
            budget.forget(path)
        raise


def account_reader(handle: Any, path: Path) -> Any:
    if budget := current_storage():
        if budget.failed:
            handle.close()
            raise TemporaryStorageError(budget.phase2, owner=budget)
        return AccountedFile(handle, path, budget, writable=False)
    return handle


def open_reader(path: Path, mode: str = "rb", **kwargs: Any) -> Any:
    if budget := current_storage():
        if budget.failed or len(budget.handles) >= MAX_IO_HANDLES:
            raise TemporaryStorageError(budget.phase2, owner=budget)
    return account_reader(path.open(mode, **kwargs), path)


def copy_artifact(repository: Any, relative: str, root: str, limit: int) -> Path:
    """Capped public-URI transfer for the qualified file and HTTP proxy profiles.

    SDK download helpers copy entire files before returning; they cannot enforce
    an admission budget. Read the public artifact URI in bounded chunks instead.
    HTTP proxy reads are always bounded, including standalone callers.
    """
    if is_remote_artifact_repository(repository):
        return _copy_remote_artifact(repository, relative, root, limit)
    budget = current_storage()
    if budget is None:
        return Path(repository.download_artifacts(relative, root))
    uri = getattr(repository, "artifact_uri", None)
    parsed = urlparse(uri) if isinstance(uri, str) else None
    if parsed is None or parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ConfigurationError(
            "artifact_resource_profile_unsupported", "bounded artifact transfer requires the qualified file URI profile"
        )
    base = Path(url2pathname(parsed.path)).resolve()
    source = (base / relative).resolve()
    if not source.is_relative_to(base) or not source.is_file():
        raise ItemImportError("artifact_verification_failed", "the artifact is unavailable within its public location")
    size = source.stat().st_size
    if size > limit:
        raise ItemImportError("artifact_verification_failed", "the artifact exceeds its declared envelope")
    target = Path(root) / source.name
    if budget:
        budget.resize(target, size)  # Reserve the known copy before writing any bytes.
    with open_reader(source) as reader, open_file(target, "wb") as writer:
        seen = 0
        while chunk := reader.read(COPY_BATCH):
            seen += len(chunk)
            if seen > limit:
                raise ItemImportError("artifact_verification_failed", "the artifact grew beyond its declared envelope")
            writer.write(chunk)
    return target


@dataclass(frozen=True)
class ArtifactEntry:
    path: str
    is_dir: bool
    file_size: int


def iter_artifacts(repository: Any, relative: str) -> Iterator[Any]:
    """Enumerate public file/proxy metadata without an SDK all-results list."""
    if is_remote_artifact_repository(repository):
        yield from _iter_remote_artifacts(repository, relative)
        return
    if current_storage() is None:
        yield from repository.list_artifacts(relative)
        return
    parsed = urlparse(getattr(repository, "artifact_uri", ""))
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ConfigurationError(
            "artifact_resource_profile_unsupported", "bounded artifact listing requires the qualified file URI profile"
        )
    base = Path(url2pathname(parsed.path)).resolve()
    directory = (base / relative).resolve()
    if not directory.is_relative_to(base):
        raise ItemImportError("artifact_verification_failed", "the artifact listing is outside its public location")
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ItemImportError("artifact_verification_failed", "artifact listing symlinks are unsupported")
                yield ArtifactEntry((Path(relative) / entry.name).as_posix(), entry.is_dir(), entry.stat().st_size)
    except FileNotFoundError:
        return


def ordered_directory(path: Path) -> Iterator[tuple[str, bool, bool]]:
    """Bounded deterministic source-header enumeration, including a wide directory."""
    from mlflow_import.storage import DiskList

    if current_storage() is None:
        # Configuration-time format detection must not create an unowned spool
        # before run_import establishes its ledger. Its header-only batch has a
        # separate finite metadata envelope. The CLI establishes a ledger before
        # source validation; standalone config callers can establish one too.
        batch: list[tuple[str, bool, bool]] = []
        try:
            with os.scandir(path) as entries:
                for item in entries:
                    if item.is_symlink():
                        continue
                    if len(batch) == MAX_TEMP_ENTRIES:
                        raise ConfigurationError(
                            "source_inventory_profile_unsupported",
                            "source inference exceeds the bounded header inventory envelope",
                        )
                    batch.append((item.name, item.is_dir(follow_symlinks=False), item.is_file(follow_symlinks=False)))
            yield from sorted(batch)
        except FileNotFoundError:
            return
        return
    try:
        with os.scandir(path) as entries:
            values = DiskList(
                (item.name, item.is_dir(follow_symlinks=False), item.is_file(follow_symlinks=False))
                for item in entries
                if not item.is_symlink()
            ).sorted_by(lambda item: item[0])
        yield from values
    except FileNotFoundError:
        return


def artifact_exists(repository: Any, relative: str) -> bool:
    if is_remote_artifact_repository(repository):
        _artifact_relative(relative)
        parent = str(PurePosixPath(relative).parent)
        return any(
            item.path == relative and not item.is_dir
            for item in iter_artifacts(repository, "" if parent == "." else parent)
        )
    if current_storage() is None:
        return any(
            item.path == relative and not item.is_dir for item in repository.list_artifacts(str(Path(relative).parent))
        )
    parsed = urlparse(getattr(repository, "artifact_uri", ""))
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ConfigurationError(
            "artifact_resource_profile_unsupported", "bounded artifact lookup requires the qualified file URI profile"
        )
    base = Path(url2pathname(parsed.path)).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ItemImportError("artifact_verification_failed", "the artifact is outside its public location")
    return path.is_file()


def is_remote_artifact_repository(repository: Any) -> bool:
    return urlparse(getattr(repository, "artifact_uri", "")).scheme in {"http", "https", "mlflow-artifacts"}


def _artifact_relative(path: str, *, empty: bool = False) -> str:
    decoded = unquote(path)
    if (
        (not path and not empty)
        or decoded != path
        or "\\" in path
        or "\x00" in path
        or "?" in path
        or "#" in path
        or any(ord(c) < 32 for c in path)
        or path.startswith("/")
        or any(part in {".", "..", ""} for part in path.split("/"))
        and path != ""
    ):
        raise ItemImportError("artifact_verification_failed", "the artifact path is outside the qualified envelope")
    return path


def validate_artifact_repository(repository: Any) -> None:
    """Check the public factory's resolved URI before any target mutation."""
    parsed = urlparse(getattr(repository, "artifact_uri", ""))
    if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
        return
    if parsed.scheme in {"http", "https"}:
        _artifact_proxy(repository)
        # Build auth/TLS now as well: unsupported auth and invalid certificate
        # configuration must not first fail after publishing an experiment.
        artifact_http_configuration()
        return
    raise ConfigurationError(
        "artifact_resource_profile_unsupported", "the artifact repository has no qualified bounded transport"
    )


def validate_artifact_uri(location: str) -> None:
    """Reject unqualified schemes before the SDK can initialize their adapters."""
    parsed = urlparse(location)
    if parsed.scheme == "mlflow-artifacts" and not parsed.netloc and not parsed.query and not parsed.fragment:
        _artifact_relative(parsed.path.lstrip("/"), empty=True)
        return
    from types import SimpleNamespace

    validate_artifact_repository(SimpleNamespace(artifact_uri=location))


def _artifact_proxy(repository: Any) -> tuple[str, str]:
    parsed = urlparse(getattr(repository, "artifact_uri", ""))
    prefix, separator, tail = parsed.path.partition(ARTIFACT_PROXY_PATH)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.params
        or not separator
        or (tail and not tail.startswith("/"))
    ):
        raise ConfigurationError(
            "artifact_resource_profile_unsupported", "bounded HTTP artifacts require the MLflow artifact proxy URI"
        )
    _artifact_relative(prefix.lstrip("/"), empty=True)
    root = tail.strip("/")
    _artifact_relative(root, empty=True)
    return f"{parsed.scheme}://{parsed.netloc}{prefix}{ARTIFACT_PROXY_PATH}", root


class _ArtifactNoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any) -> None:
        return None


def artifact_http_configuration() -> tuple[Any, dict[str, str]]:
    """Pinned standard MLflow Basic/Bearer and requests-compatible TLS settings.

    No private SDK credential helpers, ambient netrc, plugins, or redirects.
    Credential-file reads have a separate finite control-data envelope.
    """
    try:
        if os.environ.get("MLFLOW_TRACKING_AUTH") or os.environ.get("MLFLOW_TRACKING_AWS_SIGV4", "").lower() == "true":
            raise ValueError
        config = configparser.ConfigParser()
        credentials = Path.home() / ".mlflow" / "credentials"
        if credentials.exists():
            with credentials.open("rb") as handle:
                raw = handle.read(64 * 1024 + 1)
            if len(raw) > 64 * 1024:
                raise ValueError
            config.read_string(raw.decode("utf-8"))
        section = config["mlflow"] if config.has_section("mlflow") else {}
        username = os.environ.get("MLFLOW_TRACKING_USERNAME") or section.get("mlflow_tracking_username")
        password = os.environ.get("MLFLOW_TRACKING_PASSWORD") or section.get("mlflow_tracking_password")
        token = os.environ.get("MLFLOW_TRACKING_TOKEN")
        headers = {"Accept-Encoding": "identity", "User-Agent": "mlflow-import-bounded-artifacts"}
        if username and password:
            headers["Authorization"] = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        if any("\r" in value or "\n" in value for value in headers.values()):
            raise ValueError
        insecure = os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").lower() == "true"
        server_cert = os.environ.get("MLFLOW_TRACKING_SERVER_CERT_PATH")
        if insecure and server_cert:
            raise ValueError
        ca = (
            None
            if insecure
            else server_cert or os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
        )
        # Match requests' explicit verify bundle: replace the default trust
        # store, rather than add these roots to an already initialized store.
        if ca:
            context = (
                ssl.create_default_context(capath=ca) if os.path.isdir(ca) else ssl.create_default_context(cafile=ca)
            )
        else:
            context = ssl.create_default_context()
        if insecure:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        if cert := os.environ.get("MLFLOW_TRACKING_CLIENT_CERT_PATH"):
            context.load_cert_chain(certfile=cert)
        return urllib.request.build_opener(_ArtifactNoRedirect(), urllib.request.HTTPSHandler(context=context)), headers
    except (OSError, ValueError, configparser.Error, UnicodeError):
        raise ConfigurationError(
            "artifact_http_configuration_invalid", "the bounded HTTP artifact auth or TLS configuration is unsupported"
        ) from None


class _ArtifactReadFailure(Exception):
    def __init__(self, transient: bool) -> None:
        self.error_code = "TEMPORARILY_UNAVAILABLE" if transient else "PERMISSION_DENIED"
        super().__init__("the bounded HTTP artifact request failed")


@contextmanager
def _artifact_response(url: str) -> Iterator[Any]:
    from mlflow_import.retry import _BUDGET, check_budget

    check_budget()
    opener, headers = artifact_http_configuration()
    timeout = ARTIFACT_HTTP_TIMEOUT
    if budget := _BUDGET.get():
        timeout = min(timeout, max(0.001, budget.deadline - time.monotonic()))
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
            if response.status != 200 or response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise _ArtifactReadFailure(False)
            yield response
    except urllib.error.HTTPError as error:
        status = error.code
        error.close()
        raise _ArtifactReadFailure(status == 429 or 500 <= status < 600) from None
    except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException) as error:
        reason = getattr(error, "reason", error)
        raise _ArtifactReadFailure(not isinstance(reason, (ssl.SSLError, ssl.CertificateError))) from None


def _artifact_read(operation: Callable[[], _T]) -> _T:
    from mlflow_import.retry import RetryPolicy, retry_call

    try:
        return retry_call(operation, RetryPolicy())
    except _ArtifactReadFailure:
        raise ItemImportError("artifact_transport_failed", "the bounded HTTP artifact read failed") from None


def _response_length(response: Any, limit: int) -> int | None:
    length = response.headers.get("Content-Length")
    try:
        if length is not None and (int(length) < 0 or int(length) > limit):
            raise ValueError
    except ValueError:
        raise ItemImportError(
            "artifact_verification_failed", "the artifact response exceeds its declared envelope"
        ) from None
    return int(length) if length is not None else None


def _response_chunks(response: Any, limit: int) -> Iterator[bytes]:
    from mlflow_import.retry import check_budget

    length = _response_length(response, limit)
    seen = 0
    while True:
        check_budget()
        chunk = response.read1(min(COPY_BATCH, limit - seen + 1))
        if not chunk:
            break
        seen += len(chunk)
        if seen > limit:
            raise ItemImportError("artifact_verification_failed", "the artifact response exceeds its declared envelope")
        yield chunk
    if length is not None and seen != length:
        raise _ArtifactReadFailure(True)


def _copy_remote_artifact(repository: Any, relative: str, root: str, limit: int) -> Path:
    base, remote_root = _artifact_proxy(repository)
    _artifact_relative(relative)
    if limit < 0:
        raise ItemImportError("artifact_verification_failed", "the artifact envelope is invalid")
    target = Path(root) / PurePosixPath(relative).name
    url = base + "/" + quote("/".join(filter(None, (remote_root, relative))), safe="/")

    def download() -> Path:
        try:
            with _artifact_response(url) as response:
                length = _response_length(response, limit)
                if (budget := current_storage()) is not None and length is not None:
                    budget.resize(target, length)
                with open_file(target, "wb") as writer:
                    for chunk in _response_chunks(response, limit):
                        writer.write(chunk)
            return target
        except BaseException:
            remove_file(target)
            raise

    return _artifact_read(download)


def _iter_remote_artifacts(repository: Any, relative: str) -> Iterator[ArtifactEntry]:
    base, root = _artifact_proxy(repository)
    _artifact_relative(relative, empty=True)
    token = ""
    seen_tokens: set[str] = set()
    total = 0
    for _ in range(MAX_ARTIFACT_LIST_PAGES):
        params = {"path": "/".join(filter(None, (root, relative)))}
        if token:
            params["page_token"] = token

        def page(params: dict[str, str] = params) -> Any:
            with _artifact_response(base + "?" + urlencode(params)) as response:
                body = b"".join(_response_chunks(response, MAX_ARTIFACT_LIST_BYTES))
            try:
                value = json.loads(body)
            except (ValueError, UnicodeError, RecursionError):
                raise ItemImportError("artifact_listing_invalid", "the proxy artifact listing is malformed") from None
            return value

        value = _artifact_read(page)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("files", []), list)
            or set(value) - {"files", "next_page_token", "root_uri"}
        ):
            raise ItemImportError("artifact_listing_invalid", "the proxy artifact listing is malformed")
        files = value.get("files", [])
        total += len(files)
        if total > MAX_ARTIFACT_LIST_ENTRIES:
            raise ItemImportError("artifact_listing_exhausted", "the proxy artifact listing exceeds its entry envelope")
        names: set[str] = set()
        for entry in files:
            try:
                if not isinstance(entry, dict):
                    raise ValueError
                name = entry["path"]
                is_dir = entry["is_dir"]
                size = entry.get("file_size", 0)
                if (
                    not isinstance(name, str)
                    or "/" in name
                    or name in names
                    or not isinstance(is_dir, bool)
                    or isinstance(size, bool)
                    or not isinstance(size, (str, int))
                    or int(size) < 0
                ):
                    raise ValueError
                _artifact_relative(name)
                names.add(name)
                yield ArtifactEntry("/".join(filter(None, (relative, name))), is_dir, int(size))
            except (KeyError, TypeError, ValueError):
                raise ItemImportError("artifact_listing_invalid", "the proxy artifact listing is malformed") from None
        token = value.get("next_page_token", "")
        if not isinstance(token, str) or len(token) > 4096 or token in seen_tokens:
            raise ItemImportError("artifact_listing_exhausted", "the proxy artifact listing did not make progress")
        if not token:
            return
        seen_tokens.add(token)
    raise ItemImportError("artifact_listing_exhausted", "the proxy artifact listing exceeds its page envelope")


def copy_file(source: Path, target: Path) -> None:
    if budget := current_storage():
        budget.resize(target, source.stat().st_size)
    with open_reader(source) as reader, open_file(target, "wb") as writer:
        while chunk := reader.read(COPY_BATCH):
            writer.write(chunk)


def configure_reader(connection: sqlite3.Connection) -> None:
    if current_storage():
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("PRAGMA mmap_size=0")


class QuotaCursor(sqlite3.Cursor):
    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        cast(QuotaConnection, connection).cursors.add(self)

    def close(self) -> None:
        try:
            super().close()
        finally:
            cast(QuotaConnection, self.connection).cursors.discard(self)

    def execute(self, sql: str, parameters: Any = (), /) -> QuotaCursor:
        connection: QuotaConnection = self.connection  # type: ignore[assignment]
        attached_path = None
        if _statement(sql).upper().startswith("ATTACH"):
            uri = urlparse(parameters[0]) if parameters and isinstance(parameters[0], str) else None
            if uri is None or uri.scheme != "file" or parse_qs(uri.query).get("mode") != ["ro"]:
                raise ConfigurationError(
                    "temporary_sql_unsupported", "private indexes may attach only read-only databases"
                )
            attached_path = Path(url2pathname(uri.path)).resolve()
        result = cast(
            QuotaCursor,
            connection.guarded(sql, lambda statement: super(QuotaCursor, self).execute(statement, parameters)),
        )
        if attached_path is not None and connection.budget.pin(attached_path):
            connection.attachments.append(attached_path)
        return result

    def executemany(self, sql: str, parameters: Iterable[Any], /) -> QuotaCursor:
        for values in parameters:
            self.execute(sql, values)
        return self

    def _reading(self, operation: Callable[[], Any]) -> Any:
        connection = cast(QuotaConnection, self.connection)
        try:
            if connection.budget.failed:
                raise TemporaryStorageError(connection.budget.phase2, owner=connection.budget)
            from mlflow_import.retry import _BUDGET

            if deadline := _BUDGET.get():
                deadline.check()
            return operation()
        except (MemoryError, sqlite3.Error) as error:
            if connection.interrupted is not None:
                raise connection.interrupted from None
            if failure := _sqlite_failure(error, connection.budget):
                raise failure from None
            raise

    def fetchone(self) -> Any:
        return self._reading(lambda: super(QuotaCursor, self).fetchone())

    def __next__(self) -> Any:
        return self._reading(lambda: super(QuotaCursor, self).__next__())

    def _batch(self, count: int, *, complete: bool = False) -> list[Any]:
        connection = cast(QuotaConnection, self.connection)
        if count > 1024 or count < 0:
            raise TemporaryStorageError(connection.budget.phase2, owner=connection.budget)
        rows: list[Any] = []
        size = 0
        for _ in range(count + int(complete)):
            row = self.fetchone()
            if row is None:
                break
            size += sum(
                len(value.encode("utf-8")) if isinstance(value, str) else len(value) if isinstance(value, bytes) else 16
                for value in row
            )
            if size > 4 * 1024 * 1024 or len(rows) == count:
                raise TemporaryStorageError(connection.budget.phase2, owner=connection.budget)
            rows.append(row)
        return rows

    def fetchmany(self, size: int | None = None) -> list[Any]:
        return self._batch(self.arraysize if size is None else size)

    def fetchall(self) -> list[Any]:
        return self._batch(1024, complete=True)

    def executescript(self, script: str, /) -> QuotaCursor:
        self.connection.commit()
        pending = ""
        for part in script.splitlines(keepends=True):
            for char in part:
                pending += char
                if char == ";" and sqlite3.complete_statement(pending):
                    self.execute(pending)
                    pending = ""
        if pending.strip():
            self.execute(pending)
        return self


class QuotaConnection(sqlite3.Connection):
    def __init__(
        self, database: Any, *args: Any, budget: TemporaryStorage, writable: bool = True, **kwargs: Any
    ) -> None:
        self.budget, self.writable = budget, writable
        self.cursors: weakref.WeakSet[QuotaCursor] = weakref.WeakSet()
        if len(budget.connections) >= MAX_SQL_CONNECTIONS or len(budget.handles) >= MAX_IO_HANDLES:
            raise TemporaryStorageError(budget.phase2, owner=budget)
        self.resource_path = (
            Path(url2pathname(urlparse(str(database)).path)).resolve()
            if str(database).startswith("file:")
            else Path(database).resolve()
        )
        if writable:
            if (
                self.resource_path.exists()
                and self.resource_path.stat().st_size
                and self.resource_path not in budget.allocations
                and not any(self.resource_path.is_relative_to(root) for root in budget.roots)
            ):
                raise ConfigurationError(
                    "temporary_sql_unsupported", "private storage cannot adopt an existing external database"
                )
            budget.resize(self.resource_path, self.resource_path.stat().st_size if self.resource_path.exists() else 0)
        initialize = super().__init__
        if writable:
            _private_create(lambda: initialize(database, *args, **kwargs))
        else:
            initialize(database, *args, **kwargs)
        self.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 10)
        self.main_pinned = budget.pin(self.resource_path)
        self.attachments: list[Path] = []
        budget.handles.add(self)
        budget.connections.add(self)
        self.file_cap = 2**30
        self.interrupted: ImporterError | None = None
        for statement in (("PRAGMA page_size=4096", "PRAGMA journal_mode=OFF") if writable else ()) + (
            "PRAGMA temp_store=MEMORY",
            "PRAGMA cache_size=-2048",
            "PRAGMA mmap_size=0",
        ):
            super().execute(statement)

        def progress() -> int:
            from mlflow_import.retry import check_budget

            try:
                check_budget()
            except ImporterError as error:
                self.interrupted = error
                return 1
            return 0

        self.set_progress_handler(progress, 10_000)

    def cursor(self, factory: Any = None) -> Any:
        if factory is not None and (not isinstance(factory, type) or not issubclass(factory, QuotaCursor)):
            raise ConfigurationError("temporary_sql_unsupported", "private storage requires an accounted cursor")
        return super().cursor(factory or QuotaCursor)

    def execute(self, sql: str, parameters: Any = (), /) -> QuotaCursor:
        return cast(QuotaCursor, self.cursor().execute(sql, parameters))

    def executemany(self, sql: str, parameters: Iterable[Any], /) -> QuotaCursor:
        return cast(QuotaCursor, self.cursor().executemany(sql, parameters))

    def executescript(self, script: str, /) -> QuotaCursor:
        return cast(QuotaCursor, self.cursor().executescript(script))

    def guarded(self, sql: str, operation: Callable[[str], Any]) -> Any:
        normalized = " ".join(_statement(sql).upper().split())
        if normalized.startswith("PRAGMA"):
            compact = normalized.replace(" ", "")
            if "TEMP_STORE=" in compact:
                sql = "PRAGMA temp_store=MEMORY"
            elif "PAGE_SIZE=" in compact and self.writable:
                sql = "PRAGMA page_size=4096"
            elif "JOURNAL_MODE=" in compact and self.writable:
                sql = "PRAGMA journal_mode=OFF"
            elif "CACHE_SIZE=" in compact:
                sql = "PRAGMA cache_size=-2048"
            elif "MMAP_SIZE=" in compact:
                sql = "PRAGMA mmap_size=0"
            elif "MAX_PAGE_COUNT=" in compact:
                self.file_cap = int(normalized.split("=")[-1].rstrip(";"))
                sql = "PRAGMA max_page_count"
        if normalized.startswith("VACUUM"):
            raise ConfigurationError("temporary_sql_unsupported", "vacuum is not part of the private storage profile")
        with self.budget.lock:
            self.budget.sql_statements += 1
            try:
                if self.budget.failed:
                    raise TemporaryStorageError(self.budget.phase2, owner=self.budget)
                if self.budget.closed:
                    raise sqlite3.ProgrammingError("the temporary storage context is closed")
                from mlflow_import.retry import _BUDGET

                if deadline := _BUDGET.get():
                    deadline.check()
                if not self.writable:
                    return operation(sql)
                used = super().execute("PRAGMA page_count").fetchone()[0] * PAGE
                # Previously allocated (including free) pages remain charged.
                maximum = min(self.file_cap, (used + self.budget.max_bytes - self.budget.used_bytes) // PAGE)
                if maximum < max(1, used // PAGE):
                    raise TemporaryStorageError(self.budget.phase2, owner=self.budget)
                super().execute(f"PRAGMA max_page_count={maximum}")
                result = operation(sql)
                pages = super().execute("PRAGMA page_count").fetchone()[0] * PAGE
                self.budget.resize(self.resource_path, max(pages, self.resource_path.stat().st_size))
                return result
            except (MemoryError, sqlite3.Error) as error:
                if self.interrupted is not None:
                    raise self.interrupted from None
                if failure := _sqlite_failure(error, self.budget):
                    raise failure from None
                raise
            finally:
                if self.writable and self.resource_path.exists():
                    # Also account pages written by a statement which subsequently
                    # failed/rolled back. No recursive SQL or directory scan.
                    physical = self.resource_path.stat().st_size
                    charged = self.budget.allocations.get(self.resource_path, PAGE) - PAGE
                    if physical > charged:
                        self.budget.resize(self.resource_path, physical, admitted=True)

    def close(self) -> None:
        try:
            try:
                self.set_progress_handler(None, 0)
            except sqlite3.ProgrammingError:
                pass
            for cursor in tuple(self.cursors):
                cursor.close()
            super().close()
            self.interrupted = None
            if self.main_pinned:
                self.main_pinned = False
                self.budget.unpin(self.resource_path)
            for path in self.attachments:
                self.budget.unpin(path)
            self.attachments.clear()
        finally:
            self.budget.handles.discard(self)
            self.budget.connections.discard(self)

    def backup(self, *args: Any, **kwargs: Any) -> None:
        raise ConfigurationError(
            "temporary_sql_unsupported", "private storage does not support unaccounted database copies"
        )

    def blobopen(self, *args: Any, **kwargs: Any) -> Any:
        raise ConfigurationError(
            "temporary_sql_unsupported", "private storage does not support unaccounted blob handles"
        )

    def deserialize(self, *args: Any, **kwargs: Any) -> None:
        raise ConfigurationError(
            "temporary_sql_unsupported", "private storage does not accept replacement database images"
        )


def sqlite_connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
    if budget := current_storage():
        if database == "":
            directory = make_temp_dir(prefix="sqlite-validation-")
            database = directory / "index.db"
        kwargs["factory"] = lambda *a, **kw: QuotaConnection(*a, budget=budget, **kw)
        kwargs["cached_statements"] = min(64, kwargs.get("cached_statements", 64))
    try:
        return cast(sqlite3.Connection, sqlite3.connect(database, *args, **kwargs))
    except MemoryError:
        active = current_storage()
        raise TemporaryStorageError(bool(active and active.phase2)) from None


def reader_connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
    if budget := current_storage():
        parsed = urlparse(str(database))
        path = Path(url2pathname(parsed.path)).resolve() if parsed.scheme == "file" else Path(database).absolute()
        if path.is_file() and not any(path.is_relative_to(root) for root in budget.roots):
            with path.open("rb") as header:
                prefix = header.read(20)
            if (
                len(prefix) == 20
                and 2 in prefix[18:20]
                and not all(Path(str(path) + suffix).is_file() for suffix in ("-wal", "-shm"))
            ):
                raise ConfigurationError(
                    "source_wal_unavailable", "read-only WAL access requires existing WAL and shared-memory companions"
                )
        kwargs["factory"] = lambda *a, **kw: QuotaConnection(*a, budget=budget, writable=False, **kw)
        kwargs["cached_statements"] = min(64, kwargs.get("cached_statements", 64))
    try:
        return cast(sqlite3.Connection, sqlite3.connect(database, *args, **kwargs))
    except MemoryError:
        active = current_storage()
        raise TemporaryStorageError(bool(active and active.phase2)) from None
