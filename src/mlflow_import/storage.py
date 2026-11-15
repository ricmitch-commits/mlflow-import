"""Private disk collections for executable plans and sanitized results.

Pickles here are internal DTOs created by this process, never source/target wire
data. The database is owner-only in an owner-only directory; external artifacts
continue to use validated versioned JSON. Collection references share one store.
"""

from __future__ import annotations

import copyreg
import io
import itertools
import json
import pickle
import sqlite3
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import Any, Generic, TypeVar, cast

from mlflow_import import resources
from mlflow_import.errors import ImporterError

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
_CURRENT: ContextVar[Store | None] = ContextVar("private_store", default=None)
_STORES: weakref.WeakValueDictionary[str, Store] = weakref.WeakValueDictionary()


def _dump(value: object) -> bytes:
    output = io.BytesIO()
    encoder = pickle.Pickler(output, protocol=5)
    encoder.dispatch_table = copyreg.dispatch_table.copy()
    encoder.dispatch_table[type(MappingProxyType({}))] = lambda value: (dict, (dict(value),))
    encoder.dump(value)
    return output.getvalue()


def _key(value: object) -> bytes:
    """Canonical internal identity key, independent of pickle memo sharing."""

    def normalized(item: object) -> object:
        if item is None or isinstance(item, (str, int)):
            return item
        if isinstance(item, tuple):
            return ["tuple", [normalized(part) for part in item]]
        if is_dataclass(item) and not isinstance(item, type):
            return [
                type(item).__module__ + "." + type(item).__qualname__,
                [[field.name, normalized(getattr(item, field.name))] for field in fields(item)],
            ]
        raise TypeError("private identity keys require strings, integers, tuples or identity DTOs")

    return json.dumps(normalized(value), ensure_ascii=True, separators=(",", ":")).encode("ascii")


class Store:
    def __init__(self) -> None:
        try:
            self.root = resources.make_temp_dir(prefix="mlflow-import-plan-")
            self.path = self.root / "catalog.db"
            self.db = resources.sqlite_connect(self.path, isolation_level=None)
            self.path.chmod(0o600)
            self.db.executescript(
                "PRAGMA journal_mode=OFF; PRAGMA cache_size=-2048;"
                "PRAGMA temp_store=FILE; PRAGMA max_page_count=1048576;"
            )
            self.db.executescript(
                "CREATE TABLE items(ordinal INTEGER PRIMARY KEY,bucket INTEGER NOT NULL,"
                "key BLOB,key_payload BLOB,value BLOB,UNIQUE(bucket,key));"
                "CREATE INDEX item_order ON items(bucket,ordinal);"
            )
        except (OSError, sqlite3.Error):
            if hasattr(self, "db"):
                self.db.close()
            if hasattr(self, "root"):
                resources.remove_tree(self.root, ignore_errors=True)
            from mlflow_import.errors import ConfigurationError

            raise ConfigurationError(
                "temporary_storage_unavailable", "private planning storage could not be initialized"
            ) from None
        self.serial = 0
        self._closed = False
        _STORES[str(self.path)] = self

    def table(self) -> str:
        self.serial += 1
        return str(self.serial)

    def close(self) -> None:
        """Release the private result store; failures remain visible to the owner."""
        if self._closed:
            return
        budget = getattr(self.db, "budget", None)
        self.db.close()
        if budget is not None and budget.closed:
            _STORES.pop(str(self.path), None)
            self._closed = True
            return
        resources.remove_file(self.path, owner=budget)
        if self.root.exists():
            self.root.rmdir()
        if budget is not None:
            budget.forget(self.root)
            if budget.retained == self.root:
                if resources.current_storage() is not budget:
                    budget.cleanup()
                budget.retained = None
        _STORES.pop(str(self.path), None)
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError, sqlite3.Error, ImporterError):
            pass


def current_store() -> Store:
    return _CURRENT.get() or Store()


@contextmanager
def disk_collections(store: Store | None = None) -> Iterator[Store]:
    store = store or Store()
    token = _CURRENT.set(store)
    try:
        yield store
    finally:
        _CURRENT.reset(token)


def _restore(path: str, table: str, kind: str) -> Any:
    store = _STORES.get(path)
    if store is None:
        raise ValueError("private collection owner is unavailable")
    result = DiskList.__new__(DiskList) if kind == "list" else DiskMap.__new__(DiskMap)
    result.store, result.table = store, table
    return result


class DiskList(list[T], Generic[T]):
    """List-compatible lazy cursor; JSON's Python iterator can stream it."""

    def __init__(self, values: Iterable[T] = (), *, store: Store | None = None) -> None:
        self.store = store or current_store()
        self.table = self.store.table()
        self.extend(values)

    def __reduce__(self) -> Any:
        return _restore, (str(self.store.path), self.table, "list")

    def __len__(self) -> int:
        return int(self.store.db.execute(f"SELECT COUNT(*) FROM items WHERE bucket={self.table}").fetchone()[0])

    def __iter__(self) -> Iterator[T]:
        for (value,) in self.store.db.execute(f"SELECT value FROM items WHERE bucket={self.table} ORDER BY ordinal"):
            yield cast(T, pickle.loads(value))

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            length = len(self)
            start, stop, step = index.indices(length)
            if step < 0:
                return DiskList(
                    itertools.islice(reversed(self), length - 1 - start, length - 1 - stop, -step), store=self.store
                )
            return DiskList(itertools.islice(self, start, stop, step), store=self.store)
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(index)
        row = self.store.db.execute(
            f"SELECT value FROM items WHERE bucket={self.table} ORDER BY ordinal LIMIT 1 OFFSET ?", (index,)
        ).fetchone()
        if row is None:
            raise IndexError(index)
        return pickle.loads(row[0])

    def __setitem__(self, index: Any, value: Any) -> None:
        if isinstance(index, slice):
            raise TypeError("plan slices are immutable; replace the collection")
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(index)
        row = self.store.db.execute(
            f"SELECT ordinal FROM items WHERE bucket={self.table} ORDER BY ordinal LIMIT 1 OFFSET ?", (index,)
        ).fetchone()
        if row is None:
            raise IndexError(index)
        self.store.db.execute("UPDATE items SET value=? WHERE ordinal=?", (_dump(value), row[0]))

    def append(self, value: T) -> None:
        self.store.db.execute(f"INSERT INTO items(bucket,value) VALUES({self.table},?)", (_dump(value),))

    def extend(self, values: Iterable[T]) -> None:
        iterator = iter(itertools.islice(self, len(self)) if values is self else values)
        try:
            for value in iterator:
                self.append(value)
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def clear(self) -> None:
        self.store.db.execute(f"DELETE FROM items WHERE bucket={self.table}")

    def update_from(self, index: int, transform: Callable[[T], T]) -> None:
        """Rewrite a cursor suffix with one seek instead of one OFFSET scan per item."""
        for ordinal, value in self.store.db.execute(
            f"SELECT ordinal,value FROM items WHERE bucket={self.table} ORDER BY ordinal LIMIT -1 OFFSET ?", (index,)
        ):
            self.store.db.execute(
                "UPDATE items SET value=? WHERE ordinal=?", (_dump(transform(pickle.loads(value))), ordinal)
            )

    def sorted_by(self, key: Callable[[T], str | int | bytes]) -> DiskList[T]:
        """External stable sort using SQLite's finite cache and file temp store."""
        scratch = self.store.table()
        try:
            self.store.db.executemany(
                f"INSERT INTO items(bucket,key_payload,value) VALUES({scratch},?,?)",
                ((key(item), _dump(item)) for item in self),
            )
            return DiskList(
                (
                    pickle.loads(row[0])
                    for row in self.store.db.execute(
                        f"SELECT value FROM items WHERE bucket={scratch} ORDER BY key_payload,ordinal"
                    )
                ),
                store=self.store,
            )
        finally:
            self.store.db.execute(f"DELETE FROM items WHERE bucket={scratch}")

    def __reversed__(self) -> Iterator[T]:
        for (value,) in self.store.db.execute(
            f"SELECT value FROM items WHERE bucket={self.table} ORDER BY ordinal DESC"
        ):
            yield cast(T, pickle.loads(value))

    def __iadd__(self, values: Iterable[Any], /) -> DiskList[Any]:
        self.extend(values)
        return self

    def __bool__(self) -> bool:
        return self.store.db.execute(f"SELECT 1 FROM items WHERE bucket={self.table} LIMIT 1").fetchone() is not None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (list, tuple, DiskList)):
            return False
        sentinel = object()
        return all(a == b for a, b in itertools.zip_longest(self, other, fillvalue=sentinel))

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __contains__(self, value: object) -> bool:
        return any(item == value for item in self)

    def __add__(self, other: list[Any], /) -> DiskList[Any]:
        return DiskList(itertools.chain(self, other), store=self.store)

    def __repr__(self) -> str:
        prefix = list(itertools.islice(self, 8))
        return repr(prefix) if len(self) <= 8 else repr(prefix) + "..."


class DiskMap(MutableMapping[K, V], Generic[K, V]):
    def __init__(self, values: Mapping[K, V] | Iterable[tuple[K, V]] = (), *, store: Store | None = None) -> None:
        self.store = store or current_store()
        self.table = self.store.table()
        self.update(values)

    def __reduce__(self) -> Any:
        return _restore, (str(self.store.path), self.table, "map")

    def __len__(self) -> int:
        return int(self.store.db.execute(f"SELECT COUNT(*) FROM items WHERE bucket={self.table}").fetchone()[0])

    def __getitem__(self, key: K) -> V:
        row = self.store.db.execute(
            f"SELECT value FROM items WHERE bucket={self.table} AND key=?", (_key(key),)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return cast(V, pickle.loads(row[0]))

    def __setitem__(self, key: K, value: V) -> None:
        self.store.db.execute(
            f"INSERT INTO items(bucket,key,key_payload,value) VALUES({self.table},?,?,?) "
            "ON CONFLICT(bucket,key) DO UPDATE SET value=excluded.value",
            (_key(key), _dump(key), _dump(value)),
        )

    def __delitem__(self, key: K) -> None:
        if not self.store.db.execute(f"DELETE FROM items WHERE bucket={self.table} AND key=?", (_key(key),)).rowcount:
            raise KeyError(key)

    def __iter__(self) -> Iterator[K]:
        for (key,) in self.store.db.execute(
            f"SELECT key_payload FROM items WHERE bucket={self.table} ORDER BY ordinal"
        ):
            yield cast(K, pickle.loads(key))

    def items(self) -> Any:
        for key, value in self.store.db.execute(
            f"SELECT key_payload,value FROM items WHERE bucket={self.table} ORDER BY ordinal"
        ):
            yield pickle.loads(key), pickle.loads(value)

    def values(self) -> Any:
        for (value,) in self.store.db.execute(f"SELECT value FROM items WHERE bucket={self.table} ORDER BY ordinal"):
            yield pickle.loads(value)
