"""Private sequential payload storage; SQLite holds only byte locators.

Locators are invocation-local, never exported or included in fingerprints.
Each reader owns its file descriptor so nested replay cursors are independent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import BinaryIO

from mlflow_import.resources import open_file, open_reader

PREFIX = "@spool:"
MAX_SPOOL_BYTES = 8 * 1024**3


class PayloadSpool:
    """Append UTF-8 payloads with an explicit finite disk envelope."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = open_file(path, "xb", buffering=0)
        os.chmod(path, 0o600)

    def append(self, text: str) -> str:
        from mlflow_import.errors import SnapshotError

        payload = text.encode("utf-8")
        offset = self._handle.tell()
        if offset + len(payload) > MAX_SPOOL_BYTES:
            raise SnapshotError("temporary_storage_exhausted", "the source spool exceeds its disk envelope")
        if self._handle.write(payload) != len(payload):
            raise SnapshotError("temporary_storage_exhausted", "the source spool could not be fully written")
        return PREFIX + json.dumps([offset, len(payload)], separators=(",", ":"))

    def close(self) -> None:
        self._handle.close()


def read_payload(path: Path, locator: str, handle: BinaryIO | None = None) -> str:
    """Resolve a checked byte range, retaining legacy inline test snapshots."""
    if not locator.startswith(PREFIX):
        return locator
    offset, length = json.loads(locator[len(PREFIX) :])
    if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length < 0:
        raise ValueError("invalid spool locator")
    if handle is None:
        with open_reader(path) as opened:
            return read_payload(path, locator, opened)
    handle.seek(offset)
    payload = handle.read(length)
    if len(payload) != length:
        raise ValueError("incomplete spool payload")
    return payload.decode("utf-8")
