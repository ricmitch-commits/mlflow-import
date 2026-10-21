"""Shared cross-layer contracts for the OpenCode → MLflow importer.

These types form the stable seam between source normalization, content
policy, and the MLflow gateway. Submodules such as
``opencode_mlflow.content.canonical`` and ``opencode_mlflow.content.redaction``
must re-export these definitions rather than defining incompatible
duplicates.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

PROTECTED_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "session_id",
        "project_id",
        "workspace_id",
        "message_id",
        "parent_id",
        "fork_session_id",
        "seq",
        "admitted_seq",
        "enqueued_seq",
        "epoch_start",
        "through_seq",
        "hash",
        "key",
        "position",
        "type",
        "role",
        "source_id",
        "source_table",
        "source_type",
        "parent_source_table",
        "turn_key",
        "schema",
        "catalog_version",
    }
)
"""Root structural fields protected at typed/raw-row envelope boundaries."""


class RedactionMode(Enum):
    """Content policy applied before anything is persisted."""

    VERBATIM = "verbatim"
    REDACTED = "redacted"


@dataclass(frozen=True)
class CanonicalRecordEnvelope:
    """Canonical decoding of one source record with its raw provenance.

    ``decoded`` is the canonical structured value, ``raw_text`` the original
    text when the record came from a text column, ``raw_columns`` the
    permitted raw column values, and ``warnings`` the non-fatal warnings
    raised while normalizing the record.
    """

    decoded: object | None
    raw_text: str | None
    raw_columns: Mapping[str, object]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateKeyObject(Mapping[str, object]):
    """A lossless duplicate-key object with a last-occurrence projection.

    ``pairs`` remains authoritative for canonical/artifact serialization;
    the top-level wrapper may retain source ``raw_text`` while nested
    wrappers use a bounded empty marker.
    The mapping interface exists only for typed extraction of unique fields
    such as role, type, provider, and model; repeated keys project their last
    occurrence deterministically without discarding the original pairs.
    """

    pairs: tuple[tuple[str, object], ...]
    raw_text: str

    @property
    def mapping_projection(self) -> Mapping[str, object]:
        """The deterministic last-occurrence view used by typed extractors."""
        return dict(self.pairs)

    def __getitem__(self, key: str) -> object:
        for candidate, value in reversed(self.pairs):
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.mapping_projection)

    def __len__(self) -> int:
        return len(self.mapping_projection)


@runtime_checkable
class Redactor(Protocol):
    """Redaction policy applied to values before they are persisted."""

    def redact(self, value: object, protected_paths: Collection[str]) -> object: ...


class _IdentityRedactor:
    """Return every value unchanged; replayed values are already effective.

    The normalized snapshot persists post-redaction values, so every
    fingerprint or mapping pass over a replayed cursor applies this
    redactor — the redaction *mode tag* still distinguishes runs, while
    the values are never redacted twice.
    """

    def redact(self, value: object, protected_paths: Collection[str]) -> object:
        return value


IDENTITY_REDACTOR: Redactor = _IdentityRedactor()
"""The single production identity redactor.

One shared instance for every production caller — content fingerprints,
discovery, and the MLflow mapping — so no module redefines its own
duplicate (the mode a fingerprint was hashed under is carried by the
explicit ``RedactionMode`` tags, never by swapping this redactor).
"""
