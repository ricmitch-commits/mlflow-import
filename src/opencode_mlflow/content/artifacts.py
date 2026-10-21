"""Streamed, self-contained canonical turn artifacts.

Every imported turn receives one artifact: a canonical-json-v1 document
repeating the COMPLETE permitted session snapshot — the full session row
(mutable fields included, original timestamps as ``{"unit", "value"}`
pairs, permitted raw column map) plus every record visible through the
turn's three :class:`~opencode_mlflow.source.replay.ReplayTurn` cursors
— so every trace stays independently inspectable. The document is
streamed section by section through the Task 4A canonical serializer;
a multi-hundred-megabyte session is never materialized as one in-memory
value.

Document shape (one JSON object; top-level keys are emitted in
canonical-json-v1 sorted order, so the streamed bytes equal
:func:`~opencode_mlflow.content.canonical.canonical_json` of the
equivalent materialized document):

- ``artifact``: the importer/adapter/artifact/mapping versions, the
  canonical-json tag, the redaction mode, and the source database
  fingerprint;
- ``identity_map``: the generated source-to-MLflow identity map;
- ``session``: the complete session row;
- ``session_records``/``turn_records``/``unassigned_records``: every
  permitted record, each in the shared canonical record form every
  fingerprint hashes (raw text and raw columns included, malformed and
  duplicate-key values preserved through their typed wrappers);
- ``turn``: the normalized turn boundaries/summary — a derived section;
- ``warnings``: the sorted unique warning codes of the header, turn
  summary, and every record.

Determinism and the identity map: the artifact path embeds the
pre-identity-map :func:`~opencode_mlflow.content.fingerprint.artifact_snapshot_fingerprint`
— computed over the fixed ``identity_map_pending`` placeholder — so the
generated identity map is inserted into the document only AFTER that
fingerprint has fixed the hash and path; generated MLflow IDs can never
change an artifact path or fingerprint. :func:`stream_turn_artifact`
opens a fresh replay cursor for each pass (one fingerprint pass, one
write pass).

Ruling 11 (pinned in :func:`~opencode_mlflow.content.fingerprint.artifact_snapshot_fingerprint`):
the fingerprint material is complete over independent content, and the
root payloads plus turn-boundary/summary sections the artifact persists
are deterministic derivations of hashed records; any derivation-rule
change bumps the importer/mapping/artifact versions and therefore the
render fingerprint.

Relative paths are deterministic and namespaced:

``opencode-mlflow/v1/sessions/<encoded-session-id>/turns/<stable-turn-key>/\
<render-fingerprint>-<artifact-snapshot-fingerprint>.json``

Session IDs are percent-encoded byte-wise (only ``[A-Za-z0-9_-]`` stay
raw, so ``.``, ``/``, ``%``, and every non-ASCII byte encode), which
makes path traversal impossible: an encoded segment is never empty,
``.`` or ``..`` and never contains a separator. Turn keys must already
be ``turn-<64 lowercase hex>`` and both fingerprints ``<64 lowercase
hex>``; the path builder asserts rather than assumes. This module never
imports MLflow: if trace metadata needs the final artifact path, that
update happens through the gateway's documented public trace-update
operation (Task 9's capability contract), never here.

Temporary-file policy: :func:`temporary_artifact_file` creates its file
atomically with mode ``0o600`` (``tempfile.mkstemp`` opens through
``os.open`` with ``O_CREAT | O_EXCL`` and exactly that mode — there is
no create-then-chmod window), yields it for streaming, and deletes it
best-effort after the caller's upload/verification block exits.

Payload limits: the gateway capability probe reports the maximum inline
MLflow payload; this module's limit-parameterized seams
(:func:`load_root_values`, :func:`inline_value_or_reference`) enforce
the negotiated limit with the conservative
:data:`DEFAULT_INLINE_LIMIT_BYTES` (16 MiB) when the target reports
none. A COMPLETE root value above the limit fails with a safe
``payload_too_large`` :class:`~opencode_mlflow.errors.ItemImportError`
before trace creation — the write phase calls :func:`load_root_values`
first — and is never truncated. A child value above the limit stays
complete in the artifact (its record section already persists it) while
the trace receives only a digest/size reference inline.
"""

from __future__ import annotations

import hashlib
import re
import string
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, BinaryIO, cast

from opencode_mlflow import __version__ as IMPORTER_VERSION
from opencode_mlflow.content.canonical import CANONICAL_JSON_VERSION, canonical_json
from opencode_mlflow.content.fingerprint import artifact_snapshot_fingerprint, canonical_record_value
from opencode_mlflow.contracts import RedactionMode
from opencode_mlflow.errors import ItemImportError
from opencode_mlflow.source.discovery import ARTIFACT_SCHEMA_VERSION, MAPPING_VERSION
from opencode_mlflow.source.model import SessionHeader, SourceRecord, TimestampValue, TurnSummary
from opencode_mlflow.source.replay import ReplayTurn
from opencode_mlflow.source.turns import BoundedRootValue

__all__ = [
    "ARTIFACT_PATH_ROOT",
    "ArtifactManifest",
    "ArtifactProvenance",
    "ArtifactUploadChunk",
    "ArtifactValueReference",
    "DEFAULT_INLINE_LIMIT_BYTES",
    "DEFAULT_UPLOAD_CHUNK_BYTES",
    "MAX_ARTIFACT_PATH_TAG_LENGTH",
    "InlineValue",
    "RootValues",
    "artifact_relative_path",
    "inline_value_or_reference",
    "iter_artifact_chunks",
    "load_root_values",
    "resolve_inline_limit",
    "stream_turn_artifact",
    "temporary_artifact_file",
]

ARTIFACT_PATH_ROOT = "opencode-mlflow/v1"
"""Namespace root of every artifact relative path."""

DEFAULT_INLINE_LIMIT_BYTES = 16 * 1024 * 1024
"""Conservative inline payload limit when the target reports none (16 MiB)."""

DEFAULT_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
"""Default fixed upload chunk size (8 MiB)."""

MAX_ARTIFACT_PATH_TAG_LENGTH = 4096
"""Maximum length of the MLflow trace tag carrying the recorded path."""

_DOCUMENT_SECTIONS: tuple[str, ...] = (
    "artifact",
    "identity_map",
    "session",
    "session_records",
    "turn",
    "turn_records",
    "unassigned_records",
    "warnings",
)
"""Top-level artifact sections, in canonical-json-v1 UTF-8 key order."""

_ARTIFACT_REFERENCE_MARKER = "$artifact_reference"
"""Typed marker an oversized value is replaced by inline (digest + size)."""

_TURN_KEY_PATTERN = re.compile(r"turn-[0-9a-f]{64}")
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
_SAFE_PATH_CHARACTERS = frozenset(string.ascii_letters + string.digits + "-_")

_PAYLOAD_TOO_LARGE_MESSAGE = "exceeds the inline payload limit"


def _payload_too_large_message(role: str) -> str:
    """The safe per-role message for an oversized complete root value."""
    return f"the turn's complete root {role} {_PAYLOAD_TOO_LARGE_MESSAGE}"


# ---------------------------------------------------------------------------
# Manifest and upload chunks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactManifest:
    """Outcome of streaming one turn artifact.

    Carries what verification and path assembly need: the turn
    identity, the pre-identity-map artifact-snapshot fingerprint, the
    redaction mode the artifact was written under, and the exact byte
    size and SHA-256 of the written document.
    """

    session_id: str
    turn_key: str
    snapshot_fingerprint: str
    redaction_mode: RedactionMode
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactProvenance:
    """The exact mapped-request provenance persisted in a real artifact.

    The identity-map section is intentionally not part of this envelope: it
    is populated with generated MLflow IDs after the deterministic snapshot
    and path fingerprints have been fixed.
    """

    source_database_fingerprint: str | None
    source_fingerprint: str | None
    render_fingerprint: str | None
    snapshot_fingerprint: str
    artifact_path: str
    source_schema: str | None
    importer_version: str | None
    mapping_version: str | None
    artifact_schema_version: str | None
    redaction_mode: str | None
    catalog_version: str | None

    @classmethod
    def from_request(cls, request: object) -> ArtifactProvenance | None:
        metadata = getattr(request, "metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        # Requests produced by the Wave 1 hand-built test seam do not carry
        # the artifact-schema provenance key. Keep that legacy seam usable;
        # production mapped requests always include it.
        if "opencode.artifact_schema_version" not in metadata:
            return None
        mapped_request = cast(Any, request)
        return cls(
            source_database_fingerprint=_optional_metadata(metadata, "opencode.source_database_fingerprint"),
            source_fingerprint=_optional_metadata(metadata, "opencode.source_fingerprint"),
            render_fingerprint=_optional_metadata(metadata, "opencode.render_fingerprint"),
            snapshot_fingerprint=str(mapped_request.artifact_snapshot_fingerprint),
            artifact_path=str(mapped_request.artifact_relative_path),
            source_schema=_optional_metadata(metadata, "opencode.source_schema"),
            catalog_version=_optional_metadata(metadata, "opencode.catalog_version"),
            importer_version=_optional_metadata(metadata, "opencode.importer_version"),
            mapping_version=_optional_metadata(metadata, "opencode.mapping_version"),
            artifact_schema_version=_optional_metadata(metadata, "opencode.artifact_schema_version"),
            redaction_mode=_optional_metadata(metadata, "opencode.redaction_mode"),
        )


def _optional_metadata(metadata: Mapping[object, object], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None


@dataclass(frozen=True)
class ArtifactUploadChunk:
    """One fixed-size piece of an uploaded artifact.

    ``size`` is the payload length, ``digest`` the payload's SHA-256
    (lowercase hex), so a sink can verify pieces while streaming.
    """

    size: int
    digest: str
    payload: bytes


def iter_artifact_chunks(
    source: BinaryIO,
    chunk_size: int = DEFAULT_UPLOAD_CHUNK_BYTES,
) -> Iterator[ArtifactUploadChunk]:
    """Yield fixed-size upload chunks from an open artifact stream.

    Reads from the stream's current position; for a regular file (the
    artifact upload source) every chunk except the last is exactly
    ``chunk_size`` bytes. Payloads are yielded one at a time and never
    accumulated, so the full artifact is never held in memory.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    while True:
        payload = source.read(chunk_size)
        if not payload:
            return
        yield ArtifactUploadChunk(
            size=len(payload),
            digest=hashlib.sha256(payload).hexdigest(),
            payload=payload,
        )


@contextmanager
def temporary_artifact_file(
    directory: Path,
    *,
    prefix: str = "opencode-mlflow-artifact-",
) -> Iterator[tuple[Path, BinaryIO]]:
    """Open a new restrictive temporary file for one streamed artifact.

    The file is created atomically with mode ``0o600``:
    :func:`tempfile.mkstemp` opens it through ``os.open`` with
    ``O_CREAT | O_EXCL`` and exactly that mode, so no create-then-chmod
    window ever exists. The context yields ``(path, handle)``; the
    caller streams the artifact into ``handle`` and must flush before
    reading the file back. On exit the handle is closed and the file is
    deleted best-effort — deletion therefore happens after the caller's
    upload/verification block, and cleanup failures are swallowed
    (never re-raised past the upload result).
    """
    file_descriptor, name = tempfile.mkstemp(prefix=prefix, dir=directory)
    path = Path(name)
    handle = open(file_descriptor, "wb", closefd=True)
    try:
        yield path, handle
    finally:
        try:
            handle.close()
        finally:
            try:
                path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Deterministic relative paths
# ---------------------------------------------------------------------------


def artifact_relative_path(session_id: str, turn_key: str, render_fp: str, snapshot_fp: str) -> str:
    """Build the deterministic namespaced relative path of one turn artifact.

    ``opencode-mlflow/v1/sessions/<encoded-session-id>/turns/<stable-turn-key>/\
<render-fingerprint>-<artifact-snapshot-fingerprint>.json``

    The session ID is percent-encoded byte-wise so path traversal is
    impossible (see the module docstring); the turn key must already be
    ``turn-<64 lowercase hex>`` and both fingerprints ``<64 lowercase
    hex>`` — asserted here rather than assumed, so a malformed identity
    fails loudly instead of escaping the namespace.
    """
    path = "/".join(
        (
            ARTIFACT_PATH_ROOT,
            "sessions",
            _encode_session_id(session_id),
            "turns",
            _validated_turn_key(turn_key),
            f"{_validated_fingerprint(render_fp)}-{_validated_fingerprint(snapshot_fp)}.json",
        )
    )
    if len(path) > MAX_ARTIFACT_PATH_TAG_LENGTH:
        raise ItemImportError(
            "artifact_path_too_long",
            "the deterministic artifact path exceeds the target trace-tag limit",
        )
    return path


def _encode_session_id(session_id: str) -> str:
    """Percent-encode a session ID into one safe path segment.

    Only ``[A-Za-z0-9_-]`` stay raw; the dot, slash, percent sign,
    backslash, and every other character encode as uppercase-hex
    percent escapes of their UTF-8 bytes. The result can never be
    empty, ``.`` or ``..`` (the dot always encodes) and never contains
    a path separator, so no resolver can read the segment as a
    traversal component.
    """
    if not session_id:
        raise ValueError("a session ID is required for the artifact path")
    pieces: list[str] = []
    for character in session_id:
        if character in _SAFE_PATH_CHARACTERS:
            pieces.append(character)
        else:
            pieces.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
    return "".join(pieces)


def _validated_turn_key(turn_key: str) -> str:
    if _TURN_KEY_PATTERN.fullmatch(turn_key) is None:
        raise ValueError("a turn key must be turn-<64 lowercase hex characters>")
    return turn_key


def _validated_fingerprint(fingerprint: str) -> str:
    if _FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
        raise ValueError("an artifact path fingerprint must be 64 lowercase hex characters")
    return fingerprint


# ---------------------------------------------------------------------------
# The streamed artifact document
# ---------------------------------------------------------------------------


def stream_turn_artifact(
    turn: ReplayTurn,
    destination: BinaryIO,
    identity_map: Mapping[str, str],
    redaction: RedactionMode,
    provenance: ArtifactProvenance | None = None,
) -> ArtifactManifest:
    """Stream one turn's complete canonical artifact to ``destination``.

    Pass 1 opens fresh replay cursors through
    :func:`~opencode_mlflow.content.fingerprint.artifact_snapshot_fingerprint`
    to fix the pre-identity-map snapshot fingerprint (the fixed
    ``identity_map_pending`` placeholder stands in for generated MLflow
    IDs). Pass 2 reopens fresh cursors and streams the document section
    by section, inserting ``identity_map`` only after that fingerprint
    has fixed the hash and path. The destination receives incremental
    writes (bounded by the largest single record value), the byte
    digest and size are accumulated while writing, and the returned
    manifest carries both plus the snapshot fingerprint.
    """
    header = turn.session_header
    summary = turn.summary
    snapshot_fingerprint = artifact_snapshot_fingerprint(turn, redaction)
    if provenance is not None and snapshot_fingerprint != provenance.snapshot_fingerprint:
        raise ItemImportError(
            "source_consistency_mismatch",
            "the streamed artifact snapshot fingerprint does not match the mapped request",
        )

    digest = hashlib.sha256()
    size_bytes = 0

    def write(chunk: bytes) -> None:
        nonlocal size_bytes
        destination.write(chunk)
        digest.update(chunk)
        size_bytes += len(chunk)

    warnings: set[str] = set(header.warnings)
    warnings.update(summary.warnings)
    writer = _DocumentWriter(write)
    write(b"{")
    writer.section("artifact", _artifact_section(header, redaction, provenance))
    writer.section("identity_map", identity_map)
    writer.section("session", _artifact_session_value(header))
    writer.records("session_records", turn.iter_session_records(), warnings)
    writer.section("turn", _turn_section(summary))
    writer.records("turn_records", turn.iter_records(), warnings)
    writer.records("unassigned_records", turn.iter_unassigned_records(), warnings)
    writer.section("warnings", sorted(warnings))
    write(b"}")
    return ArtifactManifest(
        session_id=header.session_id,
        turn_key=summary.turn_key,
        snapshot_fingerprint=snapshot_fingerprint,
        redaction_mode=redaction,
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )


class _DocumentWriter:
    """Incremental canonical-json-v1 object writer for one artifact.

    Sections must be written in :data:`_DOCUMENT_SECTIONS` order —
    exactly the UTF-8 byte order canonical-json-v1 sorts object keys by
    — which the writer enforces, so the streamed bytes always equal
    :func:`canonical_json` of the equivalent materialized document.
    Record arrays stream one record at a time.
    """

    def __init__(self, write: Callable[[bytes], None]) -> None:
        self._write = write
        self._section_index = 0

    def section(self, name: str, value: object) -> None:
        """Write one whole section."""
        self._begin_section(name)
        self._write(canonical_json(value))

    def records(self, name: str, records: Iterator[SourceRecord], warnings: set[str]) -> None:
        """Stream one record section, absorbing its warning codes."""
        self._begin_section(name)
        self._write(b"[")
        first = True
        for record in records:
            warnings.update(record.warnings)
            if first:
                first = False
            else:
                self._write(b",")
            self._write(canonical_json(canonical_record_value(record)))
        self._write(b"]")

    def _begin_section(self, name: str) -> None:
        if name != _DOCUMENT_SECTIONS[self._section_index]:
            raise ValueError("artifact sections must be written in canonical key order")
        self._section_index += 1
        if self._section_index > 1:
            self._write(b",")
        self._write(canonical_json(name))
        self._write(b":")


def _artifact_section(
    header: SessionHeader,
    redaction: RedactionMode,
    provenance: ArtifactProvenance | None = None,
) -> dict[str, object]:
    """The artifact meta section: versions, mode, source database identity."""
    section: dict[str, object] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "canonical_json": CANONICAL_JSON_VERSION,
        "importer_version": IMPORTER_VERSION,
        "mapping_version": MAPPING_VERSION,
        "adapter_generation": header.adapter_generation,
        "redaction_mode": redaction.value,
    }
    if header.catalog_version:
        section["catalog_version"] = header.catalog_version
    if header.source_fingerprint is not None:
        section["source_database_fingerprint"] = header.source_fingerprint
    if provenance is not None:
        if provenance.source_database_fingerprint is not None:
            section["source_database_fingerprint"] = provenance.source_database_fingerprint
        if provenance.source_fingerprint is not None:
            section["source_fingerprint"] = provenance.source_fingerprint
        if provenance.render_fingerprint is not None:
            section["render_fingerprint"] = provenance.render_fingerprint
        section["artifact_snapshot_fingerprint"] = provenance.snapshot_fingerprint
        section["artifact_path"] = provenance.artifact_path
        if provenance.source_schema is not None:
            section["source_schema"] = provenance.source_schema
        if provenance.catalog_version is not None:
            section["catalog_version"] = provenance.catalog_version
        if provenance.importer_version is not None:
            section["importer_version"] = provenance.importer_version
        if provenance.mapping_version is not None:
            section["mapping_version"] = provenance.mapping_version
        if provenance.artifact_schema_version is not None:
            section["artifact_schema_version"] = provenance.artifact_schema_version
        if provenance.redaction_mode is not None:
            section["redaction_mode"] = provenance.redaction_mode
    return section


def _artifact_session_value(header: SessionHeader) -> dict[str, object]:
    """The complete session row, with original timestamp representations.

    Field parity with the fingerprint's complete-session material
    (every non-``None`` field, empty raw column map omitted, warnings
    included), differing only in instants: the artifact keeps the
    original ``{"unit", "value"}`` pair — exactly the representation
    the normalized snapshot stores and replay restores — while
    fingerprint material uses the derivable epoch nanoseconds.
    """
    value: dict[str, object] = {}
    for item in dataclass_fields(SessionHeader):
        field_value = getattr(header, item.name)
        if item.name == "raw_columns":
            raw_columns = cast(Mapping[str, object], field_value)
            if raw_columns:
                value["raw_columns"] = dict(raw_columns)
            continue
        if item.name == "catalog_version" and not field_value:
            continue
        if isinstance(field_value, TimestampValue):
            value[item.name] = {"unit": field_value.unit, "value": field_value.value}
        elif field_value is not None:
            value[item.name] = field_value
    return value


def _turn_section(summary: TurnSummary) -> dict[str, object]:
    """The derived normalized turn-boundary section (identities/counts only)."""
    section: dict[str, object] = {
        "session_id": summary.session_id,
        "turn_index": summary.turn_index,
        "turn_key": summary.turn_key,
    }
    if summary.first_user_source_id is not None:
        section["first_user_source_id"] = summary.first_user_source_id
    if summary.first_user_source_type is not None:
        section["first_user_source_type"] = summary.first_user_source_type
    if summary.start_time is not None:
        section["start_time"] = {"unit": summary.start_time.unit, "value": summary.start_time.value}
    if summary.end_time is not None:
        section["end_time"] = {"unit": summary.end_time.unit, "value": summary.end_time.value}
    section["record_count"] = summary.record_count
    section["user_record_count"] = summary.user_record_count
    section["assistant_record_count"] = summary.assistant_record_count
    if summary.terminal_outcome is not None:
        section["terminal_outcome"] = summary.terminal_outcome
    if summary.source_outcome is not None:
        section["source_outcome"] = summary.source_outcome
    if summary.root_result_source_id is not None:
        section["root_result_source_id"] = summary.root_result_source_id
    if summary.warnings:
        section["warnings"] = list(summary.warnings)
    return section


# ---------------------------------------------------------------------------
# Inline payload limits
# ---------------------------------------------------------------------------


def resolve_inline_limit(reported_limit: int | None) -> int:
    """Return the inline payload limit to enforce.

    ``None`` means the target reported no limit, so the conservative
    :data:`DEFAULT_INLINE_LIMIT_BYTES` (16 MiB) applies; a reported
    limit passes through unchanged. A non-positive reported limit is
    invalid and fails loudly instead of silently rejecting every value.
    """
    if reported_limit is None:
        return DEFAULT_INLINE_LIMIT_BYTES
    if reported_limit <= 0:
        raise ValueError("a reported inline payload limit must be a positive integer")
    return reported_limit


@dataclass(frozen=True)
class RootValues:
    """A turn's complete root input/output, loaded within the inline limit."""

    root_input: object | None
    root_output: object | None


def load_root_values(turn: ReplayTurn, inline_limit_bytes: int | None = None) -> RootValues:
    """Load the turn's complete root input/output within the inline limit.

    The limit is a parameter of this artifact layer so the write phase
    (Task 9/10) can inject the gateway's negotiated limit; the
    conservative 16 MiB default applies when the target reports none.
    Replay implementations that return a
    :class:`~opencode_mlflow.source.turns.BoundedRootValue` honor its
    ``payload_too_large`` rejection; any other returned value is
    measured by its canonical-json-v1 size. A COMPLETE root value above
    the limit raises a safe ``payload_too_large``
    :class:`~opencode_mlflow.errors.ItemImportError` — the write phase
    calls this before trace creation, so the item is rejected rather
    than truncated or partially imported.
    """
    limit = resolve_inline_limit(inline_limit_bytes)
    return RootValues(
        root_input=_root_value(turn.load_root_input(limit), limit, "input"),
        root_output=_root_value(turn.load_root_output(limit), limit, "output"),
    )


def _root_value(loaded: object | None, limit: int, role: str) -> object | None:
    if loaded is None:
        return None
    if isinstance(loaded, BoundedRootValue):
        if loaded.payload_too_large:
            raise ItemImportError("payload_too_large", _payload_too_large_message(role))
        return loaded.value
    if len(canonical_json(loaded)) > limit:
        raise ItemImportError("payload_too_large", _payload_too_large_message(role))
    return loaded


@dataclass(frozen=True)
class ArtifactValueReference:
    """Digest/size reference to a complete value retained in the artifact."""

    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class InlineValue:
    """Placement decision for one child/metadata value against the inline limit.

    ``value`` is what goes inline: the complete value when ``inline``
    is true, otherwise the typed ``{"$artifact_reference": {"sha256",
    "size_bytes"}}`` marker. ``reference`` exposes the same digest/size
    typed for inspection.
    """

    value: object
    inline: bool
    reference: ArtifactValueReference | None


def inline_value_or_reference(value: object, inline_limit_bytes: int | None = None) -> InlineValue:
    """Classify one child/metadata value against the inline limit.

    A value whose canonical-json-v1 size fits the limit stays complete
    inline. An oversized value is never truncated: ``value`` becomes
    the digest/size reference marker while the COMPLETE value remains
    in the turn's artifact — the record sections already persist it —
    so the trace keeps a verifiable pointer instead of a partial copy.
    """
    limit = resolve_inline_limit(inline_limit_bytes)
    canonical = canonical_json(value)
    if len(canonical) <= limit:
        return InlineValue(value=value, inline=True, reference=None)
    sha256 = hashlib.sha256(canonical).hexdigest()
    size_bytes = len(canonical)
    return InlineValue(
        value={_ARTIFACT_REFERENCE_MARKER: {"sha256": sha256, "size_bytes": size_bytes}},
        inline=False,
        reference=ArtifactValueReference(sha256=sha256, size_bytes=size_bytes),
    )
