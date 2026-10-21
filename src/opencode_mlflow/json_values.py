"""Lossless JSON values shared across source and content layers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from opencode_mlflow.contracts import DuplicateKeyObject


@dataclass(frozen=True)
class LosslessJsonDecode:
    """A decoded JSON value plus whether any object had duplicate keys."""

    value: object
    has_duplicate_keys: bool


def decode_lossless_json(raw_text: str) -> LosslessJsonDecode:
    """Decode JSON without collapsing duplicate object-key occurrences."""
    has_duplicate_keys = False

    def preserve_duplicates(pairs: list[tuple[str, object]]) -> object:
        nonlocal has_duplicate_keys
        seen: set[str] = set()
        duplicate = False
        for key, _value in pairs:
            if key in seen:
                duplicate = True
            seen.add(key)
        if duplicate:
            has_duplicate_keys = True
            # object_pairs_hook runs bottom-up and has no local source-span
            # offsets. Retaining the complete field text on every nested
            # wrapper would make canonical output quadratic, so nested
            # wrappers carry an explicit empty bounded marker. The full raw
            # field remains on SourceRecord/CanonicalRecordEnvelope and, when
            # the root itself is duplicate, on that one root wrapper below.
            return DuplicateKeyObject(tuple(pairs), "")
        return dict(pairs)

    value = json.loads(raw_text, object_pairs_hook=preserve_duplicates)
    if isinstance(value, DuplicateKeyObject):
        value = DuplicateKeyObject(value.pairs, raw_text)
    return LosslessJsonDecode(value=value, has_duplicate_keys=has_duplicate_keys)


def json_text_from_decoded(value: object) -> str:
    """Serialize decoded JSON while preserving duplicate-pair occurrence order."""
    if isinstance(value, DuplicateKeyObject):
        members = (_json_string(key) + ":" + json_text_from_decoded(item) for key, item in value.pairs)
        return "{" + ",".join(members) + "}"
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]).encode("utf-8", "surrogatepass"))
        members = (_json_string(str(key)) + ":" + json_text_from_decoded(item) for key, item in items)
        return "{" + ",".join(members) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(json_text_from_decoded(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
