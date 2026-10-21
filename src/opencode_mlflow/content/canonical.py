"""canonical-json-v1: deterministic canonical JSON serialization.

This module owns the versioned ``canonical-json-v1`` wire contract used
by every fingerprint and artifact. It re-exports the shared
:class:`~opencode_mlflow.contracts.CanonicalRecordEnvelope`,
:class:`~opencode_mlflow.contracts.RedactionMode`, and
:class:`~opencode_mlflow.contracts.Redactor` definitions rather than
defining duplicates.

The contract is pinned by the golden vectors in
``tests/fixtures/payloads/canonical-json-v1-vectors.json``:

- Object keys sort by their UTF-8 byte sequence. For valid scalar
  strings that equals code-point order; lone surrogates sort through
  their ``surrogatepass`` encoding, which a UTF-16 code-unit sort would
  not reproduce.
- Strings keep their original Unicode code points. Only ``"``, ``\\``,
  C0 controls (the short forms ``\\b \\t \\n \\f \\r`` or lowercase-hex
  ``\\u00xx``), and lone surrogates are escaped, so output is always
  valid UTF-8; U+2028/U+2029 and non-ASCII stay raw.
- Separators are fixed: ``,`` between members, ``:`` after keys, no
  whitespace.
- ``None``/``bool``/``int`` use their JSON scalar forms. Integers are
  exact arbitrary-precision literals and never route through double
  formatting. Finite floats use RFC 8785 (ECMAScript
  ``Number::toString``) shortest-round-trip formatting: decimal
  notation for ``1e-6 <= |x| < 1e21``, scientific notation with
  ``e+``/``e-`` and no leading exponent zeros otherwise, integral
  floats without a decimal point, and ``-0.0`` normalized to ``0``.
- Values JSON cannot express use fixed self-describing typed wrappers
  (single-key marker objects — a versioned contract):

  - non-finite floats: ``{"$nonfinite":"NaN"}``,
    ``{"$nonfinite":"Infinity"}``, ``{"$nonfinite":"-Infinity"}``;
  - SQLite BLOBs and invalid UTF-8 (``bytes``):
    ``{"$bytes":"<standard base64>"}``;
  - JSON objects with duplicate keys (:class:`DuplicateKeyObject`):
    ``{"$duplicate_keys":[["k",v],...],"$raw_text":"<original text>"}``.

  Ordinary data whose keys happen to collide with a marker serializes
  identically to the wrapper; the markers are self-describing, not
  unambiguous. SQLite ``NULL`` is JSON ``null``; absent columns are
  omitted by callers — the serializer never invents nulls.
- Mappings serialize as objects (string keys only) and sequences (lists
  and tuples) as arrays; any other type raises :class:`TypeError`
  instead of being silently coerced.
"""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping

from opencode_mlflow.contracts import (
    PROTECTED_IDENTITY_FIELDS,
    CanonicalRecordEnvelope,
    DuplicateKeyObject,
    RedactionMode,
    Redactor,
)

__all__ = [
    "CANONICAL_JSON_VERSION",
    "CanonicalRecordEnvelope",
    "DuplicateKeyObject",
    "RedactionMode",
    "Redactor",
    "canonical_json",
    "canonical_json_compatible",
    "canonical_record_envelope",
    "canonical_stream",
]

CANONICAL_JSON_VERSION = "canonical-json-v1"
"""Version tag of the serialization contract implemented here."""

_NONFINITE_NAN = "NaN"
_NONFINITE_POSITIVE_INFINITY = "Infinity"
_NONFINITE_NEGATIVE_INFINITY = "-Infinity"

_MARK_NONFINITE = "$nonfinite"
_MARK_BYTES = "$bytes"
_MARK_DUPLICATE_KEYS = "$duplicate_keys"
_MARK_RAW_TEXT = "$raw_text"

_STRING_ESCAPES: Mapping[str, str] = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}
_NEEDS_ESCAPE = re.compile(r'["\\\x00-\x1f\ud800-\udfff]')


def canonical_json(value: object) -> bytes:
    """Serialize ``value`` to canonical-json-v1 UTF-8 bytes."""
    parts: list[str] = []
    _write_value(value, parts)
    return "".join(parts).encode("utf-8")


def canonical_json_compatible(value: object) -> object:
    """Return the canonical value as ordinary JSON-compatible Python data.

    The canonical serializer is authoritative for typed wrappers and number
    spelling.  Decoding its bytes gives MLflow only JSON-native ``dict``,
    ``list``, scalar, and ``None`` values while preserving the lossless
    representation used by fingerprints and artifacts.
    """
    return json.loads(canonical_json(value))


def canonical_record_envelope(
    value: object,
    raw_text: str | None,
    raw_columns: Mapping[str, object],
) -> CanonicalRecordEnvelope:
    """Build the shared canonical record envelope for one source value.

    ``warnings`` starts empty; decoders that detect malformed or
    duplicate-key payloads add their warnings when they build envelopes.
    """
    return CanonicalRecordEnvelope(decoded=value, raw_text=raw_text, raw_columns=raw_columns, warnings=())


def canonical_stream(chunks: Iterable[object], redactor: Redactor) -> Iterator[bytes]:
    """Stream canonical serialization of ``chunks``, redacting first.

    Each chunk is redacted (with the project's fixed identity-field
    protection set, exactly like snapshot persistence) and then
    serialized to canonical-json-v1 bytes, one chunk at a time; the
    stream is never materialized, so bounded-memory artifact writers can
    consume it incrementally.
    """
    for chunk in chunks:
        yield canonical_json(redactor.redact(chunk, PROTECTED_IDENTITY_FIELDS))


def _write_value(value: object, parts: list[str]) -> None:
    if value is None:
        parts.append("null")
    elif isinstance(value, bool):
        parts.append("true" if value else "false")
    elif isinstance(value, str):
        parts.append(_dump_string(value))
    elif isinstance(value, int):
        parts.append(str(value))
    elif isinstance(value, float):
        _write_float(value, parts)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        _write_object({_MARK_BYTES: base64.b64encode(bytes(value)).decode("ascii")}, parts)
    elif isinstance(value, DuplicateKeyObject):
        _write_object(
            {
                _MARK_DUPLICATE_KEYS: [[key, item] for key, item in value.pairs],
                _MARK_RAW_TEXT: value.raw_text,
            },
            parts,
        )
    elif isinstance(value, Mapping):
        _write_object(value, parts)
    elif isinstance(value, (list, tuple)):
        _write_array(value, parts)
    else:
        raise TypeError(f"a value of type {type(value).__name__} has no canonical-json-v1 representation")


def _write_object(value: Mapping[object, object], parts: list[str]) -> None:
    items: list[tuple[str, object]] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("canonical object keys must be strings")
        items.append((key, item))
    items.sort(key=lambda item: item[0].encode("utf-8", "surrogatepass"))
    parts.append("{")
    for index, (key, item) in enumerate(items):
        if index:
            parts.append(",")
        parts.append(_dump_string(key))
        parts.append(":")
        _write_value(item, parts)
    parts.append("}")


def _write_array(value: list[object] | tuple[object, ...], parts: list[str]) -> None:
    parts.append("[")
    for index, item in enumerate(value):
        if index:
            parts.append(",")
        _write_value(item, parts)
    parts.append("]")


def _write_float(value: float, parts: list[str]) -> None:
    if math.isnan(value):
        _write_object({_MARK_NONFINITE: _NONFINITE_NAN}, parts)
    elif math.isinf(value):
        spelling = _NONFINITE_POSITIVE_INFINITY if value > 0 else _NONFINITE_NEGATIVE_INFINITY
        _write_object({_MARK_NONFINITE: spelling}, parts)
    else:
        parts.append(_format_finite_float(value))


def _dump_string(value: str) -> str:
    if _NEEDS_ESCAPE.search(value) is None:
        return f'"{value}"'
    escaped: list[str] = ['"']
    for character in value:
        short = _STRING_ESCAPES.get(character)
        if short is not None:
            escaped.append(short)
        elif character < " " or "\ud800" <= character <= "\udfff":
            escaped.append(f"\\u{ord(character):04x}")
        else:
            escaped.append(character)
    escaped.append('"')
    return "".join(escaped)


def _format_finite_float(value: float) -> str:
    """Format a finite float per RFC 8785 / ECMAScript ``Number::toString``.

    Python's ``repr`` already yields the shortest round-trip digits; the
    layout is recomputed here because ``repr``'s thresholds and exponent
    spelling differ. Decimal notation is used for
    ``1e-6 <= |x| < 1e21``, scientific notation (``e+``/``e-``, no
    leading exponent zeros, no trailing ``.0``) otherwise. ``-0.0``
    normalizes to ``0``.
    """
    if value == 0.0:
        return "0"
    text = repr(value)
    negative = text.startswith("-")
    if negative:
        text = text[1:]
    if "e" in text:
        mantissa, _, exponent_text = text.partition("e")
        exponent = int(exponent_text)
    else:
        mantissa, exponent = text, 0
    if "." in mantissa:
        integer_part, fraction_part = mantissa.split(".")
    else:
        integer_part, fraction_part = mantissa, ""
    digits = integer_part + fraction_part
    first_significant = next(index for index, character in enumerate(digits) if character != "0")
    significant = digits[first_significant:].rstrip("0")
    digit_count = len(significant)
    point = len(integer_part) - first_significant + exponent
    if digit_count <= point <= 21:
        rendered = significant + "0" * (point - digit_count)
    elif 0 < point <= 21:
        rendered = significant[:point] + "." + significant[point:]
    elif -6 < point <= 0:
        rendered = "0." + "0" * -point + significant
    else:
        exponent_value = point - 1
        sign = "+" if exponent_value >= 0 else "-"
        mantissa_text = significant if digit_count == 1 else f"{significant[0]}.{significant[1:]}"
        rendered = f"{mantissa_text}e{sign}{abs(exponent_value)}"
    return f"-{rendered}" if negative else rendered
