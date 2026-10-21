"""Deterministic redaction of transcript content before persistence.

This module owns the version-one secret detectors, the fixed
transformation pipeline that applies them, and the mode-aware
:class:`DeterministicRedactor` implementing the shared
:class:`~opencode_mlflow.contracts.Redactor` protocol. It re-exports the
shared :class:`~opencode_mlflow.contracts.RedactionMode` and
:class:`~opencode_mlflow.contracts.Redactor` definitions rather than
defining duplicates, and it runs strictly before canonicalization: every
caller serializes or hashes only the redacted value, so masked secrets
are never indirectly hashed or retained.

Replacement markers are fixed: ``[REDACTED]`` for text values and
``[REDACTED_BINARY]`` for opaque BLOB/invalid-UTF-8 bytes (unless the
value sits under a protected path). Bearer/Basic scheme prefixes are
preserved where possible — ``Authorization: Bearer [REDACTED]`` keeps
the header name, separator, and scheme.

Version-one detector categories (the only keys a report's
``detector_counts`` may carry, plus the type-based ``opaque_binary``):

- ``pem_private_key``: ``-----BEGIN ... PRIVATE KEY-----`` through the
  matching ``-----END ... PRIVATE KEY-----`` block. An unterminated
  ``BEGIN`` marker (a truncated block) masks through the end of the
  text, because the remainder is private-key fragment material.
- ``auth_header``: case-insensitive ``Authorization``/``Proxy-Authorization``
  bearer/basic values, including standalone ``Bearer <token>`` forms.
- ``structured_auth_field``: structured ``Authorization``/``Proxy-Authorization``
  keys whose scalar value begins with a bearer/basic scheme.
- ``secret_key_value``: values assigned to case-insensitive keys spelled
  ``api[-_]key``, ``api[-_]token``, ``access[-_]token``, ``refresh[-_]token``,
  ``auth[-_]token``, ``session[-_]token``, ``token``, ``access[-_]key``,
  ``client[-_]secret``, ``secret``, ``credential``, ``password``,
  ``passwd``, ``private[-_]key``, or ``bearer``. The compiled pattern
  makes the separator optional, so the concatenated spellings
  (``apikey``, ``privatekey``, ...) also match; textual assignments use
  the same spellings with delimiter-aware value bounds.
- ``connection_secret``: the password between ``scheme://user:`` and
  ``@`` in a URI or connection string, including empty-user forms
  (``redis://:password@host``) — the user part is optional, the
  password is not.
- ``known_token_prefix``: recognizable ``sk-``, ``ghp_``, ``github_pat_``,
  ``xox[baprs]-``, and AWS ``AKIA`` forms with their documented minimum
  lengths (16, 20, 20, 10, and exactly 16 respectively).
- ``opaque_binary``: bytes and other opaque non-scalar values replaced
  by ``[REDACTED_BINARY]`` because they cannot be inspected as text — a
  type-based guard, not a content match. Unknown types fail closed: a
  value the pipeline cannot inspect is masked, never passed through.

Known version-one gaps, accepted as scope rather than bugs (review
ruling): quoted ``Basic`` credentials inside malformed JSON text are
not matched — the literal auth-header form cannot cross quotes and
over-matching in quoted contexts was ruled out — while quoted ``Bearer``
values are still caught by the standalone-bearer detector;
connection strings without a password (``user:@host``) are not masked
because the password requires at least one character; and
``Digest``/negotiate schemes are outside the bearer/basic contract.

The compiled detector constants translate the design's required forms to
Python ``re`` flags (no inline ``(?i)``). The auth-header,
standalone-bearer, structured-auth-key/value, and textual-assignment
patterns carry named capture groups so replacements can preserve the
matched key, delimiters, and scheme; the matched text is otherwise
identical to the required forms. Two deliberate supersets extend the
required forms per review ruling: the connection-secret user part is
optional (empty-user forms), and the unterminated-PEM pattern masks a
``BEGIN`` marker without its matching ``END`` through the end of the
text after complete blocks were matched first.

The transformation pipeline is fixed:

1. :func:`redact_value` recursively redacts structured mappings before
   any textual matching of leaf values, preserving field names and
   masking values only. Detector precedence for a structured field is:
   protected root/path (value copied through unchanged, nested content
   included), structured auth field, secret key (the whole assigned
   value is masked, so a dict like ``{"token": "..."}`` counts as
   ``secret_key_value`` and never as a leaf value scan), then recursion.
   The same field rules apply independently to every pair of a
   :class:`~opencode_mlflow.content.canonical.DuplicateKeyObject`. Its
   wrapper type and pair order are preserved, while ``raw_text`` is
   regenerated from the redacted pairs so escaped duplicate key spellings
   cannot retain original secret-bearing text. Sequences recurse
   element-wise (tuples stay tuples), opaque bytes become
   ``[REDACTED_BINARY]``, non-protected string leaves are scanned
   textually, and any other value the pipeline cannot inspect fails
   closed to ``[REDACTED_BINARY]``.
2. Text scanning applies, in order: textual assignments (a secret-key
   spelling followed by ``:`` or ``=`` — after the separator a quoted
   value ends at its matching unescaped quote and an unquoted value ends
   at whitespace, comma, semicolon, ``}``, or ``]``; surrounding quotes
   and delimiters are preserved), ``Authorization``/``Proxy-Authorization``
   header values, PEM private-key blocks (complete blocks first, then
   unterminated ``BEGIN`` markers), standalone bearer values,
   connection-string passwords, and known token prefixes. Earlier
   stages mask first, so a value assigned to a secret key never also
   counts as a token prefix, and a PEM block replaces its whole body
   before bearer/token patterns could match inside it.
3. :func:`redact_record` applies the pipeline to one record. Typed
   structural identity fields are copied unchanged; decoded transcript
   content receives no identity-name exemption; raw-row columns are
   content and receive no typed-identity protection. Raw JSON text is
   re-serialized from the already-redacted decoded value (valid JSON —
   so the original raw text is never retained once anything was
   detected), regenerated from ordered pairs (duplicate-key JSON), or
   masked in place (malformed text, whose decoded value is absent).
   Identity, ordering, and classification fields — source,
   session, message, and part IDs, sequence values, turn keys, schema
   names, and reconciliation identifiers — survive unchanged in the
   record's typed fields, so raw record text may be redacted while the
   same identity is retained for selection, ordering, and replay.
4. :func:`redact_text` applies only the textual pipeline to one string.

:class:`RedactionReport` carries detector-category counts and affected
field/record counts only — never a matched secret, never the original
text. ``affected_field_count`` counts masked value positions: each
masked structured mapping entry, each replaced binary value, and each
scanned text value that changed (once, regardless of how many spans it
contained). ``affected_record_count`` is 1 when a call redacted anything
and 0 otherwise; reports merge cumulatively for run-level aggregation.

Verbatim mode is a no-op: the :class:`DeterministicRedactor` (default
mode verbatim) passes values through unchanged and its result-returning
helpers report zero counts. The module-level functions are always
active; the importer selects the mode by choosing the redactor.

Protected paths are root/path exact: a bare name protects only that
top-level structural field, while a dotted path protects only that path
from the root. Sequence elements contribute no path component. Identity-
like names nested in transcript content receive no blanket exemption.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from typing import cast

from opencode_mlflow.content.canonical import DuplicateKeyObject, canonical_json
from opencode_mlflow.contracts import RedactionMode, Redactor
from opencode_mlflow.json_values import json_text_from_decoded
from opencode_mlflow.source.model import SourceRecord

__all__ = [
    "AUTH_HEADER_PATTERN",
    "CONNECTION_SECRET_PATTERN",
    "DETECTOR_CATEGORIES",
    "DeterministicRedactor",
    "KNOWN_TOKEN_PREFIX_PATTERN",
    "PEM_PRIVATE_KEY_PATTERN",
    "REDACTED_BINARY_MARKER",
    "REDACTED_MARKER",
    "SECRET_KEY_NAME_PATTERN",
    "STANDALONE_BEARER_PATTERN",
    "STRUCTURED_AUTH_KEY_PATTERN",
    "STRUCTURED_AUTH_VALUE_PATTERN",
    "RedactionMode",
    "RedactionReport",
    "RedactionResult",
    "Redactor",
    "TEXT_ASSIGNMENT_PATTERN",
    "UNTERMINATED_PEM_PRIVATE_KEY_PATTERN",
    "redact_record",
    "redact_text",
    "redact_value",
]

REDACTED_MARKER = "[REDACTED]"
"""Exact replacement marker for redacted text values."""

REDACTED_BINARY_MARKER = "[REDACTED_BINARY]"
"""Exact replacement marker for opaque BLOB/invalid-UTF-8 byte values."""

DETECTOR_CATEGORIES: tuple[str, ...] = (
    "pem_private_key",
    "auth_header",
    "structured_auth_field",
    "secret_key_value",
    "connection_secret",
    "known_token_prefix",
    "opaque_binary",
)
"""Detector-category names that may appear in a report's counts."""

_SECRET_KEY_SPELLINGS = (
    "api[-_]?key",
    "api[-_]?token",
    "access[-_]?token",
    "refresh[-_]?token",
    "auth[-_]?token",
    "session[-_]?token",
    "token",
    "access[-_]?key",
    "client[-_]?secret",
    "secret",
    "credential",
    "password",
    "passwd",
    "private[-_]?key",
    "bearer",
)
_SECRET_KEY_ALTERNATION = "|".join(_SECRET_KEY_SPELLINGS)

SECRET_KEY_NAME_PATTERN = re.compile(f"^(?:{_SECRET_KEY_ALTERNATION})$", re.IGNORECASE)
"""Whole-field secret-key spelling test (structured keys)."""

_TEXT_SECRET_KEY_LITERALS = frozenset(
    {
        *(f"api{separator}{suffix}" for separator in ("", "-", "_") for suffix in ("key", "token")),
        *(
            f"{prefix}{separator}{suffix}"
            for prefix in ("access", "refresh", "auth", "session")
            for separator in ("", "-", "_")
            for suffix in ("token", "key")
        ),
        *(f"client{separator}secret" for separator in ("", "-", "_")),
        *(f"private{separator}key" for separator in ("", "-", "_")),
        "token",
        "secret",
        "credential",
        "password",
        "passwd",
        "bearer",
    }
)


def _escaped_key_alternation() -> str:
    """Build key alternatives accepting literal or JSON ``\\uXXXX`` chars."""
    alternatives = []
    for spelling in sorted(_TEXT_SECRET_KEY_LITERALS, key=len, reverse=True):
        alternatives.append("".join(f"(?:{re.escape(character)}|\\\\u{ord(character):04x})" for character in spelling))
    return "|".join(alternatives)


TEXT_ASSIGNMENT_PATTERN = re.compile(
    r"""(?P<open_quote>["']?)"""
    r"""(?<![A-Za-z0-9_])"""
    r"""(?P<name>""" + _escaped_key_alternation() + r""")"""
    r"""(?P<close_quote>["']?)"""
    r"""(?P<sep>\s*[:=]\s*)"""
    r"""(?P<value>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^,\s;}\]]+)""",
    re.IGNORECASE,
)
"""Delimiter-aware textual assignment scanner for malformed/unknown text."""

AUTH_HEADER_PATTERN = re.compile(
    r"(?P<name>authorization|proxy-authorization)"
    r"(?P<sep>\s*:\s*)"
    r"(?P<scheme>bearer|basic)"
    r"(?P<space>\s+)"
    r"(?P<credentials>[A-Za-z0-9._~+/=-]+)",
    re.IGNORECASE,
)
"""``Authorization``/``Proxy-Authorization`` bearer/basic header values."""

STANDALONE_BEARER_PATTERN = re.compile(
    r"\b(?P<scheme>bearer\s+)(?P<credentials>[A-Za-z0-9._~+/=-]{16,})\b",
    re.IGNORECASE,
)
"""Standalone ``Bearer <token>`` values (16+ credential characters)."""

STRUCTURED_AUTH_KEY_PATTERN = re.compile(r"^(?:authorization|proxy-authorization)$", re.IGNORECASE)
"""Whole-key test for structured authorization fields."""

STRUCTURED_AUTH_VALUE_PATTERN = re.compile(
    r"^(?P<scheme>bearer|basic)(?P<space>\s+)(?P<credentials>.+)$",
    re.IGNORECASE,
)
"""Scalar value of a structured authorization field: a bearer/basic scheme
followed by its credentials."""

CONNECTION_SECRET_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]*:)[^@\s/]+(@)", re.IGNORECASE)
"""Password/user-info between ``scheme://user:`` and ``@`` (design groups 1/2).

The user part is optional (``*``), so empty-user forms such as
``redis://:password@host`` are matched; the password still requires at
least one character.
"""

KNOWN_TOKEN_PREFIX_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16})\b",
    re.IGNORECASE,
)
"""Recognizable ``sk-``/``ghp_``/``github_pat_``/``xox[baprs]-``/``AKIA`` forms."""

PEM_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"
    r"[\s\S]*?"
    r"-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----"
)
"""PEM private-key block from its BEGIN to its matching END marker."""

UNTERMINATED_PEM_PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[\s\S]*$")
"""Unterminated PEM private-key BEGIN marker masking through end of text.

Applied after :data:`PEM_PRIVATE_KEY_PATTERN` so complete blocks keep
their exact bounds; a truncated block (no matching END) still masks its
remainder, which is private-key fragment material.
"""


@dataclass(frozen=True)
class RedactionReport:
    """Counts-only outcome of one or more redaction passes.

    ``detector_counts`` maps each detector category to its match count,
    ``affected_field_count`` counts masked value positions, and
    ``affected_record_count`` counts records whose redaction changed
    anything. A report never stores a matched secret or original text.
    """

    detector_counts: Mapping[str, int] = field(default_factory=dict)
    affected_field_count: int = 0
    affected_record_count: int = 0

    def merge(self, other: RedactionReport) -> RedactionReport:
        """Accumulate ``other`` into a new combined report."""
        counts = dict(self.detector_counts)
        for category, count in other.detector_counts.items():
            counts[category] = counts.get(category, 0) + count
        return RedactionReport(
            detector_counts=counts,
            affected_field_count=self.affected_field_count + other.affected_field_count,
            affected_record_count=self.affected_record_count + other.affected_record_count,
        )


@dataclass(frozen=True)
class RedactionResult:
    """One redaction outcome: the masked value and its counts-only report."""

    value: object
    report: RedactionReport


class DeterministicRedactor:
    """Mode-aware :class:`~opencode_mlflow.contracts.Redactor` implementation.

    ``RedactionMode.VERBATIM`` (the default) is a no-op passthrough —
    values return unchanged and results carry zero counts — while
    ``RedactionMode.REDACTED`` activates the deterministic pipeline. The
    protocol method :meth:`redact` returns the masked value alone; the
    result-returning helpers expose the report the run-level summary
    accumulates.
    """

    def __init__(self, mode: RedactionMode = RedactionMode.VERBATIM) -> None:
        self._mode = mode

    @property
    def mode(self) -> RedactionMode:
        """The content policy this redactor applies."""
        return self._mode

    def redact(self, value: object, protected_paths: Collection[str] = ()) -> object:
        """Protocol seam: mask ``value`` unless the mode is verbatim."""
        if self._mode is RedactionMode.VERBATIM:
            return value
        return redact_value(value, protected_paths).value

    def redact_text(self, text: str) -> RedactionResult:
        """Mode-aware :func:`redact_text`."""
        if self._mode is RedactionMode.VERBATIM:
            return RedactionResult(value=text, report=RedactionReport())
        return redact_text(text)

    def redact_value(self, value: object, protected_paths: Collection[str] = ()) -> RedactionResult:
        """Mode-aware :func:`redact_value`."""
        if self._mode is RedactionMode.VERBATIM:
            return RedactionResult(value=value, report=RedactionReport())
        return redact_value(value, protected_paths)

    def redact_record(self, record: SourceRecord) -> RedactionResult:
        """Mode-aware :func:`redact_record`."""
        if self._mode is RedactionMode.VERBATIM:
            return RedactionResult(value=record, report=RedactionReport())
        return redact_record(record)


def redact_text(text: str) -> RedactionResult:
    """Apply the textual detector pipeline to one string."""
    state = _RedactionState()
    redacted = _scan_text(text, state)
    return RedactionResult(value=redacted, report=state.report())


def redact_value(value: object, protected_paths: Collection[str] = ()) -> RedactionResult:
    """Redact one value, structurally first and textually second.

    Mappings and sequences recurse before leaf values are scanned;
    protected paths (a root field name or an exact dotted path from the
    root) are copied through unchanged; opaque bytes become
    ``[REDACTED_BINARY]``; other leaves pass through unless a detector
    matches.
    """
    state = _RedactionState()
    redacted = _redact_node(value, (), frozenset(protected_paths), state)
    return RedactionResult(value=redacted, report=state.report())


def redact_record(record: SourceRecord) -> RedactionResult:
    """Redact one record's content while preserving its identity fields.

    The decoded value and raw columns are redacted as content without a
    typed-identity exemption. The raw JSON text is
    re-serialized from the already-redacted decoded value when the raw
    text was valid JSON and anything was detected, masked in place when
    it was malformed, regenerated from lossless duplicate-key pairs, and kept
    unchanged when nothing was detected. The
    record's identity, ordering, and classification fields are copied
    through verbatim.
    """
    state = _RedactionState()
    # Typed SourceRecord fields carry structural identity separately. The
    # decoded transcript is content, so identity-looking keys inside it are
    # never exempt from redaction.
    decoded = _redact_node(record.decoded_value, (), frozenset(), state)
    raw_json = _redacted_raw_json(record, decoded, state)
    raw_columns = cast(
        Mapping[str, object],
        # Raw columns are provenance/content, not the typed identity seam.
        # Their names may be additive and may deliberately resemble a
        # structural field, so typed identity protection must not exempt
        # their values from redaction.
        _redact_node(record.raw_columns, (), frozenset(), state),
    )
    redacted = replace(record, decoded_value=decoded, raw_json=raw_json, raw_columns=raw_columns)
    return RedactionResult(value=redacted, report=state.report())


def _redacted_raw_json(record: SourceRecord, decoded: object, state: _RedactionState) -> str | None:
    """Produce the redacted raw text for one record."""
    if record.raw_json is None:
        return None
    if "malformed_json" in record.warnings:
        # The decoded value cannot represent this raw text faithfully: it is
        # absent (malformed), so scan the raw text in place.
        return _scan_text(record.raw_json, state)
    if "duplicate_json_key" in record.warnings:
        # The lossless decoder retained every pair. Rebuild raw text from the
        # already-redacted structure so escaped key spellings cannot retain an
        # original secret-bearing representation.
        return json_text_from_decoded(decoded)
    if state.counts:
        return canonical_json(decoded).decode("utf-8")
    return record.raw_json


class _RedactionState:
    """Mutable detection tallies accumulated during one redaction pass."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.affected_fields = 0

    def record(self, category: str) -> None:
        """Tally one masked value position under its detector category."""
        self.counts[category] += 1
        self.affected_fields += 1

    def report(self) -> RedactionReport:
        return RedactionReport(
            detector_counts=dict(self.counts),
            affected_field_count=self.affected_fields,
            affected_record_count=1 if self.affected_fields else 0,
        )


def _redact_node(
    value: object,
    path: tuple[str, ...],
    protected: frozenset[str],
    state: _RedactionState,
    depth: int = 0,
) -> object:
    # DuplicateKeyObject is also a Mapping for typed extraction; its lossless
    # ordered-pair form must take precedence over generic mapping traversal.
    if isinstance(value, DuplicateKeyObject):
        pairs = tuple((key, _redact_field(key, item, path, protected, state, depth + 1)) for key, item in value.pairs)
        raw_text = json_text_from_decoded(DuplicateKeyObject(pairs, "")) if depth == 0 else ""
        return DuplicateKeyObject(pairs=pairs, raw_text=raw_text)
    if isinstance(value, Mapping):
        redacted: dict[object, object] = {}
        for key, item in value.items():
            if isinstance(key, str):
                redacted[key] = _redact_field(key, item, path, protected, state, depth + 1)
            else:
                redacted[key] = _redact_node(item, path, protected, state, depth + 1)
        return redacted
    if isinstance(value, list):
        return [_redact_node(item, path, protected, state, depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_node(item, path, protected, state, depth + 1) for item in value)
    if isinstance(value, str):
        return _scan_text(value, state)
    if isinstance(value, (bytes, bytearray, memoryview)):
        state.record("opaque_binary")
        return REDACTED_BINARY_MARKER
    if value is None or isinstance(value, (bool, int, float)):
        return value
    # Fail closed: a security control must not pass through a value it
    # cannot inspect. Anything that is neither a handled container, a
    # string, bytes-like, nor a JSON scalar is masked as opaque.
    state.record("opaque_binary")
    return REDACTED_BINARY_MARKER


def _redact_field(
    name: str,
    item: object,
    path: tuple[str, ...],
    protected: frozenset[str],
    state: _RedactionState,
    depth: int,
) -> object:
    """Redact one named field's value with the structured precedence rules."""
    child_path = (*path, name)
    if _is_protected(name, child_path, protected):
        return item
    if _is_structured_auth_key(name) and isinstance(item, str):
        auth = STRUCTURED_AUTH_VALUE_PATTERN.fullmatch(item)
        if auth is not None:
            state.record("structured_auth_field")
            return auth.group("scheme") + auth.group("space") + REDACTED_MARKER
    if SECRET_KEY_NAME_PATTERN.fullmatch(name) is not None:
        state.record("secret_key_value")
        return _masked_value_marker(item)
    return _redact_node(item, child_path, protected, state, depth)


def _scan_text(text: str, state: _RedactionState) -> str:
    """Apply the textual detector pipeline in fixed order."""
    matches_before = sum(state.counts.values())
    result = _substitute(TEXT_ASSIGNMENT_PATTERN, text, _assignment_replacement, "secret_key_value", state)
    result = _substitute(AUTH_HEADER_PATTERN, result, _auth_header_replacement, "auth_header", state)
    result = _substitute(PEM_PRIVATE_KEY_PATTERN, result, _marker_replacement, "pem_private_key", state)
    result = _substitute(UNTERMINATED_PEM_PRIVATE_KEY_PATTERN, result, _marker_replacement, "pem_private_key", state)
    result = _substitute(STANDALONE_BEARER_PATTERN, result, _standalone_bearer_replacement, "auth_header", state)
    result = _substitute(CONNECTION_SECRET_PATTERN, result, _connection_replacement, "connection_secret", state)
    result = _substitute(KNOWN_TOKEN_PREFIX_PATTERN, result, _marker_replacement, "known_token_prefix", state)
    if sum(state.counts.values()) > matches_before:
        state.affected_fields += 1
    return result


def _substitute(
    pattern: re.Pattern[str],
    text: str,
    replace: Callable[[re.Match[str]], str],
    category: str,
    state: _RedactionState,
) -> str:
    """Replace every pattern match, tallying one detection per match."""

    def counted(match: re.Match[str]) -> str:
        replacement = replace(match)
        if replacement != match.group(0):
            state.counts[category] += 1
        return replacement

    return pattern.sub(counted, text)


def _marker_replacement(match: re.Match[str]) -> str:
    del match
    return REDACTED_MARKER


def _assignment_replacement(match: re.Match[str]) -> str:
    name = _decode_escaped_key(match.group("name"))
    if SECRET_KEY_NAME_PATTERN.fullmatch(name) is None:
        return match.group(0)
    value = match.group("value")
    if value.startswith('"'):
        masked = f'"{REDACTED_MARKER}"'
    elif value.startswith("'"):
        masked = f"'{REDACTED_MARKER}'"
    else:
        masked = REDACTED_MARKER
    return match.group("open_quote") + match.group("name") + match.group("close_quote") + match.group("sep") + masked


def _decode_escaped_key(value: str) -> str:
    """Decode JSON unicode escapes in a textual object key for classification."""
    return re.sub(r"\\u([0-9a-f]{4})", lambda match: chr(int(match.group(1), 16)), value, flags=re.IGNORECASE)


def _auth_header_replacement(match: re.Match[str]) -> str:
    return match.group("name") + match.group("sep") + match.group("scheme") + match.group("space") + REDACTED_MARKER


def _standalone_bearer_replacement(match: re.Match[str]) -> str:
    return match.group("scheme") + REDACTED_MARKER


def _connection_replacement(match: re.Match[str]) -> str:
    return match.group(1) + REDACTED_MARKER + match.group(2)


def _masked_value_marker(value: object) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return REDACTED_BINARY_MARKER
    return REDACTED_MARKER


def _is_protected(name: str, path: tuple[str, ...], protected: frozenset[str]) -> bool:
    return (len(path) == 1 and name in protected) or ".".join(path) in protected


def _is_structured_auth_key(name: str) -> bool:
    return STRUCTURED_AUTH_KEY_PATTERN.fullmatch(name) is not None
