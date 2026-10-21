"""Fake ``Redactor`` implementations for content-layer tests.

Real detector-backed redaction arrives in Task 7; until then these
deterministic stubs prove the injection seam: :class:`IdentityRedactor`
stands in for verbatim mode, :class:`SecretMaskingRedactor` masks the
values of ``secret`` keys (and matching ``"secret": "..."`` spans inside
raw text) the way a structured secret detector would, and
:class:`RecordingRedactor` observes every call while preserving values.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping

_SECRET_SPAN = re.compile(r'("secret"\s*:\s*")[^"]*')


class IdentityRedactor:
    """Redactor stand-in that returns every value unchanged (verbatim mode)."""

    def redact(self, value: object, protected_paths: Collection[str]) -> object:
        return value


class SecretMaskingRedactor:
    """Redactor stand-in masking detected secret values with ``[REDACTED]``.

    Mapping values under a ``secret`` key are masked (unless the key is
    protected), and raw text has its ``"secret": "..."`` spans masked in
    place, mirroring how an effective redaction pass runs before
    canonicalization: two records differing only in the original value
    of a detected secret become identical effective values.
    """

    def redact(self, value: object, protected_paths: Collection[str]) -> object:
        if isinstance(value, Mapping):
            masked: dict[object, object] = {}
            for key, item in value.items():
                if key == "secret" and key not in protected_paths:
                    masked[key] = "[REDACTED]"
                elif key in protected_paths:
                    masked[key] = item
                else:
                    masked[key] = self.redact(item, protected_paths)
            return masked
        if isinstance(value, str):
            return _SECRET_SPAN.sub(r"\1[REDACTED]", value)
        return value


class RecordingRedactor:
    """Redactor stand-in that records every call and preserves values."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, frozenset[str]]] = []

    def redact(self, value: object, protected_paths: Collection[str]) -> object:
        self.calls.append((value, frozenset(protected_paths)))
        return value
