"""Selective first-envelope JSONL headers without decoding transcript fields.

Claude's first record mixes header facts and a prompt. Discovery must not build
that prompt (or credential fields) just to decide whether its session is eligible.
This bounded lexical reader skips unselected values, including nested strings,
and decodes only the adapter's allowlisted header keys. Selected files still get
full strict schema/JSON validation during freezing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, TextIO

from mlflow_import.retry import check_budget

_STRING_SPECIAL = re.compile(r'["\\\r\n]')
HEADER_KEYS = frozenset(
    {
        "type",
        "sessionId",
        "session_id",
        "id",
        "version",
        "cli_version",
        "timestamp",
        "created_at",
        "createdAt",
        "startTime",
        "lastUpdated",
        "parentSession",
        "parentSessionId",
        "parent_session_id",
        "forked_from_id",
        "cwd",
        "directory",
        "title",
        "summary",
        "directories",
        "model",
        "model_name",
        "git",
        "kind",
        "projectHash",
        "agentId",
        "parentToolUseID",
        "parent_tool_use_id",
        "client_request_id",
        "user",
        "user_id",
        "username",
        "mlflow.trace.user",
        "mlflow.user",
    }
)


class _HeaderReader:
    def __init__(self, handle: TextIO, limit: int) -> None:
        self.handle, self.limit = handle, limit
        self.buffer = ""
        self.offset = 0
        self.consumed = 0

    def peek(self) -> str:
        if self.offset == len(self.buffer):
            check_budget()
            self.buffer = self.handle.read(8192)
            self.offset = 0
        return self.buffer[self.offset : self.offset + 1]

    def take(self, length: int = 1) -> str:
        if not self.peek():
            raise ValueError("incomplete header")
        text = self.buffer[self.offset : self.offset + length]
        self.offset += len(text)
        self.consumed += len(text.encode("utf-8"))
        if self.consumed > self.limit:
            raise ValueError("oversized header envelope")
        return text

    def spaces(self) -> None:
        while self.peek() and self.peek() in " \t":
            self.take()

    def string(self, capture: bool) -> str:
        if self.take() != '"':
            raise ValueError("invalid header key")
        parts = ['"'] if capture else []
        while self.peek():
            match = _STRING_SPECIAL.search(self.buffer, self.offset)
            length = (match.start() if match else len(self.buffer)) - self.offset
            if length:
                text = self.take(length)
                if capture:
                    parts.append(text)
            if not match:
                continue
            char = self.take()
            if char in "\r\n":
                raise ValueError("multiline header string")
            if capture:
                parts.append(char)
            if char == '"':
                return "".join(parts) if capture else ""
            escaped = self.take()
            if escaped in "\r\n":
                raise ValueError("invalid header escape")
            if capture:
                parts.append(escaped)
        raise ValueError("unterminated header string")

    def value(self, capture: bool) -> Any:
        char = self.peek()
        if char == '"':
            raw = self.string(capture)
            return json.loads(raw) if capture else None
        parts: list[str] = []
        if char in "{[" and char:
            stack = ["}" if self.take() == "{" else "]"]
            if capture:
                parts.append(char)
            while stack:
                char = self.peek()
                if not char or char in "\r\n":
                    raise ValueError("incomplete header value")
                if char == '"':
                    text = self.string(capture)
                else:
                    text = self.take()
                    if char in "{[":
                        stack.append("}" if char == "{" else "]")
                        if len(stack) > 256:
                            raise ValueError("excessive header nesting")
                    elif char in "}]" and char != stack.pop():
                        raise ValueError("invalid header nesting")
                if capture:
                    parts.append(text)
        else:
            while self.peek() and self.peek() not in " \t\r\n,}]":
                text = self.take()
                if capture:
                    parts.append(text)
        return json.loads("".join(parts)) if capture else None

    def bounded_value(self, maximum_bytes: int) -> Any:
        """Bound capture before accumulating strings/containers or decoding JSON.

        Skipped transcript values retain the outer streaming file limit; a
        metadata value gets an independent allowance at its current offset.
        """
        previous = self.limit
        self.limit = min(previous, self.consumed + maximum_bytes)
        try:
            return self.value(True)
        finally:
            self.limit = previous

    def object(self, *, payload: bool = False) -> dict[str, Any]:
        if self.take() != "{":
            raise ValueError("header must be an object")
        result: dict[str, Any] = {}
        while True:
            self.spaces()
            if self.peek() == "}":
                self.take()
                return result
            key = json.loads(self.string(True))
            self.spaces()
            if self.take() != ":":
                raise ValueError("invalid header field")
            self.spaces()
            if key in HEADER_KEYS:
                if key in result:
                    raise ValueError("duplicate header field")
                result[key] = self.value(True)
            elif key == "payload" and not payload and self.peek() == "{":
                if key in result:
                    raise ValueError("duplicate payload header")
                result[key] = self.object(payload=True)
            elif key == "messages":
                result[key] = None  # Format discriminator; never decode a legacy transcript array.
                self.value(False)
            else:
                self.value(False)
            self.spaces()
            char = self.take()
            if char == "}":
                return result
            if char != ",":
                raise ValueError("invalid header delimiter")


def read_header(handle: TextIO, limit: int) -> dict[str, Any] | None:
    return next(read_headers(handle, limit), None)


def read_headers(handle: TextIO, limit: int) -> Iterator[dict[str, Any]]:
    """Stream envelope headers with one bounded lexer, retaining read-ahead."""
    reader = _HeaderReader(handle, limit)
    while True:
        reader.consumed = 0
        while reader.peek() and reader.peek() in " \t\r\n":
            reader.take()
        if not reader.peek():
            return
        yield reader.object()
