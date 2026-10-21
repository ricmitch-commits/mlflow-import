"""Bounded transient retries for MLflow calls.

Classification contract (Task 9): typed importer errors — the whole
:class:`~opencode_mlflow.errors.ImporterError` family — are PERMANENT.
They carry the importer's own stable, safe semantics (an unsupported
schema, a payload above the negotiated limit, a capability failure) and
are never made retryable by this layer. Raw protocol exceptions are
classified by SHAPE, never by message text:

- an exception exposing a string ``error_code`` (the MLflow protocol
  property) is transient exactly when the code is a server 5xx or a
  rate-limit code (:data:`TRANSIENT_ERROR_CODES`, the vocabulary of the
  pinned MLflow 3.17.2 ``ErrorCode`` mapping); invalid-request (4xx),
  not-found, and authorization codes are permanent;
- an exception whose class-hierarchy NAME matches a timeout/connection
  signature (:data:`_TRANSIENT_EXCEPTION_NAMES`) is transient — this
  catches the transport libraries' error types without importing them;
- everything else — including control-flow interruptions such as
  ``KeyboardInterrupt``/``SystemExit`` — is permanent, so an invalid
  request or an authorization failure is never retried, and an
  unrecognized failure fails fast rather than being retried blindly.

:func:`retry_call` bounds the total number of attempts, sleeps a
bounded exponential backoff between attempts (capped at the policy's
maximum delay, with an injectable jitter fraction), and re-raises the
original exception once the attempt budget is exhausted or the failure
is permanent. Sleep and randomness are injectable so tests never really
sleep.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from opencode_mlflow.errors import ImporterError

__all__ = [
    "TRANSIENT_ERROR_CODES",
    "RetryPolicy",
    "is_transient_error",
    "retry_call",
]

T = TypeVar("T")

#: MLflow protocol ``error_code`` values that are worth retrying: server
#: 5xx responses and rate limiting, from the pinned 3.17.2
#: ``ERROR_CODE_TO_HTTP_STATUS`` vocabulary.
TRANSIENT_ERROR_CODES = frozenset(
    {
        "INTERNAL_ERROR",  # 500
        "INVALID_STATE",  # 500
        "DATA_LOSS",  # 500
        "NOT_IMPLEMENTED",  # 501
        "TEMPORARILY_UNAVAILABLE",  # 503
        "DEADLINE_EXCEEDED",  # 504
        "REQUEST_LIMIT_EXCEEDED",  # 429
        "RESOURCE_EXHAUSTED",  # 429
    }
)

#: Exception class names (matched over the full MRO) that are transport
#: timeout/connection signatures: the retry layer never imports the
#: transport libraries, it recognizes their error types by name.
_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "TimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ConnectionError",
        "ConnectError",
        "ServerDisconnectedError",
        "NetworkError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
    }
)


def is_transient_error(exc: BaseException) -> bool:
    """Whether one failure is worth a bounded retry.

    Control-flow interruptions are never transient. Typed importer
    errors are never transient (Task 9 contract). A protocol
    ``error_code`` decides transientness by the pinned code vocabulary;
    otherwise a timeout/connection class name in the exception's
    hierarchy decides; anything unrecognized is permanent.
    """
    if not isinstance(exc, Exception):
        return False
    if isinstance(exc, ImporterError):
        return False
    error_code = getattr(exc, "error_code", None)
    if isinstance(error_code, str) and error_code:
        return error_code.strip().upper() in TRANSIENT_ERROR_CODES
    return any(klass.__name__ in _TRANSIENT_EXCEPTION_NAMES for klass in type(exc).__mro__)


@dataclass(frozen=True)
class RetryPolicy:
    """One bounded exponential-backoff policy.

    ``max_attempts`` is the TOTAL number of attempts (``1`` performs no
    retry). The delay after the ``n``-th failed attempt is
    ``min(initial * multiplier**(n-1), max_delay)`` plus an injectable
    jitter fraction of that base, capped at ``max_delay``.
    """

    max_attempts: int = 4
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must not be negative")
        if self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must not be negative")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError("jitter_fraction must be between 0 and 1")

    def delay_seconds(self, attempt: int, random_source: Callable[[], float]) -> float:
        """The backoff delay after the given 1-based failed attempt."""
        if attempt < 1:
            raise ValueError("attempt must be at least 1")
        exponential = self.initial_delay_seconds * (self.backoff_multiplier ** (attempt - 1))
        base = min(exponential, self.max_delay_seconds)
        return min(base * (1.0 + self.jitter_fraction * random_source()), self.max_delay_seconds)


def retry_call(
    operation: Callable[[], T],
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    random_source: Callable[[], float] = random.random,
) -> T:
    """Run ``operation`` under ``policy``, retrying only transient failures.

    A permanent failure (or a transient failure that exhausts the
    attempt budget) re-raises the original exception. The sleep and
    randomness sources are injectable so callers and tests stay
    deterministic.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return operation()
        except Exception as exc:
            if not is_transient_error(exc) or attempt >= policy.max_attempts:
                raise
            sleep(policy.delay_seconds(attempt, random_source))
