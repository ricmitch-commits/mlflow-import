"""Unit tests for bounded transient retries.

Pins the Task 9 classification contract the retry layer must honor:
typed importer errors (the whole ``ImporterError`` family) are PERMANENT,
raw MLflow/protocol exceptions are classified by their ``error_code``
shape (server 5xx and rate-limit codes are transient; 4xx invalid
request and authorization codes are permanent), timeout/connection
exception names are transient, and everything unrecognized is permanent
so an invalid request or authorization failure is never retried
indefinitely. ``retry_call`` bounds attempts, applies exponential
backoff capped at the policy maximum, and takes injectable
sleep/jitter sources so tests never really sleep.
"""

from __future__ import annotations

import pytest

from opencode_mlflow.errors import (
    CapabilityError,
    ConfigurationError,
    ImporterError,
    ItemImportError,
    ReconciliationConflict,
)
from opencode_mlflow.retry import (
    RetryPolicy,
    is_transient_error,
    retry_call,
)
from tests.support.fake_gateway import FakeMlflowException

pytestmark = pytest.mark.unit


class _ReadTimeout(TimeoutError):
    """A requests-shaped timeout name that is not a builtin."""


class _RecordingSleep:
    """Injectable sleep source recording every requested delay."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def test_server_and_rate_limit_error_codes_are_transient() -> None:
    codes = [
        "INTERNAL_ERROR",
        "INVALID_STATE",
        "DATA_LOSS",
        "NOT_IMPLEMENTED",
        "TEMPORARILY_UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "REQUEST_LIMIT_EXCEEDED",
        "RESOURCE_EXHAUSTED",
    ]

    for code in codes:
        assert is_transient_error(FakeMlflowException("synthetic failure", code))


def test_invalid_request_and_authorization_error_codes_are_permanent() -> None:
    codes = [
        "INVALID_PARAMETER",
        "INVALID_PARAMETER_VALUE",
        "BAD_REQUEST",
        "RESOURCE_DOES_NOT_EXIST",
        "NOT_FOUND",
        "RESOURCE_ALREADY_EXISTS",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
        "CUSTOMER_UNAUTHORIZED",
    ]

    for code in codes:
        assert not is_transient_error(FakeMlflowException("synthetic failure", code))


def test_typed_importer_errors_are_permanent() -> None:
    errors: list[ImporterError] = [
        ConfigurationError("synthetic_code", "synthetic message"),
        ItemImportError("synthetic_code", "synthetic message"),
        CapabilityError("synthetic_code", "synthetic message"),
        ReconciliationConflict("synthetic_code", "synthetic message"),
    ]

    for error in errors:
        assert not is_transient_error(error)


def test_timeout_and_connection_exception_names_are_transient() -> None:
    transient = [
        TimeoutError("synthetic timeout"),
        _ReadTimeout("synthetic read timeout"),
        ConnectionError("synthetic connection failure"),
        ConnectionResetError("synthetic reset"),
        ConnectionRefusedError("synthetic refusal"),
    ]

    for error in transient:
        assert is_transient_error(error)


def test_unrecognized_and_control_flow_exceptions_are_permanent() -> None:
    assert not is_transient_error(ValueError("synthetic value error"))
    assert not is_transient_error(KeyError("synthetic_key"))
    assert not is_transient_error(KeyboardInterrupt())
    assert not is_transient_error(SystemExit(1))


def test_retry_call_returns_the_first_success_without_sleeping() -> None:
    sleep = _RecordingSleep()
    calls: list[int] = []

    def operation() -> int:
        calls.append(1)
        return 42

    result = retry_call(operation, RetryPolicy(), sleep=sleep, random_source=lambda: 0.0)

    assert result == 42
    assert len(calls) == 1
    assert sleep.delays == []


def test_retry_call_retries_transient_failures_with_bounded_backoff() -> None:
    sleep = _RecordingSleep()
    attempts: list[int] = []
    policy = RetryPolicy(
        max_attempts=4,
        initial_delay_seconds=1.0,
        max_delay_seconds=100.0,
        backoff_multiplier=2.0,
        jitter_fraction=0.0,
    )

    def operation() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise FakeMlflowException("synthetic transient failure", "TEMPORARILY_UNAVAILABLE")
        return "recovered"

    result = retry_call(operation, policy, sleep=sleep, random_source=lambda: 0.0)

    assert result == "recovered"
    assert len(attempts) == 3
    assert sleep.delays == [1.0, 2.0]


def test_retry_call_stops_after_the_policy_attempt_bound() -> None:
    sleep = _RecordingSleep()
    attempts: list[int] = []
    policy = RetryPolicy(max_attempts=3, initial_delay_seconds=0.0, max_delay_seconds=0.0)

    def operation() -> None:
        attempts.append(1)
        raise FakeMlflowException("synthetic persistent failure", "TEMPORARILY_UNAVAILABLE")

    with pytest.raises(FakeMlflowException):
        retry_call(operation, policy, sleep=sleep, random_source=lambda: 0.0)

    assert len(attempts) == 3
    assert len(sleep.delays) == 2


def test_retry_call_never_retries_permanent_errors() -> None:
    sleep = _RecordingSleep()
    attempts: list[int] = []

    def operation() -> None:
        attempts.append(1)
        raise ItemImportError("synthetic_code", "synthetic permanent failure")

    with pytest.raises(ItemImportError):
        retry_call(operation, RetryPolicy(), sleep=sleep, random_source=lambda: 0.0)

    assert len(attempts) == 1
    assert sleep.delays == []


def test_retry_call_never_retries_control_flow_interruptions() -> None:
    sleep = _RecordingSleep()
    attempts: list[int] = []

    def operation() -> None:
        attempts.append(1)
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        retry_call(operation, RetryPolicy(), sleep=sleep, random_source=lambda: 0.0)

    assert len(attempts) == 1


def test_retry_call_jitter_is_injectable_and_additive() -> None:
    sleep = _RecordingSleep()
    policy = RetryPolicy(
        max_attempts=3,
        initial_delay_seconds=1.0,
        max_delay_seconds=100.0,
        backoff_multiplier=2.0,
        jitter_fraction=0.5,
    )
    attempts: list[int] = []

    def operation() -> None:
        attempts.append(1)
        if len(attempts) < policy.max_attempts:
            raise TimeoutError("synthetic timeout")
        raise TimeoutError("synthetic timeout")

    with pytest.raises(TimeoutError):
        retry_call(operation, policy, sleep=sleep, random_source=lambda: 1.0)

    # A full jitter fraction of the exponential base is added: 1.0 -> 1.5, 2.0 -> 3.0.
    assert sleep.delays == [1.5, 3.0]


def test_retry_policy_caps_the_backoff_delay() -> None:
    policy = RetryPolicy(
        initial_delay_seconds=4.0,
        max_delay_seconds=10.0,
        backoff_multiplier=4.0,
        jitter_fraction=0.0,
    )

    assert policy.delay_seconds(1, lambda: 0.0) == 4.0
    assert policy.delay_seconds(2, lambda: 0.0) == 10.0
    assert policy.delay_seconds(9, lambda: 0.0) == 10.0


def test_retry_policy_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(initial_delay_seconds=-1.0)
    with pytest.raises(ValueError):
        RetryPolicy(max_delay_seconds=-1.0)
    with pytest.raises(ValueError):
        RetryPolicy(backoff_multiplier=0.5)
    with pytest.raises(ValueError):
        RetryPolicy(jitter_fraction=1.5)
