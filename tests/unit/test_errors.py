"""Unit tests for the safe error hierarchy."""

from __future__ import annotations

import pytest

from opencode_mlflow.errors import (
    CapabilityError,
    ConfigurationError,
    ImporterError,
    ItemImportError,
    ReconciliationConflict,
    SnapshotError,
    SourceCompatibilityError,
)

pytestmark = pytest.mark.unit


def test_importer_error_carries_code_safe_message_and_exit_status() -> None:
    error = ImporterError("invalid_argument", "arguments are invalid", exit_status=3)

    assert error.code == "invalid_argument"
    assert error.safe_message == "arguments are invalid"
    assert error.exit_status == 3


def test_importer_error_accepts_positional_construction() -> None:
    error = ImporterError("trace_create_failed", "a safe message", 1)

    assert error.code == "trace_create_failed"
    assert error.exit_status == 1


def test_importer_error_string_form_is_only_the_safe_message() -> None:
    error = ImporterError("secret_code", "a sanitized message")

    assert str(error) == "a sanitized message"
    assert "secret_code" not in str(error)


def test_importer_error_is_an_exception() -> None:
    with pytest.raises(ImporterError):
        raise ConfigurationError("invalid_timestamp", "a safe message")


@pytest.mark.parametrize(
    ("error_type", "exit_status"),
    [
        (ConfigurationError, 2),
        (SourceCompatibilityError, 2),
        (SnapshotError, 2),
        (CapabilityError, 2),
        (ItemImportError, 1),
        (ReconciliationConflict, 1),
    ],
)
def test_subclasses_default_to_plan_exit_statuses(error_type: type[ImporterError], exit_status: int) -> None:
    error = error_type("some_code", "some safe message")

    assert isinstance(error, ImporterError)
    assert error.exit_status == exit_status
    assert str(error) == "some safe message"


def test_subclass_exit_status_can_be_overridden() -> None:
    error = ItemImportError("trace_create_failed", "a safe message", exit_status=2)

    assert error.exit_status == 2
