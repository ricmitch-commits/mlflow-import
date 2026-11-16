"""Invocation timeout validation precedes any source inspection."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from mlflow_import.config import (
    ImportConfig,
    config_from_arguments,
    invocation_timeout_seconds,
    parse_arguments,
    parse_config,
)
from mlflow_import.errors import ConfigurationError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("prefix", [[], ["import"]])
@pytest.mark.parametrize("value, expected", [("1800", 1800.0), ("0.125", 0.125), ("1e2", 100.0)])
def test_timeout_flat_and_legacy_syntax(prefix: list[str], value: str, expected: float) -> None:
    arguments = [*prefix, "--experiment-name", "SessionsAI", "--timeout-seconds", value]

    assert invocation_timeout_seconds(parse_arguments(arguments)) == expected
    assert parse_config(arguments).timeout_seconds == expected


@pytest.mark.parametrize("prefix", [[], ["import"]])
def test_timeout_default(prefix: list[str]) -> None:
    config = parse_config([*prefix, "--experiment-name", "SessionsAI"])

    assert config.timeout_seconds == 600.0
    assert replace(config, dry_run=True).timeout_seconds == 600.0
    assert replace(config, timeout_seconds=1800.0).timeout_seconds == 1800.0


def test_public_config_default_and_replace() -> None:
    config = ImportConfig("SessionsAI", Path("/unused/source"))

    assert config.timeout_seconds == 600.0
    assert replace(config, timeout_seconds=0.25).timeout_seconds == 0.25


@pytest.mark.parametrize("value", [0.125, 1800.0])
def test_public_config_accepts_valid_timeout(value: float) -> None:
    config = ImportConfig("SessionsAI", Path("/unused/source"), timeout_seconds=value)

    assert config.timeout_seconds == value
    assert replace(config, timeout_seconds=0.25).timeout_seconds == 0.25


@pytest.mark.parametrize("use_replace", [False, True])
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), 0, -0.0, -1, "private-input", None, [], 10**1000],
)
def test_public_config_rejects_invalid_timeout(value: object, use_replace: bool) -> None:
    config = ImportConfig("SessionsAI", Path("/unused/source"))

    with pytest.raises(ConfigurationError) as caught:
        if use_replace:
            replace(config, timeout_seconds=value)
        else:
            ImportConfig("SessionsAI", Path("/unused/source"), timeout_seconds=value)

    assert caught.value.code == "invalid_timeout"
    assert caught.value.exit_status == 2
    assert str(caught.value) == "--timeout-seconds must be a finite positive number"
    assert caught.value.__cause__ is None
    assert config.timeout_seconds == 600.0


def test_older_namespace_retains_timeout_default() -> None:
    arguments = parse_arguments(["--experiment-name", "SessionsAI"])
    del arguments.timeout_seconds

    assert invocation_timeout_seconds(Namespace()) == 600.0
    assert config_from_arguments(arguments).timeout_seconds == 600.0


@pytest.mark.parametrize("prefix", [[], ["import"]])
@pytest.mark.parametrize("source_options", [["--source", "/unused/source"], ["--harness", "claude"]])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "0", "-1", "1e999", "1e-999", "private-input", ""])
def test_invalid_timeout_fails_before_source_reads(
    prefix: list[str], source_options: list[str], value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    detect = Mock(side_effect=AssertionError("source detection must not run"))
    adapter = Mock(side_effect=AssertionError("adapter discovery must not run"))
    monkeypatch.setattr("mlflow_import.config.detect_harness", detect)
    monkeypatch.setattr("mlflow_import.config.get_adapter", adapter)

    with pytest.raises(ConfigurationError) as caught:
        parse_config([*prefix, "--experiment-name", "SessionsAI", *source_options, f"--timeout-seconds={value}"])

    assert caught.value.code == "invalid_timeout"
    assert caught.value.exit_status == 2
    assert str(caught.value) == "--timeout-seconds must be a finite positive number"
    assert caught.value.__cause__ is None
    detect.assert_not_called()
    adapter.assert_not_called()


@pytest.mark.parametrize("value", [None, [], 10**1000, float("nan"), float("inf"), 0, -0.25])
def test_timeout_helper_rejects_invalid_namespace_values(value: object) -> None:
    with pytest.raises(ConfigurationError) as caught:
        invocation_timeout_seconds(Namespace(timeout_seconds=value))

    assert caught.value.code == "invalid_timeout"
    assert str(caught.value) == "--timeout-seconds must be a finite positive number"
