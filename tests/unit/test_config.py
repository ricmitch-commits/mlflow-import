"""Unit tests for the immutable importer configuration boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from opencode_mlflow.config import (
    ImportConfig,
    normalize_model_identity,
    normalize_path,
    parse_config,
    parse_timestamp,
)
from opencode_mlflow.errors import ConfigurationError

pytestmark = pytest.mark.unit


def test_parse_timestamp_requires_explicit_timezone():
    from datetime import timezone
    from opencode_mlflow.config import parse_timestamp

    assert parse_timestamp("2026-10-21T12:00:00Z").tzinfo == timezone.utc
    assert parse_timestamp("2026-10-21T08:00:00-04:00").isoformat() == "2026-10-21T12:00:00+00:00"

    with pytest.raises(ConfigurationError) as exc:
        parse_timestamp("2026-10-21T12:00:00")
    assert exc.value.code == "invalid_timestamp"
    assert "2026-10-21T12:00:00" not in str(exc.value)


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-21T12:00:00",
        "2026-10-21T12:00:00.500000",
        "2026-10-21",
        "not-a-timestamp",
    ],
)
def test_parse_timestamp_rejects_values_without_an_explicit_offset(value: str) -> None:
    with pytest.raises(ConfigurationError) as exc:
        parse_timestamp(value)
    assert exc.value.code == "invalid_timestamp"
    assert value not in str(exc.value)


def test_parse_timestamp_converts_explicit_offsets_to_utc() -> None:
    parsed = parse_timestamp("2026-10-21T08:00:00-04:00")

    assert parsed.tzinfo == timezone.utc
    assert parsed == datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def test_parse_timestamp_preserves_the_exact_instant() -> None:
    z_form = parse_timestamp("2026-10-21T12:00:00.123456Z")
    offset_form = parse_timestamp("2026-10-21T08:00:00.123456-04:00")

    assert z_form == offset_form
    assert z_form.microsecond == 123456
    assert z_form.tzinfo == timezone.utc


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" provider/ ", "provider/"),
        ("provider/", "provider/"),
        ("/model", "/model"),
        ("provider", "provider"),
        ("  AntHropic/Claude-4  ", "AntHropic/Claude-4"),
    ],
)
def test_normalize_model_identity_trims_whitespace_only(value: str, expected: str) -> None:
    assert normalize_model_identity(value) == expected


def test_normalize_model_identity_preserves_case() -> None:
    assert normalize_model_identity("Anthropic/Claude-Sonnet") == "Anthropic/Claude-Sonnet"
    assert normalize_model_identity("anthropic/claude-sonnet") == "anthropic/claude-sonnet"


def test_normalize_path_makes_relative_paths_absolute(
    restrictive_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(restrictive_tmp_path)
    # The OS may report the working directory through a resolved symlink
    # (e.g. /private/var on macOS); relative paths anchor at that reality.
    cwd = Path.cwd()

    assert normalize_path("db.sqlite") == cwd / "db.sqlite"
    assert normalize_path("../db.sqlite") == cwd.parent / "db.sqlite"


def test_normalize_path_resolves_dot_and_dotdot_lexically(
    restrictive_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(restrictive_tmp_path)
    cwd = Path.cwd()

    assert normalize_path("a/./b/../c") == cwd / "a" / "c"
    assert normalize_path("/does/not/exist/../exist") == Path("/does/not/exist")


def test_normalize_path_collapses_duplicate_separators() -> None:
    assert normalize_path("/var//tmp//x") == Path("/var/tmp/x")


def test_normalize_path_expands_user_home(restrictive_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(restrictive_tmp_path))

    assert normalize_path("~/data/opencode.db") == restrictive_tmp_path / "data" / "opencode.db"


def test_normalize_path_does_not_resolve_symlinks(restrictive_tmp_path: Path) -> None:
    real = restrictive_tmp_path / "real"
    real.mkdir()
    (restrictive_tmp_path / "link").symlink_to(real)
    linked = restrictive_tmp_path / "link" / "db.sqlite"

    assert normalize_path(str(linked)) == linked


def test_normalized_paths_support_component_aware_directory_comparison() -> None:
    requested = normalize_path("/repo/project")
    child = normalize_path("/repo/project/tools/./lint")
    sibling = normalize_path("/repo/project-tools")

    assert child.is_relative_to(requested)
    assert not sibling.is_relative_to(requested)


def test_import_config_is_frozen() -> None:
    config = parse_config(["import", "--experiment-name", "exp"])

    with pytest.raises(FrozenInstanceError):
        config.dry_run = True


def test_import_config_fields_match_the_contract() -> None:
    # The frozen dataclass exposes exactly the fields the plan mandates.
    assert [field for field in ImportConfig.__dataclass_fields__] == [
        "experiment_name",
        "db_path",
        "tracking_uri",
        "from_time",
        "to_time",
        "directory",
        "model",
        "min_user_turns",
        "redact",
        "dry_run",
        "output",
        "log_level",
    ]


def test_parse_config_applies_defaults(restrictive_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(restrictive_tmp_path))

    config = parse_config(["import", "--experiment-name", "exp"])

    assert config.experiment_name == "exp"
    assert config.db_path == restrictive_tmp_path / ".local/share/opencode/opencode.db"
    assert config.tracking_uri is None
    assert config.from_time is None
    assert config.to_time is None
    assert config.directory is None
    assert config.model is None
    assert config.min_user_turns is None
    assert config.redact is False
    assert config.dry_run is False
    assert config.output == "human"
    assert config.log_level == "INFO"


def test_parse_config_normalizes_every_option(restrictive_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(restrictive_tmp_path)
    monkeypatch.setenv("HOME", str(restrictive_tmp_path))
    cwd = Path.cwd()

    config = parse_config(
        [
            "import",
            "--experiment-name",
            "exp",
            "--db",
            "~/source/db.sqlite",
            "--tracking-uri",
            "http://localhost:5000",
            "--from-time",
            "2026-10-21T08:00:00-04:00",
            "--to-time",
            "2026-10-21T13:00:00Z",
            "--directory",
            "proj/../work",
            "--model",
            "  AntHropic/Claude-4  ",
            "--min-user-turns",
            "3",
            "--redact",
            "--dry-run",
            "--output",
            "json",
            "--log-level",
            "DEBUG",
        ]
    )

    assert config.experiment_name == "exp"
    assert config.db_path == restrictive_tmp_path / "source" / "db.sqlite"
    assert config.tracking_uri == "http://localhost:5000"
    assert config.from_time == datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    assert config.to_time == datetime(2026, 10, 21, 13, 0, tzinfo=timezone.utc)
    assert config.directory == cwd / "work"
    assert config.model == "AntHropic/Claude-4"
    assert config.min_user_turns == 3
    assert config.redact is True
    assert config.dry_run is True
    assert config.output == "json"
    assert config.log_level == "DEBUG"


@pytest.mark.parametrize("bound", ["--from-time", "--to-time"])
def test_parse_config_rejects_naive_timestamps(bound: str) -> None:
    value = "2026-10-21T12:00:00"

    with pytest.raises(ConfigurationError) as exc:
        parse_config(["import", "--experiment-name", "exp", bound, value])
    assert exc.value.code == "invalid_timestamp"
    assert value not in str(exc.value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0), ("2", 2)],
)
def test_parse_config_accepts_explicit_thresholds(value: str, expected: int) -> None:
    config = parse_config(["import", "--experiment-name", "exp", "--min-user-turns", value])

    assert config.min_user_turns == expected


@pytest.mark.parametrize("value", ["-1", "abc", "1.5"])
def test_parse_config_rejects_invalid_thresholds(value: str) -> None:
    with pytest.raises(ConfigurationError) as exc:
        parse_config(["import", "--experiment-name", "exp", "--min-user-turns", value])
    assert exc.value.code == "invalid_min_user_turns"
    assert value not in str(exc.value)


def test_parse_config_rejects_blank_experiment_name() -> None:
    with pytest.raises(ConfigurationError) as exc:
        parse_config(["import", "--experiment-name", "   "])
    assert exc.value.code == "invalid_experiment_name"


def test_parse_config_rejects_blank_model() -> None:
    with pytest.raises(ConfigurationError) as exc:
        parse_config(["import", "--experiment-name", "exp", "--model", "  "])
    assert exc.value.code == "invalid_model"


def test_parse_config_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit) as exc:
        parse_config([])
    assert exc.value.code == 2


def test_parse_config_requires_an_experiment_name() -> None:
    with pytest.raises(SystemExit) as exc:
        parse_config(["import"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["import", "--experiment-name", "exp", "--output", "yaml"],
        ["import", "--experiment-name", "exp", "--log-level", "TRACE"],
        ["import", "--experiment-name", "exp", "--bogus"],
    ],
)
def test_parse_config_delegates_syntax_errors_to_argparse(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_config(argv)
    assert exc.value.code == 2
