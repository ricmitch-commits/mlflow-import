"""Unit tests for the CLI boundary and safe error conversion.

These tests never touch the network, MLflow, or SQLite writes: the CLI is invoked
in-process through ``main`` (which returns the process exit status), the
``--help``/no-subcommand process-level behavior is checked through a child
interpreter that calls the console-script entry point, and the import wiring is
driven through injected fake dependencies (the real source pipeline over a
sanitized synthetic database with the in-memory MLflow/lock fakes).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from opencode_mlflow.cli import build_parser, main
from opencode_mlflow.config import ImportConfig
from tests.support.import_fakes import FakeImportDependencies

pytestmark = pytest.mark.unit

VALID_INVOCATION = ["import", "--experiment-name", "cli-unit-experiment"]


def test_main_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0

    assert "import" in capsys.readouterr().out


def test_main_without_a_subcommand_exits_two() -> None:
    assert main([]) == 2


def test_main_missing_experiment_name_exits_two() -> None:
    assert main(["import"]) == 2


def test_main_unknown_option_exits_two() -> None:
    assert main([*VALID_INVOCATION, "--no-such-option"]) == 2


def test_main_invalid_output_choice_exits_two() -> None:
    assert main([*VALID_INVOCATION, "--output", "yaml"]) == 2


def test_main_invalid_log_level_choice_exits_two() -> None:
    assert main([*VALID_INVOCATION, "--log-level", "TRACE"]) == 2


def test_main_reports_source_failure_through_the_real_dependencies(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A valid configuration reaches the real importer seam and reports safely.

    The real dependencies are built (no fakes): a missing source database
    fails at the source boundary before any MLflow work, so the run is
    deterministic on every machine.
    """
    missing = tmp_path / "missing-source.db"

    assert main(["import", "--experiment-name", "cli-real-seam", "--db", str(missing), "--output", "json"]) == 2

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["schema_version"] == "opencode-mlflow.result.v1"
    assert report["errors"][0]["code"] == "source_unavailable"
    # The failure report never echoes the source path.
    assert str(missing) not in captured.out
    assert str(missing) not in captured.err


def test_main_converts_timestamp_failure_to_safe_error(capsys: pytest.CaptureFixture[str]) -> None:
    value = "2026-10-21T12:00:00"

    assert main([*VALID_INVOCATION, "--from-time", value]) == 2

    captured = capsys.readouterr()
    assert "invalid_timestamp" in captured.err
    assert value not in captured.err


def test_main_converts_threshold_failure_to_safe_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*VALID_INVOCATION, "--min-user-turns", "-1"]) == 2

    captured = capsys.readouterr()
    assert "invalid_min_user_turns" in captured.err


def test_build_parser_exposes_the_import_subcommand() -> None:
    args = build_parser().parse_args(
        [
            "import",
            "--experiment-name",
            "exp",
            "--db",
            "/tmp/db.sqlite",
            "--tracking-uri",
            "http://localhost:5000",
            "--from-time",
            "2026-10-21T00:00:00Z",
            "--to-time",
            "2026-10-22T00:00:00Z",
            "--directory",
            "/repo/project",
            "--model",
            "provider/model",
            "--min-user-turns",
            "2",
            "--redact",
            "--dry-run",
            "--output",
            "json",
            "--log-level",
            "WARNING",
        ]
    )

    assert args.command == "import"
    assert args.experiment_name == "exp"
    assert args.db == "/tmp/db.sqlite"
    assert args.tracking_uri == "http://localhost:5000"
    assert args.from_time == "2026-10-21T00:00:00Z"
    assert args.to_time == "2026-10-22T00:00:00Z"
    assert args.directory == "/repo/project"
    assert args.model == "provider/model"
    # Thresholds stay raw strings here: semantic validation belongs to
    # parse_config so failures surface as safe errors instead of argparse
    # usage text that echoes the offending value.
    assert args.min_user_turns == "2"
    assert args.redact is True
    assert args.dry_run is True
    assert args.output == "json"
    assert args.log_level == "WARNING"


def _run_entry_point(argv: list[str]) -> int:
    """Run the console-script entry point in a child process."""
    program = "import sys; from opencode_mlflow.cli import main; sys.exit(main())"
    completed = subprocess.run(
        [sys.executable, "-c", program, *argv],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode


def test_entry_point_help_process_exits_zero() -> None:
    assert _run_entry_point(["--help"]) == 0


def test_entry_point_without_subcommand_process_exits_two() -> None:
    assert _run_entry_point([]) == 2


# ---------------------------------------------------------------------------
# Import wiring: main → run_import → selected output format → exit status
# ---------------------------------------------------------------------------


def _argv(config: ImportConfig, *extra: str) -> list[str]:
    return [
        "import",
        "--experiment-name",
        config.experiment_name,
        "--db",
        str(config.db_path),
        *extra,
    ]


def test_main_runs_the_import_with_injected_dependencies(
    capsys: pytest.CaptureFixture[str], config: ImportConfig, import_dependencies: FakeImportDependencies
) -> None:
    assert main(_argv(config, "--output", "json"), dependencies=import_dependencies) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["schema_version"] == "opencode-mlflow.result.v1"
    assert report["dry_run"] is False
    assert [turn["action"] for turn in report["sessions"][0]["turns"]] == ["imported", "imported"]
    assert import_dependencies.gateway.mutations
    assert import_dependencies.lock_factory.acquisitions


def test_main_writes_human_output_to_stdout(
    capsys: pytest.CaptureFixture[str], config: ImportConfig, import_dependencies: FakeImportDependencies
) -> None:
    assert main(_argv(config), dependencies=import_dependencies) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "opencode-mlflow import complete" in captured.out
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.out)


def test_main_dry_run_writes_json_and_no_mutations(
    capsys: pytest.CaptureFixture[str], config: ImportConfig, import_dependencies: FakeImportDependencies
) -> None:
    assert main(_argv(config, "--dry-run", "--output", "json"), dependencies=import_dependencies) == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["dry_run"] is True
    assert report["sessions"][0]["turns"][0]["action"] == "skipped"
    assert import_dependencies.gateway.mutations == []
    assert import_dependencies.lock_factory.acquisitions == []


def test_main_returns_one_for_item_failures(
    capsys: pytest.CaptureFixture[str], config: ImportConfig, import_dependencies: FakeImportDependencies
) -> None:
    import_dependencies.gateway.fail_next_trace_verification = 1

    assert main(_argv(config, "--output", "json"), dependencies=import_dependencies) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["turns_failed"] == 1
    assert report["summary"]["turns_imported"] == 1


def test_main_returns_two_for_pre_write_failures_with_a_report(
    capsys: pytest.CaptureFixture[str], config: ImportConfig, restrictive_tmp_path: Path
) -> None:
    from tests.support.fake_gateway import FakeMlflowGateway
    from tests.support.import_fakes import build_import_fakes

    dependencies = build_import_fakes(
        restrictive_tmp_path=restrictive_tmp_path, gateway=FakeMlflowGateway(compatible=False)
    )

    assert main(_argv(config, "--output", "json"), dependencies=dependencies) == 2

    report = json.loads(capsys.readouterr().out)
    assert report["errors"][0]["code"] == "mlflow_capability_unsupported"
    assert dependencies.gateway.mutations == []
