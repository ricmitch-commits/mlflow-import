"""Console entry point for the OpenCode → MLflow importer.

``main`` parses and validates the command line, runs the import through
the two-pass pipeline, writes ONLY the selected output format (human or
JSON, both to stdout), and returns the process exit status from the
result (``0`` complete/no-op, ``1`` any item failed, ``2`` invalid or
pre-write failure). Argument-parsing and semantic-validation failures
print one safe error line to stderr instead of a result document,
because no valid invocation exists to report on. The importer never
logs or echoes transcript content: the result document carries only
identifiers, counts, warning/error codes, hashes, and the experiment
name.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from argparse import Namespace
from collections.abc import Sequence

from mlflow_import import resources
from mlflow_import.config import (
    PROG,
    build_parser,
    config_from_arguments,
    invocation_timeout_seconds,
    parse_arguments,
)
from mlflow_import.errors import ImporterError, ItemImportError
from mlflow_import.importer import (
    ImportDependencies,
    ImportResult,
    _storage_failure_result,
    build_dependencies,
    exit_status,
    run_import,
)
from mlflow_import.reporting import iter_human, iter_json
from mlflow_import.retry import _BUDGET, check_budget, retry_budget, terminal_reporting

__all__ = ["PROG", "build_parser", "main"]


def main(argv: Sequence[str] | None = None, dependencies: ImportDependencies | None = None) -> int:
    """Console-script entry point; returns the process exit status.

    ``argv`` defaults to ``sys.argv[1:]``. Argument parsing and semantic
    validation run before anything else: argparse handles syntax errors and
    ``--help`` (via :class:`SystemExit` with statuses ``2`` and ``0``), and
    :func:`parse_config` converts semantic failures into safe errors with
    exit status ``2``. A valid configuration runs the import through
    ``run_import`` with ``dependencies`` (the production set built by
    :func:`build_dependencies` when ``None``), and the selected output
    format is the only thing written.
    """
    effective_argv = sys.argv[1:] if argv is None else argv
    # Syntax/help needs no workspace. Source-format validation may build a
    # header inventory, so it shares the invocation ledger and deadline too.
    try:
        arguments = parse_arguments(effective_argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 2
    status = 2
    try:
        timeout = invocation_timeout_seconds(arguments)
        with resources.temporary_storage(), retry_budget(deadline_seconds=timeout):
            status = _main(arguments, dependencies)
        return status
    except ImporterError as error:
        failure = _report_failure(error)
        return max(status, failure) if error.code == "temporary_storage_cleanup_failed" else failure


def _main(arguments: Namespace, dependencies: ImportDependencies | None) -> int:
    try:
        config = config_from_arguments(arguments)
    except ImporterError as error:
        return _report_failure(error)
    except SystemExit as error:  # argparse: --help exits 0, usage errors exit 2
        return error.code if isinstance(error.code, int) else 2
    effective_dependencies = dependencies if dependencies is not None else build_dependencies(config)
    started = time.perf_counter() if config.log_level == "DEBUG" else None
    if config.log_level == "DEBUG":
        _write_debug("mlflow-import: import pipeline started")
    try:
        result = run_import(config, effective_dependencies)
    except ImporterError as error:
        if error.code not in {"retry_budget_exhausted", "operation_budget_exhausted"}:
            return _report_failure(error)
        result = _storage_failure_result(config, error, resources.current_storage())
    if started is not None:
        elapsed = time.perf_counter() - started
        _write_debug(f"mlflow-import: import pipeline complete elapsed_s={elapsed:.3f}")
    status = exit_status(result)
    try:
        try:
            # Check the work-to-output handoff once; lazy output has its own
            # finite local allowance and cannot renew invocation network retries.
            if budget := _BUDGET.get():
                budget.check()
            output = _stage_output(result, config.output)
        except (ImporterError, OSError, RuntimeError, sqlite3.Error) as failure:
            primary = result.error
            if primary is None and (budget := _BUDGET.get()) is not None:
                primary = budget.primary_error
            if primary is None:
                primary = (
                    failure
                    if isinstance(failure, ImporterError)
                    else ItemImportError("result_render_failed", "the result could not be rendered")
                )
            emergency = _storage_failure_result(config, primary, resources.current_storage())
            # Fixed-size in-memory envelope needs neither SQLite nor file I/O,
            # including when the staging store itself exhausted its quota.
            output = "".join(iter_json(emergency)) if config.output == "json" else "\n".join(iter_human(emergency))
            output += "\n"
            status = exit_status(emergency)
        # Publish only a fully rendered document. Never append a replacement
        # after a lazy encoder has already written a partial JSON document.
        sys.stdout.write(output)
    finally:
        try:
            result.close()
        except (OSError, RuntimeError, sqlite3.Error, ImporterError):
            print(
                f"{PROG}: error (result_cleanup_failed): private result storage could not be removed", file=sys.stderr
            )
            status = max(status, 1)
    return status


def _stage_output(result: ImportResult, output_format: str) -> str:
    """Stage quota-accounted output with finite time/work/byte allowances.

    The final string is capped at 8 MiB, independent of source size. Larger or
    slower reports fall back to the constant-size incomplete envelope before
    stdout is touched. No decoded source records are retained here.
    """
    with terminal_reporting():
        handle, path = resources.open_temp_file(prefix="mlflow-import-output-", mode="w+", encoding="utf-8")
        try:
            with handle:
                chunks = iter_json(result) if output_format == "json" else (line + "\n" for line in iter_human(result))
                size = 0
                for chunk in chunks:
                    check_budget()
                    size += len(chunk.encode("utf-8"))
                    if size > 8 * 1024 * 1024 - 1:
                        raise ItemImportError("result_output_exhausted", "the terminal output allowance was exhausted")
                    handle.write(chunk)
                if output_format == "json":
                    handle.write("\n")
                handle.seek(0)
                output = str(handle.read())
                check_budget()
                return output
        finally:
            resources.remove_file(path)


def _report_failure(error: ImporterError) -> int:
    """Print one safe error line to stderr and return the exit status."""
    print(f"{PROG}: error ({error.code}): {error.safe_message}", file=sys.stderr)
    return error.exit_status


def _write_debug(message: str) -> None:
    """Write best-effort diagnostics that cannot change the import outcome."""
    try:
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass
