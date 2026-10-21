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

import sys
from collections.abc import Sequence

from opencode_mlflow.config import PROG, build_parser, parse_config
from opencode_mlflow.errors import ImporterError
from opencode_mlflow.importer import ImportDependencies, build_dependencies, exit_status, run_import
from opencode_mlflow.reporting import render_human, render_json

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
    try:
        config = parse_config(effective_argv)
    except ImporterError as error:
        return _report_failure(error)
    except SystemExit as error:  # argparse: --help exits 0, usage errors exit 2
        return error.code if isinstance(error.code, int) else 2
    effective_dependencies = dependencies if dependencies is not None else build_dependencies(config)
    result = run_import(config, effective_dependencies)
    output = render_human(result) if config.output == "human" else render_json(result)
    print(output)
    return exit_status(result)


def _report_failure(error: ImporterError) -> int:
    """Print one safe error line to stderr and return the exit status."""
    print(f"{PROG}: error ({error.code}): {error.safe_message}", file=sys.stderr)
    return error.exit_status
