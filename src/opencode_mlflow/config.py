"""Immutable, normalized importer configuration.

:func:`parse_config` converts a command line into a frozen
:class:`ImportConfig`, rejecting invalid values before any source or MLflow
object is constructed. The argument parser lives in this module because
``parse_config`` owns argument parsing; :mod:`opencode_mlflow.cli`
re-exports :func:`build_parser` as the public CLI seam (config must not
import from cli, which would create an import cycle).

Source existence is deliberately not validated here; that belongs to the
source preflight boundary.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from opencode_mlflow.errors import ConfigurationError

PROG = "opencode-mlflow"
"""Console program name used in parser output and safe error lines."""

DEFAULT_DB_PATH = "~/.local/share/opencode/opencode.db"
"""Default OpenCode database path; expanded and absolutized during parsing."""

OUTPUT_CHOICES = ("human", "json")
LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR")
DEFAULT_OUTPUT = "human"
DEFAULT_LOG_LEVEL = "INFO"

_TIMESTAMP_MESSAGE = "value must be an ISO-8601 timestamp with an explicit timezone offset or Z"
_THRESHOLD_MESSAGE = "--min-user-turns must be a non-negative integer"


@dataclass(frozen=True)
class ImportConfig:
    """Immutable, normalized representation of one importer invocation.

    ``from_time``/``to_time`` are UTC instants (inclusive bounds applied by
    the selection layer), ``db_path``/``directory`` are lexically normalized
    absolute paths, ``model`` is a whitespace-trimmed identity with case
    and completeness preserved, and ``min_user_turns`` is either ``None``
    (option omitted; no restriction) or a non-negative threshold the
    selection layer applies as a strict greater-than.
    """

    experiment_name: str
    db_path: Path
    tracking_uri: str | None = None
    from_time: datetime | None = None
    to_time: datetime | None = None
    directory: Path | None = None
    model: str | None = None
    min_user_turns: int | None = None
    redact: bool = False
    dry_run: bool = False
    output: str = DEFAULT_OUTPUT
    log_level: str = DEFAULT_LOG_LEVEL


def build_parser() -> argparse.ArgumentParser:
    """Build the importer's argument parser.

    Syntax errors (unknown options, missing required values, invalid
    ``--output``/``--log-level`` choices) are delegated to argparse, which
    exits with status ``2``; ``--help`` exits ``0``. Semantic failures are
    validated in :func:`parse_config` so they surface as safe errors instead
    of usage text that echoes the offending value.
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Import OpenCode SQLite sessions into MLflow as traces.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    import_command = subparsers.add_parser(
        "import",
        help="import sessions from an OpenCode database into MLflow traces",
    )
    import_command.add_argument(
        "--experiment-name",
        required=True,
        help="MLflow experiment name; resolved by name and created if absent",
    )
    import_command.add_argument(
        "--db",
        help=f"OpenCode SQLite database path (default: {DEFAULT_DB_PATH})",
    )
    import_command.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI; otherwise standard MLflow client/environment configuration",
    )
    import_command.add_argument(
        "--from-time",
        help="inclusive lower bound on session creation time (ISO-8601 with offset or Z)",
    )
    import_command.add_argument(
        "--to-time",
        help="inclusive upper bound on session creation time (ISO-8601 with offset or Z)",
    )
    import_command.add_argument(
        "--directory",
        help="restrict imports to this directory and all of its descendants",
    )
    import_command.add_argument(
        "--model",
        help="exact normalized provider/model identity (e.g. provider/model)",
    )
    import_command.add_argument(
        "--min-user-turns",
        help=(
            "select sessions with strictly more than N logical user turns "
            "(non-negative integer; omit for no restriction)"
        ),
    )
    import_command.add_argument(
        "--redact",
        action="store_true",
        help="mask secrets in trace values, metadata, errors, and artifacts",
    )
    import_command.add_argument(
        "--dry-run",
        action="store_true",
        help="validate, filter, and plan without mutating anything in MLflow",
    )
    import_command.add_argument(
        "--output",
        choices=OUTPUT_CHOICES,
        default=DEFAULT_OUTPUT,
        help="result format (default: human)",
    )
    import_command.add_argument(
        "--log-level",
        choices=LOG_LEVEL_CHOICES,
        default=DEFAULT_LOG_LEVEL,
        help="log verbosity (default: INFO)",
    )
    return parser


def parse_config(argv: Sequence[str]) -> ImportConfig:
    """Parse ``argv`` into a validated, frozen :class:`ImportConfig`.

    Argparse syntax errors and ``--help`` propagate :class:`SystemExit`
    (statuses ``2`` and ``0``); :func:`opencode_mlflow.cli.main` converts
    those into its return value. Semantic failures raise
    :class:`~opencode_mlflow.errors.ConfigurationError` with a safe message
    that never echoes the offending value.
    """
    args = build_parser().parse_args(argv)

    experiment_name = args.experiment_name
    if not experiment_name.strip():
        raise ConfigurationError("invalid_experiment_name", "--experiment-name must be a non-empty name")

    model: str | None = None
    if args.model is not None:
        model = normalize_model_identity(args.model)
        if not model:
            raise ConfigurationError("invalid_model", "--model must be a non-empty provider/model identity")

    return ImportConfig(
        experiment_name=experiment_name,
        db_path=normalize_path(args.db if args.db is not None else DEFAULT_DB_PATH),
        tracking_uri=args.tracking_uri,
        from_time=_parse_time_bound("--from-time", args.from_time),
        to_time=_parse_time_bound("--to-time", args.to_time),
        directory=normalize_path(args.directory) if args.directory is not None else None,
        model=model,
        min_user_turns=_parse_min_user_turns(args.min_user_turns),
        redact=args.redact,
        dry_run=args.dry_run,
        output=args.output,
        log_level=args.log_level,
    )


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp with an explicit timezone offset or ``Z``.

    The value is converted to UTC while preserving the exact instant, so the
    selection layer can apply inclusive boundaries exactly. Values without
    an explicit offset raise :class:`ConfigurationError` with code
    ``invalid_timestamp`` and a safe message that does not echo the value.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ConfigurationError("invalid_timestamp", _TIMESTAMP_MESSAGE) from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConfigurationError("invalid_timestamp", _TIMESTAMP_MESSAGE)
    return parsed.astimezone(UTC)


def normalize_path(value: str) -> Path:
    """Lexically normalize a filesystem path.

    ``~`` is expanded, relative input is made absolute against the current
    working directory, and ``.``/``..`` components plus duplicate separators
    are resolved without touching the filesystem: neither existence nor
    symlinks are resolved (``Path.resolve`` is deliberately avoided).
    """
    expanded = os.path.expanduser(value)
    return Path(os.path.abspath(expanded))


def normalize_model_identity(value: str) -> str:
    """Normalize a provider/model identity for exact matching.

    Only surrounding whitespace is removed. Case is preserved and missing
    provider or model components are never invented, so incomplete forms
    such as ``provider/`` or ``/model`` keep their exact shape and match
    only requests in the same available form.
    """
    return value.strip()


def _parse_time_bound(flag: str, value: str | None) -> datetime | None:
    """Parse one inclusive time bound, naming the flag in any safe error."""
    if value is None:
        return None
    try:
        return parse_timestamp(value)
    except ConfigurationError as error:
        raise ConfigurationError(error.code, f"{flag} {error.safe_message}") from error


def _parse_min_user_turns(value: str | None) -> int | None:
    """Validate the strict minimum-turn threshold.

    ``None`` (the option omitted) imposes no restriction, so zero-turn
    sessions remain selectable for stale-trace retirement; an explicit ``0``
    means "strictly more than 0 turns" and excludes them. The raw string is
    converted here (not via argparse's ``type=int``) so invalid values
    surface as safe errors instead of argparse usage text.
    """
    if value is None:
        return None
    try:
        threshold = int(value.strip())
    except ValueError:
        raise ConfigurationError("invalid_min_user_turns", _THRESHOLD_MESSAGE) from None
    if threshold < 0:
        raise ConfigurationError("invalid_min_user_turns", _THRESHOLD_MESSAGE)
    return threshold
