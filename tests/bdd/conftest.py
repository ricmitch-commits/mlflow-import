"""Shared BDD harness configuration and scenario fixtures.

Gherkin feature files live in ``tests/bdd/features``; step modules in
``tests/bdd/steps`` bind them with :func:`pytest_bdd.scenarios`, which is
how pytest-bdd discovery is registered for this suite.

The fixtures here are the consolidated acceptance harness every boundary
scenario drives:

- ``scenario_context`` — fresh per-scenario state shared between
  Given/When/Then steps;
- ``bdd_source_builder`` — the synthetic source-database factory
  (:class:`~tests.support.import_fakes.SyntheticSourceBuilder`) scenarios
  accumulate sanitized synthetic sessions in;
- ``bdd_gateway`` — one :class:`~tests.support.fake_gateway.FakeMlflowGateway`
  per scenario, so rerun scenarios observe exactly the gateway the runs
  used (its event and mutation histories are the observable MLflow side
  of every assertion);
- ``bdd_clock`` — the deterministic clock injected into every run;
- ``cli_import_runner`` — the ``main()``-boundary runner: it invokes the
  public CLI entry point end to end with fake dependencies, captures the
  parsed report JSON, the exit status, the gateway event marker, the
  mutation count, and the source-file state (database bytes plus WAL/SHM
  companion metadata) before and after the run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture

from opencode_mlflow.cli import main
from tests.support.fake_clock import FakeClock
from tests.support.fake_gateway import FakeMlflowGateway
from tests.support.import_fakes import (
    BDD_EXPERIMENT_NAME,
    FakeImportDependencies,
    SyntheticSourceBuilder,
    build_import_fakes,
)

FEATURES_DIR = Path(__file__).resolve().parent / "features"
"""Directory holding the Gherkin feature files bound by the step modules."""


@pytest.fixture
def scenario_context() -> dict[str, object]:
    """Fresh per-scenario state shared between Given/When/Then steps."""
    return {}


@pytest.fixture
def bdd_gateway() -> FakeMlflowGateway:
    """One in-memory MLflow gateway per scenario (event/mutation history)."""
    return FakeMlflowGateway()


@pytest.fixture
def bdd_clock() -> FakeClock:
    """The deterministic clock every boundary run is generated under."""
    return FakeClock(start_ns=1_772_000_000_000_000_000)


@pytest.fixture
def bdd_source_builder(tmp_path: Path) -> SyntheticSourceBuilder:
    """The synthetic source-database factory for this scenario."""
    return SyntheticSourceBuilder(tmp_path)


@dataclass(frozen=True)
class CliImportRun:
    """One ``main()``-boundary invocation's observable outcome."""

    source_path: Path
    exit_status: int
    report: dict[str, object]
    output: str
    result_text: str
    dependencies: FakeImportDependencies
    event_marker: int
    mutation_count_before: int
    source_state_before: dict[str, tuple[str | int, ...] | None]
    source_state_after: dict[str, tuple[str | int, ...] | None]


def _source_file_state(path: Path) -> dict[str, tuple[str | int, ...] | None]:
    """The observable file state of a source database and its companions.

    The main database and the ``-wal`` companion are captured as
    ``(sha256, size, inode)`` — the importer must never change their
    bytes. The ``-shm`` wal-index legitimately changes bytes through
    SQLite's shared read-mark protocol on any read, so only its
    ``(size, inode)`` — its metadata — is captured. A missing companion
    is captured as ``None``.
    """
    state: dict[str, tuple[str | int, ...] | None] = {}
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        key = suffix or "db"
        if not candidate.exists():
            state[key] = None
            continue
        info = candidate.stat()
        if suffix == "-shm":
            state[key] = (info.st_size, info.st_ino)
        else:
            state[key] = (hashlib.sha256(candidate.read_bytes()).hexdigest(), info.st_size, info.st_ino)
    return state


CliImportRunner = Callable[..., CliImportRun]
"""The boundary runner callable returned by the ``cli_import_runner`` fixture."""


@pytest.fixture
def cli_import_runner(
    scenario_context: dict[str, object],
    bdd_gateway: FakeMlflowGateway,
    bdd_clock: FakeClock,
    bdd_source_builder: SyntheticSourceBuilder,
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> CliImportRunner:
    """Run one import through the public ``main()`` boundary with fakes.

    The returned callable executes one in-process CLI invocation over
    the scenario's synthetic source (materializing the builder's rows
    when they changed), the scenario's single fake gateway, and the
    deterministic clock. It captures the parsed report JSON, the exit
    status, the rendered output, the gateway event marker and mutation
    count at run start, and the source-file state before and after the
    run — everything the scenario's Then steps assert on. The latest
    run is also exposed through the legacy ``cli_report`` /
    ``cli_dependencies`` / ``cli_output`` context keys.
    """

    def _run(*, dry_run: bool = False, extra_args: Sequence[str] = ()) -> CliImportRun:
        if bdd_source_builder.session_count and (bdd_source_builder.dirty or "cli_source_path" not in scenario_context):
            scenario_context["cli_source_path"] = bdd_source_builder.materialize()
        source = scenario_context.get("cli_source_path")
        assert isinstance(source, Path), "the scenario must provide a synthetic source database"
        bdd_gateway.compatible = not bool(scenario_context.get("cli_incompatible_gateway", False))
        state_before = _source_file_state(source)
        event_marker = len(bdd_gateway.events)
        mutations_before = len(bdd_gateway.mutations)
        run_root = Path(tempfile.mkdtemp(prefix="opencode-mlflow-bdd-", dir=tmp_path))
        run_root.chmod(0o700)
        try:
            dependencies = build_import_fakes(
                restrictive_tmp_path=run_root, gateway=bdd_gateway, clock=bdd_clock.time_ns
            )
            argv = [
                "import",
                "--experiment-name",
                BDD_EXPERIMENT_NAME,
                "--db",
                str(source),
                "--output",
                "json",
                *extra_args,
            ]
            if dry_run:
                argv.append("--dry-run")
            exit_status = main(argv, dependencies=dependencies)
        finally:
            shutil.rmtree(run_root, ignore_errors=True)
        captured = capsys.readouterr()
        run = CliImportRun(
            source_path=source,
            exit_status=exit_status,
            report=json.loads(captured.out),
            output=captured.out,
            result_text=captured.out + captured.err,
            dependencies=dependencies,
            event_marker=event_marker,
            mutation_count_before=mutations_before,
            source_state_before=state_before,
            source_state_after=_source_file_state(source),
        )
        scenario_context["exit_status"] = run.exit_status
        scenario_context["result"] = run.result_text
        scenario_context["cli_report"] = run.report
        scenario_context["cli_dependencies"] = run.dependencies
        scenario_context["cli_output"] = run.output
        scenario_context.setdefault("cli_runs", []).append(run)
        return run

    return _run
