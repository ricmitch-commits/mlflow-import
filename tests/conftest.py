"""Shared fixtures for the opencode-mlflow test suite."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fixtures.six_turn_session import SixTurnSession, six_turn_session_fixture
from tests.integration.server import LiveServer, live_mlflow_server
from tests.support.fake_clock import FakeClock
from tests.support.fake_gateway import FakeMlflowClient


@pytest.fixture
def restrictive_tmp_path() -> Iterator[Path]:
    """A temporary directory restricted to owner-only access (mode 0o700)."""
    path = Path(tempfile.mkdtemp(prefix="opencode-mlflow-test-"))
    os.chmod(path, 0o700)
    try:
        yield path
    finally:
        # Restore owner access (in case a test tightened it further) and clean
        # up, tolerating a test that already removed the directory.
        if path.exists():
            os.chmod(path, 0o700)
            shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def live_server() -> Iterator[LiveServer]:
    """One live MLflow tracking server for the whole test session.

    Session-scoped so every live-server module (the pinned profile contract
    test and the Task 14 integration modules) shares one disposable server —
    or one externally requested ``MLFLOW_TRACKING_URI`` — instead of each
    launching its own. See ``tests/integration/server.py`` for the launch,
    health-check, OTEL-scrub, and always-reap lifecycle.
    """
    with live_mlflow_server() as server:
        yield server


@pytest.fixture
def fake_clock() -> FakeClock:
    """A deterministic clock for reproducible timestamp behavior."""
    return FakeClock()


@pytest.fixture
def fake_mlflow_client() -> FakeMlflowClient:
    """A fresh in-memory MLflow client fake (no network, no real server).

    Pairs with ``fake_mlflow_client.trace_api`` and
    ``fake_mlflow_client.artifact_repository_factory`` when driving the real
    gateway; ``mutations`` stays empty unless a mutating client API is called.
    """
    return FakeMlflowClient()


@pytest.fixture
def six_turn_session(restrictive_tmp_path: Path) -> Iterator[SixTurnSession]:
    """The sanitized synthetic six-turn session through the real pipeline.

    Fixture database → current adapter → normalized snapshot → replay
    adapter → six replay turns plus the mapping context. The snapshot
    stays open for the scenario's lifetime; ``tests/fixtures/six_turn_session.py``
    owns the synthetic data and the factory.
    """
    with six_turn_session_fixture(restrictive_tmp_path) as fixture:
        yield fixture
