"""Disposable live MLflow tracking server for integration tests.

Generalized from the session fixture Task 0 pinned in
``tests/contract/test_mlflow_profile.py`` so every live-server integration
module shares one recipe (and, through the session-scoped ``live_server``
fixture in ``tests/conftest.py``, at most one disposable server per test
session):

- a requested server via ``MLFLOW_TRACKING_URI`` (HTTP(S) only,
  health-checked with a bounded budget) is used as-is;
- otherwise a disposable ``mlflow server`` is launched from the installed
  distribution — fresh run directory under the pre-approved temp root,
  SQLite backend, local artifact root, programmatically chosen free port,
  retried on bind races within a 60-second startup budget;
- ambient ``OTEL_EXPORTER_OTLP_*`` variables are scrubbed before the
  tracing provider initializes (they silently reroute span export to an
  external collector and degrade span creation to NoOp);
- the global tracking URI is pinned to the server for the session (span
  export follows the *global* URI, not any client's) and restored on exit;
- the server is ALWAYS reaped — SIGTERM to the process group first, then
  SIGKILL — and the run directory removed, including on failure.

A requested server that cannot be reached or launched fails the session
loudly (``pytest.fail``); nothing here skips silently. See
``tests/integration/README.md`` for the equivalent manual command
sequence and ``docs/compatibility.md`` for the pinned profile.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from tests.integration.mlflow_profile import OTEL_HIJACK_ENV_VARS

__all__ = [
    "EXTERNAL_HEALTH_TIMEOUT_SECONDS",
    "LAUNCH_RETRIES",
    "SERVER_STARTUP_TIMEOUT_SECONDS",
    "LiveServer",
    "health_ok",
    "live_mlflow_server",
    "terminate_server",
    "unique_experiment_name",
    "wait_for_health",
]

LAUNCH_RETRIES = 3
"""Bind-race retries before the session fails with the server log tail."""

SERVER_STARTUP_TIMEOUT_SECONDS = 60.0
"""Total health budget for one launch attempt."""

EXTERNAL_HEALTH_TIMEOUT_SECONDS = 15.0
"""Health budget for a server requested through ``MLFLOW_TRACKING_URI``."""


def unique_experiment_name(prefix: str) -> str:
    """A collision-proof experiment name: millisecond stamp plus random suffix.

    Integration tests must be independent and rerunnable against a shared
    server, so every test creates its own experiment under a unique name
    instead of reusing or mutating another test's state.
    """
    return f"{prefix}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def health_ok(uri: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{uri}/health", timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_for_health(uri: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if health_ok(uri):
            return True
        time.sleep(0.25)
    return False


def terminate_server(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the server process group; always escalate to SIGKILL."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait(timeout=10)


def _launch_disposable_server(run_root: Path) -> tuple[subprocess.Popen[bytes], str]:
    """Launch a disposable MLflow server from the installed distribution.

    Retries with a fresh port when the server exits before becoming healthy
    (for example on a bind race). Always fails the session with the server log
    tail when no attempt succeeds.
    """
    database = run_root / "mlflow.db"
    artifacts = run_root / "artifacts"
    last_log = ""
    for attempt in range(LAUNCH_RETRIES):
        port = _free_port()
        log_path = run_root / f"server-attempt-{attempt}.log"
        command = [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            f"sqlite:///{database}",
            "--artifacts-destination",
            f"file://{artifacts}",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
        with log_path.open("wb") as log_file:
            proc = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
        uri = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break  # server exited before becoming healthy; retry on a new port
            if health_ok(uri):
                return proc, uri
            time.sleep(0.25)
        terminate_server(proc)
        last_log = log_path.read_text(errors="replace")[-2000:]
    pytest.fail(
        "Could not start a disposable MLflow server from the installed distribution "
        f"after {LAUNCH_RETRIES} attempts; last server log:\n{last_log}",
        pytrace=False,
    )


@dataclass(frozen=True)
class LiveServer:
    """One reachable live MLflow tracking server and its client."""

    uri: str
    client: MlflowClient
    process: subprocess.Popen[bytes] | None
    run_root: Path | None


@contextmanager
def live_mlflow_server() -> Iterator[LiveServer]:
    """Yield a live MLflow server: requested via ``MLFLOW_TRACKING_URI`` or self-launched.

    Honors an externally requested server via ``MLFLOW_TRACKING_URI``
    (HTTP(S) only); otherwise launches a disposable server from the installed
    distribution that is always reaped, even on failure. Ambient OTEL
    exporter configuration is scrubbed before the tracing provider
    initializes and the global tracking URI is pinned to the server for the
    session (module-level span export follows the *global* tracking URI,
    not any individual client's URI), restoring the saved value on exit.
    """
    saved_otel = {name: os.environ.pop(name) for name in OTEL_HIJACK_ENV_VARS if name in os.environ}
    server: subprocess.Popen[bytes] | None = None
    run_root: Path | None = None
    saved_uri = mlflow.get_tracking_uri()
    try:
        requested = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
        if requested:
            if not requested.startswith(("http://", "https://")):
                pytest.fail(
                    f"MLFLOW_TRACKING_URI={requested!r} is not an HTTP tracking server URI; "
                    "the pinned profile requires a live MLflow server "
                    "(see tests/integration/README.md)",
                    pytrace=False,
                )
            if not wait_for_health(requested, EXTERNAL_HEALTH_TIMEOUT_SECONDS):
                pytest.fail(
                    f"MLFLOW_TRACKING_URI={requested!r} did not respond on /health; "
                    "a requested server that cannot be reached is a profile failure",
                    pytrace=False,
                )
            uri = requested
        else:
            base = Path(tempfile.gettempdir()).resolve() / "opencode"
            base.mkdir(parents=True, exist_ok=True)
            run_root = base / f"opencode-mlflow-integration-{uuid.uuid4().hex[:12]}"
            run_root.mkdir()
            server, uri = _launch_disposable_server(run_root)
        mlflow.set_tracking_uri(uri)
        yield LiveServer(uri=uri, client=MlflowClient(tracking_uri=uri), process=server, run_root=run_root)
    finally:
        mlflow.set_tracking_uri(saved_uri)
        if server is not None:
            terminate_server(server)
        if run_root is not None:
            shutil.rmtree(run_root, ignore_errors=True)
        os.environ.update(saved_otel)
