"""Unit tests for the local non-blocking advisory import lock.

Pins the lock contract: the lock name derives from a hash of the
NORMALIZED tracking endpoint (URI credentials stripped, never included
in the name or the file) plus the exact experiment name; acquisition is
non-blocking (a held lock fails immediately with the status-``2`` safe
error ``import_lock_held``); the lock is released after success AND
after exceptions; a held lock is never broken automatically; the lock
root honors ``XDG_RUNTIME_DIR`` and otherwise uses a restrictive
(``0o700``) subdirectory of ``tempfile.gettempdir()``; and dry-run
acquires nothing (the caller decides by passing ``dry_run=True``).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from opencode_mlflow.errors import ConfigurationError
from opencode_mlflow.locking import (
    ImportLock,
    acquire_import_lock,
    lock_path,
    normalize_tracking_endpoint,
    resolve_lock_root,
)

pytestmark = pytest.mark.unit


def _with_credentials(endpoint: str) -> str:
    scheme, rest = endpoint.split("://", 1)
    authority, _, tail = rest.partition("/")
    host = authority.split("@", 1)[-1]
    return f"{scheme}://user:secret-token@{host}/{tail}"


def test_lock_name_ignores_uri_credentials(tmp_path: Path) -> None:
    plain = "http://tracking.example.com:8080/mlflow"
    with_credentials = _with_credentials(plain)

    assert normalize_tracking_endpoint(with_credentials) == normalize_tracking_endpoint(plain)
    assert lock_path(with_credentials, "team/opencode", tmp_path) == lock_path(plain, "team/opencode", tmp_path)

    with acquire_import_lock(plain, "team/opencode", tmp_path) as lock:
        content = lock.path.read_text(encoding="utf-8")

    assert "user" not in content
    assert "secret-token" not in content


def test_malformed_endpoints_never_leak_credentials_into_the_lock_file(tmp_path: Path) -> None:
    malformed_port = "http://user:secret-token@host:notaport"
    malformed_ipv6 = "http://user:secret-token@[::1"

    for endpoint in (malformed_port, malformed_ipv6):
        normalized = normalize_tracking_endpoint(endpoint)
        assert "user" not in normalized
        assert "secret-token" not in normalized

        with acquire_import_lock(endpoint, "team/opencode", tmp_path) as lock:
            content = lock.path.read_text(encoding="utf-8")
        assert "user" not in content
        assert "secret-token" not in content


def test_lock_name_includes_the_exact_experiment_name(tmp_path: Path) -> None:
    endpoint = "http://tracking.example.com:8080/mlflow"

    assert lock_path(endpoint, "experiment-a", tmp_path) != lock_path(endpoint, "experiment-b", tmp_path)
    assert lock_path(endpoint, "experiment-a", tmp_path) != lock_path(endpoint, "experiment-a ", tmp_path)


def test_lock_name_normalizes_equivalent_endpoints(tmp_path: Path) -> None:
    assert lock_path("http://tracking.example.com:80/mlflow", "e", tmp_path) == lock_path(
        "http://tracking.example.com/mlflow", "e", tmp_path
    )
    assert lock_path("http://TRACKING.example.com/mlflow", "e", tmp_path) == lock_path(
        "http://tracking.example.com/mlflow", "e", tmp_path
    )
    assert lock_path("http://tracking.example.com/mlflow?token=hunter2", "e", tmp_path) == lock_path(
        "http://tracking.example.com/mlflow", "e", tmp_path
    )


def test_lock_file_is_restricted_to_the_owner(tmp_path: Path) -> None:
    with acquire_import_lock("http://tracking.example.com", "team/opencode", tmp_path) as lock:
        mode = stat_mode(lock.path)

    assert mode == 0o600
    assert not lock.path.exists()


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_held_lock_fails_immediately_without_breaking(tmp_path: Path) -> None:
    with acquire_import_lock("http://tracking.example.com", "team/opencode", tmp_path) as first:
        with pytest.raises(ConfigurationError) as excinfo:
            with acquire_import_lock("http://tracking.example.com", "team/opencode", tmp_path):
                raise AssertionError("a held lock must not be acquired")

        error = excinfo.value
        assert error.code == "import_lock_held"
        assert error.exit_status == 2

        # The held lock is never removed or broken by the failed acquisition.
        assert first.path.exists()
        assert first.acquired is True

    assert not first.path.exists()


def test_stale_lock_from_a_dead_process_is_never_broken_automatically(tmp_path: Path) -> None:
    lock = lock_path("http://tracking.example.com", "team/opencode", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    lock.write_text('{"experiment_name": "team/opencode"}', encoding="utf-8")

    with pytest.raises(ConfigurationError) as excinfo:
        with acquire_import_lock("http://tracking.example.com", "team/opencode", tmp_path):
            raise AssertionError("a stale lock must require operator confirmation")

    assert excinfo.value.code == "import_lock_held"
    # The stale lock survives: recovery requires explicit operator removal.
    assert lock.exists()


def test_lock_is_released_after_exceptions(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="synthetic failure inside the lock"):
        with acquire_import_lock("http://tracking.example.com", "team/opencode", tmp_path) as lock:
            assert lock.path.exists()
            raise RuntimeError("synthetic failure inside the lock")

    assert not lock.path.exists()

    # The next acquisition succeeds on the released path.
    with acquire_import_lock("http://tracking.example.com", "team/opencode", tmp_path):
        pass


def test_concurrent_locks_for_different_experiments_coexist(tmp_path: Path) -> None:
    with (
        acquire_import_lock("http://tracking.example.com", "experiment-a", tmp_path),
        acquire_import_lock("http://tracking.example.com", "experiment-b", tmp_path),
    ):
        assert len(list(tmp_path.iterdir())) == 2


def test_dry_run_acquires_nothing(tmp_path: Path) -> None:
    with acquire_import_lock("http://tracking.example.com", "team/opencode", tmp_path, dry_run=True) as lock:
        assert isinstance(lock, ImportLock)
        assert lock.acquired is False
        assert list(tmp_path.iterdir()) == []

    assert list(tmp_path.iterdir()) == []


def test_lock_root_honors_xdg_runtime_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    assert resolve_lock_root() == tmp_path


def test_lock_root_defaults_to_restrictive_tempdir_subdirectory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    root = resolve_lock_root()

    assert root == tmp_path / "opencode-mlflow-import-locks"
    with acquire_import_lock("http://tracking.example.com", "team/opencode", root):
        assert stat_mode(root) == 0o700
        children = list(root.iterdir())
        assert len(children) == 1
        assert stat_mode(children[0]) == 0o600


def test_lock_root_directory_is_created_restrictively(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "lock-root"
    os.chmod(tmp_path, 0o700)

    with acquire_import_lock("http://tracking.example.com", "team/opencode", root):
        assert stat_mode(root) == 0o700
