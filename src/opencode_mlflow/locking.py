"""Local non-blocking advisory lock for normal-mode import runs.

Before experiment creation or owned-trace lookup, a normal (non-dry-run)
run acquires one advisory lock that serializes concurrent importer
invocations for the same tracking endpoint and exact experiment name on
one host. Concurrent invocations from different hosts are explicitly not
supported and must be serialized by the operator.

Lock identity and safety:

- the lock FILE NAME is the SHA-256 of a versioned header, the
  NORMALIZED tracking endpoint, and the exact experiment name;
  normalization strips URI userinfo credentials, query strings, and
  fragments, and canonicalizes scheme/host case and default ports, so
  the name (and the lock file) never contains or depends on
  credentials;
- acquisition is non-blocking: the file is created atomically with
  ``O_CREAT | O_EXCL`` and mode ``0o600``. A held lock fails immediately
  with the status-``2`` safe error ``import_lock_held``;
- the lock is released after success AND after exceptions (the context
  manager always removes the file it created in this process);
- a held or stale lock is NEVER broken automatically: this importer
  never removes a lock file it did not acquire itself.

**Stale-lock recovery procedure (operator-confirmed, manual):**

1. Confirm no ``opencode-mlflow`` import process is still running for
   the tracking endpoint and experiment (for example with ``ps`` or
   your process manager).
2. Confirm the lock is stale — the process that created it is gone.
3. Remove the lock file manually:
   ``rm <lock-root>/import-<endpoint+experiment-sha256>.lock``.

The lock root honors ``XDG_RUNTIME_DIR`` when set (per the XDG base
directory specification, a per-user directory with mode ``0o700``);
otherwise a restrictive ``0o700`` subdirectory of
``tempfile.gettempdir()`` is used (see :func:`resolve_lock_root`).
Lock acquisition is SKIPPED for dry-run: the caller decides — pass
``dry_run=True`` and nothing is created on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from opencode_mlflow.errors import ConfigurationError

__all__ = [
    "ImportLock",
    "LOCK_FILE_PREFIX",
    "LOCK_FILE_SUFFIX",
    "LOCK_NAME_VERSION",
    "acquire_import_lock",
    "lock_path",
    "normalize_tracking_endpoint",
    "resolve_lock_root",
]

#: Version header of the lock-name hash input (versioned separators make
#: the endpoint/experiment framing unambiguous).
LOCK_NAME_VERSION = "opencode-mlflow-import-lock-v1"

LOCK_FILE_PREFIX = "import-"
LOCK_FILE_SUFFIX = ".lock"

_DEFAULT_LOCK_DIRECTORY_NAME = "opencode-mlflow-import-locks"


@dataclass(frozen=True)
class ImportLock:
    """One acquired (or dry-run-skipped) advisory import lock.

    ``endpoint_digest`` is the SHA-256 of the normalized tracking
    endpoint alone; the lock FILE name additionally folds in the exact
    experiment name (see :func:`lock_path`). ``acquired`` is ``False``
    only for the dry-run seam, which acquires nothing.
    """

    path: Path
    experiment_name: str
    endpoint_digest: str
    acquired: bool = True


def resolve_lock_root() -> Path:
    """The advisory lock root: ``XDG_RUNTIME_DIR`` or a restrictive tempdir subdirectory."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg)
    return Path(tempfile.gettempdir()) / _DEFAULT_LOCK_DIRECTORY_NAME


def normalize_tracking_endpoint(endpoint: str) -> str:
    """Normalize a tracking endpoint, stripping every credential-bearing part.

    URI userinfo (``user:password@``), query strings, and fragments are
    dropped; scheme and host are lowercased; the scheme's default port
    (80 for http, 443 for https) is removed; a malformed port is dropped
    rather than propagated. A value that does not parse as a URI with an
    authority — a bare filesystem path or DSN — is used trimmed as-is.
    The result NEVER contains credentials, including on the fallback
    paths: userinfo is stripped before the port is ever read (a
    malformed port raises), and an unparseable value is sanitized before
    being returned, because the normalized endpoint is written into the
    lock file's diagnostic payload.
    """
    text = endpoint.strip()
    try:
        parsed = urlsplit(text)
    except ValueError:
        # An unparseable value (for example a malformed IPv6 authority):
        # strip any userinfo before returning, so credentials can never
        # reach the lock file payload through the fallback.
        return _strip_userinfo(text)
    if not parsed.scheme and not parsed.netloc:
        return text
    # Read the host from the parsed value (userinfo never reaches
    # ``hostname``) and only then consult ``.port``, which raises on a
    # malformed port; a bad port is dropped, not propagated.
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        port = None
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((scheme, netloc, parsed.path, "", ""))


_USERINFO_PATTERN = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)?(?P<userinfo>[^/?#@]+@)(?P<rest>.*)$")


def _strip_userinfo(text: str) -> str:
    """Remove a leading ``scheme://user:password@`` credential section."""
    match = _USERINFO_PATTERN.match(text)
    if match is None:
        return text
    scheme = match.group("scheme") or ""
    return f"{scheme}{match.group('rest')}"


def _endpoint_digest(endpoint: str) -> str:
    return hashlib.sha256(
        LOCK_NAME_VERSION.encode("utf-8") + b"\0" + normalize_tracking_endpoint(endpoint).encode("utf-8")
    ).hexdigest()


def lock_path(tracking_endpoint: str, experiment_name: str, lock_root: Path) -> Path:
    """The lock file path derived from the normalized endpoint and exact experiment name."""
    digest = hashlib.sha256(
        LOCK_NAME_VERSION.encode("utf-8")
        + b"\0"
        + normalize_tracking_endpoint(tracking_endpoint).encode("utf-8")
        + b"\0"
        + experiment_name.encode("utf-8")
    ).hexdigest()
    return lock_root / f"{LOCK_FILE_PREFIX}{digest}{LOCK_FILE_SUFFIX}"


@contextmanager
def acquire_import_lock(
    tracking_endpoint: str,
    experiment_name: str,
    lock_root: Path,
    *,
    dry_run: bool = False,
) -> Iterator[ImportLock]:
    """Acquire the non-blocking advisory import lock for one run.

    In normal mode the lock root is created restrictively (``0o700``)
    and the lock file is created atomically with ``O_CREAT | O_EXCL``
    and mode ``0o600``; a held lock fails immediately with the
    status-``2`` safe error ``import_lock_held`` and is never broken.
    The lock file carries only credential-free metadata (the normalized
    endpoint, the experiment name, and the acquiring process id) for
    operator diagnosis. On exit — success or exception — the file this
    process created is removed.

    ``dry_run=True`` is the skip seam: nothing is created on disk and
    the yielded :class:`ImportLock` reports ``acquired=False`` with the
    path a normal run would have used.
    """
    path = lock_path(tracking_endpoint, experiment_name, lock_root)
    endpoint_digest = _endpoint_digest(tracking_endpoint)
    if dry_run:
        yield ImportLock(path=path, experiment_name=experiment_name, endpoint_digest=endpoint_digest, acquired=False)
        return

    lock_root.mkdir(parents=True, exist_ok=True)
    os.chmod(lock_root, 0o700)
    payload = json.dumps(
        {
            "experiment_name": experiment_name,
            "tracking_endpoint": normalize_tracking_endpoint(tracking_endpoint),
            "pid": os.getpid(),
        },
        sort_keys=True,
    )
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ConfigurationError(
            "import_lock_held",
            "another importer already holds the lock for this tracking endpoint and experiment; "
            f"the lock file is {path}; if no importer is running, the stale lock file must be "
            "removed manually after operator confirmation",
        ) from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        yield ImportLock(path=path, experiment_name=experiment_name, endpoint_digest=endpoint_digest, acquired=True)
    finally:
        # Release only the file this process created; a held or stale lock
        # belonging to another process is never removed here.
        try:
            path.unlink()
        except FileNotFoundError:
            pass
