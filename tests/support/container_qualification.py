"""Explicitly opted-in, exclusively owned local PostgreSQL/MLflow fixture.

Never adopts a tracking URI, existing container, network, or database. All
engine operations name resources created by this context. No global pruning.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import requests

MLFLOW_IMAGE = "ghcr.io/mlflow/mlflow:v3.17.2"
POSTGRES_IMAGE = "docker.io/library/postgres:16-alpine"
MIB = 1024**2


def engine(*args: str, timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=check)


@dataclass
class ContainerProfile:
    uri: str
    server: str
    database: str
    directory: Path

    def resources(self) -> dict:
        """Kernel high-water marks survive spikes between samples (Linux cgroup v2)."""
        result = {}
        for role, name, limit in (("server", self.server, 8192 * MIB), ("db", self.database, 1024 * MIB)):
            state = json.loads(engine("inspect", name).stdout)[0]
            assert state["HostConfig"]["Memory"] == limit
            # Read only these owned containers' cgroups. The shell reads avoid
            # requiring any optional Python/ps utility in postgres:alpine.
            script = (
                "for f in memory.current memory.peak memory.events; do "
                'while IFS= read -r line; do printf "%s %s\\n" "$f" "$line"; '
                'done < "/sys/fs/cgroup/$f"; done'
            )
            lines = engine("exec", name, "sh", "-c", script).stdout.splitlines()
            if role == "server":
                rss_script = (
                    "import pathlib; total=0\n"
                    "for p in pathlib.Path('/proc').glob('[0-9]*/status'):\n"
                    " try:\n"
                    "  for line in p.read_text().splitlines():\n"
                    "   if line.startswith('VmRSS:'): total += int(line.split()[1])*1024\n"
                    " except (FileNotFoundError, ProcessLookupError): pass\n"
                    "print('rss_bytes', total)"
                )
                lines.extend(engine("exec", name, "python", "-c", rss_script).stdout.splitlines())
            else:
                lines.extend(engine("exec", name, "ps", "-o", "pid,ppid,rss,comm").stdout.splitlines())
            metrics = {"rss_bytes": 0}
            for line in lines:
                columns = line.split()
                if columns[0] in ("memory.current", "memory.peak"):
                    metrics[columns[0].replace("memory.", "cgroup_") + "_bytes"] = int(columns[1])
                elif columns[0] == "memory.events":
                    metrics[columns[1]] = int(columns[2])
                elif columns[0] == "rss_bytes":
                    metrics["rss_bytes"] = int(columns[1])
                elif columns[0].isdigit() and len(columns) >= 3:
                    rss = columns[2].lower()
                    factor = {"k": 1024, "m": MIB, "g": 1024**3}.get(rss[-1], 1024)
                    metrics["rss_bytes"] += int(float(rss[:-1] if rss[-1].isalpha() else rss) * factor)
            metrics.update(
                running=state["State"]["Running"],
                oom_killed=state["State"].get("OOMKilled", False),
                restarts=state.get("RestartCount", 0),
                limit_bytes=limit,
            )
            result[role] = metrics
        return result


@contextmanager
def disposable_container_profile():
    """Requires local base images; never pulls implicitly or uses an external URI.

    If necessary, builds a uniquely tagged derivative with only the pinned
    PostgreSQL driver, using bounded pip retries against public PyPI. The
    derivative is removed in the same finally block as owned containers.
    """
    if os.environ.get("MLFLOW_IMPORT_CONTAINER_QUALIFICATION") != "1":
        raise RuntimeError("explicit container qualification opt-in required")
    prefix = "mlflow-qual-" + uuid.uuid4().hex[:12]
    network, db, server, image = (prefix + suffix for suffix in ("-net", "-db", "-server", ":driver"))
    containers, volumes = [], []
    network_created = derived = False
    with TemporaryDirectory(prefix=prefix + "-") as temporary:
        directory = Path(temporary)
        directory.chmod(0o700)
        try:
            engine("version", "--format", "{{json .Server}}")
            for base in (MLFLOW_IMAGE, POSTGRES_IMAGE):
                engine("image", "inspect", base)
            probe = prefix + "-driver-check"
            containers.append(probe)
            installed = (
                engine(
                    "run",
                    "--rm",
                    "--name",
                    probe,
                    "--network",
                    "none",
                    MLFLOW_IMAGE,
                    "python",
                    "-c",
                    "import psycopg2",
                    check=False,
                ).returncode
                == 0
            )
            selected_image = MLFLOW_IMAGE
            if not installed:
                (directory / "Dockerfile").write_text(
                    f"FROM {MLFLOW_IMAGE}\n"
                    "RUN python -m pip install --no-cache-dir --retries 1 --timeout 30 "
                    "--index-url https://pypi.org/simple psycopg2-binary==2.9.11\n"
                )
                derived = True
                engine("build", "--tag", image, str(directory), timeout=240)
                selected_image = image
            # Podman's internal bridge cannot route its published loopback
            # port through the VM. A uniquely owned regular bridge supports
            # the local client; telemetry remains disabled explicitly.
            engine("network", "create", network)
            network_created = True
            for suffix in ("-pgdata", "-artifacts"):
                volume = prefix + suffix
                engine("volume", "create", volume)
                volumes.append(volume)
            containers.append(db)
            engine(
                "run",
                "--detach",
                "--name",
                db,
                "--network",
                network,
                "--memory",
                "1g",
                "--memory-swap",
                "1g",
                "--restart",
                "no",
                "--env",
                "POSTGRES_USER=synthetic",
                "--env",
                "POSTGRES_PASSWORD=synthetic-local-only",
                "--env",
                "POSTGRES_DB=synthetic",
                "--volume",
                f"{volumes[0]}:/var/lib/postgresql/data",
                POSTGRES_IMAGE,
            )
            deadline = time.monotonic() + 90
            while engine("exec", db, "pg_isready", "-U", "synthetic", check=False).returncode:
                if time.monotonic() > deadline:
                    raise RuntimeError("owned PostgreSQL did not become ready")
                time.sleep(0.5)
            containers.append(server)
            engine(
                "run",
                "--detach",
                "--name",
                server,
                "--network",
                network,
                "--memory",
                "8g",
                "--memory-swap",
                "8g",
                "--restart",
                "no",
                "--publish",
                "127.0.0.1::5000",
                "--volume",
                f"{volumes[1]}:/artifacts",
                "--env",
                "MLFLOW_ENABLE_TELEMETRY=false",
                selected_image,
                "mlflow",
                "server",
                "--workers",
                "1",
                "--host",
                "0.0.0.0",
                "--port",
                "5000",
                "--backend-store-uri",
                f"postgresql+psycopg2://synthetic:synthetic-local-only@{db}:5432/synthetic",
                "--artifacts-destination",
                "/artifacts",
                "--serve-artifacts",
                "--allowed-hosts",
                "127.0.0.1,localhost,127.0.0.1:*,localhost:*",
            )
            binding = engine("port", server, "5000/tcp").stdout.strip()
            assert binding.startswith("127.0.0.1:") and "\n" not in binding
            uri = "http://" + binding
            deadline = time.monotonic() + 120
            with requests.Session() as session:
                session.trust_env = False
                last_status = None
                while True:
                    try:
                        response = session.get(uri + "/version", timeout=2)
                        last_status = response.status_code
                        if response.status_code == 200:
                            assert response.text.strip() == "3.17.2"
                            break
                    except (requests.ConnectionError, requests.Timeout):
                        pass
                    state = json.loads(engine("inspect", server).stdout)[0]["State"]
                    if time.monotonic() > deadline or not state["Running"]:
                        logs = engine("logs", "--tail", "40", server, check=False)
                        raise RuntimeError(
                            f"owned MLflow did not become ready (HTTP {last_status}): " + logs.stdout + logs.stderr
                        )
                    time.sleep(0.5)
            yield ContainerProfile(uri, server, db, directory)
        finally:
            failures = []
            for name in reversed(containers):
                result = engine("rm", "--force", "--volumes", name, check=False)
                if result.returncode and "no such container" not in result.stderr.lower():
                    failures.append("container " + name)
            for volume in reversed(volumes):
                if engine("volume", "rm", volume, check=False).returncode:
                    failures.append("volume " + volume)
            if network_created and engine("network", "rm", network, check=False).returncode:
                failures.append("network " + network)
            if derived and engine("image", "rm", image, check=False).returncode:
                failures.append("image " + image)
            if failures:
                raise RuntimeError("owned-resource cleanup failed: " + ", ".join(failures))
