# OpenCode Integration Profile

Install the project and development dependencies in the repository virtual
environment:

```bash
uv pip install -e '.[dev]'
```

Run the disposable MLflow 3.17.2 profile:

```bash
MLFLOW_DISABLE_AGENT_HINT=1 \
  python -m pytest tests/contract/test_mlflow_profile.py \
  -m '(integration or profile)' -q
```

The fixture launches `python -m mlflow server` with a fresh SQLite backend,
an absolute same-host `file://` artifact root, loopback host, and an allocated
port. It health-checks `/health`, scrubs ambient OTEL exporter variables,
restores the process tracking URI, terminates the server process group, and
removes the temporary run directory.

Set `MLFLOW_TRACKING_URI` to use an explicitly managed HTTP(S) target. The
target must already be MLflow `3.17.2`; a requested but unreachable target is
a test failure, not a skipped profile. Destructive tests use unique disposable
experiments and exact trace IDs.

The compatibility evidence is recorded in `docs/compatibility.md`.
