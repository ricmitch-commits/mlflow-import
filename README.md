# opencode-mlflow

One-shot CLI that imports OpenCode SQLite sessions into MLflow as traces.

`opencode-mlflow` reads an OpenCode session database (SQLite), reconstructs
its logical turns, and writes one MLflow trace per turn — with child spans
for messages, tool calls, and content, a streamed canonical JSON artifact
per turn, and deterministic re-run behavior: an unchanged session imports
once, a changed turn replaces only that turn's trace, and traces the source
no longer carries are retired.

The source database is treated as strictly read-only evidence. It is opened
`mode=ro` under `PRAGMA query_only` in a single deferred read transaction,
copied into a restrictive temporary normalized snapshot, and never written,
checkpointed, vacuumed, migrated, or deleted.

## Status

Version 0.1.0 — first release. Supported pair: the `mlflow==3.17.2` client
against a tracking server launched from that same installed distribution
(see [docs/compatibility.md](docs/compatibility.md)). Known limitations and
explicitly excluded scope are recorded in [CHANGELOG.md](CHANGELOG.md).

## Requirements

- Python 3.11+ (CI exercises 3.11, 3.12, and 3.13)
- An OpenCode SQLite session database (the default location is
  `~/.local/share/opencode/opencode.db`)
- A reachable MLflow tracking server satisfying the pinned compatibility
  profile

## Installation

Install from a checkout of this repository (the package is not published to
an index):

```bash
cd /path/to/mlflow-import
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
```

This installs the `opencode-mlflow` console script and its single runtime
dependency, `mlflow==3.17.2`. For development (tests, Ruff, mypy, pytest-bdd):

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

## Quick start

Preview everything the default database would do — a dry run mutates
nothing and acquires no lock:

```bash
opencode-mlflow import --experiment-name opencode-sessions --dry-run
```

Import the whole default database:

```bash
opencode-mlflow import --experiment-name opencode-sessions
```

Import one calendar month of sessions for one project directory, masking
secrets before anything is persisted:

```bash
opencode-mlflow import --experiment-name opencode-sessions \
  --from-time 2026-10-01T00:00:00Z \
  --to-time 2026-10-31T23:59:59Z \
  --directory /Users/you/Code/myproject \
  --redact
```

Point at a non-default database and tracking server, and consume the
machine-readable result document:

```bash
opencode-mlflow import --experiment-name opencode-sessions \
  --db /path/to/opencode.db \
  --tracking-uri http://mlflow.internal:5000 \
  --output json
```

## Command line

```
opencode-mlflow import --experiment-name NAME [options]
```

| Option | Meaning |
| --- | --- |
| `--experiment-name NAME` | **Required.** MLflow experiment name; resolved by name and created if absent. |
| `--db PATH` | OpenCode SQLite database path. Default: `~/.local/share/opencode/opencode.db`. |
| `--tracking-uri URI` | MLflow tracking URI; otherwise standard MLflow client/environment configuration. |
| `--from-time ISO` | Inclusive lower bound on session **creation** time (ISO-8601 with explicit offset or `Z`). |
| `--to-time ISO` | Inclusive upper bound on session creation time. |
| `--directory PATH` | Restrict to sessions in this directory and all component-descendants. |
| `--model PROVIDER/MODEL` | Exact normalized model identity match. |
| `--min-user-turns N` | Select sessions with strictly more than `N` logical user turns. Omitted = no restriction. |
| `--redact` | Mask secrets in trace values, metadata, errors, and artifacts. |
| `--dry-run` | Validate, filter, and plan without mutating anything in MLflow. |
| `--output human\|json` | Result format (default: `human`). |
| `--log-level DEBUG\|INFO\|WARNING\|ERROR` | Log verbosity (default: `INFO`). |

Selection semantics worth knowing before your first run:

- Time bounds require an explicit timezone offset or `Z`, are inclusive,
  and apply only to the session's creation time.
- Every supplied filter combines with logical AND; an omitted filter
  imposes no restriction.
- A run with **no** optional selection filter carries a
  `broad_selection` warning in its report recommending a dry run first.

Exit statuses: `0` complete or no-op, `1` any item failed (other items
continued), `2` invalid invocation or pre-write failure. `--help` exits `0`;
argument syntax errors exit `2`.

## Where to read next

- [docs/usage.md](docs/usage.md) — full usage guide: filters, output modes,
  result handling, examples.
- [docs/result-schema.md](docs/result-schema.md) — the
  `opencode-mlflow.result.v1` result document, action enum, and artifact
  path contract.
- [docs/operations.md](docs/operations.md) — operator runbook: locks,
  retries, replacement and rollback, recovery procedures.
- [docs/security.md](docs/security.md) — privacy model, redaction
  detectors, source-safety guarantees.
- [docs/architecture.md](docs/architecture.md) — pipeline phases and module
  boundaries.
- [docs/compatibility.md](docs/compatibility.md) — the pinned MLflow
  compatibility profile.
- [docs/testing.md](docs/testing.md) — test layers, markers, and the
  live-server integration recipe.

## Scope

This is a one-shot import tool. It does not watch the database, call
OpenCode's HTTP API, or keep a background process running. It does not use
an ORM, does not write to MLflow's backend store directly, and uses only
public MLflow client APIs. See [CHANGELOG.md](CHANGELOG.md) for the full
excluded-scope list.

## Source specifications

The binding requirements and design documents live in
[docs/superpowers/specs/](docs/superpowers/specs/):

- [Requirements](docs/superpowers/specs/2026-10-21-opencode-mlflow-import-requirements.md)
- [Design](docs/superpowers/specs/2026-10-21-opencode-mlflow-import-design.md)

Where those documents and this documentation differ, the shipped code and
this documentation describe the implemented behavior; the design spec's
resolution decisions are recorded in its "Decisions made" section.
