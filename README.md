# mlflow-import

One-shot CLI that imports coding-agent sessions into MLflow as traces.

`mlflow-import` reads supported session records from OpenCode, Claude Code,
Codex, PI, and Gemini CLI, reconstructs their logical turns, and writes one
MLflow trace per turn — with child spans for messages, tool calls, and
content, a streamed canonical JSON artifact per turn, and deterministic
re-run behavior: an unchanged session imports once, a changed turn replaces
only that turn's trace, and traces the source no longer carries are retired.

Each harness is handled by a source adapter. Adapters discover and normalize
the harness's supported local session format while preserving source identity,
ordering, timing, metadata, and raw records where MLflow has no native field.
The trace/session `source` value (`mlflow.source.name` in MLflow metadata) is
the canonical harness name: `opencode`, `claude`, `codex`, `pi`, or `gemini`.
The `--source PATH` option selects local input; its path is not the exported
`source` value.

Source data is treated as strictly read-only evidence. Where the source is
SQLite, it is opened `mode=ro` under `PRAGMA query_only` in a single deferred
read transaction, copied into a restrictive temporary normalized snapshot, and
never written, checkpointed, vacuumed, migrated, or deleted. Other harness
files are read without modification and normalized into a restrictive
temporary snapshot; there is no atomic filesystem snapshot across multiple
files, so changes made during a scan may be observed at different points.

## Status

Version 0.1.0. The fixture-qualified target pair is the
`mlflow==3.17.2` client against a server launched from that same
installation, with an explicit SQLite backend and a same-host `file://`
experiment artifact root (see [docs/compatibility.md](docs/compatibility.md)).
All five source adapters have synthetic fixture coverage in this
checkout; see the compatibility document for the exact schema signatures
and explicit non-support. Known limitations are recorded in
[CHANGELOG.md](CHANGELOG.md).

### Wire-contract compatibility

New traces and results use the neutral `mlflow_import.*` identifiers and
`mlflow-import.result.v1`. Legacy `opencode.*` ownership metadata and
`opencode-mlflow.result.v1` documents are decoded through
`mlflow_import.legacy`; unknown versions are never guessed into the new
contract. See
[docs/operations.md](docs/operations.md),
[docs/compatibility.md](docs/compatibility.md), and
[docs/result-schema.md](docs/result-schema.md).

## Requirements

- Python 3.11+ (CI exercises 3.11, 3.12, and 3.13)
- A supported session source from OpenCode, Claude Code, Codex, PI, or Gemini
  CLI. Default sources are discovered automatically and can be restricted
  with `--harness`.
- A reachable MLflow tracking server satisfying the pinned compatibility
  profile

## Installation

Install from a checkout of this repository (the package is not published to
an index):

```bash
cd /path/to/mlflow-import
uv venv
uv pip install .
```

This installs the sole importer console script, `mlflow-import`, and its single
runtime dependency, `mlflow==3.17.2`. For development (tests, Ruff, mypy, pytest-bdd):

```bash
uv pip install -e ".[dev]"
```

## Quick start

Preview everything discovered in the available harness default sources — a
dry run mutates nothing and acquires no lock:

```bash
mlflow-import --experiment-name coding-agent-sessions --dry-run
```

Import all discovered default harness sources:

```bash
mlflow-import --experiment-name coding-agent-sessions
```

Import only one harness from its default source:

```bash
mlflow-import --experiment-name coding-agent-sessions --harness claude
```

Import one session by its source session ID, including all of its turns:

```bash
mlflow-import --experiment-name coding-agent-sessions --session SESSION-ID
```

Import one calendar month of sessions for one project directory, masking
secrets and user identity before uploading session/trace data:

```bash
mlflow-import --experiment-name coding-agent-sessions \
  --from-time 2026-10-01T00:00:00Z \
  --to-time 2026-10-31T23:59:59Z \
  --directory /Users/you/Code/myproject \
  --redact
```

Point at a non-default source and tracking server, and consume the
machine-readable result document:

```bash
mlflow-import --experiment-name coding-agent-sessions \
  --source /path/to/session-source \
  --tracking-uri http://mlflow.internal:5000 \
  --output json
```

Import a session even when a copy of that session already exists in MLflow:

```bash
mlflow-import --experiment-name coding-agent-sessions \
  --source /path/to/session-source \
  --on-conflict duplicate
```

## Command line

```
mlflow-import --experiment-name NAME [options]
```

| Option | Meaning |
| --- | --- |
| `--experiment-name NAME` | **Required.** MLflow experiment name; resolved by name and created if absent. |
| `--harness NAME` | Select a supported harness (`opencode`, `claude`, `codex`, `pi`, or `gemini`). Without `--source`, restricts discovery to that harness's default source location(s). |
| `--source PATH` | Explicit harness session source path. The harness is detected from the path unless `--harness` is also supplied; a supplied harness must match. |
| `--session SESSION-ID` | Restrict selection to the source session with this ID. The whole session is selected, including all of its logical turns. |
| `--on-conflict duplicate\|overwrite\|ignore` | How to handle a session that already exists in MLflow, including sessions not previously imported by this tool. Default: `overwrite`. `duplicate` imports another copy, `overwrite` replaces the conflicting session, and `ignore` skips the incoming session. |
| `--tracking-uri URI` | MLflow tracking URI; otherwise standard MLflow client/environment configuration. |
| `--from-time ISO` | Inclusive lower bound on session **creation** time. Requires an explicit timezone offset or `Z`. |
| `--to-time ISO` | Inclusive upper bound on session creation time. Requires an explicit timezone offset or `Z`. |
| `--directory PATH` | Restrict to sessions in this directory and all component-descendants. |
| `--model PROVIDER/MODEL` | Exact normalized model identity match. |
| `--min-user-turns N` | Select sessions with strictly more than `N` logical user turns. Omitted = no restriction. |
| `--redact` | Mask secrets and user identity in session/trace values, metadata, errors, and artifacts, including native MLflow user fields. Without it, permitted source content and available user metadata are uploaded verbatim; use it for shared or production tracking servers. |
| `--dry-run` | Validate, filter, and plan without mutating anything in MLflow. |
| `--output human\|json` | Result format (default: `human`). |
| `--log-level DEBUG\|INFO\|WARNING\|ERROR` | Log verbosity (default: `INFO`). `DEBUG` prints live phase/count progress to stderr; the final result remains on stdout. |

With `--redact`, present user-identity values (IDs, usernames, names, and email
addresses) become `[REDACTED]` in session/trace metadata and retained raw artifact
copies, including `mlflow.trace.user` and any populated `mlflow.user` field.
Missing user values stay absent. The conversation role `user` and the identifiers
needed for session/turn reconciliation remain intact. The harness-valued `source`
is the same in either privacy mode.

Selection semantics worth knowing before your first run:

- With neither `--harness` nor `--source`, all available adapter-owned default
  harness sources are discovered and eligible.
- `--harness` selects which adapter defaults to scan; `--source` selects an
  explicit local path. They can be used together to validate that the path
  belongs to the requested harness.
- `--session` restricts selection to the matching source session ID; it selects
  the session as a whole, not a subset of its turns.
- An explicit source must match exactly one supported harness format unless
  `--harness` identifies the adapter. A harness/source mismatch fails before
  MLflow writes.
- Duplicate-session handling is controlled by `--on-conflict`: `overwrite`
  (the default) replaces the existing MLflow session, including one not
  previously imported by this tool; `duplicate` creates another copy without
  retiring the existing one; and `ignore` leaves the existing session
  unchanged and skips the incoming session.
- Timestamps require an explicit timezone offset or `Z`. Bounds are
  inclusive and apply only to the session's creation time.
- Every supplied filter combines with logical AND; an omitted filter
  imposes no restriction.
- A run with **no** optional selection filter carries a
  `broad_selection` warning in its report recommending a dry run first.

Exit statuses: `0` complete or no-op, `1` any item failed (other items
continued), `2` invalid invocation or pre-write failure. `--help` exits `0`;
argument syntax errors exit `2`.


## Scope

This is a one-shot import tool. It does not watch a source, call a harness
server/API, or keep a background process running. It does not use an ORM, does
not write to MLflow's backend store directly, and uses only public MLflow
client APIs. See [CHANGELOG.md](CHANGELOG.md) for the full excluded-scope
list.
