# Usage guide

This document describes the shipped `opencode-mlflow import` command line,
its selection semantics, output modes, and exit statuses. Every flag here
matches the parser in `src/opencode_mlflow/config.py`; the result document
is specified separately in [result-schema.md](result-schema.md).

## Installation

Python 3.11+ (3.11, 3.12, and 3.13 are exercised in CI). Install from a
repository checkout — the package is not published to an index:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .          # runtime only
.venv/bin/python -m pip install -e ".[dev]" # development tooling
```

The single runtime dependency is `mlflow==3.17.2`; the tracking server must
satisfy the pinned compatibility profile
([compatibility.md](compatibility.md)).

## Syntax

```
opencode-mlflow import --experiment-name NAME [options]
```

`import` is the only subcommand and is required. Invoking the program
without a subcommand prints usage and exits with status `2`; `--help` exits
with status `0`; argument syntax errors (unknown options, invalid
`--output`/`--log-level` choices, missing values) exit `2`.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--experiment-name NAME` | *(required)* | MLflow experiment name; resolved by exact name and created if absent. Must be non-empty after trimming. |
| `--db PATH` | `~/.local/share/opencode/opencode.db` | OpenCode SQLite database path. `~` is expanded, relative paths are made absolute against the working directory, and `.`/`..` components are resolved lexically — without touching the filesystem or resolving symlinks. |
| `--tracking-uri URI` | *(ambient MLflow configuration)* | MLflow tracking URI. When omitted, the standard MLflow client/environment configuration applies (e.g. `MLFLOW_TRACKING_URI`). |
| `--from-time ISO` | *(none)* | Inclusive lower bound on session creation time. |
| `--to-time ISO` | *(none)* | Inclusive upper bound on session creation time. |
| `--directory PATH` | *(none)* | Restrict to sessions whose directory equals this path or is a component-descendant of it. |
| `--model PROVIDER/MODEL` | *(none)* | Exact model identity match (see below). |
| `--min-user-turns N` | *(none)* | Select sessions with **strictly more than** `N` logical user turns. Non-negative integer; omitting it imposes no restriction. |
| `--redact` | off | Mask secrets in trace values, metadata, errors, and artifacts (see [security.md](security.md)). |
| `--dry-run` | off | Validate, filter, and plan without mutating anything in MLflow. |
| `--output human\|json` | `human` | Result document format, written to stdout. |
| `--log-level DEBUG\|INFO\|WARNING\|ERROR` | `INFO` | Log verbosity. |

Invalid semantic values (a naive timestamp, a negative turn threshold, an
empty experiment name or model identity) fail **before** any source or
MLflow work with a safe error line on stderr and exit status `2`. Safe
error messages never echo the offending value.

## Selection semantics

Every supplied filter combines with logical AND; an omitted filter imposes
no restriction.

### Time bounds

`--from-time` and `--to-time` are ISO-8601 timestamps that **require an
explicit timezone offset or `Z`** (for example `2026-10-01T00:00:00Z` or
`2026-10-01T02:00:00+02:00`). A timestamp without an offset is rejected.
Values are converted to UTC instants while preserving the exact instant.

Bounds are **inclusive** and apply only to the session's **creation time**
(`time_created`) — never to a message timestamp and never to
`time_updated`. A session whose creation time is missing is ineligible
whenever any time filter is supplied; without a time filter it stays
selectable.

### Directory

`--directory` matches by whole path components: a session in
`/repo/project/app` matches `--directory /repo/project`, while a session
in the similarly prefixed sibling `/repo/projector` does not. The requested
path is lexically normalized the same way as `--db`.

### Model

`--model` is matched exactly after trimming surrounding whitespace; case
is preserved and no component is ever invented. Complete identities use
`provider/model`. Incomplete forms keep their shape — `provider/` matches
only a source identity available in that same provider-only form, and a
complete `provider/model` request never matches an incomplete source
identity. A session matches when any model identity observed anywhere in
it (session default, message-level, or part-level) equals the request.

### Minimum user turns

`--min-user-turns N` selects sessions with **strictly more than** `N`
logical user turns. Omitting the option imposes no restriction, so
zero-turn sessions remain selectable — that is what allows a later run to
retire stale traces for sessions whose turns have all disappeared. An
explicit `--min-user-turns 0` means "more than zero turns" and excludes
zero-turn sessions. An eligible session always imports **every** logical
turn; there is no per-turn filter.

### Broad-selection warning

A run with no optional selection filter at all (no time bounds, directory,
model, or turn threshold) reports a `broad_selection` warning:

```text
warning (broad_selection): no selection filters were supplied, so every
session in the source is in scope; run with --dry-run first to preview the
planned actions
```

The warning appears in both output modes (`summary.warnings` in JSON) and
is the report's recommendation to preview first.

## Dry runs

`--dry-run` performs the full pipeline except every mutation: the source
snapshot, discovery, selection, capability probe (read-only), experiment
resolution (planned), owned-trace lookup, replay fingerprint verification,
and reconciliation **planning** all run; nothing is created, uploaded,
retired, or deleted, and the advisory lock is not acquired.

Every turn is reported as `action="skipped"` with `reason="dry_run"` and a
predicted `planned_action` (`imported`, `updated`, or `retired`) from the
same action enum, so automation can distinguish preview from completion
without a sixth action name. The predicted `retired` entries carry
`source_present=false` and the trace id that would be retired. The summary
carries a `planned` subcount (`imported`/`updated`/`retired`).

One caveat is inherent to a non-mutating preview: the final
unchanged-versus-replace decision for an `updated` prediction waits for
candidate verification, which a dry run does not perform. Re-run without
`--dry-run` for the authoritative outcome.

## Output modes

`--output human` (the default) prints a concise progress-oriented summary
derived entirely from the result document — status line, experiment,
source scan counts, redaction mode, turn counts, planned counts (dry run),
retired/unexamined trace counts, then warning and error lines.

`--output json` prints the full versioned result document
(`opencode-mlflow.result.v1`) as deterministic, sorted, indented JSON.
Field-by-field documentation is in [result-schema.md](result-schema.md).

Both modes write to **stdout only**; pre-invocation failures print one
safe error line to stderr instead of a result document.

## Exit statuses

| Status | Meaning |
| --- | --- |
| `0` | Complete run, or a no-op (nothing to import, update, or retire). |
| `1` | At least one session/turn item failed while later items may have continued (per-item failure isolation). |
| `2` | Invalid invocation (argument or semantic validation), or a pre-write failure: source unavailable/unsupported, snapshot failure, MLflow capability failure, lock held, replay consistency mismatch, or any other failure before writes. |

`--help` exits `0`; argument syntax errors exit `2` (argparse behavior).

## Examples

The examples below use synthetic paths and identifiers only.

Preview a filtered import (recommended first step):

```bash
opencode-mlflow import --experiment-name team-sessions \
  --from-time 2026-10-01T00:00:00Z --to-time 2026-10-31T23:59:59Z \
  --directory /Users/dev/Code/service --min-user-turns 2 \
  --dry-run --output json
```

Run it for real:

```bash
opencode-mlflow import --experiment-name team-sessions \
  --from-time 2026-10-01T00:00:00Z --to-time 2026-10-31T23:59:59Z \
  --directory /Users/dev/Code/service --min-user-turns 2
```

Re-run the same command later: unchanged turns are reported `skipped` with
reason `unchanged`, turns that changed are replaced trace-for-trace, and
turns that disappeared from the source are retired.

Import everything with secrets masked:

```bash
opencode-mlflow import --experiment-name all-sessions --redact
```

Check whether a specific tracking server can serve the importer (a dry run
performs the read-only capability probe; an incompatible pair fails with
exit status `2` and `mlflow_capability_unsupported`):

```bash
opencode-mlflow import --experiment-name probe --tracking-uri http://127.0.0.1:8791 --dry-run
```

Consume the JSON result from a script:

```bash
opencode-mlflow import --experiment-name team-sessions --output json \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["summary"])'
```

## What the importer does NOT do

These are design scope decisions, not missing features:

- It does not watch the database or run continuously — each invocation is
  one complete pass.
- It does not call OpenCode's HTTP API or any OpenCode service; the only
  input is the SQLite database.
- It does not accept arbitrary filter expressions; the filters are exactly
  the options above.
- It does not create MLflow runs; output is MLflow traces, spans, and
  trace artifacts only.
- It does not write to MLflow's backend store or artifact store except
  through the public MLflow client API.
