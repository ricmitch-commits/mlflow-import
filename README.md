# mlflow-import

One-shot CLI that imports coding-agent sessions into MLflow as traces.

`mlflow-import` targets session import from OpenCode, Claude Code,
Codex, PI, and Gemini CLI. The fixture-qualified adapters reconstruct logical turns and write one
MLflow trace per turn — with child spans for messages, tool calls, and
content, shared session JSON plus reusable turn maps and a manifest, and deterministic
re-run behavior: an unchanged session imports once, a changed turn replaces
only that turn's trace, and traces the source no longer carries are retired.

Each harness is handled by a source adapter. Adapters discover and normalize
the harness's supported local session format while preserving source identity,
ordering, timing, metadata, and raw records where MLflow has no native field.
The trace/session `source` value (`mlflow.source.name` in creation tags) is
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
The qualified implementation has synthetic end-to-end, resource and independent
review evidence; final post-audit verification/sign-off is tracked in
[docs/errors.md](docs/errors.md). OpenCode has verified reference-scale evidence. All five have
fixture-backed [source processing](docs/source-adapters.md) and independent
[native/structured readback](docs/projection-parity.md), with explicit supported
schema, pricing and unavailable-data boundaries. This is not a claim that local
storage retained every live-plugin observation. See
[docs/compatibility.md](docs/compatibility.md) for tested boundaries,
[docs/errors.md](docs/errors.md) for resolved failures and final verification status, and
[CHANGELOG.md](CHANGELOG.md).

The qualified projection is `mapping-v8` / `accounting-v2`: invocation-aware
usage, persisted-first costs, reproducible frozen API-price bases and actual
versioned coverage. Unknown USD costs remain absent native values, not zeros or
server-generated prices. Source-schema, model/rate and payload limits remain
explicit in the profile documentation.

OpenCode native trace info follows the `@mlflow/opencode` 0.4.0 reference:
bare session/parent IDs, semantic assistant completion time, plain-text previews,
input/output/reasoning totals, available cache counters, and available persisted
USD costs. Unavailable cache lifetimes and cost breakdowns remain NULL rather
than estimated. OpenCode child spans use the reference plugin vocabulary
exactly: flat `llm_call` and `tool_<name>` children under
`opencode_conversation`, the plugin's cumulative reconstructed conversation
inputs, and its tool payload shapes. Reasoning/text/step/patch/lifecycle records
remain in the canonical artifacts instead of becoming extra spans; richer
accounting remains on the trace root and in artifacts. Turns above the plugin's
1000-message fetch window keep the importer's bounded linear fallback. Native
client request IDs
are source-only; operation keys and qualified identity live in initial ownership
tags. See [projection parity](docs/projection-parity.md) for the exact contract.

### Wire-contract compatibility

New traces and results use the neutral `mlflow_import.*` identifiers and
`mlflow-import.result.v1`. Legacy `opencode.*` ownership metadata and
legacy trace fields are recognized by reconciliation. The typed `compatibility-v1`
decoder reads both `mlflow-import.result.v1` and `opencode-mlflow.result.v1`, retaining
the original schema. Historical producer compatibility is fixture/live-qualified
for revision `36ed44f`; other historical variants are not blanket-qualified, and
old unbound in-flight journals are not automatically upgraded. See
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
| `--from-time DATE\|ISO` | Inclusive creation-time lower bound; date-only means `00:00:00Z`. Timestamps require an offset or `Z`. |
| `--to-time DATE\|ISO` | Inclusive creation-time upper bound; date-only means `23:59:59Z` (excludes later fractional seconds). |
| `--directory PATH` | Restrict to sessions in this directory and all component-descendants. |
| `--model PROVIDER/MODEL` | Exact normalized model identity match. |
| `--min-user-turns N` | Select sessions with strictly more than `N` logical user turns. Omitted = no restriction. |
| `--redact` | Mask secrets and user identity in session/trace values, metadata, errors, and artifacts, including native MLflow user fields. Without it, permitted source content and available user metadata are uploaded verbatim; use it for shared or production tracking servers. |
| `--dry-run` | Validate, filter, and plan without mutating anything in MLflow. |
| `--timeout-seconds SECONDS` | Finite positive invocation wall-clock budget, including discovery, preflight, recovery, and writing (default: `600`; fractional seconds accepted). |
| `--max-trace-bytes BYTES` | Per-trace serialized span-byte admission limit (default: `67108864`; maximum: `2147483648`). Raising it is an explicit server-capacity decision. |
| `--max-trace-spans N` | Per-trace span-count admission limit including the root (default: `4096`; maximum: `32768`). |
| `--output human\|json` | Result format (default: `human`). |
| `--log-level DEBUG\|INFO\|WARNING\|ERROR` | Log verbosity (default: `INFO`). `DEBUG` prints live phase/count progress to stderr; the final result remains on stdout. |

With `--redact`, present user-identity values (IDs, usernames, names, and email
addresses) become `[REDACTED]` in session/trace metadata and retained raw artifact
copies, including `mlflow.trace.user` and any populated `mlflow.user` field.
Missing user values stay absent. The conversation role `user` and the identifiers
needed for session/turn reconciliation remain intact. The harness-valued `source`
is the same in either privacy mode.

The timeout is shared across the entire invocation. Separate shared limits of
64 retries and 1,000,000 operations still apply. For a larger workload, explicitly
allow 30 minutes while keeping the same SessionsAI import command:

```bash
mlflow-import --experiment-name SessionsAI --timeout-seconds 1800
```

Add `--log-level DEBUG` for phase/count progress on stderr. The default budget
remains 600 seconds; it is not automatically raised for larger selections. See
[import operations](docs/operations.md) for budget and recovery details.

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
- Date-only bounds use UTC day boundaries. Timestamps require an explicit
  timezone offset or `Z`. Bounds are inclusive and use source creation time.
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

## Synthetic benchmark

The strengthened disposable local gate verified **307 sessions** and
**494 / 494 / 495 / 495 turns** across fresh, unchanged, append and edit phases,
including exact paged target trace-ID cardinality. The generated workload contains
319,987,518 JSON bytes, 8,134 messages and 47,976 parts; all content is synthetic.
The four invocations took **378.763 / 308.581 / 311.298 / 311.747 seconds**,
each below 600 seconds; unchanged performed **zero** gateway mutations.
These results supersede the earlier 511.8-second fresh timing.
Hardware, complete phase timings, exact command and limits are in
[docs/compatibility.md](docs/compatibility.md#synthetic-performance-evidence).
