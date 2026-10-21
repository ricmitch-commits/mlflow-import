# Testing

This document describes the shipped test layers, the commands that run
them, the fixture provenance rules, and the live-server integration
recipe. Marker definitions live in `pyproject.toml`
(`--strict-markers`).

## The six-step TDD/BDD loop

Every implementation task in this repository was built red → green →
refactor:

1. Add one focused failing test or Gherkin scenario.
2. Run the narrow test and record the expected failure.
3. Add the smallest implementation that satisfies that contract.
4. Run the narrow test, then the task-level test group.
5. Refactor only after the tests pass; rerun unit, contract, and the
   relevant BDD tests.
6. Commit the task as one reviewable change.

New work should follow the same loop. BDD scenarios must call the public
importer/CLI boundary and never assert private helper calls.

## Test layers and commands

All commands run through the project venv from the repository root
(`PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest ...`).

### Default selection (what CI runs on every push)

```bash
.venv/bin/python -m pytest -m '(unit or contract or bdd) and not integration'
```

- **`unit`** — fast, isolated tests with no network, no live MLflow, and
  no real database: configuration/CLI validation, source models and
  ordering, catalog and probe, connection safety, snapshot, replay,
  turns, discovery, selection, canonical JSON golden vectors,
  fingerprints, redaction detectors, artifacts and inline limits,
  mapping, gateway (against fakes), capabilities, reconciliation,
  retry classification, locking, reporting, and the importer
  orchestration seams.
- **`contract`** — contract tests over public interfaces and the pinned
  compatibility profile: the same parametrized adapter assertions
  against legacy/current/mixed synthetic SQLite databases; the gateway
  contract (AST-scans the source tree so only the gateway modules
  import MLflow, and pins the gateway constants against the installed
  mlflow); the mapping contract (spec/request field parity); and the
  MLflow profile contract (markers `profile` **and** `integration` —
  excluded from the default selection).
- **`bdd`** — the Gherkin acceptance suite (41 scenarios across five
  feature files) driven through the public CLI/importer boundary with
  synthetic databases and the fake gateway:
  `tests/bdd/features/cli_and_selection.feature`,
  `mapping_and_fidelity.feature`, `privacy_and_reporting.feature`,
  `reconciliation.feature`, and `source_safety.feature`, with steps in
  `tests/bdd/steps/test_import_steps.py`.

Expected result on this checkout: **1363 passed, 32 deselected**.

### Lint and type checks

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
```

### Explicit integration/profile selection

```bash
MLFLOW_TRACKING_URI="http://127.0.0.1:8791" \
  .venv/bin/python -m pytest -m '(integration or profile)' -q
```

- **`integration`** — tests that require a real WAL snapshot or a live
  MLflow tracking server. Never silently skipped: a configured
  `MLFLOW_TRACKING_URI` that is not a healthy HTTP(S) server fails the
  session loudly.
- **`profile`** — tests pinning the exact MLflow client/server
  compatibility profile (also marked `integration`).
- **`slow`** — tests too slow for the default selection (the scale
  suite).

`tests/integration/` contains 25 integration-marked tests:
`test_mlflow_trace_contract.py` (10) and
`test_mlflow_reconciliation.py` (5) drive the COMPLETE production
pipeline against a real server; `test_wal_snapshot.py` (5) proves
live-WAL snapshot semantics (including a commit landing mid-extraction)
and needs no server; `test_final_review_gateway.py` (2) verifies live
missing-timestamp rejection and partial-child rollback.
`test_scale_memory.py` (3) measures the artifact stream (105k records /
241 MiB / <256 MiB heap), the actual 100k-record SQLite→discovery→mapping→
gateway path with a payload-discarding span sink, complete generated identity
map and streamed artifact (<256 MiB importer-owned structures), and
`payload_too_large` rejection at the real 1 MiB inline
limit. The sink deliberately excludes real SDK span-payload retention;
**this is not a complete live-MLflow 100k-span memory proof**. See the
measured limitation in [architecture](architecture.md). The five profile contract tests in
`tests/contract/test_mlflow_profile.py` also run against the server.

Discovery turn summaries are spooled to the restrictive run directory rather
than retained as a per-session Python list. Reconciliation and reporting still
retain sanitized identifiers and per-item outcomes because those entries are
the JSON result contract; their memory therefore scales with report
cardinality. The scale tests do not claim a strict full-SDK bound for a
100,000-span live import.

## Fake gateway use

The gateway boundary is injectable through three seams (the
`MlflowClient`, the module-level span API adapter, and the
artifact-repository factory), so `tests/unit/mlflow/` and
`tests/contract/test_gateway_contract.py` drive the REAL gateway against
in-memory fakes (`tests/support/fake_gateway.py`) with no network and no
server. The fakes mirror the behavioral facts the live profile pinned —
tag/metadata truncation limits, pagination tokens, `tags.`/`metadata.`
filter strings, the `mlflow.artifactLocation` tag, MLflow-shaped
protocol errors, and the real provider's child-span degradation when a
non-root span carries `experiment_id` — so drift between fakes and the
pinned pair fails the fake-backed suites rather than hiding until the
live run. The BDD suite uses the same fakes through
`tests/support/import_fakes.py`.

The live profile test remains the authoritative evidence that the pinned
pair satisfies the writer contract; the gateway contract test pins the
gateway's constants and public-API call set to that profile.

## Fixture provenance rules

- **Sanitized synthetic data only.** Every SQLite fixture is generated
  from committed SQL/builders: `tests/fixtures/sqlite/` (`current.sql`,
  `legacy.sql`, `mixed.sql`, `edge_cases.sql`) and
  `tests/support/sqlite_builder.py`; the sanitized synthetic six-turn
  session lives in `tests/fixtures/six_turn_session.py`. Payload
  vectors are committed JSON:
  `tests/fixtures/payloads/canonical-json-v1-vectors.json`,
  `redaction-positive.json`, `redaction-negative.json`.
- The live OpenCode database, local MLflow logs, and SQLiteBrowser
  project files are **research inputs only** — never committed as
  fixtures and never used as runtime dependencies.
- No fixture contains real credentials or real transcript content;
  secret-detection fixtures use obviously synthetic sentinel values.

## Integration setup (disposable pinned server)

The complete recipe lives in
[tests/integration/README.md](../tests/integration/README.md). Summary:

1. Create the environment:
   `python3 -m venv .venv` and
   `.venv/bin/python -m pip install -r tests/integration/constraints.txt -e ".[dev]"`.
2. Either export `MLFLOW_TRACKING_URI` pointing at a healthy HTTP(S)
   server, or leave it unset — the session-scoped `live_server` fixture
   (`tests/integration/server.py`) launches ONE disposable
   `mlflow server` from the installed distribution (fresh run directory,
   free port, SQLite backend, local artifact root), health-checks it,
   and always reaps it on exit.
3. Run `pytest -m '(integration or profile)'`.
4. Clean up the disposable run directory.

The pinned pair is the client and server from the same installed
`mlflow==3.17.2` distribution (the exact stable release of the API line
the plan called `3.17.2.dev0`, which is not published on PyPI —
controller ruling 2026-10-21). There are no mutable `latest` images or
tags in the recipe.

Environment hazards the harness already handles (documented for
operators who replicate the setup manually):

- Ambient `OTEL_EXPORTER_OTLP_*` variables reroute MLflow span export to
  an external OTLP collector and silently degrade span creation to NoOp
  spans; the fixtures scrub these variables before the tracing provider
  initializes and fail loudly if a NoOp span is ever observed.
- Span export follows the *global* tracking URI
  (`mlflow.set_tracking_uri` / `MLFLOW_TRACKING_URI`), not an individual
  `MlflowClient`'s URI.
- Trace export is asynchronous; the harness uses the public `flush=True`
  parameter so observations are deterministic.

For the resolved public API surface, observed 3.17.2 behaviors, named
artifact upload and fixed-chunk verification, the retirement capability,
and how an unsupported client/server pair is reported, see
[compatibility.md](compatibility.md).

## Docs verification

Documentation changes are verified mechanically against the shipped
code, in both directions. `scripts/verify_docs.py` checks that:

- every markdown link and backticked repository path in the docs suite
  exists on disk;
- every CLI option documented in `README.md`/`docs/usage.md` exists in
  the real parser (and vice versa), and every example command parses
  through `parse_config` (`src/opencode_mlflow/config.py`);
- every documented result field, action, reason, and ownership key
  matches the serializer (`src/opencode_mlflow/reporting.py`,
  `src/opencode_mlflow/mlflow/reconciliation.py`);
- the schema-catalog matrix mirrors `src/opencode_mlflow/source/catalog.py`
  exactly, per generation and per table;
- the detector categories, replacement markers, and protected identity
  fields match `src/opencode_mlflow/content/redaction.py` and
  `src/opencode_mlflow/contracts.py`;
- documented error codes and pinned limits appear in `src/` with their
  exact values.

```bash
.venv/bin/python scripts/verify_docs.py
```

Run it whenever documentation or any documented contract changes; the
[code-review checklist](code-review-checklist.md) requires it for docs
changes.

## CI

`.github/workflows/ci.yml` runs the default selection plus
`ruff check .`, `ruff format --check .`, and `mypy src` on Python
3.11/3.12/3.13. A separate gated job runs `pytest -m '(integration or
profile)'` only when repository variable `MLFLOW_INTEGRATION_ENABLED=true`
or a nonempty repository variable `MLFLOW_INTEGRATION_URI` enables the job.
Set `MLFLOW_INTEGRATION_URI` as a secret for credential-bearing targets
(or as a variable for credential-free targets), exported inside the job as
`MLFLOW_TRACKING_URI`; it stays skipped otherwise and never blocks the
default job.
