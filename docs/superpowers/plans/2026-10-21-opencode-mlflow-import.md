# OpenCode SQLite Session Importer for MLflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `opencode-mlflow`, a Python 3.11+ one-shot CLI that reads a consistent, read-only OpenCode SQLite snapshot, reconstructs logical turns, and imports each selected turn as an importer-owned MLflow trace grouped by the native OpenCode session ID.

**Architecture:** Implement the design as a layered streaming pipeline: immutable CLI configuration → read-only SQLite snapshot and versioned schema adapters → canonical source records and logical turns → sanitized disk-backed discovery plan → MLflow capability/preflight and ownership reconciliation → verified trace/span/artifact writes → safe human or versioned JSON reporting. Keep SQL, source normalization, content policy, MLflow API access, reconciliation, and reporting behind separate interfaces so each layer can be unit- and contract-tested without a live database or tracking server.

**Tech Stack:** Python 3.11+, `argparse`, `sqlite3`, `dataclasses`, `json`, `hashlib`, `datetime`, `tempfile`, and the public MLflow Python client at runtime; `pytest`, `pytest-bdd`, `pytest-cov`, Ruff, mypy, and `build` as development tools. No ORM, backend-store writes, undocumented MLflow endpoints, or MLflow run fallback.

**Spec:** `docs/superpowers/specs/2026-10-21-opencode-mlflow-import-design.md` (with acceptance requirements in `docs/superpowers/specs/2026-10-21-opencode-mlflow-import-requirements.md`)

## Global Constraints

- Target Python **3.11+** and expose the console script **`opencode-mlflow`**.
- Use standard-library components plus the MLflow Python client; do not add an ORM.
- Pin the initial MLflow compatibility profile to the tested `3.17.2.dev0` API shape and one matching client/server build; newer profiles require a new compatibility test before changing the package constraint.
- Open the live SQLite source with URI `mode=ro`, `uri=True`, `PRAGMA query_only=ON`, a bounded busy timeout, and one deferred read transaction; stream only catalogued permitted canonical rows into a restrictive temporary normalized snapshot with `snapshot_sessions` and `snapshot_records`, close the live source connection, then open the normalized snapshot read-only with `query_only=ON` and use one transaction for discovery/replay. Do not use `immutable=1` or page-for-page backup.
- Keep `-wal` and `-shm` available, never write/checkpoint/vacuum/migrate/delete in the source database, exclude secret tables before temporary persistence, and treat snapshot extraction/population failure as exit status `2`.
- Put all parameterized SQL inside the adapter that owns the matching schema catalog; never query credential-bearing values.
- Current `session_v2`/`session_message` rows take precedence over overlapping legacy rows; legacy-only session IDs remain eligible.
- A mixed database may be reported as `mixed` by the source database description, but each emitted session/owned trace uses its adapter generation (`legacy` or `current`) for `opencode.source_schema`.
- Import only records persisted for a selected fork child; never synthesize parent history.
- Use bounded-memory streaming and restrictive temporary files; never put transcript content in the sanitized discovery JSONL plan.
- Select on inclusive session `time_created`, component-aware normalized directory, exact normalized model identity, and strict `logical_turn_count > min_user_turns`, with all supplied filters combined by AND.
- A persisted, non-synthetic user prompt starts a turn; a qualifying session imports all of its turns; sessions with no user turns do not produce empty traces.
- Preserve decoded values plus permitted raw JSON/text and unknown fields; exclude `account`, `account_state`, `control_account`, `credential`, and secret-bearing `session_share` values.
- Implement and test `canonical-json-v1`, including UTF-8 byte key ordering, RFC 8785 finite-number formatting, non-finite wrappers, BLOB/invalid-UTF-8 wrappers, absent-versus-NULL distinction, and duplicate-key preservation.
- Produce one MLflow trace per logical turn, with all traces from a source session sharing the native OpenCode session ID and having distinct MLflow trace IDs.
- Use only public MLflow client/trace/artifact APIs and enforce the behavioral capability contract; never silently degrade to runs or discard data.
- Use `(opencode.session_id, opencode.turn_key)` plus the exact ownership marker for reconciliation; never mutate unowned traces.
- Populate `opencode.render_fingerprint` and `opencode.operation_key`; verify all owned candidates, choose a deterministic valid winner, and retire redundant owned candidates only after a verified winner exists.
- Replace in the order create → verify trace → upload/verify artifact → retire old; preserve the old trace on uncertainty and retry only bounded transient failures.
- Treat retirement/deletion as a global normal-write capability because rollback is required after any partial create; missing required capability aborts the whole invocation before mutation with status `2`.
- Default to verbatim content; `--redact` applies deterministic version-one detectors and `[REDACTED]` to trace values, metadata, errors, and artifacts while protecting identifiers.
- Use source-derived timestamps only, mark source-derived fallbacks with `opencode.timing_estimated=true`, and never invent source timing from the local wall clock.
- Emit `opencode-mlflow.result.v1` JSON without prompts, responses, tool output, source code, credentials, tracking-URI secrets, or redaction matches.
- Use the normative action enum `imported | updated | skipped | failed | retired`; represent unchanged as a `skipped` reason/subcount, and represent dry-run with `action=skipped`, `reason=dry_run`, and `planned_action`.
- Keep Python heap growth below 256 MiB over the empty-process baseline for a fixture with at least 100,000 records and a multi-hundred-megabyte artifact; stream artifacts in fixed upload chunks and reject complete root payloads above the target inline limit rather than truncating them.
- Exit `0` for complete work or intentional no-op, `1` for item failures, and `2` for invalid/pre-write source or MLflow capability failures.
- Keep OpenCode HTTP support, watch mode, arbitrary filter expressions, and a default MLflow run container out of this release.
- Live OpenCode databases, local MLflow logs, and SQLiteBrowser project files are research inputs only; committed fixtures must be sanitized synthetic data.

## Planning boundary

This repository currently contains only the two requirements/design documents and is not under Git control. This session creates only this plan document. The implementation phase must create the package and test scaffolding described below, execute each task with TDD, and commit task-sized changes after the repository is placed under version control. No source code, fixtures, CI files, or documentation described in the tasks are being created now.

## Repository shape after implementation

The implementation should create the following focused layout. The exact internal helper names may evolve only when the public interfaces and responsibility boundaries remain equivalent.

```text
pyproject.toml
.gitignore
README.md
CHANGELOG.md
.github/workflows/ci.yml

docs/
  architecture.md
  compatibility.md
  schema-catalog-v1.md
  usage.md
  security.md
  testing.md
  result-schema.md
  operations.md
  code-review-checklist.md

src/opencode_mlflow/
  __init__.py
  contracts.py
  cli.py
  config.py
  errors.py
  importer.py
  locking.py
  reporting.py
  retry.py
  source/
    __init__.py
    catalog.py
    connection.py
    snapshot.py
    replay.py
    probe.py
    model.py
    legacy.py
    current.py
    ordering.py
    turns.py
    selection.py
    discovery.py
  content/
    __init__.py
    canonical.py
    fingerprint.py
    redaction.py
    artifacts.py
  mlflow/
    __init__.py
    gateway.py
    capabilities.py
    mapping.py
    reconciliation.py

tests/
  conftest.py
  support/
    sqlite_builder.py
    fake_gateway.py
    fake_clock.py
    report_assertions.py
  fixtures/
    sqlite/
      legacy.sql
      current.sql
      mixed.sql
      edge_cases.sql
    payloads/
      redaction-positive.json
      redaction-negative.json
      canonical-json-v1-vectors.json
    six_turn_session.py
  unit/
    test_config.py
    test_cli.py
    test_errors.py
    source/
      test_connection.py
      test_catalog.py
      test_probe.py
      test_model.py
      test_legacy.py
      test_current.py
      test_ordering.py
      test_turns.py
      test_selection.py
      test_discovery.py
    content/
      test_canonical.py
      test_fingerprint.py
      test_redaction.py
      test_artifacts.py
    mlflow/
      test_capabilities.py
      test_gateway.py
      test_mapping.py
      test_reconciliation.py
    test_retry.py
    test_locking.py
    test_reporting.py
    test_importer.py
  contract/
    test_source_adapter_contract.py
    test_gateway_contract.py
    test_mapping_contract.py
    test_mlflow_profile.py
  bdd/
    conftest.py
    features/
      cli_and_selection.feature
      source_safety.feature
      mapping_and_fidelity.feature
      reconciliation.feature
      privacy_and_reporting.feature
    steps/
      test_import_steps.py
  integration/
    constraints.txt
    README.md
    mlflow_profile.py
    test_wal_snapshot.py
    test_scale_memory.py
    test_mlflow_trace_contract.py
    test_mlflow_reconciliation.py
```

## TDD and BDD strategy

Every implementation task follows red → green → refactor:

1. Add one focused failing test or Gherkin scenario.
2. Run the narrow test and record the expected failure.
3. Add the smallest implementation that satisfies that contract.
4. Run the narrow test, then the task-level test group.
5. Refactor only after the tests pass; rerun unit, contract, and relevant BDD tests.
6. Commit the task as one reviewable change when Git is available.

The test layers have distinct purposes:

- **Unit tests** exercise pure normalization, source models, ordering, hashing, redaction, mapping, retry classification, reconciliation planning, reporting, and CLI validation without network access.
- **Source contract tests** run the same adapter assertions against current, legacy, and mixed synthetic SQLite databases. They verify that adapters expose the same canonical semantics rather than testing SQL implementation details only.
- **Gateway contract tests** run against `FakeMlflowGateway` and verify the public gateway behavior required by the importer: capability checks, pagination, full-payload verification, deterministic artifact paths, ownership filtering, and retirement semantics.
- **BDD tests** express acceptance criteria in Gherkin and call the public importer/CLI boundary with synthetic databases and the fake gateway. They must not assert private helper calls.
- **Integration tests** run only with an explicitly configured, pinned trace-capable MLflow client/server pair and verify actual public API behavior. They are marked `integration` and excluded from the default unit/BDD command.

Configure markers in `pyproject.toml` as `unit`, `contract`, `bdd`, `integration`, `profile`, and `slow`. Default CI runs `(unit or contract or bdd) and not integration`; the profile/integration job starts or connects to the pinned MLflow pair and fails rather than silently skipping when the pair is requested.

## Implementation tasks

**Execution order:** Because the compatibility profile requires the test/package harness but must block writer work, execute **Task 1 (minimal scaffold) → Task 0 (profile gate) → Tasks 2–16**. The `Task 0` label records architectural priority; it is not a request to run tests before `pyproject.toml` and the test harness exist.

### Task 0: Establish the pinned MLflow compatibility profile before writer work

**Sequencing gate:** Create the minimal packaging/test harness from Task 1 first so this profile can run, then execute Task 0 before implementing any MLflow gateway/writer behavior. Task 0 is listed first because its result authorizes or blocks the writer; it does not add importer behavior.

**Files:**

- Create: `tests/integration/constraints.txt`
- Create: `tests/integration/README.md`
- Create: `tests/integration/mlflow_profile.py`
- Create: `tests/contract/test_mlflow_profile.py`
- Create: `docs/compatibility.md`

**Interfaces:**

- `mlflow_profile.py` produces `MlflowProfile(version: str, server_command: Sequence[str], api_table: Mapping[str, str])`, `ArtifactVerification(trace_id: str, relative_path: str, sha256: str, size: int)`, `CapabilityReport(is_compatible: bool, required_methods: frozenset[str], max_inline_payload_bytes: int, named_artifact_supported: bool, retirement_supported: bool, safe_error: str | None)`, and `verify_profile(client: MlflowClient, profile: MlflowProfile) -> CapabilityReport`.
- The profile adapter must expose these exact gateway-facing operations: `MlflowClient.get_experiment_by_name`, `MlflowClient.create_experiment`, public module-level `mlflow.start_span_no_context(name, span_type, inputs, attributes, tags, metadata, experiment_id, start_time_ns, parent_span=None)` returning `LiveSpan(trace_id, span_id)`, `LiveSpan.end(outputs, attributes, status, end_time_ns)`, `MlflowClient.get_trace(trace_id)`, paginated `MlflowClient.search_traces(experiment_ids, filter_string, max_results, page_token)`, public trace metadata/tag update, and public `MlflowClient.delete_traces(experiment_id, trace_ids)` or documented equivalent.
- The profile must also demonstrate a public named trace-artifact operation adapted to `TraceArtifactSink.upload_named_json(trace_id: str, relative_path: str, local_file: Path, chunk_size: int) -> ArtifactVerification` and `verify_named_json(trace_id: str, relative_path: str, sha256: str, size: int) -> None`. A UUID-only attachment or run-artifact method does not satisfy the profile.

- [ ] **Step 1: Write the compatibility contract before any MLflow writer code.**

```python
def test_profile_requires_retirement_for_normal_writes(profile_client):
    report = verify_profile(profile_client, MlflowProfile.initial())

    assert report.required_methods >= {
        "get_experiment_by_name",
        "create_experiment",
        "mlflow.start_span_no_context",
        "LiveSpan.end",
        "get_trace",
        "search_traces",
        "delete_traces",
        "upload_named_json",
        "verify_named_json",
    }
```

- [ ] **Step 2: Pin the initial profile reproducibly.**

Pin `mlflow==3.17.2.dev0` in `tests/integration/constraints.txt` and use a server launched from that same installed distribution with a temporary SQLite backend and local artifact root. `tests/integration/README.md` must provide the exact virtualenv/install/server command sequence, health check, artifact-root setup, and cleanup procedure; do not use a mutable `latest` image/tag. Record every resolved public method/field and its observed behavior in `docs/compatibility.md`.

- [ ] **Step 3: Probe named artifact behavior against the disposable server.**

Demonstrate that the public operation preserves the requested relative path, accepts fixed chunks without loading the whole file, and allows retrieval/verification of exact bytes. If the installed `3.17.2.dev0` pair lacks this public behavior, record the negative result and stop the implementation plan at the compatibility gate rather than substituting private artifact repositories or MLflow runs.

- [ ] **Step 4: Verify dry-run capability probing is non-mutating.**

Probe method availability, server reachability/protocol responses, full-payload retrieval, artifact behavior, and retirement support without creating a temporary experiment or uploading a probe file. A target that cannot prove the contract without mutation is unsupported for dry-run.

- [ ] **Step 5: Run the profile contract and commit the prerequisite.**

Run: `python -m pytest tests/contract/test_mlflow_profile.py -m 'profile and integration' -q`

Expected: PASS against the pinned disposable server; a missing capability is a clear global preflight failure. Commit as `test: pin mlflow compatibility profile`.

### Task 1: Create the package, test, tooling, and CI scaffold

**Files:**

- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.github/workflows/ci.yml`
- Create: `src/opencode_mlflow/__init__.py`
- Create: `src/opencode_mlflow/contracts.py`
- Create: `src/opencode_mlflow/cli.py` (importable entry-point bootstrap; Task 2 completes the parser)
- Create: `src/opencode_mlflow/source/__init__.py`
- Create: `src/opencode_mlflow/content/__init__.py`
- Create: `src/opencode_mlflow/mlflow/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/support/sqlite_builder.py`
- Create: `tests/support/fake_clock.py`
- Create: `tests/unit/test_package_scaffold.py`
- Create: `tests/bdd/conftest.py`
- Create: `tests/bdd/steps/test_import_steps.py` (fixture/step harness scaffold)
- Create: `tests/bdd/features/cli_and_selection.feature`
- Create: `tests/bdd/features/source_safety.feature`
- Create: `tests/bdd/features/mapping_and_fidelity.feature`
- Create: `tests/bdd/features/reconciliation.feature`
- Create: `tests/bdd/features/privacy_and_reporting.feature`

**Interfaces:**

- `contracts.py` produces the shared `RedactionMode` enum (`VERBATIM`, `REDACTED`), `CanonicalRecordEnvelope(decoded: object | None, raw_text: str | None, raw_columns: Mapping[str, object], warnings: tuple[str, ...])`, and `Redactor` protocol (`redact(value: object, protected_paths: Collection[str]) -> object`). `content/canonical.py` and `content/redaction.py` re-export these shared types rather than defining incompatible duplicates.
- Produces package metadata with `opencode_mlflow.__version__`, an installable `src` layout, and an importable console-script target `opencode_mlflow.cli:main`; the bootstrap must support `opencode-mlflow --help` before later CLI behavior is added.
- Produces pytest markers, shared temporary-directory fixtures, a BDD feature/step harness, and a fixture-builder seam that later tasks can use without coupling tests to committed database files.
- Produces CI jobs for Python 3.11+ unit/contract/BDD checks and a separately gated integration job.

- [ ] **Step 1: Write the failing package smoke test.**

```python
def test_package_metadata_and_entry_point_import():
    import importlib.metadata
    import subprocess
    import opencode_mlflow

    assert opencode_mlflow.__version__
    assert importlib.metadata.version("opencode-mlflow") == opencode_mlflow.__version__
    assert subprocess.run(["opencode-mlflow", "--help"], check=False, capture_output=True).returncode == 0
```

- [ ] **Step 2: Run the smoke test to verify the scaffold is absent.**

Run: `python -m pytest tests/unit/test_package_scaffold.py -q`

Expected: collection fails because the package metadata and source package do not exist.

- [ ] **Step 3: Add the minimal packaging scaffold.**

Define in `pyproject.toml` the project name, version `0.1.0`, Python requirement `>=3.11`, exact runtime dependency `mlflow==3.17.2.dev0` for the initial compatibility profile, the `opencode-mlflow` console script, optional development dependencies for pytest/pytest-bdd/pytest-cov/Ruff/mypy/`build`, and pytest marker configuration. Add package `__init__.py` files plus an importable CLI bootstrap; do not add importer behavior.

- [ ] **Step 4: Add test support and CI configuration.**

Make `tests/conftest.py` provide a restrictive temporary directory and a deterministic `fake_clock` fixture. Make `tests/bdd/conftest.py` register pytest-bdd discovery and shared scenario fixtures; give each feature file a valid `Feature:` declaration even before its scenarios are filled in. Configure `.gitignore` to exclude virtual environments, caches, coverage output, temporary artifacts, local MLflow stores, and generated SQLite databases. Configure CI to run `python -m pytest -m '(unit or contract or bdd) and not integration'`, `ruff check .`, `ruff format --check .`, and `mypy src` on the supported Python versions. Keep the profile and real MLflow integration in a separate job guarded by its explicit service/URI configuration.

- [ ] **Step 5: Run the scaffold checks.**

Run: `python -m pip install -e '.[dev]'`, then `python -m pytest tests/unit/test_package_scaffold.py -q` and `opencode-mlflow --help`

Expected: PASS. Then run `python -m build --wheel --no-isolation` and verify the wheel exposes `opencode-mlflow` without importing source data.

- [ ] **Step 6: Commit the scaffold.**

```bash
git add pyproject.toml .gitignore .github/workflows/ci.yml src tests
git commit -m "build: scaffold opencode mlflow importer"
```

### Task 2: Define safe errors, immutable configuration, and CLI validation

**Files:**

- Create: `src/opencode_mlflow/errors.py`
- Create: `src/opencode_mlflow/config.py`
- Complete: `src/opencode_mlflow/cli.py`
- Create: `tests/unit/test_errors.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_cli.py`
- Extend: `tests/bdd/features/cli_and_selection.feature`
- Extend: `tests/bdd/steps/test_import_steps.py`

**Interfaces:**

- `errors.py` produces `ImporterError(code: str, safe_message: str, exit_status: int)`, plus `ConfigurationError`, `SourceCompatibilityError`, `SnapshotError`, `CapabilityError`, `ItemImportError`, and `ReconciliationConflict` subclasses. Their string form must contain only the safe message.
- `config.py` produces frozen `ImportConfig`, `parse_config(argv: Sequence[str]) -> ImportConfig`, `parse_timestamp(value: str) -> datetime`, `normalize_path(value: str) -> Path`, and `normalize_model_identity(value: str) -> str`.
- `cli.py` produces `build_parser() -> argparse.ArgumentParser` and `main(argv: Sequence[str] | None = None) -> int`; `main` delegates execution to the importer seam added in Task 12 and never logs transcript content.

- [ ] **Step 1: Write validation tests before parser code.** Cover the required `import` subcommand, top-level invocation without a subcommand exiting `2`, `--help` exiting `0`, required experiment names, explicit-offset/Z timestamps, inclusive UTC conversion, rejection of naive timestamps, non-negative integer thresholds, absolute path normalization without filesystem resolution, component-aware directory matching, complete/provider-only/model-only identities, preserved model case, `output` choices, redaction/dry-run flags, and safe error codes.

```python
def test_parse_timestamp_requires_explicit_timezone():
    from datetime import timezone
    from opencode_mlflow.config import parse_timestamp

    assert parse_timestamp("2026-10-21T12:00:00Z").tzinfo == timezone.utc
    assert parse_timestamp("2026-10-21T08:00:00-04:00").isoformat() == "2026-10-21T12:00:00+00:00"

    with pytest.raises(ConfigurationError) as exc:
        parse_timestamp("2026-10-21T12:00:00")
    assert exc.value.code == "invalid_timestamp"
    assert "2026-10-21T12:00:00" not in str(exc.value)
```

- [ ] **Step 2: Run the focused validation tests.**

Run: `python -m pytest tests/unit/test_config.py tests/unit/test_cli.py -q`

Expected: FAIL because the configuration types and parser do not exist.

- [ ] **Step 3: Implement the immutable configuration boundary.**

Use a frozen dataclass with fields `experiment_name`, `db_path`, `tracking_uri`, `from_time`, `to_time`, `directory`, `model`, `min_user_turns`, `redact`, `dry_run`, `output`, and `log_level`. Default the database path to `~/.local/share/opencode/opencode.db`, expand and absolutize it without requiring existence or resolving symlinks, trim model whitespace while preserving case, and reject invalid values before any source or MLflow object is constructed.

- [ ] **Step 4: Implement the CLI parser and safe error conversion.**

Expose the `import` subcommand and all options from the requirements. Use `argparse` for syntax errors and convert semantic failures to status `2`. Do not validate source existence in `config.py`; that belongs to the source preflight boundary. Add the no-filter warning as a report-level event rather than a secret-bearing log line.

- [ ] **Step 5: Add BDD scenarios for CLI contracts.**

```gherkin
Feature: Validate import configuration
  Scenario: Reject a timestamp without an explicit timezone
    When I invoke the importer with "--from-time 2026-10-21T12:00:00"
    Then the process exits with status 2
    And the safe error code is "invalid_timestamp"
    And the result does not contain "2026-10-21T12:00:00"

  Scenario: Normalize an incomplete model identity exactly
    Given a session contains the model identity "provider/"
    When I select model " provider/ "
    Then the session is selected
    And a complete model request does not match "provider/"
```

- [ ] **Step 6: Run the task checks and commit.**

Run: `python -m pytest tests/unit/test_errors.py tests/unit/test_config.py tests/unit/test_cli.py -m unit -q`

Expected: PASS with no network or SQLite access. Commit as `feat: add validated importer configuration`.

### Task 3: Implement the read-only SQLite snapshot and schema probe

**Files:**

- Create: `src/opencode_mlflow/source/model.py`
- Create: `src/opencode_mlflow/source/catalog.py`
- Create: `src/opencode_mlflow/source/connection.py`
- Create: `src/opencode_mlflow/source/snapshot.py`
- Create: `src/opencode_mlflow/source/replay.py`
- Create: `src/opencode_mlflow/source/probe.py`
- Create: `docs/schema-catalog-v1.md`
- Create: `tests/support/sqlite_builder.py` updates for WAL fixtures
- Create: `tests/fixtures/sqlite/legacy.sql`
- Create: `tests/fixtures/sqlite/current.sql`
- Create: `tests/fixtures/sqlite/mixed.sql`
- Create: `tests/fixtures/sqlite/edge_cases.sql`
- Create: `tests/unit/source/test_model.py`
- Create: `tests/unit/source/test_catalog.py`
- Create: `tests/unit/source/test_connection.py`
- Create: `tests/unit/source/test_replay.py`
- Create: `tests/unit/source/test_probe.py`
- Create: `tests/integration/test_wal_snapshot.py`

**Interfaces:**

- `source/model.py` produces `SchemaGeneration`, `TimestampValue`, `SchemaDescription`, `SessionHeader` (including `adapter_generation: Literal["legacy", "current"]` for emitted sessions), `SourceRecord.from_row(source_table: str, source_type: str, source_id: str | None, raw_json: str | None, **raw_columns: object) -> SourceRecord`, `TurnSummary`, and the `SourceAdapter` protocol. `SourceRecord` includes `session_id`, `sequence_domain`, `sequence_value`, `parent_source_id`, `derived_child_ordinal`, `source_identity`, and `join_projection_key` so adapter contract tests can verify session ownership, cross-table order, and safe missing-ID identity.
- `source/catalog.py` produces `CatalogColumn`, `JoinRule`, `RecordClassification`, `SchemaCatalog`, and the versioned `OPENCODE_SCHEMA_CATALOG_V1`. Each catalog entry names required tables/columns, optional known columns, join predicates, timestamp columns and units, JSON locations, user/assistant/system/synthetic classification rules, safe session-scoped columns, authoritative sequence domains, fixed tie ranks, and raw-field handling.
- `SourceAdapter` must expose `schema_generation`, `inspect() -> SchemaDescription`, `iter_session_headers() -> Iterator[SessionHeader]`, `iter_records(session_id: str) -> Iterator[SourceRecord]`, and `iter_session_scoped_records(session_id: str) -> Iterator[SourceRecord]` over the normalized replay tables.
- `SourceExporter` must expose `schema_generation`, `inspect() -> SchemaDescription`, and `export_to_snapshot(writer: SnapshotWriter, redaction: RedactionCallback) -> ExportStats` for live-schema SQL ownership. `SourceExporter` is never reused for slow discovery/replay.
- `connection.py` produces `LiveReadOnlyConnection` and `open_live_connection(path: Path, busy_timeout_ms: int = 5000) -> ContextManager[LiveReadOnlyConnection]`.
- `snapshot.py` produces `RedactionCallback = Callable[[object, Collection[str]], object]`, `ExportStats(session_count: int, record_count: int, warning_count: int, complete: bool, schema_generation: SchemaGeneration)`, `SourceSnapshot`, `SnapshotWriter`, and `create_normalized_snapshot(path: Path, temp_root: Path, exporter: SourceExporter, redaction: RedactionCallback, busy_timeout_ms: int = 5000, write_batch: int = 256) -> ContextManager[SourceSnapshot]`. `SnapshotWriter.write_session(header: SessionHeader, canonical_session_row: CanonicalRecordEnvelope)`, `write_record(record: SourceRecord, order_key: tuple, canonical_record: CanonicalRecordEnvelope, raw_permitted_fields: Mapping[str, object])`, and `finalize() -> ExportStats` are the only temporary-snapshot writes. `SourceSnapshot` contains the non-secret source fingerprint, normalized temporary snapshot path, source schema description, and a read-only replay connection; no live source connection remains after context setup. The writer creates only `snapshot_sessions(session_id, schema, canonical_session_row)` and `snapshot_records(session_id, order_key, source_table, source_id, canonical_record, raw_permitted_fields)` in the temporary database.
- `replay.py` owns the `ReplayTurn` protocol: it exposes `session_header`, `summary`, fresh `iter_records()`, fresh `iter_session_records()`, fresh `iter_unassigned_records()`, `load_root_input(max_bytes: int)`, and `load_root_output(max_bytes: int)`, so no module creates an in-memory turn/list as a cross-layer substitute. It also produces `NormalizedSnapshotAdapter(snapshot: SourceSnapshot) -> SourceAdapter` over `snapshot_sessions`/`snapshot_records`.
- `probe.py` produces `probe_schema(connection: sqlite3.Connection, catalogs: Sequence[SchemaCatalog] = OPENCODE_SCHEMA_CATALOG_V1) -> SchemaDescription` and reports the exact catalog/version or safe missing-table/column diagnostic. Exporter/replay factories are registered and exposed by Task 4 only after their implementations exist.

- [ ] **Step 1: Add model and catalog tests for source invariants.** Verify timezone-aware UTC values retain original numeric value/unit, missing values remain absent, source IDs/sequence/raw values are distinct from decoded values, schema generations are limited to legacy/current/mixed, source type ranks are catalog data rather than query order, and the source database fingerprint hashes only schema generation plus available non-secret filesystem identity/statistics while excluding the absolute path and source content. Assert the v1 catalog contains these concrete structural columns and joins:

  - Legacy: `session(id, project_id, workspace_id, parent_id, slug, directory, path, title, version, share_url, summary_additions, summary_deletions, summary_files, summary_diffs, metadata, cost, tokens_input, tokens_output, tokens_reasoning, tokens_cache_read, tokens_cache_write, revert, permission, agent, model, time_created, time_updated, time_compacting, time_archived)`; `message(id, session_id, time_created, time_updated, data)`; and `part(id, message_id, session_id, time_created, time_updated, data)`, with `message.session_id = session.id` and `part.message_id = message.id`/`part.session_id = session.id`.
  - Current: `session_v2(id, project_id, workspace_id, parent_id, fork_session_id, fork_boundary, slug, directory, path, title, version, share_url, summary_additions, summary_deletions, summary_files, summary_diffs, metadata, cost, tokens_input, tokens_output, tokens_reasoning, tokens_cache_read, tokens_cache_write, revert, permission, agent, model, time_created, time_updated, time_idle, time_viewed, idle_outcome, time_compacting, time_archived, time_suspended, resume_attempts)` and `session_message(id, session_id, type, seq, time_created, time_updated, data)`, with `session_message.session_id = session_v2.id` and unique `(session_id, seq)`.
  - Both profiles record OpenCode `Timestamps` as epoch milliseconds in catalog v1, while the timestamp normalizer unit-tests seconds/milliseconds/microseconds/nanoseconds and rejects unknown units. JSON locations are `data`, session `model`, session `metadata`, summary diffs, and the nested current `session_message.data` content/model fields. User classification is legacy `message.data.role == user` plus non-synthetic/non-ignored parts and current `session_message.type == user` excluding synthetic content.

  The catalog also names the permitted session-scoped tables/columns: legacy `todo(session_id, content, status, priority, position, time_created, time_updated)`; current `session_pending(id, session_id, type, data, delivery, admitted_seq, time_created)`, `session_inbox(id, session_id, type, payload, delivery, enqueued_seq, time_created)`, `instruction_entry(session_id, key, value, removed, time_created, time_updated)`, `instruction_blob(hash, value)`, and `instruction_state(session_id, epoch_start, through_seq, initial_values, current_values)`. It records the unambiguous session joins for each and explicitly excludes all values from `account`, `account_state`, `control_account`, `credential`, and every `session_share` column until a reviewed safe-column catalog entry exists; an unknown `session_share` layout is omitted with a warning.

```python
def test_source_record_retains_raw_value_when_json_is_malformed():
    record = SourceRecord.from_row(
        source_table="part",
        source_type="text",
        source_id="part-1",
        raw_json="{not-json",
    )

    assert record.decoded_value is None
    assert record.raw_json == "{not-json"
    assert record.warnings == ("malformed_json",)
```

- [ ] **Step 2: Run model tests and verify they fail.**

Run: `python -m pytest tests/unit/source/test_model.py -q`

Expected: FAIL because the canonical model has not been defined.

- [ ] **Step 3: Implement the WAL-aware normalized snapshot extraction.**

Validate that the path is a regular readable SQLite file, ensure its `-wal` and `-shm` companions remain available, construct a URI containing only the normalized path and `mode=ro`, open with `uri=True`, set a bounded busy timeout and `PRAGMA query_only=ON`, and issue exactly one deferred `BEGIN`. Accept an injected `SourceExporter` contract in this task’s snapshot tests rather than instantiating the real legacy/current classes before Task 4; the exporter selects only permitted columns/records, applies the injected redaction callback before temporary persistence when enabled, and batch-writes canonical rows into a mode-`0o600` normalized snapshot containing only `snapshot_sessions` and `snapshot_records` inside a mode-`0o700` temporary directory. Close the live source connection immediately after safe extraction. Open the normalized snapshot through a read-only URI, set `query_only`, begin one replay transaction, and keep only that temporary connection open through discovery/replay and MLflow writes. Delete snapshot files and the directory in an outer `finally`/process-exit cleanup handler; report cleanup failures without paths/content. Do not include `immutable=1`, and never use the live connection for replay.

- [ ] **Step 4: Implement the concrete schema catalog and probe.**

Read only `sqlite_master` metadata and required column names. Validate `OPENCODE_SCHEMA_CATALOG_V1` table/column sets, join rules, timestamp units, JSON paths, classification rules, session-scoped allowlists, and sequence domains. Classify current when `session_v2` plus `session_message` match, legacy when `session`, `message`, and `part` match, mixed when both match, and unsupported/malformed otherwise. Return a safe compatibility diagnostic naming the missing catalog structure/column, never raw SQL values or source content. Write the same catalog matrix to `docs/schema-catalog-v1.md` so fixture DDL, SQL joins, and adapter tests have one reviewable source of truth.

- [ ] **Step 5: Build sanitized current/legacy/mixed/edge-case SQL fixtures.**

The fixture builder may create and populate temporary databases, enable WAL before inserting data, and close the writer only after the committed-WAL rows needed by the test exist. The committed `.sql` files must contain only synthetic IDs and non-sensitive prompts. Include malformed/unsupported schemas, BLOB/invalid-UTF-8/NULL/duplicate-key values, and optional/unknown session columns. The WAL test must prove that rows committed before normalized extraction are present, rows committed after extraction are absent, discovery/replay use the same normalized snapshot, source/WAL/SHM hashes and metadata are unchanged, and no importer code issues a source write/checkpoint. Assert the temporary snapshot contains no prohibited secret-table row/value.

- [ ] **Step 6: Add source snapshot failure coverage and commit.**

Run: `python -m pytest tests/unit/source tests/integration/test_wal_snapshot.py -m 'unit or integration' -q`

Expected: unit tests PASS; the WAL test is marked integration and runs only when explicitly enabled. Also assert disk-space/extraction/busy-timeout failure maps to `source_snapshot_failed`, exit `2`, and zero MLflow mutations; cleanup failures are reported without source paths/content. Commit as `feat: add read-only sqlite snapshot and schema probe`.

### Task 4: Add legacy/current adapters and canonical source normalization

**Files:**

- Create: `src/opencode_mlflow/source/legacy.py`
- Create: `src/opencode_mlflow/source/current.py`
- Create: `tests/unit/source/test_legacy.py`
- Create: `tests/unit/source/test_current.py`
- Create: `tests/contract/test_source_adapter_contract.py`
- Extend: `tests/fixtures/sqlite/legacy.sql`
- Extend: `tests/fixtures/sqlite/current.sql`
- Extend: `tests/fixtures/sqlite/mixed.sql`

**Interfaces:**

- `LegacyAdapter(connection: sqlite3.Connection, catalog: SchemaCatalog)` implements `SourceExporter` for `session`/`message`/`part`.
- `CurrentAdapter(connection: sqlite3.Connection, catalog: SchemaCatalog)` implements `SourceExporter` for `session_v2`/`session_message` and its catalogued associated records.
- `MixedAdapter(current: CurrentAdapter, legacy: LegacyAdapter)` exports all current sessions and legacy-only IDs, records suppressed overlap provenance, and never merges duplicate IDs.
- `NormalizedSnapshotAdapter(snapshot: SourceSnapshot)` implements `SourceAdapter` over the two normalized snapshot tables; it does not issue source-schema SQL or reconnect to the live database.
- `AdapterRegistry` maps `SchemaGeneration` to live exporter factories and maps normalized snapshots to the replay adapter; `build_live_exporter(connection: sqlite3.Connection, description: SchemaDescription, registry: AdapterRegistry) -> SourceExporter` and `build_replay_adapter(snapshot: SourceSnapshot, registry: AdapterRegistry) -> SourceAdapter` are implemented only after the concrete adapters are registered, so schema probing never calls a class that does not yet exist.
- Each adapter must expose catalog version, source-type rank table, timestamp-unit rules, safe session-scoped columns, and user-record classification through `inspect()`.

- [ ] **Step 1: Write shared adapter contract tests.** Apply the same parametrized assertions to legacy, current, and mixed adapters: stable session ordering, parameterized filtering by session ID, normalized headers, normalized records, raw/malformed preservation, model discovery inputs, one canonical row per source identity, parent-before-child ordering, and no secret-bearing values.

```python
@pytest.mark.parametrize("schema_name", ["legacy", "current", "mixed"])
def test_adapter_contract_emits_canonical_records(schema_name, fixture_database):
    from opencode_mlflow.source.ordering import session_sort_key

    adapter = fixture_database(schema_name).adapter
    headers = list(adapter.iter_session_headers())

    assert headers == sorted(headers, key=session_sort_key)
    for header in headers:
        for record in adapter.iter_records(header.session_id):
            assert record.session_id == header.session_id
            assert "credential-sentinel" not in repr(record)
```

- [ ] **Step 2: Run contract tests before adapter code.**

Run: `python -m pytest tests/contract/test_source_adapter_contract.py -q`

Expected: FAIL because the adapters and fixture factory do not yet satisfy the protocol.

- [ ] **Step 3: Implement the legacy exporter and cross-table merge.**

Use only catalogued, parameterized joins between `session`, `message`, and `part` while exporting from the live transaction. Emit one canonical message row into `SnapshotWriter`, then its parts immediately after it using part sequence/timestamp/ID; never emit a joined parent once per child or a part before its parent. For rows with an explicit ID, deduplicate only `(schema_generation, source_table, source_id)` projections. For a missing ID, construct `source_identity` from schema/table/session/parent, explicit sequence or timestamp, canonical row digest, and an occurrence ordinal; construct `join_projection_key` from the join path and parent/child occurrence, so repeated join projections collapse while distinct no-ID rows, including identical values in different occurrences, remain distinct. Normalize message and part rows to `SourceRecord` while retaining raw JSON/text, unknown session-row columns, and original timestamp values. Collect model identities from session defaults, messages, and parts. Classify persisted non-synthetic user prompts explicitly; classify system, synthetic, tool-only, lifecycle, compaction, and unknown rows without turning them into user prompts.

- [ ] **Step 4: Implement the current exporter and authoritative sequence domain.**

Read `session_v2` and `session_message` according to the current catalog while exporting from the live transaction, including `seq`, current JSON locations, lifecycle values, and nested model fields. Use `(session_id, seq)` as the authoritative sequence domain; nested content items receive derived IDs `<session_message_id>#content-<index>` and a child ordinal immediately after the parent, and nested user content is never another turn start. Normalize current records to the same `SourceRecord` fields and use the catalogued safe-column allowlist for session-scoped rows.

- [ ] **Step 5: Implement mixed precedence, unknown-column preservation, and secret-table exclusion.**

For a mixed database, compare only session IDs from the two session tables, select current for overlaps, and retain a warning/provenance record. The database description may be `mixed`, but each emitted `SessionHeader.adapter_generation` and owned trace `opencode.source_schema` is `current` or `legacy`. Preserve the complete permitted session-row map, including additive unknown columns, alongside typed fields. Never select values from `account`, `account_state`, `control_account`, or `credential`; omit secret-bearing `session_share` fields unless a catalogued safe-column allowlist identifies them as safe. Unassociated event/todo/instruction records must be reported and not assigned.

Use the same missing-ID identity helper for current rows and session-scoped records; never use `None` alone as a deduplication key. Include the fallback identity and occurrence ordinal in the protected canonical envelope so turn boundaries and replay are reproducible.

- [ ] **Step 6: Register factories and implement the adapter builder.**

Register `LegacyAdapter`, `CurrentAdapter`, and `MixedAdapter` exporter factories in `AdapterRegistry` after their classes exist, and register `NormalizedSnapshotAdapter` as the replay adapter. `build_live_exporter` must select the exporter from the probed schema; `build_replay_adapter` must select the normalized snapshot adapter. Reject an absent registration with a safe internal compatibility error and never fall back to heuristic SQL.

- [ ] **Step 7: Run adapter tests and commit.**

Run: `python -m pytest tests/unit/source/test_legacy.py tests/unit/source/test_current.py tests/contract/test_source_adapter_contract.py -m 'unit or contract' -q`

Expected: PASS, including mixed overlap, unknown/malformed JSON with `unknown_source_field`/`malformed_source_value` warnings, model-at-part-level, safe-table exclusion, unassociated-record warnings, and identical selection/turn counts when secret-table rows are added or removed. Commit as `feat: normalize legacy and current opencode schemas`.

### Task 4A: Establish canonicalization and fingerprint primitives before selection discovery

**Files:**

- Create: `src/opencode_mlflow/content/canonical.py`
- Create: `src/opencode_mlflow/content/fingerprint.py`
- Create: `tests/unit/content/test_canonical.py`
- Create: `tests/unit/content/test_fingerprint.py`
- Create: `tests/fixtures/payloads/canonical-json-v1-vectors.json`

**Interfaces:**

- `canonical.py` produces `canonical_json(value: object) -> bytes`, `canonical_record_envelope(value: object, raw_text: str | None, raw_columns: Mapping[str, object]) -> CanonicalRecordEnvelope`, and `canonical_stream(chunks: Iterable[object], redactor: Redactor) -> Iterator[bytes]`.
- `fingerprint.py` consumes the `ReplayTurn` protocol owned exclusively by `source/replay.py` and produces `EffectiveTurn`, `build_effective_turn(replay_turn: ReplayTurn, redactor: Redactor) -> EffectiveTurn`, `session_context_fingerprint(turn: EffectiveTurn) -> str`, `source_fingerprint(turn: EffectiveTurn, redaction_mode: RedactionMode) -> str`, `render_fingerprint(source_fp: str, artifact_version: str, mapping_version: str, importer_version: str, redaction_mode: RedactionMode) -> str`, and `artifact_snapshot_fingerprint(turn: ReplayTurn, redaction_mode: RedactionMode) -> str`.
- `EffectiveTurn` is a streaming view containing the immutable session-context fields, fresh cursors for turn/session-scoped/unassigned canonical records, counts, adapter/schema version, and redaction mode. It does not contain a list of records or generated MLflow IDs. Its producer is `build_effective_turn`; Tasks 5–6 consume that exact seam. No other module defines or re-exports a second `ReplayTurn` protocol.

- [ ] **Step 1: Write canonical golden-vector and fingerprint tests before selection code.** Cover all `canonical-json-v1` scalar/wrapper cases, effective redaction callback injection, session-context versus mutable-session fingerprint boundaries, fixed `identity_map_pending`, turn-local changes, redacted-secret equivalence, artifact-snapshot completeness, and deterministic render fingerprints.

- [ ] **Step 2: Run the foundation tests to verify they fail.**

Run: `python -m pytest tests/unit/content/test_canonical.py tests/unit/content/test_fingerprint.py -q`

Expected: FAIL because the canonical and fingerprint primitives do not exist.

- [ ] **Step 3: Implement canonical-json-v1 and streaming fingerprint primitives.**

Implement the typed wrappers, UTF-8 byte key order, number/NULL/absent/duplicate-key behavior, redactor callback seam, `session_context_fingerprint`, turn `source_fingerprint`, render fingerprint, and pre-identity-map artifact-snapshot hash without retaining complete sessions/turns in memory.

- [ ] **Step 4: Run the foundation tests and commit.**

Run: `python -m pytest tests/unit/content/test_canonical.py tests/unit/content/test_fingerprint.py -m unit -q`

Expected: PASS. Commit as `feat: establish canonical and fingerprint contracts`.

### Task 5: Implement deterministic ordering and logical-turn reconstruction

**Dependency:** Task 4A supplies `ReplayTurn`, `build_effective_turn`, canonical record serialization, and fingerprint functions; this task must not reimplement canonicalization or hash raw Python/SQLite driver values.

**Files:**

- Create: `src/opencode_mlflow/source/ordering.py`
- Create: `src/opencode_mlflow/source/turns.py`
- Create: `tests/unit/source/test_ordering.py`
- Create: `tests/unit/source/test_turns.py`
- Extend: `tests/bdd/features/source_safety.feature`

**Interfaces:**

- `ordering.py` produces `RecordOrderContext(sequence_domain: str | None, authoritative: bool)`, `record_sort_key(record: SourceRecord, context: RecordOrderContext) -> tuple`, `session_sort_key(header: SessionHeader) -> tuple`, and the fixed adapter/catalog source-type rank table `10/20/30/40/90`.
- `turns.py` produces `turn_key(session_id: str, first_user: SourceRecord, turn_index: int) -> str`, `iter_turn_summaries(records: Iterable[SourceRecord]) -> Iterator[TurnSummary]`, `iter_turn_records(records: Iterable[SourceRecord], summary: TurnSummary) -> Iterator[SourceRecord]`, and `build_replay_turn(snapshot: SourceSnapshot, header: SessionHeader, summary: TurnSummary) -> ReplayTurn`. `ReplayTurn` is the protocol owned only by `source/replay.py`; this builder returns its concrete implementation with fresh normalized-snapshot cursors and bounded root-value loaders.
- `TurnSummary` contains 1-based `turn_index`, stable `turn_key`, first-user/start and last-record/end timestamps, source counts, terminal outcome, root-result source identity, and turn-local source fingerprint; it never stores prompt/response content in the discovery plan.

- [ ] **Step 1: Write ordering and boundary tests.** Cover authoritative current `session_message` sequence ordering, legacy message/part parent-child ordering, timestamp/rank/ID ordering across streams without a shared sequence domain, missing timestamp sorting after timestamped sessions, consecutive user prompts, records before the first prompt, no-user sessions, tool-only/system/synthetic/lifecycle non-start records, fork-local rows, compaction, missing source IDs, two distinct missing-ID records with identical values remaining distinct by occurrence ordinal, and repeated join projections collapsing only when their join projection keys match.

```python
def test_consecutive_user_prompts_close_previous_turn_without_synthetic_output(records):
    summaries = list(iter_turn_summaries(records))

    assert [summary.turn_index for summary in summaries] == [1, 2]
    assert summaries[0].root_result_source_id is None
    assert summaries[0].terminal_outcome == "no_assistant_result"
    assert summaries[1].first_user_source_id == "user-2"
```

- [ ] **Step 2: Run the turn tests to verify they fail.**

Run: `python -m pytest tests/unit/source/test_ordering.py tests/unit/source/test_turns.py -q`

Expected: FAIL because ordering and turn reconstruction are not implemented.

- [ ] **Step 3: Implement deterministic ordering and timestamp fallback rules.**

Within one authoritative sequence domain, order by `(sequence, fixed_tie_rank, stable_source_id)`; do not compare sequence values from unrelated streams. Across streams without a shared sequence domain, order by `(normalized_timestamp_present/value, fixed_tie_rank, stable_source_id)`. For legacy rows, attach each part immediately after its parent using parent order plus part sequence/timestamp/ID. Normalize session keys to `(creation_time_present/value, session_id)`. Record original units and mark unknown/invalid timestamps in warnings; when a time filter is present, make the session ineligible rather than guessing from a message timestamp.

- [ ] **Step 4: Implement stable turn keys and streaming summaries.**

Use `turn-<sha256(session_id + "\\0" + first_user_source_type + "\\0" + first_user_source_id)>` for normal IDs. For missing IDs, hash the ordered tuple `(session_id, source_type, explicit_sequence_or_normalized_timestamp, canonical_effective_record, turn_index)` with NUL separators, omitting absent tuple values only after recording the fallback strategy in the artifact. If two turns still collide, add the stable occurrence ordinal to the hash input and emit `turn_key_collision`; never merge turns. Exact duplicate fallback records use their ordered occurrence index. Only persisted non-synthetic user prompts start turns. Select the last non-synthetic assistant result for root-result identity, preserve earlier results as child candidates, and represent no-result turns with `no_assistant_result` and `UNSET` unless an explicit successful terminal outcome exists.

- [ ] **Step 5: Keep reconstruction bounded.**

Make summary iteration consume records in adapter-provided source order and release each completed turn’s records before advancing. The write phase must replay records from the same temporary SQLite snapshot rather than retaining all session content. Add assertions that a discovery summary contains identifiers, counts, warnings, and hashes but not prompt/response/tool/source-code fields.

- [ ] **Step 6: Run tests and commit.**

Run: `python -m pytest tests/unit/source/test_ordering.py tests/unit/source/test_turns.py -q`

Expected: PASS. Commit as `feat: reconstruct deterministic opencode turns`.

### Task 6: Add filtering and the sanitized disk-backed discovery plan

**Dependency:** Task 4A supplies the `EffectiveTurn`/fingerprint seam. Discovery writes only the resulting hashes and identities to its plan; it never invents a second canonicalization or hashing format.

**Files:**

- Create: `src/opencode_mlflow/source/selection.py`
- Create: `src/opencode_mlflow/source/discovery.py`
- Create: `tests/unit/source/test_selection.py`
- Create: `tests/unit/source/test_discovery.py`
- Extend: `tests/bdd/features/cli_and_selection.feature`

**Interfaces:**

- `selection.py` produces frozen `SelectionFilters`, `SessionDiscoverySummary`, `matches_directory(candidate: Path, requested: Path) -> bool`, `matches_model(available: Collection[str], requested: str | None) -> bool`, and `select_session(summary: SessionDiscoverySummary, filters: SelectionFilters) -> bool`.
- `discovery.py` produces `SessionScanEntry`, `DiscoveryTurnEntry`, `DiscoveryPlanWriter`, `DiscoveryPlanReader`, and `discover_sessions(adapter: SourceAdapter, filters: SelectionFilters, plan_path: Path) -> DiscoveryStats`.
- `SessionScanEntry` is written for every compatible session, including a selected session with zero logical turns; it records `selected`, `complete_scan`, `logical_turn_count`, session identity/header fingerprint, warnings, and reconciliation-scope metadata. `DiscoveryTurnEntry` contains source/session/turn identities, normalized timestamps, counts, model identities, warnings, schema/fingerprint/render-contract values, and planned action metadata. Neither entry contains transcript or source-code content.

- [ ] **Step 1: Write selection tests for every acceptance boundary.** Test inclusive `from_time`/`to_time`, use of session creation time only, missing creation time with/without time filters, exact directory and component descendants, sibling-prefix exclusion, complete/incomplete model matching, model discovery from each source level, strict `N` versus `N+1` turns, and AND semantics.

```python
def test_selection_requires_every_supplied_filter(summary):
    filters = SelectionFilters(
        from_time=datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc),
        directory=Path("/repo/project"),
        model="provider/model",
        min_user_turns=2,
    )

    assert select_session(summary, filters) is True
    assert select_session(replace(summary, model_identities=frozenset({"other/model"})), filters) is False
```

- [ ] **Step 2: Run selection tests and verify the expected failures.**

Run: `python -m pytest tests/unit/source/test_selection.py -q`

Expected: FAIL because selection functions and discovery summaries do not exist.

- [ ] **Step 3: Implement normalized selection.**

Use path-component equality/descendant logic, not string prefixes. Compare model identities exactly after source/request normalization, including explicit `provider/` and `/model` incomplete forms. Treat `min_user_turns` as a session eligibility threshold and import every turn from an eligible session.

- [ ] **Step 4: Implement discovery as a streaming pass with complete-scan tombstones.**

Require `iter_session_headers()` to emit normalized session order and assert monotonic `(creation_time, session_id)` keys while consuming it; do not materialize all headers merely to sort them. For each compatible session, build a `ReplayTurn`, pass it through Task 4A’s `build_effective_turn`, stream model identities/source counts/warnings/complete-scan status/fingerprints, and append one sanitized `SessionScanEntry` plus one `DiscoveryTurnEntry` per logical turn. A selected complete scan with zero turns is an explicit tombstone scope that permits retirement of all previously owned turns for that session. Do not put content-bearing decoded/raw fields in the plan.

- [ ] **Step 5: Add plan replay checks.**

Make `DiscoveryPlanReader` reject malformed entries, unknown plan schema versions, duplicate `(session_id, turn_key)` identities, missing fingerprints, a turn entry without a matching session scan, and a selected complete scan that is missing its zero-turn tombstone. The write phase will later compare replayed fingerprints against these entries and treat mismatches as source-consistency failures.

- [ ] **Step 6: Add BDD selection scenarios and commit.**

```gherkin
Feature: Select sessions safely
  Scenario: Include exact time boundaries and descendants
    Given a session was created at "2026-10-21T12:00:00Z" in "/repo/project/app"
    When I filter from "2026-10-21T12:00:00Z" to "2026-10-21T12:00:00Z" and directory "/repo/project"
    Then the session is selected
    And a session in "/repo/projector" is not selected

  Scenario: Apply a strict turn threshold
    Given a session has 2 logical user turns
    When I select sessions with minimum user turns 2
    Then the session is not selected
    When I select sessions with minimum user turns 1
    Then the session is selected with all 2 turns
```

Run: `python -m pytest tests/unit/source/test_selection.py tests/unit/source/test_discovery.py tests/bdd -m 'unit or bdd' -q`

Expected: PASS. Commit as `feat: add streaming session discovery and filters`.

### Task 7: Implement deterministic redaction and protected-identity handling

**Files:**

- Create: `src/opencode_mlflow/content/redaction.py`
- Create: `tests/unit/content/test_redaction.py`
- Create: `tests/fixtures/payloads/redaction-positive.json`
- Create: `tests/fixtures/payloads/redaction-negative.json`
- Extend: `tests/bdd/features/privacy_and_reporting.feature`

**Interfaces:**

- `redaction.py` consumes/re-exports shared `RedactionMode` and `Redactor`, and produces `RedactionReport`, `RedactionResult`, compiled detector constants, `redact_text(text: str) -> RedactionResult`, `redact_value(value: object, protected_paths: Collection[str] = ()) -> RedactionResult`, and `redact_record(record: SourceRecord) -> RedactionResult`.
- `RedactionReport` contains detector-category counts and affected field/record counts only; it never stores a matched secret.
- Protected envelope fields retain source/session/message/part IDs, sequence values, turn keys, schema names, and reconciliation identifiers even when raw record text is redacted.

- [ ] **Step 1: Add positive and negative detector fixtures.** Include PEM private-key blocks, `Authorization`/`Proxy-Authorization` bearer/basic headers, standalone bearer values, structured auth fields, every version-one secret-key spelling (`api[-_]key`, `api[-_]token`, `access[-_]token`, `refresh[-_]token`, `auth[-_]token`, `session[-_]token`, `token`, `access[-_]key`, `client[-_]secret`, `secret`, `credential`, `password`, `passwd`, `private[-_]key`, `bearer`), connection-string passwords, `sk-`, `ghp_`, `github_pat_`, `xox[baprs]-`, and AWS `AKIA` forms. Include near misses, short tokens, non-secret identifiers, delimiter-preservation cases, and opaque BLOB/invalid-byte values.

- [ ] **Step 2: Write detector and protection tests.** Assert exact replacement marker `[REDACTED]`, recursive structured masking with preserved field names, detector precedence for structured values, preserved Bearer/Basic scheme where possible, protected IDs, no matched text in the report, and no-op behavior in verbatim mode.

```python
def test_redaction_masks_secret_but_preserves_reconciliation_identity():
    result = redact_value(
        {"secret": "sk-1234567890123456", "part_id": "part-7"},
        protected_paths={"part_id"},
    )

    assert result.value == {"secret": "[REDACTED]", "part_id": "part-7"}
    assert result.report.detector_counts["secret_key_value"] == 1
    assert "sk-1234567890123456" not in repr(result.report)
```

- [ ] **Step 3: Run redaction tests to verify they fail.**

Run: `python -m pytest tests/unit/content/test_redaction.py -q`

Expected: FAIL because detector constants and recursive redaction are not implemented.

- [ ] **Step 4: Implement the named version-one detectors and fixed transformation pipeline.**

Compile the exact detector semantics from the design using Python regex flags rather than relying on incidental inline-flag placement. Decode while retaining raw representation, recursively redact structured mappings before textual matching, copy protected identifiers unchanged, replace opaque BLOB/invalid-byte values with `[REDACTED_BINARY]` unless protected, serialize valid JSON only after redaction, scan malformed/unknown text with delimiter-aware assignment rules, and emit only the redacted decoded/raw envelope. Never retain original raw text in a redacted envelope or temporary upload. Record only detector categories and affected field/record counts.

- [ ] **Step 5: Add privacy BDD scenarios and commit.**

```gherkin
Feature: Redact sensitive content consistently
  Scenario: Redact trace content and artifact content without changing identity
    Given a turn contains an authorization bearer token and source ID "part-7"
    When I import with redaction enabled
    Then the trace and artifact contain "[REDACTED]"
    And neither contains the original bearer token
    And the source ID remains "part-7"
    And the report contains detector counts but not the original token
```

Run: `python -m pytest tests/unit/content/test_redaction.py tests/bdd -m 'unit or bdd' -q`

Expected: PASS. Commit as `feat: add deterministic transcript redaction`.

### Task 8: Integrate canonical fingerprints and add streamed artifacts

**Files:**

- Complete: `src/opencode_mlflow/content/canonical.py`
- Complete: `src/opencode_mlflow/content/fingerprint.py`
- Create: `src/opencode_mlflow/content/artifacts.py`
- Extend: `tests/unit/content/test_canonical.py`
- Extend: `tests/unit/content/test_fingerprint.py`
- Create: `tests/unit/content/test_artifacts.py`
- Complete: `tests/fixtures/payloads/canonical-json-v1-vectors.json`
- Create: `tests/integration/test_scale_memory.py`

**Interfaces:**

- `canonical.py` consumes/re-exports shared `CanonicalRecordEnvelope` and produces `canonical_json(value: object) -> bytes`; serialization implements the exact `canonical-json-v1` contract: UTF-8 byte key ordering, original Unicode code points with JSON escaping, fixed separators, RFC 8785 shortest-round-trip finite numbers with `-0` normalized to `0`, typed wrappers for non-finite values, SQLite BLOBs and invalid UTF-8, JSON `null` for SQLite NULL, omission for absent columns, epoch nanoseconds for normalized instants in fingerprint material, and typed ordered-pairs wrappers plus raw text for duplicate JSON keys.
- `fingerprint.py` produces `source_fingerprint(turn: EffectiveTurn, redaction_mode: RedactionMode) -> str`, `render_fingerprint(source_fp: str, artifact_version: str, mapping_version: str, importer_version: str, redaction_mode: RedactionMode) -> str`, and `artifact_snapshot_fingerprint(turn: ReplayTurn, redaction_mode: RedactionMode) -> str`. The artifact-snapshot function streams the complete canonical session payload with the fixed `identity_map_pending` placeholder; it never hashes the post-create generated identity map. `EffectiveTurn` is the one-pass canonical/redacted view used for turn-local hashing; `ReplayTurn` is a replayable header/summary plus reopenable normalized-snapshot cursors, never an in-memory record list.
- `artifacts.py` produces `ArtifactManifest`, `ArtifactUploadChunk(size: int, digest: str, payload: bytes)`, `artifact_relative_path(session_id: str, turn_key: str, render_fp: str, snapshot_fp: str) -> str`, and `stream_turn_artifact(turn: ReplayTurn, destination: BinaryIO, identity_map: Mapping[str, str], redaction: RedactionMode) -> ArtifactManifest`. The artifact writer exposes fixed-size chunk iteration and opens a fresh normalized-snapshot cursor for each pass so the gateway never materializes a multi-hundred-megabyte payload.

- [ ] **Step 1: Extend the canonical foundation tests for artifact integration.** Reuse the Task 4A golden vectors and assert artifact serialization uses the same `canonical-json-v1` bytes, raw/malformed envelopes, fixed `identity_map_pending` placeholder, importer/adapter/artifact/mapping versions, and generated-ID exclusion. Store no second canonical format or vector set.

- [ ] **Step 2: Write fingerprint tests.** Verify turn-local source changes replace only that turn, session identity/selection-context changes replace all affected turns, mutable session snapshot changes (`time_updated`, lifecycle/archival state, cost/token totals, summaries, turn counts, and other-turn counts) do not change an unchanged turn’s source fingerprint, the complete artifact-snapshot fingerprint changes for those mutable fields, changing only the original value of a detected secret does not change a redacted source/render fingerprint, switching verbatim versus redacted mode does change the render fingerprint, effective redacted values feed the source fingerprint, generated IDs are replaced by a fixed `identity_map_pending` placeholder in every fingerprint input, and source schema is compared separately during reconciliation.

```python
def test_render_fingerprint_changes_when_redaction_mode_changes():
    from opencode_mlflow.content.redaction import RedactionMode

    verbatim = render_fingerprint("a" * 64, "artifact-v1", "mapping-v1", "0.1.0", RedactionMode.VERBATIM)
    redacted = render_fingerprint("a" * 64, "artifact-v1", "mapping-v1", "0.1.0", RedactionMode.REDACTED)

    assert verbatim != redacted
    assert len(verbatim) == 64
    assert len(redacted) == 64
```

- [ ] **Step 3: Run artifact tests and verify the expected failures.**

Run: `python -m pytest tests/unit/content/test_artifacts.py -q`

Expected: FAIL because the streamed artifact writer does not exist.

- [ ] **Step 4: Implement streamed artifact paths and temporary-file policy.**

Use the Task 4A pre-identity-map artifact-snapshot fingerprint and `opencode-mlflow/v1/sessions/<encoded-session-id>/turns/<stable-turn-key>/<render-fingerprint>-<artifact-snapshot-fingerprint>.json`. Encode session IDs safely without exposing path traversal. Write with mode `0o600`, stream rather than materialize the full session snapshot, repeat the complete permitted session snapshot in each turn artifact, insert the generated identity map only after the placeholder hash/path are fixed, emit fixed upload chunks, and delete temporary files after upload/verification on a best-effort basis. If trace metadata needs the final artifact path, set it through a documented public trace-update operation and include that update in the capability contract.

- [ ] **Step 5: Add artifact privacy and payload-limit assertions.**

Assert that redacted mode never creates an unredacted upload candidate, artifacts contain raw/malformed permitted fields, excluded secret-table values never occur, large metadata is represented by digest/reference while the complete value remains in the artifact, complete root values above a reported MLflow inline limit fail with `payload_too_large` before trace creation, and child values above the limit remain complete in the artifact with only a digest/reference inline. Use a conservative 16 MiB limit when the target reports no limit.

- [ ] **Step 6: Add the bounded-memory stress test.**

Generate at least 100,000 synthetic records and a multi-hundred-megabyte artifact, stream discovery/canonicalization/upload through a fake chunk sink, and measure Python heap growth from an empty-process baseline. Assert growth remains below 256 MiB, the source fingerprint is stable, every artifact byte is preserved, and no single in-memory object scales with the complete session.

- [ ] **Step 7: Run all content tests and commit.**

Run: `python -m pytest tests/unit/content -m unit -q`

Expected: PASS, including canonical golden vectors, redaction, payload limits, and bounded-memory unit doubles. Commit as `feat: add canonical fingerprints and streamed artifacts`.

### Task 9: Build the public MLflow capability probe and gateway boundary

**Files:**

- Create: `src/opencode_mlflow/mlflow/capabilities.py`
- Create: `src/opencode_mlflow/mlflow/gateway.py`
- Create: `tests/support/fake_gateway.py`
- Create: `tests/unit/mlflow/test_capabilities.py`
- Create: `tests/unit/mlflow/test_gateway.py`
- Create: `tests/contract/test_gateway_contract.py`
- Extend: `docs/compatibility.md`
- Extend: `tests/integration/constraints.txt`
- Extend: `tests/integration/README.md`

**Interfaces:**

- `capabilities.py` consumes the profile `CapabilityReport` and produces `CapabilityRequest`, `CapabilityReport`, `probe_capabilities(client: MlflowClient, request: CapabilityRequest, dry_run: bool) -> CapabilityReport`, and `CapabilityError` outcomes. A normal write always sets `needs_retirement=True`, regardless of the initial action plan, because rollback is required after any partial create; capability failure aborts the entire invocation before mutation.
- `gateway.py` produces `TraceChildRequest(source_key: str, name: str, span_type: str, parent_source_key: str | None, inputs: object | None, attributes: Mapping[str, object], tags: Mapping[str, str], start_time_ns: int | None, outputs: object | None, status: str, end_time_ns: int | None)`, `TraceCreateRequest(client_request_id: str, experiment_id: str, name: str, native_session_id: str, root_inputs: object, root_outputs: object | None, root_attributes: Mapping[str, object], root_tags: Mapping[str, str], metadata: Mapping[str, str], start_time_ns: int | None, end_time_ns: int | None, child_specs: Callable[[], Iterator[TraceChildRequest]], expected_operation_key: str, artifact_snapshot_fingerprint: str, artifact_relative_path: str)`, `PublicTraceApi` (the adapter for module-level `mlflow.start_span_no_context` and `LiveSpan.end`), `ExperimentRef`, `CreatedTrace`, `TraceArtifactSink` with `upload_named_json(trace_id: str, relative_path: str, local_file: Path, chunk_size: int = 8 * 1024 * 1024) -> ArtifactVerification` and `verify_named_json(trace_id: str, relative_path: str, sha256: str, size: int) -> None`, and `MlflowGateway` with `resolve_experiment(name: str, create: bool) -> ExperimentRef`, `probe`, `iter_owned_traces`, `create_trace(spec: TraceCreateRequest) -> CreatedTrace`, `verify_trace(created: CreatedTrace, expected: TraceCreateRequest)`, `upload_named_json`, `verify_named_json`, and `retire_trace(experiment_id: str, trace_id: str)` methods.
- The gateway accepts `MlflowClient` plus the injected `PublicTraceApi` through dependency injection so unit and contract tests can use fake public-client/LiveSpan-shaped objects. The gateway is the only module that calls MLflow experiment/search/delete APIs or starts/ends trace spans; mapping, source, content, and orchestration modules never import MLflow.
- `create_trace` starts the root with `mlflow.start_span_no_context`, keeps a bounded `source_key -> LiveSpan` parent map, consumes the lazy child-spec iterator one child at a time, resolves each `parent_source_key`, starts the child with the returned parent `LiveSpan`, ends children/root through `LiveSpan.end`, and returns `CreatedTrace(trace_id, root_span_id, span_ids_by_source, verified_metadata)`. It never collects the full child list or source payload in memory.

- [ ] **Step 1: Write gateway contract tests against the fake client.** Verify exact experiment lookup, create-on-missing only after preflight in normal mode, read-only capability probes in both normal and dry-run modes, no experiment/trace/artifact mutation during either probe, paginated owned-trace search, complete root payload verification rather than search previews, native session metadata, public retirement calls, and deterministic named artifact-path verification.

```python
def test_dry_run_capability_probe_does_not_mutate(fake_mlflow_client):
    report = probe_capabilities(
        fake_mlflow_client,
        CapabilityRequest(needs_retirement=True, needs_named_artifacts=True),
        dry_run=True,
    )

    assert report.is_compatible
    assert fake_mlflow_client.mutations == []
```

- [ ] **Step 2: Consume the completed compatibility profile.**

Read the exact `3.17.2.dev0` API table and disposable-server evidence from Task 0. The capability probe remains behavioral and authoritative; a numeric version alone cannot make an unsupported pair pass. Do not begin gateway writer work if Task 0 did not prove named artifact upload/verification, full-payload retrieval, and retirement.

- [ ] **Step 3: Run gateway tests to verify they fail.**

Run: `python -m pytest tests/unit/mlflow/test_capabilities.py tests/unit/mlflow/test_gateway.py tests/contract/test_gateway_contract.py -q`

Expected: FAIL because the public gateway and fake-client contract are not implemented.

- [ ] **Step 4: Implement experiment and non-mutating capability handling.**

Resolve experiment names exactly, reuse existing experiments without changing unrelated settings, create missing experiments only after the complete global capability check in normal mode, and represent planned creation in dry-run. Adapt only the public `mlflow.start_span_no_context`/`LiveSpan.end` trace API and `MlflowClient` lookup/search/delete APIs through the gateway. Runtime `probe_capabilities` must be read-only: inspect client methods, server reachability/protocol responses, the recorded compatibility profile, configured artifact capability, and reported payload limits without creating probe experiments/traces, uploading probe files, or deleting traces. If full-payload retrieval, named-artifact behavior, or retirement cannot be proven non-mutatively from the pinned profile/server contract, return global `mlflow_capability_unsupported` status `2`; the disposable mutation-based proof belongs only to Task 0’s compatibility test environment. Do not create a temporary experiment or upload a probe file during dry-run.

- [ ] **Step 5: Implement the behavioral artifact sink.**

Accept only the documented public API proven by Task 0 that preserves the requested relative path, accepts fixed chunks, and allows verification. Reject UUID-only attachment behavior and run-artifact fallback with a safe capability error. Keep the exact API-version difference handling inside the gateway and assert the profile method/field table in the gateway contract.

- [ ] **Step 6: Run contract tests and commit.**

Run: `python -m pytest tests/unit/mlflow tests/contract/test_gateway_contract.py -m 'unit or contract' -q`

Expected: PASS against the fake client. Commit as `feat: add mlflow capability and gateway boundary`.

### Task 10: Implement trace/span/event mapping

**Files:**

- Create: `src/opencode_mlflow/mlflow/mapping.py`
- Create: `tests/unit/mlflow/test_mapping.py`
- Create: `tests/contract/test_mapping_contract.py`
- Create: `tests/fixtures/six_turn_session.py`
- Extend: `tests/bdd/features/mapping_and_fidelity.feature`

**Interfaces:**

- `mapping.py` consumes the `ReplayTurn` protocol owned by `source/replay.py` and produces `TraceSpec` (implementing the exact Task 9 `TraceCreateRequest` fields), `SpanSpec`, `EventSpec`, `TraceStatus`, `SourceIdentity`, `MlflowIdentityMap`, `ArtifactIdentityEnvelope`, `map_turn_to_trace(turn: ReplayTurn, context: MappingContext) -> TraceSpec`, `map_source_record(record: SourceRecord, parent_source_key: str) -> SpanSpec | EventSpec`, and pure `bind_mlflow_identity_map(source_envelope: ArtifactIdentityEnvelope, trace_id: str, span_ids_by_source: Mapping[str, str]) -> MlflowIdentityMap`. `SpanSpec.to_child_request()` and `EventSpec.to_child_request()` convert to the Task 9 `TraceChildRequest` shape; events use `span_type="event"` because the public trace API has no separate event mutation. Mapping returns lazy child-span specs keyed by source identity; it never calls MLflow.
- `TraceSpec` contains `client_request_id`, complete root input/output within the negotiated inline limit, source-derived start/end times, native session ID, namespaced ownership/identity metadata, `opencode.operation_key`, a lazy `child_specs: Callable[[], Iterator[TraceChildRequest]]`, source-to-child identity keys, the pre-identity-map `artifact_snapshot_fingerprint` and deterministic `artifact_relative_path`, and expected verification values. `MlflowGateway.create_trace(spec: TraceCreateRequest) -> CreatedTrace` consumes `spec.client_request_id`; no separate caller argument can diverge from the operation contract. All ownership/operation metadata and the artifact path are present in the initial trace-create request and therefore survive a process crash before trace completion.
- `MappingContext` contains importer/mapping/artifact versions, source schema/database fingerprint, redaction mode, experiment ID, and deterministic request/idempotency key inputs. `operation_key` is SHA-256 of the exact experiment/session/turn/render tuple with versioned separators.

- [ ] **Step 1: Write pure mapping tests for root, child, ownership, streaming, and identity-map semantics.** Cover first user prompt as root input, final non-synthetic assistant result as root output, split final-result content parts combined in order, earlier assistant results as children, no-assistant output omission/null, model/provider spans, assistant spans, tool spans, content spans/events, lifecycle events, unknown opaque records, source IDs/sequences, tokens/costs, source aggregate totals, `opencode.render_fingerprint`, `opencode.operation_key`, lazy child iteration, bounded root-value loading, and binding generated trace/span IDs without changing source/render/artifact-snapshot fingerprints. Assert the mapping module imports no MLflow module or client.

```python
def test_mapping_preserves_six_turn_session_shape(six_turn_session):
    specs = [map_turn_to_trace(turn, six_turn_session.context) for turn in six_turn_session.turns]

    assert len(specs) == 6
    assert {spec.native_session_id for spec in specs} == {"ses-synthetic-1"}
    assert len({spec.client_request_id for spec in specs}) == 6
    assert all(spec.metadata["opencode.session_id"] == "ses-synthetic-1" for spec in specs)
```

- [ ] **Step 2: Run mapping tests to verify they fail.**

Run: `python -m pytest tests/unit/mlflow/test_mapping.py -q`

Expected: FAIL because trace/span mapping types and functions do not exist.

- [ ] **Step 3: Implement source-to-trace mapping.**

Map successful outcomes to `OK`, explicit failures to `ERROR`, and interrupted/incomplete outcomes to `UNSET` with `opencode.outcome`. Use first user timestamp, first record timestamp, or source session creation as the marked start fallback; use the last record/completion timestamp or source-derived start as the end fallback; omit both when all source timestamps are absent and the target allows it. Never use local wall-clock time.

- [ ] **Step 4: Implement ownership and root metadata.**

Populate exactly `opencode.importer`, `opencode.session_id`, `opencode.turn_index`, `opencode.turn_key`, `opencode.source_fingerprint`, `opencode.source_schema`, `opencode.importer_version`, `opencode.mapping_version`, `opencode.redaction_mode`, `opencode.render_fingerprint`, and `opencode.operation_key`, plus available session/source metadata. Omit absent native user/Git/model values rather than inventing them. Pass this complete metadata set in the initial trace-create call before ending the trace.

- [ ] **Step 5: Implement typed child mapping and gateway-owned span writing.**

Create LLM/model, assistant/agent, tool, content, lifecycle, and opaque child specs according to source type. Preserve source IDs/types/times in each spec and place oversized attributes behind an artifact reference while retaining complete content in the artifact. Before trace creation, fingerprint/serialize the canonical session payload with the fixed `identity_map_pending` placeholder. The gateway, not this mapping module, calls public `mlflow.start_span_no_context` for the root and each child, passing the root/parent `LiveSpan` object and ending each span with public `LiveSpan.end`; it records the returned trace/span IDs into `MlflowIdentityMap`. The gateway then invokes the pure identity binder and the artifact writer streams a fresh `ReplayTurn` cursor to the deterministic path calculated from the pre-identity-map artifact-snapshot fingerprint; generated IDs never feed source, render, operation, or artifact-snapshot fingerprints.

- [ ] **Step 6: Run mapping contract tests and commit.**

Run: `python -m pytest tests/unit/mlflow/test_mapping.py tests/contract/test_mapping_contract.py -m 'unit or contract' -q`

Expected: PASS. Commit as `feat: map opencode turns to mlflow traces`.

### Task 11: Add retries, advisory locking, ownership reconciliation, and safe replacement

**Files:**

- Create: `src/opencode_mlflow/retry.py`
- Create: `src/opencode_mlflow/locking.py`
- Create: `src/opencode_mlflow/mlflow/reconciliation.py`
- Create: `tests/unit/test_retry.py`
- Create: `tests/unit/test_locking.py`
- Create: `tests/unit/mlflow/test_reconciliation.py`
- Extend: `tests/support/fake_gateway.py`
- Extend: `tests/bdd/features/reconciliation.feature`

**Interfaces:**

- `retry.py` produces `RetryPolicy`, `is_transient_error(exc: BaseException) -> bool`, and `retry_call(operation, policy, sleep, random_source) -> T`; retry network/timeout/rate-limit/5xx failures with bounded exponential backoff and jitter, but do not indefinitely retry invalid requests or authorization failures.
- `locking.py` produces `ImportLock` and `acquire_import_lock(tracking_endpoint: str, experiment_name: str, lock_root: Path) -> ContextManager[ImportLock]`; lock names are derived from normalized endpoint and exact experiment name, contain no credentials, are non-blocking, and are skipped for dry-run. Use `XDG_RUNTIME_DIR` when set, otherwise a restrictive subdirectory of `tempfile.gettempdir()`; never break a held lock automatically, and require operator confirmation before removing a stale file.
- `reconciliation.py` produces `OwnershipKey`, `CandidateContract(session_id: str, turn_key: str, source_fingerprint: str, source_schema: str, render_fingerprint: str, operation_key: str, artifact_relative_path: str, artifact_sha256: str, artifact_size: int)`, `OwnedTraceCandidate`, `CandidateVerification`, `ReconciliationAction`, `ReconciliationPlan`, `ReconciliationDependencies(replay: Callable[[DiscoveryTurnEntry], ReplayTurn], map_trace: Callable[[ReplayTurn], TraceCreateRequest], artifact_snapshot: Callable[[ReplayTurn], str], upload_artifact: Callable[[ReplayTurn, CreatedTrace, str, RedactionMode], ArtifactVerification])`, `upload_artifact(replay: ReplayTurn, created: CreatedTrace, artifact_relative_path: str, redaction_mode: RedactionMode) -> ArtifactVerification`, `plan_reconciliation(session_scans: Iterable[SessionScanEntry], turn_entries: Iterable[DiscoveryTurnEntry], owned_traces: Iterable[OwnedTraceCandidate])`, and `apply_reconciliation(plan: ReconciliationPlan, gateway: MlflowGateway, dependencies: ReconciliationDependencies, retry_policy: RetryPolicy)`.
- `MlflowGateway.verify_owned_candidate(candidate: OwnedTraceCandidate, expected: CandidateContract) -> CandidateVerification` retrieves/verifies ownership metadata, source/render/schema/operation fingerprints, and calls `verify_named_json(candidate.trace_id, expected.artifact_relative_path, expected.artifact_sha256, expected.artifact_size)` through public APIs before reconciliation winner selection; `plan_reconciliation` itself remains a pure grouping/action planner and never selects a winner from unverified candidates.
- `apply_reconciliation` first obtains all `CandidateVerification` results, then calls an internal `select_verified_winner(verified_candidates)` step; no create, retire, or winner decision occurs before that verification phase completes.

- [ ] **Step 1: Write retry and lock tests.** Verify transient classification, bounded attempts, injectable jitter/sleep, no retry loop for auth/invalid requests, endpoint hashing without URI credentials, non-blocking held-lock failure with status `2`, cleanup after exceptions, and no lock acquisition in dry-run.

- [ ] **Step 2: Write reconciliation-plan tests.** Cover imported, unchanged-as-`skipped`-with-reason, updated, retired, skipped, and failed actions; exact ownership marker; `(session_id, turn_key)` lookup; multiple owned candidates; artifact digest/size verification before winner selection; no winner/create/retire decision before every candidate verification result exists; deterministic newest-created/tie-by-greatest-trace-ID winner selection among verified candidates; pagination; selected-session-only stale retirement including selected zero-turn sessions; complete-scan requirement; `unexamined_owned_traces`; and protection of unowned traces sharing native session IDs.

```python
def test_filtered_scan_cannot_retire_unselected_session(fake_gateway, discovery_entries):
    plan = plan_reconciliation(
        discovery_entries.session_scans.selected_only("session-a"),
        discovery_entries.turns,
        fake_gateway.owned_traces([owned("session-a"), owned("session-b")]),
    )

    assert not any(action.trace.session_id == "session-b" for action in plan.retirements)
```

- [ ] **Step 3: Run reconciliation tests to verify they fail.**

Run: `python -m pytest tests/unit/test_retry.py tests/unit/test_locking.py tests/unit/mlflow/test_reconciliation.py -q`

Expected: FAIL because retry, lock, and reconciliation behavior are not implemented.

- [ ] **Step 4: Implement ownership indexing, candidate verification, and action planning.**

Search only exact `opencode.importer=opencode-mlflow` traces, use server-side metadata filters when available, otherwise filter each paginated page locally, and group candidates by source identity. In the effectful reconciliation phase, call `verify_owned_candidate` for every candidate before any winner selection; it uses public trace retrieval and named-artifact verification, never search previews. Store `CandidateVerification` results in the plan, then choose only among verified exact matches by newest creation timestamp, breaking ties by lexicographically greatest MLflow trace ID. If none match, create and verify one replacement through `ReconciliationDependencies`. After a verified winner exists, retire every other importer-owned candidate for that identity, including incomplete candidates left by an earlier crash; if cleanup fails, keep the valid winner/old valid trace and report `replacement_cleanup_failed` with all affected IDs. Only duplicate identities that cannot be verified or deterministically resolved become `duplicate_owned_identity` conflicts.

- [ ] **Step 5: Implement strict replacement and rollback ordering.**

For an update: obtain a fresh `ReplayTurn`, map it through `dependencies.map_trace` into the exact `TraceCreateRequest`, create the new trace with ownership and `opencode.operation_key`, end and retrieve/verify metadata/status/timing/IDs, upload and verify the deterministic artifact through `dependencies.upload_artifact`, then retire the old trace and any other redundant owned candidate. If verification or upload fails, retire the new trace as rollback and retain the old valid trace. If rollback fails, retry within policy and report both trace IDs as a reconciliation conflict. If old-trace retirement fails after a verified replacement, try bounded compensating retirement of the new trace and never delete the old trace on uncertainty. If retirement is unsupported, global preflight must already have failed before any normal mutation.

- [ ] **Step 6: Implement ambiguous-create handling.**

Use the deterministic `opencode.operation_key` and client request/idempotency key when the target supports it. After an ambiguous timeout, search for the exact owner/session/turn/render/operation tuple before retrying creation. Record any surviving importer-owned candidate for deterministic winner selection and explicit operator recovery.

- [ ] **Step 7: Run fake-gateway BDD scenarios and commit.**

```gherkin
Feature: Reconcile only importer-owned traces
  Scenario: Replace changed content without deleting the old trace first
    Given an importer-owned trace exists for session "ses-1" turn "turn-1"
    And the source fingerprint has changed
    When I import the session
    Then the event order is create, verify, upload, verify, retire-old
    And an unowned trace with the same native session remains unchanged

  Scenario: Preserve the old trace when replacement upload fails
    Given replacement artifact verification fails
    When I import the changed turn
    Then the old trace remains active
    And the new trace is rolled back or reported as a recovery conflict
```

Run: `python -m pytest tests/unit/test_retry.py tests/unit/test_locking.py tests/unit/mlflow/test_reconciliation.py tests/bdd -m 'unit or bdd' -q`

Expected: PASS. Commit as `feat: reconcile importer-owned traces safely`.

### Task 12: Orchestrate discovery, preflight, replay, writes, dry-run, and reporting

**Files:**

- Create: `src/opencode_mlflow/importer.py`
- Create: `src/opencode_mlflow/reporting.py`
- Complete: `src/opencode_mlflow/cli.py`
- Create: `tests/unit/test_importer.py`
- Create: `tests/unit/test_reporting.py`
- Create: `tests/support/report_assertions.py`
- Extend: `tests/unit/test_cli.py`
- Extend: `tests/bdd/features/privacy_and_reporting.feature`

**Interfaces:**

- `importer.py` produces `ImportDependencies`, `ImportResult`, `replay_turn(entry: DiscoveryTurnEntry, snapshot: SourceSnapshot) -> ReplayTurn`, `run_import(config: ImportConfig, dependencies: ImportDependencies) -> ImportResult`, and `exit_status(result: ImportResult) -> int`.
- `ImportDependencies` injects normalized-snapshot factory, adapter/exporter factory, MLflow gateway factory, lock factory, temporary-file/chunk factory, replay builder, pure mapper, artifact-snapshot fingerprint function, artifact uploader, clock, and retry policy for deterministic tests. It constructs the Task 11 `ReconciliationDependencies`. The live source snapshot is closed before discovery; all discovery/replay uses the temporary read-only snapshot.
- `reporting.py` produces `ReportBuilder`, `render_human(result: ImportResult) -> str`, `render_json(result: ImportResult) -> str`, and `RESULT_SCHEMA = "opencode-mlflow.result.v1"`. The top-level JSON always contains `schema_version`, `dry_run`, `experiment`, `source`, `redaction`, `summary`, `sessions`, and `errors`; `TurnResult.action` and `SessionResult.action` are exactly `imported | updated | skipped | failed | retired`; dry-run uses `action=skipped`, `reason=dry_run`, and a same-enum `planned_action`.
- `cli.main` parses args, calls `run_import`, writes only the selected output format, and returns the specified exit status.

- [ ] **Step 1: Write orchestration tests around a fake source and gateway.** Verify phase order: config → live read-only connection/`probe_schema` → registered `SourceExporter` → normalized snapshot extraction → close live source → open `NormalizedSnapshotAdapter` replay snapshot → discovery plan → global MLflow capability preflight (including retirement and payload limits) → normal-run lock → exact experiment/owned-trace lookup → replay fingerprint checks → writes → report. Assert no MLflow mutation occurs before all required pre-write checks pass and no live SQLite connection remains during MLflow work.

```python
def test_dry_run_reports_planned_actions_without_mutation(import_dependencies, config):
    result = run_import(replace(config, dry_run=True), import_dependencies)

    assert result.dry_run is True
    assert result.report["sessions"][0]["turns"][0]["planned_action"] in {
        "imported",
        "updated",
        "skipped",
        "failed",
        "retired",
    }
    assert result.report["sessions"][0]["turns"][0]["action"] == "skipped"
    assert result.report["sessions"][0]["turns"][0]["reason"] == "dry_run"
    assert import_dependencies.gateway.mutations == []
    assert import_dependencies.lock_factory.acquisitions == []
```

- [ ] **Step 2: Write report sanitization and count-unit tests.** Assert the JSON schema version, resolved safe configuration summary, experiment status/ID, source schema/fingerprint, redaction counts, disjoint current-turn counts, separate `retired_traces`, `unexamined_owned_traces`, per-session/per-turn identities, stale-retirement `source_present=false`, trace IDs where present, normative action enum, safe error codes/messages, and absence of absolute source paths, credentials, tracking URI secrets, prompts, responses, tool output, source code, and matched secrets.

- [ ] **Step 3: Run orchestration tests to verify they fail.**

Run: `python -m pytest tests/unit/test_importer.py tests/unit/test_reporting.py tests/unit/test_cli.py -q`

Expected: FAIL because the pipeline and result model are not implemented.

- [ ] **Step 4: Implement the two-pass pipeline.**

Keep the temporary normalized-snapshot transaction open across discovery and replay, but close the live source connection immediately after catalogued extraction completes. Write sanitized session-scan and turn plan entries during discovery, run the complete global capability contract before normal-mode experiment creation, acquire the non-blocking lock only for a normal write run before resolving the experiment or looking up owned traces, and replay each selected session from the same temporary snapshot. Reject any replayed source fingerprint that differs from its plan entry.

- [ ] **Step 5: Implement per-item failure isolation.**

Catch and sanitize session/turn write failures after global preflight, continue later entries, record the item as `failed`, and return exit `1` if any item failed. Treat schema/configuration/capability/snapshot/lock failures before writes as exit `2`; missing capability is invocation-global, not per-item. Ensure source snapshot, plan, canonical, and rollback temporary files are cleaned in success, failure, rollback, and ambiguous-create paths.

- [ ] **Step 6: Implement human and JSON output.**

Human output stays concise and progress-oriented, including the broad-selection warning when no optional filters are present and recommending dry-run. JSON uses the normative non-mutating `action=skipped`, `reason=dry_run`, and predicted `planned_action` in dry-run so automation can distinguish preview from completion without inventing an action outside the enum.

- [ ] **Step 7: Run the full deterministic suite and commit.**

Run: `python -m pytest -m '(unit or contract or bdd) and not integration' -q`

Expected: PASS with no network, real MLflow, live OpenCode database, or source writes. Commit as `feat: orchestrate import and versioned reporting`.

### Task 13: Complete the BDD acceptance suite

**Files:**

- Complete: `tests/bdd/conftest.py`
- Extend: `tests/bdd/features/mapping_and_fidelity.feature`
- Extend: `tests/bdd/features/reconciliation.feature`
- Extend: `tests/bdd/features/privacy_and_reporting.feature`
- Extend: `tests/bdd/features/cli_and_selection.feature`
- Extend: `tests/bdd/features/source_safety.feature`
- Complete: `tests/bdd/steps/test_import_steps.py`
- Extend: `tests/support/fake_gateway.py`

**Interfaces:**

- BDD fixtures expose a temporary synthetic database, an in-memory `FakeMlflowGateway`, deterministic clock/retry policy, and the public `run_import`/`main` boundary.
- Step definitions must assert observable reports, gateway event histories, source immutability, trace identity/mapping, artifact contents, and exit status; they must not call private implementation helpers.

- [ ] **Step 1: Add Gherkin scenarios mapped to acceptance criteria.** Include exact experiment create/reuse, WAL-safe source read, unsupported schema status `2`, all four selection dimensions, one trace per turn/shared native session, root values and child mapping, six-turn representation, unchanged rerun, replacement rollback, stale retirement scope, non-owned trace protection, redaction, dry-run no mutation, per-item continuation, and report/exit status.

- [ ] **Step 2: Implement shared BDD fixtures and step definitions.** Keep source rows synthetic and deterministic. Use the fake gateway for all behavior that does not require a real MLflow server. Assert event order and result JSON rather than incidental implementation details.

- [ ] **Step 3: Run the BDD suite independently.**

Run: `python -m pytest tests/bdd -m bdd -vv`

Expected: PASS and readable scenario names in CI. A failing scenario must identify the acceptance behavior rather than a private helper assertion.

- [ ] **Step 4: Commit the acceptance suite.**

```bash
git add tests/bdd tests/support/fake_gateway.py
git commit -m "test: cover importer acceptance criteria with bdd"
```

### Task 14: Add pinned MLflow and live-WAL integration verification

**Files:**

- Extend: `tests/integration/constraints.txt`
- Extend: `tests/integration/README.md`
- Create: `tests/integration/test_mlflow_trace_contract.py`
- Create: `tests/integration/test_mlflow_reconciliation.py`
- Complete: `tests/fixtures/six_turn_session.py`
- Complete: `tests/integration/test_wal_snapshot.py`
- Complete: `tests/integration/test_scale_memory.py`
- Extend: `docs/compatibility.md`
- Extend: `.github/workflows/ci.yml`

**Interfaces:**

- Integration tests consume `MLFLOW_TRACKING_URI` and an explicit integration dependency set; they must fail clearly when the configured server cannot satisfy the behavioral gateway contract.
- The integration fixture contains six logical turns in one synthetic session and uses only sanitized values.

- [ ] **Step 1: Consume the tested compatibility profile.** Verify that `tests/integration/constraints.txt`, `tests/integration/README.md`, and `docs/compatibility.md` all identify the exact `3.17.2.dev0` client/server build and the public named-artifact, trace, search, payload, and retirement behaviors proven in Task 0. Do not use a mutable `latest` image/tag.

- [ ] **Step 2: Add real experiment and trace contract tests.** Verify exact experiment resolution/creation/reuse, one trace per logical turn, six distinct trace IDs sharing one native session ID, full root payload retrieval, nested model/tool/content spans/events, status/timing/token/cost metadata, source IDs, and deterministic artifact path/content verification.

- [ ] **Step 3: Add real dry-run and reconciliation tests.** Verify dry-run creates no experiment/traces/artifacts, unchanged rerun creates no duplicate owner traces, changed turn replacement follows create/verify/upload/retire order, retirement failure preserves the old trace, stale retirement is scoped, and non-importer traces remain untouched.

- [ ] **Step 4: Complete WAL and scale test coverage.** With another SQLite connection writing to WAL, open the importer snapshot, verify rows committed before normalized extraction are present and rows committed after extraction are absent, assert stable discovery/replay results, source file/schema/data bytes are unchanged, and `-wal`/`-shm` are not removed or checkpointed. Run the 100,000-record/multi-hundred-megabyte artifact fixture and assert Python heap growth below 256 MiB, fixed upload chunks, complete artifact bytes, and `payload_too_large` rejection before trace creation for oversized complete root values.

- [ ] **Step 5: Run integration checks explicitly and commit.**

Run: `python -m pytest tests/integration -m integration -vv`

Expected: PASS against the documented pinned pair and a temporary WAL fixture. Commit as `test: verify mlflow and wal integration contracts`.

### Task 15: Write user, operator, security, compatibility, and contributor documentation

**Files:**

- Create: `README.md`
- Create: `CHANGELOG.md`
- Create: `docs/architecture.md`
- Complete: `docs/schema-catalog-v1.md`
- Create: `docs/usage.md`
- Create: `docs/security.md`
- Create: `docs/testing.md`
- Create: `docs/result-schema.md`
- Create: `docs/operations.md`
- Create: `docs/code-review-checklist.md`
- Complete: `docs/compatibility.md`

**Interfaces:**

- Documentation must describe the shipped CLI and result/artifact contracts without promising behavior outside the design.
- `docs/result-schema.md` is the human-readable contract for `opencode-mlflow.result.v1`, action names, safe error codes, ownership metadata, and deterministic artifact paths.

- [ ] **Step 1: Document installation and usage.** In `README.md`/`docs/usage.md`, document Python support, installation, `opencode-mlflow import` syntax, default database path, tracking configuration, all filters, inclusive/strict semantics, dry-run, output modes, exit statuses, broad-selection warning, and examples that do not contain real credentials or transcript content.

- [ ] **Step 2: Document source and architecture behavior.** In `docs/architecture.md`, show the phase diagram and module boundaries, live-source normalized-extraction lifetime, temporary replay snapshot lifetime, schema precedence, fork behavior, ordering, turn boundaries, discovery plan privacy, bounded-memory strategy, and MLflow gateway boundary. Link both source specs rather than copying conflicting requirements. In `docs/schema-catalog-v1.md`, keep the concrete table/column/join/timestamp/JSON/classification/safe-column matrix synchronized with `source/catalog.py`.

- [ ] **Step 3: Document security and privacy.** In `docs/security.md`, explain excluded tables/fields, standard MLflow authentication/TLS configuration, verbatim-mode sensitivity, `--redact`, all named detector categories/pattern semantics, protected identifiers, safe logs/reports, temporary-file permissions/cleanup, and the fact that source databases are never modified.

- [ ] **Step 4: Document test and compatibility operations.** In `docs/testing.md` and `docs/compatibility.md`, document TDD layer commands, BDD markers, fixture provenance rules, fake gateway use, the exact pinned MLflow `3.17.2.dev0` pair, the public API/method/field table, named artifact upload and fixed-chunk verification, global retirement capability, integration setup, and how an unsupported client/server pair is reported.

- [ ] **Step 5: Document report and operator runbook.** In `docs/result-schema.md` and `docs/operations.md`, document JSON fields, exact action enum `imported|updated|skipped|failed|retired`, unchanged as a skipped reason/subcount, dry-run `action=skipped`/`reason=dry_run`/`planned_action`, explicit session-versus-turn/retirement count units, `unexamined_owned_traces`, safe error codes, retry/lock behavior, concurrent-run limitation, replacement/rollback and duplicate-candidate recovery, stale-retirement scope including zero-turn tombstones, artifact cleanup, and how to inspect artifacts without exposing secrets.

- [ ] **Step 6: Add code-review checklist and changelog policy.** The checklist must require review of source write safety, SQL parameterization/catalog ownership, schema compatibility, redaction coverage, report sanitization, bounded memory, public MLflow APIs, replacement ordering, retry/idempotency behavior, tests, and docs. `CHANGELOG.md` must record the first release, supported MLflow pair, known limitations, and explicitly list excluded scope.

- [ ] **Step 7: Verify documentation links and commit.**

Run: `python -m pytest -m '(unit or contract or bdd) and not integration' -q`

Expected: PASS. Manually verify every README command matches the parser, every documented result field matches the report serializer, and every referenced path exists. Commit as `docs: document importer operations and contracts`.

### Task 16: Run layered code review, security review, and release verification

**Files:**

- Review: all `src/`, `tests/`, `docs/`, `pyproject.toml`, and CI files
- Update: `CHANGELOG.md` and documentation only when review identifies a verified discrepancy

**Review gates:**

- **Code-quality review:** module boundaries, type consistency, resource cleanup, live-source/normalized-snapshot lifetime, deterministic cross-table ordering/deduplication, stream/chunk/memory behavior, and maintainability.
- **Compatibility/API review:** both concrete schema catalogs, public MLflow-only access, the `3.17.2.dev0` method/field profile, named artifact path/chunk verification, global retirement capability, exact experiment/session/artifact semantics, and graceful unsupported-target failures.
- **Security/privacy review:** no credential-table reads, no secret-bearing output, redaction consistency, protected IDs, temp-file permissions, URI/lock sanitization, and no source writes.
- **Acceptance/verifier review:** every requirements acceptance criterion has a passing unit, contract, BDD, or integration test and a documented operator behavior, including canonical-json-v1 edge vectors, zero-turn stale retirement, duplicate-owned-candidate convergence, payload limits, and the 256 MiB scale bound.
- **Documentation review:** CLI examples, result schema, compatibility statement, redaction detector contract, limitations, and test commands are accurate.

- [ ] **Step 1: Run automated quality gates.**

Run:

```bash
ruff check .
ruff format --check .
mypy src
python -m pytest -m '(unit or contract or bdd) and not integration' -q --cov=opencode_mlflow --cov-report=term-missing
```

Expected: no lint/format/type failures, all deterministic tests pass, and coverage includes every acceptance-critical branch rather than relying on one aggregate percentage.

- [ ] **Step 2: Run the four review lenses.**

Have independent reviewers inspect the complete diff using the checklist above. Record findings as `Blocker`, `Warning`, or `Nit`; no `Blocker` or `Warning` may remain. Reviewers report findings only and do not modify production code; fixes return to the implementation owner as new red tests where applicable.

- [ ] **Step 3: Re-run integration and privacy verification after fixes.**

Run the pinned MLflow integration suite, WAL suite, BDD suite, and a sanitized-output audit that searches generated reports/logs/temp upload directories for fixture secret sentinels, absolute source paths, and transcript content.

- [ ] **Step 4: Perform final manual acceptance walkthrough.**

Exercise: new/existing experiment, no-filter dry-run, each selection filter, mixed schema, missing timestamps, no-user session, six-turn session, unchanged rerun, changed-turn replacement failure, stale retirement, unowned trace sharing a session, redacted upload, unsupported MLflow capability, held lock, and per-item continuation. Record observed exit statuses and report actions in the release notes.

- [ ] **Step 5: Update the changelog and hand off.**

Only after all checks pass, document the release version, tested Python/MLflow pair, known limitations, and commands used. Do not add OpenCode HTTP/watch/run-container behavior during cleanup.

## Spec coverage matrix

| Design area | Planned coverage |
| --- | --- |
| CLI/configuration and validation | Tasks 1–2, 12, 13, 15 |
| Read-only WAL snapshot and schema probe | Task 3, Task 14 |
| Legacy/current/mixed adapters and raw-preserving model | Task 4 |
| Ordering, timestamps, filters, and turns | Tasks 5–6, 13 |
| Canonical content, fingerprints, artifacts | Tasks 7–8, 10, 14 |
| MLflow compatibility profile/capability/experiment/trace gateway | Tasks 0, 9–10, 14 |
| Ownership, reconciliation, locking, retries | Task 11, Tasks 12–14 |
| Redaction/security/privacy | Task 7, Tasks 8–9, 12–15 |
| Reports and exit statuses | Task 12, Task 13, Task 15 |
| Bounded memory, fixed-chunk upload, payload limits, and failure isolation | Tasks 3, 5–6, 8, 11–12, 14 |
| TDD/BDD structure | Global strategy, Tasks 1–2, 5–7, 10–14 |
| Code review and release gates | Task 16 and `docs/code-review-checklist.md` |
| User/operator/contributor documentation | Task 9, Task 15, Task 16 |

## Plan self-review checklist

- [x] The plan is limited to the supplied design and requirements; no implementation is performed here.
- [x] The new-project scaffold, source package layout, test layers, fixtures, BDD features, CI, and documentation files are named.
- [x] Every design work package has a TDD task with a failing-test step, narrow command, implementation boundary, and passing-test/commit checkpoint.
- [x] BDD scenarios cover selection, source safety, mapping/fidelity, reconciliation, privacy, reporting, dry-run, and exit statuses.
- [x] The plan explicitly preserves current-over-legacy precedence, WAL snapshot behavior, raw-field fidelity, secret exclusion, bounded memory, deterministic fingerprints/artifacts, and importer ownership.
- [x] The plan explicitly covers WAL-aware normalized extraction/replay, concrete schema catalogs, canonical-json-v1 edge values, 100,000-record/256 MiB scale verification, fixed upload chunks, payload limits, zero-turn stale retirement, duplicate-owned-candidate convergence, and global retirement capability preflight.
- [x] MLflow trace/span calls are confined to the gateway’s public `mlflow.start_span_no_context`/`LiveSpan.end` adapter; mapping remains pure, replay is lazy/disk-backed, and `identity_map_pending` keeps artifact paths deterministic.
- [x] MLflow compatibility is behavioral, pinned to the initial `3.17.2.dev0` profile, and requires a reproducible client/server pair before writer/integration work is accepted.
- [x] Review lenses include code quality, compatibility/API stability, security/privacy, acceptance verification, and documentation accuracy.
- [x] No task asks for OpenCode HTTP support, watch mode, arbitrary filter expressions, or an MLflow run container.
