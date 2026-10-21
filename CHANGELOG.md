# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this
project's versions are anchored to its own release commits.

## [0.1.0] — 2026-10-22

First release: the complete one-shot import pipeline from an OpenCode
SQLite session database into MLflow traces, with ownership
reconciliation, streamed canonical artifacts, deterministic re-run
behavior, and a versioned safe result document.

### Added

- `opencode-mlflow import` CLI: `--experiment-name` (required), `--db`
  (default `~/.local/share/opencode/opencode.db`), `--tracking-uri`,
  inclusive creation-time bounds `--from-time`/`--to-time`
  (ISO-8601 with explicit offset or `Z`), `--directory`
  (component-descendant matching), `--model` (exact identity match),
  `--min-user-turns` (strictly-greater threshold), `--redact`,
  `--dry-run`, `--output human|json`, `--log-level`. Exit statuses
  0 (complete/no-op), 1 (item failures), 2 (invalid or pre-write
  failure).
- Read-only source handling: `mode=ro` + `query_only` + one deferred
  read transaction per live connection; normalized temporary snapshot
  (0700/0600) closed over the live source immediately after
  extraction; WAL/-shm companions preserved; the source database is
  never written, checkpointed, vacuumed, migrated, or deleted.
- Schema catalog v1 with legacy, current, and mixed generation
  classification; safe `unsupported_schema` diagnostics; verbatim
  retention of unknown fields, malformed JSON, and duplicate-key
  objects.
- Logical-turn reconstruction with stable turn keys, fallback and
  collision handling; sanitized streaming JSONL discovery plan.
- `canonical-json-v1` serialization (UTF-8 byte key order, RFC 8785
  floats, typed wrappers for BLOBs/non-finite floats/duplicate keys)
  and the four fingerprint domains (session-context, turn-source,
  render, artifact-snapshot) with the fixed `identity_map_pending`
  placeholder.
- Deterministic v1 redaction (`--redact`): seven detector categories
  (`pem_private_key`, `auth_header`, `structured_auth_field`,
  `secret_key_value`, `connection_secret`, `known_token_prefix`,
  `opaque_binary`), `[REDACTED]`/`[REDACTED_BINARY]` markers, protected
  identity fields, counts-only reporting.
- MLflow writer boundary: the only MLflow-importing modules, confined
  to public APIs; runtime capability gate; named trace artifacts
  through the public composition (artifactLocation tag →
  `get_artifact_repository` → `log_artifact`/`download_artifacts`)
  with fixed-chunk streaming and sha256+size verification. The gate also
  performs a read-only `GET <tracking-base>/version` attestation and requires
  exact client/server `3.17.2` agreement before trusting profile-backed
  status, artifact, retirement, and inline-payload behavior.
- Turn→trace mapping: one trace per logical turn with LLM/agent/tool/
  content/event child spans, namespaced `opencode.*` attributes,
  outcome-derived root status, digest/size references for oversized
  child payloads, and `payload_too_large` rejection of oversized
  complete root values before trace creation.
- Wave 2 fidelity corrections: normalized source completion timestamps and
  lifecycle outcomes, collision-proof `<source_table>:<source_id>` child
  lineage, full child-span verification, and `artifact-v2` provenance
  envelopes carrying the exact mapped-request fingerprints/path and source
  database fingerprint. The observable mapping contract is now
  `mapping-v3`; artifact paths retain the existing path namespace.
- Ownership reconciliation: `opencode.importer=opencode-mlflow` +
  `(session_id, turn_key)` ownership, six run-mode-tagged equivalence
  fields, artifact integrity at the candidate's own recorded path and
  digest, strict create→verify→upload→verify→record→retire ordering,
  rollback that preserves the old trace, scoped stale-turn and
  zero-turn-tombstone retirement, per-item failure isolation.
- Reliability: bounded exponential-backoff retries on transient
  failures only; non-blocking per-host advisory lock (credential-free
  naming, XDG_RUNTIME_DIR-aware, never auto-broken); ambiguous creates walk
  all survivor pages and fail closed on partial/unverifiable recovery; dry
  runs that mutate nothing and skip the lock.
- Versioned result document `opencode-mlflow.result.v1` with the exact
  action enum `imported|updated|skipped|failed|retired`, unchanged as
  a skipped reason, dry-run planned actions, separate
  `retired_traces`/`unexamined_owned_traces` counts, and safe error
  codes.
- Test suite: 1363 default-selection tests (unit/contract/BDD
  including 41 Gherkin acceptance scenarios) plus 30 integration/profile
  tests, including a live-server suite against the disposable pinned
  server and a 105k-record/241 MiB bounded-memory scale test.

### Supported MLflow pair

`mlflow==3.17.2` (PyPI) as the client, with a tracking server launched
from that same installed distribution. The plan's `3.17.2.dev0` label is
not published on PyPI; the profile pins the exact stable release of the
same API line (controller ruling, 2026-10-21). See
[docs/compatibility.md](docs/compatibility.md) for the resolved public
surface and observed behaviors.

### Known limitations

- Redaction v1 has documented gaps: quoted `Basic` credentials inside
  malformed JSON text, connection strings without a password
  (`user:@host`), and `Digest`/`negotiate` schemes (see
  [docs/security.md](docs/security.md)).
- Concurrent importer invocations are serialized only per host via the
  advisory lock; cross-host concurrency must be serialized by the
  operator.
- A dry run's `updated` prediction is provisional: the final
  unchanged-versus-replace decision waits for candidate verification,
  which a dry run does not perform.
- Retiring a trace does not delete its artifact bytes; artifact
  cleanup is a manual operator procedure (see
  [docs/operations.md](docs/operations.md)).
- The importer reports `unexamined_owned_traces` and leaves untouched
  any owned trace outside a run's reconciliation scope; narrowing or
  changing filters never retires what a filtered-out session owns.
- Ambient `OTEL_EXPORTER_OTLP_*` environment variables can silently
  reroute MLflow span export; the importer detects the resulting NoOp
  degradation and fails the item loudly (`mlflow_span_degraded`)
  rather than losing traces (see
  [docs/compatibility.md](docs/compatibility.md)).

### Excluded scope (explicit)

These are design decisions, not TODOs:

- No OpenCode HTTP API, OpenCode service calls, or OpenCode
  configuration parsing — the SQLite database is the only input.
- No watch mode, daemon, scheduler, or continuous operation — one
  invocation is one complete pass.
- No arbitrary filter expression language — selection is exactly the
  documented CLI filters.
- No MLflow run container: output is traces, spans, and trace
  artifacts only; no `mlflow.start_run`, no run artifacts, no models,
  no model registry.
- No ORM and no direct backend-store writes — SQLite is read through
  parameterized catalogued SQL; MLflow is written only through its
  public client API.
- No private or undocumented MLflow APIs; no UUID-only attachment
  channels; no presigned multipart flows.
- No source-database mutation of any kind, including checkpointing,
  vacuuming, migration, or deletion.
- No credentials accepted, stored, or logged by the importer;
  authentication rides the standard MLflow client configuration.
