# Architecture

This document describes the **shipped** pipeline: the execution phases,
module boundaries, lifetimes of the two database views, and the
cross-cutting invariants. It documents what the code does, with pointers
into `src/opencode_mlflow/`.

The binding source specifications are linked here rather than copied —
where those documents discuss alternatives, this page records only the
resolved, implemented behavior:

- Requirements:
  [docs/superpowers/specs/2026-10-21-opencode-mlflow-import-requirements.md](superpowers/specs/2026-10-21-opencode-mlflow-import-requirements.md)
- Design (including the "Decisions made" section that resolved its own
  ambiguities):
  [docs/superpowers/specs/2026-10-21-opencode-mlflow-import-design.md](superpowers/specs/2026-10-21-opencode-mlflow-import-design.md)

## Pipeline phases

One invocation of `run_import` (`importer.py`) executes the phases in
strict order. No MLflow mutation occurs before every pre-write check has
passed; a dry run performs the phases but replaces the write tail with a
plan report.

```text
1  configuration          parse_config (cli.py / config.py)
       |
2  live read-only source  open_live_connection: mode=ro URI, query_only,
       |                  one deferred BEGIN          (source/connection.py)
       v
3  schema probe + exporter registration                  (source/probe.py,
       |                  probe_schema; registry.build_live_exporter)
       v                  [probing connection closes here]
4  normalized snapshot    create_normalized_snapshot opens its own live
       |                  read-only connection (injected into the exporter
       |                  through bind_connection), streams the catalogued
       |                  extraction into a 0o700/0o600 temporary SQLite
       |                  copy, and closes the connection immediately
       |                  after extraction            (source/snapshot.py)
       v
5  replay adapter         NormalizedSnapshotAdapter over the temporary
       |                  copy, kept open across discovery and replay
       v                  (source/replay.py)
6  discovery plan         discover_sessions: streaming scan, selection,
       |                  turn summaries, fingerprints -> 0o600 JSONL plan
       v                  (source/discovery.py)
7  MLflow capability      gateway.probe (read-only) + require_capabilities
       |                  (mlflow/capabilities.py)
       v
8  advisory lock          normal runs only; dry runs skip it  (locking.py)
       v
9  experiment resolution  read-only lookup first; create-on-missing is
       |                  deferred until after the fingerprint checks
       v                  (mlflow/gateway.py)
10 owned-trace lookup     iter_owned_traces: paginated search, exact
       |                  marker re-verification            (mlflow/gateway.py)
       v
11 replay fingerprint     every plan entry's source fingerprint recomputed
       |                  from the same snapshot; mismatch aborts (exit 2)
       v                  before any write                  (importer.py)
12 reconciliation writes  plan_reconciliation (pure) -> apply_reconciliation:
       |                  verify all -> decide -> create/verify/upload/
       |                  verify/record -> retire old        (mlflow/
       v                  reconciliation.py)
13 result document        ReportBuilder -> render_human / render_json,
                          exit_status                       (reporting.py)
```

Normal mode additionally acquires the lock (phase 8) *before* experiment
resolution and owned-trace lookup; experiment creation happens only after
all pre-write checks (phases 7, 10, 11) pass.

## Module boundaries

| Boundary | Modules | Responsibility |
| --- | --- | --- |
| CLI/configuration | `cli.py`, `config.py` | Parse flags, normalize values, produce the frozen `ImportConfig`; write only the selected output. |
| Orchestration | `importer.py` | Phase order, dependency wiring, replay consistency checks, per-run reporting. |
| Source connection | `source/connection.py` | The single live read-only connection policy (see below). |
| Schema catalog/probe | `source/catalog.py`, `source/probe.py` | Pure-data catalog of both generations; metadata-only classification (`current`/`legacy`/`mixed`/unsupported). |
| Adapters | `source/legacy.py`, `source/current.py`, `source/registry.py` | Generation-specific extraction SQL (parameterized, catalog-owned); mixed-precedence composition. |
| Canonical model | `source/model.py`, `source/normalize.py`, `source/ordering.py` | Session/record types, start/completion timestamp normalization, deterministic ordering keys, source fingerprint. |
| Snapshot | `source/snapshot.py` | Temporary normalized snapshot lifecycle, redaction-before-persistence, pre-redaction selection evidence. |
| Replay | `source/replay.py`, `source/turns.py` | `ReplayTurn` protocol (single home), logical-turn reconstruction, bounded root loaders. |
| Selection/discovery | `source/selection.py`, `source/discovery.py` | Filters, streaming scan, the sanitized JSONL discovery plan. |
| Content | `content/canonical.py`, `content/fingerprint.py`, `content/redaction.py`, `content/artifacts.py` | `canonical-json-v1`, the four fingerprint domains, the v1 redactors, streamed artifacts and inline limits. |
| MLflow gateway | `mlflow/gateway.py`, `mlflow/capabilities.py` | The ONLY modules that import MLflow. Public-API calls, capability gate, artifact sink. |
| Mapping | `mlflow/mapping.py` | Pure turn→trace-request mapping; imports no MLflow at module import time. |
| Reconciliation | `mlflow/reconciliation.py` | Ownership planning, candidate verification, safe replacement ordering. |
| Reporting | `reporting.py` | `opencode-mlflow.result.v1` and its renderers. |
| Reliability | `retry.py`, `locking.py`, `errors.py` | Bounded transient retries, the advisory lock, the safe error hierarchy. |
| Shared contracts | `contracts.py`, `json_values.py` | Shared redaction/envelope types, `DuplicateKeyObject`, lossless JSON decoding, `PROTECTED_IDENTITY_FIELDS`, the single `IDENTITY_REDACTOR`. |

## Source lifetimes

**Live source (short-lived).** The live OpenCode database is opened
through `open_live_connection` with a URI containing only the normalized
path and `mode=ro` (never `immutable=1`, which can ignore WAL contents;
the `-wal`/`-shm` companions stay in place), `PRAGMA query_only=ON`, a
bounded busy timeout, and exactly one deferred `BEGIN`. The pipeline uses
this connection twice, and closes it both times immediately after use:

1. The *probing* connection (phases 2–3) closes before extraction.
2. The snapshot driver opens its *own* live connection for the
   catalogued extraction (phase 4) and closes it as soon as extraction
   completes.

No live SQLite connection remains during discovery, capability checks,
or any MLflow work. The source is never written, checkpointed, vacuumed,
migrated, or deleted; the WAL/SHM companions are preserved (their byte
content may legitimately change through SQLite's own read-mark protocol —
the integration suite asserts metadata-level invariants only).

**Temporary replay snapshot (run-scoped).** `create_normalized_snapshot`
streams the permitted, catalogued rows into a fresh SQLite file named
`snapshot.db` inside a mode-`0o700` directory under the run root, with
the file itself mode `0o600` and two tables (`snapshot_sessions`,
`snapshot_records`). Rows carry the canonical record envelope (decoded
value, raw text, raw columns, warnings) plus the typed fields needed for
lossless replay; BLOBs ride a namespaced base64 tag. The injected
redaction callback is applied **before** persistence. The snapshot lives
for the whole run — discovery, fingerprint verification, replay, and
artifact streaming all read it — and is removed in an outer `finally`.
Run-root cleanup is checked: a failure is reported as the generic
`temporary_run_cleanup_failed` warning and one `failed_items` count without
masking the underlying result. The run root also holds the
`discovery-plan.jsonl` plan file and the artifact staging subdirectory.

Duplicate-key JSON rides a namespaced ordered-pairs tag, including every
nested occurrence. Only the top-level duplicate wrapper/source envelope
retains full raw text; nested wrappers carry a bounded empty marker. For an invocation with a
directory and/or model filter, `snapshot_sessions` additionally stores
only pre-redaction match booleans. Selector values and secret-derived
hashes are never stored as selection evidence, and those booleans never
enter discovery-plan payloads, fingerprints, traces, artifacts, or reports.

## Schema precedence and fork behavior

`probe_schema` classifies the database against
[Schema catalog v1](schema-catalog-v1.md): `current` (`session_v2` +
`session_message`), `legacy` (`session` + `message` + `part`), `mixed`
(both), or unsupported (a safe `unsupported_schema` diagnostic, exit 2).

For a `mixed` database, `MixedAdapter` (`source/registry.py`) compares
only session ids between the two session tables: every current session is
exported and the current row **wins an overlapping id** (the suppressed
legacy row is retained as a `legacy_session_suppressed` warning on the
authoritative current header); legacy-only ids export through the legacy
adapter; duplicate ids are never merged. Sessions stream in globally
normalized creation-time order, interleaving generations. Each emitted
session belongs to exactly one generation.

Fork sessions include only the records actually persisted for the child.
The importer does not query a parent to synthesize history or prepend
parent turns; parent/fork ids, the fork boundary, and persisted copied
rows remain available in the canonical artifact.

## Ordering and turn boundaries

**Ordering** (`source/ordering.py`). Within one authoritative sequence
domain (current generation: `session_message (session_id, seq)`), records
order by `(sequence, fixed tie rank, stable source id)`; sequence values
from unrelated streams are never compared. Across streams without a
shared domain, records order by `(normalized timestamp, tie rank, stable
source id)`, with untimestamped records after every timestamped one. The
tie ranks are catalog data (10 message container, 20 content part, 30
lifecycle event, 40 scoped record, 90 unknown), never an incidental
query order. Parent/child attachment is carried by the exporter's
*emission order* — each part immediately after its parent — which the
snapshot persists as written order and the replay cursors replay; the
pure sort key does not re-derive it.

The live adapters make source ties explicit before that emission order is
persisted. On ordinary SQLite tables, merge-joined parts/messages and
session-scoped rows append the table's physical `rowid` as the final SQL
tie-breaker, and missing-ID identities include that same discriminator.
Secret-bearing/content columns are therefore never used to decide which
tied physical row came first. A `WITHOUT ROWID` table uses its declared
primary-key columns as the strongest catalog-stable fallback. A malformed
schema that exposes neither a physical rowid nor a declared key cannot
distinguish otherwise identical ties; those ties remain dependent on the
database's source order, which is an unavoidable compatibility limitation.

**Turn boundaries** (`source/turns.py`). A logical turn starts at a
persisted, non-synthetic user prompt (a message-container record
classified `user`) and contains every following record until the next
user prompt. System, synthetic, tool-only, lifecycle, and compaction
metadata never start a turn; records before the first prompt stay
session-level unassigned records. The normal turn key is
`turn-<sha256(session_id \0 source_type \0 source_id)>`; a first-user
record without a source id hashes a documented fallback tuple
(`turn_key_fallback_identity` warning), and two turns that would still
collide add a stable occurrence ordinal (`turn_key_collision` warning) —
turns are never merged.

Catalogued lifecycle rows (`session_pending`/`session_inbox` in current
schemas and timestamped `todo` rows in legacy schemas) stay
outside the turn-boundary cursor. Mapping consumes their unambiguously
session-associated records separately as root-parented lifecycle events, so
they do not start turns and appear exactly once in each artifact's record
sections. Lifecycle outcomes are retained as source outcome metadata when no
assistant result exists; such a trace remains `UNSET` unless the outcome is
an explicit failure, which maps to `ERROR`. The shared outcome classifier
also applies to message `error`/`status`/`finish`, error parts, tool state
errors, and opaque records. A later successful assistant result supersedes
earlier non-terminal failures, while unrelated unknown records do not fail a
turn.

Every source record carries an optional normalized `completion_timestamp`.
Adapters prefer decoded `time.completed`, `time.end`, and `state.time.end`,
then a catalogued row `time_updated`, preserving malformed-value warnings.
Turn end timing retains the last known completion/start instant when a later
record is untimed. Mapped child end times use completion when available and
carry `opencode.completion_timestamp_ns`.

Mapped child identities are qualified as `<source_table>:<raw source id>`;
parent records carry the parent table as well as the raw parent ID. Raw
`opencode.source_table`/`opencode.source_id` attributes remain available for
SQLite lookup, while qualified keys are used for parent maps, gateway
verification, and artifact identity maps. The mapping contract is
`mapping-v3`.

## Discovery plan privacy

The discovery plan (`discovery-plan.jsonl`, mode `0o600` inside the run
root) is JSONL and carries identifiers, counts, warning codes,
timestamps, model identities, and hashes only — never transcript,
prompt, response, tool, or source-code content. One `session_scan`
entry is written for every compatible session (selected or not, including
zero-turn tombstones), and one `turn` entry per logical turn of every
*selected* session. A selected session whose scan completed with zero
logical turns is an explicit tombstone, which is what permits retiring
all previously owned turns of that session while a filtered run never
retires an unselected session's traces.

## Fingerprints: turn-local versus session-complete

All fingerprints are 64-hex SHA-256 digests over versioned
scheme/domain headers plus length-framed `canonical-json-v1` bytes
(`content/fingerprint.py`). The boundary between the two families is the
core reconciliation invariant:

- **Turn-local** (`source_fingerprint`, and `render_fingerprint` which
  combines it with the render-contract versions and redaction mode):
  covers the session's *immutable identity/selection context* (session
  id, generation, project/workspace/parent/fork ids, slug, title,
  directory/path, agent, model, version, creation instant), the
  session-scoped records assigned to the turn, the turn's own ordered
  records, and the redaction-mode tag. Mutable session fields (update
  times, lifecycle state, cost/token totals, summaries), unassigned
  records, and other turns' records stay **out**. Consequence: changing
  one turn replaces only that turn's trace; changing session identity
  context replaces every turn of the session.
- **Session-complete** (`artifact_snapshot_fingerprint`): streams the
  complete canonical session payload — every session field (mutable ones
  included) plus all permitted records visible through the turn's three
  replay cursors — and terminates with the fixed `identity_map_pending`
  placeholder instead of any generated MLflow identity. Generated
  MLflow ids are inserted into the artifact document only *after* this
  fingerprint has fixed the hash and the artifact path, so generated ids
  can never move a path.

The adapter carries `SchemaDescription.catalog_version` through the
normalized snapshot, replay header, discovery plan, and `MappingContext`.
That catalog version is included in turn/session-complete fingerprints,
ownership metadata, and artifact provenance, so a catalog behavior change
cannot be treated as an unchanged trace. The filesystem database fingerprint
remains provenance-only and is deliberately not part of reconciliation or
artifact identity.

Real uploaded artifacts use `artifact-v2` and carry a typed provenance
envelope derived from the exact mapped request: source database fingerprint,
effective source/render/snapshot fingerprints, deterministic artifact path,
adapter/source schema, catalog version, mapping/artifact/importer versions,
and redaction mode. The database fingerprint is provenance only; it is
excluded from turn reconciliation identity and artifact-snapshot path
hashing.

Because the artifact path embeds the session-complete snapshot
fingerprint, a change to *any* turn of a session moves the artifact path
of *every* turn of that session — which is exactly why the artifact path
is recorded per-trace (`opencode.artifact_path`) and treated as an
integrity anchor, never as an equivalence field (the six turn-local
metadata fields are the equivalence contract; see
[result-schema.md](result-schema.md)).
The path is rejected before trace creation if its recorded MLflow tag value
would exceed 4096 characters.

## Bounded-memory strategy

No phase materializes a session or a turn as an in-memory value:

- Discovery streams headers and records one at a time, asserting
  monotonic session order instead of sorting a materialized list; turn
  summaries are yielded as soon as the next prompt closes a turn, and
  each completed turn's records are released before the next advances.
- The normalized snapshot writes rows in bounded batches.
- `build_replay_turn` returns a `ReplayTurn` whose three cursors
  (`iter_records`, `iter_session_records`, `iter_unassigned_records`)
  open **fresh snapshot queries** on every call and never retain a
  record list; root-value loaders return `BoundedRootValue`s that
  refuse (never truncate) a value above the byte bound.
- Artifacts (`content/artifacts.py`) stream section by section through
  the canonical serializer into a mode-`0o600` temporary file, in
  canonical key order; the streamed bytes equal `canonical_json` of the
  equivalent materialized document. Uploads read the staged file in
  fixed 8 MiB chunks (`DEFAULT_UPLOAD_CHUNK_BYTES`); verification
  streams downloads in fixed 64 KiB blocks.
- The mapping layer loads root values through `load_root_values`, which
  raises a safe `payload_too_large` **before trace creation** for a
  complete root value above the negotiated inline limit (16 MiB
  conservative default; 1 MiB negotiated from the pinned profile).
  Oversized *child* values are never truncated — the trace carries a
  digest/size reference (`{"$artifact_reference": ...}`) while the
  complete value stays in the artifact.

**Per-turn cursor cost trade-off.** The price of streamed payload handling
is re-reading: each replayed turn's cursors re-scan the session's
records by written order and identity, and the session-complete
artifact-snapshot pass re-reads every permitted session record for
*each* turn, so a session with *n* turns costs O(n) full-session passes
across a run (plus one fingerprint pass and one write pass per
artifact). The design deliberately trades that repeat I/O — against the
local temporary snapshot, never the live source. Replay retains only the
last requested session header and searches summaries without a run-wide
header/summary cache. Candidate verification retains one slot-based
full-field contract per child, not full `TraceChildRequest` objects or replay/
root payloads; mapped oversized values are already bounded artifact references
and are never truncated. Decision-time remapping is checked against those
contracts before mutation. Planning, reporting, identity sets, and mapping contexts
still grow with identifiers/turn counts, not complete transcript payloads.

**Measured scale limitation (not a complete live-import memory proof).**
The artifact-only stress test streams 105,000 records / a 241 MiB artifact
below 256 MiB Python heap. A separate 100,000-record test exercises the actual
SQLite exporter, snapshot, discovery, replay index, mapping, timing preflight,
and gateway parent/identity maps plus streamed artifacts with the complete
generated identity map below the same budget. Its span API discards
payloads to isolate importer-owned accumulating structures. The gateway maps
retain one handle/ID per child; this is O(spans), not a constant-memory bound.
The real MLflow SDK additionally retains complete span payloads until root
completion/export. Consequently these measurements **do not establish** a
<256 MiB complete live-MLflow import for 100k payload-bearing spans; that
literal full-path requirement remains a documented target limitation. Existing
real-server tests cover small complete traces and rollback, not that scale.

## MLflow gateway boundary

`mlflow/gateway.py` and `mlflow/capabilities.py` are the **only**
modules that import MLflow, and only through the public `mlflow` /
`mlflow.tracking` / `mlflow.artifacts` entry points (pinned by an
AST-scanning contract test). Everything else in the pipeline — mapping,
source, content, orchestration — is MLflow-free, and every gateway
operation flows through injectable seams (`MlflowClient`, the module-level
span API adapter, the artifact-repository factory, and an explicit test-only
server-version transport) so tests run against fakes with no network. The
production capability gate performs a read-only `GET
<tracking-base>/version` and requires exact client/server `3.17.2` agreement
before trusting profile-backed status coercion, named artifacts, retirement,
or the inline floor. The resolved public surface,
observed 3.17.2 behaviors (child spans must not carry `experiment_id`;
`LiveSpan.end` coerces non-ERROR status to `OK`; tag/metadata
truncation at 4096/250; the OTEL env hazard), the named-artifact
composition, and the capability gate are documented in
[compatibility.md](compatibility.md).

`mlflow/mapping.py` stays importable without MLflow: the spec types are
pure, and the request conversion imports gateway request types lazily
inside method bodies, so importing the mapper never loads `mlflow`.

## Cross-layer contracts worth knowing when reviewing

- **`PROTECTED_IDENTITY_FIELDS` has one home**: neutral `contracts.py`
  defines the structural identity/sequence field-name set. Protection is
  root/path exact at typed/raw-row envelope boundaries; the same names in
  arbitrary decoded transcript mappings receive no exemption. Source and
  content modules import the one definition rather than creating a
  content→source layering dependency.
- **`IDENTITY_REDACTOR` has one home**: `contracts.py` defines the
  single identity (no-op) redactor instance used by every production
  caller over replayed values (fingerprints, discovery, mapping).
  Replayed snapshot values are already effective (redaction ran before
  persistence), so no pass ever redacts twice; the redaction *mode tag*
  on fingerprints is what distinguishes runs, never a swapped redactor.
- **`ReplayTurn` has one home**: `source/replay.py` owns the protocol;
  `build_replay_turn` in `source/turns.py` returns the concrete
  snapshot implementation. No other module defines or re-exports a
  second `ReplayTurn`, and no cross-layer in-memory turn/list
  substitute exists.
- **Fingerprint domains are versioned and disjoint**: session-context,
  turn-source, render, and artifact-snapshot domains each carry their
  own header, so hash inputs cannot collide by concatenation; the
  catalog and canonical-JSON versions (`opencode-schema-catalog-v1`,
  `canonical-json-v1`) are recorded in artifacts and reports.

## Reliability invariants

- **Retries** (`retry.py`): bounded exponential backoff with jitter
  (default policy: 4 total attempts, 0.5 s initial, 2× multiplier, 30 s
  cap, 0.5 jitter fraction) applies only to *transient* failures —
  MLflow protocol `error_code`s in the 5xx/rate-limit vocabulary, or
  transport timeout/connection exception class names. Typed
  `ImporterError`s are permanent by contract; control-flow interruptions
  (`KeyboardInterrupt`/`SystemExit`) and unrecognized failures fail fast.
- **Advisory lock** (`locking.py`): normal runs acquire one
  non-blocking per-host lock keyed by a SHA-256 of the normalized
  tracking endpoint and exact experiment name (credential-free), under
  `XDG_RUNTIME_DIR` or a restrictive tempdir subdirectory. A held lock
  fails immediately with exit 2; a stale lock is never broken
  automatically (recovery procedure in [operations.md](operations.md)).
  Dry runs acquire nothing.
- **Per-item failure isolation**: a failing session/turn is reported as
  a failed item (exit 1) while later items continue; pre-write failures
  abort the whole invocation (exit 2) before any mutation.
- **Replacement ordering** (`mlflow/reconciliation.py`): verify →
  create → verify → upload → verify → record digest → retire old. A
  failed verification/upload rolls the **new** trace back and retains
  the old valid one; retirement failures after a verified replacement
  never delete the old trace on uncertainty.
