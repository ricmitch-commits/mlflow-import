# Design: OpenCode SQLite Session Importer for MLflow

**Date:** 2026-10-21  
**Status:** Draft for review  
**Source requirements:** `docs/superpowers/specs/2026-10-21-opencode-mlflow-import-requirements.md`  
**Working utility:** `opencode-mlflow`

## 1. Purpose and scope

This document turns the approved requirements into an implementation-ready
design for a standalone Python command-line importer. The importer reads an
OpenCode SQLite database, reconstructs logical conversation turns, and writes
those turns as MLflow traces grouped by the native OpenCode session ID.

The first release is a one-shot, read-only synchronization utility. It does
not connect to an OpenCode server, watch for changes, modify the source
database, create one MLflow run per session, or upload source data from
credential/account tables.

The design prioritizes:

1. source fidelity and raw-field preservation;
2. safe operation against a live WAL-mode database;
3. deterministic selection, ordering, fingerprints, and artifacts;
4. importer-owned reconciliation without touching other traces; and
5. bounded memory and actionable failures.

## 2. Architecture overview

The implementation is a layered streaming pipeline:

```text
CLI/configuration
        |
        v
preflight validation and capability checks
        |
        v
read-only SQLite snapshot
        |
        v
schema adapter -> canonical source records -> logical turns
        |
        v
selection and disk-backed discovery plan
        |
        v
MLflow ownership lookup and reconciliation plan
        |
        v
trace/span/artifact writes and verification
        |
        v
versioned report and exit status
```

The code is organized around independent boundaries:

| Boundary | Responsibility |
| --- | --- |
| CLI/configuration | Parse flags, normalize values, and construct immutable run configuration. |
| SQLite source | Establish a consistent read-only snapshot and expose adapters, not raw SQL. |
| Schema adapters | Read legacy/current tables and normalize source-specific rows. |
| Canonical model | Preserve source identity, order, timestamps, raw values, and warnings. |
| Selection/turns | Discover model usage, count turns, apply filters, and reconstruct boundaries. |
| Content/artifacts | Canonicalize, fingerprint, redact, and stream JSON artifacts. |
| MLflow gateway | Encapsulate all public MLflow client/API calls and capability checks. |
| Reconciliation | Locate owned traces, plan actions, and enforce safe replacement ordering. |
| Reporting | Produce safe human/JSON output and determine the exit status. |

The project targets **Python 3.11+**. It uses `argparse`, `sqlite3`,
`dataclasses`, `json`, `hashlib`, `datetime`, and `tempfile` from the standard
library, plus the MLflow Python client. An ORM is intentionally not used:
the two source schema generations, unknown fields, and raw JSON are first-class
concerns.

## 3. Execution phases

### 3.1 Configuration and validation

`ImportConfig` is an immutable, normalized representation of the command line.
The first release has one required subcommand, `import`:

```text
opencode-mlflow import --experiment-name NAME [options]
```

The top-level executable without a subcommand prints usage and exits with
status `2`; `--help` exits successfully. The `import` subcommand accepts the
required `--experiment-name`, optional `--db`,
`--tracking-uri`, time bounds, directory, model, minimum-turn threshold,
`--redact`, `--dry-run`, `--output`, and a standard verbosity/log-level option.

Validation rules are:

- ISO-8601 time bounds require an explicit offset or `Z` and are converted to
  UTC instants. Bounds are inclusive.
- `--min-user-turns` is a non-negative integer.
- Paths are normalized with expansion, absolute conversion, separator
  normalization, and `.`/`..` removal, without requiring existence or resolving
  symlinks.
- Directory matching uses exact path equality or a path-component descendant
  check; string prefixes are not sufficient.
- `--model` is normalized with surrounding whitespace removed and case
  preserved. Complete identities use `provider/model`. Explicit incomplete
  forms use `provider/` or `/model`, so an incomplete source value only matches
  the same available form.
- The default source database is normally
  `~/.local/share/opencode/opencode.db`.
- Tracking authentication and TLS settings come from standard MLflow client
  configuration. Credentials are never accepted in normal output or logs.

Invalid arguments fail before source or MLflow writes with exit status `2`.

### 3.2 Source snapshot and schema probe

The source layer validates the database path, opens SQLite using a URI with
`mode=ro` and `uri=True`, and keeps the `-wal` and `-shm` companions available.
It does not use `immutable=1`, because immutable connections can ignore WAL
contents. It sets `PRAGMA query_only=ON`, uses a bounded busy timeout, and
starts one deferred read transaction before creating a temporary snapshot.

The source transaction is streamed into a restrictive-permission temporary
normalized snapshot, not copied page-for-page. The schema adapters select only
explicit permitted columns/records and write canonical rows to two temporary
tables:

```text
snapshot_sessions(session_id, schema, canonical_session_row)
snapshot_records(session_id, order_key, source_table, source_id,
                  canonical_record, raw_permitted_fields)
```

`account`, `account_state`, `control_account`, and `credential` values are
never selected, and `session_share` is selected only through its schema-catalog
safe-column allowlist. Thus prohibited secret pages never enter the temporary
snapshot. In redacted mode, transcript values are redacted before they are
written to the snapshot; in verbatim mode, permitted transcript values remain
available with restrictive local permissions.

The normalized snapshot is populated from the live read transaction, so
committed WAL content is included and all permitted rows share one consistent
source view. The source connection is closed as soon as safe extraction
completes; discovery, capability checks, and MLflow writes never hold a reader
open on the live OpenCode database. The temporary snapshot is then opened
read-only with `query_only=ON`, and discovery/replay use one transaction over
that copy. Failure to create or fully populate the snapshot, including
disk-space or busy-timeout failure, is a preflight error with exit status `2`.

The snapshot directory is mode `0700`, files are mode `0600`, and cleanup is in
an outer `finally`/process-exit handler. Cleanup failures are reported without
exposing paths or content; no prohibited secret values are present to clean up.

The source layer never issues schema changes, checkpoints, vacuum operations,
or any other write-oriented SQLite operation. A failure to establish a
consistent snapshot is a preflight failure rather than a reason to perform
an inconsistent live read.

The schema probe classifies the database as:

- **current** when `session_v2` and its `session_message` structures are
  available;
- **legacy** when `session`, `message`, and `part` structures are available;
- **mixed** when both generations are available; or
- **unsupported/malformed** when required structures are absent or unusable.

For mixed databases, the adapter yields all current sessions and only legacy
sessions whose IDs are absent from `session_v2`. A matching current session is
authoritative; overlapping legacy rows are not merged into a second import.
The adapter reports suppressed overlap provenance and warnings where useful.

Each adapter has a versioned schema catalog covering required columns, joins,
timestamp units, JSON locations, and user-record classification. A database
whose layout cannot be matched to a catalog entry fails with a compatibility
diagnostic rather than being queried heuristically.

### 3.3 Discovery plan

Discovery is a streaming pass over compatible sessions. It computes session
creation time, normalized directory, model identities found throughout all
records, logical-turn count, source counts, warnings, and effective
fingerprints. It writes only a sanitized, disk-backed JSONL plan containing
identities and actions; prompt, response, tool, and source-code content never
enters the plan.

The plan allows the importer to complete source scanning and MLflow capability
checks before any MLflow mutation while keeping memory bounded. The write
phase replays the same temporary snapshot, not the live source database.

### 3.4 MLflow preflight and write phase

The MLflow gateway resolves the experiment by exact name. In normal mode it
creates a missing experiment through the MLflow client after required
capabilities have been checked. In dry-run mode it reports a planned creation
but does not create anything.

The gateway indexes importer-owned traces and calculates whether each selected
turn is an import, unchanged skip, replacement, or stale retirement. If a
planned operation requires a server capability that is absent, the entire
invocation aborts before any MLflow mutation with exit status `2`. Capability
failure is global preflight failure, not a per-item failure; this prevents a
partially imported experiment whose later items cannot be reconciled. Per-item
continuation and exit status `1` apply only after the complete capability
contract has passed and an individual write encounters an item-specific
failure.

During the write phase, each selected session is replayed from the source
snapshot. The replayed effective fingerprints must match the discovery plan.
A mismatch is a source-consistency failure, not permission to import an
unplanned payload.

Before experiment creation or owned-trace lookup, a normal run acquires a
non-blocking advisory lock keyed by the normalized tracking endpoint and exact
experiment name. The lock file contains no credentials. A held lock is a
pre-write operational failure with exit status `2`; dry-run does not acquire a
write lock. Concurrent invocations from different hosts are explicitly not
supported and must be serialized by the operator. If a target offers a
server-side idempotency key, the writer still uses it; if a caller bypasses the
lock, duplicate owned-trace identities are reported as a conflict and no
automatic retirement is performed.

## 4. Canonical source model

The source adapters expose a common protocol similar to:

```text
LiveSchemaAdapter
  schema_generation
  inspect() -> SchemaDescription
  export_sanitized_snapshot(destination) -> SnapshotDescription

SnapshotSource
  iter_session_headers() -> iterator[SessionHeader]
  iter_records(session_id) -> iterator[SourceRecord]
  iter_session_scoped_records(session_id) -> iterator[SourceRecord]
```

All SQL is parameterized and contained inside the adapter that owns the
schema. Adapters use explicit column allowlists rather than `SELECT *`, and
secret-bearing tables/columns are never queried for values.

The live adapter performs the export while the source read transaction is
active. `SnapshotSource` reads only the normalized temporary tables thereafter;
it never reconnects to the live database. This separates source safety from
discovery/replay and makes the complete snapshot boundary testable.

### 4.1 Session header

The normalized session header contains, when present:

- session ID, project/workspace ID, parent/fork IDs, slug, title, directory,
  and path;
- OpenCode version, agent, configured provider/model, lifecycle and archival
  state;
- creation, update, and lifecycle timestamps with original units retained;
- aggregate cost and token values;
- summary additions, deletions, files, and diffs;
- source schema generation and source database identity fingerprint.

Missing fields remain absent. The importer never invents user, Git, model, or
source values. The source database fingerprint is a non-secret SHA-256 of the
schema generation and available filesystem identity/statistics; it excludes the
absolute path and source content and is metadata only, not reconciliation
identity.

The header also retains the complete permitted source session-row map, including
additive or unknown columns, in the artifact envelope. Known fields are copied
into typed properties for selection and MLflow mapping; unknown columns remain
raw canonical fields.

### 4.2 Source record

Each permitted source record has:

- source table/generation and source type;
- source row/message/part ID and sequence value;
- normalized timestamp and original timestamp/unit;
- user/assistant/system/synthetic classification;
- provider/model, tool, finish, error, token, and cost fields where present;
- decoded structured values; and
- original raw JSON/text and raw row values for unknown or malformed fields.

An additive column, unknown JSON key, or unrecognized record type produces an
`unknown_source_field` warning identified by schema/table/record identity and
field name, while retaining the value in the artifact. Malformed JSON or
unsupported scalar decoding produces a separate `malformed_source_value`
warning. Warning messages never include the field value itself.

Allowed session-scoped event, todo, instruction, and similar records are
associated only when their session relationship is unambiguous. Unassociated
records are reported rather than silently assigned.

The following values are excluded from uploaded source artifacts:

- all values from `account`, `account_state`, `control_account`, and
  `credential` tables; and
- secret-bearing `session_share` fields. A versioned adapter safe-column
  allowlist may retain non-secret identity/lifecycle columns; if no catalogued
  safe layout exists, the adapter omits the table and reports that omission.

Their presence cannot affect selection or turn counts.

### 4.3 Ordering and timestamps

Adapters normalize known source timestamp units to timezone-aware UTC
instants, retaining the original value and unit. Record ordering is:

1. explicit source sequence, including current-schema `seq`;
2. normalized source timestamp;
3. deterministic source-type rank; and
4. stable source ID.

The source-type rank is a fixed adapter/catalog constant, not an incidental
database query order, and is recorded with the adapter version in the artifact.

Session ordering is normalized creation time followed by session ID; sessions
without a normalized creation time sort after timestamped sessions and then by
session ID. Turn ordering is source order, timestamp, and stable source ID.

If `session.time_created` is missing or invalid, the session remains eligible
when no time filter is supplied, with a warning. When either time bound is
supplied, the session is excluded because its required comparison instant is
unknown; the report records `missing_session_creation_time` rather than
guessing from a message timestamp. A source timestamp unit or value that
cannot be normalized is handled the same way.

### 4.4 Cross-table merge and deduplication

The adapter emits one canonical record for each source row and never emits a
joined parent once per child. Current-schema `session_message` rows use their
per-session `seq` as the primary order domain. Nested content items receive a
stable child ordinal after their parent and use a derived source identity such
as `<session_message_id>#content-<index>`; nested user content is not another
turn start. Legacy `message` rows are emitted once, and their `part` rows are
attached immediately after the parent using part sequence, then part
timestamp, then part ID. A part is never emitted before its parent.

The fixed tie ranks are:

```text
10  authoritative message/session-message container
20  persisted or nested content part
30  lifecycle/session event
40  todo/instruction record
90  unknown permitted record
```

Within one authoritative stream, explicit sequence is primary. Across streams
without a shared sequence domain, normalized timestamp is primary, followed by
the fixed rank and stable source ID. If timestamps are absent, the adapter's
derived parent/child order and rank are used. Duplicate identity is
`(schema_generation, source_table, source_id)`; duplicate joins are collapsed
with a warning, while identical records with distinct source IDs remain
distinct. Legacy/current overlap is suppressed before this merge and is never
deduplicated by content.

## 5. Selection and logical-turn reconstruction

### 5.1 Session filters

Filters are evaluated against the canonical session/discovery summary and
combine with logical AND:

| Filter | Semantics |
| --- | --- |
| Time | Inclusive bounds on session `time_created`, never latest message or `time_updated`. |
| Directory | Requested directory or any normalized descendant with a component boundary. |
| Model | Exact normalized identity found in any message, session-message, or part record. |
| Minimum turns | Strictly greater than `N`; exactly `N` turns is excluded. |

An omitted filter imposes no restriction. With no optional filter, human output
warns that all compatible sessions may be selected and recommends a dry run.

If a session qualifies, every logical turn in that session is imported. The
minimum-turn option is a session eligibility filter, not a request to import
only turns beyond the threshold.

### 5.2 Model identity matching

Complete model identities are formed as `providerID/modelID` after trimming
surrounding whitespace and preserving case. Source records with only a
provider normalize to `provider/`; records with only a model normalize to
`/model`. A requested incomplete form must use the same explicit trailing or
leading slash. A complete request never matches an incomplete source value.

The discovery scan collects identities from session defaults, legacy messages,
legacy parts, current session-message rows, and nested decoded model fields.

### 5.3 Turn boundaries

A persisted, non-synthetic user prompt starts a logical turn. The turn includes
all following assistant, model, tool, reasoning, text, patch, file, lifecycle,
and other records until the next persisted user prompt. System, synthetic,
tool-only, lifecycle, and compaction records do not start a turn.

Records before the first user prompt remain session-level unassigned records,
are included in the canonical artifact when permitted, and do not create an
empty synthetic trace. A session with no logical user turns is reported as
having no importable turns. If it previously had importer-owned turns and is
selected with a complete scan, those stale turns are eligible for retirement.

Each turn has:

- a 1-based `turn_index` for reporting and metadata;
- a stable `turn_key` derived from the session ID and first user record ID;
- first-user/start and last-record/end timestamps;
- terminal outcome and source counts; and
- a turn-local source fingerprint.

The root output is selected deterministically from the last non-synthetic
assistant result in source order. If a result is split into parts, the ordered
user-visible text/content parts belonging to that final assistant result are
combined for root output while every part remains separately represented in
child spans and the artifact. Earlier assistant results remain children. A
turn with no assistant result has null/omitted root output and receives
`opencode.outcome=no_assistant_result` with `UNSET`, unless the source contains
an explicit successful terminal outcome that has no assistant payload; that
explicit source status is preserved. Consecutive user prompts therefore end
the previous turn without manufacturing output and start the next turn.

The normal key formula is
`turn-<sha256(session_id || "\\0" || first_user_source_type || "\\0" || first_user_source_id)>`.
When the first user record has no source ID, the formula uses its explicit
sequence or timestamp plus its canonical effective record; the 1-based ordinal
is the final tie-breaker when all of those values are absent. This fallback is
documented in the artifact so a consumer can reproduce the key.

If two turns would produce the same normal key, the adapter adds the stable
occurrence ordinal to the hash input and emits a `turn_key_collision` warning;
it never merges the turns. Exact duplicate fallback records use their ordered
occurrence index. Repeating the same snapshot therefore reproduces the same
keys, while a source that provides no distinguishing identity may legitimately
renumber later duplicates when its order changes.

Fork sessions include only records actually persisted for the child. The
importer does not query a parent to synthesize history or prepend parent
turns. Parent/fork IDs, fork boundaries, and persisted copied rows remain
available in the canonical artifact.

Filtered reruns are non-destructive outside their selection. Stale retirement
is considered only for selected sessions whose complete source scan succeeded.
The report records the number of owned traces outside that reconciliation scope
as `unexamined_owned_traces`; those traces are explicitly left untouched.

## 6. Content, fingerprints, and artifacts

### 6.1 Canonical serialization

Canonical JSON is UTF-8 with sorted object keys, fixed separators, stable
numeric/string normalization, and an explicit artifact schema version. Each
record stores decoded values where possible and original raw JSON/text where
available. Thus valid structured data remains inspectable while malformed or
unknown fields are not silently discarded.

The versioned canonicalization contract is `canonical-json-v1`:

- object keys are sorted by their UTF-8 byte sequence; strings retain their
  original Unicode code points and are JSON-escaped with UTF-8 output;
- `null`, booleans, and integers use their JSON scalar forms; finite real
  values use RFC 8785 shortest-round-trip number formatting (`-0` becomes
  `0`), while non-finite real values use a typed wrapper with the original
  spelling;
- SQLite BLOBs and invalid-UTF-8 byte sequences use a typed base64 wrapper;
- SQLite `NULL` is JSON `null` and absent columns remain absent;
- normalized instants use integer epoch nanoseconds in fingerprint material,
  while the original timestamp and unit remain in the artifact;
- JSON objects with duplicate keys retain the original raw text and decode to a
  typed ordered-pairs wrapper rather than silently choosing one value; and
- every raw field is represented either as a decoded value plus raw source text
  or as a typed raw-value wrapper, so the serializer is lossless in verbatim
  mode.

The canonicalizer has fixed golden vectors for each scalar/wrapper case. The
effective redaction pass runs before the canonicalizer output is written in
redacted mode; the same effective values feed fingerprints and artifacts.

Each imported turn receives a self-contained canonical artifact containing:

- the source session row;
- all permitted message, part, session-message, and associated session-scoped
  records visible in the source snapshot;
- normalized logical-turn boundaries and unassigned-record information;
- the source-to-MLflow trace/span identity map for the import operation;
- importer, adapter, artifact, and mapping versions;
- source database fingerprint and schema generation; and
- redaction mode and warning summary.

The complete session snapshot is intentionally repeated in each turn artifact
so a trace remains independently inspectable. It is streamed to a temporary
file and is not held as one in-memory object. A session manifest may be
generated once and reused as an upload source, but each trace receives the
artifact path in its own trace artifact namespace.

The canonical source payload and artifact path are deterministic. The identity
map contains generated MLflow trace/span IDs, so it is deliberately excluded
from **all** source, render, and artifact-snapshot fingerprint input; the
fingerprint serializer uses a fixed `identity_map_pending` placeholder and
generated IDs are added after trace creation. This preserves a deterministic
path while still recording the source-to-MLflow mapping required for
inspection.

Relative artifact paths are deterministic and namespaced:

```text
opencode-mlflow/v1/
  sessions/<encoded-session-id>/
    turns/<stable-turn-key>/<render-fingerprint>-<artifact-snapshot-fingerprint>.json
```

The path uses a safe encoded session ID and a fingerprint rather than an
MLflow-generated trace ID. Replacements therefore never overwrite an older
trace's artifact.

The reconciliation fingerprint is turn-local so changing one turn replaces
only that turn's importer-owned trace. A separate artifact-snapshot fingerprint
may change when other session records change; that affects the new artifact's
path but does not, by itself, force unchanged turn traces to be replaced.

Fingerprint inputs are explicit. A `session_context_fingerprint` includes only
session identity/selection context: session ID, project/workspace/parent/fork
IDs, slug, title, directory/path, agent, configured model, OpenCode version,
schema generation, and creation instant. A turn `source_fingerprint` includes
`canonical-json-v1`, the adapter/schema version, that context fingerprint,
session-scoped records unambiguously assigned to the turn, the selected turn's
ordered records, and the effective redaction mode.

Mutable session snapshot fields—`time_updated`, lifecycle/archival state,
aggregate cost/token totals, summary additions/deletions/files/diffs, logical
turn count, and counts of other turns—are trace metadata and artifact content
but are deliberately excluded from the turn source fingerprint. Thus a change
to one turn's content replaces only that turn; a change to session identity or
selection context intentionally replaces all affected turn traces. The
artifact-snapshot fingerprint includes all permitted records and all session
snapshot fields for the complete session, including records assigned to other
turns and unassigned records. The artifact-snapshot hash is computed from the
complete canonical session payload before the generated identity-map section
is inserted, so generated IDs cannot change the artifact path.

Artifacts are immutable historical snapshots of the source as observed when
their trace was written; they are not a live session manifest. Consequently,
when one turn changes, its replacement contains the newer complete-session
snapshot while unchanged turn traces retain the snapshot that produced them.
The artifact-snapshot fingerprint makes that distinction visible. This is
intentional: changing one source turn replaces only its importer-owned trace,
while every artifact remains self-contained and auditable.

The bounded-memory contract is `O(header + largest active source value + upload
chunk)`, not `O(database or session)`. JSON artifacts are written incrementally
to restrictive temporary files and uploaded in fixed chunks. The gateway
capability probe reports the maximum inline MLflow payload; if it cannot accept
a complete root input/output value, the item is rejected with
`payload_too_large` before trace creation rather than truncated. When a child
span attribute exceeds the inline limit, the complete value remains in the
artifact and the span receives a digest/reference. A conservative 16 MiB
inline limit applies when a target reports no limit. A stress fixture with at
least 100,000 records and a multi-hundred-megabyte artifact must keep Python
heap growth below 256 MiB over the empty-process baseline while preserving the
full artifact.

### 6.2 Redaction

Verbatim content is the default. `--redact` applies the same deterministic
redactor to root inputs/outputs, child spans, metadata values, error strings,
and artifacts. The replacement marker is exactly `[REDACTED]`.

Version-one detector contracts are:

| Detector | Match |
| --- | --- |
| `pem_private_key` | `-----BEGIN ... PRIVATE KEY-----` through the matching `-----END ... PRIVATE KEY-----` block. |
| `auth_header` | Case-insensitive `Authorization`/`Proxy-Authorization` bearer or basic value, including standalone `Bearer <token>` forms. |
| `structured_auth_field` | Structured `Authorization`/`Proxy-Authorization` keys whose scalar value begins with bearer/basic authentication. |
| `secret_key_value` | Structured or textual values assigned to case-insensitive keys matching `api[-_]key`, `api[-_]token`, `access[-_]token`, `refresh[-_]token`, `auth[-_]token`, `session[-_]token`, `token`, `access[-_]key`, `client[-_]secret`, `secret`, `credential`, `password`, `passwd`, `private[-_]key`, or `bearer`. |
| `connection_secret` | Password/user-info in a URI or connection string between `scheme://user:` and `@`. |
| `known_token_prefix` | Recognizable `sk-`, `ghp_`, `github_pat_`, `xox[baprs]-`, and AWS `AKIA` access-key forms with their documented minimum lengths. |

The concrete regular expressions for these named detectors are constants in
the implementation and have positive/negative fixture tests. Their required
forms are:

```text
pem_private_key     = -----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[\\s\\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----
auth_header         = (?i)(?:authorization|proxy-authorization)\\s*:\\s*(?:bearer|basic)\\s+[A-Za-z0-9._~+/=-]+
standalone_bearer   = (?i)\\bbearer\\s+[A-Za-z0-9._~+/=-]{16,}\\b
structured_auth_key = (?i)^(?:authorization|proxy-authorization)$
structured_auth     = key=structured_auth_key && value=(?i)^(?:bearer|basic)\\s+.+$
secret_key_name     = (?i)^(?:api[-_]?key|api[-_]?token|access[-_]?token|refresh[-_]?token|auth[-_]?token|session[-_]?token|token|access[-_]?key|client[-_]?secret|secret|credential|password|passwd|private[-_]?key|bearer)$
connection_secret   = (?i)\\b([A-Za-z][A-Za-z0-9+.-]*://[^/\\s:@]+:)[^@\\s/]+(@)
known_token_prefix  = (?i)\\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16})\\b
```

Structured secret-key values are masked recursively while preserving field
names. The structured detector takes precedence over textual regex matching,
and the replacement preserves surrounding delimiters and the `Bearer`/`Basic`
scheme where possible. Positive and negative fixtures cover JSON, YAML-like
key/value text, and HTTP-header forms.

The transformation pipeline is fixed:

1. decode a source value while retaining its raw representation;
2. recursively redact structured mappings by secret-key and structured-auth
   rules, while copying protected identity fields through unchanged; opaque
   BLOB/invalid-byte values become `[REDACTED_BINARY]` unless the field is a
   protected identity;
3. for valid JSON raw text, serialize the already-redacted decoded value;
4. for malformed/unknown raw text, scan textual assignments using the key
   pattern above: after `:` or `=`, a quoted value ends at its matching
   unescaped quote, and an unquoted value ends at whitespace, comma, semicolon,
   `}`, or `]`; apply the header/PEM/connection/token patterns to the remaining
   text; and
5. emit only the redacted decoded/raw envelope to traces, plans, and artifacts.

The original raw text is never retained in a redacted envelope or temporary
upload file. A separate protected identity field carries source IDs needed for
reconciliation, and redaction counts are accumulated without retaining match
locations or values.

Source/session/message/part IDs, sequence values, turn keys, schema names, and
other reconciliation identifiers are protected from masking. Raw record text
may be redacted, but the same identity is retained in a protected envelope
field. Redaction reports detector categories and affected field/record counts,
never matched text.

`opencode.source_fingerprint` is a streaming SHA-256 of the canonical effective
turn representation, including relevant session-level metadata, after
secret-table exclusion and, when enabled, redaction. This avoids exposing
hashes of masked secrets. The render fingerprint is
`sha256(source_fingerprint || artifact_schema_version || mapping_version ||
importer_version || redaction_mode)`. The artifact-snapshot fingerprint hashes
the complete self-contained session artifact and is used for the artifact path,
not unchanged-trace comparison. Switching between verbatim and redacted
imports therefore creates a replacement rather than an incorrect unchanged
skip.
Reconciliation separately compares `source_schema` and the source fingerprint.

Verbatim artifacts are raw-preserving. Redacted artifacts are intentionally
not lossless with respect to detected secret values; their artifact manifest
records the redaction mode and detector counts instead of retaining an
unredacted alternate representation.

## 7. MLflow mapping and gateway

### 7.1 Public API boundary

Only `mlflow.MlflowClient` and supported public trace/artifact APIs are called
outside the MLflow gateway. The gateway hides client/server version differences
behind a narrow interface and performs an explicit capability probe. Direct
backend-store writes, run-only fallback, and undocumented mutation endpoints
are prohibited.

The capability contract covers experiment lookup/creation, trace creation,
native session metadata, nested spans or typed events, full input/output
payloads, trace metadata, trace search, trace artifact upload/verification,
and trace retirement/deletion for planned stale/replacement actions and
rollback of any partially created trace.

The full-payload check uses trace retrieval or trace artifact data rather than
the potentially truncated search preview. A target that cannot preserve and
retrieve the complete supplied root values fails the fidelity contract.

The gateway exposes a behavioral `TraceArtifactSink` contract with
`upload_named_json(trace_id, relative_path, local_file)` and verification of
that relative path. The implementation may use a public named trace-artifact
API or a public trace attachment API only when the target preserves the
requested deterministic relative name. A UUID-only attachment mechanism does
not satisfy this contract. The compatibility check is behavioral rather than
only a numeric version check because the client and tracking server can differ;
the implementation release must pin and integration-test at least one
client/server pair that satisfies this contract. Unsupported targets fail
preflight with status `2`; a run-artifact fallback is not permitted.

When dry-run targets an experiment that does not yet exist, the probe checks
client methods, server reachability, protocol/version responses, and configured
artifact capability without creating a temporary experiment or uploading a
probe file. If the target cannot prove the required contract without a
mutation, the capability check fails rather than mutating MLflow during dry
run.

If the target cannot provide the required trace/artifact fidelity, the command
exits with status `2` rather than silently degrading to runs or discarding
source data.

The initial compatibility profile is the MLflow `3.17.2.dev0` trace API shape
used by the inspected local MLflow source, with the client and tracking server
from one tested build. This version is a trace/API reference, not a claim that
the stock build already satisfies the complete importer profile; the named
artifact operation remains mandatory. The gateway contract is pinned to these
public methods and fields:

| Operation | Required public API/field behavior |
| --- | --- |
| Experiment | `MlflowClient.get_experiment_by_name` and `create_experiment`. |
| Root trace | Public `mlflow.start_span_no_context(name, span_type, inputs, attributes, tags, metadata, experiment_id, start_time_ns)` with `metadata["mlflow.trace.session"]` and ownership fields; returned `LiveSpan.trace_id` and `span_id`. |
| Child span | Public `mlflow.start_span_no_context(..., parent_span=root_or_parent_span, ...)` with full attributes/inputs and source `start_time_ns`. |
| Completion | Public `LiveSpan.end(outputs, attributes, status, end_time_ns)` for children and root. |
| Session/ownership | Searchable trace metadata containing `mlflow.trace.session` and all `opencode.*` fields, supplied at root-span creation; updates use public trace-tag/metadata methods where required. |
| Lookup | `MlflowClient.search_traces` scoped to experiment, paginated, with metadata filtering or complete page results; `get_trace` must retrieve full payloads. |
| Retirement | Public `MlflowClient.delete_traces` or the server's documented equivalent with the same soft-deletion semantics. |
| Canonical artifact | A public `TraceArtifactSink`-compatible named upload and verification operation as defined above. |

The numeric profile is a compatibility baseline, not permission to use private
MLflow internals. A server/client pair that lacks any listed behavior,
including named canonical artifact upload, is unsupported and causes global
preflight status `2`. Supporting a newer released MLflow pair requires a new
compatibility-profile test before changing the package constraint.

The first implementation task is a compatibility probe against this profile;
the writer is not authorized to substitute private artifact-repository calls or
run artifacts if the named trace-artifact operation is absent. The profile is
therefore an explicit implementation prerequisite, not an assumption hidden in
the writer.

The inspected local MLflow source establishes the public trace/span/session
metadata calls above but does not, by itself, establish a public arbitrary
named trace-artifact upload. That capability is a hard release gate: until a
concrete client/server pair exposes and passes the `TraceArtifactSink` contract,
the importer is unsupported and no lower-fidelity implementation is authorized.

### 7.2 Experiment

The requested experiment name is resolved exactly. An existing experiment is
reused without changing its name, artifact location, or unrelated tags. A
missing experiment is created through the client in normal mode and reported
with its resolved ID before trace writes begin. Dry-run reports a planned
creation without mutation.

### 7.3 Trace and root span

Every selected logical turn produces one trace. The exact OpenCode session ID
is set as the native MLflow trace session identifier (visible as
`mlflow.trace.session` on the inspected server) and is also stored in
`opencode.session_id`.

The root request/input is the first user prompt. The root response/output is
the final assistant result, with complete values passed to the creation API;
search/UI truncation is not treated as data loss. All assistant/model/tool
content remains available in child spans and the artifact.

Root metadata includes, when present:

- title, slug, directory, path, project/workspace/parent/fork IDs;
- agent, configured model, OpenCode version, and source schema;
- session creation/update/lifecycle timestamps;
- session cost/token aggregates and summary counts;
- logical turn count and source message/part counts;
- source database fingerprint without an absolute local path;
- deterministic canonical artifact path and artifact snapshot fingerprint;
- source message IDs, sequence values, timestamps, turn index/key; and
- native user, source, and Git fields only when supplied by OpenCode.

If no final assistant result exists, the root output is omitted/null and the
source terminal outcome remains non-OK when the turn is incomplete or failed.

Large metadata values are represented by a digest and artifact reference while
the complete value remains in the canonical artifact.

Session aggregates and mutable lifecycle fields are values from the snapshot
that produced this trace. They are not retroactively rewritten on unchanged
turn traces when another turn changes; the artifact-snapshot fingerprint makes
that historical boundary explicit.

### 7.4 Child spans and events

The mapping is:

| Source information | MLflow representation |
| --- | --- |
| Model/provider request and response | LLM/model child span with model, input/output, timing, finish state, tokens, and cost. |
| Assistant execution step | Assistant/agent child span with source identity and timing. |
| Tool call and result | Tool child span with tool name, call ID, input/output, status, timing, and error data. |
| Reasoning/text/patch/file part | Typed content child span or event with part ID/type, sequence, content, and timing. |
| Lifecycle record | Lifecycle event/span with source type, outcome, and timestamp. |
| Unknown record | Opaque child span/event with source type/ID and artifact reference when attributes are too large. |

Native MLflow token/cost attributes are populated where supported; namespaced
source attributes are also retained. Trace-level totals come from source
aggregates, never from truncated input/output previews.

Trace start time is the first user-record timestamp when available, otherwise
the first timestamp anywhere in the turn, then the source session creation
instant. Trace end time is the last record/completion timestamp, or the
source-derived start time when no later timestamp exists. Each fallback marks
`opencode.timing_estimated=true`; the writer never uses the local wall clock to
invent source timing. If all source timestamps are absent, the trace omits
start/end times when the target permits it; a target that requires invented
times fails that item rather than fabricating them.

Successful terminal outcomes map to `OK`, explicit failures to `ERROR`, and
interrupted/incomplete outcomes to `UNSET` plus a distinguishable
`opencode.outcome` and error/lifecycle metadata.

## 8. Ownership and reconciliation

An importer-owned trace must contain the following fields:

```text
opencode.importer          = opencode-mlflow
opencode.session_id        = <OpenCode session ID>
opencode.turn_index        = <1-based ordinal>
opencode.turn_key          = <stable first-user-derived key>
opencode.source_fingerprint = <effective turn fingerprint>
opencode.source_schema     = <legacy|current>
opencode.importer_version  = <package version>
opencode.mapping_version   = <mapping contract version>
opencode.redaction_mode    = <verbatim|redacted>
opencode.render_fingerprint = <artifact/render fingerprint>
opencode.operation_key     = <sha256 experiment/session/turn/render key>
```

The lookup key is `(opencode.session_id, opencode.turn_key)`, never the native
MLflow trace ID. The gateway searches only traces with the exact ownership
marker and paginates results. Server-side metadata filters are used when
available; otherwise owned results are filtered page by page without mutating
unmarked traces.

All ownership and operation fields are supplied at trace creation, before the
trace is ended, so a process crash leaves an identifiable owned candidate. For
each source identity, reconciliation handles multiple owned matches
deterministically:

1. verify artifacts and effective fingerprints for every candidate;
2. if one or more candidates exactly match the current source/render contract,
   select the newest creation timestamp, breaking ties by lexicographically
   greatest MLflow trace ID;
3. if no candidate matches, create and verify one replacement; and
4. after a verified winner exists, retire every other importer-owned candidate
   for that identity, including incomplete candidates left by a prior crash.

An owned candidate with missing/unverified artifacts is never treated as an
unchanged skip. If its cleanup fails, the old valid trace remains and the
report contains `replacement_cleanup_failed` with all affected trace IDs.
This recovery rule makes a crash before or after old-trace retirement
converge on one verified current candidate on the next run.

Actions are:

1. **imported**: no matching owned trace exists;
2. **skipped/unchanged**: identity and effective render contract match;
3. **updated**: identity exists but content or render contract changed;
4. **retired**: an owned trace is no longer present in a complete selected
   session scan; or
5. **failed**: the item could not be written or safely reconciled.

Replacement ordering is strict:

1. create the new trace and child spans;
2. end and retrieve/verify its metadata, status, timing, and IDs;
3. upload and verify deterministic artifacts;
4. only then retire the previous owned trace.

If creation or verification fails, the old trace remains intact. If a new trace
was created but span verification or artifact upload fails, the writer first
retires that new trace as a rollback and reports the original failure. If
rollback itself fails, it retries within the same bounded policy, reports both
trace IDs as a reconciliation conflict, and still never retires the old trace.
The new trace remains importer-owned and therefore discoverable for explicit
operator recovery; it is never treated as an untracked fallback.

If retirement fails after a verified replacement, bounded retries and
compensating retirement of the new trace are attempted; the old trace is never
deleted on uncertainty.
If retirement is unsupported, the affected replacement/stale action fails
safely before creating an untracked duplicate. Because rollback is required
after any partial create, a normal write also requires retirement capability
even when the current plan contains only initial imports.

Retirement is implemented as the target's documented public trace deletion or
soft-deletion operation (for example, `MlflowClient.delete_traces` where
supported). It is not a mutable replacement tag and does not separately purge
artifact storage. A target that cannot retire a trace is compatible only for
dry-run inspection; normal writes fail global preflight status `2`.

Existing traces without the ownership marker are never changed, even if their
native session ID matches an imported trace. A filtered run never retires
traces belonging to sessions it did not select.

## 9. Reliability, privacy, and reporting

### 9.1 Retries and failure isolation

Transient network, timeout, rate-limit, and 5xx failures use bounded
exponential backoff with jitter. Invalid requests and authorization failures
are not retried indefinitely. Trace creation uses a deterministic client
request/idempotency key when supported. After an ambiguous timeout, the writer
searches for the exact owner/session/turn/fingerprint tuple before retrying a
non-idempotent create.

Malformed source JSON produces a warning with source identity and permitted raw
data in the artifact. A source schema/configuration failure before writes is
exit `2`.
An individual session/turn failure is reported and does not prevent later
items from being attempted; any item failure produces exit `1`.

### 9.2 Security and local temporary data

Standard MLflow authentication and TLS verification are used. Credentials,
tracking-URI secrets, prompts, responses, tool output, source code, and
redaction matches are excluded from progress output, exception messages, and
JSON diagnostics.

Temporary canonical files use restrictive permissions and are deleted after
upload/verification on a best-effort basis. In redacted mode, the upload path
receives only the redacted serialization; an unredacted artifact is never
created for upload under another name. Verbatim mode is explicitly documented
as capable of sending sensitive transcript content to MLflow, with `--redact`
recommended for shared or production tracking servers.

### 9.3 Report schema

JSON output is versioned as `opencode-mlflow.result.v1` and contains:

- invocation mode, dry-run flag, and resolved configuration summary without
  credentials or absolute source path by default;
- experiment name, ID when available, and planned/actual creation status;
- source schema and non-secret database fingerprint;
- redaction enabled flag and affected field/record counts;
- selected/imported/updated/unchanged/retired/skipped/failed counts; and
- per-session and per-turn source identities, planned/actual action, MLflow
  trace ID when available, warnings, and safe error code/message.

The normative action enum is exactly:
`imported | updated | skipped | failed | retired`.
`unchanged` is not an action; it is a summary subcount of `skipped` items whose
`reason` is `unchanged`. `planned_action`, when present for dry-run output, uses
the same enum. Every turn result contains `session_id`, `turn_key`,
`turn_index`, `action`, nullable `trace_id`, and either a null error or an error
object with a stable `code` and sanitized `message`. Stable error codes include
`invalid_argument`, `source_unavailable`, `source_snapshot_failed`,
`unsupported_schema`, `mlflow_unavailable`, `mlflow_capability_unsupported`,
`concurrent_import`, `source_consistency_mismatch`, `payload_too_large`,
`trace_create_failed`, `trace_verify_failed`, `artifact_upload_failed`,
`replacement_cleanup_failed`, `duplicate_owned_identity`, and
`retirement_failed`.

Each session result contains `session_id`, `action`, `selected`,
`logical_turn_count`, `turns`, `warnings`, and `error` (nullable). The top-level
result always contains `schema_version`, `dry_run`, `experiment`, `source`,
`redaction`, `summary`, `sessions`, and `errors`.

A stale-retirement turn result uses the stored owned-trace `turn_key` and
`turn_index` when available, sets `source_present=false`, and includes the
retired trace ID. This keeps retirement diagnostics identifiable without
inventing a source record.

Summary units are explicit and do not mix sessions with turns:

- `selected_sessions` counts eligible session headers;
- `selected_turns` counts eligible logical turns;
- `imported_turns`, `updated_turns`, `skipped_turns`, and `failed_turns` are
  disjoint current-turn results;
- `unchanged_turns` is a subset of `skipped_turns`;
- `retired_traces` counts old/stale trace-retirement operations, including an
  old trace retired as part of a successful update; and
- `unexamined_owned_traces` counts importer-owned traces outside the selected
  reconciliation scope and never implies retirement.

The session-level `action` is `failed` if any turn failed, `updated` if the
session had any update or a mixture of successful import/retirement actions,
`imported` if all current turn actions were imports, `retired` if only
retirements occurred, and `skipped` if all turns were skipped or the session
had no importable turns. A dry-run uses `action=skipped`,
`reason=dry_run`, and records the predicted operation in `planned_action`.

Dry-run records the predicted action in `planned_action` and uses a non-mutating
actual action/reason so automation can distinguish a plan from completed work.
Human output remains concise and progress-oriented.

Exit status contract:

| Status | Meaning |
| --- | --- |
| `0` | Valid invocation; all selected work completed or was an intentional no-op. |
| `1` | One or more sessions/turns failed or could not be reconciled. |
| `2` | Invalid arguments, unavailable/unsupported source, or MLflow configuration/capability failure before import. |

For status `2` caused by a capability/configuration failure, the report contains
planned selection information but no experiment, trace, retirement, or
artifact mutation has occurred.

## 10. Package layout

```text
src/opencode_mlflow/
  cli.py
  config.py
  errors.py
  importer.py
  reporting.py
  retry.py
  source/
    connection.py
    probe.py
    snapshot.py
    model.py
    legacy.py
    current.py
    ordering.py
    turns.py
    selection.py
  content/
    canonical.py
    fingerprint.py
    redaction.py
    artifacts.py
  mlflow/
    gateway.py
    capabilities.py
    mapping.py
    reconciliation.py
```

The exact file split may change during implementation, but SQL ownership,
source normalization, content policy, MLflow access, reconciliation, and
reporting remain separate responsibilities. The public console entry point is
`opencode-mlflow`.

## 11. Verification strategy

### 11.1 Unit tests

Unit tests cover:

- CLI validation, timestamp parsing, path normalization, model forms, and
  strict turn thresholds, including the required `import` subcommand;
- selection boundary cases: sessions exactly at inclusive `from`/`to` bounds,
  a session outside the creation-time range whose message is inside it, a
  descendant directory, and a similarly prefixed sibling directory;
- schema detection and timestamp-unit normalization;
- deterministic cross-table ordering/deduplication and stable turn keys;
- canonical JSON, raw/malformed field preservation, unknown-field warnings, and
  fingerprints;
- fingerprint isolation: changing one turn plus derived mutable session totals
  changes only that turn's source fingerprint, while changing session identity
  context changes every affected turn fingerprint;
- every redaction detector plus protected identifiers, including structured
  and textual authorization values in JSON/YAML/header forms, generic token
  assignments, and opaque byte/BLOB fields;
- selection AND semantics and non-destructive filtered reconciliation;
- MLflow mapping, status/timing/token/cost aggregation, exact report enums,
  count units, and report sanitizing;
- missing session/record timestamps, consecutive user prompts, missing
  assistant results, duplicate source IDs, and indistinguishable fallback
  turn keys; and
- canonicalization vectors for NULL, BLOB, invalid UTF-8, non-finite real,
  duplicate JSON keys, timestamp units, and unknown session columns.

The CLI boundary test opens a malformed/unsupported database and asserts exit
status `2`, a compatibility diagnostic, and zero MLflow mutations. The same
assertion is made for a source snapshot/backup failure.

### 11.2 Source fixtures

Sanitized synthetic SQLite fixtures cover:

- current, legacy, mixed/overlap, fork, and compaction records;
- model identity at session, message, session-message, and part levels;
- unknown fields and malformed JSON;
- missing timestamps and lifecycle/error outcomes;
- secret-bearing tables/fields and unassociated session records; and
- WAL-mode reads while another connection is active.

The WAL test asserts that committed uncheckpointed WAL rows are visible in the
temporary normalized snapshot, that rows committed after snapshot creation are
absent, that discovery/replay observe the same snapshot, and that the source
database, WAL, and SHM file hashes/metadata are unchanged. It asserts that
sentinels from credential/account/session-share secret columns are absent from
the temporary snapshot and that the snapshot directory is cleaned up. It also
exercises the bounded busy-timeout failure path. A scale fixture with at least
100,000 records and a multi-hundred-megabyte artifact asserts the 256 MiB
heap-growth bound and verifies complete artifact bytes.

The live OpenCode database, local MLflow logs, and SQLiteBrowser project files
are research inputs only and are never committed as fixtures or used as runtime
dependencies.

### 11.3 MLflow integration tests

A fake gateway tests replacement ordering, idempotency/ambiguous-create
handling, retry classification, stale retirement, artifact verification, and
protection of non-importer traces.

Failure tests inject crashes/failures after trace creation, after span
verification, after artifact upload, and immediately before old-trace
retirement. A rerun must converge to one verified current owned trace and
retire incomplete/stale owned candidates without touching an unowned trace.
Tests also assert that a missing global capability produces status `2` with no
experiment/trace/artifact mutation, while a single item failure produces
status `1` and later valid items are still attempted. Discovery/replay
fingerprint mismatch and held concurrency-lock cases have the same explicit
report/error-code assertions.

Integration tests run against a pinned trace-capable MLflow client/server
combination and verify:

- exact experiment resolution/creation, plus reuse of an existing experiment
  without changing its artifact location, name, or unrelated tags;
- one trace per logical turn and distinct trace IDs sharing one native session
  ID, with `trace.info.trace_metadata["mlflow.trace.session"]` equal to the
  source session ID;
- nested model/tool/content spans and full root payloads;
- token/cost/status/timing metadata;
- deterministic artifacts containing unknown/raw fields;
- artifact-path determinism when generated trace/span IDs differ, proving the
  identity-map section is excluded from all path fingerprints;
- dry-run non-mutation; and
- safe replacement and retirement behavior.

End-to-end reconciliation assertions include an unchanged full rerun that
creates no duplicate owned traces, and a multi-turn rerun where changing one
source turn replaces only that turn's owned trace while the other turn trace
and any non-owned trace remain untouched. A complete `--redact` run places a
sentinel secret in root input/output, child content, metadata, and raw artifact
text, then asserts the sentinel is absent from every uploaded trace/artifact,
identifiers are unchanged, and redaction counts appear in the report.

The integration suite pins the compatibility profile and asserts every method
and field in its API table, including named artifact-path verification. A
server that only supports UUID attachments or run artifacts is a negative
compatibility test and must fail preflight.

A six-turn fixture represents the observed pattern of six trace IDs grouped by
one session ID without depending on a mutable live session.

## 12. Implementation work packages

The implementation should follow these independently testable units:

0. MLflow compatibility-profile probe, including named artifact upload and
   verification, before implementing the writer.
1. CLI/configuration and timestamp/path/model validation.
2. Read-only SQLite snapshot and schema detection.
3. Legacy/current adapters and canonical raw-preserving source model.
4. Selection and logical-turn reconstruction.
5. MLflow experiment/trace/span/artifact gateway.
6. Ownership, fingerprints, reconciliation, and safe replacement.
7. Redaction, retries, reporting, and exit statuses.
8. Fixture databases and MLflow integration tests.

No work package adds OpenCode HTTP support, watch mode, arbitrary filter
expressions, or an MLflow run container.

## 13. Decisions made

This section records the decisions made during brainstorming and the fixed
requirements decisions that the implementation must not reinterpret.

| Area | Decision | Reason |
| --- | --- | --- |
| Language | Python 3.11+ | Strong first-party MLflow client and SQLite support; suitable for a standalone CLI. |
| Architecture | Layered streaming pipeline, not direct streaming or export-then-import | Preserves preflight/reconciliation safety without adding a second persistent data format. |
| CLI | Standard-library `argparse` entry point named `opencode-mlflow` | Minimizes runtime dependencies while keeping validation explicit. |
| Source access | SQLite URI `mode=ro`, `query_only`, WAL-aware sanitized normalized snapshot, then read-only replay | Provides a consistent live snapshot without copying prohibited secret pages, holding a WAL reader open during slow MLflow writes, or using `immutable=1`. |
| SQL boundary | Direct parameterized SQL inside schema adapters; no ORM | Supports two generations, raw fields, and unknown columns without schema abstraction loss. |
| Schema precedence | Current `session_v2`/`session_message` wins for matching IDs; legacy-only IDs remain eligible | Prevents duplicate imports while retaining genuinely legacy sessions. |
| Forks | Import only records persisted for the selected child; do not synthesize parent history | Keeps source identity and avoids duplicate parent conversation data. |
| Memory | Stream records and use restrictive temporary files for large canonical artifacts | Meets bounded-memory requirements while retaining lossless artifacts. |
| Discovery | Two passes over the same temporary snapshot with a disk-backed sanitized plan | Allows complete preflight before writes without retaining all source content or rereading a changing live database. |
| Turn ordinal | `turn_index` is 1-based | Human-readable and stable as a separate ordinal from the source-derived key. |
| Turn identity | `turn_key` derives from session ID plus first persisted user-record ID | Prevents later turns from being renamed when an earlier prompt is inserted. |
| Missing turn IDs | Use sequence/timestamp and canonical first-user data, then ordinal as a final fallback | Keeps key generation deterministic when a source record lacks an ID. |
| Turn scope | A qualifying session imports all its logical turns | `--min-user-turns` is a session filter, not a turn slice. |
| Turn starts | Only persisted, non-synthetic user prompts start turns | Prevents empty traces for system, tool-only, lifecycle, or synthetic records. |
| Filtered reconciliation | Only selected sessions with complete scans may retire stale turns | A filtered command must not delete traces it did not examine. |
| Concurrency | Acquire a local non-blocking experiment lock; concurrent hosts must be externally serialized and duplicate identities are conflicts | The first release is one-shot and must not assume distributed locking from every tracking server. |
| Source fidelity | Preserve decoded values plus original raw JSON/text for permitted records | Unknown and malformed fields must not be silently discarded. |
| Canonicalization | `canonical-json-v1` with typed wrappers for SQLite/JSON edge values, duplicate-key preservation, and versioned fingerprint inputs | Stable hashes must not depend on driver formatting or silently lose unsupported values. |
| Unknown session fields | Preserve every permitted session-row column in the raw artifact envelope in addition to typed known fields | Additive OpenCode schema fields must not disappear silently. |
| Secret scope | Never upload account/credential/control-account/account-state values or secret-bearing `session_share` fields; retain only catalogued safe columns | Prevents credential leakage while preserving permitted non-secret provenance. |
| MLflow representation | One MLflow trace per logical turn, grouped by native session ID; no default run container | Matches the observed six-trace session pattern and requirements. |
| MLflow access | Public client/trace/artifact APIs only; no backend-store writes or run fallback | Keeps compatibility and fidelity explicit. |
| Native session field | Set `metadata["mlflow.trace.session"]` on the root `mlflow.start_span_no_context` call | Uses the public API that accepts trace metadata at creation rather than a private post-write mutation. |
| MLflow compatibility | Behavioral capability contract plus a pinned tested client/server pair; no universal numeric floor in the design | MLflow client and tracking-server releases can expose different trace/artifact features. |
| Initial MLflow profile | Test the complete contract against the `3.17.2.dev0` API shape; any newer pair requires a new compatibility profile | Makes method/field behavior explicit without assuming a client-only version guarantees server support. |
| Artifact compatibility gate | No writer implementation proceeds unless the profile exposes a public named trace-artifact upload and verification operation | UUID-only attachments and run artifacts cannot satisfy deterministic, lossless artifact requirements. |
| Capability policy | Trace/artifact fidelity is a hard preflight contract | A lower-fidelity import would silently violate the requirements. |
| Capability failure scope | Any missing capability required by any selected action aborts the whole invocation before mutation with status `2` | Prevents partial imports that cannot be reconciled safely. |
| Child mapping | Typed model, assistant, tool, content, lifecycle, and opaque spans/events | Makes UI inspection useful while retaining source identity for unknown records. |
| Status | Successful → `OK`, explicit failure → `ERROR`, interrupted/incomplete → `UNSET` plus source outcome | Preserves both MLflow status and source distinction. |
| Timing | Prefer source timestamps; use session creation only as a marked source-derived fallback | Never invents timestamps from local wall-clock time. |
| Artifact scope | Each turn trace receives a self-contained session snapshot plus its turn data and identity map | Keeps every trace independently inspectable and permits only changed turn traces to be replaced. |
| Artifact history | Artifacts are immutable historical snapshots; unchanged traces are not rewritten when another turn changes | Preserves replacement scope while making snapshot age explicit through its fingerprint. |
| Artifact paths | Safe encoded session ID, stable turn key, and render fingerprint; never trace ID | Deterministic paths do not overwrite prior replacement artifacts. |
| Artifact identity map | Add generated trace/span IDs after creation and exclude them from fingerprints/paths | Generated MLflow IDs are inherently nondeterministic but are still required in the artifact. |
| Fingerprints | Turn-local fingerprint uses session identity/context plus that turn's records; mutable aggregate/session snapshots use a separate full-session fingerprint; generated MLflow IDs excluded | A changed turn replaces only its trace while artifacts remain immutable and paths stay deterministic. |
| Database identity | Store a non-secret source database fingerprint as metadata, but do not use it as reconciliation identity | OpenCode session identity is the required cross-run identity; local paths and file identity must not create duplicate keys. |
| Redaction | Opt-in; deterministic detectors; exact replacement marker `[REDACTED]`; identifiers protected | Verbatim mode preserves fidelity, while redaction is explicit and testable. |
| Redacted fingerprint | Hash the effective redacted representation, so source changes only inside masked values do not create a new externally visible artifact | Avoids exposing secret hashes and treats indistinguishable redacted output as unchanged. |
| Reconciliation key | `(opencode.session_id, opencode.turn_key)` plus ownership marker | Native trace IDs are not stable source identities and non-owned traces must be protected. |
| Crash recovery | Set ownership/operation metadata at trace creation, verify current candidates, choose deterministically, and retire duplicate owned candidates only after a verified winner | A crash between replacement creation and old-trace retirement must converge safely on rerun. |
| Replacement order | Create → verify trace → upload/verify artifact → retire old | Prevents data loss and leaves the old trace available after creation failure. |
| Retirement | Use supported MLflow trace deletion/soft deletion for stale traces and rollback; no unsafe fallback | An unsupported target must fail safely rather than create duplicates, including after a partial initial write. |
| Retry | Bounded transient retries with idempotency keys or post-timeout ownership search | Avoids duplicate traces from ambiguous non-idempotent requests. |
| Dry run | Perform source and MLflow reads/capability checks but no experiment or trace mutation | Gives operators a safe preview. |
| Reporting | Versioned JSON without transcript content, plus concise human output | Supports automation without leaking sensitive local data. |
| Report actions/counts | Exact action enum; current-turn counts are disjoint and retirement is a separate trace-operation count | Makes JSON automation deterministic and prevents double-counting updates. |
| Exit status | `0` complete/no-op, `1` item failures, `2` pre-write invalid/unsupported invocation | Gives automation a deterministic contract. |
| Testing | Sanitized source fixtures, fake gateway tests, and pinned real MLflow integration tests | Separates deterministic logic from server/API compatibility checks. |

## 14. Acceptance mapping

The requirements acceptance criteria are covered as follows:

- experiment resolution/creation and dry-run behavior: Sections 3.4 and 7.2;
- WAL-safe read-only operation and unsupported-schema status: Section 3.2;
- inclusive time, directory boundaries, model locations, strict turn threshold,
  and AND filtering: Section 5;
- one trace per turn with shared source session ID and separate trace IDs:
  Section 7;
- root input/output, spans, source IDs, status, timing, tokens, costs, and
  unknown/raw fields: Sections 4, 6, and 7;
- unchanged rerun, safe replacement, stale retirement, and protection of other
  tools' traces: Section 8;
- redaction, no secret output, retries, per-item continuation, reporting, and
  exit status: Section 9; and
- fixture and integration coverage: Section 11.

This design intentionally leaves implementation sequencing to the subsequent
implementation plan while fixing the interfaces and decisions needed for that
plan.
