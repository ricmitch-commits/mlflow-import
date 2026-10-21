# Requirements: OpenCode SQLite Session Importer for MLflow

**Date:** 2026-10-21  
**Status:** Brainstorm-approved requirements; implementation pending  
**Working utility name:** `opencode-mlflow`

## 1. Summary

`opencode-mlflow` is a command-line utility that reads sessions from a local
OpenCode SQLite database and imports them into an MLflow experiment. It must
preserve the session identity, turn ordering, trace structure, source timing,
model/tool metadata, token usage, costs, status, and other MLflow-relevant
metadata. It must also retain a lossless JSON representation of the imported
source records, except for explicitly excluded secret-bearing tables, so that
fields unknown to the importer are not silently discarded.

The importer is a one-shot, read-only migration/synchronization tool. It is
not a live OpenCode plugin and does not modify the OpenCode database.

## 2. Goals

1. Import selected OpenCode sessions into a named MLflow experiment, creating
   the experiment when it does not already exist.
2. Represent one OpenCode session as one MLflow trace session and represent its
   logical conversation turns as separate MLflow traces.
3. Preserve enough structure for the MLflow UI and APIs to search and inspect
   prompts, responses, model calls, tools, timing, token usage, errors, and
   source metadata.
4. Support repeatable imports that reconcile changed source sessions without
   altering traces created by other tools.
5. Make selection safe and predictable for large local databases and active
   OpenCode processes.

## 3. Non-goals

- Connecting to an OpenCode HTTP server or API in the first version.
- Continuous/watch-mode synchronization.
- Writing to, vacuuming, migrating, or otherwise modifying the OpenCode
  SQLite database.
- Uploading OpenCode credential, account, access-token, or session-share
  secrets as source data.
- Creating an MLflow run for every OpenCode session by default. The primary
  representation is MLflow trace sessions and traces.
- Importing arbitrary project files that are not present in the OpenCode
  session records.
- Modifying or deleting MLflow traces that are not owned by this importer.

## 4. Terminology and fixed decisions

### 4.1 OpenCode session

An OpenCode session is the row identified by `id` in the source session table.
The importer must support both:

- the current `session_v2` table and its session-message records; and
- the legacy `session` table with related `message` and `part` records.

When both schema generations are present, the importer must use the matching
current `session_v2` representation rather than importing the same session
twice.

### 4.2 Logical conversation turn

A logical turn begins with a persisted OpenCode user prompt and includes all
subsequent assistant, model, tool, reasoning, text, patch, file, and lifecycle
records up to—but not including—the next user prompt. Synthetic, system,
tool-only, and lifecycle records do not start a new user turn.

The minimum-turn filter is strict: a session is selected only when its logical
turn count is **greater than** the supplied threshold.

### 4.3 MLflow session and trace

MLflow does not require a standalone session object for this workflow. The
OpenCode session ID is used as the MLflow trace session identifier on every
turn trace. Each logical turn receives its own MLflow trace ID, and the shared
session identifier groups those traces in the MLflow UI and APIs.

This matches the observed MLflow pattern validated against the supplied local
session `ses_ea65559e0ffe9VPzwykwPJgl6j`: six separate trace IDs shared the
same `mlflow.trace.session` metadata value, with trace-level inputs/outputs,
status, timing, token usage, user/source metadata, schema version, and
artifact locations.

### 4.4 Importer-owned trace

An MLflow trace is importer-owned only when it carries the importer ownership
marker and source identity fields defined in Section 9. Existing traces that
lack those markers must never be changed, even when they share an MLflow
session ID with an imported trace.

## 5. User stories

- As a developer, I can import all sessions created in a time range into a
  named MLflow experiment.
- As a developer, I can restrict imports to a project directory and its
  descendants.
- As a developer, I can find sessions that used a particular provider/model,
  even when the model was recorded on a message or part rather than the
  session row.
- As a developer, I can select sessions with more than a specified number of
  logical user turns.
- As a developer, I can rerun the importer after a session changes and have
  importer-owned MLflow traces reconciled rather than duplicated.
- As an operator, I can preview the selection and expected writes without
  changing MLflow.
- As an operator, I can choose verbatim content or an explicit redaction mode
  before sending local transcripts to MLflow.
- As an automation author, I can consume a deterministic JSON result and a
  meaningful exit status.

## 6. Command-line interface

The first version must provide an `import` command with the following shape:

```text
opencode-mlflow import \
  --experiment-name NAME \
  [--db PATH] \
  [--tracking-uri URI] \
  [--from-time ISO-8601] [--to-time ISO-8601] \
  [--directory PATH] \
  [--model PROVIDER/MODEL] \
  [--min-user-turns N] \
  [--redact] [--dry-run] \
  [--output human|json]
```

### 6.1 Required and connection options

| Option | Requirement |
| --- | --- |
| `--experiment-name` | Required. Resolve by name and create if absent. |
| `--db` | Optional. Use the supplied SQLite path; default to the local OpenCode database, normally `~/.local/share/opencode/opencode.db`. |
| `--tracking-uri` | Optional. Use the supplied MLflow tracking URI; otherwise honor standard MLflow client/environment configuration. |
| authentication/TLS settings | Use standard MLflow client configuration. Credentials must not appear in command output or logs. |

The command must validate the source database and MLflow trace/artifact
capabilities before beginning writes. It must never silently fall back to a
lower-fidelity run-only upload.

### 6.2 Selection options

- `--from-time` and `--to-time` accept ISO-8601 timestamps with an explicit
  timezone offset or `Z`. Boundaries are inclusive. Either boundary may be
  omitted to create an open-ended range.
- Time filtering uses the source session's `time_created` value, normalized
  from the source timestamp unit to an absolute instant. It does not use the
  latest message timestamp or `time_updated`.
- `--directory` is normalized and matches the requested directory and all
  descendants. Matching must respect path-component boundaries, so `/repo/a`
  does not match `/repo/another`. Normalization must make relative input paths
  absolute and resolve `.`/`..` and separator differences without requiring
  the path to exist or resolving symlinks.
- `--model` is an exact normalized provider/model identity. A session matches
  when that model occurs anywhere in its persisted message, session-message,
  or part records; checking only the session-level default is insufficient.
  A canonical provider/model value is formed as `providerID/modelID`, with
  surrounding whitespace removed and case preserved; incomplete source values
  are matched only when the requested value has the same available form.
- `--min-user-turns N` selects sessions whose logical turn count is strictly
  greater than `N`; `N` must be a non-negative integer.
- All supplied selection options combine with logical **AND**. An omitted
  option imposes no restriction.
- With no optional selection filter, all compatible sessions are eligible;
  the command must clearly report the potentially broad selection and support
  `--dry-run` before any writes.

### 6.3 Operational options

- `--redact` enables secret masking for trace inputs/outputs, span data,
  metadata values, and uploaded JSON artifacts. Without it, content is
  uploaded verbatim.
- `--dry-run` performs source discovery, schema validation, filtering, turn
  counting, and MLflow capability checks, but does not create an experiment,
  create traces, retire traces, or upload artifacts.
- `--output human|json` controls the result format. JSON output must include
  per-session and per-turn source identities, action (`imported`, `updated`,
  `skipped`, `failed`, or `retired`), MLflow trace IDs when available, and
  error details safe for logs.
- The command should expose a standard verbosity/log-level option, but logs
  must not print prompt/response content or secret values.

## 7. Source database requirements

### 7.1 Read consistency and safety

1. Open SQLite in read-only mode.
2. Support a consistent snapshot while OpenCode is running, including WAL
   mode and the `-wal`/`-shm` companion files.
3. Never issue schema changes, writes, checkpoints, deletes, or vacuum
   operations against the source database.
4. Use bounded memory and stream message/part data for large sessions.
5. Preserve raw JSON/text when a source field cannot be decoded; malformed or
   unknown fields must generate a warning rather than being silently dropped.

### 7.2 Schema adapters

The importer must detect the schema generation before querying session data.
The adapter layer must normalize both generations into a common internal
model containing:

- session ID, project/workspace/parent/fork IDs, slug, title, directory, path;
- OpenCode version, agent, configured model, lifecycle/archival state;
- creation/update/lifecycle timestamps;
- cost and token aggregates;
- summary additions, deletions, files, and diffs;
- ordered user prompts and all records in each logical turn;
- message/part IDs, source types, sequence values, raw JSON, and timestamps;
- model/provider, tool, finish, error, and token fields wherever present.

For the current schema, `session_v2` and `session_message` are authoritative
for a matching session. For the legacy schema, `session`, `message`, and
`part` are authoritative. Session-scoped event, todo, instruction, and
similar records must also be included in the normalized source artifact when
they can be associated without ambiguity; unassociated records must be
reported rather than silently assigned to a session.

The importer must not include values from `account`, `account_state`,
`control_account`, `credential`, or secret-bearing `session_share` fields in
the uploaded source artifact. Their existence must not affect session
selection.

## 8. MLflow data model and mapping

### 8.1 Experiment

- Resolve the requested experiment by exact name.
- Create it when it does not exist, using the MLflow API/client rather than
  writing the backend store directly.
- Reuse an existing experiment without changing its name, artifact location,
  or unrelated tags.
- Report the resolved experiment ID before importing traces.

### 8.2 Session-level identity and metadata

Every imported turn trace must carry:

- the native MLflow session identifier set to the OpenCode `session.id`; on
  the inspected MLflow server this is visible as `mlflow.trace.session`;
- an explicit namespaced `opencode.session_id` field for robust reconciliation;
- title, slug, directory, path, project/workspace/parent/fork IDs;
- agent, configured model, OpenCode version, source schema generation;
- session creation/update/lifecycle timestamps;
- session cost and token aggregates;
- logical turn count and source message/part counts;
- source database identity/fingerprint where available, represented as a
  non-secret fingerprint rather than an absolute local path by default;
- available user and Git/source metadata, using native MLflow fields when
  applicable and omitting values that are not present rather than inventing
  them.

### 8.3 Turn trace

For every logical turn:

1. Create one MLflow trace in the resolved experiment.
2. Use the first user record's timestamp as the trace start time and the last
   record's timestamp/completion time as the trace end/duration when available.
3. Set the trace status from the source terminal outcome. Successful turns map
   to `OK`; failed, interrupted, or otherwise incomplete turns retain a
   distinguishable non-OK state and error metadata.
4. Set root request/input data to the user prompt and root response/output data
   to the final assistant result. Preserve the complete values even when the
   MLflow server exposes only a truncated preview in its search response.
5. Preserve `opencode.turn_index`, source message IDs, source sequence values,
   source timestamps, and a stable turn identity.

### 8.4 Child spans and events

Reconstruct typed MLflow/OpenTelemetry-compatible child spans or events for
source records where the source contains enough information:

| OpenCode information | MLflow representation |
| --- | --- |
| Model/provider request and response | LLM/model span with model identity, inputs, outputs, timing, finish state, token usage, and cost. |
| Assistant execution step | Assistant/agent span with source message identity and timing. |
| Tool call and result | Tool span with tool name, call ID, inputs, outputs, timing, status, and error data. |
| Reasoning/text/patch/file part | Typed child span or event with content, part ID/type, sequence, and timing. |
| Session/turn lifecycle record | Event or trace metadata preserving source type, outcome, and timestamp. |
| Unknown source record | Raw normalized record in the artifact and an opaque span/event attribute when size limits allow. |

Per-span data must preserve source IDs and raw types so a consumer can map a
span back to SQLite. Per-span token usage and cost must be retained, and
trace-level aggregates must be calculated from the source rather than inferred
from truncated previews.

### 8.5 Lossless artifacts

Each imported session/turn must include a canonical JSON artifact containing:

- the source session row;
- all source message, part, session-message, and associated session-scoped
  records permitted by Section 7.2;
- the normalized logical-turn boundaries;
- the source-to-MLflow trace/span identity map;
- importer version, schema adapter, source fingerprint, and redaction mode.

Artifact paths must be deterministic and namespaced by OpenCode session ID and
turn identity. With `--redact`, redaction applies to the artifact as well as
trace data; the unredacted source must never be uploaded under another path.

## 9. Reconciliation and updates

The importer must write ownership and identity fields sufficient to locate its
own traces without relying on MLflow-generated trace IDs:

- `opencode.importer = opencode-mlflow`;
- `opencode.session_id`;
- `opencode.turn_index` or another deterministic turn key;
- `opencode.source_fingerprint`;
- `opencode.source_schema`;
- `opencode.importer_version`.

Default behavior is update/reconciliation:

1. If the source identity and fingerprint match an importer-owned trace, report
   it as unchanged/skipped rather than creating a duplicate.
2. If the source identity exists but the fingerprint changed, create and verify
   the replacement trace first, then retire the old importer-owned trace. The
   old trace must remain intact if replacement creation fails.
3. If the current source no longer contains a previously imported turn, retire
   only the stale importer-owned trace for that session.
4. If trace deletion/retirement is unsupported by the target MLflow server,
   fail the replacement safely and leave the prior trace intact; never create
   an untracked duplicate as a fallback.
5. Plugin-generated or otherwise non-importer-owned traces sharing the same
   MLflow session ID must remain untouched.

## 10. Content and redaction policy

- Verbatim upload is the default to preserve session fidelity.
- `--redact` must mask common secret categories such as bearer/API tokens,
  private keys, password assignments, and credential-bearing connection
  strings. The exact detector patterns and replacement marker must be
  documented and testable.
- Redaction must apply consistently to root inputs/outputs, child spans,
  trace metadata, error messages, and canonical JSON artifacts.
- Redaction must not change source identifiers needed for reconciliation.
- The importer must report redaction was enabled and the number of affected
  fields/records, but must never print the original secret.

## 11. Reliability and failure behavior

1. Validate all CLI arguments, timestamps, paths, schema requirements, and
   MLflow capabilities before starting the write phase.
2. Retry transient network/server failures with bounded exponential backoff.
   Do not retry invalid requests or authorization failures indefinitely.
3. Continue processing other sessions and turns after a per-item failure.
4. Return a nonzero exit status when any item fails, even if other items
   succeed.
5. Produce a summary containing selected, imported, updated, unchanged,
   retired, skipped, and failed counts, plus per-item diagnostics.
6. Use a safe write ordering for replacements: create, verify, and artifact-
   check the new trace before retiring the previous trace.
7. A session with no logical user turns must not produce an empty synthetic
   trace; it should be reported as having no importable turns.

Suggested exit-status contract:

| Exit status | Meaning |
| --- | --- |
| `0` | Valid invocation; all selected work completed or was an intentional no-op. |
| `1` | One or more sessions/turns failed or could not be reconciled. |
| `2` | Invalid arguments, unavailable source database, unsupported schema, or MLflow configuration/capability failure before import. |

## 12. Security and privacy requirements

- Use standard MLflow authentication and TLS verification; do not accept or
  echo credentials in positional arguments or normal logs.
- Never read or upload credential/account/token table values.
- Do not print prompt, response, tool output, source code, or secret values in
  progress output, errors, or JSON diagnostics.
- Clearly document that verbatim mode can upload sensitive local transcript
  content and recommend `--redact` for shared or production MLflow servers.
- Do not alter the source database or local OpenCode files.

## 13. Non-functional requirements

- Deterministic ordering: sessions by normalized creation time and session ID;
  turns by source order/timestamp and stable source ID.
- Bounded memory usage for large databases and sessions; upload artifacts and
  spans in bounded batches where the MLflow API permits.
- Clear compatibility errors for unsupported OpenCode schema or MLflow trace
  API versions.
- Stable canonicalization and hashing so a repeat import produces the same
  fingerprint when the source data is unchanged.
- Human output must be concise and progress-oriented; JSON output must be
  machine-readable and schema-versioned.

## 14. Acceptance criteria

### Experiment and connection

- Given a new experiment name, the command creates it once and imports into
  the returned experiment ID.
- Given an existing experiment name, the command reuses it without changing
  unrelated settings.
- Given a live WAL-mode OpenCode database, the command reads a consistent
  snapshot without writing to the database.
- Given an unsupported or malformed database, the command exits with status
  `2` and identifies the compatibility problem.

### Selection

- A session at exactly `fromTime` or `toTime` is included.
- A session outside the `session.time_created` range is excluded even if a
  later message falls inside the range.
- A directory filter includes descendants and excludes similarly prefixed
  sibling paths.
- A model filter matches a model recorded only on a message, session-message,
  or part record.
- A session with exactly `N` logical turns is excluded by `--min-user-turns N`;
  a session with `N + 1` turns is included.
- Multiple filters select only sessions satisfying every supplied condition.

### Mapping and fidelity

- Every selected logical turn produces one MLflow trace with the source session
  ID as its MLflow session identifier.
- Traces in one source session are separately addressable by MLflow trace ID
  and groupable by the shared session ID.
- Root input/output, timestamps, status, model/tool spans, token usage, costs,
  source IDs, and relevant metadata are preserved.
- The canonical artifact contains unknown and raw source fields that cannot be
  represented as native MLflow fields.
- The six-trace session pattern inspected in the local MLflow instance is
  representable without collapsing the traces into one run or one trace.

### Reconciliation and safety

- Re-running an unchanged import does not create duplicate importer-owned
  traces.
- Changing a source turn replaces only its importer-owned trace and leaves the
  previous trace intact if the replacement fails.
- Existing non-importer traces are never modified.
- `--redact` removes detected secrets from both trace data and artifacts while
  preserving identifiers and reporting redaction without exposing originals.
- `--dry-run` performs no MLflow mutation.
- A failed item does not prevent valid later items from being attempted, and
  the final exit status/report identifies the failure.

## 15. Implementation boundary for the next phase

The implementation plan should be organized around these independently
testable units:

1. CLI/configuration and timestamp/path/model validation.
2. Read-only SQLite snapshot and schema detection.
3. Legacy/current OpenCode schema adapters and canonical source model.
4. Session filtering and logical-turn reconstruction.
5. MLflow experiment/trace/span/artifact writer.
6. Ownership, fingerprint reconciliation, and safe replacement.
7. Redaction, retry, reporting, and exit-status handling.
8. Fixture database and MLflow integration tests.

The first implementation should not add OpenCode server/API support, live
watching, arbitrary custom filter expressions, or an MLflow run container
unless a later requirements change explicitly authorizes those features.
 