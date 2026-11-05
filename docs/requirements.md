# Requirements: Coding-Agent Session Importer for MLflow

**Date:** 2026-10-21 | **Reviewed:** 2026-11-04 | **Updated:** 2026-11-05

**Status:** Approved target requirements; the shipped v0.1 implementation has documented deviations.
**Working utility name:** `mlflow-import`

These requirements define the harness-neutral importer contract. Source-schema
examples below are explicitly scoped to the adapter they describe.
The shipped v0.1.0 writer retains legacy `opencode.*` metadata/tag keys and
the `opencode-mlflow.result.v1` identifier; these names are compatibility
identifiers, not source-harness restrictions. The namespaced
`mlflow_import.*` fields below are the target contract for a versioned
metadata migration. This checkout contains a target design, not the shipped
implementation. The snapshot, privacy, compatibility, and recovery rules in
[`architecture.md`](architecture.md) must be verified before claiming conformance.

## 1. Summary

`mlflow-import` is a command-line utility that reads sessions from supported
local coding-agent sources and imports them into an MLflow experiment. The
first supported harnesses are OpenCode, Claude Code, Codex, PI, and Gemini
CLI. It must
preserve the session identity, turn ordering, trace structure, source timing,
model/tool metadata, token usage, costs, status, and other MLflow-relevant
metadata. It must also retain a lossless JSON representation of the imported
source records, except for explicitly excluded secret-bearing records/fields, so that
fields unknown to the importer are not silently discarded.

The importer is a one-shot, read-only migration/synchronization tool. It is
not a live plugin for any supported harness and does not modify source data.

## 2. Goals

1. Import selected sessions from any supported harness into a named MLflow experiment, creating
   the experiment when it does not already exist.
2. Represent one harness session as one MLflow trace session and represent its
   logical conversation turns as separate MLflow traces.
3. Preserve enough structure for the MLflow UI and APIs to search and inspect
   prompts, responses, model calls, tools, timing, token usage, errors, and
   source metadata.
4. Support repeatable imports that reconcile changed source sessions without
   altering traces outside the selected conflicting session. Within that session,
   the effective `overwrite` policy (also the default) authorizes replacement of
   traces created by other tools, subject to Section 9's verification gate.
5. Make selection safe and predictable for large local sources and active
   harness processes.

## 3. Non-goals

- Connecting to a harness HTTP server or API in the first version.
- Continuous/watch-mode synchronization.
- Writing to, vacuuming, migrating, or otherwise modifying any source.
- Uploading harness credential, account, access-token, or session-share
  secrets as source data.
- Creating an MLflow run for every harness session by default. The primary
  representation is MLflow trace sessions and traces.
- Importing arbitrary project files that are not present in the source
  session records.
- Modifying or deleting MLflow traces that are not owned by this importer,
  except for traces in a selected conflicting session under the effective
  `overwrite` policy, whether defaulted or explicitly supplied.

## 4. Terminology and fixed decisions

### 4.1 Harness session

Each supported harness defines a source-specific session identity and adapter.
The importer must support these harnesses:

- OpenCode, including its current and legacy SQLite session schemas;
- Claude Code;
- Codex;
- PI; and
- Gemini CLI.

The adapter for each harness must detect its supported local source format,
derive a stable session identity, and avoid importing the same source session
twice when multiple compatible representations are present.

### 4.2 Logical conversation turn

A logical turn begins with a persisted user prompt and includes all subsequent
assistant, model, tool, reasoning, text, patch, file, and lifecycle records
up to—but not including—the next user prompt. The adapter maps each harness's
record types into this common model. Synthetic, system, tool-only, and
lifecycle records do not start a new user turn.

For branching or replayable formats, turns follow the authoritative persisted
conversation/resume path after applying source updates, rewinds, or compaction
semantics. Alternate branches and original update records remain in the source
artifact; they do not create additional primary turns. An adapter must document
its persisted-path rule and reject ambiguous or cyclic ancestry. In-memory state
that the harness does not persist is outside the local-source contract.

The minimum-turn filter is strict: a session is selected only when its logical
turn count is **greater than** the supplied threshold.

### 4.3 MLflow session and trace

MLflow does not require a standalone session object for this workflow. The
normalized source session identity `(harness, source_namespace, native_session_id)`
is encoded deterministically as the MLflow trace session identifier on every
turn trace, in both explicit-source and default-discovery runs. The namespace
must be stable for the same source under either selection method and must not
expose an absolute local path. Preserve the native ID separately. Each logical
turn receives its own MLflow trace ID, and the shared session identifier groups those traces in the MLflow
UI and APIs.

This matches the observed MLflow pattern in which separate trace IDs share the
same `mlflow.trace.session` metadata value, with trace-level inputs/outputs,
status, timing, token usage, user/source metadata, schema version, and
artifact locations.

### 4.4 Importer-owned trace

An MLflow trace is importer-owned only when it carries the importer ownership
marker and source identity fields defined in Section 9. Existing traces that
lack those markers are not reused or refreshed as importer-owned. The effective
`overwrite` policy, including the default, may replace traces in the selected
conflicting session, including traces created by another importer or manually, as defined
in Section 9. The `duplicate` and `ignore` policies leave those existing
traces unchanged.

## 5. User stories

- As a developer, I can import all sessions created in a time range into a
  named MLflow experiment.
- As a developer, I can import one source session by its session ID, including
  all of its logical turns.
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

The first version must provide the `mlflow-import` command with the following
shape. The tool is solely an importer, so it does not require an `import`
subcommand:

```text
mlflow-import \
  --experiment-name NAME \
  [--harness NAME] \
  [--source PATH] \
  [--session SESSION-ID] \
  [--on-conflict duplicate|overwrite|ignore] \
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
| `--harness` | Optional. Select a supported harness adapter (`opencode`, `claude`, `codex`, `pi`, or `gemini`). With no `--source`, use that adapter's default source location(s). |
| `--source` | Optional. Use the supplied local harness session source path. If `--harness` is omitted, detect exactly one compatible harness format; if both are supplied, reject a mismatch before writes. |
| `--on-conflict` | Optional. Choices are `duplicate`, `overwrite`, and `ignore`; default is `overwrite`. This effective policy applies regardless of trace ownership; see Section 9. |
| `--tracking-uri` | Optional. Use the supplied MLflow tracking URI; otherwise honor standard MLflow client/environment configuration. |
| authentication/TLS settings | Use standard MLflow client configuration. Credentials must not appear in command output or logs. |

The command must validate the selected source and MLflow trace/artifact
capabilities before beginning writes. It must never silently fall back to a
lower-fidelity run-only upload.

Source selection is intentionally separate from harness selection. With
neither `--harness` nor `--source`, the command discovers all available
adapter-owned default sources. With `--harness` only, it scans only that
harness's default source locations. With `--source` only, it detects the
harness from the explicit path. With both options, it uses the explicit path
and validates that it belongs to the requested harness. Each adapter owns its
default paths, environment-variable overrides, format detection, and
read-only opening behavior; the CLI does not encode harness-specific paths.

### 6.2 Selection options

- `--from-time` and `--to-time` accept ISO-8601 timestamps with an explicit
  timezone offset or `Z`. Boundaries are inclusive. Either boundary may be
  omitted to create an open-ended range.
- `--session SESSION-ID` restricts selection to the source session whose
  adapter-provided session ID matches the supplied ID. It selects the session
  as a whole, including all of its importable logical turns.
- Time filtering uses the source session's creation timestamp, normalized from
  the harness-specific representation to an absolute instant. It does not use
  the latest message timestamp or update time.
- `--directory` is normalized and matches the requested directory and all
  descendants. Matching must respect path-component boundaries, so `/repo/a`
  does not match `/repo/another`. Normalization must make relative input paths
  absolute and resolve `.`/`..` and separator differences without requiring
  the path to exist or resolving symlinks.
- `--model` is an exact normalized provider/model identity. A session matches
  when that model occurs anywhere in persisted records that can carry model
  identity; checking only the session-level default is insufficient. For
  OpenCode, this includes message, session-message, and part records.
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

- `--on-conflict duplicate|overwrite|ignore` controls existing sessions according
  to Section 9. Default `overwrite` is authorization to replace all conflicting
  traces in the selected session, including those created by another tool; the
  preview/report must make that scope clear.
- `--redact` enables secret and user-identity masking for session/trace
  inputs/outputs, span data, metadata values (including native MLflow user
  fields), and uploaded JSON artifacts. Without it, permitted source content
  and available user metadata are uploaded verbatim.
- `--dry-run` performs source discovery, schema validation, filtering, turn
  counting, and read-only MLflow capability checks, and previews expected
  imports, updates, skips, and retirements. It does not create an experiment,
  create traces, retire traces, or upload artifacts.
- `--output human|json` controls the result format. JSON output must include
  per-session and per-turn source identities, action (`imported`, `updated`,
  `skipped`, `failed`, or `retired`), MLflow trace IDs when available, and
  error details safe for logs. The target `schema_version` is
  `mlflow-import.result.v1`; legacy results retain their original version identifier.
- The command should expose a standard verbosity/log-level option, but logs
  must not print prompt/response content or secret values.

## 7. Source requirements

### 7.1 Read consistency and safety

1. Open each source using its adapter's read-only access mechanism.
2. Support a consistent committed read transaction for SQLite, including WAL
   mode and the `-wal`/`-shm` companion files where applicable. For file sources,
   freeze a stable per-session record/sidecar set and detect concurrent changes
   with bounded retry; persistent instability fails before writes. Independent
   files/roots need not represent one global atomic instant. Authoritative
   metadata may safely exclude nonmatching sessions before full transcript
   freezing; excluded-session churn must not invalidate an otherwise stable plan.
3. Never issue schema changes, writes, deletes, or other mutating operations
   against source data.
4. Use bounded memory and stream records for large sessions.
5. Preserve raw JSON/text when a source field cannot be decoded; malformed or
   unknown fields must generate a warning rather than being silently dropped.

### 7.2 Schema adapters

The importer must select and validate a harness adapter before reading session
data. Each adapter must normalize its source into a common internal model
containing:

- session ID, project/workspace/parent/fork IDs, slug, title, directory, path;
- harness name/version, agent, configured model, lifecycle/archival state;
- creation/update/lifecycle timestamps;
- cost and token aggregates;
- summary additions, deletions, files, and diffs;
- ordered user prompts and all records in each logical turn;
- record IDs, source types, sequence values, raw JSON/text, and timestamps;
- model/provider, tool, finish, error, and token fields wherever present.

Each adapter must also define its supported default source location(s) and
discovery rules. An explicit `--source` path may point to a non-default
installation, profile, archive, or test fixture and must be validated against
the selected adapter. The generic CLI must not expose format-specific aliases
such as an OpenCode-only `--db` option.

For OpenCode, a session's current `session_v2` and associated `session_message`/
part representation is authoritative when present, otherwise the legacy
`session`, `message`, and `part` representation is authoritative. Incomplete or
ambiguous authoritative data must fail validation rather than silently falling
back to an older representation. Claude Code, Codex, PI, and Gemini
CLI adapters must define equivalent authoritative records for their supported
formats. Session-scoped event, todo, instruction, and similar records must
also be included in the normalized source artifact when they can be associated
without ambiguity; unassociated records must be reported rather than silently
assigned to a session.

The importer must not include credential, account, access-token, session-share,
or equivalent secret-bearing records in the uploaded source artifact. Their
existence must not affect session selection.

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

- the native MLflow session identifier set to the normalized source session
  identity, consistently qualified by harness and source namespace regardless
  of how the source was selected. On the inspected MLflow server this is
  visible as `mlflow.trace.session`;
- an explicit namespaced `mlflow_import.session_id` field for robust reconciliation;
- title, slug, directory, path, project/workspace/parent/fork IDs;
- harness, agent, configured model, harness version, source schema generation;
- session creation/update/lifecycle timestamps;
- session cost and token aggregates;
- logical turn count and source record counts;
- source identity/fingerprint where available, represented as a
  non-secret fingerprint rather than an absolute local path by default;
- a `source` value equal to the canonical harness name (`opencode`, `claude`,
  `codex`, `pi`, or `gemini`), stored as native `mlflow.source.name` metadata on
  every turn trace and used consistently in normalized session/trace artifacts.
  It must not be the importer name, model/provider, or `--source` filesystem
  path; source namespace/fingerprint and Git provenance remain separate fields;
- available user and Git metadata, using native MLflow fields when applicable.
  Apply Section 10's user-identity masking before populating any native or
  namespaced user field; omit absent values rather than inventing them.

Mutable session metadata describes the current verified source projection.
Refreshing it must not replace an unchanged turn trace. Use mutable trace tags
for fields that change after logging; MLflow request metadata is not assumed mutable.
Remove obsolete importer-managed values. If a metadata value exceeds a target
field limit, preserve the full value in the canonical artifact and use an
explicit reference/overflow marker; never silently truncate identity fields or
root inputs/outputs. Identity fields that cannot fit require a compatibility error.
If an immutable native user/source value must change, use Section 9's verified
trace replacement rather than leaving an incorrect or unredacted immutable copy
behind during a metadata-only refresh.

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
5. Preserve `mlflow_import.turn_index`, source record IDs, source sequence values,
   source timestamps, and a stable turn identity.

The stable turn key derives from the initiating persisted user record, not its
current numeric index. Refreshable session-wide metadata and generated target
IDs/paths must not invalidate an unchanged turn's payload fingerprint. Retain raw timing
and usage values; normalize declared units, avoid double-counting cumulative
usage or repeated parent/child totals, and mark unknown outcomes/timing rather
than inventing source values.

### 8.4 Child spans and events

Reconstruct typed MLflow/OpenTelemetry-compatible child spans or events for
source records where the source contains enough information:

| Harness information | MLflow representation |
| --- | --- |
| Model/provider request and response | LLM/model span with model identity, inputs, outputs, timing, finish state, token usage, and cost. |
| Assistant execution step | Assistant/agent span with source message identity and timing. |
| Tool call and result | Tool span with tool name, call ID, inputs, outputs, timing, status, and error data. |
| Reasoning/text/patch/file part | Typed child span or event with content, part ID/type, sequence, and timing. |
| Session/turn lifecycle record | Event or trace metadata preserving source type, outcome, and timestamp. |
| Unknown source record | Raw normalized record in the artifact and an opaque span/event attribute when size limits allow. |

Per-span data must preserve source IDs and raw types so a consumer can map a
span back to the harness source. Per-span token usage and cost must be retained, and
trace-level aggregates must be calculated from the source rather than inferred
from truncated previews.

### 8.5 Lossless artifacts

Each imported session/turn must include a canonical JSON artifact containing:

- the normalized source session record;
- all source records and associated session-scoped records permitted by
  Section 7.2;
- the normalized logical-turn boundaries;
- the source-to-MLflow trace/span identity map;
- importer version, schema adapter, source fingerprint, and redaction mode.

The artifact may be a versioned JSON bundle: one shared session-record artifact,
reusable per-turn identity maps, and a session manifest joining them. Each trace
must resolve to its full canonical bundle. Do not repeat the full transcript in
every turn artifact or rebuild an unchanged turn map solely because another
turn changed. Prior verified generations remain intact when a write fails.

Artifact paths must be deterministic and namespaced by harness and source session ID and
turn identity. With `--redact`, redaction applies to the artifact as well as
trace data; the unredacted source must never be uploaded under another path.

## 9. Reconciliation and updates

The importer must write ownership and identity fields sufficient to locate its
own traces without relying on MLflow-generated trace IDs:

- `mlflow_import.importer = mlflow-import`;
- `mlflow_import.harness`;
- `mlflow_import.source_namespace`;
- `mlflow_import.session_id`;
- `mlflow_import.turn_index` or another deterministic turn key;
- `mlflow_import.source_fingerprint`;
- `mlflow_import.source_schema`;
- `mlflow_import.importer_version`.

Unless `--on-conflict` changes the policy, the default behavior is
update/reconciliation:

1. If source identity, turn payload fingerprint, metadata, and artifacts verify
   unchanged on an importer-owned trace, report unchanged/skipped without writes.
2. If that turn's payload changed, create and verify the replacement trace first,
   then retire the old importer-owned trace. The old trace must remain intact if
    replacement creation fails. A mutable session-metadata-only change refreshes
    managed metadata/artifact references without replacing unchanged turn traces.
3. If the current source no longer contains a previously imported turn, retire
   only the stale importer-owned trace for that session.
4. If trace deletion/retirement is unsupported by the target MLflow server,
   fail the replacement safely and leave the prior trace intact; never create
   an untracked duplicate as a fallback.
5. Plugin-generated or otherwise non-importer-owned traces sharing the same
   MLflow session ID are not reused or metadata-refreshed as importer-owned;
   retirement is authorized only within the selected conflicting session under
   the effective `overwrite` policy, including its default.

The duplicate-session policy applies only after source normalization and
selection, using the harness/session identity represented in the target
MLflow session. Adapter-level duplicate representations of the same source
session must still be deduplicated before this policy is evaluated. The
`duplicate` policy must use an additional import instance identity so the new
copy is independently addressable and does not overwrite or retire the
existing copy. The `ignore` policy never modifies existing traces. The
`overwrite` policy may retire existing traces in the selected conflicting
session, regardless of ownership, but must not modify traces outside that
session and must verify the replacement before retirement.

Conflict discovery must recognize the canonical qualified identity and supported
legacy/plugin session aliases, including native-only session IDs when harness/
source association is unambiguous. Ambiguous aliases fail before writes; an
ownership marker alone does not authorize treating another tool's trace as owned.
Repeated `overwrite` imports converge to one current copy; a `duplicate` copy
remains separately addressable by its import-instance ID until a later explicit
or default overwrite reconciles the entire selected source session.

Persist enough attempt state to recover ambiguous write responses and partial
retirement without local snapshots. Recover incomplete attempts before applying
`ignore` or allocating a new duplicate instance. If the source changes after a
failure, safely abandon pre-retirement staged work or finish an already verified
retirement batch before planning the newer projection. Reverify replacements
before resuming retirement. Serialize writers for the same target scope and
document the supported cross-process/host boundary; process-local sequential
iteration alone is not a concurrency guarantee.

Read-only recovery discovery covers requested source namespaces and any explicit
session-ID restriction even when current content filters select no sessions.
An unfinished attempt excluded by those filters must be reported as incomplete
(exit `1`), not a successful no-op. Do not mutate excluded sessions; diagnostics
must identify the session and explain which filters must be relaxed to resume it.

## 10. Content and redaction policy

- Verbatim upload is the default to preserve session fidelity.
- `--redact` must mask common secret categories such as bearer/API tokens,
  private keys, password assignments, and credential-bearing connection
  strings. The exact detector patterns and replacement marker must be
  documented and testable.
- Redaction must apply consistently to root inputs/outputs, child spans,
  trace metadata, error messages, and canonical JSON artifacts.
- `--redact` must also replace present user-identity values with `[REDACTED]`,
  even when they do not match a secret detector. This includes user IDs,
  usernames, names, email addresses, nested user objects, `mlflow.trace.user`,
  any populated `mlflow.user` field, and equivalent native/namespaced fields in
  session/trace data and retained raw/escaped artifact copies. Do not substitute
  the importing OS/SDK user. Missing user values stay absent; the conversation
  role `user` is structural data and must remain unchanged.
- Mask structured credential values by key as well as tokens in free text,
  including quoted/escaped raw JSON copies and malformed credential text.
  Every exported representation must be sanitized; selection is performed on
  permitted source facts and must not change merely because `--redact` is enabled.
- Redaction must not change source identifiers needed for reconciliation;
  user-identity metadata is not exempt under this rule. The harness-valued
  `source` remains unchanged in either privacy mode.
- The importer must report redaction was enabled and the number of affected
  fields/records, but must never print the original secret or masked user identity.

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
8. An unchanged rerun must verify and reuse existing data without trace/artifact
   writes. An ambiguous create/upload response must be looked up and verified
   before retrying a non-idempotent operation. All nested retries and recovery
   generations share bounded attempt/time limits.

Suggested exit-status contract:

| Exit status | Meaning |
| --- | --- |
| `0` | Valid invocation; all selected work completed or was an intentional no-op. |
| `1` | One or more sessions/turns failed or could not be reconciled. |
| `2` | Invalid arguments, unavailable source, unsupported format, or MLflow configuration/capability failure before import. |

## 12. Security and privacy requirements

- Use standard MLflow authentication and TLS verification; do not accept or
  echo credentials in positional arguments or normal logs.
- Never read or upload credential/account/token table values.
- Do not print prompt, response, tool output, source code, or secret values in
  progress output, errors, or JSON diagnostics.
- Clearly document that verbatim mode can upload sensitive local transcript
  content and recommend `--redact` for shared or production MLflow servers.
- Do not alter source data or local harness files.

## 13. Non-functional requirements

- Deterministic ordering: sessions by normalized creation time and session ID;
  turns by source order/timestamp and stable source ID.
- Bounded memory usage for large sources and sessions; upload artifacts and
  spans in bounded batches where the MLflow API permits.
- Record indexing, turn reconstruction, canonicalization, and reporting must
  not retain session/corpus-sized Python collections. Permit disk-backed offset/
  parent/turn indexes and spooled JSON results. Use a fixed number of streaming
  passes, indexed record access, and paginated target lookups; do not rescan the
  transcript or the experiment once per turn. A target API's irreducible one-turn
  payload is the documented memory exception, with an explicit failure on overflow.
- Clear compatibility errors for unsupported harness format or MLflow trace
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
- Given a live source, the command obtains the SQLite committed view or stable
  per-session file view described in Section 7.1, without writing to the source;
  persistent selected-source churn fails before target writes.
- Given an unsupported or malformed source, the command exits with status
  `2` and identifies the compatibility problem.
- Given any existing MLflow session matching an incoming harness/session
  identity, including one created by another tool, `--on-conflict overwrite`
  safely replaces it, `--on-conflict duplicate` creates another independently
  addressable copy without changing it, and `--on-conflict ignore` leaves it
  unchanged.

### Selection

- With no harness or source option, all available adapter-owned default sources
  are eligible.
- `--harness` alone restricts discovery to that harness's default source(s).
- `--source` alone detects exactly one compatible harness from the explicit
  path.
- Supplying both options rejects a harness/source mismatch before MLflow
  writes.
- A session at exactly `fromTime` or `toTime` is included.
- A session outside the source creation-time range is excluded even if a later
  record falls inside the range.
- A directory filter includes descendants and excludes similarly prefixed
  sibling paths.
- A model filter matches a model recorded only on a nested source record.
- `--session SESSION-ID` selects only the matching source session and imports
  all of that session's importable logical turns.
- A session with exactly `N` logical turns is excluded by `--min-user-turns N`;
  a session with `N + 1` turns is included.
- Multiple filters, including `--session`, select only sessions satisfying
  every supplied condition.

### Mapping and fidelity

- Every selected logical turn produces one MLflow trace with the consistently
  encoded qualified source identity as its MLflow session identifier and the
  native source session ID in `mlflow_import.session_id`.
- Traces in one source session are separately addressable by MLflow trace ID
  and groupable by the shared session ID.
- Every turn's `mlflow.source.name` and normalized session/trace `source` equal
  the selected/detected canonical harness name, in both privacy modes and for
  default or explicit sources. Model/provider, importer name, and input path
  never replace that value.
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
- Existing non-importer traces are never modified unless their session is
  selected under the effective `overwrite` policy, which is also the default.
- `--redact` removes detected secrets and masks present user identity in native
  MLflow fields, session/trace data, and all artifact representations, while
  preserving reconciliation IDs and conversation roles and reporting redaction
  without exposing originals. Missing user fields remain absent, and SDK/OS user
  defaults cannot reintroduce identity after redaction.
- An overwrite that changes immutable native user/source metadata creates and
  verifies a compliant replacement before retiring the previous trace.
- `--dry-run` performs no MLflow mutation.
- A failed item does not prevent valid later items from being attempted, and
  the final exit status/report identifies the failure.
- Default-discovered and explicitly selected representations of the same source
  reconcile to the same identity; an ambiguous legacy/native alias causes no writes.
- Appending one turn preserves existing trace IDs and per-turn maps, refreshing
  only required session metadata; unchanged reruns perform no target writes.
- Retrying after an ambiguous write, changed source, or partial retirement
  neither loses the previous good trace nor silently creates an untracked copy.
- Structured, raw/escaped JSON, and malformed credential fixtures are masked
  everywhere in redacted mode; excluded credential-table values are never read.
- Large linear and branching fixtures exercise bounded memory and indexed
  traversal; operation counts rule out per-turn full-source/experiment scans.

## 15. Implementation boundary for the next phase

The implementation plan should be organized around these independently
testable units:

1. CLI/configuration, harness/source selection, and timestamp/path/model validation.
2. Read-only source snapshot and format detection.
3. Harness adapters and canonical source model for OpenCode, Claude Code,
   Codex, PI, and Gemini CLI.
4. Session filtering and logical-turn reconstruction.
5. MLflow experiment/trace/span/artifact writer.
6. Ownership, fingerprint reconciliation, and safe replacement.
7. Redaction, retry, reporting, and exit-status handling.
8. Harness fixtures and MLflow integration tests.

The first implementation should not add harness server/API support, live
watching, arbitrary custom filter expressions, or an MLflow run container
unless a later requirements change explicitly authorizes those features.
