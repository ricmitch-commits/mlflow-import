# Import operations

Use the [qualified MLflow profile](compatibility.md), the
[supported source representations](source-adapters.md), and the documented
[temporary resource limits](temporary-resources.md). This is a sequential one-shot
importer using public MLflow APIs, not a synchronization service.

## Preview and import

```sh
mlflow-import --experiment-name coding-sessions --dry-run --output json
mlflow-import --experiment-name coding-sessions --harness pi --redact
mlflow-import --experiment-name coding-sessions --source /path/to/session-source --output json
```

The flat form above and an explicit `import` subcommand are accepted. The
experiment name is required. An explicit source must match one qualified harness;
`--harness` may disambiguate and must agree with the source. Without `--source`,
available adapter-owned default roots are discovered, optionally restricted by
`--harness`. Root overrides are listed in [source-adapters.md](source-adapters.md).

Selection filters combine with AND. `--session` selects the whole native session;
`--from-time`/`--to-time` are inclusive creation-time bounds; `--directory` matches
component descendants; `--model` is an exact normalized identity; and
`--min-user-turns N` means strictly more than N. Required ancestors/descendants of
direct matches are retained without selecting unrelated siblings through an
ancestor. No optional selection filter produces a `broad_selection` report warning.

All default roots freeze and complete read-only identity/conflict/projection
preflight before writes. Global ordering is creation time (missing last), native
session ID, harness, then namespace. Dry-run uses that same schedule and performs
no target writes or writer-lock acquisition. Sources are read-only; no credential
tables or excluded access/share columns are exported. SQLite input stays in one
read transaction; selected file streams receive stability checks.

Imports use a discover-first, session-at-a-time lifecycle for both explicit
sources and default roots. The complete filtered selection and read-only
validation finish before writes. The immutable source snapshot and sanitized
schedule stay on disk; execution then finishes one selected session before
starting the next, including recovery, trace verification and artifact uploads.
Each session has a disposable planning store, replay index, mapping context and
gateway verification/write caches. Its sanitized outcomes enter the separately
owned report before its execution scope exits and its planning store is closed
and removed, including after an item failure. Selected zero-turn sessions still
perform tombstone reconciliation. The writer lock and timeout cover the whole
invocation. Releasing session objects makes their memory reusable by Python;
the process's resident memory need not immediately decrease, and peak memory
still depends on the largest active trace and SDK buffers.

Source discovery evaluates attributable header metadata before freezing bodies.
Claude mode/cwd preludes and fork-context sidecars, and Gemini's qualified
persisted project-registry lookup, are described in
[source-adapters.md](source-adapters.md). Collisions in provably excluded sessions
do not block the plan. If eligible/unknown or required-relative files still claim
one native identity, `ambiguous_source` identifies the harness and fails before
writes; creation time, size and filename are not authority. Narrow the source or
filter scope, or establish authoritative source data, rather than automatically
choosing a newest file. Unsupported selected formats and metadata mutations
remain explicit failures.

The post-correction default-source production dry-run completed with exit 0:
8 sessions scanned, 4 selected, 9 selected turns, 157 source records and no error
codes. It imported/updated no turns and retired no traces. Sanitized evidence and
test commands are in [errors.md](errors.md#source-discovery-correction).

### Gemini A2A recordings with a repeated native ID

Some Gemini A2A recordings use `sessionId: "a2a-server"` in multiple files,
including multiple recordings for the same project. Distinct `startTime`
values and filenames do not establish that one recording supersedes another.
These files currently fail with `ambiguous_source`; a fresh MLflow experiment
does not resolve a source-identity collision. `--session a2a-server` still
selects every eligible claimant, and `--on-conflict` only controls target-side
conflicts.

To proceed with other harnesses, explicitly run `--harness opencode`,
`--harness claude`, `--harness codex` and `--harness pi` as separate invocations
with the same experiment and directory filters. This excludes Gemini from
those invocations; it does not import or repair its recordings. Importing all
colliding A2A recordings requires a qualified source-identity representation
or authoritative source correction. Do not treat sequential single-file
imports as independent sessions when they share the same qualified identity.

## Trace resource preflight

Both dry-run and normal mode project and admit every selected trace before
target mutation, including recovery. The default aggregate envelope is
64 MiB of conservatively bounded serialized spans, 4096 total spans (root
included), and 2 MiB of control metadata. Half the control envelope is reserved
for bounded late attempt/instance/map/session decorations; the actual request
is checked again at creation. Oversized traces fail with
`trace_resource_envelope_exceeded`; source values are not silently truncated
to make the aggregate fit. The safe error names the offending session and turn
(`session <id> turn <index> (<turn_key>): ...`) and attaches the measured span
count/bytes when that measurement stays within its bounded diagnostic ceiling
(currently 2 GiB and 32,768 spans). The measurement consumes the request
without retaining its admitted child DTOs. With `--log-level DEBUG`, each
admitted turn's byte, span and control counts are also written to stderr.

`--max-trace-bytes` and `--max-trace-spans` explicitly raise the per-trace
admission limits up to absolute caps of 2 GiB and 32,768 spans; defaults stay
64 MiB and 4096. A trace admitted under raised limits stores its actual
size/count proof and remains readable by later default invocations within the
absolute caps. Raising the limits is an explicit capacity decision: the
qualified server profile covers only the default envelope, so large traces can
exhaust importer, MLflow or PostgreSQL memory.

`trace_resource_profile_unverified` means a stored trace lacks the new writer's
recognized creation-time resource proof. Full payload retrieval, unchanged
reuse and pricing donation are blocked before downloading spans. Narrowing
source selection may avoid an unrelated legacy trace; a legacy trace needing
reuse requires a separately qualified offline migration. Do not manually add
proof tags to an old trace. The proof relies on trusted writers and immutable
span payloads; see [compatibility.md](compatibility.md#aggregate-trace-admission-and-metadata-first-reads).

## Invocation timeout

`--timeout-seconds SECONDS` sets a finite positive wall-clock budget for the whole
invocation (default: `600` seconds). Fractional seconds are accepted. The shared
budget includes source discovery, preflight, recovery, and writing; it does not
restart for each session or phase. Invalid values fail with `invalid_timeout`
before source configuration or discovery.

For a larger workload, explicitly give the same SessionsAI import 30 minutes:

```sh
mlflow-import --experiment-name SessionsAI --timeout-seconds 1800
```

Add `--log-level DEBUG` for live phase/count progress on stderr. The default stays
600 seconds and is never silently raised. The separate shared allowances remain
64 retries and 1,000,000 operations; increasing the timeout does not increase
either allowance. Exhaustion is a reported failure. Resolve the reported condition
and rerun to use normal recovery.

## Remote artifact configuration

The pinned client/server pair supports bounded reads through `mlflow-artifacts:/`
and the MLflow HTTP(S) artifact proxy. The machine running the importer does not
need access to the proxy's backing filesystem. Scheme/binding/auth/TLS validation
and a bounded read-only listing precede target writes, including in dry-run.
New experiment creation explicitly pins a qualified unique location beneath the
public Default experiment's location; an unavailable/unqualified hint fails
before creation rather than trusting an undisclosed server default.

Use standard MLflow settings for both version attestation and artifact reads:

- `MLFLOW_TRACKING_USERNAME` **and** `MLFLOW_TRACKING_PASSWORD` enable Basic
  authentication, taking precedence over `MLFLOW_TRACKING_TOKEN` (Bearer).
  Missing environment components fall back to `[mlflow]` entries
  `mlflow_tracking_username` / `mlflow_tracking_password` in
  `~/.mlflow/credentials` (bounded to 64 KiB).
- `MLFLOW_TRACKING_SERVER_CERT_PATH` selects a CA bundle/directory;
  `REQUESTS_CA_BUNDLE` and `CURL_CA_BUNDLE` are fallbacks. An explicit selection
  replaces default trust roots for artifact GETs and version attestation; it does
  not add the selected CA to the OS/default trust store.
- `MLFLOW_TRACKING_CLIENT_CERT_PATH` selects the PEM client certificate/key file.
- `MLFLOW_TRACKING_INSECURE_TLS=true` disables verification and cannot be combined
  with `MLFLOW_TRACKING_SERVER_CERT_PATH`.

Custom `MLFLOW_TRACKING_AUTH` plugins and AWS SigV4 are unqualified and fail
configuration validation. Redirects and HTTP compression are rejected. Listing
and download envelopes are in [temporary-resources.md](temporary-resources.md).
An authorization/transport error is never treated as proof that an object is
absent. `artifact_transport_failed`, `artifact_listing_invalid`,
`artifact_listing_exhausted`, `artifact_verification_failed` and
`artifact_http_configuration_invalid` are safe failures, not permission to skip
verification. Fix the reported condition and rerun for normal recovery.

## Conflict policy and reruns

| `--on-conflict` | Behavior |
| --- | --- |
| `overwrite` (default) | Reconcile the canonical imported instance; replace conflicting foreign sessions only after the complete incoming replacement verifies. |
| `ignore` | Preserve the existing conflicting copy and skip the incoming session after required recovery handling. |
| `duplicate` | Create an independent import instance; an interrupted same-input attempt resumes that instance. |

Qualified native/legacy aliases are read from both tags and metadata. Bare native
IDs require an unambiguous source identity; contradictory or cross-root ambiguous
identity fails rather than guessing. An unchanged verified import reuses trace/map
IDs and performs zero trace/artifact/tag/state mutations. An append reuses existing
turns; an edit replaces affected immutable content. Managed mutable tags are
refreshed/deleted with readback when needed.

With `mapping-v8`, the native session/parent IDs are bare source IDs. Use the
immutable `mlflow_import.qualified_session_id` creation tag to distinguish source
stores, and ownership/operation tags to diagnose recovery. Request metadata carries
only optional `parent_session_id` plus MLflow-owned/promoted API fields. OpenCode
native usage carries available cache counters and available persisted USD costs
(unavailable cache lifetimes and cost breakdowns stay NULL); canonical accounting
in the turn/session artifacts retains the richer raw observations. Absent native
client IDs
are NULL. Imported trace names use the canonical per-session harness through the
`mlflow.traceName` tag; `opencode_conversation` is the OpenCode root span name. Old
ownership metadata is still read, with conflicting aliases rejected. A mapping
version change replaces affected traces through the normal verified recovery
workflow; it does not copy or move reference traces between experiments.

## Recovery and concurrency

A session publishes a bound `prepared` journal, stages tagged traces and shared
artifacts, verifies the complete replacement set and manifest, publishes
`ready_to_retire`, retires authorized exact IDs, and publishes `complete`.
Replacement verification now runs before `ready_to_retire` is published, so a
defective batch stays `prepared` and can be aborted normally. `aborted` records
rejected pre-ready work. `discarded` records a published ready batch that no
longer verifies its own recorded replacements; its exact staged replacement
traces are removed only after attempt/ownership checks pass and only when every
recorded retirement still exists, so prior verified work is never abandoned.
The recorded `ready_to_retire` object remains as audit evidence. A source change
before ready aborts only new staging; a verified ready batch finishes before
reconciling changed input. Prior verified traces survive failed staging. Partial
deletions resume with confirmed counts, including empty/zero-turn completion.

Retry the invocation after resolving the safe reported failure. Filters do not
authorize deletion of unrelated work; pending recovery excluded by selection is
reported rather than silently ignored. A checksum alone is not retirement
authority: source/namespace/attempt/instance/chain, manifest, replacement and target
identity checks must all pass. Old unbound in-flight artifacts fail with
`attempt_state_invalid`; follow the non-destructive
[operator procedure](temporary-resources.md#old-unbound-in-flight-journals).
There is no automatic trust upgrade for unqualified journal formats; the only
automatic discard applies to a checksum-valid ready batch that fails its own
recorded replacement verification, and it removes only that attempt's staged
replacement traces.

An OS advisory lock serializes local writers per tracking endpoint/experiment.
Multi-host writers need external serialization. Retries share finite invocation
deadlines/work allowances; exhaustion is a reported failure, not a reason to
continue with incomplete state.

## Privacy, output and lifetime

Default mode is **verbatim** permitted content. `--redact` applies defined secret
and user-identity detectors before outbound hashing/publication, including encoded
JSON fragments, malformed assignments, native user metadata and retained raw
copies. Structural reconciliation IDs and the `user` conversation role remain.
The result contains safe identifiers/counts/codes, not transcript or tool content.
`--log-level DEBUG` emits phase/count progress to stderr; final output is stdout.

Exit codes: **0** complete/no-op, **1** item/operational or pending-recovery failure,
**2** invalid invocation or pre-write failure. Argument syntax errors use argparse
diagnostics; do not assume every syntax-error response is a JSON result.
See [result-schema.md](result-schema.md) for machine-readable fields, dry-run plans
and incomplete emergency accounting.

The CLI streams and closes its result. API users should close the lazy report:

```python
from contextlib import closing
from mlflow_import.config import parse_config
from mlflow_import.importer import build_dependencies, run_import
from mlflow_import.reporting import iter_json

config = parse_config(["--experiment-name", "coding-sessions", "--dry-run"])
with closing(run_import(config, build_dependencies(config))) as result:
    for chunk in iter_json(result):
        print(chunk, end="")
```

Do not access the report after close. `render_json()`/`render_human()` intentionally
materialize strings. Shared temporary exhaustion is terminal; quota increases do
not make unsupported artifact transports, schemas or journals compatible.
