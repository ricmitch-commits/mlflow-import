# Operations runbook

This runbook is for the operator running `opencode-mlflow import`
against a real tracking server. The result document is specified in
[result-schema.md](result-schema.md); the compatibility profile in
[compatibility.md](compatibility.md).

## Normal operation

1. **Preview first.** Run with `--dry-run` (ideally with the same
   filters you plan to use). A dry run mutates nothing, acquires no
   lock, and reports every turn as `skipped`/`dry_run` with a
   `planned_action`.
2. **Run the import.** Re-run the same command without `--dry-run`.
   Exit `0` = complete or no-op; `1` = some items failed (inspect
    `summary.failed_items` (and, for turn failures, `summary.turns_failed` and
    the per-turn `error_code` values); `2` =
   the run aborted before any write (inspect `errors[]`).
3. **Re-run any time.** Re-runs are convergent: unchanged turns report
   `skipped` with reason `unchanged`, changed turns are replaced, stale
   traces are retired, and crash leftovers are cleaned up after a
   verified winner exists.

## Retries

Every MLflow call runs under a bounded exponential-backoff policy with
jitter (defaults: 4 total attempts, 0.5 s initial delay, 2× multiplier,
30 s maximum, 0.5 jitter fraction). Only **transient** failures are
retried: MLflow protocol `error_code`s in the 5xx/rate-limit vocabulary
(`INTERNAL_ERROR`, `INVALID_STATE`, `DATA_LOSS`, `NOT_IMPLEMENTED`,
`TEMPORARILY_UNAVAILABLE`, `DEADLINE_EXCEEDED`,
`REQUEST_LIMIT_EXCEEDED`, `RESOURCE_EXHAUSTED`) and transport
timeout/connection failures. Typed importer errors — an unsupported
schema, an oversized payload, a capability failure — are permanent and
never retried, and neither are control-flow interruptions. A transient
failure that exhausts the budget still performs the final
ambiguity search. If an owner is found but cannot be verified, recovery stops
without another create and preserves every discovered id in the result's
`affected_trace_ids` and `conflicting_trace_ids` arrays. Only an exhausted
attempt with no discovered owner re-raises the original error, which surfaces
as a failed item (exit 1) or an invocation failure (exit 2) depending on where
it happened.

Creation is additionally idempotent under ambiguity: the create request
carries the deterministic `opencode.operation_key`
(SHA-256 of the exact experiment/session/turn/render tuple) as its
client request id, and after an ambiguous timeout the writer walks every
search page for the exact owner/session/turn/render/operation tuple **before**
retrying the create. A completed empty search is distinct from a failed or
partial search: a failed search is never treated as proof that no trace
exists. If any survivor has an ownership conflict, changed source/render
contract, missing/unverified artifact, or otherwise cannot be fully verified,
recovery stops without another non-idempotent create. Only fully verified exact
survivors can select the newest winner (creation timestamp, then greatest trace
id) and permit redundant cleanup. All old, discovered, conflicting, and
cleanup trace ids are retained in the safe conflict result for operator
recovery. The `affected_trace_ids`, `retired_trace_ids`, and
`conflicting_trace_ids` arrays are serialized on every turn result, even when
empty.

## Concurrency and the advisory lock

A normal run acquires one **non-blocking** advisory lock before
experiment resolution and owned-trace lookup, serializing concurrent
importer invocations for the same tracking endpoint and exact
experiment name **on one host**.

- Lock identity: the lock file name is a SHA-256 of a versioned header,
  the *normalized* tracking endpoint (userinfo credentials, query
  strings, and fragments stripped; scheme/host case and default ports
  canonicalized), and the exact experiment name — credential-free by
  construction. The lock file itself carries only the normalized
  endpoint, the experiment name, and the acquiring pid.
- Lock root: `XDG_RUNTIME_DIR` when set (per the XDG spec, a per-user
  `0o700` directory); otherwise a restrictive `0o700` subdirectory of
  the system tempdir. The lock file is mode `0o600`.
- A held lock fails immediately with exit `2` and code
  `import_lock_held`; the message names the lock file path.
- The lock is released after success and after exceptions (the process
  always removes the file it created).
- **Limitation:** concurrent invocations from *different hosts* are not
  supported and must be serialized by the operator. Dry runs never
  acquire the lock (they mutate nothing).

### Stale-lock recovery (operator-confirmed, manual)

The importer never removes a lock file it did not acquire itself. If a
run crashed without releasing its lock:

1. Confirm no `opencode-mlflow` import process is still running for
   that tracking endpoint and experiment (for example with `ps` or
   your process manager).
2. Confirm the lock is stale — the process that created it is gone
   (the lock file's payload records the creating pid).
3. Remove the lock file manually:
   `rm <lock-root>/import-<endpoint+experiment-sha256>.lock`.
   The exact path is printed in the `import_lock_held` error message.

A crashed run leaves no other persistent state on the importing host:
the temporary run root (snapshot, discovery plan, artifact staging) is
removed in the run's `finally` path. A cleanup failure is reported with the
generic `temporary_run_cleanup_failed` warning and contributes one
`summary.failed_items` count, but is not described as a failed source turn.
A hard kill leaves at most an ordinary restrictive temp directory for the
OS temp-cleaner.

## Scale and memory limitation

Discovery spools content-free turn summaries and the plan is replayed from
JSONL, so source transcript payloads are not accumulated in Python during
discovery. The final reconciliation plan and result document intentionally
retain sanitized identifiers and per-item outcomes because those are output
contract fields; memory therefore grows with output cardinality. The measured
scale suite isolates importer-owned structures and deliberately excludes the
MLflow SDK's live span-payload retention. It is not a strict full-SDK memory
guarantee for a 100,000-span live import.

## Replacement, rollback, and duplicate candidates

Ownership is the complete creation-time contract — importer marker,
nonempty session id, `turn-<64 lowercase hex>` turn key, positive turn index,
64-lowercase-hex source/render/operation fingerprints, supported source
schema, importer/mapping/artifact versions, redaction mode, catalog version,
and the exact experiment/trace identity — never the native MLflow trace id alone, so a replacement keeps
its source identity while getting a new trace id. The marker plus
session/turn pair is only a search index. An indexed trace with missing or
malformed required fields is `unexamined_owned_traces`/`ownership_conflict`
and is never retired. An "update" replaces one turn's trace; siblings of the
same session are untouched (their turn-local fingerprints did not change),
and historical artifacts at old paths are preserved.

**Replacement ordering is strict:** create the new trace → verify its
full payload → upload and verify the artifact → record the verified
digest on the trace → only then retire the old trace(s). Every final
retirement gateway operation re-fetches the exact experiment, importer
marker, session/turn identity, and applicable candidate contract in the
same operation boundary; any identity race is an `ownership_conflict` and
does not delete the trace.

Failure handling, in order of severity:

- **Create or verification fails** → the old trace remains intact.
- **Child creation fails after the root exists** → unfinished spans are
  ended best-effort using source timing, and the known partial trace is
  retired under bounded rollback. A failed rollback reports
  `partial_trace_cleanup_failed` and retains its trace ID for recovery.
- **A created trace fails span verification or artifact upload** → the
  NEW trace is retired as rollback and the original failure is
  reported; the old valid trace is retained. A successful rollback appears in
  the action's `retired_trace_ids` and the summary's `retired_traces` count.
- **Rollback itself fails** → the writer retries within the same
  bounded policy, reports both trace ids as a reconciliation conflict
  (`duplicate_owned_identity` when the identity cannot be
  deterministically resolved), and still never retires the old trace.
  The orphaned new trace remains importer-owned and therefore
  discoverable for explicit operator recovery — it is never an
  untracked fallback.
- **Old-trace retirement fails after a verified replacement** → bounded
  retries and compensating retirement of the new trace are attempted;
  the old trace is never deleted on uncertainty. A successful compensating
  retirement is also included in `retired_trace_ids`; the action reports
  `retirement_failed` (or `duplicate_owned_identity` if compensation fails).
- **Redundant-trace cleanup fails after finding a verified winner** → the
  verified winner is retained, but the item is `failed` with
  `replacement_cleanup_failed` and exit 1, not a successful unchanged skip.
  Stale-retirement results are reported once per trace ID, even when several
  stale traces share a session/turn identity.

**Recovery from any of these is: re-run the importer.** The next run
verifies every candidate for each identity, keeps the newest verified
exact match (ties broken by the lexicographically greatest trace id),
creates one verified replacement when none match, and — after a
verified winner exists — retires every other owned candidate for that
identity, including incomplete candidates left by a prior crash (for
example a trace whose artifact digest was never recorded because the
process died between upload and recording). An owned candidate with
missing or unverified artifacts is never treated as an unchanged skip.

**Unowned traces are never touched.** The ownership search may use the
server-side metadata filter when the target accepts it, but every page
is re-verified locally against the exact ownership marker; a trace
without the marker is never yielded as a candidate, updated, or
retired — even if it happens to share a native session id.

## Stale retirement scope

Stale-trace retirement is deliberately narrow:

- Only traces of **selected** sessions whose source scan **completed**
  are eligible. A filtered run never retires an unselected session's
  traces; a selected-but-incomplete scan cannot prove a turn stale, so
  its candidates are reported as `unexamined_owned_traces` instead.
- A turn that disappeared from a selected, completely scanned session
  retires that turn's owned traces (reason `stale_turn`,
  `source_present=false` in the report).
- A selected session whose scan completed with **zero logical turns**
  is an explicit tombstone: all of its previously owned turns are
  retired (reason `session_tombstone`). This is why omitting
  `--min-user-turns` keeps zero-turn sessions selectable.
- Owned traces for unselected sessions, unknown sessions, incomplete
  scans, and candidates without a parseable identity are counted in
  `summary.unexamined_owned_traces` and explicitly left untouched.

## Artifacts

Each imported turn has exactly one streamed canonical JSON artifact at
a deterministic relative path under the trace's artifact directory:

```text
opencode-mlflow/v1/sessions/<encoded-session-id>/turns/<turn-key>/<render-fingerprint>-<artifact-snapshot-fingerprint>.json
```

Each artifact repeats the **complete permitted session snapshot as of
its creation**, so every trace stays independently inspectable without
re-reading other turns' artifacts. Artifacts are immutable historical
snapshots: when a turn changes, the replacement artifact is written at
a new path (the session-complete snapshot fingerprint moved) and the
old trace — and its recorded artifact — are retired as history.

**Artifact cleanup is intentionally not automatic.** The importer never
deletes artifact files; retiring a trace removes the trace record, and
the artifact bytes under the experiment's artifact root remain for
audit. To reclaim space, an operator with access to the artifact store
may delete artifacts belonging to retired traces — the safest procedure
is to enumerate retired trace ids from result documents (the
`retired_traces` count and per-turn `trace_id` values with
`action="retired"`), then remove the corresponding
`traces/<trace_id>/artifacts` directories under the experiment's
artifact root using the server's artifact storage tooling. Do this only
after confirming you do not need the historical payloads; the importer
itself will never recreate them.

### Inspecting an artifact without exposing secrets

The artifact document contains whatever the run persisted — in verbatim
mode that includes full transcript content. To inspect one safely:

- Prefer inspecting **redacted-mode** artifacts (runs with `--redact`).
- Download through the MLflow client from a machine that already has
  access to the tracking server, and inspect locally — do not paste
  artifact content into tickets, logs, or chat channels.
- The document is canonical JSON with sections `artifact`,
  `identity_map`, `session`, `session_records`, `turn`, `turn_records`,
  `unassigned_records`, `warnings`. Identity information (session id,
  turn key, source ids, fingerprints) is safe to quote; record payloads
  are not.
- In redacted mode, masked values appear as `[REDACTED]` (text) or
  `[REDACTED_BINARY]` (opaque bytes); if you see either marker, the
  detector worked and the original value is not recoverable from the
  artifact.

## Common failure codes and what to do

| Code | Exit | Operator action |
| --- | --- | --- |
| `import_lock_held` | 2 | Another importer is running for this endpoint+experiment (or a stale lock remains). See stale-lock recovery above. |
| `mlflow_capability_unsupported` | 2 | The tracking target fails the pinned compatibility profile. Check the client/server versions and [compatibility.md](compatibility.md). |
| `unsupported_schema` | 2 | The source database matches no catalogued schema. Check the OpenCode version that produced it. |
| `source_unavailable` / `source_snapshot_failed` | 2 | The source path is missing/unreadable/not a database, or a consistent snapshot could not be established (busy timeout, disk space). |
| `replay_fingerprint_mismatch` | 2 | Internal inconsistency between discovery and replay over the same snapshot; aborted before any write. Re-run; if it persists, report it. |
| `payload_too_large` | 1 | One turn's complete root value exceeds the negotiated inline payload limit. The item was rejected, not truncated. |
| `ownership_metadata_too_long` | 1 | A required source/ownership/outcome metadata value exceeds MLflow's exact metadata limit. The item was rejected before trace creation, not truncated. |
| `ownership_metadata_invalid` | 1 | Warning metadata contains a code outside the documented lowercase format. The item was rejected before trace creation. |
| `duplicate_owned_identity` | 1 | Multiple owned traces for one identity could not be verified or deterministically resolved. Inspect the affected trace ids (in the report's turn entry), decide which to keep, delete the others with the MLflow client, and re-run. |
| `replacement_cleanup_failed` | 1 | A verified replacement exists but retiring redundant old traces failed; the affected trace ids are listed. Re-run — the next run converges. |
| `temporary_snapshot_cleanup_failed` | 1 | Temporary snapshot cleanup may have left sensitive data behind. Restrict access to the run's temporary root, remove it securely, and re-run. The report contains only this safe warning code/message, never a path. |
| `temporary_run_cleanup_failed` | 1 | Temporary run-root cleanup may have left sensitive data behind. Restrict access to the temporary root, remove it securely, and re-run. The warning contributes to `failed_items`, not `turns_failed`; an already-aborted preflight still retains its original exit status. The warning contains no path. |
| `mlflow_span_degraded` | 1 | Span creation degraded to a NoOp span — typically ambient `OTEL_EXPORTER_OTLP_*` configuration rerouting span export. Unset those variables and re-run. |

## Choosing an experiment strategy

One MLflow experiment can hold traces from many runs and databases;
ownership is per `(session_id, turn_key)` within the experiment, so
re-importing the same source into a *different* experiment name creates
a second, independent copy (the ownership lookup is experiment-scoped).
Pick one experiment per logical corpus and keep using it — that is what
makes re-runs converge instead of duplicate.
