# Result schema — `opencode-mlflow.result.v1`

The result document is the importer's **only** output surface. Every field
carries identifiers, counts, warning/error codes, hashes, timestamps, or
the experiment name — never prompt, response, tool, or source-code
content, absolute source paths, credentials, tracking-URI secrets, or
matched secret values.

This document is the human-readable contract for the serializer in
`src/opencode_mlflow/reporting.py`. The selected output mode (`--output
human|json`) renders the same document; JSON mode is deterministic,
sorted, and indented (`json.dumps(..., sort_keys=True, indent=2)`).

A complete JSON example (dry run, two sessions, one filtered out):

```json
{
  "dry_run": true,
  "errors": [],
  "experiment": {
    "id": "681508484302058365",
    "name": "team-sessions",
    "status": "resolved"
  },
  "redaction": {
    "affected_field_count": 0,
    "affected_record_count": 0,
    "detector_counts": {},
    "mode": "verbatim"
  },
  "schema_version": "opencode-mlflow.result.v1",
  "sessions": [
    {
      "action": "skipped",
      "reason": "not_selected",
      "selected": false,
      "session_id": "ses_01J8EXAMPLE0",
      "logical_turn_count": 0,
      "record_count": 0,
      "warnings": [],
      "error": null,
      "turns": []
    },
    {
      "action": "skipped",
      "reason": "dry_run",
      "selected": true,
      "session_id": "ses_01J8EXAMPLE1",
      "logical_turn_count": 1,
      "record_count": 2,
      "warnings": [],
      "error": null,
      "turns": [
        {
          "action": "skipped",
          "error_code": null,
          "error": null,
          "warnings": [],
          "planned_action": "imported",
          "reason": "dry_run",
          "session_id": "ses_01J8EXAMPLE1",
           "source_present": true,
           "trace_id": null,
           "affected_trace_ids": [],
           "retired_trace_ids": [],
           "conflicting_trace_ids": [],
           "turn_index": 1,
           "turn_key": "turn-1f0e3dad99993b1f7c1a..."
        }
      ]
    }
  ],
  "source": {
    "fingerprint": "9f86d081884c7d659a2fea...",
    "schema_generation": "current",
    "sessions_scanned": 2,
    "sessions_selected": 1,
    "logical_turn_count": 1,
    "record_count": 2
  },
  "summary": {
    "generated_ns": 1792713600000000000,
    "selected_sessions": 1,
    "selected_turns": 1,
    "imported_turns": 0,
    "updated_turns": 0,
    "skipped_turns": 1,
    "unchanged_turns": 0,
    "failed_turns": 0,
    "failed_retirements": 0,
    "failed_items": 0,
    "planned": {
      "imported": 1,
      "retired": 0,
      "updated": 0
    },
    "retired_traces": 0,
    "turns_failed": 0,
    "turns_imported": 0,
    "turns_skipped": 1,
    "turns_updated": 0,
    "unexamined_owned_traces": 0,
    "warnings": []
  }
}
```

(Session ids, hashes, and the experiment id above are illustrative
placeholders, not real values.)

## Top-level sections

Every top-level section is always present, including in a failure report
(unset sections carry their defaults), so a failure report is as valid a
result document as a success report.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Always `opencode-mlflow.result.v1`. |
| `dry_run` | boolean | Whether the run was a dry run. |
| `experiment` | object | `name` (the requested experiment name), `id` (resolved MLflow experiment id, `null` before resolution), `status`: `resolved` (existing), `created` (created by this run), `planned` (a dry run's planned creation), or `unresolved` (pre-write failure before resolution). |
| `source` | object | `schema_generation` (`legacy`, `current`, or `mixed`), `fingerprint` (the non-secret source database fingerprint — SHA-256 over the schema generation plus filesystem identity/statistics; never the absolute path or content), `sessions_scanned`, `sessions_selected`, `logical_turn_count`, `record_count` (counts across all compatible scanned sessions, including unselected sessions). |
| `redaction` | object | `mode` (`verbatim` or `redacted`), `detector_counts` (mapping of detector category → masked-position count), `affected_field_count`, `affected_record_count`. Counts only; verbatim mode reports zeros. |
| `summary` | object | See below. |
| `sessions` | array | One entry per scanned session, in scan order (selected or not). |
| `errors` | array | Invocation-level safe error entries `{code, message}` — the failure that aborted the run, when one did. |

## Summary

| Field | Type | Meaning |
| --- | --- | --- |
| `selected_sessions` | integer | Eligible session headers. |
| `selected_turns` | integer | Logical turns of eligible sessions. |
| `imported_turns` | integer | Current-source turns newly imported. |
| `updated_turns` | integer | Current-source turns replaced. |
| `skipped_turns` | integer | Current-source turns intentionally skipped. |
| `failed_turns` | integer | Current-source turns failed; disjoint from the other three current-turn counts. |
| `unchanged_turns` | integer | Subset of `skipped_turns` whose reason is `unchanged`. |
| `failed_items` | integer | All failed items: `failed_turns + failed_retirements`, plus one count for each sensitive temporary-data cleanup failure; cleanup-only failures do not become turn failures. With no invocation-level error this nonzero count produces exit `1`; an invocation-level error retains exit `2`. This equals `ImportResult.failed_items`. |
| `failed_retirements` | integer | Failed stale-trace retirements, excluded from the current-source-turn counts. |
| `turns_imported` | integer | Current turns newly imported this run. |
| `turns_updated` | integer | Current turns whose trace was replaced this run. |
| `turns_skipped` | integer | Compatibility alias of `skipped_turns`: current-source turns only, including dry-run current turns. Dry-run stale entries (`source_present=false`) are excluded. |
| `turns_failed` | integer | Failed turn items — including a stale-trace retirement that could not be performed. A cleanup-only failure increments `failed_items` and the overall exit status but not this turn count. |
| `retired_traces` | integer | **Trace-level actual** count of owned traces retired, including old/redundant traces and successful rollback/compensating retirements; always `0` in dry-run. Dry-run predictions live in `summary.planned.retired` and per-turn `planned_action`. Separate from turn counts. |
| `unexamined_owned_traces` | integer | **Trace-level** count of importer-owned traces outside this run's reconciliation scope (unselected sessions, unknown sessions, incomplete scans). Explicitly left untouched. |
| `warnings` | array | Report-level warning events `{code, message}` — e.g. `broad_selection`, exporter codes such as `unassociated_record`/`session_missing_id`, or temporary-data cleanup codes `temporary_snapshot_cleanup_failed`/`temporary_run_cleanup_failed`. Exporter codes are preserved as safe events, not reduced to a count. |
| `planned` | object | **Dry-run runs only**: predicted `{imported, updated, retired}` subcounts. |
| `generated_ns` | integer | Present when the run's clock provided it: result generation time as epoch nanoseconds. |

The normative counts (`imported_turns`/`updated_turns`/`skipped_turns`/
`failed_turns`) are **disjoint**: every current turn (`source_present`
true) is counted in exactly one bucket. `retired_traces` and
`unexamined_owned_traces` are trace-level counts and are deliberately
separate — a retirement removes an owned trace and is not a turn outcome
(except that a *failed* retirement also counts in the compatibility alias
`turns_failed`). A temporary snapshot or run-root cleanup failure is reported
by its warning code and contributes to `failed_items`/the overall exit status,
but does not describe a failed source turn. `turns_imported` and
`turns_updated` are compatibility aliases of `imported_turns` and `updated_turns`.

## Session entries

| Field | Type | Meaning |
| --- | --- | --- |
| `session_id` | string | The OpenCode session id. |
| `selected` | boolean | Whether the selection filters kept this session. |
| `action` | string | The session's roll-up action (see enum below). An unselected session is always `skipped` with reason `not_selected`. |
| `reason` | string \| null | `not_selected` for an unselected session; `dry_run` for a selected preview; otherwise null and per-turn reasons apply. |
| `turns` | array | Turn entries, in turn order. Unselected sessions carry an empty list. |
| `logical_turn_count` | integer | Source logical turns, even when the session is unselected. |
| `record_count` | integer | Permitted normalized source records, even when the session is unselected. |
| `warnings` | array | Safe `{code, message}` objects from the source scan. |
| `error` | object \| null | First failed item's `{code, message}`, or null. |

A selected session is `failed` if any item failed; otherwise `updated` if
any turn updated or successful imports and retirements were mixed; otherwise
`imported`, `retired` (retirement-only), or `skipped`. Dry-run sessions are
`skipped` with reason `dry_run`.

## Turn entries

| Field | Type | Meaning |
| --- | --- | --- |
| `session_id` | string | Owning session id. |
| `turn_key` | string | The turn's stable key, `turn-<64 lowercase hex>`. |
| `turn_index` | integer \| null | 1-based logical turn index within the session; stale-trace retirement entries preserve the stored owned index when available, otherwise `null`. |
| `action` | string | Exactly one of the action enum below. |
| `reason` | string \| null | See reasons below. |
| `planned_action` | string \| null | Dry-run runs only: the predicted action (`imported`, `updated`, or `retired`) from the same enum. Always `null` in a normal run. |
| `trace_id` | string \| null | The MLflow trace id wherever the run knows one (created, replaced, or to-be-retired). |
| `source_present` | boolean | `false` only for a stale-trace retirement: the trace's turn is no longer in the source. |
| `error_code` | string \| null | A failed item's safe machine-readable code. |
| `error` | object \| null | Safe `{code, message}` for a failed item; null on success. `error_code` is retained as a compatibility alias. |
| `affected_trace_ids` | array of strings | Every trace id involved in a conflict or cleanup failure; empty when none. |
| `retired_trace_ids` | array of strings | Every trace id successfully retired by this result, including a new trace removed by rollback/compensation; empty when none. |
| `conflicting_trace_ids` | array of strings | Every importer-owned trace left unresolved by an ownership or recovery conflict; empty when none. This is a subset of `affected_trace_ids` when the result also records cleanup context. |
| `warnings` | array | Safe `{code, message}` objects; never source content. |

## Action enum

`action` for both sessions and turns is exactly:

```text
imported | updated | skipped | failed | retired
```

- `imported` — the turn had no owned trace; a new trace was created,
  verified, and its artifact uploaded and verified.
- `updated` — the turn's owned trace no longer matched; one replacement
  was created and verified, then the old trace(s) retired.
- `skipped` — nothing needed to change: an unchanged turn carries reason
  `unchanged` (a subcount of `turns_skipped`, never a separate action);
  a dry-run turn carries reason `dry_run` with a `planned_action`.
- `failed` — the item failed; later items continued (exit status `1`).
- `retired` — appears on the session roll-up when a turn-level retirement
  occurred. Stale-trace retirements are reported as **turn-shaped
  entries** with `action="retired"`, `source_present=false`, and reason
  `stale_turn` or `session_tombstone`.

## Reasons

| Reason | Where | Meaning |
| --- | --- | --- |
| `unchanged` | turn | The owned trace verified exactly equivalent (unchanged re-run). Counted in `turns_skipped`. |
| `dry_run` | turn | A dry-run prediction; see `planned_action`. |
| `not_selected` | session | The filters excluded this session. |
| `stale_turn` | turn (retirement) | The trace's turn is gone from a selected, completely scanned session. |
| `session_tombstone` | turn (retirement) | The selected session now has zero logical turns, so all of its previously owned turns are retired. |

## Safe error codes

Errors are stable machine-readable codes with sanitized messages that
never include values, paths, credentials, or transcript content. They
appear in `errors[]` (invocation-level, exit status `2`) or a turn's
`error_code` (item-level, exit status `1`).

Representative invocation-level codes:

| Code | Meaning |
| --- | --- |
| `invalid_experiment_name`, `invalid_model`, `invalid_timestamp`, `invalid_min_user_turns` | Command-line semantic validation failed. |
| `import_lock_held` | Another importer holds the advisory lock for this endpoint + experiment. |
| `unsupported_schema` | The source database matches no catalogued schema. |
| `source_unavailable` | The source path is missing, not a regular file, not readable, or not a SQLite database. |
| `source_snapshot_failed` | A consistent read-only snapshot could not be established (busy timeout, disk space, population failure). |
| `mlflow_capability_unsupported` | The tracking target fails the pinned compatibility profile. |
| `replay_fingerprint_mismatch` | A replayed turn's fingerprint diverged from its discovery plan entry (internal inconsistency); aborted before any write. |
| `replay_session_unavailable`, `replay_turn_unavailable` | A discovery plan entry is absent from the snapshot. |
| `import_failed` | A sanitized catch-all for an unexpected internal failure. |

Representative item-level codes:

| Code | Meaning |
| --- | --- |
| `payload_too_large` | The turn's complete root value exceeds the negotiated inline payload limit; rejected before trace creation. |
| `ownership_metadata_too_long` | A source identity, ownership, outcome, or warning metadata value exceeds MLflow's exact metadata limit; rejected before trace creation rather than silently truncated. |
| `ownership_metadata_invalid` | Warning metadata is not a comma-separated list of documented lowercase codes; rejected before trace creation. |
| `source_timestamp_missing` | The target cannot represent absent source timing without inventing wall-clock instants; rejected before any span creation (source-derived fallbacks are allowed). |
| `partial_trace_cleanup_failed` | A partial create failed and bounded retirement of its known trace failed; the report retains the trace ID for recovery. |
| `trace_create_failed`, `turn_replay_failed`, `source_consistency_mismatch` | The write path could not produce or validate a request for this turn. |
| `trace_verification_failed`, `artifact_verification_failed` | A created trace or uploaded artifact did not round-trip; the new trace was rolled back. |
| `artifact_upload_failed` | The uploaded artifact did not match the streamed document at the requested path. |
| `duplicate_owned_identity` | Multiple owned traces for one identity could not be verified or deterministically resolved; nothing was created or retired. |
| `ownership_conflict` | An indexed trace lost or changed its importer ownership marker or safe source identity; no replacement or retirement was performed. |
| `replacement_cleanup_failed` | A verified replacement exists, but retiring redundant old traces failed (the affected trace ids are the operator's cleanup list). |
| `retirement_failed`, `trace_retirement_failed` | A stale owned trace could not be retired. |
| `mlflow_span_degraded` | Span creation degraded to a NoOp span (for example ambient OTEL exporter configuration rerouting span export); the trace was not silently lost. |

## Ownership metadata and artifact identity

Every trace the importer creates carries namespaced ownership metadata at
creation, written by the pure mapper and used by later runs for
reconciliation:

| Key | Kind | Meaning |
| --- | --- | --- |
| `opencode.importer` | trace metadata | Always `opencode-mlflow`. The marker and session/turn pair are only the lookup index; destructive cleanup additionally requires the complete creation-time ownership/source contract below. |
| `opencode.session_id` | trace metadata | The owning OpenCode session id. |
| `opencode.turn_index` | trace metadata | The positive 1-based logical turn ordinal. |
| `opencode.turn_key` | trace metadata | The turn's stable key. |
| `opencode.source_fingerprint` | trace metadata | The turn-local source fingerprint (run-mode-tagged). |
| `opencode.source_schema` | trace metadata | The source schema generation. |
| `opencode.importer_version` | trace metadata | The importer package version that wrote the trace. |
| `opencode.mapping_version` | trace metadata | The mapping contract version that wrote the trace. |
| `opencode.redaction_mode` | trace metadata | The effective `verbatim` or `redacted` mode. |
| `opencode.render_fingerprint` | trace metadata | The render fingerprint (artifact/mapping/importer versions plus redaction mode). |
| `opencode.operation_key` | trace metadata | Deterministic idempotency key: SHA-256 over the exact experiment/session/turn/render tuple. |
| `opencode.catalog_version` | trace metadata | Catalog version used by the source adapter; required for cleanup when present on production traces. |
| `opencode.artifact_path` | trace **tag** | The exact relative path of the turn's artifact. A tag (not metadata) because the pinned server silently truncates metadata values at 250 characters while the path is typically longer. |
| `opencode.artifact_sha256` | trace **tag** | The recorded digest of the uploaded artifact bytes (set after upload verification). |
| `opencode.artifact_size` | trace **tag** | The recorded artifact size in bytes. |

### Additional artifact provenance metadata

The trace also carries `opencode.artifact_schema_version` and the
non-secret `opencode.source_database_fingerprint` as provenance metadata.
The database fingerprint is not part of the turn reconciliation identity.

An indexed trace missing or carrying a malformed required ownership/source
field is reported as unexamined or an ownership conflict and is never
retired. Older traces are therefore preserved until their complete identity
can be re-verified safely.

Candidate warning metadata is accepted only as comma-separated lowercase ASCII
codes matching `[a-z][a-z0-9_]*`. Invalid warning metadata is replaced with
the generic `invalid_warning_metadata` code and treated as an ownership
conflict; arbitrary metadata values are never echoed in reports or logs.

Native session identity also rides the standard MLflow metadata key
`mlflow.trace.session` (set to the OpenCode session id), so all turns of
one session share one native MLflow session id.

## Artifact path contract

Each imported turn has exactly one streamed canonical JSON artifact at a
deterministic relative path inside the trace's artifact directory:

```text
opencode-mlflow/v1/sessions/<encoded-session-id>/turns/<turn-key>/<render-fingerprint>-<artifact-snapshot-fingerprint>.json
```

- `<encoded-session-id>` is the session id percent-encoded byte-wise (only
  `[A-Za-z0-9_-]` stay raw), so path traversal is impossible.
- `<turn-key>` is `turn-<64 lowercase hex>`; both fingerprints are
  `<64 lowercase hex>` — asserted, not assumed.
- The `<render-fingerprint>-<artifact-snapshot-fingerprint>` pair means
  the artifact document repeats the **complete session snapshot as of its
  creation** (see [architecture.md](architecture.md) for the
  turn-local versus session-complete fingerprint boundary): changing any
  turn of a session changes the snapshot fingerprint and therefore moves
  the path for **every** turn of that session. Historical artifacts at
  old paths are never deleted by the importer; each trace records the
  path its own artifact was written to, and candidate verification checks
  integrity at that recorded location.

How to inspect artifacts safely is covered in
[operations.md](operations.md).
