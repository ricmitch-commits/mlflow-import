# Result schema and compatibility

Current JSON output uses **`mlflow-import.result.v1`**. This page describes the
implemented output in `reporting.py` and the typed read contract in
`compatibility.decode_result`; it does not introduce another wire version.
Result documents contain safe IDs, counts, hashes, codes and experiment names,
not prompts, responses, raw source paths or protocol exception bodies.

## Envelope

| Field | Shape / meaning |
| --- | --- |
| `schema_version` | Current schema string above. |
| `dry_run` | Boolean. Preview actions are not completed writes. |
| `experiment` | `name`, nullable string `id`, and `status`: `planned`, `created`, `resolved` or `unresolved`. |
| `source` | Nullable schema generation/fingerprint and non-negative `sessions_scanned`, `sessions_selected`, `logical_turn_count`, `record_count`. |
| `redaction` | `mode`: `verbatim` or `redacted`; non-negative `detector_counts`, `affected_field_count`, `affected_record_count`. |
| `summary` | Accounting, coverage and safe warnings described below. |
| `sessions` | Ordered session objects with nested turn objects. |
| `errors` | Array of safe `{code, message}` invocation errors. |

Multi-root results additionally provide `sources` and session-level `harness` /
`source_namespace`. Global session order is creation time (missing last), native
session ID, harness, namespace. Additive fields may occur; readers must not infer
that every additive field is exhaustively validated by the v1 decoder.

## Session and turn entries

Sessions include string `session_id`, boolean `selected`, `action`, nullable
`reason`, non-negative `record_count`/`logical_turn_count`, `warnings`, nullable
safe `error`, and `turns`. The action vocabulary for sessions, turns and predicted
actions is exactly **`imported | updated | skipped | failed | retired`**.

Turns include:

- `session_id` matching the parent, stable string `turn_key`, nullable positive
  `turn_index`, `action`, nullable `reason` and `planned_action`;
- nullable `trace_id`, boolean `source_present`, nullable safe `error_code` and
  `{code, message}` `error`;
- string arrays `affected_trace_ids`, `retired_trace_ids`, `conflicting_trace_ids`,
  plus a safe warning-event array.

Both levels expose nullable string `attempt_id`, `import_instance_id` and
`attempt_state`. Known state is one of **`prepared | ready_to_retire | complete |
aborted`**. These reflect verified journal readback; null does not claim a known
durable transition. Safe error codes match `[a-z][a-z0-9_]*`.

V2 turn maps have per-create physical `-create-<uuid>.json` locations because
their identity maps contain generated trace/span IDs. Creation tags record the
same location in `opencode.artifact_path` and `mlflow_import.turn_map`; upload
retries retain that location. Placement does not change logical fingerprints,
operation/client request identity or frozen attempt scope. Verified reuse reads
the candidate's recorded map, including older names and a missing legacy alias;
present pointers must agree. Existing immutable map objects are never overwritten.

For dry-run turns, `action="skipped"`, `reason="dry_run"`, and `planned_action`
predicts the action using the same enum. `summary.planned` counts predicted
imports, updates and retirements. A stale retirement has `source_present=false`;
it is not another current source turn.

## Accounting and partial results

- `selected_sessions` and `selected_turns` describe selection. Current source-turn
  buckets `imported_turns`, `updated_turns`, `skipped_turns`, `failed_turns`, and
  **`unchanged_turns`** are disjoint. An unchanged turn's entry is `skipped` with
  `reason="unchanged"`, but is not counted in `skipped_turns` again.
- Legacy aliases `turns_imported`, `turns_updated`, `turns_skipped` remain;
  `turns_failed` additionally includes failed stale retirements. Do not add aliases
  to their corresponding current counters.
- `retired_traces`, `failed_retirements`, `unexamined_owned_traces`,
  `failed_sessions` and `failed_items` are separate accounting dimensions, not
  extra current turns. Session/cleanup/retirement failures can increase
  `failed_items`; it need not equal `failed_turns`.
- `summary.coverage` contains its evidence version and
  `unavailable_turn_counts`. Warnings describe known gaps, not missing data
  fabricated as zero. `summary.warnings` holds safe report-level events.
- If work or terminal reporting exhausts its allowance, a fixed-size emergency
  envelope may replace the unavailable detailed report. Its unavailable counts
  are **null, not zero**; see the narrowly qualified exception below.

Exit-code behavior and retry/recovery are documented in [operations.md](operations.md).
CLI JSON is staged under finite reporting time/work/byte allowances before it is
published, so a rendering failure cannot append a replacement to partial JSON.
The API report is lazy and must be consumed before
`ImportResult.close()`. JSON decoding into ordinary Python lists/dicts is a
deliberately materializing operation for consumers choosing it.

### Qualified emergency envelope

Only the current **`mlflow-import.result.v1`** envelope can qualify for nullable
accounting. The decoder requires all of the following together:

- `summary.incomplete` is exactly `true`, `summary.counts_complete` is exactly
  `false`, and `summary.reporting_failures` is a positive integer (not a boolean).
- `summary.recovery_pending` is a boolean. `true` means target mutation was
  entered and recovery may be required; `false` means no target mutation was
  recorded. Neither value establishes that the import completed.
- `errors` contains at least one typed safe `{code, message}` event. The first
  event preserves the primary failure, which may be an earlier source/target
  error rather than the secondary reporting failure. Error codes are not limited
  to a timeout/quota allowlist for this reason.
- `sessions` is empty; `sources`, if present, is empty. The emergency report
  cannot carry unvalidated per-session, per-turn or source-ownership claims.
- The experiment is `unresolved` with null `id`; source `schema_generation` and
  `fingerprint` are present and null; redaction `detector_counts` is empty. The
  experiment name and redaction mode retain their ordinary types.

For this qualified shape only, the four source counters, the redaction
`affected_field_count` / `affected_record_count`, and all summary accounting
counters listed above (including aliases, `unchanged_turns`, `failed_items`,
`failed_sessions`, `failed_retirements`, and `unexamined_owned_traces`) must be
present and may be null or non-negative integers. Null means unavailable, never
zero. Dry-run `summary.planned` is required and its `imported`, `updated`, and
`retired` counters follow the same rule; if supplied on a normal-run emergency,
that object is validated identically. Warnings and errors keep their typed event
validation. Missing counts, booleans, negative values, strings, or containers are
not accepted as counts.

An `incomplete` flag or error event alone does **not** permit null accounting.
Ordinary current and historical v1 results retain non-negative integer count
requirements and the full session/turn validation. This exception grants no
ownership, reuse, mutation or deletion authority.

## Turn-map input-context index

The result wire schema remains v1. New artifact envelopes are `artifact-v4` under
`mapping-v9`; their immutable turn maps declare `mlflow-import.turn-map.v2`.
The shared raw session artifact remains `mlflow-import.session.v1`. The physical
per-create map locations described above are independent of this map wire version.

Each v2 map adds `input_contexts`, an array with one entry per projected accounting
invocation (including aggregate containers and steps). Each entry has `source_key`,
its generated `span_id`, and `descriptor`, identical to that invocation's
`mlflow_import.input_context` span attribute. The descriptor contains:

| Field | Meaning |
| --- | --- |
| `version`, `policy` | `mlflow-import.input-context.v1`, `conversation-delta-v1`. |
| `context_id` | Hash of the versioned qualified session / logical turn / scope / invocation tuple. |
| `qualified_session_id`, `turn_key`, `scope`, `invocation_id` | Logical ownership; empty scope denotes the primary execution. |
| `previous_context_id` | Immediately preceding context in the same scope, or null. No cumulative identity prefixes. |
| `origin` | `derived-observed-conversation`, `recorded-request-input`, or `unavailable`. |
| `provider_request` | `unknown` for derived/unavailable inputs; `recorded-input-only` for an explicit recorded input field. Neither claims a full effective provider request. |
| `empty` | Whether projected input is null or an empty messages delta. |
| `invocation_sources` | References to all effective source occurrences contributing to this invocation. |
| `delta` | Source references contributing to this input, with `message_index` (null for an explicit input field). |

Each reference has `source_key`, nullable `occurrence_id` / `entity_id`, and a
`component` identifying the source contribution. Coalesced fragments can share a
message index. Map `records` entries additionally carry qualified `source_key` and
recorded `scope`, so references can resolve even for effective Gemini projections
or records without explicit source IDs. Complete values remain in the session
artifact; the index is provenance, not another transcript copy or an artifact-path
pointer. Empty/unavailable deltas remain explicitly described.

`content.input_context.validate_input_contexts(mapvalue, ownership, *, spans=None) -> bool` checks
v2 descriptor ownership, versions, occurrence resolution, scope-local predecessor
chains, unique source/span bindings and accounting-invocation coverage. With actual
trace `spans`, it additionally checks the exact complete identity map, one-to-one
coverage of actual span descriptors, and exact source-key, invocation and descriptor
agreement on each corresponding span. Missing/extra/duplicate contexts or identities
are rejected. Invalid shapes return false. Callers separately verify artifact integrity
and ownership. Map and ownership mapping versions must agree: changing a v9 trace's
map schema to v1 cannot bypass context validation. Historical `mlflow-import.turn-map.v1` maps remain decodable for older
pricing provenance; the helper accepts them only without a new context index and
without a `mapping-v9` claim. This read compatibility does not upgrade an old
projection or permit an old prepared attempt to resume under a new render contract.

`content.turn_map.validate_turn_map_admission(replay, request, mode) -> None` streams
a prospective v2 map before mutation. It consumes the already-admitted request's
replayable child factory once and uses its frozen accounting provenance, without
calling the mapper. The shared `build_turn_map` builder also produces upload maps;
`iter_turn_map_json` enforces the 8 MiB canonical-byte bound in both paths. Maximum
generated identity widths (trace 35, span 16 characters) conservatively reserve all
identity occurrences, including root record targets and coalesced aliases. Late
artifact nonces/pointers are outside the map. Oversized maps raise the typed
`turn_map_too_large` error, independently of whether the trace itself fits its byte
and span-count resource profile.

## Typed historical reads

```python
import json
from mlflow_import.compatibility import decode_result

decoded = decode_result(json.loads(result_json))
assert decoded.decoder_version == "compatibility-v1"
print(decoded.original_schema_version)
```

The decoder accepts **`mlflow-import.result.v1`** and
**`opencode-mlflow.result.v1`**, preserving the original schema and document. It
checks the declared envelope/session/turn types, counters, action/lifecycle enums,
optional nested lifecycle/error fields and diagnostic shapes. Unsupported versions
raise `result_schema_unsupported`; malformed declared-v1 documents raise
`result_schema_invalid`. It is a typed read view, not an artifact migration command.

Synthetic historical captures in
`tests/fixtures/payloads/historical-36ed44f.{json,md}` and a live migration test
qualify that producer revision. Ownership aliases normalize the exact historical
`opencode-mlflow` marker only with the required identity/source evidence.
Successful decoding alone grants no reuse or deletion authority. Other historical
variants and old unbound journals are not automatically upgraded; see
[temporary-resources.md](temporary-resources.md#old-unbound-in-flight-journals).
