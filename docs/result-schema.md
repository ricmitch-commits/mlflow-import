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
- Resource exhaustion may require a fixed-size emergency envelope with
  `summary.incomplete=true`, `resource_failures`, and, after target work,
  `recovery_pending=true`. Its zero counters are not exact accounting of all work
  already completed. A success must not be inferred from those zeros.

Exit-code behavior and retry/recovery are documented in [operations.md](operations.md).
JSON output is streamed; the API report is lazy and must be consumed before
`ImportResult.close()`. JSON decoding into ordinary Python lists/dicts is a
deliberately materializing operation for consumers choosing it.

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
