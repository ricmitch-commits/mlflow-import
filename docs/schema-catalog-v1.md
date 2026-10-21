# OpenCode Schema Catalog v1

> This page is part of the documentation suite: [README](../README.md) ·
> [usage](usage.md) · [result schema](result-schema.md) ·
> [architecture](architecture.md) · [security](security.md) ·
> [testing](testing.md) · [operations](operations.md).

Version: `opencode-schema-catalog-v1` (constant `SCHEMA_CATALOG_VERSION` in
`src/opencode_mlflow/source/catalog.py`, which is the authoritative data; this
document mirrors that matrix for review). Mixed-generation precedence
(current rows win overlapping session ids) is described in
[architecture.md](architecture.md).

The catalog is pure data. Each entry names required tables/columns,
optional known columns, join predicates, timestamp columns and units, JSON
locations, record classification rules, permitted session-scoped tables,
authoritative sequence domains, fixed tie ranks, and raw-field handling.
A database whose layout cannot be matched to a catalog entry fails schema
probing with a safe compatibility diagnostic (`unsupported_schema`, exit
status `2`) that names the missing table or column — never raw SQL values
or source content.

## Generation classification

`probe_schema` classifies a source database as:

| Generation | Matches when |
| --- | --- |
| `current` | `session_v2` plus `session_message` match the current entry below. |
| `legacy` | `session`, `message`, and `part` match the legacy entry below. |
| `mixed` | Both entries match. |
| unsupported | Neither entry matches (missing table or required column). |

Unknown additive columns (e.g. `session_v2.custom_flag`) are tolerated and
observed. Catalogued optional columns are read when present. Session-scoped
tables are described when present but are **not** required for a match. A
`session_share` table that exists with no catalogued safe-column entry is
omitted with the warning `session_share_omitted`.

Security exclusions are applied before any exporter projection, including
observed additive columns: `share_url`, `share_token`, `share_secret`,
`access_token`, `access_key`, `session_token`, and `session_access_token` are
never captured from a permitted table. The schema probe may report their
column names as metadata, but their values do not enter the normalized
snapshot.

## Legacy generation (`legacy`)

### `session` (required)

| Column | Notes |
| --- | --- |
| `id` | |
| `project_id` | |
| `workspace_id` | |
| `parent_id` | |
| `slug` | |
| `directory` | |
| `path` | |
| `title` | |
| `version` | |
| `summary_additions` | |
| `summary_deletions` | |
| `summary_files` | |
| `summary_diffs` | JSON |
| `metadata` | JSON |
| `cost` | |
| `tokens_input` | |
| `tokens_output` | |
| `tokens_reasoning` | |
| `tokens_cache_read` | |
| `tokens_cache_write` | |
| `revert` | |
| `permission` | |
| `agent` | |
| `model` | JSON |
| `time_created` | epoch ms |
| `time_updated` | epoch ms |
| `time_compacting` | epoch ms, optional known column |
| `time_archived` | epoch ms, optional known column |

### `message` (required)

`id`, `session_id`, `time_created` (epoch ms), `time_updated` (epoch ms),
`data` (JSON).

### `part` (required)

`id`, `message_id`, `session_id`, `time_created` (epoch ms),
`time_updated` (epoch ms), `data` (JSON).

### Joins

- `message.session_id = session.id`
- `part.message_id = message.id`
- `part.session_id = session.id`

### Sequence domains

None — legacy ordering uses normalized timestamps, the fixed tie rank,
then stable source IDs.

## Current generation (`current`)

### `session_v2` (required)

| Column | Notes |
| --- | --- |
| `id` | |
| `project_id` | |
| `workspace_id` | |
| `parent_id` | |
| `fork_session_id` | |
| `fork_boundary` | epoch ms (v1 assumption: a Timestamp; retained raw by adapters if a schema review disproves this) |
| `slug` | |
| `directory` | |
| `path` | |
| `title` | |
| `version` | |
| `summary_additions` | |
| `summary_deletions` | |
| `summary_files` | |
| `summary_diffs` | JSON |
| `metadata` | JSON |
| `cost` | |
| `tokens_input` | |
| `tokens_output` | |
| `tokens_reasoning` | |
| `tokens_cache_read` | |
| `tokens_cache_write` | |
| `revert` | |
| `permission` | |
| `agent` | |
| `model` | JSON |
| `time_created` | epoch ms |
| `time_updated` | epoch ms |
| `time_idle` | epoch ms |
| `time_viewed` | epoch ms |
| `idle_outcome` | |
| `time_compacting` | epoch ms |
| `time_archived` | epoch ms |
| `time_suspended` | epoch ms, optional known column |
| `resume_attempts` | optional known column |

### `session_message` (required)

`id`, `session_id`, `type`, `seq`, `time_created` (epoch ms),
`time_updated` (epoch ms), `data` (JSON).

### Joins

- `session_message.session_id = session_v2.id`

### Sequence domains

- `session_message (session_id, seq)` — authoritative and unique; the
  primary order domain for current records.

## Timestamp units

Both profiles record OpenCode `Timestamps` as **epoch milliseconds** in
catalog v1. The model's timestamp normalizer accepts
`s`/`ms`/`us`/`ns` and rejects unknown units; it retains the original
numeric value and unit while exposing the timezone-aware UTC instant and
exact integer nanoseconds.

Record completion normalization is catalog-aware and uses this priority:
decoded `time.completed`, decoded `time.end`, decoded `state.time.end`, then
the observed record `time_updated` column when that catalogued column exists.
Malformed candidates emit `malformed_source_value` and do not suppress a
valid lower-priority candidate.

## JSON locations

Legacy: `message.data`, `part.data`, `session.model`, `session.metadata`,
`session.summary_diffs`.

Current: `session_message.data` (whole), the nested
`session_message.data.content` items, the nested
`session_message.data.model` field, `session_v2.model`,
`session_v2.metadata`, `session_v2.summary_diffs`.

## Classification rules

A rule matches when the value at `match_column` (optionally at
`match_json_path` inside the decoded JSON) equals `match_value` (string
comparison of the decoded value; booleans compare via their lowercase JSON
spelling).

Legacy:

- `user`: `message.data.role == "user"`
- `assistant`: `message.data.role == "assistant"`
- `system`: `message.data.role == "system"`
- parts inherit their parent message's classification, except
  synthetic/ignored parts (`part.data.type` in `{"step-start",
  "step-finish"}`), which classify as `synthetic`. These marker types are
  the v1 "synthetic/ignored" set.

Current:

- `user`: `session_message.type == "user"`, excluding synthetic content
- `assistant`: `session_message.type == "assistant"`
- `system`: `session_message.type == "system"`
- `synthetic`: `session_message.data.synthetic == true`

## Session-scoped allowlists

Records from these tables are associated with a session only through the
recorded unambiguous joins; unassociated records are reported, never
assigned.

Legacy:

- `todo(session_id, content, status, priority, position, time_created,
  time_updated)` joined by `todo.session_id = session.id`.

Current:

- `session_pending(id, session_id, type, data, delivery, admitted_seq,
  time_created)` joined by `session_pending.session_id = session_v2.id`.
- `session_inbox(id, session_id, type, payload, delivery, enqueued_seq,
  time_created)` joined by `session_inbox.session_id = session_v2.id`.
- `instruction_entry(session_id, key, value, removed, time_created,
  time_updated)` joined by `instruction_entry.session_id = session_v2.id`.
- `instruction_blob(hash, value)` — carries no session column; it is
  session-scoped only transitively, through
  `instruction_entry.value = instruction_blob.hash`.
- `instruction_state(session_id, epoch_start, through_seq, initial_values,
  current_values)` joined by `instruction_state.session_id =
  session_v2.id`. `epoch_start`/`through_seq` are sequence-domain values
  (v1 assumption: they bound the message-seq range of an instruction
  epoch), not timestamps; `initial_values`/`current_values` are JSON.

## Fixed tie ranks

| Rank | Role |
| --- | --- |
| 10 | authoritative message/session-message container |
| 20 | persisted or nested content part |
| 30 | lifecycle/session event |
| 40 | todo/instruction record |
| 90 | unknown permitted record |

The rank table is catalog data (identical in both entries), never an
incidental database query order.

## Raw-field handling

Both generations retain raw JSON/text and raw column values verbatim:

- an additive column, unknown JSON key, or unrecognized record type warns
  `unknown_source_field` while the value is retained;
- malformed JSON decodes to `None` with the `malformed_json` warning and
  the raw text retained;
- JSON objects with duplicate keys warn `duplicate_json_key` and are
  retained through canonical-json-v1's typed ordered-pairs wrapper
  (`DuplicateKeyObject`, which also preserves the original text);
- unsupported scalar decoding (e.g. invalid-UTF-8 BLOBs) warns
  `malformed_source_value` while the raw bytes are retained.

Warning messages never include the field value itself.

Legacy part attachment is additionally scoped by matching `part.session_id`
to its message's session. A NULL `message_id` attaches only when that session
has exactly one missing-ID parent. With multiple such parents, the importer
preserves the parts once as unassigned records before the session's prompts
and reports `ambiguous_missing_message_id`; it does not invent parentage or
discard their artifact content. Shared current instruction blobs retain the
same source hash in a separate projection for each referencing session.

## Secret exclusions (binding)

Every catalog entry explicitly excludes all values from:

- `account`
- `account_state`
- `control_account`
- `credential`

and omits **every** `session_share` column until a reviewed safe-column
catalog entry exists (reason code `no_safe_column_catalog_entry`; an
unknown layout is omitted with the `session_share_omitted` warning).
Exporter SQL never selects those values, so prohibited secret pages never
enter a normalized snapshot, and their presence cannot affect selection
or turn counts.
The same pre-projection policy excludes permitted-table columns named
`share_url`, `share_token`, `share_secret`, `access_token`, `access_key`,
`session_token`, or `session_access_token`, including when they arrive as
additive columns.

## Source database fingerprint

The non-secret source fingerprint is a SHA-256 over the schema generation
plus available filesystem identity/statistics (file type, device, inode,
size, mtime in nanoseconds). It excludes the absolute path and never reads
source content; it is metadata only, not reconciliation identity.

## Normalized snapshot storage

`create_normalized_snapshot` writes only two tables into a mode-`0o600`
temporary file inside a mode-`0o700` directory:

```text
snapshot_sessions(session_id, schema, canonical_session_row)
snapshot_records(session_id, order_key, source_table, source_id,
                 canonical_record, raw_permitted_fields)
```

Each canonical row column holds a deterministic stdlib-JSON serialization
of the `CanonicalRecordEnvelope` (decoded, raw_text, raw_columns,
warnings) plus a reserved `session`/`record` section carrying the typed
header/record fields needed for lossless replay; SQLite BLOBs ride a
namespaced base64 tag. Rows replay in written (row) order; the replayed
raw column map is the union of the envelope's canonical raw columns and
the exporter's permitted raw fields. This storage format is internal to
the source layer; the `canonical-json-v1` hashing format does not
bind it.
