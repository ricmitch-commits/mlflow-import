# Security and privacy

This document describes the importer's privacy model as shipped: what is
never read from the source, what `--redact` masks, how logs and reports
stay safe, and how temporary data is protected. The authoritative
implementations are `src/opencode_mlflow/source/catalog.py` (exclusions),
`src/opencode_mlflow/content/redaction.py` (detectors), and
`src/opencode_mlflow/source/snapshot.py` (snapshot permissions).

## Threat model, in one paragraph

The source database contains your complete coding transcripts — prompts,
responses, tool output, and source code — and the secret tables contain
credentials. The importer treats both as sensitive: it reads only
permitted tables and fields (including observed additive columns for fidelity); it never reads the credential-bearing
tables at all; every output surface (report, logs, errors, lock files)
carries identifiers, counts, codes, hashes, and timestamps only; and the
only places transcript content ever goes are (a) a restrictive temporary
snapshot and staging files on the importing machine, and (b) the MLflow
tracking target you point it at — verbatim by default, or masked with
`--redact`.

## Excluded tables and fields

Every catalog entry (`source/catalog.py`) explicitly excludes **all
values** from these tables:

- `account`
- `account_state`
- `control_account`
- `credential`

Exporter SQL never selects those values, so prohibited secret pages never
enter a normalized snapshot, and their presence cannot affect selection
or turn counts.

`session_share` is omitted entirely — every column — until a reviewed
safe-column catalog entry exists (reason code
`no_safe_column_catalog_entry`; a `session_share` table with an unknown
layout is additionally reported through the `session_share_omitted`
warning). Within permitted tables, explicitly excluded fields (`share_url`,
`share_token`, `share_secret`, `access_token`, `access_key`, `session_token`,
and `session_access_token`) are never selected. This exclusion is applied
before the additive-column projection, so an additive column cannot bypass
the policy. Observed additive **unknown columns are
selected and retained** in raw column maps for fidelity, with an
`unknown_source_field` warning when populated. They may contain sensitive
values: verbatim mode retains them; `--redact` applies the same deterministic
policy as for known content. Unknown columns are not equivalent to excluded
credential tables or explicitly excluded fields.

The source layer additionally preserves raw values verbatim where JSON
cannot be decoded: malformed JSON keeps its raw text, duplicate-key
objects keep their original text, and opaque BLOBs keep their bytes —
see "Verbatim mode sensitivity" for what that means.

## The source database is never modified

Guarantees enforced in one place, `source/connection.py` and
`source/snapshot.py`:

- The database is opened through a URI containing only the normalized
  path and `mode=ro`; `immutable=1` is never used (it can ignore WAL
  contents), and the `-wal`/`-shm` companions are left in place so
  committed WAL rows stay readable.
- `PRAGMA query_only=ON` rejects every write, schema change, checkpoint,
  vacuum, and attach through the connection.
- Exactly one deferred `BEGIN` is issued — never `BEGIN IMMEDIATE` or
  `BEGIN EXCLUSIVE` — so the read transaction shares one consistent view
  with the extraction that follows.
- The live connection is closed immediately after extraction; no live
  connection remains during discovery or any MLflow work.
- The importer never writes, checkpoints, vacuums, migrates, or deletes
  the source database. (Opening a WAL database read-only legitimately
  mutates the `-shm` file's bytes through SQLite's own read-mark
  protocol; the integration suite therefore asserts metadata-level
  invariants, and the database and WAL files themselves are asserted
  unchanged.)

A path that is missing, not a regular file, not readable, or not a
SQLite database fails safely (`source_unavailable`, exit 2) with a
message that never echoes the path.

## Verbatim mode sensitivity (default)

By default (`--redact` omitted) the importer is **verbatim**: permitted
transcript values are persisted, hashed, and uploaded as they appear in
the source. This is the fidelity-preserving mode and the right choice
when the tracking target is at least as protected as the source database.

Be explicit about what that means: in verbatim mode, the MLflow
experiment you import into will contain your prompts, model responses,
tool output, and source code — in span payloads and in the canonical
artifact documents. Anyone who can read that experiment can read the
transcripts. Choose the target accordingly, and prefer `--redact` when
the destination has a wider audience than the source machine.

Redaction mode also affects trace identity: the render fingerprint
carries the redaction-mode tag, so switching a session between verbatim
and redacted replaces its traces (the previous render is not silently
reused).

## `--redact`

With `--redact`, the deterministic v1 redactor runs **before temporary
persistence** — the snapshot itself, the artifacts, the span payloads,
and every downstream hash see only masked values. Replacement markers
are fixed: `[REDACTED]` for text values and `[REDACTED_BINARY]` for
opaque BLOB/invalid-UTF-8 bytes. Bearer/Basic scheme prefixes are
preserved where possible — `Authorization: Bearer [REDACTED]` keeps the
header name, separator, and scheme.

Typed session fields (including title, model, metadata, and summaries) use
the same policy before persistence and trace mapping. Valid JSON raw text
and raw JSON columns are sanitized structurally and reserialized when
changed, so escaped secret-key spellings and quoted Basic credentials do
not bypass the structured policy. Malformed/duplicate text is scanned in
place; the malformed quoted-Basic gap below remains intentionally limited
to text that cannot be decoded faithfully.

The result document reports redaction as **counts only**:
`redaction.mode`, `detector_counts` (per detector category), and
affected field/record counts. No matched secret and no original text
ever reaches the report, logs, or errors.

## Detector categories and pattern semantics

Version one recognizes these categories (the only keys
`redaction.detector_counts` may carry, plus the type-based
`opaque_binary`):

| Category | What it matches |
| --- | --- |
| `pem_private_key` | A `-----BEGIN ... PRIVATE KEY-----` … `-----END ... PRIVATE KEY-----` block. An unterminated `BEGIN` marker (a truncated block) masks through the end of the text, because the remainder is private-key fragment material. |
| `auth_header` | Case-insensitive `Authorization`/`Proxy-Authorization` header values with bearer/basic schemes, including standalone `Bearer <token>` forms. |
| `structured_auth_field` | Structured `Authorization`/`Proxy-Authorization` mapping keys whose scalar value begins with a bearer/basic scheme. |
| `secret_key_value` | Values assigned to case-insensitive keys spelled `api[-_]key`, `api[-_]token`, `access[-_]token`, `refresh[-_]token`, `auth[-_]token`, `session[-_]token`, `token`, `access[-_]key`, `client[-_]secret`, `secret`, `credential`, `password`, `passwd`, `private[-_]key`, or `bearer` (the separator is optional, so concatenated spellings like `apikey` also match). Textual assignments (`key: value`, `key=value`) use the same spellings with delimiter-aware value bounds: after the separator, a quoted value ends at its matching unescaped quote and an unquoted value ends at whitespace, comma, semicolon, `}`, or `]`. |
| `connection_secret` | The password between `scheme://user:` and `@` in a URI or connection string, including empty-user forms (`redis://:password@host`) — the user part is optional, the password is not. |
| `known_token_prefix` | Recognizable token formats with their documented minimum lengths: `sk-` (16), `ghp_` (20), `github_pat_` (20), `xox[baprs]-` (10), and AWS `AKIA` (exactly 16). |
| `opaque_binary` | Bytes and other opaque non-scalar values replaced by `[REDACTED_BINARY]` because they cannot be inspected as text. This is a type-based guard, not a content match; unknown types **fail closed** — a value the pipeline cannot inspect is masked, never passed through. |

Pipeline order matters and is fixed: structured mappings are redacted
recursively before any textual matching of leaf values; within a
structured field the precedence is protected path → structured auth
field → secret key (the whole assigned value is masked, so
`{"token": "..."}` counts as `secret_key_value`) → recursion. Text
scanning then applies, in order: textual assignments, auth headers, PEM
blocks (complete blocks first, then unterminated markers), standalone
bearer values, connection-string passwords, and known token prefixes.
Earlier stages mask first, so a value assigned to a secret key never
also counts as a token prefix, and a PEM block replaces its whole body
before bearer/token patterns could match inside it.

JSON decoding is lossless for duplicate object keys. Every occurrence is
redacted independently, including keys whose source spelling used a
Unicode escape, and redacted raw JSON is regenerated from those pairs.
The original secret-bearing raw text is never retained in a redacted
snapshot or artifact.

### Known v1 gaps (accepted scope, documented not hidden)

- Quoted `Basic` credentials inside malformed JSON text are not matched
  (the literal auth-header form cannot cross quotes; over-matching in
  quoted contexts was ruled out). Quoted `Bearer` values are still
  caught by the standalone-bearer detector.
- Connection strings without a password (`user:@host`) are not masked —
  the password requires at least one character.
- `Digest` and `negotiate` authentication schemes are outside the
  bearer/basic contract.

## Protected identifiers

Redaction must not break selection, ordering, turn reconstruction, or
reconciliation, so identity and sequence fields are protected from
masking. The exact protected set — defined once as
`PROTECTED_IDENTITY_FIELDS` in `contracts.py` and imported by source and
content layers rather than duplicated — is:

`id`, `session_id`, `project_id`, `workspace_id`, `message_id`, `parent_id`, `fork_session_id`,
`source_id`, `source_table`, `source_type`, `parent_source_table`, `seq`, `admitted_seq`,
`enqueued_seq`, `epoch_start`, `through_seq`, `hash`, `key`, `position`,
`type`, `role`, `turn_key`, `schema`, `catalog_version`

— that is, the source/session/message/part ids, the sequence values,
turn keys, schema/catalog version names, classification fields, the catalogued
`hash`/`key`/`position` fields, and the reconciliation identifiers. Typed
fields retain those identities for selection, ordering, and replay. Raw
column maps and permitted raw-row fields are content/provenance
projections: additive columns named `key`, `hash`, `type`, `role`, or
`position` receive no blanket protection and are redacted normally.

Protection is root/path exact at structural envelope boundaries. Typed
`SessionHeader`/`SourceRecord` identities stay unchanged, while an
arbitrary transcript mapping named `id`, `key`, `hash`, `type`, or `role`
does not protect that mapping's subtree.

For raw text the pipeline re-serializes the already-redacted decoded
value when the JSON was valid (so the original raw text is never
retained once anything was detected), regenerates duplicate-key JSON from
its lossless redacted ordered pairs, and masks malformed text in place.

## Safe logs, reports, and errors

- The result document (both output modes) carries only identifiers,
  counts, warning/error codes, hashes, timestamps, and the experiment
  name — never prompts, responses, tool output, source code, absolute
  source paths, credentials, tracking-URI secrets, or matched secrets.
- Every error is a typed `ImporterError` with a stable machine-readable
  code and a `safe_message` sanitized for logs; `str()` of any error
  yields only the safe message, so error text can be logged directly.
  Safe messages never echo offending values (a rejected timestamp or
  model identity is described, not repeated).
- The discovery plan file carries identifiers, counts, warning codes,
  timestamps, model identities, and hashes only.
- The advisory lock file carries only credential-free metadata: the
  **normalized** tracking endpoint (userinfo, query, and fragments
  stripped; scheme/host canonicalized), the experiment name, and the
  acquiring pid. The lock *name* is a SHA-256 over the same normalized
  inputs, so it never contains or depends on credentials either.

## MLflow authentication and TLS

The importer accepts no credentials of its own. `--tracking-uri` names
the target; authentication, authorization, and TLS are whatever your
standard MLflow client configuration provides — e.g. `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_TOKEN`, `MLFLOW_TRACKING_USERNAME`/
`MLFLOW_TRACKING_PASSWORD`, `MLFLOW_TRACKING_INSECURE_TLS`,
`MLFLOW_TRACKING_SERVER_CERT_PATH`, and
`MLFLOW_TRACKING_CLIENT_CERT_PATH`, as supported by the pinned
`mlflow==3.17.2` client/requests environment. The read-only capability
attestation uses the same settings for `GET <tracking-base>/version`, with a
 bounded timeout and no redirect forwarding of authorization. The request URL
 keeps only the safe static path prefix: URI userinfo, query parameters, and
 fragments are never forwarded to `/version`. Configure them in
the invoking environment; the importer never logs credentials, URI userinfo,
certificate paths, response bodies, or transport exception details and never
writes them anywhere (the lock normalization exists precisely so a
credential-bearing URI cannot leak into a lock file).

Use HTTPS for any non-local tracking server. The capability probe treats an
unreachable, unsupported-scheme, malformed, mismatched-version, or
protocol-violating target as an incompatible pair (exit 2) rather than
falling back to an insecure path. The `/version` request is read-only and is
completed before experiment, trace, artifact, or retirement mutation.

## Temporary files: permissions and cleanup

- The run root is a `tempfile.mkdtemp` directory (prefix
  `opencode-mlflow-run-`), removed with checked cleanup in an outer
  `finally` — on success and on every failure path. A cleanup failure emits
  only the generic `temporary_run_cleanup_failed` warning and contributes to
  `failed_items`; it is not reported as a failed source turn and never
  exposes the temporary path.
- The normalized snapshot lives in a mode-`0o700` subdirectory with its
  file mode `0o600`.
- The discovery plan file is created mode `0o600`.
- Artifact staging files are created atomically with mode `0o600`
  (`tempfile.mkstemp` opens through `os.open` with `O_CREAT | O_EXCL`
  and exactly that mode — there is no create-then-chmod window) and are
  deleted best-effort after the upload/verification block.
- The advisory lock root honors `XDG_RUNTIME_DIR` when set (a per-user
  `0o700` directory per the XDG spec); otherwise a restrictive `0o700`
  subdirectory of the system tempdir. The lock file itself is mode
  `0o600`.

Nothing from the excluded secret tables ever reaches these files, so
there is no prohibited secret value to clean up even on a crash; a
crash only leaves an ordinary restrictive temp directory for the OS/job
temp-cleaner to reap (the run root path is not printed to avoid leaking
paths into logs).

## What the importer does NOT do

- It never reads the `account`, `account_state`, `control_account`, or
  `credential` tables, and never reads `session_share` columns.
- It never writes to, checkpoints, vacuums, migrates, or deletes the
  source database.
- It never accepts or logs credentials; it never puts payloads in trace
  tags or metadata (the pinned server silently truncates those at
  4096/250 characters, so payloads ride artifacts and span inputs/
  outputs instead).
- It never uses private/undocumented MLflow APIs, MLflow runs, or
  direct backend-store writes.
