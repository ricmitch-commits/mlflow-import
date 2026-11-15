# Qualified temporary resource profile

`run_import()` establishes one shared temporary-storage ledger. The CLI starts
that scope before source-format validation and keeps it through output rendering.
Nested source, target and reporting operations share the existing ledger; nesting
does not reset its allowance. No additional package dependency is required.

## Admission and ownership

- Default allowance: **16 GiB**, configurable with the positive integer environment
  variable `MLFLOW_IMPORT_TEMP_BYTES`. An explicit
  `resources.temporary_storage(max_bytes, directory=...)` scope is also available.
- The ledger charges **rounded logical file capacity** (4 KiB pages), a 4 KiB
  metadata allowance per file/directory, and the high-water allocation of SQLite
  files, including free pages. This is an importer admission quota, not an OS
  filesystem partition quota. Concurrent external disk use can still cause
  ENOSPC; that failure is sanitized and terminal too.
- Coverage includes selected source payloads, identity/relationship/replay/turn
  indexes, deferred summaries and selection catalogs, executable plans, target
  candidate/pricing indexes, result collections, canonical serialization,
  upload staging copies, and downloaded state/manifest/map/session objects.
- Original inputs, the original database's pre-existing WAL companions,
  persistent target artifacts/backend storage and persistent advisory-lock
  inodes are not temporary copies and are not charged. The importer does not
  back up an original database or modify its journal mode.
- Writers reserve growth **before** the write. SQLite receives a remaining-pool
  `max_page_count` before each statement; `executemany` admits one bounded row
  at a time. Committed and failed-statement physical high-water growth is
  reconciled with O(1) page/file metadata checks.
- Closing a file is not deletion. Open readers and read-only SQLite attachments
  pin their backing files. Deletion/credit release is deferred until the final
  descriptor closes. Source ID selection aliases identify the actual private
  index, rather than a reusable Python object ID.
- A write/allocation failure latches the invocation as failed. No subsequent
  source/plan/result writes or target mutations may continue against a possibly
  incomplete journal-free scratch database. Cleanup remains available.

Accounting updates a bounded in-memory ledger and parent indexes. It does not
walk a growing directory tree per record. A final cleanup walk removes the owned
workspace; removing a subtree visits only that subtree's recorded entries.

## SQLite and metadata envelopes

| Resource | Qualified envelope |
| --- | --- |
| SQLite main database pages | 4 KiB; each store's existing cap **and** the remaining shared quota apply |
| Private WAL/SHM/rollback journals | Disabled (`journal_mode=OFF`); a failed scratch snapshot is discarded |
| SQL transient sorting/working tables | `temp_store=MEMORY`; no unaccounted sorter spill files |
| Python `sqlite3` library heap | At most **64 MiB** across its connections, enforced with public `PRAGMA hard_heap_limit`; an existing lower ceiling is respected |
| Page cache / mmap / statement cache | At most 2 MiB per connection / mmap disabled / 64 cached statements |
| Connections / attachments | 64 accounted connections; at most 10 read-only attachments per connection |
| Accounted I/O handles | 128; cursors and retained reader descriptors are closed explicitly |
| Active filesystem ledger entries / private nesting | 8,192 entries / 16 levels below an owned root |
| Source filesystem traversal | Disk-ordered directory metadata, at most 32 directory levels below the selected source root |
| Standalone config validation without a ledger | Header-only inference batch capped at 8,192 entries per directory; it creates no unowned temporary spool |
| Snapshot writer batches | At most 256 rows or 4 MiB of buffered metadata, plus the current record |
| `fetchall` / `fetchmany` metadata batches | At most 1,024 rows and 4 MiB, checked incrementally |
| Merged source session/header metadata | 8 MiB; oversized cumulative control/sidecar metadata fails explicitly |
| State / manifest / turn-map wire metadata | 8 MiB, streamed before publication; state reference admission also caps at 65,536 |
| Verified shared-session control values | 8 MiB per session/turn-summary/scalar value, enforced during capture before JSON decoding; known top-level keys capped at 1,024 bytes |
| File transfer batches | 64 KiB, with streamed growth checks |

SQLite's public hard-heap pragma can lower but cannot restore/raise a ceiling.
The 64 MiB ceiling is therefore an explicit **process profile setting** and
remains after an API call returns. Scoped private file creation restores the
previous umask before returning or invoking target SDK operations. New temporary
files are atomically exclusive/0600; workspace directories are 0700.

Original WAL-mode inputs require existing `-wal` and `-shm` files for the
qualified read-only open. Missing companions produce `source_wal_unavailable`;
the importer does not create them or use an `immutable=1` bypass. A live writer's
external WAL growth is not importer-owned temporary storage.

The record/one-turn API envelopes documented in `source-adapters.md` and
`projection-parity.md` still apply. These limits do not claim support for
arbitrarily large records, native trace payloads, metadata documents or directory
inventories. Exceeding an envelope is a failure, never silent truncation.

Shared-session verification skips transcript values with the streaming lexer;
the complete artifact-size allowance is not a metadata-capture allowance.
Current and legacy ordered source queries, including mixed delegation, sort structural rowid or declared-primary-key
locators, then fetch each payload with an indexed lookup in the same read
transaction. This prevents transcript bodies from occupying a whole-selection
in-memory SQL sorter; the SQLite heap ceiling still applies to structural sorting
and all other connection allocations.

## Qualified artifact transport

The tested target uses a same-host public **`file://` artifact URI**. SDK download
helpers copy complete files before returning and cannot implement this quota.
The gateway consequently reads that public URI with a capped, streamed file
transport: stat/admit the known copy, check every chunk and validate exact bytes.
Artifact listings stream directory entries and use disk-backed ordering when
required; exact existence lookup does not scan the accumulated parent directory.
No target backend database is accessed.

The pinned **MLflow 3.17.2 HTTP(S) artifact proxy** is also qualified. The public
`mlflow.artifacts.get_artifact_repository` factory resolves `mlflow-artifacts:/`
locations; the bounded reader uses its exported `artifact_uri`, including a
deployment prefix, rooted at `/api/2.0/mlflow-artifacts/artifacts`. Direct HTTP(S)
URIs for that same service are supported. Arbitrary HTTP file servers, cloud
repositories, authority-bearing `mlflow-artifacts://` locations, redirects,
presigned and multipart download channels are outside this profile.

| HTTP resource | Envelope |
| --- | --- |
| Listing response | 4 MiB before JSON parsing; 8,192 entries total per enumeration |
| Continuation | At most 128 pages, 4,096-character opaque tokens; repeated tokens fail |
| Download | Caller-declared size ceiling, checked against Content-Length and every 64 KiB chunk |
| Temporary allocation | Known length admitted before reading; unknown length admitted before each chunk write; same shared ledger |
| HTTP I/O | One synchronous response at a time; 5-second socket timeout clamped to remaining invocation deadline at request start |
| Work/retry | Shared invocation operation/deadline checks on every chunk and request; at most four attempts per read, drawing from shared retries |

The pinned proxy returns an unpaginated listing. A wide native listing above the
envelope fails; it is never silently truncated. Bounded continuation handling
does not claim that the native endpoint implements pagination. Relative paths,
entry names and list structure are validated before use. HTTP compression and
redirects fail closed. Checksums and exact sizes remain independently verified by
the existing state/map/session consumers. Partial downloads are removed before
retry and on every failure; retries restart the same immutable key. Socket
timeouts bound a blocked read; the shared deadline is checked cooperatively
between chunks, so one in-flight read can last up to its socket timeout.

Uploads retain the pinned public `repository.log_artifact` binding. An ambiguous
immutable-object upload succeeds only if bounded exact-size/checksum readback of
the same key succeeds; no new generation is created blindly. SDK upload internals
retain their pinned timeout/retry behavior; this reader does not claim to impose
a hard cancellation deadline on the SDK's synchronous upload call.

Other artifact transports are unqualified for budgeted imports and fail safely
with `artifact_resource_profile_unsupported`. Low-level non-HTTP helpers outside
an invocation retain their old SDK seam for existing tests/callers; callers that
need the guarantee must use the resource scope. Injected factories must likewise
use the accounted helpers/connections. Reused gateway verification caches are
rebound to the current invocation before its capability probe.

Before mutation, experiment resolution validates the scheme and resolved proxy
binding and consumes a bounded read-only proxy listing. New experiments cannot
inherit an unknown current server default: the public Default experiment's
qualified location is used as a hint, and creation explicitly pins a unique
`<hint>/mlflow-import-experiments/<uuid>` location. If that hint is unavailable or
unqualified, creation fails before any experiment write. Dry-run performs the
same checks and represents the planned creation without writing. File transport
and the existing same-host benchmark recipe remain unchanged.

## Failure, output and lifetime

Known copies and serialization are admitted before publication where possible.
A later failure preserves prior verified traces through the existing all-turn
ready gate. Already published immutable staging objects or tagged new traces may
remain as recoverable remote work; they are not local temporary files.

`temporary_storage_exhausted` reports allowance/SQLite workspace exhaustion;
`temporary_storage_unavailable` covers sanitized private I/O failures. Before
target work the invocation exits 2; operational failures after entering target
mutation exit 1. On a full result database, a fixed-size in-memory emergency
envelope has `summary.incomplete=true`, `resource_failures`, and, after target
work, `recovery_pending=true`. Its zero item counters are **not** an assertion of
exact completed per-turn accounting. Retry/recovery must finish before success.
No raw exception frames or source values are retained by that emergency error.

Both CLI formats iterate their output. Compatibility `render_json()` and
`render_human()` deliberately materialize strings for callers requesting them.
The API's lazy result owns its remaining plan/report store until
`ImportResult.close()`; an explicit outer resource context owns everything until
that context exits. The CLI closes the result after rendering. Normal completion,
quota/I/O failure, dry-run and cooperative interruption close file readers,
writers, raw SQL cursors, connections and attachments before unlinking. Cancellation
does not start more diagnostic downloads. Hard process termination cannot execute
Python cleanup; an OS denial of cleanup is reported with the safe
`temporary_storage_cleanup_failed` code rather than claimed successful.

## Verification evidence

`tests/unit/test_temporary_resources.py` covers aggregate SQLite/file exhaustion,
terminal/nested allowance behavior, before-copy admission, read/attachment pinning,
cursor finalization, original WAL immutability, private SQL configuration, source
metadata envelopes, creation permissions/umask restoration, extended I/O failures,
buffer byte counts, dry-runs and pipeline/output interruption cleanup.

`tests/integration/test_resource_quota.py` uses the current **public historical
TraceInfo/Span binding** and disposable MLflow target. It tests phase-two state
serialization and ready-download exhaustion under a 1 MiB quota, preservation of
prior traces, clean local teardown and successful recovery. Its 32/64-turn Pi
persisted-tree and Gemini patch/rewind cases enforce a 32 MiB temporary quota,
less than 32 MiB traced Python heap, bounded metadata-frame growth, linear emitted
full-session rows and near-linear SQL statement counts. They are production-path
tests, not detached-span fakes. Exact final results are in `errors.md`.

An earlier 319,987,518-byte fresh phase failed before selection because payloads
entered a whole-selection SQL sorter. The corrected current path now passes the
strengthened reference gate in **378.762984708 / 308.581364792 / 311.297854416 /
311.746874 seconds**, with 307 sessions and exact 494/494/495/495 target trace
cardinality. Unchanged mutation calls are zero. See
[compatibility.md](compatibility.md#synthetic-performance-evidence).

Legacy/mixed ordering was independently reproduced and corrected with the same
locator-only helper. **48 synthetic 40 × 2 MiB cases** cover headers, messages,
parts, ambiguous NULL-parent parts, todos and minimal todos across rowid and
WITHOUT ROWID layouts; a separate credential-key regression ensures excluded
columns stay unread. All pass under the unchanged 64 MiB SQLite ceiling and
16 MiB temporary quota. These are adapter-scale ordering tests, not a new
307-session legacy performance certification.

Main-agent post-W1/documentation final selections passed **1663 unit/contract/BDD**
and **130 non-slow integration/profile** cases, including the production resource
cases. The separate three-case memory suite passed in 310.73s. Static checks,
wheel/sdist builds and independent qualified-profile acceptance passed; earlier
timing tables are superseded.

## Old unbound in-flight journals

An old checksum-valid journal without the required namespace/session/attempt/
instance/chain evidence is an **unqualified format**, not authority to delete.
`attempt_state_invalid` leaves its existing traces and artifacts intact. Raising
the temporary quota does not change that qualification decision.

For operator recovery, preserve the old experiment and journal objects. The
non-destructive path is to import the authorized source into a newly named
experiment and verify that completed copy, retaining the old experiment for
audit. Do not add invented binding fields/checksums to the old ready state or
run an unsafe automatic-delete implementation against it. Recovery in the same
experiment needs a separately reviewed operator procedure which reconstructs
historical references, verifies the complete replacement set, and independently
requalifies each exact retirement against the intended source scope. This
importer deliberately provides no automatic legacy-journal trust upgrade.
