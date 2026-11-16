# Final architecture-alignment errors and verification report

**Evidence date:** 2026-11-14. Implementation and corrective review work is present
in the uncommitted working tree based on
`dad31e1c0c759408a2f163a622d1b5d49b0773df`. The authoritative
[architecture](architecture.md), [requirements](requirements.md) and
[integration baseline](integration-parity.md) are preserved. This report replaces
the incremental handoff journal: historical failures are resolved as described
below, not current implementation blockers.

The approved implementation is complete for the documented source versions and
MLflow profile. The final acceptance audit's **W1 global tie ordering and
documentation drift** findings are closed. Post-correction full-suite, static,
packaging and independent acceptance checks passed; no actionable Blocker or
Warning findings remain from the completed reviews.

The prior completion evidence covered same-host file artifacts. The subsequently
reported real-command transport gap and its scoped correction are recorded
[below](#bounded-remote-artifact-correction); the earlier timings do not certify
remote proxy performance. The later [source-discovery correction](#source-discovery-correction)
records the corrected discovery behavior. The live import and persisted-trace
verification below close the reported no-trace failure.

## Historical JSON-looking string verification correction

MLflow **3.17.2** sanitizes historical OTLP span attributes by removing a JSON
encoding layer when a string parses as another string, object or array. This
reinterpreted verbatim JSON-looking tool results and caused strict
`trace_verification_failed` input/output or attribute comparisons.

The historical gateway now adds exactly one transport encoding layer for those
top-level strings, uniformly across root/child inputs, outputs and attributes.
Structured values, nested strings and numeric/boolean/null strings retain their
ordinary encoding. Source normalization, digests and exact verification are
unchanged. Synthetic pinned-localhost regressions cover Unicode/quoted/multilayer
strings, scalar/structure controls, genuinely changed payloads, native analytics,
fractional timing, and session import/unchanged reuse with artifact identity.

After updating the checkout, rerun the original import command against the same
source and experiment. Normal reconciliation verifies existing work and retries
failed work; do not edit source strings or bypass verification. If verification
still fails, retain the safe error code/message for diagnosis: other payload
differences remain errors.

## Prepared-session recovery correction

A subsequent SessionsAI retry reported `session_recovery_failed` and
`import_failed`. Read-only diagnostics observed intermittent SDK-wrapped
disconnects and timeouts on target reads that previously bypassed bounded
retries. Those reads now share the invocation's retry allowance and deadline;
exhausted reads retain a safe typed error instead of becoming an artifact
integrity nonmatch. The original sanitized output does not identify which read
failed in that invocation.

The unfinished attempt also contained a verified staged replacement and an
incomplete staged duplicate outside its frozen original-prior list. Recovery
previously included that duplicate in the original-prior retirement batch,
which journal validation correctly rejected. Recovery now validates the entire
replacement scope, re-verifies its payloads and artifacts, and separately
cleans exact same-attempt staged duplicates. Original-prior retirement still
requires the verified durable ready state. Abandoned-attempt cleanup uses the
same exact staging guards, including a final check of both tags and metadata.

Disposable-localhost tests cover prepared recovery, fresh attempts, interrupted
cleanup, changed source/policy, identity races, combined transient reads, and
unchanged reruns. A local replica of the observed unfinished attempt recovered
and completed an unchanged rerun. The normal SessionsAI import has not been
rerun against the user's tracking server as part of this correction. Rerun the
same command from this checkout; existing journals are consumed by recovery.
The `session_share_omitted` schema warning is unrelated to these failures.

## Invocation deadline and live-server diagnosis

The reported bare `retry_budget_exhausted` CLI error could also occur while
recording or rendering an earlier failure: result storage shared the expired
work deadline. Terminal reporting now has a separate bounded local allowance,
preserves the first safe error, and emits a qualified incomplete result with
unknown counts when complete counts cannot be recovered. Deadline and retry
allowance exhaustion have distinct safe messages. `--timeout-seconds` provides
an explicit finite-positive invocation deadline (default 600 seconds), including
for direct Python configurations.

Read-only profiling selected 317 SessionsAI sessions / 516 turns. The initial
full preview took 289.67 seconds, including 208.18 seconds preparing pricing
donors. Donor selection now verifies each donor with one full-trace read and
one checksummed map read instead of repeated verification. A subsequent full
read-only preview completed in 151 seconds with no retries consumed. These are
separate live observations, not a controlled network benchmark.

One normal live invocation with an explicit 1800-second deadline still failed
after 94.5 seconds during read-only planning with `read_retry_exhausted` and
capability failures. It reported no imports, updates or retirements. Subsequent
capability checks passed. Read-only OpenShift inspection then found the MLflow
container had 61 restarts and a latest `OOMKilled` termination (exit 137), with
four Uvicorn workers under a 4 GiB memory limit. This establishes server memory
instability; the exact failed request in that invocation was not retained.
No deployment configuration was changed as part of this diagnosis.

Independent paged lookup found 306 active traces in SessionsAI (experiment 2),
with historical request times from 2026-10-20 through 2026-11-04 UTC. A recent
UI date filter can hide them. There were 41 completed and two unfinished durable
attempts; full live recovery remains unverified until server stability is
addressed. These observations do not supersede the historical `test-import`
success described below.

## Persistent 8 GiB deployment and aggregate-trace blocker

With user approval, `openshift-local` commit
`9649417bb9eeea91c155ccecb00134856356150a` changes only the Git-managed MLflow
Deployment memory limit from 4 GiB to 8 GiB. Argo CD synchronized this revision;
the ready pod retained one worker. A direct Deployment patch had previously
been reverted by Argo CD self-healing.

The first monitored 8 GiB recovery reached its 1800-second deadline during
reconciliation, with 54 retries remaining and no pod restarts. Independent
inventory then verified 416 completed active winner traces and nine unfinished
attempts. Subsequent local corrections cover explicit empty native previews,
per-create immutable turn-map placement, and an 8 MiB per-value profile. These
passed 2166 unit/contract tests and the associated localhost integration cases.

A subsequent 3600-second recovery stopped after 1892.6 seconds with
`read_retry_exhausted`: 3 imported, 30 updated, 286 unchanged and 22 failed turns
were reported for the processed prefix. The MLflow pod was OOM-killed once at
8 GiB. PostgreSQL was also OOM-killed under its 1 GiB limit; database failures
and HTTP 503 responses occurred before the final failed lookup. Access logs
do not identify the exact in-flight request responsible for the memory peak.

Local reconstruction of the exact failed turn keys found one turn with 2105
child spans and 1,329,920,980 canonical child-envelope bytes, mostly accumulated
conversation history repeated in model-call inputs. Every individual value
fits within 8 MiB. Thus the per-value qualification does not qualify aggregate
trace ingestion, retrieval or database memory consumption. This measurement is
a fresh-source canonical reconstruction, not a captured OTLP wire length.

Final read-only inventory verified 436 active completed winner traces, 37 staged
traces, five legacy traces and seven unfinished attempts. Historical completed
journals also reference retired predecessor traces; their total reference count
must not be reported as the current active trace count. Full import completion
has not been established. Further normal retries were stopped pending aggregate
projection/transport and PostgreSQL capacity work; another memory increase alone
is not a demonstrated resolution.

## Verified resolution of the reported no-trace import

The original command stopped before trace creation for two independent reasons:
automatic discovery failed on metadata-only Claude preludes/unrelated Gemini
collisions, and the remote experiment's `mlflow-artifacts:/` repository exceeded
the original same-host-file-only qualification. These were importer support gaps,
not an absence of matching source turns.

After the source and bounded proxy corrections, the main agent executed the
requested import through the editable checkout's `.venv/bin/mlflow-import`,
using the original tracking URI, directory filter and experiment name, with
`--output json` solely to inspect sanitized results. No harness restriction or
selection change was needed.

| Verification | Observed result |
| --- | --- |
| Tracking server | `http://mlflow.apps-crc.testing/` |
| Experiment | `test-import`, ID **5** |
| Dry-run | Exit **0**, **4 sessions / 9 turns**, **9 planned imports**, **0 planned retirements** |
| Actual import | Exit **0**, **9 imported**, **0 updated/retired/failed**, **8.436 seconds** wall time including CLI startup |
| Independent paged server lookup | **9 persisted traces**: **7 Claude**, **2 OpenCode**; all **9 attempt states complete** |
| Unchanged invocation | Exit **0**, **9 unchanged**, **0 imported/updated/retired/failed**, **10.467 seconds** |

No prompts, responses, code, individual source IDs or target payloads were
printed by the diagnostic wrappers. Only counts, safe codes and experiment
metadata were emitted. The import intentionally uploaded the selected
`rich-test2` sessions using the user's original verbatim-mode command. No
SessionsAI data was used.

Main-agent final verification after these corrections:

```sh
env -u MLFLOW_TRACKING_URI MLFLOW_DISABLE_AGENT_HINT=1 PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m pytest -m 'not slow' -q --tb=short
# 1886 passed, 4 deselected in 222.79s; default import mode.
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
git diff --check
```

Lint/format passed for **172 Python files**; mypy passed for **65 source files**.
Independent corrective re-review passed **74 focused tests**, including real
loopback TLS handshakes, and reported no remaining Blocker/Warning findings.
The original slow benchmark was not repeated: these checks qualify remote
correctness and the reported live invocation, not remote reference-scale timing.

## Actual failures, root causes and corrections

| Failure / finding | Root cause | Resolution and verification |
| --- | --- | --- |
| Many-turn work repeated complete session scans and duplicated artifacts | Turn boundaries, summaries, hashing and payload graphs were replayed/materialized per turn | Disk-backed occurrence/entity/invocation/model/turn indexes, direct spool locators and frozen executable plans; one shared session object, reusable maps and checksummed manifest. Fixed-pass/indexed-replay and memory tests pass. |
| Source views could accept transient copied bytes or overselect siblings | Before/after identity alone did not bind the copied stream; closure and header discovery mixed content and relationships | Digest the actual copied stream against a second read; same-stat and transient-restoration regressions; header-only metadata discovery, direct-match ancestor/descendant closure, same-transaction SQLite selection and authoritative V2 schema precedence. |
| Missing/duplicate identities and unsupported formats could be guessed | Weak schema fixtures, overlapping detectors and generic fallbacks | Qualified five-adapter schemas and required identity checks; unsupported/malformed variants fail explicitly. Pi v3 trees, Gemini JSONL replay, sidecars and gzip have fixture/readback evidence. No invented live-only observations. |
| Partial replacement could retire prior verified work too early | Recovery was per-turn rather than gated by the complete session | Bound prepared/staging/ready/complete/aborted lifecycle, distinct manifest, full replacement verification before exact-ID retirement, changed-source handling and resumable partial deletion. Failure-injection tests preserve old traces before ready. |
| Checksum-valid forged states could authorize unrelated retirement | Integrity was mistaken for scope/chain/identity authority | Typed namespace/session/attempt/instance/path/transition checks; independently verified session/manifest and replacement inventory; independently qualified retirement proofs and immediate target revalidation. Forged empty-ready/out-of-scope and live unrelated-trace regressions pass. |
| Ignore/duplicate and zero-turn shortcuts bypassed recovery | Policy returned before discovering unfinished attempts | Selected empty sessions recover first; abort only their staged work, resume ready empty retirement and retain completed duplicate-instance semantics. Policy/boundary live cases pass. |
| Foreign tag-only session aliases were missed | Queries searched metadata only; bare aliases lacked all-root identity inventory | Search both tag and metadata filters with shared deduplication, qualified source identity and cross-root ambiguity rejection before mutation. Native/legacy × tag/metadata × policy cases pass. |
| Neutral tags or incomplete artifact references could be treated as unchanged | Mutable ordinal was confused with immutable ownership; missing V2 references bypassed checks | Managed set/delete/readback includes neutral/current tags; mutable ordinal separated from immutable creation evidence. Require the complete source/privacy-scoped shared-session reference tuple. Repair preserves trace/map IDs where applicable. |
| Unknown costs became native zeros/server estimates, or changed on retry | Moving SDK rates and incomplete scope/provenance rules | `mapping-v7` / `accounting-v2`, persisted-first Decimal pricing, explicit native-null unknown components, scoped invocation rollups, versioned coverage and frozen positive/negative pricing donors. Live native/structured/append/recovery checks pass. |
| Temporary growth and Python collections were unbounded | Independent per-file caps, retained plan/results/metadata blobs, whole-file SDK copies | Shared 16 GiB admission ledger with pinned lifetimes, disk-backed plans/results/pricing, streamed state/output/transfers, bounded SQL working memory and explicit terminal failure/cleanup. Aggregate quota and phase-two failure/recovery tests pass. |
| Actual remote dry-run stopped at `artifact_resource_profile_unsupported` after selecting 2 sessions / 2 turns / 26 records | The qualified artifact resource binding accepted only same-host file URIs; the existing remote experiment uses `mlflow-artifacts` | Public-factory-resolved, pinned HTTP proxy listing/existence/download now uses capped streamed responses, shared quota and deadline/retry checks. Local-only proxy lifecycle/recovery/quota and finite-wire tests pass; no user-server mutation or real transcript read was performed for this correction. |
| Startup ENOSPC/SQLite errors escaped the CLI | Resource initialization preceded the safe error boundary | Typed `temporary_storage_unavailable` startup handling, partial-root cleanup and safe exit 2. Real CLI failure regressions pass. Flat/explicit-import syntax now shares one normalization/parser path. |
| Shared-session verification could materialize huge control values | Lexer capture inherited the much larger complete artifact allowance | 8 MiB control-value capture and 1,024-byte known-key limits before accumulation/JSON decode; bounded known-key inventory. Oversized session/summary values reject; a 16 MiB transcript record streams past without full decoding. |
| Encoded, duplicate-key or malformed text leaked identities/secrets | Lossy object decoding, incomplete container consumption, start-of-string-only JSON detection and abandoned truncated quotes | Lossless duplicate pairs; quote-aware complete-container masking; bounded recursive JSON-fragment scanning with leading prose; ambiguous truncated/invalid JSON-bearing remainder masked through EOF, including later-quote borrowing. Adversarial mixed-depth/truncation, exact nonmatch, bounded-work and live artifact/span privacy regressions pass. Verbatim/canonical hash paths retain their tests. |
| Reports hid unchanged/recovery state or accepted malformed nested fields | Superseded count semantics and weak historical result validation | Disjoint unchanged accounting, safe item/coverage diagnostics, actual durable attempt/instance/state fields, streamed results and explicit close ownership. Typed `compatibility-v1` reads both declared v1 schemas; historical producer `36ed44f` has synthetic captures and live migration evidence. |
| Reference test could pass while omitting generated sessions/turns | Input-generation counts were mistaken for imported-workload assertions | Strengthened gate requires all 307 sessions, 494/494/495/495 turns, unique identities, zero failures, disjoint accounting and exact paged target trace-ID sets/cardinality in every phase. Results/connections close explicitly. |

## Source-discovery correction

This continuation preserves the authoritative requirements/architecture and the
already-implemented bounded remote transport. It adds no staging or commits.
The prior header-only inventory was used as evidence. No user data was copied
into repository fixtures or diagnostic files; the final production dry-run used
its normal private selected-source snapshot and cleanup. SessionsAI was not accessed.

The allowlisted metadata check confirms **158 Claude primaries**: **154** have
attributable cwd (**2** in the requested directory, **152** elsewhere), and **4**
remain unknown. Session-bearing mode metadata is followed by qualified metadata
such as permission mode, atis latch, file-history/system/queue, worktree state,
tools used, last prompt and cost state. Discovery lexically skips their
non-allowlisted values and transcript values. It continues until directory facts
are established when needed for filtering. Identity-less prefixes cannot supply
creation time. Fork-context sidecars require a later supported header matching
the owning primary; arbitrary context-reference paths are not opened, and later
explicit identities must match even when their values are empty/null/false/zero.

The **12 Gemini files** match their persisted registry group and the SHA-256 of
the registered absolute path. Registry evidence resolves missing cwd before
direct-match pruning. Collisions confined to provably excluded sessions no longer
block the relevant plan. Selected/unknown or required-relative collisions remain
`ambiguous_source`, with a harness-specific authority explanation; differing
creation timestamps, filenames or sizes do not resolve them. Plain repeated
metadata now receives the same immutable creation/project/kind checks as `$set`
metadata. Attributable unsupported headers remain in inventory so selection or
required closure cannot silently discard them. Entirely incompatible roots fail.

Review corrections also bind creation time to attributable session metadata,
make header inventory hash-only (plus technical IDs/locators, timestamps and
directory-match booleans), revalidate selected headers before normal writer
redaction, preserve explicit-file selection in the presence of an unqualified
sibling, and reject intermediate Claude sidecar-directory symlinks. Gemini
project/kind validation uses pre-redaction hashes. Codex auxiliary model/usage
observations also pass through redaction before SQLite persistence. Synthetic
byte-level regressions check both SQLite and payload spools where applicable.

### Verification and actual read-only result

- `tests/unit/source/test_metadata_discovery.py` supplies synthetic metadata,
  malformed-excluded-body, relative-closure/no-sibling, fork-context, registry,
  collision/authority, timestamp, unsupported-selected, identity, symlink and
  redaction regressions. Existing source and adapter parity checks also ran.
- Final full relevant selection:
  `env -u MLFLOW_TRACKING_URI PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest --import-mode=importlib -m 'not slow' -q`
  — **1,882 passed, 4 slow performance cases deselected, 219.56 s**.
  That historical invocation used importlib collection because of the
  unit/integration `test_remote_artifacts.py` basename collision. A plain initial
  `pytest -q` stopped at that collision. The collision is now corrected by a
  module rename, with default collection and the full non-slow suite verified
  [below](#review-corrections-explicit-ca-trust-and-default-collection). The
  virtualenv PATH is required for the package entry-point subprocess check.
- `ruff check src tests`, `ruff format --check src tests` and `mypy src` pass
  (**158** formatted files; **65** typed source files).
- Independent quality, compatibility, security and acceptance review passes were
  repeated after corrections; no relevant actionable Blocker/Warning findings
  remain. Reviewer startup initially failed with external plugins enabled; the
  independent review runs succeeded with `opencode run --pure`.
- **After code and reviews**, the production CLI ran with default source discovery,
  the requested directory, existing experiment/tracking configuration,
  `--dry-run --output json`. A wrapper retained the report only in memory and
  emitted numeric counts/error codes, never source IDs, paths or content.
  **Exit 0; no error codes; 8 sessions scanned, 4 selected, 9 selected turns,
  157 source records; 0 imported turns, 0 updated turns, 0 retired traces,
  0 failed sessions/turns/items.**

There was no remaining blocker for that read-only checkpoint. The subsequent
live import is verified above. Unresolved selected source
authority and unsupported metadata mutations remain explicit supported-boundary
failures, not ignored errors. See [source-adapters.md](source-adapters.md).

## Bounded remote artifact correction

The coordinator's read-only command used `--harness opencode`, a directory
selection, experiment `test-import`, an HTTP tracking URI, `--dry-run` and JSON
output. It selected **2 sessions / 2 turns / 26 records**, then failed with
`artifact_resource_profile_unsupported`. The coordinator had already established
that the experiment exists (ID **5**) and its artifact location uses
`mlflow-artifacts`. This correction did **not** re-run that command, read the
user's transcripts, call SessionsAI, or contact the user's MLflow server.

Research inspected the installed **3.17.2** distribution's HTTP artifact
repository, `mlflow-artifacts` URI resolver, proxy list handler, and standard
credential/TLS conventions. Context7's MLflow documentation corroborated Basic
authentication precedence, token settings and certificate configuration. The
pinned proxy returns an **unpaginated** `files` listing with names relative to
the requested directory. Its SDK listing buffers a whole response, and its SDK
downloads cannot participate in importer admission. Merely whitelisting a scheme
would therefore not fix the resource contract.

Implemented in `resources.py`, with gateway pre-write validation:

- Retain the approved public `mlflow.artifacts.get_artifact_repository` integration
  and consume its exported resolved URI. Support `mlflow-artifacts:/` and direct
  HTTP(S) proxy URIs, including deployment prefixes. No private helper imports,
  backend database calls, run containers or new dependencies.
- Read list responses in **64 KiB** chunks, cap a page at **4 MiB** before JSON
  decoding, cap enumeration at **8,192 entries / 128 pages**, reject malformed
  entries/traversal and repeated/oversized continuation tokens. Native listings
  over the envelope fail explicitly; no native pagination capability is invented.
- Stream downloads under each consumer's declared byte ceiling and the shared
  temporary quota. Admit known length before reading and unknown-length growth
  before every write. Check shared operations/deadline per request/chunk. Use a
  **5-second** socket timeout clamped at request start to the remaining deadline,
  at most **four attempts** per read using the invocation retry allowance,
  restart the same key on transient partial-read failures, and remove partial
  files before retry/exit. A blocked in-flight read is bounded by socket timeout,
  not hard asynchronous cancellation.
- Use standard Basic/Bearer credentials (including bounded MLflow credential-file
  fallback), custom CA bundles/directories and client PEM configuration for
  artifact reads **and version attestation**. Basic takes precedence over a token.
  Reject redirects, compressed responses, unsupported auth plugins/SigV4 and
  contradictory TLS configuration; never turn an auth/read failure into absence.
- Preserve public `repository.log_artifact` uploads. After an ambiguous immutable
  upload, accept success only after bounded exact readback of that same key.
  Deadline checks bracket SDK upload calls; SDK-internal synchronous upload
  retries/timeouts remain pinned SDK behavior, not a newly claimed hard deadline.
- Validate raw schemes before initializing adapters and validate/read the resolved
  proxy before mutation. For creation, the public Default experiment location is
  only a hint: explicitly pin a unique qualified child location rather than
  inherit an undisclosed current server default. Unqualified/unavailable hints
  fail before experiment creation. Dry-run does the same read-only validation.

Executed verification after the correction:

```sh
MLFLOW_DISABLE_AGENT_HINT=1 PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest \
  tests/unit tests/contract tests/bdd -q --tb=short
# 1711 passed in 49.37s

env -u MLFLOW_TRACKING_URI MLFLOW_DISABLE_AGENT_HINT=1 PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m pytest tests/integration/test_remote_artifacts_profile.py \
  tests/integration/test_session_recovery.py tests/integration/test_resource_quota.py \
  -q --tb=short --durations=5
# 12 passed in 115.25s

.venv/bin/mypy src/mlflow_import
# Success: no issues found in 64 source files
```

Ruff lint and format checks passed for the six Python files in this correction;
`git diff --check` passed and the staging area remains empty. The new proxy
fixture always passes `requested_uri=""` and starts a disposable loopback server;
it cannot honor an ambient remote tracking URI. Its initial **six** MLflow proxy cases
cover fresh import, unchanged zero-mutation rerun, append/prior trace reuse,
dry-run before/after creation, checksummed reads and checksum rejection through
both public URI forms, filtered pending recovery, and both phase-two quota
failure/recovery points. Unit cases cover finite bytes/pages/entries/tokens,
auth/TLS configuration, shared exhaustion, partial-download cleanup/retry,
safe permanent/transient failures and ambiguous-upload exact readback.

The first broad regression invocation omitted `.venv/bin` from subprocess PATH;
its sole failure was entry-point executable lookup, with 1,707 other cases
passing. The corrected PATH invocation above passes completely. New integration
assertions were also corrected to use the established `skipped/unchanged` report
semantics and the fixture's seven-column source schema; those failures were test
setup/assertion defects, not silently accepted transport errors.

The same-host file transfer implementation and benchmark launch recipe remain
unchanged. This is bounded synthetic proxy qualification, **not** a rerun of the
reference benchmark over network artifacts, independent new review certification,
or a production write. At that checkpoint source discovery remained a separate
sequential task; its subsequent correction and read-only result are recorded above.

### Review corrections: explicit CA trust and default collection

The independent reviewer identified two actionable transport/test warnings:

1. **An explicit CA bundle expanded rather than replaced default trust.** Calling
   `ssl.create_default_context()` first loads default roots; a subsequent
   `load_verify_locations()` adds the configured CA. That differs from Requests'
   explicit-bundle semantics and could send authenticated GETs to a server
   outside the selected trust roots. `artifact_http_configuration()` now creates
   the context directly with `cafile=` or `capath=` when a CA is selected, without
   preloading the default store. Precedence remains
   `MLFLOW_TRACKING_SERVER_CERT_PATH`, `REQUESTS_CA_BUNDLE`, then `CURL_CA_BUNDLE`.
   With no explicit CA, default trust still applies. Insecure TLS, contradictory
   profile rejection, client certificate and Basic/Bearer behavior are preserved.
   Version attestation uses this same corrected configuration.
2. **Duplicate test basenames broke pytest's default import mode.** The integration
   module was moved with `apply_patch` to
   `tests/integration/test_remote_artifacts_profile.py`; the unit module keeps
   `tests/unit/test_remote_artifacts.py`. Documentation references/commands were
   updated. No global pytest settings or import-mode workaround was introduced.

The renamed integration module adds **four real offline TLS regressions** using
two locally generated CA/leaf pairs and two loopback HTTPS servers. The existing
local OpenSSL utility generates the disposable certificates; no private SDK
helpers, new Python dependencies or external TLS endpoints are involved. Each
case first proves a real authenticated handshake with the second CA through the
default trust configuration (`SSL_CERT_FILE` / isolated `SSL_CERT_DIR`). With
the first CA explicitly selected, artifact listing/download and `/version`
attestation must accept its server and reject the default-trusted server.
Server-side request capture proves the rejected server receives **no HTTP GET
or Authorization header** after explicit trust selection. Cases cover all three
CA environment settings and a hashed CA directory. The mock configuration test
now requires direct context initialization, rather than testing additive loading.

Before the production fix, all four real TLS cases reproduced the reviewer
warning: the supposedly rejected version GET succeeded (**4 failed in 0.90s**).
After the fix, **4 passed in 0.79s**; all **30** remote-artifact unit cases passed.
The test certificates include proper subject/authority key identifiers for strict
certificate validation; an initial certificate-fixture error was corrected before
the actual failing-then-passing trust regression was established.

Final standard-import-mode verification (no `--import-mode` flag):

```sh
env -u MLFLOW_TRACKING_URI MLFLOW_DISABLE_AGENT_HINT=1 PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/pytest --collect-only -qq
# Default collection succeeds, including both distinct remote-artifact modules.

env -u MLFLOW_TRACKING_URI MLFLOW_DISABLE_AGENT_HINT=1 PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/pytest -m 'not slow' -q --tb=short
# 1886 passed, 4 deselected in 230.60s
```

Final static checks also passed: `ruff check src tests`,
`ruff format --check src tests` (**158 files**), `mypy src` (**65 source files**)
and `git diff --check`. The staging area is empty. The only change after the full
suite was formatting the corrected TLS branch and updating this evidence/docs.

This full non-slow run includes the main agent's source-discovery corrections,
which were preserved. The coordinator reports its actual read-only default-source
command now succeeds against experiment **5**, selecting **4 sessions / 9 turns /
157 records**, planning **9 imports / 0 retirements**. This corrective build pass
did not contact that server, read user transcripts, call SessionsAI or perform a
live import. No files were staged or committed.

### SQL sorter failures: current, legacy and mixed

The first resource-profile reference fresh phase failed after **0.052s**, exit 2,
zero selected sessions and zero mutations, with `temporary_storage_exhausted`.
The cause was **SQLite memory sorter workspace**, not an inadequate byte quota:
payload-bearing joined `ORDER BY` queries attempted to retain the whole selected
transcript under the 64 MiB SQLite heap ceiling.

A deterministic **40 × 2 MiB** synthetic reproduction confirmed this before
changing limits. Current ordered reads now sort structural rowid/declared-PK
locators and fetch payloads by indexed lookup in the same original read transaction.
The strengthened current reference gate subsequently passed (full evidence below).

The analogous legacy audit initially produced **34 workspace failures and two
minimal-todo SQL syntax failures**. All legacy header/message/part/ambiguous-parent/
todo joined and fallback sorts now use the existing locator-only helper. Optional
todo ordering builds a valid rowid-only clause when timestamps are absent.
**48 large-input legacy/mixed cases** cover rowid, WITHOUT ROWID children,
WITHOUT ROWID sessions and combined layouts under unchanged **64 MiB SQLite /
16 MiB temporary** limits; a separate credential-PK test proves excluded columns
are not read. Source scan-count assertions exclude only EXPLAIN-proven indexed
primary-key point reads. **509 focused source tests passed in 21.06s** after this
fix. No full reference-scale legacy timing certification is claimed.

### Final audit warning W1

The default-root execution and dry-run schedules ordered equal/missing creation
times by harness/namespace before native session ID. Distinct-time tests did not
exercise that tie. Both SQL clauses now use:

```sql
ORDER BY created IS NULL, created, session, harness, namespace
```

Four failing-then-passing cases cover equal timestamps and missing timestamps in
both dry-run and execution, multiple harnesses and multiple same-harness roots.
They assert report order and actual trace-create order, including harness and
namespace ties. The focused source-spool/importer/disk-result selection passed
**95 tests in 5.70s**:

```sh
MLFLOW_DISABLE_AGENT_HINT=1 PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest \
  tests/unit/source/test_schema_spools.py tests/unit/test_importer.py \
  tests/unit/test_disk_results.py -q --tb=short
```

W1/doc handoff static checks also passed: Ruff lint/format for the two edited
Python files, mypy across **64 source files**, and `git diff --check`. The staging
area is empty. These focused checks do not replace the main agent's final full
verification below.

This change affects default multi-root tie ordering; it does not modify the
measured explicit-source reference benchmark path. No slow benchmark rerun is
needed for W1 or the documentation refresh.

## Current benchmark and memory verification

Main-agent command:

```sh
env -u MLFLOW_TRACKING_URI MLFLOW_DISABLE_AGENT_HINT=1 PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m pytest tests/integration/test_scale_pipeline.py -m slow -q -s --durations=0
```

**1 passed in 1317.78s**. Each invocation, including full verification, was <600s:

| Invocation | Total seconds | Snapshot | Discovery | Scoped lookup | Reconciliation | Mutation authorization entrypoints |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh | 378.762984708 | 9.837 | 17.222 | 0.214 | 348.391 | 3512 |
| Unchanged | 308.581364792 | 10.306 | 17.389 | 8.927 | 127.847 | **0** |
| Append | 311.297854416 | 10.445 | 17.259 | 8.403 | 128.852 | 12 |
| Edit | 311.746874 | 10.241 | 17.680 | 8.714 | 130.591 | 13 |

Totals include pricing/donor full verification, recovery and cleanup not all
attributed to those phase columns; do not sum the columns as total runtime.
Mutation counts are gateway authorization entrypoints, not HTTP request counts.
Every invocation selected **307 sessions**; actual current turns and exact paged
trace-ID cardinality were **494 / 494 / 495 / 495**, with **zero failures**.
Unchanged trace/map IDs and append reuse were checked; edit replaced the affected
turn. The workload contains **319,987,518 generated JSON bytes**, **8134 messages**,
**47976 parts**, parent/child sessions and multiple turns. It is synthetic V2
linear OpenCode content, not copied source payloads or a tree/replay benchmark.

Hardware/profile: **Apple M4 Pro, 14 CPUs, 48 GiB RAM, Darwin 27.2 arm64,
Python 3.13.11, MLflow 3.17.2 client/server, HTTP loopback, SQLite backend,
same-host `file://` artifacts**. See [compatibility.md](compatibility.md) for the
public binding and exact profile limits.

Main memory suite: `tests/integration/test_scale_memory.py`, **3 passed in
310.73s**:

- **100000 records**, **91.6 MiB artifact**, **51.6 MiB peak importer Python heap**
  (SDK one-turn payload retention excluded).
- **105000 records**, **241.4 MiB artifact**, **16.0 MiB heap** in the streamed case.
- Explicit oversized native-root rejection.

Separate production historical-gateway Pi/Gemini **32/64-turn persisted tree/replay**
cases pass under **32 MiB temporary quota** and **<32 MiB traced Python heap** in
the non-slow integration suite. SQLite has a separate 64 MiB process ceiling.

## Suite and independent-review evidence

The main agent executed these final checks **after the legacy fix, W1 correction
and documentation refresh**:

| Check | Completed result |
| --- | --- |
| Final unit/contract/BDD selection | **1663 passed, 147 deselected, 38.76s** |
| Non-slow integration/profile selection | **130 passed, 1680 deselected, 172.74s** |
| Ruff lint/format | Clean; **168 Python files** formatted |
| Mypy | Clean; **64 source files** |
| Whitespace | `git diff --check` passed |
| Independent code review | No remaining warnings after corrective work |
| Independent security review | Reported redaction leaks and forged-ready authorization defects closed |
| Independent compatibility review | **104 unit + 6 live checks**, no remaining findings |
| Packaging | Wheel and sdist built with `.venv/bin/python -m build --no-isolation` |
| Independent acceptance audit | **99 live + 175 unit checks** initially; W1/documentation closure independently verified with **5 focused tests**; qualified-profile acceptance approved |

Final commands (run from the repository root):

```sh
MLFLOW_DISABLE_AGENT_HINT=1 PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest \
  -m '(unit or contract or bdd) and not integration' -q --tb=short
env -u MLFLOW_TRACKING_URI MLFLOW_DISABLE_AGENT_HINT=1 PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m pytest -m '(integration or profile) and not slow' -q --tb=short
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
git diff --check
.venv/bin/python -m build --no-isolation
```

The earlier 1610- and 1659-unit counts are superseded by this final 1663-unit
result. The four-phase benchmark and memory suites were executed separately as
reported above. Subsequent changes affect redaction, legacy source ordering,
default multi-root tie ordering and documentation, preserving the benchmark's
explicit-source, verbatim current-schema path. No repeat of an unchanged slow
workload was necessary.

## Historical evidence, explicitly superseded

- An early fresh run took **600.586s**, importing 475 turns and failing 19 with
  `retry_budget_exhausted`. Profiling found full-character JSON escaping and
  synchronous per-span round trips. Regex-run escaping preserved canonical bytes
  and reduced that implementation's fresh run to **511.827s**.
- The earlier four-phase timing was **511.827 / 113.937 / 112.882 / 113.831s**;
  its combined memory/pipeline selection passed **4 tests in 1044.86s**. Those
  results preceded later pricing/donor verification and resource changes and are
  not current certification. The 378.762984708/308.581364792/311.297854416/
  311.746874s strengthened gate above supersedes them.
- A previous source/structure measurement retained **64.9 MiB** traced heap;
  indexed reopening record views and streamed activity reduced the named
  100000-record case to **51.6 MiB**. Neither figure includes SDK payload retention.
- A 120-second shell timeout interrupted an intentional slow selection, and a
  console test failed when `.venv/bin` was absent from subprocess PATH. Correct
  tool timeouts/PATH resolved those execution-environment failures; they were not
  production defects.

## Qualification boundaries

The implementation is qualified for the named source variants, pinned public
MLflow APIs and same-host artifact/resource profile, not every source version,
model/rate/currency, arbitrary record/native trace size or concurrent remote writer.
Unrecorded live execution/plugin observations remain unavailable. Pricing is
persisted-first or explicitly qualified/versioned API estimation, never a universal
billing guarantee. Production uses no backend-table writes or run containers.

Old unbound in-flight journals remain unsupported and fail safely, without
automatic migration or destructive repair. Preserve them and use the documented
[non-destructive operator procedure](temporary-resources.md#old-unbound-in-flight-journals).
Typed legacy result decoding does not itself establish ownership or retirement
authority. See [operations.md](operations.md), [result-schema.md](result-schema.md),
[source-adapters.md](source-adapters.md), [projection-parity.md](projection-parity.md)
and [temporary-resources.md](temporary-resources.md) for the actual supported limits.

All corrective fixtures and benchmarks use generated synthetic data. No SessionsAI
content reads/copies, staging or commits occurred in this corrective work.
