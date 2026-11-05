---
name: mlflow-import-implementation
type: implementation-plan
status: planned
created: 2026-11-04
updated: 2026-11-05
sources:
  - docs/architecture.md
  - docs/requirements.md
---

# Implementation Plan: Coding-Agent Session Importer for MLflow

**Design:** [architecture.md](architecture.md) | **Contract:** [requirements.md](requirements.md)

## 1. Starting point and completion target

Build the **snapshot → indexed plan → sequential import** design in small, verifiable increments. This checkout has packaging and CI configuration but no `src/` package, tests, or source fixtures. All work packages below are pending. README/release-history claims are not evidence of an implementation or passing tests in this checkout; the architecture and requirements govern this work.

Completion means a public CLI supporting OpenCode, Claude Code, Codex, PI, and Gemini CLI through explicitly qualified source schemas and MLflow profiles, with recoverable reconciliation, complete canonical artifacts, privacy controls, and bounded resource use. A working first adapter is an intermediate milestone, not completion of the five-harness requirement.

Implementation constraints:

- Keep Python `>=3.11`, `mlflow==3.17.2`, and distribution `mlflow-import` at the existing version. The sole importer command, `mlflow-import`, must call `mlflow_import.cli:main`.
- Start with the architecture's small responsibility groups. Keep projection, filtering, hashing, and redaction as shared functions; use one explicit adapter registry, one disk catalog, and one `MlflowTarget` implementation.
- Complete source validation and the frozen ordered plan before target mutation. Production preflight is read-only; test fixtures may exercise writes against disposable targets.
- Preserve stable turn identities and unchanged trace/map objects. No-op reruns perform zero target writes; replacement retires only exact authorized prior IDs after the all-turn verification gate.
- Use the canonical harness name for session/trace `source` and native `mlflow.source.name`. With `--redact`, mask user identity in native/namespaced metadata and all artifact copies before persistence.
- Use the standard library and the existing MLflow dependency at runtime. Introduce no plugin framework, ORM, run container, watch mode, or distributed lock service.

## 2. Delivery order and dependency gates

IDs are work-package references, not claims that tickets already exist. Follow the order below for the first vertical slice. P13–P16 share the completed infrastructure and have no dependency on one another.

| ID | Deliverable | Depends on |
| --- | --- | --- |
| P01 | Runnable package, sole console entry point, test/build scaffold | — |
| P02 | Qualified MLflow 3.17.2 test profile and API evidence | P01 |
| P03 | DTOs, identity/canonicalization contracts, disk catalog | P01 |
| P04 | Adapter contract, safe snapshots, fixture/conformance harness | P03 |
| P05 | OpenCode adapter and schema matrix | P04 |
| P06 | Shared turn reconstruction, summaries, filters, ordering | P05 |
| P07 | Outbound redaction, diagnostics, fingerprint domains | P03, P06 |
| P08 | Production MLflow boundary, verification, shared retry budgets | P02, P03, P07 |
| P09 | Canonical artifact bundle and read-only reconciliation planner | P06, P07, P08 |
| P10 | Public CLI, complete read-only pipeline, dry-run and reports | P05, P09 |
| P11 | Durable recovery state machine and local writer exclusion | P08, P09 |
| P12 | Integrated OpenCode write/retry/reconciliation vertical slice | P10, P11 |
| P13 | Claude Code adapter | P12 |
| P14 | Codex adapter | P12 |
| P15 | PI adapter | P12 |
| P16 | Gemini CLI adapter | P12 |
| P17 | Cross-harness acceptance, failure and resource-bound evidence | P13, P14, P15, P16 |
| P18 | Installed-package checks, CI completion, operator documentation | P17 |

Milestone gates:

1. **Foundation — P01–P04:** the installed command works, index/snapshot contracts pass, and the first target profile is demonstrated on a disposable server.
2. **Read-only import planning — P05–P10:** OpenCode produces a deterministic complete plan and dry-run with no MLflow mutations, including filtered-out recovery discovery.
3. **Recoverable first import — P11–P12:** the public CLI completes and resumes the entire OpenCode lifecycle. Enable its write path only after the recovery tests pass.
4. **Five-harness coverage — P13–P16:** each adapter passes its own fixtures, the shared contracts, and the public-CLI integration scenarios.
5. **Release readiness — P17–P18:** resource/failure evidence, installed-package checks, and accurate support/operations documentation are complete.

Execution and handoff rules:

- A package closes only when its checkboxes and package acceptance have evidence. P02 closes at its **provisional API-profile gate**; final write-support qualification closes at P17, so P08 does not depend on future recovery evidence.
- Dispatch dependency-ready packages with disjoint file ownership. Shared-file edits are sequential. Record changed files, exact commands/results, profile/fixture evidence, and unresolved blockers in each handoff; a worker's completion message alone does not close a gate.
- Use §4's verification commands from P01 onward. P02 seeds `docs/compatibility.md` and P10 creates `docs/result-schema.md`; P18 updates those documents with final evidence.

## 3. Work packages

Bare production module paths below are relative to `src/mlflow_import/`; repository-level test, documentation, and configuration paths are written explicitly.

### P01 — Establish the executable scaffold

**Files:** `pyproject.toml`, `src/mlflow_import/{__init__,cli}.py`, initial tests and CI adjustments.

- [ ] Create the `src/mlflow_import` package and a minimal `main()`/help path; register only `mlflow-import` against that function.
- [ ] Use distribution `mlflow-import` and retain the MLflow pin; update package description and first-party tooling names for `mlflow_import`. Provide no alternate importer command or Python-package shim.
- [ ] Establish `tests/unit`, `tests/contract`, `tests/acceptance`, `tests/integration`, and `tests/fixtures` using the existing pytest markers. Preserve strict typing and Ruff checks; remove historical test exceptions only when their referenced code is actually replaced or unnecessary.
- [ ] Make editable installation and wheel/sdist builds work. Keep the existing Python 3.11/3.12/3.13 default CI matrix.

**Acceptance:** a clean environment installs the project with exactly one importer console entry point, `mlflow-import`; its help succeeds, default test collection succeeds, and lint/type/build checks pass. CLI work in P10 fills in the import behavior.

### P02 — Qualify the pinned target before building the writer

**OpenCode status:** provisional profile gate complete. Five disposable-server profile tests pass; final write-support qualification remains a P17 gate.

**Files:** `tests/integration/`, reusable server/repository fixtures, planned `docs/compatibility.md`.

- [x] Launch a disposable HTTP tracking server from MLflow 3.17.2 with an explicit SQLite backend, absolute artifact root, loopback host, and allocated port; record the exact launch configuration. Assert fixture-owned URI/experiment/artifact scope before destructive operations, independently of ambient MLflow settings. Qualify an initial same-host `file://` experiment artifact repository accessible to the importing process; read back its actual absolute location and test with distinct client/server working directories. Record that locality constraint. Additional artifact schemes require separate profile evidence.
- [ ] Exercise the architecture §6 bindings directly: historical root/child spans, complete structured values beyond search-preview lengths, native session metadata, span events/status/usage, explicit flush, paginated search and full trace read-back. Prove the exact escaped canonical/legacy conflict and attempt/instance/turn recovery queries needed by P08–P11; unavailable bounded lookup semantics disable that profile.
- [ ] Demonstrate managed tag set **and removal**, exact-ID deletion with absence verification, and run-free experiment-root artifact upload/list/download with streaming checksum verification. Assert no runs were created.
- [ ] Record server/client versions, repository scheme, field limits, payload envelope, timing/status behavior, consistency/verification bounds, and API failure signatures. Start with a measured envelope; do not invent a universal payload limit.
- [ ] Pin failure cases for unsupported versions/schemes, denied operations, NoOp spans, and ambient exporter configuration that prevents expected remote persistence. These tests become fixtures for P08/P11 rather than a second production target abstraction.

**Acceptance — provisional API-profile gate:** repeatable API evidence establishes an exact provisional profile consumable by P08. **Final write-support gate, closed at P17:** qualification additionally requires P11–P12 recovery tests and P17's complete evidence matrix. Any unsupported capability remains disabled; production preflight never creates probe runs, traces, or artifacts.

### P03 — Define shared contracts and the disk catalog

**OpenCode status:** implementation and contract evidence complete for the qualified OpenCode source/target path.

**Files:** `models.py`, `catalog.py`, shared canonicalization functions in `projection.py`, contract fixtures.

- [ ] Define immutable DTOs for the architecture's source candidates/formats, frozen sources, sessions/records, summaries, turns, desired trace/span projections, ordered plans, target references, attempts, and result items. Keep transcript bodies out of summary/plan DTOs.
- [ ] Implement normalized owning-root namespaces and the versioned qualified session tuple. Store native IDs separately; default discovery and explicit files must resolve the same owning root. Document the locator-derived namespace's move limitation.
- [ ] Build restricted spool storage and the temporary SQLite offset/parent/turn/target index, scoped by qualified session identity, with uniqueness constraints, indexed cursors, bounded cache/batch sizes, and streaming temporary-space checks.
- [ ] Define the versioned canonical JSON/raw-value representation, including missing versus null, raw timestamps, unknown fields, non-JSON numbers/bytes, and duplicate-key preservation where encountered. Add streaming encoders/hashes without session-sized Python values.
- [ ] Define artifact/attempt/result schema versions and the shared versioned compatibility-decoder interface. Capture sanitized legacy ownership/result fixtures from version-controlled historical contracts or authorized release exports before accepting them; keep an unavailable decoder disabled. Unknown or ambiguous versions must not be guessed into the new contract.

**Acceptance:** golden canonical bytes/hashes are deterministic; namespace equivalence and collision cases pass; duplicate IDs fail; catalog traversal uses indexes and streaming cursors. Larger fixtures increase disk usage, not in-memory record collections.

### P04 — Build safe source freezing and the adapter conformance harness

**OpenCode status:** implementation and WAL/snapshot/conformance evidence complete for the qualified OpenCode path.

**Files:** `catalog.py`, `sources/__init__.py`, snapshot/conformance tests and fixture generators.

- [ ] Implement the typed adapter protocol and explicit registry. Register only implemented, fixture-qualified formats; an incomplete adapter is not advertised as supported.
- [ ] Provide shared SQLite read-only/query-only transaction helpers. The global pass may read only allowlisted session-header metadata. Freeze one selected session per transaction/query scope; every transcript, sidecar, blob, parent, and event query must carry an indexed session predicate. A `--session` invocation must not read or copy unrelated records. Require the companions/access needed for a genuinely read-only open. Never copy the whole database or bypass live WAL consistency with `immutable=1`.
- [ ] Provide file freezing helpers that first index candidate session headers, then stream only one selected session's linked records/sidecars through schema-aware exclusions. Compare identity/size/mtime/digests against a second read. Recheck eligible inventory and enforce one bounded retry budget. Define no-follow source discovery/freezing behavior and reject symlink loops or escapes before reading payloads; test explicit-file/root namespace equivalence for each advertised locator form.
- [ ] Enforce owner-only temporary directories/files, snapshot-only source reads in phase 2, cleanup on handled failure/interrupt, and temporary-disk exhaustion handling. Source state must never be changed by cleanup.
- [ ] Create fixture manifests recording harness/schema/upstream evidence, authoritative-record/path rules, units/outcomes, associations, excluded fields, and deterministic fallback IDs. Use synthetic transcripts and credential canaries, not private sessions.
- [ ] Add conformance tests for safe pushdown equivalence, excluded-session churn, new eligible sessions, session-filtered read isolation, one-session temporary-space bounds, sidecar changes, malformed noncritical values, and required identity/order/ancestry corruption. Instrument reads to reject prohibited table/column values and unrelated-session record reads, not merely their later upload.

**Acceptance:** live-writer WAL and file-churn tests produce the documented stable view or fail before target writes. The importer performs no source mutation or prohibited-value read; cleanup and space limits are enforced.

### P05 — Implement OpenCode as the first complete source adapter

**Files:** `sources/opencode.py`, `tests/fixtures/opencode/`, adapter and snapshot tests.

- [ ] Implement default discovery and `OPENCODE_DB`, explicit database detection, stable owning-root namespaces, and source-representation deduplication.
- [ ] Support fixture-backed legacy `session`/`message`/`part` and current `session_v2`/`session_message`/part layouts. Prefer current authoritative data per session; incomplete current data fails rather than falling back.
- [ ] Handle the distinct migration-input schema and default migration location from architecture §12, including overridden database paths, `:memory:`, and an explicit upstream migration path. Never invoke migration code.
- [ ] Preserve ordered permitted messages/parts, nested model identities, session metadata, unknown fields and unambiguously associated event/todo/instruction records. Report unassociated records and exclude credential/account/token/share values before staging.
- [ ] Record exact supported schema signatures and timestamp/usage/outcome normalization in the fixture matrix; reject unsupported or ambiguous layouts. Candidate discovery must use header/index queries first, and session extraction must be scoped to the selected native session ID.

**Acceptance:** legacy/current/mixed/migration/WAL fixtures pass, default and explicit identities agree, source representations deduplicate, and complete permitted raw data survives freezing. Include a six-turn transcript for later end-to-end checks.

### P06 — Reconstruct and select turns once

**Files:** `projection.py`, catalog index operations, projection/selection tests.

- [ ] Implement linear, indexed persisted-parent-path, and adapter-provided effective replay traversal. Persist ancestry/visited/path state on disk; reject cycles, missing required parents, duplicate IDs, and ambiguous heads.
- [ ] Start turns only at persisted primary user prompts. Keep preamble, alternate branches, original replay records, subagent records, unmatched tools, and zero-turn data available to the canonical session artifact.
- [ ] Use stable initiating-record IDs as turn keys and separate display indices. Pair tool calls/results within a trace and retain external parent/call links as provenance rather than creating cross-trace parents.
- [ ] Derive each session's summary, full model index, primary-path turn count, timing/outcome/usage facts, and turn locators in fixed streaming passes. Model filters include permitted inactive branches; usage accounting avoids parent/child and cumulative-counter double-counting.
- [ ] Build the source-neutral desired trace projection: complete structured user input and final assistant output (null when absent), typed LLM/agent/tool/content spans/events, and complete source-ID provenance. Apply declared timing units and distinct non-OK outcomes; preserve missing/invalid/reversed timing and mark any API-required fallback synthetic. Preserve unsupported usage categories/resets and never invent costs.
- [ ] Derive normalized session/trace `source` and native `mlflow.source.name` from the selected/detected adapter name (`opencode`, `claude`, `codex`, `pi`, or `gemini`). Keep locator, namespace/fingerprint, model/provider, importer, and Git provenance separate; use the same mapping for default and explicit sources in either privacy mode.
- [ ] Implement timezone-required inclusive creation-time bounds, component-safe directory normalization (relative-to-absolute, dot segments, and separators, without symlink resolution/existence checks), exact normalized provider/model matching, session-ID selection, strict `turns > N`, and AND composition. Session ID, creation time, and directory filters push down to the candidate header index; model and turn-count filters read only candidate sessions. A native session-ID filter matches each qualifying harness/namespace identity; it does not merge them or make a legacy alias unambiguous.
- [ ] Persist deterministic session/turn order, including missing creation-time handling and harness/namespace tie-breakers. Recheck safely pruned header facts against the frozen view.

**Acceptance:** filter boundaries and six-turn/branch/replay/subagent/no-turn cases pass against a full-scan reference. Changing display indices does not change stable turn keys; query/pass counts rule out per-turn transcript scans.

### P07 — Apply privacy once and separate change domains

**Files:** `privacy.py`, shared fingerprint/projection functions, diagnostic helpers, privacy fixtures.

- [ ] Implement architecture §9's recursive structured-key/value redaction, free-text patterns, PEM precedence, lexical assignment handling, bounded nested-JSON decoding, and conservative opaque/malformed-value masking.
- [ ] With `--redact`, mask present user identity as `[REDACTED]` regardless of secret-pattern matches: user IDs, usernames, names, email addresses, nested user objects, `mlflow.trace.user`, any populated `mlflow.user`, and equivalent namespaced fields. Cover session/trace metadata and retained normalized/raw/escaped artifact copies. Keep missing users absent and prevent OS/SDK defaults from repopulating identity; preserve the structural role `user`.
- [ ] Apply it to every outbound normalized/raw representation, including nonidentity object keys and collision-preserving key/value wrappers. Keep reconciliation IDs/schema names stable and selection independent of redaction.
- [ ] Count distinct affected logical fields/records during projection. Preserve raw fidelity in verbatim mode except for mandatory adapter exclusions; restricted staging remains unredacted and must be documented accurately.
- [ ] Implement separate session/source, metadata, turn, and exact-artifact-byte digests. Exclude generated IDs/paths and session-wide display metadata from turn fingerprints; include immutable native user/source values, projection/privacy versions, and hash only redacted content in redacted mode.
- [ ] Install safe importer diagnostics and suppress/filter potentially sensitive SDK logs at every level. Errors contain operation codes and safe identities, not raw payloads/exceptions or credential-bearing URIs.

**Acceptance:** structured/escaped/malformed secrets, unterminated assignments/keys, key collisions, opaque bytes, reconciliation IDs, plain user identities, absent users, SDK user defaults, preserved conversation roles, and nonmatches pass. Captured logs/stdout/stderr and uploaded projections contain no fixture secret or unmasked user-identity field in redacted mode; diagnostics remain content-free in verbatim mode. Retries do not inflate counts, including user-field counts.

### P08 — Implement the single MLflow boundary

**Files:** `mlflow_target.py`, target unit/contract tests, P02 profile fixtures.

- [ ] Implement read-only profile/capability checks and exact-name experiment lookup/create/recheck. Reject deleted experiments; preserve existing settings; re-resolve a creation race or ambiguous response before retrying.
- [ ] Bind historical trace/span creation, full read-back, explicit flushing, paginated metadata lookup, managed tag diff/set/delete/read-back, and bounded exact-ID retirement with verified absence.
- [ ] Put native grouping and immutable source identity in request metadata; make mutable tags authoritative for refreshable importer/session fields, without stale immutable shadow copies. Preserve unrelated tags and omit absent source values.
- [ ] Set and read back native `mlflow.source.name` as the canonical harness name and native user fields as the privacy-projected values. Control SDK-populated `mlflow.user` before persistence; verify no unmasked user is introduced by defaults in redacted mode. Treat changes to immutable native user/source values as replacements, not metadata-only refreshes.
- [ ] Isolate the pinned exported artifact-repository factory and upload/list/download operations. Reject unavailable experiment-root integrations without private helpers, run containers, or inferred trace-storage paths.
- [ ] Verify complete values, status, timing, mapped children/events, and managed metadata against the intended projection; fingerprints/search previews alone are insufficient. Use artifact overflow references for oversized nonidentity metadata and short experiment-relative artifact keys. Reject NoOp/export degradation, identity overflow, and payloads beyond the tested envelope without truncation.
- [ ] Apply one shared deadline/retry budget across SDK retries, polling, backoff, `Retry-After`, artifact generations, and recovery calls. Explicitly control and test MLflow's internal request retry/timeout configuration so nested calls cannot reset the outer budget. Record finite defaults for attempts, deadline, backoff, generations, lock wait, redaction depth, and temporary-space limits in the applicable profile/operations evidence. Lookup ambiguous non-idempotent outcomes before any create retry; do not retry invalid/auth failures as transient.
- [ ] Select required capabilities by the forecast action: read-only no-op, metadata refresh, creation, or cleanup. Validate a new experiment's actual artifact location before session writes; first-write permission denial remains a safe operational failure.

**Acceptance:** the production wrapper passes the P02 profile tests and injected auth/network/429/5xx/partial-response cases. No adapter imports MLflow, and no operation accesses the tracking backend directly.

### P09 — Plan conflicts and publishable artifact bundles

**Files:** `projection.py`, `pipeline.py`, `catalog.py`, artifact/reconciliation contract tests.

- [ ] Build streaming session-record JSON, reusable turn-map JSON, joining session-manifest JSON, and immutable attempt-state encoders. Use versioned safe paths, separate privacy/duplicate scopes, exact-byte digests stored outside the objects, and no hash/reference cycles.
- [ ] Add existence/byte verification, partial-object generation handling, and an invocation-wide shared-artifact verification cache. Generation retries consume the same budget; referenced verified objects are never overwritten.
- [ ] Page canonical and fixture-backed legacy/plugin alias conflicts into the disk index. Resolve aliases only with unambiguous source evidence; decode legacy ownership/artifact/result versions without silently relabeling their schemas. Never reuse a foreign trace.
- [ ] Forecast reuse, metadata-only update, replacement, stale/extra retirement, ignore, duplicate, and zero-turn actions. Default overwrite covers only the selected session's resolved conflicts, including prior duplicate copies and foreign traces; exact retirement IDs must exclude reused current traces.
- [ ] Reject reuse when immutable native user/source values differ from the intended projection, including a legacy importer/path-valued source or an unredacted user under the current redaction profile. Plan verified replacement under overwrite while preserving ignore/duplicate policy semantics.
- [ ] Discover attempts by stable source/instance scope independently of current content filters/fingerprints. A zero-match plan still reports excluded unfinished attempts and filter-relaxation guidance with exit `1`.
- [ ] Recognize overwrite's true unchanged reconciliation only after full trace/artifact verification; allocate no new marker, instance, artifact, or tag write for that case. A requested duplicate still creates a new instance after completed copies. Zero-turn new/ignore/duplicate sessions produce no trace or artifact writes; overwrite cleanup uses a verified empty manifest.

**Acceptance:** planned action/retirement sets are deterministic and session-scoped across ownership/policy/alias cases. Append and metadata-only changes retain unchanged trace/map identities; one shared session artifact is verified once. Planning itself performs no mutations.

### P10 — Deliver the complete read-only public CLI

**Files:** `cli.py`, `pipeline.py`, `reporting.py`, `docs/result-schema.md`, CLI/acceptance tests.

- [ ] Implement every requirements §6 option, including `--log-level`, standard MLflow configuration, and the four harness/source selection combinations. With no sources available fail configuration; skip missing optional defaults but reject a missing explicit source or ambiguous/mismatched detection.
- [ ] Orchestrate discovery → authoritative session-header index → safe header pruning → one-session freeze/index/project/content-filter/hash → ordered disk plan → read-only target/recovery forecast. Release each session snapshot after its plan item is spooled. Validate all eligible source units before entering the eventual write phase without requiring a full-source transcript snapshot.
- [ ] Make `--dry-run` execute that complete path without acquiring a writer lock, creating an experiment, repairing state, creating traces, changing tags, retiring traces, or uploading artifacts. Identify provisional recovery forecasts rather than promising an execution result.
- [ ] Define and document `mlflow-import.result.v1`: deterministic session/turn items, safe identity/error fields, attempt/instance state, planned versus committed actions, and unambiguous count meanings. Keep legacy results decoded under their actual schema version.
- [ ] Spool result items and stream the final JSON document; send human progress to stderr and only the chosen final format to stdout. Report broad selection and default overwrite scope before future writes.
- [ ] Implement exit codes `0`/`1`/`2`, including excluded pending recovery and zero-match distinctions. Count selected sessions/turns separately; an unchanged turn increments `unchanged`, not session-policy `skipped`. Metadata-only changes count `updated` with the same trace ID; zero-turn cleanup never invents an imported turn. `retired` counts confirmed deletions even when completion is pending.

**Acceptance:** public-CLI BDD scenarios cover every filter/source-option combination, deterministic output, invalid invocation, safe diagnostics, and zero target mutations in dry-run. The CLI write path remains gated until P11–P12 pass.

Before P12 enables writes, a non-dry-run invocation fails with exit `2` and a safe `write_path_not_enabled` code before lock acquisition or experiment creation; it must not report an import as completed.

### P11 — Implement the durable writer and recovery chain

**Files:** `pipeline.py`, lock/retry helpers, recovery contract tests.

- [ ] Acquire one credential-free OS advisory lock per normalized local target/experiment name before any creation/recovery/write; hold it through cleanup. Define a deterministic per-user lock location, protected open flags, and bounded wait charged to the shared deadline; contention timeout is an operational failure before writes. Recheck capabilities/conflicts under the lock. Lock release follows process/OS semantics; do not auto-break it. Document external serialization for multi-host writers.
- [ ] Implement `prepared`: upload/verify the session object and durable preparation record before any trace call; record desired turn keys, source/projection/privacy/policy and instance identity, and prior conflicts.
- [ ] Stage or reuse each turn, recover ambiguous creates by attempt/turn identity, upload/verify maps, reconcile tags including removals, flush, and verify full traces/artifacts. Continue later turns after individual failures; retain all prior traces if staging is incomplete.
- [ ] Publish/verify the joining manifest and `ready_to_retire` only after every current turn and metadata refresh verifies. Revalidate conflicts, excluding this attempt's staged work, and record exact authorized retirement IDs.
- [ ] Reverify replacements before bounded retirement; confirm absence and count each confirmed deletion even if later work fails. Promote/read back staged tags and publish/verify `complete` only after all required work succeeds.
- [ ] Resume same-input prepared work without an extra duplicate. For source/privacy/policy changes before ready, retire only that attempt's newly created traces and verify `aborted`; after ready, finish the recorded verified batch before reconciling the latest source.
- [ ] Recover by remote source/instance/attempt state without local snapshots. Invalid/partial ready or complete objects cannot prove retirement never started; reconstruct/verify the same batch or halt. Conflicting terminal states, competing attempts, missing replacements, and failed cleanup block that session safely.

**Acceptance:** injected crashes/lost responses at every state boundary resume or halt safely, without premature prior-ID retirement or untracked duplicate creates. Prepared failures never start traces; filtered-out sessions are never repaired; lock contention is serialized on each supported host platform.

### P12 — Complete the first public-CLI write slice

**Files:** CLI/pipeline integration, `tests/acceptance/`, disposable-server integration tests.

- [ ] Connect the frozen plan, lock, experiment resolution, durable writer, and reporting through the installed `mlflow-import` command. Phase 2 reads only snapshots; target-plan changes trigger relevant read-only rechecks. Clean no-op plans perform no mutations, including no experiment creation.
- [ ] Import the six-turn OpenCode fixture into new/existing experiments; read full traces, source/map/manifest objects, timing/status/usage, native grouping, and managed metadata through public APIs.
- [ ] Exercise unchanged, append, edit, removed-turn, metadata-removal, privacy-change, duplicate, ignore, foreign overwrite, and zero-turn cleanup. Verify unchanged trace IDs/maps survive and unchanged reruns make zero writes.
- [ ] Read back harness-valued session/trace `source` and native `mlflow.source.name`; verify redacted native/namespaced user fields and every artifact copy, absent-user handling, preserved roles/IDs, and SDK default suppression. Cover overwrite replacement of immutable user/source values, followed by a zero-write unchanged rerun.
- [ ] Retry prepared/ready failures with same and changed source, policy, and privacy, including partial retirement after local snapshots have been removed. Include ignore's own incomplete attempt and filtered-out pending recovery with zero selected sessions.
- [ ] Verify per-item failure continuation, first-write denial, sanitized diagnostics, cleanup/interrupt behavior, committed-turn counts, and partial-retirement reporting. Staged work is not reported as a completed import.

**Acceptance:** the full OpenCode lifecycle passes through the public CLI against the qualified profile, with fault-injected invariants as well as happy-path fidelity. This is the first write-enabled milestone.

### P13 — Add Claude Code

**Files:** `sources/claude.py`, `tests/fixtures/claude/`, shared contract/CLI scenarios.

- [ ] Discover project JSONL sessions under the default or `CLAUDE_CONFIG_DIR`; normalize owning-root identity and detect supported line schemas.
- [ ] Freeze unambiguously linked subagent transcripts and `tool-results` sidecars; preserve UUID/parent/order links, multipart records, usage/outcomes, unknown fields, and primary-versus-subagent roles.
- [ ] Define authoritative persisted conversation/fork semantics from fixture-backed evidence; handle missing/retained/deleted-history cases explicitly and reject ambiguous linkage.
- [ ] Add stable-file/sidecar, namespace-equivalence, secret-exclusion, projection, and public-CLI import/retry/unchanged scenarios.

**Acceptance:** the adapter supplies facts through the existing protocol and passes the same safety/reconciliation contracts as OpenCode without another turn/filter implementation.

### P14 — Add Codex

**Files:** `sources/codex.py`, `tests/fixtures/codex/`, shared contract/CLI scenarios.

- [ ] Honor `CODEX_HOME`; discover supported dated rollouts, archives, and fixture-qualified compressed siblings. Deduplicate representations before conflict policy; never use `history.jsonl` as the full transcript.
- [ ] Preserve session metadata, response items, events, turn context, compaction, usage, and fork/revert links. Define authoritative replay/path and timestamp/counter semantics per schema.
- [ ] Stream supported decompression under explicit byte/disk limits; unsupported encodings/versions fail compatibility rather than silently omitting sessions.
- [ ] Test plain/archive/compressed identity equality, compaction/reverts, counter resets, malformed records, source churn, raw fidelity, and shared CLI lifecycle cases.

**Acceptance:** each advertised rollout/decoder schema has evidence and conformance fixtures; equivalent representations import once with stable turn identity.

### P15 — Add PI

**Files:** `sources/pi.py`, `tests/fixtures/pi/`, shared contract/CLI scenarios.

- [ ] Implement default/environment session roots and explicit `--source` handling for alternate session directories, including encoded-cwd identity/metadata.
- [ ] Support fixture-backed linear v1 and parent-linked v2/v3 formats. Follow the documented persisted resume-head rule using the shared disk ancestry index; preserve alternate branches.
- [ ] Preserve source IDs, model/usage/cost/outcome semantics, compaction/branch records, and unknown values. Never call loaders that migrate/rewrite source files.
- [ ] Exercise ambiguous/missing heads, cycles, large branching histories, branch-only model matches, rewind/index changes, and shared CLI lifecycle cases.

**Acceptance:** observed persisted paths produce the expected primary turns; alternates remain in artifacts; traversal remains indexed and bounded without a Python session graph.

### P16 — Add Gemini CLI

**Files:** `sources/gemini.py`, `tests/fixtures/gemini/`, shared contract/CLI scenarios.

- [ ] Discover supported chat storage under the default, `GEMINI_CLI_HOME`, and fixture-backed sandbox layouts. Validate project registry identity and legacy hash-directory migration/deduplication rules.
- [ ] Replay supported message and `$set`/`$patch`/`$rewindTo` records in the shared disk working store, retaining original updates and effective records.
- [ ] Pin source signatures, ordering, associations, model/usage/outcome semantics, and incomplete/invalid replay behavior; do not infer support from product/account availability.
- [ ] Exercise rewinds, patches, unknown records, ambiguous registries, moved/alternate representations, excluded source fields, and shared CLI lifecycle cases.

**Acceptance:** effective conversations and retained raw replay history match fixture expectations; support claims enumerate only tested storage/replay versions.

### P17 — Prove cross-harness correctness and resource bounds

**Files:** generated scale fixtures, cross-harness BDD/contract/integration tests, profile evidence.

- [ ] Run the complete selection/fidelity/reconciliation matrix across all qualified adapters, including same native IDs in different harnesses/namespaces and ambiguous legacy/plugin aliases.
- [ ] For each harness, verify canonical `source` values under default/explicit selection and both privacy modes, plus user-identity masking across native/namespaced fields and normalized/raw/escaped artifacts. Include plain names/IDs/emails without secret-pattern matches, missing users, SDK defaults, preserved `user` roles/reconciliation IDs, and immutable-field replacement.
- [ ] Exercise failures/lost responses at artifact/session/map/state upload, trace start/end/flush, tag set/delete, ready/complete publication, and each partial-delete boundary. Cover invalid terminal objects, missing replacement data, multiple attempts, and external conflict changes.
- [ ] Measure peak memory for large linear/tree/replay sources, separately accounting for the profile's one-turn payload envelope. Test explicit oversize failure without truncation and safe temporary-disk exhaustion.
- [ ] Instrument source passes, indexed ancestry lookups, target pages/search scopes, upload/download counts, retry/deadline consumption, and spooled JSON output. Require fixed transcript passes, no whole-experiment lookup per turn, one shared-artifact verification, and zero unchanged-run writes.
- [ ] Confirm append changes one new turn trace/map plus shared session/manifest state and necessary O(turns) small tag diffs, rather than rebuilding earlier maps or duplicating transcripts per turn. Measure unrelated-source reads and require session-filtered imports to remain bounded by candidate headers plus selected-session records.
- [ ] Run Linux/macOS lock and cleanup checks and the Python compatibility matrix. Use deterministic operation counts and resource budgets, not brittle wall-clock thresholds.

**Acceptance:** architecture §11's evidence matrix is satisfied for the declared source/target profiles. Any unsupported format/scheme remains an explicit pre-write compatibility error, not a skipped assertion or silent downgrade.

### P18 — Finish distribution, CI, and operating guidance

**Files:** `README.md`, `docs/compatibility.md`, `docs/operations.md`, `docs/result-schema.md`, fixture support matrix, `.github/workflows/ci.yml`.

- [ ] Update README claims/examples to the implemented contract: timezone-required bounds, shared artifact bundle, actual source/profile support, and the sole `mlflow-import` console entry point in the `mlflow_import` package. Preserve historical changelog entries; add accurate new release notes through the project's release process.
- [ ] Document source schemas and path/authority/branch rules, MLflow profile/scheme/envelope limits, pinned exported artifact integration, legacy ownership/result migration, and unsupported-profile behavior.
- [ ] Document overwrite's default foreign-conflict scope, dry-run forecasts, zero-turn behavior, recovery/filter relaxation, local versus multi-host serialization, retry/resource bounds, safe error codes, and experiment/trace/artifact inspection.
- [ ] Explain verbatim sensitivity, mandatory exclusions, secret and user-identity masking (including native MLflow user fields), redaction versions/counts, restricted unredacted staging, and the fact that changing privacy mode does not erase old uploads. Document canonical harness-valued session/trace `source` separately from `--source PATH`. Record artifact retention/chunking as deferred scope.
- [ ] Keep default unit/contract/BDD/lint/type checks unconditional. Add a required disposable-server profile/integration lane for release evidence; preserve a separately configured external-profile lane that fails loudly when enabled against an unsupported target. Remove stale CI references to absent historical tasks/tests.
- [ ] Build and install wheel and sdist in clean environments; verify the sole console entry point, packaged files, dependency pin, help and representative fixture imports through the installed `mlflow-import` command.

**Acceptance:** documented support corresponds to passing fixtures/profile tests, CI actually exercises the published contract, and fresh installations reproduce the demonstrated lifecycle.

## 4. Verification commands and test placement

These are implementation checks to establish during P01–P18; this planning change does not claim they already pass. Install the development dependencies once the scaffold exists:

```bash
python -m pip install -e '.[dev]'
```

Run the relevant unit/contract/BDD selection for each package as it lands, then the complete default selection at milestone gates:

```bash
python -m pytest -m '(unit or contract or bdd) and not integration'
ruff check .
ruff format --check .
mypy src
```

Run live WAL/server/profile tests in the disposable fixture environment, including slow resource tests for release qualification:

```bash
python -m pytest -m '(integration or profile)' -q
python -m pytest -m slow -q
python -m build --no-isolation
```

Live fixtures must distinguish the disposable server they own from an explicitly configured external test target. Destructive/failure-injection tests use isolated experiment/artifact scopes and never a normal user experiment. Unit markers cover pure functions; SQLite/WAL tests use contract/integration markers as appropriate rather than contradicting the existing `unit` definition.

## 5. Requirements-to-work traceability

| Contract area | Architecture | Primary implementation | Required evidence |
| --- | --- | --- | --- |
| CLI/source combinations and exact filters, requirements §§6–7 | §§2–3, 10 | P04–P06, P10, P13–P16 | Boundary/AND filters, namespace equality, ambiguity, complete dry-run |
| Read-only stable sources and exclusions, requirements §§7, 12 | §§2–3, 9 | P04–P05, P07, P13–P16 | WAL/live writer, file/sidecar churn, denied secret-value reads, cleanup |
| Session/turn fidelity, requirements §§4, 7–8 | §§3–7 | P03, P05–P09, P13–P16 | Full payload/artifact/map read-back, harness-valued source, six turns, branch/replay, usage/outcome units |
| Identity/legacy compatibility, requirements §§4, 8–9 | §§1, 4–5, 8 | P03, P05, P08–P09, P12 | Default/explicit equality, legacy/plugin aliases, ambiguity, managed tag removals |
| Conflict policy and durable recovery, requirements §§9, 11 | §§2, 7–8 | P09–P12, P17 | Exact retirement sets, all-turn ready gate, same/changed input, remote-only recovery |
| Privacy and diagnostics, requirements §§10, 12 | §§3–5, 9 | P04, P07–P09, P10, P12, P17 | Every raw/normalized representation, malformed/escaped secrets and user identity, native fields/SDK defaults, preserved roles/IDs, immutable-field replacement, log canaries |
| Results/failure isolation, requirements §§6, 11, 13 | §§2, 8, 10 | P08, P10–P12, P17–P18 | Committed counts, pending excluded attempts, partial retirements, exit statuses |
| Bounded performance, requirements §§7, 11, 13 | §§3–4, 7–8, 10 | P03–P04, P06, P08–P12, P17 | Pass/query/page/write counts, RAM/disk envelope, no-op zero writes, shared retry budgets |
| Five-harness and target support, requirements §§4, 7, 14–15 | §§6, 11–12 | P02, P05, P12–P18 | Fixture/version matrix, real-server profile, installed-CLI acceptance |

## 6. Completion checklist

- [ ] Exactly one importer console command, `mlflow-import`, installs and executes `mlflow_import.cli:main`.
- [ ] All five adapters have explicit fixture-backed schema, authority/path, exclusion, and normalization rules.
- [ ] Source safety and complete planning precede target writes; dry-run remains entirely read-only.
- [ ] Qualified targets preserve full root/span/artifact content and support verified mutable metadata and exact retirement.
- [ ] Every harness exports its canonical name as session/trace `source` and native `mlflow.source.name`, independently of locator and privacy mode.
- [ ] Unchanged reruns perform zero writes; append/edit/metadata-only cases reuse the intended traces/maps.
- [ ] Remote durable state recovers ambiguous writes and interrupted retirement without unsafe cleanup or extra duplicates.
- [ ] Secret and user-identity redaction (including native user fields and artifact copies), source exclusions, log safety, deterministic result counts/order, and exit statuses pass public-CLI tests.
- [ ] Resource/operation-count tests demonstrate the architecture's bounds and local writer-serialization boundary.
- [ ] Documentation, packaged entry points, CI, and declared support agree with the tested implementation.
