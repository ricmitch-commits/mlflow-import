# Source adapter qualification (2026-11-14)

This records source-processing qualifications. The
[projection qualification](projection-parity.md) records independent native/structured
MLflow readback for the listed representations. The authoritative contracts remain `architecture.md`,
`requirements.md`, and `integration-parity.md`. Machine-readable source-format
declarations and counter-scope limitations are in `sources/schemas.py`.

## Qualified source representations

| Harness | Source facts and effective-history rule | Explicit limits |
| --- | --- | --- |
| OpenCode | Existing current/legacy/mixed SQLite fixtures; V2 header/record precedence; candidate selection and all required-relative extraction in the same read transaction | Incomplete authoritative V2, missing/duplicate session identities and distinct physical rows sharing a required logical ID fail; repeated projections of the same row still deduplicate; no whole-database backup or credential-table queries |
| Claude | Session-ID JSONL envelopes, including metadata preludes before cwd; UUID/parentUuid paths; message.id invocation identities retained across split fragments; tool-result/injected/sidechain records do not open primary turns | Earliest attributable non-null session facts supply creation/header metadata; identity-less prefixes cannot supply creation time; a metadata-only session can use its standard UUID filename, explicitly marked as filename identity; missing creation is not inferred from update time |
| Codex | session_meta/payload rollout and the explicit-session-ID export fixture; response_item messages/tools; event_msg prompt observations are not duplicate prompts; historical Git/model-provider facts retained | Known rollback/rewind variants fail with unsupported_replay; last/cumulative token observations do not prove per-call totals |
| Pi | Version 3 session headers and parentId entry trees; last persisted entry is the resume head; indexed cycle/missing/forward-parent validation | Versions 1/2 are rejected; compaction/context-edit records remain historical evidence and do not invent prompts or erase historical turns; no official-plugin parity claim |
| Gemini | Partial metadata JSONL; repeated-ID replacement preserves insertion order; content/tool-result patches, remove/order, rewind and `$set.messages` checkpoints; plain and `$set` metadata refresh | Monolithic legacy JSON and creation-time/session-kind/project-identity-changing metadata updates are rejected; live OTLP-only timing/cost/user data is not fabricated |

The four file adapters accept `.jsonl.gz` as well as plain JSONL, including Codex
archived rollouts. After safe header filters, competing eligible/unknown files
claiming the same native ID fail as ambiguous; timestamps, file size and filenames
are not authority. Provably excluded collisions do not block an unrelated plan;
required-relative collisions still fail. ZIP/TAR and automatic legacy migrations are not qualified. Malformed,
duplicate-key, non-JSON-constant, oversized, or unsafe-boundary file records fail
explicitly. Existing SQLite tagged malformed/duplicate/non-JSON preservation is
retained. Unknown **permitted, well-formed** fields/records survive in the spool.

## Discovery, identities and stable views

- Default discovery covers all available adapter roots, or only the requested
  harness's roots. Codex scans `sessions` and `archived_sessions`, not global
  `history.jsonl`/configuration. Gemini scans project `chats` directories, not
  runtime logs. All roots freeze before the first target operation.
- Overrides: `XDG_DATA_HOME/opencode/opencode.db`,
  `CLAUDE_CONFIG_DIR/projects`, `CODEX_HOME/{sessions,archived_sessions}`,
  `PI_CODING_AGENT_DIR/sessions` or `PI_CODING_AGENT_SESSION_DIR`, and
  `GEMINI_CLI_HOME/{.gemini/tmp,.cache/.gemini/tmp}`. Gemini's override denotes
  a **home**, not the `.gemini` directory. Tildes/relative paths normalize
  lexically. Explicit file and default-root discovery share the owning-root
  namespace, including custom configured roots and Codex archives.
  Explicit primary files use the known owning-store header inventory for required
  relatives, with that file as the implicit direct-match seed. They do not read
  unrelated sibling bodies. A standalone subagent file is not a primary source;
  select its owning session/root instead.
- Namespaces use the versioned normalized-owning-root hash; no qualified persisted
  store/profile-ID schema is assumed. Moving the store changes this fallback.
  For arbitrary nonstandard nested files, configure the owning root using its
  harness override rather than guessing an undiscoverable profile boundary.
- File header discovery lexically skips transcript/credential values in mixed
  envelopes; it decodes only allowlisted metadata. Claude mode records bearing a
  session ID do not end discovery: supported prefixes include `permission-mode`,
  `atis-latch`, `worktree-state`, `tools-used`, `last-prompt`, `cost-state`,
  file-history/queue/summary/system/progress and `fork-context-ref`. For directory
  selection it continues through known conversation envelopes until cwd is known
  or EOF leaves it unknown. Unrecognized selected header variants fail explicitly.
  Inventory persists only locators, native relationships, numeric creation time,
  directory match booleans and validation hashes; selected header values are
  reread, hash-validated and redacted before snapshot persistence. Only header-eligible
  candidates undergo full record decoding. Content filters run before relative
  freezing; closure includes ancestors and descendants of direct matches, never
  siblings reached through an ancestor. Missing creation under a time bound
  emits a safe warning; source and CLI-bound comparisons retain nanoseconds,
  including offset remainders, without datetime rounding.
- Gemini cwd lookup, when absent from the header, supports the persisted
  `.gemini/projects.json` schema `{"projects":{"/absolute/path":"allocated-group"}}`
  beside the owning `tmp` directory. Both the allocated group in
  `tmp/<group>/chats` and SHA-256 of the exact registered path must match the
  header's `projectHash`. No basename inference, directory rebasing, changed
  default-root interpretation or newest-file preference is used. Missing mappings
  remain unknown; malformed/ambiguous/symlinked registries fail with
  `unsupported_project_registry`. Registry input is bounded to 1 MiB before JSON
  decoding. Discovery and final selection use the same established cwd; supported
  OpenCode `directory`-or-`path` fallback likewise agrees at both selection seams.
- Each selected record/sidecar set receives identity/size/mtime and a digest of
  the **actual copied stream** compared with a second source read (also for gzip);
  inventory/header/relationship changes
  invalidate it. Construction retries at most twice within the invocation's
  shared retry/deadline budget. Consumer/target exceptions never trigger a live
  reread. Unrelated-session churn does not invalidate selected files. Newly
  associated sidecars invalidate the selected view; new unrelated sessions after
  its inventory are left to a later invocation.
- Claude `subagents/agent-*.jsonl`, inline sidechains and `tool-results/*.txt`,
  and Gemini's parent-session subagent directory are retained. Parent UUID,
  recorded parent tool-use ID, or recorded toolUseResult.agentId establishes turn
  attribution. Unknown/ambiguous attribution remains session-level evidence with
  a warning, rather than being assigned to the last turn. Subagent assistant
  results cannot replace a primary root output.
  Claude fork-context references are recognized metadata markers, not permission
  to open arbitrary reference paths. A selected sidecar's later session ID must
  agree with its owning primary, and all later explicit identity fields must
  agree with that ID, including presence with invalid empty/null/false/zero values.
  Intermediate Claude session sidecar directories cannot be symlinks.

## Actual storage and access bounds

`snapshot.payloads` contains permitted header/record envelopes and raw-field
projections. SQLite stores numeric byte offsets/lengths, physical versus derived
or effective occurrence kinds, logical entities/parents, invocation/call links,
active-path ordinals, model identities and small summaries. Original occurrences
remain addressable separately through `iter_original_records`; generated Gemini
effective projections reference **source-stable** occurrence IDs, not SQLite
rowids. Outbound hashes exclude temporary offsets and locators.

Codex model-context and cumulative-usage auxiliary observations also pass through
the invocation's redaction policy before SQLite persistence. Pre-redaction model
selection evidence is captured from the original turn-context observation.

`NormalizedSnapshotAdapter.occurrences()` exposes immutable content-free
`OccurrenceRef` values. OpenCode parent/model-container links are table-qualified
in this index, preserving its existing outbound record representation; multi-call
records use the call-fragment index. Summaries distinguish physical, derived and
effective occurrence counts from logical entity counts.

Header relationship/selected-ID sets, legacy explicit/join deduplication and
fallback occurrence counters, instruction-reference sets, deferred discovery
state and plan-reader uniqueness checks are disk-backed. A turn cursor joins its
indexed occurrences directly to spool locators with an independent file handle;
it does not replay the session to rediscover boundaries. The legacy all-other-
records cursor uses an indexed anti-join, not another boundary scan.

The current file-record envelope is 32 MiB (no truncation), source payload spool
8 GiB, ID/turn indexes 1 GiB each, with finite SQLite caches. Standard JSON decoding
still requires one physical record/header value in memory. These per-store limits
also share the invocation-wide admission ledger covering target plans, pricing,
results, serialization and transfers; see [temporary-resources.md](temporary-resources.md).

Synthetic regressions in `tests/unit/source/test_metadata_discovery.py` cover
mode/cwd preludes, identity-less timestamps, fork-context sidecars, excluded
malformed-body guards, ancestor/descendant closure without siblings, registry
qualification, filter-before-collision, unresolved/required duplicate failure,
plain/`$set` metadata identity validation and redaction-before-persistence.
Regressions in `tests/unit/source/test_schema_spools.py` cover all of
these source paths, actual seek/read offsets, mixed-header decoding bounds,
same-stat edits/retries, sidecar attribution, independent selection fingerprints
and a doubled 150/300-entry tree with 16 KiB records (peak Python heap <8 MiB,
growth <2 MiB). Existing OpenCode source/contract/indexed-replay tests remain.

Current, legacy and mixed OpenCode paths order structural rowid/declared-key
locators before indexed payload reads in the same original transaction. Header,
message, part, ambiguous-parent part and todo ordering is covered, including
WITHOUT ROWID and optional-column fallback paths. Synthetic 40 × 2 MiB tests run
under the unchanged 64 MiB SQLite ceiling and 16 MiB temporary quota. Excluded
credential columns cannot become locator reads.

Default-root discovery uses one global schedule: present creation instants first
in ascending order, then native session ID, harness and namespace; missing-time
sessions follow timestamped sessions with the same tie keys. Equal/missing-time
tests check dry-run report and execution trace-create order across roots.
The current strengthened reference gate verifies all 307 sessions and exact live
trace cardinality across four phases. Full timings and separate memory/tree/replay
evidence are in [compatibility.md](compatibility.md#synthetic-performance-evidence).
## Resource envelope

The invocation's [temporary resource profile](temporary-resources.md) applies to
every qualified adapter. Source payloads and indexes share one quota; original
inputs are not counted as temporary copies. Merged session/header facts are
limited to 8 MiB and fail with `source_metadata_too_large` if cumulative control
or sidecar metadata exceeds that envelope. Filesystem traversal is disk-ordered
and capped at 32 directory levels; this is not a limit on persisted Pi ancestry
depth. Large records retain their separate record/native payload envelopes.
