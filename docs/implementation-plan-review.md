# Implementation-plan review — 2026-11-05

**Reviewed:** `docs/implementation-plan.md` against `docs/architecture.md`, `docs/requirements.md`, packaging, and CI. Three independent OpenCode workers used `enmaas/rits/deepseek-ai/DeepSeek-V4.1-Flash-1017`. This report records plan review, not implementation or compatibility evidence.

## Outcome

The plan covers the governing design. Clarifications were added for provisional versus final profile gates, disposable-target ownership/backend configuration, exact lookup queries, SDK retry control, lock bounds, source locator/symlink handling, unavailable legacy evidence, and the interim write-disabled CLI. The full P01–P18 scope remains intact.

The user-requested contract follow-up below adds explicit user-identity redaction
and harness-valued source mapping to the requirements, architecture, README, and
implementation gates. Runtime verification remains pending with P01–P18.

Several findings overlap across lenses. Several others restate an existing requirement or propose a behavior change the contract does not require; those are retained with their disposition rather than silently counted as gaps.

## Contract follow-up — 2026-11-05

These changes record the user's updated contract and its implementation coverage.

| ID | Required behavior | Plan coverage and acceptance evidence |
| --- | --- | --- |
| C01 | `--redact` masks present user identity as `[REDACTED]`, even without a secret-pattern match, throughout session/trace metadata and normalized/raw/escaped artifacts. Includes `mlflow.trace.user`, populated `mlflow.user`, and equivalent namespaced user fields; missing users stay absent and SDK/OS defaults cannot reintroduce identity. Preserve the structural role `user` and reconciliation IDs. | P07 defines projection/counting; P08 controls and verifies native/SDK fields; P09/P12 cover verified replacement when immutable user metadata differs; P17 proves per-harness masking, absent-user/default cases, role/ID preservation, and unchanged reruns. P18 documents the behavior. |
| C02 | Session/trace `source`, including native `mlflow.source.name`, equals the canonical harness name: `opencode`, `claude`, `codex`, `pi`, or `gemini`. Locator, namespace/fingerprint, importer, model/provider, and Git provenance remain separate. | P06 defines the projection; P08 writes and reads back native metadata; P09/P12 cover correction of immutable legacy source values; P17 verifies all harnesses, default/explicit selection, and both privacy modes. P18 documents the distinction from `--source PATH`. |
| C03 | `mlflow-import` is the distribution name and sole importer console command, targeting `mlflow_import.cli:main`. No alternate importer command or Python-package shim is provided. Persisted legacy wire identifiers remain versioned data contracts. | P01 requires exactly one installed console entry point; P12 exercises that command; P18 and the completion checklist verify the same packaging contract. This user decision supersedes earlier alias-preservation guidance. |

These gates refine A17's metadata coverage and E04's privacy-transition cases.
User/source values stored in immutable request metadata participate in turn
fingerprints and full verification; an overwrite must verify a replacement before
retiring the old trace. A mutable-tag-only refresh cannot satisfy either rule
while an incorrect or unredacted immutable copy remains visible.

## Adversarial findings and disposition

| ID | Location | Trigger / consequence | Suggested guard | Disposition |
| --- | --- | --- | --- | --- |
| A01 | P02/P08 dependency | Final qualification references later recovery work; a literal completion rule could deadlock. | Separate provisional API acceptance from final support closure. | Clarified P02 and execution rules; final closure stays P17. |
| A02 | P02 server fixture | Unspecified backend/launch flags could test the wrong trace semantics. | Pin backend, artifacts, loopback and port. | Added explicit SQLite-backed owned fixture configuration. |
| A03 | P02 file artifacts | Relative/client-local paths could hide artifact-location mistakes. | Read actual absolute location and vary client/server CWD. | Added; same-host repository APIs remain the intended binding, not a new server upload API. |
| A04 | P02/P08/P09 searches | Generic pagination evidence might omit required conflict/recovery filters. | Prove exact escaped queries. | Added profile evidence requirement. |
| A05 | P11 ambiguous create | Attempt/turn lookup might be unsearchable. | Contract-test request identity lookup. | Overlaps A04; creation identity already required by architecture §8. |
| A06 | P08 retries | SDK internal retries could escape the outer budget. | Control internal retries/timeouts and prove consumption. | Added explicit configuration/evidence requirement. |
| A07 | P11 lock | Different lock locations or unlimited waiting could defeat exclusion. | Deterministic protected location and finite wait. | Added. |
| A08 | P06/P10 native ID | Multiple qualified sessions share a native ID. | Reviewer proposed uniqueness failure. | Proposal rejected: a native-ID filter applies independently to each qualified identity. Added explicit selection semantics; legacy alias ambiguity still fails. |
| A09 | P06 directories | Relative/dot-segment normalization might be missed. | Make lexical absolute normalization explicit. | Clarified; already required by requirements §6.2. |
| A10 | P05 migration roots | Override and in-memory behavior might be misinterpreted. | Separate global-data resolver and precedence tests. | Already explicit in architecture §12 and P05; retain as implementation cases. |
| A11 | P03 legacy fixtures | No current source/tests establish the old wire schema. | Name authorized historical evidence or disable decoder. | Clarified historical-contract/export source and disabled behavior. |
| A12 | P04 source links | Recursive source links could escape or loop. | No-follow discovery/freezing and locator tests. | Added. |
| A13 | P11 staging tags | Exact staging keys/promotion sequence are not named. | Define and test them. | Implementation-level schema decision within P03/P08/P11; architecture already requires staged identity and verified promotion. |
| A14 | P11 invalid state | Corrupt ready state might not reconstruct authorized IDs. | Duplicate trusted evidence or halt. | Safe halt is already explicitly allowed; no new journal/redundancy requirement introduced. |
| A15 | P14 compression | Some codecs are unavailable in Python 3.11 stdlib. | Scope codecs or authorize a dependency. | No new dependency authorized. Only evidenced decoders may be advertised; unsupported encodings must fail explicitly. Actual upstream codec evidence remains a P14 gate. |
| A16 | P10/P17 matrix | “Complete matrix” could be interpreted loosely. | Enumerate named decision-table scenarios. | Implementation test-design responsibility; retain requirement-to-evidence mapping and use explicit scenario manifests. |
| A17 | P03/adapters metadata | Normalized field coverage could be incomplete. | Fixture-map each required field to a value or explicit absence. | Required by shared conformance and fidelity acceptance; retain as test-design guidance. |
| A18 | P10 tracking URI | Credential-bearing URI might leak. | Reject or sanitize before diagnostics. | Already covered by P07 diagnostics and requirements §12; do not echo URI/raw exceptions. |
| A19 | P10 interim write path | Gate behavior before P12 is undefined. | Exit 2 before lock/creation. | Added `write_path_not_enabled` behavior. |
| A20 | P04/P07/P08/P11 bounds | Unrecorded limits cannot be asserted. | Record finite defaults and test exact consumption. | Added profile/operations evidence requirement; measured payload limits remain profile-specific. |
| A21 | Adapter source locators | Explicit file/directory/root forms might diverge. | Per-adapter locator/namespace conformance. | Added to P04; do not claim unsupported locator forms. |
| A22 | P01 packaging | Entry point and first-party names might remain stale or expose an extra importer command. | Verify exactly one `mlflow-import` entry point and `mlflow_import` tooling names. | Updated for C03; the earlier dual-command guard is superseded. Retain installed-command checks. |

## Edge-case findings and disposition

| ID | Location | Trigger / consequence | Suggested guard | Disposition |
| --- | --- | --- | --- | --- |
| E01 | P10 | Non-dry-run before writer qualification could partially write. | Fail before lock/creation. | Added; overlaps A19. |
| E02 | P06/P10 | Same native ID across namespaces. | Reviewer proposed ambiguity failure. | Rejected for the same reason as A08; test independent qualified matches. |
| E03 | P10 defaults | Missing environment-selected root could hide misconfiguration. | Treat every harness/environment root as explicit. | Not adopted as a blanket rule: requirements distinguish optional defaults from explicit sources. Record each adapter override's availability/error semantics and always reject missing `--source`. |
| E04 | P09/P12 privacy | Privacy changes under conflict policies could reuse verbatim content. | Replace for overwrite; preserve completed copies for ignore/duplicate. | Already explicit in architecture §§4, 7–8 and P12 privacy-change coverage. |
| E05 | P11 | Contended lock might wait forever. | Finite wait charged to shared budget. | Added; overlaps A07/A20. |
| E06 | Test fixtures | Ambient tracking URI could redirect destructive tests. | Assert fixture-owned target/experiment/artifact scope. | Added to P02; reinforces existing §4 isolation rule. |

## Structure lens

Word metrics reported 4,893 words for the original plan. The reader is an implementation agent. Content decisions remain governed by requirements and architecture.

| Finding | Suggested structural change | Disposition |
| --- | --- | --- |
| P02 provisional/final acceptance appears combined. | Split gate labels. | Applied; overlaps A01. |
| Dependency closure rule is implicit. | Name acceptance evidence required before dependents. | Applied with provisional P02 exception. |
| “First vertical slice” could obscure parallel readiness. | Explain dependency ordering. | Added disjoint ownership/handoff rules; original P13–P16 parallel statement retained. |
| P02/P10/P18 share documentation paths. | Separate creation from final update. | Applied in execution rules. |
| Some Files entries describe helpers rather than paths. | Normalize paths or rename heading. | Deferred; implementation may keep cohesive helpers in existing responsibility groups. |
| Verification commands appear after work packages. | Move or cross-link early. | Added early §4 pointer. |
| Milestones repeat package acceptance. | Condense. | Retained: cross-package gates provide useful execution context. |
| Traceability evidence resembles architecture matrix. | Replace with references. | Retained: local traceability aids worker handoffs. |
| Console-entry-point rule repeats. | Remove checklist repetition. | Retained as acceptance reinforcement, updated for C03's sole-command rule. |
| Traceability rows overlap package sets. | Partition or explain. | Retained: capability coverage is intentionally many-to-many. |
| P13–P16 parallel statement looks redundant. | Preserve. | Preserved. |
| Completion checklist looks redundant. | Preserve as release omission check. | Preserved. |

## Canonical behavioral findings

```json
[
  {"lens":"adversarial","location":"P02/P08","trigger_condition":"Final qualification is read as the prerequisite for P08","guard_snippet":"Separate provisional API-profile acceptance and P17 final closure","potential_consequence":"Dependency deadlock"},
  {"lens":"adversarial","location":"P02","trigger_condition":"Disposable server backend/launch configuration is unspecified","guard_snippet":"Pin SQLite backend, absolute artifacts, loopback and allocated port","potential_consequence":"Profile tests prove different persistence semantics"},
  {"lens":"adversarial","location":"P02 artifacts","trigger_condition":"Client and server resolve relative artifact paths differently","guard_snippet":"Read actual absolute artifact location; vary working directories","potential_consequence":"Bundle data cannot be verified"},
  {"lens":"adversarial","location":"P02/P08/P09","trigger_condition":"Required conflict/recovery filter queries are not qualified","guard_snippet":"Exercise exact escaped paginated queries","potential_consequence":"Missed conflicts or unbounded scans"},
  {"lens":"adversarial","location":"P11","trigger_condition":"Ambiguous create identity is not remotely searchable","guard_snippet":"Prove attempt/instance/turn lookup","potential_consequence":"Duplicate creates"},
  {"lens":"adversarial","location":"P08","trigger_condition":"SDK retries outlive outer budget","guard_snippet":"Control request retries/timeouts and test consumption","potential_consequence":"Unbounded operations or ambiguous retries"},
  {"lens":"adversarial","location":"P11 lock","trigger_condition":"Lock path/wait policy is not deterministic and finite","guard_snippet":"Protected per-user location and shared deadline","potential_consequence":"Concurrent writers or indefinite wait"},
  {"lens":"adversarial","location":"P06/P10","trigger_condition":"Same native ID exists in different qualified identities","guard_snippet":"Reviewer proposed uniqueness failure; triage instead preserves independent qualified matches","potential_consequence":"Incorrect identity selection if merged"},
  {"lens":"adversarial","location":"P06 directory","trigger_condition":"Relative/dot paths are not normalized","guard_snippet":"Lexical absolute component normalization","potential_consequence":"Incorrect filter matches"},
  {"lens":"adversarial","location":"P05 migration","trigger_condition":"Migration root is derived from an overridden database path","guard_snippet":"Use separate global-data root and explicit precedence","potential_consequence":"Wrong migration source"},
  {"lens":"adversarial","location":"P03 legacy","trigger_condition":"Legacy schema evidence is absent","guard_snippet":"Capture authorized historical fixtures or disable decoder","potential_consequence":"Guessed ownership/result decoding"},
  {"lens":"adversarial","location":"P04 source links","trigger_condition":"Discovery follows loops or out-of-root links","guard_snippet":"No-follow source traversal and tests","potential_consequence":"Unintended reads or nontermination"},
  {"lens":"adversarial","location":"P11 staged tags","trigger_condition":"Staging/promotion keys are unspecified","guard_snippet":"Define versioned keys and test promotion failures","potential_consequence":"Staged work confused with completion"},
  {"lens":"adversarial","location":"P11 invalid states","trigger_condition":"Corrupt ready/complete object obscures retirement authorization","guard_snippet":"Reconstruct from verified evidence or halt","potential_consequence":"Unsafe retirement if guessed"},
  {"lens":"adversarial","location":"P14 codecs","trigger_condition":"Upstream compression lacks an allowed decoder","guard_snippet":"Advertise only qualified decoders; explicitly reject others","potential_consequence":"Undeclared dependency or missing sessions"},
  {"lens":"adversarial","location":"P10/P17 scenarios","trigger_condition":"Complete matrix lacks an explicit scenario manifest","guard_snippet":"Enumerate dimensions and named cases","potential_consequence":"Unproved combinations"},
  {"lens":"adversarial","location":"P03/adapters","trigger_condition":"Required normalized metadata is omitted","guard_snippet":"Map fixture values or explicit absence for required fields","potential_consequence":"Incomplete fidelity"},
  {"lens":"adversarial","location":"P10 tracking URI","trigger_condition":"Credential-bearing URI reaches diagnostics","guard_snippet":"Sanitize or reject before output","potential_consequence":"Credential leakage"},
  {"lens":"adversarial","location":"P10 write gate","trigger_condition":"Non-dry-run occurs before P12","guard_snippet":"Exit 2 with write_path_not_enabled before lock/creation","potential_consequence":"Unqualified writes or misleading success"},
  {"lens":"adversarial","location":"Resource/retry bounds","trigger_condition":"Finite defaults are unrecorded","guard_snippet":"Document and assert exact configured limits","potential_consequence":"Unverifiable boundedness"},
  {"lens":"adversarial","location":"Adapter locator forms","trigger_condition":"Explicit file/root forms resolve different namespaces","guard_snippet":"Per-adapter locator and identity conformance","potential_consequence":"Duplicate source identities"},
  {"lens":"adversarial","location":"P01 packaging","trigger_condition":"Console target/tool names remain stale or an extra importer command is registered","guard_snippet":"Require exactly one mlflow-import entry point targeting mlflow_import.cli:main and aligned first-party tooling","potential_consequence":"Broken or unexpected installed importer commands"},
  {"lens":"edge-case-hunter","location":"P10","trigger_condition":"Write invocation before qualified writer exists","guard_snippet":"Fail before lock/experiment creation","potential_consequence":"Partial writes"},
  {"lens":"edge-case-hunter","location":"P06/P10","trigger_condition":"Native ID collides across qualified namespaces","guard_snippet":"Keep qualified identities independent; do not merge","potential_consequence":"Wrong-session reconciliation if conflated"},
  {"lens":"edge-case-hunter","location":"Default discovery","trigger_condition":"Configured root is unavailable","guard_snippet":"Define override errors; reject missing explicit source","potential_consequence":"Misleading empty import"},
  {"lens":"edge-case-hunter","location":"P09/P12 privacy","trigger_condition":"Privacy mode changes under a conflict policy","guard_snippet":"Apply recorded overwrite/ignore/duplicate semantics without reusing wrong payload","potential_consequence":"Wrong current privacy projection"},
  {"lens":"edge-case-hunter","location":"P11 lock","trigger_condition":"Another writer holds the lock","guard_snippet":"Bounded wait charged to shared deadline","potential_consequence":"Indefinite blocking"},
  {"lens":"edge-case-hunter","location":"Disposable fixtures","trigger_condition":"Ambient target configuration redirects tests","guard_snippet":"Assert owned URI, experiment and artifact scope","potential_consequence":"Destructive tests reach user data"}
]
```
