# Fixture-qualified content and MLflow projection

This support document records the qualified projection and final evidence. The authoritative
requirements, architecture and integration baseline remain unchanged. The
implementation is `mapping-v9`, `artifact-v4`, `accounting-v2`, with coverage evidence version
`integration-baseline-2026-11-13-v1`.

## Qualified representations and independent readback

`tests/integration/test_adapter_parity.py` asserts literal expected values through
public MLflow retrieval, rather than using the production mapper as its oracle:

| Adapter/version | Qualified target evidence |
| --- | --- |
| `opencode-schema-catalog-v1` | Current, legacy and mixed SQLite; authoritative V2 precedence; part-level usage/cost; recorded zeros; reasoning totals; structured tools; exact parent/user/Git metadata, native timing/name/previews/status and USD costs |
| `claude-persisted-v2` | Plain/gzip JSONL, both privacy modes; split invocation messages and coalesced root output; repeated usage; cache subsets and the native extended creation bucket; tools/skills/thinking; recorded nested subagent parentage |
| `codex-persisted-v2` | Rollout `session_meta` and explicit-session-ID exports; gzip and both privacy modes; context before/after the initiating prompt; historical Git/user/request/parent fields; paired tools; last observations versus cumulative deltas across turns |
| `pi-persisted-v2` | Pi v3 JSONL/gzip and both privacy modes; the source-qualified persisted path; structured messages/tools/thinking; input cache subsets and reasoning already included in output |
| `gemini-persisted-v2` | Partial-metadata JSONL/gzip and both privacy modes; effective repeated-ID/patch/rewind accounting; thinking and structured tools; recorded subagent/tool hierarchy |

All four file formats explicitly withstand a conflicting raw `source` value;
the native source and normalized session artifact retain the canonical harness.
Unknown source facts remain unavailable. These fixtures do not certify every
combination of source updates, lifecycle events and failures. Pi still has no
verified official-plugin baseline.

## Public pinned target binding

Production historical imports use public `mlflow.entities.TraceInfo`,
`Span.from_dict`, and `mlflow.tracing.client.TracingClient.start_trace/log_spans`.
This pinned object-ingestion binding is checked at preflight and exercised against
the disposable 3.17.2 SQLite/file-artifact profile. No backend tables, private
client methods, trace-storage paths, or run containers are used.

Native info is published with the attempt identity before span upload, then
finalized through the same public API after upload. This matters because the
pinned server can otherwise derive trace timing from earlier child events on a
trace without usage observations. Explicit reference previews are preserved. Full remote
root/child values, native timing/state/session/request/name/usage/cost and exact
immutable metadata are verified; ready-state references also retain native-field
expectations. Ambiguous writes use the existing durable attempt recovery.
Absent historical native request/response previews are physically represented by
an explicit empty string. MLflow 3.17.2 otherwise autofills NULL previews from
structured span content, and finalizing with NULL cannot clear that generated
text. The gateway applies this qualifier at creation, finalization and strict
reuse verification; mapped source previews remain `None` and source artifacts,
fingerprints and digests retain their existing meaning. Nonempty previews must
still match exactly. Older NULL or autofilled candidates fail this physical
contract. Prepared candidates that failed before turn-map publication are
replaced through normal reconciliation; verified empty-string winners are
reused. Local evidence is in
`tests/integration/test_native_preview_transport.py`.
Historical spans preserve `UNSET` child execution outcomes; a missing/unknown
terminal turn outcome is represented as a non-OK trace. Recorded tool-state
execution intervals are measured; other recorded timestamp intervals are derived,
and API-required container fallbacks are explicitly synthetic.

MLflow's storage translator otherwise calculates costs from a moving model table
when a model/usage span lacks the cost convention. Explicit empty cost conventions
suppress that estimator; final public `TraceInfo` publication sets unavailable
native cost components to null. OpenCode always projects native cache/cost as NULL
to match `@mlflow/opencode` 0.4.0; richer observed accounting remains canonical.
Empty conventions alone caused native zeros in
the initial experiment and are **not** accepted as evidence of zero cost. Unknown
currency/rates remain native-null; for other harnesses recorded zeros remain numeric zero, including
total-only persisted costs without invented input/output breakdowns.

The detached-span API remains exercised by the older compatibility/failure
seams. It alone does not establish the production native-field contract.
`update_current_trace(client_request_id=...)` on a detached span did not persist
that native field in the tested profile and is not used as a substitute.

The native request ID limit is 50 characters. Only attributable source request
IDs populate it; absent IDs stay NULL and operation keys live in initial tags.
Missing user/Git facts remain absent; direct historical ingestion does not merge
SDK/OS defaults. Historical facts and importer ownership are creation tags, with
only optional bare `parent_session_id` as arbitrary physical request metadata.
Native session/usage/cost keys are promoted by MLflow and `infoFinalized` is
server-owned. Unsupported request lengths fail during read-only planning.

### OpenCode reference info

`tests/integration/test_opencode_reference_projection.py` reproduces the observed
reference values locally: start `1794580115471`, duration `15966` despite a user
row update five milliseconds later, input `73202`, output `148`, total `73380`
(including reasoning `30`). Canonical explicit totals `75104 + 75204 = 150308`
and cache read `76928` remain in the retained accounting. Native cache and all
cost columns are NULL even for recorded cost zero. The root span is
`opencode_conversation`; native trace name is `opencode` via the `mlflow.traceName`
tag, and absent client request ID is NULL.

OpenCode child spans also reproduce the plugin's exact tree and payload shapes:
flat `llm_call` (`LLM`) and `tool_<tool>` (`TOOL`) children of
`opencode_conversation`, no synthetic reasoning/text/step/patch children, no
lifecycle events, and no nesting under `llm_call`. LLM inputs carry the
plugin's cumulative reconstructed conversation; LLM outputs use the
plugin's assistant-message shape; tool inputs/outputs/attributes and measured
tool intervals match the plugin. Per-span source/accounting provenance remains
on the trace root and canonical artifacts. A turn above the plugin's
1000-message fetch window uses the importer's bounded linear activity
projection because the plugin has no reference behavior for that size.
Native session and parent IDs are bare, while qualified source identity stays
immutable in tags/artifacts. Previews select the last nonempty user/assistant
text, newline-join parts (including synthetic user text), omit whitespace-only
assistant parts, and use the plugin's 1000 UTF-16-unit limit. SQL readback checks
the physical metadata whitelist; a rerun reuses the same trace without mutation.

A UTF-16 boundary that splits a surrogate pair is converted to U+FFFD before
protobuf transport (the well-formed UTF-8 behavior of JavaScript `TextEncoder`).
Complete structured text remains unchanged. Reasoning follows the same
parent-authoritative, missing-field step rollup as input/output. Semantic message
times use validated epoch milliseconds; malformed/boolean/out-of-range values
retain the adapter's normalized fallback. Native session/parent metadata is
limited to 250 characters, historical/qualified creation tags to 4096, and native
client request IDs to 50; the metadata limit does not constrain tag-only facts.

## Usage and costs

The shared projection coalesces invocations before summing counters. Message
observations take precedence over repeated part observations; parts fill missing
facts. Split messages retain their original occurrences and share a mapped model
span. A turn map includes occurrence/entity/invocation references and resulting
span IDs, including aliases from coalesced fragments and tool results.
Multiple OpenCode `step-finish` observations become distinct LLM steps beneath
an assistant AGENT rollup. Parent observations take precedence at their recorded
scope; children are not added a second time. Pi cache-warm/compaction usage
operations retain their independent accounting scope. Model chat inputs retain
correlated tool results, and Gemini tool calls render in structured assistant
outputs. Derived turn-context inputs are labelled as such, not claimed to be a
complete copy of an inaccessible provider request.

### Linear observed conversation context

`conversation-delta-v1` replaces cumulative derived model-input histories. Within
each recorded execution scope, the first call receives the complete observed
prompt contributions available before that call. Each later call receives new
user/tool contributions and the immediately preceding assistant message once.
Pending contributions are consumed at each conversation model call; an old prompt, result or source
identity prefix is not copied into all later inputs. A contribution therefore has
constant total replication, independent of call count. Tool IDs and actual result
values remain correlated. Recorded subagent scopes have independent context chains.
Standalone Codex function calls are retained as new assistant tool-call contributions.
This policy applies to the shared activity projector and the legacy mapping fallback.
Nonconversation usage/compaction/branch-summary observations advance the context
provenance chain with unavailable inputs, but preserve pending user/tool values and
the last actual assistant for the next model call. Accounting observations do not
consume or replace conversation contributions.

Explicit recorded `input` fields, including null, empty containers and full recorded
request histories, retain their complete source value. They are labelled
`recorded-request-input` / `recorded-input-only`; that does not establish that an
entire provider request was captured. Derived context is labelled
`derived-observed-conversation` with `provider_request="unknown"`. A call without
observed context has `inputs=None`, an empty delta and an explicit empty descriptor.
Accounting-only steps may likewise report unavailable input rather than inventing it.

Each invocation's `mlflow_import.input_context` attribute is a versioned descriptor
linking qualified session, logical turn, recorded scope, invocation and preceding
context. Its delta references effective occurrence/entity/source identities and
message positions; coalesced fragments reference all contributing occurrences.
Gemini patches/checkpoints reference their effective identities, with physical
originals retained in the full session artifact. Context IDs depend on logical
turn-local identities, never a session snapshot hash, artifact location or turn
index. Updating a sibling turn does not change those IDs.

New `mlflow-import.turn-map.v2` objects persist the same descriptors in a linear
`input_contexts` index; see [result-schema.md](result-schema.md#turn-map-input-context-index).
Full source/session artifacts and complete root values remain authoritative and
are not compacted. The raw session schema stays `mlflow-import.session.v1` and
accounting stays `accounting-v2`. Projection/artifact version inputs change render
fingerprints without changing source fingerprints or logical turn keys. Historical
v1 maps remain readable for pricing provenance; that alone grants no current
projection reuse or in-flight attempt migration authority.

Activity graph construction calls the shared pure `SpanBudget.account_spec()`
before retaining each child, reserving a root span slot. This bounds mapped bytes
and span count before the complete graph exists. The final transport admission
also includes the root and wire/control overhead. Local regression evidence is in
`tests/unit/content/test_input_context.py` (10/100/1000 calls, a 2105-child shape,
early span/byte failures) and `test_input_context_sources.py` (six source cases,
both privacy modes, retained artifacts and local map publication).

- Claude input includes cache-read and cache-creation counts once. Extended
  creation is a subset, not an additional token count.
- Pi normalized input includes cache counts once. Its output already includes
  reasoning, so reasoning is not added again.
- OpenCode canonical input/output remain source-defined; explicit source totals
  win in artifacts. Native reference total is input + output + reasoning, with
  available cache counters and available persisted USD costs published natively;
  unavailable cache lifetimes and cost breakdowns remain NULL.
- Gemini records `candidatesTokenCount` as `tokens.output` and
  `thoughtsTokenCount` separately; normalized generated output includes both.
  Source totals remain authoritative. Evidence: pinned upstream
  [recordMessageTokens implementation](https://github.com/google-gemini/gemini-cli/blob/b9776560ea5710bc673891499aaec68c737d160a/packages/core/src/services/chatRecordingService.ts).
- Codex cumulative counters are differenced using disk-indexed preceding
  observations, with initial/session and reset provenance retained. Missing
  preceding fields are not silently treated as zero. A first cumulative reading
  in a later turn does not charge that turn for earlier turns. A `last_token_usage`
  observation remains turn-scoped and is never replicated onto each LLM span;
  it does not prove complete turn or per-call usage.

Persisted costs win at their recorded scope. `origin` (`persisted` or
`importer-derived`) is separate from `method` (`estimated`, `provider-billed`, or
`unknown`). A persisted estimate remains an estimate. Original amounts and units
remain in the bundle. Non-USD source costs need an explicit positive, versioned
USD conversion basis; unsupported currency/conversion stays unavailable.

Native cost aggregation sums **observed** amounts, as MLflow does; it does not
allocate a missing cost or fabricate zero. Provenance records completeness per
cost component and missing invocations. A partial sum, or a priced Codex last
observation, does not establish the full-turn cost: coverage diagnoses that gap.
Persisted breakdowns are never guessed from a source total.

### Bundled estimate basis

`bundled-api-usd-2025-08-v1` uses Decimal arithmetic (precision 50) and USD per
million tokens. `bundled-model-map-v1` records exact qualified IDs and the explicit
bare-ID aliases for this small table:

| Model | Input | Output | Cache read |
| --- | ---: | ---: | ---: |
| `openai/gpt-4o`, `openai/gpt-4o-2024-08-06` | 2.50 | 10 | 1.25 |
| `openai/gpt-5` | 1.25 | 10 | 0.125 |
| `anthropic/claude-sonnet-4-20250514` | 3 | 15 | 0.30 |

Claude's qualified creation rates are 3.75 (default TTL) and 6 (extended 1-hour
subset); the base-context estimate is limited to 200,000 input tokens. Pi's
unspecified cache-write lifetime is not assigned Claude's TTL. OpenCode cached/
separately reasoned **estimates** remain unavailable where the bundled billing
scope is not qualified; its persisted USD costs/cache counts remain in canonical
accounting, while the native reference cache/cost projection remains NULL.
Unknown models/tiering/rates yield a coverage gap. These are API list-price
estimates, never actual subscription spend or proof of provider billing. Applied
rates, model mapping, units, source URLs, inputs and calculation identity are
stored with each invocation, so reproduction requires no network lookup.
Unspecified Claude creation TTL is likewise unavailable unless source lifetime
counters establish the split. Bare model-ID and normalized `/model` aliases are
explicit versioned aliases. Unsupported/invalid recorded pricing bases do not
silently fall back to newly installed rates.

Read-only planning freezes the verified instance-scoped donor selection in a
private SQLite pricing catalog before trace payload hashes/writes. Recovery uses
its prepared basis first. Overwrite prefers a verified completed canonical
instance, otherwise the lexicographically smallest completed duplicate instance;
within it, the smallest trace ID supplies each stable turn's applicable basis.
New invocations use the declared bundled basis. A fresh duplicate is independent.
The prepared state records the complete selected per-invocation basis before
trace calls. Changing installed rates or aliases cannot silently reprice existing
or recovering invocations; foreign traces never donate pricing provenance.
New invocation bases and negative/unavailable decisions are frozen during
read-only planning too, so even an installed-table change immediately after
`prepared` cannot alter the staged payload. After finishing a ready duplicate,
a changed overwrite reselects the canonical donor rather than inheriting that
duplicate's independent basis. The current recoverable turn-map envelope is
8 MiB; both fresh publication and donor reads enforce it explicitly, preserving
prior traces on failure rather than claiming an unrecoverable mapping succeeded.
Global preflight additionally calls the pure
`content.turn_map.validate_turn_map_admission(replay, request, mode)` on the already
admitted request snapshot. It streams the same canonical v2 builder as upload,
including complete accounting, contexts, records, aliases and identity maps, with
35-character trace-ID and 16-character span-ID reservations at every occurrence.
An oversized prospective map raises `turn_map_too_large` before target mutation;
upload retains its actual-byte check. Admission does not re-run the mapper or
pricing projection. Late artifact-placement nonces and attempt/session pointer
tags do not enter immutable map content.

## Coverage and limits

`sources/schemas.py` binds coverage declarations to source-format versions.
The shared projector combines them with actual missing observations. Bundles
store per-turn provenance; reports expose `coverage_unavailable_*` warnings and
`summary.coverage.unavailable_turn_counts`. Schema capability is not evidence
that a particular session retained a value. Exact live execution timing remains
unavailable when only timestamps exist. API-required fallback intervals are
explicitly `synthetic-container`; an absent/unknown terminal outcome is non-OK.
Declarations enumerate every baseline item with source fields, units, scope,
transformation, target, evidence version and limitations; OpenCode declares both
catalog generations. Actual root-usage completeness and per-call completeness are
separate. Partial/last observations and missing cost components cannot establish
full-turn coverage. An unknown source declaration never reports available parity.

The 100,000-record source/structure stress gate now uses indexed reopening record
views and streams plain activity, rather than retaining a transcript tuple and a
second span-spec graph. The measured heap bound excludes the target API's actual
one-turn payload retention. Whole-invocation source/target/report admission and
foreign tag/metadata alias policies are implemented and tested. Independent
quality/security/compatibility reviews and final acceptance checks are recorded
in [errors.md](errors.md); final verification passed with 1663 unit/contract/BDD
and 130 non-slow integration/profile checks, clean static checks, successful
package builds and independent qualified-profile acceptance.
Bound recovery tests cover the named durable boundaries without claiming every
possible external race or unsupported historical artifact format.

The source/structure stress check passed with 100,000 records, a 91.6 MiB artifact,
and **51.6 MiB peak traced heap** (SDK payloads excluded). The final three-case
memory selection passed in **310.73s**, also covering the 105,000-record streamed
case and oversized-root rejection. Production historical-gateway Pi/Gemini
32/64-turn tree/replay gates pass under 32 MiB quota and <32 MiB Python heap.
The strengthened linear OpenCode reference gate separately passed all four
invocations in **378.762984708 / 308.581364792 / 311.297854416 / 311.746874s**, with
307 sessions, 494/494/495/495 turns, exact live trace-ID cardinality and zero
unchanged mutation calls. See [compatibility.md](compatibility.md) for the complete
profile, accounting and main-agent suite/review evidence.
