# MLflow integration data parity

**Reviewed:** 2026-11-13  
**Status:** Target requirements and evidence baseline; not implementation certification.

This document supplements [requirements.md, Section 8.6](requirements.md#86-integration-data-parity).
`mlflow-import` must preserve at least the documented integration data that is
available in its supported local sources, plus additional permitted source data.
It must identify gaps where a live integration observes facts that were not
persisted. Full plugin parity cannot be claimed merely because raw JSON is saved.

## 1. Official documentation baseline

The [MLflow integration index](https://mlflow.org/docs/latest/genai/tracing/integrations/)
lists four of the five supported harnesses. Gemini uses built-in OTLP export;
the others below use MLflow plugins/hooks. Pi has no verified official listing.

| Harness and official guide | Explicitly documented capture | Importer obligation |
| --- | --- | --- |
| [OpenCode](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/opencode/) | Prompts/responses; file, shell, edit and other tools; conversation timing/duration per turn; input/output/total tokens; session grouping and current-system user. The guide also describes token/cost displays. | All persisted turns, structured model/tool activity, timing, usage, attributable user/session metadata, and available cost/cache data. A display capability alone does not establish a persisted cost field. |
| [Claude Code](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/claude_code/) | Prompts/responses; tool inputs/outputs; nested subagents and their steps; skills; per-call/session tokens and costs; step timing; working directory, user and model. | Reconstruct nested model/tool/agent activity, skill usage, associated subagent transcripts, source-attributable environment facts, usage and cost provenance. |
| [Codex CLI](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/codex/) | Prompts/responses; shell/file tools; input/output/total tokens; API-pricing cost estimates with a subscription-spend caveat; session/user metadata; Git repository provenance. Traces are exported after conversation turns. | Reconstruct all persisted turns, correlate tools/results, preserve available model/usage, provide qualified cost estimates or report the gap, and map historical user/Git fields when present. |
| [Gemini CLI](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/gemini_cli/) | Prompts/responses including thinking; LLM model, usage and costs; tool calls/results; session metadata. OTLP operation mapping includes `llm_call` → `LLM`, `tool_call` → `TOOL`, `agent_call` → `AGENT`. | Preserve equivalent structured conversation/model/tool/agent data from persisted records; explicitly identify telemetry-only fields missing from local storage. |
| PI | No official Pi-specific integration guide or plugin baseline was verified in the reviewed index/catalog. | Use the upstream persisted-session schema and the shared MLflow contract. Do not describe this as verified official-plugin parity. |

The guides describe different export mechanisms and do not establish a common
historical import algorithm. One trace per persisted logical user turn remains
the importer contract, regardless of idle, Stop, notify, or OTLP export boundaries.

## 2. Shared MLflow representation

- **Trace data:** Use the `trace_info` fields in requirements Section 8.0 for
  native identity, timing, status, previews, sessions, token usage and costs.
- **Span structure:** `AGENT` turn roots; `LLM` model calls; `TOOL` invocations
  with correlated results/errors; nested `AGENT` subagents where recorded.
  Preserve skills, structured content and thinking without introducing
  synthetic user turns. Plugin-specific span names are not required.
- **Chat rendering:** Supply structured messages, tool calls/results and
  supported model/provider attributes. Relevant MLflow conventions include
  `mlflow.llm.model`, `mlflow.chat.tokenUsage`, and `mlflow.llm.cost` at span
  scope, with trace aggregates exposed through the supported target API.
- **Metadata:** Preserve attributable session, working-directory, harness
  version, user, and Git facts where present. Git URL, commit and branch use
  applicable `mlflow.source.git.*` fields. Do not substitute the importer
  process's environment for historical facts.
- **Costs:** Preserve source costs first. Track persisted/importer-derived
  origin separately from estimated/provider-billed/unknown calculation method.
  Estimates require deterministic pricing provenance and populate supported
  span and native trace USD cost fields with associated provenance. Preserve
  original units and conversion basis; report missing data. See
  [MLflow token/cost guidance](https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/).
- **Coverage:** For each source version and baseline item, record direct,
  derived, or unavailable status, the source fields, transformation, target,
  and limitations in adapter documentation and canonical artifact provenance.
  Summarize unavailable baseline items in reports. Redaction and secret
  exclusions apply consistently to every representation.

### Canonical trace source

The importer always explicitly supplies these values:

| Harness | Native `mlflow.source.name` and normalized `source` |
| --- | --- |
| OpenCode | `opencode` |
| Claude Code | `claude` |
| Codex | `codex` |
| PI | `pi` |
| Gemini CLI | `gemini` |

These are importer requirements, not claims about plugin defaults. An OTLP
`service.name` such as `gemini-cli`, a source filename, raw source-record field,
SDK default, model/provider, or importer entry point must not replace them.
Keep source namespace, original provenance and Git metadata in separate fields.

## 3. Source-verified implementation qualifications

These observations clarify documentation promises; they do not certify every
installed plugin/source version. Adapters must validate the formats they support.

### OpenCode and Claude Code

Reference: MLflow revision `0bbd404a84408cf4e19ea489b953fd9159f6a159`, reported
by the reviewed published TypeScript integration packages, version 0.5.0.

- [OpenCode plugin](https://github.com/mlflow/mlflow/blob/0bbd404a84408cf4e19ea489b953fd9159f6a159/libs/typescript/integrations/opencode/src/index.ts)
  builds agent roots with model/tool children on idle. Its observed message
  fetch limit is 1,000 and processing state is in memory; neither limits offline
  historical import. Its usage total includes recorded reasoning in addition
  to input/output. Preserve source-defined totals rather than imposing one
  universal formula. Tool errors must remain distinguishable even if a plugin
  root output labels the conversation completed.
- [Claude trace builder](https://github.com/mlflow/mlflow/blob/0bbd404a84408cf4e19ea489b953fd9159f6a159/libs/typescript/integrations/claude-code/src/tracing.ts)
  and [transcript parser](https://github.com/mlflow/mlflow/blob/0bbd404a84408cf4e19ea489b953fd9159f6a159/libs/typescript/integrations/claude-code/src/transcript.ts)
  reconstruct model/tool spans and nested subagents from sidecars or correlated
  progress records. Coalesce assistant fragments sharing a message identity;
  discriminate actual prompts from tool results, compact summaries and skill
  injections. The reviewed hook runs on `Stop`; the guide's session-end wording
  is not a historical turn-boundary algorithm.
- [Claude usage normalization](https://github.com/mlflow/mlflow/blob/0bbd404a84408cf4e19ea489b953fd9159f6a159/libs/typescript/integrations/claude-code/src/_internal.ts)
  includes cache-read and cache-creation counts in normalized input tokens.
  Those separate cache counters are subsets, not extra tokens to add again.
  Its [pricing helper](https://github.com/mlflow/mlflow/blob/0bbd404a84408cf4e19ea489b953fd9159f6a159/libs/typescript/integrations/claude-code/src/pricing.ts)
  calculates estimates from model/usage; they are not proof of billed cost in
  a transcript. Reconstructed/padded plugin timing is not measured execution.

### Codex and Gemini CLI

MLflow reference revision: `8171679f34de004e2ec62b759b8591f5e089f15f`.
Gemini CLI reference revision: `b9776560ea5710bc673891499aaec68c737d160a`.

- [Codex trace builder](https://github.com/mlflow/mlflow/blob/8171679f34de004e2ec62b759b8591f5e089f15f/libs/typescript/integrations/codex/src/tracing.ts)
  and [transcript parser](https://github.com/mlflow/mlflow/blob/8171679f34de004e2ec62b759b8591f5e089f15f/libs/typescript/integrations/codex/src/transcript.ts)
  enrich notify events from rollout JSONL and select the last task interval.
  The inspected builder places the last observed `info.last_token_usage` on the
  root, despite the guide's per-call wording; it does not aggregate all calls
  in the turn. Determine actual counter scope before aggregating, and never
  replicate a turn-wide counter onto each model span.
  Model timing reconstructed from neighboring records must be labeled derived.
- [Gemini OTLP translation](https://github.com/mlflow/mlflow/blob/8171679f34de004e2ec62b759b8591f5e089f15f/mlflow/tracing/otel/translation/gemini_cli.py)
  consumes live model/tool/agent operations. Upstream
  [telemetry documentation](https://github.com/google-gemini/gemini-cli/blob/b9776560ea5710bc673891499aaec68c737d160a/docs/cli/telemetry.md)
  describes additional live telemetry and capture settings; this does not imply
  those details exist in stored conversations.
- Gemini [persisted record types](https://github.com/google-gemini/gemini-cli/blob/b9776560ea5710bc673891499aaec68c737d160a/packages/core/src/services/chatRecordingTypes.ts)
  and [recording/replay implementation](https://github.com/google-gemini/gemini-cli/blob/b9776560ea5710bc673891499aaec68c737d160a/packages/core/src/services/chatRecordingService.ts)
  expose structured parts, tools and usage, including cached and thinking data.
  Apply repeated-ID, metadata update, patch and rewind semantics before turn
  reconstruction. Do not treat every raw line as a new conversation event.
  Stored message/tool timestamps do not establish exact live request durations.

### PI evidence boundary

The reviewed MLflow
[integration catalog](https://github.com/mlflow/mlflow/blob/8171679f34de004e2ec62b759b8591f5e089f15f/docs/src/components/TracingIntegrations/index.tsx)
contains no Pi entry. Generic OTLP ingestion is not a Pi-specific parity baseline.

Use upstream Pi revision `21e4cb80dddd4330806247818b1f4ec19af95fa5` as the
reviewed source-format reference:

- [Session format](https://github.com/earendil-works/pi/blob/21e4cb80dddd4330806247818b1f4ec19af95fa5/packages/coding-agent/docs/session-format.md):
  session header, parent-linked entry tree, messages, model changes, compaction,
  branch summaries and related records. Reconstruct the persisted active path.
- [Message/usage types](https://github.com/earendil-works/pi/blob/21e4cb80dddd4330806247818b1f4ec19af95fa5/packages/ai/src/types.ts):
  model/provider, structured content, tool identities, usage, costs, stop/error
  information and optional timing. Reasoning usage is already included in
  output; avoid double-counting. A one-hour cache bucket must not be assumed
  equivalent to a target bucket with different lifetime semantics.

A [third-party Pi tracer](https://github.com/chrisdjscott/pi-mlflow-tracer/blob/95393622639684ff7f075aa2a7eb01f8e9d9d0ba/index.ts)
provides an agent/model/tool span example. It is not an official MLflow
integration and is not the completeness baseline for Pi support.

## 4. Parity verification

For each adapter/source version, fixture-based integration checks must verify:

1. All persisted logical turns and structured messages survive import, including
   tool-only assistant records, multiple tool calls and relevant source sidecars.
2. Model/tool/agent types, tool call/result pairing, nested subagents and skills
   render with the correct identities and parentage when recorded.
3. Per-call/turn/session usage agrees with source semantics, including cached
   subsets, split messages, repeated counters, cumulative totals and zeros.
4. Source costs are retained; origin and calculation method remain distinct.
   Estimates populate supported structured/native USD cost fields and reproduce
   from recorded pricing/conversion snapshots; unknown values remain unknown.
5. User, directory, Git, timing and outcome values are source-attributable;
   unavailable live-only facts are diagnosed rather than invented.
6. Native fields and supported structured spans carry available baseline data;
   raw artifacts supplement them and preserve permitted additional fields.
7. Canonical harness source values win over conflicting source fields, OTLP
   service names and automatic SDK defaults, in both privacy modes.
8. Coverage diagnostics and artifacts agree; official Pi plugin parity is not
   claimed without a separately identified and verified integration baseline.
