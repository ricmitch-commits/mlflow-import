# MLflow compatibility profile — resolved surface and observed behavior

> This page is part of the documentation suite: [README](../README.md) ·
> [usage](usage.md) · [result schema](result-schema.md) ·
> [architecture](architecture.md) · [security](security.md) ·
> [testing](testing.md) · [operations](operations.md).

**Pinned pair:** client `mlflow==3.17.2` (PyPI) and a tracking server launched
from that same installed distribution. The plan prose referenced
`mlflow==3.17.2.dev0`, which is **not published on PyPI**; per the controller
ruling recorded 2026-10-21 the profile pins **`mlflow==3.17.2`**, the exact
stable release of the same API line. See `tests/integration/constraints.txt`
and `tests/integration/README.md` for the reproducible recipe.

**How this was verified (2026-10-21, macOS/aarch64, Python 3.13.11):** every
row below was exercised against a disposable server
(`mlflow server --backend-store-uri sqlite:///<run-dir>/mlflow.db
--artifacts-destination file://<run-dir>/artifacts --host 127.0.0.1 --port
<port>`, `--serve-artifacts` enabled by default in 3.17.2) launched from the
same venv. The committed proofs live in `tests/contract/test_mlflow_profile.py`
(markers `profile` and `integration`); `verify_profile` in
`tests/integration/mlflow_profile.py` re-proves the surface on every run.

## Resolved public surface

| Gateway-facing operation | Resolved public API in 3.17.2 | Observed behavior |
| --- | --- | --- |
| Server version attestation | Read-only `GET <tracking-base>/version` | Returns the server version as plain-text semantic version (`3.17.2` on the pinned pair); the importer also accepts the documented JSON `{ "version": "3.17.2" }` shape and requires an exact `3.17.2` match before trusting profile-backed behavior. |
| Experiment lookup | `MlflowClient.get_experiment_by_name(name: str) -> Experiment \| None` | Returns `None` for a missing name; the created `Experiment` exposes `.experiment_id` and `.artifact_location` (`mlflow-artifacts:/<id>` for a proxied server). |
| Experiment creation | `MlflowClient.create_experiment(name: str, artifact_location: str \| None = None, tags: dict \| None = None) -> str` | Returns the new experiment id as a string. |
| Span creation | module-level `mlflow.start_span_no_context(name, span_type='UNKNOWN', parent_span=None, inputs=None, attributes=None, tags=None, metadata=None, experiment_id=None, start_time_ns=None, log_level=None, links=None, description=None) -> LiveSpan` | Returns a `LiveSpan`. `experiment_id` must name an existing experiment. A root span starts a trace; `parent_span=<LiveSpan>` links the child (same trace id, `child.parent_id == root.span_id`). `start_time_ns` is preserved verbatim. |
| Span completion | `LiveSpan.end(outputs=None, attributes=None, status=None, end_time_ns=None)` | `status="OK"` (string) is accepted; `outputs`/`attributes`/`end_time_ns` round-trip byte-exact through `get_trace`. |
| Trace retrieval | `MlflowClient.get_trace(trace_id: str, display: bool = True, flush: bool = False) -> Trace` | Full payload: `trace.info` (state, tags, metadata) plus `trace.data.spans` with inputs/outputs/attributes/parent id and nanosecond timestamps. `flush=True` drains the client-side async export queue first — without it, a trace just ended may still be served from the in-process in-memory buffer and appear "corrupted" (missing server-side tags). `display=False` avoids notebook display. |
| Trace search (paginated) | `MlflowClient.search_traces(experiment_ids=None, filter_string=None, max_results=100, order_by=None, page_token=None, run_id=None, include_spans=True, model_id=None, locations=None, flush=False) -> PagedList[Trace]` | Pagination via `PagedList.token` (opaque offset token) + `page_token=` on the next call; pages covered every probe trace. `filter_string` supports ``tags.`mlflow.traceName` = '<name>'``. `include_spans=False` returns metadata-only results (much cheaper). |
| Trace tag update | `MlflowClient.set_trace_tag(trace_id: str, key: str, value: str)` / `MlflowClient.delete_trace_tag(trace_id: str, key: str)` | Both persist and are visible in a subsequent `get_trace`. Tag values are silently truncated to 4096 characters and metadata values to 250 (`mlflow.tracing.constant`), so payloads must never ride in tags/metadata. The importer's `MlflowGateway.record_artifact_verification(trace_id, sha256, size)` records the verified artifact digest through `set_trace_tag` (the `opencode.artifact_sha256`/`opencode.artifact_size` tags) so candidate verification can check the stored artifact's integrity against the digest recorded at write time. |
| Retirement | `MlflowClient.delete_traces(experiment_id: str, max_timestamp_millis=None, max_traces=None, trace_ids: list[str] \| None = None) -> int` | `delete_traces(exp_id, trace_ids=[...])` deletes exactly the listed traces and returns the count; a deleted trace then raises `RESOURCE_DOES_NOT_EXIST` from `get_trace`. |
| Named trace artifacts | composition (see below) | Upload preserves the requested relative path, streams in fixed chunks, and retrieves exact bytes. |

## Named trace artifacts (plan Step 3) — positive result

The importer records the exact mapped artifact provenance (including the
catalog version) in the artifact document and re-stream verification passes
the same provenance envelope. The deterministic relative path is validated
against MLflow's 4096-character tag limit before trace creation; an oversized
path is a safe per-item `artifact_path_too_long` failure rather than a
silently truncated tag.

`MlflowClient` in 3.17.2 exposes **no single method** named anything like
"upload trace artifact". The public named-artifact channel is a composition of
public pieces, and it satisfies the profile:

1. Every server-backed trace carries the public tag `mlflow.artifactLocation`
   (constant `MLFLOW_ARTIFACT_LOCATION` from `mlflow.utils.mlflow_tags`)
   pointing at the trace's artifact directory
   (`<experiment artifact root>/traces/<trace_id>/artifacts`, served through
   the server's proxied-artifacts endpoint).
2. `mlflow.artifacts.get_artifact_repository(uri, tracking_uri=...)` (public)
   resolves that URI to an `ArtifactRepository`
   (`MlflowArtifactsRepository` → `HttpArtifactRepository` against a proxied
   server).
3. `ArtifactRepository.log_artifact(local_file, artifact_path)` (public
   interface method) stores the file at `artifact_path/<basename>` — a
   caller-chosen relative path, preserved exactly.
4. `ArtifactRepository.download_artifacts(artifact_path, dst_path)` /
   `mlflow.artifacts.download_artifacts(artifact_uri=...)` / `list_artifacts`
   retrieve and list the same paths.

Observed behavior (empirically instrumented): the client passes an open file
handle to the HTTP layer, which streams the body in bounded 16,384-byte reads
(17 reads for a 274,472-byte file) — the whole file is never loaded into
memory; the server-side proxied-upload handler likewise reads the request
stream in fixed chunks (`ARTIFACT_STREAM_CHUNK_SIZE`) and has no size cap on
this route. Exact-byte retrieval verified by sha256 and length, including
nested relative paths (`sessions/2026/10/21/session-payload.json`, 274,472
bytes). `tests/integration/mlflow_profile.py` adapts this channel to the
plan's `TraceArtifactSink` contract as `upload_named_json(client, trace_id,
relative_path, local_file, chunk_size) -> ArtifactVerification` (streams the
source in fixed `chunk_size` blocks for hashing and staging) and
`verify_named_json(client, trace_id, relative_path, sha256, size) -> None`
(streamed digest comparison; raises on mismatch, accepts nothing tampered).

### Negative results recorded alongside (plan Step 3 honesty requirement)

- The UUID-only attachment channel (`ArtifactRepository.upload_attachment`,
  `download_trace_attachment*`, and the `mlflow.tracing.attachments.Attachment`
  span-input mechanism) validates that the path **is a UUID** and stores under
  `attachments/<uuid>`; it rejects caller-chosen relative paths and therefore
  does **not** satisfy the profile (verified: a non-UUID path raises).
- Run-artifact methods (`MlflowClient.log_artifact(run_id, ...)`, the
  `/api/2.0/mlflow/tracking-artifacts/...` PUT route) require a run uuid and
  are capped at 10 MB; they are not trace-scoped and do not satisfy the
  profile.
- No presigned multipart upload applies to the pinned `file://` artifact
  destination (`MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD` targets S3-backed
  presigned flows).

The composition above is the same channel MLflow itself uses to store span
payloads (`traces.json`) for artifact-backed traces, so it is the sanctioned
public route, not a private-API workaround.

## Inline payload capacity

`CapabilityReport.max_inline_payload_bytes` pins **1 MiB (1,048,576)**: the
profile verifies — on every `verify_profile` run — that a payload of exactly
that size placed in span `outputs` round-trips byte-exact through
`get_trace` (observed `spansLocation=TRACKING_STORE` on the pinned server, so
span payloads are ingested through the OTLP endpoint into SQLite). This is a
verified floor and pinned policy threshold, not a discovered ceiling; the
writer must send larger session payloads as named trace artifacts. Trace
metadata/tag storage is unsuitable for payloads (250/4096-character silent
truncation, see above).

## Dry-run probing (plan Step 4) — non-mutating

`verify_profile(client, profile, dry_run=True)` proves, strictly through
read-only calls: the exact `GET <tracking-base>/version` server attestation;
static availability of every required method; server reachability
(`get_experiment_by_name` on a name that must not exist);
protocol responses (`get_trace` on an unknown id raises a proper MLflow
`RESOURCE_DOES_NOT_EXIST`, not a transport error); full-payload retrieval of
an existing trace discovered via `search_traces`; the artifact channel
(`list_artifacts` on a discovered trace's artifact directory); and retirement
*availability*. It never creates an experiment, trace, or artifact, and never
uploads a probe file. A target with no readable trace cannot prove
full-payload retrieval without mutation and is reported incompatible for
dry-run. The committed test proves non-mutation by snapshotting the full
experiment set, trace set, and artifact trees before and after the probe and
asserting equality.

## How an unsupported client/server pair is reported

The runtime gate is `probe_capabilities` (read-only in both modes) plus
`require_capabilities`. It derives the configured tracking URI from the
public client and performs the same read-only `/version` attestation. A pair
that fails any check aborts the **whole
invocation before any mutation**:

- the failure is a typed `CapabilityError` with the stable code
  `mlflow_capability_unsupported` and **exit status 2**;
- in a normal CLI run the failure appears in the result document's
  `errors[]` section (code + safe message) and as one safe error line
  on stderr when no valid invocation exists to report on;
- the message carries only the probe's safe causes, joined with
  `; ` — for example an installed client version that does not match
  the pinned profile version, a missing client method, a
  `mlflow.start_span_no_context` signature missing required
   parameters, an unreachable server, a missing/malformed/mismatched
   `/version` response, unsupported URI scheme, authentication/TLS/transport
   failure, a trace-retrieval response that is not an MLflow protocol error,
   or a required capability (retirement / named artifacts) that is unavailable;
- nothing is ever created, uploaded, or deleted by the probe itself, so
  the operator can fix the environment and re-run with no cleanup.

A **version mismatch alone** is always incompatible: the probe trusts
the recorded profile's mutation-based evidence (`LiveSpan.end`, named
artifacts, retirement semantics, and the 1 MiB inline floor) only while both
the installed client and the attested server match `PINNED_MLFLOW_VERSION`
exactly. A numeric client match alone never bypasses the read-only server
attestation or the existing method/protocol checks. A newer client/server pair
requires recording a new profile before the package constraint changes.

## Brief-vs-reality mismatches (recorded verbatim)

- **`LiveSpan(trace_id, span_id)`** — the brief's construction shape. Actual
  3.17.2 constructor is `LiveSpan(otel_span, trace_id, span_type="UNKNOWN")`;
  `trace_id` and `span_id` are properties (both populated on live spans;
  `parent_id` is `None` for roots). The writer only consumes the properties.
- **`mlflow.start_span_no_context(...)` parameter order** — the brief lists
  `parent_span` last; in 3.17.2 it is the third positional parameter, and the
  signature also accepts `log_level`, `links`, and `description`. Every
  parameter the brief names exists and works when passed by keyword.
- **`MlflowClient.update_trace_info`** — does **not exist** in 3.17.2. The
  public trace metadata/tag update surface is `set_trace_tag` /
  `delete_trace_tag` (tags only). There is no public method that updates
  `trace_metadata` after creation in 3.17.2; the profile records this absence
  and the writer must not depend on post-hoc metadata updates.
- **`search_traces(experiment_ids=...)`** — exists and works, but is
  deprecated in 3.17.2 in favor of `locations=...` (emits `FutureWarning`).
  The profile verifies both forms and uses `locations` everywhere except one
  deliberate exercise of the deprecated form.
- **`MlflowClient.delete_traces(experiment_id, trace_ids)`** — exists under
  exactly that name (keyword `trace_ids`), returning the deleted count.

## Live-verified gateway constraints (Task 14, 2026-10-22)

Task 14's real-server integration suite
(`tests/integration/test_mlflow_trace_contract.py` and
`tests/integration/test_mlflow_reconciliation.py`) drove the COMPLETE
production pipeline — the real gateway, not the in-memory fakes — against a
disposable 3.17.2 server. Two gateway call shapes that the fake-backed unit
and contract tests had encoded could not survive contact with the real pair
and were corrected:

1. **A non-root span must not carry `experiment_id`.**
   `mlflow.start_span_no_context(..., parent_span=<LiveSpan>,
   experiment_id=<id>)` fails on the pinned pair: the provider's
   `start_detached_span` logs its "experiment_id can only be used for root
   spans" warning and then dereferences a `LiveSpan` attribute
   (`span._span`) on the raw OTel span, raising `AttributeError`. The
   fluent wrapper swallows the exception and returns a NoOp span
   (`trace_id == MLFLOW_NO_OP_SPAN_TRACE_ID`), so every child span of every
   trace silently degrades — the importer's loud NoOp check surfaced it as
   `mlflow_span_degraded` item failures. The profile's proven child-call
   shape (Task 0's `_step_spans`) omits `experiment_id` — a child joins its
   parent's trace and inherits the experiment fixed by the root — and
   `MlflowGateway.create_trace` now passes it only for the root span. The
   client-level fake (`FakeMlflowClient.create_span`) models the
   degradation — a non-root span carrying `experiment_id` warns with the
   real provider's wording and returns a NoOp-shaped span (sentinel trace
   id, nothing registered server-side) — so this class of drift can no
   longer pass the fake-backed suites (pinned by
   `tests/unit/mlflow/test_gateway.py::
   test_child_span_with_experiment_id_degrades_in_the_fake_and_fails_loudly`).
2. **`LiveSpan.end` forces every non-`ERROR` span status to `OK`.** Ending
   a span with `status="UNSET"` (an incomplete turn's root, or a
   pending/running tool child) retrieves as span status `OK`: 3.17.2's
   `LiveSpan.end` unconditionally re-sets any non-`ERROR` status to `OK`
   (its own documented workaround for spans that never set one). The
   gateway's `RETRIEVED_SPAN_STATUS` mapping records this coercion;
   `verify_trace` compares the retrieved root span status through it
   (exactly like the pre-existing trace-state mapping), while the
   source-known outcome stays recorded on `CreatedTrace.root_status` and in
   the `opencode.outcome` trace metadata. The client-level fakes
   (`FakeLiveSpan.end`) mirror the coercion so the fake-backed tests encode
   the real behavior.
3. **Absent span timing cannot be represented faithfully.** Passing `None`
   to the pinned SDK invents wall-clock instants. The gateway validates root
   and child start/end timing in a read-only iterator pass before creating
   any span. Source-derived mapping fallbacks remain allowed; otherwise the
   item fails with `source_timestamp_missing` (exit 1), never synthetic time.
4. **A child failure is a partial create, not an unknown trace.** Once a real
   root exists, the gateway retains its identity, attempts to end unfinished
   spans with source-derived end timing and ERROR status, and raises a safe
   partial-create error. Reconciliation retires that known trace under the
   bounded retry policy while preserving previous traces. Pending async
   exports are flushed before deletion so a late export cannot resurrect a
   retired partial trace. Failed cleanup is
   reported as `partial_trace_cleanup_failed` with the known trace ID.

The corrected render contract is `mapping-v3` with `artifact-v2` provenance
envelopes. Existing `mapping-v1`/`mapping-v2` traces
are replaced once rather than incorrectly skipped with old root-input,
model-mapping, qualified source-lineage, or artifact-provenance semantics. Within
`mapping-v3`, turn-local
equivalence and historical sibling artifact preservation are unchanged.

These behaviors are exercised by the committed integration tests on every
run against the pinned pair.

## Environment hazards (all handled by the harness, documented for operators)

1. **Ambient OTLP configuration hijacks span export.** With
   `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (or `OTEL_EXPORTER_OTLP_ENDPOINT`) set
   — as it was in this machine's shell — MLflow routes spans to that OTLP
   collector; the `opentelemetry-exporter-otlp-proto-http` package is not a
   dependency of `mlflow` 3.17.2, so span creation silently degrades to
   `NoOpSpan` (`trace_id == MLFLOW_NO_OP_SPAN_TRACE_ID`) and nothing reaches
   the tracking server. The fixture and `verify_profile` scrub these
   variables before the tracing provider initializes and additionally fail
   loudly if a NoOp span is ever observed.
2. **Span export follows the global tracking URI.** `start_span_no_context`
   exports through `mlflow.get_tracking_uri()` (env `MLFLOW_TRACKING_URI`),
   not through any `MlflowClient` instance's URI. Constructing
   `MlflowClient(tracking_uri=...)` alone leaves spans going to the default
   local store. The harness pins the global URI to the server under test.
3. **Trace export is asynchronous.** Immediately after `span.end()` the trace
   typically exists only in the process-local `InMemoryTraceManager`
   (missing server-side tags): `get_trace` then reports "span data is
   corrupted", and later "not found" once the buffer is popped. Passing
   `flush=True` (or calling `mlflow.flush_trace_async_logging()`) makes
   retrieval deterministic.
4. **Silent tag/metadata truncation** (4096/250 characters) — payloads must
   use artifacts, never tags or metadata.

## Gateway boundary (Task 9)

`src/opencode_mlflow/mlflow/gateway.py` and
`src/opencode_mlflow/mlflow/capabilities.py` are the importer's writer
boundary and the **only** modules that import MLflow — through the public
`mlflow` / `mlflow.tracking` / `mlflow.artifacts` entry points only (never
the semi-private `mlflow.store.artifact.artifact_repo` import Task 0's
availability check used; the public factory
`mlflow.artifacts.get_artifact_repository` is used instead). A contract
test (`tests/contract/test_gateway_contract.py`) pins both facts by
AST-scanning the source tree, and pins the gateway's constants (version,
inline floor, required method set, `mlflow.artifactLocation` tag, NoOp
sentinel, `mlflow.trace.session` key, 4096/250 truncation limits) against
this profile and the installed mlflow.

**Artifact sink composition.** `TraceArtifactSink.upload_named_json` /
`verify_named_json` reimplement the profile-proven composition identically:
resolve the trace's `mlflow.artifactLocation` tag via `get_trace` →
`mlflow.artifacts.get_artifact_repository(uri, tracking_uri=...)` →
`ArtifactRepository.log_artifact(local_file, artifact_path=<parent>)` /
`download_artifacts`. The source is read and staged in fixed `chunk_size`
blocks (default 8 MiB) and never loaded whole; verification streams the
download in fixed 64 KiB blocks and compares sha256 + size, rejecting
anything tampered. UUID-only attachment channels and run-artifact fallbacks
are never used. Relative paths are traversal-guarded.

**Span work.** `create_trace` starts the root and each child through the
injectable `PublicTraceApi` adapter (module-level
`mlflow.start_span_no_context`), resolves each child's `parent_source_key`
against a per-call `source_key -> LiveSpan` map bounded by the trace's span
count (span handles only — never the child list or a source payload, which
are consumed one child at a time), and ends every span through
`LiveSpan.end`. Because span export follows the *global* tracking URI
(hazard 2 above), the gateway pins the global URI to the client's target
for the whole create and restores the saved value after. Because failed
span creation degrades to a NoOp span rather than raising (`trace_id ==
MLFLOW_NO_OP_SPAN_TRACE_ID`), the gateway fails loudly with a safe
`mlflow_span_degraded` capability error instead of silently losing the
trace. The root ends after all children with the status from the gateway's
one derivation helper `_effective_root_status`: the request's explicit
`root_status` (set by Task 10's mapping, which is authoritative over the
source terminal outcome — an explicit failure whose last child ended `OK`
stays `ERROR`) is preferred whenever it is set, otherwise the status is
derived from the verbatim request fields: the last child's status when
children exist, otherwise `OK` with a root output and `UNSET` without one.

**Ownership search.** `iter_owned_traces` searches with the exact
server-side metadata filter ``metadata.`opencode.importer` =
'opencode-mlflow'`` when the target accepts it — falling back to unfiltered
pages when the target rejects the filter or when the filtered first page
comes back empty (a filter that silently matches nothing is not meaningfully
available, and the ownership marker may legitimately live in tags rather
than metadata) — always uses the non-deprecated `locations=` form, walks
pagination tokens to exhaustion, and re-verifies the exact ownership marker
(tags or metadata) locally on every candidate — an unowned trace is never
yielded.

**Verification.** `verify_trace` retrieves the complete payload with
`get_trace(display=False, flush=True)` — never a search preview — and
compares the complete root inputs/outputs, span ids, timing, the root span
status, the trace state, the root attributes, the native
`mlflow.trace.session` metadata, and the expected tags/metadata. The
expected root status is single-sourced on `CreatedTrace.root_status`
(the mapping's explicit `root_status` request field is preferred in the
gateway's one derivation helper whenever it is set). The trace state is compared
through the pinned mapping MLflow itself applies
(`maybe_update_trace_state`): `OK`→`OK`, `ERROR`→`ERROR`, and a root ended
`UNSET` surfaces as state `OK` (MLflow overrides `STATE_UNSPECIFIED` to
`OK`), so an incomplete turn preserves source status `UNSET` on
`CreatedTrace.root_status` and `opencode.outcome`, but its retrieved span
status is `OK` (the `LiveSpan.end` coercion above) while its trace
state reads `OK`. Root attributes are compared as a subset
(`expected ⊆ actual`) because MLflow adds its own internal attribute keys
to server-side spans; each of the importer's attributes must survive with
an equal value. Tag and metadata values at or below the documented
4096/250 truncation limits are compared exactly; longer values only
require the key's presence, so verification never relies on full-length
tag round-trips.

**Recorded artifact digests.** `record_artifact_verification(trace_id,
sha256, size)` records a verified artifact digest on its trace through the
public `set_trace_tag` operation (the `opencode.artifact_sha256` /
`opencode.artifact_size` tags). Candidate verification
(`verify_owned_candidate`) reads that recorded digest back from full trace
retrieval and checks the stored artifact's exact bytes against it — an
integrity check against the digest recorded at write time, since the
artifact document embeds the created trace's generated identity map and
its bytes are therefore not deterministic across runs. The two tag updates
are not atomic: a crash between them leaves an incomplete candidate (no
usable recorded digest) that the next run replaces.

**Runtime capability probe.** `probe_capabilities` (in `capabilities.py`)
is read-only in both modes: exact server-version attestation through
`GET <tracking-base>/version`, client-method availability, the module-level
span API's availability and signature, server reachability
(`get_experiment_by_name` on a reserved name that must not exist), protocol
responses (`get_trace` on an unknown id must raise an MLflow protocol error
carrying a machine-readable `error_code`, not a transport failure), and the
public artifact factory's availability. Facts that cannot be re-proven
without mutation (`LiveSpan.end`, the named-artifact composition's
server-side behavior, the 1 MiB inline floor) are taken from this recorded
profile and apply only while both the installed client and attested server
match the pinned version; a normal write always requires retirement (rollback
after partial creates), and an incompatible report fails the whole invocation with
`mlflow_capability_unsupported`, exit status `2` (`require_capabilities`
performs that abort). The gateway additionally gates every mutating method:
the capability check must pass first (in the gateway's own mode), and a
dry-run gateway never mutates — `resolve_experiment` represents a planned
creation instead.
