# MLflow Compatibility Profile

**Profile:** MLflow `3.17.2` client and server
**Verified evidence:** 2026-11-14 baseline; 2026-11-16 activated 8 MiB localhost qualification below
**Scope:** Qualified five-adapter projection, resource/recovery profile and strengthened OpenCode reference gate; final verification and independent acceptance complete

## Disposable Server

### Aggregate trace admission and metadata-first reads

`mlflow.trace_envelope` admits at most **64 MiB of serialized
span upper bound** and **4096 spans including the root** by default,
independently of the existing 8 MiB per-value policy. `--max-trace-bytes` and
`--max-trace-spans` can raise those admission limits up to absolute caps of
2 GiB and 32,768 spans. The local PostgreSQL qualification below supports only
the default aggregate constants for its tested shapes; a version match alone
is not that qualification, and raised limits require separate capacity
validation. These are importer limits, not arbitrary proxy allowances.
The counter derives JSON escape expansion (including historical JSON-looking
strings' extra layer), protobuf field framing, IDs and fixed SDK fields. It
counts strings in small chunks, rejects non-finite/unsupported values and deep
containers, and stops before retaining an over-budget span graph. The writer
materializes only admitted pure DTOs and consumes the child factory once.

Control data has a separate 2 MiB serialized envelope. Global, pure projection
preflight reserves 1 MiB for late attempt/instance identities, map nonce/path,
session artifact tags and final target identity. Creation repeats admission on
the actual decorated request. Global preflight runs in both modes before any
experiment creation, trace/session artifact publication or recovery-journal
mutation, using the same mapping contexts and frozen pricing bases as writing.
No unresolved-target placeholder fingerprint is persisted.

New traces carry creation-only tags `mlflow_import.trace_resource_profile`,
`mlflow_import.trace_resource_bound_bytes`, and
`mlflow_import.trace_resource_span_count`. Metadata-only lookup uses public
`TracingClient.get_trace_info` on the pinned SDK (or an adapter's metadata-only
method/exact-ID search with `include_spans=False`). Full reads require this
recognized proof before calling `get_trace`; actual returned protobuf bytes,
full-read JSON size and span count are checked against the proof and hard caps.

This tag proof assumes a **trusted importer writer and immutable span payloads**
after creation. It is not a cryptographic attestation against a party able to
forge tags or append/replace spans. A tagless legacy trace is never stamped or
read speculatively: it fails permanently with
`trace_resource_profile_unverified`, including old small v1 pricing maps unless
an explicitly offline-qualified migration is performed separately. Such a
failure is not a candidate mismatch authorizing replacement. Metadata-only
existence, artifact-location and exact retirement-identity checks do not fetch
span bodies; prepared/ready replacement references still require full verified
payloads at their gates. Existing ownership/native identity and version-decoder
checks remain required in addition to resource admission.

### Local aggregate and compact-projection qualification (2026-11-16)

Disposable loopback containers exercised MLflow 3.17.2 with one worker and an
8 GiB limit against PostgreSQL 16 with a 1 GiB limit. Generated cases covered
2106 total spans, 4096 small spans, few large spans near the byte bound, and a
combined near-byte/count boundary. The largest measured OTLP body was
66,849,707 bytes; all accepted request and full-read bodies remained below their
admission upper bounds. Kernel cgroup peaks were approximately 2.95 GiB for
MLflow and 323 MiB for PostgreSQL, with zero OOM events or restarts. Evidence
and measurement limitations are in `tests/fixtures/container_trace_qualification.json`.

Separate tests use the actual production mapper and `conversation-delta-v1`
projection at 10, 100 and 1000 calls, including a 2105-child shape. Lifecycle
checks exercise turn-map.v2 context resolution, complete retained source,
prepared-attempt interruption/resume, unchanged reuse, append/edit isolation and
immutable maps. The full mapper-driven lifecycle runs at 10 calls on SQLite
and 100 calls on PostgreSQL. It does not claim a 1000-call database lifecycle or
long-duration/concurrent capacity qualification.

The final opt-in container and compact-recovery selection passed **14 tests in
162.35 seconds**. Unit/contract checks passed **2394 tests**, BDD checks passed
**41 tests**, and static checks were clean. All targets were newly created local
fixtures; no user tracking-server requests, deployment changes, commits or
pushes were made for this correction. A live retry against an existing legacy
experiment remains subject to the metadata-first migration restriction above.

### Mapping-v8 native projection migration

The historical object-ingestion API now carries bare native session/parent IDs,
source-only nullable client request identity, canonical per-session harness trace name
(via the `mlflow.traceName` tag, independent of root span names),
explicit previews and separate native analytics. OpenCode matches the 0.4.0
plugin reference: semantic message duration, newline text previews, reasoning in
native total, and NULL native cache/cost columns. Canonical `accounting-v2` retains
explicit source totals, cache and costs. Other harness cache/cost mappings persist.

Initial tags carry immutable qualified identity, ownership, operation, version,
source/user/Git and instance facts before spans upload. Arbitrary physical request
metadata is restricted to optional `parent_session_id`; MLflow owns finalization
and promotes native session/usage/cost API keys. Public readback checks exact values
and absence. Ownership reads support new tags and legacy metadata together, reject
contradictions, and union both paginated search locations. Replacement references
use a versioned v2 envelope protecting immutable ownership and native identity;
unverifiable legacy references fail before retirement. The local SQL reproduction
is `tests/integration/test_opencode_reference_projection.py`; no remote experiment
is needed to verify this migration.

The profile tests launch the server from the installed MLflow distribution with:

```text
python -m mlflow server \
  --backend-store-uri sqlite:///<run-dir>/mlflow.db \
  --artifacts-destination file://<run-dir>/artifacts \
  --default-artifact-root file://<run-dir>/artifacts \
  --host 127.0.0.1 \
  --port <free-port>
```

The server is health-checked at `/health`, uses a fresh temporary directory,
and is terminated and removed after the test session. The verified artifact
repository for this original/benchmark recipe is the same-host `file://`
experiment artifact root. The importing process must be able to access that
local root. The additional bounded proxy qualification below does not change
this benchmark recipe or recertify its timings for network artifacts.

### Bounded remote artifact profile

`tests/integration/test_remote_artifacts_profile.py` starts a separate **local-only**
MLflow 3.17.2 server using the recipe above without `--default-artifact-root`.
Its experiments expose `mlflow-artifacts:/` locations. The public repository
factory resolves those into the HTTP artifact service; direct HTTP(S) proxy
repository URIs use the same bounded listing/existence/download binding.
No private MLflow helpers, backend access, run containers, new dependencies or
whole-file SDK download methods are used for these importer-owned objects.

Qualification covers fresh import, zero-mutation unchanged rerun, append with
prior trace reuse, dry-run before/after creation, independent checksummed reads,
filtered pending recovery, and phase-two quota exhaustion with prior-trace
preservation and successful retry. Unit evidence covers response/entry/page/token
exhaustion, malformed paths, partial-read retries, bounded upload ambiguity,
shared operation/deadline exhaustion and standard auth/TLS configuration.
Four additional offline HTTPS cases verify that an explicit CA file/directory
replaces default trust for authenticated artifact and version GETs, rejecting a
second server that the default context demonstrably trusts.
See [temporary-resources.md](temporary-resources.md#qualified-artifact-transport)
for exact envelopes and new-experiment default handling, and
[errors.md](errors.md#bounded-remote-artifact-correction) for executed checks.

In addition to synthetic loopback evidence, the reported `test-import` invocation
was executed against its existing remote MLflow proxy: **9 traces imported across
4 sessions in 8.436 seconds**, exit 0; independent paged lookup confirmed all
9 persisted traces and complete attempt states. An unchanged rerun reported
9 unchanged and no imported/updated/retired turns in 10.467 seconds. The final
corrective non-slow suite passed **1886 tests**, with 4 slow cases deselected.
See [errors.md](errors.md#verified-resolution-of-the-reported-no-trace-import).
This small live import is not network-scale performance certification. Arbitrary
HTTP repositories and other cloud schemes remain unqualified.

## Verified Operations

The disposable profile and integration suites cover:

- Experiment lookup and creation.
- Context-free root and child span creation with historical timestamps.
- Complete structured inputs, outputs, attributes, status, and parent links.
- Explicit flush, full trace read-back, and paginated trace search.
- Mutable managed tag set and removal.
- Run-free named artifact upload, listing, streaming checksum verification, and tamper rejection.
- Exact trace deletion with absence verification.
- No-op span rejection and ambient OTEL exporter isolation.
- Read-only dry-run profile checks with unchanged experiment, trace, and artifact state.

## API Surface

The importer is restricted to the public APIs below:

| Operation | Binding |
| --- | --- |
| Experiment lookup | `MlflowClient.get_experiment_by_name` |
| Experiment creation | `MlflowClient.create_experiment` |
| Production historical trace/native fields and spans | Public `TraceInfo`, `Span.from_dict`, and `TracingClient.start_trace/log_spans` in the pinned distribution |
| Detached-span compatibility/failure seam | `mlflow.start_span_no_context` and `LiveSpan.end` |
| Full trace read | `MlflowClient.get_trace` |
| Paged search | `MlflowClient.search_traces` with `locations` and `page_token` |
| Tags | `MlflowClient.set_trace_tag` and `delete_trace_tag` |
| Retirement | `MlflowClient.delete_traces(..., trace_ids=[...])` |
| Named artifacts | `mlflow.artifacts.get_artifact_repository` plus public repository methods |

The configured inline per-value envelope is **8 MiB** (8,388,608 canonical UTF-8
bytes), qualified by the localhost historical-ingest cases below. The earlier
synthetic live benchmark round-tripped a message over 2,004,362 bytes and a part
over 1,205,918 bytes; those earlier observations alone did not qualify the new
boundary or arbitrarily large one-turn traces. Oversized root values fail
with `payload_too_large`; artifacts cannot substitute for complete native root
values. No MLflow runs are created by the profile or importer boundary.

### Supplemental 8 MiB localhost qualification (2026-11-16)

**Activated:** both `capabilities.py::PINNED_MAX_INLINE_PAYLOAD_BYTES` and
`tests/integration/mlflow_profile.py::MAX_INLINE_PAYLOAD_BYTES` are **8,388,608
canonical UTF-8 bytes per complete value**. The existing gateway equality
contract remains in force. The new localhost tests assert both constants equal
the production capability probe's reported bound, use that report for root and
structured-child guards, and retain fixed 8 MiB + 1 rejection cases. No runtime
limit override is used. The exact-version and read-only capability gates still
apply before normal writes.

`tests/integration/test_inline_payload_profile.py` forces
`live_mlflow_server(requested_uri="", proxy_artifacts=True)`, asserts a live
local process and MLflow 3.17.2, and starts one worker. An unusable ambient remote
URI is deliberately set; the measured requests are restricted to that process's
loopback host and port. All values are generated synthetics containing nested
objects/lists, Unicode, quotes, backslashes and newlines. No SessionsAI payloads
are loaded or logged and no user-target writes are performed.

The real `MlflowGateway.create_trace(historical=True)` uses public
`TraceInfo`/`Span.from_dict` and `TracingClient.start_trace/log_spans`. Each case
puts the complete value simultaneously in root and TOOL-child inputs, outputs,
and an attribute (six placements). The unchanged full gateway verifier and a
second public `get_trace` both succeed; the tests independently compare all six
retrieved values, their types, and the parent link. The pinned SQL backend serves
these spans through its complete-trace HTTP endpoint, even when the experiment's
artifact root is `mlflow-artifacts:/`; these span reads are not named-artifact
downloads.

Initial local qualification: **7 passed in 16.04 seconds**. Numeric observations from that
run (generation, ingest, full verification and independent readback included in
each case's elapsed time):

| Value | Canonical bytes | Largest HTTP request bytes | Seconds | Sampled server-tree RSS bytes | Sampled client RSS bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured, old bound + 1 | 4,194,305 | 19,359,195 | 1.733 | 3,111,878,656 | 745,177,088 |
| Structured, first reported blocker size | 4,196,595 | 19,369,767 | 1.563 | 2,954,575,872 | 865,386,496 |
| Structured, second reported blocker size | 4,215,023 | 19,454,811 | 1.603 | 2,684,043,264 | 873,021,440 |
| Structured, qualified bound | 8,388,608 | 38,717,511 | 3.134 | 2,971,795,456 | 1,267,613,696 |
| String, qualified bound | 8,388,608 | 38,717,199 | 3.043 | 2,882,076,672 | 1,268,662,272 |

RSS is sampled with `ps` at approximately 100 ms intervals, summing the server
process tree; it can count shared pages more than once and miss between-sample
peaks. These are Darwin localhost observations, not an enforced container memory
limit or Linux performance certification. The tests assert server and client RSS
each below 8 GiB, per-case elapsed time below 300 seconds, and measured requests
below 64 MiB for this fixed two-span workload. They do not establish a 64 MiB
universal HTTP-request bound or a bound on arbitrary span counts.

The two remaining live cases exercise the production structured-activity guard
at **8,388,609 bytes**, separately for inputs and outputs: lazy child expansion
raises `payload_too_large` before any historical-ingest HTTP request. Complete
roots are also rejected before transport. The new unit boundary suite additionally
covers both string and structured roots at all four accepted sizes, rejection
at bound + 1, real tool-part projection for both structured directions, and the
existing digest/size artifact-reference policy for oversized non-activity values.
The focused envelope/capability/gateway-contract/mapping/artifact selection passed
**230 tests in 9.80 seconds**. No fragmentation, mapping, gateway, artifact-map,
pricing-map, journal, or upload/download limits are changed by these tests.

**Activated-profile verification (2026-11-16):** after synchronizing both
constants, the envelope/capability/gateway-contract/mapping/artifact selection
plus the existing negotiated-limit oversized-root regression passed **231 tests
in 9.96 seconds**. The existing regression derives its oversized value from the
profile, so it now rejects a complete root above 8 MiB without changing its
expectation. The new inline qualification, original MLflow profile contracts,
historical string transport and native-preview transport selection passed
**95 tests in 43.88 seconds**, using disposable localhost servers with the
ambient tracking URI removed. This includes all seven inline qualification
cases with production-reported limits and the original profile's larger live
payload. The six-placement requests remained at most **38,717,511 bytes**;
per-case elapsed times were **1.937–4.135 seconds**, sampled server-tree RSS
was at most **3,837,952,000 bytes**, and sampled client RSS was at most
**1,137,180,672 bytes**. Ruff lint/format passed for all five affected Python
files; `mypy src` passed for **65 source files**. No existing boundary-test
expectations needed relaxation. These are local activation results, not new
behavioral evidence about the existing remote route or its 8 GiB deployment.

Transport/configuration checks are deliberately narrower than remote behavioral
qualification. Installed vendor MLflow 3.17.2 `RestStore.log_spans` serializes the
entire supplied span list into one OTLP protobuf POST to `/v1/traces`, without
splitting it at the importer per-value limit. Its SQL ingestion stores the
translated span values. The vendor's distinct trace metadata/tag constraints
(250-character metadata, 250-character tag keys and 4096-character tag values)
are not an inline span-value allowance. The observed round trips, rather than
an inferred vendor size promise, support the activated 8 MiB floor.

Read-only Kubernetes configuration inspection confirmed the current MLflow
deployment's **8 GiB** memory limit and **`MLFLOW_WORKERS=1`**, with its healthy
GitOps application at `9649417bb9eeea91c155ccecb00134856356150a`. Its OpenShift
Route has no custom body-limit annotation; the default ingress controller's
reported tuning contains only `reloadInterval: 0s`, and router environment names
show no explicit body-limit override. This identifies no configured smaller
request-body cap, but does **not** prove the route accepts the measured 38.7 MB
requests or certify every remote/proxy/vendor deployment. No large remote
request was sent. A per-value profile change cannot itself bound a whole turn's
aggregate HTTP request or memory amplification; the existing whole-turn resource
and failure boundaries continue to apply.

## Recovery and artifact evidence

Production imports use experiment-root shared session objects, reusable turn
maps, checksummed manifests, and immutable prepared/ready/complete/aborted state
objects. `test_attempt_boundaries.py` exercises preparation/ready publication
response loss, completion publication failure, ready-batch recovery after a
source edit, duplicate-instance resume, and zero-turn overwrite cleanup.
`test_session_recovery.py` verifies that a later-turn failure preserves all prior
traces and that a zero-match filtered invocation reports pending recovery.

One OS advisory lock serializes local writers per target/experiment. Multi-host
writers require external serialization. Neither artifact publication nor trace
tag operations are a distributed lock.

## Source qualification limits

- OpenCode: fixture-backed current/legacy/mixed catalog schemas. Incomplete
  authoritative V2 schemas fail rather than silently falling back to legacy.
- All five now have fixture-backed **source** processing, documented in
  [source-adapters.md](source-adapters.md), including Pi v3 trees, Gemini JSONL
  replay, Claude sidecars and gzip archives. Unsupported variants fail explicitly.
  Independent native/structured readback now covers these representations,
  including gzip/privacy variants, source-specific accounting, recorded agents,
  and reproducible pricing. Exact evidence and qualifications are in
  [projection-parity.md](projection-parity.md). The scale benchmark is OpenCode-only.

Native/legacy foreign session aliases in tags and metadata are qualified under
overwrite/ignore/duplicate, including empty-source recovery, failed staging and
partial retirement. Global multi-root identity preflight, disk-backed plans/results,
streamed reporting and shared temporary admission are implemented and regression-tested.
No test establishes unavailable live-only observations, universal pricing/model
coverage, arbitrary historical producer variants, every possible identity race,
or unrestricted resource sizes. See [errors.md](errors.md#qualification-boundaries)
and [temporary-resources.md](temporary-resources.md).

## Synthetic performance evidence

The **current strengthened reference gate** passed **1 test in 1317.78 seconds**.
Each of the four separate importer invocations completed in less than 600 seconds.
Command (prepend `.venv/bin` to `PATH` for console-entrypoint subprocesses):

```sh
env -u MLFLOW_TRACKING_URI MLFLOW_DISABLE_AGENT_HINT=1 PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m pytest tests/integration/test_scale_pipeline.py -m slow -q -s --durations=0
```

2026-11-14 profile: Apple M4 Pro, 14 logical CPUs, 48 GiB RAM, Darwin 27.2
arm64, Python 3.13.11, MLflow 3.17.2 client/server, disposable loopback HTTP
server, SQLite backend and same-host file artifacts; synchronous SDK trace
logging and full native/artifact verification. No source project content is
used or copied from SessionsAI. The generated V2 linear OpenCode workload contains
307 sessions, 494 primary user turns,
8,134 messages, 47,976 derived parts and **319,987,518 JSON bytes**. It includes
parent/child sessions and two-turn sessions. Persisted tree/replay semantics are
covered by separate production-path Pi/Gemini gates, not this linear workload.

| Phase | Total seconds | Snapshot | Discovery | Scoped lookup | Reconciliation | Mutation entrypoints |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh | 378.762984708 | 9.837 | 17.222 | 0.214 | 348.391 | 3,512 |
| Unchanged | 308.581364792 | 10.306 | 17.389 | 8.927 | 127.847 | **0** |
| Append | 311.297854416 | 10.445 | 17.259 | 8.403 | 128.852 | 12 |
| Edit | 311.746874 | 10.241 | 17.680 | 8.714 | 130.591 | 13 |

Totals include pricing/donor full verification, recovery and cleanup work that is
not all attributed to the four phase columns; the columns are not a sum of total
runtime. All phases selected **307 sessions**, verified **494/494/495/495 current
turns**, and had zero failures. Exact paged live trace-ID sets/cardinality match
the report in every phase. All phases returned exit 0. Tests assert unchanged
IDs/maps, zero unchanged mutation
entrypoints, append reuse of all existing IDs/maps, and exactly one replaced
trace after editing the appended turn. A mutation entrypoint is a gateway
mutation authorization call, not an individual network request count.

### Memory and production replay gates

`tests/integration/test_scale_memory.py`: **3 passed in 310.73s**. The cases cover
100,000 records / 91.6 MiB artifact / **51.6 MiB peak traced importer Python heap**;
105,000 records / 241.4 MiB artifact / **16.0 MiB heap**; and explicit oversized-root
rejection. The structure/discarding sink excludes SDK one-turn payload retention;
these figures do not certify arbitrary live 100,000-span turns.
Separately, production historical-gateway Pi/Gemini 32/64-turn tree/replay cases
pass under a 32 MiB temporary quota with <32 MiB traced Python heap in the
non-slow integration suite. SQLite has its separate unchanged 64 MiB ceiling.

### Final suite and review evidence

Main-agent results **after the legacy-sorter fix, W1 correction and documentation refresh**:

- **1663 passed, 147 deselected in 38.76s** in the unit/contract/BDD selection.
- **130 passed, 1680 deselected in 172.74s** in non-slow integration/profile checks.
- Ruff lint/format clean (**168 Python files**); mypy clean (**64 source files**);
  whitespace check clean.
- Independent code/security/compatibility reviews reported no remaining findings;
  compatibility validation included 104 unit and 6 live checks. Final acceptance
  validation ran 99 live and 175 unit checks. W1 tie ordering and documentation
  findings were then independently closed with 5 focused tests; qualified-profile
  acceptance is approved. See
  [errors.md](errors.md#final-audit-warning-w1).

Final wheel and sdist builds passed with `python -m build --no-isolation`.
The earlier 511.827 / 113.937 / 112.882 / 113.831-second benchmark and
1044.86-second combined slow run are superseded historical evidence, not the
current performance claim.

## Unsupported Profiles

Different MLflow client/server versions, unqualified artifact schemes, unavailable
experiment-root repositories, NoOp span export, and targets without exact-ID
retirement or public named-artifact support are incompatible and must fail
before importer writes.

## Lazy result ownership

The qualified temporary-storage limits, shared invocation admission rules,
SQLite process heap setting and bounded file/proxy transports are specified in
[temporary-resources.md](temporary-resources.md). Resource exhaustion fails rather
than truncating or retiring prior verified work before its ready gate.

The CLI streams JSON from private SQLite result collections and releases their
storage after rendering. Programmatic callers of `run_import()` should call
`ImportResult.close()` after reading or rendering the result. Closing is
idempotent; the lazy report must not be read after it is closed. The compatibility
`render_json()` helper returns a complete string; use `reporting.iter_json()` for
bounded-memory serialization. The typed `compatibility-v1` result decoder supports
the two declared v1 envelopes and preserves the original schema. Historical
ownership/producer migration is qualified for `36ed44f`; unsupported historical
variants and unbound journals are not silently trusted. See
[result-schema.md](result-schema.md) and [operations.md](operations.md).
