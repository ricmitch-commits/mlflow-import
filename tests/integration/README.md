# Integration environment: pinned MLflow compatibility profile

The importer's MLflow writer is gated on a pinned compatibility profile. The
pinned pair is the **client and server from the same `mlflow==3.17.2`
distribution** — the exact stable release of the API line the plan referred to
as `3.17.2.dev0`, which is not published on PyPI (controller ruling,
2026-10-21). There are no mutable `latest` images or tags anywhere in this
recipe; everything runs from the installed distribution in `.venv`.

The profile is verified by `tests/contract/test_mlflow_profile.py` (markers
`profile` and `integration`); the verification library lives in
`tests/integration/mlflow_profile.py` and the resolved public API surface is
recorded in `docs/compatibility.md`. Every profile run first performs a
read-only `GET <tracking-base>/version` attestation and requires the exact
plain-text (or documented JSON) value `3.17.2`; a mismatched, unavailable, or
malformed endpoint is a profile failure.

The importer's gateway boundary (`src/opencode_mlflow/mlflow/gateway.py`,
`src/opencode_mlflow/mlflow/capabilities.py`) does **not** need this server
for its own tests: `tests/unit/mlflow/` and
`tests/contract/test_gateway_contract.py` drive the real gateway against
in-memory fakes (`tests/support/fake_gateway.py`) that mirror the behavioral
facts this profile pinned (tag/metadata truncation limits, pagination
tokens, `tags.`/`metadata.` filter strings, the `mlflow.artifactLocation`
tag, MLflow-shaped protocol errors). The live profile test above remains
the authoritative evidence that the pinned pair satisfies the writer
contract, and the gateway contract test pins its constants and public-API
call set to this profile.

## Live-server integration modules (Task 14)

Beyond the profile contract, two integration modules verify the COMPLETE
import pipeline against a real server:

- `tests/integration/test_mlflow_trace_contract.py` — one real import of
  the sanitized synthetic six-turn session, then verification through the
  public client API: exact experiment creation/reuse, one trace per
  logical turn, six distinct trace ids sharing one native session id
  (`mlflow.trace.session`), FULL root payload retrieval through
  `get_trace`, nested model/tool/content spans and events with source
  ids, status/timing/token/cost metadata, and deterministic artifact
  paths with exact-byte verification (including re-streaming the document
  from the retrieved identity map).
- `tests/integration/test_mlflow_reconciliation.py` — dry-run
  non-mutation (three-axis server-state snapshot), unchanged reruns with
  no duplicate owned traces and Ruling 16 recorded-digest verification,
  changed-turn replacement order (create/verify/upload/verify/record then
  retire, asserted through gateway-boundary event capture plus observable
  server state), retirement-failure preservation of the old trace
  (failure injected at the gateway boundary, never by breaking the
  server), scoped stale-turn retirement, and untouched non-importer
  traces.

Both consume the session-scoped `live_server` fixture (defined in
`tests/conftest.py`, machinery in `tests/integration/server.py`), which
follows the same dual-mode pattern as the profile test: it uses
`MLFLOW_TRACKING_URI` when it names a healthy HTTP(S) server and otherwise
launches one disposable server for the whole test session — so a full
`pytest -m integration` run needs no external configuration at all. The
real-pipeline runner both modules share is `tests/integration/import_runner.py`
(production seams everywhere; only the report clock and retry policy are
deterministic).

`tests/integration/test_wal_snapshot.py` (live-WAL snapshot semantics,
including a commit landing mid-extraction) and
`tests/integration/test_scale_memory.py` (105k records / 241 MiB streamed
artifact / <256 MiB heap growth, plus the `payload_too_large` rejection at
the real 1 MiB inline limit) need no server.

## 1. Create the environment

```bash
cd "$(git rev-parse --show-toplevel)"
python3 -m venv .venv
# Pin the profile pair plus the project's dev tooling:
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r tests/integration/constraints.txt -e ".[dev]"
# (or, with uv: uv pip install -r tests/integration/constraints.txt -e ".[dev]")
```

Verify the pin:

```bash
.venv/bin/mlflow --version          # must print: mlflow, version 3.17.2
.venv/bin/python -c "import mlflow; print(mlflow.__version__)"
```

## 2. Launch the disposable server

Choose a fresh, unique run directory under the pre-approved temp dir and a
free port. `--serve-artifacts` is enabled by default in 3.17.2, so artifacts
are proxied through the server to `--artifacts-destination`:

```bash
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/opencode/opencode-mlflow-profile-XXXXXX")"
mkdir -p "$RUN_DIR/artifacts"
.venv/bin/mlflow server \
  --backend-store-uri "sqlite:///$RUN_DIR/mlflow.db" \
  --artifacts-destination "file://$RUN_DIR/artifacts" \
  --host 127.0.0.1 \
  --port 8791 \
  >"$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
```

Health-check before running anything (retry until it prints `OK`):

```bash
curl --fail --silent http://127.0.0.1:8791/health
curl --fail --silent http://127.0.0.1:8791/version   # must print: 3.17.2
```

If port 8791 is taken, pick another free port (the committed fixture chooses
one programmatically and retries on bind failure).

## 3. Run the profile contract test and the integration suite

```bash
PATH="$PWD/.venv/bin:$PATH" \
MLFLOW_TRACKING_URI="http://127.0.0.1:8791" \
.venv/bin/python -m pytest tests/contract/test_mlflow_profile.py \
  -m 'profile and integration' -q

# The Task 14 real-server suite (self-launches its own disposable server
# when MLFLOW_TRACKING_URI is unset):
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest tests/integration \
  -m integration -vv
```

`MLFLOW_TRACKING_URI` must point at a live server and must be an HTTP(S)
URI whose `/version` endpoint attests `3.17.2`; a non-HTTP value, version
mismatch, malformed/missing endpoint, or unreachable server fails the session loudly —
the profile (and every live-server integration module) never silently
skips. All tests must pass.

### Self-launched mode

If `MLFLOW_TRACKING_URI` is unset, the test session launches its own
disposable server from the installed distribution (fresh run directory under
`$TMPDIR/opencode`, free port, SQLite backend, local artifact root),
health-checks it, and always reaps it on exit — including on failure. The
session-scoped `live_server` fixture (`tests/integration/server.py`) shares
ONE server across the profile contract test and every live-server
integration module, so a full run still launches at most one.

## 4. Cleanup

```bash
kill "$SERVER_PID"            # always, even after a failed run
rm -rf "$RUN_DIR"             # disposable: database, artifacts, and server log
```

## Environment hazards the harness already handles

- **Ambient `OTEL_EXPORTER_OTLP_*` variables** reroute MLflow span export to
  an external OTLP collector; without the `opentelemetry-exporter-otlp-proto-http`
  package installed, `mlflow.start_span_no_context` silently degrades to NoOp
  spans. The test fixture and `verify_profile` scrub these variables before
  the tracing provider initializes.
- **Span export follows the global tracking URI** (`mlflow.set_tracking_uri` /
  `MLFLOW_TRACKING_URI`), not an individual `MlflowClient`'s URI. The harness
  pins the global URI to the server under test.
- **Trace export is asynchronous**; the harness uses the public
  `flush=True` parameter on `get_trace`/`search_traces` (and
  `mlflow.flush_trace_async_logging()`) so observations are deterministic.
