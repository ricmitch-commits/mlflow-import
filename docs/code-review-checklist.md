# Code review checklist

Every change to `src/opencode_mlflow/` (or to the contracts the tests
pin) must be reviewed against this checklist. A "no" to any question is
a change request, not a comment. Each item names the invariant and
where it lives.

## Source write safety

- [ ] The live source is only ever opened through
      `open_live_connection` (`source/connection.py`): `mode=ro` URI
      (never `immutable=1`), `PRAGMA query_only=ON`, bounded busy
      timeout, exactly one deferred `BEGIN`.
- [ ] No new code writes, checkpoints, vacuums, migrates, or deletes the
      source database, or holds a live connection open past extraction.
- [ ] Snapshot/plan/staging files are created with mode `0o700`
      directories / `0o600` files and cleaned up in `finally` paths.

## SQL parameterization and catalog ownership

- [ ] All new source SQL is parameterized; no string interpolation of
      values into SQL.
- [ ] Table/column names come from the schema catalog
      (`source/catalog.py`), not from unvalidated input or discovery at
      runtime. A new source column/table means a catalog change plus a
      `docs/schema-catalog-v1.md` update (both generations considered).
- [ ] The excluded tables (`account`, `account_state`,
      `control_account`, `credential`) and the `session_share` omission
      are still never selected. If a new safe column is being added for
      `session_share`, that requires an explicit reviewed catalog entry
      — not a drive-by.

## Schema compatibility

- [ ] A schema-layout change fails loudly with `unsupported_schema`
      (exit 2) and a safe message; unknown additive columns/keys stay
      tolerated with warnings while values are retained verbatim.
- [ ] Legacy, current, and mixed precedence semantics are unchanged
      unless the change intends to change them (current wins overlapping
      ids; legacy-only ids export; duplicate ids never merge).
- [ ] `docs/schema-catalog-v1.md` still mirrors `source/catalog.py`
      exactly.

## Redaction coverage

- [ ] New content paths run redaction **before** persistence (the
      snapshot callback boundary), never after.
- [ ] Any new value shape fails closed: a value the pipeline cannot
      inspect becomes `[REDACTED_BINARY]`, never a pass-through.
- [ ] Identity/sequence fields stay protected — the single
      `PROTECTED_IDENTITY_FIELDS` set in `contracts.py` is the only
      definition; protection remains root/path exact and never exempts
      arbitrary transcript subtrees by field name.
- [ ] New detector behavior updates `docs/security.md`'s detector table
      (including accepted gaps) and the positive/negative fixture
      vectors.

## Report sanitization

- [ ] Every new report/error field carries identifiers, counts, codes,
      hashes, timestamps, or the experiment name only — never prompt,
      response, tool, or source-code content, absolute source paths,
      credentials, tracking-URI secrets, or matched secrets.
- [ ] Safe messages never echo the offending value; `str()` of any
      error yields only the safe message.
- [ ] `docs/result-schema.md` is updated for any result-field change,
      and field names match the serializer in `src/opencode_mlflow/
      reporting.py` exactly.

## Bounded memory

- [ ] No new code materializes a session, a turn, or a whole artifact
      as an in-memory value; cursors stream, summaries release completed
      turns, uploads read fixed chunks (8 MiB), verification streams
      fixed blocks (64 KiB).
- [ ] A complete root value above the inline limit is rejected with
      `payload_too_large` **before trace creation** — never truncated.
      Oversized child values become digest/size references.
- [ ] The per-turn cursor cost trade-off is acknowledged: re-reading
      the snapshot is the sanctioned price for flat memory; do not
      "optimize" it by caching session content in memory.

## Public MLflow APIs only

- [ ] Only `mlflow/gateway.py` and `mlflow/capabilities.py` import
      MLflow (the AST-scanning contract test enforces this); no new
      module imports `mlflow`.
- [ ] Only public `mlflow`/`mlflow.tracking`/`mlflow.artifacts` entry
      points are used — no private/undocumented modules, no backend-store
      writes, no MLflow runs, no UUID-only attachment channels or
      run-artifact fallbacks.
- [ ] Payloads never ride trace tags or metadata (silent 4096/250
      truncation); new trace facts ride span payloads, attributes
      within limits, or artifacts.
- [ ] Non-root spans never carry `experiment_id` (real 3.17.2
      degradation, mirrored in the fakes); `LiveSpan.end`'s non-ERROR →
      `OK` coercion and the root-status→state mapping are not
      re-derived ad hoc.

## Replacement ordering and ownership

- [ ] The strict order holds: verify → create → verify → upload →
      verify → record digest → retire old. No retirement before a
      verified winner exists.
- [ ] A failed create/verification/upload rolls back the NEW trace and
      retains the old one; the old trace is never deleted on
      uncertainty.
- [ ] Unowned traces are never yielded, updated, or retired; every
      search page is re-verified locally against the exact ownership
      marker.
- [ ] Stale retirement stays scoped to selected, completely scanned
      sessions; everything else stays `unexamined_owned_traces`.
- [ ] Ownership keys are `(session_id, turn_key)` — never the native
      MLflow trace id.

## Retry and idempotency

- [ ] Typed `ImporterError`s stay permanent; only the documented
      transient vocabulary (5xx/rate-limit protocol codes,
      timeout/connection class names) is retried; control-flow
      interruptions never are.
- [ ] New mutating calls are safe under bounded retry: they either run
      under the deterministic `opencode.operation_key` idempotency key
      or are verified-after-write; an ambiguous create searches every page
      for the exact owner/session/turn/render/operation/source contract,
      distinguishes failed search from authoritative empty search, and never
      creates again around an unverifiable survivor.
- [ ] The advisory lock is still non-blocking, credential-free, never
      auto-broken, and skipped for dry runs.

## Tests

- [ ] The change follows the six-step loop (failing test first; see
      [testing.md](testing.md)); the narrow test and then the task
      group were run.
- [ ] The default selection passes:
      `python -m pytest -m '(unit or contract or bdd) and not integration'`
      (baseline 1363 passed, 32 deselected) with zero warnings.
- [ ] `ruff check .`, `ruff format --check .`, and `mypy src` are clean.
- [ ] Behavior that touches the MLflow boundary has both a fake-backed
      test (fakes mirror the pinned pair's observed behavior) and, when
      it changes wire behavior, a live-server integration test; the
      fakes were updated in lockstep, not loosened.
- [ ] Fixtures are sanitized synthetic data only; no real credentials
      or transcript content (see the provenance rules in
      [testing.md](testing.md)).

## Docs

- [ ] Every documented CLI flag/default still matches
      `src/opencode_mlflow/config.py`; every documented result field
      still matches the serializer; every referenced path exists.
      `scripts/verify_docs.py` passes (it checks all three, plus the
      schema-catalog mirror and the detector/protected-field sets).
- [ ] Changes to contracts update the matching document in the same
      commit: [result-schema.md](result-schema.md),
      [schema-catalog-v1.md](schema-catalog-v1.md),
      [security.md](security.md), [compatibility.md](compatibility.md),
      [operations.md](operations.md), [testing.md](testing.md),
      [architecture.md](architecture.md), and
      [CHANGELOG.md](../CHANGELOG.md) as appropriate.
- [ ] No documentation promises out-of-scope behavior: no OpenCode
      HTTP API, no watch mode, no arbitrary filter expressions, no
      MLflow run container, no ORM, no private MLflow APIs.
