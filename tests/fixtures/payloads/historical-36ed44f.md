# Historical synthetic compatibility capture

`historical-36ed44f.json` captures the result, ownership metadata and artifact
shapes from repository revision `36ed44f`. Sources inspected with `git show`:

- `docs/result-schema.md`: the explicitly illustrative result example and
  ownership/artifact-path contract (not user data).
- `src/opencode_mlflow/content/artifacts.py`: streamed artifact section shape.
- `src/opencode_mlflow/source/discovery.py`: `mapping-v4` / `artifact-v2`.
- `tests/unit/test_reporting.py` and `tests/unit/mlflow/test_reconciliation.py`:
  synthetic fixture conventions and complete ownership qualification.

All identifiers, hashes, experiment names and transcript values in the capture
are synthetic substitutions. The original schema and producer marker are
retained. No external source database or developer sessions were used.

The decoder retains the original wire schema/marker. Recognition of the historical
marker is an index hint, not deletion authority: migration still requires a full
ownership contract, unambiguous qualified source association and the same verified
replacement/ready retirement gate. V1 payloads are replaced rather than declared
equivalent to V2 projections; their historical artifact objects are retained.
