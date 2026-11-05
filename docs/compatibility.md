# MLflow Compatibility Profile

**Profile:** MLflow `3.17.2` client and server
**Verified:** 2026-11-05
**Scope:** OpenCode importer target

## Disposable Server

The profile tests launch the server from the installed MLflow distribution with:

```text
python -m mlflow server \
  --backend-store-uri sqlite:///<run-dir>/mlflow.db \
  --artifacts-destination file://<run-dir>/artifacts \
  --host 127.0.0.1 \
  --port <free-port>
```

The server is health-checked at `/health`, uses a fresh temporary directory,
and is terminated and removed after the test session. The verified artifact
repository is the same-host `file://` experiment artifact root. The importing
process must be able to access that local root; remote artifact schemes are not
qualified by this profile.

## Verified Operations

`tests/contract/test_mlflow_profile.py` passed 5 tests against the disposable
server. The evidence covers:

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
| Historical spans | `mlflow.start_span_no_context` and `LiveSpan.end` |
| Full trace read | `MlflowClient.get_trace` |
| Paged search | `MlflowClient.search_traces` with `locations` and `page_token` |
| Tags | `MlflowClient.set_trace_tag` and `delete_trace_tag` |
| Retirement | `MlflowClient.delete_traces(..., trace_ids=[...])` |
| Named artifacts | `mlflow.artifacts.get_artifact_repository` plus public repository methods |

The verified inline payload envelope is 1 MiB. Larger values must use the
named-artifact path; values are never silently truncated. No MLflow runs are
created by the profile or importer boundary.

## Unsupported Profiles

Different MLflow client/server versions, artifact schemes, unavailable
experiment-root repositories, NoOp span export, and targets without exact-ID
retirement or public named-artifact support are incompatible and must fail
before importer writes.
