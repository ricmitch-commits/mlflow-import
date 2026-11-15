"""Quoted/prose/truncated content stays private through public live readback."""

import json
from contextlib import closing
from pathlib import Path

import pytest
from mlflow.artifacts import get_artifact_repository

from mlflow_import.config import parse_config
from mlflow_import.importer import build_dependencies, exit_status, run_import
from tests.integration.server import unique_experiment_name
from tests.integration.test_adapter_parity import source_file

pytestmark = pytest.mark.integration


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


@pytest.mark.parametrize("redact", [False, True])
def test_mixed_encoded_fragments_live_artifacts_and_spans(live_server, tmp_path, redact):
    encoded = json.dumps({"user_id": "fixture-user", "password": "fixture-secret"})
    deeper = json.dumps("inner message: " + json.dumps("nested prose " + encoded))
    prompt = "\n".join(
        [
            "prefix " + json.dumps("message " + encoded) + " suffix",
            "prefix " + json.dumps(encoded)[:-1],
            "prefix " + deeper + " suffix",
            "prefix " + json.dumps("wrapper " + deeper[:-2]) + " suffix",
        ]
    )
    path, _ = source_file(tmp_path, "pi")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[1]["message"]["content"] = prompt
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("quoted-fragment-privacy"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
            *(["--redact"] if redact else []),
        ]
    )
    with closing(run_import(config, build_dependencies(config))) as result:
        assert exit_status(result) == 0, result.report
        trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
        repository = get_artifact_repository(
            live_server.client.get_experiment(result.report["experiment"]["id"]).artifact_location,
            tracking_uri=live_server.uri,
        )
        artifact = Path(repository.download_artifacts(trace.info.tags["mlflow_import.session_artifact"], str(tmp_path)))
        bundle = artifact.read_text()
        spans = json.dumps([[span.inputs, span.outputs, span.attributes] for span in trace.data.spans], default=str)
        for content in (bundle, spans):
            for sensitive in ("fixture-user", "fixture-secret"):
                assert (sensitive in content) is not redact
        if redact:
            assert "[REDACTED]" in bundle and "[REDACTED]" in spans
        else:
            root = next(span for span in trace.data.spans if span.parent_id is None)
            assert prompt in list(_strings(root.inputs))
