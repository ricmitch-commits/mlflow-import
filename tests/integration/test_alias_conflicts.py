"""Resolved plugin/legacy conflicts obey session policy on disposable targets."""

import json
import sqlite3
from collections import Counter
from dataclasses import replace

import mlflow
import pytest
from mlflow.tracing.client import TracingClient

from mlflow_import.config import ImportConfig, parse_config
from mlflow_import.errors import ItemImportError
from mlflow_import.importer import build_dependencies, exit_status, run_import
from mlflow_import.sources import Adapter
from tests.integration.server import unique_experiment_name
from tests.support.import_fakes import build_current_source_db

pytestmark = pytest.mark.integration


def _setup(live_server, tmp_path, policy, alias, location="metadata"):
    source = build_current_source_db(
        tmp_path,
        sessions=[("plugin", 1000)],
        messages=[
            ("u1", "plugin", "user", 1, '{"text":"synthetic prompt"}'),
            ("a1", "plugin", "assistant", 2, '{"text":"synthetic answer"}'),
        ],
    )
    config = ImportConfig(unique_experiment_name("foreign-alias"), source, tracking_uri=live_server.uri)
    object.__setattr__(config, "_on_conflict", policy)
    experiment = live_server.client.create_experiment(config.experiment_name)
    span = mlflow.start_span_no_context(name="plugin-copy", experiment_id=experiment)
    span.end(outputs={"text": "plugin answer"})
    info = live_server.client.get_trace(span.trace_id, display=False).info
    if location == "metadata":
        TracingClient(tracking_uri=live_server.uri).start_trace(
            replace(info, trace_metadata={**info.trace_metadata, alias: "plugin"})
        )
    else:
        live_server.client.set_trace_tag(span.trace_id, alias, "plugin")
    matches = live_server.client.search_traces(
        locations=[experiment], filter_string=f"{location}.`{alias}` = 'plugin'", include_spans=False
    )
    assert [trace.trace_id for trace in matches] == [span.trace_id]
    return config, experiment, span.trace_id


@pytest.mark.parametrize("alias", ["mlflow.trace.session", "opencode.session_id"])
@pytest.mark.parametrize("policy", ["overwrite", "ignore", "duplicate"])
@pytest.mark.parametrize("location", ["metadata", "tags"])
def test_foreign_alias_session_policy(live_server, tmp_path, policy, alias, location):
    config, experiment, foreign = _setup(live_server, tmp_path, policy, alias, location)
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    traces = live_server.client.search_traces(locations=[experiment], include_spans=False)
    ids = {trace.trace_id for trace in traces}
    if policy == "overwrite":
        assert foreign not in ids
        assert len(ids) == 1
        assert result.report["summary"]["retired_traces"] == 1
    elif policy == "ignore":
        assert ids == {foreign}
        assert result.report["sessions"][0]["turns"][0]["reason"] == "session_conflict_ignored"
    else:
        assert foreign in ids
        assert len(ids) == 2


def test_failed_replacement_preserves_foreign_alias(live_server, tmp_path):
    config, experiment, foreign = _setup(live_server, tmp_path, "overwrite", "mlflow.trace.session")
    dependencies = build_dependencies(config)

    def factory(dry_run):
        gateway = dependencies.gateway_factory(dry_run)

        def fail_create(request):
            raise ItemImportError("injected_create_failure", "synthetic failure")

        gateway.create_trace = fail_create
        return gateway

    result = run_import(config, replace(dependencies, gateway_factory=factory))
    assert exit_status(result) == 1
    assert live_server.client.get_trace(foreign, display=False).trace_id == foreign
    resumed = run_import(config, dependencies)
    assert exit_status(resumed) == 0, resumed.report
    ids = {trace.trace_id for trace in live_server.client.search_traces(locations=[experiment], include_spans=False)}
    assert len(ids) == 1
    assert foreign not in ids


@pytest.mark.parametrize("policy", ["overwrite", "ignore", "duplicate"])
def test_zero_turn_foreign_conflict_policy(live_server, tmp_path, policy):
    config, experiment, foreign = _setup(live_server, tmp_path, policy, "mlflow.trace.session")
    with sqlite3.connect(config.db_path) as connection:
        connection.execute("UPDATE session_message SET type='system'")
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    ids = {trace.trace_id for trace in live_server.client.search_traces(locations=[experiment], include_spans=False)}
    assert ids == (set() if policy == "overwrite" else {foreign})
    assert result.report["summary"]["retired_traces"] == (1 if policy == "overwrite" else 0)


def _default_roots(tmp_path, monkeypatch, *, same_id=False):
    roots = {"claude": tmp_path / "claude", "pi": tmp_path / "pi"}
    for directory in roots.values():
        directory.mkdir()
    records = {
        "claude": [
            {
                "sessionId": "plugin" if same_id else "late",
                "type": "user",
                "uuid": "u",
                "timestamp": "2026-02-01T00:00:00Z",
            }
        ],
        "pi": [
            {
                "type": "session",
                "version": 3,
                "id": "plugin" if same_id else "early",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {"type": "message", "id": "u", "parentId": None, "message": {"role": "user", "content": "synthetic"}},
        ],
    }
    for name, rows in records.items():
        (roots[name] / "session.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        Adapter, "default_locations", lambda self, session_id=None: (roots.get(self.name, tmp_path / "absent"),)
    )


def test_default_roots_write_in_global_order(live_server, tmp_path, monkeypatch):
    _default_roots(tmp_path, monkeypatch)
    config = parse_config(
        ["--experiment-name", unique_experiment_name("global-order"), "--tracking-uri", live_server.uri]
    )
    dependencies = build_dependencies(config)
    created = []

    def factory(dry_run):
        gateway = dependencies.gateway_factory(dry_run)
        create = gateway.create_trace

        def record(request):
            created.append(request.root_tags["mlflow_import.session_id"])
            return create(request)

        gateway.create_trace = record
        return gateway

    result = run_import(config, replace(dependencies, gateway_factory=factory))
    assert exit_status(result) == 0, result.report
    assert created == ["early", "late"]
    assert [session["session_id"] for session in result.report["sessions"]] == created
    assert result.report["summary"]["imported_turns"] == 2
    expected_coverage = Counter(
        warning["code"].removeprefix("coverage_unavailable_")
        for session in result.report["sessions"]
        for turn in session["turns"]
        for warning in turn["warnings"]
        if warning["code"].startswith("coverage_unavailable_")
    )
    assert result.report["summary"]["coverage"]["unavailable_turn_counts"] == dict(expected_coverage)


def test_all_default_identities_disambiguate_aliases_before_any_write(live_server, tmp_path, monkeypatch):
    _, experiment, foreign = _setup(live_server, tmp_path, "overwrite", "mlflow.trace.session")
    name = live_server.client.get_experiment(experiment).name
    _default_roots(tmp_path, monkeypatch, same_id=True)
    config = parse_config(["--experiment-name", name, "--tracking-uri", live_server.uri])
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 2, result.report
    assert result.error.code == "alias_ambiguous"
    traces = live_server.client.search_traces(locations=[experiment], include_spans=False)
    assert {trace.trace_id for trace in traces} == {foreign}
