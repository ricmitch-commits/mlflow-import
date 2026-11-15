"""Small real four-phase counterpart to the slow reference workload assertions."""

import sqlite3
from contextlib import closing

import pytest

from mlflow_import.config import ImportConfig
from mlflow_import.importer import build_dependencies, exit_status, run_import
from mlflow_import.mlflow.gateway import MlflowGateway
from tests.integration.server import unique_experiment_name
from tests.support.import_fakes import build_current_source_db
from tests.support.scale_assertions import assert_imported_workload

pytestmark = pytest.mark.integration


def test_small_reference_phases_preserve_full_accounting_and_cardinality(live_server, tmp_path, monkeypatch):
    source = build_current_source_db(
        tmp_path,
        sessions=[("one", 1000), ("two", 1001)],
        messages=[
            ("u1", "one", "user", 1, '{"text":"synthetic first"}'),
            ("a1", "one", "assistant", 2, '{"text":"synthetic reply"}'),
            ("u2", "one", "user", 3, '{"text":"synthetic second"}'),
            ("a2", "one", "assistant", 4, '{"text":"synthetic reply"}'),
            ("u3", "two", "user", 1, '{"text":"synthetic third"}'),
            ("a3", "two", "assistant", 2, '{"text":"synthetic reply"}'),
        ],
    )
    config = ImportConfig(unique_experiment_name("small-scale-accounting"), source, tracking_uri=live_server.uri)
    dependencies = build_dependencies(config)
    writes = 0
    require = MlflowGateway._require_mutation_allowed

    def mutation(gateway):
        nonlocal writes
        writes += 1
        return require(gateway)

    monkeypatch.setattr(MlflowGateway, "_require_mutation_allowed", mutation)
    previous_ids, previous_maps = {}, {}
    for phase in ("fresh", "unchanged", "append", "edit"):
        if phase in {"append", "edit"}:
            with closing(sqlite3.connect(source)) as db, db:
                if phase == "append":
                    db.execute(
                        "INSERT INTO session_message VALUES(?,?,?,?,?,?,?)",
                        (
                            "u4",
                            "two",
                            "user",
                            99,
                            2000,
                            2000,
                            '{"text":"synthetic append"}',
                        ),
                    )
                else:
                    db.execute("UPDATE session_message SET data=? WHERE id='u4'", ('{"text":"synthetic edit"}',))
        writes = 0
        with closing(run_import(config, dependencies)) as result:
            assert exit_status(result) == 0, result.report
            rows = live_server.client.search_traces(locations=[result.report["experiment"]["id"]], include_spans=False)
            ids = {
                turn["turn_key"]: turn["trace_id"]
                for session in result.report["sessions"]
                for turn in session["turns"]
                if turn["source_present"]
            }
            maps = {trace.trace_id: trace.info.tags["opencode.artifact_path"] for trace in rows}
            assert_imported_workload(result, 2, 3 if phase in {"fresh", "unchanged"} else 4, maps)
            if phase == "unchanged":
                assert writes == 0 and ids == previous_ids and maps == previous_maps
            elif phase == "append":
                assert previous_ids.items() <= ids.items()
            elif phase == "edit":
                assert sum(ids[key] != previous_ids[key] for key in ids) == 1
            assert all(maps[trace_id] == path for trace_id, path in previous_maps.items() if trace_id in maps)
            previous_ids, previous_maps = ids, maps
