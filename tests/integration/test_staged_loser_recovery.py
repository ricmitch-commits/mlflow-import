"""Prepared/winner/incomplete-loser recovery on an explicitly local pinned server."""

import sqlite3
from dataclasses import replace
from http.client import RemoteDisconnected

import mlflow
import pytest
from mlflow.exceptions import MlflowException
from requests.exceptions import ConnectionError as RequestsConnectionError
from urllib3 import HTTPConnectionPool
from urllib3.exceptions import MaxRetryError

from mlflow_import.config import ImportConfig
from mlflow_import.errors import ItemImportError
from mlflow_import.importer import build_dependencies, exit_status, run_import
from mlflow_import.mlflow.attempts import AttemptJournal
from mlflow_import.reporting import render_json
from mlflow_import.retry import RetryPolicy, retry_budget
from mlflow_import.sources import Adapter
from tests.integration.server import live_mlflow_server, unique_experiment_name
from tests.support.import_fakes import build_current_source_db

pytestmark = [pytest.mark.integration, pytest.mark.profile]


def _sdk_disconnect():
    """The observed SDK exception chain, containing synthetic constants only."""
    try:
        raise RemoteDisconnected("synthetic disconnect")
    except RemoteDisconnected as disconnect:
        try:
            raise MaxRetryError(HTTPConnectionPool("synthetic.invalid"), "/synthetic", disconnect) from None
        except MaxRetryError as exhausted:
            try:
                raise RequestsConnectionError(exhausted) from None
            except RequestsConnectionError:
                try:
                    raise MlflowException("synthetic SDK disconnect") from None
                except MlflowException as wrapped:
                    return wrapped


@pytest.fixture(scope="module")
def staged_local_server():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLFLOW_TRACKING_URI", "http://ambient.invalid:1")
        with live_mlflow_server(requested_uri="") as server:
            assert server.process is not None
            assert server.uri.startswith("http://127.0.0.1:")
            assert mlflow.__version__ == "3.17.2"
            yield server


@pytest.mark.parametrize(
    "with_prior,fault",
    [
        (True, None),
        (False, None),
        (True, "cleanup"),
        (True, "before_ready"),
        (True, "winner"),
        (True, "snapshot"),
        (True, "combined"),
        *[(True, f"ready_{shape}") for shape in ("missing", "extra", "duplicate")],
        *[
            (True, f"abort_{path}_{fault}")
            for path in ("changed", "empty", "policy")
            for fault in ("none", "instance", "attempt_race", "state_race")
        ],
    ],
)
def test_prepared_staged_loser_default_source_recovery(staged_local_server, tmp_path, monkeypatch, with_prior, fault):
    server = staged_local_server
    source = build_current_source_db(
        tmp_path,
        sessions=[("staged-recovery", 1000)],
        messages=[
            ("u", "staged-recovery", "user", 1, '{"text":"synthetic prompt"}'),
            ("a", "staged-recovery", "assistant", 2, '{"text":"original reply"}'),
        ],
    )
    # Exercise the default-source preview/normal orchestration, with discovery
    # restricted to this synthetic database rather than any real source root.
    monkeypatch.setattr("mlflow_import.sources.ADAPTERS", {"opencode": Adapter("opencode", True)})
    monkeypatch.setattr(Adapter, "default_locations", lambda self: (source,))
    config = ImportConfig(unique_experiment_name("staged-recovery"), source, tracking_uri=server.uri)
    object.__setattr__(config, "_discover_defaults", True)
    object.__setattr__(config, "_on_conflict", "overwrite")
    dependencies = build_dependencies(config)
    prior_ids = set()
    if with_prior:
        first = run_import(config, dependencies)
        assert exit_status(first) == 0, render_json(first)
        prior_ids = {turn["trace_id"] for session in first.report["sessions"] for turn in session["turns"]}
        assert len(prior_ids) == 1
        with sqlite3.connect(source) as connection:
            connection.execute("UPDATE session_message SET data=? WHERE id='a'", ('{"text":"changed reply"}',))

    captured = {}
    publish = AttemptJournal.publish

    def seed_factory(dry_run):
        gateway = dependencies.gateway_factory(dry_run)
        create = gateway.create_trace

        def capture(request):
            created = create(request)
            if "mlflow_import.attempt_id" in request.root_tags:
                captured.update(request=request, winner=created.trace_id, create=create)
            return created

        gateway.create_trace = capture
        return gateway

    def stop_before_ready(journal, state, values):
        if state == "ready_to_retire":
            # Leave the real prepared record and fully verified winner intact;
            # create another same-attempt trace but never record its digest.
            assert set(values["prior_ids"]) == prior_ids
            loser = captured["create"](captured["request"])
            captured.update(loser=loser.trace_id, attempt_id=values["attempt_id"])
            raise ItemImportError("injected_staging_interruption", "synthetic interruption")
        return publish(journal, state, values)

    with monkeypatch.context() as patch:
        patch.setattr(AttemptJournal, "publish", stop_before_ready)
        seeded = run_import(config, replace(dependencies, gateway_factory=seed_factory))
    assert exit_status(seeded) == 1, seeded.report
    winner, loser = captured["winner"], captured["loser"]
    assert winner != loser
    for trace_id in (winner, loser):
        tags = server.client.get_trace(trace_id, display=False).info.tags
        assert tags["mlflow_import.attempt_id"] == captured["attempt_id"]
        assert tags["mlflow_import.attempt_state"] == "staged"
        assert bool(tags.get("opencode.artifact_sha256")) == (trace_id == winner)

    if fault and fault.startswith("abort_"):
        _check_abort_recovery(server, source, config, dependencies, prior_ids, captured, monkeypatch, fault)
        return

    events = []
    verify_ready = AttemptJournal.verify_ready
    validate = AttemptJournal.validate

    def checked_validate(journal, value, path=None):
        if (
            fault
            and fault.startswith("ready_")
            and value["state"] == "ready_to_retire"
            and value["attempt_id"] == captured["attempt_id"]
        ):
            assert journal.gateway.trace_exists(loser)
            references = list(value["replacements"])
            if fault == "ready_missing":
                references.clear()
            elif fault == "ready_extra":
                extra = {**references[0], "trace_id": "extra-winner", "metadata": dict(references[0]["metadata"])}
                for key in ("opencode.turn_key", "mlflow_import.turn_key"):
                    extra["metadata"][key] = "turn-" + "e" * 64
                references.append(extra)
            else:
                references.append(references[0])
            value = {**value, "replacements": references}
            events.append("invalid_replacement_set")
        return validate(journal, value, path)

    def checked_ready(journal, ready):
        result = verify_ready(journal, ready)
        events.append("ready_verified")
        return result

    def checked_publish(journal, state, values):
        if state == "ready_to_retire":
            assert not journal.gateway.trace_exists(loser)
            assert set(values["prior_ids"]) == prior_ids
            assert {item["trace_id"] for item in values["retirements"]} == prior_ids
            assert [item["trace_id"] for item in values["replacements"]] == [winner]
            if fault == "before_ready":
                raise ItemImportError("injected_ready_interruption", "synthetic interruption")
        published = publish(journal, state, values)
        if state == "ready_to_retire":
            events.append("ready_published")
        return published

    def recovery_factory(dry_run):
        gateway = dependencies.gateway_factory(dry_run)
        gateway._read_retry_policy = RetryPolicy(max_attempts=3, initial_delay_seconds=0, max_delay_seconds=0)
        retire = gateway.retire_trace
        verify = gateway.verify_replacement_reference
        reference = gateway.replacement_reference
        payload = gateway.verify_candidate_payload
        get_trace = gateway._client.get_trace
        search_traces = gateway._client.search_traces
        in_reference = False

        def checked_payload(candidate, request):
            result = payload(candidate, request)
            if candidate.trace_id == winner:
                events.append("candidate_verified")
            return result

        def checked_reference(trace_id):
            nonlocal in_reference
            if trace_id == winner and fault == "snapshot":
                assert "candidate_verified" in events
                gateway._client.set_trace_tag(winner, "mlflow_import.import_instance_id", "f" * 32)
                events.append("snapshot_scope_changed")
            in_reference = True
            try:
                return reference(trace_id)
            finally:
                in_reference = False

        def flaky_get(trace_id, **kwargs):
            if fault == "combined" and in_reference and trace_id == winner and "reference_disconnect" not in events:
                events.append("reference_disconnect")
                raise _sdk_disconnect()
            return get_trace(trace_id, **kwargs)

        def flaky_search(**kwargs):
            if fault == "combined" and "reference_disconnect" in events and "search_disconnect" not in events:
                assert not in_reference
                events.append("search_disconnect")
                raise _sdk_disconnect()
            return search_traces(**kwargs)

        def checked_verify(reference):
            if reference["trace_id"] == winner and fault == "winner":
                raise ItemImportError("replacement_verification_failed", "synthetic failure")
            verify(reference)
            if reference["trace_id"] == winner:
                events.append("winner_verified")

        def checked_retire(experiment_id, trace_id, **kwargs):
            assert "winner_verified" in events
            assert trace_id != winner
            if trace_id == loser:
                assert "ready_verified" not in events
                expected = kwargs["expected_metadata"]
                assert expected["mlflow_import.attempt_id"] == captured["attempt_id"]
                assert expected["mlflow_import.attempt_state"] == "staged"
                assert kwargs["session_id"] == "staged-recovery"
                if fault == "cleanup":
                    raise ItemImportError("injected_cleanup_failure", "synthetic failure")
            else:
                assert trace_id in prior_ids
                assert "ready_verified" in events
            retire(experiment_id, trace_id, **kwargs)
            events.append(trace_id)

        def unexpected_create(request):
            if "mlflow_import.attempt_id" in request.root_tags:
                pytest.fail("recovery must reuse the verified winner")
            return create(request)

        create = gateway.create_trace
        gateway.create_trace = unexpected_create
        gateway.retire_trace = checked_retire
        gateway.verify_replacement_reference = checked_verify
        gateway.verify_candidate_payload = checked_payload
        gateway.replacement_reference = checked_reference
        gateway._client.get_trace = flaky_get
        gateway._client.search_traces = flaky_search
        return gateway

    with monkeypatch.context() as patch:
        patch.setattr(AttemptJournal, "publish", checked_publish)
        patch.setattr(AttemptJournal, "verify_ready", checked_ready)
        patch.setattr(AttemptJournal, "validate", checked_validate)
        with retry_budget(max_retries=2) as budget:
            recovered = run_import(config, replace(dependencies, gateway_factory=recovery_factory))
            if fault == "combined":
                assert budget.retries_left == 0

    if fault and fault != "combined":
        assert exit_status(recovered) == 1, recovered.report
        assert "ready_published" not in events
        assert recovered.report["sessions"][0]["turns"][0]["attempt_state"] == "prepared"
        assert server.client.get_trace(winner, display=False).trace_id == winner
        for trace_id in prior_ids:
            assert server.client.get_trace(trace_id, display=False).trace_id == trace_id
        expected_cleanups = int(fault == "before_ready")
        assert recovered.report["summary"]["retired_traces"] == expected_cleanups
        if expected_cleanups:
            assert loser in events
            reported = {
                trace_id
                for session in recovered.report["sessions"]
                for turn in session["turns"]
                for trace_id in turn["retired_trace_ids"]
            }
            assert reported == {loser}
        else:
            assert server.client.get_trace(loser, display=False).trace_id == loser
        if fault == "snapshot":
            assert "snapshot_scope_changed" in events
            assert recovered.report["sessions"][0]["turns"][0]["error_code"] == "attempt_state_invalid"
            server.client.delete_trace_tag(winner, "mlflow_import.import_instance_id")
        if fault.startswith("ready_"):
            assert "invalid_replacement_set" in events
            assert recovered.report["sessions"][0]["turns"][0]["error_code"] == "attempt_state_invalid"
        recovered = run_import(config, dependencies)
    else:
        assert recovered.report["summary"]["retired_traces"] == len(prior_ids) + 1
        assert events.index(loser) < events.index("ready_published") < events.index("ready_verified")
        if fault == "combined":
            assert events.index("reference_disconnect") < events.index("search_disconnect") < events.index(loser)
            assert events.count("ready_published") == 1
            assert events.count(loser) == 1
            assert all(events.count(trace_id) == 1 for trace_id in prior_ids)

    assert exit_status(recovered) == 0, recovered.report
    turn = recovered.report["sessions"][0]["turns"][0]
    assert turn["trace_id"] == winner
    assert turn["attempt_state"] == "complete"
    experiment_id = recovered.report["experiment"]["id"]
    traces = server.client.search_traces(locations=[experiment_id], include_spans=False)
    assert {trace.trace_id for trace in traces} == {winner}
    unchanged = run_import(config, dependencies)
    assert exit_status(unchanged) == 0, unchanged.report
    assert unchanged.report["summary"]["unchanged_turns"] == 1, render_json(unchanged)
    assert unchanged.report["summary"]["retired_traces"] == 0
    assert unchanged.report["sessions"][0]["turns"][0]["trace_id"] == winner


def _check_abort_recovery(server, source, config, dependencies, prior_ids, captured, monkeypatch, fault):
    _, path, injected = fault.split("_", 2)
    with sqlite3.connect(source) as connection:
        if path == "empty":
            connection.execute("DELETE FROM session_message")
        elif path == "changed":
            connection.execute("UPDATE session_message SET data=? WHERE id='a'", ('{"text":"new desired reply"}',))
    if path == "policy":
        object.__setattr__(config, "_on_conflict", "ignore")
    staged_ids = {captured["winner"], captured["loser"]}
    changes = {}
    if injected == "instance":
        changes[captured["loser"]] = ("mlflow_import.import_instance_id", "f" * 32)
        server.client.set_trace_tag(captured["loser"], *changes[captured["loser"]])
    retire_calls = []

    def factory(dry_run):
        gateway = dependencies.gateway_factory(dry_run)
        retire = gateway.retire_trace

        def guarded_retire(experiment_id, trace_id, **kwargs):
            if trace_id in staged_ids:
                expected = kwargs["expected_metadata"]
                assert expected["mlflow_import.attempt_id"] == captured["attempt_id"]
                assert expected["mlflow_import.attempt_state"] == "staged"
                assert expected["mlflow_import.import_instance_id"] == ""
                assert expected["mlflow_import.harness"] == "opencode"
                if injected in {"attempt_race", "state_race"}:
                    key = "mlflow_import.attempt_id" if injected == "attempt_race" else "mlflow_import.attempt_state"
                    value = "f" * 32 if injected == "attempt_race" else "complete"
                    original_flush = mlflow.flush_trace_async_logging

                    def flush_then_change():
                        original_flush()
                        server.client.set_trace_tag(trace_id, key, value)
                        changes[trace_id] = (key, value)

                    with monkeypatch.context() as patch:
                        patch.setattr(mlflow, "flush_trace_async_logging", flush_then_change)
                        return retire(experiment_id, trace_id, **kwargs)
            result = retire(experiment_id, trace_id, **kwargs)
            retire_calls.append(trace_id)
            return result

        gateway.retire_trace = guarded_retire
        return gateway

    result = run_import(config, replace(dependencies, gateway_factory=factory))
    if injected != "none":
        assert exit_status(result) == 1, render_json(result)
        assert retire_calls == []
        assert result.report["summary"]["retired_traces"] == 0
        for trace_id in prior_ids | staged_ids:
            assert server.client.get_trace(trace_id, display=False).trace_id == trace_id
        for trace_id, (key, _) in changes.items():
            if key == "mlflow_import.import_instance_id":
                server.client.delete_trace_tag(trace_id, key)
            else:
                value = captured["attempt_id"] if key == "mlflow_import.attempt_id" else "staged"
                server.client.set_trace_tag(trace_id, key, value)
        result = run_import(config, dependencies)
    else:
        assert staged_ids <= set(retire_calls)
    assert exit_status(result) == 0, render_json(result)
    expected_retired = staged_ids if path == "policy" else staged_ids | prior_ids
    assert result.report["summary"]["retired_traces"] == len(expected_retired)
    traces = server.client.search_traces(locations=[result.report["experiment"]["id"]], include_spans=False)
    assert not {trace.trace_id for trace in traces} & expected_retired
    if path == "policy":
        assert {trace.trace_id for trace in traces} == prior_ids
    assert len(traces) == (0 if path == "empty" else 1)
