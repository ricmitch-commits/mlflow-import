"""Exhaustion reporting uses only synthetic local sources and fake targets."""

import json
from dataclasses import replace

import pytest

from mlflow_import import cli, importer, resources
from mlflow_import.compatibility import decode_result
from mlflow_import.errors import CapabilityError, ImporterError
from mlflow_import.reporting import ReportBuilder, render_json
from mlflow_import.retry import _BUDGET, RetryPolicy, check_budget, retry_budget, retry_call, terminal_reporting

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("reason", ["deadline", "retries", "operations"])
def test_budget_reason_counters_and_sticky_exhaustion(reason):
    with retry_budget() as budget:
        if reason == "deadline":
            budget.deadline = -1
        elif reason == "retries":
            budget.retries_left = 0
        else:
            budget.operations_left = 0
        with pytest.raises(ImporterError) as caught:
            budget.check(retry=reason == "retries", operation=reason == "operations")
        error = caught.value
        assert error.code == ("operation_budget_exhausted" if reason == "operations" else "retry_budget_exhausted")
        assert error.diagnostics == {
            "reason": reason,
            "phase": "work",
            "retries_left": budget.retries_left,
            "operations_left": budget.operations_left,
        }
        expected = {"deadline": "deadline", "retries": "retry allowance", "operations": "work-operation"}
        assert expected[reason] in str(error)
        calls = []
        with pytest.raises(ImporterError) as repeated:
            retry_call(lambda: calls.append(True), RetryPolicy())
        assert repeated.value is error
        assert calls == []


@pytest.mark.parametrize("guard", ["sqlite", "file"])
@pytest.mark.parametrize("mutated", [False, True])
def test_primary_failure_survives_expiration_in_error_storage(
    config, import_dependencies, monkeypatch, guard, mutated
):
    primary = CapabilityError("synthetic_target_failure", "the target read could not be completed")

    def fail_pipeline(*args, **kwargs):
        resources.current_storage().phase2 = mutated
        raise primary from RuntimeError("PRIVATE exception/source/credential sentinel")

    original = ReportBuilder.add_error
    guarded = []

    def expire_recording(builder, code, message):
        budget = _BUDGET.get()
        if guard == "sqlite":
            budget.deadline = -1
            try:
                original(builder, code, message)
            except ImporterError:
                guarded.append("sqlite")
                raise
        else:
            handle, path = resources.open_temp_file(prefix="synthetic-report-", mode="w")
            try:
                with handle:
                    budget.deadline = -1
                    try:
                        handle.write("safe")
                    except ImporterError:
                        guarded.append("file")
                        raise
            finally:
                resources.remove_file(path)

    monkeypatch.setattr(importer, "_run_pipeline", fail_pipeline)
    monkeypatch.setattr(ReportBuilder, "add_error", expire_recording)
    with retry_budget() as work:
        result = importer.run_import(config, import_dependencies)
        assert work.retries_left == 64
        assert work.deadline == -1
        with pytest.raises(ImporterError):
            retry_call(lambda: pytest.fail("work restarted after exhaustion"), RetryPolicy())
    document = json.loads(render_json(result))
    assert decode_result(document).document == document
    assert guarded == [guard]
    assert result.error.code == primary.code
    assert importer.exit_status(result) == (1 if mutated else 2)
    assert document["summary"]["incomplete"] is True
    assert document["summary"]["counts_complete"] is False
    assert document["summary"]["recovery_pending"] is mutated
    assert document["summary"]["turns_imported"] is None
    assert "PRIVATE" not in render_json(result)
    assert import_dependencies.gateway.mutations == []


@pytest.mark.parametrize("output", ["human", "json"])
@pytest.mark.parametrize("mutated", [False, True])
def test_cli_deadline_after_work_before_lazy_output(
    config, import_dependencies, monkeypatch, capsys, output, mutated
):
    original = cli.run_import

    def finish_then_expire(config, dependencies):
        result = original(config, dependencies)
        assert result.error is None
        resources.current_storage().phase2 = mutated
        _BUDGET.get().deadline = -1
        return result

    monkeypatch.setattr(cli, "run_import", finish_then_expire)
    status = cli.main(
        ["--experiment-name", "synthetic", "--db", str(config.db_path), "--dry-run", "--output", output],
        import_dependencies,
    )
    captured = capsys.readouterr()
    assert status == (1 if mutated else 2)
    assert captured.err == ""
    if output == "json":
        document = json.loads(captured.out)
        assert decode_result(document).document == document
        assert document["errors"][0]["code"] == "retry_budget_exhausted"
        assert document["errors"][0]["diagnostics"]["retries_left"] == 64
        assert document["summary"]["recovery_pending"] is mutated
    else:
        assert "report: incomplete; counts unavailable" in captured.out
        assert "invocation deadline" in captured.out
    assert import_dependencies.gateway.mutations == []


@pytest.mark.parametrize("reason", ["deadline", "retries", "operations"])
def test_partial_lazy_json_is_never_published(config, import_dependencies, monkeypatch, capsys, reason):
    original = cli.iter_json
    calls = []

    def failing_renderer(result):
        if calls:
            yield from original(result)
            return
        calls.append(True)
        yield '{"PRIVATE_PARTIAL":'
        budget = _BUDGET.get()
        assert budget.terminal
        if reason == "deadline":
            budget.deadline = -1
        elif reason == "operations":
            budget.operations_left = 0
        budget.check(retry=reason == "retries", operation=reason == "operations")

    monkeypatch.setattr(cli, "iter_json", failing_renderer)
    status = cli.main(
        ["--experiment-name", "synthetic", "--db", str(config.db_path), "--dry-run", "--output", "json"],
        import_dependencies,
    )
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert decode_result(document).document == document
    assert status == 2
    assert document["summary"]["counts_complete"] is False
    assert document["errors"][0]["diagnostics"]["reason"] == reason
    assert document["errors"][0]["diagnostics"]["phase"] == "rendering"
    assert "PRIVATE_PARTIAL" not in captured.out
    assert captured.err == ""
    assert import_dependencies.gateway.mutations == []


def test_terminal_allowance_is_finite_and_does_not_reset_work():
    with retry_budget(max_retries=7, max_operations=3) as work:
        work.deadline = -1
        with terminal_reporting() as terminal:
            check_budget()
            assert terminal.retries_left == 0
            with pytest.raises(ImporterError, match="remote work is forbidden"):
                retry_call(lambda: pytest.fail("remote work during rendering"), RetryPolicy())
            terminal.operations_left = 0
            with pytest.raises(ImporterError):
                check_budget()
        assert _BUDGET.get() is work
        assert (work.retries_left, work.operations_left, work.deadline) == (7, 3, -1)
        with pytest.raises(ImporterError):
            check_budget()


@pytest.mark.parametrize("output", ["human", "json"])
def test_cli_preserves_primary_typed_error_when_reporting_expires(
    config, import_dependencies, monkeypatch, capsys, output
):
    original = ReportBuilder.add_error

    def fail_pipeline(*args, **kwargs):
        raise CapabilityError("synthetic_primary", "the primary target read failed") from RuntimeError("PRIVATE")

    def expire_recording(builder, code, message):
        _BUDGET.get().deadline = -1
        original(builder, code, message)

    monkeypatch.setattr(importer, "_run_pipeline", fail_pipeline)
    monkeypatch.setattr(ReportBuilder, "add_error", expire_recording)
    status = cli.main(
        ["--experiment-name", "synthetic", "--db", str(config.db_path), "--output", output], import_dependencies
    )
    captured = capsys.readouterr()
    assert status == 2
    assert "PRIVATE" not in captured.out + captured.err
    assert "synthetic_primary" in captured.out
    assert "the primary target read failed" in captured.out
    assert captured.err == ""
    if output == "json":
        assert json.loads(captured.out)["errors"][0]["code"] == "synthetic_primary"
    assert import_dependencies.gateway.mutations == []


@pytest.mark.parametrize("reason", ["retries", "operations"])
def test_exhaustion_is_reportable_when_the_error_store_is_also_guarded(
    config, import_dependencies, monkeypatch, reason
):
    def exhaust_pipeline(*args, **kwargs):
        budget = _BUDGET.get()
        if reason == "retries":
            budget.retries_left = 0
        else:
            budget.operations_left = 0
        budget.check(retry=reason == "retries", operation=reason == "operations")

    monkeypatch.setattr(importer, "_run_pipeline", exhaust_pipeline)
    result = importer.run_import(config, import_dependencies)
    document = json.loads(render_json(result))
    assert decode_result(document).document == document
    assert document["errors"][0]["diagnostics"]["reason"] == reason
    assert document["summary"]["counts_complete"] is False
    assert document["summary"]["turns_imported"] is None
    assert document["summary"]["failed_items"] is None
    assert import_dependencies.gateway.mutations == []


def test_output_byte_cap_falls_back_before_stdout(config, import_dependencies, monkeypatch, capsys):
    original = cli.iter_json
    calls = []

    def oversized(result):
        if calls:
            yield from original(result)
            return
        calls.append(True)
        for _ in range(9):
            yield " " * (1024 * 1024)

    monkeypatch.setattr(cli, "iter_json", oversized)
    status = cli.main(
        ["--experiment-name", "synthetic", "--db", str(config.db_path), "--dry-run", "--output", "json"],
        import_dependencies,
    )
    captured = capsys.readouterr()
    assert status == 2
    assert len(captured.out) < 4096
    assert json.loads(captured.out)["errors"][0]["code"] == "result_output_exhausted"
    assert import_dependencies.gateway.mutations == []


def test_run_import_uses_configured_timeout(config, import_dependencies, monkeypatch):
    observed = []

    def capture_pipeline(*args, **kwargs):
        from mlflow_import.retry import time

        observed.append(_BUDGET.get().deadline - time.monotonic())
        raise CapabilityError("synthetic_stop", "the synthetic pipeline stopped")

    monkeypatch.setattr(importer, "_run_pipeline", capture_pipeline)
    result = importer.run_import(replace(config, timeout_seconds=37), import_dependencies)
    try:
        assert result.error.code == "synthetic_stop"
        assert 36 < observed[0] <= 37
    finally:
        result.close()
