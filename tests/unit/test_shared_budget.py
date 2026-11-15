import pytest

from mlflow_import.errors import ItemImportError
from mlflow_import.retry import RetryPolicy, check_budget, retry_budget, retry_call

pytestmark = pytest.mark.unit


def test_nested_operations_share_retry_allowance():
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        raise TimeoutError("synthetic")

    with retry_budget(max_retries=2, deadline_seconds=60):
        with pytest.raises(ItemImportError) as caught:
            retry_call(
                lambda: retry_call(fail, RetryPolicy(), sleep=lambda _: None), RetryPolicy(), sleep=lambda _: None
            )
        assert caught.value.code == "retry_budget_exhausted"
    assert calls == 3


def test_nested_operations_cannot_reset_the_work_allowance():
    with retry_budget(max_operations=2):
        check_budget()
        with retry_budget(max_operations=100):
            check_budget()
            with pytest.raises(ItemImportError) as caught:
                check_budget()
    assert caught.value.code == "operation_budget_exhausted"


def test_transport_bounds_are_restored_after_failure(monkeypatch):
    import os

    from mlflow_import.privacy import safe_client_diagnostics

    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "99")
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)
    with pytest.raises(RuntimeError), safe_client_diagnostics():
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "0"
        assert float(os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"]) <= 30
        raise RuntimeError("synthetic failure")
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "99"
    assert "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ
