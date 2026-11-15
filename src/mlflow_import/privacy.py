"""Public privacy seam for deterministic redaction."""

import logging
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

from .content.redaction import *  # noqa: F403


@contextmanager
def safe_client_diagnostics() -> Iterator[None]:
    """Suppress SDK/transport exception bodies while emitting importer-owned codes.

    The one-shot CLI owns this process-wide logging boundary for the import.
    Restoring the prior threshold also covers interrupted and failed imports.
    """
    threshold = logging.root.manager.disable
    settings = {
        "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING": "false",
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "30",
    }
    previous = {key: os.environ.get(key) for key in settings}
    # Serial import has no throughput benefit from SDK batching. Configure
    # the supported synchronous exporter before the first span is created.
    os.environ.update(settings)
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(threshold)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
