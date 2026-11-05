"""Scaffold smoke test: package metadata, install, and console script."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_package_metadata_and_entry_point_import():
    import importlib.metadata
    import subprocess

    import mlflow_import

    assert mlflow_import.__version__
    assert importlib.metadata.version("mlflow-import") == mlflow_import.__version__
    assert subprocess.run(["mlflow-import", "--help"], check=False, capture_output=True).returncode == 0
