"""Scaffold smoke test: package metadata, install, and console script."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_package_metadata_and_entry_point_import():
    import importlib.metadata
    import subprocess
    import opencode_mlflow

    assert opencode_mlflow.__version__
    assert importlib.metadata.version("opencode-mlflow") == opencode_mlflow.__version__
    assert subprocess.run(["opencode-mlflow", "--help"], check=False, capture_output=True).returncode == 0
