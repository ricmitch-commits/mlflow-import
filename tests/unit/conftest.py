"""Shared fixtures for the importer, reporting, and CLI unit tests.

``source_db`` materializes a sanitized synthetic current-generation
source database with ONE selected session of TWO logical turns (the
minimum that exercises per-item failure isolation), ``config`` is the
frozen invocation over it, and ``import_dependencies`` wires the real
source pipeline to the in-memory MLflow/lock fakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opencode_mlflow.config import ImportConfig
from tests.support.import_fakes import (
    DEFAULT_SESSION_CREATED_MS,
    UNIT_PROMPT_JSON,
    UNIT_REPLY_JSON,
    UNIT_SESSION_ID,
    FakeImportDependencies,
    build_current_source_db,
    build_import_fakes,
)


@pytest.fixture
def source_db(tmp_path: Path) -> Path:
    """A current-generation source database with one two-turn session."""
    return build_current_source_db(
        tmp_path,
        sessions=[(UNIT_SESSION_ID, DEFAULT_SESSION_CREATED_MS)],
        messages=[
            ("unit-user-1", UNIT_SESSION_ID, "user", 1, UNIT_PROMPT_JSON),
            ("unit-assistant-1", UNIT_SESSION_ID, "assistant", 2, UNIT_REPLY_JSON),
            ("unit-user-2", UNIT_SESSION_ID, "user", 3, UNIT_PROMPT_JSON),
            ("unit-assistant-2", UNIT_SESSION_ID, "assistant", 4, UNIT_REPLY_JSON),
        ],
    )


@pytest.fixture
def config(source_db: Path) -> ImportConfig:
    """A valid import invocation over the fixture source (no optional filters)."""
    return ImportConfig(experiment_name="unit-import-experiment", db_path=source_db)


@pytest.fixture
def import_dependencies(restrictive_tmp_path: Path) -> FakeImportDependencies:
    """The importer dependencies: real source pipeline, fake MLflow/lock seams.

    Nothing needs teardown: the fake gateway and lock factory are
    in-memory, and ``run_import`` removes every temporary directory it
    created under the restrictive root.
    """
    return build_import_fakes(restrictive_tmp_path=restrictive_tmp_path)
