"""Unit tests for the shared test-support seams (fixtures and builders)."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from tests.support.fake_clock import FakeClock
from tests.support.sqlite_builder import build_sqlite_db

pytestmark = pytest.mark.unit


def test_restrictive_tmp_path_is_owner_only(restrictive_tmp_path: Path) -> None:
    mode = stat.S_IMODE(restrictive_tmp_path.stat().st_mode)

    assert mode == 0o700


def test_fake_clock_fixture_is_deterministic(fake_clock: FakeClock) -> None:
    first = fake_clock.time_ns()
    second = fake_clock.time_ns()

    assert first == second

    fake_clock.advance(1_000_000)

    assert fake_clock.time_ns() == first + 1_000_000


def test_fake_clock_advances_only_when_arranged() -> None:
    clock = FakeClock(start_ns=5, auto_step_ns=2)

    assert [clock.time_ns() for _ in range(3)] == [5, 7, 9]


def test_build_sqlite_db_executes_script(restrictive_tmp_path: Path) -> None:
    db_path = restrictive_tmp_path / "source.db"

    connection = build_sqlite_db(
        db_path,
        "CREATE TABLE session (id TEXT); INSERT INTO session VALUES ('s1');",
    )
    try:
        rows = connection.execute("SELECT id FROM session").fetchall()
    finally:
        connection.close()

    assert rows == [("s1",)]
    assert db_path.exists()
