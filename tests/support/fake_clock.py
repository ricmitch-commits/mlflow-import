"""Deterministic, injectable clock for reproducible timestamp tests."""

from __future__ import annotations


class FakeClock:
    """A deterministic stand-in for ``time.time`` and ``time.time_ns``.

    The clock never advances on its own unless ``auto_step_ns`` is set, so
    code under test observes exactly the timestamps the test arranges.
    Pass an explicit ``start_ns`` when a realistic epoch matters.
    """

    def __init__(self, start_ns: int = 0, auto_step_ns: int = 0) -> None:
        self._now_ns = start_ns
        self._auto_step_ns = auto_step_ns

    def time_ns(self) -> int:
        now = self._now_ns
        self._now_ns += self._auto_step_ns
        return now

    def time(self) -> float:
        return self.time_ns() / 1e9

    def advance(self, delta_ns: int) -> None:
        """Move the clock forward (or backward) by ``delta_ns``."""
        self._now_ns += delta_ns
