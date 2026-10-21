"""Unit tests for normalized session selection.

The boundaries pinned here come from the design spec's session filter
table: inclusive time bounds applied to the session creation instant
only, path-component directory equality/descendant matching, exact
normalized model identities (including the incomplete ``provider/`` and
``/model`` forms), a strict minimum-turn threshold where an omitted
filter imposes no restriction, and AND combination of every supplied
filter.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from opencode_mlflow.source.selection import (
    SelectionEvidence,
    SelectionFilters,
    SessionDiscoverySummary,
    matches_directory,
    matches_model,
    select_session,
)

pytestmark = pytest.mark.unit

CREATION = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


@pytest.fixture
def summary() -> SessionDiscoverySummary:
    """A discovery summary that satisfies every filter in the plan's example test."""
    return SessionDiscoverySummary(
        session_id="ses-1",
        creation_time=CREATION,
        directory=Path("/repo/project"),
        model_identities=frozenset({"provider/model"}),
        logical_turn_count=3,
    )


def test_selection_requires_every_supplied_filter(summary):
    filters = SelectionFilters(
        from_time=datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc),
        directory=Path("/repo/project"),
        model="provider/model",
        min_user_turns=2,
    )

    assert select_session(summary, filters) is True
    assert select_session(replace(summary, model_identities=frozenset({"other/model"})), filters) is False


def test_no_filters_select_every_compatible_session(summary: SessionDiscoverySummary) -> None:
    assert select_session(summary, SelectionFilters()) is True


class TestTimeBounds:
    def test_from_time_is_inclusive(self, summary: SessionDiscoverySummary) -> None:
        assert select_session(summary, SelectionFilters(from_time=CREATION)) is True
        assert select_session(summary, SelectionFilters(from_time=CREATION - timedelta(microseconds=1))) is True
        assert select_session(summary, SelectionFilters(from_time=CREATION + timedelta(microseconds=1))) is False

    def test_to_time_is_inclusive(self, summary: SessionDiscoverySummary) -> None:
        assert select_session(summary, SelectionFilters(to_time=CREATION)) is True
        assert select_session(summary, SelectionFilters(to_time=CREATION + timedelta(microseconds=1))) is True
        assert select_session(summary, SelectionFilters(to_time=CREATION - timedelta(microseconds=1))) is False

    def test_bounds_apply_to_the_session_creation_instant_only(self) -> None:
        # The summary exposes exactly one time field — the session creation
        # instant. Message timestamps and time_updated never enter selection.
        time_fields = [item.name for item in dataclass_fields(SessionDiscoverySummary) if "time" in item.name]
        assert time_fields == ["creation_time"]

    def test_missing_creation_time_is_selectable_without_time_filters(self, summary: SessionDiscoverySummary) -> None:
        missing = replace(summary, creation_time=None)

        assert select_session(missing, SelectionFilters()) is True
        assert select_session(missing, SelectionFilters(model="provider/model")) is True

    @pytest.mark.parametrize(
        "filters",
        [
            SelectionFilters(from_time=CREATION),
            SelectionFilters(to_time=CREATION),
            SelectionFilters(from_time=CREATION - timedelta(days=1), to_time=CREATION + timedelta(days=1)),
        ],
    )
    def test_missing_creation_time_is_ineligible_with_any_time_filter(
        self, summary: SessionDiscoverySummary, filters: SelectionFilters
    ) -> None:
        assert select_session(replace(summary, creation_time=None), filters) is False


class TestDirectoryMatching:
    def test_candidates_use_cli_equivalent_lexical_normalization(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        requested = tmp_path / "repo" / "project"

        matching = (
            Path(str(requested) + "/./child"),
            Path(str(requested) + "/nested/../child//grandchild"),
        )
        non_matching = (
            Path(str(tmp_path / "repo" / "projector") + "/child"),
            Path("repo/project-sibling/../project-sibling"),
        )

        assert all(matches_directory(candidate, requested) for candidate in matching)
        assert not any(matches_directory(candidate, requested) for candidate in non_matching)

    def test_relative_source_directory_is_not_rebased_for_absolute_request(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        requested = tmp_path / "repo" / "project"

        assert not matches_directory(Path("repo/project/child"), requested)

    def test_exact_directory_matches(self) -> None:
        assert matches_directory(Path("/repo/project"), Path("/repo/project")) is True

    def test_component_descendants_match(self) -> None:
        assert matches_directory(Path("/repo/project/app"), Path("/repo/project")) is True
        assert matches_directory(Path("/repo/project/app/nested"), Path("/repo/project")) is True

    def test_sibling_prefix_does_not_match(self) -> None:
        assert matches_directory(Path("/repo/projector"), Path("/repo/project")) is False

    def test_unrelated_directories_do_not_match(self) -> None:
        assert matches_directory(Path("/repo/other"), Path("/repo/project")) is False
        assert matches_directory(Path("/repo"), Path("/repo/project")) is False

    def test_directory_filter_selects_by_components(self, summary: SessionDiscoverySummary) -> None:
        nested = replace(summary, directory=Path("/repo/project/app"))
        projector = replace(summary, directory=Path("/repo/projector"))
        filters = SelectionFilters(directory=Path("/repo/project"))

        assert select_session(summary, filters) is True
        assert select_session(nested, filters) is True
        assert select_session(projector, filters) is False

    def test_missing_session_directory_never_matches_a_directory_filter(self, summary: SessionDiscoverySummary) -> None:
        missing = replace(summary, directory=None)

        assert select_session(missing, SelectionFilters(directory=Path("/repo/project"))) is False
        assert select_session(missing, SelectionFilters()) is True


class TestModelMatching:
    def test_no_requested_model_imposes_no_restriction(self) -> None:
        assert matches_model((), None) is True
        assert matches_model(frozenset(), None) is True

    def test_complete_identity_matches_exactly(self) -> None:
        assert matches_model(("provider/model",), "provider/model") is True

    def test_the_request_is_whitespace_normalized(self) -> None:
        assert matches_model(("provider/model",), "  provider/model\t") is True

    def test_case_is_preserved(self) -> None:
        assert matches_model(("Provider/Model",), "provider/model") is False
        assert matches_model(("Provider/Model",), "Provider/Model") is True

    def test_incomplete_source_identities_match_only_their_exact_form(self) -> None:
        assert matches_model(("provider/",), "provider/") is True
        assert matches_model(("provider/",), "provider/model") is False
        assert matches_model(("/model",), "/model") is True
        assert matches_model(("/model",), "provider/model") is False

    def test_complete_source_identities_never_match_incomplete_requests(self) -> None:
        assert matches_model(("provider/model",), "provider/") is False
        assert matches_model(("provider/model",), "/model") is False

    def test_no_available_identity_never_matches(self) -> None:
        assert matches_model((), "provider/model") is False

    def test_model_filter_selects_via_the_summary(self, summary: SessionDiscoverySummary) -> None:
        assert select_session(summary, SelectionFilters(model="provider/model")) is True
        assert select_session(summary, SelectionFilters(model="provider/")) is False


class TestMinimumUserTurns:
    def test_no_threshold_keeps_zero_turn_sessions_selectable(self, summary: SessionDiscoverySummary) -> None:
        zero = replace(summary, logical_turn_count=0)

        assert select_session(zero, SelectionFilters()) is True

    def test_the_threshold_is_strictly_greater(self, summary: SessionDiscoverySummary) -> None:
        exact = replace(summary, logical_turn_count=2)
        above = replace(summary, logical_turn_count=3)

        assert select_session(exact, SelectionFilters(min_user_turns=2)) is False
        assert select_session(above, SelectionFilters(min_user_turns=2)) is True

    def test_a_zero_threshold_excludes_zero_turn_sessions(self, summary: SessionDiscoverySummary) -> None:
        zero = replace(summary, logical_turn_count=0)
        one = replace(summary, logical_turn_count=1)

        assert select_session(zero, SelectionFilters(min_user_turns=0)) is False
        assert select_session(one, SelectionFilters(min_user_turns=0)) is True


class TestAndSemantics:
    def test_pre_redaction_boolean_evidence_controls_directory_and_model_filters(
        self, summary: SessionDiscoverySummary
    ) -> None:
        filters = SelectionFilters(directory=Path("/repo/project"), model="provider/model")
        redacted_summary = replace(
            summary,
            directory=Path("/[REDACTED]"),
            model_identities=frozenset(),
        )

        assert select_session(
            redacted_summary,
            filters,
            SelectionEvidence(directory_match=True, model_match=True),
        )
        assert not select_session(
            summary,
            filters,
            SelectionEvidence(directory_match=False, model_match=True),
        )
        assert not select_session(
            summary,
            filters,
            SelectionEvidence(directory_match=True, model_match=False),
        )

    @pytest.mark.parametrize(
        "changed",
        [
            {"creation_time": CREATION + timedelta(days=1)},
            {"directory": Path("/repo/projector")},
            {"model_identities": frozenset({"other/model"})},
            {"logical_turn_count": 2},
        ],
    )
    def test_one_failing_filter_rejects_the_session(
        self, summary: SessionDiscoverySummary, changed: dict[str, object]
    ) -> None:
        filters = SelectionFilters(
            from_time=CREATION,
            to_time=CREATION,
            directory=Path("/repo/project"),
            model="provider/model",
            min_user_turns=2,
        )

        assert select_session(replace(summary, **changed), filters) is False

    def test_every_supplied_filter_passing_selects_the_session(self, summary: SessionDiscoverySummary) -> None:
        filters = SelectionFilters(
            from_time=CREATION,
            to_time=CREATION,
            directory=Path("/repo/project"),
            model="provider/model",
            min_user_turns=2,
        )

        assert select_session(summary, filters) is True
