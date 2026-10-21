"""Unit tests validating the BDD feature-file scaffold.

pytest-bdd does not parse the feature files until scenarios are bound, so
these checks keep the ``Feature:`` declarations valid and descriptive in
the default selection. They use the same Gherkin parser pytest-bdd uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from gherkin.parser import Parser

pytestmark = pytest.mark.unit

FEATURES_DIR = Path(__file__).resolve().parents[1] / "bdd" / "features"


@pytest.mark.parametrize(
    ("filename", "expected_feature"),
    [
        ("cli_and_selection.feature", "CLI invocation and session selection"),
        ("source_safety.feature", "Read-only source snapshot safety"),
        ("mapping_and_fidelity.feature", "Turn-to-trace mapping and content fidelity"),
        ("reconciliation.feature", "Reconciliation of previously imported traces"),
        ("privacy_and_reporting.feature", "Privacy, redaction, and reporting"),
    ],
)
def test_feature_file_declares_a_named_feature(filename: str, expected_feature: str) -> None:
    text = (FEATURES_DIR / filename).read_text(encoding="utf-8")

    document = Parser().parse(text)

    assert document["feature"]["name"] == expected_feature
