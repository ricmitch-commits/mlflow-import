from datetime import UTC, datetime

import pytest

from mlflow_import.config import parse_config
from mlflow_import.errors import ConfigurationError

pytestmark = pytest.mark.unit


def test_date_bounds_are_utc_and_upper_bound_excludes_fractional_seconds():
    config = parse_config(["--experiment-name", "synthetic", "--from-time", "2026-11-13", "--to-time", "2026-11-13"])
    assert config.from_time == datetime(2026, 11, 13, tzinfo=UTC)
    assert config.to_time == datetime(2026, 11, 13, 23, 59, 59, tzinfo=UTC)


def test_reversed_bounds_fail_before_sources_are_opened():
    with pytest.raises(ConfigurationError) as caught:
        parse_config(["--experiment-name", "synthetic", "--from-time", "2026-11-14", "--to-time", "2026-11-13"])
    assert caught.value.code == "invalid_time_range"
