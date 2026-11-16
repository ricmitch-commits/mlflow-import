"""Exact canonical-byte guards for complete roots and structured activities."""

import pytest

from mlflow_import.content.activity import project_activity
from mlflow_import.content.artifacts import inline_value_or_reference, load_root_values
from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.capabilities import PINNED_MAX_INLINE_PAYLOAD_BYTES
from mlflow_import.mlflow.mapping import _shared_child_specs
from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_summary
from tests.support.inline_payloads import (
    QUALIFICATION_SIZES,
    exact_inline_value,
    synthetic_tool_turn,
)

pytestmark = pytest.mark.unit


def _turn(**kwargs):
    return FakeReplayTurn(session_header=fake_header(), summary=fake_summary(), **kwargs)


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("size", QUALIFICATION_SIZES)
def test_complete_roots_fit_pinned_envelope(size, structured):
    value = exact_inline_value(size, structured=structured)
    loaded = load_root_values(_turn(root_input=value, root_output=value), PINNED_MAX_INLINE_PAYLOAD_BYTES)
    assert loaded.root_input == value
    assert loaded.root_output == value
    inline = inline_value_or_reference(value, PINNED_MAX_INLINE_PAYLOAD_BYTES)
    assert inline.inline
    assert inline.value == value


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("field", ["root_input", "root_output"])
def test_complete_root_above_eight_mib_rejected(field, structured):
    value = exact_inline_value(8_388_609, structured=structured)
    with pytest.raises(ItemImportError) as caught:
        load_root_values(_turn(**{field: value}), PINNED_MAX_INLINE_PAYLOAD_BYTES)
    assert caught.value.code == "payload_too_large"


@pytest.mark.parametrize("field", ["input", "output"])
@pytest.mark.parametrize("size", [*QUALIFICATION_SIZES, 8_388_609])
def test_projected_structured_tool_guard(size, field):
    value = exact_inline_value(size)
    turn = synthetic_tool_turn(value, field)
    activity = project_activity(turn)
    if size == 8_388_609:
        with pytest.raises(ItemImportError) as caught:
            list(_shared_child_specs(turn, activity, PINNED_MAX_INLINE_PAYLOAD_BYTES))
        assert caught.value.code == "payload_too_large"
    else:
        specs = list(_shared_child_specs(turn, activity, PINNED_MAX_INLINE_PAYLOAD_BYTES))
        tool = next(spec for spec in specs if spec.span_type == "TOOL")
        assert getattr(tool, field + "s") == value


def test_oversized_non_activity_values_keep_existing_artifact_reference_policy():
    value = exact_inline_value(8_388_609, structured=False)
    inline = inline_value_or_reference(value, PINNED_MAX_INLINE_PAYLOAD_BYTES)
    assert not inline.inline
    assert inline.reference is not None
    assert inline.reference.size_bytes == 8_388_609
    assert inline.value == {"$artifact_reference": {"sha256": inline.reference.sha256, "size_bytes": 8_388_609}}
