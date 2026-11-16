"""The pinned transport compensation changes only ambiguous top-level strings."""

import json

import pytest

from mlflow_import.mlflow.gateway import _encode_historical_attribute

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "value",
    [
        "[1, 2]",
        '{"x": 1, "x": 2}',
        ' \n ["雪 😀"] \t',
        '""',
        '"null"',
        '"1"',
        '"true"',
        '"\\u96ea"',
        json.dumps(json.dumps('雪 "quoted" \\ path\nnext', ensure_ascii=False), ensure_ascii=False),
    ],
)
def test_ambiguous_source_string_gets_exactly_one_extra_layer(value):
    encoded = _encode_historical_attribute(value)
    assert json.loads(encoded) == json.dumps(value, ensure_ascii=False)
    assert json.loads(json.loads(encoded)) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "plain 雪 😀",
        "[incomplete",
        "{} trailing",
        "01",
        "1",
        "-1",
        "1.5",
        "1e3",
        "true",
        "false",
        "null",
        "NaN",
        "Infinity",
        "-Infinity",
        " \n null \t",
        {"array": "[1]", "quoted": '"x"'},
        ["[1]", {"object": "{}"}],
        {},
        [],
        0,
        1.5,
        True,
        False,
        None,
    ],
)
def test_structures_and_scalar_controls_keep_ordinary_encoding(value):
    assert _encode_historical_attribute(value) == json.dumps(value, ensure_ascii=False)
