"""Independent expected values for versioned usage/cost semantics."""

from dataclasses import replace
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest

from mlflow_import.content.accounting import cost_projection, cumulative_delta, normalize_usage, pricing_bases
from mlflow_import.content.activity import project_activity
from mlflow_import.content.coverage import coverage
from mlflow_import.content.fingerprint import build_effective_turn, session_context_fingerprint
from mlflow_import.contracts import IDENTITY_REDACTOR
from mlflow_import.identity import native_metadata
from mlflow_import.source.model import SessionHeader, SourceRecord, TurnSummary

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("harness", "value", "expected"),
    [
        (
            "claude",
            {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 2,
                    "cache_creation_input_tokens": 4,
                }
            },
            {
                "input_tokens": 16,
                "output_tokens": 3,
                "total_tokens": 19,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 4,
            },
        ),
        (
            "pi",
            {"usage": {"input": 10, "output": 3, "reasoning": 2, "cacheRead": 2, "totalTokens": 15}},
            {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15, "cache_read_input_tokens": 2},
        ),
        (
            "opencode",
            {"tokens": {"input": 10, "output": 3, "reasoning": 2, "cache": {"read": 2}}},
            {"input_tokens": 10, "output_tokens": 3, "total_tokens": 15, "cache_read_input_tokens": 2},
        ),
        (
            "gemini",
            {"tokens": {"input": 10, "output": 3, "thoughts": 2, "cached": 2, "total": 17}},
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 17, "cache_read_input_tokens": 2},
        ),
        (
            "codex",
            {
                "info": {
                    "last_token_usage": {
                        "input_tokens": 10,
                        "output_tokens": 0,
                        "total_tokens": 13,
                        "cached_input_tokens": 2,
                    }
                }
            },
            {"input_tokens": 10, "output_tokens": 0, "total_tokens": 13, "cache_read_input_tokens": 2},
        ),
        ("pi", {"usage": {"input": 0, "output": 0}}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
        ("claude", {"usage": {"input_tokens": True, "output_tokens": -1}}, {}),
        ("codex", {"info": {"total_token_usage": {"input_tokens": 100}}}, {}),
    ],
)
def test_declared_usage_rules(harness, value, expected):
    assert normalize_usage(harness, value)[0] == expected


def test_decimal_estimate_reproduces_with_snapshot_and_preserves_original_cost():
    usage = {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 20}
    value = cost_projection("codex", {}, "openai/gpt-4o", usage)
    assert value["usd"] == {"input_cost": "0.000225", "output_cost": "0.0001", "total_cost": "0.000325"}
    assert value["origin"] == "importer-derived"
    assert value["method"] == "estimated"
    with localcontext() as context:
        context.prec = 3
        assert cost_projection("codex", {}, "openai/gpt-4o", usage, value["basis"]) == value
    persisted = cost_projection("pi", {"cost": {"total": 0, "method": "estimated"}}, "openai/gpt-4o", usage)
    assert persisted["usd"] == {"total_cost": "0"}
    assert persisted["origin"] == "persisted"
    assert persisted["method"] == "estimated"
    unknown = cost_projection("claude", {"cost": 1}, "openai/gpt-4o", usage)
    assert unknown["usd"] == {}  # Currency is not guessed, and no estimate replaces it.
    assert unknown["original"] == 1


def test_cumulative_reset_is_not_a_negative_delta():
    assert cumulative_delta({"input_tokens": 15}, {"input_tokens": 10}) == ({"input_tokens": 5}, False)
    assert cumulative_delta({"input_tokens": 2}, {"input_tokens": 10}) == ({"input_tokens": 2}, True)


def replay(records, harness="claude", metadata=None):
    return SimpleNamespace(
        session_header=SessionHeader("s", "current", catalog_version=harness + "-persisted-v2", metadata=metadata),
        summary=TurnSummary("s", 1, "u", record_count=len(records)),
        iter_records=lambda: iter(records),
        iter_all_records=lambda: iter(records),
    )


def test_split_invocation_coalesces_latest_present_observations_and_preserves_zero():
    first = SourceRecord(
        "jsonl_message",
        "assistant",
        "a",
        decoded_value={"model": "openai/gpt-4o", "usage": {"input_tokens": 10, "output_tokens": 1}},
        classification="assistant",
        invocation_id="call",
    )
    second = replace(first, source_id="b", decoded_value={"usage": {"output_tokens": 0}})
    activity = project_activity(replay([first, second]))
    assert list(activity.calls) == [":call"]
    assert activity.usage == {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10}
    assert len(activity.calls[":call"].records) == 2


def test_native_metadata_is_historical_and_included_in_immutable_hash():
    turn = replay(
        [],
        metadata={
            "user": {"name": "synthetic-user"},
            "git": {"repository_url": "https://example.test/repo", "commit_hash": "abc", "branch": "historical"},
        },
    )
    header = replace(turn.session_header, parent_id="parent")
    turn.session_header = header
    native = native_metadata(header)
    assert native["mlflow.trace.user"] == "synthetic-user"
    assert native["mlflow.source.git.commit"] == "abc"
    assert native["parent_session_id"] == "parent"
    first = session_context_fingerprint(build_effective_turn(turn, IDENTITY_REDACTOR))
    turn.session_header = replace(header, metadata={"user": "another"})
    assert session_context_fingerprint(build_effective_turn(turn, IDENTITY_REDACTOR)) != first
    empty = coverage(turn.session_header, project_activity(turn))
    assert empty["source_version"] == "claude-persisted-v2"
    assert empty["items"]["usage"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "field,size,accepted",
    [
        ("session", 250, True),
        ("session", 251, False),
        ("parent", 250, True),
        ("parent", 251, False),
        ("git", 4096, True),
        ("git", 4097, False),
        ("request", 50, True),
        ("request", 51, False),
    ],
)
def test_historical_fact_limits_follow_the_actual_destination(field, size, accepted):
    from mlflow_import.errors import SourceCompatibilityError

    header = replace(replay([]).session_header, source_fingerprint="f" * 64)
    if field == "session":
        header = replace(header, session_id="s" * size)
    elif field == "parent":
        header = replace(header, parent_id="p" * size)
    elif field == "git":
        header = replace(header, metadata={"git": {"repository_url": "r" * size}})
    else:
        header = replace(header, metadata={"client_request_id": "q" * size})
    if accepted:
        native_metadata(header)
    else:
        with pytest.raises(SourceCompatibilityError):
            native_metadata(header)


def test_selected_per_invocation_basis_used_before_fingerprinting():
    record = SourceRecord(
        "jsonl_message",
        "assistant",
        "a",
        decoded_value={"model": "openai/gpt-4o", "usage": {"input_tokens": 10, "output_tokens": 1}},
        classification="assistant",
        invocation_id="call",
    )
    turn = replay([record])
    value = project_activity(turn)
    basis = dict(value.calls[":call"].cost["basis"], input="100")
    token = pricing_bases.set({"u": {":call": basis}})
    try:
        assert Decimal(project_activity(turn).calls[":call"].cost["usd"]["input_cost"]) == Decimal("0.001")
    finally:
        pricing_bases.reset(token)


def test_basis_donor_order_is_instance_then_stable_turn_then_trace_id():
    from mlflow_import.mlflow.reconciliation import _select_pricing_basis

    def reference(trace_id, turn="turn"):
        return {"trace_id": trace_id, "metadata": {"mlflow_import.turn_key": turn}}

    attempts = [
        {"complete": {"instance_id": "duplicate-z", "replacements": [reference("a")]}},
        {"complete": {"instance_id": "", "replacements": [reference("z"), reference("b"), reference("c", "other")]}},
        {"complete": {"instance_id": "duplicate-a", "replacements": [reference("x")]}},
    ]
    gateway = SimpleNamespace(
        trace_exists=lambda _: True,
        verify_replacement_reference=lambda _: None,
        read_pricing_basis=lambda ref: {"inv": {"donor": ref["trace_id"]}},
    )
    assert _select_pricing_basis(attempts, gateway, "overwrite") == {
        "turn": {"inv": {"donor": "b"}},
        "other": {"inv": {"donor": "c"}},
    }
    assert _select_pricing_basis([attempts[0], attempts[2]], gateway, "overwrite") == {"turn": {"inv": {"donor": "x"}}}
    assert _select_pricing_basis(attempts, gateway, "duplicate") == {}
    saved = {"turn": {"inv": {"version": "recorded"}}}
    assert _select_pricing_basis([{"prepared": {"pricing_bases": saved}}, *attempts], gateway, "duplicate") == saved


def test_persisted_usd_conversion_requires_version_and_reproduces():
    value = cost_projection(
        "claude",
        {
            "cost": {
                "total": "2.50",
                "currency": "EUR",
                "method": "estimated",
                "conversion": {"version": "synthetic-fx-v1", "from": "EUR", "to": "USD", "factor": "1.2"},
            }
        },
        None,
        {},
    )
    assert value["usd"] == {"total_cost": "3.000"}
    assert value["original"]["total"] == "2.50"
    assert value["origin"] == "persisted"
    assert value["method"] == "estimated"
    assert value["conversion"]["version"] == "synthetic-fx-v1"


def test_claude_creation_ttl_subsets_are_priced_once():
    usage, _ = normalize_usage(
        "claude",
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 4,
                "cache_creation": {"ephemeral_1h_input_tokens": 1},
            }
        },
    )
    assert usage["input_tokens"] == 16
    value = cost_projection("claude", {}, "anthropic/claude-sonnet-4-20250514", usage)
    assert value["usd"] == {"input_cost": "0.00004785", "output_cost": "0.000045", "total_cost": "0.00009285"}
    assert value["basis"]["cache_creation"] == "3.75"
    assert value["basis"]["cache_creation_1h"] == "6"
    assert cost_projection("pi", {}, "anthropic/claude-sonnet-4-20250514", usage)["usd"] == {}


def test_versioned_model_alias_is_part_of_reused_basis(monkeypatch):
    from mlflow_import.content.accounting import MODEL_ALIASES

    usage = {"input_tokens": 1, "output_tokens": 1}
    value = cost_projection("claude", {}, "claude-sonnet-4-20250514", usage)
    assert value["usd"]["total_cost"] == "0.000018"
    assert value["basis"]["source_model"] == "claude-sonnet-4-20250514"
    assert value["basis"]["model"] == "anthropic/claude-sonnet-4-20250514"
    assert value["basis"]["model_mapping_version"] == "bundled-model-map-v1"
    monkeypatch.setitem(MODEL_ALIASES, "claude-sonnet-4-20250514", "openai/gpt-4o")
    assert cost_projection("claude", {}, "claude-sonnet-4-20250514", usage, value["basis"]) == value


def test_unavailable_basis_is_frozen_even_when_installed_alias_becomes_priced(monkeypatch):
    from mlflow_import.content.accounting import MODEL_ALIASES

    usage = {"input_tokens": 10, "output_tokens": 2}
    original = cost_projection("claude", {}, "/unpriced-model", usage)
    assert original["usd"] == {}
    monkeypatch.setitem(MODEL_ALIASES, "/unpriced-model", "openai/gpt-4o")
    assert cost_projection("claude", {}, "/unpriced-model", usage, original["basis"]) == original


def test_creation_lifetime_and_new_cumulative_fields_are_not_guessed():
    value = cost_projection(
        "claude",
        {},
        "anthropic/claude-sonnet-4-20250514",
        {"input_tokens": 20, "output_tokens": 2, "cache_creation_input_tokens": 4},
    )
    assert value["gap"] == "cache_creation_lifetime_unavailable"
    assert value["usd"] == {}
    assert cumulative_delta({"input_tokens": 2, "output_tokens": 5}, {"input_tokens": 10}) == (
        {"input_tokens": 2},
        True,
    )


def test_multi_step_usage_and_cost_are_not_parent_child_double_counted():
    user = SourceRecord("message", "user", "u", classification="user", decoded_value={"content": "prompt"})
    parent = SourceRecord(
        "message", "assistant", "a", classification="assistant", decoded_value={"model": "synthetic/model"}
    )
    parts = [
        SourceRecord(
            "part",
            "step-finish",
            f"step-{index}",
            parent_source_id="a",
            decoded_value={
                "type": "step-finish",
                "tokens": {"input": 10 * index, "output": index},
                "cost": str(Decimal(index) / 10),
            },
        )
        for index in (1, 2)
    ]
    turn = replay([user, parent, *parts], harness="opencode")
    activity = project_activity(turn)
    assert activity.usage == {"input_tokens": 30, "output_tokens": 3, "total_tokens": 33}
    assert activity.cost == {"total_cost": 0.3}
    assert len(activity.calls[":a"].aggregate_children) == 2
    parent = replace(
        parent,
        decoded_value={"model": "synthetic/model", "tokens": {"input": 40, "output": 5, "total": 49}, "cost": "0.7"},
    )
    activity = project_activity(replay([user, parent, *parts], harness="opencode"))
    assert activity.usage == {"input_tokens": 40, "output_tokens": 5, "total_tokens": 49}
    assert activity.cost == {"total_cost": 0.7}


def test_pi_usage_operations_and_converted_cache_costs_are_retained():
    record = SourceRecord(
        "jsonl",
        "usage",
        "warm",
        decoded_value={
            "type": "usage",
            "kind": "cache_warm",
            "model": "synthetic/model",
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 10,
                "totalTokens": 10,
                "cost": {"input": "0", "output": "0", "cacheRead": "0.1", "cacheWrite": "0", "total": "0.1"},
            },
        },
    )
    activity = project_activity(replay([record], harness="pi"))
    assert activity.usage == {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10, "cache_read_input_tokens": 10}
    assert activity.cost == {"input_cost": 0.1, "output_cost": 0.0, "total_cost": 0.1}
    value = cost_projection(
        "pi",
        {
            "cost": {
                "input": "1",
                "cacheRead": "0.2",
                "total": "1.2",
                "currency": "EUR",
                "conversion": {"from": "EUR", "to": "USD", "factor": "2", "version": "fx-fixture-v1"},
            }
        },
        None,
        {"cache_read_input_tokens": 10},
    )
    assert value["usd"] == {"input_cost": "2.4", "total_cost": "2.4"}


def test_unknown_execution_is_not_success_and_declarations_are_exhaustive():
    from mlflow_import.content.activity import observed_status
    from mlflow_import.content.coverage import BASELINE
    from mlflow_import.sources.schemas import COVERAGE_TABLES

    assert observed_status({}) == "UNSET"
    assert observed_status({"stop_reason": "vendor-unknown"}) == "UNSET"
    assert observed_status({"stop_reason": "tool_use"}) == "OK"
    for declaration in COVERAGE_TABLES.values():
        assert set(declaration["items"]) == set(BASELINE)
        assert all(
            item["fields"] and item["units"] and item["scope"] and item["evidence_version"]
            for item in declaration["items"].values()
        )


def test_recorded_basis_validation_and_all_new_invocations_freeze(monkeypatch):
    from mlflow_import.content.accounting import PRICES, freeze_pricing_bases
    from mlflow_import.errors import ItemImportError

    record = SourceRecord(
        "jsonl_message",
        "assistant",
        "a",
        classification="assistant",
        invocation_id="i",
        decoded_value={"model": "openai/gpt-4o", "usage": {"input_tokens": 10, "output_tokens": 1}},
    )
    turn = replay([record])
    selected = freeze_pricing_bases([turn], {})
    monkeypatch.setitem(PRICES, "openai/gpt-4o", ("100", "200", "50"))
    token = pricing_bases.set(selected)
    try:
        assert project_activity(turn).cost == {"input_cost": 0.000025, "output_cost": 0.00001, "total_cost": 0.000035}
    finally:
        pricing_bases.reset(token)
    bad = {**selected["u"][":i"], "input": "NaN"}
    with pytest.raises(ItemImportError, match="cannot be reproduced"):
        cost_projection("claude", {}, "openai/gpt-4o", {"input_tokens": 10, "output_tokens": 1}, bad)
