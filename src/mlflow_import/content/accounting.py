"""Shared, source-version-bound usage and reproducible USD accounting.

Only a single turn's invocation state is retained. Raw observations remain in
the session spool; derived values never overwrite their source observations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from mlflow_import.source.model import SourceRecord

ACCOUNTING_VERSION = "accounting-v2"
PRICING_VERSION = "bundled-api-usd-2025-08-v1"
MAX_PRICING_MAP_BYTES = 8 * 1024 * 1024
# USD per million tokens. Deliberately exact model IDs, never fuzzy aliases.
# Published API list prices, not subscription charges. The applied copy is
# retained in each invocation's provenance and reused independently of this table.
PRICES = {
    "openai/gpt-4o-2024-08-06": ("2.50", "10", "1.25"),
    "openai/gpt-4o": ("2.50", "10", "1.25"),
    "openai/gpt-5": ("1.25", "10", "0.125"),
    "anthropic/claude-sonnet-4-20250514": ("3", "15", "0.30"),
}
MODEL_MAPPING_VERSION = "bundled-model-map-v1"
MODEL_ALIASES = {alias: key for key in PRICES for alias in (key.partition("/")[2], "/" + key.partition("/")[2])}

# Selected before final mapping by the sequential reconciliation boundary.
# Keys are stable turn keys, then adapter-qualified invocation IDs.
pricing_bases: ContextVar[Mapping[str, dict[str, Any]] | None] = ContextVar("pricing_bases", default=None)


def object_map(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def merge_observation(target: dict[str, Any], source: Mapping[str, Any], *, override: bool = True) -> None:
    """Merge present counter fields, retaining absent nested cache observations."""
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            merge_observation(target[key], value, override=override)
        elif override or key not in target:
            target[key] = dict(value) if isinstance(value, Mapping) else value


def counts(value: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, int]:
    return {
        target: value[key]
        for key, target in aliases.items()
        if isinstance(value.get(key), int) and not isinstance(value[key], bool) and value[key] >= 0
    }


def normalize_usage(
    harness: str, decoded: Mapping[str, Any], previous: Mapping[str, Any] | None = None, *, initial: bool = False
) -> tuple[dict[str, int], str]:
    """Return native token keys and scope, preserving absent versus zero.

    Codex last_token_usage is a turn observation, never copied to model spans.
    A cumulative-only observation cannot establish a turn delta without its
    preceding observation; it is retained as unavailable, rather than guessed.
    """
    usage = object_map(decoded.get("usage", decoded.get("tokens")))
    scope = "invocation"
    if harness == "codex":
        info = object_map(decoded.get("info"))
        usage = object_map(info.get("last_token_usage"))
        scope = "turn-last-observation"
    aliases = {
        "input": "input_tokens",
        "input_tokens": "input_tokens",
        "promptTokenCount": "input_tokens",
        "output": "output_tokens",
        "output_tokens": "output_tokens",
        "candidatesTokenCount": "output_tokens",
        "total": "total_tokens",
        "totalTokens": "total_tokens",
        "total_tokens": "total_tokens",
        "totalTokenCount": "total_tokens",
        "cacheRead": "cache_read_input_tokens",
        "cache_read_input_tokens": "cache_read_input_tokens",
        "cached_input_tokens": "cache_read_input_tokens",
        "cached": "cache_read_input_tokens",
        "cachedContentTokenCount": "cache_read_input_tokens",
        "cacheWrite": "cache_creation_input_tokens",
        "cache_creation_input_tokens": "cache_creation_input_tokens",
    }
    result = counts(usage, aliases)
    if harness == "codex" and previous is not None and isinstance(info.get("total_token_usage"), Mapping):
        current = counts(info["total_token_usage"], aliases)
        prior = counts(previous, aliases)
        if initial:
            result, scope = current, "cumulative-initial"
        else:
            delta, reset = cumulative_delta(current, prior)
            if delta or not result:
                result = delta
                scope = "cumulative-reset" if reset else "cumulative-delta"
    cache = object_map(usage.get("cache"))
    result.update(counts(cache, {"read": "cache_read_input_tokens", "write": "cache_creation_input_tokens"}))
    if harness == "claude" and "input_tokens" in result:
        result["input_tokens"] += result.get("cache_read_input_tokens", 0) + result.get(
            "cache_creation_input_tokens", 0
        )
        creation = object_map(usage.get("cache_creation"))
        result.update(counts(creation, {"ephemeral_1h_input_tokens": "cache_creation_input_tokens_above_1hr"}))
    if harness == "pi" and "input_tokens" in result:
        result["input_tokens"] += result.get("cache_read_input_tokens", 0) + result.get(
            "cache_creation_input_tokens", 0
        )
    if harness == "gemini" and "output_tokens" in result:
        # RecordingService stores candidatesTokenCount as output and
        # thoughtsTokenCount separately. Both are generated output tokens.
        thoughts = usage.get("thoughts", usage.get("thoughtsTokenCount"))
        if isinstance(thoughts, int) and not isinstance(thoughts, bool) and thoughts >= 0:
            result["output_tokens"] += thoughts
    if "total_tokens" not in result and {"input_tokens", "output_tokens"} <= result.keys():
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
        if harness == "opencode":
            reasoning = usage.get("reasoning")
            if isinstance(reasoning, int) and not isinstance(reasoning, bool) and reasoning >= 0:
                result["total_tokens"] += reasoning
    return result, scope


def cumulative_delta(current: Mapping[str, int], previous: Mapping[str, int]) -> tuple[dict[str, int], bool]:
    """Difference proven cumulative counters, recording reset rather than negative usage."""
    reset = any(value < previous[key] for key, value in current.items() if key in previous)
    return (
        {key: value if reset else value - previous[key] for key, value in current.items() if key in previous},
        reset,
    )


def sum_fields(values: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        for key, amount in value.items():
            result[key] = result.get(key, 0) + amount
    return result


def invocation_key(record: SourceRecord) -> str:
    scope = str(record.raw_columns.get("subagent") or "")
    return scope + ":" + (record.invocation_id or record.source_id or record.source_identity or "unknown")


def applied_bases(provenance: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: observation["cost"]["basis"]
        for key, observation in provenance.get("invocations", {}).items()
        if "basis" in observation.get("cost", {})
    }
    turn_cost = provenance.get("turn_cost") or {}
    if "basis" in turn_cost:
        result["turn-usage"] = turn_cost["basis"]
    return result


def _decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        result = Decimal(str(value))
        return result if result.is_finite() and result >= 0 else None
    except InvalidOperation:
        return None


def freeze_pricing_bases(turns: Iterable[Any], selected: Mapping[str, dict[str, Any]]) -> Mapping[str, dict[str, Any]]:
    """Freeze estimates and negative coverage decisions before final hashes/writes."""
    from mlflow_import.content.activity import project_activity

    token = pricing_bases.set(selected)
    try:
        from mlflow_import.storage import DiskMap

        frozen: DiskMap[str, dict[str, Any]] = DiskMap()
        for turn in turns:
            from mlflow_import.errors import SourceCompatibilityError
            from mlflow_import.identity import native_metadata

            native = native_metadata(turn.session_header)
            first = next(turn.iter_records(), None)
            value = object_map(first.decoded_value) if first is not None else {}
            request = native.get(
                "mlflow_import.client_request_id",
                object_map(turn.session_header.metadata).get(
                    "client_request_id", value.get("request_id", value.get("requestID"))
                ),
            )
            if isinstance(request, str) and len(request) > 50:
                raise SourceCompatibilityError(
                    "native_request_id_too_long", "the source request identity exceeds the qualified native limit"
                )
            frozen[turn.summary.turn_key] = applied_bases(project_activity(turn).provenance)
        return frozen
    finally:
        pricing_bases.reset(token)


def cost_projection(
    harness: str,
    decoded: Mapping[str, Any],
    model: str | None,
    usage: Mapping[str, int],
    basis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persisted cost wins even when currency/breakdown is unavailable.

    Decimal strings are the reproducible artifact representation; conversion to
    floating point happens only at the MLflow numeric-attribute boundary.
    """
    raw = decoded.get("cost", object_map(decoded.get("usage")).get("cost"))
    if raw is not None:
        source = object_map(raw)
        currency = source.get("currency", decoded.get("cost_currency"))
        if isinstance(currency, str):
            currency = currency.strip().upper()
        if currency is None and harness in {"opencode", "pi"}:
            currency = "USD"  # Qualified source schema defines the unit.
        method = source.get("method", decoded.get("cost_method", "unknown"))
        if method not in {"estimated", "provider-billed", "unknown"}:
            method = "unknown"
        amounts = (
            {"total_cost": raw}
            if not source
            else {
                target: source[key]
                for key, target in {
                    "input": "input_cost",
                    "output": "output_cost",
                    "total": "total_cost",
                    "input_cost": "input_cost",
                    "output_cost": "output_cost",
                    "total_cost": "total_cost",
                }.items()
                if key in source
            }
        )
        if harness == "pi" and "input_cost" in amounts:
            cache_keys = (("cacheRead", "cache_read_input_tokens"), ("cacheWrite", "cache_creation_input_tokens"))
            if any(usage.get(counter, 0) > 0 and _decimal(source.get(key)) is None for key, counter in cache_keys):
                amounts.pop("input_cost")
            else:
                with localcontext() as context:
                    context.prec = 50
                    original_input = _decimal(amounts["input_cost"])
                    if original_input is not None:
                        amounts["input_cost"] = str(
                            original_input
                            + sum((_decimal(source.get(key)) or Decimal(0) for key, _ in cache_keys), Decimal(0))
                        )
        usd = (
            {key: str(number) for key, value in amounts.items() if (number := _decimal(value)) is not None}
            if currency == "USD"
            else {}
        )
        conversion = object_map(source.get("conversion", decoded.get("cost_conversion")))
        factor = _decimal(conversion.get("factor"))
        if (
            not usd
            and currency != "USD"
            and conversion.get("from") == currency
            and conversion.get("to") == "USD"
            and conversion.get("version")
            and factor is not None
            and factor > 0
        ):
            with localcontext() as context:
                context.prec = 50
                usd = {
                    key: str(number * factor)
                    for key, amount in amounts.items()
                    if (number := _decimal(amount)) is not None
                }
        return {
            "origin": "persisted",
            "method": method,
            "original": raw,
            "currency": currency,
            "conversion": dict(conversion) if conversion else None,
            "usd": usd,
            "gap": None if usd else "cost_currency_or_amount_unavailable",
        }
    source_model = model
    applicable = basis is not None and basis.get("source_model", basis.get("model")) == source_model
    if applicable:
        assert basis is not None
        model = basis.get("model")
    else:
        model = MODEL_ALIASES.get(model, model) if model is not None else None
    selected = dict(basis) if applicable and basis is not None else None
    if selected is not None and selected.get("model") != model:
        selected = None
    if selected is None and model in PRICES:
        input_rate, output_rate, cache_rate = PRICES[model]
        selected = {
            "version": PRICING_VERSION,
            "model": model,
            "source_model": source_model,
            "model_mapping_version": MODEL_MAPPING_VERSION,
            "decimal_precision": 50,
            "currency": "USD",
            "unit": "million-tokens",
            "input": input_rate,
            "output": output_rate,
            "cache_read": cache_rate,
            "rule": "input-minus-cache-read-v1",
            "effective": "2025-08",
            "source": "https://openai.com/api/pricing/"
            if model.startswith("openai/")
            else "https://www.anthropic.com/pricing",
        }
        if model.startswith("anthropic/"):
            selected.update(cache_creation="3.75", cache_creation_1h="6", max_input_tokens=200000)

    def unavailable(reason: str) -> dict[str, Any]:
        return {
            "origin": "importer-derived",
            "method": "estimated",
            "usd": {},
            "gap": reason,
            "basis": selected
            or {
                "version": PRICING_VERSION,
                "model": model,
                "source_model": source_model,
                "model_mapping_version": MODEL_MAPPING_VERSION,
                "available": False,
                "gap": reason,
            },
        }

    if selected is not None and selected.get("available") is False:
        return unavailable(str(selected.get("gap", "pricing_or_usage_unavailable")))
    if selected is not None:
        from mlflow_import.errors import ItemImportError

        if (
            not selected.get("version")
            or selected.get("currency") != "USD"
            or selected.get("unit") != "million-tokens"
            or selected.get("rule") != "input-minus-cache-read-v1"
            or any(_decimal(selected.get(key)) is None for key in ("input", "output", "cache_read"))
        ):
            raise ItemImportError("pricing_basis_invalid", "the recorded pricing basis cannot be reproduced")

    if harness == "opencode" and (
        usage.get("cache_read_input_tokens", 0)
        or usage.get("cache_creation_input_tokens", 0)
        or object_map(decoded.get("usage", decoded.get("tokens"))).get("reasoning")
    ):
        return unavailable("billing_cache_or_reasoning_scope_unavailable")
    if selected and usage.get("input_tokens", 0) > selected.get("max_input_tokens", float("inf")):
        return unavailable("pricing_context_tier_unqualified")
    if (
        not selected
        or not {"input_tokens", "output_tokens"} <= usage.keys()
        or (usage.get("cache_creation_input_tokens", 0) and (harness != "claude" or "cache_creation" not in selected))
    ):
        return unavailable("pricing_or_usage_unavailable")
    if (
        selected.get("model") != model
        or selected.get("currency") != "USD"
        or selected.get("rule") != "input-minus-cache-read-v1"
    ):
        return unavailable("pricing_basis_inapplicable")
    cached = usage.get("cache_read_input_tokens", 0)
    created = usage.get("cache_creation_input_tokens", 0)
    extended = usage.get("cache_creation_input_tokens_above_1hr", 0)
    if cached + created > usage["input_tokens"] or extended > created:
        return unavailable("invalid_cache_subset")
    if created and harness == "claude" and "cache_creation_input_tokens_above_1hr" not in usage:
        creation = object_map(object_map(decoded.get("usage")).get("cache_creation"))
        if creation.get("ephemeral_5m_input_tokens") != created:
            return unavailable("cache_creation_lifetime_unavailable")
    with localcontext() as context:
        context.prec = 50
        input_cost = (
            Decimal(usage["input_tokens"] - cached - created) * Decimal(selected["input"])
            + Decimal(cached) * Decimal(selected["cache_read"])
            + Decimal(created - extended) * Decimal(selected.get("cache_creation", "0"))
            + Decimal(extended) * Decimal(selected.get("cache_creation_1h", "0"))
        ) / Decimal(1_000_000)
        output_cost = Decimal(usage["output_tokens"]) * Decimal(selected["output"]) / Decimal(1_000_000)
        total_cost = input_cost + output_cost
    return {
        "origin": "importer-derived",
        "method": "estimated",
        "basis": selected,
        "inputs": dict(usage),
        "usd": {
            "input_cost": str(input_cost),
            "output_cost": str(output_cost),
            "total_cost": str(total_cost),
        },
        "gap": None,
    }
