"""Versioned baseline declarations combined with observed (not assumed) values."""

from __future__ import annotations

from typing import Any

from mlflow_import.content.accounting import ACCOUNTING_VERSION
from mlflow_import.content.activity import Activity
from mlflow_import.identity import harness_name, native_metadata
from mlflow_import.source.model import SessionHeader
from mlflow_import.sources.schemas import COVERAGE_TABLES

COVERAGE_VERSION = "integration-baseline-2026-11-13-v1"
BASELINE = {
    "messages": ("content,parts,message.content", "structured root and chat messages", "direct"),
    "model": ("model,modelID,provider", "mlflow.llm.model/provider", "direct"),
    "tools": ("tool,toolCalls,tool_use,function_call", "correlated TOOL inputs/outputs", "derived"),
    "usage": ("tokens,usage,info.last_token_usage", "mlflow.chat.tokenUsage and native usage", "derived"),
    "cost": ("cost,usage.cost,model,usage", "mlflow.llm.cost and native USD costs", "derived"),
    "timing": ("timestamp,time", "native timing; timestamp intervals, not live execution", "derived"),
    "user": ("metadata.user,user,mlflow.trace.user", "native user metadata", "direct"),
    "git": ("git,mlflow.source.git.*", "native Git metadata", "direct"),
    "thinking": ("reasoning,thinking,thoughts", "structured assistant reasoning", "direct"),
    "agents": ("subagent,parentToolUseID,agentId", "nested AGENT scope", "derived"),
    "skills": ("tool_use.name,tool.name", "TOOL skill activity", "direct"),
    "directory": ("cwd,directory", "current source directory tags/session artifact", "direct"),
    "harness_version": ("cli_version,version", "current harness version tags/session artifact", "direct"),
    "git_repo_url": ("git.repository_url", "mlflow.source.git.repoURL", "direct"),
    "git_commit": ("git.commit_hash", "mlflow.source.git.commit", "direct"),
    "git_branch": ("git.branch", "mlflow.source.git.branch", "direct"),
    "per_call_usage": ("invocation-scoped tokens/usage", "LLM mlflow.chat.tokenUsage", "derived"),
    "execution_timing": ("measured execution intervals", "measured span timing", "unavailable"),
}


def coverage(header: SessionHeader, activity: Activity) -> dict[str, Any]:
    native = native_metadata(header)
    records = activity.records
    observed = {
        "messages": bool(records),
        "model": any(call.model for call in activity.calls.values()),
        "usage": bool(activity.usage) and activity.provenance.get("usage_aggregation", {}).get("complete", False),
        "cost": bool(activity.cost) and activity.provenance["cost_aggregation"]["complete"].get("total_cost", False),
        "timing": bool(records) and all(record.timestamp is not None for record in records),
        "user": bool(native.get("mlflow.trace.user") or native.get("mlflow.user")),
        "git": any(native.get("mlflow.source.git." + key) for key in ("repoURL", "commit", "branch")),
        "git_repo_url": bool(native.get("mlflow.source.git.repoURL")),
        "git_commit": bool(native.get("mlflow.source.git.commit")),
        "git_branch": bool(native.get("mlflow.source.git.branch")),
        "directory": header.directory is not None,
        "harness_version": header.version is not None,
        "per_call_usage": bool(activity.calls)
        and all(
            observation.get("usage_complete", False)
            for observation in activity.provenance["invocations"].values()
            if not observation.get("aggregate_children")
        ),
        "execution_timing": False,
    }
    # Structured evidence comes from the same projection, with independent
    # fixture tests asserting its content; raw presence alone is not coverage.
    # Tools/agents/skills/thinking are optional observations, not fabricated events.
    from mlflow_import.content.accounting import object_map

    observed["agents"] = any(record.raw_columns.get("subagent") for record in records)
    observed.update(tools=False, thinking=False, skills=False)
    for record in records:
        value = object_map(record.decoded_value)
        parts = [value]
        content = value.get("content")
        if isinstance(content, list):
            parts.extend(object_map(part) for part in content)
        if isinstance(value.get("toolCalls"), list):
            parts.extend({**object_map(tool), "type": "toolCall"} for tool in value["toolCalls"])
        observed["thinking"] |= bool(value.get("thoughts"))
        for part in parts:
            kind = str(part.get("type"))
            observed["tools"] |= (
                kind in {"tool", "toolCall", "tool_use", "tool_result", "function_call", "function_call_output"}
                or part.get("role") == "toolResult"
            )
            observed["thinking"] |= kind in {"reasoning", "thinking"} or part.get("thought") is True
            observed["skills"] |= str(part.get("name", part.get("tool", ""))).lower() in {"skill", "skills"}
    harness = harness_name(header)
    declaration = object_map(COVERAGE_TABLES.get(header.catalog_version, {}))
    declared_items = object_map(declaration.get("items"))
    return {
        "version": COVERAGE_VERSION,
        "source_version": header.catalog_version,
        "source_variant": header.adapter_generation,
        "declaration_supported": bool(declaration),
        "declaration": declaration,
        "baseline": "persisted-shared-contract" if harness == "pi" else "official-integration-2026-11-13",
        "items": {
            item: {
                "status": object_map(declared_items.get(item)).get("status", status)
                if observed[item] and declaration
                else "unavailable",
                "fields": object_map(declaration.get("source_fields")).get(item, fields),
                "target": target,
                "transformation": ACCOUNTING_VERSION if item in {"usage", "cost", "per_call_usage"} else "activity-v2",
                "units": object_map(declared_items.get(item)).get("units"),
                "scope": object_map(declared_items.get(item)).get("scope"),
                "observed": observed[item],
                "details": (
                    activity.provenance.get("usage_aggregation")
                    if item in {"usage", "per_call_usage"}
                    else activity.provenance.get("cost_aggregation")
                    if item == "cost"
                    else None
                ),
                "limitation": (
                    "partial observed costs; full-turn cost unavailable"
                    if item == "cost" and activity.cost and not observed[item]
                    else None
                    if observed[item]
                    else "not observed in this effective turn"
                ),
            }
            for item, (fields, target, status) in BASELINE.items()
        },
    }
