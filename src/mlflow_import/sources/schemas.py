"""Versioned source qualifications, independent of MLflow target parity.

These declarations describe persisted facts, not live-plugin completeness.
Normalization must retain original counters alongside these semantics.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceFormat:
    identifier: str
    harness: str
    variants: tuple[str, ...]
    persisted_path: str
    usage_scope: str
    limitations: tuple[str, ...]


FORMATS = {
    "opencode": SourceFormat(
        "opencode-schema-catalog-v1",
        "opencode",
        ("legacy-sqlite", "current-sqlite", "mixed-sqlite"),
        "V2 rows precede legacy rows; catalogued source order",
        "model message/part; source total may additionally include reasoning",
        ("live-only user and exact execution intervals may be unavailable",),
    ),
    "claude": SourceFormat(
        "claude-persisted-v2",
        "claude",
        ("session-jsonl", "session-jsonl-gzip", "metadata-prelude-jsonl", "fork-context-ref-sidecar"),
        "last persisted conversational UUID head per primary/subagent scope, or explicit linear export",
        "message.id invocation shared by assistant fragments; cache counters are input subsets after normalization",
        (
            "no guessed in-memory leaf",
            "unassociated sidecars are not assigned to a session",
            "fork-context-ref locators are not opened; later session identity must agree with the owning primary",
        ),
    ),
    "codex": SourceFormat(
        "codex-persisted-v2",
        "codex",
        ("rollout-session-meta-jsonl", "explicit-session-id-export", "jsonl-gzip"),
        "persisted response_item order; event_msg.user_message is not a second prompt",
        "token_count last/cumulative observations do not establish complete per-invocation totals",
        ("line fallback identity changes when earlier no-ID records are inserted", "unknown rollback variants reject"),
    ),
    "pi": SourceFormat(
        "pi-persisted-v2",
        "pi",
        ("session-v3-jsonl", "session-v3-jsonl-gzip"),
        "last persisted entry is head; indexed parentId path; compaction/context edits do not erase historical turns",
        "assistant invocation; reasoning already included in output; preserve source cost components",
        ("v1/v2 migrations unsupported", "no official MLflow plugin parity claim"),
    ),
    "gemini": SourceFormat(
        "gemini-persisted-v2",
        "gemini",
        ("partial-metadata-jsonl", "partial-metadata-jsonl-gzip", "registered-project-jsonl"),
        "ordered IDs with replacement, patch, remove/reorder, rewind and checkpoint replay",
        "message tokens; cached/thinking/tool counters retained with source totals",
        (
            "legacy monolithic JSON unsupported",
            "creation-time-changing metadata updates unsupported",
            "selected/unknown duplicate native identities require authority; timestamps do not choose a representation",
            "directory lookup requires matching persisted registry group and SHA-256 registered-path projectHash",
            "live OTLP operation timing/cost/user fields may be unavailable",
        ),
    ),
}

# Source-version-bound baseline declarations. The target convention is shared;
# adapters declare only where the evidence resides and its accounting scope.
COVERAGE_FIELDS = {
    "opencode": {
        "messages": "message/part.data.content",
        "usage": "message.data.tokens",
        "cost": "message.data.cost (USD)",
        "thinking": "part.type=reasoning",
        "tools": "part.type=tool,state,callID",
    },
    "claude": {
        "messages": "message.content",
        "usage": "message.id + message.usage; cache input subsets",
        "cost": "persisted cost with declared currency, or versioned API rates",
        "agents": "subagents + parentToolUseID/agentId",
        "skills": "message.content[type=tool_use,name=Skill]",
    },
    "codex": {
        "messages": "response_item.payload.content",
        "usage": "event_msg.info.total_token_usage differences; last_token_usage fallback",
        "cost": "turn_context.model + qualified turn usage + API rates",
        "git": "session_meta.payload.git",
        "tools": "response_item.function_call/function_call_output.call_id",
    },
    "pi": {
        "messages": "message.content",
        "usage": "message.usage; input + cache subsets; reasoning included in output",
        "cost": "message.usage.cost (USD)",
        "tools": "content.toolCall + message.toolResult.toolCallId",
    },
    "gemini": {
        "messages": "content/parts",
        "usage": "tokens.input/output/cached/thoughts/total",
        "cost": "declared persisted costs or model/usage API estimate",
        "thinking": "thoughts/parts.thought",
        "tools": "toolCalls.id,args,result",
        "agents": "subagent files + toolCalls.agentId",
    },
}
COVERAGE_TABLES: dict[str, dict[str, Any]] = {
    descriptor.identifier: {
        "harness": name,
        "evidence_version": "integration-baseline-2026-11-13-v1",
        "source_fields": COVERAGE_FIELDS[name],
        "scope": descriptor.usage_scope,
        "limitations": descriptor.limitations,
    }
    for name, descriptor in FORMATS.items()
}

# Every declaration binds field locations, units, scope and target to a fixture
# profile. Missing observations are resolved by the shared projector, not this
# capability table. Codex does not prove per-call accounting or live timing.
_BASELINE_TARGETS = {
    "messages": ("structured root/chat content", "source content", "turn", "direct"),
    "model": ("mlflow.llm.model/provider", "model identity", "invocation", "direct"),
    "tools": ("TOOL inputs/outputs/call ID", "structured JSON", "call", "derived"),
    "usage": ("mlflow.chat.tokenUsage + native usage", "tokens", "turn", "derived"),
    "cost": ("mlflow.llm.cost + native costs", "USD", "turn", "derived"),
    "timing": ("native trace/span intervals", "nanoseconds to milliseconds", "turn", "derived"),
    "user": ("mlflow.trace.user/mlflow.user", "source identity", "session", "direct"),
    "git": ("mlflow.source.git.*", "historical Git identity", "session", "direct"),
    "thinking": ("assistant reasoning_content/details", "source content", "invocation", "direct"),
    "agents": ("nested AGENT parentage", "recorded IDs", "agent", "derived"),
    "skills": ("TOOL skill activity", "structured JSON", "call", "direct"),
    "directory": ("source directory metadata", "source path", "session", "direct"),
    "harness_version": ("source version metadata", "version", "session", "direct"),
    "git_repo_url": ("mlflow.source.git.repoURL", "URL", "session", "direct"),
    "git_commit": ("mlflow.source.git.commit", "commit ID", "session", "direct"),
    "git_branch": ("mlflow.source.git.branch", "branch", "session", "direct"),
    "per_call_usage": ("LLM mlflow.chat.tokenUsage", "tokens", "invocation", "derived"),
    "execution_timing": ("measured span timing", "nanoseconds", "invocation", "unavailable"),
}
for _version, _declaration in COVERAGE_TABLES.items():
    _harness = _declaration["harness"]
    _declaration["items"] = {
        item: {
            "status": "unavailable" if item == "per_call_usage" and _harness == "codex" else status,
            "fields": COVERAGE_FIELDS[_harness].get(
                item,
                {
                    "timing": "source timestamps",
                    "user": "metadata.user/user_id/mlflow.trace.user",
                    "git": "metadata.git",
                    "directory": "cwd/directory",
                    "harness_version": "version/cli_version",
                    "model": "model/provider/modelID",
                    "per_call_usage": COVERAGE_FIELDS[_harness]["usage"],
                }.get(item, COVERAGE_FIELDS[_harness].get("git", item)),
            ),
            "units": units,
            "scope": scope,
            "target": target,
            "transformation": "accounting-v2" if item in {"usage", "cost", "per_call_usage"} else "activity-v2",
            "evidence_version": "integration-baseline-2026-11-13-v1",
            "limitations": _declaration["limitations"],
        }
        for item, (target, units, scope, status) in _BASELINE_TARGETS.items()
    }
    if _harness == "opencode":
        _declaration["variants"] = {
            "legacy": {"record_tables": ["message", "part"], "model_scope": "assistant message/step"},
            "current": {"record_tables": ["session_message"], "model_scope": "assistant message/derived step"},
        }
