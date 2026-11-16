"""Read back literal source expectations through public MLflow APIs."""

import gzip
import json
import sqlite3
from dataclasses import replace

import pytest

from mlflow_import.config import parse_config
from mlflow_import.importer import build_dependencies, exit_status, run_import
from tests.integration.server import unique_experiment_name

pytestmark = pytest.mark.integration


def diagnosed_dependencies(config):
    dependencies = build_dependencies(config)
    failures = []
    factory = dependencies.gateway_factory

    def gateway_factory(dry_run):
        gateway = factory(dry_run)
        for name in ("create_trace", "verify_trace"):
            original = getattr(gateway, name)

            def call(*args, operation=original):
                try:
                    return operation(*args)
                except Exception as error:
                    failures.append(str(error.__cause__ or error))
                    if operation.__name__ == "verify_trace":
                        created, request = args
                        trace = live_server_trace = gateway._client.get_trace(created.trace_id, display=False)
                        root = next(span for span in live_server_trace.data.spans if span.parent_id is None)
                        failures.extend(
                            (key, value, root.attributes.get(key))
                            for key, value in request.root_attributes.items()
                            if root.attributes.get(key) != value
                        )
                        failures.append(("native", trace.info.token_usage, trace.info.cost))
                    raise

            setattr(gateway, name, call)
        return gateway

    return replace(dependencies, gateway_factory=gateway_factory), failures


def test_opencode_native_literal_readback(live_server, tmp_path):
    from mlflow_import.mlflow.gateway import MlflowGateway
    from tests.fixtures.six_turn_session import build_six_turn_source_db

    source = build_six_turn_source_db(tmp_path)
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("parity-opencode"),
            "--source",
            str(source),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    dependencies = build_dependencies(config)
    failures = []

    class CheckedGateway(MlflowGateway):
        def verify_trace(self, created, request):
            try:
                return super().verify_trace(created, request)
            except Exception as error:
                failures.append((request.name, str(error)))
                trace = live_server.client.get_trace(created.trace_id, display=False)
                failures.append((
                    "previews", request.request_preview, trace.info.request_preview,
                    request.response_preview, trace.info.response_preview,
                ))
                failures.append(
                    (
                        "native",
                        request.root_attributes.get("mlflow.llm.cost"),
                        trace.info.cost,
                        request.root_attributes.get("mlflow.chat.tokenUsage"),
                        trace.info.token_usage,
                    )
                )
                root = next(span for span in trace.data.spans if span.parent_id is None)
                failures.extend(
                    (key, value, root.attributes.get(key))
                    for key, value in request.root_attributes.items()
                    if root.attributes.get(key) != value
                )
                raise

        def create_trace(self, request):
            try:
                return super().create_trace(request)
            except Exception as error:
                failures.append((request.name, repr(error.__cause__ or error)))
                raise

    result = run_import(
        config,
        replace(dependencies, gateway_factory=lambda dry_run: CheckedGateway(live_server.client, dry_run=dry_run)),
    )
    if exit_status(result) != 0:
        pytest.fail(repr(failures or result.report))
    first = result.report["sessions"][0]["turns"][0]
    trace = live_server.client.get_trace(first["trace_id"], display=False)
    assert trace.info.token_usage == {
        "input_tokens": 120,
        "output_tokens": 80,
        "total_tokens": 212,
        "cache_read_input_tokens": 40,
        "cache_creation_input_tokens": 4,
    }
    assert trace.info.cost == {"total_cost": 0.0025}


def source_file(tmp_path, harness):
    common = {"model": "openai/gpt-4o"}
    if harness == "pi":
        values = [
            {"type": "session", "version": 3, "id": "s", "cwd": "/synthetic", "timestamp": "2026-01-01T00:00:00Z"},
            {"type": "message", "id": "u", "parentId": None, "message": {"role": "user", "content": "hello"}},
            {
                "type": "message",
                "id": "a",
                "parentId": "u",
                "message": {
                    "role": "assistant",
                    **common,
                    "content": [{"type": "text", "text": "answer"}],
                    "usage": {"input": 100, "output": 10, "cacheRead": 20, "totalTokens": 130},
                    "stopReason": "stop",
                },
            },
        ]
        expected = {"input_tokens": 120, "output_tokens": 10, "total_tokens": 130, "cache_read_input_tokens": 20}
    elif harness == "claude":
        values = [
            {
                "type": "user",
                "uuid": "u",
                "sessionId": "s",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": "hello"},
            },
            {
                "type": "assistant",
                "uuid": "a",
                "parentUuid": "u",
                "sessionId": "s",
                "message": {
                    "id": "invocation",
                    "role": "assistant",
                    **common,
                    "content": [{"type": "text", "text": "answer"}],
                    "usage": {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 20},
                    "stopReason": "stop",
                },
            },
        ]
        expected = {"input_tokens": 120, "output_tokens": 10, "total_tokens": 130, "cache_read_input_tokens": 20}
    elif harness == "gemini":
        values = [
            {"sessionId": "s", "projectHash": "synthetic", "startTime": "2026-01-01T00:00:00Z"},
            {"id": "u", "type": "user", "content": "hello"},
            {
                "id": "a",
                "type": "gemini",
                **common,
                "content": "answer",
                "tokens": {"input": 120, "output": 10, "cached": 20, "total": 130},
            },
        ]
        expected = {"input_tokens": 120, "output_tokens": 10, "total_tokens": 130, "cache_read_input_tokens": 20}
    else:
        values = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "s",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "cwd": "/synthetic",
                    "git": {
                        "commit_hash": "abc",
                        "repository_url": "https://example.test/repo",
                        "branch": "historical",
                    },
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "id": "u",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
            {"type": "turn_context", "payload": {"model": "gpt-4o", "model_provider": "openai"}},
            {
                "type": "response_item",
                "payload": {
                    "id": "a",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 120,
                            "output_tokens": 10,
                            "cached_input_tokens": 20,
                            "total_tokens": 130,
                        }
                    },
                },
            },
        ]
        expected = {"input_tokens": 120, "output_tokens": 10, "total_tokens": 130, "cache_read_input_tokens": 20}
    if harness == "codex":
        values[0]["payload"]["source"] = "conflicting-source"
    else:
        values[0]["source"] = "conflicting-source"
    path = tmp_path / f"{harness}.jsonl"
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    return path, expected


@pytest.mark.parametrize("harness", ["claude", "codex", "pi", "gemini"])
@pytest.mark.parametrize("archive", [False, True])
@pytest.mark.parametrize("redact", [False, True])
def test_literal_native_and_structured_readback(live_server, tmp_path, harness, archive, redact):
    source, usage = source_file(tmp_path, harness)
    if archive:
        compressed = source.with_suffix(".jsonl.gz")
        with gzip.open(compressed, "wt") as handle:
            handle.write(source.read_text())
        source.unlink()
        source = compressed
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("parity-" + harness),
            "--source",
            str(source),
            "--tracking-uri",
            live_server.uri,
            "--harness",
            harness,
            *(["--redact"] if redact else []),
        ]
    )
    dependencies = build_dependencies(config)
    failures = []
    factory = dependencies.gateway_factory

    def gateway_factory(dry_run):
        gateway = factory(dry_run)
        original = gateway.create_trace

        def create(request):
            try:
                return original(request)
            except Exception as error:
                failures.append(repr(error.__cause__ or error))
                raise

        gateway.create_trace = create
        verify = gateway.verify_trace

        def verify_trace(created, request):
            try:
                return verify(created, request)
            except Exception as error:
                failures.append(str(error))
                raise

        gateway.verify_trace = verify_trace
        return gateway

    result = run_import(config, replace(dependencies, gateway_factory=gateway_factory))
    assert exit_status(result) == 0, (result.report, failures)
    (session,) = result.report["sessions"]
    (turn,) = session["turns"]
    trace = live_server.client.get_trace(turn["trace_id"], display=False)
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert root.span_type == "AGENT"
    assert trace.info.tags["mlflow.source.name"] == harness
    assert trace.info.tags["mlflow_import.source"] == harness
    assert "mlflow.trace.user" not in trace.info.trace_metadata
    assert "mlflow.user" not in trace.info.trace_metadata
    assert trace.info.token_usage == usage
    assert trace.info.cost == {"input_cost": 0.000275, "output_cost": 0.0001, "total_cost": 0.000375}
    assert trace.info.client_request_id is None
    assert trace.info.tags["mlflow_import.client_request_id_origin"] == "absent"
    assert trace.info.request_time == 1767225600000
    assert trace.info.execution_duration == 0
    assert trace.info.experiment_id == result.report["experiment"]["id"]
    assert "hello" in trace.info.request_preview and "answer" in trace.info.response_preview
    assert trace.info.tags["mlflow.traceName"] == harness
    assert trace.info.trace_metadata["mlflow.trace.session"] == "s"
    identity = json.loads(trace.info.tags["mlflow_import.qualified_session_id"])
    assert identity[:2] == ["v1", harness] and identity[-1] == "s" and len(identity[2]) == 64
    assert root.attributes["mlflow_import.timing"] in {"timestamp-derived", "synthetic-container"}
    (model,) = [span for span in trace.data.spans if span.span_type == "LLM"]
    assert model.attributes["mlflow.llm.model"] == "gpt-4o"
    assert model.outputs["choices"][0]["message"]["content"]
    assert model.parent_id == root.span_id
    if harness in {"gemini", "codex"}:
        assert model.status.status_code.value == "UNSET"
    from mlflow.artifacts import get_artifact_repository

    repository = get_artifact_repository(
        live_server.client.get_experiment(result.report["experiment"]["id"]).artifact_location,
        tracking_uri=live_server.uri,
    )
    bundle_path = repository.download_artifacts(trace.info.tags["mlflow_import.session_artifact"], str(tmp_path))
    with open(bundle_path) as handle:
        bundle = json.load(handle)
    assert bundle["session"]["source"] == harness
    assert bundle["projections"][0]["coverage"]["source_version"] == harness + "-persisted-v2"
    if harness == "codex":
        assert "mlflow.chat.tokenUsage" not in model.attributes
        assert trace.info.tags["mlflow.source.git.commit"] == "abc"
    else:
        assert model.attributes["mlflow.chat.tokenUsage"] == usage
    repeated = run_import(config, dependencies)
    assert exit_status(repeated) == 0, repeated.report
    assert repeated.report["sessions"][0]["turns"][0]["trace_id"] == turn["trace_id"]


@pytest.mark.parametrize("generation", ["legacy", "current", "mixed"])
def test_opencode_part_usage_cost_native_fields_and_immutable_metadata(live_server, tmp_path, generation):
    from tests.support.import_fakes import CURRENT_DDL, LEGACY_DDL

    path = tmp_path / "opencode.db"
    parts = [
        {
            "type": "step-finish",
            "tokens": {"input": 100, "output": 10, "reasoning": 2, "cache": {"read": 20, "write": 0}},
            "cost": 0.25,
        },
        {
            "type": "tool",
            "tool": "shell",
            "callID": "call",
            "state": {"input": {"command": "synthetic"}, "output": "tool answer", "status": "completed"},
        },
        {"type": "text", "text": "literal answer"},
    ]
    assistant = {"role": "assistant", "model": "openai/gpt-4o", "finish": "stop"}
    metadata = json.dumps(
        {
            "user": "historical-user",
            "mlflow.source.name": "conflicting-source",
            "git": {"repository_url": "https://example.test/repo", "commit_hash": "abc", "branch": "historical"},
        }
    )
    with sqlite3.connect(path) as connection:
        if generation in {"legacy", "mixed"}:
            connection.executescript(LEGACY_DDL)
            connection.execute(
                "INSERT INTO session(id,parent_id,time_created,metadata) VALUES('s','parent',1000,?)", (metadata,)
            )
            for id_, at, value in [("u", 1000, {"role": "user"}), ("a", 1002, assistant)]:
                connection.execute("INSERT INTO message VALUES(?, 's', ?, ?, ?)", (id_, at, at, json.dumps(value)))
            for id_, message_id, at, value in [
                ("prompt", "u", 1000, {"type": "text", "text": "literal prompt"}),
                *((f"p{index}", "a", 1003 + index, value) for index, value in enumerate(parts)),
            ]:
                connection.execute(
                    "INSERT INTO part VALUES(?,?,'s',?,?,?)", (id_, message_id, at, at, json.dumps(value))
                )
        if generation in {"current", "mixed"}:
            connection.executescript(CURRENT_DDL)
            connection.execute(
                "INSERT INTO session_v2(id,parent_id,time_created,metadata) VALUES('s','parent',1000,?)", (metadata,)
            )
            connection.execute(
                "INSERT INTO session_message VALUES('u','s','user',0,1000,1000,?)",
                (json.dumps({"content": [{"type": "text", "text": "literal prompt"}]}),),
            )
            connection.execute(
                "INSERT INTO session_message VALUES('a','s','assistant',1,1002,1005,?)",
                (json.dumps({**assistant, "content": parts}),),
            )
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("opencode-" + generation),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    (turn,) = result.report["sessions"][0]["turns"]
    trace = live_server.client.get_trace(turn["trace_id"], display=False)
    assert trace.info.experiment_id == result.report["experiment"]["id"]
    assert trace.info.request_time == 1000
    assert trace.info.execution_duration == 2
    assert trace.info.state.value == "OK"
    assert "literal prompt" in trace.info.request_preview
    assert "literal answer" in trace.info.response_preview
    assert trace.info.tags["mlflow.traceName"] == "opencode"
    assert trace.info.token_usage == {
        "input_tokens": 100,
        "output_tokens": 10,
        "total_tokens": 112,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 0,
    }
    assert trace.info.cost == {"total_cost": 0.25}
    assert trace.info.tags["mlflow.trace.user"] == "historical-user"
    assert trace.info.tags["mlflow.source.name"] == "opencode"
    assert trace.info.tags["mlflow.source.git.commit"] == "abc"
    assert trace.info.trace_metadata["parent_session_id"] == "parent"
    (model,) = [span for span in trace.data.spans if span.span_type == "LLM"]
    assert model.attributes["mlflow.chat.tokenUsage"] == {
        **trace.info.token_usage, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 0
    }
    assert model.attributes["mlflow.llm.cost"] == {"total_cost": 0.25}
    assert model.attributes["mlflow_import.accounting"]["origin"] == "persisted"
    assert model.attributes["mlflow_import.accounting"]["method"] == "unknown"
    assert [
        part["text"] for part in model.outputs["choices"][0]["message"]["content"] if part.get("type") == "text"
    ] == ["literal answer"]
    assert any(span.span_type == "TOOL" and span.outputs == "tool answer" for span in trace.data.spans)


def test_codex_explicit_session_export_native_readback(live_server, tmp_path):
    path, usage = source_file(tmp_path, "codex")
    values = [json.loads(line) for line in path.read_text().splitlines()][1:]
    for value in values:
        value["session_id"] = "s"
    values[0]["timestamp"] = "2026-01-01T00:00:00Z"
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("codex-export"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    assert trace.info.token_usage == usage
    assert trace.info.cost["total_cost"] == 0.000375


def test_codex_model_context_before_primary_prompt_is_attributable(live_server, tmp_path):
    path, _ = source_file(tmp_path, "codex")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    context = values.pop(2)
    values.insert(1, context)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("codex-prior-context"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    (model,) = [span for span in trace.data.spans if span.span_type == "LLM"]
    assert model.attributes["mlflow.llm.model"] == "gpt-4o"
    assert trace.info.cost["total_cost"] == 0.000375


def test_claude_split_invocation_root_and_record_map_are_coalesced(live_server, tmp_path):
    from mlflow.artifacts import get_artifact_repository

    path, _ = source_file(tmp_path, "claude")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[1]["message"].pop("stopReason")
    values.append(
        {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "a",
            "sessionId": "s",
            "message": {
                "id": "invocation",
                "role": "assistant",
                "content": [{"type": "text", "text": " remainder"}],
                "usage": {"input_tokens": 100, "output_tokens": 11, "cache_read_input_tokens": 20},
                "stopReason": "stop",
            },
        }
    )
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("split-call"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    (model,) = [span for span in trace.data.spans if span.span_type == "LLM"]
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert root.outputs == [{"type": "text", "text": "answer"}, {"type": "text", "text": " remainder"}]
    assert model.outputs["choices"][0]["message"]["content"] == root.outputs
    assert trace.info.token_usage == {
        "input_tokens": 120,
        "output_tokens": 11,
        "total_tokens": 131,
        "cache_read_input_tokens": 20,
    }
    assert trace.info.cost["total_cost"] == 0.000385
    experiment = live_server.client.get_experiment(result.report["experiment"]["id"])
    repository = get_artifact_repository(experiment.artifact_location, tracking_uri=live_server.uri)
    map_path = repository.download_artifacts(trace.info.tags["mlflow_import.turn_map"], str(tmp_path))
    with open(map_path) as handle:
        records = {record["source_id"]: record for record in json.load(handle)["records"]}
    assert records["a"]["span_ids"] == records["a2"]["span_ids"] == [model.span_id]


def test_claude_native_creation_cache_ttl_and_estimate(live_server, tmp_path):
    path, _ = source_file(tmp_path, "claude")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[1]["message"].update(
        model="claude-sonnet-4-20250514",
        usage={
            "input_tokens": 10,
            "output_tokens": 3,
            "cache_read_input_tokens": 2,
            "cache_creation_input_tokens": 4,
            "cache_creation": {"ephemeral_1h_input_tokens": 1},
        },
    )
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("claude-cache-ttl"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    dependencies, failures = diagnosed_dependencies(config)
    result = run_import(config, dependencies)
    if exit_status(result) != 0:
        pytest.fail(repr(failures))
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    assert trace.info.token_usage == {
        "input_tokens": 16,
        "output_tokens": 3,
        "total_tokens": 19,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 4,
        "cache_creation_input_tokens_above_1hr": 1,
    }
    assert trace.info.cost == {"input_cost": 0.00004785, "output_cost": 0.000045, "total_cost": 0.00009285}


def test_gemini_effective_replay_native_accounting_and_thinking(live_server, tmp_path):
    path, _ = source_file(tmp_path, "gemini")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[2]["model"] = "superseded-model"
    values.extend(
        [
            {
                "id": "a",
                "type": "gemini",
                "model": "openai/gpt-4o",
                "content": "draft",
                "tokens": {"input": 7, "output": 2, "thoughts": 2, "cached": 1, "total": 11},
                "thoughts": [{"description": "synthetic reasoning"}],
            },
            {"$patch": {"id": "a", "content": "effective answer"}},
            {"id": "discard", "type": "user", "content": "rewound"},
            {"$rewindTo": "discard"},
        ]
    )
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("gemini-replay"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    assert result.report["source"]["logical_turn_count"] == 1
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    assert trace.info.token_usage == {
        "input_tokens": 7,
        "output_tokens": 4,
        "total_tokens": 11,
        "cache_read_input_tokens": 1,
    }
    assert trace.info.cost["total_cost"] == 0.00005625
    (model,) = [span for span in trace.data.spans if span.span_type == "LLM"]
    assert model.outputs["choices"][0]["message"]["content"] == [{"type": "text", "text": "effective answer"}]
    assert model.outputs["choices"][0]["message"]["reasoning_content"] == "synthetic reasoning"
    assert model.outputs["choices"][0]["message"]["reasoning_details"] == [{"description": "synthetic reasoning"}]


@pytest.mark.parametrize("harness", ["claude", "codex", "pi", "gemini"])
def test_correlated_tools_and_thinking_are_structured(live_server, tmp_path, harness):
    path, _ = source_file(tmp_path, harness)
    values = [json.loads(line) for line in path.read_text().splitlines()]
    if harness == "claude":
        values[1]["message"]["content"] += [
            {"type": "thinking", "thinking": "synthetic reasoning"},
            {"type": "tool_use", "id": "call", "name": "Skill", "input": {"name": "synthetic"}},
        ]
        values.extend(
            [
                {
                    "type": "user",
                    "uuid": "result",
                    "parentUuid": "a",
                    "sessionId": "s",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "call", "content": "tool answer", "is_error": True}
                        ],
                    },
                },
                {
                    "type": "assistant",
                    "uuid": "final",
                    "parentUuid": "result",
                    "sessionId": "s",
                    "message": {
                        "id": "final-call",
                        "role": "assistant",
                        "content": "final answer",
                        "stopReason": "stop",
                    },
                },
            ]
        )
    elif harness == "pi":
        values[2]["message"]["content"] += [
            {"type": "thinking", "thinking": "synthetic reasoning"},
            {"type": "toolCall", "id": "call", "name": "Skill", "arguments": {"name": "synthetic"}},
        ]
        values.extend(
            [
                {
                    "type": "message",
                    "id": "result",
                    "parentId": "a",
                    "message": {"role": "toolResult", "toolCallId": "call", "content": "tool answer", "isError": True},
                },
                {
                    "type": "message",
                    "id": "final",
                    "parentId": "result",
                    "message": {"role": "assistant", "content": "final answer", "stopReason": "stop"},
                },
            ]
        )
    elif harness == "gemini":
        values[2]["toolCalls"] = [
            {"id": "call", "name": "Skill", "args": {"name": "synthetic"}, "result": "tool answer", "isError": True}
        ]
        values[2]["thoughts"] = [{"subject": "synthetic", "description": "synthetic reasoning"}]
    else:
        values[-1:-1] = [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "call",
                    "name": "Skill",
                    "arguments": '{"name":"synthetic"}',
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call",
                    "output": "tool answer",
                    "is_error": True,
                },
            },
        ]
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("tools-" + harness),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    dependencies = build_dependencies(config)
    failures = []
    factory = dependencies.gateway_factory

    def checked_factory(dry_run):
        gateway = factory(dry_run)
        original = gateway.verify_trace

        def verify(created, request):
            try:
                return original(created, request)
            except Exception as error:
                failures.append(str(error))
                raise

        gateway.verify_trace = verify
        return gateway

    result = run_import(config, replace(dependencies, gateway_factory=checked_factory))
    assert exit_status(result) == 0, failures or result.report
    assert result.report["source"]["logical_turn_count"] == 1
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    (tool,) = [span for span in trace.data.spans if span.span_type == "TOOL"]
    assert tool.name == "Skill"
    assert tool.outputs == "tool answer"
    assert tool.status.status_code.value == "ERROR"
    assert tool.attributes["mlflow_import.tool_call_id"] == "call"
    assert tool.attributes["mlflow_import.pairing"] == "paired"
    assert tool.inputs in ({"name": "synthetic"}, '{"name":"synthetic"}')
    if harness != "codex":
        assert any(
            span.outputs and span.outputs.get("choices", [{}])[0].get("message", {}).get("reasoning_content")
            for span in trace.data.spans
            if span.span_type == "LLM"
        )


def test_pricing_basis_reused_for_unchanged_and_append(live_server, tmp_path, monkeypatch):
    from mlflow_import.content.accounting import PRICES

    path, _ = source_file(tmp_path, "claude")
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("pricing-reuse"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    first = run_import(config, build_dependencies(config))
    assert exit_status(first) == 0, first.report
    first_id = first.report["sessions"][0]["turns"][0]["trace_id"]
    monkeypatch.setitem(PRICES, "openai/gpt-4o", ("100", "200", "50"))
    unchanged = run_import(config, build_dependencies(config))
    assert exit_status(unchanged) == 0, unchanged.report
    assert unchanged.report["sessions"][0]["turns"][0]["trace_id"] == first_id
    with path.open("a") as handle:
        for value in [
            {
                "type": "user",
                "uuid": "u2",
                "parentUuid": "a",
                "sessionId": "s",
                "message": {"role": "user", "content": "new prompt"},
            },
            {
                "type": "assistant",
                "uuid": "a2",
                "parentUuid": "u2",
                "sessionId": "s",
                "message": {
                    "id": "new-call",
                    "role": "assistant",
                    "model": "openai/gpt-4o",
                    "content": "new answer",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stopReason": "stop",
                },
            },
        ]:
            handle.write(json.dumps(value) + "\n")
    appended = run_import(config, build_dependencies(config))
    assert exit_status(appended) == 0, appended.report
    turns = appended.report["sessions"][0]["turns"]
    assert turns[0]["trace_id"] == first_id
    assert live_server.client.get_trace(first_id, display=False).info.cost["total_cost"] == 0.000375
    assert live_server.client.get_trace(turns[1]["trace_id"], display=False).info.cost["total_cost"] == 0.0003


def test_claude_recorded_subagent_parentage(live_server, tmp_path):
    path, _ = source_file(tmp_path, "claude")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[1]["message"]["content"].append(
        {"type": "tool_use", "id": "call", "name": "Task", "input": {"prompt": "delegate"}}
    )
    values.extend(
        [
            {
                "type": "user",
                "uuid": "r",
                "parentUuid": "a",
                "sessionId": "s",
                "toolUseResult": {"agentId": "child"},
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "call", "content": "done"}],
                },
            },
            {
                "type": "assistant",
                "uuid": "final",
                "parentUuid": "r",
                "sessionId": "s",
                "message": {"id": "final-call", "role": "assistant", "content": "final", "stopReason": "stop"},
            },
        ]
    )
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    side = tmp_path / "claude" / "subagents" / "agent-child.jsonl"
    side.parent.mkdir(parents=True)
    side.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "nested",
                "parentUuid": "a",
                "parentToolUseID": "call",
                "sessionId": "s",
                "message": {
                    "id": "nested-call",
                    "role": "assistant",
                    "model": "openai/gpt-4o",
                    "content": "nested answer",
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                    "stopReason": "stop",
                },
            }
        )
        + "\n"
    )
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("nested-agent"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    assert result.report["source"]["logical_turn_count"] == 1
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    (tool,) = [span for span in trace.data.spans if span.span_type == "TOOL"]
    (agent,) = [span for span in trace.data.spans if "mlflow_import.agent_scope" in span.attributes]
    assert agent.parent_id == tool.span_id, agent.attributes
    assert agent.attributes["mlflow_import.execution_parentage"] == "recorded"
    (model,) = [span for span in trace.data.spans if span.parent_id == agent.span_id and span.span_type == "LLM"]
    assert model.attributes["mlflow.chat.tokenUsage"] == {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
    assert trace.info.token_usage == {
        "input_tokens": 122,
        "output_tokens": 11,
        "total_tokens": 133,
        "cache_read_input_tokens": 20,
    }


def test_gemini_recorded_subagent_parentage(live_server, tmp_path):
    from mlflow_import.source.file_snapshot import create_file_snapshot
    from mlflow_import.source.replay import NormalizedSnapshotAdapter
    from mlflow_import.sources.jsonl_common import JsonlExporter

    main = tmp_path / "main.jsonl"
    main.write_text(
        "".join(
            json.dumps(value) + "\n"
            for value in [
                {"sessionId": "main", "projectHash": "synthetic", "startTime": "2026-01-01T00:00:00Z"},
                {"id": "u", "type": "user", "content": "first"},
                {
                    "id": "a",
                    "type": "gemini",
                    "model": "openai/gpt-4o",
                    "content": "delegating",
                    "toolCalls": [{"id": "call", "name": "delegate", "agentId": "child", "args": {}, "result": "done"}],
                },
                {"id": "final", "type": "gemini", "content": "final"},
            ]
        )
    )
    side = tmp_path / "main" / "child.jsonl"
    side.parent.mkdir()
    side.write_text(
        "".join(
            json.dumps(value) + "\n"
            for value in [
                {"sessionId": "child", "projectHash": "synthetic", "kind": "subagent"},
                {"id": "su", "type": "user", "content": "delegated"},
                {
                    "id": "sa",
                    "type": "gemini",
                    "model": "openai/gpt-4o",
                    "content": "nested",
                    "tokens": {"input": 2, "output": 1, "total": 3},
                },
            ]
        )
    )
    with create_file_snapshot(main, tmp_path, JsonlExporter(main, "gemini"), lambda value, _: value) as snapshot:
        header = NormalizedSnapshotAdapter(snapshot).session_header("main")
        assert header.metadata["execution_links"]["agent:child.jsonl:"]["call_id"] == "call", header.metadata
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("gemini-agent"),
            "--source",
            str(main),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    dependencies, failures = diagnosed_dependencies(config)
    result = run_import(config, dependencies)
    assert exit_status(result) == 0, failures or result.report
    assert result.report["source"]["logical_turn_count"] == 1
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    (agent,) = [span for span in trace.data.spans if "mlflow_import.agent_scope" in span.attributes]
    (tool,) = [span for span in trace.data.spans if span.span_type == "TOOL"]
    assert agent.parent_id == tool.span_id, agent.attributes
    assert any(span.span_type == "LLM" and span.parent_id == agent.span_id for span in trace.data.spans)


def test_codex_cumulative_observations_difference_across_turns(live_server, tmp_path):
    path, _ = source_file(tmp_path, "codex")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[-1]["payload"]["info"] = {
        "total_token_usage": {"input_tokens": 120, "output_tokens": 10, "total_tokens": 130}
    }
    values.extend(
        [
            {"type": "response_item", "payload": {"id": "u2", "type": "message", "role": "user", "content": "second"}},
            {
                "type": "response_item",
                "payload": {
                    "id": "a2",
                    "type": "message",
                    "role": "assistant",
                    "model": "openai/gpt-4o",
                    "content": "second answer",
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {"total_token_usage": {"input_tokens": 125, "output_tokens": 12, "total_tokens": 137}},
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {"total_token_usage": {"input_tokens": 125, "output_tokens": 12, "total_tokens": 137}},
                },
            },
        ]
    )
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("cumulative"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    turns = result.report["sessions"][0]["turns"]
    assert len(turns) == 2
    assert live_server.client.get_trace(turns[0]["trace_id"], display=False).info.token_usage == {
        "input_tokens": 120,
        "output_tokens": 10,
        "total_tokens": 130,
    }
    assert live_server.client.get_trace(turns[1]["trace_id"], display=False).info.token_usage == {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }


def test_prepared_recovery_retains_recorded_pricing(live_server, tmp_path, monkeypatch):
    from mlflow_import.content.accounting import PRICES
    from mlflow_import.errors import ItemImportError
    from mlflow_import.mlflow.attempts import AttemptJournal

    path, _ = source_file(tmp_path, "claude")
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("basis-recovery"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    original = AttemptJournal.publish

    def fail_after_prepare(journal, state, payload):
        result = original(journal, state, payload)
        if state == "prepared":
            raise ItemImportError("synthetic_after_prepared", "synthetic failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(AttemptJournal, "publish", fail_after_prepare)
        first = run_import(config, build_dependencies(config))
        assert exit_status(first) == 1
    monkeypatch.setitem(PRICES, "openai/gpt-4o", ("100", "200", "50"))
    resumed = run_import(config, build_dependencies(config))
    assert exit_status(resumed) == 0, resumed.report
    trace = live_server.client.get_trace(resumed.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    assert trace.info.cost["total_cost"] == 0.000375


def test_overwrite_basis_donor_is_smallest_completed_duplicate_instance(live_server, tmp_path, monkeypatch):
    from mlflow_import.content.accounting import PRICES

    path, _ = source_file(tmp_path, "claude")
    args = [
        "--experiment-name",
        unique_experiment_name("basis-donor"),
        "--source",
        str(path),
        "--tracking-uri",
        live_server.uri,
    ]
    duplicate = parse_config([*args, "--on-conflict", "duplicate"])
    costs = {}
    for rates in [("2.50", "10", "1.25"), ("100", "200", "50")]:
        monkeypatch.setitem(PRICES, "openai/gpt-4o", rates)
        result = run_import(duplicate, build_dependencies(duplicate))
        assert exit_status(result) == 0, result.report
        trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
        costs[trace.info.tags["mlflow_import.import_instance_id"]] = trace.info.cost["total_cost"]
    expected = costs[min(costs)]
    overwrite = parse_config(args)
    result = run_import(overwrite, build_dependencies(overwrite))
    assert exit_status(result) == 0, result.report
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    assert trace.info.cost["total_cost"] == expected
    assert "mlflow_import.import_instance_id" not in trace.info.trace_metadata
    assert (
        len(live_server.client.search_traces(locations=[result.report["experiment"]["id"]], include_spans=False)) == 1
    )


def test_immutable_user_git_parent_metadata_replaces_and_redacts(live_server, tmp_path):
    path, _ = source_file(tmp_path, "codex")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[0]["payload"].update(
        user={"name": "synthetic-user"}, parent_session_id="parent", client_request_id="source-request"
    )
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    args = [
        "--experiment-name",
        unique_experiment_name("immutable"),
        "--source",
        str(path),
        "--tracking-uri",
        live_server.uri,
    ]
    config = parse_config(args)
    first = run_import(config, build_dependencies(config))
    assert exit_status(first) == 0, first.report
    first_id = first.report["sessions"][0]["turns"][0]["trace_id"]
    trace = live_server.client.get_trace(first_id, display=False)
    assert trace.info.client_request_id == "source-request"
    assert trace.info.tags["mlflow.trace.user"] == "synthetic-user"
    assert trace.info.trace_metadata["parent_session_id"] == "parent"
    values[0]["payload"]["git"]["commit_hash"] = "changed"
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    changed = run_import(config, build_dependencies(config))
    assert exit_status(changed) == 0, changed.report
    changed_id = changed.report["sessions"][0]["turns"][0]["trace_id"]
    assert changed_id != first_id
    redacted_config = parse_config([*args, "--redact"])
    redacted = run_import(redacted_config, build_dependencies(redacted_config))
    assert exit_status(redacted) == 0, redacted.report
    trace = live_server.client.get_trace(redacted.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    assert trace.info.tags["mlflow.trace.user"] == "[REDACTED]"
    assert trace.info.tags["mlflow.source.git.commit"] == "changed"
    assert trace.info.tags["mlflow.source.name"] == "codex"


def test_unavailable_usd_cost_never_uses_server_prices_or_fabricates_zero(live_server, tmp_path, monkeypatch):
    from mlflow_import.mlflow.gateway import MlflowGateway

    path, _ = source_file(tmp_path, "claude")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[1]["message"]["cost"] = {"total": "7", "currency": "EUR", "method": "estimated"}
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("no-server-reprice"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    trace_id = result.report["sessions"][0]["turns"][0]["trace_id"]
    trace = live_server.client.get_trace(trace_id, display=False)
    assert trace.info.cost is None
    model = next(span for span in trace.data.spans if span.span_type == "LLM")
    assert model.attributes["mlflow.llm.cost"] == {}
    root = next(span for span in trace.data.spans if span.parent_id is None)
    assert root.attributes["mlflow_import.coverage"]["items"]["cost"]["status"] == "unavailable"

    def forbidden(*args, **kwargs):
        raise AssertionError("unchanged import attempted a mutation")

    monkeypatch.setattr(MlflowGateway, "_require_mutation_allowed", forbidden)
    repeated = run_import(config, build_dependencies(config))
    assert exit_status(repeated) == 0, repeated.report
    assert repeated.report["sessions"][0]["turns"][0]["trace_id"] == trace_id


def test_unknown_price_remains_frozen_when_installed_mapping_adds_it(live_server, tmp_path, monkeypatch):
    from mlflow_import.content.accounting import MODEL_ALIASES
    from mlflow_import.mlflow.gateway import MlflowGateway

    path, _ = source_file(tmp_path, "claude")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[1]["message"]["model"] = "unknown-model-v1"
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("negative-basis"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    first_id = result.report["sessions"][0]["turns"][0]["trace_id"]
    assert live_server.client.get_trace(first_id, display=False).info.cost is None
    monkeypatch.setitem(MODEL_ALIASES, "/unknown-model-v1", "openai/gpt-4o")
    monkeypatch.setattr(
        MlflowGateway, "_require_mutation_allowed", lambda *args, **kwargs: pytest.fail("negative basis was repriced")
    )
    repeated = run_import(config, build_dependencies(config))
    assert exit_status(repeated) == 0, repeated.report
    assert repeated.report["sessions"][0]["turns"][0]["trace_id"] == first_id


def test_new_basis_is_frozen_before_prepared_publication(live_server, tmp_path, monkeypatch):
    from mlflow_import.content.accounting import PRICES
    from mlflow_import.mlflow.attempts import AttemptJournal

    path, _ = source_file(tmp_path, "claude")
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("freeze-before-writes"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    original = AttemptJournal.publish

    def change_installed_snapshot(journal, state, payload):
        result = original(journal, state, payload)
        if state == "prepared":
            monkeypatch.setitem(PRICES, "openai/gpt-4o", ("100", "200", "50"))
        return result

    monkeypatch.setattr(AttemptJournal, "publish", change_installed_snapshot)
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    assert trace.info.cost == {"input_cost": 0.000275, "output_cost": 0.0001, "total_cost": 0.000375}


def test_unrepresentable_request_identity_is_rejected_before_experiment_creation(live_server, tmp_path):
    path, _ = source_file(tmp_path, "claude")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[0]["request_id"] = "r" * 51
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    name = unique_experiment_name("unsupported-native-request")
    config = parse_config(["--experiment-name", name, "--source", str(path), "--tracking-uri", live_server.uri])
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 2
    assert result.error.code == "native_request_id_too_long"
    assert live_server.client.get_experiment_by_name(name) is None


def test_claude_followup_model_inputs_include_correlated_tool_result(live_server, tmp_path):
    path, _ = source_file(tmp_path, "claude")
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[1]["message"]["content"] = [{"type": "tool_use", "id": "c", "name": "read", "input": {"path": "synthetic"}}]
    values[1]["message"]["stop_reason"] = "tool_use"
    values.extend(
        [
            {
                "sessionId": "s",
                "uuid": "r",
                "parentUuid": "a",
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "c", "content": "literal tool result"}],
                },
            },
            {
                "sessionId": "s",
                "uuid": "final",
                "parentUuid": "r",
                "type": "assistant",
                "message": {
                    "id": "final-inv",
                    "role": "assistant",
                    "model": "openai/gpt-4o",
                    "content": "done",
                    "stop_reason": "end_turn",
                },
            },
        ]
    )
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("chat-tool-history"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    model = next(
        span for span in trace.data.spans if span.attributes.get("mlflow_import.invocation_id") == ":final-inv"
    )
    assert {"role": "tool", "tool_call_id": "c", "content": "literal tool result"} in model.inputs["messages"]
    assert [message["role"] for message in model.inputs["messages"]] == ["assistant", "tool"]
    assert model.inputs["messages"][0]["tool_calls"][0]["id"] == "c"
    context = model.attributes["mlflow_import.input_context"]
    assert context["policy"] == "conversation-delta-v1"
    assert context["previous_context_id"]
    assert context["provider_request"] == "unknown"
    assert trace.info.tags["mlflow_import.turn_fingerprint"]
    assert trace.info.tags["mlflow_import.metadata_fingerprint"]


def test_first_cumulative_observation_in_a_later_turn_does_not_claim_earlier_usage(live_server, tmp_path):
    path, _ = source_file(tmp_path, "codex")
    values = [json.loads(line) for line in path.read_text().splitlines()][:-1]
    values.extend(
        [
            {"type": "response_item", "payload": {"id": "u2", "type": "message", "role": "user", "content": "later"}},
            {
                "type": "response_item",
                "payload": {"id": "a2", "type": "message", "role": "assistant", "content": "later answer"},
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {"total_token_usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}},
                },
            },
        ]
    )
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("late-counter-baseline"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    turns = result.report["sessions"][0]["turns"]
    assert live_server.client.get_trace(turns[1]["trace_id"], display=False).info.token_usage is None


def test_opencode_multiple_steps_and_measured_tool_interval(live_server, tmp_path):
    from tests.support.import_fakes import CURRENT_DDL

    path = tmp_path / "opencode.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(CURRENT_DDL)
        connection.execute("INSERT INTO session_v2(id,time_created) VALUES('s',1000)")
        connection.execute(
            "INSERT INTO session_message VALUES('u','s','user',0,1000,1000,?)", (json.dumps({"content": "prompt"}),)
        )
        connection.execute(
            "INSERT INTO session_message VALUES('a','s','assistant',1,1001,1010,?)",
            (
                json.dumps(
                    {
                        "model": "synthetic/model",
                        "finish": "stop",
                        "content": [
                            {
                                "type": "step-finish",
                                "reason": "tool-calls",
                                "tokens": {"input": 10, "output": 1, "reasoning": 2},
                                "cost": 0.1,
                            },
                            {
                                "type": "tool",
                                "tool": "read",
                                "callID": "call",
                                "state": {
                                    "status": "completed",
                                    "input": {},
                                    "output": "result",
                                    "time": {"start": 1002, "end": 1004},
                                },
                            },
                            {
                                "type": "step-finish",
                                "reason": "stop",
                                "tokens": {"input": 20, "output": 2, "reasoning": 3},
                                "cost": 0.2,
                            },
                            {"type": "text", "text": "answer"},
                        ],
                    }
                ),
            ),
        )
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("multi-step"),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    assert trace.info.token_usage == {"input_tokens": 30, "output_tokens": 3, "total_tokens": 38}
    assert trace.info.cost == {"total_cost": 0.3}
    models = [span for span in trace.data.spans if span.span_type == "LLM"]
    assert len(models) == 2
    assert sorted(model.attributes["mlflow.llm.cost"]["total_cost"] for model in models) == [0.1, 0.2]
    tool = next(span for span in trace.data.spans if span.span_type == "TOOL")
    assert (tool.start_time_ns, tool.end_time_ns) == (1002000000, 1004000000)
    assert tool.attributes["mlflow_import.timing"] == "measured"


@pytest.mark.parametrize("harness", ["claude", "codex", "pi", "gemini"])
@pytest.mark.parametrize("redact", [False, True])
def test_all_file_profiles_native_historical_identity_and_parent(live_server, tmp_path, harness, redact):
    path, _ = source_file(tmp_path, harness)
    values = [json.loads(line) for line in path.read_text().splitlines()]
    header = values[0]["payload"] if harness == "codex" else values[0]
    header.update(
        user={"name": "historical-identity"},
        parent_session_id="recorded-parent",
        client_request_id="historical-request",
        git={
            "repository_url": "https://example.test/actual",
            "commit_hash": "commit-literal",
            "branch": "branch-literal",
        },
    )
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("identity-" + harness),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
            *(["--redact"] if redact else []),
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    from mlflow_import.compatibility import decode_ownership

    metadata = decode_ownership(trace.info.tags, trace.info.trace_metadata)
    assert metadata["mlflow.trace.user"] == ("[REDACTED]" if redact else "historical-identity")
    assert metadata["mlflow.source.name"] == harness
    assert metadata["mlflow.source.git.repoURL"] == "https://example.test/actual"
    assert metadata["mlflow.source.git.commit"] == "commit-literal"
    assert metadata["mlflow.source.git.branch"] == "branch-literal"
    assert metadata["parent_session_id"] == "recorded-parent"
    assert trace.info.client_request_id == "historical-request"
    assert metadata["mlflow_import.client_request_id_origin"] == "source"


def test_ready_duplicate_finishes_then_changed_overwrite_uses_canonical_basis(live_server, tmp_path, monkeypatch):
    from mlflow_import.content.accounting import PRICES
    from mlflow_import.errors import ItemImportError
    from mlflow_import.mlflow.attempts import AttemptJournal

    path, _ = source_file(tmp_path, "claude")
    args = [
        "--experiment-name",
        unique_experiment_name("ready-basis-donor"),
        "--source",
        str(path),
        "--tracking-uri",
        live_server.uri,
    ]
    canonical = parse_config(args)
    first = run_import(canonical, build_dependencies(canonical))
    assert exit_status(first) == 0, first.report
    monkeypatch.setitem(PRICES, "openai/gpt-4o", ("100", "200", "50"))
    original = AttemptJournal.publish

    def fail_ready(journal, state, payload):
        result = original(journal, state, payload)
        if state == "ready_to_retire":
            raise ItemImportError("synthetic_ready_interruption", "synthetic interruption")
        return result

    duplicate = parse_config([*args, "--on-conflict", "duplicate"])
    with monkeypatch.context() as patch:
        patch.setattr(AttemptJournal, "publish", fail_ready)
        failed = run_import(duplicate, build_dependencies(duplicate))
        assert exit_status(failed) == 1
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[0]["message"]["content"] = "changed prompt"
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    repaired = run_import(canonical, build_dependencies(canonical))
    assert exit_status(repaired) == 0, repaired.report
    trace = live_server.client.get_trace(repaired.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    assert trace.info.cost["total_cost"] == 0.000375
    assert "mlflow_import.import_instance_id" not in trace.info.trace_metadata


@pytest.mark.parametrize("harness", ["claude", "pi"])
def test_recorded_zero_cost_is_not_missing_or_an_invented_breakdown(live_server, tmp_path, harness):
    path, _ = source_file(tmp_path, harness)
    values = [json.loads(line) for line in path.read_text().splitlines()]
    assistant = values[-1]["message"]
    assistant["usage"] = (
        {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
        if harness == "claude"
        else {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0}
    )
    assistant["cost"] = {"total": 0, "currency": "USD", "method": "estimated"}
    path.write_text("".join(json.dumps(value) + "\n" for value in values))
    config = parse_config(
        [
            "--experiment-name",
            unique_experiment_name("zero-" + harness),
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 0, result.report
    trace = live_server.client.get_trace(result.report["sessions"][0]["turns"][0]["trace_id"], display=False)
    assert trace.info.cost == {"total_cost": 0.0}
    assert trace.info.token_usage == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    model = next(span for span in trace.data.spans if span.span_type == "LLM")
    assert model.attributes["mlflow_import.accounting"]["origin"] == "persisted"
    assert model.attributes["mlflow_import.accounting"]["method"] == "estimated"


def test_fresh_import_refuses_a_map_outside_its_recoverable_profile(live_server, tmp_path, monkeypatch):
    from mlflow_import.content import accounting
    from mlflow_import.mlflow.gateway import MlflowGateway

    path, _ = source_file(tmp_path, "claude")
    monkeypatch.setattr(accounting, "MAX_PRICING_MAP_BYTES", 128)
    trace_creates = []

    def unexpected_create(*args, **kwargs):
        trace_creates.append(True)
        pytest.fail("an oversized map reached trace creation instead of failing preflight")

    monkeypatch.setattr(MlflowGateway, "create_trace", unexpected_create)
    name = unique_experiment_name("map-limit")
    config = parse_config(
        [
            "--experiment-name",
            name,
            "--source",
            str(path),
            "--tracking-uri",
            live_server.uri,
        ]
    )
    result = run_import(config, build_dependencies(config))
    assert exit_status(result) == 1, result.report
    assert result.error is not None
    assert result.error.code == "turn_map_too_large"
    assert result.report["summary"]["imported_turns"] == 0
    assert result.report["experiment"]["id"] is None
    assert live_server.client.get_experiment_by_name(name) is None
    assert trace_creates == []
