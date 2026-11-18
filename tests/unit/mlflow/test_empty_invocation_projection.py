"""Contentless assistant envelopes are excluded from the bound accounting.

An assistant message can exist with a model identity and no text/tool/reasoning
parts (aborted, auth failure, invalid request). The plugin-shaped projection
emits no span for it, so the trace root and turn-map accounting must not claim
its invocation; otherwise the reader's input-context binding rejects the whole
session. The canonical artifacts retain the full observation.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from mlflow_import import __version__ as IMPORTER_VERSION
from mlflow_import.compatibility import request_ownership
from mlflow_import.content.input_context import validate_input_contexts
from mlflow_import.content.turn_map import _require_context_bindings, build_turn_map
from mlflow_import.contracts import RedactionMode
from mlflow_import.errors import ItemImportError
from mlflow_import.mlflow.mapping import MappingContext, map_turn_to_trace
from mlflow_import.source.catalog import CURRENT_CATALOG_V1
from mlflow_import.source.current import CurrentAdapter
from mlflow_import.source.discovery import ARTIFACT_SCHEMA_VERSION, MAPPING_VERSION
from mlflow_import.source.replay import NormalizedSnapshotAdapter
from mlflow_import.source.snapshot import create_normalized_snapshot
from mlflow_import.source.turns import build_replay_turn, iter_turn_summaries
from tests.fixtures.six_turn_session import _CURRENT_DDL
from tests.support.fake_source import identity_redaction

pytestmark = pytest.mark.unit

SESSION_ID = "ses-empty-invocation"
ASSISTANT_ID = "msg-empty-assistant"
TRACE_ID = "t" * 35
ROOT_SPAN_ID = "0" * 16


def _build_source(tmp_path):
    source = tmp_path / "empty-invocation.db"
    connection = sqlite3.connect(source)
    try:
        connection.executescript(_CURRENT_DDL)
        connection.execute(
            "INSERT INTO session_v2 (id, project_id, directory, title, version, time_created, time_updated)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (SESSION_ID, "proj-empty", "/repo/empty", "Empty invocation", "1.0", 1_772_000_000_000, 1_772_000_010_000),
        )
        rows = (
            ("msg-empty-user", "user", 1, {"content": [{"type": "text", "text": "synthetic prompt"}]}),
            (
                ASSISTANT_ID,
                "assistant",
                2,
                {
                    "content": [],
                    "model": {"providerID": "synthetic-provider", "id": "synthetic-model"},
                    "finish": "error",
                    "error": {"type": "provider.invalid-request", "message": "synthetic failure"},
                },
            ),
        )
        for message_id, message_type, seq, payload in rows:
            created = 1_772_000_000_000 + seq * 1_000
            connection.execute(
                "INSERT INTO session_message VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, SESSION_ID, message_type, seq, created, created, json.dumps(payload)),
            )
        connection.commit()
    finally:
        connection.close()
    return source


@contextmanager
def _mapped(tmp_path):
    source = _build_source(tmp_path)
    connection = sqlite3.connect(source)
    try:
        exporter = CurrentAdapter(connection, CURRENT_CATALOG_V1)
    finally:
        connection.close()
    with create_normalized_snapshot(source, tmp_path, exporter, identity_redaction) as snapshot:
        adapter = NormalizedSnapshotAdapter(snapshot)
        header = next(adapter.iter_session_headers())
        records = list(adapter.iter_records(SESSION_ID))
        (summary,) = list(iter_turn_summaries(records))
        turn = build_replay_turn(snapshot, header, summary)
        context = MappingContext(
            importer_version=IMPORTER_VERSION,
            mapping_version=MAPPING_VERSION,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            source_schema=header.adapter_generation,
            redaction_mode=RedactionMode.VERBATIM,
            experiment_id="exp-empty-invocation",
            catalog_version=snapshot.schema_description.catalog_version,
            source_database_fingerprint=snapshot.fingerprint,
            inline_limit_bytes=None,
            logical_turn_count=1,
            source_message_count=sum(1 for record in records if record.derived_child_ordinal is None),
            source_part_count=sum(1 for record in records if record.derived_child_ordinal is not None),
        )
        yield turn, map_turn_to_trace(turn, context)


def _synthetic_spans(children, span_ids):
    spans = [SimpleNamespace(span_id=ROOT_SPAN_ID, trace_id=TRACE_ID, parent_id=None, attributes={})]
    for child in children:
        attributes = dict(child.attributes)
        attributes.setdefault("mlflow_import.source_key", child.source_key)
        spans.append(
            SimpleNamespace(
                span_id=span_ids[child.source_key],
                trace_id=TRACE_ID,
                parent_id=ROOT_SPAN_ID,
                attributes=attributes,
            )
        )
    return spans


def test_contentless_assistant_invocation_is_excluded_from_bound_accounting(tmp_path):
    with _mapped(tmp_path) as (turn, request):
        children = list(request.child_specs())
        assert all(child.source_key != f"session_message:{ASSISTANT_ID}" for child in children)
        accounting = request.root_attributes["mlflow_import.accounting"]
        assert accounting["invocations"] == {}

        span_ids = {child.source_key: f"{index + 1:016x}" for index, child in enumerate(children)}
        value = build_turn_map(
            turn,
            children=children,
            accounting=request.root_attributes["mlflow_import.accounting"],
            ownership=request_ownership(request),
            trace_id=TRACE_ID,
            root_span_id=ROOT_SPAN_ID,
            span_ids_by_source=span_ids,
            mode=RedactionMode.VERBATIM,
        )
        mapped = {
            **value,
            "records": list(value["records"].values),
            "input_contexts": list(value["input_contexts"].values),
        }
        assert mapped["input_contexts"] == []
        assert validate_input_contexts(mapped, request_ownership(request), spans=_synthetic_spans(children, span_ids))


def test_unrepresented_invocation_is_rejected_before_any_create():
    accounting = {"invocations": {":msg-empty-assistant": {"usage": {}, "cost": {}}}}
    with pytest.raises(ItemImportError) as caught:
        _require_context_bindings(accounting, set())
    assert caught.value.code == "turn_map_binding_failed"


def test_aggregate_child_invocation_may_omit_its_span():
    accounting = {
        "invocations": {
            ":parent": {"aggregate_children": [":step"]},
            ":step": {"aggregate_children": []},
        }
    }
    _require_context_bindings(accounting, {":parent"})
