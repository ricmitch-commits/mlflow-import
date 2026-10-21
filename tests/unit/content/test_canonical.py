"""Unit tests for canonical-json-v1 serialization and its golden vectors.

``tests/fixtures/payloads/canonical-json-v1-vectors.json`` is the
reviewable contract table: each vector's ``input`` is materialized to a
Python value (JSON numbers parse to int/float exactly as written;
``$$``-prefixed marker objects stand for the values JSON cannot express)
and its ``expected`` is the exact canonical output text, compared as
UTF-8 bytes. The vectors pin key ordering, Unicode preservation and
escaping, separators, RFC 8785 number formatting, the typed wrappers,
null-versus-absent, duplicate keys, nesting, and empty containers.

The Task 8 artifact-integration class reuses the same vector table — no
second canonical format or vector set — to pin that streamed turn
artifacts embed the exact canonical-json-v1 bytes for the scalar and
wrapper cases.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import pytest

from opencode_mlflow.content.canonical import (
    CANONICAL_JSON_VERSION,
    DuplicateKeyObject,
    canonical_json,
    canonical_record_envelope,
    canonical_stream,
)
from opencode_mlflow.contracts import (
    CanonicalRecordEnvelope,
    RedactionMode,
    Redactor,
)
from opencode_mlflow.contracts import (
    DuplicateKeyObject as SharedDuplicateKeyObject,
)
from opencode_mlflow.source.snapshot import PROTECTED_IDENTITY_FIELDS
from tests.support.fake_redactor import IdentityRedactor, RecordingRedactor, SecretMaskingRedactor

pytestmark = pytest.mark.unit

VECTORS_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "payloads" / "canonical-json-v1-vectors.json"


def _load_vectors() -> list[dict[str, Any]]:
    document = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    vectors = document.get("vectors") if isinstance(document, dict) else None
    if not isinstance(vectors, list) or not vectors:
        raise AssertionError("the canonical-json-v1 vector table is missing or empty")
    return vectors


def _materialize(value: Any) -> Any:
    """Turn a fixture vector input into the Python value it stands for."""
    if isinstance(value, dict):
        if set(value) == {"$$nonfinite"}:
            return float(value["$$nonfinite"])
        if set(value) == {"$$bytes_hex"}:
            return bytes.fromhex(value["$$bytes_hex"])
        if set(value) == {"$$duplicate_keys", "$$raw_text"}:
            return DuplicateKeyObject(
                pairs=tuple((key, _materialize(item)) for key, item in value["$$duplicate_keys"]),
                raw_text=value["$$raw_text"],
            )
        return {key: _materialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item) for item in value]
    return value


VECTORS = _load_vectors()


@pytest.mark.parametrize("vector", VECTORS, ids=lambda vector: str(vector["name"]))
def test_canonical_json_matches_golden_vector(vector: dict[str, Any]) -> None:
    assert canonical_json(_materialize(vector["input"])) == vector["expected"].encode("utf-8")


class TestContractSurface:
    def test_version_tag_is_canonical_json_v1(self) -> None:
        assert CANONICAL_JSON_VERSION == "canonical-json-v1"

    def test_reexports_the_shared_contract_types(self) -> None:
        from opencode_mlflow.content import canonical

        assert canonical.CanonicalRecordEnvelope is CanonicalRecordEnvelope
        assert canonical.DuplicateKeyObject is SharedDuplicateKeyObject
        assert canonical.RedactionMode is RedactionMode
        assert canonical.Redactor is Redactor

    def test_returns_utf8_bytes(self) -> None:
        assert canonical_json("é") == '"é"'.encode()


class TestScalarBehavior:
    def test_tuples_serialize_as_arrays(self) -> None:
        assert canonical_json((1, "two")) == b'[1,"two"]'

    def test_bytearray_and_memoryview_wrap_like_bytes(self) -> None:
        assert canonical_json(bytearray(b"\x00")) == b'{"$bytes":"AA=="}'
        assert canonical_json(memoryview(b"\x00")) == b'{"$bytes":"AA=="}'

    def test_rejects_unsupported_value_types(self) -> None:
        with pytest.raises(TypeError):
            canonical_json(object())

    def test_rejects_non_string_mapping_keys(self) -> None:
        with pytest.raises(TypeError):
            canonical_json({1: "one"})

    def test_rejects_unsupported_values_nested_in_structures(self) -> None:
        with pytest.raises(TypeError):
            canonical_json({"fine": 1, "bad": {object()}})


class TestDuplicateKeyObject:
    def test_retains_every_occurrence_in_order_with_raw_text(self) -> None:
        duplicate = DuplicateKeyObject(
            pairs=(("text", "first"), ("text", "second"), ("other", 1)),
            raw_text='{"text": "first", "text": "second", "other": 1}',
        )

        assert canonical_json(duplicate) == (
            b'{"$duplicate_keys":[["text","first"],["text","second"],["other",1]],'
            b'"$raw_text":"{\\"text\\": \\"first\\", \\"text\\": \\"second\\", \\"other\\": 1}"}'
        )


class TestCanonicalRecordEnvelope:
    def test_builds_the_shared_envelope_with_empty_warnings(self) -> None:
        envelope = canonical_record_envelope({"role": "user"}, '{"role": "user"}', {"id": "m1"})

        assert isinstance(envelope, CanonicalRecordEnvelope)
        assert envelope.decoded == {"role": "user"}
        assert envelope.raw_text == '{"role": "user"}'
        assert dict(envelope.raw_columns) == {"id": "m1"}
        assert envelope.warnings == ()


class TestCanonicalStream:
    def test_redacts_each_chunk_before_serialization(self) -> None:
        chunks = canonical_stream([{"secret": "alpha-sentinel", "text": "hi"}], SecretMaskingRedactor())

        assert list(chunks) == [canonical_json({"secret": "[REDACTED]", "text": "hi"})]

    def test_passes_the_fixed_identity_protection_set_to_the_redactor(self) -> None:
        recorder = RecordingRedactor()

        assert list(canonical_stream([{"id": "m1"}], recorder)) == [canonical_json({"id": "m1"})]
        assert recorder.calls == [({"id": "m1"}, frozenset(PROTECTED_IDENTITY_FIELDS))]

    def test_never_materializes_the_stream(self) -> None:
        def endless() -> Iterator[object]:
            while True:
                yield {"seq": 1}

        first_two = list(islice(canonical_stream(endless(), IdentityRedactor()), 2))

        assert first_two == [canonical_json({"seq": 1}), canonical_json({"seq": 1})]

    def test_yields_fresh_deterministic_bytes_per_pass(self) -> None:
        chunks: list[object] = [{"a": 1}, {"b": [1e21]}]
        first = list(canonical_stream(chunks, IdentityRedactor()))
        second = list(canonical_stream(chunks, IdentityRedactor()))

        assert first == second == [b'{"a":1}', b'{"b":[1e+21]}']


class TestArtifactSerialization:
    """Task 8 integration: artifacts embed the same canonical-json-v1 bytes.

    The streamed artifact writer serializes record values through the
    same :func:`canonical_json` Task 4A pinned with the golden vectors,
    so a record carrying golden-vector values embeds those vectors'
    exact expected bytes inside the artifact document.
    """

    _VECTOR_NAMES = (
        "integer-arbitrary-precision",
        "float-integral-large",
        "float-scientific-threshold-1e21",
        "nonfinite-nan",
        "string-escapes",
        "string-unicode-preserved",
        "string-lone-surrogate-escaped",
        "bytes-blob",
        "duplicate-keys-ordered-pairs-with-raw-text",
        "object-key-order-utf8-bytes-not-utf16-units",
    )

    def test_artifact_records_embed_golden_vector_bytes(self) -> None:
        from opencode_mlflow.content.artifacts import stream_turn_artifact
        from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_record, fake_summary

        vectors = {vector["name"]: vector for vector in VECTORS}
        record = fake_record(
            "msg-vector",
            raw_json=None,
            decoded={
                "integral": _materialize(vectors["integer-arbitrary-precision"]["input"]),
                "large": _materialize(vectors["float-integral-large"]["input"]),
                "scientific": _materialize(vectors["float-scientific-threshold-1e21"]["input"]),
                "nan": _materialize(vectors["nonfinite-nan"]["input"]),
                "escaped": _materialize(vectors["string-escapes"]["input"]),
                "unicode": _materialize(vectors["string-unicode-preserved"]["input"]),
                "surrogate": _materialize(vectors["string-lone-surrogate-escaped"]["input"]),
                "blob": _materialize(vectors["bytes-blob"]["input"]),
                "duplicate": _materialize(vectors["duplicate-keys-ordered-pairs-with-raw-text"]["input"]),
                "order": _materialize(vectors["object-key-order-utf8-bytes-not-utf16-units"]["input"]),
            },
        )
        turn = FakeReplayTurn(
            session_header=fake_header(),
            summary=fake_summary(),
            turn_records=(record,),
        )

        buffer = io.BytesIO()
        stream_turn_artifact(turn, buffer, {"msg-vector": "tr_1"}, RedactionMode.VERBATIM)
        artifact_bytes = buffer.getvalue()

        for name in self._VECTOR_NAMES:
            assert vectors[name]["expected"].encode("utf-8") in artifact_bytes

    def test_artifact_document_parses_as_valid_json(self) -> None:
        from opencode_mlflow.content.artifacts import stream_turn_artifact
        from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_record, fake_summary

        record = fake_record(
            "msg-vector",
            raw_json=None,
            decoded={"nan": _materialize({"$$nonfinite": "NaN"}), "blob": _materialize({"$$bytes_hex": "ff"})},
        )
        turn = FakeReplayTurn(
            session_header=fake_header(),
            summary=fake_summary(),
            turn_records=(record,),
        )

        buffer = io.BytesIO()
        stream_turn_artifact(turn, buffer, {}, RedactionMode.VERBATIM)

        # Even with typed wrappers inside, the whole document is one valid
        # canonical-json-v1 JSON document (wrappers are self-describing JSON).
        document = json.loads(buffer.getvalue())
        assert document["turn_records"][0]["decoded"]["nan"] == {"$nonfinite": "NaN"}
        assert document["turn_records"][0]["decoded"]["blob"] == {"$bytes": "/w=="}
