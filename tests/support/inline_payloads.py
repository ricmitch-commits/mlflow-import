"""Generated canonical-byte boundary values; never read source sessions."""

from mlflow_import.content.canonical import canonical_json
from tests.support.fake_turn import FakeReplayTurn, fake_header, fake_record, fake_summary

QUALIFICATION_SIZES = (4_194_305, 4_196_595, 4_215_023, 8_388_608)


def exact_inline_value(size: int, *, structured: bool = True) -> object:
    """Include Unicode and escaping while fixing the complete canonical size."""

    def wrap(text: str) -> object:
        if not structured:
            return text
        return {"messages": [{"role": "user", "content": text}], "synthetic": True, "count": 7}

    unit = '雪😀"\\\n'
    remaining = size - len(canonical_json(wrap("")))
    unit_bytes = len(canonical_json(unit)) - 2
    value = wrap(unit * (remaining // unit_bytes) + "x" * (remaining % unit_bytes))
    assert len(canonical_json(value)) == size
    return value


def synthetic_tool_turn(value: object, field: str) -> FakeReplayTurn:
    """Project a real tool part without an enclosing larger assistant value."""
    state = {"input": {"query": "synthetic"}, "output": {"done": True}, "status": "completed", field: value}
    records = (
        fake_record("u", decoded={"role": "user", "content": "synthetic prompt"}),
        fake_record(
            "t",
            source_table="part",
            source_type="tool",
            classification=None,
            decoded={"type": "tool", "call_id": "synthetic-call", "state": state},
        ),
    )
    return FakeReplayTurn(session_header=fake_header(), summary=fake_summary(), turn_records=records)
