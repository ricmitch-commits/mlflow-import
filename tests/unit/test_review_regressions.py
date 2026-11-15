"""Reviewer security and safe-boundary reproductions, all synthetic."""

import errno
import sqlite3

import pytest

from mlflow_import.cli import main
from mlflow_import.content.redaction import redact_text
from mlflow_import.json_values import decode_lossless_json

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text",
    [
        '{"user":"SYNTHETIC-FIRST","user":"SYNTHETIC-SECOND"}',
        '{"user_id":"SYNTHETIC-FIRST",',
        r'{"user\u005fid":"SYNTHETIC-FIRST",',
        r'{"p\u0061ssword":"SYNTHETIC-FIRST multiword SYNTHETIC-SECOND',
        r'username="SYNTHETIC-FIRST multiword SYNTHETIC-SECOND',
        r"user_id=SYNTHETIC-FIRST multiword SYNTHETIC-SECOND",
        r'x-p\u0061ssword="SYNTHETIC-FIRST multiword SYNTHETIC-SECOND',
    ],
)
def test_embedded_or_malformed_identity_and_escaped_secrets(text):
    result = redact_text(text).value
    assert "SYNTHETIC-FIRST" not in result
    assert "SYNTHETIC-SECOND" not in result


def test_embedded_duplicate_identity_values_are_losslessly_masked():
    result = redact_text('{"user":"first","user":"second","role":"user"}').value
    decoded = decode_lossless_json(result)
    assert decoded.has_duplicate_keys
    assert [value for name, value in decoded.value.pairs if name == "user"] == ["[REDACTED]", "[REDACTED]"]
    assert decoded.value["role"] == "user"


@pytest.mark.parametrize("boundary", ["mkdtemp", "sqlite"])
def test_cli_store_initialization_failure_is_safe(boundary, monkeypatch, capsys, config, import_dependencies, tmp_path):
    from mlflow_import import resources

    partial_root = tmp_path / "partial-store"
    original_connect = resources.sqlite3.connect
    if boundary == "sqlite":
        partial_root.mkdir()
        original_mkdtemp = resources.tempfile.mkdtemp
        monkeypatch.setattr(
            resources.tempfile,
            "mkdtemp",
            lambda **kwargs: (
                str(partial_root) if kwargs.get("prefix") == "mlflow-import-plan-" else original_mkdtemp(**kwargs)
            ),
        )

    def fail(*args, **kwargs):
        if boundary == "sqlite":
            from pathlib import Path

            if args[0] == ":memory:":
                return original_connect(*args, **kwargs)
            Path(args[0]).write_bytes(b"synthetic partial initialization")
            raise sqlite3.OperationalError("SYNTHETIC-PRIVATE-PATH disk full")
        raise OSError(errno.ENOSPC, "SYNTHETIC-PRIVATE-PATH")

    monkeypatch.setattr(
        resources.tempfile if boundary == "mkdtemp" else resources.sqlite3,
        "mkdtemp" if boundary == "mkdtemp" else "connect",
        fail,
    )
    status = main(["import", "--experiment-name", "synthetic", "--db", str(config.db_path)], import_dependencies)
    assert status == 2
    output = capsys.readouterr()
    assert "temporary_storage_unavailable" in output.err
    assert "SYNTHETIC-PRIVATE-PATH" not in output.err + output.out
    assert "Traceback" not in output.err
    assert not partial_root.exists()
