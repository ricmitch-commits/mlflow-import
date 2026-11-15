import json

import pytest

from mlflow_import.content.redaction import DeterministicRedactor
from mlflow_import.contracts import RedactionMode

pytestmark = pytest.mark.unit


def test_nested_user_identity_and_escaped_copies_are_masked_but_roles_survive():
    raw = {
        "user": {"name": "Synthetic Person", "email": "fixture@example.test"},
        "role": "user",
        "mlflow.trace.user": "synthetic-user",
    }
    redactor = DeterministicRedactor(RedactionMode.REDACTED)
    result = redactor.redact({"metadata": raw, "escaped": json.dumps(json.dumps(raw))}, ())
    encoded = json.dumps(result)
    assert "Synthetic Person" not in encoded
    assert "fixture@example.test" not in encoded
    assert "synthetic-user" not in encoded
    assert result["metadata"]["role"] == "user"


def test_unterminated_credential_assignment_masks_the_whole_remainder():
    result = DeterministicRedactor(RedactionMode.REDACTED).redact('password="several secret words', ())
    assert "secret" not in result
    assert "words" not in result
