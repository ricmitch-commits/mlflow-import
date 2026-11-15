"""Put invocation workspaces in a known synthetic-test directory."""

from mlflow_import import resources


def isolate_workspaces(monkeypatch, directory, limit):
    directory.mkdir()
    original = resources.tempfile.mkdtemp

    def create(*args, **kwargs):
        if kwargs.get("prefix") == "mlflow-import-workspace-":
            kwargs["dir"] = directory
        return original(*args, **kwargs)

    monkeypatch.setattr(resources.tempfile, "mkdtemp", create)
    monkeypatch.setenv("MLFLOW_IMPORT_TEMP_BYTES", str(limit))
    return directory
