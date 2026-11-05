"""Public MLflow boundary seam.

All MLflow SDK access remains implemented in ``mlflow_import.mlflow.gateway``;
this module gives callers one stable target import without introducing a second
client abstraction.
"""

import importlib


def __getattr__(name: str) -> object:
    return getattr(importlib.import_module("mlflow_import.mlflow.gateway"), name)
