"""Public projection seam for canonical values, turns, and trace mapping."""

import importlib

from .content.canonical import *  # noqa: F403
from .content.fingerprint import *  # noqa: F403
from .source.selection import *  # noqa: F403
from .source.turns import *  # noqa: F403


def __getattr__(name: str) -> object:
    return getattr(importlib.import_module("mlflow_import.mlflow.mapping"), name)
