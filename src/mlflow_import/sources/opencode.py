"""Qualified OpenCode adapter surface.

The schema-specific readers remain in ``mlflow_import.source`` so the public
adapter module can expose one harness-level entry point without duplicating
SQL or turn reconstruction.
"""

from ..source.current import CurrentAdapter
from ..source.legacy import LegacyAdapter
from ..source.registry import AdapterRegistry, default_adapter_registry

NAME = "opencode"

__all__ = ["AdapterRegistry", "CurrentAdapter", "LegacyAdapter", "NAME", "default_adapter_registry"]
