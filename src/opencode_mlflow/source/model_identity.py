"""Provider/model identity extraction shared by selection and mapping."""

from __future__ import annotations

from collections.abc import Mapping


def extract_model_identity(value: object) -> str | None:
    """Build an exact provider/model identity from a decoded source value."""
    if isinstance(value, str):
        identity = value.strip()
        return identity if identity else None
    if isinstance(value, Mapping):
        nested = value.get("model")
        provider = (
            _identity_component(value.get("providerID"))
            or _identity_component(value.get("provider"))
            or _identity_component(value.get("provider_id"))
        )
        if isinstance(nested, Mapping):
            provider = (
                _identity_component(nested.get("providerID"))
                or _identity_component(nested.get("provider"))
                or _identity_component(nested.get("provider_id"))
                or provider
            )
            model = _identity_component(nested.get("modelID")) or _identity_component(nested.get("model"))
        else:
            model = _identity_component(value.get("modelID")) or _identity_component(nested)
        if provider is None and model is not None and "/" in model:
            return model
        if provider is None and model is None:
            return None
        return f"{provider or ''}/{model or ''}"
    return None


def _identity_component(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    component = value.strip()
    return component if component else None
