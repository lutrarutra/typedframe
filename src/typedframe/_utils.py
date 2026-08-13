from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, create_model


def field_names(model: type[BaseModel]) -> frozenset[str]:
    return frozenset(model.model_fields)


def require_fields(model: type[BaseModel], fields: tuple[str, ...]) -> None:
    missing = [name for name in fields if name not in model.model_fields]
    if missing:
        joined = ", ".join(repr(name) for name in missing)
        raise AttributeError(f"{model.__name__} has no field(s): {joined}")


def freeze(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return freeze(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return tuple(sorted((key, freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(freeze(item) for item in value))
    return value


def annotation_for(values: list[Any]) -> Any:
    types = {type(value) for value in values if value is not None}
    has_none = any(value is None for value in values)
    if not types:
        return Any
    if len(types) == 1:
        inferred = next(iter(types))
        return inferred | None if has_none else inferred
    return Any


def dynamic_model(
    name: str,
    fields: Mapping[str, Any],
    *,
    base: type[BaseModel] | None = None,
    config: ConfigDict | None = None,
) -> type[BaseModel]:
    """Build a Pydantic model from field definitions without leaking create_model typing."""
    definitions: dict[str, Any] = dict(fields)
    if base is not None:
        definitions["__base__"] = base
    if config is not None:
        definitions["__config__"] = config
    return cast(Any, create_model)(name, **definitions)
