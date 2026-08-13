from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from typing_extensions import TypeAliasType

_ColumnT = TypeVar("_ColumnT")

UNSET = object()
_UNSET_KEY = "x-typedframe-unset"

Future = TypeAliasType("Future", _ColumnT | None, type_params=(_ColumnT,))
"""Optional column type for values you will assign later.

Use with :func:`deferred` so the field exists on the row model (and is visible
to type checkers) before you populate it:

    class User(BaseModel):
        extra: Future[int] = deferred()
        active: Future[bool] = deferred(False)
"""


def deferred(default: Any = UNSET) -> Any:
    """Default for a :data:`Future` column.

    ``deferred()`` leaves the column unset (hidden from the frame display).
    ``deferred(value)`` fills the column with ``value`` immediately.
    """
    if default is UNSET:
        return Field(default=None, json_schema_extra={_UNSET_KEY: True})
    return Field(default=default)


def is_unset_future(row: BaseModel, name: str, info: FieldInfo | None = None) -> bool:
    """Return True when ``name`` is a future column that has not been assigned."""
    field = info or type(row).model_fields[name]
    extra = field.json_schema_extra
    if not isinstance(extra, dict) or not extra.get(_UNSET_KEY):
        return False
    return name not in row.model_fields_set


def dump_row(row: BaseModel) -> dict[str, Any]:
    """``model_dump()`` without future columns that are still unset."""
    data = row.model_dump()
    for name, info in type(row).model_fields.items():
        if is_unset_future(row, name, info):
            data.pop(name, None)
    return data


def format_row(row: BaseModel) -> str:
    """Repr a row, omitting unset future columns."""
    parts = [
        f"{name}={getattr(row, name)!r}"
        for name, info in type(row).model_fields.items()
        if not is_unset_future(row, name, info)
    ]
    return f"{type(row).__name__}({', '.join(parts)})"
