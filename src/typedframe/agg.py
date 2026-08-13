from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence
from typing import Any


class Aggregator:
    """Computes a single value from a sequence of rows or field values."""

    def __init__(
        self,
        fn: Callable[[Sequence[Any]], Any],
        *,
        field: str | None = None,
    ) -> None:
        self._fn = fn
        self.field = field

    def __call__(self, rows: Sequence[Any]) -> Any:
        if self.field is None:
            return self._fn(rows)
        values = [getattr(row, self.field) for row in rows]
        return self._fn(values)


def _skip_none(values: Sequence[Any], *, skipna: bool) -> list[Any]:
    if skipna:
        return [value for value in values if value is not None]
    return list(values)


def _numeric_sum(values: Sequence[Any]) -> Any:
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


def count() -> Aggregator:
    """Number of rows in the group."""
    return Aggregator(len)


def sum(field: str, *, skipna: bool = True) -> Aggregator:
    """Sum of ``field``. Empty / all-null groups return ``0`` or ``None``."""

    def _sum(values: Sequence[Any]) -> Any:
        xs = _skip_none(values, skipna=skipna)
        if not xs:
            return None if skipna else 0
        return _numeric_sum(xs)

    return Aggregator(_sum, field=field)


def mean(field: str, *, skipna: bool = True) -> Aggregator:
    """Arithmetic mean of ``field``. Empty / all-null groups return ``None``."""

    def _mean(values: Sequence[Any]) -> Any:
        xs = _skip_none(values, skipna=skipna)
        if not xs:
            return None
        return _numeric_sum(xs) / len(xs)

    return Aggregator(_mean, field=field)


def min(field: str, *, skipna: bool = True) -> Aggregator:
    """Minimum of ``field``. Empty / all-null groups return ``None``."""

    def _min(values: Sequence[Any]) -> Any:
        xs = _skip_none(values, skipna=skipna)
        return builtins.min(xs) if xs else None

    return Aggregator(_min, field=field)


def max(field: str, *, skipna: bool = True) -> Aggregator:
    """Maximum of ``field``. Empty / all-null groups return ``None``."""

    def _max(values: Sequence[Any]) -> Any:
        xs = _skip_none(values, skipna=skipna)
        return builtins.max(xs) if xs else None

    return Aggregator(_max, field=field)


def first(field: str | None = None) -> Aggregator:
    """First row, or the first value of ``field``. Empty groups return ``None``."""
    if field is None:
        return Aggregator(lambda rows: rows[0] if rows else None)
    return Aggregator(lambda values: values[0] if values else None, field=field)


def last(field: str | None = None) -> Aggregator:
    """Last row, or the last value of ``field``. Empty groups return ``None``."""
    if field is None:
        return Aggregator(lambda rows: rows[-1] if rows else None)
    return Aggregator(lambda values: values[-1] if values else None, field=field)


def nunique(field: str) -> Aggregator:
    """Number of distinct values in ``field``."""
    return Aggregator(lambda values: len(set(values)), field=field)
