from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload

from pydantic import BaseModel, ConfigDict, TypeAdapter

from typedframe._utils import annotation_for, dynamic_model, freeze, require_fields
from typedframe.fields import dump_row, format_row, is_unset_future
from typedframe.grouped import AggLike, GroupedTypedFrame

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=BaseModel)


class TypedFrame(Generic[T], Sequence[T]):
    """A sequence of Pydantic models with table-oriented operations.

    Rows are validated when they enter the frame. Iteration and indexing return
    the stored model instances so application code can use normal attributes.
    """

    _model: type[T]
    _unique: frozenset[str]

    def __init__(
        self,
        model: type[T],
        rows: Iterable[T | dict[str, Any]] | None = None,
        *,
        unique: str | Iterable[str] | None = None,
        _trusted: bool = False,
    ) -> None:
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError("model must be a Pydantic BaseModel subclass")
        self._model = cast(type[T], model)
        collected: list[T]
        if rows is None:
            collected = []
        elif _trusted:
            collected = list(cast(Iterable[T], rows))
        else:
            collected = [self._coerce(row) for row in rows]
        self._rows = collected
        self._unique = self._normalize_unique(unique)
        self._check_unique()

    @classmethod
    def empty(
        cls,
        model: type[T],
        *,
        unique: str | Iterable[str] | None = None,
    ) -> TypedFrame[T]:
        return cls(model, [], unique=unique)

    @classmethod
    def from_models(
        cls,
        model: type[T],
        rows: Iterable[T],
        *,
        unique: str | Iterable[str] | None = None,
    ) -> TypedFrame[T]:
        return cls(model, rows, unique=unique)

    @classmethod
    def from_dicts(
        cls,
        model: type[T],
        rows: Iterable[dict[str, Any]],
        *,
        unique: str | Iterable[str] | None = None,
    ) -> TypedFrame[T]:
        return cls(model, rows, unique=unique)

    @classmethod
    def from_json(
        cls,
        model: type[T],
        data: str | bytes,
        *,
        unique: str | Iterable[str] | None = None,
    ) -> TypedFrame[T]:
        adapter: TypeAdapter[list[T]] = TypeAdapter(list[model])  # type: ignore[valid-type]
        return cls(model, adapter.validate_json(data), unique=unique, _trusted=True)

    @classmethod
    def load_json(
        cls,
        model: type[T],
        path: str | Path,
        *,
        unique: str | Iterable[str] | None = None,
    ) -> TypedFrame[T]:
        """Load a JSON array of rows from ``path``."""
        return cls.from_json(model, Path(path).read_text(encoding="utf-8"), unique=unique)

    @classmethod
    def from_pandas(
        cls,
        model: type[T],
        df: pd.DataFrame,
        *,
        unique: str | Iterable[str] | None = None,
    ) -> TypedFrame[T]:
        from typedframe.conversions import from_pandas

        return cls(model, from_pandas(df), unique=unique)

    @classmethod
    def from_polars(
        cls,
        model: type[T],
        df: pl.DataFrame,
        *,
        unique: str | Iterable[str] | None = None,
    ) -> TypedFrame[T]:
        from typedframe.conversions import from_polars

        return cls(model, from_polars(df), unique=unique)

    @classmethod
    def from_csv(
        cls,
        model: type[T],
        source: str | bytes | Path,
        *,
        sep: str = ",",
        header: int | None = 0,
        comment: str | None = "#",
        index_col: int | str | Literal[False] | None = False,
        encoding: str = "utf-8",
        na_values: Sequence[str] | None = None,
        quotechar: str = '"',
        skip_blank_lines: bool = True,
        unique: str | Iterable[str] | None = None,
    ) -> TypedFrame[T]:
        """Load rows from CSV text, bytes, or a file path."""
        from typedframe.csv_io import csv_to_records

        return cls(
            model,
            csv_to_records(
                source,
                model,
                sep=sep,
                header=header,
                comment=comment,
                index_col=index_col,
                encoding=encoding,
                na_values=na_values,
                quotechar=quotechar,
                skip_blank_lines=skip_blank_lines,
            ),
            unique=unique,
        )

    @property
    def model(self) -> type[T]:
        return self._model

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._model.model_fields)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._rows), len(self.columns))

    @property
    def rows(self) -> tuple[T, ...]:
        return tuple(self._rows)

    @property
    def unique_columns(self) -> frozenset[str]:
        return self._unique

    def _coerce(self, row: T | dict[str, Any] | Any) -> T:
        if isinstance(row, self._model):
            return cast(T, row)
        return cast(T, self._model.model_validate(row))

    def _clone(self, rows: list[T]) -> TypedFrame[T]:
        return TypedFrame(self._model, rows, unique=self._unique, _trusted=True)

    def _normalize_unique(self, unique: str | Iterable[str] | None) -> frozenset[str]:
        if unique is None:
            return frozenset()
        fields = (unique,) if isinstance(unique, str) else tuple(unique)
        if fields:
            require_fields(self._model, fields)
        return frozenset(fields)

    def _check_unique(self) -> None:
        for name in self._unique:
            seen: set[Any] = set()
            for index, row in enumerate(self._rows):
                value = getattr(row, name)
                key = freeze(value)
                if key in seen:
                    raise ValueError(f"Column {name!r} must be unique; duplicate value {value!r} at row {index}")
                seen.add(key)

    def require_unique(self, *fields: str) -> TypedFrame[T]:
        """Require that each named column has no duplicate values.

        The constraint is stored on the frame and re-checked when rows or
        those columns change. Returns ``self`` for chaining.
        """
        if not fields:
            raise ValueError("require_unique() requires at least one column")
        require_fields(self._model, fields)
        self._unique = self._unique | frozenset(fields)
        self._check_unique()
        return self

    def __iter__(self) -> Iterator[T]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __bool__(self) -> bool:
        return bool(self._rows)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> TypedFrame[T]: ...

    @overload
    def __getitem__(self, index: str) -> list[Any]: ...

    def __getitem__(self, index: int | slice | str) -> T | TypedFrame[T] | list[Any]:
        if isinstance(index, str):
            return self.column(index)
        if isinstance(index, slice):
            return self._clone(self._rows[index])
        return self._rows[index]

    def __setitem__(
        self,
        key: str,
        value: Sequence[Any] | Callable[[T], Any] | Any,
    ) -> None:
        """Add or replace a column, like ``df["col"] = values``.

        ``value`` may be a sequence of row-aligned values, a callable applied
        to each row, or a scalar broadcast to every row.

        If ``key`` is already a field on the row model (including
        :class:`~typedframe.fields.Future` columns), values are written onto
        that field. Otherwise a subclass is created at runtime so existing
        attributes and ``isinstance`` checks keep working. Declare future
        columns on the model when you want typed attribute access.
        """
        if not isinstance(key, str):
            raise TypeError("columns must be assigned with a string name")
        self._assign_column(key, self._resolve_column_values(value))

    @property
    def loc(self) -> LocIndexer[T]:
        """Pandas-style row/column indexer for quick masked reads and writes."""
        return LocIndexer(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypedFrame):
            return NotImplemented
        return self._model is other._model and self._rows == other._rows

    def __add__(self, other: TypedFrame[T]) -> TypedFrame[T]:
        if not isinstance(other, TypedFrame):
            return NotImplemented
        return concat([self, other])

    def __repr__(self) -> str:
        name = self._model.__name__
        n = len(self._rows)
        if n == 0:
            return f"TypedFrame[{name}](empty)"
        preview = 5
        lines = [f"TypedFrame[{name}]({n} row{'s' if n != 1 else ''})"]
        lines.extend(f"  {format_row(row)}" for row in self._rows[:preview])
        if n > preview:
            lines.append(f"  ... ({n - preview} more)")
        return "\n".join(lines)

    def filter(self, predicate: Callable[[T], bool]) -> TypedFrame[T]:
        return self._clone([row for row in self._rows if predicate(row)])

    def exclude(self, predicate: Callable[[T], bool]) -> TypedFrame[T]:
        return self._clone([row for row in self._rows if not predicate(row)])

    def update_where(
        self,
        predicate: Callable[[T], bool],
        fn: Callable[[T], T],
    ) -> TypedFrame[T]:
        """Return a new frame where matching rows are replaced by ``fn(row)``.

        Both callables receive a typed row. ``fn`` must return a row of the same
        model (typically ``row.model_copy(update=...)``).
        """
        return self._clone([self._coerce(fn(row)) if predicate(row) else row for row in self._rows])

    def apply_where(
        self,
        predicate: Callable[[T], bool],
        fn: Callable[[T], None],
    ) -> TypedFrame[T]:
        """Call ``fn`` on each matching row (in place). Returns ``self``.

        Use this to assign typed attributes: ``fn`` receives the model instance,
        so ``user.age = 99`` is checked by the type checker when the field
        exists on the model.
        """
        for row in self._rows:
            if predicate(row):
                fn(row)
        self._check_unique()
        return self

    def apply(self, fn: Callable[[T], Any]) -> list[Any]:
        """Apply ``fn`` to each row and return a list of results."""
        return [fn(row) for row in self._rows]

    def map(
        self,
        fn: Callable[[T], U | dict[str, Any]],
        *,
        model: type[U] | None = None,
    ) -> TypedFrame[U]:
        mapped = [fn(row) for row in self._rows]
        resolved = model
        if resolved is None:
            if mapped and isinstance(mapped[0], BaseModel):
                resolved = type(mapped[0])
            else:
                raise TypeError("model is required when map() does not return BaseModel instances")
        inherited = [name for name in self._unique if name in resolved.model_fields]
        return TypedFrame(resolved, mapped, unique=inherited)

    def project(self, model: type[U]) -> TypedFrame[U]:
        """Validate each row as ``model``, typically a field subset or reshape."""
        inherited = [name for name in self._unique if name in model.model_fields]
        return TypedFrame(model, [dump_row(row) for row in self._rows], unique=inherited)

    def sort_by(
        self,
        *keys: str | Callable[[T], Any],
        reverse: bool = False,
    ) -> TypedFrame[T]:
        if not keys:
            raise ValueError("sort_by() requires at least one key")
        field_keys = tuple(key for key in keys if isinstance(key, str))
        require_fields(self._model, field_keys)

        def key_fn(row: T) -> Any:
            parts = [getattr(row, key) if isinstance(key, str) else key(row) for key in keys]
            return parts[0] if len(parts) == 1 else tuple(parts)

        return self._clone(sorted(self._rows, key=key_fn, reverse=reverse))

    def reverse(self) -> TypedFrame[T]:
        return self._clone(list(reversed(self._rows)))

    def group_by(
        self,
        *keys: str | Callable[[T], Any],
        names: Sequence[str] | None = None,
    ) -> GroupedTypedFrame[T]:
        if not keys:
            raise ValueError("group_by() requires at least one key")
        if names is not None and len(names) != len(keys):
            raise ValueError("names must have one entry per group key")

        field_keys = tuple(key for key in keys if isinstance(key, str))
        require_fields(self._model, field_keys)

        key_fns: list[Callable[[T], Any]] = []
        key_names: list[str] = []
        for index, key in enumerate(keys):
            if isinstance(key, str):
                key_fns.append(lambda row, field=key: getattr(row, field))
                key_names.append(key)
            else:
                key_fns.append(key)
                default = "key" if len(keys) == 1 else f"key_{index}"
                key_names.append(default)
        if names is not None:
            key_names = list(names)

        groups: dict[Any, list[T]] = {}
        order: list[Any] = []
        for row in self._rows:
            key = key_fns[0](row) if len(key_fns) == 1 else tuple(fn(row) for fn in key_fns)
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError(f"group key {key!r} is unhashable; use hashable field values") from exc
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(row)

        return GroupedTypedFrame(self._model, groups, order, tuple(key_names), unique=self._unique)

    def agg(self, **aggregations: AggLike) -> TypedFrame[Any]:
        """Aggregate the whole frame to a single row."""
        if not aggregations:
            raise ValueError("agg() requires at least one aggregation")
        record = {name: aggregator(self._rows) for name, aggregator in aggregations.items()}
        fields = {name: (annotation_for([value]), ...) for name, value in record.items()}
        model = dynamic_model(
            "AggRow",
            fields,
            config=ConfigDict(arbitrary_types_allowed=True),
        )
        return TypedFrame(model, [record])

    def unique(self, *fields: str) -> TypedFrame[T]:
        """Keep the first row for each distinct value of ``fields`` (or all fields)."""
        require_fields(self._model, fields)
        seen: set[Any] = set()
        result: list[T] = []
        for row in self._rows:
            key = freeze(tuple(getattr(row, field) for field in fields)) if fields else freeze(dump_row(row))
            if key not in seen:
                seen.add(key)
                result.append(row)
        return self._clone(result)

    def column(self, name: str) -> list[Any]:
        require_fields(self._model, (name,))
        return [getattr(row, name) for row in self._rows]

    def map_columns(
        self,
        src: TypedFrame[Any],
        idx_columns: str | Sequence[str] | None,
        col: str,
    ) -> list[Any]:
        """Look up ``src[col]`` by key and return values in this frame's row order.

        ``idx_columns`` is the join key on both frames (one name, several names,
        or ``None`` to align by row position). Missing keys yield ``None``.
        Duplicate keys in ``src`` keep the last value.
        """
        require_fields(src.model, (col,))
        if idx_columns is None:
            values = src.column(col)
            return [values[index] if index < len(values) else None for index in range(len(self._rows))]

        names = (idx_columns,) if isinstance(idx_columns, str) else tuple(idx_columns)
        if not names:
            raise ValueError("idx_columns must not be empty")
        require_fields(self._model, names)
        require_fields(src.model, names)

        mapping: dict[Any, Any] = {}
        for row in src:
            mapping[_row_key(row, names)] = getattr(row, col)

        if isinstance(idx_columns, str):
            result: list[Any] = []
            for row in self._rows:
                key = getattr(row, idx_columns)
                result.append(None if key is None else mapping.get(freeze(key)))
            return result

        return [mapping.get(_row_key(row, names)) for row in self._rows]

    def _resolve_column_values(
        self,
        value: Sequence[Any] | Callable[[T], Any] | Any,
    ) -> list[Any]:
        if callable(value) and not isinstance(value, type):
            return [value(row) for row in self._rows]
        if isinstance(value, (str, bytes, bytearray)):
            return [value] * len(self._rows)
        if isinstance(value, Sequence):
            if len(value) != len(self._rows):
                raise ValueError(f"Length of values ({len(value)}) does not match number of rows ({len(self._rows)})")
            return list(value)
        return [value] * len(self._rows)

    def _resolve_mask(self, predicate: Callable[[T], bool] | Sequence[bool]) -> list[bool]:
        if callable(predicate) and not isinstance(predicate, type):
            return [bool(predicate(row)) for row in self._rows]
        if isinstance(predicate, Sequence) and not isinstance(predicate, (str, bytes, bytearray)):
            mask = [bool(flag) for flag in predicate]
            if len(mask) != len(self._rows):
                raise ValueError(f"Length of mask ({len(mask)}) does not match number of rows ({len(self._rows)})")
            return mask
        raise TypeError("loc predicate must be a callable or a sequence of bools")

    def _set_where(
        self,
        predicate: Callable[[T], bool] | Sequence[bool],
        column: str,
        value: Sequence[Any] | Callable[[T], Any] | Any,
    ) -> None:
        if not isinstance(column, str):
            raise TypeError("column must be a string")
        mask = self._resolve_mask(predicate)
        if not any(mask):
            return
        matched = [row for row, flag in zip(self._rows, mask, strict=True) if flag]
        assigned = self._resolve_where_values(value, matched)
        name_exists = column in self._model.model_fields
        if not name_exists:
            annotation = annotation_for(assigned)
            self._model = cast(
                type[T],
                dynamic_model(
                    self._model.__name__,
                    {column: (annotation | None, None)},
                    base=self._model,
                ),
            )

        assigned_iter = iter(assigned)
        rows: list[T] = []
        for row, flag in zip(self._rows, mask, strict=True):
            data = dump_row(row)
            if flag:
                data[column] = next(assigned_iter)
            elif name_exists and not is_unset_future(row, column):
                data[column] = getattr(row, column)
            rows.append(cast(T, self._model.model_validate(data)))
        self._rows = rows
        self._check_unique()

    def _resolve_where_values(
        self,
        value: Sequence[Any] | Callable[[T], Any] | Any,
        matched: list[T],
    ) -> list[Any]:
        if callable(value) and not isinstance(value, type):
            return [value(row) for row in matched]
        if isinstance(value, (str, bytes, bytearray)):
            return [value] * len(matched)
        if isinstance(value, Sequence):
            if len(value) != len(matched):
                raise ValueError(
                    f"Length of values ({len(value)}) does not match number of selected rows ({len(matched)})"
                )
            return list(value)
        return [value] * len(matched)

    def _assign_column(self, name: str, values: list[Any]) -> None:
        if name not in self._model.model_fields:
            annotation = annotation_for(values)
            self._model = cast(
                type[T],
                dynamic_model(
                    self._model.__name__,
                    {name: (annotation, ...)},
                    base=self._model,
                ),
            )
        self._rows = [
            cast(T, self._model.model_validate({**dump_row(row), name: value}))
            for row, value in zip(self._rows, values, strict=True)
        ]
        self._check_unique()

    def head(self, n: int = 5) -> TypedFrame[T]:
        if n < 0:
            raise ValueError("n must be non-negative")
        return self._clone(self._rows[:n])

    def tail(self, n: int = 5) -> TypedFrame[T]:
        if n < 0:
            raise ValueError("n must be non-negative")
        return self._clone(self._rows[-n:] if n else [])

    def append(self, row: T | dict[str, Any]) -> TypedFrame[T]:
        return self._clone([*self._rows, self._coerce(row)])

    def extend(self, rows: Iterable[T | dict[str, Any]]) -> TypedFrame[T]:
        extra = [self._coerce(row) for row in rows]
        return self._clone([*self._rows, *extra])

    def to_list(self) -> list[T]:
        return list(self._rows)

    def to_dicts(self, *, mode: Literal["python", "json"] = "python") -> list[dict[str, Any]]:
        return [dump_row(row, mode=mode) for row in self._rows]

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize rows to a JSON array. Unset future columns are omitted."""
        return json.dumps(self.to_dicts(mode="json"), indent=indent)

    def dump_json(self, path: str | Path, *, indent: int | None = None) -> None:
        """Write a JSON array of rows to ``path``."""
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    def to_csv(
        self,
        path: str | Path | None = None,
        *,
        sep: str = ",",
        index: bool = False,
        index_label: str = "",
        header: bool = True,
        na_rep: str = "",
        encoding: str = "utf-8",
        lineterminator: str = "\n",
        quotechar: str = '"',
        columns: Sequence[str] | None = None,
    ) -> str:
        """Serialize rows to CSV. If ``path`` is given, also write the file."""
        from typedframe.csv_io import frame_to_csv

        fieldnames = list(columns) if columns is not None else list(self.columns)
        if columns is not None:
            require_fields(self._model, tuple(columns))
        text = frame_to_csv(
            self.to_dicts(mode="json"),
            fieldnames,
            sep=sep,
            index=index,
            index_label=index_label,
            header=header,
            na_rep=na_rep,
            lineterminator=lineterminator,
            quotechar=quotechar,
        )
        if path is not None:
            Path(path).write_text(text, encoding=encoding)
        return text

    def to_pandas(self) -> pd.DataFrame:
        from typedframe.conversions import to_pandas

        return to_pandas(self.to_dicts())

    def to_polars(self) -> pl.DataFrame:
        from typedframe.conversions import to_polars

        return to_polars(self.to_dicts())


class LocIndexer(Generic[T]):
    """Boolean-row indexer: ``frame.loc[predicate]`` or ``frame.loc[predicate, column]``."""

    def __init__(self, frame: TypedFrame[T]) -> None:
        self._frame = frame

    @overload
    def __getitem__(self, key: Callable[[T], bool] | Sequence[bool]) -> TypedFrame[T]: ...

    @overload
    def __getitem__(self, key: tuple[Callable[[T], bool] | Sequence[bool], str]) -> list[Any]: ...

    def __getitem__(
        self,
        key: Callable[[T], bool] | Sequence[bool] | tuple[Callable[[T], bool] | Sequence[bool], str],
    ) -> TypedFrame[T] | list[Any]:
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], str):
            predicate = cast(Callable[[T], bool] | Sequence[bool], key[0])
            column = key[1]
            mask = self._frame._resolve_mask(predicate)
            require_fields(self._frame.model, (column,))
            return [getattr(row, column) for row, flag in zip(self._frame, mask, strict=True) if flag]
        if isinstance(key, tuple):
            raise TypeError("loc getitem requires loc[predicate] or loc[predicate, column]")
        mask = self._frame._resolve_mask(key)
        return self._frame._clone([row for row, flag in zip(self._frame, mask, strict=True) if flag])

    def __setitem__(
        self,
        key: tuple[Callable[[T], bool] | Sequence[bool], str],
        value: Sequence[Any] | Callable[[T], Any] | Any,
    ) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError('loc assignment requires frame.loc[predicate, "column"] = value')
        predicate, column = key
        self._frame._set_where(predicate, column, value)


def _row_key(row: BaseModel, names: tuple[str, ...]) -> Any:
    if len(names) == 1:
        return freeze(getattr(row, names[0]))
    return freeze(tuple(getattr(row, name) for name in names))


def concat(frames: Iterable[TypedFrame[T]]) -> TypedFrame[T]:
    """Concatenate frames that share the same row model."""
    collected = list(frames)
    if not collected:
        raise ValueError("concat() requires at least one frame")
    model = collected[0].model
    unique: set[str] = set()
    rows: list[T] = []
    for frame in collected:
        if frame.model is not model:
            raise TypeError(f"Cannot concat TypedFrame[{model.__name__}] with TypedFrame[{frame.model.__name__}]")
        unique |= set(frame.unique_columns)
        rows.extend(frame)
    return TypedFrame(model, rows, unique=unique, _trusted=True)
