from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from typedframe._utils import annotation_for, dynamic_model
from typedframe.agg import Aggregator
from typedframe.agg import count as count_agg

if TYPE_CHECKING:
    from typedframe.frame import TypedFrame

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=BaseModel)

AggLike = Aggregator | Callable[[Sequence[Any]], Any]


class GroupedTypedFrame(Generic[T]):
    """Rows of a :class:`TypedFrame` partitioned by one or more keys."""

    def __init__(
        self,
        model: type[T],
        groups: Mapping[Any, Sequence[T]],
        order: list[Any],
        key_names: tuple[str, ...],
        unique: frozenset[str] = frozenset(),
    ) -> None:
        self._model = model
        self._groups = {key: list(rows) for key, rows in groups.items()}
        self._order = order
        self._key_names = key_names
        self._unique = unique

    @property
    def model(self) -> type[T]:
        return self._model

    @property
    def key_names(self) -> tuple[str, ...]:
        return self._key_names

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self) -> Iterator[tuple[Any, TypedFrame[T]]]:
        for key in self._order:
            yield key, self._frame_for(key)

    def __getitem__(self, key: Any) -> TypedFrame[T]:
        if key not in self._groups:
            raise KeyError(key)
        return self._frame_for(key)

    def __contains__(self, key: object) -> bool:
        return key in self._groups

    def __repr__(self) -> str:
        n = len(self._order)
        keys = ", ".join(self._key_names)
        return f"GroupedTypedFrame[{self._model.__name__}]({n} groups by {keys})"

    def keys(self) -> list[Any]:
        return list(self._order)

    def to_dict(self) -> dict[Any, TypedFrame[T]]:
        return {key: self._frame_for(key) for key in self._order}

    def count(self) -> TypedFrame[Any]:
        """Return one row per group with a ``count`` column."""
        return self.agg(count=count_agg())

    def agg(self, **aggregations: AggLike) -> TypedFrame[Any]:
        """Reduce each group to a single row of key columns plus aggregations.

        Values may be :class:`~typedframe.agg.Aggregator` instances or callables
        that receive the group's rows.
        """
        if not aggregations:
            raise ValueError("agg() requires at least one aggregation")

        overlap = set(aggregations) & set(self._key_names)
        if overlap:
            names = ", ".join(sorted(repr(name) for name in overlap))
            raise ValueError(f"aggregation name(s) collide with group key(s): {names}")

        records: list[dict[str, Any]] = []
        for key in self._order:
            rows = self._groups[key]
            record = self._key_as_dict(key)
            for name, aggregator in aggregations.items():
                record[name] = aggregator(rows)
            records.append(record)

        model = self._agg_model(records, tuple(aggregations))
        from typedframe.frame import TypedFrame

        return TypedFrame(model, records)

    def map_groups(
        self,
        fn: Callable[[TypedFrame[T]], TypedFrame[U]],
        *,
        model: type[U] | None = None,
    ) -> TypedFrame[U]:
        """Apply ``fn`` to each group and concatenate the resulting frames."""
        from typedframe.frame import TypedFrame, concat

        frames = [fn(self._frame_for(key)) for key in self._order]
        if not frames:
            if model is None:
                raise ValueError("map_groups() on an empty grouping requires model=")
            return TypedFrame(model, [])
        return concat(frames)

    def _frame_for(self, key: Any) -> TypedFrame[T]:
        from typedframe.frame import TypedFrame

        return TypedFrame(self._model, self._groups[key], unique=self._unique, _trusted=True)

    def _key_as_dict(self, key: Any) -> dict[str, Any]:
        if len(self._key_names) == 1:
            return {self._key_names[0]: key}
        return dict(zip(self._key_names, key, strict=True))

    def _agg_model(
        self,
        records: list[dict[str, Any]],
        agg_names: tuple[str, ...],
    ) -> type[BaseModel]:
        field_names = (*self._key_names, *agg_names)
        if not records:
            fields = {name: (Any, ...) for name in field_names}
        else:
            fields = {name: (annotation_for([record[name] for record in records]), ...) for name in field_names}
        return dynamic_model(
            "AggRow",
            fields,
            config=ConfigDict(arbitrary_types_allowed=True),
        )
