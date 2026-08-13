# TypedFrame

TypedFrame is a lightweight, type-friendly table abstraction for Python applications.

It stores rows as Pydantic models and provides convenient operations for iteration, filtering, grouping, aggregation, sorting, and conversion to analytical libraries when needed.

It is designed for datasets of up to a few thousand rows where:

- Strong row-level typing is more important than maximum DataFrame performance
- Application/business logic should use normal Python attributes
- Data must be validated at boundaries
- Iteration should return typed model instances

Rows entering the frame are validated by Pydantic. Rows already stored as the target model are kept as-is. Most frame operations return new frames. Column assignment (`frame["col"] = ...`) updates the frame in place.

## Install

```bash
uv add typedframe
```

Optional extras for DataFrame conversion:

```bash
uv add 'typedframe[pandas]'
uv add 'typedframe[polars]'
uv add 'typedframe[all]'
```

## Define a row model

```python
from pydantic import BaseModel

from typedframe import Future, TypedFrame, agg, concat, deferred


class User(BaseModel):
    id: int
    name: str
    age: int
    city: str
    score: float | None = None
    adult: Future[bool] = deferred()
    bonus: Future[int] = deferred(0)
```

`Future[T] = deferred()` declares a column you will fill later. Type checkers accept `user.adult`, and unset values are omitted from the frame display. `deferred(value)` fills every row with that default immediately.

## Construct a frame

```python
users = TypedFrame(
    User,
    [
        {"id": 1, "name": "Alice", "age": 30, "city": "NYC", "score": 9.0},
        User(id=2, name="Bob", age=25, city="LA", score=7.5),
        {"id": 3, "name": "Carol", "age": 30, "city": "NYC", "score": 8.0},
        {"id": 4, "name": "Dan", "age": 40, "city": "LA"},
    ],
    unique="id",
)

empty = TypedFrame.empty(User, unique="id")
from_dicts = TypedFrame.from_dicts(User, [{"id": 5, "name": "Eve", "age": 22, "city": "Austin"}])
from_models = TypedFrame.from_models(User, [User(id=6, name="Fay", age=29, city="Austin")])
```

Invalid rows raise a Pydantic `ValidationError`. `unique="id"` (or `unique=["id", "email"]`) requires distinct values and is re-checked when rows or those columns change.

```python
users.require_unique("id")
```

## Iterate, index, and inspect

```python
for user in users:
    print(user.name, user.age)

users[0]          # User
users[1:3]        # TypedFrame[User]
len(users)        # 4
users.shape       # (4, 7)
users.columns     # ("id", "name", "age", "city", "score", "adult", "bonus")
users.model       # User
```

## Filter, sort, and slice

```python
adults = users.filter(lambda user: user.age >= 18)
not_nyc = users.exclude(lambda user: user.city == "NYC")

by_age = users.sort_by("age")
by_city_age = users.sort_by("city", "age", reverse=True)
by_name_len = users.sort_by(lambda user: len(user.name))
reversed_users = users.reverse()

users.head(2)
users.tail(2)
```

`unique()` **deduplicates** (keeps the first row). That is different from the `unique=` constraint, which **rejects** duplicates.

```python
users.unique("city")
users.unique()          # distinct on all fields
```

## Columns

Read a column as a list:

```python
users["name"]
users.column("age")
users.all_none("score")
users.any_none("score")
```

Assign a column from a sequence, a per-row callable, `apply`, or a scalar:

```python
users["adult"] = users.apply(lambda user: user.age >= 18)
users["adult"] = lambda user: user.age >= 18
users["active"] = True
users["label"] = ["a", "b", "c", "d"]
```

If the name already exists on the model (including `Future` fields), values are written onto that field and `user.adult` type-checks. Undeclared names still work at runtime via a subclass, but attribute access is not typed.

Update only matching rows with typed callables. `update_where` returns a new frame; `apply_where` assigns attributes in place (the field should exist on the model, e.g. `Future`):

```python
older = users.update_where(
    lambda user: user.age > 10,
    lambda user: user.model_copy(update={"age": 10}),
)

def mark_adult(user: User) -> None:
    user.adult = user.age >= 18

users.apply_where(lambda user: user.city == "NYC", mark_adult)
```

`loc` is the shorter pandas-style form when you want to set a column by name:

```python
users.loc[lambda user: user.age > 10, "over_10"] = 10
users.loc[lambda user: user.city == "NYC", "tag"] = lambda user: user.name
users.loc[[True, False, True, False], "flag"] = [1, 2]
users.loc[lambda user: user.city == "NYC"]          # matching rows
users.loc[lambda user: user.city == "NYC", "name"]  # that column for those rows
```

## Transform rows

`apply` returns a list. `map` returns a `TypedFrame`. `project` validates each row as another model.

```python
names = users.apply(lambda user: user.name.upper())

older = users.map(lambda user: user.model_copy(update={"age": user.age + 1}))


class NameOnly(BaseModel):
    name: str

names_frame = users.project(NameOnly)
```

When `map` returns dicts, pass `model=`:

```python
users.map(lambda user: {"name": user.name, "age": user.age}, model=NameOnly)
```

## Look up values from another frame

`map_columns` looks up `src[col]` by key and returns a list in this frame's row order. Missing keys are `None`. Duplicate keys in `src` keep the last value.

```python
class Bonus(BaseModel):
    id: int
    points: int


bonuses = TypedFrame(Bonus, [{"id": 1, "points": 10}, {"id": 2, "points": 3}])

users["bonus"] = users.map_columns(bonuses, "id", "points")
users.map_columns(bonuses, ["id"], "points")   # composite key
users.map_columns(bonuses, None, "points")     # align by row position
```

## Group and aggregate

```python
summary = users.group_by("city").agg(
    n=agg.count(),
    avg_age=agg.mean("age"),
    total_score=agg.sum("score"),
    youngest=agg.min("age"),
    oldest=agg.max("age"),
    first_name=agg.first("name"),
    last_name=agg.last("name"),
    distinct_ages=agg.nunique("age"),
)

for row in summary:
    print(row.city, row.n, row.avg_age)
```

Built-in aggregators (`from typedframe import agg`): `count`, `sum`, `mean`, `min`, `max`, `first`, `last`, `nunique`. `sum` / `mean` / `min` / `max` skip `None` by default (`skipna=False` to keep them).

Custom callables receive the group's rows:

```python
users.group_by("city").agg(
    names=lambda rows: [row.name for row in rows],
)
```

Several keys, a computed key, or the whole frame:

```python
users.group_by("city", "age").agg(n=agg.count())
users.group_by(lambda user: user.age // 10, names=["decade"]).agg(n=agg.count())
users.group_by("city").count()
users.agg(n=agg.count(), avg_age=agg.mean("age"))
```

Iterate groups, or transform each group and concatenate:

```python
grouped = users.group_by("city")
grouped.keys()
grouped["NYC"]
grouped.to_dict()

for city, group in grouped:
    print(city, len(group))

oldest = grouped.map_groups(lambda group: group.sort_by("age", reverse=True).head(1))
```

The aggregation name cannot collide with a group key (`city=agg.count()` raises).

## Combine frames

`append`, `extend`, `concat`, and `+` return new frames. Unique constraints are inherited and re-checked.

```python
users.append({"id": 5, "name": "Eve", "age": 22, "city": "Austin"})
users.extend([User(id=6, name="Fay", age=29, city="Austin")])

more = TypedFrame(User, [{"id": 9, "name": "Zed", "age": 50, "city": "NYC"}])
concat([users, more])
users + more
```

## Convert

```python
users.to_list()
users.to_dicts()
users.to_json()
users.to_json(indent=2)
TypedFrame.from_json(User, users.to_json())
users.dump_json("users.json")
TypedFrame.load_json(User, "users.json", unique="id")

users.to_csv()
users.to_csv("users.csv", sep=";", index=False)
TypedFrame.from_csv(User, "users.csv", comment="#", header=0)
TypedFrame.from_csv(User, "1,Ada,36,London\n", header=None)

users.to_pandas()
users.to_polars()
TypedFrame.from_pandas(User, users.to_pandas())
TypedFrame.from_polars(User, users.to_polars())
```

Pandas and Polars are optional. Install `typedframe[pandas]`, `typedframe[polars]`, or `typedframe[all]`.

## API

| Operation | Method |
| --- | --- |
| Construct | `TypedFrame(Model, rows, unique=...)`, `from_dicts`, `from_models`, `from_json`, `load_json`, `from_csv`, `from_pandas`, `from_polars`, `empty` |
| Unique columns | `unique="id"`, `require_unique("id")` |
| Iterate / index | `for row in frame`, `frame[i]`, `frame[i:j]`, `shape`, `columns` |
| Columns | `frame["col"]`, `frame["col"] = values`, `loc[mask, "col"]`, `column`, `all_none`, `any_none`, `Future[T] = deferred()` |
| Filter | `filter`, `exclude` |
| Transform | `apply`, `map`, `project`, `map_columns`, `update_where`, `apply_where` |
| Sort | `sort_by`, `reverse`, `head`, `tail` |
| Group / aggregate | `group_by`, `agg`, `count`, `map_groups` |
| Deduplicate | `unique` |
| Combine | `append`, `extend`, `concat`, `+` |
| Convert | `to_list`, `to_dicts`, `to_json`, `dump_json`, `to_csv`, `to_pandas`, `to_polars` |

## Development

```bash
uv sync --group dev
uv run ruff check .
uv run pytest
```
