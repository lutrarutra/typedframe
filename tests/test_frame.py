from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from tests.models import User
from typedframe import Future, TypedFrame, concat, deferred


def test_construct_from_dicts_validates() -> None:
    frame = TypedFrame(
        User,
        [{"id": 1, "name": "Ada", "age": 36, "city": "London"}],
    )
    assert len(frame) == 1
    assert isinstance(frame[0], User)
    assert frame[0].name == "Ada"


def test_construct_rejects_invalid_rows() -> None:
    with pytest.raises(ValidationError):
        TypedFrame(User, [{"id": "x", "name": "Ada"}])


def test_construct_requires_pydantic_model() -> None:
    with pytest.raises(TypeError, match="BaseModel"):
        TypedFrame(dict, [])  # type: ignore[type-var]


def test_empty_and_bool(user_model: type[User]) -> None:
    frame = TypedFrame.empty(user_model)
    assert len(frame) == 0
    assert not frame
    assert frame.columns == ("id", "name", "age", "city", "score")
    assert frame.shape == (0, 5)


def test_iteration_returns_models(users: TypedFrame[User]) -> None:
    names = [user.name for user in users]
    assert names == ["Alice", "Bob", "Carol", "Dan"]
    assert all(isinstance(user, User) for user in users)


def test_index_and_slice(users: TypedFrame[User]) -> None:
    assert users[1].name == "Bob"
    subset = users[1:3]
    assert isinstance(subset, TypedFrame)
    assert [user.name for user in subset] == ["Bob", "Carol"]


def test_filter_and_exclude(users: TypedFrame[User]) -> None:
    nyc = users.filter(lambda user: user.city == "NYC")
    assert [user.name for user in nyc] == ["Alice", "Carol"]
    not_nyc = users.exclude(lambda user: user.city == "NYC")
    assert [user.name for user in not_nyc] == ["Bob", "Dan"]


def test_sort_by_field_and_callable(users: TypedFrame[User]) -> None:
    by_age = users.sort_by("age", "name")
    assert [user.name for user in by_age] == ["Bob", "Alice", "Carol", "Dan"]
    by_name_len = users.sort_by(lambda user: len(user.name), reverse=True)
    assert by_name_len[0].name == "Alice"


def test_sort_by_unknown_field(users: TypedFrame[User]) -> None:
    with pytest.raises(AttributeError, match="country"):
        users.sort_by("country")


def test_map_same_model(users: TypedFrame[User]) -> None:
    older = users.map(lambda user: user.model_copy(update={"age": user.age + 1}))
    assert older.model is User
    assert older[0].age == 31
    assert users[0].age == 30


def test_map_to_dicts_requires_model(users: TypedFrame[User]) -> None:
    with pytest.raises(TypeError, match="model is required"):
        users.map(lambda user: {"name": user.name})


def test_project(users: TypedFrame[User]) -> None:
    class NameOnly(BaseModel):
        name: str

    names = users.project(NameOnly)
    assert names.model is NameOnly
    assert names.to_dicts() == [
        {"name": "Alice"},
        {"name": "Bob"},
        {"name": "Carol"},
        {"name": "Dan"},
    ]


def test_unique_by_field(users: TypedFrame[User]) -> None:
    unique_cities = users.unique("city")
    assert [user.city for user in unique_cities] == ["NYC", "LA"]


def test_head_tail_reverse(users: TypedFrame[User]) -> None:
    assert [user.name for user in users.head(2)] == ["Alice", "Bob"]
    assert [user.name for user in users.tail(2)] == ["Carol", "Dan"]
    assert [user.name for user in users.reverse()] == ["Dan", "Carol", "Bob", "Alice"]


def test_column(users: TypedFrame[User]) -> None:
    assert users.column("age") == [30, 25, 30, 40]
    assert users["age"] == [30, 25, 30, 40]


def test_shape(users: TypedFrame[User]) -> None:
    assert users.shape == (4, 5)


def test_apply(users: TypedFrame[User]) -> None:
    names = users.apply(lambda user: user.name.upper())
    assert names == ["ALICE", "BOB", "CAROL", "DAN"]
    flags = users.apply(lambda user: user.age >= 30)
    assert flags == [True, False, True, True]


def test_map_columns_single_key(users: TypedFrame[User]) -> None:
    class Score(BaseModel):
        id: int
        points: int

    src = TypedFrame(
        Score,
        [
            {"id": 2, "points": 10},
            {"id": 1, "points": 20},
            {"id": 2, "points": 15},
        ],
    )
    assert users.map_columns(src, "id", "points") == [20, 15, None, None]


def test_map_columns_multiple_keys(users: TypedFrame[User]) -> None:
    class Bonus(BaseModel):
        city: str
        age: int
        bonus: int

    src = TypedFrame(
        Bonus,
        [
            {"city": "NYC", "age": 30, "bonus": 5},
            {"city": "LA", "age": 25, "bonus": 3},
        ],
    )
    assert users.map_columns(src, ["city", "age"], "bonus") == [5, 3, 5, None]


def test_map_columns_none_key_and_positional(users: TypedFrame[User]) -> None:
    class Label(BaseModel):
        id: int | None
        label: str

    src = TypedFrame(
        Label,
        [
            {"id": 1, "label": "a"},
            {"id": None, "label": "missing"},
        ],
    )
    dst = TypedFrame(
        Label,
        [
            {"id": 1, "label": ""},
            {"id": None, "label": ""},
            {"id": 9, "label": ""},
        ],
    )
    assert dst.map_columns(src, "id", "label") == ["a", None, None]
    assert dst.map_columns(src, None, "label") == ["a", "missing", None]


def test_update_where_returns_new_frame(users: TypedFrame[User]) -> None:
    updated = users.update_where(
        lambda user: user.age > 30,
        lambda user: user.model_copy(update={"age": 99}),
    )
    assert users["age"] == [30, 25, 30, 40]
    assert updated["age"] == [30, 25, 30, 99]
    assert updated[3].name == "Dan"


def test_loc_assign_scalar_and_keep_others(users: TypedFrame[User]) -> None:
    users.loc[lambda user: user.age >= 30, "flag"] = 10
    assert [user.flag for user in users] == [10, None, 10, 10]
    assert users[1].age == 25


def test_loc_assign_callable_and_sequence(users: TypedFrame[User]) -> None:
    users.loc[lambda user: user.city == "NYC", "tag"] = lambda user: user.name[0]
    assert users.loc[lambda user: user.city == "NYC", "tag"] == ["A", "C"]
    users.loc[[False, True, False, True], "tag"] = ["B", "D"]
    assert [getattr(user, "tag", None) for user in users] == ["A", "B", "C", "D"]


def test_loc_updates_existing_column(users: TypedFrame[User]) -> None:
    users.loc[lambda user: user.age > 30, "age"] = 99
    assert users["age"] == [30, 25, 30, 99]


def test_loc_filter(users: TypedFrame[User]) -> None:
    nyc = users.loc[lambda user: user.city == "NYC"]
    assert [user.name for user in nyc] == ["Alice", "Carol"]


def test_loc_keeps_unset_future_hidden() -> None:
    class UserWithFlag(BaseModel):
        id: int
        age: int
        over_30: Future[int] = deferred()

    frame = TypedFrame(UserWithFlag, [{"id": 1, "age": 20}, {"id": 2, "age": 40}])
    frame.loc[lambda user: user.age > 30, "over_30"] = 10
    assert frame[0].over_30 is None
    assert frame[1].over_30 == 10
    lines = repr(frame).splitlines()
    assert "over_30=" not in lines[1]
    assert "over_30=10" in lines[2]


def test_loc_value_length_mismatch(users: TypedFrame[User]) -> None:
    with pytest.raises(ValueError, match="selected rows"):
        users.loc[lambda user: user.city == "NYC", "x"] = [1]


def test_apply_where_assigns_typed_field() -> None:
    class UserWithFlag(BaseModel):
        id: int
        age: int
        over_30: Future[int] = deferred()

    frame = TypedFrame(UserWithFlag, [{"id": 1, "age": 20}, {"id": 2, "age": 40}])

    def mark(user: UserWithFlag) -> None:
        user.over_30 = 10

    frame.apply_where(lambda user: user.age > 30, mark)
    assert frame[0].over_30 is None
    assert frame[1].over_30 == 10
    lines = repr(frame).splitlines()
    assert "over_30=" not in lines[1]
    assert "over_30=10" in lines[2]


def test_assign_column_from_sequence(users: TypedFrame[User]) -> None:
    users["adult"] = [True, False, True, True]
    assert "adult" in users.columns
    assert users.shape == (4, 6)
    assert users["adult"] == [True, False, True, True]
    assert users[0].adult is True
    assert isinstance(users[0], User)


def test_assign_column_from_apply(users: TypedFrame[User]) -> None:
    users["label"] = users.apply(lambda user: f"{user.city}:{user.age}")
    assert users["label"] == ["NYC:30", "LA:25", "NYC:30", "LA:40"]


def test_assign_column_from_callable_and_scalar(users: TypedFrame[User]) -> None:
    users["adult"] = lambda user: user.age >= 30
    users["active"] = True
    assert users["adult"] == [True, False, True, True]
    assert users["active"] == [True, True, True, True]
    assert users[1].adult is False
    assert users[1].active is True


def test_assign_column_replace_existing(users: TypedFrame[User]) -> None:
    users["age"] = [1, 2, 3, 4]
    assert users["age"] == [1, 2, 3, 4]
    assert users.model is User


def test_assign_column_length_mismatch(users: TypedFrame[User]) -> None:
    with pytest.raises(ValueError, match="does not match"):
        users["extra"] = [1, 2]


def test_assign_declared_future_column() -> None:
    class UserWithExtra(BaseModel):
        id: int
        name: str
        age: int
        city: str
        extra: Future[int] = deferred()

    frame = TypedFrame(
        UserWithExtra,
        [{"id": 1, "name": "Ada", "age": 36, "city": "London"}],
    )
    assert frame[0].extra is None
    assert "extra=" not in repr(frame)
    assert "extra" not in frame.to_dicts()[0]
    frame["extra"] = [99]
    assert frame.model is UserWithExtra
    assert frame[0].extra == 99
    assert isinstance(frame[0], UserWithExtra)
    assert "extra=99" in repr(frame)


def test_assign_other_column_keeps_future_hidden() -> None:
    class UserWithExtra(BaseModel):
        id: int
        name: str
        extra: Future[int] = deferred()

    frame = TypedFrame(UserWithExtra, [{"id": 1, "name": "Ada"}])
    frame["label"] = ["x"]
    assert frame[0].extra is None
    assert "extra=" not in repr(frame)
    assert "label='x'" in repr(frame)


def test_future_column_with_default() -> None:
    class UserWithExtra(BaseModel):
        id: int
        name: str
        extra: Future[int] = deferred(0)

    frame = TypedFrame(UserWithExtra, [{"id": 1, "name": "Ada"}])
    assert frame[0].extra == 0
    assert "extra=0" in repr(frame)
    frame["extra"] = [7]
    assert frame[0].extra == 7


def test_assign_column_on_empty_frame() -> None:
    frame = TypedFrame.empty(User)
    frame["flag"] = []
    assert frame.shape == (0, 6)
    assert "flag" in frame.columns


def test_unique_constraint_on_construct(users: TypedFrame[User]) -> None:
    frame = TypedFrame(User, users.to_list(), unique="id")
    assert frame.unique_columns == frozenset({"id"})


def test_unique_constraint_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Column 'city' must be unique"):
        TypedFrame(
            User,
            [
                {"id": 1, "name": "Ada", "age": 36, "city": "NYC"},
                {"id": 2, "name": "Bob", "age": 25, "city": "NYC"},
            ],
            unique="city",
        )


def test_require_unique_after_construct(users: TypedFrame[User]) -> None:
    users.require_unique("id")
    assert "id" in users.unique_columns
    with pytest.raises(ValueError, match="Column 'city' must be unique"):
        users.require_unique("city")


def test_require_unique_on_new_column(users: TypedFrame[User]) -> None:
    users["code"] = [10, 20, 30, 40]
    users.require_unique("code")
    with pytest.raises(ValueError, match="Column 'code' must be unique"):
        users["code"] = [10, 20, 10, 40]


def test_unique_constraint_on_append(users: TypedFrame[User]) -> None:
    frame = TypedFrame(User, users.to_list(), unique="id")
    with pytest.raises(ValueError, match="duplicate value 1"):
        frame.append({"id": 1, "name": "Eve", "age": 22, "city": "Austin"})


def test_unique_constraint_on_concat(users: TypedFrame[User]) -> None:
    left = TypedFrame(User, users.to_list()[:2], unique="id")
    right = TypedFrame(User, users.to_list()[1:3])
    with pytest.raises(ValueError, match="Column 'id' must be unique"):
        concat([left, right])


def test_unique_constraint_survives_filter(users: TypedFrame[User]) -> None:
    frame = TypedFrame(User, users.to_list(), unique="id")
    nyc = frame.filter(lambda user: user.city == "NYC")
    assert nyc.unique_columns == frozenset({"id"})
    assert [user.id for user in nyc] == [1, 3]


def test_append_extend_are_immutable(users: TypedFrame[User]) -> None:
    grown = users.append({"id": 5, "name": "Eve", "age": 22, "city": "Austin"})
    assert len(users) == 4
    assert len(grown) == 5
    assert grown[-1].name == "Eve"

    more = grown.extend([User(id=6, name="Fay", age=29, city="Austin")])
    assert len(grown) == 5
    assert len(more) == 6


def test_concat_and_add(users: TypedFrame[User]) -> None:
    extra = TypedFrame(User, [User(id=9, name="Zed", age=50, city="NYC")])
    combined = concat([users, extra])
    assert len(combined) == 5
    assert (users + extra)[-1].name == "Zed"


def test_concat_rejects_mismatched_models(users: TypedFrame[User]) -> None:
    class Other(BaseModel):
        name: str

    with pytest.raises(TypeError, match="Cannot concat"):
        concat([users, TypedFrame(Other, [{"name": "x"}])])


def test_to_dicts_and_json_roundtrip(users: TypedFrame[User]) -> None:
    records = users.to_dicts()
    assert records[0]["name"] == "Alice"
    restored = TypedFrame.from_json(User, users.to_json())
    assert restored == users
    parsed = json.loads(users.to_json())
    assert parsed[1]["city"] == "LA"
    pretty = users.to_json(indent=2)
    assert "\n" in pretty


def test_dump_and_load_json(users: TypedFrame[User], tmp_path: Path) -> None:
    path = tmp_path / "users.json"
    users.dump_json(path, indent=2)
    loaded = TypedFrame.load_json(User, path, unique="id")
    assert loaded == users
    assert loaded.unique_columns == frozenset({"id"})


def test_json_omits_unset_future() -> None:
    class Row(BaseModel):
        id: int
        extra: Future[int] = deferred()

    frame = TypedFrame(Row, [{"id": 1}])
    assert "extra" not in json.loads(frame.to_json())[0]
    frame["extra"] = [9]
    assert json.loads(frame.to_json())[0]["extra"] == 9


def test_equality(users: TypedFrame[User]) -> None:
    clone = TypedFrame(User, users.to_list())
    assert clone == users
    assert users != users.filter(lambda user: user.age > 100)


def test_repr_empty_and_preview(users: TypedFrame[User]) -> None:
    assert "empty" in repr(TypedFrame.empty(User))
    text = repr(users)
    assert "TypedFrame[User](4 rows)" in text
    assert "Alice" in text
    assert "score=None" in text
