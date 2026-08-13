from __future__ import annotations

import pytest

from tests.models import User
from typedframe import TypedFrame, agg


def test_group_by_field(users: TypedFrame[User]) -> None:
    grouped = users.group_by("city")
    assert len(grouped) == 2
    assert grouped.keys() == ["NYC", "LA"]
    assert [user.name for user in grouped["NYC"]] == ["Alice", "Carol"]
    assert "Austin" not in grouped


def test_group_by_multiple_fields(users: TypedFrame[User]) -> None:
    grouped = users.group_by("city", "age")
    assert ("NYC", 30) in grouped
    assert len(grouped["NYC", 30]) == 2


def test_group_by_callable_with_name(users: TypedFrame[User]) -> None:
    grouped = users.group_by(lambda user: user.age // 10, names=["decade"])
    assert grouped.key_names == ("decade",)
    result = grouped.agg(n=agg.count())
    decades = {row.decade: row.n for row in result}
    assert decades == {3: 2, 2: 1, 4: 1}


def test_group_by_requires_keys(users: TypedFrame[User]) -> None:
    with pytest.raises(ValueError, match="at least one key"):
        users.group_by()


def test_agg_count_mean_sum(users: TypedFrame[User]) -> None:
    result = users.group_by("city").agg(
        n=agg.count(),
        avg_age=agg.mean("age"),
        total_score=agg.sum("score"),
    )
    by_city = {row.city: row for row in result}
    assert by_city["NYC"].n == 2
    assert by_city["NYC"].avg_age == 30
    assert by_city["NYC"].total_score == 17.0
    assert by_city["LA"].n == 2
    assert by_city["LA"].avg_age == 32.5
    assert by_city["LA"].total_score == 7.5


def test_agg_min_max_first_last_nunique(users: TypedFrame[User]) -> None:
    result = users.group_by("city").agg(
        youngest=agg.min("age"),
        oldest=agg.max("age"),
        first_name=agg.first("name"),
        last_name=agg.last("name"),
        distinct_ages=agg.nunique("age"),
    )
    nyc = next(row for row in result if row.city == "NYC")
    assert nyc.youngest == 30
    assert nyc.oldest == 30
    assert nyc.first_name == "Alice"
    assert nyc.last_name == "Carol"
    assert nyc.distinct_ages == 1


def test_agg_callable(users: TypedFrame[User]) -> None:
    result = users.group_by("city").agg(
        names=lambda rows: [row.name for row in rows],
    )
    nyc = next(row for row in result if row.city == "NYC")
    assert nyc.names == ["Alice", "Carol"]


def test_agg_name_collision(users: TypedFrame[User]) -> None:
    with pytest.raises(ValueError, match="collide"):
        users.group_by("city").agg(city=agg.count())


def test_grouped_count_and_iteration(users: TypedFrame[User]) -> None:
    grouped = users.group_by("city")
    counts = {row.city: row.count for row in grouped.count()}
    assert counts == {"NYC": 2, "LA": 2}
    seen = [key for key, _frame in grouped]
    assert seen == ["NYC", "LA"]


def test_map_groups(users: TypedFrame[User]) -> None:
    def keep_oldest(group: TypedFrame[User]) -> TypedFrame[User]:
        return group.sort_by("age", reverse=True).head(1)

    oldest = users.group_by("city").map_groups(keep_oldest)
    assert {user.name for user in oldest} == {"Alice", "Dan"}


def test_frame_agg(users: TypedFrame[User]) -> None:
    summary = users.agg(n=agg.count(), avg_age=agg.mean("age"))
    assert len(summary) == 1
    assert summary[0].n == 4
    assert summary[0].avg_age == 31.25


def test_empty_group_agg() -> None:
    grouped = TypedFrame.empty(User).group_by("city")
    result = grouped.agg(n=agg.count())
    assert len(result) == 0
    assert "city" in result.columns
    assert "n" in result.columns
