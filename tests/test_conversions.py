from __future__ import annotations

import pytest

from tests.models import User
from typedframe import TypedFrame


def test_to_from_pandas(users: TypedFrame[User]) -> None:
    pd = pytest.importorskip("pandas")
    df = users.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert list(df["name"]) == ["Alice", "Bob", "Carol", "Dan"]
    restored = TypedFrame.from_pandas(User, df)
    assert restored == users


def test_to_from_polars(users: TypedFrame[User]) -> None:
    pl = pytest.importorskip("polars")
    df = users.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert df["name"].to_list() == ["Alice", "Bob", "Carol", "Dan"]
    restored = TypedFrame.from_polars(User, df)
    assert restored == users
