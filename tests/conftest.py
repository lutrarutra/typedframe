from __future__ import annotations

import pytest

from tests.models import User
from typedframe import TypedFrame


@pytest.fixture
def user_model() -> type[User]:
    return User


@pytest.fixture
def users() -> TypedFrame[User]:
    return TypedFrame(
        User,
        [
            User(id=1, name="Alice", age=30, city="NYC", score=9.0),
            User(id=2, name="Bob", age=25, city="LA", score=7.5),
            User(id=3, name="Carol", age=30, city="NYC", score=8.0),
            User(id=4, name="Dan", age=40, city="LA", score=None),
        ],
    )
