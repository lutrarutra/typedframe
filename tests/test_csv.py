from __future__ import annotations

from pathlib import Path

from tests.models import User
from typedframe import TypedFrame


def test_csv_roundtrip(users: TypedFrame[User]) -> None:
    text = users.to_csv()
    restored = TypedFrame.from_csv(User, text)
    assert restored == users
    assert "id,name,age,city,score" in text.splitlines()[0]


def test_csv_custom_sep_and_file(users: TypedFrame[User], tmp_path: Path) -> None:
    path = tmp_path / "users.csv"
    users.to_csv(path, sep=";")
    loaded = TypedFrame.from_csv(User, path, sep=";")
    assert loaded == users


def test_csv_comment_and_header_none() -> None:
    raw = "# ignore\nid;name;age;city\n1;Ada;36;London\n"
    frame = TypedFrame.from_csv(User, raw, sep=";", comment="#")
    assert frame[0].name == "Ada"

    no_header = "1,Ada,36,London,\n"
    frame = TypedFrame.from_csv(User, no_header, header=None, comment=None)
    assert frame[0].id == 1
    assert frame[0].city == "London"


def test_csv_header_row_index() -> None:
    raw = "skip\nid,name,age,city,score\n2,Bob,25,LA,7.5\n"
    frame = TypedFrame.from_csv(User, raw, header=1, comment=None)
    assert frame[0].name == "Bob"


def test_csv_index_col_dropped(users: TypedFrame[User]) -> None:
    text = users.to_csv(index=True, index_label="idx")
    restored = TypedFrame.from_csv(User, text, index_col="idx")
    assert restored == users
    restored_pos = TypedFrame.from_csv(User, text, index_col=0)
    assert restored_pos == users


def test_csv_na_and_column_subset(users: TypedFrame[User]) -> None:
    text = users.to_csv(columns=["id", "name"], na_rep="NA")
    assert "score" not in text
    assert text.splitlines()[0] == "id,name"
