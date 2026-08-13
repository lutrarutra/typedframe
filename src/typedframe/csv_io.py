from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

DEFAULT_NA_VALUES = frozenset({"", "NA", "NaN", "nan", "null", "None"})


def _as_text(source: str | bytes | Path, *, encoding: str) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding=encoding)
    if isinstance(source, bytes):
        return source.decode(encoding)
    if "\n" not in source and Path(source).is_file():
        return Path(source).read_text(encoding=encoding)
    return source


def _cell(value: Any, na_rep: str) -> Any:
    if value is None:
        return na_rep
    return value


def frame_to_csv(
    records: list[dict[str, Any]],
    fieldnames: Sequence[str],
    *,
    sep: str = ",",
    index: bool = False,
    index_label: str = "",
    header: bool = True,
    na_rep: str = "",
    lineterminator: str = "\n",
    quotechar: str = '"',
) -> str:
    names = list(fieldnames)
    if index:
        names = [index_label, *names]
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=names,
        extrasaction="ignore",
        delimiter=sep,
        lineterminator=lineterminator,
        quotechar=quotechar,
        quoting=csv.QUOTE_MINIMAL,
    )
    if header:
        writer.writeheader()
    for row_index, record in enumerate(records):
        row = {name: _cell(record.get(name), na_rep) for name in fieldnames}
        if index:
            row[index_label] = row_index
        writer.writerow(row)
    return buf.getvalue()


def csv_to_records(
    source: str | bytes | Path,
    model: type[BaseModel],
    *,
    sep: str = ",",
    header: int | None = 0,
    comment: str | None = "#",
    index_col: int | str | Literal[False] | None = False,
    encoding: str = "utf-8",
    na_values: Sequence[str] | None = None,
    quotechar: str = '"',
    skip_blank_lines: bool = True,
) -> list[dict[str, Any]]:
    text = _as_text(source, encoding=encoding)
    lines = text.splitlines()
    if comment:
        lines = [line for line in lines if not line.lstrip().startswith(comment)]
    if skip_blank_lines:
        lines = [line for line in lines if line.strip()]

    reader_kwargs = {"delimiter": sep, "quotechar": quotechar}
    if header is None:
        fieldnames = list(model.model_fields)
        body = lines
        parsed = list(csv.reader(body, **reader_kwargs))
        records = [dict(zip(fieldnames, row, strict=False)) for row in parsed]
    else:
        if header < 0:
            raise ValueError("header must be None or a non-negative row index")
        if header >= len(lines):
            raise ValueError(f"header row {header} is past the end of the file")
        fieldnames = next(csv.reader([lines[header]], **reader_kwargs))
        body = lines[header + 1 :]
        parsed = list(csv.reader(body, **reader_kwargs))
        records = [dict(zip(fieldnames, row, strict=False)) for row in parsed]

    drop: str | None = None
    if index_col is not False and index_col is not None:
        if isinstance(index_col, int):
            if index_col < 0 or index_col >= len(fieldnames):
                raise ValueError(f"index_col {index_col} is out of range")
            drop = fieldnames[index_col]
        else:
            drop = index_col
        records = [{key: value for key, value in record.items() if key != drop} for record in records]

    missing = DEFAULT_NA_VALUES if na_values is None else frozenset(na_values)
    cleaned: list[dict[str, Any]] = []
    for record in records:
        cleaned.append({key: None if value in missing else value for key, value in record.items()})
    return cleaned
