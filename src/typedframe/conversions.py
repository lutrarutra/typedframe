from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def to_pandas(records: list[dict[str, Any]]) -> pd.DataFrame:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for to_pandas(). Install it with: pip install 'typedframe[pandas]'"
        ) from exc
    return pd.DataFrame(records)


def from_pandas(df: pd.DataFrame) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for from_pandas(). Install it with: pip install 'typedframe[pandas]'"
        ) from exc
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df).__name__}")
    return [
        {key: None if _is_missing(value) or value is pd.NA else value for key, value in record.items()}
        for record in df.to_dict(orient="records")
    ]


def to_polars(records: list[dict[str, Any]]) -> pl.DataFrame:
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "polars is required for to_polars(). Install it with: pip install 'typedframe[polars]'"
        ) from exc
    return pl.DataFrame(records)


def from_polars(df: pl.DataFrame) -> list[dict[str, Any]]:
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "polars is required for from_polars(). Install it with: pip install 'typedframe[polars]'"
        ) from exc
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected polars.DataFrame, got {type(df).__name__}")
    return df.to_dicts()
