"""
market_loader.py
================
Load MT5-exported EURUSD M1 OHLCV+spread CSV files into a UTC-indexed
DataFrame enriched with metadata needed by the gating and environment
modules.

MT5 export format (space-separated)::

    <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>

Example row::

    2025.01.06 00:00:00 1.02345 1.02390 1.02310 1.02360 312 0 8

``SPREAD`` is expressed in **points** (5-digit EURUSD → 1 point = 0.00001).
``spread_price`` (in price units) = ``spread_points * POINT``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

# ---------------------------------------------------------------------------
# EURUSD instrument constants
# ---------------------------------------------------------------------------

POINT = 0.00001   # 5-digit EURUSD
PIP_SIZE = 0.0001  # 1 pip = 10 points

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_mt5_m1_csv(
    path: Union[str, Path],
    unsafe_weeks_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Parse an MT5 M1 CSV export and return an enriched DataFrame.

    Parameters
    ----------
    path:
        Path to the MT5 space-separated CSV file.
    unsafe_weeks_path:
        Optional path to ``unsafe_weeks.csv`` (columns ``iso_year, iso_week``).
        When provided, a boolean ``is_safe_week`` column is added.

    Returns
    -------
    DataFrame with a UTC-aware ``dt`` column (and UTC DatetimeIndex) plus
    the following added columns:

    * ``spread_price``   — spread in price units (spread_points * POINT).
    * ``iso_year``       — ISO calendar year.
    * ``iso_week``       — ISO calendar week.
    * ``weekday``        — Day of week (Mon=0 … Sun=6).
    * ``time_of_day``    — Pandas Timedelta representing time within the day.
    * ``is_safe_week``   — 1 if no high-impact news on Wed/Thu; else 0.
                           Only present when *unsafe_weeks_path* is supplied.
    """
    df = _parse_raw_csv(path)
    df = _add_datetime_fields(df)

    if unsafe_weeks_path is not None:
        df = _merge_safe_week(df, unsafe_weeks_path)

    df = df.set_index("dt", drop=False)
    df.index.name = "datetime"
    return df


def _parse_raw_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read the MT5 raw CSV into a DataFrame with canonical column names."""
    df = pd.read_csv(path, sep=r"\s+", engine="python", header=0)
    df.columns = [c.strip("<>").lower() for c in df.columns]

    # MT5 may export either 'date'+'time' or a combined 'datetime' column
    if "date" in df.columns and "time" in df.columns:
        df["dt"] = pd.to_datetime(
            df["date"] + " " + df["time"],
            format="%Y.%m.%d %H:%M:%S",
            utc=True,
        )
    elif "datetime" in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"], utc=True)
    else:
        raise ValueError(
            "MT5 CSV must contain either ('date','time') or 'datetime' columns."
        )

    # Ensure required OHLCV columns are present
    required = {"open", "high", "low", "close", "spread"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in MT5 CSV: {missing}")

    df["spread_points"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0)
    df["spread_price"] = df["spread_points"] * POINT

    # Keep a consistent float dtype for OHLC
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _add_datetime_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add ISO year/week, weekday, and time-of-day columns."""
    iso = df["dt"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int)
    df["weekday"] = df["dt"].dt.weekday  # Mon=0 … Sun=6
    df["time_of_day"] = df["dt"].dt.hour * 60 + df["dt"].dt.minute  # minutes since midnight
    return df


def _merge_safe_week(
    df: pd.DataFrame,
    unsafe_weeks_path: Union[str, Path],
) -> pd.DataFrame:
    """Join unsafe-week flags onto bar data."""
    unsafe = pd.read_csv(unsafe_weeks_path)
    unsafe = unsafe[["iso_year", "iso_week"]].drop_duplicates()
    unsafe["_unsafe"] = 1

    df = df.merge(unsafe, on=["iso_year", "iso_week"], how="left")
    df["is_safe_week"] = (df["_unsafe"].isna()).astype(int)
    df = df.drop(columns=["_unsafe"])
    return df
