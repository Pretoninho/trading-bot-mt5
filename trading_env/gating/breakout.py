"""
breakout.py
===========
Weekly breakout gate: for each ISO week determine whether Tuesday's close
broke outside the Monday high–low range.

``has_breakout = 1`` if ``tue_close > mon_high`` OR ``tue_close < mon_low``.
``has_breakout = 0`` otherwise (including weeks with insufficient data).

The result is stored in ``week_meta.csv`` and also merged per-bar into the
market DataFrame via :func:`add_breakout_flag`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_week_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Monday range + Tuesday close and derive ``has_breakout`` per ISO week.

    Parameters
    ----------
    df:
        Market DataFrame as returned by
        :func:`~trading_env.data.market_loader.load_mt5_m1_csv`.
        Must contain columns: ``iso_year, iso_week, weekday, high, low, close``.

    Returns
    -------
    DataFrame with columns ``iso_year, iso_week, mon_high, mon_low,
    tue_close, has_breakout`` — one row per ISO week.
    """
    # Monday bars (weekday == 0)
    mon = df[df["weekday"] == 0].groupby(["iso_year", "iso_week"]).agg(
        mon_high=("high", "max"),
        mon_low=("low", "min"),
    )

    # Tuesday bars (weekday == 1) — last close of the day
    tue_bars = df[df["weekday"] == 1].copy()
    # Sort to ensure last bar is indeed the latest Tuesday bar
    tue_bars = tue_bars.sort_values("dt")
    tue = tue_bars.groupby(["iso_year", "iso_week"]).agg(
        tue_close=("close", "last"),
    )

    week_meta = mon.join(tue, how="left")
    week_meta = week_meta.reset_index()

    # has_breakout: 1 if Tuesday close is outside the Monday range
    def _breakout(row: pd.Series) -> int:
        if pd.isna(row["tue_close"]) or pd.isna(row["mon_high"]) or pd.isna(row["mon_low"]):
            return 0
        if row["tue_close"] > row["mon_high"] or row["tue_close"] < row["mon_low"]:
            return 1
        return 0

    week_meta["has_breakout"] = week_meta.apply(_breakout, axis=1)
    return week_meta


def add_breakout_flag(
    df: pd.DataFrame,
    week_meta: pd.DataFrame | None = None,
    week_meta_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Merge ``has_breakout`` onto a per-bar market DataFrame.

    Exactly one of *week_meta* or *week_meta_path* must be provided.

    Parameters
    ----------
    df:
        Market DataFrame (must contain ``iso_year, iso_week``).
    week_meta:
        Pre-computed week-metadata DataFrame (from :func:`compute_week_breakout`).
    week_meta_path:
        Path to a CSV containing at least ``iso_year, iso_week, has_breakout``.

    Returns
    -------
    The input DataFrame with a new ``has_breakout`` column (0 or 1).
    """
    if week_meta is None and week_meta_path is None:
        raise ValueError("Provide either week_meta or week_meta_path.")

    if week_meta is None:
        week_meta = pd.read_csv(week_meta_path)

    flags = week_meta[["iso_year", "iso_week", "has_breakout"]].drop_duplicates()
    df = df.drop(columns=["has_breakout"], errors="ignore")
    df = df.merge(flags, on=["iso_year", "iso_week"], how="left")
    df["has_breakout"] = df["has_breakout"].fillna(0).astype(int)
    return df


def save_week_meta(
    week_meta: pd.DataFrame,
    path: Union[str, Path] = "week_meta.csv",
) -> None:
    """Persist week metadata to CSV.

    Parameters
    ----------
    week_meta:
        DataFrame as returned by :func:`compute_week_breakout`.
    path:
        Destination CSV path.
    """
    week_meta.to_csv(path, index=False)
    print(f"Saved: {path} ({len(week_meta)} weeks)")
