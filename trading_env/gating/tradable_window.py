"""
tradable_window.py
==================
Compute the minute-level ``tradable_now`` flag:

    tradable_now = is_safe_week & has_breakout
                   & (weekday in {Wednesday=2, Thursday=3})
                   & (08:00 <= time_of_day <= 22:00 UTC)

``time_of_day`` is expressed as **minutes since midnight** (integer).

If ``is_safe_week`` is absent from the DataFrame, all bars are treated as
safe (useful for back-tests that do not use the news filter).
"""

from __future__ import annotations

import pandas as pd

# Tradable weekdays: Wednesday=2, Thursday=3
TRADABLE_WEEKDAYS = {2, 3}

# Tradable time window [08:00, 22:00] UTC expressed as minutes since midnight
TRADABLE_START_MIN = 8 * 60    # 480
TRADABLE_END_MIN = 22 * 60     # 1320


def add_tradable_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``tradable_now`` column to a bar DataFrame.

    Parameters
    ----------
    df:
        Market DataFrame containing at minimum:
        ``weekday``, ``time_of_day``, ``has_breakout``.
        Optionally: ``is_safe_week`` (defaults to 1 for all bars when absent).

    Returns
    -------
    The input DataFrame with a new ``tradable_now`` integer column (0 or 1).
    """
    if "is_safe_week" in df.columns:
        safe = df["is_safe_week"].eq(1)
    else:
        safe = pd.Series(True, index=df.index)

    breakout = df["has_breakout"].eq(1)
    wd = df["weekday"].isin(TRADABLE_WEEKDAYS)
    tod = df["time_of_day"].between(TRADABLE_START_MIN, TRADABLE_END_MIN)

    df = df.copy()
    df["tradable_now"] = (safe & breakout & wd & tod).astype(int)
    return df
