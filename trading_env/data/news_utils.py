"""
news_utils.py
=============
Convert FXStreet calendar CSV exports into:
  - ``news_events.csv``  — filtered, normalised high-impact news events.
  - ``unsafe_weeks.csv`` — ISO weeks that contain a high-impact EUR/USD
    CPI/PPI/NFP/FOMC event falling on **Wednesday or Thursday (UTC)**.

FXStreet CSV columns expected::

    Id, Start, Name, Impact, Currency

``Start`` must be in UTC with format ``%m/%d/%Y %H:%M:%S``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"

CURRENCIES = {"USD", "EUR"}

# Keywords that identify CPI / PPI / NFP / FOMC events
_PATTERNS = [
    r"\bCPI\b",
    r"Consumer Price Index",
    r"Harmonized Index of Consumer Prices",
    r"\bPPI\b",
    r"Producer Price Index",
    r"Nonfarm Payrolls",
    r"Non-Farm Payrolls",
    r"\bNFP\b",
    r"\bFOMC\b",
    r"Federal Open Market Committee",
    r"FOMC Minutes",
    r"FOMC Statement",
    r"Federal Funds Rate",
    r"Rate Decision",
]
NEWS_REGEX = re.compile("|".join(_PATTERNS), flags=re.IGNORECASE)

# Wednesday=2, Thursday=3 (Python .weekday(): Mon=0 … Sun=6)
UNSAFE_WEEKDAYS = {2, 3}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_fxstreet_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a single FXStreet calendar CSV export.

    Parameters
    ----------
    path:
        Path to the raw FXStreet CSV file.

    Returns
    -------
    DataFrame with columns: ``Id, Start, Name, Impact, Currency``.
    """
    df = pd.read_csv(path)
    required = {"Start", "Name", "Impact", "Currency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in FXStreet CSV: {missing}")
    return df


def concat_fxstreet_exports(paths: list[Union[str, Path]]) -> pd.DataFrame:
    """Concatenate multiple FXStreet quarterly CSV exports.

    Parameters
    ----------
    paths:
        List of paths to individual FXStreet CSV files.

    Returns
    -------
    Deduplicated DataFrame of all events.
    """
    frames = [load_fxstreet_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Id"])
    return df


def filter_news_events(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a raw FXStreet DataFrame to CPI/PPI/NFP/FOMC high-impact events.

    Applies three filters:
    1. ``Impact == "HIGH"``
    2. ``Currency in {"USD", "EUR"}``
    3. ``Name`` matches CPI/PPI/NFP/FOMC keyword pattern

    Also parses ``Start`` to a UTC-aware ``time_utc`` column and normalises
    ``impact`` to lower-case ``"high"``.

    Parameters
    ----------
    df:
        Raw FXStreet DataFrame (from :func:`load_fxstreet_csv` or
        :func:`concat_fxstreet_exports`).

    Returns
    -------
    Filtered DataFrame with schema ``time_utc, currency, impact, title``.
    """
    df = df.copy()

    # Normalise string columns
    df["Impact"] = df["Impact"].astype(str).str.upper().str.strip()
    df["Currency"] = df["Currency"].astype(str).str.upper().str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()

    # Parse datetime (UTC)
    df["time_utc"] = pd.to_datetime(
        df["Start"], format=DATETIME_FORMAT, errors="raise", utc=True
    )

    # Apply filters
    mask = (
        df["Impact"].eq("HIGH")
        & df["Currency"].isin(CURRENCIES)
        & df["Name"].apply(lambda s: bool(NEWS_REGEX.search(s)))
    )
    df = df[mask].copy()

    # Build output schema
    events = df[["time_utc", "Currency", "Impact", "Name"]].rename(
        columns={"Currency": "currency", "Impact": "impact", "Name": "title"}
    )
    events["impact"] = "high"
    events = events.sort_values("time_utc").reset_index(drop=True)
    return events


def build_unsafe_weeks(events: pd.DataFrame) -> pd.DataFrame:
    """Derive unsafe ISO weeks from a filtered news-events DataFrame.

    A week is *unsafe* when at least one high-impact EUR/USD CPI/PPI/NFP/FOMC
    event falls on **Wednesday or Thursday (UTC)** within that ISO week.

    Parameters
    ----------
    events:
        Filtered events DataFrame as returned by :func:`filter_news_events`.

    Returns
    -------
    DataFrame with columns ``iso_year, iso_week`` (one row per unsafe week).
    """
    ev = events.copy()
    ev["weekday"] = ev["time_utc"].dt.weekday
    iso = ev["time_utc"].dt.isocalendar()
    ev["iso_year"] = iso.year.astype(int)
    ev["iso_week"] = iso.week.astype(int)

    unsafe = (
        ev[ev["weekday"].isin(UNSAFE_WEEKDAYS)][["iso_year", "iso_week"]]
        .drop_duplicates()
        .sort_values(["iso_year", "iso_week"])
        .reset_index(drop=True)
    )
    return unsafe


def convert_fxstreet_to_news_csv(
    input_paths: list[Union[str, Path]],
    out_events: Union[str, Path] = "news_events.csv",
    out_unsafe_weeks: Union[str, Path] = "unsafe_weeks.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end pipeline: FXStreet CSV(s) → ``news_events.csv`` + ``unsafe_weeks.csv``.

    Parameters
    ----------
    input_paths:
        One or more FXStreet quarterly CSV export paths.
    out_events:
        Output path for the normalised news events CSV.
    out_unsafe_weeks:
        Output path for the unsafe-weeks CSV.

    Returns
    -------
    Tuple of ``(events_df, unsafe_weeks_df)``.
    """
    raw = concat_fxstreet_exports(input_paths)
    events = filter_news_events(raw)
    unsafe = build_unsafe_weeks(events)

    events.to_csv(out_events, index=False)
    unsafe.to_csv(out_unsafe_weeks, index=False)

    print(f"Saved: {out_events} ({len(events)} events)")
    print(f"Saved: {out_unsafe_weeks} ({len(unsafe)} unsafe weeks)")
    return events, unsafe
