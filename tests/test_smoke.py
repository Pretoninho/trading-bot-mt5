"""
test_smoke.py
=============
Smoke tests for the trading_env package.

Covers:
* :mod:`trading_env.data.news_utils`   — FXStreet parsing + unsafe-week detection
* :mod:`trading_env.data.market_loader` — MT5 M1 CSV loading
* :mod:`trading_env.gating.breakout`    — weekly breakout gate
* :mod:`trading_env.gating.tradable_window` — tradable_now flag
* :mod:`trading_env.utils.position_sizing`  — lot computation
* :class:`trading_env.env.trading_env.EURUSDTradingEnv`
  — single-day episode with gating / early-stop / stop-out checks
"""

from __future__ import annotations

import io
import textwrap
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trading_env.data.news_utils import filter_news_events, build_unsafe_weeks
from trading_env.data.market_loader import POINT
from trading_env.gating.breakout import compute_week_breakout, add_breakout_flag
from trading_env.gating.tradable_window import (
    add_tradable_flag,
    TRADABLE_START_MIN,
    TRADABLE_END_MIN,
)
from trading_env.utils.position_sizing import compute_lots, MIN_LOT
from trading_env.env.trading_env import (
    EURUSDTradingEnv,
    HOLD,
    OPEN_LONG,
    OPEN_SHORT,
    CLOSE,
    PROTECT,
    MANAGE_TP,
    EPISODE_EQUITY_GAIN_LIMIT,
    EPISODE_EQUITY_LOSS_LIMIT,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FXSTREET_SAMPLE = textwrap.dedent(
    """\
    Id,Start,Name,Impact,Currency
    aaa,01/08/2025 19:00:00,FOMC Minutes,HIGH,USD
    bbb,01/09/2025 10:00:00,Core CPI (MoM),HIGH,USD
    ccc,01/07/2025 10:00:00,Harmonized Index of Consumer Prices (YoY),HIGH,EUR
    ddd,01/06/2025 09:00:00,ISM Manufacturing PMI,HIGH,USD
    eee,01/08/2025 13:00:00,ADP Employment Change,HIGH,USD
    fff,02/05/2025 13:30:00,Nonfarm Payrolls,HIGH,USD
    """
)

# A minimal MT5-style CSV: one "full" Wednesday so we can test gating.
# Format:  <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
# 2025-01-06 is a Monday; 2025-01-07 Tuesday; 2025-01-08 Wednesday.

def _make_mt5_csv(n_days: int = 3) -> str:
    """Generate a minimal MT5 M1 CSV with `n_days` days starting 2025-01-06."""
    lines = ["<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>"]
    base_date = date(2025, 1, 6)  # Monday

    price = 1.05000
    for day_offset in range(n_days):
        d = base_date.replace(day=base_date.day + day_offset)
        for hour in range(1):  # just 1 bar per day to keep it small
            dt_str = f"{d.year}.{d.month:02d}.{d.day:02d}"
            tm_str = f"{hour:02d}:00:00"
            h = price + 0.00050
            lo = price - 0.00050
            lines.append(
                f"{dt_str} {tm_str} {price:.5f} {h:.5f} {lo:.5f} {price:.5f} 100 0 8"
            )
            price += 0.00005
    return "\n".join(lines)


def _make_full_day_mt5(
    day: date, n_bars: int = 80, base_price: float = 1.05000
) -> pd.DataFrame:
    """Return a DataFrame with *n_bars* M1 bars for *day* starting at 00:00 UTC."""
    rows = []
    price = base_price
    for i in range(n_bars):
        dt = datetime(day.year, day.month, day.day, i // 60, i % 60, 0,
                      tzinfo=timezone.utc)
        spread_pts = 8
        rows.append(
            {
                "dt": pd.Timestamp(dt),
                "open": price,
                "high": price + 0.00050,
                "low": price - 0.00050,
                "close": price,
                "spread_points": spread_pts,
                "spread_price": spread_pts * POINT,
                "iso_year": dt.isocalendar().year,
                "iso_week": dt.isocalendar().week,
                "weekday": dt.weekday(),
                "time_of_day": dt.hour * 60 + dt.minute,
                "tradable_now": 1,
                "is_safe_week": 1,
                "has_breakout": 1,
            }
        )
        price += 0.00001
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Tests: news_utils
# ---------------------------------------------------------------------------


class TestNewsUtils:
    def test_filter_keeps_fomc(self) -> None:
        raw = pd.read_csv(io.StringIO(FXSTREET_SAMPLE))
        events = filter_news_events(raw)
        titles = events["title"].tolist()
        assert any("FOMC" in t for t in titles)

    def test_filter_keeps_nfp(self) -> None:
        raw = pd.read_csv(io.StringIO(FXSTREET_SAMPLE))
        events = filter_news_events(raw)
        titles = events["title"].tolist()
        assert any("Nonfarm" in t for t in titles)

    def test_filter_keeps_cpi(self) -> None:
        raw = pd.read_csv(io.StringIO(FXSTREET_SAMPLE))
        events = filter_news_events(raw)
        titles = events["title"].tolist()
        assert any("CPI" in t or "Consumer Price" in t for t in titles)

    def test_filter_drops_ism(self) -> None:
        raw = pd.read_csv(io.StringIO(FXSTREET_SAMPLE))
        events = filter_news_events(raw)
        titles = events["title"].tolist()
        assert not any("ISM" in t for t in titles)

    def test_filter_drops_adp(self) -> None:
        raw = pd.read_csv(io.StringIO(FXSTREET_SAMPLE))
        events = filter_news_events(raw)
        titles = events["title"].tolist()
        assert not any("ADP" in t for t in titles)

    def test_impact_normalised_to_high(self) -> None:
        raw = pd.read_csv(io.StringIO(FXSTREET_SAMPLE))
        events = filter_news_events(raw)
        assert (events["impact"] == "high").all()

    def test_unsafe_weeks_wed_thu(self) -> None:
        """FOMC (Wed 2025-01-08) and NFP (Wed 2025-02-05) should be unsafe."""
        raw = pd.read_csv(io.StringIO(FXSTREET_SAMPLE))
        events = filter_news_events(raw)
        unsafe = build_unsafe_weeks(events)
        assert len(unsafe) >= 1
        # 2025-01-08 is a Wednesday → week should be in unsafe
        dt = pd.Timestamp("2025-01-08", tz="UTC")
        iso_year = dt.isocalendar().year
        iso_week = dt.isocalendar().week
        assert ((unsafe["iso_year"] == iso_year) & (unsafe["iso_week"] == iso_week)).any()

    def test_unsafe_weeks_excludes_non_wed_thu(self) -> None:
        """Event on Tuesday (2025-01-07) should NOT create an unsafe week."""
        raw = pd.read_csv(io.StringIO(FXSTREET_SAMPLE))
        events = filter_news_events(raw)
        unsafe = build_unsafe_weeks(events)
        # 2025-01-07 is a Tuesday
        dt = pd.Timestamp("2025-01-07", tz="UTC")
        iso_year = dt.isocalendar().year
        iso_week = dt.isocalendar().week
        # The same ISO week also contains FOMC on Wed 01/08, so the week IS unsafe;
        # the point of this test is that a Tuesday-only event does not create unsafe.
        # We test with a CSV that only has a Tuesday event.
        only_tuesday = pd.DataFrame(
            [
                {
                    "Id": "x",
                    "Start": "01/07/2025 10:00:00",
                    "Name": "CPI y/y",
                    "Impact": "HIGH",
                    "Currency": "USD",
                }
            ]
        )
        ev = filter_news_events(only_tuesday)
        u = build_unsafe_weeks(ev)
        assert len(u) == 0


# ---------------------------------------------------------------------------
# Tests: market_loader
# ---------------------------------------------------------------------------


class TestMarketLoader:
    def test_load_parses_columns(self) -> None:
        from trading_env.data.market_loader import load_mt5_m1_csv

        csv_text = _make_mt5_csv(3)
        buf = io.StringIO(csv_text)
        df = load_mt5_m1_csv(buf)
        for col in ("dt", "open", "high", "low", "close", "spread_price",
                    "iso_year", "iso_week", "weekday", "time_of_day"):
            assert col in df.columns, f"Missing column: {col}"

    def test_spread_price_computed(self) -> None:
        from trading_env.data.market_loader import load_mt5_m1_csv

        csv_text = _make_mt5_csv(1)
        buf = io.StringIO(csv_text)
        df = load_mt5_m1_csv(buf)
        assert (df["spread_price"] > 0).all()
        # spread = 8 points → 8 * 0.00001 = 0.00008
        assert abs(df["spread_price"].iloc[0] - 8 * POINT) < 1e-9


# ---------------------------------------------------------------------------
# Tests: breakout gate
# ---------------------------------------------------------------------------


class TestBreakoutGate:
    def _make_week_df(self, tue_close_above_mon: bool) -> pd.DataFrame:
        """Craft a minimal DataFrame with 1 Monday and 1 Tuesday bar."""
        mon_high = 1.06000
        mon_low = 1.04000
        tue_close = 1.07000 if tue_close_above_mon else 1.05000
        rows = [
            {
                "dt": pd.Timestamp("2025-01-06 12:00:00", tz="UTC"),
                "open": 1.04500,
                "high": mon_high,
                "low": mon_low,
                "close": 1.05500,
                "iso_year": 2025,
                "iso_week": 2,
                "weekday": 0,  # Monday
            },
            {
                "dt": pd.Timestamp("2025-01-07 18:00:00", tz="UTC"),
                "open": 1.05500,
                "high": 1.07100 if tue_close_above_mon else 1.05800,
                "low": 1.05000,
                "close": tue_close,
                "iso_year": 2025,
                "iso_week": 2,
                "weekday": 1,  # Tuesday
            },
        ]
        return pd.DataFrame(rows)

    def test_breakout_detected(self) -> None:
        df = self._make_week_df(tue_close_above_mon=True)
        meta = compute_week_breakout(df)
        assert meta.iloc[0]["has_breakout"] == 1

    def test_no_breakout(self) -> None:
        df = self._make_week_df(tue_close_above_mon=False)
        meta = compute_week_breakout(df)
        assert meta.iloc[0]["has_breakout"] == 0

    def test_add_breakout_flag_merges(self) -> None:
        df = self._make_week_df(tue_close_above_mon=True)
        meta = compute_week_breakout(df)
        df2 = add_breakout_flag(df, week_meta=meta)
        assert "has_breakout" in df2.columns
        assert (df2["has_breakout"] == 1).all()


# ---------------------------------------------------------------------------
# Tests: tradable window gate
# ---------------------------------------------------------------------------


class TestTradableWindow:
    def _make_bars(
        self,
        weekday: int,
        time_of_day: int,
        is_safe: int = 1,
        has_bo: int = 1,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "weekday": weekday,
                    "time_of_day": time_of_day,
                    "is_safe_week": is_safe,
                    "has_breakout": has_bo,
                }
            ]
        )

    def test_tradable_on_wed_0900(self) -> None:
        df = self._make_bars(weekday=2, time_of_day=9 * 60)
        out = add_tradable_flag(df)
        assert out["tradable_now"].iloc[0] == 1

    def test_not_tradable_on_mon(self) -> None:
        df = self._make_bars(weekday=0, time_of_day=9 * 60)
        out = add_tradable_flag(df)
        assert out["tradable_now"].iloc[0] == 0

    def test_not_tradable_before_0800(self) -> None:
        df = self._make_bars(weekday=2, time_of_day=7 * 60 + 59)
        out = add_tradable_flag(df)
        assert out["tradable_now"].iloc[0] == 0

    def test_not_tradable_after_2200(self) -> None:
        df = self._make_bars(weekday=3, time_of_day=22 * 60 + 1)
        out = add_tradable_flag(df)
        assert out["tradable_now"].iloc[0] == 0

    def test_not_tradable_unsafe_week(self) -> None:
        df = self._make_bars(weekday=2, time_of_day=10 * 60, is_safe=0)
        out = add_tradable_flag(df)
        assert out["tradable_now"].iloc[0] == 0

    def test_not_tradable_no_breakout(self) -> None:
        df = self._make_bars(weekday=2, time_of_day=10 * 60, has_bo=0)
        out = add_tradable_flag(df)
        assert out["tradable_now"].iloc[0] == 0

    def test_tradable_at_exact_0800(self) -> None:
        df = self._make_bars(weekday=2, time_of_day=TRADABLE_START_MIN)
        out = add_tradable_flag(df)
        assert out["tradable_now"].iloc[0] == 1

    def test_tradable_at_exact_2200(self) -> None:
        df = self._make_bars(weekday=3, time_of_day=TRADABLE_END_MIN)
        out = add_tradable_flag(df)
        assert out["tradable_now"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Tests: position sizing
# ---------------------------------------------------------------------------


class TestPositionSizing:
    def test_basic_sizing(self) -> None:
        equity = 10_000.0
        entry = 1.05000
        sl = 1.04800  # 20 pips away
        lots = compute_lots(equity, entry, sl)
        # 1% of 10k = 100 USD risk; 20 pips * $10/pip/lot = $200/lot → 0.5 lot
        assert lots == pytest.approx(0.50, abs=0.01)

    def test_rejects_zero_stop(self) -> None:
        assert compute_lots(10_000.0, 1.05000, 1.05000) == 0.0

    def test_minimum_lot(self) -> None:
        # Very tiny equity → should reject (below min lot after rounding)
        result = compute_lots(1.0, 1.05000, 1.04800)
        assert result == 0.0 or result >= MIN_LOT


# ---------------------------------------------------------------------------
# Tests: EURUSDTradingEnv
# ---------------------------------------------------------------------------


class TestEURUSDTradingEnv:
    def _make_env(
        self, n_bars: int = 80, day: date | None = None
    ) -> EURUSDTradingEnv:
        if day is None:
            day = date(2025, 1, 8)  # Wednesday
        df = _make_full_day_mt5(day, n_bars=n_bars)
        env = EURUSDTradingEnv(df, initial_equity=10_000.0)
        return env

    def test_reset_returns_valid_obs(self) -> None:
        env = self._make_env()
        obs, info = env.reset(options={"day_date": date(2025, 1, 8)})
        assert obs.shape == env.observation_space.shape
        assert "equity" in info

    def test_hold_step(self) -> None:
        env = self._make_env()
        env.reset(options={"day_date": date(2025, 1, 8)})
        obs, reward, terminated, truncated, info = env.step(HOLD)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert not truncated

    def test_day_terminates(self) -> None:
        env = self._make_env(n_bars=70)
        env.reset(options={"day_date": date(2025, 1, 8)})
        terminated = False
        for _ in range(200):
            _obs, _r, terminated, _trunc, _info = env.step(HOLD)
            if terminated:
                break
        assert terminated, "Episode should terminate at end of day"

    def test_gating_forces_hold(self) -> None:
        """When tradable_now=0, any non-HOLD action is silently forced to HOLD."""
        day = date(2025, 1, 8)
        df = _make_full_day_mt5(day, n_bars=80)
        # Force all bars to be non-tradable
        df["tradable_now"] = 0
        env = EURUSDTradingEnv(df, initial_equity=10_000.0)
        env.reset(options={"day_date": day})
        # Try to open a long — position should remain 0
        for _ in range(5):
            env.step(OPEN_LONG)
        assert env._pos == 0, "Position should be 0 when gated"

    def test_open_long_then_close(self) -> None:
        env = self._make_env()
        env.reset(options={"day_date": date(2025, 1, 8)})
        env.step(OPEN_LONG)
        assert env._pos == 1
        env.step(CLOSE)
        assert env._pos == 0

    def test_open_short_then_close(self) -> None:
        env = self._make_env()
        env.reset(options={"day_date": date(2025, 1, 8)})
        env.step(OPEN_SHORT)
        assert env._pos == -1
        env.step(CLOSE)
        assert env._pos == 0

    def test_invalid_open_when_in_position(self) -> None:
        env = self._make_env()
        env.reset(options={"day_date": date(2025, 1, 8)})
        env.step(OPEN_LONG)
        _obs, reward, _term, _trunc, _info = env.step(OPEN_LONG)
        # Should receive a penalty for invalid action
        assert reward < 0

    def test_protect_sets_break_even(self) -> None:
        env = self._make_env(n_bars=80)
        env.reset(options={"day_date": date(2025, 1, 8)})
        env.step(OPEN_LONG)
        # Simulate that tp_state >= 1 by forcing it
        env._tp_state = 1
        env.step(PROTECT)
        assert env._break_even_set

    def test_early_stop_gain(self) -> None:
        """Episode ends when equity reaches +2% of start-of-day."""
        day = date(2025, 1, 8)
        df = _make_full_day_mt5(day, n_bars=80)
        env = EURUSDTradingEnv(df, initial_equity=10_000.0)
        env.reset(options={"day_date": day})
        # Manually inflate equity past 2% threshold
        env._equity = env._start_equity * (1 + EPISODE_EQUITY_GAIN_LIMIT + 0.001)
        _obs, _r, terminated, _trunc, _info = env.step(HOLD)
        assert terminated, "Episode should terminate on +2% equity gain"

    def test_early_stop_loss(self) -> None:
        """Episode ends when equity reaches -2% of start-of-day."""
        day = date(2025, 1, 8)
        df = _make_full_day_mt5(day, n_bars=80)
        env = EURUSDTradingEnv(df, initial_equity=10_000.0)
        env.reset(options={"day_date": day})
        # Manually deflate equity past -2% threshold
        env._equity = env._start_equity * (1 + EPISODE_EQUITY_LOSS_LIMIT - 0.001)
        _obs, _r, terminated, _trunc, _info = env.step(HOLD)
        assert terminated, "Episode should terminate on -2% equity loss"

    def test_stopout_closes_position(self) -> None:
        """When bar low <= SL the position is closed automatically."""
        day = date(2025, 1, 8)
        df = _make_full_day_mt5(day, n_bars=80)
        env = EURUSDTradingEnv(df, initial_equity=10_000.0)
        env.reset(options={"day_date": day})
        env.step(OPEN_LONG)
        assert env._pos == 1
        # Force SL above current bar high to guarantee stop-out
        env._sl_price = env.df.iloc[env._bar_idx]["high"] + 1.0
        env._pos = 1  # keep long
        # step — stop-out should close position
        env.step(HOLD)
        assert env._pos == 0, "Position should be closed after stop-out"

    def test_obs_shape_consistent(self) -> None:
        env = self._make_env()
        obs, _ = env.reset(options={"day_date": date(2025, 1, 8)})
        for _ in range(10):
            obs, _, terminated, _, _ = env.step(HOLD)
            assert obs.shape == env.observation_space.shape
            if terminated:
                break

    def test_manage_tp_partial_close(self) -> None:
        """MANAGE_TP at tp_state=0 should close 25% lots if TP1 is reached."""
        day = date(2025, 1, 8)
        df = _make_full_day_mt5(day, n_bars=80)
        env = EURUSDTradingEnv(df, initial_equity=10_000.0)
        env.reset(options={"day_date": day})
        env.step(OPEN_LONG)
        initial_lots = env._lots_initial
        assert initial_lots > 0
        # Force bar high above TP1
        env._tp1_price = env.df.iloc[env._bar_idx]["low"]  # guaranteed reached
        env.step(MANAGE_TP)
        # Should have closed 25% of initial lots
        expected_remaining = round(initial_lots - initial_lots * 0.25, 10)
        assert abs(env._lots_remaining - expected_remaining) < 1e-6 or env._tp_state == 1
