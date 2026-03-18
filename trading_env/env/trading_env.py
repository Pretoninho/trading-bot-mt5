"""
trading_env.py
==============
Gymnasium-compatible EURUSD M1 trading environment.

Episode
-------
One UTC trading session per day: **08:00–22:00 UTC** (inclusive).
Terminates early when equity reaches +2 % or -2 % relative to
session-start equity.

Actions (discrete)
------------------
0  HOLD          — do nothing.
1  OPEN_LONG     — open a long position (if no position open).
2  OPEN_SHORT    — open a short position (if no position open).
3  CLOSE         — close the current position at market.
4  PROTECT       — move SL to break-even (and trailing once tp_state == 2).
5  MANAGE_TP     — close partial lots at TP1 / TP2 if reached.

Gate
----
When ``tradable_now == 0``, any action other than HOLD is silently forced to
HOLD (no penalty for the agent; it simply cannot act).

Execution pricing
-----------------
``mid = close``,
``ask = mid + spread_price / 2``,
``bid = mid - spread_price / 2``.
Long entries fill at *ask*; exits at *bid*.
Short entries fill at *bid*; exits at *ask*.

Stop-loss (ATR-based)
---------------------
On entry the stop-loss is placed at ``entry ± 2 * ATR14``.
ATR14 is computed from the last 14 completed bars preceding entry.

Position sizing
---------------
1 % equity risk per trade (see :mod:`~trading_env.utils.position_sizing`).

TP management (Option B — action-driven)
-----------------------------------------
* TP1 at ``entry + 1R`` (long) / ``entry - 1R`` (short).
* TP2 at ``entry + 2R`` / ``entry - 2R``.
* ``MANAGE_TP`` checks the current bar's High/Low:
  - ``tp_state 0→1``: close 25 % of initial lots at TP1 if bar reaches it.
  - ``tp_state 1→2``: close 25 % of initial lots at TP2 if bar reaches it.
* Remaining ~50 % is the *runner*.

PROTECT behaviour
-----------------
* If ``tp_state >= 1`` OR ``move_in_favor >= 1R`` (unrealised), and break-even
  not yet set → move SL to entry ± be_buffer (default 0).
* If ``tp_state == 2`` → trailing: SL updated to
  ``max(SL, close - 2*ATR14)`` for longs / ``min(SL, close + 2*ATR14)`` for
  shorts.

Stop-out priority
-----------------
Before each step action, if the bar hits the SL, the entire remaining
position is closed at the SL price (ignoring the agent's requested action).

Observation space
-----------------
* 64-bar window of normalised OHLC returns + raw spread + ATR14 (shape 64×6).
* Scalar features appended: ``is_tradable_now, is_safe_week, has_breakout,
  tp_state_norm, break_even_set, runner_trailing_active, r_multiple_unrealized``
  (total length 64*6 + 7 = 391).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from trading_env.utils.position_sizing import compute_lots, PIP_VALUE_PER_LOT, PIP_SIZE

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

HOLD = 0
OPEN_LONG = 1
OPEN_SHORT = 2
CLOSE = 3
PROTECT = 4
MANAGE_TP = 5

N_ACTIONS = 6

# ---------------------------------------------------------------------------
# Episode / risk constants
# ---------------------------------------------------------------------------

EPISODE_EQUITY_GAIN_LIMIT = 0.02   # +2 % → episode ends
EPISODE_EQUITY_LOSS_LIMIT = -0.02  # -2 % → episode ends
EPISODE_START_HOUR = 8             # episode begins at 08:00 UTC
EPISODE_END_HOUR = 22              # episode ends at 22:00 UTC (inclusive)
INVALID_ACTION_PENALTY = -1e-4
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 2.0
TP1_R = 1.0
TP2_R = 2.0
PARTIAL_CLOSE_FRACTION = 0.25  # each TP closes 25 % of initial lots

WINDOW = 64  # observation bar-window size
N_BAR_FEATURES = 6  # open_ret, high_ret, low_ret, close_ret, spread, atr14_norm
SCALAR_FEATURES = 7

OBS_SIZE = WINDOW * N_BAR_FEATURES + SCALAR_FEATURES


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class EURUSDTradingEnv(gym.Env):
    """EURUSD M1 Gymnasium trading environment.

    Parameters
    ----------
    df:
        Market DataFrame produced by the data-loading pipeline (must contain
        columns ``dt, open, high, low, close, spread_price, tradable_now,
        is_safe_week, has_breakout, iso_year, iso_week, weekday``).
    initial_equity:
        Starting account equity in USD (default 10 000).
    be_buffer:
        Break-even buffer in price units (default 0 → exact break-even).
    render_mode:
        Gymnasium render mode (only ``"human"`` is supported and simply prints
        step info).
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_equity: float = 10_000.0,
        be_buffer: float = 0.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_equity = initial_equity
        self.be_buffer = be_buffer
        self.render_mode = render_mode

        # Precompute ATR14 for the whole dataset
        self._atr = self._compute_atr14(self.df)

        # Action space
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Observation space (flattened)
        obs_low = np.full(OBS_SIZE, -np.inf, dtype=np.float32)
        obs_high = np.full(OBS_SIZE, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # State — initialised in reset()
        self._day_indices: list[int] = []
        self._step_idx: int = 0          # index within the current day
        self._bar_idx: int = 0           # global bar index in df
        self._equity: float = initial_equity
        self._start_equity: float = initial_equity

        # Position state
        self._pos: int = 0               # +1 long, -1 short, 0 flat
        self._entry_price: float = 0.0
        self._sl_price: float = 0.0
        self._tp1_price: float = 0.0
        self._tp2_price: float = 0.0
        self._lots_initial: float = 0.0
        self._lots_remaining: float = 0.0
        self._tp_state: int = 0          # 0,1,2
        self._break_even_set: bool = False
        self._atr_at_entry: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Choose a day at random (or from options["day_date"])
        all_days = sorted(self.df["dt"].dt.date.unique())
        if options and "day_date" in options:
            day = options["day_date"]
        else:
            idx = self.np_random.integers(0, len(all_days))
            day = all_days[idx]

        # Filter to bars within the episode window: 08:00–22:00 UTC
        episode_start_min = EPISODE_START_HOUR * 60  # 480
        episode_end_min = EPISODE_END_HOUR * 60       # 1320

        day_mask = self.df["dt"].dt.date == day

        if "time_of_day" in self.df.columns:
            # Preferred: pre-computed column added by market_loader or test helpers
            time_of_day = self.df["time_of_day"]
        else:
            # Fallback: compute on-the-fly for DataFrames built without the loader
            time_of_day = self.df["dt"].dt.hour * 60 + self.df["dt"].dt.minute

        time_mask = time_of_day.between(episode_start_min, episode_end_min)

        self._day_indices = self.df.index[day_mask & time_mask].tolist()

        if len(self._day_indices) < WINDOW + 1:
            # Not enough history — fall back to a full day if possible
            self._day_indices = []

        self._step_idx = 0
        if self._day_indices:
            self._bar_idx = self._day_indices[0]
        else:
            self._bar_idx = WINDOW  # safe fallback

        self._equity = self.initial_equity
        self._start_equity = self.initial_equity
        self._reset_position()

        obs = self._get_obs()
        episode_start_dt = (
            str(self.df.iloc[self._day_indices[0]]["dt"]) if self._day_indices else ""
        )
        episode_end_dt = (
            str(self.df.iloc[self._day_indices[-1]]["dt"]) if self._day_indices else ""
        )
        info = {
            "day": str(day),
            "equity": self._equity,
            "episode_start_dt": episode_start_dt,
            "episode_end_dt": episode_end_dt,
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one minute bar.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        prev_equity = self._equity
        penalty = 0.0

        bar = self.df.iloc[self._bar_idx]

        # ------------------------------------------------------------------
        # 1. Stop-out check (before agent action)
        # ------------------------------------------------------------------
        stopped_out = False
        if self._pos != 0:
            stopped_out = self._check_stopout(bar)

        # ------------------------------------------------------------------
        # 2. Apply agent action (only if not stopped out this bar)
        # ------------------------------------------------------------------
        if not stopped_out:
            tradable = (
                int(bar["tradable_now"]) == 1
                if "tradable_now" in bar.index
                else True
            )

            # Force HOLD when not in tradable window
            if not tradable and action != HOLD:
                action = HOLD

            penalty = self._apply_action(action, bar)

        # ------------------------------------------------------------------
        # 3. Mark-to-market equity
        # ------------------------------------------------------------------
        self._mark_equity(bar)

        # ------------------------------------------------------------------
        # 4. Reward
        # ------------------------------------------------------------------
        equity_return = (self._equity - prev_equity) / prev_equity
        reward = float(equity_return + penalty)

        # ------------------------------------------------------------------
        # 5. Advance indices
        # ------------------------------------------------------------------
        self._step_idx += 1
        if self._step_idx < len(self._day_indices):
            self._bar_idx = self._day_indices[self._step_idx]
        else:
            self._bar_idx = min(self._bar_idx + 1, len(self.df) - 1)

        # ------------------------------------------------------------------
        # 6. Termination
        # ------------------------------------------------------------------
        day_done = self._step_idx >= len(self._day_indices)
        pnl_frac = (self._equity - self._start_equity) / self._start_equity
        equity_limit_hit = (
            pnl_frac >= EPISODE_EQUITY_GAIN_LIMIT
            or pnl_frac <= EPISODE_EQUITY_LOSS_LIMIT
        )
        terminated = day_done or equity_limit_hit
        truncated = False

        obs = self._get_obs()
        info = {
            "equity": self._equity,
            "pos": self._pos,
            "tp_state": self._tp_state,
            "break_even_set": self._break_even_set,
            "pnl_frac": pnl_frac,
        }

        if self.render_mode == "human":
            self._render_step(bar, action, reward, info)

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass  # rendering is done inline in step()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_position(self) -> None:
        self._pos = 0
        self._entry_price = 0.0
        self._sl_price = 0.0
        self._tp1_price = 0.0
        self._tp2_price = 0.0
        self._lots_initial = 0.0
        self._lots_remaining = 0.0
        self._tp_state = 0
        self._break_even_set = False
        self._atr_at_entry = 0.0

    # ------------------------------------------------------------------
    # Pricing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bar_prices(bar: pd.Series) -> tuple[float, float, float]:
        """Return (mid, ask, bid) from a bar."""
        mid = float(bar["close"])
        half_spread = float(bar["spread_price"]) / 2.0
        return mid, mid + half_spread, mid - half_spread

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr14(df: pd.DataFrame) -> np.ndarray:
        """Vectorised ATR14 (Wilder smoothing) for the full DataFrame."""
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        n = len(df)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr = np.empty(n)
        atr[: ATR_PERIOD] = np.nan
        if n >= ATR_PERIOD:
            atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
            for i in range(ATR_PERIOD, n):
                atr[i] = (atr[i - 1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD

        return atr

    def _atr_at(self, global_idx: int) -> float:
        """Return ATR14 at *global_idx*; falls back to TR if ATR is NaN."""
        val = self._atr[global_idx] if global_idx < len(self._atr) else np.nan
        if np.isnan(val):
            bar = self.df.iloc[global_idx]
            return float(bar["high"]) - float(bar["low"])
        return float(val)

    # ------------------------------------------------------------------
    # Open position
    # ------------------------------------------------------------------

    def _open_position(self, direction: int, bar: pd.Series) -> float:
        """Open a new position.  Returns invalid-action penalty (0 if OK)."""
        if self._pos != 0:
            return INVALID_ACTION_PENALTY  # already in a trade

        _mid, ask, bid = self._bar_prices(bar)
        entry = ask if direction == 1 else bid

        atr = self._atr_at(self._bar_idx)
        sl_dist = ATR_SL_MULTIPLIER * atr

        if sl_dist <= 0:
            return INVALID_ACTION_PENALTY

        sl = entry - sl_dist if direction == 1 else entry + sl_dist

        lots = compute_lots(self._equity, entry, sl)
        if lots <= 0:
            return INVALID_ACTION_PENALTY

        r_dist = abs(entry - sl)
        tp1 = entry + TP1_R * r_dist if direction == 1 else entry - TP1_R * r_dist
        tp2 = entry + TP2_R * r_dist if direction == 1 else entry - TP2_R * r_dist

        self._pos = direction
        self._entry_price = entry
        self._sl_price = sl
        self._tp1_price = tp1
        self._tp2_price = tp2
        self._lots_initial = lots
        self._lots_remaining = lots
        self._tp_state = 0
        self._break_even_set = False
        self._atr_at_entry = atr
        return 0.0

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    def _close_position(self, close_lots: float, exit_price: float) -> float:
        """Close *close_lots* of the current position at *exit_price*.

        Returns realised PnL in USD.
        """
        close_lots = min(close_lots, self._lots_remaining)
        if close_lots <= 0:
            return 0.0

        pip_pnl = (
            (exit_price - self._entry_price)
            if self._pos == 1
            else (self._entry_price - exit_price)
        )
        pips = pip_pnl / PIP_SIZE
        pnl = pips * PIP_VALUE_PER_LOT * close_lots

        self._equity += pnl
        self._lots_remaining -= close_lots
        self._lots_remaining = max(0.0, round(self._lots_remaining, 10))

        if self._lots_remaining == 0:
            self._reset_position()

        return pnl

    # ------------------------------------------------------------------
    # Stop-out check
    # ------------------------------------------------------------------

    def _check_stopout(self, bar: pd.Series) -> bool:
        """If SL is hit by the current bar, close all remaining lots at SL."""
        bar_low = float(bar["low"])
        bar_high = float(bar["high"])

        hit = (self._pos == 1 and bar_low <= self._sl_price) or (
            self._pos == -1 and bar_high >= self._sl_price
        )
        if hit:
            self._close_position(self._lots_remaining, self._sl_price)
        return hit

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def _mark_equity(self, bar: pd.Series) -> None:
        """No-op: equity is updated only on realised closes (SL/TP/CLOSE).

        The reward signal is based purely on realised P&L so that the agent
        does not receive mark-to-market floating equity changes between bars.
        """
        pass

    # ------------------------------------------------------------------
    # Action handler
    # ------------------------------------------------------------------

    def _apply_action(self, action: int, bar: pd.Series) -> float:
        """Execute action and return penalty (0 if valid)."""
        penalty = 0.0

        if action == HOLD:
            pass  # nothing to do

        elif action == OPEN_LONG:
            if self._pos != 0:
                penalty = INVALID_ACTION_PENALTY
            else:
                penalty = self._open_position(1, bar)

        elif action == OPEN_SHORT:
            if self._pos != 0:
                penalty = INVALID_ACTION_PENALTY
            else:
                penalty = self._open_position(-1, bar)

        elif action == CLOSE:
            if self._pos == 0:
                penalty = INVALID_ACTION_PENALTY
            else:
                _mid, ask, bid = self._bar_prices(bar)
                exit_price = bid if self._pos == 1 else ask
                self._close_position(self._lots_remaining, exit_price)

        elif action == PROTECT:
            penalty = self._apply_protect(bar)

        elif action == MANAGE_TP:
            penalty = self._apply_manage_tp(bar)

        return penalty

    # ------------------------------------------------------------------
    # PROTECT logic
    # ------------------------------------------------------------------

    def _apply_protect(self, bar: pd.Series) -> float:
        """Move SL to break-even (and trail if tp_state == 2)."""
        if self._pos == 0:
            return INVALID_ACTION_PENALTY

        _mid, _ask, _bid = self._bar_prices(bar)
        close_price = float(bar["close"])

        r_dist = abs(self._entry_price - self._sl_price)
        move_in_favor = (
            (close_price - self._entry_price)
            if self._pos == 1
            else (self._entry_price - close_price)
        )

        eligible = self._tp_state >= 1 or move_in_favor >= r_dist

        if eligible and not self._break_even_set:
            if self._pos == 1:
                new_sl = self._entry_price + self.be_buffer
            else:
                new_sl = self._entry_price - self.be_buffer

            # Only move SL in favour direction
            if self._pos == 1:
                self._sl_price = max(self._sl_price, new_sl)
            else:
                self._sl_price = min(self._sl_price, new_sl)
            self._break_even_set = True

        # Trailing stop when runner is active (tp_state == 2)
        if self._tp_state == 2:
            atr_now = self._atr_at(self._bar_idx)
            if self._pos == 1:
                trail_sl = close_price - ATR_SL_MULTIPLIER * atr_now
                self._sl_price = max(self._sl_price, trail_sl)
            else:
                trail_sl = close_price + ATR_SL_MULTIPLIER * atr_now
                self._sl_price = min(self._sl_price, trail_sl)

        return 0.0

    # ------------------------------------------------------------------
    # MANAGE_TP logic
    # ------------------------------------------------------------------

    def _apply_manage_tp(self, bar: pd.Series) -> float:
        """Close partial lots at TP1/TP2 if the bar reaches them."""
        if self._pos == 0:
            return INVALID_ACTION_PENALTY

        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        penalty = 0.0

        if self._tp_state == 0:
            tp1_hit = (self._pos == 1 and bar_high >= self._tp1_price) or (
                self._pos == -1 and bar_low <= self._tp1_price
            )
            if tp1_hit:
                lots_to_close = round(
                    self._lots_initial * PARTIAL_CLOSE_FRACTION, 10
                )
                lots_to_close = max(0.0, min(lots_to_close, self._lots_remaining))
                self._close_position(lots_to_close, self._tp1_price)
                self._tp_state = 1
            else:
                penalty = INVALID_ACTION_PENALTY  # TP not reached yet

        elif self._tp_state == 1:
            tp2_hit = (self._pos == 1 and bar_high >= self._tp2_price) or (
                self._pos == -1 and bar_low <= self._tp2_price
            )
            if tp2_hit:
                lots_to_close = round(
                    self._lots_initial * PARTIAL_CLOSE_FRACTION, 10
                )
                lots_to_close = max(0.0, min(lots_to_close, self._lots_remaining))
                self._close_position(lots_to_close, self._tp2_price)
                self._tp_state = 2
            else:
                penalty = INVALID_ACTION_PENALTY  # TP not reached yet

        else:
            # tp_state == 2: both TPs already taken, runner remaining
            penalty = INVALID_ACTION_PENALTY

        return penalty

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the flat observation vector."""
        # --- 64-bar OHLC window (right-aligned; zero-padded on the left) ---
        end_idx = self._bar_idx
        # Include the current bar: [end_idx - WINDOW + 1 … end_idx]
        start_idx = max(0, end_idx - WINDOW + 1)
        window_df = self.df.iloc[start_idx : end_idx + 1]
        n_rows = len(window_df)

        bar_features = np.zeros((WINDOW, N_BAR_FEATURES), dtype=np.float32)

        # Place bars right-aligned in the WINDOW buffer
        offset = WINDOW - n_rows  # left-padding with zeros
        for local_i in range(n_rows):
            win_i = offset + local_i
            idx_global = start_idx + local_i
            row = window_df.iloc[local_i]

            if local_i == 0:
                # No previous bar available → use intra-bar deltas from open
                o = float(row["open"]) + 1e-10
                bar_features[win_i, 0] = 0.0
                bar_features[win_i, 1] = (float(row["high"]) - o) / o
                bar_features[win_i, 2] = (float(row["low"]) - o) / o
                bar_features[win_i, 3] = (float(row["close"]) - o) / o
            else:
                prev_close = float(window_df.iloc[local_i - 1]["close"]) + 1e-10
                bar_features[win_i, 0] = (float(row["open"]) - prev_close) / prev_close
                bar_features[win_i, 1] = (float(row["high"]) - prev_close) / prev_close
                bar_features[win_i, 2] = (float(row["low"]) - prev_close) / prev_close
                bar_features[win_i, 3] = (float(row["close"]) - prev_close) / prev_close

            sp = row["spread_price"] if "spread_price" in row.index else 0.0
            bar_features[win_i, 4] = float(sp)
            atr_val = self._atr[idx_global] if idx_global < len(self._atr) else 0.0
            bar_features[win_i, 5] = 0.0 if np.isnan(atr_val) else float(atr_val)

        bar_flat = bar_features.flatten()

        # --- Scalar features ---
        cur_bar = self.df.iloc[self._bar_idx]

        def _get_bar(col: str, default: float = 1.0) -> float:
            return float(cur_bar[col]) if col in cur_bar.index else default

        is_tradable = _get_bar("tradable_now", 1.0)
        is_safe = _get_bar("is_safe_week", 1.0)
        has_bo = _get_bar("has_breakout", 1.0)
        tp_state_norm = self._tp_state / 2.0
        be_set = float(self._break_even_set)
        runner_trailing = float(self._tp_state == 2 and self._break_even_set)

        if self._pos != 0 and self._atr_at_entry > 0:
            r_dist = ATR_SL_MULTIPLIER * self._atr_at_entry
            mid_now = float(cur_bar["close"])
            move = (
                (mid_now - self._entry_price)
                if self._pos == 1
                else (self._entry_price - mid_now)
            )
            r_multiple = move / (r_dist + 1e-10)
        else:
            r_multiple = 0.0

        scalars = np.array(
            [
                is_tradable,
                is_safe,
                has_bo,
                tp_state_norm,
                be_set,
                runner_trailing,
                r_multiple,
            ],
            dtype=np.float32,
        )

        return np.concatenate([bar_flat, scalars])

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def _render_step(
        self, bar: pd.Series, action: int, reward: float, info: dict
    ) -> None:
        action_names = ["HOLD", "OPEN_LONG", "OPEN_SHORT", "CLOSE", "PROTECT", "MANAGE_TP"]
        name = action_names[action] if 0 <= action < N_ACTIONS else str(action)
        dt_str = str(bar["dt"]) if "dt" in bar.index else ""
        print(
            f"[{dt_str}] action={name} reward={reward:.6f} "
            f"equity={info['equity']:.2f} pos={info['pos']} "
            f"tp_state={info['tp_state']} be={info['break_even_set']}"
        )
