"""
reward_functions.py
===================
Advanced reward function components for EURUSD trading MDP.

Combines multiple reward signals:
  - Equity return (base)
  - Sharpe ratio component (risk-adjusted returns)
  - Win-rate & streak bonuses
  - Trade duration reward
  - Drawdown penalty
  - Opportunity cost (holding penalty)
  - Invalid action penalty

All components are configurable via RewardConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Configuration for multi-component reward function."""

    # Base component
    equity_scale: float = 1.0

    # Sharpe Ratio component (window-based risk-adjusted returns)
    use_sharpe: bool = True
    sharpe_scale: float = 0.1
    sharpe_window: int = 100  # look-back for Sharpe calculation

    # Win-rate & streak bonus
    use_streak: bool = True
    streak_scale: float = 0.05
    streak_min_length: int = 3  # minimum streak length for bonus

    # Trade duration reward (encourages efficient trading)
    use_duration: bool = True
    duration_scale: float = 0.02
    duration_optimal_bars: int = 60  # optimal trade length (~1 hour for M1)

    # Drawdown penalty
    use_drawdown: bool = True
    drawdown_scale: float = 0.1
    drawdown_window: int = 500  # look-back for max equity

    # Opportunity cost (holding penalty)
    use_opportunity_cost: bool = True
    opportunity_scale: float = 0.01
    holding_threshold_bars: int = 120  # penalty if position held > 2 hours

    # Invalid action penalty
    use_invalid_penalty: bool = True
    invalid_penalty: float = -1e-4

    # Clip final reward
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0

    def to_dict(self) -> dict:
        """Convert config to dictionary for saving."""
        return {
            "equity_scale": self.equity_scale,
            "use_sharpe": self.use_sharpe,
            "sharpe_scale": self.sharpe_scale,
            "sharpe_window": self.sharpe_window,
            "use_streak": self.use_streak,
            "streak_scale": self.streak_scale,
            "streak_min_length": self.streak_min_length,
            "use_duration": self.use_duration,
            "duration_scale": self.duration_scale,
            "duration_optimal_bars": self.duration_optimal_bars,
            "use_drawdown": self.use_drawdown,
            "drawdown_scale": self.drawdown_scale,
            "drawdown_window": self.drawdown_window,
            "use_opportunity_cost": self.use_opportunity_cost,
            "opportunity_scale": self.opportunity_scale,
            "holding_threshold_bars": self.holding_threshold_bars,
            "use_invalid_penalty": self.use_invalid_penalty,
            "invalid_penalty": self.invalid_penalty,
            "reward_clip_min": self.reward_clip_min,
            "reward_clip_max": self.reward_clip_max,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> RewardConfig:
        """Create config from dictionary."""
        return cls(**config_dict)


# ---------------------------------------------------------------------------
# Component Calculators
# ---------------------------------------------------------------------------


class RewardComponents:
    """Calculates individual reward components."""

    @staticmethod
    def equity_return(prev_equity: float, curr_equity: float, scale: float = 1.0) -> float:
        """Base equity return component.

        Parameters
        ----------
        prev_equity : float
            Previous step equity
        curr_equity : float
            Current step equity
        scale : float
            Scaling factor

        Returns
        -------
        float
            Equity return component
        """
        if prev_equity <= 0:
            return 0.0
        return scale * (curr_equity - prev_equity) / prev_equity

    @staticmethod
    def sharpe_ratio_component(
        equity_history: np.ndarray,
        window: int = 100,
        scale: float = 0.1,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Risk-adjusted return component (Sharpe ratio-based).

        Parameters
        ----------
        equity_history : np.ndarray
            Historical equity values
        window : int
            Lookback window for Sharpe calculation
        scale : float
            Scaling factor
        risk_free_rate : float
            Risk-free rate (default 0 for trading)

        Returns
        -------
        float
            Sharpe ratio bonus component (0 if window too small)
        """
        if len(equity_history) < window + 1:
            return 0.0

        window_equity = equity_history[-window:]
        returns = np.diff(window_equity) / (window_equity[:-1] + 1e-10)

        if len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return < 1e-10:
            return 0.0

        sharpe = (avg_return - risk_free_rate) / std_return
        # Normalize Sharpe to reasonable range; cap at 2.0 to prevent overflow
        sharpe_bonus = scale * np.tanh(sharpe / 2.0)

        return float(sharpe_bonus)

    @staticmethod
    def win_rate_streak_bonus(
        pnl_history: list[float],
        streak_min: int = 3,
        scale: float = 0.05,
    ) -> float:
        """Bonus for winning streaks.

        Parameters
        ----------
        pnl_history : list[float]
            Recent P&L from closed trades
        streak_min : int
            Minimum streak length for bonus
        scale : float
            Scaling factor

        Returns
        -------
        float
            Streak bonus component
        """
        if len(pnl_history) < streak_min:
            return 0.0

        # Count consecutive winning trades at the end
        streak_length = 0
        for pnl in reversed(pnl_history):
            if pnl > 0:
                streak_length += 1
            else:
                break

        if streak_length >= streak_min:
            # Exponential bonus: 3-streak = 0.05x, 5-streak = 0.1x, etc.
            bonus = scale * (1.0 - np.exp(-(streak_length - streak_min) / 2.0))
            return float(bonus)

        return 0.0

    @staticmethod
    def trade_duration_reward(
        bars_in_trade: int,
        optimal_bars: int = 60,
        scale: float = 0.02,
    ) -> float:
        """Reward for trades of optimal duration.

        Encourages neither scalping nor holding too long.

        Parameters
        ----------
        bars_in_trade : int
            Number of bars the current/recent trade lasted
        optimal_bars : int
            Optimal trade duration in bars
        scale : float
            Scaling factor

        Returns
        -------
        float
            Duration reward component
        """
        if bars_in_trade <= 0:
            return 0.0

        # Gaussian reward centered at optimal_bars
        ratio = bars_in_trade / optimal_bars
        duration_bonus = scale * np.exp(-0.5 * ((ratio - 1.0) ** 2))

        return float(duration_bonus)

    @staticmethod
    def drawdown_penalty(
        equity_history: np.ndarray,
        window: int = 500,
        scale: float = 0.1,
    ) -> float:
        """Penalty during drawdown periods.

        Parameters
        ----------
        equity_history : np.ndarray
            Historical equity values
        window : int
            Lookback window for max equity
        scale : float
            Scaling factor

        Returns
        -------
        float
            Drawdown penalty (negative)
        """
        if len(equity_history) < 2:
            return 0.0

        window_equity = equity_history[-window:] if len(equity_history) >= window else equity_history
        max_equity = np.max(window_equity)
        curr_equity = equity_history[-1]

        if max_equity <= 0:
            return 0.0

        drawdown_pct = (max_equity - curr_equity) / max_equity

        if drawdown_pct > 0:
            # Penalty increases quadratically with drawdown
            penalty = -scale * (drawdown_pct ** 2)
            return float(penalty)

        return 0.0

    @staticmethod
    def opportunity_cost_penalty(
        bars_held: int,
        threshold_bars: int = 120,
        scale: float = 0.01,
    ) -> float:
        """Penalty for holding position too long without closing.

        Parameters
        ----------
        bars_held : int
            Number of bars current position has been held
        threshold_bars : int
            Threshold for penalty application
        scale : float
            Scaling factor

        Returns
        -------
        float
            Opportunity cost penalty (negative)
        """
        if bars_held <= threshold_bars:
            return 0.0

        # Linear penalty after threshold
        excess_bars = bars_held - threshold_bars
        penalty = -scale * (excess_bars / threshold_bars)

        return float(penalty)

    @staticmethod
    def invalid_action_penalty(is_invalid: bool, penalty: float = -1e-4) -> float:
        """Penalty for invalid actions.

        Parameters
        ----------
        is_invalid : bool
            Whether action was invalid
        penalty : float
            Penalty value

        Returns
        -------
        float
            Invalid action penalty
        """
        return float(penalty if is_invalid else 0.0)


# ---------------------------------------------------------------------------
# Reward Calculator
# ---------------------------------------------------------------------------


class RewardCalculator:
    """Orchestrates calculation of multi-component reward."""

    def __init__(self, config: RewardConfig | None = None):
        """Initialize calculator.

        Parameters
        ----------
        config : RewardConfig, optional
            Configuration for reward components. Uses defaults if None.
        """
        self.config = config or RewardConfig()
        self.equity_history: list[float] = []
        self.pnl_history: list[float] = []  # P&L from closed trades
        self.bars_in_trade = 0  # Bars since entry

    def reset(self) -> None:
        """Reset tracking for new episode."""
        self.equity_history.clear()
        self.pnl_history.clear()
        self.bars_in_trade = 0

    def record_equity(self, equity: float) -> None:
        """Record equity for history tracking."""
        self.equity_history.append(float(equity))

    def record_pnl(self, pnl: float) -> None:
        """Record closed trade P&L."""
        self.pnl_history.append(float(pnl))
        self.bars_in_trade = 0

    def increment_bars_held(self) -> None:
        """Increment bars held counter."""
        self.bars_in_trade += 1

    def calculate_reward(
        self,
        prev_equity: float,
        curr_equity: float,
        position_open: bool,
        is_invalid_action: bool,
    ) -> Tuple[float, dict]:
        """Calculate total reward and components breakdown.

        Parameters
        ----------
        prev_equity : float
            Previous equity
        curr_equity : float
            Current equity
        position_open : bool
            Whether position is currently open
        is_invalid_action : bool
            Whether action was invalid

        Returns
        -------
        total_reward : float
            Total multi-component reward (clipped)
        components : dict
            Breakdown of all components
        """
        components = {}

        # Record equity
        self.record_equity(curr_equity)

        # 1. Base equity return
        if self.config.equity_scale > 0:
            components["equity"] = RewardComponents.equity_return(
                prev_equity, curr_equity, self.config.equity_scale
            )
        else:
            components["equity"] = 0.0

        # 2. Sharpe ratio bonus
        if self.config.use_sharpe:
            components["sharpe"] = RewardComponents.sharpe_ratio_component(
                np.array(self.equity_history),
                window=self.config.sharpe_window,
                scale=self.config.sharpe_scale,
            )
        else:
            components["sharpe"] = 0.0

        # 3. Streak bonus
        if self.config.use_streak:
            components["streak"] = RewardComponents.win_rate_streak_bonus(
                self.pnl_history,
                streak_min=self.config.streak_min_length,
                scale=self.config.streak_scale,
            )
        else:
            components["streak"] = 0.0

        # 4. Duration reward
        if self.config.use_duration and position_open:
            components["duration"] = RewardComponents.trade_duration_reward(
                self.bars_in_trade,
                optimal_bars=self.config.duration_optimal_bars,
                scale=self.config.duration_scale,
            )
        else:
            components["duration"] = 0.0

        # 5. Drawdown penalty
        if self.config.use_drawdown:
            components["drawdown"] = RewardComponents.drawdown_penalty(
                np.array(self.equity_history),
                window=self.config.drawdown_window,
                scale=self.config.drawdown_scale,
            )
        else:
            components["drawdown"] = 0.0

        # 6. Opportunity cost
        if self.config.use_opportunity_cost and position_open:
            components["opportunity"] = RewardComponents.opportunity_cost_penalty(
                self.bars_in_trade,
                threshold_bars=self.config.holding_threshold_bars,
                scale=self.config.opportunity_scale,
            )
        else:
            components["opportunity"] = 0.0

        # 7. Invalid action penalty
        if self.config.use_invalid_penalty:
            components["invalid"] = RewardComponents.invalid_action_penalty(
                is_invalid_action, self.config.invalid_penalty
            )
        else:
            components["invalid"] = 0.0

        # Sum all components
        total_reward = sum(components.values())

        # Clip reward
        total_reward = float(
            np.clip(total_reward, self.config.reward_clip_min, self.config.reward_clip_max)
        )

        return total_reward, components


# ---------------------------------------------------------------------------
# Preset Configurations
# ---------------------------------------------------------------------------


def reward_config_baseline() -> RewardConfig:
    """Baseline config: close to original reward function."""
    return RewardConfig(
        equity_scale=1.0,
        use_sharpe=False,
        use_streak=False,
        use_duration=False,
        use_drawdown=False,
        use_opportunity_cost=False,
        use_invalid_penalty=True,
        invalid_penalty=-1e-4,
    )


def reward_config_enhanced() -> RewardConfig:
    """Enhanced config: experimental new components."""
    return RewardConfig(
        equity_scale=1.0,
        use_sharpe=True,
        sharpe_scale=0.05,
        use_streak=True,
        streak_scale=0.03,
        use_duration=True,
        duration_scale=0.02,
        use_drawdown=True,
        drawdown_scale=0.05,
        use_opportunity_cost=True,
        opportunity_scale=0.005,
        use_invalid_penalty=True,
        invalid_penalty=-1e-4,
    )


def reward_config_aggressive() -> RewardConfig:
    """Aggressive config: higher weights for all components."""
    return RewardConfig(
        equity_scale=1.0,
        use_sharpe=True,
        sharpe_scale=0.15,
        use_streak=True,
        streak_scale=0.10,
        use_duration=True,
        duration_scale=0.05,
        use_drawdown=True,
        drawdown_scale=0.15,
        use_opportunity_cost=True,
        opportunity_scale=0.02,
        use_invalid_penalty=True,
        invalid_penalty=-5e-4,
    )


if __name__ == "__main__":
    # Quick test
    config = reward_config_enhanced()
    calc = RewardCalculator(config)

    # Simulate 100 steps
    equity = 10000.0
    for i in range(100):
        prev_equity = equity
        # Simulate random equity change
        change = np.random.randn() * 100
        equity += change

        calc.record_equity(equity)

        reward, components = calc.calculate_reward(
            prev_equity, equity, position_open=(i % 10 < 7), is_invalid_action=(i % 20 == 0)
        )

        if i % 20 == 0:
            print(
                f"Step {i}: Equity={equity:.0f}, Reward={reward:.6f}, "
                f"Components: {', '.join(f'{k}={v:.4f}' for k, v in components.items() if v != 0)}"
            )

    print("\n✅ Reward calculator working!")
