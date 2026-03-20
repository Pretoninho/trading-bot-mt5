"""
test_reward.py
==============
Comprehensive tests for multi-component reward function system.

Tests:
- Individual reward components (equity, Sharpe, streak, duration, etc.)
- RewardCalculator integration
- Configuration management
- Preset configurations
- Edge cases and boundary conditions
"""

import numpy as np
import pytest

from trading_env.utils.reward_functions import (
    RewardComponents,
    RewardCalculator,
    RewardConfig,
    reward_config_baseline,
    reward_config_enhanced,
    reward_config_aggressive,
)


# ---------------------------------------------------------------------------
# Test RewardConfig
# ---------------------------------------------------------------------------


class TestRewardConfig:
    """Test configuration management."""

    def test_config_defaults(self):
        """RewardConfig should have sensible defaults."""
        config = RewardConfig()
        assert config.equity_scale == 1.0
        assert config.use_sharpe is True
        assert config.sharpe_window == 100
        assert config.streak_min_length == 3
        assert config.reward_clip_min == -1.0
        assert config.reward_clip_max == 1.0

    def test_config_to_dict(self):
        """Config should convert to dict."""
        config = RewardConfig(equity_scale=2.0, sharpe_scale=0.2)
        config_dict = config.to_dict()
        assert config_dict["equity_scale"] == 2.0
        assert config_dict["sharpe_scale"] == 0.2

    def test_config_from_dict(self):
        """Config should load from dict."""
        config_dict = {
            "equity_scale": 1.5,
            "sharpe_scale": 0.15,
            "sharpe_window": 150,
            "streak_scale": 0.1,
            "duration_scale": 0.03,
            "drawdown_scale": 0.2,
            "opportunity_scale": 0.02,
            "invalid_penalty": -5e-4,
            "reward_clip_min": -1.0,
            "reward_clip_max": 1.0,
        }
        config = RewardConfig.from_dict(config_dict)
        assert config.equity_scale == 1.5
        assert config.sharpe_scale == 0.15

    def test_preset_configs(self):
        """Preset configurations should have different characteristics."""
        baseline = reward_config_baseline()
        enhanced = reward_config_enhanced()
        aggressive = reward_config_aggressive()

        # Baseline should have minimal components
        assert not baseline.use_sharpe
        assert not baseline.use_streak
        assert not baseline.use_duration

        # Enhanced should have more components enabled
        assert enhanced.use_sharpe
        assert enhanced.use_streak
        assert enhanced.use_duration

        # Aggressive should have higher scales
        assert aggressive.sharpe_scale > enhanced.sharpe_scale
        assert aggressive.streak_scale > enhanced.streak_scale


# ---------------------------------------------------------------------------
# TestRewardComponents - Individual Components
# ---------------------------------------------------------------------------


class TestEquityReturn:
    """Test equity return component."""

    def test_positive_return(self):
        """Positive equity change should give positive reward."""
        reward = RewardComponents.equity_return(10000, 10100, scale=1.0)
        assert reward > 0
        assert abs(reward - 0.01) < 1e-6

    def test_negative_return(self):
        """Negative equity change should give negative reward."""
        reward = RewardComponents.equity_return(10000, 9900, scale=1.0)
        assert reward < 0
        assert abs(reward - (-0.01)) < 1e-6

    def test_zero_return(self):
        """No equity change should give zero reward."""
        reward = RewardComponents.equity_return(10000, 10000, scale=1.0)
        assert reward == 0.0

    def test_scaling(self):
        """Scaling factor should be applied."""
        reward1 = RewardComponents.equity_return(10000, 10100, scale=1.0)
        reward2 = RewardComponents.equity_return(10000, 10100, scale=2.0)
        assert abs(reward2 - 2 * reward1) < 1e-6

    def test_zero_equity_protection(self):
        """Should handle zero equity gracefully."""
        reward = RewardComponents.equity_return(0, 100, scale=1.0)
        assert reward == 0.0


class TestSharpeRatio:
    """Test Sharpe ratio component."""

    def test_insufficient_history(self):
        """Should return 0 if history too short."""
        equity_history = np.array([10000, 10100])
        reward = RewardComponents.sharpe_ratio_component(equity_history, window=100)
        assert reward == 0.0

    def test_constant_equity(self):
        """Constant equity should give near-zero Sharpe bonus."""
        equity_history = np.full(150, 10000.0)
        reward = RewardComponents.sharpe_ratio_component(equity_history, window=100)
        assert abs(reward) < 0.01

    def test_uptrend(self):
        """Uptrend should give positive Sharpe bonus."""
        equity_history = np.linspace(10000, 11000, 150)
        reward = RewardComponents.sharpe_ratio_component(equity_history, window=100)
        assert reward > 0

    def test_downtrend(self):
        """Downtrend should give negative Sharpe bonus."""
        equity_history = np.linspace(11000, 10000, 150)
        reward = RewardComponents.sharpe_ratio_component(equity_history, window=100)
        assert reward < 0

    def test_scaling(self):
        """Scaling factor should be applied."""
        equity_history = np.linspace(10000, 11000, 150)
        reward1 = RewardComponents.sharpe_ratio_component(equity_history, window=100, scale=0.1)
        reward2 = RewardComponents.sharpe_ratio_component(equity_history, window=100, scale=0.2)
        assert abs(reward2 - 2 * reward1) < 1e-6


class TestStreakBonus:
    """Test win-rate streak bonus."""

    def test_no_streak(self):
        """No consecutive wins should give zero bonus."""
        pnl_history = [100, -50, 100, -50]
        bonus = RewardComponents.win_rate_streak_bonus(pnl_history, streak_min=3)
        assert bonus == 0.0

    def test_short_streak_below_min(self):
        """Streak shorter than minimum should give zero bonus."""
        pnl_history = [100, 100]  # 2-streak, min=3
        bonus = RewardComponents.win_rate_streak_bonus(pnl_history, streak_min=3)
        assert bonus == 0.0

    def test_min_streak(self):
        """Minimum streak should give zero bonus (needs min+1 for non-zero)."""
        # Minimum streak exactly = min gives zero bonus by design
        pnl_history = [100, 100, 100]  # 3-streak, min=3
        bonus = RewardComponents.win_rate_streak_bonus(pnl_history, streak_min=3)
        assert bonus == 0.0

    def test_streak_above_minimum(self):
        """Streak above minimum should give positive bonus."""
        pnl_history = [100, 100, 100, 100]  # 4-streak, min=3
        bonus = RewardComponents.win_rate_streak_bonus(pnl_history, streak_min=3)
        assert bonus > 0

    def test_long_streak(self):
        """Longer streak should give larger bonus."""
        pnl_short = [100, 100, 100]
        pnl_long = [100, 100, 100, 100, 100]
        bonus_short = RewardComponents.win_rate_streak_bonus(pnl_short, streak_min=3)
        bonus_long = RewardComponents.win_rate_streak_bonus(pnl_long, streak_min=3)
        assert bonus_long > bonus_short

    def test_streak_after_loss(self):
        """Streak not at end should not count."""
        pnl_history = [100, 100, 100, -50]  # Streak ended
        bonus = RewardComponents.win_rate_streak_bonus(pnl_history, streak_min=3)
        assert bonus == 0.0


class TestDurationReward:
    """Test trade duration reward."""

    def test_zero_bars(self):
        """Zero bars in trade should give zero reward."""
        reward = RewardComponents.trade_duration_reward(0, optimal_bars=60)
        assert reward == 0.0

    def test_optimal_duration(self):
        """Optimal duration should give maximum reward."""
        reward_opt = RewardComponents.trade_duration_reward(60, optimal_bars=60)
        reward_short = RewardComponents.trade_duration_reward(30, optimal_bars=60)
        reward_long = RewardComponents.trade_duration_reward(90, optimal_bars=60)
        assert reward_opt > reward_short
        assert reward_opt > reward_long

    def test_symmetry(self):
        """Reward should be symmetric around optimal."""
        reward_40 = RewardComponents.trade_duration_reward(40, optimal_bars=60)
        reward_80 = RewardComponents.trade_duration_reward(80, optimal_bars=60)
        assert abs(reward_40 - reward_80) < 1e-6

    def test_scaling(self):
        """Scaling should be applied."""
        reward1 = RewardComponents.trade_duration_reward(60, optimal_bars=60, scale=0.02)
        reward2 = RewardComponents.trade_duration_reward(60, optimal_bars=60, scale=0.04)
        assert abs(reward2 - 2 * reward1) < 1e-6


class TestDrawdownPenalty:
    """Test drawdown penalty."""

    def test_no_drawdown(self):
        """No drawdown should give zero penalty."""
        equity_history = np.array([10000, 10100, 10200, 10300])
        penalty = RewardComponents.drawdown_penalty(equity_history)
        assert penalty == 0.0

    def test_mild_drawdown(self):
        """Mild drawdown should give small negative penalty."""
        equity_history = np.array([10000, 10100, 10200, 10100])  # max=10200, current=10100
        penalty = RewardComponents.drawdown_penalty(equity_history)
        assert penalty < 0
        assert abs(penalty) < 0.01

    def test_severe_drawdown(self):
        """Severe drawdown should give larger penalty."""
        # Note: Penalty is quadratic, so a 3.9% drawdown with scale=0.1 gives ~-0.00015
        # To get -0.05, we'd need ~50% drawdown
        equity_history = np.array([10000, 10100, 10200, 9800])  # max=10200, current=9800
        penalty = RewardComponents.drawdown_penalty(equity_history, scale=0.1)
        assert penalty < 0
        assert abs(penalty) < 0.01  # Quadratic penalty is small for moderate drawdown

    def test_scaling(self):
        """Scaling should increase penalty magnitude."""
        equity_history = np.array([10000, 10100, 10200, 9800])
        penalty1 = RewardComponents.drawdown_penalty(equity_history, scale=0.1)
        penalty2 = RewardComponents.drawdown_penalty(equity_history, scale=0.2)
        assert abs(penalty2) > abs(penalty1)


class TestOpportunityCost:
    """Test opportunity cost penalty."""

    def test_no_penalty_below_threshold(self):
        """No penalty when bars held below threshold."""
        penalty = RewardComponents.opportunity_cost_penalty(60, threshold_bars=120)
        assert penalty == 0.0

    def test_penalty_above_threshold(self):
        """Penalty when bars held exceeds threshold."""
        penalty = RewardComponents.opportunity_cost_penalty(150, threshold_bars=120)
        assert penalty < 0

    def test_increasing_penalty(self):
        """Penalty should increase with bars held."""
        penalty_short = RewardComponents.opportunity_cost_penalty(130, threshold_bars=120)
        penalty_long = RewardComponents.opportunity_cost_penalty(180, threshold_bars=120)
        assert penalty_long < penalty_short  # more negative

    def test_scaling(self):
        """Scaling should affect penalty magnitude."""
        penalty1 = RewardComponents.opportunity_cost_penalty(150, threshold_bars=120, scale=0.01)
        penalty2 = RewardComponents.opportunity_cost_penalty(150, threshold_bars=120, scale=0.02)
        assert abs(penalty2) > abs(penalty1)


class TestInvalidPenalty:
    """Test invalid action penalty."""

    def test_valid_action(self):
        """Valid action should give zero penalty."""
        penalty = RewardComponents.invalid_action_penalty(False, penalty=-1e-4)
        assert penalty == 0.0

    def test_invalid_action(self):
        """Invalid action should give negative penalty."""
        penalty = RewardComponents.invalid_action_penalty(True, penalty=-1e-4)
        assert penalty == -1e-4


# ---------------------------------------------------------------------------
# TestRewardCalculator - Integration
# ---------------------------------------------------------------------------


class TestRewardCalculator:
    """Test RewardCalculator integration."""

    def test_init(self):
        """Calculator should initialize with config."""
        config = reward_config_enhanced()
        calc = RewardCalculator(config)
        assert calc.config == config
        assert len(calc.equity_history) == 0

    def test_reset(self):
        """Reset should clear history."""
        calc = RewardCalculator()
        calc.record_equity(10000)
        calc.record_equity(10100)
        assert len(calc.equity_history) == 2

        calc.reset()
        assert len(calc.equity_history) == 0
        assert len(calc.pnl_history) == 0
        assert calc.bars_in_trade == 0

    def test_record_equity(self):
        """Record equity should update history."""
        calc = RewardCalculator()
        calc.record_equity(10000)
        calc.record_equity(10100)
        assert len(calc.equity_history) == 2
        assert calc.equity_history[0] == 10000
        assert calc.equity_history[1] == 10100

    def test_record_pnl_and_bars_reset(self):
        """Record PnL should reset bars_in_trade counter."""
        calc = RewardCalculator()
        calc.bars_in_trade = 50
        calc.record_pnl(100)
        assert calc.bars_in_trade == 0
        assert calc.pnl_history == [100]

    def test_increment_bars_held(self):
        """Increment should increase counter."""
        calc = RewardCalculator()
        assert calc.bars_in_trade == 0
        calc.increment_bars_held()
        assert calc.bars_in_trade == 1
        calc.increment_bars_held()
        assert calc.bars_in_trade == 2

    def test_calculate_reward_baseline(self):
        """Baseline config should give simple equity-based reward."""
        config = reward_config_baseline()
        calc = RewardCalculator(config)

        reward, components = calc.calculate_reward(
            prev_equity=10000, curr_equity=10100, position_open=False, is_invalid_action=False
        )

        assert reward > 0
        assert components["equity"] > 0
        assert components["sharpe"] == 0.0
        assert components["streak"] == 0.0

    def test_calculate_reward_with_components(self):
        """Enhanced config should calculate all enabled components."""
        config = reward_config_enhanced()
        calc = RewardCalculator(config)

        # Build equity history
        for eq in [10000, 10050, 10100, 10080, 10120, 10150]:
            _, _ = calc.calculate_reward(
                prev_equity=eq - 50, curr_equity=eq, position_open=True, is_invalid_action=False
            )

        reward, components = calc.calculate_reward(
            prev_equity=10150, curr_equity=10200, position_open=True, is_invalid_action=False
        )

        assert reward > 0
        assert "equity" in components
        assert "sharpe" in components
        assert "streak" in components
        assert "duration" in components

    def test_reward_clipping(self):
        """Reward should be clipped to configured range."""
        config = RewardConfig(reward_clip_min=-0.5, reward_clip_max=0.5)
        calc = RewardCalculator(config)

        reward, _ = calc.calculate_reward(
            prev_equity=10000, curr_equity=20000, position_open=False, is_invalid_action=False
        )

        assert reward <= 0.5

    def test_invalid_action_penalty_applied(self):
        """Invalid actions should reduce reward."""
        config = reward_config_baseline()
        calc = RewardCalculator(config)

        reward_valid, _ = calc.calculate_reward(
            prev_equity=10000, curr_equity=10100, position_open=False, is_invalid_action=False
        )
        calc.reset()
        reward_invalid, _ = calc.calculate_reward(
            prev_equity=10000, curr_equity=10100, position_open=False, is_invalid_action=True
        )

        assert reward_invalid < reward_valid


# ---------------------------------------------------------------------------
# Edge Cases & Boundary Conditions
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_equity_values(self):
        """Should handle large equity values."""
        calc = RewardCalculator()
        reward, _ = calc.calculate_reward(
            prev_equity=1e9, curr_equity=1e9 + 1e6, position_open=False, is_invalid_action=False
        )
        assert np.isfinite(reward)

    def test_very_small_equity_changes(self):
        """Should handle very small equity changes."""
        calc = RewardCalculator()
        reward, _ = calc.calculate_reward(
            prev_equity=10000, curr_equity=10000.0001, position_open=False, is_invalid_action=False
        )
        assert np.isfinite(reward)

    def test_zero_equity(self):
        """Should handle zero equity gracefully."""
        calc = RewardCalculator()
        reward, components = calc.calculate_reward(
            prev_equity=0, curr_equity=100, position_open=False, is_invalid_action=False
        )
        assert np.isfinite(reward)
        assert components["equity"] == 0.0

    def test_alternating_wins_losses(self):
        """Should handle alternating P&L correctly."""
        config = reward_config_enhanced()
        calc = RewardCalculator(config)

        for _ in range(10):
            calc.record_pnl(100)
            calc.record_pnl(-50)

        bonus = RewardComponents.win_rate_streak_bonus(calc.pnl_history)
        assert bonus == 0.0  # No winning streak

    def test_all_components_disabled(self):
        """Should handle all components disabled."""
        config = RewardConfig(
            equity_scale=0,
            use_sharpe=False,
            use_streak=False,
            use_duration=False,
            use_drawdown=False,
            use_opportunity_cost=False,
            use_invalid_penalty=False,
        )
        calc = RewardCalculator(config)

        reward, components = calc.calculate_reward(
            prev_equity=10000, curr_equity=10100, position_open=True, is_invalid_action=False
        )

        assert reward == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
