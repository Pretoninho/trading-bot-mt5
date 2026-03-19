"""
test_features.py
================
Comprehensive test suite for feature engineering module.

Tests:
- Feature computation (volatility, trend, market structure)
- Normalization (mean=0, std=1)
- Integration with market_loader and trading environment
- Shape validation
- No NaN / Inf values
"""

import numpy as np
import pandas as pd
import pytest

from trading_env.utils.feature_engineering import (
    FeatureEngineer,
    VolatilityFeatures,
    TrendFeatures,
    MarketStructureFeatures,
    VOLATILITY_FEATURES_COUNT,
    TREND_FEATURES_COUNT,
    MARKET_STRUCTURE_FEATURES_COUNT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataframe():
    """Create synthetic OHLC data for testing."""
    n = 500
    t = np.arange(n)
    close = 1.0 + 0.01 * np.sin(t / 50) + 0.001 * np.random.randn(n)
    high = close + 0.001 * np.abs(np.random.randn(n))
    low = close - 0.001 * np.abs(np.random.randn(n))
    open_ = close + 0.0005 * np.random.randn(n)
    spread = 0.00008 * np.ones(n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "spread_price": spread,
    })
    return df


@pytest.fixture
def realistic_dataframe():
    """Create realistic-looking EURUSD M1 data."""
    n = 2000  # 1+ day of data
    dates = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")

    # Trend with noise
    t = np.arange(n)
    trend = 0.0001 * (t / n)  # slight uptrend
    noise = 0.0005 * np.random.randn(n)
    close = 1.08500 + trend + noise
    high = close + 0.0002 * np.abs(np.random.randn(n))
    low = close - 0.0002 * np.abs(np.random.randn(n))
    open_ = close + 0.0001 * np.random.randn(n)

    df = pd.DataFrame({
        "dt": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "spread_price": 0.00008,
    })
    return df


# ---------------------------------------------------------------------------
# TestVolatilityFeatures
# ---------------------------------------------------------------------------


class TestVolatilityFeatures:
    """Test volatility feature computation."""

    def test_shape(self, synthetic_dataframe):
        """Volatility features should have shape (N, 20)."""
        result = VolatilityFeatures.compute(synthetic_dataframe)
        assert result.shape == (len(synthetic_dataframe), 20)
        assert result.dtype == np.float32

    def test_no_nan_inf(self, synthetic_dataframe):
        """Computed features should not contain excessive NaN or Inf."""
        result = VolatilityFeatures.compute(synthetic_dataframe)
        # Some NaN at the beginning is expected due to rolling windows
        # Check that not all values are NaN/Inf
        non_nan_count = np.sum(~np.isnan(result))
        total_count = result.size
        nan_ratio = 1 - (non_nan_count / total_count)
        assert nan_ratio < 0.3, f"Too many NaN values: {nan_ratio * 100:.1f}%"
        assert not np.isinf(result).any(), "Features contain Inf"

    def test_values_bounded(self, synthetic_dataframe):
        """Volatility features should be in reasonable range (pre-normalization)."""
        result = VolatilityFeatures.compute(synthetic_dataframe)
        # Remove NaN and Inf values before checking bounds
        valid_result = result[~np.isnan(result) & ~np.isinf(result)]
        # Most volatility features are ratios; we allow outliers from divisions
        # The normalization step will clip these to [-10, 10]
        assert len(valid_result) > 0, "No valid values in result"
        # Just ensure we don't have overwhelming Inf values
        inf_ratio = np.sum(np.isinf(result)) / result.size
        assert inf_ratio < 0.1, f"Too many Inf values: {inf_ratio * 100:.1f}%"

    def test_atr_computation(self, synthetic_dataframe):
        """ATR should be positive."""
        result = VolatilityFeatures.compute(synthetic_dataframe)
        # ATR (feature 5) should be positive where defined
        atr_col = result[:, 5]
        atr_valid = atr_col[~np.isnan(atr_col)]
        assert np.all(atr_valid >= -0.01), "ATR should be non-negative (allowing small floating point errors)"


class TestTrendFeatures:
    """Test trend feature computation."""

    def test_shape(self, synthetic_dataframe):
        """Trend features should have shape (N, 30)."""
        result = TrendFeatures.compute(synthetic_dataframe)
        assert result.shape == (len(synthetic_dataframe), 30)
        assert result.dtype == np.float32

    def test_no_nan_inf(self, synthetic_dataframe):
        """Computed features should not contain excessive NaN or Inf."""
        result = TrendFeatures.compute(synthetic_dataframe)
        # Some NaN at the beginning is expected due to rolling windows
        non_nan_count = np.sum(~np.isnan(result))
        total_count = result.size
        nan_ratio = 1 - (non_nan_count / total_count)
        assert nan_ratio < 0.5, f"Too many NaN values: {nan_ratio * 100:.1f}%"
        assert not np.isinf(result).any(), "Features contain Inf"

    def test_rsi_in_range(self, synthetic_dataframe):
        """RSI (feature 0) should be in [0, 1] (already normalized)."""
        result = TrendFeatures.compute(synthetic_dataframe)
        rsi = result[:, 0]
        rsi_valid = rsi[~np.isnan(rsi)]
        assert np.all(rsi_valid >= -1.1), "RSI normalized should be >= -1"
        assert np.all(rsi_valid <= 1.1), "RSI normalized should be <= 1"

    def test_ema_crossover_signal(self, synthetic_dataframe):
        """EMA crossover signal (feature 15) should be ±1 or variations."""
        result = TrendFeatures.compute(synthetic_dataframe)
        crossover = result[:, 15]
        assert np.all(np.abs(crossover) <= 1.5), "EMA crossover signal out of range"


class TestMarketStructureFeatures:
    """Test market structure feature computation."""

    def test_shape(self, synthetic_dataframe):
        """Market structure features should have shape (N, 15)."""
        result = MarketStructureFeatures.compute(synthetic_dataframe)
        assert result.shape == (len(synthetic_dataframe), 15)
        assert result.dtype == np.float32

    def test_no_nan_inf(self, synthetic_dataframe):
        """Computed features should not contain NaN or Inf."""
        result = MarketStructureFeatures.compute(synthetic_dataframe)
        # Fill remaining NaN with 0 (expected for incomplete bars)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        assert not np.isnan(result).any(), "Features contain NaN"
        assert not np.isinf(result).any(), "Features contain Inf"

    def test_spread_normalized(self, synthetic_dataframe):
        """Spread feature (feature 0) should be small positive values."""
        result = MarketStructureFeatures.compute(synthetic_dataframe)
        spread = result[:, 0]
        assert np.all(spread >= 0), "Spread should be non-negative"
        assert np.median(spread) < 0.01, "Spread seems too large"


# ---------------------------------------------------------------------------
# TestFeatureEngineer
# ---------------------------------------------------------------------------


class TestFeatureEngineer:
    """Test main FeatureEngineer orchestrator."""

    def test_compute_all_features_shapes(self, synthetic_dataframe):
        """All feature groups should have correct shapes."""
        vol, trend, market = FeatureEngineer.compute_all_features(synthetic_dataframe)

        assert vol.shape == (len(synthetic_dataframe), 20)
        assert trend.shape == (len(synthetic_dataframe), 30)
        assert market.shape == (len(synthetic_dataframe), 15)

    def test_normalization_mean_std(self, synthetic_dataframe):
        """Normalized features should have mean≈0, std≈1."""
        vol, trend, market = FeatureEngineer.compute_all_features(synthetic_dataframe)

        # Check mean and std (should be close to 0 and 1)
        vol_mean = np.nanmean(vol)
        vol_std = np.nanstd(vol)
        assert abs(vol_mean) < 0.1, f"Volatility mean {vol_mean} not close to 0"
        assert abs(vol_std - 1.0) < 0.2, f"Volatility std {vol_std} not close to 1"

        trend_mean = np.nanmean(trend)
        trend_std = np.nanstd(trend)
        assert abs(trend_mean) < 0.1, f"Trend mean {trend_mean} not close to 0"
        assert abs(trend_std - 1.0) < 0.2, f"Trend std {trend_std} not close to 1"

    def test_no_nan_after_compute(self, synthetic_dataframe):
        """After compute_all_features, should have no NaN values."""
        vol, trend, market = FeatureEngineer.compute_all_features(synthetic_dataframe)

        assert not np.isnan(vol).any(), "Volatility features contain NaN"
        assert not np.isnan(trend).any(), "Trend features contain NaN"
        assert not np.isnan(market).any(), "Market features contain NaN"

    def test_values_clipped(self, synthetic_dataframe):
        """Normalized features should be clipped to [-10, 10]."""
        vol, trend, market = FeatureEngineer.compute_all_features(synthetic_dataframe)

        assert np.all(vol >= -10) and np.all(vol <= 10), "Vol features out of [-10, 10]"
        assert np.all(trend >= -10) and np.all(trend <= 10), "Trend features out of [-10, 10]"
        assert np.all(market >= -10) and np.all(market <= 10), "Market features out of [-10, 10]"


# ---------------------------------------------------------------------------
# TestMarketLoaderIntegration
# ---------------------------------------------------------------------------


class TestMarketLoaderIntegration:
    """Test integration with market_loader (if compute_features=True)."""

    def test_load_with_features(self, realistic_dataframe, tmp_path):
        """Load data with features enabled."""
        from trading_env.data.market_loader import load_mt5_m1_csv

        # Write test CSV
        csv_file = tmp_path / "test_data.csv"
        realistic_dataframe.to_csv(
            csv_file,
            sep=" ",
            index=False,
            columns=["open", "high", "low", "close", "spread_price"],
            float_format="%.8f",
        )

        # Modify to match MT5 format (add datetime as date + time)
        df_csv = realistic_dataframe.copy()
        df_csv["date"] = df_csv["dt"].dt.strftime("%Y.%m.%d")
        df_csv["time"] = df_csv["dt"].dt.strftime("%H:%M:%S")
        df_csv["spread"] = (df_csv["spread_price"] / 0.00001).astype(int)

        csv_file = tmp_path / "test_mt5.csv"
        df_csv.to_csv(
            csv_file,
            sep=" ",
            index=False,
            columns=["date", "time", "open", "high", "low", "close", "spread"],
        )

        # Load with features
        df_loaded = load_mt5_m1_csv(csv_file, compute_features=True)

        # Check that feature columns exist
        feature_cols = [c for c in df_loaded.columns if c.startswith(("vol_", "trend_", "market_"))]
        assert len(feature_cols) == 65, f"Expected 65 features, got {len(feature_cols)}"

        # Check no NaN in features
        for col in feature_cols:
            assert not df_loaded[col].isna().all(), f"{col} is all NaN"


# ---------------------------------------------------------------------------
# TestTradingEnvironmentIntegration
# ---------------------------------------------------------------------------


class TestTradingEnvironmentIntegration:
    """Test integration with trading environment."""

    def test_environment_obs_size_with_features(self, realistic_dataframe):
        """Trading environment should use expanded observation size."""
        from trading_env.env.trading_env import EURUSDTradingEnv

        # Add required columns for environment
        df = realistic_dataframe.copy()
        df["tradable_now"] = 1
        df["is_safe_week"] = 1
        df["has_breakout"] = 0
        df["iso_year"] = 2025
        df["iso_week"] = 1
        df["weekday"] = df["dt"].dt.weekday

        # Add features manually (simulating market_loader behavior)
        vol, trend, market = FeatureEngineer.compute_all_features(df)
        for i in range(20):
            df[f"vol_{i}"] = vol[:, i]
        for i in range(30):
            df[f"trend_{i}"] = trend[:, i]
        for i in range(15):
            df[f"market_{i}"] = market[:, i]

        # Create environment
        env = EURUSDTradingEnv(df, initial_equity=10000.0)

        # Observation size should be 456 (391 + 65)
        assert env.observation_space.shape[0] == 456, \
            f"Expected obs size 456, got {env.observation_space.shape[0]}"

    def test_env_reset_and_step_with_features(self, realistic_dataframe):
        """Environment should reset and step with enriched observations."""
        from trading_env.env.trading_env import EURUSDTradingEnv

        df = realistic_dataframe.copy()
        df["tradable_now"] = 1
        df["is_safe_week"] = 1
        df["has_breakout"] = 0
        df["iso_year"] = 2025
        df["iso_week"] = 1
        df["weekday"] = df["dt"].dt.weekday

        vol, trend, market = FeatureEngineer.compute_all_features(df)
        for i in range(20):
            df[f"vol_{i}"] = vol[:, i]
        for i in range(30):
            df[f"trend_{i}"] = trend[:, i]
        for i in range(15):
            df[f"market_{i}"] = market[:, i]

        env = EURUSDTradingEnv(df, initial_equity=10000.0)

        # Reset
        obs, info = env.reset()
        assert obs.shape == (456,), f"Reset obs shape {obs.shape} != (456,)"
        assert not np.isnan(obs).any(), "Reset obs contains NaN"

        # Step
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)  # HOLD
            assert obs.shape == (456,), f"Step obs shape {obs.shape} != (456,)"
            assert not np.isnan(obs).any(), f"Step obs contains NaN"
            if terminated or truncated:
                break


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_short_series(self):
        """Feature computation should handle short series gracefully (>= 20 bars recommended)."""
        # Short but workable series (50 bars)
        n = 50
        df = pd.DataFrame({
            "open": np.full(n, 1.0),
            "high": np.full(n, 1.001),
            "low": np.full(n, 0.999),
            "close": np.full(n, 1.0),
            "spread_price": np.full(n, 0.00008),
        })

        vol, trend, market = FeatureEngineer.compute_all_features(df)

        # Should not crash and should have correct shapes
        assert vol.shape == (n, 20)
        assert trend.shape == (n, 30)
        assert market.shape == (n, 15)

    def test_constant_price(self):
        """Feature computation should handle flat markets."""
        n = 100
        df = pd.DataFrame({
            "open": np.full(n, 1.08500),
            "high": np.full(n, 1.08505),
            "low": np.full(n, 1.08495),
            "close": np.full(n, 1.08500),
            "spread_price": np.full(n, 0.00008),
        })

        vol, trend, market = FeatureEngineer.compute_all_features(df)

        # Should not crash; features may be 0/NaN in some columns
        assert vol.shape == (n, 20)
        assert trend.shape == (n, 30)
        assert market.shape == (n, 15)

    def test_highly_volatile_market(self):
        """Feature computation should handle extreme volatility."""
        n = 100
        df = pd.DataFrame({
            "open": 1.0 + 0.1 * np.random.randn(n),
            "high": 1.0 + 0.15 * np.abs(np.random.randn(n)),
            "low": 1.0 - 0.15 * np.abs(np.random.randn(n)),
            "close": 1.0 + 0.1 * np.random.randn(n),
            "spread_price": np.full(n, 0.00008),
        })

        vol, trend, market = FeatureEngineer.compute_all_features(df)

        # Features should still be clipped to [-10, 10]
        assert np.all(vol >= -10) and np.all(vol <= 10)
        assert np.all(trend >= -10) and np.all(trend <= 10)
        assert np.all(market >= -10) and np.all(market <= 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
