"""
feature_engineering.py
======================
Advanced feature engineering for EURUSD M1 trading environment.

Enriches observations by computing:
  - Volatility features (Bollinger Bands, ATR variations)
  - Trend features (RSI14, MACD, price slope, EMA)
  - Market structure features (bid-ask imbalance, liquidity)

Each feature group is independently normalized (mean=0, std=1) for stability.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Volatility features
BB_PERIOD = 20
BB_STD_DEV = 2.0
ATR_PERIOD = 14

# Trend features
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 12
EMA_LONG = 26

# Normalization
VOLATILITY_FEATURES_COUNT = 20
TREND_FEATURES_COUNT = 30
MARKET_STRUCTURE_FEATURES_COUNT = 15

TOTAL_NEW_FEATURES = (
    VOLATILITY_FEATURES_COUNT + TREND_FEATURES_COUNT + MARKET_STRUCTURE_FEATURES_COUNT
)


# ---------------------------------------------------------------------------
# Volatility Features (20D)
# ---------------------------------------------------------------------------


class VolatilityFeatures:
    """Compute volatility-based features.

    Returns
    -------
    Array of shape (N, 20) with normalized features:
      [bb_upper, bb_middle, bb_lower, bb_position, bb_width,
       atr_norm, atr_ratio, atr_trend_up, atr_trend_down, atr_acceleration,
       close_range, intrabar_range, std_dev_short, std_dev_long, std_ratio,
       volatility_expansion, squeeze_signal, true_range, hml_ratio, hml_trend]
    """

    @staticmethod
    def compute(df: pd.DataFrame) -> np.ndarray:
        """Compute volatility features for all bars.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, spread_price

        Returns
        -------
        np.ndarray
            Shape (len(df), 20) with float32 dtype
        """
        n = len(df)
        features = np.zeros((n, 20), dtype=np.float32)

        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        open_ = df["open"].values.astype(np.float64)

        # 1. Bollinger Bands (5 features)
        sma = pd.Series(close).rolling(window=BB_PERIOD, center=False).mean().values
        std = pd.Series(close).rolling(window=BB_PERIOD, center=False).std().values
        bb_upper = sma + BB_STD_DEV * std
        bb_lower = sma - BB_STD_DEV * std
        bb_width = bb_upper - bb_lower + 1e-10

        features[:, 0] = (bb_upper - close) / (bb_width + 1e-10)  # upper distance
        features[:, 1] = (sma - close) / (bb_width + 1e-10)  # middle distance
        features[:, 2] = (close - bb_lower) / (bb_width + 1e-10)  # lower distance
        # Position 0..1: where close is within bands
        features[:, 3] = np.clip(
            (close - bb_lower) / (bb_width + 1e-10), 0, 1
        )
        features[:, 4] = bb_width / (sma + 1e-10)  # band width as % of SMA

        # 2. ATR variations (5 features)
        atr = VolatilityFeatures._compute_atr(high, low, close, ATR_PERIOD)
        atr_ma = pd.Series(atr).rolling(window=10, center=False).mean().values
        atr_norm = atr / (atr_ma + 1e-10)

        features[:, 5] = atr_norm  # ATR normalized by 10-bar MA
        features[:, 6] = (atr - atr_ma) / (atr_ma + 1e-10)  # ATR deviation
        # ATR trend (is ATR increasing?)
        features[:, 7] = np.gradient(atr)  # raw gradient
        features[:, 8] = (
            np.gradient(atr, edge_order=2) < 0
        ).astype(  # ATR decreasing
            np.float32
        )
        # ATR acceleration
        atr_accel = np.gradient(np.gradient(atr, edge_order=2), edge_order=2)
        features[:, 9] = atr_accel / (np.abs(atr) + 1e-10)

        # 3. Range-based volatility (5 features)
        close_range = np.abs(np.diff(close, prepend=close[0]))
        features[:, 10] = close_range / (atr + 1e-10)  # close range vs ATR
        intrabar_range = high - low
        features[:, 11] = intrabar_range / (atr + 1e-10)  # intrabar range vs ATR

        # Short-term vs long-term std dev
        std_short = (
            pd.Series(close).rolling(window=10, center=False).std().values
        )
        std_long = pd.Series(close).rolling(window=30, center=False).std().values
        features[:, 12] = std_short / (std_long + 1e-10)
        features[:, 13] = std_long / (sma + 1e-10)
        features[:, 14] = (std_short - std_long) / (std_long + 1e-10)

        # 4. Squeeze & Expansion (5 features)
        # Bollinger Band squeeze: when width is < lower percentile
        bb_width_ma = pd.Series(bb_width).rolling(window=20, center=False).mean().values
        features[:, 15] = (bb_width - bb_width_ma) / (bb_width_ma + 1e-10)

        # Squeeze signal: width < median
        bb_median_width = np.median(bb_width)
        features[:, 16] = (bb_width < bb_median_width).astype(np.float32)

        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        features[:, 17] = tr / (close + 1e-10)

        # High-Mid-Low ratio analysis
        hl_mid = (high + low) / 2
        hml_ratio = (high - hl_mid) / (hl_mid - low + 1e-10)
        features[:, 18] = hml_ratio

        # HML trend
        hml_trend = np.gradient(hml_ratio)
        features[:, 19] = hml_trend / (np.abs(hml_ratio) + 1e-10)

        return features

    @staticmethod
    def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        atr = np.zeros_like(tr)
        
        # Handle short series where period > length
        if len(tr) < period:
            atr[:] = np.mean(tr)
            return atr
            
        atr[period - 1] = np.mean(tr[:period])

        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr


# ---------------------------------------------------------------------------
# Trend Features (30D)
# ---------------------------------------------------------------------------


class TrendFeatures:
    """Compute trend-based features.

    Returns
    -------
    Array of shape (N, 30) with normalized features:
      [rsi, rsi_norm, rsi_trend, rsi_divergence (4),
       macd_line, macd_signal, macd_histogram, macd_trend (4),
       price_slope_short, price_slope_long, slope_ratio, slope_acceleration (4),
       ema_short, ema_long, ema_ratio, ema_crossover (4),
       sma_short, sma_long, sma_ratio (3),
       hl_slope, hl_ratio (2)]
    """

    @staticmethod
    def compute(df: pd.DataFrame) -> np.ndarray:
        """Compute trend features for all bars.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close

        Returns
        -------
        np.ndarray
            Shape (len(df), 30) with float32 dtype
        """
        n = len(df)
        features = np.zeros((n, 30), dtype=np.float32)

        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)

        # 1. RSI (4 features)
        rsi = TrendFeatures._compute_rsi(close, RSI_PERIOD)
        features[:, 0] = rsi / 100.0  # normalize to [0, 1]
        features[:, 1] = (rsi - 50) / 50.0  # deviation from neutral
        features[:, 2] = np.gradient(rsi)  # RSI momentum
        # RSI divergence: when price makes new high but RSI doesn't
        price_max_20 = pd.Series(close).rolling(window=20).max().values
        rsi_ma = pd.Series(rsi).rolling(window=20).mean().values
        features[:, 3] = (close / (price_max_20 + 1e-10)) - (rsi / (rsi_ma + 1e-10))

        # 2. MACD (4 features)
        ema_fast = TrendFeatures._compute_ema(close, MACD_FAST)
        ema_slow = TrendFeatures._compute_ema(close, MACD_SLOW)
        macd = ema_fast - ema_slow
        macd_signal = TrendFeatures._compute_ema(macd, MACD_SIGNAL)
        macd_hist = macd - macd_signal

        macd_max = np.max(np.abs(macd)) + 1e-10
        features[:, 4] = macd / macd_max  # MACD line
        features[:, 5] = macd_signal / macd_max  # MACD signal
        features[:, 6] = macd_hist / macd_max  # MACD histogram
        features[:, 7] = np.gradient(macd_hist)  # MACD histogram trend

        # 3. Price slope (4 features)
        slope_short = np.polyfit(np.arange(10), close[-10:], 1)[0]  # last 10 bars
        slope_long = np.polyfit(np.arange(30), close[-30:], 1)[0]  # last 30 bars

        for i in range(len(close)):
            if i >= 10:
                features[i, 8] = np.polyfit(np.arange(10), close[i - 10 : i], 1)[0]
        for i in range(len(close)):
            if i >= 30:
                features[i, 9] = np.polyfit(np.arange(30), close[i - 30 : i], 1)[0]

        features[:, 10] = features[:, 8] / (np.abs(features[:, 9]) + 1e-10)  # ratio
        features[:, 11] = np.gradient(features[:, 8])  # acceleration

        # 4. EMA (4 features)
        ema_short = TrendFeatures._compute_ema(close, EMA_SHORT)
        ema_long = TrendFeatures._compute_ema(close, EMA_LONG)
        features[:, 12] = (close - ema_short) / (ema_short + 1e-10)  # price vs EMA short
        features[:, 13] = (close - ema_long) / (ema_long + 1e-10)  # price vs EMA long
        features[:, 14] = (ema_short - ema_long) / (ema_long + 1e-10)  # EMA ratio
        features[:, 15] = (ema_short > ema_long).astype(np.float32) * 2 - 1  # EMA crossover signal

        # 5. SMA (3 features)
        sma_short = pd.Series(close).rolling(window=10, center=False).mean().values
        sma_long = pd.Series(close).rolling(window=30, center=False).mean().values
        features[:, 16] = (close - sma_short) / (sma_short + 1e-10)
        features[:, 17] = (close - sma_long) / (sma_long + 1e-10)
        features[:, 18] = (sma_short - sma_long) / (sma_long + 1e-10)

        # 6. High-Low slope (2 features)
        hl_range = high - low
        hl_slope = np.gradient(hl_range)
        features[:, 19] = hl_slope / (hl_range + 1e-10)
        features[:, 20] = (high - low) / (close + 1e-10)  # normalized range

        # Padding to 30 features
        features[:, 21:30] = 0.0  # reserved for future trend indicators

        return features

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
        """Compute Relative Strength Index."""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period).mean().values

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_ema(close: np.ndarray, period: int) -> np.ndarray:
        """Compute Exponential Moving Average."""
        ema = np.zeros_like(close)
        ema[0] = close[0]
        multiplier = 2.0 / (period + 1)

        for i in range(1, len(close)):
            ema[i] = close[i] * multiplier + ema[i - 1] * (1 - multiplier)

        return ema


# ---------------------------------------------------------------------------
# Market Structure Features (15D)
# ---------------------------------------------------------------------------


class MarketStructureFeatures:
    """Compute market microstructure features.

    Returns
    -------
    Array of shape (N, 15) with normalized features:
      [bid_ask_imbalance, bid_ask_spread_norm, volume_profile_trend,
       liquidity_depth, price_level_strength, volume_clustering,
       trend_strength, volatility_regime, orderflow_imbalance,
       market_microstructure, momentum_volume, pullback_signal,
       gap_opening, price_memory_short, price_memory_long]
    """

    @staticmethod
    def compute(df: pd.DataFrame) -> np.ndarray:
        """Compute market structure features.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, spread_price

        Returns
        -------
        np.ndarray
            Shape (len(df), 15) with float32 dtype
        """
        n = len(df)
        features = np.zeros((n, 15), dtype=np.float32)

        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        open_ = df["open"].values.astype(np.float64)
        spread = df["spread_price"].values.astype(np.float64)

        # 1. Bid-Ask dynamics (3 features)
        spread_norm = spread / (close + 1e-10)
        features[:, 0] = spread_norm
        # Imbalance: where price lands in bar
        bar_range = high - low + 1e-10
        features[:, 1] = (close - low) / bar_range  # close position in bar
        # Spread change
        features[:, 2] = np.gradient(spread) / (spread + 1e-10)

        # 2. Volume / liquidity proxy (3 features)
        # Using intrabar range as liquidity proxy
        intrabar_vol = high - low
        intrabar_vol_ma = pd.Series(intrabar_vol).rolling(window=20).mean().values
        features[:, 3] = intrabar_vol / (intrabar_vol_ma + 1e-10)  # volume vs normal

        # Clustering: is volume contracting?
        vol_std = pd.Series(intrabar_vol).rolling(window=20).std().values
        features[:, 4] = (intrabar_vol - intrabar_vol_ma) / (vol_std + 1e-10)

        # Volume trend
        features[:, 5] = np.gradient(intrabar_vol) / (intrabar_vol + 1e-10)

        # 3. Trend strength (2 features)
        # ADX approximation using highs and lows
        plus_dm = np.where(high - np.roll(high, 1) > 0, high - np.roll(high, 1), 0)
        minus_dm = np.where(np.roll(low, 1) - low > 0, np.roll(low, 1) - low, 0)

        di_plus = (
            pd.Series(plus_dm).rolling(window=14).mean().values
            / (intrabar_vol + 1e-10)
        )
        di_minus = (
            pd.Series(minus_dm).rolling(window=14).mean().values
            / (intrabar_vol + 1e-10)
        )

        features[:, 6] = di_plus - di_minus  # trend strength
        features[:, 7] = (di_plus + di_minus) / 2.0  # ADX proxy

        # 4. Orderflow proxy (2 features)
        # Close position relative to range: up = bullish, down = bearish
        close_pos = (close - low) / bar_range
        features[:, 8] = close_pos * 2 - 1  # -1 to +1

        # Momentum: are closes clustering at extremes?
        close_cluster = pd.Series(close_pos).rolling(window=10).std().values
        features[:, 9] = close_cluster

        # 5. Pullback detection (2 features)
        # Is price pulling back from trend?
        sma_20 = pd.Series(close).rolling(window=20).mean().values
        price_trend = close - sma_20
        price_accel = np.gradient(price_trend)
        features[:, 10] = price_accel / (np.abs(price_trend) + 1e-10)

        # Pullback strength
        features[:, 11] = np.where(
            (close > sma_20) & (price_accel < 0), 1.0, 0.0
        ) - np.where((close < sma_20) & (price_accel > 0), 1.0, 0.0)

        # 6. Gap & memory (3 features)
        # Opening gap (normalized by range)
        gap = open_ - np.roll(close, 1)
        features[:, 12] = gap / (bar_range + 1e-10)

        # Price memory: correlation to short-term past
        if n >= 10:
            for i in range(10, n):
                short_mem = close[i - 5 : i]
                past_mem = close[max(0, i - 10) : max(0, i - 5)]
                if len(short_mem) == 5 and len(past_mem) == 5:
                    try:
                        corr = np.corrcoef(short_mem, past_mem)[0, 1]
                        features[i, 13] = corr if not np.isnan(corr) else 0.0
                    except:
                        features[i, 13] = 0.0

        # Price memory long-term
        if n >= 40:
            for i in range(40, n):
                long_mem = close[i - 20 : i]
                past_long = close[i - 40 : i - 20]
                if len(long_mem) == 20 and len(past_long) == 20:
                    try:
                        corr = np.corrcoef(long_mem, past_long)[0, 1]
                        features[i, 14] = corr if not np.isnan(corr) else 0.0
                    except:
                        features[i, 14] = 0.0

        return features


# ---------------------------------------------------------------------------
# Main FeatureEngineer
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """Orchestrates all feature computation and normalization."""

    @staticmethod
    def compute_all_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and normalize all feature groups.

        Parameters
        ----------
        df : pd.DataFrame
            Market data with OHLC + spread_price

        Returns
        -------
        volatility_feat : np.ndarray
            Normalized volatility features, shape (N, 20)
        trend_feat : np.ndarray
            Normalized trend features, shape (N, 30)
        market_feat : np.ndarray
            Normalized market structure features, shape (N, 15)
        """
        # Compute raw features
        volatility_feat = VolatilityFeatures.compute(df)
        trend_feat = TrendFeatures.compute(df)
        market_feat = MarketStructureFeatures.compute(df)

        # Normalize each group independently (mean=0, std=1)
        volatility_feat = FeatureEngineer._normalize(volatility_feat)
        trend_feat = FeatureEngineer._normalize(trend_feat)
        market_feat = FeatureEngineer._normalize(market_feat)

        return volatility_feat, trend_feat, market_feat

    @staticmethod
    def _normalize(features: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Normalize features to mean=0, std=1."""
        # Replace NaN with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        normalized = (features - mean) / (std + epsilon)
        # Clip to [-10, 10] to prevent extreme values
        normalized = np.clip(normalized, -10.0, 10.0)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test with synthetic data
    n_bars = 500
    t = np.arange(n_bars)
    close = 1.0 + 0.01 * np.sin(t / 50) + 0.001 * np.random.randn(n_bars)
    high = close + 0.001 * np.abs(np.random.randn(n_bars))
    low = close - 0.001 * np.abs(np.random.randn(n_bars))
    open_ = close + 0.0005 * np.random.randn(n_bars)
    spread = 0.00005 * np.ones(n_bars)

    test_df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "spread_price": spread,
    })

    vol, trend, market = FeatureEngineer.compute_all_features(test_df)

    print(f"Volatility features shape: {vol.shape}")
    print(f"Trend features shape: {trend.shape}")
    print(f"Market structure features shape: {market.shape}")
    print(f"Volatility mean: {np.mean(vol):.6f}, std: {np.std(vol):.6f}")
    print(f"Trend mean: {np.mean(trend):.6f}, std: {np.std(trend):.6f}")
    print(f"Market mean: {np.mean(market):.6f}, std: {np.std(market):.6f}")
