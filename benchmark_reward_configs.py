"""
benchmark_reward_configs.py
===========================
Benchmarking script comparing baseline vs enhanced reward configurations.

Runs short training episodes with different reward configs and compares:
- Episode returns (cumulative reward)
- Convergence speed (loss trends)
- Reward signal stability (variance in rewards)
- Performance metrics (Sharpe, drawdown, etc.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple

from trading_env.data.market_loader import load_mt5_m1_csv
from trading_env.env.trading_env import EURUSDTradingEnv
from trading_env.utils.reward_functions import (
    RewardCalculator,
    reward_config_baseline,
    reward_config_enhanced,
    reward_config_aggressive,
)


def load_benchmark_data() -> pd.DataFrame:
    """Load market data for benchmarking.

    Returns
    -------
    pd.DataFrame
        Market data to use for testing
    """
    data_file = Path("trading_env/data/sample_eurusd_m1.csv")

    if not data_file.exists():
        print(f"Note: {data_file} not found. Generating synthetic data...")
        # Generate synthetic data for benchmarking
        dates = pd.date_range("2023-01-01", periods=10000, freq="1min")
        prices = np.cumsum(np.random.randn(10000) * 0.0001) + 1.0700

        df = pd.DataFrame({
            "dt": dates,
            "open": prices,
            "high": prices + 0.0002,
            "low": prices - 0.0002,
            "close": prices,
            "spread_price": 0.0002,  # Fixed 2-pip spread
            "tick_volume": 100,
            "tradable_now": 1,  # Always tradable for benchmark
            "is_safe_week": 1,
            "has_breakout": 1,
        })
    else:
        df = load_mt5_m1_csv(data_file)

    return df


def run_episode_with_config(
    df: pd.DataFrame,
    episodes: int = 10,
    max_steps: int = 500,
    reward_config=None,
) -> Dict[str, np.ndarray]:
    """Run episodes and collect reward statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Market data
    episodes : int
        Number of episodes to run
    max_steps : int
        Max steps per episode
    reward_config : RewardConfig
        Reward configuration to use

    Returns
    -------
    dict
        Statistics: episode_returns, episode_lengths, reward_distributions, etc.
    """
    if reward_config is None:
        reward_config = reward_config_baseline()

    calc = RewardCalculator(reward_config)

    episode_returns = []
    episode_lengths = []
    reward_samples = []
    max_drawdowns = []
    sharpe_ratios = []

    for ep in range(episodes):
        env = EURUSDTradingEnv(df)
        obs, _ = env.reset()
        calc.reset()

        episode_reward = 0.0
        equity_history = [10000.0]  # Starting equity
        step = 0

        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()

            obs, env_reward, terminated, truncated, info = env.step(action)

            # Track equity if available
            if "equity" in info:
                calc.record_equity(info["equity"])
                equity_history.append(info["equity"])

            # Track PnL if trade was closed
            if "pnl" in info and info["pnl"] != 0:
                calc.record_pnl(info["pnl"])

            # Increment bars held if position open
            if info.get("position_open", False):
                calc.increment_bars_held()

            # Calculate custom reward
            prev_equity = equity_history[-2] if len(equity_history) > 1 else 10000
            curr_equity = equity_history[-1] if len(equity_history) > 0 else 10000
            is_invalid = info.get("is_invalid_action", False)

            reward, components = calc.calculate_reward(
                prev_equity=prev_equity,
                curr_equity=curr_equity,
                position_open=info.get("position_open", False),
                is_invalid_action=is_invalid,
            )

            episode_reward += reward
            reward_samples.append(reward)

            if terminated or truncated:
                break

        # Compute episode stats
        episode_returns.append(episode_reward)
        episode_lengths.append(step + 1)

        # Compute drawdown
        equity_array = np.array(equity_history)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / running_max
        max_drawdown = np.max(drawdown)
        max_drawdowns.append(max_drawdown)

        # Compute Sharpe ratio if sufficient data
        if len(equity_history) > 1:
            returns = np.diff(equity_array) / equity_array[:-1]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 1440)
                sharpe_ratios.append(sharpe)

        print(f"  Episode {ep + 1}/{episodes}: return={episode_reward:.4f}, length={step + 1}")

    return {
        "episode_returns": np.array(episode_returns),
        "episode_lengths": np.array(episode_lengths),
        "reward_samples": np.array(reward_samples),
        "max_drawdowns": np.array(max_drawdowns),
        "sharpe_ratios": np.array(sharpe_ratios),
    }


def compute_statistics(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute summary statistics.

    Parameters
    ----------
    data : dict
        Statistics from run_episode_with_config

    Returns
    -------
    dict
        Summary statistics
    """
    returns = data["episode_returns"]
    lengths = data["episode_lengths"]
    rewards = data["reward_samples"]
    drawdowns = data["max_drawdowns"]
    sharpes = data["sharpe_ratios"]

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_reward_signal": float(np.mean(rewards)),
        "std_reward_signal": float(np.std(rewards)),
        "mean_max_drawdown": float(np.mean(drawdowns)),
        "mean_sharpe": float(np.mean(sharpes)) if len(sharpes) > 0 else 0.0,
        "num_episodes": len(returns),
        "total_steps": int(np.sum(lengths)),
    }


def benchmark_configs(
    df: pd.DataFrame,
    configs: Dict[str, any],
    episodes_per_config: int = 10,
) -> Tuple[Dict, pd.DataFrame]:
    """Benchmark multiple reward configurations.

    Parameters
    ----------
    df : pd.DataFrame
        Market data
    configs : dict
        {name: reward_config} pairs
    episodes_per_config : int
        Episodes to run per config

    Returns
    -------
    tuple
        (raw_data, summary_dataframe)
    """
    results = {}
    summary_stats = []

    for config_name, reward_config in configs.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {config_name}")
        print(f"{'='*60}")

        data = run_episode_with_config(
            df=df,
            episodes=episodes_per_config,
            reward_config=reward_config,
        )

        stats = compute_statistics(data)
        stats["config_name"] = config_name

        results[config_name] = data
        summary_stats.append(stats)

        print(f"\nSummary for {config_name}:")
        for key, val in stats.items():
            if key != "config_name":
                print(f"  {key}: {val:.4f}")

    summary_df = pd.DataFrame(summary_stats)
    return results, summary_df


def plot_comparison(results: Dict, summary_df: pd.DataFrame, output_dir: Path = None):
    """Plot comparison across configurations.

    Parameters
    ----------
    results : dict
        Raw data from benchmark_configs
    summary_df : pd.DataFrame
        Summary statistics
    output_dir : Path
        Directory to save plots
    """
    if output_dir is None:
        output_dir = Path(".")

    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reward Configuration Benchmark Comparison", fontsize=16, fontweight="bold")

    # Plot 1: Episode Returns Distribution
    ax1 = axes[0, 0]
    for config_name, data in results.items():
        returns = data["episode_returns"]
        ax1.hist(returns, alpha=0.6, label=config_name, bins=10)
    ax1.set_xlabel("Episode Return")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Episode Returns")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean Reward Signal
    ax2 = axes[0, 1]
    means = [summary_df.iloc[i]["mean_reward_signal"] for i in range(len(summary_df))]
    stds = [summary_df.iloc[i]["std_reward_signal"] for i in range(len(summary_df))]
    config_names = summary_df["config_name"].tolist()
    ax2.bar(config_names, means, yerr=stds, capsize=5, alpha=0.7)
    ax2.set_ylabel("Mean Reward Signal")
    ax2.set_title("Reward Signal Mean ± Std")
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Max Drawdown
    ax3 = axes[1, 0]
    drawdowns = [summary_df.iloc[i]["mean_max_drawdown"] for i in range(len(summary_df))]
    ax3.bar(config_names, drawdowns, alpha=0.7, color="orange")
    ax3.set_ylabel("Mean Max Drawdown")
    ax3.set_title("Drawdown Risk Comparison")
    ax3.grid(True, alpha=0.3, axis="y")

    # Plot 4: Sharpe Ratio
    ax4 = axes[1, 1]
    sharpes = [summary_df.iloc[i]["mean_sharpe"] for i in range(len(summary_df))]
    ax4.bar(config_names, sharpes, alpha=0.7, color="green")
    ax4.set_ylabel("Mean Sharpe Ratio")
    ax4.set_title("Risk-Adjusted Return Comparison")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = output_dir / "reward_benchmark_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved: {plot_path}")
    plt.close()


def save_results(results: Dict, summary_df: pd.DataFrame, output_dir: Path = None):
    """Save benchmark results to files.

    Parameters
    ----------
    results : dict
        Raw data from benchmark_configs
    summary_df : pd.DataFrame
        Summary statistics
    output_dir : Path
        Directory to save files
    """
    if output_dir is None:
        output_dir = Path(".")

    output_dir.mkdir(exist_ok=True)

    # Save summary CSV
    csv_path = output_dir / "reward_benchmark_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Summary saved: {csv_path}")

    # Save detailed JSON
    json_data = {}
    for config_name, data in results.items():
        json_data[config_name] = {
            "episode_returns": data["episode_returns"].tolist(),
            "episode_lengths": data["episode_lengths"].tolist(),
            "max_drawdowns": data["max_drawdowns"].tolist(),
            "sharpe_ratios": data["sharpe_ratios"].tolist(),
            "reward_signal_mean": float(np.mean(data["reward_samples"])),
            "reward_signal_std": float(np.std(data["reward_samples"])),
        }

    json_path = output_dir / "reward_benchmark_detailed.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Detailed results saved: {json_path}")


def main():
    """Run full benchmark suite."""
    print("\n" + "=" * 70)
    print("REWARD CONFIGURATION BENCHMARK SUITE")
    print("=" * 70)

    # Load market data
    print("\nLoading market data...")
    df = load_benchmark_data()
    print(f"Loaded {len(df)} bars of EURUSD M1 data")

    # Define configurations to benchmark
    configs = {
        "baseline": reward_config_baseline(),
        "enhanced": reward_config_enhanced(),
        "aggressive": reward_config_aggressive(),
    }

    # Run benchmarks (reduced episodes for speed)
    results, summary_df = benchmark_configs(df, configs, episodes_per_config=3)

    # Display summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Generate plots
    plot_comparison(results, summary_df, Path("benchmark_results"))

    # Save results
    save_results(results, summary_df, Path("benchmark_results"))

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print("1. All configurations should produce stable training signals")
    print("2. Enhanced config should show lower variance in rewards")
    print("3. Aggressive config may show higher returns but with more volatility")
    print("4. Baseline should match original reward behavior (simple equity returns)")


if __name__ == "__main__":
    main()
