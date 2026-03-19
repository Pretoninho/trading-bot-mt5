"""
Example: Training with bot_config

Demonstrates how to load and use BotConfig in training scripts.
"""

import sys
import os
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from bot_config import BotConfig
from trading_env.data.market_loader import load_mt5_m1_csv
from trading_env.gating.breakout import compute_week_breakout, add_breakout_flag
from trading_env.gating.tradable_window import add_tradable_flag
from trading_env.env.trading_env import EURUSDTradingEnv


def main():
    """Training example with BotConfig"""
    
    # ==== STEP 1: Load configuration ====
    print("📋 Loading configuration...")
    config = BotConfig.load("bot_config.json")
    
    print(f"  Initial equity: ${config.initial_equity:,.0f}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Risk per trade: {config.risk_per_trade * 100:.1f}%")
    print(f"  Device: {config.device}")
    
    # ==== STEP 2: Load market data ====
    print("\n📊 Loading market data...")
    if not Path(config.eurusd_csv).exists():
        print(f"❌ Market data file not found: {config.eurusd_csv}")
        print("   Please upload via interface first!")
        return
    
    df = load_mt5_m1_csv(
        config.eurusd_csv,
        unsafe_weeks_path=config.unsafe_weeks_csv if Path(config.unsafe_weeks_csv).exists() else None
    )
    
    print(f"  Loaded {len(df)} bars")
    print(f"  Date range: {df['dt'].min()} to {df['dt'].max()}")
    
    # ==== STEP 3: Add gating flags ====
    print("\n🎯 Adding trading gates...")
    
    if config.use_breakout_detection:
        print("  ✓ Computing breakout flags...")
        week_meta = compute_week_breakout(df)
        df = add_breakout_flag(df, week_meta=week_meta)
    else:
        df["has_breakout"] = 0
    
    if config.use_tradable_window_gating:
        print("  ✓ Adding tradable window...")
        df = add_tradable_flag(df)
    else:
        df["tradable_now"] = 1
    
    print(f"  Safe weeks gating: {'✓' if config.use_safe_week_gating else '✗'}")
    
    # ==== STEP 4: Create environment ====
    print("\n🎮 Creating environment...")
    env = EURUSDTradingEnv(
        df,
        initial_equity=config.initial_equity,
        be_buffer=config.break_even_buffer,
        render_mode=config.render_mode,
    )
    
    print(f"  Observation size: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n} actions")
    
    # ==== STEP 5: Quick test ====
    print("\n🧪 Running quick smoke test (1 episode)...")
    
    obs, info = env.reset(options={"day_date": date(2025, 1, 8)})
    
    total_reward = 0
    step_count = 0
    terminated = False
    
    while not terminated and step_count < 1440:  # Max 1 day
        action = env.action_space.sample()  # Random policy
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
    
    print(f"  ✓ Episode complete: {step_count} steps")
    print(f"  Total reward: {total_reward:.6f}")
    print(f"  Final equity: ${info['equity']:,.2f}")
    
    # ==== SUMMARY ====
    print("\n✅ Configuration loaded successfully!")
    print("\n💡 Next steps:")
    print("   1. Modify config via: streamlit run bot_config_interface.py")
    print("   2. Or edit bot_config.json directly")
    print("   3. Create training loop with PPOAgentPyTorch")
    print("   4. Save checkpoints to: checkpoints/ppo_agent.pt")


if __name__ == "__main__":
    main()
