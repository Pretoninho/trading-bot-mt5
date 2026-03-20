"""
DEPLOYMENT_CONFIG.py
====================
Production deployment configuration for trading bot with reward-optimized agent.

Defines:
- Production reward configurations
- Agent initialization parameters
- Logging and monitoring setup
- Risk management settings
- Integration checkpoints
"""

from trading_env.utils.reward_functions import (
    reward_config_baseline,
    reward_config_enhanced,
    reward_config_aggressive,
    RewardConfig,
)


# ============================================================================
# PRODUCTION REWARD PROFILES
# ============================================================================

REWARD_PROFILES = {
    "conservative": reward_config_baseline(),
    "balanced": reward_config_enhanced(),
    "aggressive": reward_config_aggressive(),
}


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

AGENT_CONFIG = {
    "observation_dim": 456,  # 391 base + 65 engineered features
    "action_dim": 6,  # HOLD, OPEN_LONG, OPEN_SHORT, CLOSE, PROTECT, MANAGE_TP
    "hidden_dim": 256,  # Increased for production
    "learning_rate": 1e-4,  # Reduced for stability
    "gamma": 0.99,  # Standard discount factor
    "lambda_": 0.95,  # GAE parameter
    "clip_ratio": 0.2,  # PPO clipping
    "entropy_coef": 0.01,  # Exploration encouragement
    "critic_coef": 0.5,  # Critic loss weight
    "max_grad_norm": 1.0,  # Gradient clipping
    "epochs": 10,  # More training per update
    "batch_size": 256,  # Larger batches for stability
    "device": "cuda",  # Use GPU if available
}


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    "num_episodes": 100,  # Production training episodes
    "update_interval": 5,  # Update every N episodes
    "save_interval": 10,  # Save checkpoint every N episodes
    "eval_interval": 20,  # Evaluation every N episodes
    "max_steps_per_episode": 1440,  # Full trading day (1440 min M1)
    "reward_profile": "balanced",  # Which reward config to use
    "enable_tensorboard": True,
    "log_dir": "./logs/production",
    "checkpoint_dir": "./checkpoints/production",
}


# ============================================================================
# RISK MANAGEMENT
# ============================================================================

RISK_CONFIG = {
    "max_position_size": 2.0,  # Max lots per trade
    "max_daily_loss_pct": 5.0,  # Stop trading if down 5%
    "max_daily_gain_pct": 10.0,  # Stop trading if up 10%
    "max_concurrent_positions": 1,  # Only one position at a time
    "max_holding_time_bars": 1440,  # Max 1 day holding
    "min_trade_profit_threshold": 10.0,  # Min profit in USD
    "max_drawdown_threshold": 15.0,  # Max equity drawdown %
}


# ============================================================================
# MARKET DATA CONFIGURATION
# ============================================================================

MARKET_DATA_CONFIG = {
    "timeframe": "M1",  # 1-minute candles
    "symbol": "EURUSD",
    "trading_hours_start": 8,  # UTC 08:00
    "trading_hours_end": 22,  # UTC 22:00
    "enable_news_filter": True,  # Avoid high-impact news
    "data_source": "mt5",  # MetaTrader 5
    "data_buffer_size": 10000,  # Max bars in memory
    "feature_computation": True,  # Compute 65D engineered features
}


# ============================================================================
# MONITORING & LOGGING
# ============================================================================

MONITORING_CONFIG = {
    "enable_metrics_logging": True,
    "metrics_interval_seconds": 60,  # Log every 60s
    "enable_performance_tracking": True,
    "enable_equity_tracking": True,
    "log_level": "INFO",
    "alert_threshold_drawdown": 10.0,  # Alert if drawdown > 10%
    "alert_threshold_loss": -2.0,  # Alert if loss > -2%
    "slack_notifications": False,  # Optional Slack integration
    "email_notifications": False,  # Optional email alerts
}


# ============================================================================
# MODEL VALIDATION CHECKPOINTS
# ============================================================================

VALIDATION_CHECKPOINTS = {
    "pre_deployment": {
        "min_training_episodes": 50,
        "min_avg_reward": -0.02,  # Not too negative
        "max_reward_std": 0.1,  # Reasonably stable
        "min_actor_convergence": True,
        "check_nan_inf": True,
    },
    "early_stopping": {
        "patience_episodes": 20,  # Stop if no improvement for 20 episodes
        "min_improvement_rate": 0.001,  # Minimum improvement per episode
    },
    "production": {
        "max_inference_latency_ms": 100,  # Action selection < 100ms
        "min_sharpe_ratio": -1.0,  # Not too negative
        "max_consecutive_losses": 5,  # Stop after 5 losses
    },
}


# ============================================================================
# REPRODUCIBILITY & VERSIONING
# ============================================================================

VERSION_CONFIG = {
    "agent_version": "v1.0",
    "reward_system_version": "v1.0",
    "features_version": "v1.0",
    "model_hash": None,  # Set after training
    "deployment_date": None,  # Set at deployment
    "git_commit": None,  # Set from git
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_agent_config(profile="balanced"):
    """Get complete agent configuration.
    
    Parameters
    ----------
    profile : str
        Reward profile: conservative, balanced, or aggressive
    
    Returns
    -------
    dict
        Complete agent configuration
    """
    config = AGENT_CONFIG.copy()
    config["reward_config"] = REWARD_PROFILES.get(profile, REWARD_PROFILES["balanced"])
    return config


def validate_production_config():
    """Validate that all production settings are appropriate.
    
    Returns
    -------
    dict
        Validation results
    """
    results = {
        "valid": True,
        "warnings": [],
        "errors": [],
    }
    
    # Check reward config exists
    if TRAINING_CONFIG["reward_profile"] not in REWARD_PROFILES:
        results["errors"].append(f"Unknown reward profile: {TRAINING_CONFIG['reward_profile']}")
        results["valid"] = False
    
    # Check observation dimension
    if AGENT_CONFIG["observation_dim"] != 456:
        results["warnings"].append(f"Non-standard observation_dim: {AGENT_CONFIG['observation_dim']}")
    
    # Check learning rate
    if AGENT_CONFIG["learning_rate"] > 1e-3:
        results["warnings"].append("Learning rate may be too high for stability")
    
    # Check training duration
    if TRAINING_CONFIG["num_episodes"] < 20:
        results["warnings"].append("Training episodes may be insufficient for convergence")
    
    # Check risk limits
    if RISK_CONFIG["max_daily_loss_pct"] < 1.0:
        results["warnings"].append("Very tight daily loss limit may be too restrictive")
    
    return results


def print_deployment_report():
    """Print deployment readiness report."""
    print("\n" + "="*80)
    print("PRODUCTION DEPLOYMENT CONFIGURATION REPORT")
    print("="*80)
    
    validation = validate_production_config()
    
    print("\n📊 AGENT CONFIGURATION:")
    print(f"  Observation Dimension: {AGENT_CONFIG['observation_dim']}")
    print(f"  Action Dimension: {AGENT_CONFIG['action_dim']}")
    print(f"  Hidden Size: {AGENT_CONFIG['hidden_dim']}")
    print(f"  Learning Rate: {AGENT_CONFIG['learning_rate']}")
    print(f"  Device: {AGENT_CONFIG['device']}")
    
    print("\n📈 TRAINING CONFIGURATION:")
    print(f"  Target Episodes: {TRAINING_CONFIG['num_episodes']}")
    print(f"  Reward Profile: {TRAINING_CONFIG['reward_profile']}")
    print(f"  Update Interval: {TRAINING_CONFIG['update_interval']} episodes")
    print(f"  Save Interval: {TRAINING_CONFIG['save_interval']} episodes")
    
    print("\n⚖️  RISK MANAGEMENT:")
    print(f"  Max Position Size: {RISK_CONFIG['max_position_size']} lots")
    print(f"  Max Daily Loss: {RISK_CONFIG['max_daily_loss_pct']}%")
    print(f"  Max Daily Gain: {RISK_CONFIG['max_daily_gain_pct']}%")
    print(f"  Max Drawdown: {RISK_CONFIG['max_drawdown_threshold']}%")
    
    print("\n📊 MARKET DATA:")
    print(f"  Symbol: {MARKET_DATA_CONFIG['symbol']}")
    print(f"  Timeframe: {MARKET_DATA_CONFIG['timeframe']}")
    print(f"  Trading Hours: {MARKET_DATA_CONFIG['trading_hours_start']:02d}:00 - {MARKET_DATA_CONFIG['trading_hours_end']:02d}:00 UTC")
    print(f"  Features: {MARKET_DATA_CONFIG['feature_computation']} ({65} engineered features)")
    
    print("\n✅ VALIDATION STATUS:")
    if validation["valid"]:
        print("  ✓ Configuration is VALID")
    else:
        print("  ✗ Configuration has ERRORS:")
        for err in validation["errors"]:
            print(f"    - {err}")
    
    if validation["warnings"]:
        print("  ⚠ Warnings:")
        for warn in validation["warnings"]:
            print(f"    - {warn}")
    
    print("\n" + "="*80)
    
    return validation["valid"]


if __name__ == "__main__":
    print_deployment_report()
