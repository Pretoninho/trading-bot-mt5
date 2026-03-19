"""Configuration management for trading bot"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class BotConfig:
    """Centralised configuration for the trading bot"""
    
    # ========== DATA FILES ==========
    eurusd_csv: str = "EURUSD_M1.csv"
    fxstreet_csvs: List[str] = None
    unsafe_weeks_csv: str = "unsafe_weeks.csv"
    
    # ========== ENVIRONMENT PARAMETERS ==========
    # Episode limits
    equity_gain_limit: float = 0.02      # +2%
    equity_loss_limit: float = -0.02     # -2%
    
    # Technical parameters
    atr_period: int = 14
    atr_sl_multiplier: float = 2.0
    tp1_r_multiple: float = 1.0
    tp2_r_multiple: float = 2.0
    partial_close_fraction: float = 0.25
    
    # Observation window
    observation_window: int = 64
    
    # ========== POSITION SIZING ==========
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 100.0
    risk_per_trade: float = 0.01  # 1% equity risk
    
    # ========== TRAINING PARAMETERS ==========
    initial_equity: float = 10_000.0
    learning_rate: float = 3e-4
    gamma: float = 1.0
    lambda_gae: float = 0.95
    hidden_dim: int = 128
    batch_size: int = 64
    epochs_per_update: int = 4
    max_grad_norm: float = 0.5
    
    # ========== INFERENCE PARAMETERS ==========
    checkpoint_path: Optional[str] = None
    render_mode: Optional[str] = None
    device: str = "cpu"  # "cuda" ou "cpu"
    
    # ========== TRADING GATES (on/off) ==========
    use_safe_week_gating: bool = True
    use_tradable_window_gating: bool = True
    use_breakout_detection: bool = True
    
    # ========== ADDITIONAL TUNING ==========
    invalid_action_penalty: float = -1e-4
    break_even_buffer: float = 0.0
    trailing_stop_enabled: bool = True
    
    def __post_init__(self):
        if self.fxstreet_csvs is None:
            self.fxstreet_csvs = ["fx_q1.csv", "fx_q2.csv", "fx_q3.csv", "fx_q4.csv"]
    
    def to_dict(self) -> Dict[str, Any]:
        config_dict = asdict(self)
        return config_dict
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'BotConfig':
        return BotConfig(**data)
    
    def save(self, path: str = "bot_config.json"):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def load(path: str = "bot_config.json") -> 'BotConfig':
        """Load configuration from JSON file or return defaults"""
        if Path(path).exists():
            with open(path, 'r') as f:
                return BotConfig.from_dict(json.load(f))
        return BotConfig()
