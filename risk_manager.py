"""
risk_manager.py
===============
Real-time risk management system for production trading.

Enforces:
- Position sizing limits
- Drawdown protection
- Daily loss limits
- Emergency stop conditions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path


@dataclass
class Position:
    """Represents an open position."""
    position_id: int
    entry_price: float
    entry_time: datetime
    size: float
    direction: str  # "LONG" or "SHORT"
    stop_loss: float
    take_profit: float


class RiskManager:
    """Production risk management system."""
    
    def __init__(self, initial_equity: float = 10000):
        """Initialize risk manager.
        
        Parameters
        ----------
        initial_equity : float
            Starting account equity
        """
        from DEPLOYMENT_CONFIG import RISK_CONFIG, MARKET_DATA_CONFIG
        
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.risk_config = RISK_CONFIG
        self.market_config = MARKET_DATA_CONFIG
        
        # Position tracking
        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.position_counter = 0
        
        # Session tracking
        self.session_start = datetime.now()
        self.session_pnl = 0
        self.session_max_equity = initial_equity
        
        # Daily tracking
        self.daily_start = datetime.now()
        self.daily_pnl = 0
        self.daily_max_equity = initial_equity
        
        # Violation log
        self.violations = []
    
    def check_can_open_position(self) -> Tuple[bool, str]:
        """Check if a new position can be opened.
        
        Returns
        -------
        bool
            Whether position can be opened
        str
            Reason if denied
        """
        # Check concurrent positions limit
        if len(self.open_positions) >= self.risk_config["max_concurrent_positions"]:
            return False, f"Max concurrent positions ({self.risk_config['max_concurrent_positions']}) reached"
        
        # Check daily loss limit
        daily_loss_pct = -self.daily_pnl / self.initial_equity * 100
        if daily_loss_pct > self.risk_config["max_daily_loss_pct"]:
            return False, f"Daily loss limit ({self.risk_config['max_daily_loss_pct']}%) exceeded"
        
        # Check drawdown limit
        drawdown = (self.session_max_equity - self.current_equity) / self.session_max_equity * 100
        if drawdown > self.risk_config["max_drawdown_threshold"]:
            return False, f"Drawdown limit ({self.risk_config['max_drawdown_threshold']}%) exceeded"
        
        return True, "OK"
    
    def calculate_position_sizing(
        self,
        entry_price: float,
        stop_loss: float,
        risk_pct: float = 2.0,
    ) -> float:
        """Calculate position size using risk management.
        
        Parameters
        ----------
        entry_price : float
            Entry price
        stop_loss : float
            Stop loss price
        risk_pct : float
            Risk percentage of account (default 2%)
        
        Returns
        -------
        float
            Position size in base currency units
        """
        # Risk per trade in USD
        risk_amount = self.current_equity * risk_pct / 100
        
        # Price difference
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0
        
        # Position size
        position_size = risk_amount / price_diff
        
        # Max allowed position size (in lots, convert to units)
        # Assuming 1 lot = 100,000 units for FX
        max_size = self.risk_config["max_position_size"] * 100000 / entry_price
        
        return min(position_size, max_size)
    
    def open_position(
        self,
        entry_price: float,
        direction: str,
        stop_loss: float,
        take_profit: float,
        size: Optional[float] = None,
    ) -> Tuple[bool, Optional[Position], str]:
        """Open a new position.
        
        Parameters
        ----------
        entry_price : float
            Entry price
        direction : str
            "LONG" or "SHORT"
        stop_loss : float
            Stop loss price
        take_profit : float
            Take profit price
        size : float, optional
            Position size (auto-calculated if None)
        
        Returns
        -------
        bool
            Success
        Position or None
            Opened position or None
        str
            Status message
        """
        # Validate can open
        can_open, reason = self.check_can_open_position()
        if not can_open:
            self.violations.append({
                "timestamp": datetime.now().isoformat(),
                "type": "POSITION_DENIED",
                "reason": reason,
            })
            return False, None, reason
        
        # Calculate size if not provided
        if size is None:
            size = self.calculate_position_sizing(entry_price, stop_loss, risk_pct=2.0)
        
        # Validate stop loss
        if direction == "LONG":
            if stop_loss >= entry_price:
                return False, None, "SL must be below entry for LONG"
            if take_profit <= entry_price:
                return False, None, "TP must be above entry for LONG"
        else:  # SHORT
            if stop_loss <= entry_price:
                return False, None, "SL must be above entry for SHORT"
            if take_profit >= entry_price:
                return False, None, "TP must be below entry for SHORT"
        
        # Create position
        self.position_counter += 1
        position = Position(
            position_id=self.position_counter,
            entry_price=entry_price,
            entry_time=datetime.now(),
            size=size,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        self.open_positions.append(position)
        
        return True, position, f"Position {self.position_counter} opened"
    
    def close_position(
        self,
        position_id: int,
        exit_price: float,
    ) -> Tuple[bool, float, str]:
        """Close an open position.
        
        Parameters
        ----------
        position_id : int
            Position ID to close
        exit_price : float
            Exit price
        
        Returns
        -------
        bool
            Success
        float
            PnL amount
        str
            Status message
        """
        # Find position
        position = None
        idx = None
        for i, p in enumerate(self.open_positions):
            if p.position_id == position_id:
                position = p
                idx = i
                break
        
        if position is None:
            return False, 0, f"Position {position_id} not found"
        
        # Calculate PnL
        price_diff = exit_price - position.entry_price
        if position.direction == "SHORT":
            price_diff = -price_diff
        
        pnl = price_diff * position.size
        
        # Update equity
        self.current_equity += pnl
        self.session_pnl += pnl
        self.daily_pnl += pnl
        
        # Update equity tracker
        if self.current_equity > self.session_max_equity:
            self.session_max_equity = self.current_equity
        if self.current_equity > self.daily_max_equity:
            self.daily_max_equity = self.current_equity
        
        # Move to closed positions
        position.exit_price = exit_price  # Add as attribute
        self.closed_positions.append(position)
        self.open_positions.pop(idx)
        
        return True, pnl, f"Position {position_id} closed with PnL: ${pnl:+.2f}"
    
    def check_risk_violations(self) -> Dict:
        """Check for active risk violations.
        
        Returns
        -------
        dict
            Violation details
        """
        violations = {
            "stop_loss_hits": [],
            "take_profit_hits": [],
            "equity_alerts": [],
            "emergency_stop": False,
        }
        
        # Check open positions for SL/TP hits (mock current prices)
        for position in self.open_positions:
            current_price = position.entry_price + np.random.randn() * 0.001  # Mock
            
            if position.direction == "LONG":
                if current_price <= position.stop_loss:
                    violations["stop_loss_hits"].append(position.position_id)
                if current_price >= position.take_profit:
                    violations["take_profit_hits"].append(position.position_id)
            else:  # SHORT
                if current_price >= position.stop_loss:
                    violations["stop_loss_hits"].append(position.position_id)
                if current_price <= position.take_profit:
                    violations["take_profit_hits"].append(position.position_id)
        
        # Check equity alerts
        daily_loss_pct = -self.daily_pnl / self.initial_equity * 100
        if daily_loss_pct > self.risk_config["max_daily_loss_pct"] * 0.8:  # 80% of limit
            violations["equity_alerts"].append(f"Daily loss at {daily_loss_pct:.1f}% (limit: {self.risk_config['max_daily_loss_pct']}%)")
        
        # Check drawdown
        drawdown = (self.session_max_equity - self.current_equity) / self.session_max_equity * 100
        if drawdown > self.risk_config["max_drawdown_threshold"] * 0.8:
            violations["equity_alerts"].append(f"Drawdown at {drawdown:.1f}% (limit: {self.risk_config['max_drawdown_threshold']}%)")
        
        # Emergency stop condition
        if (daily_loss_pct > self.risk_config["max_daily_loss_pct"] or 
            drawdown > self.risk_config["max_drawdown_threshold"]):
            violations["emergency_stop"] = True
        
        return violations
    
    def get_status(self) -> Dict:
        """Get current risk manager status.
        
        Returns
        -------
        dict
            Status summary
        """
        daily_loss_pct = -self.daily_pnl / self.initial_equity * 100
        session_loss_pct = -self.session_pnl / self.initial_equity * 100
        drawdown = (self.session_max_equity - self.current_equity) / self.session_max_equity * 100
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_equity": self.current_equity,
            "session_pnl": self.session_pnl,
            "session_pnl_pct": session_loss_pct,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": daily_loss_pct,
            "max_drawdown": drawdown,
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions),
            "daily_loss_limit_remaining_pct": self.risk_config["max_daily_loss_pct"] - daily_loss_pct,
            "drawdown_limit_remaining_pct": self.risk_config["max_drawdown_threshold"] - drawdown,
        }


def main():
    """Demonstrate risk manager."""
    print("\n" + "=" * 80)
    print("RISK MANAGER - SYSTEM TEST")
    print("=" * 80)
    
    manager = RiskManager(initial_equity=10000)
    
    print("\n1. Testing position opening...")
    can_open, reason = manager.check_can_open_position()
    print(f"   Can open position: {can_open} ({reason})")
    
    print("\n2. Opening first position...")
    success, position, msg = manager.open_position(
        entry_price=1.0700,
        direction="LONG",
        stop_loss=1.0695,
        take_profit=1.0710,
    )
    print(f"   {msg}")
    if position:
        print(f"   Position size: {position.size:.2f}")
    
    print("\n3. Calculating position sizing...")
    size = manager.calculate_position_sizing(
        entry_price=1.0700,
        stop_loss=1.0690,
        risk_pct=2.0,
    )
    print(f"   Calculated size: {size:.2f} base units for 2% risk")
    
    print("\n4. Closing position...")
    if position:
        success, pnl, msg = manager.close_position(
            position_id=position.position_id,
            exit_price=1.0705,
        )
        print(f"   {msg}")
    
    print("\n5. Risk status...")
    status = manager.get_status()
    print(f"   Current equity: ${status['current_equity']:,.2f}")
    print(f"   Open positions: {status['open_positions']}")
    print(f"   Session P&L: ${status['session_pnl']:+,.2f}")
    print(f"   Max drawdown: {status['max_drawdown']:.2f}%")
    
    print("\n6. Risk violations...")
    violations = manager.check_risk_violations()
    print(f"   Emergency stop: {violations['emergency_stop']}")
    print(f"   S/L hits: {len(violations['stop_loss_hits'])}")
    print(f"   T/P hits: {len(violations['take_profit_hits'])}")
    
    print("\n✓ Risk manager test complete")


if __name__ == "__main__":
    main()
