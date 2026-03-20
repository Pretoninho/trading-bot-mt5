"""
production_monitor.py
====================
Real-time production monitoring and risk management dashboard.

Monitors:
- Equity levels and drawdowns
- Trade performance metrics
- Risk limit violations
- Agent convergence
- System health
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class ProductionMonitor:
    """Real-time production monitoring system."""
    
    def __init__(self, log_dir: str = "./logs/production"):
        """Initialize monitor.
        
        Parameters
        ----------
        log_dir : str
            Directory for monitoring logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session state
        self.session_start = datetime.now()
        self.trades = []
        self.equity_history = []
        self.alerts = []
        
        # Load configuration
        from DEPLOYMENT_CONFIG import RISK_CONFIG, MONITORING_CONFIG, TRAINING_CONFIG
        self.risk_config = RISK_CONFIG
        self.monitoring_config = MONITORING_CONFIG
        self.training_config = TRAINING_CONFIG
    
    def record_trade(
        self,
        trade_id: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        bars_held: int,
    ) -> None:
        """Record completed trade.
        
        Parameters
        ----------
        trade_id : int
            Unique trade identifier
        entry_price : float
            Entry price
        exit_price : float
            Exit price
        pnl : float
            Profit/loss in USD
        bars_held : int
            Number of bars position was held
        """
        trade = {
            "timestamp": datetime.now().isoformat(),
            "trade_id": trade_id,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": (exit_price - entry_price) / entry_price * 100,
            "bars_held": bars_held,
            "duration_minutes": bars_held,  # M1 candles
        }
        self.trades.append(trade)
    
    def record_equity(self, equity: float) -> None:
        """Record equity level.
        
        Parameters
        ----------
        equity : float
            Current account equity
        """
        self.equity_history.append({
            "timestamp": datetime.now().isoformat(),
            "equity": equity,
        })
    
    def check_risk_limits(self, equity: float, initial_equity: float = 10000) -> Dict:
        """Check if risk limits are violated.
        
        Parameters
        ----------
        equity : float
            Current equity
        initial_equity : float
            Starting equity
        
        Returns
        -------
        dict
            Risk check results
        """
        results = {
            "violations": [],
            "warnings": [],
            "ok": True,
        }
        
        daily_return = (equity - initial_equity) / initial_equity * 100
        
        # Check daily loss limit
        if daily_return < -self.risk_config["max_daily_loss_pct"]:
            results["violations"].append(
                f"Daily loss limit exceeded: {daily_return:.2f}% (limit: -{self.risk_config['max_daily_loss_pct']}%)"
            )
            results["ok"] = False
            self.alerts.append({
                "timestamp": datetime.now().isoformat(),
                "severity": "CRITICAL",
                "message": f"STOP TRADING: Daily loss limit exceeded ({daily_return:.2f}%)",
            })
        
        # Check daily gain limit
        if daily_return > self.risk_config["max_daily_gain_pct"]:
            results["warnings"].append(
                f"Daily gain target reached: {daily_return:.2f}% (target: {self.risk_config['max_daily_gain_pct']}%)"
            )
            self.alerts.append({
                "timestamp": datetime.now().isoformat(),
                "severity": "WARNING",
                "message": f"Daily gain target reached ({daily_return:.2f}%)",
            })
        
        # Check drawdown limits
        if len(self.equity_history) > 1:
            equities = [e["equity"] for e in self.equity_history]
            max_equity = np.max(equities)
            curr_equity = equities[-1]
            drawdown_pct = (max_equity - curr_equity) / max_equity * 100
            
            if drawdown_pct > self.risk_config["max_drawdown_threshold"]:
                results["violations"].append(
                    f"Drawdown limit exceeded: {drawdown_pct:.2f}% (limit: {self.risk_config['max_drawdown_threshold']}%)"
                )
                results["ok"] = False
                self.alerts.append({
                    "timestamp": datetime.now().isoformat(),
                    "severity": "CRITICAL",
                    "message": f"STOP TRADING: Drawdown limit exceeded ({drawdown_pct:.2f}%)",
                })
        
        return results
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics from trades.
        
        Returns
        -------
        dict
            Performance summary
        """
        if not self.trades:
            return {"trades_total": 0, "win_rate": 0, "avg_pnl": 0}
        
        trades_df = pd.DataFrame(self.trades)
        
        winning_trades = len(trades_df[trades_df["pnl"] > 0])
        losing_trades = len(trades_df[trades_df["pnl"] < 0])
        total_trades = len(trades_df)
        
        return {
            "trades_total": total_trades,
            "trades_winning": winning_trades,
            "trades_losing": losing_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "avg_pnl": float(trades_df["pnl"].mean()),
            "total_pnl": float(trades_df["pnl"].sum()),
            "avg_bars_held": float(trades_df["bars_held"].mean()),
            "min_pnl": float(trades_df["pnl"].min()),
            "max_pnl": float(trades_df["pnl"].max()),
            "std_pnl": float(trades_df["pnl"].std()),
        }
    
    def print_status_report(self, equity: float, initial_equity: float = 10000) -> str:
        """Generate status report.
        
        Parameters
        ----------
        equity : float
            Current equity
        initial_equity : float
            Starting equity
        
        Returns
        -------
        str
            Formatted report
        """
        report_lines = [
            "\n" + "=" * 80,
            "PRODUCTION MONITORING - STATUS REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session Duration: {datetime.now() - self.session_start}",
        ]
        
        # Equity section
        daily_return = (equity - initial_equity) / initial_equity * 100
        report_lines.append("\n📊 EQUITY STATUS:")
        report_lines.append(f"  Starting Equity: ${initial_equity:,.2f}")
        report_lines.append(f"  Current Equity:  ${equity:,.2f}")
        report_lines.append(f"  Daily Return:    {daily_return:+.2f}%")
        
        # Risk check
        risk_check = self.check_risk_limits(equity, initial_equity)
        report_lines.append("\n⚠️  RISK STATUS:")
        if risk_check["ok"]:
            report_lines.append("  ✓ All risk limits OK")
        else:
            report_lines.append("  ✗ VIOLATIONS DETECTED:")
            for violation in risk_check["violations"]:
                report_lines.append(f"    - {violation}")
        
        if risk_check["warnings"]:
            report_lines.append("  ⚠ WARNINGS:")
            for warning in risk_check["warnings"]:
                report_lines.append(f"    - {warning}")
        
        # Performance metrics
        metrics = self.get_performance_metrics()
        if metrics["trades_total"] > 0:
            report_lines.append("\n📈 PERFORMANCE METRICS:")
            report_lines.append(f"  Total Trades:      {metrics['trades_total']}")
            report_lines.append(f"  Winning Trades:    {metrics['trades_winning']}")
            report_lines.append(f"  Losing Trades:     {metrics['trades_losing']}")
            report_lines.append(f"  Win Rate:          {metrics['win_rate']*100:.1f}%")
            report_lines.append(f"  Avg P&L:           ${metrics['avg_pnl']:+,.2f}")
            report_lines.append(f"  Total P&L:         ${metrics['total_pnl']:+,.2f}")
            report_lines.append(f"  Avg Bars Held:     {metrics['avg_bars_held']:.0f}")
        
        # Recent alerts
        if self.alerts:
            report_lines.append("\n🚨 RECENT ALERTS:")
            for alert in self.alerts[-5:]:  # Last 5 alerts
                report_lines.append(f"  [{alert['severity']:8s}] {alert['message']}")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)
    
    def save_session_logs(self) -> Path:
        """Save session logs to file.
        
        Returns
        -------
        Path
            Path to saved log file
        """
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "session_start": self.session_start.isoformat(),
            "duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "trades": self.trades,
            "equity_history": self.equity_history,
            "alerts": self.alerts,
            "metrics": self.get_performance_metrics(),
        }
        
        log_file = self.log_dir / f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w") as f:
            json.dump(session_data, f, indent=2, default=str)
        
        return log_file


def main():
    """Demonstrate monitoring system."""
    print("\n" + "=" * 80)
    print("PRODUCTION MONITORING - SYSTEM TEST")
    print("=" * 80)
    
    monitor = ProductionMonitor()
    
    # Simulate trading session
    print("\nSimulating trading session...")
    
    initial_equity = 10000
    equity = initial_equity
    
    # Simulate 10 trades
    for i in range(10):
        entry_price = 1.0700 + np.random.randn() * 0.001
        bars_held = np.random.randint(5, 240)
        pnl = np.random.randn() * 100 + 50
        exit_price = entry_price + pnl / 100000  # Scale for currency
        
        monitor.record_trade(i, entry_price, exit_price, pnl, bars_held)
        
        equity += pnl
        monitor.record_equity(equity)
        
        print(f"  Trade {i+1}: PnL=${pnl:+.2f}")
    
    # Print status report
    report = monitor.print_status_report(equity, initial_equity)
    print(report)
    
    # Save logs
    log_file = monitor.save_session_logs()
    print(f"\n✓ Session logs saved to: {log_file}")
    
    # Verify monitoring config
    print("\nMonitoring Configuration:")
    print(f"  Equity check interval: {monitor.monitoring_config['metrics_interval_seconds']}s")
    print(f"  Drawdown alert threshold: {monitor.risk_config['max_drawdown_threshold']}%")
    print(f"  Daily loss limit: {monitor.risk_config['max_daily_loss_pct']}%")


if __name__ == "__main__":
    main()
