"""
monitoring_dashboard.py
========================
Integrated production monitoring dashboard.

Combines:
- Production monitor
- Risk manager
- Metrics collector
- Real-time alerts
"""

from production_monitor import ProductionMonitor
from risk_manager import RiskManager
from metrics_collector import MetricsCollector, RealtimeMetricsMonitor
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict


class IntegratedDashboard:
    """Complete production monitoring dashboard."""
    
    def __init__(self, initial_equity: float = 10000, log_dir: str = "./logs/production"):
        """Initialize dashboard.
        
        Parameters
        ----------
        initial_equity : float
            Starting equity
        log_dir : str
            Logging directory
        """
        self.initial_equity = initial_equity
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.monitor = ProductionMonitor(log_dir)
        self.risk_manager = RiskManager(initial_equity)
        self.metrics_collector = MetricsCollector(log_dir)
        self.metrics_monitor = RealtimeMetricsMonitor(self.metrics_collector)
        
        # Session state
        self.session_start = datetime.now()
        self.alerts_log = []
    
    def record_trade(
        self,
        trade_id: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        bars_held: int,
        direction: str = "LONG",
    ) -> None:
        """Record completed trade across all systems.
        
        Parameters
        ----------
        trade_id : int
            Trade identifier
        entry_price : float
            Entry price
        exit_price : float
            Exit price
        pnl : float
            Profit/loss
        bars_held : int
            Number of bars held
        direction : str
            LONG or SHORT
        """
        # Record in monitor
        self.monitor.record_trade(trade_id, entry_price, exit_price, pnl, bars_held)
        
        # Record in risk manager
        self.risk_manager.close_position(trade_id, exit_price)
        
        # Record in metrics
        win = pnl > 0
        return_pct = (exit_price - entry_price) / entry_price * 100
        self.metrics_collector.record_trade_metrics(
            win=win,
            pnl=pnl,
            duration=bars_held,
            entry_price=entry_price,
            exit_price=exit_price,
        )
        
        # Check for alerts
        if pnl > 100:
            self.log_alert("SUCCESS", f"Large profit: ${pnl:+.2f}")
        elif pnl < -100:
            self.log_alert("RISK", f"Large loss: ${pnl:+.2f}")
    
    def request_trade(
        self,
        entry_price: float,
        direction: str,
        stop_loss: float,
        take_profit: float,
    ) -> Dict:
        """Request to open a trade (validates risk limits).
        
        Parameters
        ----------
        entry_price : float
            Entry price
        direction : str
            LONG or SHORT
        stop_loss : float
            Stop loss price
        take_profit : float
            Take profit price
        
        Returns
        -------
        dict
            Trade approval status
        """
        # Check risk limits
        can_open, reason = self.risk_manager.check_can_open_position()
        
        if not can_open:
            self.log_alert("DENIED", f"Trade request denied: {reason}")
            return {
                "approved": False,
                "reason": reason,
            }
        
        # Calculate position size
        size = self.risk_manager.calculate_position_sizing(
            entry_price, stop_loss, risk_pct=2.0
        )
        
        # Open position
        success, position, msg = self.risk_manager.open_position(
            entry_price, direction, stop_loss, take_profit, size
        )
        
        if success:
            self.log_alert("SUCCESS", f"Trade approved: {msg}")
            return {
                "approved": True,
                "position_id": position.position_id,
                "size": position.size,
                "message": msg,
            }
        else:
            self.log_alert("DENIED", msg)
            return {
                "approved": False,
                "reason": msg,
            }
    
    def update_equity(self, equity: float) -> None:
        """Update equity and check for violations.
        
        Parameters
        ----------
        equity : float
            Current equity
        """
        self.monitor.record_equity(equity)
        self.risk_manager.current_equity = equity
        
        # Check risk limits
        risk_check = self.monitor.check_risk_limits(equity, self.initial_equity)
        
        if not risk_check["ok"]:
            for violation in risk_check["violations"]:
                self.log_alert("CRITICAL", violation)
        
        for warning in risk_check["warnings"]:
            self.log_alert("WARNING", warning)
    
    def check_violations(self) -> Dict:
        """Check all active violations.
        
        Returns
        -------
        dict
            Violation details
        """
        violations = self.risk_manager.check_risk_violations()
        
        # Log any critical violations
        if violations["emergency_stop"]:
            self.log_alert("CRITICAL", "EMERGENCY STOP: Critical risk limits exceeded!")
        
        return violations
    
    def log_alert(self, severity: str, message: str) -> None:
        """Log alert.
        
        Parameters
        ----------
        severity : str
            Alert severity (INFO, WARNING, RISK, CRITICAL, SUCCESS, DENIED)
        message : str
            Alert message
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "message": message,
        }
        self.alerts_log.append(alert)
        
        # Save to file
        alert_file = self.log_dir / "alerts.log"
        with open(alert_file, "a") as f:
            f.write(json.dumps(alert) + "\n")
    
    def print_full_dashboard(self, equity: float) -> None:
        """Print complete dashboard.
        
        Parameters
        ----------
        equity : float
            Current equity
        """
        # Header
        print("\n" + "=" * 100)
        print("█" * 100)
        print(" " * 30 + "🤖 PRODUCTION MONITORING DASHBOARD 🤖")
        print(" " * 32 + f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("█" * 100)
        print("=" * 100)
        
        # Risk manager status
        rm_status = self.risk_manager.get_status()
        print("\n📊 EQUITY & POSITIONS:")
        print(f"  Starting Equity:        ${self.initial_equity:>12,.2f}")
        print(f"  Current Equity:         ${equity:>12,.2f}")
        print(f"  Session P&L:            ${rm_status['session_pnl']:>12,.2f} ({rm_status['session_pnl_pct']:+6.2f}%)")
        print(f"  Daily P&L:              ${rm_status['daily_pnl']:>12,.2f} ({rm_status['daily_pnl_pct']:+6.2f}%)")
        print(f"  Max Drawdown:           {rm_status['max_drawdown']:>12.2f}%")
        print(f"  Open Positions:         {rm_status['open_positions']:>12}")
        print(f"  Closed Positions:       {rm_status['closed_positions']:>12}")
        
        # Risk status
        print("\n⚠️  RISK LIMITS:")
        print(f"  Daily loss limit:       ${-self.risk_manager.risk_config['max_daily_loss_pct'] * self.initial_equity / 100:>12,.2f}")
        print(f"  Daily loss remaining:   ${rm_status['daily_loss_limit_remaining_pct'] * self.initial_equity / 100:>12,.2f}")
        print(f"  Drawdown limit:         {self.risk_manager.risk_config['max_drawdown_threshold']:>12.2f}%")
        print(f"  Drawdown remaining:     {rm_status['drawdown_limit_remaining_pct']:>12.2f}%")
        
        # Violations check
        violations = self.check_violations()
        print("\n⚡ ACTIVE VIOLATIONS:")
        if violations["stop_loss_hits"]:
            print(f"  Stop loss hits:         {len(violations['stop_loss_hits']):>12} positions")
        if violations["take_profit_hits"]:
            print(f"  Take profit hits:       {len(violations['take_profit_hits']):>12} positions")
        if violations["equity_alerts"]:
            for alert in violations["equity_alerts"]:
                print(f"  ⚠️  {alert}")
        
        print("\n" + ("🛑 EMERGENCY STOP ACTIVE!" if violations["emergency_stop"] else "✅ Emergency stop NOT active"))
        
        # Performance metrics
        stats = self.metrics_collector.get_summary_stats()
        if stats["total_trades"] > 0:
            print("\n📈 PERFORMANCE METRICS:")
            print(f"  Total Trades:           {stats['total_trades']:>12}")
            print(f"  Win Rate:               {stats['win_rate']*100:>12.2f}%")
            print(f"  Avg P&L:                ${stats['avg_pnl']:>12,.2f}")
            print(f"  Std P&L:                ${stats['std_pnl']:>12,.2f}")
            print(f"  Avg Trade Duration:     {stats['avg_duration']:>12.0f} minutes")
            print(f"  Sharpe Ratio:           {stats['sharpe_ratio']:>12.3f}")
        
        # Recent alerts (last 5)
        if self.alerts_log:
            print("\n🚨 RECENT ALERTS (Last 5):")
            for alert in self.alerts_log[-5:]:
                severity_emoji = {
                    "INFO": "ℹ️ ",
                    "WARNING": "⚠️ ",
                    "RISK": "📉 ",
                    "CRITICAL": "🛑 ",
                    "SUCCESS": "✓ ",
                    "DENIED": "❌ ",
                }.get(alert["severity"], "○ ")
                print(f"  {severity_emoji} [{alert['severity']:8s}] {alert['message']}")
        
        # Session duration
        duration = datetime.now() - self.session_start
        print(f"\nSession Duration: {duration}")
        print("=" * 100 + "\n")
    
    def get_status(self) -> Dict:
        """Get current dashboard status.
        
        Returns
        -------
        dict
            Status information
        """
        rm_status = self.risk_manager.get_status()
        violations = self.check_violations()
        
        return {
            "current_loss": rm_status.get("daily_pnl", 0),
            "current_gain": rm_status.get("session_pnl", 0),
            "drawdown": rm_status.get("max_drawdown", 0),
            "positions_open": len(self.risk_manager.open_positions),
            "violations": [v for v in violations.get("equity_alerts", [])],
            "alerts": [a["message"] for a in self.alerts_log[-10:]],
            "open_positions": [
                {
                    "symbol": "EURUSD",
                    "entry_price": p.entry_price,
                    "current_price": p.entry_price,  # Placeholder
                    "size": p.size,
                    "pnl": 0,  # Placeholder
                }
                for p in self.risk_manager.open_positions
            ],
        }
    
    def get_rich_metrics(self) -> Dict:
        """Get comprehensive metrics for dashboard.
        
        Returns
        -------
        dict
            Rich metrics data
        """
        stats = self.metrics_collector.get_summary_stats()
        rm_status = self.risk_manager.get_status()
        
        return {
            "metrics": {
                "trade_stats": {
                    "total_trades": stats.get("total_trades", 0),
                    "winning_trades": int(stats.get("total_trades", 0) * stats.get("win_rate", 0)),
                    "losing_trades": int(stats.get("total_trades", 0) * (1 - stats.get("win_rate", 0))),
                    "win_rate": stats.get("win_rate", 0) * 100,
                    "avg_win": stats.get("avg_pnl", 0) if stats.get("win_rate", 0) > 0 else 0,
                    "avg_loss": -stats.get("std_pnl", 0) if stats.get("win_rate", 0) < 1 else 0,
                    "profit_factor": 1.0,  # Placeholder
                    "trades": [
                        {
                            "pnl": t.get("pnl", 0), 
                            "entry_price": t.get("entry", 0), 
                            "exit_price": t.get("exit", 0)
                        }
                        for t in self.monitor.trades[-20:]  # Last 20 trades
                    ],
                },
                "sharpe_ratio": stats.get("sharpe_ratio", 0),
            },
        }
    
    def export_session_report(self) -> Path:
        """Export complete session report.
        
        Returns
        -------
        Path
            Path to report
        """
        report_path = self.log_dir / f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "initial_equity": self.initial_equity,
            "risk_manager_status": self.risk_manager.get_status(),
            "performance_metrics": self.metrics_collector.get_summary_stats(),
            "trades": self.monitor.trades,
            "alerts": self.alerts_log[-100:],  # Last 100 alerts
            "open_positions": [
                {
                    "id": p.position_id,
                    "direction": p.direction,
                    "entry": p.entry_price,
                    "sl": p.stop_loss,
                    "tp": p.take_profit,
                    "size": p.size,
                }
                for p in self.risk_manager.open_positions
            ],
        }
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return report_path


def demo_run():
    """Run demonstration."""
    print("\n" + "=" * 100)
    print("INTEGRATED DASHBOARD - DEMONSTRATION")
    print("=" * 100)
    
    dashboard = IntegratedDashboard(initial_equity=10000)
    
    print("\n1. Simulating 10 trades...")
    equity = 10000
    for i in range(10):
        # Request trade
        approval = dashboard.request_trade(
            entry_price=1.0700,
            direction="LONG",
            stop_loss=1.0695,
            take_profit=1.0710,
        )
        
        if approval["approved"]:
            # Simulate trade result
            import numpy as np
            pnl = np.random.randn() * 50 + 30
            dashboard.record_trade(
                trade_id=i,
                entry_price=1.0700,
                exit_price=1.0700 + pnl / 100000,
                pnl=pnl,
                bars_held=np.random.randint(10, 120),
            )
            
            equity += pnl
            dashboard.update_equity(equity)
    
    print("\n2. Printing full dashboard...")
    dashboard.print_full_dashboard(equity)
    
    print("\n3. Exporting reports...")
    report_path = dashboard.export_session_report()
    print(f"✓ Session report: {report_path}")
    
    print("\n✓ Demo complete")


if __name__ == "__main__":
    demo_run()
