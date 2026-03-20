"""
metrics_collector.py
====================
Collects and logs production metrics for TensorBoard visualization.

Metrics:
- Equity curve
- Win rate
- Drawdown
- P&L distribution
- Trade duration
"""

import numpy as np
from typing import Dict, List
from datetime import datetime
from pathlib import Path
import json


class MetricsCollector:
    """Collects metrics for production monitoring."""
    
    def __init__(self, log_dir: str = "./logs/production"):
        """Initialize collector.
        
        Parameters
        ----------
        log_dir : str
            Directory for metric logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics_history = []
        self.episode_metrics = {
            "episodes": 0,
            "avg_reward": 0,
            "avg_episode_length": 0,
            "max_episode_reward": -np.inf,
            "min_episode_reward": np.inf,
        }
        self.train_metrics = {
            "loss": [],
            "value_loss": [],
            "policy_loss": [],
            "entropy": [],
            "learning_rate": [],
        }
    
    def record_train_step(
        self,
        step: int,
        loss: float,
        value_loss: float,
        policy_loss: float,
        entropy: float,
        learning_rate: float,
    ) -> None:
        """Record training step metrics.
        
        Parameters
        ----------
        step : int
            Global training step
        loss : float
            Total loss
        value_loss : float
            Value function loss
        policy_loss : float
            Policy loss
        entropy : float
            Policy entropy
        learning_rate : float
            Current learning rate
        """
        self.train_metrics["loss"].append(loss)
        self.train_metrics["value_loss"].append(value_loss)
        self.train_metrics["policy_loss"].append(policy_loss)
        self.train_metrics["entropy"].append(entropy)
        self.train_metrics["learning_rate"].append(learning_rate)
        
        # Log to file every 100 steps
        if step % 100 == 0:
            self._log_train_metrics(step)
    
    def record_episode(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        win_rate: float,
        avg_trade_duration: float,
    ) -> None:
        """Record episode metrics.
        
        Parameters
        ----------
        episode : int
            Episode number
        total_reward : float
            Total reward for episode
        episode_length : int
            Length of episode in steps
        win_rate : float
            Win rate for episode (0-1)
        avg_trade_duration : float
            Average trade duration
        """
        self.episode_metrics["episodes"] = episode
        self.episode_metrics["avg_reward"] = (
            (self.episode_metrics["avg_reward"] * (episode - 1) + total_reward) / episode
        )
        self.episode_metrics["avg_episode_length"] = (
            (self.episode_metrics["avg_episode_length"] * (episode - 1) + episode_length) / episode
        )
        self.episode_metrics["max_episode_reward"] = max(
            self.episode_metrics["max_episode_reward"], total_reward
        )
        self.episode_metrics["min_episode_reward"] = min(
            self.episode_metrics["min_episode_reward"], total_reward
        )
        
        # Log every 10 episodes
        if episode % 10 == 0:
            self._log_episode_metrics(episode, total_reward, episode_length, win_rate, avg_trade_duration)
    
    def record_trade_metrics(
        self,
        win: bool,
        pnl: float,
        duration: int,
        entry_price: float,
        exit_price: float,
    ) -> None:
        """Record individual trade metrics.
        
        Parameters
        ----------
        win : bool
            Whether trade was profitable
        pnl : float
            Profit/loss amount
        duration : int
            Trade duration in bars
        entry_price : float
            Entry price
        exit_price : float
            Exit price
        """
        trade_metric = {
            "timestamp": datetime.now().isoformat(),
            "win": win,
            "pnl": pnl,
            "duration": duration,
            "entry": entry_price,
            "exit": exit_price,
            "return_pct": (exit_price - entry_price) / entry_price * 100,
        }
        self.metrics_history.append(trade_metric)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics.
        
        Returns
        -------
        dict
            Summary statistics
        """
        if not self.metrics_history:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "avg_duration": 0,
                "sharpe_ratio": 0,
            }
        
        metrics_array = np.array([m["pnl"] for m in self.metrics_history])
        wins = sum(1 for m in self.metrics_history if m["win"])
        
        # Sharpe ratio (annualized)
        returns = np.array([m["return_pct"] for m in self.metrics_history])
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)  # Annualized for M1
        else:
            sharpe = 0
        
        return {
            "total_trades": len(self.metrics_history),
            "win_rate": wins / len(self.metrics_history),
            "avg_pnl": float(metrics_array.mean()),
            "std_pnl": float(metrics_array.std()),
            "avg_duration": float(np.mean([m["duration"] for m in self.metrics_history])),
            "sharpe_ratio": sharpe,
            "max_pnl": float(metrics_array.max()),
            "min_pnl": float(metrics_array.min()),
        }
    
    def export_tensorboard_format(self, output_dir: str = "./logs/production") -> Path:
        """Export metrics in TensorBoard-compatible format.
        
        Parameters
        ----------
        output_dir : str
            Output directory for metrics
        
        Returns
        -------
        Path
            Path to exported metrics file
        """
        output_path = Path(output_dir) / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "training_metrics": {
                "avg_loss": float(np.mean(self.train_metrics["loss"])) if self.train_metrics["loss"] else 0,
                "avg_value_loss": float(np.mean(self.train_metrics["value_loss"])) if self.train_metrics["value_loss"] else 0,
                "avg_policy_loss": float(np.mean(self.train_metrics["policy_loss"])) if self.train_metrics["policy_loss"] else 0,
                "avg_entropy": float(np.mean(self.train_metrics["entropy"])) if self.train_metrics["entropy"] else 0,
            },
            "episode_metrics": self.episode_metrics,
            "trade_summary": self.get_summary_stats(),
            "recent_trades": self.metrics_history[-50:],  # Last 50 trades
        }
        
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return output_path
    
    def _log_train_metrics(self, step: int) -> None:
        """Log training metrics to file."""
        log_file = self.log_dir / "training_metrics.log"
        log_entry = {
            "step": step,
            "loss": float(np.mean(self.train_metrics["loss"][-100:])),  # Last 100 steps
            "value_loss": float(np.mean(self.train_metrics["value_loss"][-100:])),
            "policy_loss": float(np.mean(self.train_metrics["policy_loss"][-100:])),
            "entropy": float(np.mean(self.train_metrics["entropy"][-100:])),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _log_episode_metrics(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        win_rate: float,
        avg_trade_duration: float,
    ) -> None:
        """Log episode metrics to file."""
        log_file = self.log_dir / "episode_metrics.log"
        log_entry = {
            "episode": episode,
            "total_reward": float(total_reward),
            "episode_length": episode_length,
            "win_rate": float(win_rate),
            "avg_trade_duration": float(avg_trade_duration),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


class RealtimeMetricsMonitor:
    """Real-time metrics monitoring dashboard."""
    
    def __init__(self, collector: MetricsCollector):
        """Initialize monitor.
        
        Parameters
        ----------
        collector : MetricsCollector
            Metrics collector instance
        """
        self.collector = collector
    
    def print_dashboard(self) -> None:
        """Print real-time dashboard."""
        stats = self.collector.get_summary_stats()
        
        dashboard = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PRODUCTION METRICS DASHBOARD                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 TRADING PERFORMANCE:
  Total Trades:      {stats['total_trades']:>6} trades
  Win Rate:          {stats['win_rate']*100:>6.2f}%
  Avg P&L:           ${stats['avg_pnl']:>10,.2f}
  Std P&L:           ${stats['std_pnl']:>10,.2f}
  Max P&L:           ${stats['max_pnl']:>10,.2f}
  Min P&L:           ${stats['min_pnl']:>10,.2f}

⏱️  DURATION METRICS:
  Avg Trade Time:    {stats['avg_duration']:>6.0f} minutes

📈 RISK-ADJUSTED RETURNS:
  Sharpe Ratio:      {stats['sharpe_ratio']:>10,.3f}

"""
        print(dashboard)
    
    def export_report(self, output_path: str = "./logs/production/report.md") -> Path:
        """Export markdown report.
        
        Parameters
        ----------
        output_path : str
            Output file path
        
        Returns
        -------
        Path
            Path to report
        """
        stats = self.collector.get_summary_stats()
        
        report = f"""# Production Metrics Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Trading Performance

- **Total Trades**: {stats['total_trades']}
- **Win Rate**: {stats['win_rate']*100:.2f}%
- **Average P&L**: ${stats['avg_pnl']:,.2f}
- **P&L Std Dev**: ${stats['std_pnl']:,.2f}
- **Max P&L**: ${stats['max_pnl']:,.2f}
- **Min P&L**: ${stats['min_pnl']:,.2f}

## Risk Metrics

- **Avg Trade Duration**: {stats['avg_duration']:.0f} minutes
- **Sharpe Ratio**: {stats['sharpe_ratio']:.3f}

## Training Performance

- **Avg Loss**: {float(np.mean(self.collector.train_metrics['loss']) if self.collector.train_metrics['loss'] else 0):.6f}
- **Avg Value Loss**: {float(np.mean(self.collector.train_metrics['value_loss']) if self.collector.train_metrics['value_loss'] else 0):.6f}
- **Avg Policy Loss**: {float(np.mean(self.collector.train_metrics['policy_loss']) if self.collector.train_metrics['policy_loss'] else 0):.6f}
- **Avg Entropy**: {float(np.mean(self.collector.train_metrics['entropy']) if self.collector.train_metrics['entropy'] else 0):.6f}

## Recent Trades

| Trade | Win | P&L | Duration | Return |
|-------|-----|-----|----------|--------|
"""
        
        for i, trade in enumerate(self.collector.metrics_history[-20:], 1):
            win_str = "✓" if trade["win"] else "✗"
            report += f"| {i} | {win_str} | ${trade['pnl']:,.2f} | {trade['duration']}m | {trade['return_pct']:+.2f}% |\n"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(report)
        
        return output_file


def main():
    """Demonstrate metrics collector."""
    print("\n" + "=" * 80)
    print("METRICS COLLECTOR - SYSTEM TEST")
    print("=" * 80)
    
    collector = MetricsCollector()
    
    # Simulate training metrics
    print("\nSimulating training metrics...")
    for step in range(0, 1001, 100):
        collector.record_train_step(
            step=step,
            loss=0.5 * (1 - step/1000),
            value_loss=0.3 * (1 - step/1000),
            policy_loss=0.2 * (1 - step/1000),
            entropy=0.5 * (1 - step/1000),
            learning_rate=1e-4,
        )
    
    # Simulate episodes
    print("Simulating episodes...")
    for episode in range(1, 101):
        collector.record_episode(
            episode=episode,
            total_reward=50 + np.random.randn() * 10,
            episode_length=1000,
            win_rate=0.55 + np.random.rand() * 0.1,
            avg_trade_duration=60,
        )
    
    # Simulate trades
    print("Simulating trades...")
    for i in range(50):
        win = np.random.rand() > 0.45
        pnl = np.random.randn() * 50 + (50 if win else -30)
        collector.record_trade_metrics(
            win=win,
            pnl=pnl,
            duration=np.random.randint(10, 240),
            entry_price=1.0700,
            exit_price=1.0700 + np.random.randn() * 0.001,
        )
    
    # Display dashboard
    monitor = RealtimeMetricsMonitor(collector)
    monitor.print_dashboard()
    
    # Export formats
    print("\nExporting metrics...")
    json_path = collector.export_tensorboard_format()
    print(f"✓ JSON export: {json_path}")
    
    report_path = monitor.export_report()
    print(f"✓ Markdown report: {report_path}")


if __name__ == "__main__":
    main()
