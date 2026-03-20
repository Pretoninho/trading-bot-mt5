"""
test_monitoring.py
==================
Tests for production monitoring and risk management.

Tests:
- Production monitoring
- Risk management
- Metrics collection
- Integrated dashboard
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import tempfile
import shutil

from production_monitor import ProductionMonitor
from risk_manager import RiskManager
from metrics_collector import MetricsCollector, RealtimeMetricsMonitor
from monitoring_dashboard import IntegratedDashboard


class TestProductionMonitor:
    """Test production monitoring system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = ProductionMonitor(log_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_record_trade(self):
        """Test trade recording."""
        self.monitor.record_trade(
            trade_id=1,
            entry_price=1.0700,
            exit_price=1.0705,
            pnl=50.0,
            bars_held=10,
        )
        
        assert len(self.monitor.trades) == 1
        assert self.monitor.trades[0]["trade_id"] == 1
        assert self.monitor.trades[0]["pnl"] == 50.0
    
    def test_record_equity(self):
        """Test equity recording."""
        self.monitor.record_equity(10050.0)
        
        assert len(self.monitor.equity_history) == 1
        assert self.monitor.equity_history[0]["equity"] == 10050.0
    
    def test_check_risk_limits(self):
        """Test risk limit checking."""
        # Normal case
        result = self.monitor.check_risk_limits(10500, 10000)
        assert result["ok"] is True
        
        # Violation case
        result = self.monitor.check_risk_limits(9400, 10000)
        assert result["ok"] is False
        assert len(result["violations"]) > 0
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        for i in range(10):
            self.monitor.record_trade(
                trade_id=i,
                entry_price=1.0700,
                exit_price=1.0700 + (0.0005 if i % 2 == 0 else -0.0003),
                pnl=50 if i % 2 == 0 else -30,
                bars_held=50,
            )
        
        metrics = self.monitor.get_performance_metrics()
        assert metrics["trades_total"] == 10
        assert metrics["win_rate"] == 0.5
        assert metrics["avg_pnl"] > 0
    
    def test_save_session_logs(self):
        """Test session log saving."""
        self.monitor.record_equity(10100)
        log_file = self.monitor.save_session_logs()
        
        assert log_file.exists()
        with open(log_file) as f:
            data = json.load(f)
        
        assert "trades" in data
        assert "equity_history" in data


class TestRiskManager:
    """Test risk management system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.manager = RiskManager(initial_equity=10000)
    
    def test_can_open_position(self):
        """Test position opening permission."""
        can_open, reason = self.manager.check_can_open_position()
        assert can_open is True
    
    def test_position_sizing(self):
        """Test position size calculation."""
        size = self.manager.calculate_position_sizing(
            entry_price=1.0700,
            stop_loss=1.0695,
            risk_pct=2.0,
        )
        
        assert size > 0
        assert isinstance(size, float)
    
    def test_open_position(self):
        """Test opening a position."""
        success, position, msg = self.manager.open_position(
            entry_price=1.0700,
            direction="LONG",
            stop_loss=1.0695,
            take_profit=1.0710,
        )
        
        assert success is True
        assert position is not None
        assert len(self.manager.open_positions) == 1
    
    def test_close_position(self):
        """Test closing a position."""
        # Open position
        success, position, _ = self.manager.open_position(
            entry_price=1.0700,
            direction="LONG",
            stop_loss=1.0695,
            take_profit=1.0710,
        )
        
        # Close position
        success, pnl, msg = self.manager.close_position(
            position_id=position.position_id,
            exit_price=1.0705,
        )
        
        assert success is True
        assert pnl > 0
        assert len(self.manager.open_positions) == 0
        assert len(self.manager.closed_positions) == 1
    
    def test_check_risk_violations(self):
        """Test violation checking."""
        violations = self.manager.check_risk_violations()
        
        assert "stop_loss_hits" in violations
        assert "take_profit_hits" in violations
        assert "emergency_stop" in violations
    
    def test_get_status(self):
        """Test status reporting."""
        status = self.manager.get_status()
        
        assert "current_equity" in status
        assert "session_pnl" in status
        assert "open_positions" in status
        assert status["current_equity"] == 10000


class TestMetricsCollector:
    """Test metrics collection system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = MetricsCollector(log_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_record_train_step(self):
        """Test training step recording."""
        self.collector.record_train_step(
            step=0,
            loss=0.5,
            value_loss=0.3,
            policy_loss=0.2,
            entropy=0.7,
            learning_rate=1e-4,
        )
        
        assert len(self.collector.train_metrics["loss"]) == 1
    
    def test_record_episode(self):
        """Test episode recording."""
        self.collector.record_episode(
            episode=1,
            total_reward=100.0,
            episode_length=1000,
            win_rate=0.55,
            avg_trade_duration=60,
        )
        
        assert self.collector.episode_metrics["episodes"] == 1
    
    def test_record_trade_metrics(self):
        """Test trade metrics recording."""
        self.collector.record_trade_metrics(
            win=True,
            pnl=50.0,
            duration=60,
            entry_price=1.0700,
            exit_price=1.0705,
        )
        
        assert len(self.collector.metrics_history) == 1
    
    def test_get_summary_stats(self):
        """Test summary statistics."""
        for i in range(10):
            self.collector.record_trade_metrics(
                win=i % 2 == 0,
                pnl=(50 if i % 2 == 0 else -30),
                duration=60,
                entry_price=1.0700,
                exit_price=1.0700 + (0.0005 if i % 2 == 0 else -0.0003),
            )
        
        stats = self.collector.get_summary_stats()
        assert stats["total_trades"] == 10
        assert stats["win_rate"] == 0.5
    
    def test_export_tensorboard_format(self):
        """Test TensorBoard format export."""
        self.collector.record_trade_metrics(
            win=True,
            pnl=50.0,
            duration=60,
            entry_price=1.0700,
            exit_price=1.0705,
        )
        
        export_path = self.collector.export_tensorboard_format(self.temp_dir)
        assert export_path.exists()


class TestIntegratedDashboard:
    """Test integrated dashboard."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dashboard = IntegratedDashboard(
            initial_equity=10000,
            log_dir=self.temp_dir,
        )
    
    def teardown_method(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_request_trade(self):
        """Test trade request."""
        approval = self.dashboard.request_trade(
            entry_price=1.0700,
            direction="LONG",
            stop_loss=1.0695,
            take_profit=1.0710,
        )
        
        assert "approved" in approval
    
    def test_record_trade_integration(self):
        """Test integrated trade recording."""
        # Request trade
        approval = self.dashboard.request_trade(
            entry_price=1.0700,
            direction="LONG",
            stop_loss=1.0695,
            take_profit=1.0710,
        )
        
        if approval["approved"]:
            # Record trade
            self.dashboard.record_trade(
                trade_id=1,
                entry_price=1.0700,
                exit_price=1.0705,
                pnl=50.0,
                bars_held=60,
            )
            
            assert len(self.dashboard.monitor.trades) == 1
    
    def test_update_equity(self):
        """Test equity update."""
        self.dashboard.update_equity(10100)
        
        assert len(self.dashboard.monitor.equity_history) == 1
    
    def test_check_violations(self):
        """Test violation checking."""
        violations = self.dashboard.check_violations()
        
        assert "stop_loss_hits" in violations
        assert "emergency_stop" in violations
    
    def test_export_session_report(self):
        """Test session report export."""
        # Add some data
        self.dashboard.update_equity(10100)
        self.dashboard.log_alert("INFO", "Test alert")
        
        report_path = self.dashboard.export_session_report()
        assert report_path.exists()
        
        with open(report_path) as f:
            report = json.load(f)
        
        assert "timestamp" in report
        assert "alerts" in report


class TestIntegration:
    """Integration tests combining all components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dashboard = IntegratedDashboard(
            initial_equity=10000,
            log_dir=self.temp_dir,
        )
    
    def teardown_method(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_trading_session(self):
        """Test complete trading session."""
        initial_equity = 10000
        equity = initial_equity
        
        # Simulate 5 trades
        for i in range(5):
            # Request trade
            approval = self.dashboard.request_trade(
                entry_price=1.0700,
                direction="LONG",
                stop_loss=1.0695,
                take_profit=1.0710,
            )
            
            if approval["approved"]:
                # Simulate trade result
                pnl = np.random.randn() * 50 + 30
                
                # Record trade
                self.dashboard.record_trade(
                    trade_id=i,
                    entry_price=1.0700,
                    exit_price=1.0700 + pnl / 100000,
                    pnl=pnl,
                    bars_held=np.random.randint(10, 120),
                )
                
                # Update equity
                equity += pnl
                self.dashboard.update_equity(equity)
        
        # Verify results
        assert len(self.dashboard.monitor.trades) > 0
        assert len(self.dashboard.monitor.equity_history) > 0
        # Verify metrics recorded
        assert len(self.dashboard.metrics_collector.metrics_history) > 0
    
    def test_risk_limit_enforcement(self):
        """Test risk limit enforcement."""
        # Simulate large loss
        self.dashboard.update_equity(9400)  # 6% loss, exceeds 5% limit
        
        # Check violations
        violations = self.dashboard.check_violations()
        
        # Should have alerts
        assert len(self.dashboard.alerts_log) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
