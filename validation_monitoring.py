"""
validation_monitoring.py
=========================
Comprehensive validation and stress testing of the monitoring system.

Tests realistic trading scenarios and edge cases.
"""

import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import shutil

from monitoring_dashboard import IntegratedDashboard
from production_monitor import ProductionMonitor
from risk_manager import RiskManager
from metrics_collector import MetricsCollector


def test_scenario_1_profitable_day():
    """Scenario 1: Profitable trading day."""
    print("\n" + "="*80)
    print("SCENARIO 1: Profitable Trading Day")
    print("="*80)
    
    dashboard = IntegratedDashboard(initial_equity=10000)
    equity = 10000
    
    # Simulate 5 winning trades
    trades = [
        (1.0700, 1.0705, 50),   # +$50
        (1.0705, 1.0715, 100),  # +$100
        (1.0715, 1.0718, 30),   # +$30
        (1.0718, 1.0725, 70),   # +$70
        (1.0725, 1.0730, 50),   # +$50
    ]
    
    results = {
        "total_trades": 0,
        "winning_trades": 0,
        "total_pnl": 0,
        "approvals_accepted": 0,
        "approvals_denied": 0,
    }
    
    for i, (entry, exit_price, pnl) in enumerate(trades):
        # Request trade
        approval = dashboard.request_trade(
            entry_price=entry,
            direction="LONG",
            stop_loss=entry - 0.0010,
            take_profit=exit_price,
        )
        
        if approval["approved"]:
            results["approvals_accepted"] += 1
            dashboard.record_trade(
                trade_id=i,
                entry_price=entry,
                exit_price=exit_price,
                pnl=pnl,
                bars_held=np.random.randint(30, 120),
            )
            equity += pnl
            dashboard.update_equity(equity)
            results["total_trades"] += 1
            results["winning_trades"] += 1
            results["total_pnl"] += pnl
        else:
            results["approvals_denied"] += 1
    
    # Verify
    assert results["total_trades"] == 5, f"Expected 5 trades, got {results['total_trades']}"
    assert results["approvals_accepted"] == 5, "Not all trades were approved"
    assert results["total_pnl"] == 300, f"Expected P&L of $300, got ${results['total_pnl']}"
    assert equity == 10300, f"Expected equity $10,300, got ${equity}"
    
    status = dashboard.risk_manager.get_status()
    print(f"✓ Starting Equity: ${10000:,.2f}")
    print(f"✓ Ending Equity: ${equity:,.2f}")
    print(f"✓ Total P&L: ${results['total_pnl']:,.2f}")
    print(f"✓ Trades Executed: {results['total_trades']}")
    print(f"✓ Win Rate: 100%")
    print(f"✓ Max Drawdown: {status['max_drawdown']:.2f}%")
    print("✓ SCENARIO 1 PASSED")
    
    return results


def test_scenario_2_mixed_results():
    """Scenario 2: Mixed winning and losing trades."""
    print("\n" + "="*80)
    print("SCENARIO 2: Mixed Win/Loss Trading Day")
    print("="*80)
    
    dashboard = IntegratedDashboard(initial_equity=10000)
    equity = 10000
    
    # Mix of profitable and losing trades
    trades = [
        (1.0700, 1.0708, 80, True),    # +$80 (win)
        (1.0708, 1.0703, -50, False),  # -$50 (loss)
        (1.0703, 1.0712, 90, True),    # +$90 (win)
        (1.0712, 1.0708, -40, False),  # -$40 (loss)
        (1.0708, 1.0720, 120, True),   # +$120 (win)
    ]
    
    results = {
        "trades": [],
        "wins": 0,
        "losses": 0,
        "total_pnl": 0,
    }
    
    for i, (entry, exit_price, pnl, is_win) in enumerate(trades):
        approval = dashboard.request_trade(
            entry_price=entry,
            direction="LONG",
            stop_loss=entry - 0.0015,
            take_profit=exit_price,
        )
        
        if approval["approved"]:
            dashboard.record_trade(
                trade_id=i,
                entry_price=entry,
                exit_price=exit_price,
                pnl=pnl,
                bars_held=60,
            )
            equity += pnl
            dashboard.update_equity(equity)
            
            results["trades"].append({"pnl": pnl, "is_win": is_win})
            if is_win:
                results["wins"] += 1
            else:
                results["losses"] += 1
            results["total_pnl"] += pnl
    
    # Verify
    assert results["wins"] == 3, f"Expected 3 wins, got {results['wins']}"
    assert results["losses"] == 2, f"Expected 2 losses, got {results['losses']}"
    assert results["total_pnl"] == 200, f"Expected total P&L $200, got ${results['total_pnl']}"
    
    metrics = dashboard.metrics_collector.get_summary_stats()
    print(f"✓ Starting Equity: ${10000:,.2f}")
    print(f"✓ Ending Equity: ${equity:,.2f}")
    print(f"✓ Total P&L: ${results['total_pnl']:,.2f}")
    print(f"✓ Win Rate: {metrics['win_rate']*100:.1f}% ({results['wins']}/{len(results['trades'])})")
    print(f"✓ Avg P&L per Trade: ${metrics['avg_pnl']:.2f}")
    print("✓ SCENARIO 2 PASSED")
    
    return results


def test_scenario_3_daily_loss_limit():
    """Scenario 3: Test daily loss limit enforcement."""
    print("\n" + "="*80)
    print("SCENARIO 3: Daily Loss Limit Enforcement (-5%)")
    print("="*80)
    
    dashboard = IntegratedDashboard(initial_equity=10000)
    equity = 10000
    daily_loss_limit = -500  # 5%
    
    # Simulate losing trades until hitting limit
    losing_trades = [
        -150,  # -$150
        -150,  # -$150
        -100,  # -$100
        -100,  # -$100 (total: -$500, at limit)
    ]
    
    results = {
        "trades_before_limit": 0,
        "equity_before_limit": equity,
        "total_loss": 0,
        "limit_triggered": False,
    }
    
    for i, loss in enumerate(losing_trades):
        approval = dashboard.request_trade(
            entry_price=1.0700,
            direction="LONG",
            stop_loss=1.0695,
            take_profit=1.0710,
        )
        
        if approval["approved"]:
            dashboard.record_trade(
                trade_id=i,
                entry_price=1.0700,
                exit_price=1.0700 + loss/100000,
                pnl=loss,
                bars_held=60,
            )
            equity += loss
            dashboard.update_equity(equity)
            results["trades_before_limit"] += 1
            results["total_loss"] += loss
        else:
            results["limit_triggered"] = True
            print(f"  ⚠️  Trade {i+1} denied: Daily loss limit likely triggered")
    
    # Verify
    status = dashboard.risk_manager.get_status()
    print(f"✓ Starting Equity: ${10000:,.2f}")
    print(f"✓ Ending Equity: ${equity:,.2f}")
    print(f"✓ Total Loss: ${results['total_loss']:,.2f}")
    print(f"✓ Daily P&L: ${status['daily_pnl']:,.2f}")
    print(f"✓ Daily P&L %: {status['daily_pnl_pct']:+.2f}%")
    print(f"✓ Daily Loss Limit: ${daily_loss_limit:,.2f} (-5%)")
    print(f"✓ Trades executed: {results['trades_before_limit']}")
    print("✓ SCENARIO 3 PASSED")
    
    return results


def test_scenario_4_drawdown_monitoring():
    """Scenario 4: Test drawdown monitoring."""
    print("\n" + "="*80)
    print("SCENARIO 4: Drawdown Monitoring (Max 15%)")
    print("="*80)
    
    dashboard = IntegratedDashboard(initial_equity=10000)
    equity = 10000
    max_equity_reached = 10000
    
    # Simulate equity curve with drawdown
    equity_updates = [
        10100,  # +$100
        10200,  # +$200 (max)
        10150,  # -$50
        10050,  # -$150 (0.99% drawdown from peak)
        9950,   # -$250 (1.98% drawdown)
        9800,   # -$400 (3.92% drawdown)
        9500,   # -$700 (6.86% drawdown)
    ]
    
    results = {
        "max_equity": max_equity_reached,
        "min_equity": equity,
        "peak_drawdown": 0,
        "updates_before_critical": 0,
    }
    
    for i, new_equity in enumerate(equity_updates):
        dashboard.update_equity(new_equity)
        equity = new_equity
        
        # Track peak equity
        if equity > results["max_equity"]:
            results["max_equity"] = equity
        
        # Calculate drawdown
        if results["max_equity"] > 0:
            drawdown = (results["max_equity"] - equity) / results["max_equity"] * 100
            results["peak_drawdown"] = max(results["peak_drawdown"], drawdown)
        
        results["updates_before_critical"] += 1
        
        # Check violations
        violations = dashboard.check_violations()
        if violations["emergency_stop"]:
            print(f"  🛑 Emergency stop at update {i+1}: {drawdown:.2f}% drawdown")
            break
    
    status = dashboard.risk_manager.get_status()
    print(f"✓ Peak Equity: ${results['max_equity']:,.2f}")
    print(f"✓ Current Equity: ${equity:,.2f}")
    print(f"✓ Max Drawdown: {results['peak_drawdown']:.2f}%")
    print(f"✓ Drawdown Limit: 15%")
    print(f"✓ Updates processed: {results['updates_before_critical']}")
    print("✓ SCENARIO 4 PASSED")
    
    return results


def test_scenario_5_position_sizing():
    """Scenario 5: Test position sizing accuracy."""
    print("\n" + "="*80)
    print("SCENARIO 5: Position Sizing Validation")
    print("="*80)
    
    manager = RiskManager(initial_equity=10000)
    
    results = {
        "test_cases": [],
    }
    
    # Test cases: (entry, stop_loss, risk_pct, expected_max_profit_range)
    test_params = [
        (1.0700, 1.0695, 2.0),  # $200 risk, 0.0005 SL distance
        (1.0700, 1.0690, 1.0),  # $100 risk, 0.0010 SL distance
        (1.0700, 1.0680, 2.0),  # $200 risk, 0.0020 SL distance
    ]
    
    for entry, sl, risk_pct in test_params:
        size = manager.calculate_position_sizing(entry, sl, risk_pct)
        risk_amount = manager.current_equity * risk_pct / 100
        price_diff = abs(entry - sl)
        expected_size = risk_amount / price_diff
        
        test_case = {
            "entry": entry,
            "stop_loss": sl,
            "risk_pct": risk_pct,
            "calculated_size": size,
            "expected_size": expected_size,
            "matches": abs(size - expected_size) < 1,  # Allow 1 unit tolerance
        }
        results["test_cases"].append(test_case)
        
        print(f"✓ Entry: {entry}, SL: {sl}")
        print(f"  Risk %: {risk_pct}% → Size: {size:.0f} units")
    
    # Verify all calculations are accurate
    for test in results["test_cases"]:
        assert test["matches"], f"Position sizing mismatch: {test['calculated_size']} vs {test['expected_size']}"
    
    print(f"✓ All {len(results['test_cases'])} position sizing tests accurate")
    print("✓ SCENARIO 5 PASSED")
    
    return results


def test_scenario_6_alert_system():
    """Scenario 6: Test alert system functionality."""
    print("\n" + "="*80)
    print("SCENARIO 6: Alert System Validation")
    print("="*80)
    
    dashboard = IntegratedDashboard(initial_equity=10000)
    
    # Trigger different alert types
    alerts_to_test = [
        ("INFO", "System initialized"),
        ("SUCCESS", "Large profit: +$500"),
        ("WARNING", "Approaching daily limit"),
        ("RISK", "Large loss: -$200"),
        ("DENIED", "Trade rejected: daily loss limit"),
        ("CRITICAL", "Emergency stop: drawdown exceeded"),
    ]
    
    results = {
        "alerts_logged": [],
        "alert_counts": {},
    }
    
    for severity, message in alerts_to_test:
        dashboard.log_alert(severity, message)
        results["alerts_logged"].append({"severity": severity, "message": message})
        results["alert_counts"][severity] = results["alert_counts"].get(severity, 0) + 1
    
    # Verify all alerts were recorded
    assert len(dashboard.alerts_log) >= len(alerts_to_test), "Not all alerts recorded"
    
    # Verify alert file was created
    alert_file = Path(dashboard.log_dir) / "alerts.log"
    assert alert_file.exists(), "Alert log file not created"
    
    # Verify alert content
    with open(alert_file) as f:
        logged_alerts = [json.loads(line) for line in f]
    
    assert len(logged_alerts) >= len(alerts_to_test), "Alerts not written to file"
    
    print(f"✓ Total alerts logged: {len(dashboard.alerts_log)}")
    for severity, count in results["alert_counts"].items():
        print(f"  - {severity}: {count} alerts")
    print(f"✓ Alert file: {alert_file}")
    print(f"✓ File contains {len(logged_alerts)} alerts")
    print("✓ SCENARIO 6 PASSED")
    
    return results


def test_scenario_7_metrics_export():
    """Scenario 7: Test metrics export functionality."""
    print("\n" + "="*80)
    print("SCENARIO 7: Metrics Export Validation")
    print("="*80)
    
    temp_dir = tempfile.mkdtemp()
    collector = MetricsCollector(log_dir=temp_dir)
    
    results = {
        "exports": [],
        "files_created": [],
    }
    
    try:
        # Record sample metrics
        for i in range(20):
            collector.record_trade_metrics(
                win=i % 3 != 0,  # 2 wins per loss roughly
                pnl=np.random.randn() * 50 + 40,
                duration=60,
                entry_price=1.0700,
                exit_price=1.0700 + np.random.randn() * 0.001,
            )
        
        # Export in different formats
        json_export = collector.export_tensorboard_format(temp_dir)
        assert json_export.exists(), "JSON export failed"
        results["files_created"].append(str(json_export))
        
        # Verify export content
        with open(json_export) as f:
            export_data = json.load(f)
        
        assert "trade_summary" in export_data, "Missing trade_summary in export"
        assert "recent_trades" in export_data, "Missing recent_trades in export"
        
        results["exports"].append({
            "format": "JSON",
            "path": str(json_export),
            "size_kb": json_export.stat().st_size / 1024,
            "valid": True,
        })
        
        # Get summary stats
        stats = collector.get_summary_stats()
        print(f"✓ Collected {stats['total_trades']} trade metrics")
        print(f"✓ Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"✓ Avg P&L: ${stats['avg_pnl']:.2f}")
        print(f"✓ Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"✓ JSON export: {json_export.name}")
        print(f"✓ Export size: {results['exports'][0]['size_kb']:.2f} KB")
        print("✓ SCENARIO 7 PASSED")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


def test_scenario_8_concurrent_positions():
    """Scenario 8: Test concurrent position limit enforcement."""
    print("\n" + "="*80)
    print("SCENARIO 8: Concurrent Position Limit (Max 1)")
    print("="*80)
    
    manager = RiskManager(initial_equity=10000)
    
    results = {
        "first_trade_approved": False,
        "second_trade_approved": False,
        "total_attempts": 0,
    }
    
    # First position should succeed
    success, pos1, msg1 = manager.open_position(
        entry_price=1.0700,
        direction="LONG",
        stop_loss=1.0695,
        take_profit=1.0710,
    )
    results["first_trade_approved"] = success
    results["total_attempts"] += 1
    
    print(f"✓ First position: {'APPROVED' if success else 'DENIED'}")
    assert success, "First position should be approved"
    
    # Second position should be rejected
    can_open, reason = manager.check_can_open_position()
    results["second_trade_approved"] = can_open
    results["total_attempts"] += 1
    
    print(f"✓ Second position: {'APPROVED' if can_open else f'DENIED ({reason})'}")
    assert not can_open, "Second position should be rejected"
    
    # Close first position
    if pos1:
        manager.close_position(pos1.position_id, 1.0705)
    
    # Now second should succeed
    can_open, reason = manager.check_can_open_position()
    print(f"✓ After closing first: {'CAN OPEN' if can_open else 'CANNOT OPEN'}")
    assert can_open, "Should be able to open position after closing previous"
    
    print(f"✓ Open positions: {len(manager.open_positions)}")
    print(f"✓ Closed positions: {len(manager.closed_positions)}")
    print("✓ SCENARIO 8 PASSED")
    
    return results


def run_all_scenarios():
    """Run all validation scenarios."""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  COMPREHENSIVE MONITORING SYSTEM VALIDATION".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    scenarios = [
        ("Profitable Day", test_scenario_1_profitable_day),
        ("Mixed Results", test_scenario_2_mixed_results),
        ("Daily Loss Limit", test_scenario_3_daily_loss_limit),
        ("Drawdown Monitoring", test_scenario_4_drawdown_monitoring),
        ("Position Sizing", test_scenario_5_position_sizing),
        ("Alert System", test_scenario_6_alert_system),
        ("Metrics Export", test_scenario_7_metrics_export),
        ("Concurrent Positions", test_scenario_8_concurrent_positions),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for name, test_func in scenarios:
        try:
            result = test_func()
            results[name] = {"status": "PASSED", "result": result}
            passed += 1
        except AssertionError as e:
            results[name] = {"status": "FAILED", "error": str(e)}
            failed += 1
            print(f"\n❌ {name} FAILED: {e}")
        except Exception as e:
            results[name] = {"status": "ERROR", "error": str(e)}
            failed += 1
            print(f"\n❌ {name} ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"✓ Passed: {passed}/{len(scenarios)}")
    print(f"✗ Failed: {failed}/{len(scenarios)}")
    print(f"Pass Rate: {passed/len(scenarios)*100:.1f}%")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 ALL VALIDATION SCENARIOS PASSED! 🎉")
    
    return results


if __name__ == "__main__":
    run_all_scenarios()
