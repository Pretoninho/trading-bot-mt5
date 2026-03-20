"""
QUICK_START_MONITORING.md
==========================
Quick start guide for using the production monitoring system.
"""

# Production Monitoring - Quick Start Guide

## 5-Minute Setup

### 1. Import the Dashboard

```python
from monitoring_dashboard import IntegratedDashboard

# Create dashboard
dashboard = IntegratedDashboard(initial_equity=10000)
```

### 2. Trading Loop Integration

```python
# Before trading
current_equity = 10000

for bar in trading_data:
    # Get agent action
    action = agent(observation)
    
    # STEP 1: Request trade approval
    if should_open_trade:
        approval = dashboard.request_trade(
            entry_price=bar.close,
            direction="LONG",
            stop_loss=bar.close - 0.0010,
            take_profit=bar.close + 0.0015
        )
        
        if not approval["approved"]:
            continue  # Skip trade
    
    # STEP 2: Execute trade (your broker code)
    # ...
    
    # STEP 3: Record result when trade closes
    if position_closed:
        dashboard.record_trade(
            trade_id=trade_num,
            entry_price=entry,
            exit_price=exit,
            pnl=pnl,
            bars_held=duration
        )
    
    # STEP 4: Update equity
    current_equity = get_current_equity()
    dashboard.update_equity(current_equity)
    
    # STEP 5: Check violations
    violations = dashboard.check_violations()
    if violations["emergency_stop"]:
        print("🛑 EMERGENCY STOP!")
        break

# STEP 6: Export report
report = dashboard.export_session_report()
```

## Components Quick Reference

### ProductionMonitor
```python
from production_monitor import ProductionMonitor

monitor = ProductionMonitor()

# Record trades
monitor.record_trade(1, 1.0700, 1.0705, 50, 10)

# Check risk
result = monitor.check_risk_limits(10500, 10000)

# Get metrics
metrics = monitor.get_performance_metrics()
# → {"trades_total": 10, "win_rate": 0.7, ...}

# Print report
print(monitor.print_status_report(10500))
```

### RiskManager
```python
from risk_manager import RiskManager

manager = RiskManager(initial_equity=10000)

# Calculate size
size = manager.calculate_position_sizing(1.0700, 1.0695, risk_pct=2)

# Open position
success, position, msg = manager.open_position(
    entry_price=1.0700,
    direction="LONG",
    stop_loss=1.0695,
    take_profit=1.0710
)

# Close position
success, pnl, msg = manager.close_position(1, 1.0705)

# Get status
status = manager.get_status()
```

### MetricsCollector
```python
from metrics_collector import MetricsCollector, RealtimeMetricsMonitor

collector = MetricsCollector()

# Record metrics
collector.record_trade_metrics(
    win=True,
    pnl=50,
    duration=60,
    entry_price=1.0700,
    exit_price=1.0705
)

# Get stats
stats = collector.get_summary_stats()

# Export
collector.export_tensorboard_format()

# Display dashboard
monitor = RealtimeMetricsMonitor(collector)
monitor.print_dashboard()
```

## Risk Limits Reference

**Daily Limits**:
- Daily Loss: -5.0% (trading stops)
- Daily Gain: +10.0% (trading stops)

**Position Limits**:
- Max Position Size: 2.0 lots
- Max Concurrent: 1 position
- Risk Per Trade: 2% allocation

**Equity Protection**:
- Max Drawdown: 15.0% (emergency stop)
- Min Trade Profit: $10

## Alert Severities

| Level    | Usage |
|----------|-------|
| INFO     | Informational messages |
| SUCCESS  | Successful trades (large profit) |
| WARNING  | Caution needed |
| RISK     | Risk notification |
| DENIED   | Trade rejected |
| CRITICAL | Emergency stop |

## Output Files

**Automatically Generated**:
- `logs/production/session_*.json` - Session data
- `logs/production/alerts.log` - Alert history
- `logs/production/metrics_*.json` - Metrics snapshot
- `logs/production/report.md` - Markdown report

## Common Patterns

### Check if Can Trade
```python
can_open, reason = dashboard.risk_manager.check_can_open_position()
if can_open:
    # Open trade
```

### Get Current Status
```python
status = dashboard.risk_manager.get_status()
print(f"Equity: ${status['current_equity']:.2f}")
print(f"Drawdown: {status['max_drawdown']:.2f}%")
```

### Export Report
```python
report_path = dashboard.export_session_report()
print(f"Report saved to: {report_path}")
```

### Print Dashboard
```python
dashboard.print_full_dashboard(current_equity)
```

## Testing

```bash
# Run monitoring tests
pytest tests/test_monitoring.py -v

# Run specific component
pytest tests/test_monitoring.py::TestRiskManager -v
```

## Troubleshooting

**Issue**: "Max concurrent positions reached"
- Solution: Close existing position before opening new one
- Risk limit: `max_concurrent_positions = 1`

**Issue**: "Daily loss limit exceeded"
- Solution: Trading stops at -5% loss
- Check: `status['daily_pnl_pct']`

**Issue**: "Emergency stop triggered"
- Solution: Drawdown exceeded 15%
- Check: `status['max_drawdown']`

## Production Checklist

Before going live:
- [ ] Test monitoring in backtest
- [ ] Verify risk limits are appropriate
- [ ] Check alert notifications work
- [ ] Validate position sizing calculation
- [ ] Review session reports
- [ ] Confirm log file permissions
- [ ] Set up monitoring dashboard
- [ ] Test emergency stop procedure

## Next Steps

1. **Phase 6.5**: Visual dashboard with equity curves
2. **Phase 7**: Rollback procedures and incident response
3. **Live Trading**: Deploy with real broker connection

## Support Resources

- Full Guide: `MONITORING_DEPLOYMENT_GUIDE.md`
- Tests: `tests/test_monitoring.py`
- Components:
  - `production_monitor.py`
  - `risk_manager.py`
  - `metrics_collector.py`
  - `monitoring_dashboard.py`
