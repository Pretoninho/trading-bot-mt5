"""
MONITORING_DEPLOYMENT_GUIDE.md
================================
Guide to deploying and using the production monitoring infrastructure.

Created in Phase 6.4 - Monitoring Infrastructure Setup
"""

# Production Monitoring Deployment Guide

## Overview

Phase 6.4 created a comprehensive monitoring and risk management system for production trading. This system consists of four integrated components:

1. **Production Monitor** - Tracks equity, trades, and risk metrics
2. **Risk Manager** - Enforces position sizing, risk limits, and stop conditions
3. **Metrics Collector** - Collects trading and training metrics
4. **Integrated Dashboard** - Combines all components with unified alerting

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Integrated Monitoring Dashboard                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Real-time Risk Management                           │ │
│  ├─────────────────────────────────────────────────────┤ │
│  │ • Equity tracking & drawdown monitoring            │ │
│  │ • Position sizing & limit enforcement              │ │
│  │ • Daily loss & gain stopping                       │ │
│  │ • Emergency stop conditions                        │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Performance Metrics & Logging                       │ │
│  ├─────────────────────────────────────────────────────┤ │
│  │ • Trade metrics (P&L, win rate, duration)          │ │
│  │ • Training metrics (loss, entropy)                 │ │
│  │ • Sharpe ratio & risk-adjusted returns             │ │
│  │ • TensorBoard export & reporting                   │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Unified Alerting & Logging                          │ │
│  ├─────────────────────────────────────────────────────┤ │
│  │ • Real-time severity-based alerts                  │ │
│  │ • Session logging & report generation              │ │
│  │ • JSON exports for analysis                        │ │
│  │ • Emergency notification system ready              │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Components Overview

### 1. ProductionMonitor (`production_monitor.py`)

**Class**: `ProductionMonitor`

**Responsibilities**:
- Record individual trades
- Track equity levels
- Monitor risk limits
- Generate status reports

**Key Methods**:

```python
# Record a completed trade
monitor.record_trade(
    trade_id=1,
    entry_price=1.0700,
    exit_price=1.0705,
    pnl=50.0,
    bars_held=10
)

# Record equity checkpoint
monitor.record_equity(10050.0)

# Check if risk limits are violated
check_result = monitor.check_risk_limits(equity=10500, initial_equity=10000)
# Returns: {"violations": [...], "warnings": [...], "ok": True/False}

# Generate status report
report = monitor.print_status_report(equity=10500, initial_equity=10000)
```

**Output Files**:
- `logs/production/session_YYYYMMDD_HHMMSS.json` - Session logs
- `logs/production/alerts.log` - Alert history

### 2. RiskManager (`risk_manager.py`)

**Class**: `RiskManager`

**Responsibilities**:
- Position sizing based on risk %
- Enforce position limits
- Check daily loss/gain limits
- Monitor drawdown
- Emergency stop conditions

**Key Methods**:

```python
# Create risk manager
manager = RiskManager(initial_equity=10000)

# Check if position can be opened
can_open, reason = manager.check_can_open_position()

# Calculate position size
size = manager.calculate_position_sizing(
    entry_price=1.0700,
    stop_loss=1.0695,
    risk_pct=2.0  # 2% of account per trade
)

# Open position
success, position, msg = manager.open_position(
    entry_price=1.0700,
    direction="LONG",  # or "SHORT"
    stop_loss=1.0695,
    take_profit=1.0710,
    size=100000  # optional, auto-calculated if None
)

# Close position
success, pnl, msg = manager.close_position(
    position_id=1,
    exit_price=1.0705
)

# Get current status
status = manager.get_status()
# Returns: {
#     "current_equity": 10100,
#     "session_pnl": 100,
#     "daily_pnl": 50,
#     "max_drawdown": 0.5,
#     "open_positions": 1,
#     "closed_positions": 10
# }

# Check violations
violations = manager.check_risk_violations()
# Returns: {
#     "stop_loss_hits": [...],
#     "take_profit_hits": [...],
#     "equity_alerts": [...],
#     "emergency_stop": True/False
# }
```

**Risk Limits** (from DEPLOYMENT_CONFIG):
- Daily Loss Limit: -5.0% (stops trading if exceeded)
- Daily Gain Target: +10.0% (stops trading if exceeded)
- Max Drawdown: 15.0% (emergency stop)
- Max Concurrent Positions: 1
- Max Position Size: 2.0 lots

### 3. MetricsCollector (`metrics_collector.py`)

**Class**: `MetricsCollector`

**Responsibilities**:
- Collect training metrics
- Collect episode metrics
- Collect trade metrics
- Generate reports and exports

**Key Methods**:

```python
# Create collector
collector = MetricsCollector()

# Record training step
collector.record_train_step(
    step=100,
    loss=0.25,
    value_loss=0.15,
    policy_loss=0.10,
    entropy=0.5,
    learning_rate=1e-4
)

# Record episode
collector.record_episode(
    episode=5,
    total_reward=150.0,
    episode_length=1000,
    win_rate=0.55,
    avg_trade_duration=60
)

# Record trade
collector.record_trade_metrics(
    win=True,
    pnl=50.0,
    duration=60,  # minutes
    entry_price=1.0700,
    exit_price=1.0705
)

# Get summary statistics
stats = collector.get_summary_stats()
# Returns: {
#     "total_trades": 50,
#     "win_rate": 0.55,
#     "avg_pnl": 25.50,
#     "std_pnl": 50.0,
#     "avg_duration": 60,
#     "sharpe_ratio": 0.51
# }

# Export to TensorBoard format
export_path = collector.export_tensorboard_format()
```

**Output Files**:
- `logs/production/metrics_YYYYMMDD_HHMMSS.json` - Metrics snapshot
- `logs/production/training_metrics.log` - Training history (JSON lines)
- `logs/production/episode_metrics.log` - Episode history (JSON lines)
- `logs/production/report.md` - Markdown report

### 4. IntegratedDashboard (`monitoring_dashboard.py`)

**Class**: `IntegratedDashboard`

**Responsibilities**:
- Unified interface to all monitoring components
- Trade approval workflow
- Alert management
- Session reporting

**Key Methods**:

```python
# Create integrated dashboard
dashboard = IntegratedDashboard(initial_equity=10000)

# Request to open a trade (validates risk limits)
approval = dashboard.request_trade(
    entry_price=1.0700,
    direction="LONG",
    stop_loss=1.0695,
    take_profit=1.0710
)
# Returns: {"approved": True/False, "position_id": 1, "reason": "..."}

# Record completed trade
dashboard.record_trade(
    trade_id=1,
    entry_price=1.0700,
    exit_price=1.0705,
    pnl=50.0,
    bars_held=60
)

# Update equity
dashboard.update_equity(10050.0)

# Check violations
violations = dashboard.check_violations()

# Log custom alert
dashboard.log_alert("INFO", "Trade executed successfully")

# Print full dashboard
dashboard.print_full_dashboard(equity=10050.0)

# Export session report
report_path = dashboard.export_session_report()
```

## Usage Example

```python
from monitoring_dashboard import IntegratedDashboard
import numpy as np

# Initialize
dashboard = IntegratedDashboard(initial_equity=10000)
current_equity = 10000

# Trading loop
for bar in market_data:
    # 1. Get agent action
    action = agent.get_action(observation)
    
    # 2. Request trade approval
    if action in [OPEN_LONG, OPEN_SHORT]:
        approval = dashboard.request_trade(
            entry_price=bar.close,
            direction="LONG" if action == OPEN_LONG else "SHORT",
            stop_loss=calculate_stop_loss(bar),
            take_profit=calculate_take_profit(bar)
        )
        
        if not approval["approved"]:
            continue  # Skip this action
    
    # 3. Execute trade (with broker)
    # ... broker execution code ...
    
    # 4. Record result
    if trade_closed:
        dashboard.record_trade(
            trade_id=trade_id,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            bars_held=bars_held
        )
    
    # 5. Update equity
    current_equity = backtest.equity
    dashboard.update_equity(current_equity)
    
    # 6. Check for violations
    violations = dashboard.check_violations()
    if violations["emergency_stop"]:
        break  # Stop trading immediately
    
    # 7. Print dashboard (every N bars)
    if bar_num % 1440 == 0:  # Every day
        dashboard.print_full_dashboard(current_equity)

# 8. Export session report
report_path = dashboard.export_session_report()
```

## Configuration

**Risk Configuration** (`DEPLOYMENT_CONFIG.py`):

```python
RISK_CONFIG = {
    "max_position_size": 2.0,  # Max lots per trade
    "max_daily_loss_pct": 5.0,  # Stop if down 5%
    "max_daily_gain_pct": 10.0,  # Stop if up 10%
    "max_concurrent_positions": 1,
    "max_holding_time_bars": 1440,  # 24h for M1
    "min_trade_profit_threshold": 10.0,  # $10 min profit
    "max_drawdown_threshold": 15.0,  # Emergency stop at 15% drawdown
}

MONITORING_CONFIG = {
    "enable_metrics_logging": True,
    "metrics_interval_seconds": 60,
    "enable_performance_tracking": True,
    "enable_equity_tracking": True,
    "log_level": "INFO",
    "alert_threshold_drawdown": 10.0,  # Alert at 10%
    "alert_threshold_loss": -2.0,  # Alert at -2%
}
```

## Alert System

**Alert Severity Levels**:
- `INFO` - Informational (trade opened, updated)
- `SUCCESS` - Successful operation (large profit)
- `WARNING` - Caution needed (approaching limits)
- `RISK` - Risk notification (large loss)
- `DENIED` - Trade rejected (risk limit)
- `CRITICAL` - Stop conditions (emergency stop)

**Alert Triggers**:
- Risk limit violations
- Trade rejections
- Large losses/profits
- Drawdown warnings
- Emergency stop conditions

**Alert Log**: `logs/production/alerts.log`

## Monitoring Outputs

### Session Reports (.json)

```json
{
  "timestamp": "2024-01-20T15:30:00",
  "session_duration": 3600,
  "initial_equity": 10000,
  "risk_manager_status": {...},
  "performance_metrics": {...},
  "trades": [...],
  "alerts": [...],
  "open_positions": [...]
}
```

### Performance Report (.md)

```markdown
# Production Metrics Report

## Trading Performance
- Total Trades: 50
- Win Rate: 55.0%
- Avg P&L: $25.50
- Sharpe Ratio: 0.51

## Risk Metrics
- Avg Trade Duration: 60 minutes
- Max Drawdown: 5.2%
```

## Testing

23 tests implemented in `tests/test_monitoring.py`:

```bash
# Run monitoring tests
pytest tests/test_monitoring.py -v

# Run all tests
pytest tests/ -v
```

**Test Coverage**:
- ✓ Trade recording & equity tracking
- ✓ Risk limit enforcement
- ✓ Position sizing calculations
- ✓ Violation detection
- ✓ Metrics collection & export
- ✓ Integrated dashboard workflow
- ✓ Full trading session simulation
- ✓ Risk limit enforcement

## Next Steps: Phase 6.5

**Risk Management Dashboard** will:
- Real-time equity curve visualization
- Daily P&L tracking
- Drawdown monitoring
- Position management interface
- Risk limit status
- Historical performance charts

## Production Deployment Checklist

- [x] Monitoring infrastructure created
- [x] Risk management system functional
- [x] Metrics collection working
- [x] Dashboard integration complete
- [x] 23 unit tests passing
- [x] 149 total tests passing
- [ ] Phase 6.5: Risk dashboard implementation
- [ ] Phase 7: Rollback procedures
- [ ] Live trading testing
- [ ] Performance optimization

## Resources

- **Production Monitor**: `production_monitor.py` (380 lines)
- **Risk Manager**: `risk_manager.py` (410 lines)
- **Metrics Collector**: `metrics_collector.py` (420 lines)
- **Integrated Dashboard**: `monitoring_dashboard.py` (520 lines)
- **Tests**: `tests/test_monitoring.py` (380+ lines)

## Support

For production issues:
1. Check `logs/production/alerts.log` for recent alerts
2. Review session report in `logs/production/session_*.json`
3. Export metrics: `collector.export_tensorboard_format()`
4. Print dashboard: `dashboard.print_full_dashboard(equity)`
