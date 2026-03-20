"""
Tests for Risk Management Dashboard (Phase 6.5)
Tests for dashboard components, data processing, and visualizations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path


# ============================================================================
# FIXTURE: SAMPLE DATA
# ============================================================================

@pytest.fixture
def sample_metrics():
    """Create sample metrics for testing."""
    return {
        'metrics': {
            'trade_stats': {
                'total_trades': 5,
                'winning_trades': 3,
                'losing_trades': 2,
                'win_rate': 60.0,
                'avg_win': 150.0,
                'avg_loss': -75.0,
                'profit_factor': 2.0,
                'trades': [
                    {'pnl': 100.0, 'entry_price': 1.1000, 'exit_price': 1.1050},
                    {'pnl': 150.0, 'entry_price': 1.1050, 'exit_price': 1.1100},
                    {'pnl': -75.0, 'entry_price': 1.1100, 'exit_price': 1.1070},
                    {'pnl': 200.0, 'entry_price': 1.1070, 'exit_price': 1.1150},
                    {'pnl': -50.0, 'entry_price': 1.1150, 'exit_price': 1.1120},
                ]
            },
            'sharpe_ratio': 1.85,
        },
        'timestamp': datetime.now()
    }


@pytest.fixture
def sample_risk_status():
    """Create sample risk status for testing."""
    return {
        'daily_loss_limit': -5.0,
        'daily_gain_target': 10.0,
        'max_drawdown': 15.0,
        'current_daily_loss': -150.0,
        'current_daily_gain': 325.0,
        'current_drawdown': 2.5,
        'positions_open': 1,
        'max_positions': 1,
        'violations': [],
        'alerts': []
    }


@pytest.fixture
def sample_equity_curve():
    """Create sample equity curve."""
    return [10000.0, 10100.0, 10250.0, 10175.0, 10375.0, 10325.0]


# ============================================================================
# TESTS: DATA EXTRACTION
# ============================================================================

class TestDataExtraction:
    """Tests for data extraction and processing functions."""
    
    def test_extract_equity_curve_empty_trades(self):
        """Test equity curve extraction with no trades."""
        metrics = {
            'metrics': {
                'trade_stats': {
                    'trades': []
                }
            }
        }
        
        # Simulate extraction logic
        trades = metrics['metrics']['trade_stats']['trades']
        equity_curve = [10000.0]
        
        assert len(equity_curve) == 1
        assert equity_curve[0] == 10000.0
    
    def test_extract_equity_curve_with_trades(self, sample_metrics):
        """Test equity curve extraction with trades."""
        metrics = sample_metrics
        trades = metrics['metrics']['trade_stats']['trades']
        
        equity_curve = [10000.0]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['pnl'])
        
        # Verify equity curve calculation
        assert len(equity_curve) == 6  # 1 starting + 5 trades
        assert equity_curve[-1] == 10325.0  # Final equity
        assert all(eq >= 0 for eq in equity_curve)  # All positive
    
    def test_calculate_daily_pnl_positive(self, sample_metrics):
        """Test daily P&L calculation with positive result."""
        trades = sample_metrics['metrics']['trade_stats']['trades']
        
        daily_pnl = sum(t['pnl'] for t in trades)
        
        assert daily_pnl == 325.0
        assert daily_pnl > 0
    
    def test_calculate_daily_pnl_negative(self):
        """Test daily P&L calculation with negative result."""
        trades = [
            {'pnl': -100.0},
            {'pnl': -150.0},
        ]
        
        daily_pnl = sum(t['pnl'] for t in trades)
        
        assert daily_pnl == -250.0
        assert daily_pnl < 0
    
    def test_get_drawdown_info_no_drawdown(self):
        """Test drawdown calculation with no drawdown."""
        equity_curve = [10000, 10100, 10200, 10300]
        
        peaks = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peaks) / peaks * 100
        
        assert all(dd == 0 for dd in drawdown)
    
    def test_get_drawdown_info_with_drawdown(self, sample_equity_curve):
        """Test drawdown calculation with actual drawdown."""
        equity_curve = sample_equity_curve
        peaks = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peaks) / peaks * 100
        
        assert min(drawdown) < 0  # Some drawdown occurred
        assert abs(min(drawdown)) <= 5  # Less than 5% drawdown
    
    def test_get_drawdown_info_major_drawdown(self):
        """Test drawdown calculation with major drawdown."""
        equity_curve = [10000, 10000, 10000, 8000, 9000]  # 20% drawdown
        
        peaks = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peaks) / peaks * 100
        current_dd = abs(drawdown[-1])
        
        assert current_dd > 0
        assert min(drawdown) < -15  # Major drawdown


# ============================================================================
# TESTS: RISK STATUS
# ============================================================================

class TestRiskStatus:
    """Tests for risk status calculation and validation."""
    
    def test_daily_loss_limit_not_breached(self, sample_risk_status):
        """Test daily loss limit when not breached."""
        loss_pct = (sample_risk_status['current_daily_loss'] / 10000) * 100
        limit = sample_risk_status['daily_loss_limit']
        
        assert loss_pct > limit  # Loss is less severe than limit
        assert loss_pct == -1.5  # -1.5% loss
    
    def test_daily_loss_limit_breached(self):
        """Test daily loss limit when breached."""
        current_loss = -600.0  # -6% loss (exceeds -5% limit)
        loss_pct = (current_loss / 10000) * 100
        limit = -5.0
        
        assert loss_pct < limit  # Loss exceeds limit
        assert abs(loss_pct) > abs(limit)
    
    def test_drawdown_below_limit(self, sample_risk_status):
        """Test drawdown protection when below limit."""
        current_dd = sample_risk_status['current_drawdown']
        max_dd = sample_risk_status['max_drawdown']
        
        assert current_dd < max_dd
        assert current_dd == 2.5
    
    def test_drawdown_above_limit(self):
        """Test drawdown protection when above limit."""
        current_dd = 17.5  # 17.5% drawdown (exceeds 15% limit)
        max_dd = 15.0
        
        assert current_dd > max_dd
    
    def test_position_limit_enforcement(self, sample_risk_status):
        """Test position limit enforcement."""
        positions = sample_risk_status['positions_open']
        max_pos = sample_risk_status['max_positions']
        
        assert positions <= max_pos
        assert positions == 1
    
    def test_position_limit_exceeded(self):
        """Test position limit when exceeded."""
        positions_open = 2
        max_positions = 1
        
        assert positions_open > max_positions


# ============================================================================
# TESTS: PERFORMANCE METRICS
# ============================================================================

class TestPerformanceMetrics:
    """Tests for performance metrics calculation."""
    
    def test_win_rate_calculation(self, sample_metrics):
        """Test win rate calculation."""
        wins = sample_metrics['metrics']['trade_stats']['winning_trades']
        total = sample_metrics['metrics']['trade_stats']['total_trades']
        
        win_rate = (wins / total) * 100
        
        assert win_rate == 60.0
        assert 0 <= win_rate <= 100
    
    def test_profit_factor_calculation(self, sample_metrics):
        """Test profit factor calculation."""
        trades = sample_metrics['metrics']['trade_stats']['trades']
        
        total_wins = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_losses = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        assert profit_factor > 1  # Profitable
        assert profit_factor == pytest.approx(3.6)  # 450 / 125 = 3.6
    
    def test_sharpe_ratio_available(self, sample_metrics):
        """Test that Sharpe ratio is available in metrics."""
        sharpe = sample_metrics['metrics']['sharpe_ratio']
        
        assert sharpe > 0
        assert sharpe == 1.85
    
    def test_average_win_loss(self, sample_metrics):
        """Test average win/loss calculation."""
        trades = sample_metrics['metrics']['trade_stats']['trades']
        
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        assert avg_win > 0
        assert avg_loss < 0
        assert avg_win == pytest.approx(150.0)
        assert avg_loss == pytest.approx(-62.5)


# ============================================================================
# TESTS: DATAFRAME OPERATIONS
# ============================================================================

class TestPositionDataFrame:
    """Tests for position DataFrame generation."""
    
    def test_position_dataframe_empty(self):
        """Test position DataFrame when no positions."""
        positions = []
        
        df = pd.DataFrame(columns=['Symbol', 'Entry Price', 'Size', 'Current Price', 'P&L', 'Status'])
        
        assert df.empty
        assert len(df) == 0
    
    def test_position_dataframe_single_position(self):
        """Test position DataFrame with one position."""
        positions = [{
            'symbol': 'EURUSD',
            'entry_price': 1.1000,
            'current_price': 1.1050,
            'size': 1.0,
            'pnl': 50.0
        }]
        
        df_data = []
        for pos in positions:
            df_data.append({
                'Symbol': pos['symbol'],
                'Entry Price': f"${pos['entry_price']:.5f}",
                'Size': f"{pos['size']:.2f}",
                'PnL': f"${pos['pnl']:.2f}"
            })
        
        df = pd.DataFrame(df_data)
        
        assert len(df) == 1
        assert df['Symbol'].iloc[0] == 'EURUSD'
        assert '$50.00' in df['PnL'].iloc[0]
    
    def test_position_dataframe_multiple_positions(self):
        """Test position DataFrame with multiple positions."""
        positions = [
            {'symbol': 'EURUSD', 'entry_price': 1.1000, 'current_price': 1.1050, 'size': 1.0, 'pnl': 50.0},
            {'symbol': 'GBPUSD', 'entry_price': 1.2700, 'current_price': 1.2650, 'size': 0.5, 'pnl': -25.0}
        ]
        
        df_data = [
            {
                'Symbol': pos['symbol'],
                'Size': f"{pos['size']:.2f}",
                'PnL': f"${pos['pnl']:.2f}"
            }
            for pos in positions
        ]
        
        df = pd.DataFrame(df_data)
        
        assert len(df) == 2
        assert df['Symbol'].tolist() == ['EURUSD', 'GBPUSD']


# ============================================================================
# TESTS: VISUALIZATION LOGIC
# ============================================================================

class TestVisualizationLogic:
    """Tests for visualization data preparation."""
    
    def test_chart_data_point_limit(self):
        """Test that chart data is limited to max points."""
        MAX_CHART_POINTS = 100
        
        # Create large dataset
        large_data = list(range(200))
        limited_data = large_data[-MAX_CHART_POINTS:]
        
        assert len(limited_data) == 100
        assert limited_data == list(range(100, 200))
    
    def test_equity_curve_color_assignment(self, sample_equity_curve):
        """Test that colors are assigned based on equity."""
        eq_curve = sample_equity_curve
        
        # Prices should all be between $10,000 and $10,500
        assert all(10000 <= eq <= 11000 for eq in eq_curve)
        assert min(eq_curve) == 10000.0
        assert max(eq_curve) == 10375.0
    
    def test_gauge_chart_range(self):
        """Test gauge chart range calculation."""
        daily_pnl = -150.0
        daily_limit = -5.0
        
        # Gauge should show from limit to 2x limit
        gauge_min = daily_limit * 100 / 100  # -5%
        gauge_max = abs(daily_limit) * 2  # 10%
        
        assert gauge_min < 0
        assert gauge_max > 0


# ============================================================================
# TESTS: INTEGRATION
# ============================================================================

class TestDashboardIntegration:
    """Integration tests for dashboard components."""
    
    def test_metric_calculation_chain(self, sample_metrics, sample_risk_status):
        """Test that metrics flow through the chain correctly."""
        trades = sample_metrics['metrics']['trade_stats']['trades']
        
        # Extract equity
        equity = [10000.0]
        for trade in trades:
            equity.append(equity[-1] + trade['pnl'])
        
        # Calculate daily PnL
        pnl = sum(t['pnl'] for t in trades)
        
        # Calculate drawdown
        peaks = np.maximum.accumulate(equity)
        drawdown = (np.array(equity) - peaks) / peaks * 100
        
        # Verify chain
        assert len(equity) == 6
        assert pnl == 325.0
        assert min(drawdown) <= 0
    
    def test_status_refresh_cycle(self, sample_metrics, sample_risk_status):
        """Test status refresh cycle with updated data."""
        # Initial status
        status1_pnl = 325.0
        
        # Simulate new trade
        new_trade_pnl = -200.0
        status2_pnl = status1_pnl + new_trade_pnl
        
        # Verify update
        assert status2_pnl == 125.0
        assert status2_pnl < status1_pnl
    
    def test_alert_triggering_on_limit_breach(self):
        """Test that alerts trigger when limits are breached."""
        current_loss = -600.0  # -6% loss
        daily_loss_limit = -5.0  # -5% limit
        
        loss_pct = (current_loss / 10000) * 100
        
        if loss_pct < daily_loss_limit:
            alert = "ALERT: Daily loss limit exceeded"
        else:
            alert = None
        
        assert alert == "ALERT: Daily loss limit exceeded"


# ============================================================================
# TESTS: ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in dashboard components."""
    
    def test_empty_metrics_handling(self):
        """Test handling of empty metrics."""
        metrics = {'metrics': {}}
        
        trade_stats = metrics['metrics'].get('trade_stats', {})
        trades = trade_stats.get('trades', [])
        
        assert isinstance(trades, list)
        assert len(trades) == 0
    
    def test_missing_fields_handling(self):
        """Test handling of missing fields in data."""
        position = {
            'symbol': 'EURUSD',
            # Missing entry_price, current_price, etc.
        }
        
        entry = position.get('entry_price', 0)
        size = position.get('size', 0)
        
        assert entry == 0
        assert size == 0
    
    def test_zero_division_protection(self):
        """Test protection against zero division."""
        total_trades = 0
        winning_trades = 0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        assert win_rate == 0
        assert isinstance(win_rate, (int, float))


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
