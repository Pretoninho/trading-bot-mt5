"""
Risk Management Dashboard - Phase 6.5
Real-time monitoring dashboard for trading bot risk management.

Streamlit application providing:
  - Real-time equity curve visualization
  - Daily P&L tracking and limits
  - Drawdown monitoring
  - Position management interface
  - Risk limit status display
  - Historical performance analysis
  - Interactive metrics dashboard

Author: Trading Bot Team
Date: March 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from monitoring_dashboard import IntegratedDashboard


# ============================================================================
# CONFIGURATION
# ============================================================================

REFRESH_INTERVAL = 1  # seconds
MAX_CHART_POINTS = 100  # limit historical points for performance
RISK_COLORS = {
    'safe': '#2ecc71',        # green
    'warning': '#f39c12',      # orange
    'danger': '#e74c3c',       # red
    'critical': '#c0392b'      # dark red
}


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_dashboard():
    """Initialize the monitoring dashboard with caching."""
    try:
        dashboard = IntegratedDashboard()
        return dashboard
    except Exception as e:
        st.error(f"Failed to initialize dashboard: {e}")
        return None


def get_session_state():
    """Get or initialize session state variables."""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'trade_data' not in st.session_state:
        st.session_state.trade_data = []
    if 'equity_history' not in st.session_state:
        st.session_state.equity_history = []
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    return st.session_state


# ============================================================================
# DATA RETRIEVAL & PROCESSING
# ============================================================================

def get_live_metrics(dashboard: IntegratedDashboard) -> Dict:
    """Get current live metrics from the monitoring system."""
    try:
        status = dashboard.get_status()
        metrics = dashboard.get_rich_metrics()
        return {
            'status': status,
            'metrics': metrics,
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.error(f"Error retrieving metrics: {e}")
        return None


def extract_equity_curve(metrics: Dict) -> Tuple[List[float], List[datetime]]:
    """Extract equity history from metrics."""
    trades = metrics.get('metrics', {}).get('trade_stats', {}).get('trades', [])
    
    equity_curve = []
    starting_equity = 10000.0  # Default starting equity
    current_equity = starting_equity
    timestamps = []
    
    for trade in trades:
        pnl = trade.get('pnl', 0)
        current_equity += pnl
        equity_curve.append(current_equity)
        # Create realistic timestamps
        timestamps.append(datetime.now() - timedelta(seconds=len(equity_curve)))
    
    # Add current equity
    if equity_curve:
        equity_curve.append(current_equity)
        timestamps.append(datetime.now())
    else:
        equity_curve = [starting_equity]
        timestamps = [datetime.now()]
    
    return equity_curve[-MAX_CHART_POINTS:], timestamps[-MAX_CHART_POINTS:]


def calculate_daily_pnl(metrics: Dict) -> Tuple[float, float, float]:
    """Calculate daily P&L, running total, and return percentage."""
    trades = metrics.get('metrics', {}).get('trade_stats', {}).get('trades', [])
    
    daily_pnl = 0.0
    winning_trades = 0
    losing_trades = 0
    
    for trade in trades:
        pnl = trade.get('pnl', 0)
        daily_pnl += pnl
        if pnl > 0:
            winning_trades += 1
        elif pnl < 0:
            losing_trades += 1
    
    starting_equity = 10000.0
    return_pct = (daily_pnl / starting_equity) * 100
    
    return daily_pnl, winning_trades, losing_trades, return_pct


def get_drawdown_info(equity_curve: List[float]) -> Tuple[float, float]:
    """Calculate current drawdown and peak drawdown."""
    if not equity_curve:
        return 0.0, 0.0
    
    peaks = np.maximum.accumulate(equity_curve)
    drawdown = (np.array(equity_curve) - peaks) / peaks * 100
    current_dd = drawdown[-1]
    peak_dd = np.min(drawdown)
    
    return abs(current_dd), abs(peak_dd)


def get_risk_status(dashboard: IntegratedDashboard) -> Dict:
    """Get current risk management status."""
    try:
        status = dashboard.get_status()
        return {
            'daily_loss_limit': -5.0,
            'daily_gain_target': 10.0,
            'max_drawdown': 15.0,
            'current_daily_loss': status.get('current_loss', 0),
            'current_daily_gain': status.get('current_gain', 0),
            'current_drawdown': status.get('drawdown', 0),
            'positions_open': status.get('positions_open', 0),
            'max_positions': 1,
            'violations': status.get('violations', []),
            'alerts': status.get('alerts', [])
        }
    except Exception as e:
        st.warning(f"Error retrieving risk status: {e}")
        return None


def get_position_details(dashboard: IntegratedDashboard) -> pd.DataFrame:
    """Get current open positions as DataFrame."""
    try:
        status = dashboard.get_status()
        positions = status.get('open_positions', [])
        
        if not positions:
            return pd.DataFrame(columns=['Symbol', 'Entry', 'Size', 'Current Price', 'PnL', 'Status'])
        
        df_data = []
        for pos in positions:
            df_data.append({
                'Symbol': pos.get('symbol', 'N/A'),
                'Entry Price': f"${pos.get('entry_price', 0):.5f}",
                'Size': f"{pos.get('size', 0):.2f} lots",
                'Current Price': f"${pos.get('current_price', 0):.5f}",
                'P&L': f"${pos.get('pnl', 0):.2f}",
                'Status': 'Open'
            })
        
        return pd.DataFrame(df_data)
    except Exception as e:
        st.warning(f"Error retrieving positions: {e}")
        return pd.DataFrame()


def get_performance_metrics(dashboard: IntegratedDashboard) -> Dict:
    """Get detailed performance metrics."""
    try:
        metrics = dashboard.get_rich_metrics()
        trade_stats = metrics.get('metrics', {}).get('trade_stats', {})
        
        return {
            'total_trades': trade_stats.get('total_trades', 0),
            'winning_trades': trade_stats.get('winning_trades', 0),
            'losing_trades': trade_stats.get('losing_trades', 0),
            'win_rate': trade_stats.get('win_rate', 0),
            'avg_win': trade_stats.get('avg_win', 0),
            'avg_loss': trade_stats.get('avg_loss', 0),
            'profit_factor': trade_stats.get('profit_factor', 0),
            'sharpe_ratio': metrics.get('metrics', {}).get('sharpe_ratio', 0),
        }
    except Exception as e:
        st.warning(f"Error retrieving performance metrics: {e}")
        return {}


# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================

def plot_equity_curve(equity_curve: List[float], timestamps: List[datetime]) -> go.Figure:
    """Create interactive equity curve plot."""
    fig = go.Figure()
    
    # Calculate running values for the curve
    starting_equity = 10000.0
    returns = [(v - starting_equity) / starting_equity * 100 for v in equity_curve]
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=equity_curve,
        mode='lines+markers',
        name='Equity',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.1)',
        hovertemplate='<b>Time:</b> %{x|%H:%M:%S}<br>' +
                      '<b>Equity:</b> $%{y:.2f}<br>' +
                      f'<b>Return:</b> %{{customdata:.2f}}%<extra></extra>',
        customdata=returns
    ))
    
    fig.update_layout(
        title='Real-Time Equity Curve',
        xaxis_title='Time',
        yaxis_title='Account Equity ($)',
        hovermode='x unified',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        height=400,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    return fig


def plot_daily_pnl(daily_pnl: float, daily_limit: float) -> go.Figure:
    """Create daily P&L gauge chart."""
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=daily_pnl,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'Daily P&L'},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [daily_limit, abs(daily_limit) * 2]},
            'bar': {'color': '#3498db'},
            'steps': [
                {'range': [daily_limit, daily_limit * 0.5], 'color': '#e74c3c'},
                {'range': [daily_limit * 0.5, 0], 'color': '#f39c12'},
                {'range': [0, abs(daily_limit)], 'color': '#2ecc71'},
                {'range': [abs(daily_limit), abs(daily_limit) * 2], 'color': '#27ae60'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': daily_limit
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig


def plot_drawdown_monitor(equity_curve: List[float]) -> go.Figure:
    """Create drawdown visualization."""
    if not equity_curve or len(equity_curve) < 2:
        return go.Figure()
    
    peaks = np.maximum.accumulate(equity_curve)
    drawdown = (np.array(equity_curve) - peaks) / peaks * 100
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=np.abs(drawdown),
        marker=dict(
            color=[RISK_COLORS['danger'] if dd < -10 else 
                  RISK_COLORS['warning'] if dd < -5 else 
                  RISK_COLORS['safe'] for dd in drawdown]
        ),
        name='Drawdown %',
        hovertemplate='<b>Drawdown:</b> %{y:.2f}%<extra></extra>'
    ))
    
    # Add max drawdown limit line
    fig.add_hline(y=15, line_dash='dash', line_color=RISK_COLORS['danger'],
                  annotation_text='Max Drawdown Limit (15%)', annotation_position='right')
    
    fig.update_layout(
        title='Historical Drawdown',
        xaxis_title='Trade Number',
        yaxis_title='Drawdown (%)',
        height=300,
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        margin=dict(l=60, r=100, t=60, b=60),
        showlegend=False
    )
    
    return fig


def plot_win_rate(metrics: Dict) -> go.Figure:
    """Create win rate visualization."""
    wins = metrics.get('winning_trades', 0)
    losses = metrics.get('losing_trades', 0)
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Winning Trades', 'Losing Trades'],
            values=[wins, losses],
            marker=dict(colors=[RISK_COLORS['safe'], RISK_COLORS['danger']]),
            hole=0.4,
            textposition='inside',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
        )
    ])
    
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    fig.update_layout(
        title=f'Win Rate: {win_rate:.1f}%',
        height=350,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig


# ============================================================================
# DASHBOARD LAYOUT
# ============================================================================

def render_dashboard():
    """Main dashboard rendering function."""
    
    # Page configuration
    st.set_page_config(
        page_title='Risk Management Dashboard',
        page_icon='📊',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .metric-card-danger {
            border-left-color: #e74c3c;
        }
        .metric-card-warning {
            border-left-color: #f39c12;
        }
        .metric-card-success {
            border-left-color: #2ecc71;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    session_state = get_session_state()
    dashboard = initialize_dashboard()
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title('📊 Risk Management Dashboard')
    with col2:
        refresh_button = st.button('🔄 Refresh Now', use_container_width=True)
    with col3:
        session_state.auto_refresh = st.checkbox('Auto Refresh', value=True)
    
    if not dashboard:
        st.error('Failed to initialize monitoring dashboard')
        return
    
    # Get current metrics
    live_metrics = get_live_metrics(dashboard)
    if not live_metrics:
        st.error('Unable to retrieve live metrics')
        return
    
    # ========================================================================
    # STATUS OVERVIEW
    # ========================================================================
    
    st.markdown('---')
    st.subheader('📈 Status Overview')
    
    metrics = live_metrics['metrics']
    equity_curve, timestamps = extract_equity_curve(metrics)
    daily_pnl, winning, losing, return_pct = calculate_daily_pnl(metrics)
    current_dd, peak_dd = get_drawdown_info(equity_curve)
    risk_status = get_risk_status(dashboard)
    perf_metrics = get_performance_metrics(dashboard)
    
    # Key metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            'Daily P&L',
            f'${daily_pnl:.2f}',
            f'{return_pct:+.2f}%',
            delta_color='inverse'
        )
    
    with col2:
        st.metric(
            'Current Equity',
            f'${equity_curve[-1]:.2f}' if equity_curve else '$10,000.00',
            f'${equity_curve[-1] - 10000:.2f}' if equity_curve else '$0.00'
        )
    
    with col3:
        st.metric(
            'Drawdown',
            f'{current_dd:.2f}%',
            f'Peak: {peak_dd:.2f}%'
        )
    
    with col4:
        st.metric(
            'Win Rate',
            f'{perf_metrics.get("win_rate", 0):.1f}%',
            f'{winning}W - {losing}L'
        )
    
    with col5:
        st.metric(
            'Total Trades',
            perf_metrics.get('total_trades', 0),
            'Today'
        )
    
    # ========================================================================
    # CHARTS - ROW 1
    # ========================================================================
    
    st.markdown('---')
    st.subheader('📉 Performance Charts')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            plot_equity_curve(equity_curve, timestamps),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            plot_daily_pnl(daily_pnl, risk_status['daily_loss_limit']),
            use_container_width=True
        )
    
    # ========================================================================
    # CHARTS - ROW 2
    # ========================================================================
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            plot_drawdown_monitor(equity_curve),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            plot_win_rate(perf_metrics),
            use_container_width=True
        )
    
    # ========================================================================
    # RISK MANAGEMENT STATUS
    # ========================================================================
    
    st.markdown('---')
    st.subheader('🛡️ Risk Management Status')
    
    col1, col2, col3 = st.columns(3)
    
    # Daily Loss Limit
    with col1:
        loss_pct = (risk_status['current_daily_loss'] / 10000) * 100 if risk_status else 0
        loss_remaining = risk_status['daily_loss_limit'] - loss_pct if risk_status else -5
        
        safe_level = loss_remaining / risk_status['daily_loss_limit'] * 100 if risk_status else 100
        color = RISK_COLORS['safe'] if safe_level > 50 else \
                RISK_COLORS['warning'] if safe_level > 20 else \
                RISK_COLORS['danger']
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color};">
        <h4>Daily Loss Limit</h4>
        <p><b>{loss_pct:.2f}%</b> / {risk_status['daily_loss_limit']:.2f}%</p>
        <p style="color: {color}; font-size: 12px;">Remaining: {loss_remaining:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Drawdown Protection
    with col2:
        dd_remaining = risk_status['max_drawdown'] - current_dd if risk_status else 15
        safe_level = dd_remaining / risk_status['max_drawdown'] * 100 if risk_status else 100
        
        color = RISK_COLORS['safe'] if safe_level > 50 else \
                RISK_COLORS['warning'] if safe_level > 20 else \
                RISK_COLORS['danger']
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color};">
        <h4>Drawdown Protection</h4>
        <p><b>{current_dd:.2f}%</b> / {risk_status['max_drawdown']:.2f}%</p>
        <p style="color: {color}; font-size: 12px;">Remaining: {dd_remaining:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Concurrent Positions
    with col3:
        color = RISK_COLORS['safe'] if risk_status['positions_open'] < risk_status['max_positions'] else \
                RISK_COLORS['warning']
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color};">
        <h4>Concurrent Positions</h4>
        <p><b>{risk_status['positions_open']}</b> / {risk_status['max_positions']}</p>
        <p style="color: {color}; font-size: 12px;">
        {'✓ Within limits' if risk_status['positions_open'] <= risk_status['max_positions'] else '⚠ At limit'}
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # POSITION DETAILS
    # ========================================================================
    
    st.markdown('---')
    st.subheader('💼 Current Positions')
    
    positions_df = get_position_details(dashboard)
    if not positions_df.empty:
        st.dataframe(positions_df, use_container_width=True, hide_index=True)
    else:
        st.info('No open positions')
    
    # ========================================================================
    # DETAILED METRICS
    # ========================================================================
    
    st.markdown('---')
    st.subheader('📊 Detailed Performance Metrics')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric('Profit Factor', f"{perf_metrics.get('profit_factor', 0):.2f}")
    
    with col2:
        st.metric('Sharpe Ratio', f"{perf_metrics.get('sharpe_ratio', 0):.2f}")
    
    with col3:
        avg_win = perf_metrics.get('avg_win', 0)
        st.metric('Avg Win', f'${avg_win:.2f}' if avg_win else 'N/A')
    
    with col4:
        avg_loss = perf_metrics.get('avg_loss', 0)
        st.metric('Avg Loss', f'${avg_loss:.2f}' if avg_loss else 'N/A')
    
    # ========================================================================
    # ALERTS & VIOLATIONS
    # ========================================================================
    
    if risk_status.get('violations') or risk_status.get('alerts'):
        st.markdown('---')
        st.subheader('⚠️ Alerts & Violations')
        
        if risk_status.get('violations'):
            for violation in risk_status['violations']:
                st.warning(f"🚨 {violation}")
        
        if risk_status.get('alerts'):
            for alert in risk_status['alerts']:
                st.info(f"ℹ️ {alert}")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown('---')
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.caption(f'Last Updated: {live_metrics["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}')
    
    with col2:
        st.caption('Trading Bot MT5 - Phase 6.5 Dashboard')
    
    with col3:
        st.caption('🟢 System Status: Online' if dashboard else '🔴 System Status: Offline')
    
    # ========================================================================
    # AUTO-REFRESH
    # ========================================================================
    
    if session_state.auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    render_dashboard()
