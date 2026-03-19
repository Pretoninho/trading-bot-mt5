"""Streamlit interface for bot configuration and data upload"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json
import tempfile
from bot_config import BotConfig

st.set_page_config(page_title="🤖 Bot Configurator", layout="wide")

st.title("🤖 Trading Bot Configurator")
st.markdown("Complete setup interface for EURUSD M1 PPO agent")

# ============= SIDEBAR NAVIGATION =============
st.sidebar.title("🎛️ Bot Setup")
page = st.sidebar.radio(
    "Choose Section", 
    [
        "📊 Data Files", 
        "⚙️ Environment", 
        "💰 Position Sizing", 
        "🧠 Training", 
        "🎯 Gates & Features"
    ]
)

# Load current config
config = BotConfig.load("bot_config.json")

# ============= PAGE: DATA FILES =============
if page == "📊 Data Files":
    st.header("📊 Data Files Upload")
    st.markdown("""
    Upload your market and economic calendar data. Files will be saved to the project root.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕯️ Market Data (EURUSD M1)")
        st.info("Format: MT5 export (space-separated): DATE TIME OPEN HIGH LOW CLOSE TICKVOL VOL SPREAD")
        eurusd_file = st.file_uploader("Upload EURUSD_M1.csv", type="csv", key="eurusd")
        if eurusd_file:
            config.eurusd_csv = eurusd_file.name
            st.success(f"✅ Loaded: {eurusd_file.name}")
            try:
                df_preview = pd.read_csv(eurusd_file, sep=r"\s+", nrows=5)
                st.dataframe(df_preview)
                st.caption(f"📈 File size: {eurusd_file.size / 1024 / 1024:.2f} MB")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col2:
        st.subheader("📅 Unsafe Weeks (Optional)")
        st.info("File with high-impact news weeks (CPI/NFP/FOMC/PPI)")
        unsafe_file = st.file_uploader(
            "Upload unsafe_weeks.csv (optional)", 
            type="csv", 
            key="unsafe"
        )
        if unsafe_file:
            config.unsafe_weeks_csv = unsafe_file.name
            st.success(f"✅ Loaded: {unsafe_file.name}")
            try:
                df_unsafe = pd.read_csv(unsafe_file, nrows=5)
                st.dataframe(df_unsafe)
            except Exception as e:
                st.error(f"Error reading file: {e}")

    st.divider()
    st.subheader("📑 FXStreet Calendar Files (Quarterly)")
    st.info("Upload FXStreet economic calendar exports (fx_q1.csv, fx_q2.csv, etc.)")
    
    uploaded_quarters = st.file_uploader(
        "Upload FXStreet quarterly files", 
        type="csv", 
        accept_multiple_files=True,
        key="fxstreet"
    )
    
    if uploaded_quarters:
        config.fxstreet_csvs = [f.name for f in uploaded_quarters]
        st.success(f"✅ Loaded {len(uploaded_quarters)} FXStreet files")
        
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        for idx, f in enumerate(uploaded_quarters):
            with [col_q1, col_q2, col_q3, col_q4][idx % 4]:
                st.caption(f"📄 {f.name}")
                st.caption(f"Size: {f.size / 1024:.1f} KB")

# ============= PAGE: ENVIRONMENT =============
elif page == "⚙️ Environment":
    st.header("⚙️ Environment Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Episode Limits")
        config.equity_gain_limit = st.slider(
            "Equity gain limit (%)", 
            min_value=0.01, max_value=0.1, 
            value=config.equity_gain_limit,
            step=0.01,
            help="Episode terminates if equity gains this much (+X%)"
        ) 
        config.equity_loss_limit = st.slider(
            "Equity loss limit (%)", 
            min_value=-0.1, max_value=-0.01, 
            value=config.equity_loss_limit,
            step=0.01,
            help="Episode terminates if equity loses this much (-X%)"
        )
        
        st.caption(f"Range: [{config.equity_loss_limit*100:.1f}%, {config.equity_gain_limit*100:.1f}%]")
    
    with col2:
        st.subheader("🎯 ATR & Stop Loss")
        config.atr_period = st.slider(
            "ATR Period", 
            min_value=5, max_value=30, 
            value=config.atr_period,
            help="ATR calculation period (bars)"
        )
        config.atr_sl_multiplier = st.slider(
            "ATR SL Multiplier", 
            min_value=1.0, max_value=4.0, 
            value=config.atr_sl_multiplier,
            step=0.1,
            help="SL distance = ATR × this multiplier (e.g., 2 ATR)"
        )
        
        st.caption(f"SL = {config.atr_period}-bar ATR × {config.atr_sl_multiplier}")
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🎲 Take Profit Targets")
        config.tp1_r_multiple = st.slider(
            "TP1 R-multiple", 
            min_value=0.5, max_value=3.0, 
            value=config.tp1_r_multiple,
            step=0.1,
            help="TP1 profit target = 1R (where R = SL distance)"
        )
        config.tp2_r_multiple = st.slider(
            "TP2 R-multiple", 
            min_value=1.0, max_value=5.0, 
            value=config.tp2_r_multiple,
            step=0.1,
            help="TP2 profit target = 2R"
        )
    
    with col4:
        st.subheader("💧 Position Management")
        config.partial_close_fraction = st.slider(
            "Partial close at TP (%)", 
            min_value=0.1, max_value=0.5, 
            value=config.partial_close_fraction,
            step=0.05,
            help="% of original position closed at each TP level"
        )
        config.observation_window = st.number_input(
            "Observation window (bars)", 
            min_value=32, max_value=256, 
            value=config.observation_window,
            step=32,
            help="Historical bars in state observation (larger = more context)"
        )

# ============= PAGE: POSITION SIZING =============
elif page == "💰 Position Sizing":
    st.header("💰 Position Sizing & Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📏 Lot Constraints")
        config.min_lot = st.number_input(
            "Minimum lot size", 
            value=config.min_lot, 
            step=0.01,
            format="%.2f",
            help="Smallest position size allowed (0.01 = 1 microlot)"
        )
        config.lot_step = st.number_input(
            "Lot step size", 
            value=config.lot_step, 
            step=0.01,
            format="%.2f",
            help="Position size increment (0.01 = 1 microlot steps)"
        )
        config.max_lot = st.number_input(
            "Maximum lot size", 
            value=config.max_lot, 
            step=1.0,
            format="%.1f",
            help="Largest position size allowed"
        )
        
        st.caption(f"Range: {config.min_lot} - {config.max_lot} lots")
    
    with col2:
        st.subheader("⚠️ Risk Management")
        risk_input = st.slider(
            "Risk per trade (%)", 
            min_value=0.1, max_value=5.0, 
            value=config.risk_per_trade * 100,
            step=0.1,
            help="% of equity risked per trade (standard: 1%)"
        )
        config.risk_per_trade = risk_input / 100
        
        # Show calculation
        loss_per_percent = config.initial_equity / 100
        st.info(f"💡 At ${config.initial_equity:,.0f} equity, each 1% loss = ${loss_per_percent:,.0f}")

    st.divider()
    
    st.subheader("📌 Position Sizing Formula")
    st.code("""
Equity Risk = Equity × Risk%  (default 1%)
Stop Loss Distance = |Entry - SL| in pips
Loss per Lot = Stop Loss Pips × $10/pip
Lots = Equity Risk / Loss per Lot
Lots_Final = Clamp(Lots, MIN, MAX) rounded to LOT_STEP
    """, language="text")

# ============= PAGE: TRAINING =============
elif page == "🧠 Training":
    st.header("🧠 PPO Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Environment Setup")
        config.initial_equity = st.number_input(
            "Initial equity ($)", 
            value=config.initial_equity, 
            step=1000.0,
            format="%.0f",
            help="Starting account balance for each episode"
        )
        config.device = st.selectbox(
            "Device", 
            ["cpu", "cuda"],
            help="CPU = slower but universal, CUDA = faster (requires NVIDIA GPU)"
        )
    
    with col2:
        st.subheader("🧠 Neural Network")
        config.hidden_dim = st.slider(
            "Hidden layer dimension", 
            min_value=64, max_value=512, 
            value=config.hidden_dim,
            step=64,
            help="Network capacity (64=lightweight, 256=balanced, 512=powerful)"
        )
        
        obs_size = config.observation_window * 6 + 7
        st.caption(f"Input: {obs_size} dims (391 default) → Hidden: {config.hidden_dim}")

    st.divider()
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.subheader("📚 Optimization")
        lr_options = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        config.learning_rate = st.select_slider(
            "Learning rate", 
            options=lr_options,
            value=config.learning_rate,
            help="Step size for weight updates (smaller = safer, slower)"
        )
        config.batch_size = st.slider(
            "Batch size", 
            min_value=32, max_value=256, 
            value=config.batch_size,
            step=32,
            help="Samples per gradient update"
        )
    
    with col4:
        st.subheader("📊 Discount & GAE")
        config.gamma = st.slider(
            "Gamma (discount)", 
            min_value=0.9, max_value=1.0, 
            value=config.gamma,
            step=0.01,
            help="How much to value future rewards vs immediate"
        )
        config.lambda_gae = st.slider(
            "Lambda (GAE)", 
            min_value=0.9, max_value=0.99, 
            value=config.lambda_gae,
            step=0.01,
            help="Advantage estimation smoothing (0.95 default)"
        )
    
    with col5:
        st.subheader("🔧 Gradient & Updates")
        config.epochs_per_update = st.slider(
            "Epochs per update", 
            min_value=1, max_value=10, 
            value=config.epochs_per_update,
            help="Training passes over same batch data"
        )
        config.max_grad_norm = st.slider(
            "Max gradient norm", 
            min_value=0.1, max_value=1.0, 
            value=config.max_grad_norm,
            step=0.1,
            help="Gradient clipping threshold (prevents explosion)"
        )

# ============= PAGE: GATES & FEATURES =============
elif page == "🎯 Gates & Features":
    st.header("🎯 Trading Gates & Advanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔐 Gating Filters")
        st.markdown("Restrict agent actions based on market conditions:")
        
        config.use_safe_week_gating = st.checkbox(
            "❌ Skip unsafe weeks", 
            value=config.use_safe_week_gating,
            help="Block trades during weeks with high-impact news (CPI/NFP/FOMC/PPI on Wed/Thu)"
        )
        config.use_tradable_window_gating = st.checkbox(
            "⏰ Tradable window only", 
            value=config.use_tradable_window_gating,
            help="Only allow trades Wed-Thu 13:00-21:00 UTC (reduced slippage zone)"
        )
        config.use_breakout_detection = st.checkbox(
            "📈 Breakout detection", 
            value=config.use_breakout_detection,
            help="Detect and flag weekly breakout patterns"
        )
        
        st.caption("Gated actions are forced to HOLD without penalty")
    
    with col2:
        st.subheader("🎮 Advanced Features")
        config.trailing_stop_enabled = st.checkbox(
            "🔄 Trailing stop", 
            value=config.trailing_stop_enabled,
            help="Automatically trail SL as price moves in profitable direction"
        )
        config.break_even_buffer = st.number_input(
            "Break-even buffer (pips)", 
            value=config.break_even_buffer * 10000,
            step=0.5,
            format="%.1f",
            help="Buffer added when moving SL to break-even (0 = exact)"
        ) / 10000
        
        st.divider()
        config.invalid_action_penalty = st.number_input(
            "Invalid action penalty", 
            value=config.invalid_action_penalty,
            format="%.2e",
            help="Penalty for actions blocked by gates (negative = discourages)"
        )

# ============= SUMMARY CARDS =============
st.divider()

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Initial Equity", f"${config.initial_equity:,.0f}")

with col_info2:
    st.metric("Observation Window", f"{config.observation_window} bars")

with col_info3:
    st.metric("Risk per Trade", f"{config.risk_per_trade*100:.2f}%")

# ============= BOTTOM CONTROLS =============
st.divider()

col_save, col_export, col_reset, col_load = st.columns(4)

with col_save:
    if st.button("💾 Save Configuration", use_container_width=True):
        config.save("bot_config.json")
        st.success("✅ Configuration saved to `bot_config.json`")

with col_export:
    config_json = json.dumps(config.to_dict(), indent=2)
    st.download_button(
        label="⬇️ Download JSON",
        data=config_json,
        file_name="bot_config.json",
        mime="application/json",
        use_container_width=True
    )

with col_reset:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        config = BotConfig()
        st.rerun()

with col_load:
    config_upload = st.file_uploader(
        "📥 Load JSON config",
        type="json",
        label_visibility="collapsed"
    )
    if config_upload:
        try:
            config = BotConfig.from_dict(json.load(config_upload))
            st.success("✅ Configuration loaded")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading config: {e}")

# Show current config summary in expandable section
st.divider()

with st.expander("📋 Full Configuration (JSON)"):
    st.json(config.to_dict())

# Footer
st.markdown("""
---
**🚀 Usage**: After configuring, save and use in your training script:
```python
from bot_config import BotConfig
config = BotConfig.load("bot_config.json")
env = EURUSDTradingEnv(..., initial_equity=config.initial_equity, ...)
```
""")
