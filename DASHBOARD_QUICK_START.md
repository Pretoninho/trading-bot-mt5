# 📊 Dashboard Quick Start Guide - Phase 6.5

**Get started with the Risk Management Dashboard in 5 minutes**

---

## Step 1: Install Dependencies (1 minute)

```bash
cd /workspaces/trading-bot-mt5
pip install streamlit plotly pandas numpy
```

**Verify installation:**
```bash
streamlit --version
python -c "import plotly; print('Plotly OK')"
```

---

## Step 2: Start the Dashboard (1 minute)

```bash
streamlit run dashboard_app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Open in browser:** http://localhost:8501

---

## Step 3: Dashboard Overview (2 minutes)

### Top Section - Key Metrics
```
Daily P&L    │ Equity     │ Drawdown   │ Win Rate  │ Trades
-$150.00     │ $10,325.00 │ 2.50%      │ 60%       │ 5
+2.50% return│ +$325.00   │ Peak: 8%   │ 3W-2L     │ Today
```

**What it shows:**
- Today's profit/loss
- Current account balance
- Maximum lost value so far
- Percentage of profitable trades

### Charts Section
1. **Left Chart - Equity Curve**
   - Shows account balance over time
   - Green area = profit accumulation
   - Hover for exact values

2. **Right Chart - Daily P&L Gauge**
   - Color indicates risk status
   - Green = Safe, Orange = Warning, Red = Danger
   - Line shows daily loss limit

3. **Bottom Left - Drawdown**
   - Shows maximum loss from peak
   - Red dashed line = 15% limit
   - Lower is better

4. **Bottom Right - Win Rate**
   - Donut chart of wins vs losses
   - Blue = Winners percentage
   - Red = Losses percentage

### Risk Status Cards
Three status indicators:
```
Daily Loss Limit    │ Drawdown Protection │ Position Limit
-1.50% / -5.00%      │ 2.50% / 15.00%       │ 1 / 1
Remaining: 3.50%     │ Remaining: 12.50%    │ ✓ Within Limits
🟢 Safe              │ 🟢 Safe              │ 🟢 Safe
```

**What it means:**
- Green = Safe to trade
- Orange = Approaching limit
- Red = Limit breached

---

## Step 4: Basic Usage (1 minute)

### View Current Positions
Scroll down to **"💼 Current Positions"** table:
```
Symbol   │ Entry      │ Size      │ Current    │ P&L
EURUSD   │ $1.10000   │ 1.00 lot  │ $1.10500   │ $50.00
```

### Check Performance Metrics
Below positions, you'll see:
- **Profit Factor**: 3.6 (wins/losses ratio)
- **Sharpe Ratio**: 1.85 (risk-adjusted return)
- **Avg Win**: $150.00
- **Avg Loss**: -$62.50

### Refresh Controls
Top-right corner:
- **"🔄 Refresh Now"** - Manual refresh
- **"☑ Auto Refresh"** - Toggle automatic updates (default: ON)

---

## Quick Reference

### Color Meanings
| Color  | Meaning | Action |
|--------|---------|--------|
| 🟢 Green | Safe | Continue trading |
| 🟡 Orange | Warning | Be cautious |
| 🔴 Red | Danger | Stop/Review |

### Limits You're Monitoring
| Limit | Value | Alert |
|-------|-------|-------|
| Daily Loss | -5.0% | Exceeded |
| Max Drawdown | 15.0% | Exceeded |
| Open Positions | 1 max | At limit |
| Position Size | 2.0% risk | Per trade |

### What Each Section Shows
| Section | Shows | Frequency |
|---------|-------|-----------|
| Key Metrics | Summary stats | Real-time |
| Equity Curve | Account value over time | Real-time |
| Daily P&L | Today's profit/loss | Real-time |
| Drawdown | Max loss from peak | Real-time |
| Win Rate | Trade winners % | Real-time |
| Risk Cards | Limit status | Real-time |
| Positions | Open trades | Real-time |
| Performance | Strategy stats | Real-time |
| Alerts | Violations/Warnings | Real-time |

---

## Common Questions

### Q: How do I stop the dashboard?
**A:** Press `Ctrl+C` in the terminal where it's running.

### Q: Does the dashboard trade?
**A:** No, it only displays data. Trading happens in the main bot.

### Q: How often does it update?
**A:** Every 1 second (if Auto Refresh is enabled).

### Q: What if it shows "No open positions"?
**A:** This is normal if no trades are active. Just means the bot isn't trading.

### Q: Can I access it from another computer?
**A:** Yes, use the Network URL shown when starting (e.g., http://192.168.x.x:8501)

### Q: What if numbers don't update?
**A:** Click "Refresh Now" button or check if monitoring system is running.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit
```

### "Failed to initialize dashboard"
Check monitoring system is running:
```bash
# Verify trading environment is loaded
python -c "from checkpoints.production.monitoring_dashboard import IntegratedDashboard; print('OK')"
```

### "No data appearing in charts"
1. Ensure trades have been executed
2. Click "Refresh Now" button
3. Verify monitoring system has trade data
4. Check System Status indicator at bottom

### Charts showing but no data points
- This is normal if no trades yet
- Metrics still show current P&L
- Data will appear after first trade

---

## Advanced Tips

### Monitor Multiple Accounts
Run multiple dashboard instances:
```bash
streamlit run dashboard_app.py --server.port=8501  # Account 1
streamlit run dashboard_app.py --server.port=8502  # Account 2
```

Access at:
- Account 1: http://localhost:8501
- Account 2: http://localhost:8502

### Share Dashboard Link
Get network URL from startup message:
```bash
Network URL: http://192.168.1.100:8501
```
Share this with team members on same network.

### Customize Refresh Rate
Edit `dashboard_app.py`, line ~20:
```python
REFRESH_INTERVAL = 1  # Change to 2 for 2-second updates
```

### Dark Mode
Streamlit has built-in dark mode:
- Top-right menu → Settings → Theme → Dark
- Or automatic (follows system theme)

---

## What to Look For

### 🟢 Healthy System
- Green risk status cards
- Equity curve trending up
- Win rate > 50%
- Drawdown < 10%
- Minimal alerts

### ⚠️ Warning Signs
- Orange status cards
- Drawdown approaching 15%
- Win rate dropping
- Equity curve flat or down
- Multiple alerts

### 🔴 Critical
- Red risk cards
- Loss limit exceeded
- Drawdown > 15%
- Emergency stop alert
- Immediate action needed

---

## Next Steps

1. **Monitor daily**: Check dashboard each morning
2. **Review alerts**: Act on any warnings
3. **Analyze performance**: Review weekly stats
4. **Adjust limits**: Update if needed (DEPLOYMENT_CONFIG.py)
5. **Scale gradually**: Only increase position size with confidence

---

## Dashboard Features Summary

✅ **Real-time Monitoring**
- Live equity curve updates
- Instant P&L calculation
- Drawdown tracking
- Position monitoring

✅ **Risk Management**
- Daily loss protection (-5%)
- Drawdown monitoring (15% max)
- Position limits (1 max)
- Visual alerts

✅ **Performance Analysis**
- Win rate calculation
- Profit factor display
- Sharpe ratio tracking
- Trade statistics

✅ **Easy to Use**
- One-click refresh
- Auto-refresh available
- Clear color coding
- Intuitive layout

---

## Support

If you encounter issues:

1. **Check the logs**
   - Terminal output while running
   - Browser console (F12)

2. **Verify setup**
   - All dependencies installed
   - Monitoring system running
   - Trade data available

3. **Restart**
   ```bash
   # Stop current instance (Ctrl+C)
   # Clear cache
   rm -rf .streamlit
   # Restart
   streamlit run dashboard_app.py
   ```

4. **Review full documentation**
   - See `DASHBOARD_DEPLOYMENT_GUIDE.md` for complete details

---

## Production Checklist

Before using with real trading:

- [x] Dashboard starts without errors
- [x] Data connects from monitoring system
- [x] Charts display properly
- [x] Risk limits show correctly
- [x] Auto-refresh is working
- [x] Alerts trigger on violations
- [x] Understand all metrics displayed
- [x] Know how to stop bot if needed

---

**Happy Trading! 📈**

Dashboard v1.0 | Phase 6.5 | March 2026
