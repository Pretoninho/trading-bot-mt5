# trading-bot-mt5

Gymnasium-compatible EURUSD M1 reinforcement-learning trading environment with PPO support.

---

## Overview

This repository implements a Gymnasium-compatible reinforcement-learning trading
environment trained on MT5-exported EURUSD M1 OHLCV+spread data (UTC).

Key features:

* **News gating** — filter FXStreet calendar CSV exports to CPI/PPI/NFP/FOMC
  events and generate `unsafe_weeks.csv` (weeks that are skipped by the agent).
* **Weekly breakout gate** — each ISO week is marked tradable only if Tuesday's
  close breaks outside Monday's high–low range.
* **Tradable-window gate** — agent can trade only on Wednesday/Thursday between
  08:00–22:00 UTC in a safe, breakout week.
* **Full position management** — ATR-based SL, partial TP closes, break-even
  protection, and trailing stop via discrete actions.
* **Gymnasium `Env`** — `reset()` / `step()` API with a 64-bar observation
  window and early-stop on ±2 % daily equity change.

---

## Package structure

```tree
trading_env/
├── data/
│   ├── market_loader.py   # MT5 M1 CSV → enriched DataFrame
│   └── news_utils.py      # FXStreet CSV → news_events.csv + unsafe_weeks.csv
├── gating/
│   ├── breakout.py        # Mon range / Tue close → has_breakout
│   └── tradable_window.py # tradable_now gate
├── env/
│   └── trading_env.py     # EURUSDTradingEnv (Gymnasium)
└── utils/
    └── position_sizing.py # 1 % equity risk lot calculator
tests/
└── test_smoke.py          # Unit + smoke tests
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Python ≥ 3.11** is recommended.

---

## Step 1 — Export MT5 M1 data

1. Open **MetaTrader 5** → *Tools* → *History Center*.
2. Select **EURUSD** in the left panel, timeframe **M1**.
3. Ensure you have downloaded the data (right-click → *Download*).
4. Right-click on the EURUSD data → *Export* → select **Comma- or space-separated** CSV format and save.

The exported file has the format:

```text
<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
2025.01.06 00:00:00 1.02345 1.02390 1.02310 1.02360 312 0 8
```

---

## Step 2 — Download FXStreet calendar

1. Go to [FXStreet Economic Calendar](https://www.fxstreet.com/economic-calendar) (or search "FXStreet Economic Calendar").
2. Set the timezone to **UTC / GMT+0** in the calendar settings.
3. Filter events: **Impact = High**, **Currencies = USD, EUR**.
4. Export by quarter (**max ~3 months per download**) — you will receive files named
   e.g. `fx_q1.csv`, `fx_q2.csv`, `fx_q3.csv`, `fx_q4.csv` (exact naming varies).

FXStreet CSV columns: `Id, Start, Name, Impact, Currency`
(`Start` is `MM/DD/YYYY HH:MM:SS` in UTC).

---

## Step 3 — Generate `unsafe_weeks.csv`

```python
from trading_env.data.news_utils import convert_fxstreet_to_news_csv

events, unsafe_weeks = convert_fxstreet_to_news_csv(
    input_paths=["fx_q1.csv", "fx_q2.csv", "fx_q3.csv", "fx_q4.csv"],
    out_events="news_events.csv",
    out_unsafe_weeks="unsafe_weeks.csv",
)
print(unsafe_weeks.head())
```

This produces:

* `news_events.csv` — filtered CPI/PPI/NFP/FOMC events (columns:
  `time_utc, currency, impact, title`).
* `unsafe_weeks.csv` — ISO weeks that contain a high-impact event on
  **Wednesday or Thursday UTC** (columns: `iso_year, iso_week`).

---

## Step 4 — Load market data

```python
from trading_env.data.market_loader import load_mt5_m1_csv

df = load_mt5_m1_csv("EURUSD_M1.csv", unsafe_weeks_path="unsafe_weeks.csv")
print(df[["dt", "close", "spread_price", "iso_week", "is_safe_week"]].head())
```

---

## Step 5 — Add breakout and tradable-window flags

```python
from trading_env.gating.breakout import compute_week_breakout, add_breakout_flag
from trading_env.gating.tradable_window import add_tradable_flag

week_meta = compute_week_breakout(df)
df = add_breakout_flag(df, week_meta=week_meta)
df = add_tradable_flag(df)

print(df[["dt", "has_breakout", "tradable_now"]].head(20))
```

---

## Step 6 — Quick environment smoke test

```python
from datetime import date
from trading_env.env.trading_env import EURUSDTradingEnv

# df must already have tradable_now, is_safe_week, has_breakout columns
env = EURUSDTradingEnv(df, initial_equity=10_000.0, render_mode="human")

obs, info = env.reset(options={"day_date": date(2025, 1, 8)})
print("Initial obs shape:", obs.shape)  # (391,)

terminated = False
while not terminated:
    action = env.action_space.sample()   # random agent
    obs, reward, terminated, truncated, info = env.step(action)

print("Final equity:", info["equity"])
```

---

## Markov Decision Process (MDP)

This environment is formulated as a **finite-horizon Markov Decision Process** for learning trading policies.

### MDP Interaction Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADING DAY LOOP (1440 M1 bars)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  S(t) = State                                                   │
│  (391D = 384 bar features + 7 context)                          │
│     │                                                           │
│     ├─ h(t): 64 bars × 6 features (OHLC rets, spread, ATR14)    │
│     └─ c(t): 7 context (tradable_now, safe_week, position...)   │
│     │                                                           │
│     ↓                                                           │
│  Agent's Neural Network:                                        │
│  π_θ(·|s) = Softmax(Actor Output)  ← Policy (action probs)      │
│  V_φ(s)   = Critic Output           ← Value (state eval)        │
│     │                                                           │
│     ↓                                                           │
│  a(t) ∈ {0,1,2,3,4,5}  ← Action sampled or argmax               │
│  │ HOLD│LONG│SHORT│CLOSE│PROTECT│MANAGE_TP│                     │
│     │                                                           │
│     ↓                                                           │
│  Environment.step(a(t)):                                        │
│  • Apply action (subject to gating constraints)                 │
│  • Mark-to-market position                                      │
│  • Check SL (close if hit)                                      │
│  • Check TP (partial close if hit)                              │
│     │                                                           │
│     ├─ Next bar price: p(t+1)                                   │
│     ├─ Realized PnL: π(t)                                       │
│     └─ Unrealized value: u(t)                                   │
│     │                                                           │
│     ↓                                                           │
│  r(t) = Reward                                                  │
│  Primary: ΔEquity(t) / Equity(t-1)   ← Profit/loss %            │
│  Penalty: -10⁻⁴ × 𝟙[invalid action]   ← Action enforcement      │
│     │                                                           │
│     ↓                                                           │
│  S(t+1) = Next State                                            │
│  (Updated bars, position state, equity)                         │
│     │                                                           │
│     ↓                                                           │
│  Check Termination:                                             │
│  • IF Equity moved ±2% → Episode ends ✓                         │
│  • IF t=1440 (end of day) → Episode ends ✓                      │
│  • ELSE → Continue to S(t+1)                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **State Space** $\mathcal{S} \subset \mathbb{R}^{391}$

Each state $s_t = (h_t, c_t)$ contains full information to decide actions:

**History Component** $h_t \in \mathbb{R}^{384}$ (64 bars × 6 features):
- `open_return`, `high_return`, `low_return`, `close_return` (normalized price changes)
- `spread_price` (bid-ask cost)
- `atr14` (14-bar volatility measure)

**Context Component** $c_t \in \mathbb{R}^{7}$ (trading environment):
- `is_tradable_now` ∈ {0,1} — Can agent open/close positions now?
- `is_safe_week` ∈ {0,1} — Is current ISO week free of high-impact news?
- `has_breakout` ∈ {0,1} — Did Tuesday close break Monday's range?
- `tp_state_norm` ∈ {0, 0.5, 1} — Take-profit level (none/TP1/TP2 taken)
- `break_even_set` ∈ {0,1} — Stop-loss moved to break-even?
- `runner_trailing_active` ∈ {0,1} — Trailing stop active?
- `r_multiple_unrealized` ∈ ℝ — Unrealized PnL as risk multiples (e.g., 0.5R, 1.5R)

**Example State Value:**
```python
S(t) = [
  h(t),     # 64 bars of 6 features each = 384D
            # e.g., [0.001, 0.002, -0.0015, 0.0008, 0.00012, 8.5, ...]
  c(t)      # 7 context features = 7D
            # e.g., [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.015]
]           # Total: 391D
```

**Markov Property:** Future state depends only on $s_t$ and $a_t$, not on prior history. The 64-bar window encodes all necessary temporal information.

---

#### 2. **Action Space** $\mathcal{A} = \{0, 1, 2, 3, 4, 5\}$

Six discrete actions with preconditions (enforced via gating):

| ID | Action | Precondition | Effect |
|-------|------------|-----------------|--------|
| 0 | **HOLD** | Always | Do nothing; advance to next bar |
| 1 | **OPEN_LONG** | No position + tradable_now | SL = 2×ATR below; TP1 at 1R, TP2 at 2R |
| 2 | **OPEN_SHORT** | No position + tradable_now | SL = 2×ATR above; TP1 at 1R, TP2 at 2R |
| 3 | **CLOSE** | Position open | Close at market price (bid for long, ask for short) |
| 4 | **PROTECT** | Position open | Move SL to break-even; enable trailing if TP≥1 taken |
| 5 | **MANAGE_TP** | Position open | Close 25% at TP1 or TP2 if price reached |

**Gating Constraint:** If `tradable_now=0`, actions 1 and 2 (OPEN_LONG/SHORT) are masked—agent cannot open positions outside designated trading windows (Wed-Thu 08:00-22:00 UTC in safe weeks).

---

#### 3. **Reward Function** $R(s_t, a_t, s_{t+1}) \to \mathbb{R}$

Composite reward combining trading profit and action validity:

$$r_t = \underbrace{\frac{\text{Equity}_t - \text{Equity}_{t-1}}{\text{Equity}_{t-1}}}_{\text{Primary: profit/loss}} - \underbrace{10^{-4} \times \mathbb{1}[\text{invalid action}]}_{\text{Penalty: enforce constraints}}$$

**Components:**
- **Primary reward:** Fractional equity change (e.g., +50 pips on 1 lot = +0.0005 on $10k)
- **Invalid action penalty:** -0.0001 if agent attempts forbidden action (e.g., OPEN_LONG when not tradable)

**Example Trajectory:**
```python
# Episode starts: Equity = $10,000, position = flat, tradable_now = 1

t=0:  S(0) = [..., tradable_now=1, position=0, ...]
      Agent samples a(0) = OPEN_LONG
      → Entry at 1.0850, SL at 1.0830 (-2×ATR = 20 pips)
      r(0) = 0 (just opened, no PnL yet)

t=1:  Price moved to 1.0860
      S(1) = [..., unrealized=+10pips, position=1, ...]
      Agent samples a(1) = HOLD
      Equity now = $10,000 + (10 pips × 1 lot) = $10,000.01
      r(1) = 0.000001 ✓

t=2:  Price at 1.0870, TP1 triggered
      S(2) = [..., unrealized=+20pips, tp_state=0.5, ...]
      Agent samples a(2) = MANAGE_TP
      Close 25% (0.25 lots) at TP1 → Realized +5 pips
      r(2) = 0.000005 ✓

t=14: Price hits SL at 1.0830
      S(14) = [...]
      Remaining 0.75 lots auto-closed at SL → Realized -20 pips on 0.75 = -15 pips total
      Equity = $10,000 + 5 - 15 = $9,990
      r(14) = -0.0001 ✗
```

---

#### 4. **Episode Termination Conditions**

| Condition | Trigger | Reason |
|-----------|---------|--------|
| **Daily Loss** | $\|\Delta \text{Equity}\| \geq 2\%$ | Risk management (prevent large daily drawdowns) |
| **End-of-Day** | $t = 1440$ bars (24 hours M1) | Natural episode boundary |

Episode ends after whichever occurs first. On termination:
- Agent receives final reward $r_T$
- Learning algorithm processes entire trajectory $\{(s_t, a_t, r_t)\}_{t=0}^{T}$
- Episode starts fresh next training iteration

---

### Policy & Value Learning

**Policy** $\pi_\theta(a \mid s)$: Neural network that outputs probability distribution over 6 actions

$$\pi_\theta(a \mid s) = \text{Softmax}(\text{Actor}(s))$$

**Value** $V_\phi(s)$: Neural network that estimates expected return from state $s$

$$V_\phi(s) \approx \mathbb{E}[G_t \mid S_t = s]$$

**Learning Objective:** Maximize expected cumulative return (cumulative reward with discount factor $\gamma=1.0$ for finite horizon):

$$G_t = \sum_{k=0}^{T-1} r_{t+k}$$

---

### Actor-Critic PPO+GAE Learning

**Default Algorithm:** PPO (Proximal Policy Optimization) with GAE (Generalized Advantage Estimation)

**Components:**
- **Actor** network: learns policy $\pi_\theta(a|s)$ that selects high-reward actions
- **Critic** network: learns value $V_\phi(s)$ to bootstrap TD learning and reduce variance
- **Advantage**: $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — credit assignment signal
- **PPO-Clip**: Trust region via importance sampling — prevents too-large policy updates
- **GAE**: $\lambda$-weighted temporal-difference accumulation — variance reduction (λ=0.95 optimal)

**Why this works:**
- ✅ **Sample efficiency:** TD learning vs pure Monte-Carlo
- ✅ **Stability:** Advantage normalization + gradient clipping
- ✅ **Exploration:** Softmax policy enables stochastic action sampling
- ✅ **Fast convergence:** PPO trust region prevents divergence

See [`docs/MDP_FORMALIZATION.md`](docs/MDP_FORMALIZATION.md) and [`docs/RL_MATHEMATICAL_DERIVATIONS.md`](docs/RL_MATHEMATICAL_DERIVATIONS.md) for complete theory.

---

## Actions

| Index | Name         | Description                                    |
|-------|--------------|----------------------------------------------- |
| 0     | `HOLD`       | Do nothing                                     |
| 1     | `OPEN_LONG`  | Open a long position                           |
| 2     | `OPEN_SHORT` | Open a short position                          |
| 3     | `CLOSE`      | Close the current position at market           |
| 4     | `PROTECT`    | Move SL to break-even; trail if TP2 taken      |
| 5     | `MANAGE_TP`  | Close 25 % lots at TP1 or TP2 if price reached |

---

## Observation space

Shape: `(391,)` = 64 bars × 6 features + 7 scalars.

**Bar features** (normalized, per bar): `open_ret, high_ret, low_ret, close_ret, spread_price, atr14`

**Scalar features** (7 total):

* `is_tradable_now` — Boolean flag (1.0 or 0.0) if agent can trade now
* `is_safe_week` — Boolean flag if current ISO week is free of high-impact news
* `has_breakout` — Boolean flag if this week's Tuesday closed outside Monday's range
* `tp_state_norm` — Current TP level (0.0 = no TP, 0.5 = TP1 taken, 1.0 = TP2 taken)
* `break_even_set` — Boolean flag if stop-loss has been moved to break-even
* `runner_trailing_active` — Boolean flag if trailing stop is active
* `r_multiple_unrealized` — Current unrealized PnL in risk multiples (e.g., 0.5R, 1.5R)

---

## Production Monitoring (Phase 6.4)

Complete production-ready monitoring and risk management system:

- **Risk Manager**: Position sizing (2% risk), daily loss limits (-5%), drawdown protection (15%)
- **Production Monitor**: Equity tracking, trade analysis, performance metrics
- **Metrics Collector**: Training metrics, episode tracking, TensorBoard export
- **Integrated Dashboard**: Unified monitoring with real-time alerts

### Quick Start

```python
from monitoring_dashboard import IntegratedDashboard

dashboard = IntegratedDashboard(initial_equity=10000)

# Request trade approval (validates risk limits)
approval = dashboard.request_trade(
    entry_price=1.0700, direction="LONG",
    stop_loss=1.0695, take_profit=1.0710
)

if approval["approved"]:
    # ... execute trade ...
    dashboard.record_trade(1, 1.0700, 1.0705, pnl=50, bars_held=60)
    dashboard.update_equity(10050)
```

**Resources**:
- `QUICK_START_MONITORING.md` — 5-minute guide
- `MONITORING_DEPLOYMENT_GUIDE.md` — Full documentation
- `tests/test_monitoring.py` — 23 comprehensive tests (149 total)

---

## Risk Management Dashboard (Phase 6.5)

Real-time Streamlit dashboard for complete trading visibility:

- **Real-Time Equity Curve** — Interactive chart showing account balance over time
- **Daily P&L Display** — Color-coded gauge showing profit/loss and risk status
- **Drawdown Monitor** — Historical drawdown visualization with limit lines
- **Position Management** — Current open positions with entry/exit details
- **Risk Status Cards** — Visual indicators for daily loss limit, drawdown, positions
- **Performance Analytics** — Win rate, profit factor, Sharpe ratio, trade statistics
- **Alert System** — Real-time notifications for violations and warnings

### Quick Start

```bash
# Install dependencies
pip install streamlit plotly

# Start dashboard
streamlit run dashboard_app.py

# Access at: http://localhost:8501
```

### Features

- **Auto-Refresh**: 1-second real-time updates (configurable)
- **Manual Refresh**: Click "Refresh Now" for immediate update
- **Risk Display**: Shows distance to all risk limits (visual color coding)
- **Integration**: Direct connection to IntegratedDashboard monitoring system
- **Responsive**: Works on all screen sizes and browsers

**Resources**:
- `DASHBOARD_QUICK_START.md` — 5-minute setup guide
- `DASHBOARD_DEPLOYMENT_GUIDE.md` — Complete technical documentation
- `tests/test_dashboard.py` — 29 comprehensive tests

---

## Running tests

```bash
pytest tests/ -v                          # All tests (178)
pytest tests/test_monitoring.py -v        # Monitoring tests (23)
pytest tests/test_dashboard.py -v         # Dashboard tests (29)
```

---

## Dependencies

See `requirements.txt`:

```text
gymnasium>=0.29.0
numpy>=1.24.0
pandas>=2.0.0
pytest>=7.0.0
```
