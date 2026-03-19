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
  window and early-stop on ±2 % daily equity change.  Each episode spans
  **08:00–22:00 UTC** of a single calendar day.

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

This environment is formulated as a **finite-horizon Markov Decision Process**:

### Agent Interaction Loop

```text
State S(t) → Agent selects Action a(t) → Environment executes → Reward r(t) ↓
State S(t+1) → Agent selects Action a(t+1) → ... → Reward r(t+1) ↓
...continues until episode terminates
```

### Key Components

**State $S(t)$**: `(391,)` observation = 64-bar history + 7 context scalars

* Encodes all information needed to decide next action
* Satisfies Markov property: future only depends on current state, not history

**Action $a(t)$**: Discrete choice from {0=HOLD, 1=OPEN_LONG, 2=OPEN_SHORT, 3=CLOSE, 4=PROTECT, 5=MANAGE_TP}

* Subject to **gating constraints**: cannot open positions outside tradable window

**Reward $r(t)$**:
$$r(t) = \frac{\text{equity}(t) - \text{equity}(t-1)}{\text{equity}(t-1)} - 10^{-4} \times \mathbb{1}[\text{invalid action}]$$

* Primary: fractional equity change (profit/loss)
* Secondary: small penalty for attempting invalid actions

**Episode Termination**:

* End-of-day (1,440 M1 bars processed)
* OR equity reaches ±2% threshold (early stop)

### Policy & Learning

Agent learns a **policy** $\pi(a|s)$ that maps states to action probabilities.

Objective: Maximize expected cumulative return
$$G_t = \sum_{k=0}^{T-1} \gamma^k r_{t+k} \quad (\gamma=1.0 \text{ for trading-bot})$$

See [`docs/MDP_FORMALIZATION.md`](docs/MDP_FORMALIZATION.md) for complete mathematical specification.

### Actor-Critic Learning

Default implementation: **Actor-Critic A2C**

* **Actor** network: learns policy $\pi_\theta(a|s)$ (action selection)
* **Critic** network: learns value $V_\phi(s)$ (state evaluation)
* **Advantage**: $A(s,a) = r + \gamma V(s') - V(s)$ (credit assignment)

This approach balances:

* ✅ Sample efficiency (Temporal Difference learning)
* ✅ Stability (value-based component)
* ✅ Exploration (stochastic policy via softmax)

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

## Running tests

```bash
pytest tests/test_smoke.py -v
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
