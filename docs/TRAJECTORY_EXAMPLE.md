# Trajectory Example: One Trading Day

## Overview

This document walks through a **complete trading day trajectory** showing how states, actions, rewards, and learning work in practice.

**Date:** 2025-01-08 (Wednesday, safe week, breakout week)
**Initial Equity:** $10,000 USD
**Market:** EURUSD M1

---

## State Sequence

### Time 08:00 UTC — Step 0: Market Opens

```python
S(0) = {
    # History: 64 bars of M1 data from previous day's close
    h_t[0:6] = [open_ret, high_ret, low_ret, close_ret, spread, atr14],  # 1st bar
    h_t[6:12] = [...],  # 2nd bar
    # ... 64 bars total (384 dims)
    
    # Context
    is_tradable_now=1.0,      # ✅ Wed 08:00 UTC is in tradable window
    is_safe_week=1.0,         # ✅ Week 01 has no high-impact news
    has_breakout=1.0,         # ✅ Tue close broke Mon range
    tp_state_norm=0.0,        # No position open yet
    break_even_set=0.0,       # N/A
    runner_trailing_active=0.0,  # N/A
    r_multiple_unrealized=0.0,   # Flat
}

equity = 10,000 USD
position = FLAT (closed)
```

**Observation shape:** (391,)

---

### Step 0 → Step 1: First Action (08:01 UTC)

**Policy Decision:** Agent's actor network reads observation & selects action

```python
# Actor forward pass
hidden = relu(observation @ W_shared + b_shared)
logits = hidden @ W_actor + b_actor  # shape (6,)
logits = logits / temperature  # temperature = 1.0 initially

probs = softmax(logits)
# Example: probs = [0.10, 0.60, 0.05, 0.15, 0.05, 0.05]
#          actions= [HOLD, OPEN_LONG, OPEN_SHORT, CLOSE, PROTECT, MANAGE_TP]

a(0) = sample(probs) = action 1 (OPEN_LONG)  # 60% probability
```

**Action:** OPEN_LONG
- Entry: mid (close from bar M1) = 1.0850
- Bid-Ask: Bid=1.08485, Ask=1.08515 (spread=3 pips)
- **Fill at Ask:** 1.08515
- ATR14 (previous) = 0.0025
- **SL:** 1.08515 - 2 × 0.0025 = 1.08015 ✓
- **TP1:** 1.08515 + 0.005 (1R) = 1.09015 ✓
- **TP2:** 1.08515 + 0.010 (2R) = 1.09515 ✓
- **Position Size:** 1% risk = (10,000 × 0.01) / (1.08515 - 1.08015) = 200,000 units

**State after Action:**

```python
S(1) = {
    # New bar (08:01): OHLCV data
    # Returns: (1.0850 - 1.0849) / 1.0849 = tiny open_ret
    # ... (new bar added to history)
    
    # Updated context
    is_tradable_now=1.0,       # Still in window
    is_safe_week=1.0,          # Still safe
    has_breakout=1.0,          # Still true
    tp_state_norm=0.0,         # TP1/TP2 not reached yet
    break_even_set=0.0,        # SL not at break-even
    runner_trailing_active=0.0,    # Trailing not active
    r_multiple_unrealized = (1.0850 - 1.08515) / 0.005 = -0.001R,  # Negative from fill
}

equity = 10,000 - (1.08515 - 1.08515) × 200,000 = 10,000 USD (no immediate change)
position = LONG, entry=1.08515, sl=1.08015, tp1=1.09015, tp2=1.09515
```

**Reward at t=0:**

$$r(0) = \frac{\text{equity}(1) - \text{equity}(0)}{\text{equity}(0)} - 10^{-4} \times \mathbb{1}[\text{invalid}]$$

- Equity unchanged: Δequity = 0
- Action valid: action in [0,5] ✓
- **r(0) = 0.0**

---

### Step 1 → Step 2: Position Moves (08:02-08:15 UTC, Various Steps)

```
08:02: Price = 1.0852 (+2 pips)
  S(2): r_multiple_unrealized = (1.0852 - 1.08515) / 0.005 = 0.001R
  Mark-to-market: PnL = 0.00005 × 200,000 = $10 ✓
  Equity = 10,010 USD
  
  Policy: agent_select_action(S(2)) = action 5 (MANAGE_TP)
  → Check if High reaches TP1 (1.09015) → NO, continue
  r(1) = 0.001 = +0.1%

08:03-08:10: Price consolidates around 1.0851-1.0857
  Several steps with action=HOLD (safest during consolidation)
  Rewards ≈ 0 (no significant equity change)
  
08:11: Price spikes to 1.0905
  S(12): r_multiple_unrealized = (1.0905 - 1.08515) / 0.005 = 0.908R
  Equity = 10,000 + 0.908 × 500 = 10,454 USD  ✓✓✓
  
  Policy: action=MANAGE_TP detects High reached TP1 (1.09015)
  → Close 25% of lots (50,000 units) at TP1 price
  Realized PnL: 50,000 × (1.09015 - 1.08515) = $250 ✓
  tp_state_norm: 0.0 → 0.5  (TP1 taken)
  
  Remaining position: 150,000 units, 75% of initial
  New SL: 1.08515 (still 2×ATR below entry, could move to break-even later)
  r(11) = +0.025 = +2.5%
```

---

### Step 12 → Step 50: Extended Position (08:12-09:00 UTC)

```python
# Agent manages trailing position

08:30: Price pulls back to 1.0878
  Unrealized on runner: (1.0878 - 1.08515) × 150,000 = $1,125
  But TP1 already realized → total equity = 10,454 USD (not recalculated)
  
  Policy: action=4 (PROTECT)
  → Check if tp_state >= 1 (YES) → Move SL to break-even
  → SL becomes 1.08515 + small buffer (e.g., 1 pip) = 1.08525
  break_even_set=1.0
  
  r(26) = 0 (no change, just protection)

09:00: Price reaches TP2 (1.09515)
  Agent detects High touched TP2 → action=MANAGE_TP
  → Close remaining 150,000 units at TP2
  Realized PnL: 150,000 × (1.09515 - 1.08515) = $1,500 ✓✓
  tp_state_norm: 0.5 → 1.0  (both TPs taken)
  Position: CLOSED ✅
  Total Daily PnL: $250 + $1,500 = $1,750
  
  r(59) = +0.175 = +17.5%
```

---

## Episode Aggregation

### Full Trajectory Summary

```
S(0) → a(0)=OPEN_LONG → r(0)=0.000
S(1) → a(1)=MANAGE_TP → r(1)=+0.001
S(2) → a(2)=HOLD       → r(2)=0.000
...
S(11) → a(11)=MANAGE_TP → r(11)=+0.025 [TP1 taken]
...
S(26) → a(26)=PROTECT   → r(26)=0.000  [SL to BE]
...
S(59) → a(59)=MANAGE_TP → r(59)=+0.175 [TP2 taken]
S(60) → a(60)=HOLD      → r(60)=0.000

... rest of day (mostly HOLD until close)

S(1439) → Episode END (EOD)
```

### Return Calculation (γ = 1.0)

$$G(0) = r(0) + r(1) + r(2) + ... + r(59) + ... + r(1439)$$

Assuming:
- Non-zero rewards only at entry/TP closes
- Rest are ~0 (holding)

$$G(0) ≈ 0 + 0.001 + 0 + ... + 0.025 + 0 + ... + 0.175 + 0 + ... + 0$$
$$G(0) ≈ +0.175 \text{ (total return = +17.5%)}$$

But equity was:
- Start: $10,000
- After TP1: $10,250 (2.5% gain)
- After TP2: $11,750 (17.5% gain)
- Final: $11,750

**Daily Return:** $11,750 / $10,000 = 1.175 = **+17.5%** ✅

---

## Critic (Value) Estimation

As agent traverses, **critic network** estimates **V(s)** for each state:

```python
# Critic predictions (estimated expected return from each state)

V(S(0)) = critic(S(0)) ≈ 0.10  # Market just opened, potential +10%?
V(S(1)) = critic(S(1)) ≈ 0.15  # Position entered, trending up
V(S(11)) = critic(S(11)) ≈ 0.05 # TP1 taken, less upside remaining
V(S(26)) = critic(S(26)) ≈ 0.02 # Protected, waiting for TP2
V(S(59)) = critic(S(59)) ≈ 0.00 # Closed, day ending
```

### Advantage Calculation (Credit Assignment)

```python
# Step 0: Open position
advantage(S(0), a(0)) = r(0) + γ V(S(1)) - V(S(0))
                      = 0 + 1.0 × 0.15 - 0.10
                      = +0.05

# ✓ Positive advantage: opening this position was good choice
#   (critic estimates better future from S(1))

# Step 11: Take TP1
advantage(S(11), a(11)) = r(11) + γ V(S(12)) - V(S(11))
                        = 0.025 + 1.0 × 0.05 - 0.05
                        = +0.025

# ✓ Positive advantage: taking TP1 was justified
#   (got +0.025 reward, future unchanged)

# Step 59: Close position
advantage(S(59), a(59)) = r(59) + γ V(S(60)) - V(S(59))
                        = 0.175 + 1.0 × 0.00 - 0.00
                        = +0.175

# ✓✓✓ Very positive: closing at TP2 was excellent
#     (large reward, position closed appropriately)
```

---

## Learning Update

After episode ends, actor-critic network updates:

```python
actor_loss = sum(- log(π(a_t|s_t)) × advantage_t) / T
           = -[log(π(OPEN_LONG|S0)) × 0.05 + log(π(MANAGE_TP|S11)) × 0.025 + ... 
             + log(π(MANAGE_TP|S59)) × 0.175] / 1440

critic_loss = sum(advantage_t²) / T
            = (0.05² + 0.025² + ... + 0.175²) / 1440

optimizer.zero_grad()
loss = actor_loss + critic_loss
loss.backward()  # ← compute gradients ∇W
optimizer.step()  # ← update weights: W := W - α∇W
```

### Result

After this episode:
- **Actor learns:** opening long in S(0), taking TPs in S(11) & S(59), protecting in S(26) had **high advantage**
  - Policy will increase probability of similar actions in similar states
  - π(OPEN_LONG | "breakout week, safe, tradable") ↑
  - π(MANAGE_TP | "position trending up") ↑

- **Critic learns:** V(S) estimates are calibrated better
  - V(S(0)) had prediction error vs realized → adjusted
  - More accurate future predictions next episode

- **Temperature anneal:** τ *= 0.995 → policy becomes more deterministic (less exploration)

---

## Key Takeaways

1. **Full Trajectory:** S→a→r→S' flow shows how RL agent navigates trading day
2. **Reward Signal:** Reflects equity changes (real profit/loss signal)
3. **Advantage:** Credit assignment isolates which actions were good
4. **Learning:** Positive advantages → increase action probability; negative → decrease
5. **Exploration:** Softmax policy (stochastic) + temperature decay → explore then exploit

This one-day example shows why actor-critic works well:
- Actor gets policy gradient signal from **actual rewards** ✓
- Critic stabilizes learning via value bootstrapping ✓
- Advantage reduces variance ✓
- No experience replay needed (on-policy) → simpler than DQN ✓

---

See **[MDP_FORMALIZATION.md](MDP_FORMALIZATION.md)** for complete mathematical notation.
