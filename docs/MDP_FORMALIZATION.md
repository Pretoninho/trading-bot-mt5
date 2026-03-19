# Markov Decision Process Formalization

## Overview

The **trading-bot-mt5** environment implements a **finite-horizon Markov Decision Process (MDP)** for EURUSD M1 (1-minute) trading simulation. This document provides the complete mathematical formulation.

---

## 1. State Space $\mathcal{S}$

### 1.1 Definition

The state space is continuous and finite-dimensional:

$$\mathcal{S} \subseteq \mathbb{R}^{391}$$

Each state $s_t \in \mathcal{S}$ at time step $t$ consists of two components:

$$s_t = (h_t, c_t)$$

where:
- **$h_t \in \mathbb{R}^{384}$** is the **observation history** — a window of normalized OHLC data
- **$c_t \in \mathbb{R}^{7}$** is the **context/scalar features** — position and market state information

### 1.2 History Component $h_t$

The history window contains **64 bars** of price data with **6 features per bar**:

$$h_t = [b_1, b_2, \ldots, b_{64}] \in \mathbb{R}^{64 \times 6}$$

where each bar $b_i = [\text{open\_ret}_i, \text{high\_ret}_i, \text{low\_ret}_i, \text{close\_ret}_i, \text{spread}_i, \text{atr14\_norm}_i]$.

**Feature definitions:**

| Feature | Type | Range | Definition |
|---------|------|-------|------------|
| `open_ret` | Float | [-1, 1] | $(O_i - C_{i-1}) / C_{i-1}$ |
| `high_ret` | Float | [-1, 1] | $(H_i - C_{i-1}) / C_{i-1}$ |
| `low_ret` | Float | [-1, 1] | $(L_i - C_{i-1}) / C_{i-1}$ |
| `close_ret` | Float | [-1, 1] | $(C_i - C_{i-1}) / C_{i-1}$ |
| `spread` | Float | [0, ∞) | $(A_i - B_i)$ in price units |
| `atr14_norm` | Float | [0, 1] | $\text{ATR}_{14,i} / \text{ATR}_{14,\max}$ |

where $O_i, H_i, L_i, C_i$ are open, high, low, close prices; $A_i, B_i$ are ask/bid prices.

### 1.3 Context Component $c_t$

The context vector contains 7 scalar features describing the trading environment:

$$c_t = [c_1, c_2, c_3, c_4, c_5, c_6, c_7]^T$$

| Index | Feature | Type | Range | Meaning |
|-------|---------|------|-------|---------|
| 1 | `is_tradable_now` | Binary | {0, 1} | Agent can trade (Wed/Thu, 08:00-22:00 UTC) |
| 2 | `is_safe_week` | Binary | {0, 1} | Current ISO week free of high-impact news |
| 3 | `has_breakout` | Binary | {0, 1} | This week: Tue close broke Mon range |
| 4 | `tp_state_norm` | Float | {0, 0.5, 1} | TP state (0=none, 0.5=TP1 taken, 1=TP2 taken) |
| 5 | `break_even_set` | Binary | {0, 1} | SL moved to break-even |
| 6 | `runner_trailing_active` | Binary | {0, 1} | Trailing stop is active |
| 7 | `r_multiple_unrealized` | Float | ℝ | Unrealized PnL in risk multiples (e.g., 0.5R, 1.5R) |

### 1.4 State Dynamics

States are generated sequentially from the MT5 market data. State transitions are **deterministic** given the agent's action and market data:

$$s_{t+1} = f(s_t, a_t, b_{t+1})$$

where $b_{t+1}$ is the next market bar (fixed from historical data).

---

## 2. Action Space $\mathcal{A}$

### 2.1 Definition

The action space is **discrete** and finite:

$$\mathcal{A} = \{0, 1, 2, 3, 4, 5\}$$

| Action | Name | Precondition | Effect |
|--------|------|--------------|--------|
| 0 | `HOLD` | Always allowed | Do nothing; move to next bar |
| 1 | `OPEN_LONG` | No position open | Enter long at ask; compute SL (entry - 2×ATR), TP1 (entry + R), TP2 (entry + 2R) |
| 2 | `OPEN_SHORT` | No position open | Enter short at bid; compute SL (entry + 2×ATR), TP targets |
| 3 | `CLOSE` | Position open | Close position at market (bid for long, ask for short) |
| 4 | `PROTECT` | Position open | Move SL to break-even; enable trailing if TP state ≥ 1 |
| 5 | `MANAGE_TP` | Position open | Check bar's High/Low; close 25% of lots at TP1 or TP2 if reached |

### 2.2 Action Constraints (Gating)

The environment enforces a **trading window gate**. If the agent attempts an invalid action outside permitted times, it is silently converted to `HOLD`:

$$a'_t = \begin{cases}
a_t & \text{if } c_t[\text{is\_tradable\_now}] = 1 \\
0 \text{ (HOLD)} & \text{if } a_t \neq 0 \text{ and } c_t[\text{is\_tradable\_now}] = 0
\end{cases}$$

**Note:** There is **no penalty** for forced `HOLD` actions; the agent simply cannot trade outside permitted windows.

---

## 3. Transition Function $P(s' | s, a)$

### 3.1 Deterministic Transitions

Given the fixed market data (historical MT5 bars), the transition function is **deterministic**:

$$P(s' | s, a) = \begin{cases}
1 & \text{if } s' = f(s, a, \beta_{t+1}) \\
0 & \text{otherwise}
\end{cases}$$

where $\beta_{t+1}$ is the next market bar in the historical dataset.

### 3.2 Transition Mechanics

The transition proceeds as follows:

#### Step 1: Stop-Loss Check (before action)

If a position is open and the current bar touches the stop-loss level:

$$\text{stopped\_out} = \begin{cases}
\text{True} & \text{if } \text{pos} = 1 \text{ and } L_t \leq \text{sl\_price} \\
\text{True} & \text{if } \text{pos} = -1 \text{ and } H_t \geq \text{sl\_price} \\
\text{False} & \text{otherwise}
\end{cases}$$

If stopped out, close entire position at SL price; skip agent action for this bar.

#### Step 2: Apply Action

Execute the agent's action (after constraint enforcement):

- **OPEN_LONG:**
  - Entry price: $P_{\text{entry}} = C_t + \frac{\text{spread}_t}{2}$ (ask price)
  - SL: $P_{\text{SL}} = P_{\text{entry}} - 2 \times \text{ATR}_{14,(t-1)}$
  - TP1: $P_{\text{TP1}} = P_{\text{entry}} + R$ where $R = 2 \times \text{ATR}_{14,(t-1)}$
  - TP2: $P_{\text{TP2}} = P_{\text{entry}} + 2R$
  - Lots: $\text{lots} = \frac{0.01 \times \text{equity}}{(P_{\text{entry}} - P_{\text{SL}}) \times 10}$ (1% risk sizing)
  - $\text{tp\_state} \gets 0$, $\text{break\_even\_set} \gets \text{False}$

- **OPEN_SHORT:** Symmetric to LONG (entry at bid, SL above)

- **CLOSE:**
  - Exit price: $(C_t - \frac{\text{spread}_t}{2})$ for long, $(C_t + \frac{\text{spread}_t}{2})$ for short
  - Compute realized PnL: $\Delta P = (P_{\text{exit}} - P_{\text{entry}}) \times \text{lots}$
  - Update equity: $\text{equity} \gets \text{equity} + \Delta P$
  - Reset position state: $\text{pos} \gets 0$

- **MANAGE_TP:**
  - Check if bar High reaches TP1: if so and $\text{tp\_state} = 0$, close 25% at TP1, $\text{tp\_state} \gets 1$
  - Check if bar High reaches TP2: if so and $\text{tp\_state} = 1$, close 25% at TP2, $\text{tp\_state} \gets 2$

- **PROTECT:**
  - If $\text{tp\_state} \geq 1$ or unrealized move $\geq 1R$: move SL to entry ± `be_buffer`; set $\text{break\_even\_set} \gets \text{True}$
  - If $\text{tp\_state} = 2$ (both TPs taken): enable trailing stop
    - For long: $\text{SL} \gets \max(\text{SL}, C_t - 2 \times \text{ATR}_{14,t})$
    - For short: $\text{SL} \gets \min(\text{SL}, C_t + 2 \times \text{ATR}_{14,t})$

#### Step 3: Mark-to-Market

Update unrealized equity based on current position's mark-to-market value:

$$\text{equity}_t = \text{equity}_{\text{realized}} + \begin{cases}
\text{lots} \times (C_t - P_{\text{entry}}) & \text{if pos} = 1 \\
\text{lots} \times (P_{\text{entry}} - C_t) & \text{if pos} = -1 \\
0 & \text{if pos} = 0
\end{cases}$$

#### Step 4: Generate Next State

Construct $s_{t+1}$ from:
- Updated history window $h_{t+1}$ (shift old bars, append bar $t+1$)
- Updated context $c_{t+1}$ from dataset and position state

---

## 4. Reward Function $R(s_t, a_t, s_{t+1})$

### 4.1 Definition

The reward at time step $t$ is:

$$r_t = r_t^{\text{equity}} + r_t^{\text{penalty}}$$

### 4.2 Equity Return Component

$$r_t^{\text{equity}} = \frac{\text{equity}_t - \text{equity}_{t-1}}{\text{equity}_{t-1}}$$

This is the fractional change in equity from the mark-to-market at step $t-1$ to step $t$.

**Interpretation:**
- **Positive reward:** Equity increased (profitable position or closed trade)
- **Negative reward:** Equity decreased (unprofitable position or loss)
- **Zero reward:** No change (e.g., holding with no mark-to-market change)

### 4.3 Penalty Component

$$r_t^{\text{penalty}} = \begin{cases}
-10^{-4} & \text{if agent attempted non-HOLD action while } c_t[\text{tradable\_now}] = 0 \\
0 & \text{otherwise}
\end{cases}$$

**Rationale:** Small penalty for invalid actions to discourage the agent from violating trading constraints, but not severe enough to dominate the equity-based reward.

### 4.4 Example Reward Sequence

Episode with 10 steps (simplified):

| Step | Equity | Equity Return | Action Penalty | Total Reward |
|------|--------|----------------|---|----------|
| 1 | $10,000 | — | 0 | — |
| 2 | $10,005 | +0.0005 | 0 | +0.0005 |
| 3 | $10,003 | -0.0002 | 0 | -0.0002 |
| 4 | $10,008 | +0.0005 | $-10^{-4}$ | +0.0003 |
| ... | ... | ... | ... | ... |

---

## 5. Episode Termination

### 5.1 Time Horizon

Each episode corresponds to **one UTC calendar day** (approximately 1,440 minutes of M1 data).

$$T_{\max} = \text{number of M1 bars in the day}$$

### 5.2 Termination Conditions

An episode terminates (flag `terminated = True`) when **either** condition is met:

#### Condition 1: End of Day
$$t \geq T_{\max}$$

All bars for the day have been processed.

#### Condition 2: Equity Limit Hit

$$\frac{\text{equity}_t - \text{equity}_0}{\text{equity}_0} \in [-0.02, +0.02]$$

Specifically:

$$\text{terminated} = \begin{cases}
\text{True} & \text{if } \frac{\text{equity}_t - \text{equity}_0}{\text{equity}_0} \geq +0.02 \text{ (gain limit)} \\
\text{True} & \text{if } \frac{\text{equity}_t - \text{equity}_0}{\text{equity}_0} \leq -0.02 \text{ (loss limit)} \\
\text{False} & \text{otherwise}
\end{cases}$$

**Rationale:**
- Agents that achieve 2% daily gains/losses are considered "converged" for the day
- Prevents excessively long/short episodes
- Models realistic risk management stop-outs

### 5.3 Truncation

The `truncated` flag is set to `False` for all steps in this environment. (Gymnasium standard: used for time limits imposed by the environment, not by the problem.)

---

## 6. Initial State Distribution

### 6.1 Random Day Selection

At reset, an episode is initialized by **randomly selecting a calendar day** from the available data:

$$d^* \sim \text{Uniform}(\{d_1, d_2, \ldots, d_N\})$$

where $d_i$ are all unique dates in the MT5 dataset.

### 6.2 Day Validation

If the selected day has **fewer than $65$ bars** (enough for 64-bar history + current bar), a fallback empty episode is returned, and the agent is expected to reset again.

$$\text{day\_valid} = (\text{number of bars in day}) \geq 65$$

### 6.3 Initial Equity

$$\text{equity}_0 \in \mathbb{R}^+ \quad \text{(default: 10,000 USD)}$$

Specified at environment construction; agent starts flat (no position).

---

## 7. Discount Factor

### 7.1 Definition

For finite-horizon MDPs (episodic), the discount factor is:

$$\gamma = 1.0$$

**Rationale:** Within an episode, all rewards are treated equally (no temporal discounting). The episode naturally ends at the time horizon $T_{\max}$ or equity limits.

### 7.2 Return Calculation

The return at time $t$ is:

$$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k} = \sum_{k=0}^{T-t-1} r_{t+k}$$

(sum of all future rewards in the episode)

---

## 8. Observation Function

### 8.1 Flattened Observation

The environment returns observations as a **flattened vector** of shape $(391,)$:

$$o_t = \text{flatten}(h_t) \oplus c_t \in \mathbb{R}^{391}$$

where:
- $\text{flatten}(h_t)$: Flatten $64 \times 6$ history matrix to length 384
- $\oplus$: Concatenation operator
- $c_t$: 7-dimensional context vector

**Order in flattened vector:**
```
[bar_1_open_ret, bar_1_high_ret, ..., bar_1_atr14,
 bar_2_open_ret, ..., bar_2_atr14,
 ...
 bar_64_open_ret, ..., bar_64_atr14,   # Total 384 elements
 is_tradable_now, is_safe_week, has_breakout,
 tp_state_norm, break_even_set, runner_trailing_active, r_multiple_unrealized]  # 7 elements
```

### 8.2 Normalization

- **OHLC returns:** Normalized to $[-1, 1]$ (naturally bounded from price changes)
- **Spread:** Raw price units (typically 0.0001–0.0005 for EURUSD)
- **ATR14:** Normalized to $[0, 1]$ by max observed ATR
- **Binary features:** $\{0, 1\}$
- **tp_state_norm:** $\{0, 0.5, 1\}$
- **r_multiple:** Raw float (unbounded, e.g., $-0.5$ to $+3.0$)

---

## 9. Gymnasium API Compliance

### 9.1 reset() Method

```python
observation, info = env.reset(
    seed=None,
    options={"day_date": date(2025, 1, 8)}  # Optional: specify day
)
```

**Returns:**
- `observation`: $o_0 \in \mathbb{R}^{391}$ (initial state observation)
- `info`: Dict with keys `{"day": str, "equity": float}`

### 9.2 step() Method

```python
observation, reward, terminated, truncated, info = env.step(action)
```

**Inputs:**
- `action`: $a \in \{0, 1, 2, 3, 4, 5\}$

**Returns:**
- `observation`: $o_{t+1} \in \mathbb{R}^{391}$
- `reward`: $r_t \in \mathbb{R}$ (scalar)
- `terminated`: Boolean (episode finished)
- `truncated`: Boolean (always False)
- `info`: Dict with keys `{"equity": float, ...}`

---

## 10. Summary Table

| Component | Definition | Type | Range/Size |
|-----------|-----------|------|-----------|
| **State Space** $\mathcal{S}$ | Continuous | $\mathbb{R}^{391}$ | Unbounded |
| **Action Space** $\mathcal{A}$ | Discrete | $\{0,1,2,3,4,5\}$ | 6 actions |
| **Transition** $P(s' \| s,a)$ | Deterministic | Function | Fixed by data |
| **Reward** $r_t$ | Equity return + penalty | $\mathbb{R}$ | Unbounded |
| **Initial State** $s_0$ | Random day + flat position | Distribution | Varies |
| **Episode Length** $T$ | Variable | Integer | 60–1440 bars ≈ 1 day |
| **Discount Factor** $\gamma$ | Finite-horizon | Float | 1.0 |
| **Observation** $o_t$ | Flattened state | $\mathbb{R}^{391}$ | Mixed norms |

---

## 11. References

- **Gymnasium API:** https://gymnasium.farama.org/
- **Sutton & Barto (2018):** *Reinforcement Learning: An Introduction*. Chapters 3–4 on MDPs.
- **MT5 Documentation:** https://www.mql5.com/en/docs/

---

## Appendix: Notation Glossary

| Symbol | Meaning |
|--------|---------|
| $\mathcal{S}$ | State space |
| $\mathcal{A}$ | Action space |
| $s_t$ | State at time $t$ |
| $a_t$ | Action at time $t$ |
| $r_t$ | Reward at time $t$ |
| $o_t$ | Observation (flattened state) at time $t$ |
| $P(s' \| s, a)$ | Transition probability |
| $\gamma$ | Discount factor |
| $T$ | Episode length (time horizon) |
| $h_t$ | History component of state |
| $c_t$ | Context component of state |
| ATR | Average True Range (volatility) |
| $R$ | Risk multiple (SL distance in units) |
| PnL | Profit and Loss |