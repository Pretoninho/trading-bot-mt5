import math

PIP_SIZE = 0.0001
PIP_VALUE_PER_LOT = 10.0  # USD per pip per 1.0 lot (EURUSD, USD account) - V1 approximation

MIN_LOT = 0.01
LOT_STEP = 0.01
MAX_LOT = 100.0

def round_down_to_step(x: float, step: float) -> float:
    return math.floor(x / step) * step

def compute_lots_for_1pct_risk(equity: float, entry_price: float, sl_price: float) -> float:
    risk_amount = 0.01 * equity
    r_price = abs(entry_price - sl_price)
    stop_pips = r_price / PIP_SIZE

    # Guardrails
    if stop_pips <= 0:
        return 0.0

    loss_per_lot = stop_pips * PIP_VALUE_PER_LOT
    if loss_per_lot <= 0:
        return 0.0

    lots_raw = risk_amount / loss_per_lot

    # Clamp then round down to step
    lots_clamped = max(MIN_LOT, min(MAX_LOT, lots_raw))
    lots = round_down_to_step(lots_clamped, LOT_STEP)

    # If rounding pushed below min lot, reject
    if lots < MIN_LOT:
        return 0.0

    return lots