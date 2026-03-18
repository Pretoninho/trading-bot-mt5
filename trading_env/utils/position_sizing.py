"""
position_sizing.py
==================
Compute EURUSD lot size for a 1 % equity risk trade.

EURUSD specifics
----------------
* 5-digit pricing: ``point = 0.00001``, ``pip_size = 0.0001``
* Pip value per 1.0 standard lot (USD account): **$10 / pip**
* Minimum lot: 0.01 — Lot step: 0.01 — Maximum lot: 100.0
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Instrument constants
# ---------------------------------------------------------------------------

POINT = 0.00001
PIP_SIZE = 0.0001
PIP_VALUE_PER_LOT = 10.0   # USD per pip per 1.0 lot

MIN_LOT = 0.01
LOT_STEP = 0.01
MAX_LOT = 100.0

RISK_FRACTION = 0.01  # risk 1 % of equity per trade

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def round_down_to_step(value: float, step: float) -> float:
    """Round *value* down to the nearest multiple of *step*.

    Uses a precision guard (round to 10 decimal places before floor) to avoid
    floating-point rounding artefacts such as ``0.5 / 0.01 = 49.9999...``.
    """
    return round(math.floor(round(value / step, 10)) * step, 10)


def compute_lots(
    equity: float,
    entry_price: float,
    sl_price: float,
    risk_fraction: float = RISK_FRACTION,
) -> float:
    """Compute the lot size that risks *risk_fraction* of *equity*.

    Parameters
    ----------
    equity:
        Current account equity in USD.
    entry_price:
        Trade entry price.
    sl_price:
        Initial stop-loss price.
    risk_fraction:
        Fraction of equity to risk (default 0.01 = 1 %).

    Returns
    -------
    Lot size rounded down to the nearest ``LOT_STEP``.
    Returns ``0.0`` when the trade should be rejected (stop too tight or
    sizing falls below ``MIN_LOT`` after rounding).
    """
    risk_amount = risk_fraction * equity
    r_price = abs(entry_price - sl_price)
    stop_pips = r_price / PIP_SIZE

    if stop_pips <= 0:
        return 0.0

    loss_per_lot = stop_pips * PIP_VALUE_PER_LOT
    if loss_per_lot <= 0:
        return 0.0

    lots_raw = risk_amount / loss_per_lot

    # Clamp to [MIN_LOT, MAX_LOT] before rounding
    lots_clamped = max(MIN_LOT, min(MAX_LOT, lots_raw))
    lots = round_down_to_step(lots_clamped, LOT_STEP)

    if lots < MIN_LOT:
        return 0.0

    return lots
