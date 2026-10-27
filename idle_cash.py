"""
idle_cash.py
============
Idle-cash parking for the core sleeve (data/trading.db): uninvested cash is
held in config.IDLE_CASH_PARK_TICKER (XIU.TO) instead of earning nothing.

- virtual_buy.py sizes from cash + the parked holding's market value, and
  before writing the morning's buys calls raise_cash() to sell just enough of
  the holding to pay for them (one sale, plus the cash buffer).
- position_monitor.py's pre-close run calls park_excess() after its sells:
  cash above the buffer is parked once it exceeds IDLE_CASH_MIN_TRADE_PCT of
  equity, so small amounts don't trigger a trade every day.

The holding lives in db.py's `parked` table, never in `positions`, so no exit
rule ever applies to it; its trades are BUY/SELL rows in `transactions` with
reason PARK/UNPARK. Backtested as idle_cash_mode="xiu_always" in
backtest_runner.py (reports/idle_cash/03_matrix/). Prices are the caller's
live quote -- this module never fetches.
"""

from __future__ import annotations

import math
from typing import Optional

from config import IDLE_CASH_BUFFER_PCT, IDLE_CASH_MIN_TRADE_PCT, IDLE_CASH_PARK_TICKER
from db import get_cash, get_parked, park_trade


def parked_shares() -> float:
    """Shares of the park ticker currently held (0 when parking is off or nothing is parked)."""
    if IDLE_CASH_PARK_TICKER is None:
        return 0.0
    return get_parked().get(IDLE_CASH_PARK_TICKER.upper(), {}).get("shares", 0.0)


def funds_estimate() -> float:
    """Cash + every parked holding at its last traded price -- no network call.
    For display and the pipeline's indicative sizing; virtual_buy.py values
    the holding at a live quote instead."""
    return get_cash() + sum(h["shares"] * h["last_price"] for h in get_parked().values())


def raise_cash(needed: float, price: Optional[float], trade_date: str, equity: float,
               dry_run: bool = False) -> int:
    """Sell enough parked shares at price that cash covers `needed` plus the
    buffer. Returns the number of shares sold (0 if cash already covers it,
    nothing is parked, or there's no price)."""
    cash = get_cash()
    held = parked_shares()
    if cash >= needed or not held or not price or price <= 0:
        return 0
    buffer = IDLE_CASH_BUFFER_PCT / 100 * equity
    n = int(min(held, math.ceil((needed - cash + buffer) / price)))
    if n > 0 and not dry_run:
        park_trade(IDLE_CASH_PARK_TICKER, trade_date, price, -n)
    return n


def park_excess(price: Optional[float], trade_date: str, equity: float, dry_run: bool = False) -> int:
    """Buy the park ticker at price with the cash above the buffer, if that
    excess exceeds IDLE_CASH_MIN_TRADE_PCT of equity. Returns shares bought."""
    if IDLE_CASH_PARK_TICKER is None or not price or price <= 0 or equity <= 0:
        return 0
    excess = get_cash() - IDLE_CASH_BUFFER_PCT / 100 * equity
    if excess <= IDLE_CASH_MIN_TRADE_PCT / 100 * equity:
        return 0
    n = int(excess / price)
    if n > 0 and not dry_run:
        park_trade(IDLE_CASH_PARK_TICKER, trade_date, price, n)
    return n
