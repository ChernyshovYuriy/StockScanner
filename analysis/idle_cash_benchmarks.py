"""
analysis/idle_cash_benchmarks.py -- the XIU-only benchmarks of the idle-cash
plan's Phase 3 matrix (reports/idle_cash/03_matrix/DECISION_RULES.md):

    D  buy-and-hold XIU (all cash at the fold's first open)
    E  XIU while its prior close is >= its 200-day SMA, else cash (switch at the open)
    F  55% XIU / 45% cash, rebalanced to 55% at each month's first open (descriptive only)

Run on the strategy's own fold days, $50k per fold, whole XIU shares; cash
earns CASH.TO's daily total return on its closing balance. XIU fills move
against the trade by spread_bps; the commission is per order. The gate is
backtest_runner._is_market_in_uptrend on the prior completed close, the same
test the strategy's park gate and entry gate use.
"""
from __future__ import annotations

import pandas as pd

from backtest_runner import _day_close_price, _is_market_in_uptrend, _next_open_price
from market_data import HistoricalSliceProvider

BENCHMARK = "XIU.TO"
F_XIU_WEIGHT = 0.55


def run_fold(kind: str, days: list, provider: HistoricalSliceProvider, cash_returns: dict,
             capital: float, spread_bps: float, commission: float) -> pd.DataFrame:
    """One fold of benchmark `kind` ("D"/"E"/"F") over `days`; one row per day."""
    s = spread_bps / 1e4
    cash, shares, costs, traded = capital, 0, 0.0, 0.0
    rows = []

    def trade(n: int, px: float) -> None:
        nonlocal cash, shares, costs, traded
        if n == 0:
            return
        fill = px * (1 + s) if n > 0 else px * (1 - s)
        cash -= n * fill + commission
        shares += n
        costs += abs(n) * abs(fill - px) + commission
        traded += abs(n) * px

    def buy_all(px: float) -> int:
        return max(int((cash - commission) / (px * (1 + s))), 0)

    prev_month = None
    for i, day in enumerate(days):
        ts = pd.Timestamp(day)
        prior = ts - pd.Timedelta(days=1)
        px = _next_open_price(BENCHMARK, provider, prior)
        if px is not None and px > 0:
            if kind == "D" and i == 0:
                trade(buy_all(px), px)
            elif kind == "E":
                if _is_market_in_uptrend(BENCHMARK, provider, prior):
                    if shares == 0:
                        trade(buy_all(px), px)
                elif shares:
                    trade(-shares, px)
            elif kind == "F" and (i == 0 or day.month != prev_month):
                target = int(F_XIU_WEIGHT * (cash + shares * px) / px)
                trade(target - shares, px)
        prev_month = day.month
        close = _day_close_price(BENCHMARK, provider, ts)
        r = cash_returns.get(day, 0.0)
        y = cash * r if cash > 0 else 0.0
        cash += y
        xiu_value = shares * close
        rows.append(dict(date=day, cash=cash, open_value=0.0, xiu_value=xiu_value, total_equity=cash + xiu_value,
                         cash_yield=y, cum_costs=costs, cum_xiu_traded=traded))
    return pd.DataFrame(rows)

