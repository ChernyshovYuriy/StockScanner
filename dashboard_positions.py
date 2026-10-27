"""
dashboard_positions.py
=======================
Read-only assembly of the "live positions" view for the web dashboard.

Mirrors position_monitor.py's per-position pipeline (load_or_fetch_data ->
fetch_intraday_snapshot if market open -> compute_signals) without importing
or modifying position_monitor.py's main() — this repo already has a
precedent for this (backtest_runner.py duplicates the same loop rather than
sharing one orchestrator), so the dashboard follows the same pattern.

Never calls execute_virtual_sells(): a "SELL" status here is informational
only. Positions are only closed by an explicit call to
manual_sell.sell_position().
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List

import pandas as pd

from config import DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS
from db import get_parked
from position_monitor import (
    LOOKBACK_DAYS_BEFORE_ENTRY,
    MIN_BARS_REQUIRED,
    check_stale_data,
    compute_signals,
    fetch_intraday_snapshot,
    load_or_fetch_data,
    parse_positions_from_db,
    stale_no_data_row,
)
from schema_keys import (
    POSITION_COL_ENTRY_DATE,
    POSITION_COL_ENTRY_PRICE,
    POSITION_COL_LAST_CLOSE,
    POSITION_COL_PNL_DOLLARS,
    POSITION_COL_PNL_PCT,
    POSITION_COL_REASON,
    POSITION_COL_SHARES,
    POSITION_COL_STATUS,
    SIGNAL_COL_TICKER,
)
from time_utils import is_market_open

_cache_lock = threading.Lock()
_cache: Dict[str, object] = {"ts": 0.0, "rows": None}
_parked_cache: Dict[str, object] = {"ts": 0.0, "rows": None}


def _best_effort_price(ticker: str, df: pd.DataFrame, use_intraday: bool) -> tuple[float, str] | None:
    """Last-resort valuation price for a ticker without enough history for
    a full compute_signals() analysis (stop/ATR/trend all need
    MIN_BARS_REQUIRED bars) — so a broken or thin live feed still
    contributes the position's real value to Positions Value / Unrealized
    P&L instead of it being silently valued at $0. Not used for exit
    decisions, only for the dashboard's aggregate totals."""
    if use_intraday:
        snap = fetch_intraday_snapshot(ticker)
        if snap is not None:
            return snap.close, "5m-intraday"
    if not df.empty and "Close" in df.columns:
        return float(df["Close"].iloc[-1]), "stale-cache"
    return None


def _no_data_row(pos, reason: str, use_intraday: bool, df: pd.DataFrame) -> Dict:
    row = {SIGNAL_COL_TICKER: pos.ticker, POSITION_COL_STATUS: "NO_DATA", POSITION_COL_REASON: reason}
    # A zero/negative entry_price would divide-by-zero below and crash the
    # whole dashboard page for every position, not just this one — same
    # class of bug as position_monitor.compute_signals(), independently
    # reachable here since this fallback path doesn't go through it.
    if pos.entry_price <= 0:
        return row
    fallback = _best_effort_price(pos.ticker, df, use_intraday)
    if fallback is not None:
        price, source = fallback
        row[POSITION_COL_REASON] = f"{reason}; valued at last known price ({source})"
        row.update({
            POSITION_COL_ENTRY_DATE: pos.entry_date.isoformat(),
            POSITION_COL_ENTRY_PRICE: pos.entry_price,
            POSITION_COL_SHARES: pos.shares,
            POSITION_COL_LAST_CLOSE: round(price, 4),
            POSITION_COL_PNL_PCT: round((price / pos.entry_price - 1.0) * 100.0, 2),
            POSITION_COL_PNL_DOLLARS: round((price - pos.entry_price) * pos.shares, 2),
        })
    return row


def _build_live_positions() -> List[Dict]:
    """One uncached pass over open positions — same steps as
    position_monitor.py's main() loop, read-only."""
    positions = parse_positions_from_db()
    if not positions:
        return []

    use_intraday = is_market_open()
    rows: List[Dict] = []

    for pos in positions:
        start = (
            pd.Timestamp(pos.entry_date) - pd.Timedelta(days=LOOKBACK_DAYS_BEFORE_ENTRY)
        ).date()
        df = load_or_fetch_data(pos.ticker, start=start)

        if df.empty or len(df) < MIN_BARS_REQUIRED:
            rows.append(_no_data_row(pos, f"Insufficient bars ({len(df)})", use_intraday, df))
            continue

        needed = {"High", "Low", "Close"}
        if not needed.issubset(df.columns):
            missing = sorted(needed - set(df.columns))
            rows.append(_no_data_row(pos, f"Missing columns: {missing}", use_intraday, df))
            continue

        today_bar = fetch_intraday_snapshot(pos.ticker) if use_intraday else None
        stale_rec = check_stale_data(pos, df, today_bar)
        if stale_rec is not None:
            rows.append(_no_data_row(pos, stale_no_data_row(stale_rec)[POSITION_COL_REASON], use_intraday, df))
            continue
        result = compute_signals(pos, df, today_bar=today_bar, planned_stop=pos.stop_price)
        rows.append(result)

    return rows


def build_live_positions() -> List[Dict]:
    """TTL-cached wrapper around _build_live_positions() so repeated page
    loads (auto-refresh, multiple open tabs) within the cache window don't
    each re-fetch every open position from Yahoo Finance."""
    with _cache_lock:
        now = time.monotonic()
        if _cache["rows"] is not None and now - _cache["ts"] < DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS:
            return _cache["rows"]

        rows = _build_live_positions()
        _cache["rows"] = rows
        _cache["ts"] = now
        return rows


def build_parked_holdings() -> List[Dict]:
    """Idle cash parked in an ETF (idle_cash.py), one dict per holding:
    ticker, shares, price, value, price_source. Valued at a live 5-min
    snapshot while the market is open, else at its last traded price.
    TTL-cached like build_live_positions(). Read-only."""
    with _cache_lock:
        now = time.monotonic()
        if _parked_cache["rows"] is not None and now - _parked_cache["ts"] < DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS:
            return _parked_cache["rows"]
        rows = []
        use_intraday = is_market_open()
        for ticker, h in get_parked().items():
            snap = fetch_intraday_snapshot(ticker) if use_intraday else None
            price, source = (snap.close, "5m-intraday") if snap is not None else (h["last_price"], "last-trade")
            rows.append({"ticker": ticker, "shares": h["shares"], "price": price,
                         "value": h["shares"] * price, "price_source": source})
        _parked_cache["rows"] = rows
        _parked_cache["ts"] = now
        return rows
