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
from position_monitor import (
    LOOKBACK_DAYS_BEFORE_ENTRY,
    MIN_BARS_REQUIRED,
    compute_signals,
    fetch_intraday_snapshot,
    load_or_fetch_data,
    parse_positions_from_db,
)
from schema_keys import POSITION_COL_REASON, POSITION_COL_STATUS, SIGNAL_COL_TICKER
from time_utils import is_market_open

_cache_lock = threading.Lock()
_cache: Dict[str, object] = {"ts": 0.0, "rows": None}


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
            rows.append({SIGNAL_COL_TICKER: pos.ticker, POSITION_COL_STATUS: "NO_DATA",
                         POSITION_COL_REASON: f"Insufficient bars ({len(df)})"})
            continue

        needed = {"High", "Low", "Close"}
        if not needed.issubset(df.columns):
            missing = sorted(needed - set(df.columns))
            rows.append({SIGNAL_COL_TICKER: pos.ticker, POSITION_COL_STATUS: "NO_DATA",
                         POSITION_COL_REASON: f"Missing columns: {missing}"})
            continue

        today_bar = fetch_intraday_snapshot(pos.ticker) if use_intraday else None
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
