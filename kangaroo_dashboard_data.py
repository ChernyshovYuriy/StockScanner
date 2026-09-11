"""
kangaroo_dashboard_data.py
============================
Read-only assembly of the Kangaroo Tail sleeve's "live positions" view for
the web dashboard's /kangaroo route.

Same isolation pattern as momentum_dashboard_data.py/macro_dashboard_data.py:
opens its own read-only DuckDB connection straight at KANGAROO_DB_PATH — never
imports db.py or calls init_db(), so this long-running dashboard process
never races kangaroo_pipeline.py/kangaroo_buy.py/kangaroo_monitor.py's own
writer connections. No manual-sell action is exposed here either, for the
same reason.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List

import duckdb
import pandas as pd

from config import DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS, KANGAROO_DB_PATH, KANGAROO_MAX_HOLD_DAYS
from kangaroo_monitor import KangarooPosition, compute_kangaroo_signal
from position_monitor import LOOKBACK_DAYS_BEFORE_ENTRY, MIN_BARS_REQUIRED, fetch_intraday_snapshot, load_or_fetch_data
from schema_keys import (
    POSITION_COL_ENTRY_DATE,
    POSITION_COL_ENTRY_PRICE,
    POSITION_COL_REASON,
    POSITION_COL_SHARES,
    POSITION_COL_STATUS,
    SIGNAL_COL_TICKER,
)
from time_utils import is_market_open

_cache_lock = threading.Lock()
_cache: Dict[str, object] = {"ts": 0.0, "rows": None}


def _read_kangaroo_db(sql: str) -> pd.DataFrame:
    if not KANGAROO_DB_PATH.exists():
        return pd.DataFrame()
    conn = duckdb.connect(str(KANGAROO_DB_PATH), read_only=True)
    try:
        return conn.execute(sql).df()
    finally:
        conn.close()


def get_kangaroo_cash() -> float:
    df = _read_kangaroo_db("SELECT cash FROM account WHERE id = 1")
    return float(df["cash"].iloc[0]) if not df.empty else 0.0


def get_kangaroo_transactions() -> pd.DataFrame:
    return _read_kangaroo_db(
        "SELECT side, ticker, trade_date, price, shares, amount, pnl_dollars, pnl_pct, reason, recorded_at"
        " FROM transactions ORDER BY trade_date, recorded_at"
    )


def get_kangaroo_pending_intents() -> pd.DataFrame:
    """Pending breakout setups not yet triggered — surfaced on the
    dashboard so a "watchlist" of untriggered tails is visible, not just
    filled positions."""
    return _read_kangaroo_db(
        "SELECT ticker, signal_date, entry_price_planned, stop_price, target_price, rr"
        " FROM intents WHERE intent_status = 'PENDING' ORDER BY signal_date DESC"
    )


def _parse_kangaroo_positions() -> list[KangarooPosition]:
    df = _read_kangaroo_db(
        "SELECT ticker, entry_date, entry_price, shares, stop_price, target_price FROM positions ORDER BY entry_date"
    )
    if df.empty:
        return []
    positions: list[KangarooPosition] = []
    for _, row in df.iterrows():
        ticker = str(row[SIGNAL_COL_TICKER]).strip()
        if not ticker or ticker.lower() == "nan":
            continue
        stop_price = row.get("stop_price")
        target_price = row.get("target_price")
        positions.append(KangarooPosition(
            ticker=ticker,
            entry_date=pd.to_datetime(row[POSITION_COL_ENTRY_DATE]).date(),
            entry_price=float(row[POSITION_COL_ENTRY_PRICE]),
            shares=float(row[POSITION_COL_SHARES]),
            stop_price=float(stop_price) if pd.notna(stop_price) else None,
            target_price=float(target_price) if pd.notna(target_price) else None,
        ))
    return positions


def _no_data_row(pos: KangarooPosition, reason: str) -> Dict:
    return {SIGNAL_COL_TICKER: pos.ticker, POSITION_COL_STATUS: "NO_DATA", POSITION_COL_REASON: reason}


def _build_kangaroo_positions() -> List[Dict]:
    positions = _parse_kangaroo_positions()
    if not positions:
        return []

    use_intraday = is_market_open()
    rows: List[Dict] = []

    for pos in positions:
        start = (pd.Timestamp(pos.entry_date) - pd.Timedelta(days=LOOKBACK_DAYS_BEFORE_ENTRY)).date()
        df = load_or_fetch_data(pos.ticker, start=start)

        if df.empty or len(df) < MIN_BARS_REQUIRED:
            rows.append(_no_data_row(pos, f"Insufficient bars ({len(df)})"))
            continue

        needed = {"High", "Low", "Close"}
        if not needed.issubset(df.columns):
            missing = sorted(needed - set(df.columns))
            rows.append(_no_data_row(pos, f"Missing columns: {missing}"))
            continue

        today_bar = fetch_intraday_snapshot(pos.ticker) if use_intraday else None
        rows.append(compute_kangaroo_signal(pos, df, today_bar=today_bar, max_hold_days=KANGAROO_MAX_HOLD_DAYS))

    return rows


def build_kangaroo_positions() -> List[Dict]:
    """TTL-cached wrapper — same rationale as dashboard_positions.build_live_positions()."""
    with _cache_lock:
        now = time.monotonic()
        if _cache["rows"] is not None and now - _cache["ts"] < DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS:
            return _cache["rows"]

        rows = _build_kangaroo_positions()
        _cache["rows"] = rows
        _cache["ts"] = now
        return rows
