"""
momentum_dashboard_data.py
============================
Read-only assembly of the momentum sleeve's "live positions" view for the
web dashboard's /momentum route.

Mirrors dashboard_positions.py's pattern exactly, with one deliberate
difference: it never calls db.py's init_db()/get_*() functions. db.py keeps
a single module-global DB_PATH — fine for the three core scheduled services
(each its own process), but dashboard_app.py is one long-running process
that must serve the core sleeve's pages AND this one concurrently, and
Waitress can run multiple requests at once. Repointing db.py's global state
per-request would race. Instead this module opens its own read-only DuckDB
connection straight at MOMENTUM_DB_PATH — db.py is never imported here.

Never calls execute_virtual_sells(): informational only, same rule as
dashboard_positions.py. No manual-sell action is exposed for this sleeve
yet (see build plan) — exactly for the same global-state reason above.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List

import duckdb
import pandas as pd

from config import DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS, MOMENTUM_DB_PATH
from momentum_monitor import MOMENTUM_EXIT_PARAMS
from position_monitor import (
    LOOKBACK_DAYS_BEFORE_ENTRY,
    MIN_BARS_REQUIRED,
    Position,
    compute_signals,
    fetch_intraday_snapshot,
    load_or_fetch_data,
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


def _read_momentum_db(sql: str) -> pd.DataFrame:
    """Open, query, close — read_only so this never contends with the
    momentum services' own writer connections."""
    if not MOMENTUM_DB_PATH.exists():
        return pd.DataFrame()
    conn = duckdb.connect(str(MOMENTUM_DB_PATH), read_only=True)
    try:
        return conn.execute(sql).df()
    finally:
        conn.close()


def get_momentum_cash() -> float:
    df = _read_momentum_db("SELECT cash FROM account WHERE id = 1")
    return float(df["cash"].iloc[0]) if not df.empty else 0.0


def get_momentum_transactions() -> pd.DataFrame:
    return _read_momentum_db(
        "SELECT side, ticker, trade_date, price, shares, amount, pnl_dollars, pnl_pct, reason, recorded_at"
        " FROM transactions ORDER BY trade_date, recorded_at"
    )


def _parse_momentum_positions() -> list[Position]:
    df = _read_momentum_db(
        "SELECT ticker, entry_date, entry_price, shares, stop_price FROM positions ORDER BY entry_date"
    )
    if df.empty:
        return []
    positions: list[Position] = []
    for _, row in df.iterrows():
        ticker = str(row[SIGNAL_COL_TICKER]).strip()
        if not ticker or ticker.lower() == "nan":
            continue
        raw_stop = row["stop_price"] if "stop_price" in row else None
        stop_price = float(raw_stop) if pd.notna(raw_stop) else None
        positions.append(Position(
            ticker=ticker,
            entry_date=pd.to_datetime(row[POSITION_COL_ENTRY_DATE]).date(),
            entry_price=float(row[POSITION_COL_ENTRY_PRICE]),
            shares=float(row[POSITION_COL_SHARES]),
            stop_price=stop_price,
        ))
    return positions


def _best_effort_price(ticker: str, df: pd.DataFrame, use_intraday: bool) -> tuple[float, str] | None:
    """Same rationale as dashboard_positions.py's helper of the same name —
    a broken/thin feed still contributes to the dashboard's totals instead
    of silently valuing the position at $0."""
    if use_intraday:
        snap = fetch_intraday_snapshot(ticker)
        if snap is not None:
            return snap.close, "5m-intraday"
    if not df.empty and "Close" in df.columns:
        return float(df["Close"].iloc[-1]), "stale-cache"
    return None


def _no_data_row(pos: Position, reason: str, use_intraday: bool, df: pd.DataFrame) -> Dict:
    row = {SIGNAL_COL_TICKER: pos.ticker, POSITION_COL_STATUS: "NO_DATA", POSITION_COL_REASON: reason}
    # A zero/negative entry_price would divide-by-zero below and crash the
    # whole dashboard page for every position, not just this one — same
    # bug as dashboard_positions.py's helper of the same name.
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


def _build_momentum_positions() -> List[Dict]:
    positions = _parse_momentum_positions()
    if not positions:
        return []

    use_intraday = is_market_open()
    rows: List[Dict] = []

    for pos in positions:
        start = (pd.Timestamp(pos.entry_date) - pd.Timedelta(days=LOOKBACK_DAYS_BEFORE_ENTRY)).date()
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
        result = compute_signals(
            pos, df, today_bar=today_bar, exit_params=MOMENTUM_EXIT_PARAMS, planned_stop=pos.stop_price,
        )
        rows.append(result)

    return rows


def build_momentum_positions() -> List[Dict]:
    """TTL-cached wrapper — same rationale as dashboard_positions.build_live_positions()."""
    with _cache_lock:
        now = time.monotonic()
        if _cache["rows"] is not None and now - _cache["ts"] < DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS:
            return _cache["rows"]

        rows = _build_momentum_positions()
        _cache["rows"] = rows
        _cache["ts"] = now
        return rows
