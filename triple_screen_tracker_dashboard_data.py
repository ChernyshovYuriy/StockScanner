"""
triple_screen_tracker_dashboard_data.py
=========================================
Read-only assembly of triple_screen_tracker.db for the web dashboard's
/triple-screen route.

Same isolation pattern as demand_dashboard_data.py: opens its OWN read-only
sqlite3 connection straight at TRIPLE_SCREEN_TRACKER_DB_PATH -- never imports
triple_screen_tracker/store.py's connect() (a write-shaped self-migrate the
dashboard's long-running process shouldn't do on every page load). Purely a
display layer over whatever triple_screen_tracker_service.py has already
populated; never writes, never fetches a new price itself.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import date
from typing import Dict, List

from config import DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS, TRIPLE_SCREEN_TRACKER_DB_PATH

_cache_lock = threading.Lock()
_cache: Dict[str, object] = {"ts": 0.0, "rows": None}


def _pct(buy_price, other_price):
    if not buy_price:
        return None
    return (other_price / buy_price - 1.0) * 100.0


def _days_held(buy_date_str, other_date_str):
    try:
        return (date.fromisoformat(other_date_str) - date.fromisoformat(buy_date_str)).days
    except (TypeError, ValueError):
        return None


def _read_tracked() -> list[dict]:
    """All tracked_signals rows + their price history. [] if the DB doesn't
    exist yet (no scheduled run has happened) -- same "not yet available"
    convention as demand_dashboard_data.py's DB-missing guard."""
    if not TRIPLE_SCREEN_TRACKER_DB_PATH.exists():
        return []
    conn = sqlite3.connect(f"file:{TRIPLE_SCREEN_TRACKER_DB_PATH}?mode=ro", uri=True)
    try:
        tracked = conn.execute(
            "SELECT id, ticker, buy_date, buy_price, status, sell_date, sell_price "
            "FROM tracked_signals ORDER BY buy_date DESC, id DESC"
        ).fetchall()
        history_rows = conn.execute(
            "SELECT tracked_id, date, close_price FROM price_history ORDER BY tracked_id, date"
        ).fetchall()
    finally:
        conn.close()

    history_by_id: Dict[int, List[Dict]] = {}
    for tracked_id, hdate, close_price in history_rows:
        history_by_id.setdefault(tracked_id, []).append({"date": hdate, "close_price": close_price})

    out = []
    for tid, ticker, buy_date, buy_price, status, sell_date, sell_price in tracked:
        out.append({
            "id": tid, "ticker": ticker, "buy_date": buy_date, "buy_price": buy_price,
            "status": status, "sell_date": sell_date, "sell_price": sell_price,
            "price_history": history_by_id.get(tid, []),
        })
    return out


def _build_triple_screen_tracker_state() -> Dict[str, List[Dict]]:
    open_rows, closed_rows = [], []
    for r in _read_tracked():
        history = r["price_history"]
        latest_price = history[-1]["close_price"] if history else r["buy_price"]
        latest_date = history[-1]["date"] if history else r["buy_date"]
        if r["status"] == "OPEN":
            open_rows.append({
                "ticker": r["ticker"], "buy_date": r["buy_date"], "buy_price": r["buy_price"],
                "latest_price": latest_price, "pct_change": _pct(r["buy_price"], latest_price),
                "days_held": _days_held(r["buy_date"], latest_date),
                "price_history": history,
            })
        else:
            closed_rows.append({
                "ticker": r["ticker"], "buy_date": r["buy_date"], "buy_price": r["buy_price"],
                "sell_date": r["sell_date"], "sell_price": r["sell_price"],
                "pct_change": _pct(r["buy_price"], r["sell_price"]),
                "days_held": _days_held(r["buy_date"], r["sell_date"]),
                "price_history": history,
            })
    return {"open": open_rows, "closed": closed_rows}


def build_triple_screen_tracker_state() -> Dict[str, List[Dict]]:
    """TTL-cached wrapper -- same rationale as
    demand_dashboard_data.build_demand_signals_by_ticker()."""
    with _cache_lock:
        now = time.monotonic()
        if _cache["rows"] is not None and now - _cache["ts"] < DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS:
            return _cache["rows"]

        rows = _build_triple_screen_tracker_state()
        _cache["rows"] = rows
        _cache["ts"] = now
        return rows
