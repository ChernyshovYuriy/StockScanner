"""
demand_dashboard_data.py
==========================
Read-only assembly of demand_signals.db for the web dashboard's /demand
route.

Mirrors momentum_dashboard_data.py's isolation exactly: opens its OWN
read-only sqlite3 connection straight at DEMAND_DB_PATH -- never imports
demand_signals/store.py's connect() (which self-migrates/creates the
schema, a write-shaped operation dashboard_app.py's long-running process
shouldn't be doing on every page load) and never imports edgar/ or db.py.
Purely a display layer over whatever demand_signals_service.py has already
populated; never writes, never fetches a new signal itself.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict
from typing import Dict, List

from config import DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS, DEMAND_DB_PATH, EDGAR_CACHE_PATH

_cache_lock = threading.Lock()
_cache: Dict[str, object] = {"ts": 0.0, "rows": None}
_names_cache: Dict[str, object] = {"file": None, "names": {}}


def _read_demand_signals() -> list[dict]:
    """All stored signals, newest first. [] if the DB doesn't exist yet (no
    scheduled run has happened) -- same "not yet available" convention as
    momentum_dashboard_data.py's DB-missing guard, not an error."""
    if not DEMAND_DB_PATH.exists():
        return []
    conn = sqlite3.connect(f"file:{DEMAND_DB_PATH}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT ticker, source, signal_type, direction, strength, "
            "lag_days, date, detail FROM demand_signals "
            "ORDER BY ticker, date DESC"
        ).fetchall()
    finally:
        conn.close()

    out = []
    for ticker, source, signal_type, direction, strength, lag_days, date, detail in rows:
        out.append({
            "ticker": ticker, "source": source, "signal_type": signal_type,
            "direction": direction, "strength": strength, "lag_days": lag_days,
            "date": date, "detail": json.loads(detail) if detail else {},
        })
    return out


def _build_demand_signals_by_ticker() -> Dict[str, List[Dict]]:
    by_ticker = defaultdict(list)
    for row in _read_demand_signals():
        by_ticker[row["ticker"]].append(row)
    return dict(by_ticker)


def build_demand_signals_by_ticker() -> Dict[str, List[Dict]]:
    """TTL-cached wrapper -- same rationale as
    momentum_dashboard_data.build_momentum_positions()."""
    with _cache_lock:
        now = time.monotonic()
        if _cache["rows"] is not None and now - _cache["ts"] < DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS:
            return _cache["rows"]

        rows = _build_demand_signals_by_ticker()
        _cache["rows"] = rows
        _cache["ts"] = now
        return rows


def read_company_names() -> Dict[str, str]:
    """{TICKER: company name} from the newest SEC company_tickers_*.json that
    edgar/core.py's load_ticker_map() already caches daily -- the same file
    demand_signals_service.py resolves its watchlist tickers from, so every
    ticker on the page has an entry. Read straight off disk (never fetched
    from here, never imports edgar/); {} if no cached copy exists yet."""
    files = sorted(EDGAR_CACHE_PATH.glob("company_tickers_*.json"))
    if not files:
        return {}
    latest = files[-1]
    with _cache_lock:
        if _names_cache["file"] == latest:
            return _names_cache["names"]
    try:
        data = json.loads(latest.read_text())
        names = {row["ticker"].upper(): row["title"] for row in data.values()}
    except (OSError, ValueError, KeyError, AttributeError):
        return {}
    with _cache_lock:
        _names_cache["file"] = latest
        _names_cache["names"] = names
    return names
