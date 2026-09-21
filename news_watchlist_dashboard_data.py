"""
news_watchlist_dashboard_data.py
==================================
Read-only assembly of news_watchlist.db for the web dashboard's
/news-watchlist route.

Same isolation pattern as triple_screen_tracker_dashboard_data.py: opens its
OWN read-only sqlite3 connection straight at NEWS_WATCHLIST_DB_PATH -- never
imports news_watchlist/store.py's connect() (a write-shaped self-migrate the
dashboard's long-running process shouldn't do on every page load). Purely a
display layer over whatever news_watchlist_service.py has already seeded/
updated, plus the confirm/dismiss/note/manual-add writes dashboard_app.py's
own routes make through news_watchlist/store.py directly.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import date
from typing import Dict, List

from config import DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS, NEWS_WATCHLIST_DB_PATH

_cache_lock = threading.Lock()
_cache: Dict[str, object] = {"ts": 0.0, "rows": None}

_ITEM_COLUMNS = (
    "id", "guid", "ticker", "company", "category", "materiality", "summary",
    "source_link", "status", "note", "flagged_at", "flag_price",
    "status_changed_at", "created_at",
)


def _pct(flag_price, other_price):
    if not flag_price:
        return None
    return (other_price / flag_price - 1.0) * 100.0


def _days_since(flagged_at_str):
    try:
        return (date.today() - date.fromisoformat(flagged_at_str)).days
    except (TypeError, ValueError):
        return None


def _read_items() -> list[dict]:
    """Every watchlist_items row + its price history. [] if the DB doesn't
    exist yet (no scheduled run has happened) -- same "not yet available"
    convention as triple_screen_tracker_dashboard_data.py's DB-missing
    guard."""
    if not NEWS_WATCHLIST_DB_PATH.exists():
        return []
    conn = sqlite3.connect(f"file:{NEWS_WATCHLIST_DB_PATH}?mode=ro", uri=True)
    try:
        items = conn.execute(
            f"SELECT {','.join(_ITEM_COLUMNS)} FROM watchlist_items "
            "ORDER BY flagged_at DESC, id DESC"
        ).fetchall()
        history_rows = conn.execute(
            "SELECT item_id, date, close_price FROM price_history ORDER BY item_id, date"
        ).fetchall()
    finally:
        conn.close()

    history_by_id: Dict[int, List[Dict]] = {}
    for item_id, hdate, close_price in history_rows:
        history_by_id.setdefault(item_id, []).append({"date": hdate, "close_price": close_price})

    out = []
    for row in items:
        item = dict(zip(_ITEM_COLUMNS, row))
        item["price_history"] = history_by_id.get(item["id"], [])
        out.append(item)
    return out


def _build_news_watchlist_state() -> Dict[str, List[Dict]]:
    inbox, watching, dismissed = [], [], []
    for item in _read_items():
        history = item["price_history"]
        latest_price = history[-1]["close_price"] if history else item["flag_price"]
        item["latest_price"] = latest_price
        item["pct_change"] = _pct(item["flag_price"], latest_price)
        item["days_since_flagged"] = _days_since(item["flagged_at"])
        if item["status"] == "inbox":
            inbox.append(item)
        elif item["status"] == "watching":
            watching.append(item)
        else:
            dismissed.append(item)
    return {"inbox": inbox, "watching": watching, "dismissed": dismissed}


def build_news_watchlist_state() -> Dict[str, List[Dict]]:
    """TTL-cached wrapper -- same rationale as
    triple_screen_tracker_dashboard_data.build_triple_screen_tracker_state()."""
    with _cache_lock:
        now = time.monotonic()
        if _cache["rows"] is not None and now - _cache["ts"] < DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS:
            return _cache["rows"]

        rows = _build_news_watchlist_state()
        _cache["rows"] = rows
        _cache["ts"] = now
        return rows


def invalidate_news_watchlist_cache() -> None:
    """Called by dashboard_app.py's own confirm/dismiss/note/add routes
    after a write, so the page reflects the change on the very next load
    instead of waiting out the TTL -- same need triple_screen_tracker's
    read-only tab never has (that tab has no write routes at all)."""
    with _cache_lock:
        _cache["rows"] = None
