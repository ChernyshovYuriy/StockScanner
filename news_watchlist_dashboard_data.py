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
from datetime import date, timedelta
from typing import Dict, List, Optional

from config import DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS, NEWS_WATCHLIST_DB_PATH
from market_data import DEFAULT_PROVIDER
from time_utils import date_to_iso_extended, market_today

_cache_lock = threading.Lock()
_cache: Dict[str, object] = {"ts": 0.0, "rows": None}

_ITEM_COLUMNS = (
    "id", "guid", "ticker", "company", "category", "materiality", "summary",
    "source_link", "status", "note", "flagged_at", "flag_price",
    "status_changed_at", "created_at",
)

# Same trailing window and lookback padding as volume_spike_scanner.py's
# own AVG_VOLUME_DAYS/LOOKBACK_CALENDAR_DAYS -- "above/below average" means
# the same thing on both tabs.
_AVG_VOLUME_DAYS = 20
_VOLUME_LOOKBACK_CALENDAR_DAYS = 45


def _pct(flag_price, other_price):
    if not flag_price:
        return None
    return (other_price / flag_price - 1.0) * 100.0


def _fetch_volumes(tickers: List[str]) -> Dict[str, Dict[str, float]]:
    """Today's volume-so-far + trailing _AVG_VOLUME_DAYS average, one
    batched download_range() call -- same source/window
    volume_spike_scanner.py uses. Restricted to Watching-tab tickers only
    (called with a handful, not the whole inbox), since a live per-ticker
    fetch across a 100+ row inbox would slow the page the same way a full
    /volume-spikes scan does (see that page's own async-fetch workaround).
    A ticker missing from the result (network failure, no history yet)
    just gets no entry -- caller renders '—'."""
    if not tickers:
        return {}
    today = market_today()
    start = date_to_iso_extended(today - timedelta(days=_VOLUME_LOOKBACK_CALENDAR_DAYS))
    # end is exclusive in yfinance -- +1 day so today's still-filling bar
    # is actually included, same as volume_spike_scanner.py.
    end = date_to_iso_extended(today + timedelta(days=1))
    data, _failed = DEFAULT_PROVIDER.download_range(sorted(set(tickers)), start=start, end=end)

    out: Dict[str, Dict[str, float]] = {}
    for ticker, df in data.items():
        if df.empty:
            continue
        current_volume = float(df["Volume"].iloc[-1])
        history = df["Volume"].iloc[-(_AVG_VOLUME_DAYS + 1):-1]
        if history.empty:
            continue
        average_volume = float(history.mean())
        if average_volume <= 0:
            continue
        out[ticker] = {"current_volume": current_volume, "average_volume": average_volume}
    return out


def fetch_ticker_volume(ticker: str) -> Optional[Dict[str, float]]:
    """One-ticker lookup for dashboard_app.py's confirm route -- a
    freshly-confirmed item has no cached row yet to read this off of."""
    return _fetch_volumes([ticker]).get(ticker)


def fetch_volumes(tickers: List[str]) -> Dict[str, Dict[str, float]]:
    """Public multi-ticker wrapper around _fetch_volumes(), used by
    dashboard_app.py's /news-watchlist/volumes route -- the Inbox table's
    async volume fill (see templates/news_watchlist.html). Inbox can run
    to 100+ rows, so unlike Watching's own fetch (folded into the page's
    already-cached read), this one is deliberately NOT part of
    build_news_watchlist_state() -- it's fetched client-side, after the
    page has already rendered, same defer-after-render shape
    /volume-spikes/data uses for its own full-universe scan."""
    return _fetch_volumes(tickers)


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
    items = _read_items()
    watching_tickers = [item["ticker"] for item in items if item["status"] == "watching"]
    volumes = _fetch_volumes(watching_tickers)

    inbox, watching, dismissed = [], [], []
    for item in items:
        history = item["price_history"]
        latest_price = history[-1]["close_price"] if history else item["flag_price"]
        item["latest_price"] = latest_price
        item["pct_change"] = _pct(item["flag_price"], latest_price)
        item["days_since_flagged"] = _days_since(item["flagged_at"])
        vol = volumes.get(item["ticker"])
        item["current_volume"] = vol["current_volume"] if vol else None
        item["average_volume"] = vol["average_volume"] if vol else None
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
