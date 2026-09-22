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
from datetime import date, datetime, timedelta
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

# download_range() batches 30 tickers/call with a 0.5s sleep between
# batches (see market_data.py's LiveDataProvider) -- on a Pi's weaker
# CPU/network, an ~88-ticker Inbox can take the better part of a minute.
# Per-ticker TTL cache so a page reload / confirm / dismiss within this
# window (the common case -- the Inbox's ticker set rarely changes
# between clicks) is served instantly instead of re-paying that cost.
_volume_cache_lock = threading.Lock()
_volume_cache: Dict[str, "tuple[float, Dict[str, float]]"] = {}
_VOLUME_CACHE_TTL_SECONDS = 90


def _pct(flag_price, other_price):
    if not flag_price:
        return None
    return (other_price / flag_price - 1.0) * 100.0


def _fetch_volumes(tickers: List[str]) -> Dict[str, Dict[str, float]]:
    """Today's volume-so-far + trailing _AVG_VOLUME_DAYS average, same
    source/window volume_spike_scanner.py uses. Called both for Watching
    (a handful of tickers, folded into the page's own TTL-cached read) and
    for Inbox (up to 100+, fetched client-side -- see fetch_volumes()).
    Per-ticker results are served from _volume_cache when fresh, so only
    tickers actually missing/stale get a real download_range() call. A
    ticker still missing from the result after that (network failure, no
    history yet) just gets no entry -- caller renders '—'."""
    if not tickers:
        return {}
    unique = sorted(set(tickers))
    now = time.monotonic()
    out: Dict[str, Dict[str, float]] = {}
    to_fetch: List[str] = []
    with _volume_cache_lock:
        for ticker in unique:
            cached = _volume_cache.get(ticker)
            if cached is not None and now - cached[0] < _VOLUME_CACHE_TTL_SECONDS:
                out[ticker] = cached[1]
            else:
                to_fetch.append(ticker)

    if to_fetch:
        today = market_today()
        start = date_to_iso_extended(today - timedelta(days=_VOLUME_LOOKBACK_CALENDAR_DAYS))
        # end is exclusive in yfinance -- +1 day so today's still-filling bar
        # is actually included, same as volume_spike_scanner.py.
        end = date_to_iso_extended(today + timedelta(days=1))

        data, _failed = DEFAULT_PROVIDER.download_range(to_fetch, start=start, end=end)

        # A bare ticker (no exchange suffix) straight from the press-release
        # LLM parser (see news_watchlist_service._resolve_market_price()'s
        # own docstring for why it's bare) sometimes has no Yahoo Finance
        # data under that bare symbol at all -- e.g. "AC" (Air Canada) only
        # trades as "AC.TO". Only probed for tickers that came back
        # genuinely empty above, so the common case (an already-correct
        # ticker, bare or suffixed) stays a single batch call -- this
        # dashboard's ~88-ticker Inbox already takes the better part of a
        # minute to fetch once (see this function's own docstring); trying
        # .TO/.V for every bare ticker unconditionally would triple that.
        missing = [t for t in to_fetch if "." not in t and (t not in data or data[t].empty)]
        for suffix in (".TO", ".V"):
            if not missing:
                break
            probe = [t + suffix for t in missing]
            data_p, _failed = DEFAULT_PROVIDER.download_range(probe, start=start, end=end)
            still_missing = []
            for t in missing:
                df = data_p.get(t + suffix)
                if df is not None and not df.empty:
                    data[t] = df
                else:
                    still_missing.append(t)
            missing = still_missing

        fresh: Dict[str, Dict[str, float]] = {}
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
            fresh[ticker] = {"current_volume": current_volume, "average_volume": average_volume}

        with _volume_cache_lock:
            for ticker, vol in fresh.items():
                _volume_cache[ticker] = (now, vol)
        out.update(fresh)

    return out


def format_volume_compact(v: Optional[float]) -> str:
    """Compact K/M/B display for a volume number -- the raw comma-grouped
    form ("1,038,054 / 276,764") was wide enough that adding the Volume
    column pushed the Inbox/Watching tables into horizontal scroll.
    Registered as the `volfmt` Jinja filter by dashboard_app.py for the
    Watching table's server-rendered cells; mirrored in JS by
    templates/news_watchlist.html's own formatVolumeCompact() for the
    Inbox's async-filled cells and the Watching row builder."""
    if v is None:
        return "—"
    v = float(v)
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{v:.0f}"


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


def _flagged_date_time(created_at_str):
    """Inbox's Flagged column shows created_at's date and HH-MM time on two
    lines instead of days_since_flagged -- almost every item is flagged the
    same day it's seeded, so "0d" carried no information."""
    try:
        dt = datetime.fromisoformat(created_at_str)
    except (TypeError, ValueError):
        return None, None
    return dt.date().isoformat(), dt.strftime("%H-%M")


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
        item["flagged_date"], item["flagged_time"] = _flagged_date_time(item["created_at"])
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
