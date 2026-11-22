"""
dashboard_market_data.py
========================
Batched, memoized Yahoo Finance reads for the web dashboard's live position
views (dashboard_positions.py and the momentum/kangaroo/macro
*_dashboard_data.py modules).

Those views used position_monitor.load_or_fetch_data() and
fetch_intraday_snapshot() per position: two Yahoo round trips each, one
position after another, and the full daily history downloaded again on
every page load even though past daily bars never change. With ~8 tickers
that is ~16 sequential calls per cold load -- seconds when Yahoo is quick,
dozens of seconds when it is slow or rate-limiting during market hours.

Here a view calls prefetch() once with every ticker it is about to value:
- daily bars: one batched download for every ticker not already held in
  memory, reused for up to DAILY_BARS_MAX_AGE_SECONDS. While the market is
  open today's still-forming daily row is rebuilt from the intraday snapshot
  on every read, so a memo up to that old never shows a stale "today".
  A memo taken while the market was open is refetched once it has closed,
  so the final daily bar replaces the partial one.
- intraday snapshots: one batched 5-min download for every ticker, reused
  for DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS (the views' own page cache).

load_daily_bars()/intraday_snapshot() then serve from memory, with the same
signatures (and fallbacks) as the position_monitor functions they stand in
for. A ticker the batch could not load falls back to
position_monitor.load_or_fetch_data() -- which also reads the CSV cache, so
a delisted name (SES.TO) still shows its last known bars and the stale-data
check still fires.

Read-only: never writes the CSV cache or any DB. Dashboard process only --
the scheduled monitors keep fetching fresh on every run.
"""

from __future__ import annotations

import threading
import time
from datetime import date, timedelta
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from config import DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS
from market_data import DEFAULT_PROVIDER, TodayBar
from position_monitor import fetch_intraday_snapshot as _fetch_one_snapshot
from position_monitor import load_or_fetch_data as _load_one
from time_utils import is_market_open, market_today

DAILY_BARS_MAX_AGE_SECONDS = 30 * 60

_lock = threading.Lock()
# ticker -> (fetched_at monotonic, fetched while market open, start, bars)
_daily: Dict[str, Tuple[float, bool, date, pd.DataFrame]] = {}
# ticker -> (fetched_at monotonic, snapshot or None)
_intraday: Dict[str, Tuple[float, Optional[TodayBar]]] = {}


def _daily_is_fresh(ticker: str, start: date, now: float, market_open: bool) -> bool:
    memo = _daily.get(ticker)
    if memo is None:
        return False
    fetched_at, fetched_open, memo_start, _ = memo
    if memo_start > start or now - fetched_at > DAILY_BARS_MAX_AGE_SECONDS:
        return False
    # Fetched mid-session but the market has since closed: pick up the
    # final daily bar instead of the partial one.
    return not (fetched_open and not market_open)


def prefetch(daily: Dict[str, date], intraday: Iterable[str] = ()) -> None:
    """Load every ticker's bars a view is about to read, in as few Yahoo
    calls as possible. daily maps ticker -> earliest bar date needed;
    intraday lists the tickers needing a live snapshot (pass none while the
    market is closed)."""
    with _lock:
        now = time.monotonic()
        market_open = is_market_open()

        stale = {t: s for t, s in daily.items() if not _daily_is_fresh(t, s, now, market_open)}
        if stale:
            end = (market_today().date() + timedelta(days=1)).isoformat()  # yfinance end is exclusive
            data, failed = DEFAULT_PROVIDER.download_range(
                sorted(stale), start=min(stale.values()).isoformat(), end=end,
            )
            for ticker, start in stale.items():
                df = data.get(ticker)
                if df is None or df.empty or ticker in failed:
                    df = _load_one(ticker, start=start)
                _daily[ticker] = (now, market_open, start, df.sort_index())

        wanted = [t for t in dict.fromkeys(intraday)
                  if t not in _intraday or now - _intraday[t][0] >= DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS]
        if wanted:
            snaps = DEFAULT_PROVIDER.get_intraday_snapshots(wanted)
            for ticker in wanted:
                _intraday[ticker] = (now, snaps.get(ticker))


def intraday_snapshot(ticker: str) -> Optional[TodayBar]:
    """Drop-in for position_monitor.fetch_intraday_snapshot(): the prefetched
    snapshot, or a single fetch for a ticker prefetch() wasn't given."""
    with _lock:
        memo = _intraday.get(ticker)
    if memo is not None:
        return memo[1]
    return _fetch_one_snapshot(ticker)


def load_daily_bars(ticker: str, start: date) -> pd.DataFrame:
    """Drop-in for position_monitor.load_or_fetch_data(): the prefetched
    daily bars (a single fetch for a ticker prefetch() wasn't given). While
    the market is open, today's row is replaced by one built from the
    prefetched intraday snapshot -- the partial daily bar a fresh download
    would carry; with no snapshot the memo's own row is left as it is."""
    with _lock:
        memo = _daily.get(ticker)
        snap_memo = _intraday.get(ticker)
    if memo is None or memo[2] > start:
        return _load_one(ticker, start=start)
    df = memo[3].copy()
    if df.empty or not is_market_open():
        return df

    snap = snap_memo[1] if snap_memo is not None else None
    if snap is None:
        return df
    today = pd.Timestamp(market_today().date())
    today_rows = df[df.index.normalize() == today]
    df = df[df.index.normalize() != today]
    row = {"Open": float("nan"), "High": snap.high, "Low": snap.low, "Close": snap.close, "Volume": float("nan")}
    if not today_rows.empty:
        old = today_rows.iloc[-1]
        row.update(Open=old.get("Open", float("nan")), Volume=old.get("Volume", float("nan")),
                   High=max(snap.high, float(old["High"])), Low=min(snap.low, float(old["Low"])))
    return pd.concat([df, pd.DataFrame([row], index=[today]).reindex(columns=df.columns)])
