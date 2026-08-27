"""
market_data_cache.py
=====================
Local DuckDB cache of daily OHLCV bars for backtest data, so repeated
backtest runs don't re-download unchanged history from Yahoo Finance on
every invocation.

sync_and_load() is the entry point: given a ticker list and a date range,
it fetches from Yahoo Finance only what's missing from the cache (a brand
new ticker gets its full history; a ticker already cached only needs its
tail topped up to `end`, plus a few days of overlap re-fetched in case of
late corrections) and returns the full requested range read back out of
DuckDB.

Cache lives in its own file, data/market_cache.db — separate from
data/trading.db (that one is live trading state; this one is just a
mirror of Yahoo Finance and is safe to delete any time, it will simply be
rebuilt on the next run).

Usage
-----
    from market_data_cache import sync_and_load

    data = sync_and_load(["RY.TO", "TD.TO", "XIU.TO"],
                          start="2022-01-01", end="2026-08-26")
    # -> Dict[str, pd.DataFrame], ready for HistoricalSliceProvider(data)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import yfinance as yf

CACHE_DB_PATH = "data/market_cache.db"

_BATCH_SIZE = 30
_SLEEP_SECONDS = 0.5

# When a cached ticker's tail is being topped up, re-fetch this many days
# before the cached max date too, in case Yahoo revises recent bars
# (splits/dividends applied a day or two late) after they first appear.
_TAIL_OVERLAP_DAYS = 5

# `start`/`end` are calendar dates that may land on a weekend or market
# holiday, so the cached min/max trading date is legitimately a few days
# inside the requested range even when the cache is fully up to date. Only
# treat a ticker as needing a fetch once the gap exceeds this tolerance —
# otherwise every call with the same (start, end) would spuriously re-fetch.
_END_COVERAGE_TOLERANCE_DAYS = 5
_START_COVERAGE_TOLERANCE_DAYS = 5


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION / SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

def _connect(db_path: str = CACHE_DB_PATH) -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_daily (
            ticker VARCHAR NOT NULL,
            date   DATE    NOT NULL,
            open   DOUBLE,
            high   DOUBLE,
            low    DOUBLE,
            close  DOUBLE,
            volume BIGINT,
            PRIMARY KEY (ticker, date)
        )
    """)
    return conn


def _cached_range(
    conn: duckdb.DuckDBPyConnection, ticker: str
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    row = conn.execute(
        "SELECT MIN(date), MAX(date) FROM ohlcv_daily WHERE ticker = ?", [ticker]
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return pd.Timestamp(row[0]), pd.Timestamp(row[1])


def _upsert(conn: duckdb.DuckDBPyConnection, ticker: str, df: pd.DataFrame) -> None:
    """
    Delete-then-insert df's date range for ticker (idempotent re-sync).

    Inserts via a registered DataFrame + INSERT...SELECT rather than
    executemany() — DuckDB's Python executemany() issues one row at a time,
    which is fine for a handful of rows but ~1000x too slow for a
    multi-year, multi-ticker sync (this is the same "register a DataFrame,
    bulk-insert with SQL" pattern DuckDB's own docs recommend for pandas
    ingestion).
    """
    if df.empty:
        return
    lo, hi = df.index.min(), df.index.max()
    conn.execute(
        "DELETE FROM ohlcv_daily WHERE ticker = ? AND date >= ? AND date <= ?",
        [ticker, lo.date(), hi.date()],
    )
    insert_df = pd.DataFrame({
        "ticker": ticker,
        "date":   df.index.date,
        "open":   df["Open"].astype(float),
        "high":   df["High"].astype(float),
        "low":    df["Low"].astype(float),
        "close":  df["Close"].astype(float),
        "volume": df["Volume"].astype("int64"),
    })
    conn.register("_upsert_batch", insert_df)
    try:
        conn.execute("INSERT INTO ohlcv_daily SELECT * FROM _upsert_batch")
    finally:
        conn.unregister("_upsert_batch")


# ─────────────────────────────────────────────────────────────────────────────
# SYNC — fetch only what's missing
# ─────────────────────────────────────────────────────────────────────────────

def sync_tickers(
    tickers: List[str],
    start: str,
    end: str,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    force_refresh: bool = False,
    quiet: bool = False,
) -> None:
    """
    Ensure the cache covers [start, end] for every ticker, fetching from
    Yahoo Finance only the missing pieces (new tickers: full range;
    already-cached tickers: just the tail past their cached max date, plus
    an earlier backfill if `start` predates what's cached).

    force_refresh=True ignores existing cache coverage and re-downloads the
    full [start, end] window for every ticker.
    """
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        to_fetch: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}

        for ticker in tickers:
            if force_refresh:
                to_fetch[ticker] = (start_ts, end_ts)
                continue

            cached = _cached_range(conn, ticker)
            if cached is None:
                to_fetch[ticker] = (start_ts, end_ts)
                continue

            cached_lo, cached_hi = cached
            needs_backfill = (cached_lo - start_ts).days > _START_COVERAGE_TOLERANCE_DAYS
            needs_tail = (end_ts - cached_hi).days > _END_COVERAGE_TOLERANCE_DAYS
            lo = start_ts if needs_backfill else None
            hi = end_ts if needs_tail else None
            if lo is None and hi is None:
                continue  # already fully covered

            fetch_lo = lo if lo is not None else \
                max(start_ts, cached_hi - pd.Timedelta(days=_TAIL_OVERLAP_DAYS))
            to_fetch[ticker] = (fetch_lo, end_ts)

        if not to_fetch:
            if not quiet:
                print("  [market_data_cache] cache already covers requested range")
            return

        if not quiet:
            print(f"  [market_data_cache] fetching {len(to_fetch)}/{len(tickers)} "
                  f"tickers needing update...")

        # Group tickers that need the identical fetch window so they can be
        # batch-downloaded together (the common case: every new ticker needs
        # the same full range, every already-cached ticker needs the same
        # tail top-up).
        groups: Dict[Tuple[str, str], List[str]] = {}
        for ticker, (lo, hi) in to_fetch.items():
            key = (lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))
            groups.setdefault(key, []).append(ticker)

        # One transaction for the whole sync rather than one per ticker —
        # each _upsert() otherwise auto-commits (and fsyncs) on its own.
        conn.execute("BEGIN TRANSACTION")
        try:
            for (g_start, g_end), g_tickers in groups.items():
                for i in range(0, len(g_tickers), _BATCH_SIZE):
                    batch = g_tickers[i : i + _BATCH_SIZE]
                    try:
                        raw = yf.download(
                            batch,
                            start=g_start,
                            end=g_end,
                            auto_adjust=True,
                            progress=False,
                            threads=True,
                            timeout=10,
                        )
                        for ticker in batch:
                            try:
                                if isinstance(raw.columns, pd.MultiIndex):
                                    df = pd.DataFrame({
                                        "Open":   raw["Open"][ticker],
                                        "High":   raw["High"][ticker],
                                        "Low":    raw["Low"][ticker],
                                        "Close":  raw["Close"][ticker],
                                        "Volume": raw["Volume"][ticker],
                                    }).dropna()
                                else:
                                    if ticker != batch[0]:
                                        continue
                                    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

                                if not df.empty:
                                    df.index = pd.to_datetime(df.index).tz_localize(None)
                                    _upsert(conn, ticker, df)
                            except Exception:
                                if not quiet:
                                    print(f"  [market_data_cache] failed to fetch {ticker}")
                    except Exception:
                        if not quiet:
                            print(f"  [market_data_cache] batch fetch failed: {batch}")

                    time.sleep(_SLEEP_SECONDS)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        if not quiet:
            print("  [market_data_cache] sync complete")
    finally:
        if own_conn:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD — read the requested range back out of the cache
# ─────────────────────────────────────────────────────────────────────────────

def load_from_cache(
    tickers: List[str],
    start: str,
    end: str,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> Dict[str, pd.DataFrame]:
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        start_d, end_d = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        data: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = conn.execute(
                "SELECT date, open, high, low, close, volume FROM ohlcv_daily "
                "WHERE ticker = ? AND date >= ? AND date <= ? ORDER BY date",
                [ticker, start_d, end_d],
            ).df()
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
            data[ticker] = df
        return data
    finally:
        if own_conn:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE — sync then load in one call
# ─────────────────────────────────────────────────────────────────────────────

def sync_and_load(
    tickers: List[str],
    start: str,
    end: str,
    force_refresh: bool = False,
    quiet: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Sync the cache to cover [start, end] for `tickers`, then return it."""
    conn = _connect()
    try:
        sync_tickers(tickers, start, end, conn=conn,
                     force_refresh=force_refresh, quiet=quiet)
        return load_from_cache(tickers, start, end, conn=conn)
    finally:
        conn.close()
