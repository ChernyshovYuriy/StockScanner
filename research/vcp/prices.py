"""
One-time price download for the VCP study, into its own DuckDB file
(cache/vcp_prices.db) -- never market_cache.db, which other tools share.

Reuses market_data_cache's sync/load and its schema through a connection to
this file. sync_tickers() itself has no pacing fit for a bulk job and swallows
download errors, so this module adds what a bulk job needs: at most
YAHOO_PER_SECOND symbols per second, a resumable status table, and a stop when
Yahoo refuses requests (a chunk with no bars at all is checked against the
benchmark before its names are recorded as having no data).

Usage:
    python -m research.vcp.prices --tickers FILE            # download / resume
    python -m research.vcp.prices --tickers FILE --retry    # ask again for names with no data
    python -m research.vcp.prices --status
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import duckdb
import pandas as pd

import market_data_cache as mdc

DB_PATH = str(Path(__file__).resolve().parents[2] / "cache" / "vcp_prices.db")
BENCHMARK = "XIU.TO"
START = "2018-01-01"
END = "2026-11-20"
YAHOO_PER_SECOND = 0.4            # the repo's bulk-job ceiling, counted per symbol
CHUNK = 10
PROBE_DAYS = 14                   # benchmark window used to tell "no data" from "blocked"

OK, NO_DATA = "ok", "no_data"

# Same patterns as the Financing repo's swing_universe.is_excluded_instrument.
_NON_COMMON_RE = re.compile(r"-(UN|U|WT|RT|DB)[A-Z]?\.|-P[A-Z]{1,2}\.")
_LISTED_RE = re.compile(r"\.(TO|V|CN)$")


class YahooBlocked(RuntimeError):
    """Yahoo is refusing requests; stop rather than make the block longer."""


class Pacer:
    def __init__(self, per_second: float, sleep=time.sleep, clock=time.monotonic):
        self._gap, self._next = 1.0 / per_second, 0.0
        self._sleep, self._clock = sleep, clock

    def wait(self, n: int) -> None:
        """Book `n` requests; sleep until the previous booking is paid off."""
        now = self._clock()
        delay = max(0.0, self._next - now)
        self._next = max(now, self._next) + self._gap * n
        if delay:
            self._sleep(delay)


def is_candidate(ticker: str) -> bool:
    """A TSX/TSXV/CSE common share, judged from its symbol."""
    return bool(_LISTED_RE.search(ticker)) and not _NON_COMMON_RE.search(ticker)


def candidate_tickers(raw: Iterable[str]) -> List[str]:
    names = {t.strip().upper() for t in raw}
    return sorted(t for t in names if is_candidate(t) and t != BENCHMARK)


def connect(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    conn = mdc._connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS download_status ("
                 "ticker VARCHAR PRIMARY KEY, status VARCHAR NOT NULL, n_rows INTEGER NOT NULL)")
    return conn


def _record(conn, ticker: str, status: str, n_rows: int) -> None:
    conn.execute("INSERT OR REPLACE INTO download_status VALUES (?, ?, ?)", [ticker, status, n_rows])


def statuses(conn) -> Dict[str, str]:
    return dict(conn.execute("SELECT ticker, status FROM download_status").fetchall())


def _row_counts(conn, tickers: List[str]) -> Dict[str, int]:
    marks = ",".join("?" * len(tickers))
    return dict(conn.execute(f"SELECT ticker, COUNT(*) FROM ohlcv_daily WHERE ticker IN ({marks}) "
                             "GROUP BY ticker", tickers).fetchall())


def _fetch(conn, tickers, start, end, sync_fn, pacer) -> Dict[str, int]:
    pacer.wait(len(tickers))
    sync_fn(tickers, start, end, conn=conn, force_refresh=True, quiet=True)
    return _row_counts(conn, tickers)


def _require_not_blocked(end, sync_fn, pacer) -> None:
    """Fetch a short stretch of the benchmark into a throwaway database; no bars
    means Yahoo refuses us (the stored benchmark can't pass for a fresh answer)."""
    start = (pd.Timestamp(end) - pd.Timedelta(days=PROBE_DAYS)).strftime("%Y-%m-%d")
    probe = mdc._connect(":memory:")
    try:
        if not _fetch(probe, [BENCHMARK], start, end, sync_fn, pacer):
            raise YahooBlocked("benchmark probe returned no bars")
    finally:
        probe.close()


def download(conn, tickers: List[str], start: str = START, end: str = END, retry: bool = False,
             sync_fn: Optional[Callable] = None, pacer: Optional[Pacer] = None,
             progress: Callable[[str], None] = print) -> Dict[str, int]:
    """Download every ticker not already recorded. Raises YahooBlocked (with
    everything fetched so far kept) if Yahoo stops answering."""
    sync_fn = sync_fn or mdc.sync_tickers
    pacer = pacer or Pacer(YAHOO_PER_SECOND)

    done = statuses(conn)
    if done.get(BENCHMARK) != OK:
        if not _fetch(conn, [BENCHMARK], start, end, sync_fn, pacer).get(BENCHMARK):
            raise YahooBlocked("no benchmark bars at the start")
        _record(conn, BENCHMARK, OK, _row_counts(conn, [BENCHMARK])[BENCHMARK])

    todo = [t for t in tickers if t != BENCHMARK and
            (t not in done or (retry and done[t] == NO_DATA))]
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        counts = _fetch(conn, chunk, start, end, sync_fn, pacer)
        if not counts:
            _require_not_blocked(end, sync_fn, pacer)
        for t in chunk:
            n = counts.get(t, 0)
            _record(conn, t, OK if n else NO_DATA, n)
        progress(f"{min(i + CHUNK, len(todo))}/{len(todo)} done")

    final = list(statuses(conn).values())
    return {OK: final.count(OK), NO_DATA: final.count(NO_DATA)}


def load_all(conn) -> Dict[str, pd.DataFrame]:
    """{ticker: OHLCV frame} for every ticker in the file, benchmark included."""
    df = conn.execute("SELECT ticker, date, open, high, low, close, volume "
                      "FROM ohlcv_daily ORDER BY ticker, date").df()
    df["date"] = pd.to_datetime(df["date"])
    out: Dict[str, pd.DataFrame] = {}
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.drop(columns="ticker").set_index("date")
        g.columns = ["Open", "High", "Low", "Close", "Volume"]
        out[ticker] = g
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Paced price download for the VCP study")
    p.add_argument("--tickers", help="file with one ticker per line")
    p.add_argument("--retry", action="store_true", help="ask again for names recorded with no data")
    p.add_argument("--status", action="store_true")
    args = p.parse_args(argv)
    conn = connect()
    try:
        if args.status or not args.tickers:
            s = list(statuses(conn).values())
            print({OK: s.count(OK), NO_DATA: s.count(NO_DATA)})
            return 0
        tickers = candidate_tickers(Path(args.tickers).read_text().splitlines())
        print(f"{len(tickers)} candidates, about {len(tickers) / YAHOO_PER_SECOND / 60:.0f} min if none is cached")
        try:
            print(download(conn, tickers, retry=args.retry,
                           progress=lambda m: print(m, flush=True)))
        except YahooBlocked as exc:
            print(f"STOPPED, Yahoo is refusing requests: {exc}. Re-run later to resume.")
            return 2
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
