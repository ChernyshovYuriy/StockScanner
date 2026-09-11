"""
scanner_pipeline.py
=====================
Ticker Indicator Board — daily fetch/compute/persist entrypoint, the 9th
StockScanner service (see scanner_board/PLAN.md). Runs ~17:15 ET, after
the Triple Screen tracker's 17:00 slot — no file/DB collision, this
service owns its own SQLite DB, data/scanner_board.db.

For every ticker in the CAN_TICKERS_URL universe: fetch/top-up daily OHLCV
via market_data_cache.py (the same incremental DuckDB cache run_backtest.py
uses, so a daily run here never re-downloads unchanged history), resample
to weekly (scanner_board/weekly.py — "perform weekly studies each day",
Ch.23), compute one Board row (scanner_board/row.py), and persist the
whole day's snapshot in one transaction (scanner_board/store.py).

Read-only research board: no capital, no positions, no buy/sell — a
failed fetch/compute for one ticker is logged and skipped, never aborts
the run (same per-ticker resilience precedent as
triple_screen_tracker_service.py). The daily-history lookback window
reuses canadian_stock_screener.CONFIG.lookback_days (504 calendar days)
rather than a fresh, unvalidated number of our own.

Usage
-----
  python scanner_pipeline.py                        # daily run, full CAN_TICKERS_URL universe
  python scanner_pipeline.py --dry-run               # scan + print a summary, write nothing
  python scanner_pipeline.py AAPL MSFT --dry-run      # manual test against explicit tickers
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date, timedelta

from canadian_stock_screener import CONFIG as SCREENER_CONFIG
from concurrent_utils import acquire_lock
from config import CAN_TICKERS_URL
from log_utils import log
from market_data_cache import sync_and_load
from time_utils import TSX_HOLIDAYS, TSX_TZ, market_now

from research.triple_screen.batch import load_tickers

from scanner_board import row as row_mod
from scanner_board import store
from scanner_board.weekly import resample_ohlcv_weekly

# scanner_board/row.py assumes both `daily` and `weekly` are non-empty (see
# its own module docstring) — a handful of daily bars is nowhere near
# enough to seed anything meaningful anyway, and resample_ohlcv_weekly()
# drops a still-in-progress trailing week, so too few daily bars can
# resample to an EMPTY weekly frame. 30 daily bars (~6 calendar weeks)
# safely guarantees at least several complete weekly bars survive that
# drop.
MIN_DAILY_BARS = 30


def _is_trading_day(today: date) -> bool:
    return today.weekday() < 5 and today.isoformat() not in TSX_HOLIDAYS


def run_pipeline(run_id: str, dry_run: bool = False, tickers=None, conn=None) -> None:
    """Event loop: fetch/top-up daily bars for the universe, compute one
    Board row per ticker, persist the whole snapshot in one transaction.

    `tickers` overrides the CAN_TICKERS_URL universe (manual/dry-run
    testing only; the daily systemd run always scans the full universe).
    `conn` overrides the default store.connect() — a test-only seam,
    mirroring triple_screen_tracker_service.run_collector's own.
    """
    today = market_now(TSX_TZ).date()
    today_str = today.isoformat()
    if not _is_trading_day(today):
        log("scanner_board", run_id, "not_a_trading_day", date=today_str)
        return

    universe = tickers if tickers is not None else load_tickers(CAN_TICKERS_URL)

    start = (today - timedelta(days=SCREENER_CONFIG.lookback_days)).isoformat()
    daily_by_ticker = sync_and_load(universe, start=start, end=today_str, quiet=True)

    rows = []
    skipped = 0
    for ticker in universe:
        daily = daily_by_ticker.get(ticker)
        if daily is None or len(daily) < MIN_DAILY_BARS:
            skipped += 1
            continue
        try:
            weekly = resample_ohlcv_weekly(daily)
            if weekly.empty:
                skipped += 1
                continue
            rows.append(row_mod.compute_row(ticker, daily, weekly))
        except Exception as e:
            log("scanner_board", run_id, "ticker_failed", ticker=ticker,
                error=f"{type(e).__name__}: {e}")
            skipped += 1

    log("scanner_board", run_id, "computed", universe=len(universe),
        rows=len(rows), skipped=skipped)

    if dry_run:
        print(f"scanner_board dry-run {today_str}: {len(rows)} rows, {skipped} skipped")
        for r in rows[:10]:
            print(f"  {r['ticker']:10s} price={r['price']!s:>10} "
                  f"trend_health={r['trend_health']:8s} "
                  f"triple_screen={r['triple_screen']:16s} "
                  f"impulse_daily={r['impulse_daily']}")
        return

    if not rows:
        log("scanner_board", run_id, "no_rows_to_write")
        return

    own_conn = conn is None
    conn = conn or store.connect()
    try:
        store.upsert_rows(conn, today_str, rows, updated_at=market_now().isoformat())
        log("scanner_board", run_id, "persisted", rows=len(rows), run_date=today_str)
    finally:
        if own_conn:
            conn.close()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ticker Indicator Board — 9th StockScanner service")
    p.add_argument("tickers", nargs="*",
                   help="explicit tickers, e.g. AAPL MSFT SLF.TO (overrides CAN_TICKERS_URL; manual/dry-run testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="Scan and print a summary without writing to the DB.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    service = "scanner_board"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "start", dry_run=args.dry_run)
    try:
        run_pipeline(run_id, dry_run=args.dry_run, tickers=args.tickers or None)
        log(service, run_id, "completed")
    except Exception as e:
        log(service, run_id, "error", error=str(e))
        raise
    finally:
        lock_file.close()
