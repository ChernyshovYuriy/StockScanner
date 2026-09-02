"""
demand_signals_service.py
==========================
Demand-signals collector — the 5th StockScanner service (independent of
edgar_service.py and the TSX trio; see demand_signals/__init__.py).

Runs daily: for each watchlisted ticker (reuses edgar.store's watchlist --
the same CIKs already tracked for insider buys), normalizes EDGAR insider
buys + FINRA dark-pool ratio + an options-flow proxy into
demand_signals.db, so demand_signals.store.signals_for_ticker() has
something to screen.

No email digest yet -- same "interim" staging edgar_service.py itself
started with. This just keeps the DB current.

Usage
-----
  python demand_signals_service.py              # daily run
  python demand_signals_service.py --dry-run    # print computed signals, write nothing
"""
from __future__ import annotations

import argparse
import sys
import uuid

from concurrent_utils import acquire_lock
from log_utils import log
from time_utils import market_now

from demand_signals import darkpool, edgar_adapter, options_flow, store
from demand_signals.options_flow import YahooOptionsProvider
from demand_signals.ticker_map import get_us_ticker

from edgar import store as edgar_store
from edgar.core import load_cik_to_ticker


def run_collector(run_id, dry_run=False):
    conn = store.connect()
    e_conn = edgar_store.connect()

    fetched_at = market_now().isoformat()
    all_signals = []

    # ── EDGAR insider buys: adapt what edgar_service.py already collected,
    #    never re-fetched here ──
    insider_signals = edgar_adapter.normalize_recent_insider_buys(e_conn)
    all_signals.extend(insider_signals)
    log("demand_signals", run_id, "edgar_adapted", count=len(insider_signals))

    # ── dark-pool + options-flow, driven by the same watchlist CIKs edgar
    #    already tracks (no separate watchlist to maintain) ──
    cik2tic = load_cik_to_ticker()
    tickers = sorted({cik2tic[cik] for cik in edgar_store.watchlist_ciks(e_conn) if cik in cik2tic})

    provider = YahooOptionsProvider()
    for ticker in tickers:
        us_ticker = get_us_ticker(ticker)
        if not us_ticker:
            log("demand_signals", run_id, "no_us_line", ticker=ticker)
            continue

        ats_weekly = darkpool.fetch_weekly_ats_volume(us_ticker)
        if ats_weekly:
            all_signals.extend(darkpool.build_signals(ticker, us_ticker, ats_weekly, fetched_at))

        try:
            snap = provider.snapshot(us_ticker)
        except Exception as exc:
            log("demand_signals", run_id, "options_flow_failed", ticker=ticker, error=str(exc))
            snap = None
        if snap:
            all_signals.extend(options_flow.build_signals(ticker, us_ticker, snap, fetched_at))

    log("demand_signals", run_id, "computed", total_signals=len(all_signals))

    if dry_run:
        for s in all_signals:
            print(f"{s.ticker:<10} {s.source:<16} {s.signal_type:<24} "
                  f"{s.direction:<8} {s.strength:.2f}  {s.date}")
        return

    store.save_signals(conn, all_signals)
    log("demand_signals", run_id, "stored", count=len(all_signals))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Demand-signals collector — 5th StockScanner service")
    p.add_argument("--dry-run", action="store_true",
                   help="Print computed signals without writing to the DB.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    service = "demand_signals"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "start", dry_run=args.dry_run)
    try:
        run_collector(run_id, dry_run=args.dry_run)
        log(service, run_id, "completed")
    except Exception as e:
        log(service, run_id, "error", error=str(e))
        raise
    finally:
        lock_file.close()
