"""
triple_screen_tracker_service.py
=================================
Triple Screen tracker — the 7th StockScanner service (see
triple_screen_tracker/__init__.py).

Runs daily after market close (~5pm ET, after main.py's 16:30 pipeline and
position_monitor.py's 15:50 run — no file/DB collision, this service owns
its own SQLite DB): scans the CAN_TICKERS_URL universe with
research/triple_screen's reference engine, "buys" every fresh BUY signal at
that day's close, appends today's close to every already-tracked (OPEN)
record, and "sells" (closes) the first day a record's close drops below its
buy price. Always emails a plain-text report — a report every day, even a
quiet one, is the point of a paper-tracking experiment (unlike edgar's
"quiet day = no email").

Usage
-----
  python triple_screen_tracker_service.py                  # daily run, full CAN_TICKERS_URL universe
  python triple_screen_tracker_service.py --dry-run         # scan + print, write nothing
  python triple_screen_tracker_service.py AAPL MSFT --dry-run   # manual test against explicit tickers
"""
from __future__ import annotations

import argparse
import math
import sys
import uuid
from datetime import date

from concurrent_utils import acquire_lock
from config import CAN_TICKERS_URL
from log_utils import log
from send_report import send_text_email
from time_utils import ISO_DATE_EXTENDED, TSX_HOLIDAYS, TSX_TZ, market_now

from research.triple_screen.batch import build_default_engine, load_tickers
from research.triple_screen.types import Signal, TimeframeConfig
from research.triple_screen.yfinance_provider import YFinanceDataProvider

from triple_screen_tracker import digest, store


def _is_trading_day(today: date) -> bool:
    return today.weekday() < 5 and today.isoformat() not in TSX_HOLIDAYS


def _latest_close(price_data) -> tuple[str, float] | None:
    """The latest bar's (date, Close) off a PriceData's daily bars, or None
    if there's nothing usable -- defensively sorted ascending first, same
    discipline research/triple_screen/indicators.py applies (nothing
    upstream enforces the "sorted ascending" contract; see
    research/triple_screen/TRIPLE_SCREEN_STRESS_FINDINGS.md), and a
    non-finite Close is treated as unusable, not a real price.
    """
    daily = price_data.bars.sort_index()
    if daily.empty:
        return None
    close = float(daily["Close"].iloc[-1])
    if not math.isfinite(close):
        return None
    return daily.index[-1].strftime(ISO_DATE_EXTENDED), close


def run_collector(run_id, dry_run=False, tickers=None, provider=None, conn=None):
    """Event loop: scan for fresh BUYs, roll forward every open record's
    price, close the ones that dropped below entry, then always email a
    report.

    `tickers` overrides the CAN_TICKERS_URL universe (manual/dry-run testing
    only; the daily systemd run always scans the full universe). `provider`
    and `conn` override the default YFinanceDataProvider / store.connect()
    -- test-only seams, mirroring research/triple_screen/batch.run_batch's
    own DataProvider injection.
    """
    today = market_now(TSX_TZ).date()
    today_str = today.isoformat()
    if not _is_trading_day(today):
        log("triple_screen_tracker", run_id, "not_a_trading_day", date=today_str)
        return

    conn = conn or store.connect()
    provider = provider or YFinanceDataProvider()
    engine = build_default_engine()
    timeframes = TimeframeConfig()

    open_records = store.open_records(conn)
    open_tickers = {r["ticker"] for r in open_records}

    universe = tickers if tickers is not None else load_tickers(CAN_TICKERS_URL)
    candidates = [t for t in universe if t not in open_tickers]

    # ── Fresh BUYs: only tickers with no currently-OPEN record are scanned
    #    at all -- this IS the "if a record exists, skip it" rule. ──
    new_buys = []
    for ticker in candidates:
        try:
            aligned = provider.get_bars(ticker, timeframes)
            latest = _latest_close(aligned.entry)
            if latest is None:
                continue
            latest_date, close_price = latest
            if latest_date != today_str:
                # Stale data (provider hasn't published today's bar yet, or
                # returned an old cached value): never record a stale price
                # as "today's" -- skip this ticker this run rather than
                # silently mislabeling it.
                log("triple_screen_tracker", run_id, "stale_data",
                    ticker=ticker, latest_date=latest_date)
                continue
            result = engine.evaluate(aligned, position_open=False)
        except Exception as e:
            print(f"WARNING: skipping {ticker} ({type(e).__name__}: {e})")
            continue
        if result.signal != Signal.BUY:
            continue
        if not dry_run:
            tracked_id = store.create_tracked(
                conn, ticker, today_str, close_price, created_at=market_now().isoformat())
            store.append_price(conn, tracked_id, today_str, close_price)
        new_buys.append({
            "ticker": ticker, "buy_price": close_price,
            "trend": result.trend.direction.value,
            "pullback": result.entry.pullback_present,
            "trigger": result.trigger.triggered,
        })

    # ── Roll forward every already-OPEN record: append today's close, sell
    #    (close) it if that close is below the buy price. This is a pure
    #    price rule -- the engine's own trend-flip SELL verdict is never
    #    consulted for a held ticker (the engine isn't even called here). ──
    new_sells = []
    open_rows = []
    for r in open_records:
        try:
            aligned = provider.get_bars(r["ticker"], timeframes)
            latest = _latest_close(aligned.entry)
        except Exception as e:
            print(f"WARNING: skipping {r['ticker']} ({type(e).__name__}: {e})")
            continue
        if latest is None:
            continue
        latest_date, close_price = latest
        if latest_date != today_str:
            log("triple_screen_tracker", run_id, "stale_data",
                ticker=r["ticker"], latest_date=latest_date)
            continue

        if not dry_run:
            store.append_price(conn, r["id"], today_str, close_price)

        days_held = (today - date.fromisoformat(r["buy_date"])).days
        if close_price < r["buy_price"]:
            if not dry_run:
                store.close_tracked(conn, r["id"], today_str, close_price)
            new_sells.append({
                "ticker": r["ticker"], "buy_date": r["buy_date"], "buy_price": r["buy_price"],
                "sell_date": today_str, "sell_price": close_price,
            })
        else:
            open_rows.append({
                "ticker": r["ticker"], "buy_date": r["buy_date"], "buy_price": r["buy_price"],
                "latest_price": close_price, "days_held": days_held,
            })

    log("triple_screen_tracker", run_id, "scanned", candidates=len(candidates),
        open_before=len(open_records), new_buys=len(new_buys), new_sells=len(new_sells))

    subject, body = digest.build_digest(today_str, new_buys, new_sells, open_rows)

    if dry_run:
        print(subject)
        print(body)
        log("triple_screen_tracker", run_id, "dry_run",
            new_buys=len(new_buys), new_sells=len(new_sells), open=len(open_rows))
        return

    # Idempotent same-day re-run guard -- writes above are already
    # idempotent themselves (open-ticker exclusion, INSERT OR REPLACE,
    # status flip), this only prevents a duplicate email.
    if store.already_sent(conn, today_str):
        log("triple_screen_tracker", run_id, "already_sent_today", date=today_str)
        return

    sent = send_text_email(subject, body)
    store.record_email(conn, today_str, len(new_buys) + len(new_sells), sent=sent)
    log("triple_screen_tracker", run_id, "sent" if sent else "send_skipped",
        new_buys=len(new_buys), new_sells=len(new_sells), open=len(open_rows))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Triple Screen tracker — 7th StockScanner service")
    p.add_argument("tickers", nargs="*",
                   help="explicit tickers, e.g. AAPL MSFT SLF.TO (overrides CAN_TICKERS_URL; manual/dry-run testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="Scan and print the digest without writing to the DB or sending email.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    service = "triple_screen_tracker"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "start", dry_run=args.dry_run)
    try:
        run_collector(run_id, dry_run=args.dry_run, tickers=args.tickers or None)
        log(service, run_id, "completed")
    except Exception as e:
        log(service, run_id, "error", error=str(e))
        raise
    finally:
        lock_file.close()
