"""
manual_sell.py
===============
Manually sell one open position at the current market price.

Usage:
    python manual_sell.py TICKER
    python manual_sell.py TICKER --dry-run   # print the planned sell, write nothing

Fetches a best-effort current price (live 5-min intraday snapshot, falling back
to the last completed daily close), then closes the position through the same
execute_virtual_sells() path used by position_monitor.py — removing the
position, crediting cash, recording the trade, and sending the usual
transaction email.
"""

from __future__ import annotations

import sys
import uuid
from datetime import date, timedelta

from colorama import Fore, Style, init

from concurrent_utils import acquire_lock
from log_utils import log
from position_monitor import (
    download_ohlc,
    execute_virtual_sells,
    fetch_intraday_snapshot,
    parse_positions_from_db,
)
from schema_keys import (
    POSITION_COL_ENTRY_DATE,
    POSITION_COL_ENTRY_PRICE,
    POSITION_COL_LAST_CLOSE,
    POSITION_COL_PNL_DOLLARS,
    POSITION_COL_PNL_PCT,
    POSITION_COL_REASON,
    POSITION_COL_SHARES,
    SIGNAL_COL_TICKER,
)

init(autoreset=True)


def get_market_price(ticker: str) -> tuple[float, str] | tuple[None, None]:
    """Best-effort current price: live intraday snapshot, else last daily close."""
    snap = fetch_intraday_snapshot(ticker)
    if snap is not None:
        return snap.close, snap.source
    df = download_ohlc(ticker, start=date.today() - timedelta(days=10))
    if df.empty:
        return None, None
    return float(df["Close"].iloc[-1]), "daily-close"


def sell_position(ticker: str, dry_run: bool = False) -> dict:
    """
    Close one open position at the current market price.

    Acquires the "manual_sell" lock itself (the same lock name used
    regardless of caller, so the CLI and the web dashboard can never race
    each other or double-sell the same ticker).

    Returns a dict:
      ok=True  : {"ok": True, "ticker", "price", "source",
                  "pnl_dollars", "pnl_pct", "funds_state"}
      ok=False : {"ok": False, "ticker",
                  "error": "locked" | "no_position" | "no_price",
                  "message": "<human readable>"}

    Does not raise for expected failure paths — callers branch on result["ok"].

    Does NOT call db.init_db() — the caller must have already initialised
    the database (db.init_db()'s bare/no-arg form always resets DB_PATH back
    to the production default, which would clobber a caller's explicit path,
    e.g. a test's tmp_path DB or the web dashboard's own startup init).
    """
    ticker = ticker.strip().upper()
    service = "manual_sell"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "locked", ticker=ticker)
        return {"ok": False, "ticker": ticker, "error": "locked",
                "message": "Another manual_sell run is already in progress."}

    try:
        log(service, run_id, "start", ticker=ticker)

        positions = {p.ticker: p for p in parse_positions_from_db()}
        pos = positions.get(ticker)
        if pos is None:
            log(service, run_id, "no_position", ticker=ticker)
            return {"ok": False, "ticker": ticker, "error": "no_position",
                    "message": f"No open position for {ticker}."}

        price, source = get_market_price(ticker)
        if price is None:
            log(service, run_id, "no_price", ticker=ticker)
            return {"ok": False, "ticker": ticker, "error": "no_price",
                    "message": f"Could not fetch a market price for {ticker}."}

        print(f"  Price source: {source}  →  ${price:.4f}")

        pnl_dollars = (price - pos.entry_price) * pos.shares
        pnl_pct = (price / pos.entry_price - 1.0) * 100.0

        sell_row = {
            SIGNAL_COL_TICKER: ticker,
            POSITION_COL_ENTRY_DATE: pos.entry_date.isoformat(),
            POSITION_COL_ENTRY_PRICE: pos.entry_price,
            POSITION_COL_SHARES: pos.shares,
            POSITION_COL_LAST_CLOSE: price,
            POSITION_COL_PNL_DOLLARS: pnl_dollars,
            POSITION_COL_PNL_PCT: pnl_pct,
            POSITION_COL_REASON: "MANUAL_SELL",
        }

        funds_state = execute_virtual_sells([sell_row], dry_run=dry_run)

        log(service, run_id, "done", ticker=ticker, price=price, dry_run=dry_run)
        return {
            "ok": True,
            "ticker": ticker,
            "price": price,
            "source": source,
            "pnl_dollars": pnl_dollars,
            "pnl_pct": pnl_pct,
            "funds_state": funds_state,
        }
    finally:
        # Always release the lock, including on an unexpected exception —
        # required now that this function is also called from the long-lived
        # web dashboard process, where a leaked lock would wedge every future
        # sell (CLI and web) until the service is restarted.
        lock_file.close()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Manually sell an open position at market price")
    parser.add_argument("ticker", help="Ticker of the open position to sell (e.g. SLF.TO)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the planned sell without writing to the database")
    args = parser.parse_args()

    from db import init_db
    init_db()

    result = sell_position(args.ticker, dry_run=args.dry_run)

    if not result["ok"]:
        print(f"{Fore.RED}{result['message']}{Style.RESET_ALL}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
