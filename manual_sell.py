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
from db import init_db
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Manually sell an open position at market price")
    parser.add_argument("ticker", help="Ticker of the open position to sell (e.g. SLF.TO)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the planned sell without writing to the database")
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    service = "manual_sell"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        print(f"{Fore.YELLOW}Another manual_sell run is already in progress.{Style.RESET_ALL}")
        sys.exit(1)

    log(service, run_id, "start", ticker=ticker)
    init_db()

    positions = {p.ticker: p for p in parse_positions_from_db()}
    pos = positions.get(ticker)
    if pos is None:
        print(f"{Fore.RED}No open position for {ticker}.{Style.RESET_ALL}")
        log(service, run_id, "no_position", ticker=ticker)
        lock_file.close()
        sys.exit(1)

    price, source = get_market_price(ticker)
    if price is None:
        print(f"{Fore.RED}Could not fetch a market price for {ticker}.{Style.RESET_ALL}")
        log(service, run_id, "no_price", ticker=ticker)
        lock_file.close()
        sys.exit(1)

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

    execute_virtual_sells([sell_row], dry_run=args.dry_run)

    log(service, run_id, "done", ticker=ticker, price=price, dry_run=args.dry_run)
    lock_file.close()


if __name__ == "__main__":
    main()
