"""
kangaroo_buy.py
================
Breakout-trigger execution for the Kangaroo Tail sleeve — separate
DB/capital from every other sleeve (see config.py KANGAROO_* and
CLAUDE.md).

Deliberate deviation from virtual_buy.py/momentum_buy.py's "buy at market
open off a confirmed intent" pattern: a Kangaroo Tail intent is a PENDING
BREAKOUT (see kangaroo_pipeline.py), not a same-day buy signal — Phase 2
found no edge for buying at the tail's own close. This script instead
watches each pending intent's live intraday session HIGH and only fills
once it has traded through entry_price_planned (the tail's own high, a
buy-stop level), at that exact trigger price — matching the assumption the
R-multiple backtest was validated under (see
research/kangaroo_tail/KANGAROO_TAIL_VERIFICATION_FINDINGS.md). Real-world
slippage on a buy-stop fill is NOT modelled.

Because the trigger check needs to know whether TODAY's session has
already traded through the level, this runs at the pre-close slot
(~15:50 ET, same time as kangaroo_monitor.py) rather than virtual_buy.py's
9:45 AM — there is nothing to check yet at the open.

Run manually or via system/stockscanner-kangaroo-buy.{service,timer}.

Usage
-----
  python kangaroo_buy.py
  python kangaroo_buy.py --dry-run
"""

from __future__ import annotations

import sys
import uuid
from typing import Optional

from colorama import Fore, Style, init

from concurrent_utils import acquire_lock
from config import KANGAROO_DB_PATH, KANGAROO_MAX_POSITIONS, KANGAROO_RISK_PER_TRADE_PCT
from db import (
    get_cash,
    get_open_positions_df,
    init_db,
    insert_position,
    load_pending_intents,
    mark_intent_executed,
)
from log_utils import log
from market_data import DEFAULT_PROVIDER
from schema_keys import (
    INTENT_COL_ENTRY_PRICE_PLANNED,
    INTENT_COL_STOP_PRICE,
    INTENT_COL_TARGET_PRICE,
    POSITION_COL_ENTRY_DATE,
    POSITION_COL_ENTRY_PRICE,
    POSITION_COL_SHARES,
    SIGNAL_COL_TICKER,
)
from send_report import send_transaction_email
from time_utils import is_market_open, market_today

init(autoreset=True)


def run_kangaroo_buy(dry_run: bool, run_id: Optional[str] = None) -> None:
    service = "kangaroo_buy"
    run_id = run_id or uuid.uuid4().hex
    log(service, run_id, "start", dry_run=dry_run)

    print(f"\n{'=' * 60}")
    print(f"  {Fore.YELLOW}🦘  Kangaroo Tail Breakout Watcher{Style.RESET_ALL}")
    print(f"{'=' * 60}\n")

    if not dry_run and not is_market_open():
        print(f"{Fore.YELLOW}Market closed — no breakout can be confirmed, nothing to buy.{Style.RESET_ALL}")
        log(service, run_id, "skip_market_closed")
        return

    intents_df = load_pending_intents()
    if intents_df.empty:
        print(f"{Fore.YELLOW}No pending breakout intents — nothing to watch.{Style.RESET_ALL}")
        return

    owned = set(get_open_positions_df()["ticker"].str.upper())
    total_funds = get_cash()
    if total_funds <= 0:
        print(f"{Fore.YELLOW}Available funds is ${total_funds:,.2f} — nothing to buy.{Style.RESET_ALL}")
        return

    current_position_count = len(owned)
    remaining_slots = KANGAROO_MAX_POSITIONS - current_position_count
    if remaining_slots <= 0:
        print(f"{Fore.YELLOW}Portfolio full — {current_position_count} of "
              f"{KANGAROO_MAX_POSITIONS} positions occupied. Nothing to buy.{Style.RESET_ALL}")
        return

    max_position_value = total_funds / remaining_slots
    dollar_risk = total_funds * (KANGAROO_RISK_PER_TRADE_PCT / 100)

    print(f"  Pending intents watched : {len(intents_df)}")
    print(f"  Positions               : {current_position_count} / {KANGAROO_MAX_POSITIONS} occupied, "
          f"{remaining_slots} slot(s) open")
    print(f"  Risk/trade              : ${dollar_risk:,.2f} ({KANGAROO_RISK_PER_TRADE_PCT}% of ${total_funds:,.2f})")
    print(f"  Max position            : ${max_position_value:,.2f} (1/{remaining_slots} of funds)\n")

    today = market_today().date()
    buy_records: list[dict] = []
    triggered = 0

    for _, row in intents_df.iterrows():
        ticker = str(row[SIGNAL_COL_TICKER]).strip().upper()
        intent_id = int(row["id"])
        if not ticker or ticker == "NAN" or ticker in owned:
            continue
        if len(buy_records) >= remaining_slots:
            break

        try:
            trigger = float(row[INTENT_COL_ENTRY_PRICE_PLANNED])
            stop_price = float(row[INTENT_COL_STOP_PRICE])
            target_price = float(row[INTENT_COL_TARGET_PRICE])
        except (TypeError, ValueError):
            continue
        per_share_risk = trigger - stop_price
        if per_share_risk <= 0:
            continue

        snapshot = DEFAULT_PROVIDER.get_intraday_snapshot(ticker)
        if snapshot is None:
            print(f"  {Fore.YELLOW}{ticker:<10}{Style.RESET_ALL} no live intraday data — skipped this run")
            continue

        print(f"  {Fore.CYAN}{ticker:<10}{Style.RESET_ALL} trigger=${trigger:.2f}  "
              f"session_high=${snapshot.high:.2f}", end="  ")

        if snapshot.high < trigger:
            print("not triggered yet")
            continue

        triggered += 1
        shares_by_risk = int(dollar_risk / per_share_risk)
        shares_by_cap = int(max_position_value / trigger)
        shares = min(shares_by_risk, shares_by_cap)
        if shares <= 0:
            print(f"{Fore.YELLOW}triggered, but allocation too small — skipped{Style.RESET_ALL}")
            continue

        cost = shares * trigger
        print(f"{Fore.GREEN}{'[DRY RUN] ' if dry_run else ''}BREAKOUT — BUY {shares} sh = ${cost:,.2f}{Style.RESET_ALL}")

        buy_records.append({
            "intent_id": intent_id,
            "pattern": "kangaroo_tail",
            SIGNAL_COL_TICKER: ticker,
            POSITION_COL_ENTRY_DATE: today,
            POSITION_COL_ENTRY_PRICE: trigger,
            POSITION_COL_SHARES: shares,
            "stop_price": stop_price,
            "target_price": target_price,
        })
        owned.add(ticker)

    print()
    if not buy_records:
        print(f"{Fore.YELLOW}No breakouts triggered this run ({triggered} intent(s) crossed the trigger "
              f"but couldn't be sized/filled).{Style.RESET_ALL}" if triggered else
              f"{Fore.YELLOW}No breakouts triggered this run.{Style.RESET_ALL}")
        log(service, run_id, "completed", bought=0, triggered=triggered)
        return

    if dry_run:
        print(f"{Fore.CYAN}[DRY RUN] Would insert {len(buy_records)} position(s) into database{Style.RESET_ALL}")
    else:
        for rec in buy_records:
            insert_position(
                rec[SIGNAL_COL_TICKER],
                rec[POSITION_COL_ENTRY_DATE].isoformat(),
                rec[POSITION_COL_ENTRY_PRICE],
                rec[POSITION_COL_SHARES],
                pattern=rec.get("pattern"),
                stop_price=rec.get("stop_price"),
                target_price=rec.get("target_price"),
                cash_delta=-(rec[POSITION_COL_SHARES] * rec[POSITION_COL_ENTRY_PRICE]),
            )
            mark_intent_executed(rec["intent_id"], rec[POSITION_COL_ENTRY_PRICE], rec[POSITION_COL_SHARES])
        print(f"{Fore.GREEN}✓ Inserted {len(buy_records)} position(s) into database{Style.RESET_ALL}")

    total_invested = sum(r[POSITION_COL_SHARES] * r[POSITION_COL_ENTRY_PRICE] for r in buy_records)
    remaining = total_funds - total_invested
    print(f"  Tickers bought : {len(buy_records)}")
    print(f"  Total invested : ${total_invested:,.2f}")
    print(f"  Cash remaining : ${remaining:,.2f}\n")

    print(f"{Fore.RED}⚠  VIRTUAL TRANSACTIONS ONLY — not financial advice.{Style.RESET_ALL}\n")
    send_transaction_email(
        buys=buy_records,
        sells=[],
        cash_before=total_funds,
        cash_after=remaining,
        open_positions_count=len(get_open_positions_df()),
        label="Kangaroo Tail",
    )
    log(service, run_id, "completed", bought=len(buy_records), triggered=triggered)


if __name__ == "__main__":
    import argparse

    service = "kangaroo_buy"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "lock_acquired", lock_file=str(lock_path))

    parser = argparse.ArgumentParser(description="Kangaroo Tail Breakout Watcher")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be bought without writing")
    args = parser.parse_args()

    init_db(path=KANGAROO_DB_PATH)
    try:
        run_kangaroo_buy(dry_run=args.dry_run, run_id=run_id)
    finally:
        lock_file.close()
