"""
momentum_buy.py
================
Virtual entry execution for the momentum sleeve — separate DB/capital from
the core sleeve's virtual_buy.py (see config.py MOMENTUM_* and CLAUDE.md).
Structurally mirrors virtual_buy.py, forked rather than parameterised: the
core script reads MAX_POSITIONS / RISK_PER_TRADE_PCT from config.py at
module level (not injectable), and the "services stay independent" rule
already established for EDGAR applies here too.

No sector cap or gap filter for this sleeve — small 5-position book, kept
out of scope for the first live iteration (see build plan).

Usage
-----
  python momentum_buy.py
  python momentum_buy.py --top 3
  python momentum_buy.py --dry-run
"""

from __future__ import annotations

import sys
import uuid
from typing import Optional

import pandas as pd
from colorama import Fore, Style, init

from concurrent_utils import acquire_lock
from config import MOMENTUM_DB_PATH, MOMENTUM_MAX_POSITIONS, MOMENTUM_RISK_PER_TRADE_PCT
from market_data import DEFAULT_PROVIDER
from db import (
    get_cash,
    get_open_positions_df,
    init_db,
    insert_position,
    load_pending_intents,
    mark_intent_executed,
    mark_intent_skipped,
)
from log_utils import log
from send_report import send_transaction_email
from schema_keys import (
    INTENT_COL_ENTRY_PRICE_PLANNED,
    INTENT_COL_STOP_PRICE,
    POSITION_COL_ENTRY_DATE,
    POSITION_COL_ENTRY_PRICE,
    POSITION_COL_SHARES,
    SIGNAL_COL_PATTERN,
    SIGNAL_COL_TICKER,
)
from time_utils import is_market_open, market_today, previous_trading_day

init(autoreset=True)


def _parse_signal_date(row: pd.Series):
    raw = str(row.get("signal_date", "") or "").strip()
    try:
        return pd.Timestamp(raw).date()
    except (ValueError, TypeError):
        return None


def fetch_latest_price(ticker: str) -> Optional[float]:
    """Same strategy as virtual_buy.py — see its docstring. Delegates to
    market_data.DEFAULT_PROVIDER.get_quote(), the single place this fetch
    logic now lives (previously a byte-identical duplicate of
    virtual_buy.fetch_latest_price)."""
    return DEFAULT_PROVIDER.get_quote(ticker)


def run_momentum_buy(top_n: Optional[int], dry_run: bool, run_id: Optional[str] = None) -> None:
    service = "momentum_buy"
    run_id = run_id or uuid.uuid4().hex
    log(service, run_id, "start", dry_run=dry_run)

    print(f"\n{'=' * 60}")
    print(f"  {Fore.YELLOW}🚀  Momentum Buy Runner{Style.RESET_ALL}")
    print(f"{'=' * 60}\n")

    if not dry_run and not is_market_open():
        print(f"{Fore.YELLOW}Market closed — no buys executed.{Style.RESET_ALL}")
        log(service, run_id, "skip_market_closed")
        return

    intents_df = load_pending_intents()
    if intents_df.empty:
        print(f"{Fore.YELLOW}No pending intents in queue — nothing to buy.{Style.RESET_ALL}")
        return

    if top_n is not None and top_n > 0:
        intents_df = intents_df.head(top_n).copy()

    pending_tickers = intents_df[SIGNAL_COL_TICKER].tolist()
    print(f"  Pending intents found: {', '.join(pending_tickers)}\n")

    owned_tickers = set(get_open_positions_df()["ticker"].str.upper())

    duplicate_seen: set[str] = set()
    run_seen: set[str] = set()
    actionable: list[dict] = []
    skipped_count = 0

    for _, row in intents_df.iterrows():
        ticker = str(row[SIGNAL_COL_TICKER]).strip().upper()
        intent_id = int(row["id"])

        if not ticker or ticker == "NAN":
            if not dry_run:
                mark_intent_skipped(intent_id, "invalid_ticker")
            skipped_count += 1
            continue

        signal_date = _parse_signal_date(row)
        if signal_date is not None and signal_date < previous_trading_day(market_today().date()):
            if not dry_run:
                mark_intent_skipped(intent_id, "stale_intent")
            skipped_count += 1
            continue

        if ticker in duplicate_seen:
            if not dry_run:
                mark_intent_skipped(intent_id, "duplicate_pending")
            skipped_count += 1
            continue
        duplicate_seen.add(ticker)

        if ticker in owned_tickers:
            if not dry_run:
                mark_intent_skipped(intent_id, "already_owned")
            skipped_count += 1
            continue

        if ticker in run_seen:
            if not dry_run:
                mark_intent_skipped(intent_id, "duplicate_run")
            skipped_count += 1
            continue

        run_seen.add(ticker)
        actionable.append({"intent_id": intent_id, "ticker": ticker, "row": row})

    total_funds = get_cash()
    if total_funds <= 0:
        print(f"{Fore.YELLOW}Available funds is ${total_funds:,.2f} — nothing to buy.{Style.RESET_ALL}")
        return

    print(f"  Total funds  : ${total_funds:,.2f}")

    if not actionable:
        print(f"{Fore.YELLOW}No actionable pending intents — nothing to buy.{Style.RESET_ALL}")
        return

    current_position_count = len(owned_tickers)
    remaining_slots = MOMENTUM_MAX_POSITIONS - current_position_count
    if remaining_slots <= 0:
        print(
            f"{Fore.YELLOW}Portfolio full — {current_position_count} of "
            f"{MOMENTUM_MAX_POSITIONS} positions occupied. Nothing to buy.{Style.RESET_ALL}"
        )
        return

    actionable = actionable[:remaining_slots]

    # Divide cash by REMAINING slots, not MOMENTUM_MAX_POSITIONS — same fix
    # already validated and applied to the core sleeve's virtual_buy.py
    # (walk-forward 2026-07): dividing by the total leaves an ever-shrinking
    # fraction of capital permanently undeployed as the book fills.
    max_position_value = total_funds / remaining_slots
    dollar_risk = total_funds * (MOMENTUM_RISK_PER_TRADE_PCT / 100)

    print(f"  Positions    : {current_position_count} / {MOMENTUM_MAX_POSITIONS} occupied, {remaining_slots} slot(s) open")
    print(f"  Risk/trade   : ${dollar_risk:,.2f} ({MOMENTUM_RISK_PER_TRADE_PCT}% of ${total_funds:,.2f})")
    print(f"  Max position : ${max_position_value:,.2f} (1/{remaining_slots} of funds)\n")

    today = market_today().date()
    buy_records: list[dict] = []

    for item in actionable:
        ticker = item["ticker"]
        intent_id = item["intent_id"]
        row = item["row"]
        print(f"  {Fore.CYAN}{ticker:<14}{Style.RESET_ALL}", end=" ", flush=True)

        price = fetch_latest_price(ticker)
        if price is None or price <= 0:
            print(f"{Fore.RED}no price data — skipped{Style.RESET_ALL}")
            if not dry_run:
                mark_intent_skipped(intent_id, "no_price_data")
            skipped_count += 1
            continue

        raw_entry = str(row[INTENT_COL_ENTRY_PRICE_PLANNED]).replace("$", "").strip()
        raw_stop = str(row[INTENT_COL_STOP_PRICE]).replace("$", "").strip()

        stop_price: Optional[float] = None
        try:
            planned_entry = float(raw_entry)
            planned_stop = float(raw_stop)
            per_share_risk = planned_entry - planned_stop
        except (ValueError, TypeError):
            per_share_risk = None

        if per_share_risk is None or per_share_risk <= 0:
            # Stop missing/invalid (NaN, or at/above entry) — the position
            # can't be risk-sized. Buying it anyway at full equal-split
            # allocation used to silently promote a broken setup to the
            # largest position size in the book with no stop_price persisted
            # at all (same bug fixed in virtual_buy.py). Skip the candidate
            # instead; a lower-priority intent in the same run can still
            # fill the slot.
            print(f"{Fore.YELLOW}invalid/missing stop data — skipped{Style.RESET_ALL}")
            if not dry_run:
                mark_intent_skipped(intent_id, "invalid_stop_data")
            skipped_count += 1
            continue

        stop_price = planned_stop
        shares_by_risk = int(dollar_risk / per_share_risk)
        shares_by_cap = int(max_position_value / price)
        shares = min(shares_by_risk, shares_by_cap)

        if shares <= 0:
            print(f"{Fore.YELLOW}price ${price:.2f} too high for available allocation — skipped{Style.RESET_ALL}")
            if not dry_run:
                mark_intent_skipped(intent_id, "allocation_too_small")
            skipped_count += 1
            continue

        cost = shares * price
        print(
            f"price=${price:.2f}  shares={shares}  cost=${cost:,.2f}  "
            f"{Fore.GREEN}{'[DRY RUN] ' if dry_run else ''}BUY{Style.RESET_ALL}"
        )

        buy_records.append({
            "intent_id": intent_id,
            "pattern": str(row.get(SIGNAL_COL_PATTERN, "")) or None,
            SIGNAL_COL_TICKER: ticker,
            POSITION_COL_ENTRY_DATE: today,
            POSITION_COL_ENTRY_PRICE: price,
            POSITION_COL_SHARES: shares,
            "stop_price": stop_price,
        })

    print()
    if not buy_records:
        print(f"{Fore.YELLOW}No valid buy records generated — nothing written.{Style.RESET_ALL}")
        log(service, run_id, "completed", bought=0, skipped=skipped_count)
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
                cash_delta=-(rec[POSITION_COL_SHARES] * rec[POSITION_COL_ENTRY_PRICE]),
            )
            mark_intent_executed(rec["intent_id"], rec[POSITION_COL_ENTRY_PRICE], rec[POSITION_COL_SHARES])
        print(f"{Fore.GREEN}✓ Inserted {len(buy_records)} position(s) into database{Style.RESET_ALL}")

    print(f"\n{'─' * 60}")
    total_invested = sum(r[POSITION_COL_SHARES] * r[POSITION_COL_ENTRY_PRICE] for r in buy_records)
    remaining = total_funds - total_invested

    if dry_run:
        print(f"{Fore.CYAN}[DRY RUN] Would update cash → ${remaining:,.2f}{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✓ Cash updated → ${remaining:,.2f} remaining{Style.RESET_ALL}")

    print(f"  Tickers bought : {len(buy_records)}")
    print(f"  Total invested : ${total_invested:,.2f}")
    print(f"  Cash remaining : ${remaining:,.2f}")
    print(f"  {'─' * 56}")
    for rec in buy_records:
        cost = rec[POSITION_COL_SHARES] * rec[POSITION_COL_ENTRY_PRICE]
        print(
            f"  {rec[SIGNAL_COL_TICKER]:<14} {rec[POSITION_COL_SHARES]:>6} shares @ "
            f"${rec[POSITION_COL_ENTRY_PRICE]:.4f}  =  ${cost:,.2f}"
        )
    print(f"{'─' * 60}\n")

    print(f"{Fore.RED}⚠  VIRTUAL TRANSACTIONS ONLY — not financial advice.{Style.RESET_ALL}\n")
    send_transaction_email(
        buys=buy_records,
        sells=[],
        cash_before=total_funds,
        cash_after=remaining,
        open_positions_count=len(get_open_positions_df()),
        label="Momentum",
    )
    log(service, run_id, "completed", bought=len(buy_records), skipped=skipped_count)


if __name__ == "__main__":
    import argparse

    service = "momentum_buy"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "lock_acquired", lock_file=str(lock_path))

    parser = argparse.ArgumentParser(description="Momentum Sleeve Buy Runner")
    parser.add_argument("--top", type=int, default=10, help="Max pending intents to process (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be bought without writing")
    args = parser.parse_args()

    init_db(path=MOMENTUM_DB_PATH)
    try:
        run_momentum_buy(top_n=args.top, dry_run=args.dry_run, run_id=run_id)
    finally:
        lock_file.close()
