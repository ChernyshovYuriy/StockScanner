"""
virtual_buy.py
==============
Executes virtual buy transactions for tickers found in the intent queue.

Reads available cash and pending intents from the database, fetches the
current price for each ticker via Yahoo Finance, computes how many whole
shares can be purchased, and writes new positions to the database.

If the intent queue is empty or contains no valid tickers — does nothing.
If cash balance is 0 — does nothing.

Usage
-----
  python virtual_buy.py
  python virtual_buy.py --top 3   # buy only the top-N pending intents
  python virtual_buy.py --dry-run
  python virtual_buy.py --help

Arguments
---------
  --top      (optional) Limit processing to the first N pending intent rows.
             Default: process all pending intents.
  --dry-run  Print what would be bought without writing anything.
"""

from __future__ import annotations

import sys
import uuid
from typing import Optional

import pandas as pd
from colorama import Fore, Style, init

from concurrent_utils import acquire_lock
from config import MAX_POSITIONS, MAX_POSITIONS_PER_SECTOR, RISK_PER_TRADE_PCT, GAP_FILTER_PCT
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
from sector_lookup import get_sector
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
    """Return the intent's signal_date as a date, or None if missing/unparseable."""
    raw = str(row.get("signal_date", "") or "").strip()
    try:
        return pd.Timestamp(raw).date()
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PRICE FETCH  — latest available price (~15-min delayed quote from Yahoo)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_latest_price(ticker: str) -> Optional[float]:
    """
    Fetch the latest available market price for a ticker via Yahoo Finance.

    Strategy (in order of preference):
      1. yf.Ticker.fast_info["last_price"]  — fastest, returns the most recent
         delayed quote (~15 min) directly without downloading OHLCV bars.
      2. Fallback: download 1-minute bars for the last 1 trading day and take
         the last bar's Close — useful outside regular hours when fast_info
         may return None.

    This intentionally does NOT use daily bars so the price reflects the
    current session, not yesterday's close.

    Delegates to market_data.DEFAULT_PROVIDER.get_quote() — the single place
    this fetch logic now lives (see market_data.py).
    """
    return DEFAULT_PROVIDER.get_quote(ticker)


# ─────────────────────────────────────────────────────────────────────────────
# CORE BUY LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def run_virtual_buy(
        top_n: Optional[int],
        dry_run: bool,
        run_id: Optional[str] = None,
) -> None:
    service = "virtual_buy"
    run_id = run_id or uuid.uuid4().hex
    log(service, run_id, "start", dry_run=dry_run)

    print(f"\n{'=' * 60}")
    print(f"  {Fore.YELLOW}💸  Virtual Buy Runner{Style.RESET_ALL}")
    print(f"{'=' * 60}\n")

    # ── Market-hours guard ───────────────────────────────────────────────────
    # Never fill live buys outside the TSX session — an off-hours trigger (e.g.
    # a systemd Persistent catch-up, a manual run, or a holiday) would size
    # against a stale yfinance quote.  --dry-run is still allowed any time so the
    # runner can be inspected off-hours without writing anything.
    if not dry_run and not is_market_open():
        print(f"{Fore.YELLOW}Market closed — no buys executed.{Style.RESET_ALL}")
        log(service, run_id, "skip_market_closed")
        return

    # ── 1. Read intents ──────────────────────────────────────────────────────
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

        # Expire intents that missed their execution slot: an intent is only
        # valid on the first trading day after its signal date (same rule as
        # the backtest, and the Persistent=false philosophy — a missed slot is
        # skipped, never executed late on a stale setup).
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

    # ── 2. Read available funds ──────────────────────────────────────────────
    total_funds = get_cash()
    if total_funds <= 0:
        print(f"{Fore.YELLOW}Available funds is ${total_funds:,.2f} — nothing to buy.{Style.RESET_ALL}")
        return

    print(f"  Total funds  : ${total_funds:,.2f}")

    if not actionable:
        print(f"{Fore.YELLOW}No actionable pending intents — nothing to buy.{Style.RESET_ALL}")
        return

    current_position_count = len(owned_tickers)
    remaining_slots = MAX_POSITIONS - current_position_count
    if remaining_slots <= 0:
        print(
            f"{Fore.YELLOW}Portfolio full — {current_position_count} of "
            f"{MAX_POSITIONS} positions occupied. Nothing to buy.{Style.RESET_ALL}"
        )
        return

    # Sector concentration cap — skip (not defer) a candidate whose sector
    # already holds MAX_POSITIONS_PER_SECTOR open positions, preserving
    # priority order so a lower-priority candidate in another sector fills
    # the slot instead. Mirrors backtest_runner.py's max_per_sector, walk-
    # forward validated 2026-08 (no return cost, significant max-drawdown
    # reduction) — prevents correlated same-sector clusters (e.g. a bank
    # earnings week) from entering the book together. A candidate beyond
    # remaining_slots is left untouched (stays pending), same as before.
    sector_counts: dict[str, int] = {}
    if MAX_POSITIONS_PER_SECTOR is not None:
        for t in owned_tickers:
            sec = get_sector(t)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

    filtered_actionable: list[dict] = []
    for item in actionable:
        if len(filtered_actionable) >= remaining_slots:
            break
        if MAX_POSITIONS_PER_SECTOR is not None:
            sec = get_sector(item["ticker"])
            if sector_counts.get(sec, 0) >= MAX_POSITIONS_PER_SECTOR:
                if not dry_run:
                    mark_intent_skipped(item["intent_id"], f"sector_cap_{sec}")
                skipped_count += 1
                continue
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        filtered_actionable.append(item)
    actionable = filtered_actionable

    if not actionable:
        print(f"{Fore.YELLOW}No actionable pending intents after the sector cap — nothing to buy.{Style.RESET_ALL}")
        return

    # Divide cash by REMAINING slots, not MAX_POSITIONS: dividing by the total
    # made each fill take 1/8 of a shrinking cash pile, so ~(7/8)^8 ≈ 34% of
    # capital could never be deployed even at a full book (walk-forward
    # validated 2026-07: 12/16 OOS windows better, +8.6pts full-period).
    max_position_value = total_funds / remaining_slots
    dollar_risk = total_funds * (RISK_PER_TRADE_PCT / 100)

    print(f"  Positions    : {current_position_count} / {MAX_POSITIONS} occupied, {remaining_slots} slot(s) open")
    print(f"  Risk/trade   : ${dollar_risk:,.2f} ({RISK_PER_TRADE_PCT}% of ${total_funds:,.2f})")
    print(f"  Max position : ${max_position_value:,.2f} (1/{remaining_slots} of funds)\n")

    # ── 3. Fetch prices & compute shares ────────────────────────────────────
    today = market_today().date()
    buy_records: list[dict] = []

    for item in actionable:
        ticker   = item["ticker"]
        intent_id = item["intent_id"]
        row      = item["row"]
        print(f"  {Fore.CYAN}{ticker:<14}{Style.RESET_ALL}", end=" ", flush=True)

        price = fetch_latest_price(ticker)
        if price is None or price <= 0:
            print(f"{Fore.RED}no price data — skipped{Style.RESET_ALL}")
            if not dry_run:
                mark_intent_skipped(intent_id, "no_price_data")
            skipped_count += 1
            continue

        # ── Gap filter: skip if price has moved too far above planned entry ───
        # A gap-up invalidates the R:R — the stop hasn't moved but the entry
        # price has, so the trade no longer has the expected reward profile.
        if GAP_FILTER_PCT is not None:
            try:
                planned_entry_for_gap = float(
                    str(row[INTENT_COL_ENTRY_PRICE_PLANNED]).replace("$", "").strip()
                )
                max_allowed = planned_entry_for_gap * (1 + GAP_FILTER_PCT / 100)
                if price > max_allowed:
                    print(
                        f"{Fore.YELLOW}gap-up skip: open ${price:.2f} > "
                        f"entry ${planned_entry_for_gap:.2f} + {GAP_FILTER_PCT}% "
                        f"(${max_allowed:.2f}){Style.RESET_ALL}"
                    )
                    if not dry_run:
                        mark_intent_skipped(intent_id, f"gap_up_{price:.2f}_vs_{planned_entry_for_gap:.2f}")
                    skipped_count += 1
                    continue
            except (ValueError, TypeError):
                pass  # no planned entry recorded — proceed without gap check

        # ── Risk-based sizing using stop from the candidates queue ────────────
        raw_entry = str(row[INTENT_COL_ENTRY_PRICE_PLANNED]).replace("$", "").strip()
        raw_stop  = str(row[INTENT_COL_STOP_PRICE]).replace("$", "").strip()

        stop_price: Optional[float] = None
        try:
            planned_entry = float(raw_entry)
            planned_stop  = float(raw_stop)
            per_share_risk = planned_entry - planned_stop
        except (ValueError, TypeError):
            per_share_risk = None

        if per_share_risk is None or not (per_share_risk > 0):
            # Stop missing/invalid (NaN, or at/above entry) — the position
            # can't be risk-sized. Buying it anyway at full equal-split
            # allocation used to silently promote a broken setup to the
            # largest position size in the book with no stop_price persisted
            # at all. Skip the candidate instead; a lower-priority intent in
            # the same run can still fill the slot.
            print(f"{Fore.YELLOW}invalid/missing stop data — skipped{Style.RESET_ALL}")
            if not dry_run:
                mark_intent_skipped(intent_id, "invalid_stop_data")
            skipped_count += 1
            continue

        stop_price = planned_stop  # persist so the exit honours the planned stop
        shares_by_risk = int(dollar_risk / per_share_risk)
        # Cap: position value must not exceed max_position_value
        shares_by_cap  = int(max_position_value / price)
        shares = min(shares_by_risk, shares_by_cap)

        if shares <= 0:
            print(
                f"{Fore.YELLOW}price ${price:.2f} too high for available allocation — skipped{Style.RESET_ALL}"
            )
            if not dry_run:
                mark_intent_skipped(intent_id, "allocation_too_small")
            skipped_count += 1
            continue

        cost = shares * price
        print(
            f"price=${price:.2f}  shares={shares}  "
            f"cost=${cost:,.2f}  "
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

    # ── 4. Write to database ─────────────────────────────────────────────────
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

    # ── 5. Update cash & print summary ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    total_invested = sum(r[POSITION_COL_SHARES] * r[POSITION_COL_ENTRY_PRICE] for r in buy_records)
    remaining = total_funds - total_invested

    if dry_run:
        print(f"{Fore.CYAN}[DRY RUN] Would update cash → ${remaining:,.2f}{Style.RESET_ALL}")
    else:
        # Cash was already deducted atomically with each insert_position above.
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
    )
    log(service, run_id, "completed", bought=len(buy_records), skipped=skipped_count)


if __name__ == "__main__":
    import argparse

    service = "virtual_buy"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "lock_acquired", lock_file=str(lock_path))

    parser = argparse.ArgumentParser(description="Virtual Buy Runner")
    parser.add_argument("--top", type=int, default=10, help="Max pending intents to process (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be bought without writing")
    args = parser.parse_args()

    init_db()
    try:
        run_virtual_buy(
            top_n=args.top,
            dry_run=args.dry_run,
            run_id=run_id,
        )
    finally:
        lock_file.close()
