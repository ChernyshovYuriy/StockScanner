"""
macro_buy.py
============
Virtual entry execution for the macro conviction sleeve — separate DB/capital
from the core sleeve's virtual_buy.py (see config.py MACRO_* and CLAUDE.md).

Unlike momentum_buy.py, this sleeve has NO own screener/pipeline: it reads
the CORE sleeve's own already-confirmed intents (data/trading.db) READ-ONLY,
via a raw duckdb.connect(..., read_only=True) query -- never through
db.py's init_db()/load_pending_intents() against the core DB (that would
mutate db.py's module-global DB_PATH, and load_pending_intents() only
returns PENDING rows, which races against virtual_buy.py's own
mark_intent_executed/mark_intent_skipped calls in the same 9:45 slot).

***THIS MODULE MUST NEVER CALL mark_intent_executed()/mark_intent_skipped()
AGAINST THE CORE trading.db.*** Doing so would make the core sleeve's own
virtual_buy.py silently skip or lose real intents. This sleeve only READS
core intents and only WRITES to its own macro.db.

Regime-gated: takes NO buys at all unless macro_regime.get_macro_regime()
currently reads "risk_on" -- capital preservation first. When it does buy,
candidates are ranked by `rr` (reward:risk) descending -- the only per-
intent quality signal that reaches the intents table today (the screener's
composite_score is dropped by auto_pipeline._read_screener_file() before it
gets there; ranking by composite_score instead would need that plumbing
added, which this module deliberately does not do -- see CLAUDE.md).

No sector cap or gap filter for this sleeve -- a 1-2 position book with a
regime gate already in front of it, kept out of scope for the first live
iteration (same precedent as momentum_buy.py dropping both entirely rather
than disabling via flag).

Usage
-----
  python macro_buy.py
  python macro_buy.py --dry-run
"""

from __future__ import annotations

import sys
import uuid
from datetime import date
from typing import Optional

import duckdb
import pandas as pd
from colorama import Fore, Style, init

from concurrent_utils import acquire_lock
from config import MACRO_DB_PATH, MACRO_INITIAL_CAPITAL, MACRO_MAX_POSITIONS, MACRO_RISK_PER_TRADE_PCT
from market_data import DEFAULT_PROVIDER
from db import DB_PATH as CORE_DB_PATH  # captured at import time -- see module docstring
from db import (
    get_cash,
    get_open_positions_df,
    init_db,
    insert_position,
    set_cash,
)
from log_utils import log
from macro_regime import get_macro_regime
from send_report import send_transaction_email
from schema_keys import (
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
    market_data.DEFAULT_PROVIDER.get_quote()."""
    return DEFAULT_PROVIDER.get_quote(ticker)


def _read_core_intents(as_of_date: date) -> pd.DataFrame:
    """Read-only raw SQL against the CORE sleeve's intents table
    (data/trading.db), unfiltered by intent_status (so this doesn't race
    virtual_buy.py's own status mutations in the same 9:45 slot), filtered
    by signal_date >= as_of_date (the same staleness window virtual_buy.py
    enforces inline), ranked by rr descending -- the only per-intent quality
    signal available (see module docstring).

    Never calls db.init_db(path=CORE_DB_PATH) -- this is a plain read-only
    connection, entirely separate from this module's own db.py writes
    against MACRO_DB_PATH.
    """
    if not CORE_DB_PATH.exists():
        return pd.DataFrame()
    conn = duckdb.connect(str(CORE_DB_PATH), read_only=True)
    try:
        return conn.execute(
            "SELECT id, ticker, signal_date, pattern, entry_price_planned, "
            "stop_price, target_price, rr, intent_status "
            "FROM intents WHERE signal_date >= ? ORDER BY rr DESC NULLS LAST",
            [as_of_date.isoformat()],
        ).df()
    finally:
        conn.close()


def run_macro_buy(dry_run: bool, run_id: Optional[str] = None) -> None:
    service = "macro_buy"
    run_id = run_id or uuid.uuid4().hex
    log(service, run_id, "start", dry_run=dry_run)

    print(f"\n{'=' * 60}")
    print(f"  {Fore.BLUE}🎯  Macro Conviction Sleeve — Buy Runner{Style.RESET_ALL}")
    print(f"{'=' * 60}\n")

    if not dry_run and not is_market_open():
        print(f"{Fore.YELLOW}Market closed — no buys executed.{Style.RESET_ALL}")
        log(service, run_id, "skip_market_closed")
        return

    # ── Regime gate: no buys at all unless the macro backdrop is supportive ──
    regime = get_macro_regime()
    print(
        f"  Macro regime : {regime['label'].upper()} "
        f"(composite={regime['composite']}, votes={regime['votes']})\n"
    )
    if regime["label"] != "risk_on":
        print(f"{Fore.YELLOW}Regime is not risk_on — no buys this run (capital preservation).{Style.RESET_ALL}")
        log(service, run_id, "skip_regime_not_risk_on", composite=regime["composite"])
        return

    today = market_today().date()
    intents_df = _read_core_intents(previous_trading_day(today))
    if intents_df.empty:
        print(f"{Fore.YELLOW}No core-sleeve intents found — nothing to buy.{Style.RESET_ALL}")
        return

    candidate_tickers = intents_df[SIGNAL_COL_TICKER].tolist()
    print(f"  Core-sleeve candidates (by rr): {', '.join(candidate_tickers)}\n")

    owned_tickers = set(get_open_positions_df()["ticker"].str.upper())

    duplicate_seen: set[str] = set()
    actionable: list[dict] = []

    for _, row in intents_df.iterrows():
        ticker = str(row[SIGNAL_COL_TICKER]).strip().upper()
        if not ticker or ticker == "NAN":
            continue

        signal_date = _parse_signal_date(row)
        if signal_date is not None and signal_date < previous_trading_day(today):
            continue  # stale -- never touches the core intent row, just skipped locally

        if ticker in duplicate_seen:
            continue
        duplicate_seen.add(ticker)

        if ticker in owned_tickers:
            continue

        actionable.append({"ticker": ticker, "row": row})

    total_funds = get_cash()
    if total_funds <= 0:
        print(f"{Fore.YELLOW}Available funds is ${total_funds:,.2f} — nothing to buy.{Style.RESET_ALL}")
        return

    print(f"  Total funds  : ${total_funds:,.2f}")

    if not actionable:
        print(f"{Fore.YELLOW}No actionable candidates — nothing to buy.{Style.RESET_ALL}")
        return

    current_position_count = len(owned_tickers)
    remaining_slots = MACRO_MAX_POSITIONS - current_position_count
    if remaining_slots <= 0:
        print(
            f"{Fore.YELLOW}Portfolio full — {current_position_count} of "
            f"{MACRO_MAX_POSITIONS} positions occupied. Nothing to buy.{Style.RESET_ALL}"
        )
        return

    # Already sorted by rr DESC from the SQL query -- take the top
    # `remaining_slots` candidates, same "remaining slots not total" fix
    # already validated in virtual_buy.py/momentum_buy.py.
    actionable = actionable[:remaining_slots]

    max_position_value = total_funds / remaining_slots
    dollar_risk = total_funds * (MACRO_RISK_PER_TRADE_PCT / 100)

    print(f"  Positions    : {current_position_count} / {MACRO_MAX_POSITIONS} occupied, {remaining_slots} slot(s) open")
    print(f"  Risk/trade   : ${dollar_risk:,.2f} ({MACRO_RISK_PER_TRADE_PCT}% of ${total_funds:,.2f})")
    print(f"  Max position : ${max_position_value:,.2f} (1/{remaining_slots} of funds)\n")

    buy_records: list[dict] = []

    for item in actionable:
        ticker = item["ticker"]
        row = item["row"]
        print(f"  {Fore.CYAN}{ticker:<14}{Style.RESET_ALL}", end=" ", flush=True)

        price = fetch_latest_price(ticker)
        if price is None or price <= 0:
            print(f"{Fore.RED}no price data — skipped{Style.RESET_ALL}")
            continue

        raw_entry = str(row["entry_price_planned"]).replace("$", "").strip()
        raw_stop = str(row["stop_price"]).replace("$", "").strip()

        try:
            planned_entry = float(raw_entry)
            planned_stop = float(raw_stop)
            per_share_risk = planned_entry - planned_stop
        except (ValueError, TypeError):
            per_share_risk = None

        if per_share_risk is None or per_share_risk <= 0:
            print(f"{Fore.YELLOW}invalid/missing stop data — skipped{Style.RESET_ALL}")
            continue

        stop_price = planned_stop
        shares_by_risk = int(dollar_risk / per_share_risk)
        shares_by_cap = int(max_position_value / price)
        shares = min(shares_by_risk, shares_by_cap)

        if shares <= 0:
            print(f"{Fore.YELLOW}price ${price:.2f} too high for available allocation — skipped{Style.RESET_ALL}")
            continue

        cost = shares * price
        print(
            f"price=${price:.2f}  shares={shares}  cost=${cost:,.2f}  "
            f"{Fore.GREEN}{'[DRY RUN] ' if dry_run else ''}BUY{Style.RESET_ALL}"
        )

        buy_records.append({
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
        log(service, run_id, "completed", bought=0)
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
            # NOTE: deliberately no mark_intent_executed() call here -- see
            # module docstring. This intent belongs to the core sleeve; this
            # sleeve's fill is entirely independent of what the core sleeve
            # decides to do with the same candidate.
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
        label="Macro",
    )
    log(service, run_id, "completed", bought=len(buy_records))


if __name__ == "__main__":
    import argparse

    service = "macro_buy"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "lock_acquired", lock_file=str(lock_path))

    parser = argparse.ArgumentParser(description="Macro Conviction Sleeve Buy Runner")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be bought without writing")
    args = parser.parse_args()

    init_db(path=MACRO_DB_PATH)
    if get_cash() <= 0:
        # First-time init for this sleeve's DB -- see CLAUDE.md init pattern.
        # No macro_pipeline.py exists to do this the way momentum_pipeline.py
        # does for the momentum sleeve, so this buy script (the earliest
        # daily touchpoint) seeds it instead.
        set_cash(MACRO_INITIAL_CAPITAL)
        print(f"  Initialised macro conviction sleeve account with ${MACRO_INITIAL_CAPITAL:,.2f}")

    try:
        run_macro_buy(dry_run=args.dry_run, run_id=run_id)
    finally:
        lock_file.close()
