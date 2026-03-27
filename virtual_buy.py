"""
virtual_buy.py
==============
Executes virtual buy transactions for tickers found in a signals file.

Reads available funds from a plain-text funds file, fetches the current
price for each ticker via Yahoo Finance (same pattern as the rest of this
project), computes how many whole shares can be purchased, and appends a
record to a positions CSV file.

If the signals file is empty or contains no valid tickers — does nothing.
If the funds file is missing or contains 0 — does nothing.

Usage
-----
  python virtual_buy.py \\
      --signals  screener_outputs/20260312T1600.csv \\
      --funds    funds.txt \\
      --positions positions/positions.csv

  python virtual_buy.py \\
      --signals  screener_outputs/20260312T1600.csv \\
      --funds    funds.txt \\
      --positions positions/positions.csv \\
      --top      3          # buy only the top-N tickers by score (default: all)

  python virtual_buy.py --help

Arguments
---------
  --signals    Path to the structured entry-intent CSV emitted by the
               pipeline (requires intent_status and related intent columns).
  --funds      Path to a plain-text file whose first non-blank line is the
               total available capital in CAD (e.g. "50000" or "50000.00").
               Funds are split equally across all purchased tickers.
  --positions  Path to the output positions CSV.  The file is APPENDED to
               (never overwritten) so it accumulates across multiple runs.
               Created with a header row if it does not yet exist.
  --top        (optional) Limit processing to the first N pending intent rows.
               Default: process all pending intents.
  --dry-run    Print what would be bought without writing anything.

Output CSV columns
------------------
  ticker, entry_date, entry_price, shares

  This format is intentionally compatible with position_monitor.py.
"""

from __future__ import annotations

import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from colorama import Fore, Style, init

from concurrent_utils import acquire_lock
from config import CANDIDATES_QUEUE_PATH, FUNDS_PATH, OWN_PATH, MAX_POSITIONS
from log_utils import log
from schema_keys import INTENT_COL_EXECUTED_PRICE, INTENT_COL_EXECUTED_SHARES, INTENT_COL_PROCESSED_AT, INTENT_COL_REASON, \
    INTENT_COL_STATUS, INTENT_REQUIRED_COLS, POSITION_COL_ENTRY_DATE, POSITION_COL_ENTRY_PRICE, POSITION_COL_SHARES, \
    POSITIONS_COLS, SIGNAL_COL_TICKER
from time_utils import market_today

init(autoreset=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# FUNDS FILE
# ─────────────────────────────────────────────────────────────────────────────

def read_funds(path: Path) -> float:
    """
    Read available capital from a plain-text file.
    The first non-blank, non-comment line is parsed as a float.
    Returns 0.0 if the file is missing, empty, or unparseable.
    """
    if not path.exists():
        print(f"{Fore.YELLOW}Funds file not found: {path}{Style.RESET_ALL}")
        return 0.0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            value = float(line.replace(",", "").replace("$", ""))
            return value
        except ValueError:
            print(
                f"{Fore.YELLOW}Could not parse funds value '{line}' "
                f"in {path}{Style.RESET_ALL}"
            )
            return 0.0

    print(f"{Fore.YELLOW}Funds file is empty: {path}{Style.RESET_ALL}")
    return 0.0


def write_funds(path: Path, amount: float) -> None:
    """
    Overwrite the funds file with the updated remaining balance.
    Preserves any comment lines (starting with #) that were in the original.
    """
    comments: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#"):
                comments.append(line)

    lines = comments + [f"{amount:.2f}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# SIGNALS FILE
# ─────────────────────────────────────────────────────────────────────────────

def load_intents_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=INTENT_REQUIRED_COLS)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=INTENT_REQUIRED_COLS)


def persist_intent_updates(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def validate_intents_csv(df: pd.DataFrame, path: Path) -> bool:
    missing = [col for col in INTENT_REQUIRED_COLS if col not in df.columns]
    if missing:
        print(
            f"{Fore.RED}Structured intents CSV missing required columns: {missing}. "
            f"Found: {list(df.columns)} in {path}{Style.RESET_ALL}"
        )
        return False
    return True


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
    """
    t = yf.Ticker(ticker)

    # ── Primary: fast_info delayed quote ────────────────────────────────────
    try:
        price = t.fast_info["last_price"]
        if price is not None and float(price) > 0:
            return float(price)
    except Exception:
        pass

    # ── Fallback: latest 1-minute intraday bar ───────────────────────────────
    try:
        df = yf.download(
            tickers=ticker,
            period="1d",
            interval="1m",
            auto_adjust=True,
            progress=False,
        )
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            if not close.empty:
                return float(close.iloc[-1])
    except Exception as e:
        print(f"  {Fore.RED}{ticker}: fallback download error — {e}{Style.RESET_ALL}")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS CSV  — append-only, position_monitor.py compatible
# ─────────────────────────────────────────────────────────────────────────────

def load_positions(path: Path) -> pd.DataFrame:
    """Load existing positions CSV, or return an empty DataFrame."""
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=POSITIONS_COLS)


def append_position(path: Path, ticker: str, entry_date: date,
                    entry_price: float, shares: float) -> None:
    """Append a single new position row to the CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([{
        SIGNAL_COL_TICKER: ticker,
        POSITION_COL_ENTRY_DATE: entry_date.isoformat(),
        POSITION_COL_ENTRY_PRICE: round(entry_price, 4),
        POSITION_COL_SHARES: shares,
    }])

    write_header = not path.exists() or path.stat().st_size == 0
    new_row.to_csv(path, mode="a", index=False, header=write_header)


# ─────────────────────────────────────────────────────────────────────────────
# CORE BUY LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def run_virtual_buy(
        signals_path: Path,
        funds_path: Path,
        positions_path: Path,
        top_n: Optional[int],
        dry_run: bool,
        run_id: Optional[str] = None,
) -> None:
    service = "virtual_buy"
    run_id = run_id or uuid.uuid4().hex
    log(service, run_id, "start", signals_path=str(signals_path), funds_path=str(funds_path),
        positions_path=str(positions_path), dry_run=dry_run)

    print(f"\n{'=' * 60}")
    print(f"  {Fore.YELLOW}💸  Virtual Buy Runner{Style.RESET_ALL}")
    print(f"{'=' * 60}\n")

    # ── 1. Read intents ──────────────────────────────────────────────────────
    intents_df = load_intents_csv(signals_path)
    if intents_df.empty:
        print(f"{Fore.YELLOW}No intents found in signals file — nothing to buy.{Style.RESET_ALL}")
        return
    if not validate_intents_csv(intents_df, signals_path):
        return

    intents_df = intents_df.copy()
    intents_df[SIGNAL_COL_TICKER] = intents_df[SIGNAL_COL_TICKER].astype(str).str.strip().str.upper()
    intents_df[INTENT_COL_STATUS] = intents_df[INTENT_COL_STATUS].astype(str).str.strip().str.lower()
    intents_df[INTENT_COL_REASON] = intents_df[INTENT_COL_REASON].fillna("").astype(str)
    for optional_col in (INTENT_COL_EXECUTED_PRICE, INTENT_COL_EXECUTED_SHARES, INTENT_COL_PROCESSED_AT):
        if optional_col not in intents_df.columns:
            intents_df[optional_col] = ""

    pending_df = intents_df[intents_df[INTENT_COL_STATUS] == "pending"].copy()
    if top_n is not None and top_n > 0:
        pending_df = pending_df.head(top_n)
    if pending_df.empty:
        print(f"{Fore.YELLOW}No pending intents in signals file — nothing to buy.{Style.RESET_ALL}")
        return

    pending_indices = pending_df.index.tolist()
    pending_tickers = pending_df[SIGNAL_COL_TICKER].tolist()

    print(f"  Signals file : {signals_path}")
    print(f"  Pending intents found: {', '.join(pending_tickers)}\n")

    positions_df = load_positions(positions_path)
    owned_tickers = set()
    if not positions_df.empty and SIGNAL_COL_TICKER in positions_df.columns:
        owned_tickers = {
            t.strip().upper() for t in positions_df[SIGNAL_COL_TICKER].dropna().astype(str).tolist() if t.strip()
        }

    duplicate_seen: set[str] = set()
    run_seen: set[str] = set()
    actionable_indices: list[int] = []
    processed_at = market_today().isoformat()
    skipped_count = 0
    for idx in pending_indices:
        ticker = str(intents_df.at[idx, SIGNAL_COL_TICKER]).strip().upper()
        if not ticker or ticker == "NAN":
            intents_df.at[idx, INTENT_COL_STATUS] = "skipped"
            intents_df.at[idx, INTENT_COL_REASON] = "invalid_ticker"
            intents_df.at[idx, INTENT_COL_PROCESSED_AT] = processed_at
            skipped_count += 1
            continue
        if ticker in duplicate_seen:
            intents_df.at[idx, INTENT_COL_STATUS] = "skipped"
            intents_df.at[idx, INTENT_COL_REASON] = "duplicate_pending"
            intents_df.at[idx, INTENT_COL_PROCESSED_AT] = processed_at
            skipped_count += 1
            continue
        duplicate_seen.add(ticker)

        if ticker in owned_tickers:
            intents_df.at[idx, INTENT_COL_STATUS] = "skipped"
            intents_df.at[idx, INTENT_COL_REASON] = "already_owned"
            intents_df.at[idx, INTENT_COL_PROCESSED_AT] = processed_at
            skipped_count += 1
            continue

        if ticker in run_seen:
            intents_df.at[idx, INTENT_COL_STATUS] = "skipped"
            intents_df.at[idx, INTENT_COL_REASON] = "duplicate_run"
            intents_df.at[idx, INTENT_COL_PROCESSED_AT] = processed_at
            skipped_count += 1
            continue

        run_seen.add(ticker)
        actionable_indices.append(idx)

    # ── 2. Read available funds ──────────────────────────────────────────────
    total_funds = read_funds(funds_path)
    if total_funds <= 0:
        print(
            f"{Fore.YELLOW}Available funds is ${total_funds:,.2f} — nothing to buy.{Style.RESET_ALL}"
        )
        if not dry_run:
            persist_intent_updates(signals_path, intents_df)
        return

    print(f"  Funds file   : {funds_path}")
    print(f"  Total funds  : ${total_funds:,.2f}")

    # Slot - based allocation: reserve capacity for future opportunities.
    # The portfolio is divided into MAX_POSITIONS slots. Only the remaining
    # (unfilled) slots may be used for new buys, so capital is never fully
    # committed when fewer tickers than slots are detected today.
    if not actionable_indices:
        print(f"{Fore.YELLOW}No actionable pending intents — nothing to buy.{Style.RESET_ALL}")
        if not dry_run:
            persist_intent_updates(signals_path, intents_df)
        return

    current_position_count = len(owned_tickers)
    remaining_slots = MAX_POSITIONS - current_position_count
    if remaining_slots <= 0:
        print(
            f"{Fore.YELLOW}Portfolio full — {current_position_count} of "
            f"{MAX_POSITIONS} positions occupied. Nothing to buy.{Style.RESET_ALL}"
        )
        if not dry_run:
            persist_intent_updates(signals_path, intents_df)
        return

    # Cap the number of buys to available slots
    if len(actionable_indices) > remaining_slots:
        actionable_indices = actionable_indices[:remaining_slots]

    allocation_per_ticker = total_funds / remaining_slots
    print(f"  Positions    : {current_position_count} / {MAX_POSITIONS} occupied, {remaining_slots} slot(s) open")
    print(f"  Per ticker   : ${allocation_per_ticker:,.2f}  ({len(actionable_indices)} ticker(s) to buy)\n")
    # ── 3. Fetch prices & compute shares ────────────────────────────────────
    today = market_today().date()
    buy_records: list[dict] = []

    for idx in actionable_indices:
        ticker = str(intents_df.at[idx, SIGNAL_COL_TICKER]).strip().upper()
        print(f"  {Fore.CYAN}{ticker:<14}{Style.RESET_ALL}", end=" ", flush=True)

        price = fetch_latest_price(ticker)
        if price is None or price <= 0:
            print(f"{Fore.RED}no price data — skipped{Style.RESET_ALL}")
            intents_df.at[idx, INTENT_COL_STATUS] = "skipped"
            intents_df.at[idx, INTENT_COL_REASON] = "no_price_data"
            intents_df.at[idx, INTENT_COL_PROCESSED_AT] = processed_at
            continue

        shares = int(allocation_per_ticker / price)  # whole shares only
        if shares <= 0:
            print(
                f"{Fore.YELLOW}price ${price:.2f} exceeds allocation "
                f"${allocation_per_ticker:,.2f} — skipped{Style.RESET_ALL}"
            )
            intents_df.at[idx, INTENT_COL_STATUS] = "skipped"
            intents_df.at[idx, INTENT_COL_REASON] = "allocation_too_small"
            intents_df.at[idx, INTENT_COL_PROCESSED_AT] = processed_at
            continue

        cost = shares * price
        print(
            f"price=${price:.2f}  shares={shares}  "
            f"cost=${cost:,.2f}  "
            f"{Fore.GREEN}{'[DRY RUN] ' if dry_run else ''}BUY{Style.RESET_ALL}"
        )

        buy_records.append({
            "intent_index": idx,
            SIGNAL_COL_TICKER: ticker,
            POSITION_COL_ENTRY_DATE: today,
            POSITION_COL_ENTRY_PRICE: price,
            POSITION_COL_SHARES: shares,
        })

    # ── 4. Write to positions CSV ────────────────────────────────────────────
    print()
    if not buy_records:
        print(f"{Fore.YELLOW}No valid buy records generated — positions file unchanged.{Style.RESET_ALL}")
        if not dry_run:
            persist_intent_updates(signals_path, intents_df)
        return

    if dry_run:
        print(f"{Fore.CYAN}[DRY RUN] Would append {len(buy_records)} record(s) to {positions_path}{Style.RESET_ALL}")
    else:
        for rec in buy_records:
            append_position(
                positions_path,
                rec[SIGNAL_COL_TICKER],
                rec[POSITION_COL_ENTRY_DATE],
                rec[POSITION_COL_ENTRY_PRICE],
                rec[POSITION_COL_SHARES],
            )
            intents_df.at[rec["intent_index"], INTENT_COL_STATUS] = "executed"
            intents_df.at[rec["intent_index"], INTENT_COL_REASON] = ""
            intents_df.at[rec["intent_index"], INTENT_COL_EXECUTED_PRICE] = str(round(rec[POSITION_COL_ENTRY_PRICE], 4))
            intents_df.at[rec["intent_index"], INTENT_COL_EXECUTED_SHARES] = str(rec[POSITION_COL_SHARES])
            intents_df.at[rec["intent_index"], INTENT_COL_PROCESSED_AT] = processed_at
        print(
            f"{Fore.GREEN}✓ Appended {len(buy_records)} record(s) → {positions_path.resolve()}{Style.RESET_ALL}"
        )

    # ── 5. Update funds file & print summary ────────────────────────────────
    print(f"\n{'─' * 60}")
    total_invested = sum(r[POSITION_COL_SHARES] * r[POSITION_COL_ENTRY_PRICE] for r in buy_records)
    remaining = total_funds - total_invested

    if dry_run:
        print(f"{Fore.CYAN}[DRY RUN] Would update {funds_path} → ${remaining:,.2f}{Style.RESET_ALL}")
    else:
        write_funds(funds_path, remaining)
        print(
            f"{Fore.GREEN}✓ Funds updated → {funds_path.resolve()}  "
            f"(${remaining:,.2f} remaining){Style.RESET_ALL}"
        )

    if dry_run:
        print(f"{Fore.CYAN}[DRY RUN] Would update intent statuses in {signals_path}{Style.RESET_ALL}")
    else:
        persist_intent_updates(signals_path, intents_df)
        print(
            f"{Fore.GREEN}✓ Intent queue updated → statuses persisted in "
            f"{signals_path.resolve()}{Style.RESET_ALL}"
        )

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
    log(service, run_id, "completed", bought=len(buy_records), skipped=skipped_count)


if __name__ == "__main__":
    service = "virtual_buy"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "lock_acquired", lock_file=str(lock_path))
    try:
        run_virtual_buy(
            signals_path=Path(CANDIDATES_QUEUE_PATH),
            funds_path=Path(FUNDS_PATH),
            positions_path=Path(OWN_PATH),
            top_n=10,
            dry_run=False,
            run_id=run_id,
        )
    finally:
        lock_file.close()
