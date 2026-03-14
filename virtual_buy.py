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
  --signals    Path to the screener/pipeline output CSV (must have a
               'ticker' or 'Ticker' column; optionally 'composite_score').
  --funds      Path to a plain-text file whose first non-blank line is the
               total available capital in CAD (e.g. "50000" or "50000.00").
               Funds are split equally across all purchased tickers.
  --positions  Path to the output positions CSV.  The file is APPENDED to
               (never overwritten) so it accumulates across multiple runs.
               Created with a header row if it does not yet exist.
  --top        (optional) Limit buys to the top-N tickers by score column.
               Default: buy all tickers in the signals file.
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
from config import CANDIDATES_QUEUE_PATH, FUNDS_PATH, OWN_PATH
from log_utils import log
from time_utils import market_today

init(autoreset=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

POSITIONS_COLS = ["ticker", "entry_date", "entry_price", "shares"]


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

def read_tickers(path: Path, top_n: Optional[int]) -> list[str]:
    """
    Read tickers from a signals file.  Two formats are supported:

    1. Plain text  — one ticker per line (blank lines and # comments ignored).
       Example:
           RY.TO
           TD.TO
           # skip this one
           BNS.TO

    2. CSV  — must contain a 'Ticker', 'ticker', 'symbol', or 'Symbol' column.
       If a 'composite_score' column is present, rows are sorted descending
       before the top-N limit is applied.

    The format is detected automatically: if the first non-blank, non-comment
    line contains a comma the file is treated as CSV, otherwise as plain text.
    """
    if not path.exists():
        print(f"{Fore.RED}Signals file not found: {path}{Style.RESET_ALL}")
        return []

    raw_text = path.read_text(encoding="utf-8")

    # ── Detect format ────────────────────────────────────────────────────────
    is_csv = False
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            is_csv = "," in stripped
            break

    # ── Plain-text path ──────────────────────────────────────────────────────
    if not is_csv:
        tickers = []
        for line in raw_text.splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                tickers.append(t.upper())
        if top_n is not None and top_n > 0:
            tickers = tickers[:top_n]
        return tickers

    # ── CSV path ─────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"{Fore.RED}Could not read signals file: {e}{Style.RESET_ALL}")
        return []

    ticker_col: Optional[str] = None
    for candidate in ("Ticker", "ticker", "symbol", "Symbol", "TICKER"):
        if candidate in df.columns:
            ticker_col = candidate
            break

    if ticker_col is None:
        print(
            f"{Fore.YELLOW}No ticker column found in {path}. "
            f"Columns: {list(df.columns)}{Style.RESET_ALL}"
        )
        return []

    score_col: Optional[str] = None
    for candidate in ("composite_score", "score", "Score"):
        if candidate in df.columns:
            score_col = candidate
            break

    if score_col:
        df = df.sort_values(score_col, ascending=False)

    tickers = (
        df[ticker_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )
    tickers = [t for t in tickers if t and t != "NAN"]

    if top_n is not None and top_n > 0:
        tickers = tickers[:top_n]

    return tickers


def remove_tickers_from_queue(path: Path, bought_tickers: list[str]) -> int:
    """
    Remove bought tickers from a plain-text queue file (one ticker per line).

    Returns the number of queue rows removed.
    Notes:
      - Matching is case-insensitive.
      - Blank lines and comment lines (#...) are preserved.
      - If the queue file is missing, no-op.
    """
    if not path.exists() or not bought_tickers:
        return 0

    bought = {t.strip().upper() for t in bought_tickers if t and t.strip()}
    if not bought:
        return 0

    lines = path.read_text(encoding="utf-8").splitlines()
    kept_lines: list[str] = []
    removed = 0

    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            kept_lines.append(raw)
            continue

        if stripped.upper() in bought:
            removed += 1
            continue

        kept_lines.append(raw)

    path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    return removed


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
        "ticker": ticker,
        "entry_date": entry_date.isoformat(),
        "entry_price": round(entry_price, 4),
        "shares": shares,
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

    # ── 1. Read tickers ──────────────────────────────────────────────────────
    tickers = read_tickers(signals_path, top_n)
    if not tickers:
        print(f"{Fore.YELLOW}No tickers found in signals file — nothing to buy.{Style.RESET_ALL}")
        return

    print(f"  Signals file : {signals_path}")
    print(f"  Tickers found: {', '.join(tickers)}\n")

    # ── 2. Read available funds ──────────────────────────────────────────────
    total_funds = read_funds(funds_path)
    if total_funds <= 0:
        print(
            f"{Fore.YELLOW}Available funds is ${total_funds:,.2f} — nothing to buy.{Style.RESET_ALL}"
        )
        return

    print(f"  Funds file   : {funds_path}")
    print(f"  Total funds  : ${total_funds:,.2f}")

    # Equal allocation across all tickers
    allocation_per_ticker = total_funds / len(tickers)
    print(f"  Per ticker   : ${allocation_per_ticker:,.2f}  ({len(tickers)} ticker(s))\n")

    # ── 3. Fetch prices & compute shares ────────────────────────────────────
    today = market_today().date()
    buy_records: list[dict] = []

    for ticker in tickers:
        print(f"  {Fore.CYAN}{ticker:<14}{Style.RESET_ALL}", end=" ", flush=True)

        price = fetch_latest_price(ticker)
        if price is None or price <= 0:
            print(f"{Fore.RED}no price data — skipped{Style.RESET_ALL}")
            continue

        shares = int(allocation_per_ticker / price)  # whole shares only
        if shares <= 0:
            print(
                f"{Fore.YELLOW}price ${price:.2f} exceeds allocation "
                f"${allocation_per_ticker:,.2f} — skipped{Style.RESET_ALL}"
            )
            continue

        cost = shares * price
        print(
            f"price=${price:.2f}  shares={shares}  "
            f"cost=${cost:,.2f}  "
            f"{Fore.GREEN}{'[DRY RUN] ' if dry_run else ''}BUY{Style.RESET_ALL}"
        )

        buy_records.append({
            "ticker": ticker,
            "entry_date": today,
            "entry_price": price,
            "shares": shares,
        })

    # ── 4. Write to positions CSV ────────────────────────────────────────────
    print()
    if not buy_records:
        print(f"{Fore.YELLOW}No valid buy records generated — positions file unchanged.{Style.RESET_ALL}")
        return

    if dry_run:
        print(f"{Fore.CYAN}[DRY RUN] Would append {len(buy_records)} record(s) to {positions_path}{Style.RESET_ALL}")
    else:
        for rec in buy_records:
            append_position(
                positions_path,
                rec["ticker"],
                rec["entry_date"],
                rec["entry_price"],
                rec["shares"],
            )
        print(
            f"{Fore.GREEN}✓ Appended {len(buy_records)} record(s) → {positions_path.resolve()}{Style.RESET_ALL}"
        )

    # ── 5. Update funds file & print summary ────────────────────────────────
    print(f"\n{'─' * 60}")
    total_invested = sum(r["shares"] * r["entry_price"] for r in buy_records)
    remaining = total_funds - total_invested

    if dry_run:
        print(f"{Fore.CYAN}[DRY RUN] Would update {funds_path} → ${remaining:,.2f}{Style.RESET_ALL}")
    else:
        write_funds(funds_path, remaining)
        print(
            f"{Fore.GREEN}✓ Funds updated → {funds_path.resolve()}  "
            f"(${remaining:,.2f} remaining){Style.RESET_ALL}"
        )

    removed = remove_tickers_from_queue(
        signals_path,
        [rec["ticker"] for rec in buy_records],
    )
    if removed:
        print(
            f"{Fore.GREEN}✓ Queue updated → removed {removed} bought ticker(s) "
            f"from {signals_path.resolve()}{Style.RESET_ALL}"
        )

    print(f"  Tickers bought : {len(buy_records)}")
    print(f"  Total invested : ${total_invested:,.2f}")
    print(f"  Cash remaining : ${remaining:,.2f}")
    print(f"  {'─' * 56}")
    for rec in buy_records:
        cost = rec["shares"] * rec["entry_price"]
        print(
            f"  {rec['ticker']:<14} {rec['shares']:>6} shares @ "
            f"${rec['entry_price']:.4f}  =  ${cost:,.2f}"
        )
    print(f"{'─' * 60}\n")

    print(f"{Fore.RED}⚠  VIRTUAL TRANSACTIONS ONLY — not financial advice.{Style.RESET_ALL}\n")
    log(service, run_id, "completed", bought=len(buy_records))


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
