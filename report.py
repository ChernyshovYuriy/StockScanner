"""
report.py
=========
Console report of the live trading database.

Prints, in one place, everything the system has recorded:
  - cash balance
  - open positions (current holdings, marked to live market price)
  - the unified BUY/SELL transaction ledger
  - closed trades with realised P&L
  - pending buy intents
  - a short summary (cash, market value of open positions, P&L)

Usage
-----
  python report.py            # full report
  python report.py --section transactions   # one section only
  python report.py --no-quotes              # skip live yfinance lookups
  python report.py --help

Read-only on the database: opens it through db.py accessors, never writes.
Safe to run while the scheduled services are running. By default it makes
one live yfinance quote call per open position to compute unrealised P&L;
pass --no-quotes to value positions at cost instead.
"""

from __future__ import annotations

import argparse

import pandas as pd

import db
from virtual_buy import fetch_latest_price


def _print_df(title: str, df: pd.DataFrame) -> None:
    """Print a titled, full-width table — or a friendly note when empty."""
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)
    if df is None or df.empty:
        print("  (none)")
        return
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.float_format", lambda x: f"{x:,.2f}",
    ):
        print(df.to_string(index=False))


def _mark_to_market(positions: pd.DataFrame) -> pd.DataFrame:
    """
    Add live-price columns to the open-positions DataFrame.

    Makes one yfinance quote call per ticker via fetch_latest_price().
    Adds: cost, last, market_value, unrealized_pnl, unrealized_pct.
    last/market_value/P&L are NaN for any ticker whose quote failed.
    """
    df = positions.copy()
    df["cost"] = df["entry_price"] * df["shares"]
    if df.empty:
        df["last"] = []
        df["market_value"] = []
        df["unrealized_pnl"] = []
        df["unrealized_pct"] = []
        return df

    df["last"] = df["ticker"].apply(fetch_latest_price)
    df["market_value"] = df["last"] * df["shares"]
    df["unrealized_pnl"] = df["market_value"] - df["cost"]
    df["unrealized_pct"] = df["unrealized_pnl"] / df["cost"] * 100.0
    return df


def _print_summary(
    cash: float,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    with_quotes: bool,
) -> None:
    """Print a compact bottom-line summary."""
    cost_basis = float(positions["cost"].sum()) if not positions.empty else 0.0
    realised = float(trades["pnl_dollars"].sum()) if not trades.empty else 0.0

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print("=" * 72)
    print(f"  Cash balance .............. {cash:>14,.2f}")
    if with_quotes and not positions.empty:
        market_value = float(positions["market_value"].sum(skipna=True))
        unrealised = float(positions["unrealized_pnl"].sum(skipna=True))
        print(f"  Open positions (at cost) .. {cost_basis:>14,.2f}")
        print(f"  Open positions (market) ... {market_value:>14,.2f}")
        print(f"  Account value (market) .... {cash + market_value:>14,.2f}")
        print(f"  Unrealised P&L ............ {unrealised:>14,.2f}")
    else:
        print(f"  Open positions (at cost) .. {cost_basis:>14,.2f}")
        print(f"  Account value (at cost) ... {cash + cost_basis:>14,.2f}")
    print(f"  Open positions ............ {len(positions):>14d}")
    print(f"  Closed trades ............. {len(trades):>14d}")
    print(f"  Realised P&L .............. {realised:>14,.2f}")


SECTIONS = ("summary", "positions", "transactions", "trades", "intents")


def main() -> None:
    parser = argparse.ArgumentParser(description="Console report of the live trading database.")
    parser.add_argument(
        "--section",
        choices=SECTIONS,
        help="Print only one section (default: all).",
    )
    parser.add_argument(
        "--no-quotes",
        action="store_true",
        help="Skip live yfinance lookups; value open positions at cost.",
    )
    args = parser.parse_args()

    db.init_db()

    cash = db.get_cash()
    positions = _mark_to_market(db.get_open_positions_df()) if not args.no_quotes \
        else db.get_open_positions_df().assign(cost=lambda d: d["entry_price"] * d["shares"])
    transactions = db.get_transactions()
    trades = db.get_all_trades()
    intents = db.load_pending_intents()

    with_quotes = not args.no_quotes
    want = args.section
    if want in (None, "summary"):
        _print_summary(cash, positions, trades, with_quotes)
    if want in (None, "positions"):
        _print_df("OPEN POSITIONS", positions)
    if want in (None, "transactions"):
        _print_df("TRANSACTIONS (buy/sell ledger)", transactions)
    if want in (None, "trades"):
        _print_df("CLOSED TRADES", trades)
    if want in (None, "intents"):
        _print_df("PENDING INTENTS", intents)

    print()


if __name__ == "__main__":
    main()
