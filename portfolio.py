"""
portfolio.py
============
In-memory portfolio state for the backtest refactor (Phase 3).

Replaces the three on-disk state files used in live mode:
  - funds.txt          → PortfolioState.cash
  - positions.csv      → PortfolioState.open_positions
  - signal_history.csv → passed in/out of the backtest runner separately

Design rules
------------
- No file I/O in this module.  Persistence (live mode) is handled by callers
  that load from disk at startup and save at shutdown.
- All business arithmetic (equal allocation, whole-share sizing, P&L formula)
  is identical to virtual_buy.py and position_monitor.py.  This module does
  NOT reimplement those rules — it is a state container that callers populate
  using the existing rule outputs.
- snapshot() returns a deep copy so the backtest runner can checkpoint and
  restore state per simulated day without aliasing bugs.

Public API
----------
  PortfolioState(initial_cash)
  .buy(ticker, entry_date, price, shares)
  .sell(ticker, sell_date, price)
  .cash                → current cash balance
  .realized_pnl        → cumulative realised P&L across all closed trades
  .open_positions      → dict {ticker: OpenPosition}
  .trade_log           → list of ClosedTrade records
  .snapshot()          → deep copy of the current state
  .to_dict()           → serialisable summary (for reporting)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    """A position that is currently held."""
    ticker:      str
    entry_date:  date
    entry_price: float
    shares:      int       # whole shares only — matches virtual_buy.py rule
    stop_price:  Optional[float] = None  # planned exit stop carried from the buy intent

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.shares

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (current_price / self.entry_price - 1.0) * 100.0


@dataclass
class ClosedTrade:
    """A fully closed position with realised P&L."""
    ticker:      str
    entry_date:  date
    sell_date:   date
    entry_price: float
    sell_price:  float
    shares:      int
    pnl:         float       # (sell_price - entry_price) * shares
    pnl_pct:     float       # (sell_price / entry_price - 1) * 100

    @property
    def holding_days(self) -> int:
        return (self.sell_date - self.entry_date).days


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO STATE
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioState:
    """
    Mutable, in-memory portfolio state.

    Tracks cash, open positions, and the closed-trade log.  No file I/O.

    Parameters
    ----------
    initial_cash : starting capital in CAD
    """

    def __init__(self, initial_cash: float):
        if initial_cash < 0:
            raise ValueError(f"initial_cash must be ≥ 0, got {initial_cash}")
        self._cash:          float                     = float(initial_cash)
        self._realized_pnl:  float                     = 0.0
        self._open:          Dict[str, OpenPosition]   = {}
        self._trade_log:     List[ClosedTrade]         = []

    # ── read-only properties ───────────────────────────────────────────────

    @property
    def cash(self) -> float:
        """Current uninvested cash balance."""
        return self._cash

    @property
    def realized_pnl(self) -> float:
        """Cumulative realised profit/loss across all closed trades."""
        return self._realized_pnl

    @property
    def open_positions(self) -> Dict[str, OpenPosition]:
        """Dict of currently held positions keyed by ticker."""
        return dict(self._open)   # shallow copy — callers must not mutate

    @property
    def trade_log(self) -> List[ClosedTrade]:
        """List of all closed trades, oldest first."""
        return list(self._trade_log)

    # ── mutations ─────────────────────────────────────────────────────────

    def buy(
        self,
        ticker:     str,
        entry_date: date,
        price:      float,
        shares:     int,
        stop_price: Optional[float] = None,
    ) -> None:
        """
        Open a new position.

        Deducts price * shares from cash.  Raises ValueError if cash would go
        negative or if ticker is already held (no averaging-down support yet).

        Parameters
        ----------
        ticker     : Yahoo Finance symbol
        entry_date : trade date (date of execution, typically D+1 open in backtest)
        price      : execution price per share
        shares     : number of whole shares  (must be > 0)
        stop_price : planned exit stop carried from the buy intent (optional)
        """
        if shares <= 0:
            raise ValueError(f"shares must be > 0, got {shares}")
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")
        if ticker in self._open:
            raise ValueError(
                f"Cannot buy {ticker!r}: position already open. "
                "Close the existing position before opening a new one."
            )
        cost = price * shares
        if cost > self._cash + 1e-6:   # tiny tolerance for float arithmetic
            raise ValueError(
                f"Insufficient cash to buy {shares} shares of {ticker!r} "
                f"at {price:.4f} (cost={cost:.2f}, cash={self._cash:.2f})"
            )
        self._cash -= cost
        self._open[ticker] = OpenPosition(
            ticker=ticker,
            entry_date=entry_date,
            entry_price=float(price),
            shares=int(shares),
            stop_price=float(stop_price) if stop_price is not None else None,
        )

    def sell(
        self,
        ticker:    str,
        sell_date: date,
        price:     float,
    ) -> ClosedTrade:
        """
        Close an open position at price.

        Adds proceeds (price * shares) to cash and records the closed trade.

        Parameters
        ----------
        ticker    : must match an open position
        sell_date : trade date
        price     : execution price per share

        Returns
        -------
        The ClosedTrade record for this transaction.
        """
        if ticker not in self._open:
            raise KeyError(f"Cannot sell {ticker!r}: no open position found")
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")

        pos      = self._open.pop(ticker)
        proceeds = price * pos.shares
        pnl      = (price - pos.entry_price) * pos.shares
        pnl_pct  = (price / pos.entry_price - 1.0) * 100.0 if pos.entry_price > 0 else 0.0

        self._cash         += proceeds
        self._realized_pnl += pnl

        trade = ClosedTrade(
            ticker      = ticker,
            entry_date  = pos.entry_date,
            sell_date   = sell_date,
            entry_price = pos.entry_price,
            sell_price  = float(price),
            shares      = pos.shares,
            pnl         = round(pnl, 4),
            pnl_pct     = round(pnl_pct, 4),
        )
        self._trade_log.append(trade)
        return trade

    # ── snapshot / restore ────────────────────────────────────────────────

    def snapshot(self) -> "PortfolioState":
        """
        Return a deep-copy of the current state.

        The copy is completely independent — mutations to self after this call
        do not affect the snapshot, and vice versa.  Used by the backtest runner
        to checkpoint state before simulating each day.
        """
        return copy.deepcopy(self)

    # ── serialisation helpers ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Return a plain-dict summary suitable for JSON serialisation or logging.
        Does not include the full trade log (use .trade_log for that).
        """
        return {
            "cash":                round(self._cash, 2),
            "realized_pnl":        round(self._realized_pnl, 2),
            "open_positions_count": len(self._open),
            "closed_trades_count":  len(self._trade_log),
            "open_tickers":        list(self._open.keys()),
        }

    # ── dunder helpers ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"PortfolioState("
            f"cash={self._cash:,.2f}, "
            f"open={list(self._open.keys())}, "
            f"realized_pnl={self._realized_pnl:+,.2f})"
        )
