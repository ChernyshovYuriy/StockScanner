"""
db.py
=====
DuckDB persistence layer for the trading system.

Replaces five file-based state stores:
  data/funds                       → account table
  data/own.csv                     → positions table
  out/logs/sells_YYYYMMDD.csv      → trades table
  out/signal_db/signal_history.csv → signals table
  data/candidates_queue.csv        → intents table

Usage
-----
Each service calls init_db() once at startup — it is idempotent.
All functions open and close their own connection, so the three scheduled
services (main.py, virtual_buy.py, position_monitor.py) can run as
separate processes safely. DuckDB enforces a single-writer file lock
automatically; concurrent reads are always allowed.

Column names in returned DataFrames match the constants in schema_keys.py
so callers need minimal changes when switching from CSV reads.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Dict

import duckdb
import pandas as pd

from config import ROOT_DIR

DB_PATH = ROOT_DIR / "data" / "trading.db"


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _connect() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Open a connection, begin a transaction, commit or rollback, then close."""
    conn = duckdb.connect(str(DB_PATH))
    conn.begin()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

_DDL: list[str] = [
    # sequences for auto-increment IDs
    "CREATE SEQUENCE IF NOT EXISTS seq_positions",
    "CREATE SEQUENCE IF NOT EXISTS seq_trades",
    "CREATE SEQUENCE IF NOT EXISTS seq_signals",
    "CREATE SEQUENCE IF NOT EXISTS seq_intents",
    "CREATE SEQUENCE IF NOT EXISTS seq_transactions",

    """
    CREATE TABLE IF NOT EXISTS account (
        id         INTEGER PRIMARY KEY,
        cash       DOUBLE  NOT NULL,
        updated_at TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        id          INTEGER DEFAULT nextval('seq_positions') PRIMARY KEY,
        ticker      TEXT   NOT NULL UNIQUE,
        entry_date  TEXT   NOT NULL,
        entry_price DOUBLE NOT NULL,
        shares      DOUBLE NOT NULL,
        opened_at   TEXT   NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id          INTEGER DEFAULT nextval('seq_trades') PRIMARY KEY,
        ticker      TEXT   NOT NULL,
        entry_date  TEXT   NOT NULL,
        entry_price DOUBLE NOT NULL,
        shares      DOUBLE NOT NULL,
        sell_date   TEXT   NOT NULL,
        sell_price  DOUBLE NOT NULL,
        proceeds    DOUBLE NOT NULL,
        pnl_dollars DOUBLE NOT NULL,
        pnl_pct     DOUBLE NOT NULL,
        reason      TEXT   NOT NULL,
        closed_at   TEXT   NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
        id                        INTEGER DEFAULT nextval('seq_signals') PRIMARY KEY,
        ticker                    TEXT    NOT NULL,
        pattern                   TEXT    NOT NULL,
        state                     TEXT    NOT NULL,
        first_seen                TEXT    NOT NULL,
        last_seen                 TEXT    NOT NULL,
        days_in_state             INTEGER NOT NULL DEFAULT 0,
        consecutive_screener_days INTEGER NOT NULL DEFAULT 0,
        screener_days             INTEGER NOT NULL DEFAULT 0,
        entry                     DOUBLE,
        stop                      DOUBLE,
        target_2r                 DOUBLE,
        target_3r                 DOUBLE,
        risk_pct                  DOUBLE,
        pivot_price               DOUBLE,
        detail                    TEXT,
        alert_sent                BOOLEAN NOT NULL DEFAULT false,
        UNIQUE (ticker, pattern)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS intents (
        id                  INTEGER DEFAULT nextval('seq_intents') PRIMARY KEY,
        ticker              TEXT    NOT NULL,
        signal_date         TEXT    NOT NULL,
        alert_state         TEXT,
        priority            INTEGER,
        pattern             TEXT,
        entry_price_planned DOUBLE,
        stop_price          DOUBLE,
        target_price        DOUBLE,
        rr                  DOUBLE,
        intent_status       TEXT    NOT NULL DEFAULT 'PENDING',
        intent_reason       TEXT,
        created_at          TEXT    NOT NULL,
        executed_price      DOUBLE,
        executed_shares     DOUBLE,
        processed_at        TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS transactions (
        id          INTEGER DEFAULT nextval('seq_transactions') PRIMARY KEY,
        side        TEXT    NOT NULL,  -- 'BUY' or 'SELL'
        ticker      TEXT    NOT NULL,
        trade_date  TEXT    NOT NULL,
        price       DOUBLE  NOT NULL,
        shares      DOUBLE  NOT NULL,
        amount      DOUBLE  NOT NULL,  -- price * shares
        pnl_dollars DOUBLE,            -- SELL only
        pnl_pct     DOUBLE,            -- SELL only
        reason      TEXT,              -- SELL: exit reason; BUY: pattern name
        recorded_at TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_trades_ticker        ON trades       (ticker)",
    "CREATE INDEX IF NOT EXISTS idx_trades_sell_date     ON trades       (sell_date)",
    "CREATE INDEX IF NOT EXISTS idx_signals_ticker       ON signals      (ticker)",
    "CREATE INDEX IF NOT EXISTS idx_intents_status       ON intents      (intent_status)",
    "CREATE INDEX IF NOT EXISTS idx_transactions_ticker  ON transactions (ticker)",
    "CREATE INDEX IF NOT EXISTS idx_transactions_date    ON transactions (trade_date)",
    "CREATE INDEX IF NOT EXISTS idx_transactions_side    ON transactions (side)",
]


def init_db(path: Path = DB_PATH) -> None:
    """
    Create the database and all tables if they do not exist.

    Safe to call multiple times — all statements use IF NOT EXISTS.
    Call once at the top of each service's main block.
    """
    global DB_PATH
    DB_PATH = path
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        for stmt in _DDL:
            conn.execute(stmt)


# ─────────────────────────────────────────────────────────────────────────────
# ACCOUNT (cash balance)
# ─────────────────────────────────────────────────────────────────────────────

def get_cash() -> float:
    """Return the current cash balance. Returns 0.0 if not initialised."""
    with _connect() as conn:
        row = conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()
        return float(row[0]) if row else 0.0


def set_cash(cash: float) -> None:
    """Upsert the cash balance."""
    from time_utils import market_now
    now = market_now().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO account (id, cash, updated_at) VALUES (1, ?, ?)
            ON CONFLICT (id) DO UPDATE SET cash = excluded.cash,
                                           updated_at = excluded.updated_at
            """,
            [round(cash, 4), now],
        )


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTIONS (unified buy + sell ledger)
# ─────────────────────────────────────────────────────────────────────────────

def _record_transaction(
    conn: duckdb.DuckDBPyConnection,
    side: str,
    ticker: str,
    trade_date: str,
    price: float,
    shares: float,
    pnl_dollars: float | None = None,
    pnl_pct: float | None = None,
    reason: str | None = None,
) -> None:
    """Write one row to the transactions ledger. Always called inside an existing connection."""
    from time_utils import market_now
    conn.execute(
        """
        INSERT INTO transactions
          (side, ticker, trade_date, price, shares, amount, pnl_dollars, pnl_pct, reason, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            side,
            ticker.upper(),
            trade_date,
            round(price, 4),
            round(shares, 4),
            round(price * shares, 2),
            round(pnl_dollars, 2) if pnl_dollars is not None else None,
            round(pnl_pct, 2) if pnl_pct is not None else None,
            reason,
            market_now().isoformat(),
        ],
    )


def get_transactions() -> pd.DataFrame:
    """
    Return the full unified transaction ledger — every BUY and SELL in date order.

    Columns: side, ticker, trade_date, price, shares, amount,
             pnl_dollars, pnl_pct, reason, recorded_at
    """
    with _connect() as conn:
        return conn.execute(
            "SELECT side, ticker, trade_date, price, shares, amount, pnl_dollars, pnl_pct, reason, recorded_at"
            " FROM transactions ORDER BY trade_date, recorded_at"
        ).df()


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS (open trades)
# ─────────────────────────────────────────────────────────────────────────────

def get_open_positions() -> List[Dict]:
    """
    Return all open positions as a list of dicts.

    Keys match schema_keys.py POSITION_COL_* constants:
      ticker, entry_date, entry_price, shares
    """
    with _connect() as conn:
        return (
            conn.execute(
                "SELECT ticker, entry_date, entry_price, shares FROM positions ORDER BY entry_date"
            )
            .df()
            .to_dict("records")
        )


def get_open_positions_df() -> pd.DataFrame:
    """Return open positions as a DataFrame (drop-in for pd.read_csv(own.csv))."""
    with _connect() as conn:
        df = conn.execute(
            "SELECT ticker, entry_date, entry_price, shares FROM positions ORDER BY entry_date"
        ).df()
        if df.empty:
            return pd.DataFrame(columns=["ticker", "entry_date", "entry_price", "shares"])
        return df


def insert_position(ticker: str, entry_date: str, entry_price: float, shares: float, pattern: str | None = None) -> None:
    """Add a new open position and record a BUY transaction. Raises if ticker is already open."""
    from time_utils import market_now
    now = market_now().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO positions (ticker, entry_date, entry_price, shares, opened_at) VALUES (?, ?, ?, ?, ?)",
            [ticker.upper(), entry_date, round(entry_price, 4), round(shares, 4), now],
        )
        _record_transaction(conn, "BUY", ticker, entry_date, entry_price, shares, reason=pattern)


def delete_position(ticker: str) -> None:
    """Remove a position after it has been closed."""
    with _connect() as conn:
        conn.execute("DELETE FROM positions WHERE ticker = ?", [ticker.upper()])


# ─────────────────────────────────────────────────────────────────────────────
# TRADES (closed positions)
# ─────────────────────────────────────────────────────────────────────────────

def insert_trade(
        ticker: str,
        entry_date: str,
        entry_price: float,
        shares: float,
        sell_date: str,
        sell_price: float,
        proceeds: float,
        pnl_dollars: float,
        pnl_pct: float,
        reason: str,
) -> None:
    """Append one closed trade to the permanent trade log and record a SELL transaction."""
    from time_utils import market_now
    now = market_now().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO trades
              (ticker, entry_date, entry_price, shares,
               sell_date, sell_price, proceeds, pnl_dollars, pnl_pct, reason, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ticker.upper(),
                entry_date,
                round(entry_price, 4),
                round(shares, 4),
                sell_date,
                round(sell_price, 4),
                round(proceeds, 2),
                round(pnl_dollars, 2),
                round(pnl_pct, 2),
                reason,
                now,
            ],
        )
        _record_transaction(conn, "SELL", ticker, sell_date, sell_price, shares, pnl_dollars, pnl_pct, reason)


def get_all_trades() -> pd.DataFrame:
    """Return the full trade history as a DataFrame."""
    with _connect() as conn:
        return conn.execute(
            """
            SELECT ticker, entry_date, entry_price, shares,
                   sell_date, sell_price, proceeds,
                   pnl_dollars, pnl_pct, reason, closed_at
            FROM trades
            ORDER BY sell_date, ticker
            """
        ).df()


# ─────────────────────────────────────────────────────────────────────────────
# SIGNALS (auto_pipeline state machine)
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_COLS = [
    "ticker", "pattern", "state",
    "first_seen", "last_seen", "days_in_state", "consecutive_screener_days",
    "entry", "stop", "target_2r", "target_3r",
    "risk_pct", "pivot_price", "detail", "alert_sent", "screener_days",
]


def load_signals() -> pd.DataFrame:
    """
    Return the signal history as a DataFrame.

    Drop-in replacement for pd.read_csv(signal_history_path).
    Column names and dtypes match what auto_pipeline expects.
    """
    with _connect() as conn:
        df = conn.execute(f"SELECT {', '.join(_SIGNAL_COLS)} FROM signals").df()
        if df.empty:
            return pd.DataFrame(columns=_SIGNAL_COLS)
        return df


def save_signals(df: pd.DataFrame) -> None:
    """
    Replace the entire signals table with the contents of df.

    Drop-in replacement for df.to_csv(signal_history_path, index=False).
    auto_pipeline owns the signal DB in full — it rewrites on every run.
    """
    with _connect() as conn:
        conn.execute("DELETE FROM signals")
        if df.empty:
            return
        insert_df = df.reindex(columns=_SIGNAL_COLS).copy()
        insert_df["alert_sent"] = insert_df["alert_sent"].fillna(False).astype(bool)
        for col in ["days_in_state", "consecutive_screener_days", "screener_days"]:
            insert_df[col] = insert_df[col].fillna(0).astype(int)
        cols = ", ".join(_SIGNAL_COLS)
        conn.register("_signals_tmp", insert_df)
        conn.execute(f"INSERT INTO signals ({cols}) SELECT {cols} FROM _signals_tmp")


# ─────────────────────────────────────────────────────────────────────────────
# INTENTS (buy candidate queue)
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_COLS = [
    "ticker", "signal_date", "alert_state", "priority", "pattern",
    "entry_price_planned", "stop_price", "target_price", "rr",
    "intent_status", "intent_reason", "created_at",
]


def save_intents(intents: List[Dict]) -> None:
    """
    Write a fresh batch of PENDING buy intents produced by auto_pipeline.

    Replaces only the PENDING rows so that EXECUTED/SKIPPED/EXPIRED history
    is preserved. Call once per pipeline run.
    """
    from time_utils import market_now
    now = market_now().isoformat()
    with _connect() as conn:
        conn.execute("DELETE FROM intents WHERE intent_status = 'PENDING'")
        if not intents:
            return
        df = pd.DataFrame(intents)
        df["ticker"] = df["ticker"].str.upper()
        df["intent_status"] = "PENDING"
        df["created_at"] = now
        if "intent_reason" not in df.columns:
            df["intent_reason"] = None
        insert_df = df.reindex(columns=_INTENT_COLS)
        cols = ", ".join(_INTENT_COLS)
        conn.register("_intents_tmp", insert_df)
        conn.execute(f"INSERT INTO intents ({cols}) SELECT {cols} FROM _intents_tmp")


def load_pending_intents() -> pd.DataFrame:
    """
    Return all PENDING buy intents as a DataFrame.

    Drop-in replacement for reading candidates_queue.csv in virtual_buy.py.
    Includes the internal `id` column so callers can pass it to mark_intent_*.
    """
    with _connect() as conn:
        return conn.execute(
            """
            SELECT id, ticker, signal_date, alert_state, priority, pattern,
                   entry_price_planned, stop_price, target_price, rr,
                   intent_status, intent_reason, created_at
            FROM intents
            WHERE intent_status = 'PENDING'
            ORDER BY priority
            """
        ).df()


def mark_intent_executed(intent_id: int, executed_price: float, executed_shares: float) -> None:
    """Mark one intent as EXECUTED after virtual_buy fills it."""
    from time_utils import market_now
    now = market_now().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE intents
               SET intent_status   = 'EXECUTED',
                   executed_price  = ?,
                   executed_shares = ?,
                   processed_at    = ?
             WHERE id = ?
            """,
            [round(executed_price, 4), round(executed_shares, 4), now, intent_id],
        )


def mark_intent_skipped(intent_id: int, reason: str) -> None:
    """Mark one intent as SKIPPED (e.g. gap-up filter, insufficient funds)."""
    from time_utils import market_now
    now = market_now().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE intents
               SET intent_status = 'SKIPPED',
                   intent_reason = ?,
                   processed_at  = ?
             WHERE id = ?
            """,
            [reason, now, intent_id],
        )
