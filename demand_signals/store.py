"""
Persistence (SQLite), keyed on ticker+date+source+signal_type. Same
convention as edgar/store.py: one schema string, connect() self-migrates,
plain functions -- no ORM.
"""

import sqlite3
from pathlib import Path

# In StockScanner the path comes from config.py (DEMAND_DB_PATH); the
# fallback keeps this package runnable standalone, same as edgar/store.py.
try:
    from config import DEMAND_DB_PATH as DB_PATH
except Exception:
    DB_PATH = Path(__file__).resolve().parent.parent / "demand_signals.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS demand_signals (
    ticker TEXT, us_ticker TEXT, date TEXT,
    source TEXT, signal_type TEXT,
    direction TEXT, strength REAL, lag_days INTEGER,
    detail TEXT, fetched_at TEXT,
    PRIMARY KEY (ticker, date, source, signal_type)
);
"""


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def save_signal(conn, signal) -> None:
    """Upsert one DemandSignal (schema.DemandSignal). Re-runs for the same
    ticker/date/source/signal_type replace the row (a source may refine its
    own read of the same day, e.g. FINRA restating a prior week)."""
    row = signal.to_row()
    conn.execute(
        "INSERT OR REPLACE INTO demand_signals"
        "(ticker,us_ticker,date,source,signal_type,direction,strength,"
        "lag_days,detail,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (row["ticker"], row["us_ticker"], row["date"], row["source"],
         row["signal_type"], row["direction"], row["strength"],
         row["lag_days"], row["detail"], row["fetched_at"]),
    )
    conn.commit()


def save_signals(conn, signals) -> None:
    """Upsert many DemandSignals in one transaction."""
    rows = [s.to_row() for s in signals]
    conn.executemany(
        "INSERT OR REPLACE INTO demand_signals"
        "(ticker,us_ticker,date,source,signal_type,direction,strength,"
        "lag_days,detail,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
        [(r["ticker"], r["us_ticker"], r["date"], r["source"], r["signal_type"],
          r["direction"], r["strength"], r["lag_days"], r["detail"], r["fetched_at"])
         for r in rows],
    )
    conn.commit()


def signals_for_ticker(conn, ticker, since_date=None):
    """All stored signals for one ticker (screener-facing query), newest
    first. `ticker` matches the DemandSignal.ticker column as originally
    written (CAN or US symbol, whichever the screener passed in)."""
    from demand_signals.schema import DemandSignal

    if since_date:
        rows = conn.execute(
            "SELECT ticker,us_ticker,date,source,signal_type,direction,"
            "strength,lag_days,detail,fetched_at FROM demand_signals "
            "WHERE ticker=? AND date>=? ORDER BY date DESC",
            (ticker, since_date),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ticker,us_ticker,date,source,signal_type,direction,"
            "strength,lag_days,detail,fetched_at FROM demand_signals "
            "WHERE ticker=? ORDER BY date DESC",
            (ticker,),
        ).fetchall()

    cols = ["ticker", "us_ticker", "date", "source", "signal_type",
            "direction", "strength", "lag_days", "detail", "fetched_at"]
    return [DemandSignal.from_row(dict(zip(cols, r))) for r in rows]
