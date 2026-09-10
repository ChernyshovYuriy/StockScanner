"""
Persistence (SQLite). Same convention as edgar/store.py / demand_signals/store.py:
one schema string, connect() self-migrates, plain functions -- no ORM.

`id` (not `ticker`) is the primary key on tracked_signals: a ticker that gets
sold can be picked up again on a later BUY signal, and each buy->sell cycle
is its own row so the full history survives across cycles.
"""

import sqlite3
from pathlib import Path

# In StockScanner the path comes from config.py (TRIPLE_SCREEN_TRACKER_DB_PATH);
# the fallback keeps this package runnable standalone, same as edgar/store.py.
try:
    from config import TRIPLE_SCREEN_TRACKER_DB_PATH as DB_PATH
except Exception:
    DB_PATH = Path(__file__).resolve().parent.parent / "triple_screen_tracker.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracked_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    buy_date TEXT NOT NULL,
    buy_price REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    sell_date TEXT,
    sell_price REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS price_history (
    tracked_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    close_price REAL NOT NULL,
    PRIMARY KEY (tracked_id, date)
);
CREATE TABLE IF NOT EXISTS email_log (
    digest_date TEXT PRIMARY KEY,
    hit_count INTEGER,
    sent INTEGER
);
"""


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def open_tracked_tickers(conn) -> set:
    """Tickers with a currently-OPEN record -- the "already tracked, skip
    on this scan" set the daily service checks before considering a fresh
    BUY signal."""
    return {r[0] for r in conn.execute(
        "SELECT ticker FROM tracked_signals WHERE status='OPEN'")}


def open_records(conn) -> list[dict]:
    """All currently-OPEN rows, for the daily price-append + sell-check loop."""
    rows = conn.execute(
        "SELECT id, ticker, buy_date, buy_price, created_at "
        "FROM tracked_signals WHERE status='OPEN'"
    ).fetchall()
    return [{"id": r[0], "ticker": r[1], "buy_date": r[2], "buy_price": r[3],
              "created_at": r[4]} for r in rows]


def create_tracked(conn, ticker, buy_date, buy_price, created_at) -> int:
    """Open a new tracked record at buy_price/buy_date. Returns its id."""
    cur = conn.execute(
        "INSERT INTO tracked_signals(ticker,buy_date,buy_price,status,created_at) "
        "VALUES(?,?,?,'OPEN',?)",
        (ticker, buy_date, buy_price, created_at),
    )
    conn.commit()
    return cur.lastrowid


def append_price(conn, tracked_id, date, close_price) -> None:
    """Record one day's close for one tracked record. INSERT OR REPLACE so a
    same-day rerun (e.g. a systemd retry) updates that day's row instead of
    duplicating it."""
    conn.execute(
        "INSERT OR REPLACE INTO price_history(tracked_id,date,close_price) VALUES(?,?,?)",
        (tracked_id, date, close_price),
    )
    conn.commit()


def close_tracked(conn, tracked_id, sell_date, sell_price) -> None:
    """Close a tracked record: price closed below its buy_price."""
    conn.execute(
        "UPDATE tracked_signals SET status='SOLD', sell_date=?, sell_price=? WHERE id=?",
        (sell_date, sell_price, tracked_id),
    )
    conn.commit()


def price_history_for(conn, tracked_id) -> list[dict]:
    """One tracked record's daily close history, oldest first."""
    rows = conn.execute(
        "SELECT date, close_price FROM price_history WHERE tracked_id=? ORDER BY date",
        (tracked_id,),
    ).fetchall()
    return [{"date": r[0], "close_price": r[1]} for r in rows]


def all_tracked(conn) -> list[dict]:
    """Every tracked record, OPEN and SOLD -- the dashboard's full-state read."""
    rows = conn.execute(
        "SELECT id, ticker, buy_date, buy_price, status, sell_date, sell_price, created_at "
        "FROM tracked_signals ORDER BY buy_date DESC, id DESC"
    ).fetchall()
    return [{"id": r[0], "ticker": r[1], "buy_date": r[2], "buy_price": r[3],
              "status": r[4], "sell_date": r[5], "sell_price": r[6],
              "created_at": r[7]} for r in rows]


def already_sent(conn, digest_date) -> bool:
    """True if a digest for digest_date was already SENT -- guards a same-day
    systemd retry from double-emailing (same pattern as edgar/store.py)."""
    row = conn.execute(
        "SELECT sent FROM email_log WHERE digest_date=?", (digest_date,)
    ).fetchone()
    return bool(row and row[0])


def record_email(conn, digest_date, hit_count, sent) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO email_log(digest_date,hit_count,sent) VALUES(?,?,?)",
        (digest_date, int(hit_count), int(bool(sent))),
    )
    conn.commit()
