"""
news_watchlist/store.py
=========================
Persistence (SQLite). Same convention as press_release_tracker/store.py /
triple_screen_tracker/store.py: one schema string, connect() self-migrates,
plain functions -- no ORM.

`id` (not `ticker`) is the primary key on watchlist_items, same reasoning as
triple_screen_tracker/store.py's tracked_signals: a ticker dismissed once
can be re-added (manually, or by a fresh press release) without losing its
prior history.

`guid` (nullable -- null for a manually-added item) carries a unique partial
index rather than being the primary key itself: it's the idempotency key for
auto-seeding (news_watchlist_service.py re-checks press_releases.db every
run, same "re-fetched every cycle" shape press_release_tracker/store.py's
own guid handling has), but a manual add has no guid to key off.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

try:
    from config import NEWS_WATCHLIST_DB_PATH as DB_PATH
except Exception:
    DB_PATH = Path(__file__).resolve().parent.parent / "news_watchlist.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guid TEXT,
    ticker TEXT NOT NULL,
    company TEXT,
    category TEXT,
    materiality TEXT,
    summary TEXT,
    source_link TEXT,
    status TEXT NOT NULL DEFAULT 'inbox',
    note TEXT,
    flagged_at TEXT NOT NULL,
    flag_price REAL,
    status_changed_at TEXT,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_items_guid
    ON watchlist_items(guid) WHERE guid IS NOT NULL;
CREATE TABLE IF NOT EXISTS price_history (
    item_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    close_price REAL NOT NULL,
    PRIMARY KEY (item_id, date)
);
"""

_ITEM_COLUMNS = (
    "id", "guid", "ticker", "company", "category", "materiality", "summary",
    "source_link", "status", "note", "flagged_at", "flag_price",
    "status_changed_at", "created_at",
)


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def _row_to_item(row) -> dict:
    return dict(zip(_ITEM_COLUMNS, row))


def seeded_guids(conn) -> set:
    """Every guid already present -- news_watchlist_service.py's own
    "already seeded, skip it" set, checked before considering a fresh
    parsed_releases row."""
    return {r[0] for r in conn.execute(
        "SELECT guid FROM watchlist_items WHERE guid IS NOT NULL")}


def seed_inbox_item(conn, *, guid, ticker, company, category, materiality,
                     summary, source_link, flagged_at, flag_price,
                     created_at) -> int:
    """Insert a fresh auto-seeded candidate as status='inbox'. Returns its
    id. Caller (news_watchlist_service.py) is responsible for the
    idempotency check via seeded_guids() -- INSERT here relies on the
    unique guid index only as a last-resort guard against a same-run race,
    not the primary dedup mechanism."""
    cur = conn.execute(
        "INSERT INTO watchlist_items"
        "(guid,ticker,company,category,materiality,summary,source_link,"
        " status,flagged_at,flag_price,status_changed_at,created_at) "
        "VALUES(?,?,?,?,?,?,?,'inbox',?,?,?,?)",
        (guid, ticker, company, category, materiality, summary, source_link,
         flagged_at, flag_price, created_at, created_at),
    )
    conn.commit()
    return cur.lastrowid


def add_manual(conn, *, ticker, note, flagged_at, flag_price,
                created_at) -> int:
    """A user-typed ticker the feed missed -- inserted straight into
    'watching' (no inbox step) since the act of typing it in already IS the
    triage decision, same immediacy as conviction_watchlist's manual holding
    add. guid is NULL (nothing to dedup against)."""
    cur = conn.execute(
        "INSERT INTO watchlist_items"
        "(guid,ticker,company,category,materiality,summary,source_link,"
        " status,note,flagged_at,flag_price,status_changed_at,created_at) "
        "VALUES(NULL,?,NULL,NULL,NULL,NULL,NULL,'watching',?,?,?,?,?)",
        (ticker, note, flagged_at, flag_price, created_at, created_at),
    )
    conn.commit()
    return cur.lastrowid


def list_by_status(conn, status: str) -> list[dict]:
    rows = conn.execute(
        f"SELECT {','.join(_ITEM_COLUMNS)} FROM watchlist_items "
        "WHERE status=? ORDER BY flagged_at DESC, id DESC",
        (status,),
    ).fetchall()
    return [_row_to_item(r) for r in rows]


def get_item(conn, item_id: int) -> dict | None:
    row = conn.execute(
        f"SELECT {','.join(_ITEM_COLUMNS)} FROM watchlist_items WHERE id=?",
        (item_id,),
    ).fetchone()
    return _row_to_item(row) if row else None


def set_status(conn, item_id: int, status: str, changed_at: str) -> None:
    conn.execute(
        "UPDATE watchlist_items SET status=?, status_changed_at=? WHERE id=?",
        (status, changed_at, item_id),
    )
    conn.commit()


def set_note(conn, item_id: int, note: str) -> None:
    conn.execute("UPDATE watchlist_items SET note=? WHERE id=?", (note, item_id))
    conn.commit()


def append_price(conn, item_id: int, date: str, close_price: float) -> None:
    """Record one day's close for one watching item. INSERT OR REPLACE so a
    same-day rerun updates that day's row instead of duplicating it -- same
    convention as triple_screen_tracker/store.py.append_price()."""
    conn.execute(
        "INSERT OR REPLACE INTO price_history(item_id,date,close_price) VALUES(?,?,?)",
        (item_id, date, close_price),
    )
    conn.commit()


def price_history_for(conn, item_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT date, close_price FROM price_history WHERE item_id=? ORDER BY date",
        (item_id,),
    ).fetchall()
    return [{"date": r[0], "close_price": r[1]} for r in rows]
