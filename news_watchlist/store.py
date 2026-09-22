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
index, but a manual add has no guid to key off.

`seeded_release_guids` (added 2026-09) is the actual idempotency log for
auto-seeding (news_watchlist_service.py re-checks press_releases.db every
run, same "re-fetched every cycle" shape press_release_tracker/store.py's
own seen_items has) -- kept as a separate permanent table rather than
reading watchlist_items.guid directly, because a fresh press release for a
ticker that already has an unreviewed ('inbox') row now COLLAPSES into
that row via update_inbox_item() (see find_pending_inbox_item()) instead
of inserting a new one; that overwrites the row's guid column, so
watchlist_items alone would forget every guid it had already processed
and start reseeding old candidates. connect()'s schema script backfills
this table from watchlist_items.guid once, for a DB created before this
table existed.
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
    created_at TEXT NOT NULL,
    yahoo_ticker TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_items_guid
    ON watchlist_items(guid) WHERE guid IS NOT NULL;
CREATE TABLE IF NOT EXISTS price_history (
    item_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    close_price REAL NOT NULL,
    PRIMARY KEY (item_id, date)
);
CREATE TABLE IF NOT EXISTS seeded_release_guids (
    guid TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL
);
INSERT OR IGNORE INTO seeded_release_guids (guid, processed_at)
    SELECT guid, created_at FROM watchlist_items WHERE guid IS NOT NULL;
"""

_ITEM_COLUMNS = (
    "id", "guid", "ticker", "company", "category", "materiality", "summary",
    "source_link", "status", "note", "flagged_at", "flag_price",
    "status_changed_at", "created_at", "yahoo_ticker",
)


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    # Migration for a watchlist_items table created before yahoo_ticker was
    # added -- no-op on a fresh DB, where the column already exists from
    # the CREATE TABLE above. SQLite's ALTER TABLE has no "ADD COLUMN IF
    # NOT EXISTS" (unlike DuckDB's, see db.py's own stop_price/target_price
    # migrations) -- PRAGMA table_info is the standard SQLite way to check
    # first. Holds the suffixed symbol (e.g. "AYA.TO") that
    # news_watchlist_service.py's _resolve_market_price() actually found
    # data under for a bare LLM-parsed ticker (e.g. "AYA") -- the
    # dashboard's Ticker column link needs this to point at a real Yahoo
    # Finance quote page instead of 404ing or landing on an unrelated
    # same-letters security. NULL for a manually-added item
    # (news_watchlist_store.add_manual) or a pre-existing row not yet
    # backfilled -- the dashboard falls back to the stored ticker itself.
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(watchlist_items)")}
    if "yahoo_ticker" not in existing_columns:
        conn.execute("ALTER TABLE watchlist_items ADD COLUMN yahoo_ticker TEXT")
        conn.commit()
    return conn


def _row_to_item(row) -> dict:
    return dict(zip(_ITEM_COLUMNS, row))


def seeded_guids(conn) -> set:
    """Every guid ever processed from the press-release feed --
    news_watchlist_service.py's own "already seeded, skip it" set, checked
    before considering a fresh parsed_releases row. Read from
    seeded_release_guids (a permanent log), not watchlist_items.guid --
    see this module's own docstring for why."""
    return {r[0] for r in conn.execute("SELECT guid FROM seeded_release_guids")}


def mark_guid_processed(conn, guid, processed_at) -> None:
    """Records that a press-release guid has been handled (either seeded
    as a new inbox item or collapsed into an existing one) -- the write
    side of seeded_guids()'s idempotency check."""
    conn.execute(
        "INSERT OR IGNORE INTO seeded_release_guids(guid, processed_at) VALUES(?,?)",
        (guid, processed_at),
    )
    conn.commit()


def find_pending_inbox_item(conn, ticker: str) -> dict | None:
    """Most recent still-unreviewed ('inbox') item for ticker, if any.
    news_watchlist_service.seed_inbox() uses this to collapse a fresh
    press release into the existing pending row instead of adding
    another -- a busy ticker (e.g. several procedural filings for the
    same M&A deal) no longer floods the inbox with one row per article."""
    row = conn.execute(
        f"SELECT {','.join(_ITEM_COLUMNS)} FROM watchlist_items "
        "WHERE ticker=? AND status='inbox' ORDER BY id DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    return _row_to_item(row) if row else None


def update_inbox_item(conn, item_id: int, *, guid, company, category, materiality,
                       summary, source_link, flagged_at, flag_price, created_at,
                       yahoo_ticker=None) -> None:
    """Refreshes an existing pending inbox row with a newer press release
    about the same ticker (see find_pending_inbox_item()) -- content and
    the flag date/price move to this latest catalyst; id/note are left
    untouched. created_at also moves to this latest catalyst's timestamp
    -- the dashboard's Inbox "Flagged" column and its sort order are both
    driven by created_at (see news_watchlist_dashboard_data.py's
    _flagged_date_time()), so leaving the original seed timestamp in
    place made a same-day refreshed catalyst display/sort as if it were
    the old, already-reviewed article it collapsed into. yahoo_ticker
    (default None, for a caller that hasn't resolved one) also moves to
    whatever this latest catalyst's own price lookup resolved -- see
    seed_inbox_item()'s own docstring."""
    conn.execute(
        "UPDATE watchlist_items SET guid=?, company=?, category=?, materiality=?, "
        "summary=?, source_link=?, flagged_at=?, flag_price=?, created_at=?, "
        "yahoo_ticker=? WHERE id=?",
        (guid, company, category, materiality, summary, source_link,
         flagged_at, flag_price, created_at, yahoo_ticker, item_id),
    )
    conn.commit()


def seed_inbox_item(conn, *, guid, ticker, company, category, materiality,
                     summary, source_link, flagged_at, flag_price,
                     created_at, yahoo_ticker=None) -> int:
    """Insert a fresh auto-seeded candidate as status='inbox'. Returns its
    id. Caller (news_watchlist_service.py) is responsible for the
    idempotency check via seeded_guids() -- INSERT here relies on the
    unique guid index only as a last-resort guard against a same-run race,
    not the primary dedup mechanism.

    yahoo_ticker (default None) is the suffixed symbol
    news_watchlist_service._resolve_market_price() actually found this
    item's flag_price under (e.g. "AYA.TO" for the bare parsed ticker
    "AYA") -- kept separate from `ticker` itself, which stays the literal
    bare/as-parsed form so find_pending_inbox_item()'s dedup keeps
    matching future releases for the same company (see that function's
    own docstring)."""
    cur = conn.execute(
        "INSERT INTO watchlist_items"
        "(guid,ticker,company,category,materiality,summary,source_link,"
        " status,flagged_at,flag_price,status_changed_at,created_at,yahoo_ticker) "
        "VALUES(?,?,?,?,?,?,?,'inbox',?,?,?,?,?)",
        (guid, ticker, company, category, materiality, summary, source_link,
         flagged_at, flag_price, created_at, created_at, yahoo_ticker),
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


def set_yahoo_ticker(conn, item_id: int, yahoo_ticker: str) -> None:
    """Backfill hook for a row seeded before yahoo_ticker existed -- see
    the column's own migration comment in _SCHEMA. Not called anywhere in
    the regular seed/update-prices flow (that path sets it at insert/
    update time instead); used by the one-off backfill script for
    pre-existing rows."""
    conn.execute("UPDATE watchlist_items SET yahoo_ticker=? WHERE id=?", (yahoo_ticker, item_id))
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
