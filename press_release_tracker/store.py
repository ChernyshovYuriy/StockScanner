"""
press_release_tracker/store.py
================================
Persistence (SQLite). Same convention as edgar/store.py /
triple_screen_tracker/store.py: one schema string, connect() self-
migrates, plain functions -- no ORM.

guid (the feed item's own dedup key, falling back to link -- see
feeds.py) is the primary key on seen_items: this service polls every few
minutes (see the systemd timer), so the same item is refetched on every
run until it ages out of the feed's own window -- guid is what "already
seen" means here.

Unlike triple_screen_tracker/store.py's day-keyed email_log (one row per
daily run), this service polls many times a day, so "already emailed" is
tracked per item (seen_items.emailed) rather than per day.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

try:
    from config import PRESS_RELEASE_DB_PATH as DB_PATH
except Exception:
    DB_PATH = Path(__file__).resolve().parent.parent / "press_releases.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_items (
    guid TEXT PRIMARY KEY,
    feed_url TEXT NOT NULL,
    title TEXT NOT NULL,
    link TEXT,
    pubdate TEXT,
    description TEXT,
    categories TEXT,
    first_seen_at TEXT NOT NULL,
    emailed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS parsed_releases (
    guid TEXT PRIMARY KEY,
    ticker TEXT,
    company TEXT,
    category TEXT,
    materiality TEXT,
    summary TEXT,
    llm_model TEXT,
    parsed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS release_analysis (
    guid TEXT PRIMARY KEY,
    ticker TEXT,
    release_type TEXT,
    verdict TEXT,
    trend TEXT,
    dilution_level TEXT,
    confidence TEXT,
    analysis_json TEXT NOT NULL,
    body_chars INTEGER,
    llm_model TEXT,
    analyzed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batch_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_sent_at TEXT
);
"""


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def is_seen(conn, guid: str) -> bool:
    row = conn.execute("SELECT 1 FROM seen_items WHERE guid=?", (guid,)).fetchone()
    return row is not None


def mark_seen(conn, item, first_seen_at: str) -> None:
    """Record a freshly-fetched feed item as seen. INSERT OR IGNORE --
    the caller already checked is_seen(), but staying idempotent here too
    costs nothing and avoids a race across concurrent feeds."""
    conn.execute(
        "INSERT OR IGNORE INTO seen_items"
        "(guid,feed_url,title,link,pubdate,description,categories,first_seen_at,emailed) "
        "VALUES(?,?,?,?,?,?,?,?,0)",
        (item.guid, item.feed_url, item.title, item.link, item.pubdate,
         item.description, "|".join(item.categories), first_seen_at),
    )
    conn.commit()


def save_parsed(conn, guid: str, parsed: dict, model: str, parsed_at: str) -> None:
    """Store an LLM parse result for guid. INSERT OR REPLACE so a retried
    parse (e.g. after a prior run's crash) overwrites rather than
    duplicating."""
    conn.execute(
        "INSERT OR REPLACE INTO parsed_releases"
        "(guid,ticker,company,category,materiality,summary,llm_model,parsed_at) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (guid, parsed.get("ticker"), parsed.get("company"), parsed.get("category"),
         parsed.get("materiality"), parsed.get("summary"), model, parsed_at),
    )
    conn.commit()


def save_analysis(conn, guid: str, ticker: str, analysis: dict, model: str, analyzed_at: str) -> None:
    """Store analyst.py's full-article read for guid. The few columns a
    later scoring pass will filter on (verdict/trend/dilution/confidence)
    are pulled out of the JSON; the whole result stays in analysis_json.
    INSERT OR REPLACE, same retry reasoning as save_parsed()."""
    dilution = analysis.get("dilution")
    conn.execute(
        "INSERT OR REPLACE INTO release_analysis"
        "(guid,ticker,release_type,verdict,trend,dilution_level,confidence,"
        " analysis_json,body_chars,llm_model,analyzed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (guid, ticker, analysis.get("release_type"), analysis.get("verdict"), analysis.get("trend"),
         dilution.get("level") if isinstance(dilution, dict) else None, analysis.get("confidence"),
         json.dumps(analysis, ensure_ascii=False), analysis.get("body_chars"), model, analyzed_at),
    )
    conn.commit()


def unemailed(conn) -> list[dict]:
    """Every seen item not yet emailed, oldest first, LEFT JOINed against
    its LLM parse (None fields when unparsed -- e.g. OPENAI_API_KEY isn't
    configured, or the parse call failed). Includes items left over from a
    prior run that fetched them but crashed/failed before sending, so a
    quiet fetch cycle can still catch up on a backlog."""
    rows = conn.execute(
        "SELECT s.guid, s.feed_url, s.title, s.link, s.pubdate, "
        "       p.ticker, p.company, p.category, p.materiality, p.summary, a.analysis_json "
        "FROM seen_items s LEFT JOIN parsed_releases p ON p.guid = s.guid "
        "LEFT JOIN release_analysis a ON a.guid = s.guid "
        "WHERE s.emailed = 0 ORDER BY s.first_seen_at"
    ).fetchall()
    cols = ["guid", "feed_url", "title", "link", "pubdate",
            "ticker", "company", "category", "materiality", "summary", "analysis_json"]
    out = []
    for r in rows:
        row = dict(zip(cols, r))
        row["analysis"] = json.loads(row.pop("analysis_json")) if row["analysis_json"] else None
        out.append(row)
    return out


def mark_emailed(conn, guids: list[str]) -> None:
    conn.executemany("UPDATE seen_items SET emailed=1 WHERE guid=?", [(g,) for g in guids])
    conn.commit()


def get_last_batch_sent_at(conn) -> str | None:
    """When the hourly batch digest (see press_release_service.py) last
    actually sent, or None if it never has -- None means "due
    immediately" rather than waiting out a full interval on first run."""
    row = conn.execute("SELECT last_sent_at FROM batch_state WHERE id=1").fetchone()
    return row[0] if row else None


def set_last_batch_sent_at(conn, ts: str) -> None:
    conn.execute(
        "INSERT INTO batch_state(id,last_sent_at) VALUES(1,?) "
        "ON CONFLICT(id) DO UPDATE SET last_sent_at=excluded.last_sent_at",
        (ts,),
    )
    conn.commit()
