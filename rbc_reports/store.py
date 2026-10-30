"""
rbc_reports/store.py
====================
Persistence (SQLite, data/rbc_reports.db). Same convention as
press_release_tracker/store.py: one schema string, connect() self-
migrates, plain functions -- no ORM.

`reports` is keyed on the report's own ResearchViewer link (see
parser.ReportItem.key), so re-ingesting the same digest -- or the same
email saved twice under two filenames -- never duplicates a row.
`report_tickers.position` is the ticker's place in the report's own list
-- 0 is the report's subject; the rest are often a sector note's peer list.
`digests` is keyed on the digest's "as of" timestamp plus its
subscription name (several alerts can all arrive at ~7pm) and keeps its
shown/total counts (a digest only lists the first 100 reports).
`report_tags` holds tagger.py's LLM labels, one row per tagged report.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

try:
    from config import RBC_REPORTS_DB_PATH as DB_PATH
except Exception:
    DB_PATH = Path(__file__).resolve().parent.parent / "rbc_reports.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS digests (
    as_of TEXT NOT NULL,
    subscription TEXT NOT NULL DEFAULT '',
    shown INTEGER,
    total INTEGER,
    source_file TEXT,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (as_of, subscription)
);
CREATE TABLE IF NOT EXISTS reports (
    report_key TEXT PRIMARY KEY,
    published_at TEXT NOT NULL,
    source TEXT,
    title TEXT NOT NULL,
    company TEXT,
    headline TEXT,
    link TEXT,
    view TEXT,
    tickers_raw TEXT,
    analysts TEXT,
    pdf_size TEXT,
    digest_as_of TEXT,
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reports_published ON reports(published_at);
CREATE TABLE IF NOT EXISTS report_tickers (
    report_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL DEFAULT '',
    position INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (report_key, symbol, exchange)
);
CREATE INDEX IF NOT EXISTS idx_report_tickers_symbol ON report_tickers(symbol);
CREATE TABLE IF NOT EXISTS report_tags (
    report_key TEXT PRIMARY KEY,
    report_type TEXT,
    action TEXT,
    tone TEXT,
    rating TEXT,
    theme TEXT,
    model TEXT,
    tagged_at TEXT NOT NULL
);
"""


def connect(db_path=DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    # One transaction for the whole schema: run as separate statements,
    # each CREATE commits (and fsyncs) on its own, which took ~7s on the
    # Pi's SD card.
    conn.executescript("BEGIN;" + _SCHEMA + "COMMIT;")
    return conn


def ingested_files(conn) -> set:
    """Source filenames already ingested -- ingest.py skips re-parsing
    them (a digest file never changes once saved). A digest that yielded
    no reports doesn't count, so a parser fix (e.g. after an RBC template
    change) re-reads it on the next run instead of skipping it forever."""
    return {r[0] for r in conn.execute(
        "SELECT d.source_file FROM digests d WHERE d.source_file IS NOT NULL "
        "AND EXISTS (SELECT 1 FROM reports r WHERE r.digest_as_of = d.as_of)")}


def save_digest(conn, digest, source_file: str, ingested_at: str, commit: bool = True) -> int:
    """Upserts the digest row and inserts every not-yet-stored report.
    Returns how many reports were new. commit=False leaves the commit to
    the caller -- ingest.py commits once per run, since each commit costs
    1-5s on the Pi's SD card."""
    conn.execute(
        "INSERT INTO digests(as_of, subscription, shown, total, source_file, ingested_at) "
        "VALUES(?,?,?,?,?,?) ON CONFLICT(as_of, subscription) DO UPDATE SET "
        "shown=excluded.shown, total=excluded.total",
        (digest.as_of, digest.subscription, digest.shown, digest.total, source_file, ingested_at),
    )
    new = 0
    for it in digest.items:
        cur = conn.execute(
            "INSERT OR IGNORE INTO reports(report_key, published_at, source, title, company, "
            "headline, link, view, tickers_raw, analysts, pdf_size, digest_as_of, ingested_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (it.key, it.published_at, it.source, it.title, it.company, it.headline, it.link,
             it.view, it.tickers_raw, it.analysts, it.pdf_size, digest.as_of, ingested_at),
        )
        if cur.rowcount:
            new += 1
            conn.executemany(
                "INSERT OR IGNORE INTO report_tickers(report_key, symbol, exchange, position) "
                "VALUES(?,?,?,?)",
                [(it.key, sym, exch, pos) for pos, (sym, exch) in enumerate(it.tickers)],
            )
    if commit:
        conn.commit()
    return new


def untagged_reports(conn, limit: int | None = None, retag: bool = False) -> list:
    """[(report_key, title, view, tickers_raw)] newest first -- reports
    with no report_tags row (every report with retag=True)."""
    sql = "SELECT r.report_key, r.title, r.view, r.tickers_raw FROM reports r "
    if not retag:
        sql += "WHERE NOT EXISTS (SELECT 1 FROM report_tags g WHERE g.report_key = r.report_key) "
    sql += "ORDER BY r.published_at DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql).fetchall()


def save_tags(conn, labels: dict, model: str, tagged_at: str, commit: bool = True) -> int:
    """labels: {report_key: {"report_type", "action", "tone", "rating",
    "theme"}}. Upserts each; returns how many."""
    conn.executemany(
        "INSERT OR REPLACE INTO report_tags(report_key, report_type, action, tone, rating, theme, "
        "model, tagged_at) VALUES(?,?,?,?,?,?,?,?)",
        [(k, v["report_type"], v["action"], v["tone"], v["rating"], v["theme"], model, tagged_at)
         for k, v in labels.items()],
    )
    if commit:
        conn.commit()
    return len(labels)
