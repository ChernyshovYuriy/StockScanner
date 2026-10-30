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
`symbol_map` / `price_context` / `report_prices` are prices.py's (step 3):
an RBC symbol's Yahoo listing (NULL = none found), each symbol's latest
price context, and each note's base close. symbol_map is keyed on the
company name RBC printed too: a bare symbol can name two companies (CFR
is Cullen/Frost in the US and Richemont in Switzerland).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

try:
    from config import RBC_REPORTS_DB_PATH as DB_PATH
except Exception:
    DB_PATH = Path(__file__).resolve().parent.parent / "rbc_reports.db"

# A report listing more tickers than this is a sector note or a daily
# digest whose list is a peer list, not its subject (see
# rbc_dashboard_data.py) -- a "company note" lists at most this many, and
# its first-listed ticker is its subject.
COMPANY_NOTE_MAX_TICKERS = 2
COMPANY_NOTE_SQL = (f"report_key IN (SELECT report_key FROM report_tickers GROUP BY report_key "
                    f"HAVING COUNT(*) <= {COMPANY_NOTE_MAX_TICKERS})")

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
CREATE TABLE IF NOT EXISTS symbol_map (
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL DEFAULT '',
    company TEXT NOT NULL,
    yahoo_symbol TEXT,
    yahoo_name TEXT,
    currency TEXT,
    checked_at TEXT NOT NULL,
    PRIMARY KEY (symbol, exchange, company)
);
CREATE TABLE IF NOT EXISTS price_context (
    yahoo_symbol TEXT PRIMARY KEY,
    as_of TEXT NOT NULL,
    close REAL,
    high_52w REAL,
    low_52w REAL,
    pct_off_high REAL,
    range_pos REAL,
    ret_20d REAL,
    above_sma50 INTEGER,
    turning_up INTEGER,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS report_prices (
    report_key TEXT PRIMARY KEY,
    yahoo_symbol TEXT NOT NULL,
    base_date TEXT NOT NULL,
    base_close REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
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


def unresolved_subjects(conn, retry: bool = False, limit: int | None = None) -> list:
    """[(symbol, exchange, company)] -- company notes' subjects (ticker +
    the company name RBC printed) not yet looked up (also the misses,
    with retry=True), most recently covered first."""
    known = ("SELECT 1 FROM symbol_map m WHERE m.symbol = t.symbol AND m.exchange = t.exchange "
             "AND m.company = r.company" + (" AND m.yahoo_symbol IS NOT NULL" if retry else ""))
    sql = ("SELECT t.symbol, t.exchange, r.company, MAX(r.published_at) last FROM report_tickers t "
           "JOIN reports r ON r.report_key = t.report_key "
           f"WHERE t.position = 0 AND t.{COMPANY_NOTE_SQL} AND r.company IS NOT NULL "
           f"AND NOT EXISTS ({known}) GROUP BY t.symbol, t.exchange, r.company ORDER BY last DESC")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [(sym, exch, company) for sym, exch, company, _ in conn.execute(sql)]


def save_symbol(conn, symbol, exchange, company, yahoo_symbol, yahoo_name, currency, checked_at,
                commit: bool = True):
    conn.execute(
        "INSERT OR REPLACE INTO symbol_map(symbol, exchange, company, yahoo_symbol, yahoo_name, "
        "currency, checked_at) VALUES(?,?,?,?,?,?,?)",
        (symbol, exchange, company, yahoo_symbol, yahoo_name, currency, checked_at))
    if commit:
        conn.commit()


def price_window_notes(conn, since: str, positive_actions) -> list:
    """[(report_key, yahoo_symbol, published_at)] -- company notes since
    `since` tagged bullish or with a positive action, whose subject has a
    Yahoo listing."""
    marks = ",".join("?" * len(positive_actions))
    return conn.execute(
        "SELECT r.report_key, m.yahoo_symbol, r.published_at FROM reports r "
        "JOIN report_tags g ON g.report_key = r.report_key "
        "JOIN report_tickers t ON t.report_key = r.report_key AND t.position = 0 "
        "JOIN symbol_map m ON m.symbol = t.symbol AND m.exchange = t.exchange "
        "AND m.company = r.company "
        f"WHERE r.published_at >= ? AND r.{COMPANY_NOTE_SQL} AND m.yahoo_symbol IS NOT NULL "
        f"AND (g.tone = 'bullish' OR g.action IN ({marks})) ORDER BY r.published_at",
        [since, *positive_actions]).fetchall()


def save_price_context(conn, contexts: dict, updated_at: str, commit: bool = True):
    conn.executemany(
        "INSERT OR REPLACE INTO price_context(yahoo_symbol, as_of, close, high_52w, low_52w, "
        "pct_off_high, range_pos, ret_20d, above_sma50, turning_up, updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        [(ys, c["as_of"], c["close"], c["high_52w"], c["low_52w"], c["pct_off_high"],
          c["range_pos"], c["ret_20d"], c["above_sma50"], c["turning_up"], updated_at)
         for ys, c in contexts.items()])
    if commit:
        conn.commit()


def save_report_prices(conn, rows: list, commit: bool = True):
    """rows: [(report_key, yahoo_symbol, base_date, base_close)]."""
    conn.executemany("INSERT OR REPLACE INTO report_prices(report_key, yahoo_symbol, base_date, "
                     "base_close) VALUES(?,?,?,?)", rows)
    if commit:
        conn.commit()


def get_meta(conn, key: str):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn, key: str, value: str):
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)", (key, value))
    conn.commit()
