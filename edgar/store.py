"""
Layer 4: persistence (SQLite, keyed on CIK).

Lets the scanner and screeners diff over time instead of re-fetching. Everything
joins on CIK -- the shared key across fundamentals, insiders, and scan hits.
"""

import sqlite3
from pathlib import Path

# In StockScanner the path comes from config.py (EDGAR_DB_PATH); the fallback
# keeps this package runnable standalone in the edgar sandbox.
try:
    from config import EDGAR_DB_PATH as DB_PATH
except Exception:
    DB_PATH = Path(__file__).resolve().parent.parent / "edgar.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    cik INTEGER PRIMARY KEY,
    ticker TEXT
);
CREATE TABLE IF NOT EXISTS fundamentals (
    cik INTEGER, metric TEXT, val REAL, period_end TEXT,
    tag TEXT, fetched_at TEXT,
    PRIMARY KEY (cik, metric, period_end)
);
CREATE TABLE IF NOT EXISTS insider_buys (
    cik INTEGER, accession TEXT, owner TEXT, shares REAL, price REAL,
    txn_date TEXT, is_officer INTEGER, is_director INTEGER,
    PRIMARY KEY (accession, owner, txn_date)
);
CREATE TABLE IF NOT EXISTS scan_hits (
    cik INTEGER, ticker TEXT, form TEXT, category TEXT,
    filing_date TEXT, url TEXT,
    PRIMARY KEY (cik, form, filing_date, url)
);
CREATE TABLE IF NOT EXISTS email_log (
    digest_date TEXT PRIMARY KEY,
    hit_count INTEGER,
    sent INTEGER
);
CREATE TABLE IF NOT EXISTS activist_filings (
    accession TEXT PRIMARY KEY,
    cik INTEGER, ticker TEXT, form TEXT,
    filer TEXT, subject TEXT, pct REAL,
    filing_date TEXT, url TEXT,
    raw_text TEXT       -- full submission text, kept for later (LLM) analysis
);
"""


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def set_watchlist(conn, ticker_cik_pairs):
    conn.executemany("INSERT OR REPLACE INTO watchlist(cik,ticker) VALUES(?,?)",
                     [(c, t) for t, c in ticker_cik_pairs])
    conn.commit()


def watchlist_ciks(conn):
    return {r[0] for r in conn.execute("SELECT cik FROM watchlist")}


def save_scan_hits(conn, hits):
    rows = [(h.get("cik"), h.get("ticker"), h.get("form"), h.get("category"),
             h.get("date"), h.get("url")) for h in hits if h.get("cik")]
    conn.executemany(
        "INSERT OR IGNORE INTO scan_hits"
        "(cik,ticker,form,category,filing_date,url) VALUES(?,?,?,?,?,?)", rows)
    conn.commit()


def save_insider_buys(conn, cik, buys):
    rows = [(cik, b.get("accession"), b.get("owner"), b.get("shares"),
             b.get("price"), b.get("date"),
             int(bool(b.get("is_officer"))), int(bool(b.get("is_director"))))
            for b in buys]
    conn.executemany(
        "INSERT OR IGNORE INTO insider_buys"
        "(cik,accession,owner,shares,price,txn_date,is_officer,is_director)"
        " VALUES(?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def save_activist_filing(conn, row):
    """Upsert one parsed 13D/13G filing (structured fields + raw evidence text).

    Keyed on accession (INSERT OR IGNORE) so re-scans within the backfill window
    are idempotent. `raw_text` is retained so a later analysis layer can re-read
    the filing for anything the v1 regex parse missed.
    """
    conn.execute(
        "INSERT OR IGNORE INTO activist_filings"
        "(accession,cik,ticker,form,filer,subject,pct,filing_date,url,raw_text)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (row.get("accession"), row.get("cik"), row.get("ticker"), row.get("form"),
         row.get("filer"), row.get("subject"), row.get("pct"),
         row.get("filing_date") or row.get("date"), row.get("url"), row.get("raw_text")),
    )
    conn.commit()


def already_sent(conn, digest_date):
    """True if a digest for digest_date was already SENT — guards same-day re-runs.

    A quiet day recorded with sent=0 does NOT count as sent, so a later run the
    same day can still email if genuine hits appear.
    """
    row = conn.execute(
        "SELECT sent FROM email_log WHERE digest_date=?", (digest_date,)
    ).fetchone()
    return bool(row and row[0])


def record_email(conn, digest_date, hit_count, sent):
    """Record one digest run: date, number of flagged hits, whether it was sent."""
    conn.execute(
        "INSERT OR REPLACE INTO email_log(digest_date,hit_count,sent) VALUES(?,?,?)",
        (digest_date, int(hit_count), int(bool(sent))),
    )
    conn.commit()
