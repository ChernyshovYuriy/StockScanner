"""
rbc_dashboard_data.py
=====================
Read-only assembly of the RBC research tracker (rbc_reports/) for the web
dashboard's /rbc route.

Same isolation pattern as scanner_dashboard_data.py: opens its OWN
read-only sqlite3 connection straight at RBC_REPORTS_DB_PATH -- never
rbc_reports/store.py's connect() (a write-shaped self-migrate the
dashboard's long-running process shouldn't do on every page load). A
missing DB (nothing ingested yet) reads as an empty state, not an error.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from config import RBC_REPORTS_DB_PATH

FEED_LIMIT = 500
TOP_TICKERS = 25
# A report listing more tickers than this is a sector note or a daily
# digest whose list is a peer list (often alphabetical), not its subject --
# kept out of the "Most covered" ranking. On the first 13 digests: 847
# reports had 1 ticker, 62 had 2, and 3+ were sector notes/morning digests.
COMPANY_NOTE_MAX_TICKERS = 2


def _connect(db_path):
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _empty_state(days, ticker, query):
    return {"reports": [], "digests": [], "top_tickers": [], "days": days, "ticker": ticker,
            "query": query, "total_reports": 0, "truncated": False}


def build_rbc_state(days: int | None = 7, ticker: str | None = None, query: str | None = None,
                    db_path=RBC_REPORTS_DB_PATH, now: datetime | None = None) -> dict:
    """Feed rows for the last `days` days (None = all), optionally only one
    ticker symbol's reports or those matching a free-text `query` (company,
    headline, view or analyst). A ticker view ignores `days` -- it's that
    ticker's whole timeline. Also returns the digests' shown/total counts
    and the most-covered tickers over the same window -- counted only over
    company notes (see COMPANY_NOTE_MAX_TICKERS), by their first-listed
    ticker, so sector notes' peer lists don't swamp the ranking."""
    ticker = (ticker or "").strip().upper() or None
    query = (query or "").strip() or None
    if not Path(db_path).exists():
        return _empty_state(days, ticker, query)

    since = None
    if days and not ticker:
        since = ((now or datetime.now()) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M")

    where, params = [], []
    if since:
        where.append("r.published_at >= ?")
        params.append(since)
    if ticker:
        where.append("r.report_key IN (SELECT report_key FROM report_tickers WHERE symbol = ?)")
        params.append(ticker)
    if query:
        where.append("(r.title LIKE ? OR r.view LIKE ? OR r.analysts LIKE ?)")
        params += [f"%{query}%"] * 3
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    conn = _connect(db_path)
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM reports r {clause}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT r.report_key, r.published_at, r.company, r.headline, r.title, r.link, r.view, "
            f"r.analysts FROM reports r {clause} ORDER BY r.published_at DESC LIMIT ?",
            params + [FEED_LIMIT],
        ).fetchall()
        keys = [r[0] for r in rows]
        tickers_by_key: dict = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            for key, sym, exch in conn.execute(
                f"SELECT report_key, symbol, exchange FROM report_tickers "
                f"WHERE report_key IN ({','.join('?' * len(chunk))}) ORDER BY position", chunk,
            ):
                tickers_by_key.setdefault(key, []).append({"symbol": sym, "exchange": exch})

        top = conn.execute(
            "SELECT t.symbol, COUNT(DISTINCT t.report_key) n FROM report_tickers t "
            "JOIN reports r ON r.report_key = t.report_key WHERE t.position = 0 "
            f"AND t.report_key IN (SELECT report_key FROM report_tickers GROUP BY report_key "
            f"HAVING COUNT(*) <= {COMPANY_NOTE_MAX_TICKERS}) "
            + ("AND r.published_at >= ? " if since else "")
            + "GROUP BY t.symbol ORDER BY n DESC, t.symbol LIMIT ?",
            ([since] if since else []) + [TOP_TICKERS],
        ).fetchall()

        digests = conn.execute(
            "SELECT as_of, subscription, shown, total FROM digests "
            + ("WHERE as_of >= ? " if since else "")
            + "ORDER BY as_of DESC", [since] if since else [],
        ).fetchall()
    finally:
        conn.close()

    reports = []
    for key, published, company, headline, title, link, view, analysts in rows:
        reports.append({
            "published_at": published, "date": published[:10], "time": published[11:16],
            "company": company or title, "headline": headline, "link": link, "view": view,
            "analysts": analysts, "tickers": tickers_by_key.get(key, []),
        })
    return {
        "reports": reports,
        "digests": [{"date": a[:10], "subscription": sub, "shown": s, "total": t,
                     "missed": (t - s) if (s is not None and t is not None) else 0}
                    for a, sub, s, t in digests],
        "top_tickers": [{"symbol": s, "count": n} for s, n in top],
        "days": days, "ticker": ticker, "query": query,
        "total_reports": total, "truncated": total > len(reports),
    }
