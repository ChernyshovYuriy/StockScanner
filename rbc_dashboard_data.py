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
tagger.py's labels (report_tags) are optional too: an untagged report
shows no labels, and a DB from before the tagger reads as all-untagged.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from config import RBC_REPORTS_DB_PATH
from rbc_reports.tagger import ACTIONS, REPORT_TYPES, THEMES, TONES

FEED_LIMIT = 500
TOP_TICKERS = 25
# A report listing more tickers than this is a sector note or a daily
# digest whose list is a peer list (often alphabetical), not its subject --
# kept out of the "Most covered" ranking. On the first 13 digests: 847
# reports had 1 ticker, 62 had 2, and 3+ were sector notes/morning digests.
COMPANY_NOTE_MAX_TICKERS = 2
# Filter dropdown choices -- the tagger's own vocabularies, so a filter can
# never offer a value it doesn't produce.
TAG_FILTERS = {"tone": TONES, "action": ACTIONS, "theme": THEMES, "report_type": REPORT_TYPES}
# Action filter shortcuts covering several actions at once.
ACTION_GROUPS = {
    "positive": ("upgrade", "pt_raise", "initiation", "results_beat"),
    "negative": ("downgrade", "pt_cut", "results_miss"),
}


def _connect(db_path):
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _empty_state(days, ticker, query, tags):
    return {"reports": [], "digests": [], "top_tickers": [], "themes": [], "days": days,
            "ticker": ticker, "query": query, "tags": tags, "total_reports": 0, "truncated": False,
            "tagged": False, "untagged": 0}


def _tag_filters(tone, action, theme, report_type) -> dict:
    """Only known values survive -- anything else (a hand-edited URL) is
    dropped rather than matching nothing."""
    out = {}
    for field, value in (("tone", tone), ("theme", theme), ("report_type", report_type)):
        value = (value or "").strip().lower()
        if value in TAG_FILTERS[field]:
            out[field] = value
    action = (action or "").strip().lower()
    if action in ACTION_GROUPS or action in ACTIONS:
        out["action"] = action
    return out


def build_rbc_state(days: int | None = 7, ticker: str | None = None, query: str | None = None,
                    db_path=RBC_REPORTS_DB_PATH, now: datetime | None = None,
                    tone: str | None = None, action: str | None = None, theme: str | None = None,
                    report_type: str | None = None) -> dict:
    """Feed rows for the last `days` days (None = all), optionally only one
    ticker symbol's reports or those matching a free-text `query` (company,
    headline, view or analyst). A ticker view ignores `days` -- it's that
    ticker's whole timeline. Also returns the digests' shown/total counts
    and the most-covered tickers over the same window -- counted only over
    company notes (see COMPANY_NOTE_MAX_TICKERS), by their first-listed
    ticker, so sector notes' peer lists don't swamp the ranking.
    tone/action/theme/report_type filter on the tagger's labels (action
    also takes an ACTION_GROUPS name); "themes" counts the window's tagged
    reports per theme with their bullish/bearish split."""
    ticker = (ticker or "").strip().upper() or None
    query = (query or "").strip() or None
    tags = _tag_filters(tone, action, theme, report_type)
    if not Path(db_path).exists():
        return _empty_state(days, ticker, query, tags)

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

    conn = _connect(db_path)
    try:
        tagged = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='report_tags'").fetchone() is not None
        if tags and not tagged:
            where.append("0")
        for field, value in (tags.items() if tagged else ()):
            if field == "action" and value in ACTION_GROUPS:
                group = ACTION_GROUPS[value]
                where.append(f"g.action IN ({','.join('?' * len(group))})")
                params += list(group)
            else:
                where.append(f"g.{field} = ?")
                params.append(value)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        join = "LEFT JOIN report_tags g ON g.report_key = r.report_key " if tagged else ""
        tag_cols = ("g.report_type, g.action, g.tone, g.rating, g.theme" if tagged
                    else "NULL, NULL, NULL, NULL, NULL")
        total = conn.execute(f"SELECT COUNT(*) FROM reports r {join}{clause}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT r.report_key, r.published_at, r.company, r.headline, r.title, r.link, r.view, "
            f"r.analysts, {tag_cols} FROM reports r {join}{clause} "
            f"ORDER BY r.published_at DESC LIMIT ?",
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

        themes, untagged = [], 0
        if tagged:
            window = "WHERE r.published_at >= ? " if since else ""
            themes = conn.execute(
                "SELECT g.theme, COUNT(*), SUM(g.tone = 'bullish'), SUM(g.tone = 'bearish') "
                f"FROM reports r JOIN report_tags g ON g.report_key = r.report_key {window}"
                "GROUP BY g.theme ORDER BY COUNT(*) DESC, g.theme", [since] if since else [],
            ).fetchall()
            untagged = conn.execute(
                "SELECT COUNT(*) FROM reports r WHERE NOT EXISTS "
                "(SELECT 1 FROM report_tags g WHERE g.report_key = r.report_key)").fetchone()[0]

        digests = conn.execute(
            "SELECT as_of, subscription, shown, total FROM digests "
            + ("WHERE as_of >= ? " if since else "")
            + "ORDER BY as_of DESC", [since] if since else [],
        ).fetchall()
    finally:
        conn.close()

    reports = []
    for (key, published, company, headline, title, link, view, analysts,
         rtype, act, tone_, rating, theme_) in rows:
        reports.append({
            "published_at": published, "date": published[:10], "time": published[11:16],
            "company": company or title, "headline": headline, "link": link, "view": view,
            "analysts": analysts, "tickers": tickers_by_key.get(key, []),
            "report_type": rtype, "action": act, "tone": tone_, "rating": rating, "theme": theme_,
        })
    return {
        "reports": reports,
        "digests": [{"date": a[:10], "subscription": sub, "shown": s, "total": t,
                     "missed": (t - s) if (s is not None and t is not None) else 0}
                    for a, sub, s, t in digests],
        "top_tickers": [{"symbol": s, "count": n} for s, n in top],
        "themes": [{"theme": t, "count": n, "bullish": b or 0, "bearish": x or 0}
                   for t, n, b, x in themes if t != "other"],
        "days": days, "ticker": ticker, "query": query, "tags": tags,
        "total_reports": total, "truncated": total > len(reports),
        "tagged": tagged, "untagged": untagged,
    }
