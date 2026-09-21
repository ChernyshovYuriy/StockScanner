"""
news_watchlist/digest.py
==========================
Plain-text email for the News Watchlist's own immediate alert: fires
whenever news_watchlist_service.py's `seed` step (see its module docstring)
lands one or more fresh candidates in the inbox. This is a DIFFERENT email
from press_release_tracker's own (see press_release_tracker/digest.py) --
the user explicitly asked to still get a fast, direct nudge that something
landed in the watchlist specifically, even though press_release_tracker
already emailed the same underlying catalyst. Distinct subject prefix
(not press_release_tracker's "Stock News Results") so the two are
filterable separately in an email client.
"""
from __future__ import annotations

SUBJECT_PREFIX = "News Watchlist"


def build_digest(rows: list[dict]) -> tuple[str, str]:
    """rows: dicts with ticker/company/category/materiality/summary/flag_price
    (see news_watchlist_service.seed_inbox's return shape). Returns
    (subject, body)."""
    n = len(rows)
    plural = "s" if n != 1 else ""
    subject = f"{SUBJECT_PREFIX} — {n} new item{plural} to review"

    lines = []
    for r in rows:
        header = r.get("ticker") or "(ticker unknown)"
        company = f" — {r['company']}" if r.get("company") else ""
        lines.append(f"[{header}]{company}")
        if r.get("category") or r.get("materiality"):
            lines.append(f"  category: {r.get('category')}  materiality: {r.get('materiality')}")
        if r.get("summary"):
            lines.append(f"  {r['summary']}")
        if r.get("flag_price") is not None:
            lines.append(f"  flagged at ${r['flag_price']:.2f}")
        lines.append("")
    lines.append("Review and confirm/dismiss at /news-watchlist.")

    body = "\n".join(lines)
    return subject, body
