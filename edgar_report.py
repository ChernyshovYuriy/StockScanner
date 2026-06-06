"""
edgar_report.py
===============
On-demand digest of ALREADY-COLLECTED EDGAR records — a separate command,
independent of the daily collector (edgar_service.py). It reads the fresh 13D
references the collector stored in `scan_hits`, parses the latest N filing bodies
(persisting them to `activist_filings`), and emails (or prints) the digest. It
does NOT scan EDGAR's daily index and does NOT change the daily collection logic.

Usage
-----
  python edgar_report.py                     # email the latest 20 fresh 13D
  python edgar_report.py --latest 50         # email the latest 50
  python edgar_report.py --latest 50 --dry-run   # print only, send nothing

Note: `scan_hits` is populated by any collector run (even --dry-run), so run
edgar_service.py at least once before expecting records here.
"""
from __future__ import annotations

import argparse
import uuid

from log_utils import log
from send_report import send_text_email
from time_utils import market_today_str

from edgar import store
from edgar.activist import dedup_by_accession, fetch_activist_filing
from edgar.digest import build_digest

# scan_hits.category for a fresh SC/SCHEDULE 13D (activist intent).
_FRESH_13D = "activist_stake"


def latest_activist(conn, n, enrich=True):
    """The N most recently filed fresh 13D records, newest first.

    Sourced from `scan_hits` (populated by every collector run, dry-run or not),
    not `activist_filings` (which only holds bodies already parsed). When
    `enrich`, each filing's body is fetched/parsed for filer+percent and the
    parsed evidence is persisted to `activist_filings`.
    """
    rows = conn.execute(
        "SELECT cik, ticker, form, filing_date, url FROM scan_hits "
        "WHERE category=? ORDER BY filing_date DESC, url DESC LIMIT ?",
        (_FRESH_13D, max(n * 4, n)),  # over-fetch; dedup collapses per-CIK dupes
    ).fetchall()
    hits = dedup_by_accession([
        {"cik": c, "ticker": t, "form": f, "date": d, "url": u}
        for (c, t, f, d, u) in rows
    ])[:n]

    if enrich:
        for h in hits:
            try:
                p = fetch_activist_filing(h["url"])
            except Exception:
                continue
            h["filer"], h["pct"] = p.get("filer"), p.get("pct")
            h["subject"], h["accession"] = p.get("subject"), p.get("accession")
            store.save_activist_filing(conn, {**h, "raw_text": p.get("raw_text")})
    return hits


def latest_insider(conn, n):
    """The N most recent stored insider open-market buys, newest first."""
    rows = conn.execute(
        "SELECT cik, owner, shares, price, txn_date "
        "FROM insider_buys ORDER BY txn_date DESC LIMIT ?",
        (n,),
    ).fetchall()
    return [{"cik": c, "owner": o, "shares": s, "price": p, "date": d}
            for (c, o, s, p, d) in rows]


def run_report(latest, dry_run=False, run_id=None):
    run_id = run_id or uuid.uuid4().hex
    conn = store.connect()
    activist = latest_activist(conn, latest)
    insiders = latest_insider(conn, latest)

    digest = build_digest(f"latest {latest} records (as of {market_today_str()})",
                          insiders, activist)
    if digest is None:
        print("No stored records to report — run edgar_service.py first.")
        log("edgar_report", run_id, "empty")
        return

    subject, body = digest
    if dry_run:
        print(subject)
        print(body)
        return

    sent = send_text_email(subject, body)
    log("edgar_report", run_id, "sent" if sent else "send_skipped",
        activist=len(activist), insiders=len(insiders))
    print(f"{'sent' if sent else 'NOT sent (Gmail not configured)'}: "
          f"{len(activist)} activist, {len(insiders)} insider")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Email a digest of already-collected EDGAR records (read-only)."
    )
    p.add_argument("--latest", type=int, default=20,
                   help="How many recent records of each kind to include (default 20).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the digest instead of emailing.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_report(args.latest, dry_run=args.dry_run)
