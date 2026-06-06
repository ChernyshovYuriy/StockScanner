"""
edgar_report.py
===============
On-demand digest of ALREADY-COLLECTED EDGAR records — a separate, read-only
command, independent of the daily collector (edgar_service.py). It reads
data/edgar.db and emails (or prints) the latest N stored records. It does NOT
scan EDGAR and does NOT change the daily collection logic — it only reports what
the collector has already stored.

Usage
-----
  python edgar_report.py                     # email the latest 20 activist filings
  python edgar_report.py --latest 50         # email the latest 50
  python edgar_report.py --latest 50 --dry-run   # print only, send nothing

Note: activist_filings are populated by real (non-dry-run) collector runs, so
run edgar_service.py at least once before expecting records here.
"""
from __future__ import annotations

import argparse
import uuid

from log_utils import log
from send_report import send_text_email
from time_utils import market_today_str

from edgar import store
from edgar.digest import build_digest


def latest_activist(conn, n):
    """The N most recently filed activist (13D/G) records, newest first."""
    rows = conn.execute(
        "SELECT ticker, cik, form, filer, pct, filing_date, url "
        "FROM activist_filings ORDER BY filing_date DESC, accession DESC LIMIT ?",
        (n,),
    ).fetchall()
    return [{"ticker": t, "cik": c, "form": f, "filer": fl, "pct": p,
             "date": d, "url": u} for (t, c, f, fl, p, d, u) in rows]


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
