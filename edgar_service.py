"""
edgar_service.py
================
EDGAR collector — the 4th StockScanner service (US SEC filings).

Runs daily after the US market close (Mon-Fri, ~6:30pm ET): sweeps EDGAR's daily
index for forms of interest, stores them in a SEPARATE SQLite DB (data/edgar.db),
and emails a plain-text digest of FLAGGED HITS ONLY through StockScanner's shared
Gmail sender. Quiet day -> no email.

This is the fundamentals/ownership counterweight to the TSX momentum system: it
surfaces *footprints* of big money (insider open-market buys, activist stakes)
earlier and more systematically than the crowd. Every filing is a lagged
disclosure — a research trigger, never a price predictor or financial advice.

Interim flagging (Step 2 deferred): watchlist insider open-market buys + all
SC 13D/13G market-wide. Set a watchlist with:
    python -m edgar.run watchlist MU,KEY,AMD

Usage
-----
  python edgar_service.py              # daily run (scan, store, email digest)
  python edgar_service.py --dry-run    # build & print the digest, send nothing
  python edgar_service.py --days 10    # widen the storage backfill window
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import timedelta

from concurrent_utils import acquire_lock
from config import EDGAR_BACKFILL_DAYS, EDGAR_FORMS
from log_utils import log
from send_report import send_text_email
from time_utils import TSX_TZ, market_now

from edgar import store
from edgar.activist import accession_from_url, fetch_activist_filing
from edgar.core import load_cik_to_ticker
from edgar.digest import build_digest
from edgar.insiders import get_recent_insider_activity, open_market_buys
from edgar.scanner import scan_range


def _dedup_by_accession(hits):
    """Collapse the per-CIK duplicate index rows for one filing (same accession).

    A 13D is listed under both the filer and the subject-company CIK; keep one,
    preferring the row that resolved a ticker (the subject company).
    """
    by_acc = {}
    for h in hits:
        acc = accession_from_url(h.get("url", ""))
        cur = by_acc.get(acc)
        if cur is None or (not cur.get("ticker") and h.get("ticker")):
            by_acc[acc] = h
    return list(by_acc.values())


def run_collector(run_id, dry_run=False, backfill_days=None):
    """Event loop: scan the daily index, store hits, build + send the digest.

    The scan is market-wide (the whole point is catching names you don't watch),
    but the EMAIL flags only the flag date's hits: insider open-market buys for
    watchlisted CIKs + all SC 13D/13G. The flag date is normally today, but
    today's master.idx is often not yet published when this runs, so it falls
    back to the most recent day in the scan window that fetched successfully
    and hasn't already been sent (see flag_date below) — otherwise that day's
    hits would land in storage on tomorrow's backfill but never get emailed,
    since tomorrow's "today" no longer matches their filing date. The 5-day
    backfill feeds STORAGE so the DB self-heals after downtime; only the flag
    date's flagged hits are emailed.
    """
    backfill = backfill_days or EDGAR_BACKFILL_DAYS
    # ET ~ US Eastern; the existing services already run on this clock.
    today = market_now(TSX_TZ).date()
    today_str = today.isoformat()
    # Generous calendar window so the last `backfill` business days are covered;
    # scan_range skips weekends and tolerates a missing/partial daily index.
    start = today - timedelta(days=backfill + 4)

    conn = store.connect()

    hits = scan_range(start, today, watchlist_ciks=None, forms=list(EDGAR_FORMS))
    real = [h for h in hits if "error" not in h]
    for e in (h for h in hits if "error" in h):
        # Missing-index tolerance: skip + log, never crash.
        log("edgar", run_id, "index_skipped", date=e.get("date"), error=e.get("error"))
    store.save_scan_hits(conn, real)
    log("edgar", run_id, "scanned", window_days=(today - start).days, stored_hits=len(real))

    # Pick the flag date: the most recent day with successfully-scanned data
    # that hasn't already been sent. Usually today, but falls back when today's
    # index isn't published yet (see docstring above).
    candidate_dates = sorted({h["date"] for h in real if h.get("date")}, reverse=True)
    flag_date = next(
        (d for d in candidate_dates if not store.already_sent(conn, d)), today_str
    )

    # Idempotent same-flag-date re-run guard.
    if not dry_run and store.already_sent(conn, flag_date):
        log("edgar", run_id, "already_sent_today", date=flag_date)
        return

    # ── Activist: the EMAIL flags FRESH 13D only (activist intent, highest
    #    signal). 13D/A amendments and all 13G are already collected in scan_hits
    #    as references for the analysis layer; they are not emailed. Each filing
    #    is indexed under both the filer and subject CIK, so dedup by accession.
    #    Fetch + parse each flagged body for filer / percent and keep the raw
    #    text as evidence; one bad filing must not sink the run, so guard it. ──
    fresh_13d = _dedup_by_accession([
        h for h in real
        if h.get("category") == "activist_stake" and h.get("date") == flag_date
    ])
    activist_hits = []
    for h in fresh_13d:
        try:
            parsed = fetch_activist_filing(h.get("url", ""))
        except Exception as exc:
            log("edgar", run_id, "activist_parse_failed", url=h.get("url"), error=str(exc))
            parsed = {}
        h["filer"] = parsed.get("filer")
        h["pct"] = parsed.get("pct")
        h["subject"] = parsed.get("subject")
        h["accession"] = parsed.get("accession") or accession_from_url(h.get("url", ""))
        if not dry_run:
            store.save_activist_filing(conn, {**h, "raw_text": parsed.get("raw_text")})
        activist_hits.append(h)

    # ── Insider buys: watchlist CIKs with a Form 4 on the flag date (bounded fetch) ──
    insider_buys = []
    watchlist = store.watchlist_ciks(conn)
    if watchlist:
        cik2tic = load_cik_to_ticker()
        todays_form4_ciks = {
            h["cik"] for h in real
            if h.get("form") == "4" and h.get("date") == flag_date and h["cik"] in watchlist
        }
        for cik in sorted(todays_form4_ciks):
            buys = open_market_buys(get_recent_insider_activity(cik))
            todays = [b for b in buys if b.get("filing_date") == flag_date]
            if todays:
                store.save_insider_buys(conn, cik, todays)
            for b in todays:
                b["cik"] = cik
                b["ticker"] = cik2tic.get(cik)
                insider_buys.append(b)

    # ── Build + send digest (quiet day = silence) ─────────────────────────────
    digest = build_digest(flag_date, insider_buys, activist_hits)
    hit_count = len(insider_buys) + len(activist_hits)

    if digest is None:
        log("edgar", run_id, "quiet_day", date=flag_date)
        if not dry_run:
            store.record_email(conn, flag_date, 0, sent=0)
        return

    subject, body = digest
    if dry_run:
        print(subject)
        print(body)
        log("edgar", run_id, "dry_run", insiders=len(insider_buys), activist=len(activist_hits))
        return

    sent = send_text_email(subject, body)
    store.record_email(conn, flag_date, hit_count, sent=sent)
    log("edgar", run_id, "sent" if sent else "send_skipped",
        insiders=len(insider_buys), activist=len(activist_hits))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EDGAR collector — 4th StockScanner service")
    p.add_argument("--dry-run", action="store_true",
                   help="Build and print the digest without sending or recording.")
    p.add_argument("--days", type=int, default=None,
                   help=f"Storage backfill window in business days (default {EDGAR_BACKFILL_DAYS}).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    service = "edgar"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "start", dry_run=args.dry_run)
    try:
        run_collector(run_id, dry_run=args.dry_run, backfill_days=args.days)
        log(service, run_id, "completed")
    except Exception as e:
        # Surface failures to the journal so a silent break doesn't go unnoticed.
        log(service, run_id, "error", error=str(e))
        raise
    finally:
        lock_file.close()
