"""
rbc_reports_service.py
======================
RBC research tracker -- 12th StockScanner service (see rbc_reports/).
Pulls the latest RBC Insight digest emails from Gmail into
config.RBC_REPORTS_DIR (rbc_reports/gmail_fetch.py, read-only on the
mailbox), then parses every digest in that folder into
data/rbc_reports.db (rbc_reports/ingest.py), then labels every new
report with an LLM (rbc_reports/tagger.py; skipped if OPENAI_API_KEY is
unset), then -- once per trading day -- refreshes the price context of the
stocks behind bullish notes (rbc_reports/prices.py). No capital, no
positions, no email of its own. Scheduled via stockscanner-rbc-reports.timer (19:30 and
23:30 daily -- the digest is sent ~19:01 ET; the second run catches a late
one). The /rbc dashboard tab's "Fetch new digests" button runs the same
run_collector(), minus the price step (minutes of paced Yahoo requests --
too slow for a button; the next timer run does it).

  python rbc_reports_service.py              # fetch + ingest
  python rbc_reports_service.py --dry-run    # list what would be fetched, write nothing
"""
from __future__ import annotations

import argparse
import sys
import uuid

from concurrent_utils import acquire_lock
from config import RBC_GMAIL_FETCH_DAYS, RBC_REPORTS_DIR
from log_utils import log
from rbc_reports.gmail_fetch import fetch_digests
from rbc_reports import prices
from rbc_reports.ingest import ingest_dir
from rbc_reports.tagger import tag_untagged


def run_collector(run_id, dry_run=False, with_prices=True) -> dict:
    """Returns {"fetch": fetch_digests() result, "ingest": ingest_dir()
    result or None, "tags": tag_untagged() result or None, "prices":
    prices.run() result or None (not due today, dry run, or
    with_prices=False)}. An
    unconfigured Gmail still ingests whatever is already in the folder.
    The first run (no fetched digest in the folder yet) backfills every
    RBC digest in the mailbox; later runs look back RBC_GMAIL_FETCH_DAYS."""
    first_run = not any(RBC_REPORTS_DIR.glob("rbc_digest_*.html")) if RBC_REPORTS_DIR.is_dir() else True
    fetched = fetch_digests(days=None if first_run else RBC_GMAIL_FETCH_DAYS, dry_run=dry_run)
    if not fetched["configured"]:
        log("rbc_reports", run_id, "gmail_not_configured")
    else:
        log("rbc_reports", run_id, "fetched", backfill=first_run, messages=fetched["messages"],
            saved=len(fetched["saved"]), existing=fetched["existing"], skipped=fetched["skipped"])
    ingested = None
    if RBC_REPORTS_DIR.is_dir():
        ingested = ingest_dir(RBC_REPORTS_DIR, dry_run=dry_run)
        log("rbc_reports", run_id, "ingested", digests=ingested["digests"],
            reports_new=ingested["reports_new"], skipped=len(ingested["skipped"]))
    tags = None
    if ingested is not None:
        tags = tag_untagged(dry_run=dry_run)
        log("rbc_reports", run_id, "tagged", configured=tags["configured"],
            pending=tags["pending"], tagged=tags["tagged"])
    priced = None
    if ingested is not None and with_prices and not dry_run:
        try:
            priced = prices.run()
        except Exception as e:   # e.g. market_cache.db locked by another job
            log("rbc_reports", run_id, "prices_error", error=str(e))
        if priced is not None:
            log("rbc_reports", run_id, "prices", **priced["resolve"], **{
                f"sync_{k}": v for k, v in priced["sync"].items()})
    if dry_run:
        print(fetched)
        print(ingested)
        print(tags)
    return {"fetch": fetched, "ingest": ingested, "tags": tags, "prices": priced}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RBC research tracker -- 12th StockScanner service")
    p.add_argument("--dry-run", action="store_true",
                   help="Fetch + parse, print what would happen, write nothing.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    service = "rbc_reports"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "start", dry_run=args.dry_run)
    try:
        run_collector(run_id, dry_run=args.dry_run)
        log(service, run_id, "completed")
    except Exception as e:
        log(service, run_id, "error", error=str(e))
        raise
    finally:
        lock_file.close()
