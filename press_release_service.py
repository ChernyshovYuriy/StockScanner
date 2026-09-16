"""
press_release_service.py
==========================
Press-release wire tracker -- the 10th StockScanner service, structurally
independent of every trading sleeve (no capital, no positions -- same
"pure collector" shape as edgar_service.py / demand_signals_service.py).

Polls every feed in config.PRESS_RELEASE_FEEDS (starts with one --
GlobeNewswire's "News from Canada" feed; designed to take more feeds
later with no code change, just appending another URL to that list),
dedupes new items by guid (press_release_tracker/store.py), and parses
each new item with an LLM (press_release_tracker/llm_parser.py --
OPENAI_API_KEY optional; unconfigured means the raw RSS title/link still
go out, un-classified).

Two delivery lanes, split on the LLM's own materiality read:
  - "high"  -- emailed immediately, every run that finds one, no batching.
               Catching a market-moving catalyst fast is the point of this
               sleeve, so a high item never waits.
  - other   -- (medium/low/unclassified) accumulated and flushed as one
               digest at most every PRESS_RELEASE_BATCH_INTERVAL_MINUTES,
               so a busy newswire morning doesn't produce an email every
               5 minutes (the timer's own poll interval) for routine
               releases.
Both lanes share one subject prefix (press_release_tracker/digest.py's
SUBJECT_PREFIX, "Stock News Results") so every email this sleeve sends,
either lane, can be filtered on that one string in an email client.

A run that finds nothing to send in either lane sends nothing -- same
"quiet run = no email" convention as edgar_service.py/
demand_signals_service.py, unlike triple_screen_tracker_service.py's
once-daily "always email" choice.

Usage
-----
  python press_release_service.py              # poll all feeds, parse, email new items
  python press_release_service.py --dry-run     # poll + parse + print, write nothing, send nothing
"""
from __future__ import annotations

import argparse
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import requests

from concurrent_utils import acquire_lock
from config import PRESS_RELEASE_BATCH_INTERVAL_MINUTES, PRESS_RELEASE_FEEDS
from log_utils import log
from send_report import send_text_email
from time_utils import market_now

from press_release_tracker import digest, feeds, llm_parser, store

BATCH_INTERVAL = timedelta(minutes=PRESS_RELEASE_BATCH_INTERVAL_MINUTES)


def run_collector(run_id, dry_run=False, conn=None):
    """Event loop: fetch each feed, dedupe+store new items, parse each new
    item with the LLM, then route each not-yet-emailed item into one of
    two lanes -- materiality='high' emails immediately, everything else
    batches into one digest at most every BATCH_INTERVAL (see the module
    docstring).

    `conn` overrides the default store.connect() -- a test-only seam,
    mirroring triple_screen_tracker_service.run_collector's own `conn`
    injection.
    """
    conn = conn or store.connect()
    now_dt = market_now()
    now = now_dt.isoformat()

    new_items = []
    for feed_url in PRESS_RELEASE_FEEDS:
        try:
            items = feeds.fetch_feed_items(feed_url)
        except (requests.RequestException, ET.ParseError) as exc:
            log("press_release", run_id, "feed_fetch_failed", feed_url=feed_url, error=str(exc))
            continue
        for item in items:
            if store.is_seen(conn, item.guid):
                continue
            if not dry_run:
                store.mark_seen(conn, item, first_seen_at=now)
            new_items.append(item)
    log("press_release", run_id, "new_items", count=len(new_items))

    dry_run_rows = []
    for item in new_items:
        parsed = llm_parser.parse_release(item.title, item.description, item.categories)
        if parsed and not dry_run:
            store.save_parsed(conn, item.guid, parsed, llm_parser.MODEL, now)
        dry_run_rows.append({
            "guid": item.guid, "feed_url": item.feed_url, "title": item.title,
            "link": item.link, "pubdate": item.pubdate,
            **(parsed or {"ticker": None, "company": None, "category": None,
                           "materiality": None, "summary": None}),
        })

    rows = dry_run_rows if dry_run else store.unemailed(conn)
    if not rows:
        log("press_release", run_id, "quiet")
        return

    high_rows = [r for r in rows if r.get("materiality") == "high"]
    batch_rows = [r for r in rows if r.get("materiality") != "high"]

    # dry-run never persists batch_state (no DB writes at all), so the
    # hourly gate always reads as "due" -- a preview run should show both
    # lanes' full pending content, not simulate the real-world wait.
    last_sent = None if dry_run else store.get_last_batch_sent_at(conn)
    batch_due = last_sent is None or (now_dt - datetime.fromisoformat(last_sent)) >= BATCH_INTERVAL

    if high_rows:
        subject, body = digest.build_digest(high_rows, kind="high")
        if dry_run:
            print(subject)
            print(body)
        else:
            sent = send_text_email(subject, body)
            if sent:
                store.mark_emailed(conn, [r["guid"] for r in high_rows])
            log("press_release", run_id, "high_sent" if sent else "high_send_skipped",
                count=len(high_rows))

    if batch_rows and batch_due:
        subject, body = digest.build_digest(batch_rows, kind="batch")
        if dry_run:
            print(subject)
            print(body)
        else:
            sent = send_text_email(subject, body)
            if sent:
                store.mark_emailed(conn, [r["guid"] for r in batch_rows])
                store.set_last_batch_sent_at(conn, now)
            log("press_release", run_id, "batch_sent" if sent else "batch_send_skipped",
                count=len(batch_rows))
    elif batch_rows:
        log("press_release", run_id, "batch_pending", count=len(batch_rows))

    if dry_run:
        log("press_release", run_id, "dry_run", high=len(high_rows), batch=len(batch_rows))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Press-release wire tracker -- 10th StockScanner service")
    p.add_argument("--dry-run", action="store_true",
                   help="Poll + parse + print the digest without writing to the DB or sending email.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    service = "press_release"
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
