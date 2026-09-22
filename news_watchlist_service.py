"""
news_watchlist_service.py
===========================
News watchlist — the 11th StockScanner service (see news_watchlist/__init__.py).

Two independent jobs, run on two different schedules via `--mode` (same
split-schedule-single-script precedent as position_monitor.py's
`--mode pre-close`/`--mode post-close`):

  --mode seed           Frequent (every ~10 minutes, all day, every day --
                         no trading-day gate, same reasoning as
                         press_release_tracker's own "no market-hours gate":
                         news can break any time). Reads data/press_releases.db
                         READ-ONLY for parsed_releases rows with a non-null
                         ticker not already seeded into this watchlist and
                         no older than NEWS_WATCHLIST_MAX_ARTICLE_AGE_DAYS
                         (config.py), and inserts each as a fresh
                         status='inbox' candidate stamped with today's
                         price. If it finds anything new, emails an
                         immediate alert (news_watchlist/
                         digest.py) -- a direct nudge to go review the
                         watchlist tab, separate from press_release_tracker's
                         own email for the same underlying catalyst (the
                         user explicitly asked to keep getting notified
                         fast, even with the dashboard tab in place).

  --mode update-prices   Once daily (~17:10 ET, after triple_screen_tracker
                         's 17:00 run, before scanner_pipeline's 17:15;
                         trading-day gated, since a daily price-history row
                         only makes sense on a trading day). Appends today's
                         price to every already-confirmed status='watching'
                         item's price history. Inbox items get NO price
                         tracking until a human confirms them from the
                         dashboard -- see news_watchlist/__init__.py for why
                         that gate exists.

  --mode both (default)  Runs both steps once, in order -- convenient for
                         manual testing / --dry-run; not what either
                         scheduled timer actually uses.

Usage
-----
  python news_watchlist_service.py --mode seed              # frequent job
  python news_watchlist_service.py --mode update-prices      # daily job
  python news_watchlist_service.py --dry-run                 # both steps, print, write/send nothing
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from datetime import date, timedelta, timezone
from email.utils import parsedate_to_datetime

from concurrent_utils import acquire_lock
from config import NEWS_WATCHLIST_MAX_ARTICLE_AGE_DAYS, PRESS_RELEASE_DB_PATH
from log_utils import log
from manual_sell import get_market_price
from send_report import send_text_email
from time_utils import TSX_HOLIDAYS, market_now, market_today_str

from news_watchlist import digest, store


def _is_trading_day(today: date) -> bool:
    return today.weekday() < 5 and today.isoformat() not in TSX_HOLIDAYS


def _resolve_market_price(ticker: str) -> tuple[float, str] | tuple[None, None]:
    """get_market_price(), but first disambiguates a bare ticker (no
    exchange suffix) -- press_release_tracker/llm_parser.py extracts
    whatever symbol the release text states with no Yahoo-Finance-suffix
    awareness (see its own docstring; that's by design, not a parser
    bug), and on Yahoo Finance a bare Canadian symbol commonly collides
    with an unrelated US-listed security of the same letters (e.g. "AYA"
    resolves to a different company than TSX-listed "AYA.TO"; "ARE" to
    Alexandria Real Estate instead of TSX-listed Aecon Group). Since this
    feed is GlobeNewswire's "News from Canada", a bare symbol is tried as
    TSX main board (.TO) then TSX Venture (.V) first, falling back to the
    bare symbol only if neither suffixed form has data -- a genuinely
    non-Canadian name (e.g. "XENE", Nasdaq-only) still resolves
    correctly. A ticker already carrying a suffix (e.g. "KTO.V", already
    correct from the parser) is passed through unchanged."""
    if "." in ticker:
        return get_market_price(ticker)
    for suffix in (".TO", ".V"):
        price, source = get_market_price(ticker + suffix)
        if price is not None:
            return price, source
    return get_market_price(ticker)


def _is_stale(pubdate: str, now, max_age_days: int) -> bool:
    """True if pubdate (a seen_items.pubdate value -- the feed's own RFC
    822 <pubDate>, see press_release_tracker/feeds.py) is older than
    max_age_days relative to now. An empty or unparseable pubdate is
    treated as NOT stale -- a parse edge case shouldn't silently drop a
    genuine candidate."""
    if not pubdate:
        return False
    try:
        published = parsedate_to_datetime(pubdate)
    except (TypeError, ValueError):
        return False
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return (now - published) > timedelta(days=max_age_days)


def _read_parsed_candidates(pr_db_path) -> list[dict]:
    """Every parsed_releases row with a non-null ticker, joined against
    seen_items for its link and pubdate -- [] if press_releases.db doesn't
    exist yet (press_release_service.py hasn't run), same "not yet
    available" convention every read-only cross-DB reader in this repo
    uses."""
    try:
        conn = sqlite3.connect(f"file:{pr_db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return []
    try:
        rows = conn.execute(
            "SELECT p.guid, p.ticker, p.company, p.category, p.materiality, p.summary, "
            "       s.link, s.pubdate "
            "FROM parsed_releases p JOIN seen_items s ON s.guid = p.guid "
            "WHERE p.ticker IS NOT NULL AND p.ticker != ''"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    cols = ["guid", "ticker", "company", "category", "materiality", "summary", "link", "pubdate"]
    return [dict(zip(cols, r)) for r in rows]


def seed_inbox(run_id, conn, pr_db_path, price_fetcher, dry_run=False) -> list[dict]:
    """Insert every not-yet-processed parsed_releases candidate as a fresh
    inbox item -- or, if that ticker already has an unreviewed inbox item
    (store.find_pending_inbox_item()), collapse into that row instead
    (store.update_inbox_item()) so a busy ticker (several procedural
    filings for the same story) doesn't flood the inbox with one row per
    article. A candidate whose own pubDate is older than
    NEWS_WATCHLIST_MAX_ARTICLE_AGE_DAYS is skipped entirely (marked
    processed, never seeded) -- this service is about fast follow-through
    on a fresh catalyst, and a stale article surfacing late (e.g. after
    the service was down for a while) has no such catalyst left to follow.
    Returns the list of newly seeded/updated items (ticker/company/
    category/materiality/summary/flag_price) for the caller to alert on --
    [] if nothing new."""
    today_str = market_today_str()
    now_dt = market_now()
    now = now_dt.isoformat()
    already_seeded = store.seeded_guids(conn)
    candidates = [c for c in _read_parsed_candidates(pr_db_path) if c["guid"] not in already_seeded]

    seeded = []
    for c in candidates:
        if _is_stale(c["pubdate"], now_dt, NEWS_WATCHLIST_MAX_ARTICLE_AGE_DAYS):
            log("news_watchlist", run_id, "candidate_too_stale", ticker=c["ticker"], pubdate=c["pubdate"])
            if not dry_run:
                store.mark_guid_processed(conn, c["guid"], now)
            continue
        try:
            price, _source = price_fetcher(c["ticker"])
        except Exception as e:
            log("news_watchlist", run_id, "price_fetch_error", ticker=c["ticker"], error=str(e))
            continue
        if price is None:
            log("news_watchlist", run_id, "price_unavailable", ticker=c["ticker"])
            continue
        if not dry_run:
            existing = store.find_pending_inbox_item(conn, c["ticker"])
            if existing:
                store.update_inbox_item(
                    conn, existing["id"], guid=c["guid"], company=c["company"],
                    category=c["category"], materiality=c["materiality"], summary=c["summary"],
                    source_link=c["link"], flagged_at=today_str, flag_price=price,
                    created_at=now,
                )
            else:
                store.seed_inbox_item(
                    conn, guid=c["guid"], ticker=c["ticker"], company=c["company"],
                    category=c["category"], materiality=c["materiality"], summary=c["summary"],
                    source_link=c["link"], flagged_at=today_str, flag_price=price, created_at=now,
                )
            store.mark_guid_processed(conn, c["guid"], now)
        seeded.append({
            "ticker": c["ticker"], "company": c["company"], "category": c["category"],
            "materiality": c["materiality"], "summary": c["summary"], "flag_price": price,
        })
    return seeded


def update_watching_prices(run_id, conn, price_fetcher, dry_run=False) -> list[dict]:
    """Append today's price to every status='watching' item. Returns the
    list of items actually priced this run."""
    today_str = market_today_str()
    updated = []
    for item in store.list_by_status(conn, "watching"):
        try:
            price, _source = price_fetcher(item["ticker"])
        except Exception as e:
            log("news_watchlist", run_id, "price_fetch_error", ticker=item["ticker"], error=str(e))
            continue
        if price is None:
            log("news_watchlist", run_id, "price_unavailable", ticker=item["ticker"])
            continue
        if not dry_run:
            store.append_price(conn, item["id"], today_str, price)
        updated.append({"ticker": item["ticker"], "price": price})
    return updated


def run_collector(run_id, mode="both", dry_run=False, conn=None, press_release_db_path=None,
                   price_fetcher=None):
    """Dispatches to seed_inbox()/update_watching_prices() per `mode` (see
    module docstring), then handles the seed step's immediate-alert email.

    `conn` overrides the default store.connect(), `press_release_db_path`
    overrides PRESS_RELEASE_DB_PATH, `price_fetcher` overrides
    _resolve_market_price() -- test-only seams, same injection pattern
    triple_screen_tracker_service.run_collector's `provider`/`conn` use.
    """
    conn = conn or store.connect()
    pr_db_path = press_release_db_path or PRESS_RELEASE_DB_PATH
    price_fetcher = price_fetcher or _resolve_market_price

    if mode in ("seed", "both"):
        seeded = seed_inbox(run_id, conn, pr_db_path, price_fetcher, dry_run=dry_run)
        log("news_watchlist", run_id, "seeded", count=len(seeded))
        if seeded:
            subject, body = digest.build_digest(seeded)
            if dry_run:
                print(subject)
                print(body)
            else:
                sent = send_text_email(subject, body)
                log("news_watchlist", run_id, "alert_sent" if sent else "alert_send_skipped",
                    count=len(seeded))

    if mode in ("update-prices", "both"):
        today = market_now().date()
        if _is_trading_day(today):
            updated = update_watching_prices(run_id, conn, price_fetcher, dry_run=dry_run)
            log("news_watchlist", run_id, "updated", count=len(updated))
            if dry_run:
                print(f"Would update {len(updated)} watching item(s): {[u['ticker'] for u in updated]}")
        else:
            log("news_watchlist", run_id, "not_a_trading_day", date=today.isoformat())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="News watchlist — 11th StockScanner service")
    p.add_argument("--mode", choices=["seed", "update-prices", "both"], default="both",
                   help="seed: check press_releases.db for fresh candidates, email an immediate "
                        "alert if any (the frequent timer). update-prices: append today's price to "
                        "every confirmed watching item (the once-daily timer). both (default): run "
                        "each step once -- manual/dry-run convenience, not what either timer uses.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run the selected step(s) and print what would happen without writing to "
                        "the DB or sending email.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    service = "news_watchlist"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "start", mode=args.mode, dry_run=args.dry_run)
    try:
        run_collector(run_id, mode=args.mode, dry_run=args.dry_run)
        log(service, run_id, "completed")
    except Exception as e:
        log(service, run_id, "error", error=str(e))
        raise
    finally:
        lock_file.close()
