"""
press_release_tracker/archive.py
==================================
Historical backfill (added 2026-10): runs the press-release pipeline over
GlobeNewswire's past Canadian-listed releases so the outcome report
(news_watchlist/outcomes.py) has thousands of events instead of the few
dozen the live feed has gathered since 2026-09.

Source: GlobeNewswire's monthly English sitemaps
(sitemaps.globenewswire.com/news/en/YYYY-MM.xml, back to 2023-11) -- the
crawler-sanctioned listing (robots.txt disallows /search, allows
/news-release/). Each entry has url, publication time, title and
exchange-qualified tickers ("TSX Venture Exchange:ABC"), so the Yahoo
symbol comes straight from the exchange instead of the live path's
.TO/.V guessing.

Everything lives in its OWN DB (config.PRESS_RELEASE_ARCHIVE_DB_PATH),
never read by a live service: press_release_tracker/store.py's tables
(seen_items/parsed_releases/financing_terms) plus news_watchlist/store.py's
release_outcomes, in one file -- the table names don't collide.

Steps (python -m press_release_tracker.archive <step>):
  list       download the monthly sitemaps (cached; only the newest month
             is re-fetched) and import Canadian-listed releases
  classify   batch-classify titles with the cheap model (40 titles/call).
             Titles only -- the sitemap carries no teaser text; financings
             nearly always say so in the title
  financing  full article + financing.extract_terms() for every
             'financing' release, with the share count AT THE TIME (Yahoo
             shares history), not today's. No analyst verdicts on old
             releases: the model may know what happened next
  score      forward returns via news_watchlist_service.score_release_outcomes()
  report     outcome report + how many financings were lost to missing
             Yahoo data (survivorship -- a delisted junior is usually a
             bad outcome, so the report flatters)
  all        every step in order
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Optional

import requests

from press_release_tracker import article, financing, store
from press_release_tracker.llm_parser import OPENAI_API_KEY

try:
    from config import PRESS_RELEASE_ARCHIVE_CACHE_PATH as CACHE_PATH
    from config import PRESS_RELEASE_ARCHIVE_DB_PATH as DB_PATH
    from config import PRESS_RELEASE_LLM_MODEL as CLASSIFY_MODEL
    from config import PRESS_RELEASE_USER_AGENT as USER_AGENT
except Exception:
    _ROOT = Path(__file__).resolve().parent.parent
    DB_PATH = _ROOT / "data" / "press_release_archive.db"
    CACHE_PATH = _ROOT / "cache" / "press_release_archive"
    CLASSIFY_MODEL = "gpt-5-nano"
    USER_AGENT = "StockScanner-PressRelease/0.1"

SITEMAP_URL = "https://sitemaps.globenewswire.com/news/en/{month}.xml"
FIRST_MONTH = "2023-11"
# The live feed starts 2026-09-16 (data/press_releases.db) -- the archive
# stops before it so the two never overlap.
LAST_MONTH = "2026-08"
FEED_URL = "archive:globenewswire-sitemap"

_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
       "news": "http://www.google.com/schemas/sitemap-news/0.9"}

# Exchange name as GlobeNewswire writes it -> Yahoo suffix, in order of
# preference when a release lists several Canadian lines.
EXCHANGE_SUFFIX = {
    "Toronto Stock Exchange": ".TO",
    "TSX Venture Exchange": ".V",
    "Canadian Stock Exchange": ".CN",
    "Aequitas Neo Exchange": ".NE",
}
_ALL_SUFFIXES = (".TO", ".V", ".CN", ".NE")

# A common-share line: base symbol plus an optional share-class/unit tag.
# Warrants (WT), preferreds (PR/P?), debentures (DB), rights (R) are skipped.
_SYMBOL_RE = re.compile(r"^([A-Z0-9]+)(?:[.-](UN|U|A|B|H))?$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS archive_symbols (
    guid TEXT PRIMARY KEY,
    candidates TEXT NOT NULL,
    raw_tickers TEXT
);
"""

_CATEGORIES = ("exploration_drilling", "financing", "ma_acquisition", "earnings",
               "contract_award", "regulatory", "personnel", "other")

_CLASSIFY_PROMPT = (
    "You classify Canadian company press-release HEADLINES. For each item, reply with its "
    "category (one of: " + ", ".join(_CATEGORIES) + ") and materiality (one of: high, medium, "
    "low -- how likely this news is to move the stock's price). financing means the company "
    "raising money: private placements, LIFE offerings, bought deals, public offerings, "
    "flow-through, units, convertible debentures, notes or loans -- including closings and "
    "amendments of them. A share buyback (normal course issuer bid) or dividend is not "
    "financing. Reply with ONLY a JSON object: {\"items\": [{\"id\": <id>, \"category\": ..., "
    "\"materiality\": ...}, ...]} with one entry per input id."
)
CLASSIFY_BATCH = 40


# ── sitemap parsing ──────────────────────────────────────────────────────

def yahoo_candidates(stock_tickers: str) -> list[str]:
    """Yahoo symbols for a sitemap <news:stock_tickers> value, best first:
    the first Canadian common-share line (exchange preference order) with
    its own suffix, then the same base on the other Canadian exchanges (a
    company that moved boards since). [] if no Canadian common line."""
    lines = []
    for part in (stock_tickers or "").split(","):
        if ":" not in part:
            continue
        exchange, symbol = (p.strip() for p in part.split(":", 1))
        if exchange not in EXCHANGE_SUFFIX:
            continue
        for suffix in _ALL_SUFFIXES:          # some entries already carry one ("HG.CN")
            if symbol.endswith(suffix):
                symbol = symbol[: -len(suffix)]
                break
        m = _SYMBOL_RE.match(symbol.upper())
        if not m:
            continue
        base = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
        lines.append((list(EXCHANGE_SUFFIX).index(exchange), base, EXCHANGE_SUFFIX[exchange]))
    if not lines:
        return []
    _, base, suffix = min(lines)
    return [base + suffix] + [base + s for s in _ALL_SUFFIXES if s != suffix]


def parse_sitemap(xml_text: str) -> list[dict]:
    """Canadian-listed entries of one monthly sitemap:
    [{link, title, published (aware UTC datetime), candidates, raw_tickers}]."""
    root = ET.fromstring(xml_text)
    out = []
    for url in root.findall("sm:url", _NS):
        link = url.findtext("sm:loc", default="", namespaces=_NS).strip()
        news = url.find("news:news", _NS)
        if news is None or not link:
            continue
        raw = news.findtext("news:stock_tickers", default="", namespaces=_NS)
        candidates = yahoo_candidates(raw)
        if not candidates:
            continue
        pub = news.findtext("news:publication_date", default="", namespaces=_NS).strip()
        try:
            published = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        except ValueError:
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        out.append({
            "link": link,
            "title": (news.findtext("news:title", default="", namespaces=_NS) or "").strip(),
            "published": published,
            "candidates": candidates,
            "raw_tickers": raw,
        })
    return out


def months(first: str, last: str) -> list[str]:
    y, m = map(int, first.split("-"))
    ly, lm = map(int, last.split("-"))
    out = []
    while (y, m) <= (ly, lm):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# ── DB ───────────────────────────────────────────────────────────────────

def connect(db_path=None):
    conn = store.connect(db_path or DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def import_entries(conn, entries: list[dict], now: str) -> int:
    """Insert entries not yet in the DB. guid = the release URL. Returns
    the number added."""
    added = 0
    for e in entries:
        if store.is_seen(conn, e["link"]):
            continue
        conn.execute(
            "INSERT OR IGNORE INTO seen_items"
            "(guid,feed_url,title,link,pubdate,description,categories,first_seen_at,emailed) "
            "VALUES(?,?,?,?,?,'','',?,1)",
            (e["link"], FEED_URL, e["title"], e["link"], format_datetime(e["published"]), now),
        )
        conn.execute(
            "INSERT OR REPLACE INTO archive_symbols(guid,candidates,raw_tickers) VALUES(?,?,?)",
            (e["link"], "|".join(e["candidates"]), e["raw_tickers"]),
        )
        added += 1
    conn.commit()
    return added


def candidates_by_guid(conn) -> dict:
    return {g: c.split("|") for g, c in conn.execute("SELECT guid, candidates FROM archive_symbols")}


# ── steps ────────────────────────────────────────────────────────────────

def _fetch_sitemap(month: str, refresh: bool) -> str:
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    path = CACHE_PATH / f"{month}.xml"
    if path.exists() and not refresh:
        return path.read_text()
    resp = requests.get(SITEMAP_URL.format(month=month), headers={"User-Agent": USER_AGENT}, timeout=120)
    resp.raise_for_status()
    path.write_text(resp.text)
    time.sleep(1.0)
    return resp.text


def step_list(conn, first=FIRST_MONTH, last=LAST_MONTH) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for month in months(first, last):
        entries = parse_sitemap(_fetch_sitemap(month, refresh=(month == last)))
        added = import_entries(conn, entries, now)
        print(f"{month}: {len(entries)} Canadian-listed, {added} new")


def _unclassified(conn) -> list[tuple]:
    return conn.execute(
        "SELECT s.guid, s.title FROM seen_items s LEFT JOIN parsed_releases p ON p.guid = s.guid "
        "WHERE s.feed_url = ? AND p.guid IS NULL ORDER BY s.pubdate", (FEED_URL,)
    ).fetchall()


def classify_batch(client, titles: list[str]) -> Optional[dict]:
    """{index: (category, materiality)} for one batch of titles, or None
    on a failed call. Unknown categories/materialities fall back to
    other/low."""
    payload = json.dumps([{"id": i, "title": t} for i, t in enumerate(titles)], ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[{"role": "system", "content": _CLASSIFY_PROMPT},
                      {"role": "user", "content": payload}],
            response_format={"type": "json_object"},
            timeout=180,
        )
        items = json.loads(resp.choices[0].message.content).get("items")
    except Exception:
        return None
    out = {}
    for it in items or []:
        try:
            i = int(it.get("id"))
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(titles):
            cat = it.get("category") if it.get("category") in _CATEGORIES else "other"
            mat = it.get("materiality") if it.get("materiality") in ("high", "medium", "low") else "low"
            out[i] = (cat, mat)
    return out


def step_classify(conn, workers: int = 6, limit: Optional[int] = None) -> None:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    rows = _unclassified(conn)[:limit] if limit else _unclassified(conn)
    symbols = candidates_by_guid(conn)
    batches = [rows[i:i + CLASSIFY_BATCH] for i in range(0, len(rows), CLASSIFY_BATCH)]
    print(f"{len(rows)} titles to classify in {len(batches)} batches")
    now = datetime.now(timezone.utc).isoformat()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (batch, result) in enumerate(
                pool.map(lambda b: (b, classify_batch(client, [t for _, t in b])), batches), 1):
            if result is None:
                print(f"batch {n}/{len(batches)}: FAILED (will retry on the next run)")
                continue
            for i, (guid, _title) in enumerate(batch):
                if i not in result:
                    continue
                cat, mat = result[i]
                store.save_parsed(conn, guid, {"ticker": symbols[guid][0], "category": cat,
                                               "materiality": mat}, CLASSIFY_MODEL, now)
                done += 1
            if n % 25 == 0 or n == len(batches):
                print(f"batch {n}/{len(batches)}: {done} classified")


class _RateLimiter:
    def __init__(self, per_second: float):
        self._gap = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._gap
        if delay:
            time.sleep(delay)


def shares_at(candidates: list[str], published: datetime, fetch=None) -> tuple[Optional[str], Optional[float]]:
    """(symbol, shares outstanding) from the last Yahoo share-count
    reading on or before publication (looking back up to 180 days), from
    the first candidate symbol that has one."""
    if fetch is None:
        from market_data import DEFAULT_PROVIDER
        fetch = DEFAULT_PROVIDER.get_shares_history
    start = (published.date() - timedelta(days=180)).isoformat()
    end = (published.date() + timedelta(days=1)).isoformat()
    for symbol in candidates:
        try:
            series = fetch(symbol, start, end)
        except Exception:
            continue
        if series is None or len(series) == 0:
            continue
        series = series.sort_index()
        idx = series.index.tz_convert("UTC") if series.index.tz is not None else series.index.tz_localize("UTC")
        before = series[idx <= published]
        if len(before):
            return symbol, float(before.iloc[-1])
    return None, None


def step_financing(conn, workers: int = 4, limit: Optional[int] = None) -> None:
    from news_watchlist.outcomes import parse_published

    todo = conn.execute(
        "SELECT s.guid, s.title, s.link, s.pubdate FROM parsed_releases p "
        "JOIN seen_items s ON s.guid = p.guid LEFT JOIN financing_terms f ON f.guid = p.guid "
        "WHERE s.feed_url = ? AND p.category = 'financing' AND f.guid IS NULL ORDER BY s.pubdate",
        (FEED_URL,)).fetchall()
    if limit:
        todo = todo[:limit]
    symbols = candidates_by_guid(conn)
    limiter = _RateLimiter(per_second=1.0)
    print(f"{len(todo)} financing release(s) to extract")

    def work(row):
        guid, title, link, pubdate = row
        limiter.wait()
        body = article.fetch_article_text(link)
        if not body:
            return row, None
        symbol, shares = shares_at(symbols[guid], parse_published(pubdate))
        context = {"yahoo_ticker": symbol, "shares_outstanding": shares}
        return row, financing.extract_terms(symbols[guid][0], title, link, body=body, context=context)

    ok = 0
    now = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (row, terms) in enumerate(pool.map(work, todo), 1):
            if terms is None:
                print(f"[{n}/{len(todo)}] {symbols[row[0]][0]}: FAILED")
                continue
            store.save_financing_terms(conn, row[0], symbols[row[0]][0], terms, financing.MODEL, now)
            ok += 1
            if n % 50 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}] {ok} extracted")


def step_score(conn, db_path=None) -> None:
    import news_watchlist_service
    from news_watchlist import store as nw_store

    db_path = db_path or DB_PATH
    symbols = candidates_by_guid(conn)
    rows = news_watchlist_service.score_release_outcomes(
        uuid.uuid4().hex, nw_store.connect(db_path), db_path,
        news_watchlist_service._download_bars,
        candidates_fn=lambda e: symbols.get(e["guid"]) or [e["ticker"]])
    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"scored {len(rows)} release(s): {by_status}")


def survivorship(conn) -> dict:
    """How many classified financings made it into the outcome sample."""
    q = ("SELECT COUNT(*), SUM(o.entry_date IS NOT NULL), SUM(o.status = 'no_data') "
         "FROM parsed_releases p JOIN seen_items s ON s.guid = p.guid "
         "LEFT JOIN release_outcomes o ON o.guid = p.guid "
         "WHERE s.feed_url = ? AND p.category = 'financing'")
    total, scored, no_data = conn.execute(q, (FEED_URL,)).fetchone()
    return {"financing_releases": total or 0, "scored": scored or 0, "no_yahoo_data": no_data or 0}


def step_report(db_path=None) -> str:
    from news_watchlist import outcomes
    from news_watchlist import store as nw_store

    db_path = db_path or DB_PATH
    nw = nw_store.connect(db_path)
    rows = outcomes.attach_financing(nw_store.list_scored_outcomes(nw),
                                     outcomes.read_financing_terms(db_path))
    s = survivorship(nw)
    head = (f"ARCHIVE ({FIRST_MONTH}..{LAST_MONTH}, GlobeNewswire sitemaps). Financing releases: "
            f"{s['financing_releases']}, scored: {s['scored']}, no Yahoo data: {s['no_yahoo_data']} "
            "(survivorship -- missing names are often delisted juniors, so results flatter).\n")
    return head + outcomes.format_report(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GlobeNewswire archive backfill (see module docstring)")
    p.add_argument("step", choices=["list", "classify", "financing", "score", "report", "all"])
    p.add_argument("--first", default=FIRST_MONTH, help="First month (YYYY-MM) for list.")
    p.add_argument("--last", default=LAST_MONTH, help="Last month (YYYY-MM) for list.")
    p.add_argument("--limit", type=int, default=None, help="classify/financing: only the first N.")
    p.add_argument("--workers", type=int, default=None, help="Parallel LLM calls.")
    args = p.parse_args()

    conn = connect()
    if args.step in ("list", "all"):
        step_list(conn, args.first, args.last)
    if args.step in ("classify", "all"):
        step_classify(conn, workers=args.workers or 6, limit=args.limit)
    if args.step in ("financing", "all"):
        step_financing(conn, workers=args.workers or 4, limit=args.limit)
    if args.step in ("score", "all"):
        step_score(conn)
    if args.step in ("report", "all"):
        print(step_report())
