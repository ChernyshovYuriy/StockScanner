"""
press_release_tracker/cse_news.py
===================================
Canadian Securities Exchange company news as feed items (added 2026-10).

The CSE's own RSS feed (issuers.thecse.com/en/issuer_news.xml) now
redirects to a login page. Its public Company News page instead loads one
JSON file holding EVERY CSE listed-company release since 1991, newest first
(~98k entries, ~44 MB, no gzip, no conditional GET). This reader asks for
only the first config.PRESS_RELEASE_CSE_NEWS_RANGE_BYTES via an HTTP Range
request (and stops reading there even if the server ignores Range), then
decodes the complete entries in that prefix one by one -- the Pi can't
afford to parse the whole file.

Each entry has id/date/title/language/mainSymbol and a PDF link signed
for 5 minutes, so there's no teaser text: the item's description just
names the CSE symbol for llm_parser.py, and its link is the company's CSE
page (there's no per-release page). French releases are dropped
('neutral' ones are English). Mechanical only, same split as feeds.py.

fetch_release_text() gets one release's full text for the analyst read:
it re-reads the prefix for a freshly signed file link (an item can wait
minutes behind the LLM calls before its turn), then extracts the text with
article.document_to_text() -- ~98% of releases are PDFs, ~2% Word .docx.
"""
from __future__ import annotations

import json
from datetime import datetime
from email.utils import format_datetime
from typing import List, Optional

import requests

from press_release_tracker import article
from press_release_tracker.feeds import USER_AGENT, FeedItem

try:
    from config import PRESS_RELEASE_CSE_NEWS_RANGE_BYTES as RANGE_BYTES
except Exception:
    RANGE_BYTES = 80_000

COMPANY_PAGE = "https://thecse.com/s/{symbol}"


def is_cse_news_url(feed_url: str) -> bool:
    return feed_url.endswith("/news-releases.json")


def parse_entries(text: str) -> List[dict]:
    """Every complete entry in a (possibly truncated) prefix of the JSON
    file, in file order. Raises ValueError if the prefix has no "list"."""
    start = text.find('"list"')
    if start < 0:
        raise ValueError("CSE news JSON: no \"list\" in the response")
    i = text.index("[", start) + 1
    decoder = json.JSONDecoder()
    entries = []
    while True:
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text) or text[i] == "]":
            break
        try:
            entry, i = decoder.raw_decode(text, i)
        except ValueError:
            break  # the entry cut off by the byte limit
        entries.append(entry)
    return entries


def to_feed_item(entry: dict, feed_url: str) -> FeedItem | None:
    if entry.get("language") == "fr" or not entry.get("id") or not entry.get("title"):
        return None
    symbol = (entry.get("mainSymbol") or "").strip()
    try:
        pubdate = format_datetime(datetime.fromisoformat(entry["date"]))
    except (KeyError, TypeError, ValueError):
        pubdate = ""
    return FeedItem(
        guid=f"cse-news:{entry['id']}",
        feed_url=feed_url,
        title=entry["title"].strip(),
        link=COMPANY_PAGE.format(symbol=symbol) if symbol else "https://thecse.com/news-events/company-news/",
        pubdate=pubdate,
        description=(f"Canadian Securities Exchange listed company (CSE: {symbol}) news release."
                     if symbol else "Canadian Securities Exchange listed company news release."),
    )


def _read_prefix(feed_url: str, max_bytes: int, timeout: int) -> str:
    resp = requests.get(feed_url, headers={"User-Agent": USER_AGENT, "Range": f"bytes=0-{max_bytes - 1}"},
                        timeout=timeout, stream=True)
    try:
        resp.raise_for_status()
        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=16384):
            buf.extend(chunk)
            if len(buf) >= max_bytes:
                break
    finally:
        resp.close()
    return bytes(buf[:max_bytes]).decode("utf-8", errors="ignore")


def fetch_cse_items(feed_url: str, max_bytes: int = RANGE_BYTES, timeout: int = 30) -> List[FeedItem]:
    """The newest releases that fit in the first max_bytes of the file.
    Raises requests.RequestException on a fetch failure, ValueError on an
    unrecognised response."""
    text = _read_prefix(feed_url, max_bytes, timeout)
    items = (to_feed_item(e, feed_url) for e in parse_entries(text))
    return [it for it in items if it is not None]


def fetch_release_text(item: FeedItem, max_bytes: int = RANGE_BYTES, timeout: int = 30) -> Optional[str]:
    """Full text of one CSE item's release file (PDF or .docx), or None if
    the release has left the prefix, a fetch fails, or there's no
    extractable text."""
    entry_id = item.guid.split(":", 1)[-1]
    try:
        entries = parse_entries(_read_prefix(item.feed_url, max_bytes, timeout))
    except (requests.RequestException, ValueError):
        return None
    url = next((e.get("fileUrl") for e in entries if str(e.get("id")) == entry_id), None)
    if not url:
        return None
    article.wait_turn(url)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    return article.document_to_text(resp.content)
