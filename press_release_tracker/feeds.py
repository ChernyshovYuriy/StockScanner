"""
press_release_tracker/feeds.py
================================
Mechanical RSS fetch + parse -- no interpretation happens here (that's
llm_parser.py's job; same fetch-vs-interpret split as
demand_signals/short_volume.py's fetch_daily_short_volume() vs
build_signals()). Stdlib XML only (xml.etree.ElementTree): RSS 2.0's
<item> fields used here (title, link, guid, pubDate, description,
category) aren't namespaced, so a feedparser dependency isn't needed for
this well-formed feed.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List

import requests

try:
    from config import PRESS_RELEASE_USER_AGENT as USER_AGENT
except Exception:
    USER_AGENT = "StockScanner-PressRelease/0.1 (chernyshov.yuriy@gmail.com)"


@dataclass
class FeedItem:
    guid: str
    feed_url: str
    title: str
    link: str
    pubdate: str
    description: str
    categories: List[str] = field(default_factory=list)


def _text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "").strip() if el is not None and el.text else ""


def fetch_feed_items(feed_url: str, timeout: int = 30) -> List[FeedItem]:
    """One feed -> its current <item> list, in the order the feed
    publishes them (not re-sorted). Raises requests.RequestException /
    ET.ParseError on a fetch/parse failure -- the service layer catches
    per-feed so one broken feed doesn't stop the others (same isolation
    guarantee as research/triple_screen/batch.run_batch's per-ticker try).

    guid falls back to link when a feed omits <guid> -- either way it's
    the value store.py dedupes on.
    """
    resp = requests.get(feed_url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    items = []
    for item in root.findall("./channel/item"):
        guid = _text(item, "guid") or _text(item, "link")
        if not guid:
            continue
        items.append(FeedItem(
            guid=guid,
            feed_url=feed_url,
            title=_text(item, "title"),
            link=_text(item, "link"),
            pubdate=_text(item, "pubDate"),
            description=_text(item, "description"),
            categories=[(c.text or "").strip() for c in item.findall("category") if c.text],
        ))
    return items
