"""
CSE company news (2026-10): the CSE's public company-news JSON (the whole
archive, newest first) read from a byte-limited prefix, turned into normal
feed items, gated like Newsfile (only 'high' reaches the user), never
article-fetched. Plus: the parser's ".CSE" suffix resolves to Yahoo's ".CN".
"""
import json
from email.utils import parsedate_to_datetime

import pytest
import requests

import config
import news_watchlist_service
import press_release_service
from news_watchlist import outcomes
from press_release_tracker import analyst, cse_news, feeds, store

URL = config.PRESS_RELEASE_CSE_NEWS_FEED


def _entry(i, title="Hertz Energy Announces Private Placement", lang="en", symbol="HZ",
           date="2026-10-09T08:46:44-04:00"):
    return {"id": i, "date": date, "title": title, "language": lang, "slug": "s",
            "mainSymbol": symbol, "symbols": [symbol], "fileUrl": "https://pdf?X-Amz-Expires=300"}


def _doc(entries):
    return json.dumps({"totalItems": 98449, "list": entries})


# ── parsing ─────────────────────────────────────────────────────────────────

def test_truncated_prefix_yields_only_complete_entries():
    text = _doc([_entry(3), _entry(2), _entry(1)])
    cut = text[: text.index('"id": 1') + 5]  # mid-way through the third entry
    assert [e["id"] for e in cse_news.parse_entries(cut)] == [3, 2]


def test_full_document_is_parsed_to_the_end():
    assert [e["id"] for e in cse_news.parse_entries(_doc([_entry(2), _entry(1)]))] == [2, 1]


def test_unrecognised_response_raises_value_error():
    with pytest.raises(ValueError):
        cse_news.parse_entries("<html>login</html>")


def test_feed_item_fields():
    it = cse_news.to_feed_item(_entry(299369), URL)
    assert it.guid == "cse-news:299369"
    assert it.feed_url == URL
    assert it.link == "https://thecse.com/s/HZ"
    assert "(CSE: HZ)" in it.description
    # RSS-style date, as news_watchlist parses it
    assert parsedate_to_datetime(it.pubdate).isoformat() == "2026-10-09T08:46:44-04:00"


def test_french_dropped_neutral_kept():
    assert cse_news.to_feed_item(_entry(1, lang="fr"), URL) is None
    assert cse_news.to_feed_item(_entry(1, lang="neutral"), URL) is not None


def test_missing_symbol_or_date_still_an_item():
    it = cse_news.to_feed_item(_entry(1, symbol=None, date=None), URL)
    assert it.pubdate == "" and "CSE:" not in it.description


# ── fetching: a Range request, and never more than max_bytes read ───────────

class _Resp:
    def __init__(self, body, status=206):
        self.body, self.status_code, self.closed, self.read = body, status, False, 0

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            self.read += chunk_size
            yield self.body[i:i + chunk_size]

    def close(self):
        self.closed = True


def test_fetch_sends_range_and_stops_reading_at_the_limit(monkeypatch):
    # A server that ignores Range and sends the whole (big) archive.
    body = _doc([_entry(i) for i in range(5000, 0, -1)]).encode()
    seen = {}

    def fake_get(url, headers, timeout, stream):
        seen.update(headers=headers, stream=stream)
        seen["resp"] = _Resp(body, status=200)
        return seen["resp"]

    monkeypatch.setattr(cse_news.requests, "get", fake_get)
    items = cse_news.fetch_cse_items(URL, max_bytes=50_000)
    assert seen["headers"]["Range"] == "bytes=0-49999" and seen["stream"]
    assert seen["resp"].read < 50_000 + 16384 * 2 and seen["resp"].closed
    assert 0 < len(items) < 5000
    assert items[0].guid == "cse-news:5000"


def test_feeds_dispatches_the_cse_url(monkeypatch):
    monkeypatch.setattr(cse_news.requests, "get",
                        lambda url, **k: _Resp(_doc([_entry(7)]).encode()))
    assert [i.guid for i in feeds.fetch_feed_items(URL)] == ["cse-news:7"]


def test_login_redirect_is_logged_and_other_feeds_still_run(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", [URL, "https://feed-b"])
    rss = _Resp(b"", 200)
    rss.content = b"<rss><channel><item><guid>b1</guid><title>T</title></item></channel></rss>"
    # cse_news and feeds share one requests module: route by URL
    monkeypatch.setattr(requests, "get",
                        lambda url, **k: _Resp(b"<html>login</html>", 200) if url == URL else rss)
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release", lambda *a: None)
    monkeypatch.setattr(press_release_service, "send_text_email", lambda s, b: True)
    press_release_service.run_collector("r", conn=conn)
    assert [r[0] for r in conn.execute("SELECT guid FROM seen_items")] == ["b1"]


# ── service: gated like Newsfile, never article-fetched ─────────────────────

def test_cse_config_is_high_only():
    assert URL in config.PRESS_RELEASE_FEEDS
    assert URL in config.PRESS_RELEASE_HIGH_ONLY_FEEDS


@pytest.mark.parametrize("materiality,emailed", [("high", True), ("medium", False)])
def test_cse_items_gated_and_never_article_fetched(tmp_path, monkeypatch, materiality, emailed):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", [URL])
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_HIGH_ONLY_FEEDS", frozenset({URL}))
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items",
                        lambda url: [cse_news.to_feed_item(_entry(1), URL)])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                        lambda *a: {"ticker": "HZ.CSE", "company": "Hertz", "category": "financing",
                                    "materiality": materiality, "summary": "s"})
    monkeypatch.setattr(press_release_service.article, "fetch_article_text",
                        lambda link: pytest.fail("CSE item must not be article-fetched"))
    monkeypatch.setattr(press_release_service.analyst, "analyze_release",
                        lambda *a, **k: pytest.fail("no analyst read for a CSE item"))
    monkeypatch.setattr(press_release_service.financing, "extract_terms",
                        lambda *a, **k: pytest.fail("no financing extraction for a CSE item"))
    sent = []
    monkeypatch.setattr(press_release_service, "send_text_email", lambda s, b: sent.append(b) or True)
    press_release_service.run_collector("r", conn=conn)
    assert bool(sent) == emailed
    assert store.unemailed(conn) == []


# ── ".CSE" resolves to Yahoo's ".CN" everywhere a price is looked up ────────

def test_outcomes_candidates_map_cse_to_cn():
    assert outcomes.symbol_candidates("hz.cse") == ["HZ.CN"]
    assert outcomes.symbol_candidates("KTO.V") == ["KTO.V"]


def test_analyst_candidates_map_cse_to_cn():
    assert analyst._candidates("HZ.CSE") == ["HZ.CN"]


def test_watchlist_price_resolver_maps_cse_to_cn(monkeypatch):
    asked = []
    monkeypatch.setattr(news_watchlist_service, "get_market_price",
                        lambda t: asked.append(t) or ((0.5, "daily-close") if t == "HZ.CN" else (None, None)))
    assert news_watchlist_service._resolve_market_price("HZ.CSE") == (0.5, "daily-close", "HZ.CN")
    assert asked == ["HZ.CN"]
