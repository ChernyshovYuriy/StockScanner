"""Offline tests for press_release_tracker.feeds (no network -- requests.get
is always monkeypatched to a canned response)."""

import requests

from press_release_tracker import feeds

_SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>GlobeNewswire - News from Canada</title>
<item>
<title>Orosur Mining Announces El Cedro Discovery</title>
<link>https://www.globenewswire.com/news-release/2026/09/15/orosur.html</link>
<guid isPermaLink="true">https://www.globenewswire.com/news-release/2026/09/15/orosur.html</guid>
<pubDate>Tue, 15 Sep 2026 08:36:00 GMT</pubDate>
<description><![CDATA[Orosur Mining Inc. announced today drilling results.]]></description>
<category domain="Ticker">OMI</category>
<category domain="Subject">Mining</category>
</item>
<item>
<title>No Guid Item</title>
<link>https://www.globenewswire.com/news-release/2026/09/15/noguid.html</link>
<pubDate>Tue, 15 Sep 2026 09:00:00 GMT</pubDate>
<description>Plain text description.</description>
</item>
</channel>
</rss>
"""


class _FakeResponse:
    def __init__(self, content, status=200):
        self.content = content.encode("utf-8")
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise requests.HTTPError(f"status {self._status}")


def test_fetch_feed_items_parses_expected_fields(monkeypatch):
    monkeypatch.setattr(feeds.requests, "get", lambda *a, **k: _FakeResponse(_SAMPLE_RSS))

    items = feeds.fetch_feed_items("https://example.com/feed")

    assert len(items) == 2
    first = items[0]
    assert first.guid == "https://www.globenewswire.com/news-release/2026/09/15/orosur.html"
    assert first.title == "Orosur Mining Announces El Cedro Discovery"
    assert first.pubdate == "Tue, 15 Sep 2026 08:36:00 GMT"
    assert first.description == "Orosur Mining Inc. announced today drilling results."
    assert first.categories == ["OMI", "Mining"]
    assert first.feed_url == "https://example.com/feed"


def test_fetch_feed_items_falls_back_to_link_when_guid_missing(monkeypatch):
    monkeypatch.setattr(feeds.requests, "get", lambda *a, **k: _FakeResponse(_SAMPLE_RSS))

    items = feeds.fetch_feed_items("https://example.com/feed")

    second = items[1]
    assert second.guid == "https://www.globenewswire.com/news-release/2026/09/15/noguid.html"
    assert second.categories == []


def test_fetch_feed_items_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(feeds.requests, "get", lambda *a, **k: _FakeResponse("", status=404))

    try:
        feeds.fetch_feed_items("https://example.com/feed")
        assert False, "expected HTTPError"
    except requests.HTTPError:
        pass
