"""
TMX Newsfile feeds (2026-10): fetch everything, analyse it, notify only on
what matters. Every item from a config.PRESS_RELEASE_HIGH_ONLY_FEEDS feed is
parsed, analysed and stored like any other, but only a materiality 'high'
one is emailed or seeded into News Watchlist's inbox. GlobeNewswire items
keep both email lanes and are seeded as before.
"""
from datetime import datetime

import pytest

import config
import news_watchlist_service
import press_release_service
from news_watchlist import store as nw_store
from press_release_tracker import article
from press_release_tracker import store
from press_release_tracker.feeds import FeedItem
from time_utils import TSX_TZ, set_backtest_clock

GNW = "https://gnw-feed"
NEWSFILE = "https://newsfile-feed"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(press_release_service.analyst, "analyze_release", lambda *a, **k: None)
    monkeypatch.setattr(press_release_service.financing, "extract_terms", lambda *a, **k: None)
    monkeypatch.setattr(press_release_service.article, "fetch_article_text", lambda link: "full body")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", [GNW, NEWSFILE])
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_HIGH_ONLY_FEEDS", frozenset({NEWSFILE}))
    monkeypatch.setattr(news_watchlist_service, "PRESS_RELEASE_HIGH_ONLY_FEEDS", frozenset({NEWSFILE}))
    set_backtest_clock(datetime(2026, 9, 21, 17, 10, tzinfo=TSX_TZ))
    yield
    set_backtest_clock(None)


def _item(guid, feed_url):
    return FeedItem(guid=guid, feed_url=feed_url, title=f"Release {guid}",
                    link=f"https://example.com/{guid}", pubdate="Mon, 21 Sep 2026 08:36:00 GMT",
                    description="desc", categories=[])


# guid -> (feed, materiality)
ITEMS = {
    "gnw-high": (GNW, "high"),
    "gnw-low": (GNW, "low"),
    "nf-high": (NEWSFILE, "high"),
    "nf-medium": (NEWSFILE, "medium"),
    "nf-low": (NEWSFILE, "low"),
}


def _collect(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items",
                        lambda url: [_item(g, f) for g, (f, _) in ITEMS.items() if f == url])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                        lambda title, desc, cats: {"ticker": title.split()[-1].upper().replace("-", ""),
                                                    "company": "Co", "category": "financing",
                                                    "materiality": ITEMS[title.split()[-1]][1],
                                                    "summary": "s"})
    calls = []
    monkeypatch.setattr(press_release_service, "send_text_email",
                        lambda subject, body: calls.append((subject, body)) or True)
    press_release_service.run_collector("run1", conn=conn)
    return conn, calls


def test_only_high_newsfile_items_are_emailed(tmp_path, monkeypatch):
    conn, calls = _collect(tmp_path, monkeypatch)
    body = "\n".join(b for _, b in calls)
    assert "NFHIGH" in body
    assert "NFMEDIUM" not in body and "NFLOW" not in body
    # GlobeNewswire unchanged: both lanes still go out
    assert "GNWHIGH" in body and "GNWLOW" in body
    assert store.unemailed(conn) == []  # filed items don't wait in the batch lane


def test_filed_newsfile_items_are_still_parsed_and_analysed(tmp_path, monkeypatch):
    analysed = []
    monkeypatch.setattr(press_release_service.analyst, "analyze_release",
                        lambda ticker, *a, **k: analysed.append(ticker) or None)
    monkeypatch.setattr(press_release_service.financing, "extract_terms",
                        lambda ticker, *a, **k: analysed.append("terms:" + ticker) or None)
    conn, _ = _collect(tmp_path, monkeypatch)
    parsed = {r[0] for r in conn.execute("SELECT guid FROM parsed_releases")}
    assert parsed == set(ITEMS)
    assert "NFLOW" in analysed and "terms:NFLOW" in analysed


def test_filed_items_are_not_resent_on_the_next_run(tmp_path, monkeypatch):
    conn, calls = _collect(tmp_path, monkeypatch)
    n = len(calls)
    press_release_service.run_collector("run2", conn=conn)
    assert len(calls) == n


def test_dry_run_hides_filed_items_and_writes_nothing(tmp_path, monkeypatch, capsys):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items",
                        lambda url: [_item(g, f) for g, (f, _) in ITEMS.items() if f == url])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                        lambda title, desc, cats: {"ticker": title.split()[-1].upper().replace("-", ""),
                                                    "company": "Co", "category": "other",
                                                    "materiality": ITEMS[title.split()[-1]][1],
                                                    "summary": "s"})
    press_release_service.run_collector("run1", dry_run=True, conn=conn)
    out = capsys.readouterr().out
    assert "NFHIGH" in out and "NFLOW" not in out and "GNWLOW" in out
    assert conn.execute("SELECT COUNT(*) FROM seen_items").fetchone()[0] == 0


def test_inbox_seeds_only_high_newsfile_items(tmp_path, monkeypatch):
    pr_conn, _ = _collect(tmp_path, monkeypatch)
    pr_conn.close()
    conn = nw_store.connect(tmp_path / "nw.db")
    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=tmp_path / "pr.db",
        price_fetcher=lambda t: (1.0, "daily-close", t))
    tickers = {r["ticker"] for r in nw_store.list_by_status(conn, "inbox")}
    assert tickers == {"GNWHIGH", "GNWLOW", "NFHIGH"}


def test_config_gates_exactly_the_newsfile_feeds():
    assert config.PRESS_RELEASE_HIGH_ONLY_FEEDS == frozenset(config.PRESS_RELEASE_NEWSFILE_FEEDS)
    assert set(config.PRESS_RELEASE_NEWSFILE_FEEDS) <= set(config.PRESS_RELEASE_FEEDS)
    assert config.PRESS_RELEASE_FEEDS[0].startswith("https://www.globenewswire.com/")
    assert all(u.startswith("https://feeds.newsfilecorp.com/industry/")
               for u in config.PRESS_RELEASE_NEWSFILE_FEEDS)


_NEWSFILE_HTML = """<html><body>
<div class="nf-container left-content"><p>Related news sidebar</p></div>
<article id="release" class="nf-container nf-content" data-print="show">
<p>Vancouver--(Newsfile Corp. - October 9, 2026) - Kingsmen Resources Ltd. (TSXV: KNG) announces.</p>
<table><tr><td>Shares</td><td>1,000,000</td></tr></table>
<p>Source: Kingsmen Resources Ltd</p>
</article>
<div class="footer"><p>Subscribe to Newsfile</p></div>
</body></html>"""


def test_article_reads_newsfile_release_body_only():
    text = article.extract_article_text(_NEWSFILE_HTML)
    assert "Kingsmen Resources Ltd. (TSXV: KNG) announces." in text
    assert "Shares | 1,000,000" in text
    assert "sidebar" not in text and "Subscribe" not in text


def test_other_article_elements_are_not_a_body():
    assert article.extract_article_text('<article id="other"><p>x</p></article>') is None
