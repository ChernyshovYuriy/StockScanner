"""Offline integration tests for press_release_service.run_collector (no
network -- feeds.fetch_feed_items / llm_parser.parse_release / send_text_email
are always monkeypatched; a real call to any of them would be a test bug)."""

import requests

import press_release_service
from press_release_tracker import store
from press_release_tracker.feeds import FeedItem


def _item(guid, feed_url="https://feed-a", **overrides):
    defaults = dict(
        guid=guid, feed_url=feed_url, title=f"Release {guid}",
        link=f"https://example.com/{guid}.html", pubdate="Tue, 15 Sep 2026 08:36:00 GMT",
        description="desc", categories=[],
    )
    defaults.update(overrides)
    return FeedItem(**defaults)


def _no_email(monkeypatch, sent=True):
    calls = []
    monkeypatch.setattr(press_release_service, "send_text_email",
                         lambda subject, body: calls.append((subject, body)) or sent)
    return calls


def test_run_collector_emails_new_items_and_marks_emailed(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", ["https://feed-a"])
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items",
                         lambda url: [_item("g1")])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                         lambda title, desc, cats: {"ticker": "OMI.V", "company": "Orosur",
                                                     "category": "exploration_drilling",
                                                     "materiality": "high", "summary": "s"})
    calls = _no_email(monkeypatch)

    press_release_service.run_collector("run1", conn=conn)

    assert len(calls) == 1
    assert "HIGH" in calls[0][0]  # materiality='high' -> the immediate lane
    assert "OMI.V" in calls[0][1]
    assert store.unemailed(conn) == []  # marked emailed


def test_run_collector_second_run_does_not_resend_same_item(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", ["https://feed-a"])
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items",
                         lambda url: [_item("g1")])  # same item every poll, RSS-feed-style
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                         lambda title, desc, cats: None)
    calls = _no_email(monkeypatch)

    press_release_service.run_collector("run1", conn=conn)
    press_release_service.run_collector("run2", conn=conn)

    assert len(calls) == 1  # only the first run actually sent anything


def test_run_collector_quiet_run_sends_nothing(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", ["https://feed-a"])
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items", lambda url: [])
    calls = _no_email(monkeypatch)

    press_release_service.run_collector("run1", conn=conn)

    assert calls == []


def test_run_collector_dry_run_writes_nothing_and_sends_nothing(tmp_path, monkeypatch, capsys):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", ["https://feed-a"])
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items",
                         lambda url: [_item("g1")])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                         lambda title, desc, cats: {"ticker": "OMI.V", "company": None,
                                                     "category": "other", "materiality": "low",
                                                     "summary": ""})
    calls = _no_email(monkeypatch)

    press_release_service.run_collector("run1", dry_run=True, conn=conn)

    assert calls == []  # never sent
    assert store.is_seen(conn, "g1") is False  # never persisted
    out = capsys.readouterr().out
    assert "OMI.V" in out  # still previewed to stdout


def test_run_collector_one_feed_failure_does_not_abort_the_others(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS",
                         ["https://feed-bad", "https://feed-good"])

    def fake_fetch(url):
        if url == "https://feed-bad":
            raise requests.RequestException("boom")
        return [_item("g1", feed_url=url)]

    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items", fake_fetch)
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                         lambda title, desc, cats: None)
    calls = _no_email(monkeypatch)

    press_release_service.run_collector("run1", conn=conn)

    assert len(calls) == 1
    assert store.is_seen(conn, "g1") is True


def test_run_collector_unconfigured_llm_still_emails_raw_item(tmp_path, monkeypatch):
    """OPENAI_API_KEY unset -> llm_parser.parse_release returns None -- the
    item must still go out with its raw RSS title/link, not get dropped."""
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", ["https://feed-a"])
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items",
                         lambda url: [_item("g1")])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                         lambda title, desc, cats: None)
    calls = _no_email(monkeypatch)

    press_release_service.run_collector("run1", conn=conn)

    assert len(calls) == 1
    assert "Release g1" in calls[0][1]
    assert "(ticker unknown)" in calls[0][1]


def test_run_collector_send_failure_leaves_item_unemailed_for_next_run(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", ["https://feed-a"])
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items",
                         lambda url: [_item("g1")])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                         lambda title, desc, cats: None)
    _no_email(monkeypatch, sent=False)  # simulates Gmail not configured

    press_release_service.run_collector("run1", conn=conn)

    assert len(store.unemailed(conn)) == 1  # still pending, ready for a later run to pick up


# ── two-lane delivery: high fires immediately, everything else batches ────

def _fake_parse_by_guid(mapping):
    def _parse(title, desc, cats):
        return mapping.get(title.split()[-1])
    return _parse


def test_high_item_sent_immediately_while_batch_item_waits_for_the_hourly_window(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "pr.db")
    # A batch digest "just sent" -- the hourly window isn't due again yet.
    store.set_last_batch_sent_at(conn, press_release_service.market_now().isoformat())

    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", ["https://feed-a"])
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items",
                         lambda url: [_item("high1"), _item("med1")])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release", _fake_parse_by_guid({
        "high1": {"ticker": "AAA", "company": None, "category": "other",
                  "materiality": "high", "summary": "urgent"},
        "med1": {"ticker": "BBB", "company": None, "category": "other",
                 "materiality": "medium", "summary": "routine"},
    }))
    calls = _no_email(monkeypatch)

    press_release_service.run_collector("run1", conn=conn)

    assert len(calls) == 1  # only the high lane fired this run
    assert "HIGH" in calls[0][0]
    assert "AAA" in calls[0][1]
    remaining = store.unemailed(conn)
    assert len(remaining) == 1 and remaining[0]["guid"] == "med1"  # still waiting on the batch window


def test_batch_fires_once_the_hourly_window_has_elapsed(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "pr.db")
    long_ago = (press_release_service.market_now() - press_release_service.timedelta(hours=2)).isoformat()
    store.set_last_batch_sent_at(conn, long_ago)

    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", ["https://feed-a"])
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items", lambda url: [_item("med1")])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                         lambda title, desc, cats: {"ticker": "BBB", "company": None,
                                                     "category": "other", "materiality": "medium",
                                                     "summary": "routine"})
    calls = _no_email(monkeypatch)

    press_release_service.run_collector("run1", conn=conn)

    assert len(calls) == 1
    assert "hourly digest" in calls[0][0]
    assert store.unemailed(conn) == []


def test_batch_does_not_fire_before_the_hourly_window_elapses(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "pr.db")
    store.set_last_batch_sent_at(conn, press_release_service.market_now().isoformat())

    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", ["https://feed-a"])
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items", lambda url: [_item("med1")])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                         lambda title, desc, cats: {"ticker": "BBB", "company": None,
                                                     "category": "other", "materiality": "medium",
                                                     "summary": "routine"})
    calls = _no_email(monkeypatch)

    press_release_service.run_collector("run1", conn=conn)

    assert calls == []
    assert len(store.unemailed(conn)) == 1


def test_dry_run_previews_batch_lane_even_when_the_real_window_is_not_due(tmp_path, monkeypatch, capsys):
    conn = store.connect(tmp_path / "pr.db")
    store.set_last_batch_sent_at(conn, press_release_service.market_now().isoformat())  # not due

    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", ["https://feed-a"])
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items", lambda url: [_item("med1")])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                         lambda title, desc, cats: {"ticker": "BBB", "company": None,
                                                     "category": "other", "materiality": "medium",
                                                     "summary": "routine"})
    calls = _no_email(monkeypatch)

    press_release_service.run_collector("run1", dry_run=True, conn=conn)

    assert calls == []
    out = capsys.readouterr().out
    assert "hourly digest" in out
    assert "BBB" in out
