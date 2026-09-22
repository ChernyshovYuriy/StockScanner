"""
Offline integration tests for news_watchlist_service.run_collector (no
network -- price_fetcher is always a fake/stub and send_text_email is always
monkeypatched; a real market_data/manual_sell/email call would be a test bug).
"""
from datetime import datetime

import pytest

import news_watchlist_service
from news_watchlist import store
from press_release_tracker import store as pr_store
from press_release_tracker.feeds import FeedItem
from time_utils import TSX_TZ, set_backtest_clock

# A known weekday (Monday), so update-prices' trading-day gate never has to
# be reasoned about per-test -- same fixed-anchor convention
# triple_screen_tracker's own tests use for non-trading-day cases.
_A_MONDAY = datetime(2026, 9, 21, 17, 10, tzinfo=TSX_TZ)
_A_SATURDAY = datetime(2026, 9, 19, 17, 10, tzinfo=TSX_TZ)


@pytest.fixture(autouse=True)
def _reset_backtest_clock():
    set_backtest_clock(_A_MONDAY)
    yield
    set_backtest_clock(None)


@pytest.fixture(autouse=True)
def _no_real_email(monkeypatch):
    calls = []
    monkeypatch.setattr(news_watchlist_service, "send_text_email",
                         lambda subject, body: calls.append((subject, body)) or True)
    return calls


def _seed_parsed_release(pr_db_path, guid, ticker, pubdate="Mon, 21 Sep 2026 08:36:00 GMT", **overrides):
    conn = pr_store.connect(pr_db_path)
    item = FeedItem(
        guid=guid, feed_url="https://feed-a", title=f"Release {guid}",
        link=f"https://example.com/{guid}.html", pubdate=pubdate,
        description="desc", categories=[],
    )
    pr_store.mark_seen(conn, item, first_seen_at="2026-09-21T08:40:00")
    parsed = {"ticker": ticker, "company": "Some Co", "category": "exploration_drilling",
              "materiality": "high", "summary": "Drilling results announced."}
    parsed.update(overrides)
    pr_store.save_parsed(conn, guid, parsed, model="gpt-5-nano", parsed_at="2026-09-21T08:41:00")
    conn.close()


def _fake_price(price_map):
    def _fetch(ticker):
        price = price_map.get(ticker)
        return (price, "daily-close", ticker) if price is not None else (None, None, None)
    return _fetch


# ── seed: inbox seeding ──────────────────────────────────────────────────

def test_new_parsed_ticker_is_seeded_into_inbox(tmp_path):
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "OMI.V")
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db,
        price_fetcher=_fake_price({"OMI.V": 0.12}))

    inbox = store.list_by_status(conn, "inbox")
    assert len(inbox) == 1
    assert inbox[0]["ticker"] == "OMI.V"
    assert inbox[0]["flag_price"] == 0.12
    assert inbox[0]["flagged_at"] == "2026-09-21"


def test_seeded_item_persists_the_resolved_yahoo_ticker(tmp_path):
    """price_fetcher's third return value (the symbol that actually had
    data -- see _resolve_market_price()) lands on the new row's
    yahoo_ticker, not just the bare ticker the parser produced."""
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "AYA")
    conn = store.connect(tmp_path / "nw.db")

    def price_fetcher(ticker):
        return (40.24, "daily-close", "AYA.TO")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db, price_fetcher=price_fetcher)

    inbox = store.list_by_status(conn, "inbox")
    assert inbox[0]["ticker"] == "AYA"
    assert inbox[0]["yahoo_ticker"] == "AYA.TO"


def test_already_seeded_guid_is_not_reseeded(tmp_path):
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "OMI.V")
    conn = store.connect(tmp_path / "nw.db")
    price_fetcher = _fake_price({"OMI.V": 0.12})

    news_watchlist_service.run_collector("run1", mode="seed", conn=conn, press_release_db_path=pr_db, price_fetcher=price_fetcher)
    news_watchlist_service.run_collector("run2", mode="seed", conn=conn, press_release_db_path=pr_db, price_fetcher=price_fetcher)

    assert len(store.list_by_status(conn, "inbox")) == 1


def test_second_release_for_pending_ticker_collapses_into_the_same_row(tmp_path):
    """A ticker that already has an unreviewed inbox item shouldn't flood
    the inbox with one row per press release -- a fresh one for the same
    ticker updates the existing row's content instead."""
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "CYG.V", summary="First update.")
    conn = store.connect(tmp_path / "nw.db")
    price_fetcher = _fake_price({"CYG.V": 0.17})

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db, price_fetcher=price_fetcher)

    _seed_parsed_release(pr_db, "g2", "CYG.V", summary="Second update.")
    news_watchlist_service.run_collector(
        "run2", mode="seed", conn=conn, press_release_db_path=pr_db, price_fetcher=price_fetcher)

    inbox = store.list_by_status(conn, "inbox")
    assert len(inbox) == 1
    assert inbox[0]["summary"] == "Second update."
    assert inbox[0]["guid"] == "g2"
    # Both guids stay remembered even though only g2 is now on the row --
    # otherwise g1 would look "not yet seeded" again on the next run.
    assert store.seeded_guids(conn) == {"g1", "g2"}


def test_release_for_a_watching_ticker_starts_a_fresh_inbox_item(tmp_path):
    """Collapsing only applies to a still-unreviewed ('inbox') item -- once
    a ticker has been confirmed to Watching, a new release is a genuinely
    fresh thing to review, not an update to something already decided."""
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "CYG.V")
    conn = store.connect(tmp_path / "nw.db")
    price_fetcher = _fake_price({"CYG.V": 0.17})

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db, price_fetcher=price_fetcher)
    first_id = store.list_by_status(conn, "inbox")[0]["id"]
    store.set_status(conn, first_id, "watching", "2026-09-21T09:00:00")

    _seed_parsed_release(pr_db, "g2", "CYG.V")
    news_watchlist_service.run_collector(
        "run2", mode="seed", conn=conn, press_release_db_path=pr_db, price_fetcher=price_fetcher)

    assert len(store.list_by_status(conn, "inbox")) == 1
    assert len(store.list_by_status(conn, "watching")) == 1


def test_stale_candidate_is_skipped_and_marked_processed(tmp_path):
    """A release whose own pubDate is older than
    NEWS_WATCHLIST_MAX_ARTICLE_AGE_DAYS (7) is never seeded -- and its guid
    is still marked processed, so it isn't re-evaluated on every future
    run either."""
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "OMI.V", pubdate="Mon, 07 Sep 2026 08:36:00 GMT")
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db,
        price_fetcher=_fake_price({"OMI.V": 0.12}))

    assert store.list_by_status(conn, "inbox") == []
    assert store.seeded_guids(conn) == {"g1"}


def test_stale_candidate_dry_run_leaves_no_trace(tmp_path):
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "OMI.V", pubdate="Mon, 07 Sep 2026 08:36:00 GMT")
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", dry_run=True, conn=conn, press_release_db_path=pr_db,
        price_fetcher=_fake_price({"OMI.V": 0.12}))

    assert store.list_by_status(conn, "inbox") == []
    assert store.seeded_guids(conn) == set()


def test_candidate_within_max_age_is_still_seeded(tmp_path):
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "OMI.V", pubdate="Wed, 16 Sep 2026 08:36:00 GMT")
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db,
        price_fetcher=_fake_price({"OMI.V": 0.12}))

    assert len(store.list_by_status(conn, "inbox")) == 1


def test_missing_pubdate_is_not_treated_as_stale(tmp_path):
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "OMI.V", pubdate="")
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db,
        price_fetcher=_fake_price({"OMI.V": 0.12}))

    assert len(store.list_by_status(conn, "inbox")) == 1


def test_parsed_item_with_no_ticker_is_never_seeded(tmp_path):
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", None)
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db, price_fetcher=_fake_price({}))

    assert store.list_by_status(conn, "inbox") == []


def test_price_unavailable_skips_seeding_this_run_not_permanently(tmp_path):
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "DELISTED.V")
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db, price_fetcher=_fake_price({}))
    assert store.list_by_status(conn, "inbox") == []
    assert store.seeded_guids(conn) == set()  # not marked seeded -- eligible again next run

    news_watchlist_service.run_collector(
        "run2", mode="seed", conn=conn, press_release_db_path=pr_db,
        price_fetcher=_fake_price({"DELISTED.V": 0.05}))
    assert len(store.list_by_status(conn, "inbox")) == 1


def test_missing_press_releases_db_seeds_nothing_but_does_not_crash(tmp_path):
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=tmp_path / "does_not_exist.db",
        price_fetcher=_fake_price({}))

    assert store.list_by_status(conn, "inbox") == []


def test_seed_mode_runs_on_a_non_trading_day(tmp_path):
    """News can break any time -- unlike update-prices, seed has no
    trading-day gate, same reasoning as press_release_tracker's own timer."""
    set_backtest_clock(_A_SATURDAY)
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "OMI.V")
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db,
        price_fetcher=_fake_price({"OMI.V": 0.12}))

    assert len(store.list_by_status(conn, "inbox")) == 1


# ── seed: immediate alert email ──────────────────────────────────────────

def test_seed_with_new_items_sends_an_immediate_alert(tmp_path, _no_real_email):
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "OMI.V")
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db,
        price_fetcher=_fake_price({"OMI.V": 0.12}))

    assert len(_no_real_email) == 1
    subject, body = _no_real_email[0]
    assert "News Watchlist" in subject
    assert "OMI.V" in body


def test_seed_with_nothing_new_sends_no_alert(tmp_path, _no_real_email):
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=tmp_path / "does_not_exist.db",
        price_fetcher=_fake_price({}))

    assert _no_real_email == []


def test_seed_dry_run_previews_the_alert_but_sends_nothing(tmp_path, _no_real_email, capsys):
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "OMI.V")
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", dry_run=True, conn=conn, press_release_db_path=pr_db,
        price_fetcher=_fake_price({"OMI.V": 0.12}))

    assert _no_real_email == []
    assert store.list_by_status(conn, "inbox") == []
    out = capsys.readouterr().out
    assert "News Watchlist" in out
    assert "OMI.V" in out


# ── update-prices: daily roll-forward for watching items ────────────────

def test_watching_item_gets_daily_price_appended(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.add_manual(
        conn, ticker="AAPL", note="", flagged_at="2026-09-18", flag_price=190.0,
        created_at="2026-09-18T17:10:00",
    )

    news_watchlist_service.run_collector(
        "run1", mode="update-prices", conn=conn, price_fetcher=_fake_price({"AAPL": 195.0}))

    history = store.price_history_for(conn, item_id)
    assert history == [{"date": "2026-09-21", "close_price": 195.0}]


def test_update_prices_backfills_yahoo_ticker_for_a_pre_existing_watching_item(tmp_path):
    """A Watching row seeded before yahoo_ticker existed (add_manual never
    sets it) self-heals its dead Ticker-column link on the next daily
    price update, using the same symbol this run's price lookup already
    resolved -- no separate backfill script needed for an item still
    open in Watching."""
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.add_manual(
        conn, ticker="AYA", note="", flagged_at="2026-09-18", flag_price=38.0,
        created_at="2026-09-18T17:10:00",
    )
    assert store.get_item(conn, item_id)["yahoo_ticker"] is None

    def price_fetcher(ticker):
        return (40.24, "daily-close", "AYA.TO")

    news_watchlist_service.run_collector(
        "run1", mode="update-prices", conn=conn, price_fetcher=price_fetcher)

    assert store.get_item(conn, item_id)["yahoo_ticker"] == "AYA.TO"


def test_update_prices_does_not_overwrite_an_already_set_yahoo_ticker(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.add_manual(
        conn, ticker="AYA.TO", note="", flagged_at="2026-09-18", flag_price=38.0,
        created_at="2026-09-18T17:10:00",
    )
    store.set_yahoo_ticker(conn, item_id, "AYA.TO")

    def price_fetcher(ticker):
        return (40.24, "daily-close", "AYA.V")  # would be wrong if it won

    news_watchlist_service.run_collector(
        "run1", mode="update-prices", conn=conn, price_fetcher=price_fetcher)

    assert store.get_item(conn, item_id)["yahoo_ticker"] == "AYA.TO"


def test_inbox_item_gets_no_price_history_at_all(tmp_path):
    """The review gate: an unconfirmed inbox item must never accumulate a
    price_history row, even after update-prices runs."""
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "OMI.V")
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", mode="seed", conn=conn, press_release_db_path=pr_db,
        price_fetcher=_fake_price({"OMI.V": 0.12}))
    item_id = store.list_by_status(conn, "inbox")[0]["id"]
    news_watchlist_service.run_collector(
        "run2", mode="update-prices", conn=conn, price_fetcher=_fake_price({"OMI.V": 0.12}))

    assert store.price_history_for(conn, item_id) == []


def test_dismissed_item_gets_no_price_history(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.add_manual(
        conn, ticker="AAPL", note="", flagged_at="2026-09-18", flag_price=190.0,
        created_at="2026-09-18T17:10:00",
    )
    store.set_status(conn, item_id, "dismissed", "2026-09-19T09:00:00")

    news_watchlist_service.run_collector(
        "run1", mode="update-prices", conn=conn, price_fetcher=_fake_price({"AAPL": 195.0}))

    assert store.price_history_for(conn, item_id) == []


def test_update_prices_does_not_run_on_a_non_trading_day(tmp_path):
    set_backtest_clock(_A_SATURDAY)
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.add_manual(
        conn, ticker="AAPL", note="", flagged_at="2026-09-18", flag_price=190.0,
        created_at="2026-09-18T17:10:00",
    )

    class _ExplodingFetcher:
        def __call__(self, ticker):
            raise AssertionError("price_fetcher must not be called on a non-trading day")

    news_watchlist_service.run_collector(
        "run1", mode="update-prices", conn=conn, price_fetcher=_ExplodingFetcher())

    assert store.price_history_for(conn, item_id) == []


# ── both: manual/dry-run convenience ─────────────────────────────────────

def test_dry_run_defaults_to_both_modes_and_writes_nothing(tmp_path, capsys):
    pr_db = tmp_path / "pr.db"
    _seed_parsed_release(pr_db, "g1", "OMI.V")
    conn = store.connect(tmp_path / "nw.db")

    news_watchlist_service.run_collector(
        "run1", dry_run=True, conn=conn, press_release_db_path=pr_db,
        price_fetcher=_fake_price({"OMI.V": 0.12}))

    assert store.list_by_status(conn, "inbox") == []
    out = capsys.readouterr().out
    assert "OMI.V" in out


# ── _resolve_market_price: bare-ticker suffix disambiguation ─────────────
#
# press_release_tracker/llm_parser.py extracts whatever ticker a release's
# text states, with no Yahoo-Finance-suffix awareness (by design -- see
# press-release-parser-scope in project memory). A bare symbol like "AYA"
# commonly collides with an unrelated US-listed security on Yahoo Finance
# instead of the intended TSX-listed one ("AYA.TO"). These test the
# disambiguation in isolation, stubbing manual_sell.get_market_price (the
# module-level name news_watchlist_service imported it under) so no real
# network call happens.

def test_resolve_market_price_passes_through_an_already_suffixed_ticker(monkeypatch):
    calls = []

    def _fake(ticker):
        calls.append(ticker)
        return (0.85, "daily-close")

    monkeypatch.setattr(news_watchlist_service, "get_market_price", _fake)
    price, source, yahoo_ticker = news_watchlist_service._resolve_market_price("KTO.V")

    assert (price, source, yahoo_ticker) == (0.85, "daily-close", "KTO.V")
    assert calls == ["KTO.V"]


def test_resolve_market_price_prefers_dot_to_for_a_bare_ticker(monkeypatch):
    calls = []

    def _fake(ticker):
        calls.append(ticker)
        return (40.24, "daily-close") if ticker == "AYA.TO" else (None, None)

    monkeypatch.setattr(news_watchlist_service, "get_market_price", _fake)
    price, source, yahoo_ticker = news_watchlist_service._resolve_market_price("AYA")

    assert (price, source, yahoo_ticker) == (40.24, "daily-close", "AYA.TO")
    assert calls == ["AYA.TO"]


def test_resolve_market_price_falls_back_to_dot_v_when_dot_to_has_no_data(monkeypatch):
    calls = []

    def _fake(ticker):
        calls.append(ticker)
        return (3.62, "daily-close") if ticker == "KRY.V" else (None, None)

    monkeypatch.setattr(news_watchlist_service, "get_market_price", _fake)
    price, source, yahoo_ticker = news_watchlist_service._resolve_market_price("KRY")

    assert (price, source, yahoo_ticker) == (3.62, "daily-close", "KRY.V")
    assert calls == ["KRY.TO", "KRY.V"]


def test_resolve_market_price_falls_back_to_the_bare_ticker_when_neither_suffix_has_data(monkeypatch):
    calls = []

    def _fake(ticker):
        calls.append(ticker)
        return (38.99, "daily-close") if ticker == "XENE" else (None, None)

    monkeypatch.setattr(news_watchlist_service, "get_market_price", _fake)
    price, source, yahoo_ticker = news_watchlist_service._resolve_market_price("XENE")

    assert (price, source, yahoo_ticker) == (38.99, "daily-close", "XENE")
    assert calls == ["XENE.TO", "XENE.V", "XENE"]
