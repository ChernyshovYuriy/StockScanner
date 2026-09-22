"""Offline tests for news_watchlist_dashboard_data (no network)."""

import pandas as pd
import pytest

import news_watchlist_dashboard_data as nwd
from news_watchlist import store

# Captured before the autouse fixture below ever patches nwd._fetch_volumes
# -- the caching tests further down need to call the REAL implementation
# (with DEFAULT_PROVIDER.download_range mocked instead), not the blanket
# no-network stub every other test in this file gets.
_REAL_FETCH_VOLUMES = nwd._fetch_volumes


@pytest.fixture(autouse=True)
def _no_network_volume_fetch(monkeypatch):
    """_build_news_watchlist_state() now calls _fetch_volumes() (a live
    download_range() against Yahoo Finance) for every watching-status
    ticker. Stub it so this file's offline/no-network guarantee holds;
    a test that actually wants to exercise volume data overrides this."""
    monkeypatch.setattr(nwd, "_fetch_volumes", lambda tickers: {})


def test_read_returns_empty_when_db_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(nwd, "NEWS_WATCHLIST_DB_PATH", tmp_path / "does_not_exist.db")
    assert nwd._read_items() == []


def test_build_state_splits_by_status(tmp_path, monkeypatch):
    db_path = tmp_path / "nw.db"
    conn = store.connect(db_path)
    store.seed_inbox_item(
        conn, guid="g1", ticker="OMI.V", company=None, category=None, materiality="high",
        summary=None, source_link=None, flagged_at="2026-09-20", flag_price=0.12,
        created_at="2026-09-20T17:10:00",
    )
    watching_id = store.add_manual(
        conn, ticker="AAPL", note="", flagged_at="2026-09-18", flag_price=190.0,
        created_at="2026-09-18T17:10:00",
    )
    dismissed_id = store.add_manual(
        conn, ticker="MSFT", note="", flagged_at="2026-09-15", flag_price=400.0,
        created_at="2026-09-15T17:10:00",
    )
    store.set_status(conn, dismissed_id, "dismissed", "2026-09-16T09:00:00")
    monkeypatch.setattr(nwd, "NEWS_WATCHLIST_DB_PATH", db_path)

    state = nwd._build_news_watchlist_state()
    assert [r["ticker"] for r in state["inbox"]] == ["OMI.V"]
    assert [r["ticker"] for r in state["watching"]] == ["AAPL"]
    assert [r["ticker"] for r in state["dismissed"]] == ["MSFT"]
    assert state["watching"][0]["id"] == watching_id


def test_watching_row_reports_latest_price_and_pct_change(tmp_path, monkeypatch):
    db_path = tmp_path / "nw.db"
    conn = store.connect(db_path)
    item_id = store.add_manual(
        conn, ticker="AAPL", note="", flagged_at="2026-09-18", flag_price=100.0,
        created_at="2026-09-18T17:10:00",
    )
    store.append_price(conn, item_id, "2026-09-19", 100.0)
    store.append_price(conn, item_id, "2026-09-21", 110.0)
    monkeypatch.setattr(nwd, "NEWS_WATCHLIST_DB_PATH", db_path)

    row = nwd._build_news_watchlist_state()["watching"][0]
    assert row["latest_price"] == 110.0
    assert row["pct_change"] == pytest.approx(10.0)
    assert len(row["price_history"]) == 2


def test_inbox_row_with_no_price_history_falls_back_to_flag_price(tmp_path, monkeypatch):
    db_path = tmp_path / "nw.db"
    conn = store.connect(db_path)
    store.seed_inbox_item(
        conn, guid="g1", ticker="OMI.V", company=None, category=None, materiality="high",
        summary=None, source_link=None, flagged_at="2026-09-20", flag_price=0.12,
        created_at="2026-09-20T17:10:00",
    )
    monkeypatch.setattr(nwd, "NEWS_WATCHLIST_DB_PATH", db_path)

    row = nwd._build_news_watchlist_state()["inbox"][0]
    assert row["latest_price"] == 0.12
    assert row["pct_change"] == pytest.approx(0.0)


def test_build_news_watchlist_state_caches_within_ttl(tmp_path, monkeypatch):
    db_path = tmp_path / "nw.db"
    conn = store.connect(db_path)
    store.add_manual(conn, ticker="AAPL", note="", flagged_at="2026-09-18", flag_price=100.0,
                      created_at="2026-09-18T17:10:00")
    monkeypatch.setattr(nwd, "NEWS_WATCHLIST_DB_PATH", db_path)
    monkeypatch.setattr(nwd, "DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS", 999)
    nwd._cache["ts"] = 0.0
    nwd._cache["rows"] = None

    first = nwd.build_news_watchlist_state()
    # A second item added after the first read must NOT appear yet -- confirms
    # the TTL cache is actually being served, not re-read every call.
    store.add_manual(conn, ticker="MSFT", note="", flagged_at="2026-09-18", flag_price=400.0,
                      created_at="2026-09-18T17:10:00")
    second = nwd.build_news_watchlist_state()

    assert first == second
    assert len(second["watching"]) == 1


def test_watching_row_gets_volume_from_fetch_volumes(tmp_path, monkeypatch):
    db_path = tmp_path / "nw.db"
    conn = store.connect(db_path)
    store.add_manual(conn, ticker="AAPL", note="", flagged_at="2026-09-18", flag_price=100.0,
                      created_at="2026-09-18T17:10:00")
    monkeypatch.setattr(nwd, "NEWS_WATCHLIST_DB_PATH", db_path)
    monkeypatch.setattr(
        nwd, "_fetch_volumes",
        lambda tickers: {"AAPL": {"current_volume": 5_000_000.0, "average_volume": 2_000_000.0}},
    )

    row = nwd._build_news_watchlist_state()["watching"][0]
    assert row["current_volume"] == 5_000_000.0
    assert row["average_volume"] == 2_000_000.0


def test_watching_row_volume_is_none_when_fetch_has_no_entry(tmp_path, monkeypatch):
    db_path = tmp_path / "nw.db"
    conn = store.connect(db_path)
    store.add_manual(conn, ticker="AAPL", note="", flagged_at="2026-09-18", flag_price=100.0,
                      created_at="2026-09-18T17:10:00")
    monkeypatch.setattr(nwd, "NEWS_WATCHLIST_DB_PATH", db_path)

    row = nwd._build_news_watchlist_state()["watching"][0]
    assert row["current_volume"] is None
    assert row["average_volume"] is None


def test_inbox_rows_never_trigger_a_volume_fetch(tmp_path, monkeypatch):
    """Volume is a Watching-only enrichment -- fetching it for a 100+ row
    inbox on every page load would slow the page the way a full
    /volume-spikes scan does. _build_news_watchlist_state() must only
    pass watching-status tickers to _fetch_volumes()."""
    db_path = tmp_path / "nw.db"
    conn = store.connect(db_path)
    store.seed_inbox_item(
        conn, guid="g1", ticker="OMI.V", company=None, category=None, materiality="high",
        summary=None, source_link=None, flagged_at="2026-09-20", flag_price=0.12,
        created_at="2026-09-20T17:10:00",
    )
    monkeypatch.setattr(nwd, "NEWS_WATCHLIST_DB_PATH", db_path)
    seen_tickers = []
    monkeypatch.setattr(nwd, "_fetch_volumes", lambda tickers: seen_tickers.extend(tickers) or {})

    nwd._build_news_watchlist_state()
    assert seen_tickers == []


def _fake_volume_df(volume=1_000_000.0, rows=21):
    return pd.DataFrame({"Volume": [volume] * rows})


def test_fetch_volumes_caches_per_ticker_within_ttl(monkeypatch):
    """A repeat call for the same ticker within _VOLUME_CACHE_TTL_SECONDS
    must be served from _volume_cache, not re-hit download_range -- the
    fix for the Pi's ~88-ticker Inbox otherwise re-paying a ~minute-long
    batched yfinance fetch on every single page load/confirm/dismiss."""
    monkeypatch.setattr(nwd, "_fetch_volumes", _REAL_FETCH_VOLUMES)
    nwd._volume_cache.clear()
    calls = []

    def fake_download_range(tickers, start, end):
        calls.append(list(tickers))
        return {"AAPL": _fake_volume_df()}, []

    monkeypatch.setattr(nwd.DEFAULT_PROVIDER, "download_range", fake_download_range)

    first = nwd._fetch_volumes(["AAPL"])
    second = nwd._fetch_volumes(["AAPL"])

    assert len(calls) == 1
    assert first == second == {"AAPL": {"current_volume": 1_000_000.0, "average_volume": 1_000_000.0}}


def test_fetch_volumes_refetches_once_cache_entry_expires(monkeypatch):
    monkeypatch.setattr(nwd, "_fetch_volumes", _REAL_FETCH_VOLUMES)
    monkeypatch.setattr(nwd, "_VOLUME_CACHE_TTL_SECONDS", 0)
    nwd._volume_cache.clear()
    calls = []

    def fake_download_range(tickers, start, end):
        calls.append(list(tickers))
        return {"AAPL": _fake_volume_df()}, []

    monkeypatch.setattr(nwd.DEFAULT_PROVIDER, "download_range", fake_download_range)

    nwd._fetch_volumes(["AAPL"])
    nwd._fetch_volumes(["AAPL"])

    assert len(calls) == 2


# ── build_quote_link: Ticker column Yahoo Finance link ───────────────────

def test_quote_link_prefers_the_resolved_yahoo_ticker():
    link = nwd.build_quote_link("AYA", "AYA.TO", "Aya Gold & Silver")
    assert link == "https://ca.finance.yahoo.com/quote/AYA.TO"


def test_quote_link_falls_back_to_an_already_suffixed_ticker():
    link = nwd.build_quote_link("KTO.V", None, "Kootenay Silver")
    assert link == "https://ca.finance.yahoo.com/quote/KTO.V"


def test_quote_link_falls_back_to_a_company_name_search_when_unresolved():
    """A legacy row seeded before yahoo_ticker existed and never
    backfilled, with a bare ticker -- must not build a dead quote-page
    link out of a symbol nothing ever confirmed has data."""
    link = nwd.build_quote_link("AYA", None, "Aya Gold & Silver")
    assert link == "https://ca.finance.yahoo.com/lookup?s=Aya%20Gold%20%26%20Silver"


def test_quote_link_search_fallback_uses_the_ticker_when_company_is_unknown():
    link = nwd.build_quote_link("AYA", None, None)
    assert link == "https://ca.finance.yahoo.com/lookup?s=AYA"


def test_build_state_computes_quote_link_for_every_item(tmp_path, monkeypatch):
    db_path = tmp_path / "nw.db"
    conn = store.connect(db_path)
    store.seed_inbox_item(
        conn, guid="g1", ticker="AYA", company="Aya Gold & Silver", category=None,
        materiality="high", summary=None, source_link=None, flagged_at="2026-09-20",
        flag_price=40.24, created_at="2026-09-20T17:10:00", yahoo_ticker="AYA.TO",
    )
    monkeypatch.setattr(nwd, "NEWS_WATCHLIST_DB_PATH", db_path)

    row = nwd._build_news_watchlist_state()["inbox"][0]
    assert row["quote_link"] == "https://ca.finance.yahoo.com/quote/AYA.TO"


def test_invalidate_cache_forces_a_fresh_read(tmp_path, monkeypatch):
    db_path = tmp_path / "nw.db"
    conn = store.connect(db_path)
    store.add_manual(conn, ticker="AAPL", note="", flagged_at="2026-09-18", flag_price=100.0,
                      created_at="2026-09-18T17:10:00")
    monkeypatch.setattr(nwd, "NEWS_WATCHLIST_DB_PATH", db_path)
    monkeypatch.setattr(nwd, "DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS", 999)
    nwd._cache["ts"] = 0.0
    nwd._cache["rows"] = None

    nwd.build_news_watchlist_state()
    store.add_manual(conn, ticker="MSFT", note="", flagged_at="2026-09-18", flag_price=400.0,
                      created_at="2026-09-18T17:10:00")
    nwd.invalidate_news_watchlist_cache()

    assert len(nwd.build_news_watchlist_state()["watching"]) == 2


# ── _fetch_volumes: bare-ticker suffix fallback ──────────────────────────
#
# A bare ticker straight from the press-release LLM parser (no ".TO"/".V"
# suffix -- see news_watchlist_service._resolve_market_price()'s own
# docstring for why) sometimes has no Yahoo Finance data at all under
# that bare symbol (e.g. "AC" -- Air Canada only trades as "AC.TO").

def test_fetch_volumes_falls_back_to_dot_to_when_bare_ticker_has_no_data(monkeypatch):
    monkeypatch.setattr(nwd, "_fetch_volumes", _REAL_FETCH_VOLUMES)
    nwd._volume_cache.clear()
    calls = []

    def fake_download_range(tickers, start, end):
        calls.append(list(tickers))
        if tickers == ["AC.TO"]:
            return {"AC.TO": _fake_volume_df(volume=500_000.0)}, []
        return {}, list(tickers)

    monkeypatch.setattr(nwd.DEFAULT_PROVIDER, "download_range", fake_download_range)

    result = nwd._fetch_volumes(["AC"])

    assert calls == [["AC"], ["AC.TO"]]
    assert result == {"AC": {"current_volume": 500_000.0, "average_volume": 500_000.0}}


def test_fetch_volumes_falls_back_to_dot_v_when_dot_to_also_has_no_data(monkeypatch):
    monkeypatch.setattr(nwd, "_fetch_volumes", _REAL_FETCH_VOLUMES)
    nwd._volume_cache.clear()
    calls = []

    def fake_download_range(tickers, start, end):
        calls.append(list(tickers))
        if tickers == ["KRY.V"]:
            return {"KRY.V": _fake_volume_df(volume=25_000.0)}, []
        return {}, list(tickers)

    monkeypatch.setattr(nwd.DEFAULT_PROVIDER, "download_range", fake_download_range)

    result = nwd._fetch_volumes(["KRY"])

    assert calls == [["KRY"], ["KRY.TO"], ["KRY.V"]]
    assert result == {"KRY": {"current_volume": 25_000.0, "average_volume": 25_000.0}}


def test_fetch_volumes_does_not_probe_suffixes_when_bare_ticker_already_has_data(monkeypatch):
    """The common case (an already-correct bare ticker, e.g. a genuine US
    name) must stay a single batch call -- suffix-probing every bare
    ticker unconditionally would multiply the cost of a 100+-row Inbox."""
    monkeypatch.setattr(nwd, "_fetch_volumes", _REAL_FETCH_VOLUMES)
    nwd._volume_cache.clear()
    calls = []

    def fake_download_range(tickers, start, end):
        calls.append(list(tickers))
        return {"AAPL": _fake_volume_df()}, []

    monkeypatch.setattr(nwd.DEFAULT_PROVIDER, "download_range", fake_download_range)

    result = nwd._fetch_volumes(["AAPL"])

    assert calls == [["AAPL"]]
    assert result == {"AAPL": {"current_volume": 1_000_000.0, "average_volume": 1_000_000.0}}


def test_fetch_volumes_never_probes_suffixes_for_an_already_suffixed_ticker(monkeypatch):
    monkeypatch.setattr(nwd, "_fetch_volumes", _REAL_FETCH_VOLUMES)
    nwd._volume_cache.clear()
    calls = []

    def fake_download_range(tickers, start, end):
        calls.append(list(tickers))
        return {}, list(tickers)

    monkeypatch.setattr(nwd.DEFAULT_PROVIDER, "download_range", fake_download_range)

    result = nwd._fetch_volumes(["KTO.V"])

    assert calls == [["KTO.V"]]
    assert result == {}
