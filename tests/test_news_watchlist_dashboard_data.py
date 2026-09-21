"""Offline tests for news_watchlist_dashboard_data (no network)."""

import pytest

import news_watchlist_dashboard_data as nwd
from news_watchlist import store


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
