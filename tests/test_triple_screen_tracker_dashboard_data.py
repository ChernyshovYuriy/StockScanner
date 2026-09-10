"""Offline tests for triple_screen_tracker_dashboard_data (no network)."""

import pytest

import triple_screen_tracker_dashboard_data as tsd
from triple_screen_tracker import store


def test_read_returns_empty_when_db_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(tsd, "TRIPLE_SCREEN_TRACKER_DB_PATH", tmp_path / "does_not_exist.db")
    assert tsd._read_tracked() == []


def test_build_state_splits_open_and_closed(tmp_path, monkeypatch):
    db_path = tmp_path / "ts.db"
    conn = store.connect(db_path)
    open_id = store.create_tracked(conn, "AAPL", "2026-09-01", 100.0, created_at="2026-09-01T17:00:00")
    store.append_price(conn, open_id, "2026-09-01", 100.0)
    store.append_price(conn, open_id, "2026-09-02", 105.0)

    closed_id = store.create_tracked(conn, "MSFT", "2026-08-20", 200.0, created_at="2026-08-20T17:00:00")
    store.append_price(conn, closed_id, "2026-08-20", 200.0)
    store.close_tracked(conn, closed_id, "2026-08-25", 190.0)
    monkeypatch.setattr(tsd, "TRIPLE_SCREEN_TRACKER_DB_PATH", db_path)

    state = tsd._build_triple_screen_tracker_state()
    assert [r["ticker"] for r in state["open"]] == ["AAPL"]
    assert [r["ticker"] for r in state["closed"]] == ["MSFT"]


def test_open_row_reports_latest_price_and_pct_change(tmp_path, monkeypatch):
    db_path = tmp_path / "ts.db"
    conn = store.connect(db_path)
    tid = store.create_tracked(conn, "AAPL", "2026-09-01", 100.0, created_at="2026-09-01T17:00:00")
    store.append_price(conn, tid, "2026-09-01", 100.0)
    store.append_price(conn, tid, "2026-09-03", 110.0)
    monkeypatch.setattr(tsd, "TRIPLE_SCREEN_TRACKER_DB_PATH", db_path)

    row = tsd._build_triple_screen_tracker_state()["open"][0]
    assert row["latest_price"] == 110.0
    assert row["pct_change"] == pytest.approx(10.0)
    assert row["days_held"] == 2
    assert len(row["price_history"]) == 2


def test_closed_row_reports_sell_price_and_pct_change(tmp_path, monkeypatch):
    db_path = tmp_path / "ts.db"
    conn = store.connect(db_path)
    tid = store.create_tracked(conn, "AAPL", "2026-09-01", 100.0, created_at="2026-09-01T17:00:00")
    store.close_tracked(conn, tid, "2026-09-05", 90.0)
    monkeypatch.setattr(tsd, "TRIPLE_SCREEN_TRACKER_DB_PATH", db_path)

    row = tsd._build_triple_screen_tracker_state()["closed"][0]
    assert row["sell_price"] == 90.0
    assert row["pct_change"] == pytest.approx(-10.0)
    assert row["days_held"] == 4


def test_build_triple_screen_tracker_state_caches_within_ttl(tmp_path, monkeypatch):
    db_path = tmp_path / "ts.db"
    conn = store.connect(db_path)
    store.create_tracked(conn, "AAPL", "2026-09-01", 100.0, created_at="2026-09-01T17:00:00")
    monkeypatch.setattr(tsd, "TRIPLE_SCREEN_TRACKER_DB_PATH", db_path)
    monkeypatch.setattr(tsd, "DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS", 999)
    tsd._cache["ts"] = 0.0
    tsd._cache["rows"] = None

    first = tsd.build_triple_screen_tracker_state()
    # A second ticker created after the first read must NOT appear yet --
    # confirms the TTL cache is actually being served, not re-read every call.
    store.create_tracked(conn, "MSFT", "2026-09-01", 200.0, created_at="2026-09-01T17:00:00")
    second = tsd.build_triple_screen_tracker_state()

    assert first == second
    assert len(second["open"]) == 1
