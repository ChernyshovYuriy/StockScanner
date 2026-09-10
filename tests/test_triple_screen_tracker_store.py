"""Offline tests for triple_screen_tracker.store persistence (no network)."""

from triple_screen_tracker import store


def test_create_tracked_appears_in_open_tracked_tickers(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    store.create_tracked(conn, "AAPL", "2026-09-01", 100.0, created_at="2026-09-01T17:00:00")
    assert store.open_tracked_tickers(conn) == {"AAPL"}


def test_open_records_returns_open_rows_with_expected_fields(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    store.create_tracked(conn, "AAPL", "2026-09-01", 100.0, created_at="2026-09-01T17:00:00")
    rows = store.open_records(conn)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["buy_date"] == "2026-09-01"
    assert rows[0]["buy_price"] == 100.0


def test_append_price_idempotent_on_same_day_rerun(tmp_path):
    """A same-day rerun (e.g. a systemd retry) must replace that day's row,
    not accumulate a duplicate."""
    conn = store.connect(tmp_path / "ts.db")
    tid = store.create_tracked(conn, "AAPL", "2026-09-01", 100.0, created_at="2026-09-01T17:00:00")
    store.append_price(conn, tid, "2026-09-02", 101.0)
    store.append_price(conn, tid, "2026-09-02", 103.0)  # rerun, different close

    history = store.price_history_for(conn, tid)
    assert len(history) == 1
    assert history[0] == {"date": "2026-09-02", "close_price": 103.0}


def test_price_history_for_ordered_oldest_first(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    tid = store.create_tracked(conn, "AAPL", "2026-09-01", 100.0, created_at="2026-09-01T17:00:00")
    store.append_price(conn, tid, "2026-09-03", 103.0)
    store.append_price(conn, tid, "2026-09-01", 100.0)
    store.append_price(conn, tid, "2026-09-02", 101.0)

    dates = [h["date"] for h in store.price_history_for(conn, tid)]
    assert dates == ["2026-09-01", "2026-09-02", "2026-09-03"]


def test_close_tracked_transitions_status_and_excludes_from_open(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    tid = store.create_tracked(conn, "AAPL", "2026-09-01", 100.0, created_at="2026-09-01T17:00:00")
    store.close_tracked(conn, tid, "2026-09-05", 95.0)

    assert store.open_tracked_tickers(conn) == set()
    assert store.open_records(conn) == []

    all_rows = store.all_tracked(conn)
    assert len(all_rows) == 1
    assert all_rows[0]["status"] == "SOLD"
    assert all_rows[0]["sell_date"] == "2026-09-05"
    assert all_rows[0]["sell_price"] == 95.0


def test_ticker_can_be_tracked_again_after_being_sold(tmp_path):
    """A ticker's PK is `id`, not `ticker` -- a sold ticker can start a fresh
    cycle, preserving the old row as history rather than overwriting it."""
    conn = store.connect(tmp_path / "ts.db")
    tid1 = store.create_tracked(conn, "AAPL", "2026-09-01", 100.0, created_at="2026-09-01T17:00:00")
    store.close_tracked(conn, tid1, "2026-09-05", 95.0)
    tid2 = store.create_tracked(conn, "AAPL", "2026-09-10", 110.0, created_at="2026-09-10T17:00:00")

    assert tid2 != tid1
    assert store.open_tracked_tickers(conn) == {"AAPL"}
    assert len(store.all_tracked(conn)) == 2


def test_all_tracked_returns_both_statuses_newest_first(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    store.create_tracked(conn, "AAPL", "2026-09-01", 100.0, created_at="2026-09-01T17:00:00")
    tid2 = store.create_tracked(conn, "MSFT", "2026-09-05", 200.0, created_at="2026-09-05T17:00:00")
    store.close_tracked(conn, tid2, "2026-09-06", 190.0)

    rows = store.all_tracked(conn)
    assert {r["ticker"] for r in rows} == {"AAPL", "MSFT"}
    assert {r["status"] for r in rows} == {"OPEN", "SOLD"}
    assert rows[0]["buy_date"] == "2026-09-05"  # newest buy_date first


def test_already_sent_false_until_recorded_sent(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    assert store.already_sent(conn, "2026-09-01") is False

    store.record_email(conn, "2026-09-01", hit_count=0, sent=0)
    assert store.already_sent(conn, "2026-09-01") is False  # sent=0 doesn't count

    store.record_email(conn, "2026-09-01", hit_count=2, sent=1)
    assert store.already_sent(conn, "2026-09-01") is True
    assert store.already_sent(conn, "2026-09-02") is False  # different date
