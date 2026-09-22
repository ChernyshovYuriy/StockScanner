"""Offline tests for news_watchlist.store persistence (no network)."""

import sqlite3

from news_watchlist import store


def test_seed_inbox_item_lands_in_inbox_status(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.seed_inbox_item(
        conn, guid="g1", ticker="OMI.V", company="Orosur Mining", category="exploration_drilling",
        materiality="high", summary="Drilling results announced.", source_link="https://example.com/g1",
        flagged_at="2026-09-20", flag_price=0.12, created_at="2026-09-20T17:10:00",
    )
    items = store.list_by_status(conn, "inbox")
    assert len(items) == 1
    assert items[0]["id"] == item_id
    assert items[0]["ticker"] == "OMI.V"
    assert items[0]["flag_price"] == 0.12


def test_mark_guid_processed_and_seeded_guids_round_trip(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    store.mark_guid_processed(conn, "g1", "2026-09-20T17:10:00")
    assert store.seeded_guids(conn) == {"g1"}
    # Idempotent -- a same-run race re-marking the same guid must not raise.
    store.mark_guid_processed(conn, "g1", "2026-09-20T17:11:00")
    assert store.seeded_guids(conn) == {"g1"}


def test_seeded_guids_backfills_from_existing_watchlist_items_on_connect(tmp_path):
    """A DB created before seeded_release_guids existed still has its
    already-seeded guids remembered -- connect()'s schema script backfills
    them from watchlist_items.guid the first time it runs against it."""
    db_path = tmp_path / "nw.db"
    conn = store.connect(db_path)
    store.seed_inbox_item(
        conn, guid="g1", ticker="A", company=None, category=None, materiality=None,
        summary=None, source_link=None, flagged_at="2026-09-20", flag_price=1.0,
        created_at="2026-09-20T17:10:00",
    )
    conn.close()

    # Simulate a pre-migration DB by dropping the new table, then
    # reconnect through store.connect() the way the service actually would.
    raw = sqlite3.connect(db_path)
    raw.execute("DROP TABLE seeded_release_guids")
    raw.commit()
    raw.close()

    conn = store.connect(db_path)
    assert store.seeded_guids(conn) == {"g1"}


def test_find_pending_inbox_item_returns_none_when_there_is_no_inbox_row(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    assert store.find_pending_inbox_item(conn, "A") is None


def test_find_pending_inbox_item_ignores_a_non_inbox_status(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.seed_inbox_item(
        conn, guid="g1", ticker="A", company=None, category=None, materiality=None,
        summary=None, source_link=None, flagged_at="2026-09-20", flag_price=1.0,
        created_at="2026-09-20T17:10:00",
    )
    store.set_status(conn, item_id, "watching", "2026-09-21T09:00:00")
    assert store.find_pending_inbox_item(conn, "A") is None


def test_seed_inbox_item_persists_yahoo_ticker(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.seed_inbox_item(
        conn, guid="g1", ticker="AYA", company="Aya Gold & Silver", category="exploration_drilling",
        materiality="high", summary="Drilling results announced.", source_link="https://example.com/g1",
        flagged_at="2026-09-20", flag_price=40.24, created_at="2026-09-20T17:10:00",
        yahoo_ticker="AYA.TO",
    )
    item = store.get_item(conn, item_id)
    assert item["ticker"] == "AYA"
    assert item["yahoo_ticker"] == "AYA.TO"


def test_seed_inbox_item_yahoo_ticker_defaults_to_none(tmp_path):
    """Every existing call site (and every pre-561102e row) omits this kwarg
    -- must not raise, and must read back as None rather than requiring a
    caller to pass it explicitly."""
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.seed_inbox_item(
        conn, guid="g1", ticker="A", company=None, category=None, materiality=None,
        summary=None, source_link=None, flagged_at="2026-09-20", flag_price=1.0,
        created_at="2026-09-20T17:10:00",
    )
    assert store.get_item(conn, item_id)["yahoo_ticker"] is None


def test_update_inbox_item_refreshes_yahoo_ticker(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.seed_inbox_item(
        conn, guid="g1", ticker="AYA", company="Old Co", category="old_cat", materiality="medium",
        summary="Old summary.", source_link="https://example.com/g1",
        flagged_at="2026-09-18", flag_price=1.0, created_at="2026-09-18T17:10:00",
        yahoo_ticker="AYA.TO",
    )
    store.update_inbox_item(
        conn, item_id, guid="g2", company="New Co", category="new_cat", materiality="high",
        summary="New summary.", source_link="https://example.com/g2",
        flagged_at="2026-09-21", flag_price=2.0, created_at="2026-09-21T09:15:00",
        yahoo_ticker="AYA.V",
    )
    assert store.get_item(conn, item_id)["yahoo_ticker"] == "AYA.V"


def test_set_yahoo_ticker_updates_only_that_field(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.seed_inbox_item(
        conn, guid="g1", ticker="AYA", company=None, category=None, materiality=None,
        summary=None, source_link=None, flagged_at="2026-09-20", flag_price=1.0,
        created_at="2026-09-20T17:10:00",
    )
    store.set_yahoo_ticker(conn, item_id, "AYA.TO")

    item = store.get_item(conn, item_id)
    assert item["yahoo_ticker"] == "AYA.TO"
    assert item["ticker"] == "AYA"


def test_update_inbox_item_refreshes_content_and_created_at_but_keeps_id(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.seed_inbox_item(
        conn, guid="g1", ticker="A", company="Old Co", category="old_cat", materiality="medium",
        summary="Old summary.", source_link="https://example.com/g1",
        flagged_at="2026-09-18", flag_price=1.0, created_at="2026-09-18T17:10:00",
    )

    store.update_inbox_item(
        conn, item_id, guid="g2", company="New Co", category="new_cat", materiality="high",
        summary="New summary.", source_link="https://example.com/g2",
        flagged_at="2026-09-21", flag_price=2.0, created_at="2026-09-21T09:15:00",
    )

    item = store.get_item(conn, item_id)
    assert item["id"] == item_id
    assert item["created_at"] == "2026-09-21T09:15:00"
    assert item["guid"] == "g2"
    assert item["company"] == "New Co"
    assert item["summary"] == "New summary."
    assert item["flagged_at"] == "2026-09-21"
    assert item["flag_price"] == 2.0
    assert len(store.list_by_status(conn, "inbox")) == 1


def test_add_manual_lands_directly_in_watching(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.add_manual(
        conn, ticker="AAPL", note="missed by the feed", flagged_at="2026-09-20",
        flag_price=190.0, created_at="2026-09-20T17:10:00",
    )
    assert store.list_by_status(conn, "inbox") == []
    watching = store.list_by_status(conn, "watching")
    assert len(watching) == 1
    assert watching[0]["id"] == item_id
    assert watching[0]["guid"] is None
    assert watching[0]["note"] == "missed by the feed"


def test_set_status_moves_item_between_lists(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.seed_inbox_item(
        conn, guid="g1", ticker="A", company=None, category=None, materiality=None,
        summary=None, source_link=None, flagged_at="2026-09-20", flag_price=1.0,
        created_at="2026-09-20T17:10:00",
    )
    store.set_status(conn, item_id, "watching", "2026-09-21T09:00:00")

    assert store.list_by_status(conn, "inbox") == []
    watching = store.list_by_status(conn, "watching")
    assert len(watching) == 1 and watching[0]["id"] == item_id
    assert store.get_item(conn, item_id)["status_changed_at"] == "2026-09-21T09:00:00"


def test_set_note_updates_only_the_note_field(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.add_manual(
        conn, ticker="AAPL", note="", flagged_at="2026-09-20", flag_price=190.0,
        created_at="2026-09-20T17:10:00",
    )
    store.set_note(conn, item_id, "bought a starter position")

    assert store.get_item(conn, item_id)["note"] == "bought a starter position"


def test_append_price_is_idempotent_for_a_rerun_same_day(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.add_manual(
        conn, ticker="AAPL", note="", flagged_at="2026-09-20", flag_price=190.0,
        created_at="2026-09-20T17:10:00",
    )
    store.append_price(conn, item_id, "2026-09-21", 195.0)
    store.append_price(conn, item_id, "2026-09-21", 197.0)  # same-day rerun overwrites, not duplicates

    history = store.price_history_for(conn, item_id)
    assert history == [{"date": "2026-09-21", "close_price": 197.0}]


def test_price_history_for_is_ordered_oldest_first(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.add_manual(
        conn, ticker="AAPL", note="", flagged_at="2026-09-18", flag_price=190.0,
        created_at="2026-09-18T17:10:00",
    )
    store.append_price(conn, item_id, "2026-09-20", 200.0)
    store.append_price(conn, item_id, "2026-09-19", 195.0)

    dates = [h["date"] for h in store.price_history_for(conn, item_id)]
    assert dates == ["2026-09-19", "2026-09-20"]


def test_dismissed_item_is_kept_not_deleted(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    item_id = store.seed_inbox_item(
        conn, guid="g1", ticker="A", company=None, category=None, materiality=None,
        summary=None, source_link=None, flagged_at="2026-09-20", flag_price=1.0,
        created_at="2026-09-20T17:10:00",
    )
    store.set_status(conn, item_id, "dismissed", "2026-09-21T09:00:00")

    dismissed = store.list_by_status(conn, "dismissed")
    assert len(dismissed) == 1 and dismissed[0]["id"] == item_id
