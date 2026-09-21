"""Offline tests for news_watchlist.store persistence (no network)."""

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


def test_seeded_guids_tracks_every_guid_ever_seeded(tmp_path):
    conn = store.connect(tmp_path / "nw.db")
    store.seed_inbox_item(
        conn, guid="g1", ticker="A", company=None, category=None, materiality=None,
        summary=None, source_link=None, flagged_at="2026-09-20", flag_price=1.0,
        created_at="2026-09-20T17:10:00",
    )
    assert store.seeded_guids(conn) == {"g1"}


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
