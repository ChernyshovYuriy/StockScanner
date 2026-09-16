"""Offline tests for press_release_tracker.store persistence (no network)."""

from press_release_tracker import store
from press_release_tracker.feeds import FeedItem


def _item(guid="g1", **overrides):
    defaults = dict(
        guid=guid, feed_url="https://example.com/feed", title="Some Release",
        link="https://example.com/a.html", pubdate="Tue, 15 Sep 2026 08:36:00 GMT",
        description="desc", categories=["OMI", "Mining"],
    )
    defaults.update(overrides)
    return FeedItem(**defaults)


def test_is_seen_false_until_marked(tmp_path):
    conn = store.connect(tmp_path / "pr.db")
    assert store.is_seen(conn, "g1") is False
    store.mark_seen(conn, _item("g1"), first_seen_at="2026-09-15T08:40:00")
    assert store.is_seen(conn, "g1") is True


def test_mark_seen_is_idempotent(tmp_path):
    conn = store.connect(tmp_path / "pr.db")
    store.mark_seen(conn, _item("g1"), first_seen_at="2026-09-15T08:40:00")
    store.mark_seen(conn, _item("g1"), first_seen_at="2026-09-15T08:45:00")  # re-fetched, same guid

    rows = conn.execute("SELECT COUNT(*) FROM seen_items WHERE guid='g1'").fetchone()
    assert rows[0] == 1


def test_unemailed_includes_unparsed_item_with_null_llm_fields(tmp_path):
    conn = store.connect(tmp_path / "pr.db")
    store.mark_seen(conn, _item("g1"), first_seen_at="2026-09-15T08:40:00")

    rows = store.unemailed(conn)
    assert len(rows) == 1
    assert rows[0]["guid"] == "g1"
    assert rows[0]["ticker"] is None
    assert rows[0]["summary"] is None


def test_unemailed_joins_parsed_fields_when_present(tmp_path):
    conn = store.connect(tmp_path / "pr.db")
    store.mark_seen(conn, _item("g1"), first_seen_at="2026-09-15T08:40:00")
    store.save_parsed(conn, "g1", {
        "ticker": "OMI.V", "company": "Orosur Mining Inc.", "category": "exploration_drilling",
        "materiality": "high", "summary": "Drilling results announced.",
    }, model="gpt-5-nano", parsed_at="2026-09-15T08:41:00")

    rows = store.unemailed(conn)
    assert rows[0]["ticker"] == "OMI.V"
    assert rows[0]["materiality"] == "high"


def test_mark_emailed_excludes_from_unemailed(tmp_path):
    conn = store.connect(tmp_path / "pr.db")
    store.mark_seen(conn, _item("g1"), first_seen_at="2026-09-15T08:40:00")
    store.mark_seen(conn, _item("g2"), first_seen_at="2026-09-15T08:41:00")

    store.mark_emailed(conn, ["g1"])

    remaining = {r["guid"] for r in store.unemailed(conn)}
    assert remaining == {"g2"}


def test_save_parsed_overwrites_on_reparse(tmp_path):
    conn = store.connect(tmp_path / "pr.db")
    store.mark_seen(conn, _item("g1"), first_seen_at="2026-09-15T08:40:00")
    store.save_parsed(conn, "g1", {"ticker": "A", "company": None, "category": "other",
                                    "materiality": "low", "summary": "x"},
                       model="gpt-5-nano", parsed_at="t1")
    store.save_parsed(conn, "g1", {"ticker": "B", "company": None, "category": "other",
                                    "materiality": "high", "summary": "y"},
                       model="gpt-5-nano", parsed_at="t2")

    rows = store.unemailed(conn)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "B"
    assert rows[0]["materiality"] == "high"


def test_unemailed_ordered_oldest_first(tmp_path):
    conn = store.connect(tmp_path / "pr.db")
    store.mark_seen(conn, _item("g2"), first_seen_at="2026-09-15T09:00:00")
    store.mark_seen(conn, _item("g1"), first_seen_at="2026-09-15T08:00:00")

    guids = [r["guid"] for r in store.unemailed(conn)]
    assert guids == ["g1", "g2"]


def test_last_batch_sent_at_none_until_set(tmp_path):
    conn = store.connect(tmp_path / "pr.db")
    assert store.get_last_batch_sent_at(conn) is None

    store.set_last_batch_sent_at(conn, "2026-09-15T09:00:00-04:00")
    assert store.get_last_batch_sent_at(conn) == "2026-09-15T09:00:00-04:00"


def test_set_last_batch_sent_at_overwrites_not_duplicates(tmp_path):
    conn = store.connect(tmp_path / "pr.db")
    store.set_last_batch_sent_at(conn, "2026-09-15T09:00:00-04:00")
    store.set_last_batch_sent_at(conn, "2026-09-15T10:00:00-04:00")

    assert store.get_last_batch_sent_at(conn) == "2026-09-15T10:00:00-04:00"
    assert conn.execute("SELECT COUNT(*) FROM batch_state").fetchone()[0] == 1
