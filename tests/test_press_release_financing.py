"""Offline tests for press_release_tracker/financing.py -- no network, no
OpenAI (extract_terms itself is only exercised with the key unset)."""
from press_release_tracker import financing, store
from press_release_tracker.feeds import FeedItem


def test_should_extract_only_english_financing_with_a_ticker():
    p = {"ticker": "ABC.V", "category": "financing"}
    assert financing.should_extract(p, "https://www.globenewswire.com/news-release/2026/10/05/1/0/en/x.html")
    assert not financing.should_extract(p, "https://www.globenewswire.com/news-release/2026/10/05/1/0/fr/x.html")
    assert not financing.should_extract({**p, "category": "earnings"}, "https://x/en")
    assert not financing.should_extract({**p, "ticker": None}, "https://x/en")
    assert not financing.should_extract(None, "https://x/en")


def test_normalize_coerces_numbers_and_currency():
    t = financing.normalize({"gross_proceeds": "2,500,000", "issue_price": "$0.10",
                             "securities_offered": 25000000, "warrant_coverage": "0.5",
                             "currency": "c$", "flow_through": True, "junk": 1})
    assert t["gross_proceeds"] == 2_500_000.0
    assert t["issue_price"] == 0.10
    assert t["warrant_coverage"] == 0.5
    assert t["currency"] == "CAD"
    assert t["flow_through"] is True
    assert "junk" not in t


def test_normalize_unparseable_number_is_none():
    assert financing.normalize({"issue_price": "TBD"})["issue_price"] is None


def test_normalize_flattens_lists_so_sqlite_can_store_them():
    t = financing.normalize({"use_of_proceeds": ["exploration", "working capital"],
                             "strategic_investor": []})
    assert t["use_of_proceeds"] == "exploration; working capital"
    assert t["strategic_investor"] is None


def test_extract_terms_without_key_is_none(monkeypatch):
    monkeypatch.setattr(financing, "OPENAI_API_KEY", None)
    assert financing.extract_terms("ABC.V", "t", "https://x", body="text", context={}) is None


def test_store_round_trip_and_backfill_query(tmp_path):
    conn = store.connect(tmp_path / "pr.db")
    for guid, cat in (("g1", "financing"), ("g2", "financing"), ("g3", "earnings")):
        store.mark_seen(conn, FeedItem(guid=guid, feed_url="f", title=guid, link=f"https://x/{guid}",
                                       pubdate="", description="", categories=[]), "2026-10-01T00:00:00")
        store.save_parsed(conn, guid, {"ticker": "ABC.V", "category": cat}, "m", "2026-10-01T00:00:00")
    store.save_financing_terms(conn, "g1", "ABC.V", {"is_financing": True, "issue_price": 0.1}, "m",
                               "2026-10-01T00:00:00")
    assert [r["guid"] for r in store.financing_without_terms(conn)] == ["g2"]
    assert conn.execute("SELECT is_financing, issue_price FROM financing_terms").fetchone() == (1, 0.1)
