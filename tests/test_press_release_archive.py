"""Offline tests for press_release_tracker/archive.py -- no network, no
OpenAI (the LLM client and the shares/bars fetchers are always fakes)."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import news_watchlist_service
from news_watchlist import store as nw_store
from press_release_tracker import archive, store

_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
  <url>
    <loc>https://www.globenewswire.com/news-release/2025/03/31/1/0/en/abc-closes-placement.html</loc>
    <news:news>
      <news:publication_date>2025-03-31T13:00:00+00:00</news:publication_date>
      <news:title>ABC Closes $2M Private Placement</news:title>
      <news:stock_tickers>Other OTC:ABCCF, TSX Venture Exchange:ABC</news:stock_tickers>
    </news:news>
  </url>
  <url>
    <loc>https://www.globenewswire.com/news-release/2025/03/31/2/0/en/us-co.html</loc>
    <news:news>
      <news:publication_date>2025-03-31T14:00:00+00:00</news:publication_date>
      <news:title>US Co Reports</news:title>
      <news:stock_tickers>Nasdaq:USCO</news:stock_tickers>
    </news:news>
  </url>
  <url>
    <loc>https://www.globenewswire.com/news-release/2025/03/31/3/0/en/no-tickers.html</loc>
    <news:news>
      <news:publication_date>2025-03-31T15:00:00+00:00</news:publication_date>
      <news:title>Private Co News</news:title>
    </news:news>
  </url>
</urlset>"""


# ── symbols / sitemap ────────────────────────────────────────────────────

def test_yahoo_candidates_maps_exchange_and_adds_fallbacks():
    assert archive.yahoo_candidates("TSX Venture Exchange:ABC") == ["ABC.V", "ABC.TO", "ABC.CN", "ABC.NE"]


def test_yahoo_candidates_prefers_tsx_over_other_canadian_lines():
    c = archive.yahoo_candidates("Canadian Stock Exchange:XYZ, Toronto Stock Exchange:XYZ")
    assert c[0] == "XYZ.TO"


def test_yahoo_candidates_class_and_unit_symbols():
    assert archive.yahoo_candidates("Toronto Stock Exchange:RCI.B")[0] == "RCI-B.TO"
    assert archive.yahoo_candidates("Toronto Stock Exchange:CAR.UN")[0] == "CAR-UN.TO"
    assert archive.yahoo_candidates("Toronto Stock Exchange:CAR-UN")[0] == "CAR-UN.TO"
    assert archive.yahoo_candidates("Canadian Stock Exchange:HG.CN")[0] == "HG.CN"


def test_yahoo_candidates_skips_warrants_and_non_canadian():
    assert archive.yahoo_candidates("TSX Venture Exchange:OGG.WT.V") == []
    assert archive.yahoo_candidates("Nasdaq:AAPL, Other OTC:ABCF") == []
    assert archive.yahoo_candidates("") == []
    c = archive.yahoo_candidates("TSX Venture Exchange:OGG.WT.V, TSX Venture Exchange:OGG.V")
    assert c[0] == "OGG.V"


def test_parse_sitemap_keeps_canadian_listed_only():
    [e] = archive.parse_sitemap(_SITEMAP)
    assert e["title"] == "ABC Closes $2M Private Placement"
    assert e["candidates"][0] == "ABC.V"
    assert e["published"] == datetime(2025, 3, 31, 13, 0, tzinfo=timezone.utc)


def test_months_range_crosses_year():
    assert archive.months("2023-11", "2024-02") == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_import_is_idempotent_and_pubdate_round_trips(tmp_path):
    from news_watchlist.outcomes import parse_published

    conn = archive.connect(tmp_path / "a.db")
    entries = archive.parse_sitemap(_SITEMAP)
    assert archive.import_entries(conn, entries, "now") == 1
    assert archive.import_entries(conn, entries, "now") == 0
    pubdate = conn.execute("SELECT pubdate FROM seen_items").fetchone()[0]
    assert parse_published(pubdate) == datetime(2025, 3, 31, 13, 0, tzinfo=timezone.utc)
    assert archive.candidates_by_guid(conn)[entries[0]["link"]][0] == "ABC.V"


# ── classification ───────────────────────────────────────────────────────

def _fake_client(reply):
    def create(**kwargs):
        content = reply(json.loads(kwargs["messages"][1]["content"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_classify_batch_maps_ids_and_sanitizes_labels():
    client = _fake_client(lambda items: json.dumps({"items": [
        {"id": 0, "category": "financing", "materiality": "medium"},
        {"id": 1, "category": "made_up", "materiality": "huge"},
        {"id": 9, "category": "earnings", "materiality": "low"},
    ]}))
    out = archive.classify_batch(client, ["a", "b"])
    assert out == {0: ("financing", "medium"), 1: ("other", "low")}


def test_classify_batch_failed_call_is_none():
    def boom(**kwargs):
        raise RuntimeError("x")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    assert archive.classify_batch(client, ["a"]) is None


# ── shares at publication ────────────────────────────────────────────────

def test_shares_at_uses_last_reading_before_publication_and_falls_through_candidates():
    pub = datetime(2025, 3, 31, 13, 0, tzinfo=timezone.utc)
    series = pd.Series([100.0, 120.0, 200.0],
                       index=pd.DatetimeIndex(["2025-01-01", "2025-03-01", "2025-04-15"], tz="America/New_York"))

    def fetch(symbol, start, end):
        if symbol == "ABC.V":
            return None
        return series

    assert archive.shares_at(["ABC.V", "ABC.TO"], pub, fetch=fetch) == ("ABC.TO", 120.0)


def test_shares_at_nothing_found():
    pub = datetime(2025, 3, 31, tzinfo=timezone.utc)
    assert archive.shares_at(["X.V"], pub, fetch=lambda *a: None) == (None, None)


# ── scoring through the shared scorer ────────────────────────────────────

def test_score_uses_archive_candidates_and_reports_survivorship(tmp_path, monkeypatch):
    db = tmp_path / "a.db"
    conn = archive.connect(db)
    archive.import_entries(conn, archive.parse_sitemap(_SITEMAP), "now")
    guid = conn.execute("SELECT guid FROM seen_items").fetchone()[0]
    store.save_parsed(conn, guid, {"ticker": "ABC.V", "category": "financing", "materiality": "medium"},
                      "m", "now")

    idx = pd.bdate_range("2025-03-20", periods=90)
    bars = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.1, "Volume": 1}, index=idx)
    bench = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1}, index=idx)
    requested = []

    def fake_download(symbols, start, end):
        requested.extend(symbols)
        return {"ABC.TO": bars, "XIU.TO": bench}   # moved to TSX since: only the fallback has data

    monkeypatch.setattr(news_watchlist_service, "_download_bars", fake_download)
    archive.step_score(conn, db_path=db)

    assert "ABC.V" in requested and "ABC.TO" in requested
    [row] = nw_store.list_scored_outcomes(nw_store.connect(db))
    assert row["yahoo_ticker"] == "ABC.TO"
    assert row["status"] == "complete"
    assert archive.survivorship(conn) == {"financing_releases": 1, "scored": 1, "no_yahoo_data": 0}
    assert "Financing releases: 1, scored: 1" in archive.step_report(db_path=db)


def test_bare_url_sitemap_yields_nothing():
    bare = ('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url>'
            '<loc>/news-release/2026/06/30/1/0/en/x.html</loc></url></urlset>')
    assert archive.parse_sitemap(bare) == []
