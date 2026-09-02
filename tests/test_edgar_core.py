"""Offline test for edgar.core's caching layer (no network)."""

import json

from edgar import core


def test_fetch_submissions_refetches_on_a_new_day(tmp_path, monkeypatch):
    """Regression: submissions were cached with no date component, so a new
    Form 4 filed after the first fetch for a CIK was never seen again. The
    cache key must vary with _cache_date() so each day re-fetches."""
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)

    calls = {"n": 0}

    class FakeResp:
        def __init__(self, body):
            self.text = body

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=30):
        calls["n"] += 1
        return FakeResp(json.dumps({"call": calls["n"]}))

    monkeypatch.setattr(core._session, "get", fake_get)
    monkeypatch.setattr(core._limiter, "acquire", lambda: None)

    monkeypatch.setattr(core, "_cache_date", lambda: "20260604")
    first = core.fetch_submissions(1234)
    assert first == {"call": 1}
    assert calls["n"] == 1

    # Same day, same CIK: served from cache, no second network call.
    same_day_again = core.fetch_submissions(1234)
    assert same_day_again == first
    assert calls["n"] == 1

    # New day: cache key changes, so it actually re-fetches.
    monkeypatch.setattr(core, "_cache_date", lambda: "20260605")
    next_day = core.fetch_submissions(1234)
    assert next_day == {"call": 2}
    assert calls["n"] == 2


def test_load_ticker_map_refetches_on_a_new_day(tmp_path, monkeypatch):
    """Regression: the ticker/CIK map was cached forever (cache_key was a
    fixed filename), so a new IPO or ticker/CIK reassignment would never
    resolve once the map had been fetched once. Same date-scoping fix as
    fetch_submissions above."""
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)

    calls = {"n": 0}

    class FakeResp:
        def __init__(self, body):
            self.text = body

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=30):
        calls["n"] += 1
        # Day 2 adds a new listing that day-1's snapshot couldn't have had.
        rows = {"0": {"ticker": "MU", "cik_str": 723125}}
        if calls["n"] > 1:
            rows["1"] = {"ticker": "NEWCO", "cik_str": 999999}
        return FakeResp(json.dumps(rows))

    monkeypatch.setattr(core._session, "get", fake_get)
    monkeypatch.setattr(core._limiter, "acquire", lambda: None)

    monkeypatch.setattr(core, "_cache_date", lambda: "20260604")
    first = core.load_ticker_map()
    assert first == {"MU": 723125}
    assert calls["n"] == 1

    # Same day: served from cache, no second network call.
    same_day_again = core.load_ticker_map()
    assert same_day_again == first
    assert calls["n"] == 1

    # New day: re-fetches and picks up the new listing.
    monkeypatch.setattr(core, "_cache_date", lambda: "20260605")
    next_day = core.load_ticker_map()
    assert next_day == {"MU": 723125, "NEWCO": 999999}
    assert calls["n"] == 2
