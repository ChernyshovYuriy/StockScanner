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
