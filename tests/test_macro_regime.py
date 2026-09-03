"""Offline tests for macro_regime.py (no network — CachedClient.request is monkeypatched)."""

import pytest

import macro_regime


# ── _consecutive_trend ────────────────────────────────────────────────────────

def test_consecutive_trend_rising_true_for_strictly_increasing_window():
    assert macro_regime._consecutive_trend([0.1, 0.2, 0.3], rising=True) is True


def test_consecutive_trend_rising_false_for_flat_or_mixed_window():
    assert macro_regime._consecutive_trend([0.1, 0.1, 0.3], rising=True) is False
    assert macro_regime._consecutive_trend([0.1, 0.3, 0.2], rising=True) is False


def test_consecutive_trend_falling_true_for_strictly_decreasing_window():
    assert macro_regime._consecutive_trend([0.3, 0.2, 0.1], rising=False) is True


def test_consecutive_trend_insufficient_history_returns_false():
    assert macro_regime._consecutive_trend([], rising=True) is False
    assert macro_regime._consecutive_trend([0.5], rising=True) is False
    assert macro_regime._consecutive_trend([], rising=False) is False


# ── _fetch_series ──────────────────────────────────────────────────────────────

def test_fetch_series_skips_dot_sentinel_missing_values(monkeypatch):
    monkeypatch.setattr(macro_regime, "FRED_API_KEY", "dummy")
    payload = {"observations": [
        {"date": "2026-08-30", "value": "0.6"},
        {"date": "2026-08-29", "value": "."},  # FRED's missing-value sentinel
        {"date": "2026-08-28", "value": "0.5"},
    ]}
    monkeypatch.setattr(macro_regime._client, "request", lambda *a, **k: payload)
    rows = macro_regime._fetch_series("T10Y2Y", 3)
    assert rows == [{"date": "2026-08-28", "value": 0.5}, {"date": "2026-08-30", "value": 0.6}]


def test_fetch_series_returns_none_without_api_key(monkeypatch):
    monkeypatch.setattr(macro_regime, "FRED_API_KEY", "")
    assert macro_regime._fetch_series("T10Y2Y", 3) is None


def test_get_macro_regime_force_bypasses_cache(monkeypatch):
    """force=True must reach CachedClient.request's own force kwarg -- was
    previously a dead parameter, never actually plumbed through."""
    monkeypatch.setattr(macro_regime, "FRED_API_KEY", "dummy")
    seen_force = []

    def fake_request(url, *, cache_key=None, force=False, is_json=True):
        seen_force.append(force)
        return {"observations": [{"date": "d1", "value": "0.1"}, {"date": "d2", "value": "0.2"}]}

    monkeypatch.setattr(macro_regime._client, "request", fake_request)
    macro_regime.get_macro_regime(force=True)
    assert seen_force and all(f is True for f in seen_force)

    seen_force.clear()
    macro_regime.get_macro_regime(force=False)
    assert seen_force and all(f is False for f in seen_force)


# ── _vote_series ─────────────────────────────────────────────────────────────

def test_vote_series_returns_plus1_when_trend_matches_bullish_direction(monkeypatch):
    rows = [{"date": "d1", "value": 0.1}, {"date": "d2", "value": 0.2}, {"date": "d3", "value": 0.3}]
    monkeypatch.setattr(macro_regime, "_fetch_series", lambda series_id, n, force=False: rows)
    vote, detail = macro_regime._vote_series("T10Y2Y", 3, rising_is_bullish=True)
    assert vote == 1
    assert detail["latest"] == 0.3
    assert detail["as_of"] == "d3"


def test_vote_series_returns_minus1_when_trend_matches_bearish_direction(monkeypatch):
    rows = [{"date": "d1", "value": 0.3}, {"date": "d2", "value": 0.2}, {"date": "d3", "value": 0.1}]
    monkeypatch.setattr(macro_regime, "_fetch_series", lambda series_id, n, force=False: rows)
    vote, _ = macro_regime._vote_series("T10Y2Y", 3, rising_is_bullish=True)
    assert vote == -1


def test_vote_series_returns_zero_on_mixed_trend(monkeypatch):
    rows = [{"date": "d1", "value": 0.1}, {"date": "d2", "value": 0.3}, {"date": "d3", "value": 0.2}]
    monkeypatch.setattr(macro_regime, "_fetch_series", lambda series_id, n, force=False: rows)
    vote, _ = macro_regime._vote_series("T10Y2Y", 3, rising_is_bullish=True)
    assert vote == 0


def test_vote_series_returns_zero_and_error_detail_when_unavailable(monkeypatch):
    monkeypatch.setattr(macro_regime, "_fetch_series", lambda series_id, n, force=False: None)
    vote, detail = macro_regime._vote_series("T10Y2Y", 3, rising_is_bullish=True)
    assert vote == 0
    assert "error" in detail


# ── get_macro_regime ────────────────────────────────────────────────────────

@pytest.mark.parametrize("votes,expected_label", [
    ({"T10Y2Y": 1, "BAMLH0A0HYM2": 1, "WALCL": 1}, "risk_on"),
    ({"T10Y2Y": 1, "BAMLH0A0HYM2": 0, "WALCL": 0}, "risk_on"),
    ({"T10Y2Y": 0, "BAMLH0A0HYM2": 0, "WALCL": 0}, "neutral"),
    ({"T10Y2Y": 1, "BAMLH0A0HYM2": -1, "WALCL": 0}, "neutral"),
    ({"T10Y2Y": -1, "BAMLH0A0HYM2": 0, "WALCL": 0}, "risk_off"),
    ({"T10Y2Y": -1, "BAMLH0A0HYM2": -1, "WALCL": -1}, "risk_off"),
])
def test_get_macro_regime_composite_thresholds(monkeypatch, votes, expected_label):
    def fake_vote_series(series_id, lookback_n, rising_is_bullish, force=False):
        return votes[series_id], {}
    monkeypatch.setattr(macro_regime, "_vote_series", fake_vote_series)
    result = macro_regime.get_macro_regime()
    assert result["composite"] == sum(votes.values())
    assert result["label"] == expected_label
    assert result["votes"] == votes


def test_get_macro_regime_returns_neutral_when_fred_api_key_unset(monkeypatch):
    monkeypatch.setattr(macro_regime, "FRED_API_KEY", "")
    result = macro_regime.get_macro_regime()
    assert result["label"] == "neutral"
    assert result["composite"] == 0
    assert all(v == 0 for v in result["votes"].values())


def test_get_macro_regime_one_series_fetch_failure_does_not_crash_whole_call(monkeypatch):
    def fake_vote_series(series_id, lookback_n, rising_is_bullish, force=False):
        if series_id == "WALCL":
            raise RuntimeError("boom")
        return 1, {}
    monkeypatch.setattr(macro_regime, "_vote_series", fake_vote_series)
    result = macro_regime.get_macro_regime()
    assert result["votes"]["WALCL"] == 0
    assert "error" in result["detail"]["WALCL"]
    assert result["votes"]["T10Y2Y"] == 1
    assert result["votes"]["BAMLH0A0HYM2"] == 1
    # 1 + 1 + 0 (failed series) = 2 -> still risk_on despite the one failure
    assert result["label"] == "risk_on"
