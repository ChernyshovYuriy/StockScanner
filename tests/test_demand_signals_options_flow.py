"""Offline tests for demand_signals.options_flow (no network)."""

import pytest

from demand_signals.options_flow import (
    OptionsFlowProvider,
    OptionsSnapshot,
    YahooOptionsProvider,
    build_signals,
)

# Representative yfinance option_chain() leg shape (post fillna/to_dict).
FIXTURE_CHAIN = {
    "expiration": "2026-06-19",
    "calls": [
        {"volume": 500, "openInterest": 100, "strike": 100.0},
        {"volume": 300, "openInterest": 50, "strike": 105.0},
    ],
    "puts": [
        {"volume": 40, "openInterest": 200, "strike": 95.0},
    ],
}


# ── OptionsFlowProvider abstraction ─────────────────────────────────────────

def test_provider_is_abstract():
    with pytest.raises(TypeError):
        OptionsFlowProvider()


# ── YahooOptionsProvider._to_snapshot ───────────────────────────────────────

def test_to_snapshot_sums_volume_and_oi_across_legs():
    snap = YahooOptionsProvider._to_snapshot("MU", FIXTURE_CHAIN)
    assert snap.call_volume == 800
    assert snap.call_oi == 150
    assert snap.put_volume == 40
    assert snap.put_oi == 200
    assert len(snap.legs) == 3


def test_yahoo_provider_caches_fetch_result(tmp_path, monkeypatch):
    provider = YahooOptionsProvider(cache_dir=tmp_path)
    calls = {"n": 0}

    def fake_fetch(us_ticker):
        calls["n"] += 1
        return FIXTURE_CHAIN

    monkeypatch.setattr(provider, "_fetch", fake_fetch)

    first = provider.snapshot("MU")
    second = provider.snapshot("MU")

    assert first == second
    assert calls["n"] == 1  # second call served from cache


def test_yahoo_provider_returns_none_when_fetch_finds_nothing(tmp_path, monkeypatch):
    provider = YahooOptionsProvider(cache_dir=tmp_path)
    monkeypatch.setattr(provider, "_fetch", lambda us_ticker: None)
    assert provider.snapshot("NOOPTIONS") is None


# ── build_signals ────────────────────────────────────────────────────────────

def _snapshot(**overrides):
    fields = dict(us_ticker="MU", as_of_date="2026-06-05",
                  call_volume=0, put_volume=0, call_oi=0, put_oi=0)
    fields.update(overrides)
    return OptionsSnapshot(**fields)


def test_unusual_call_volume_fires_above_ratio_threshold():
    snap = _snapshot(call_volume=300, call_oi=100)  # ratio 3.0 >= 2.0 default
    signals = build_signals("MU", "MU", snap, fetched_at="t")
    types = {s.signal_type for s in signals}
    assert "unusual_call_volume" in types
    call_sig = next(s for s in signals if s.signal_type == "unusual_call_volume")
    assert call_sig.direction == "bullish"
    assert call_sig.source == "options_flow"
    assert call_sig.lag_days == 0


def test_unusual_put_volume_fires_above_ratio_threshold():
    snap = _snapshot(put_volume=250, put_oi=100)  # ratio 2.5 >= 2.0
    signals = build_signals("MU", "MU", snap, fetched_at="t")
    put_sig = next(s for s in signals if s.signal_type == "unusual_put_volume")
    assert put_sig.direction == "bearish"


def test_no_unusual_signal_below_threshold():
    snap = _snapshot(call_volume=100, call_oi=100, put_volume=50, put_oi=100)  # ratios 1.0, 0.5
    signals = build_signals("MU", "MU", snap, fetched_at="t")
    types = {s.signal_type for s in signals}
    assert "unusual_call_volume" not in types
    assert "unusual_put_volume" not in types


def test_call_put_skew_bullish_when_calls_dominate():
    snap = _snapshot(call_volume=800, call_oi=1000, put_volume=200, put_oi=1000)
    signals = build_signals("MU", "MU", snap, fetched_at="t")
    skew = next(s for s in signals if s.signal_type == "call_put_skew")
    assert skew.direction == "bullish"
    assert skew.strength == pytest.approx(0.6)  # |800-200|/1000


def test_call_put_skew_bearish_when_puts_dominate():
    snap = _snapshot(call_volume=100, call_oi=1000, put_volume=900, put_oi=1000)
    signals = build_signals("MU", "MU", snap, fetched_at="t")
    skew = next(s for s in signals if s.signal_type == "call_put_skew")
    assert skew.direction == "bearish"


def test_no_skew_signal_when_zero_total_volume():
    snap = _snapshot()  # all zero
    signals = build_signals("MU", "MU", snap, fetched_at="t")
    assert signals == []


def test_zero_open_interest_does_not_divide_by_zero():
    snap = _snapshot(call_volume=100, call_oi=0, put_volume=50, put_oi=0)
    # Must not raise ZeroDivisionError; only the skew signal (no OI needed) fires.
    signals = build_signals("MU", "MU", snap, fetched_at="t")
    assert {s.signal_type for s in signals} == {"call_put_skew"}
