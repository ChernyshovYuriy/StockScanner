"""
tests/test_volume_spike_scanner.py
====================================
Offline tests for volume_spike_scanner.py — compute_spikes() is pure
(no network), and scan_volume_spikes() is tested with
DEFAULT_PROVIDER.download_range monkeypatched to a fixture, never a real
network call.
"""
from __future__ import annotations

import pandas as pd
import pytest

import volume_spike_scanner as vss


def _ohlcv(volumes, closes=None) -> pd.DataFrame:
    """Build a minimal OHLCV frame (Close + Volume) for compute_spikes().
    Defaults to a flat price with today's close nudged up 1% above
    yesterday's, so tests focused on the volume math aren't incidentally
    excluded by the price-direction filter — tests that care about price
    direction pass their own `closes`.
    """
    idx = pd.bdate_range(end="2026-09-15", periods=len(volumes))
    if closes is None:
        closes = [100.0] * (len(volumes) - 1) + [101.0]
    return pd.DataFrame({"Close": closes, "Volume": volumes}, index=idx)


def test_spike_above_average_is_included_and_ranked():
    data_by_ticker = {
        # 20 days of 100k, today 250k -> +150% spike
        "AAA": _ohlcv([100_000] * 20 + [250_000]),
        # 20 days of 100k, today 130k -> +30% spike
        "BBB": _ohlcv([100_000] * 20 + [130_000]),
    }
    rows = vss.compute_spikes(data_by_ticker)
    assert [r.ticker for r in rows] == ["AAA", "BBB"]
    assert rows[0].spike_pct == pytest.approx(150.0)
    assert rows[0].current_volume == 250_000
    assert rows[0].average_volume == pytest.approx(100_000.0)


def test_at_or_below_average_is_excluded():
    data_by_ticker = {
        "FLAT": _ohlcv([100_000] * 20 + [100_000]),   # exactly average
        "LOW": _ohlcv([100_000] * 20 + [50_000]),      # below average
    }
    rows = vss.compute_spikes(data_by_ticker)
    assert rows == []


def test_zero_average_is_excluded_not_a_division_error():
    data_by_ticker = {"ZERO": _ohlcv([0] * 20 + [1_000])}
    rows = vss.compute_spikes(data_by_ticker)
    assert rows == []


def test_insufficient_history_is_skipped():
    data_by_ticker = {"NEW": _ohlcv([500_000])}
    rows = vss.compute_spikes(data_by_ticker)
    assert rows == []


def test_average_window_only_uses_trailing_avg_volume_days():
    # 30 days of 200k far in the past, then 20 days of 100k, then today's
    # 150k spike -- the average must be based on the trailing 20 days
    # (100k), not diluted by the older 200k history.
    volumes = [200_000] * 30 + [100_000] * 20 + [150_000]
    rows = vss.compute_spikes({"AAA": _ohlcv(volumes)})
    assert len(rows) == 1
    assert rows[0].average_volume == pytest.approx(100_000.0)
    assert rows[0].spike_pct == pytest.approx(50.0)


def test_price_down_excludes_despite_volume_spike():
    # Volume clearly spikes (+150%), but the close is BELOW yesterday's --
    # a volume spike on a down day reads as selling, not the buying
    # interest this scanner is for, so it must be excluded.
    volumes = [100_000] * 20 + [250_000]
    closes = [100.0] * 20 + [95.0]
    rows = vss.compute_spikes({"AAA": _ohlcv(volumes, closes)})
    assert rows == []


def test_price_flat_excludes_despite_volume_spike():
    volumes = [100_000] * 20 + [250_000]
    closes = [100.0] * 21  # today's close == yesterday's -- not "up"
    rows = vss.compute_spikes({"AAA": _ohlcv(volumes, closes)})
    assert rows == []


def test_price_up_with_volume_spike_is_included():
    volumes = [100_000] * 20 + [250_000]
    closes = [100.0] * 20 + [100.01]  # barely up still counts
    rows = vss.compute_spikes({"AAA": _ohlcv(volumes, closes)})
    assert len(rows) == 1
    assert rows[0].ticker == "AAA"


class _FakeProvider:
    def __init__(self, data):
        self._data = data

    def download_range(self, tickers, start, end):
        return {t: self._data[t] for t in tickers if t in self._data}, []


def test_scan_volume_spikes_uses_download_range(monkeypatch):
    df = _ohlcv([100_000] * 20 + [300_000])
    fake = _FakeProvider({"AAA": df})
    monkeypatch.setattr(vss, "DEFAULT_PROVIDER", fake)

    rows = vss.scan_volume_spikes(["AAA"])
    assert len(rows) == 1
    assert rows[0].ticker == "AAA"
    assert rows[0].spike_pct == pytest.approx(200.0)


def test_scan_volume_spikes_uses_ticker_list_url_by_default(monkeypatch):
    calls = {}

    def fake_load_tickers(url):
        calls["url"] = url
        return ["AAA"]

    monkeypatch.setattr(vss, "load_tickers", fake_load_tickers)
    monkeypatch.setattr(vss, "DEFAULT_PROVIDER", _FakeProvider({}))

    vss.scan_volume_spikes()
    assert calls["url"] == vss.VOLUME_SPIKE_TICKERS_URL


def test_scan_volume_spikes_liquid_universe_skips_ticker_list_url(monkeypatch):
    def fail_load_tickers(url):
        raise AssertionError("liquid universe must not fetch VOLUME_SPIKE_TICKERS_URL")

    seen = {}

    class _CapturingProvider(_FakeProvider):
        def download_range(self, tickers, start, end):
            seen["tickers"] = tickers
            return super().download_range(tickers, start, end)

    monkeypatch.setattr(vss, "load_tickers", fail_load_tickers)
    monkeypatch.setattr(vss, "DEFAULT_PROVIDER", _CapturingProvider({}))

    vss.scan_volume_spikes(universe="liquid")
    assert seen["tickers"] == vss.LIQUID_TICKERS


def test_scan_volume_spikes_explicit_tickers_override_universe(monkeypatch):
    monkeypatch.setattr(vss, "DEFAULT_PROVIDER", _FakeProvider({}))

    def fail_load_tickers(url):
        raise AssertionError("explicit tickers must not trigger a universe fetch")

    monkeypatch.setattr(vss, "load_tickers", fail_load_tickers)

    # Should not raise even though universe="liquid" is also passed --
    # an explicit tickers list always wins.
    vss.scan_volume_spikes(["AAA"], universe="liquid")
