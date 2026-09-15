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


def _volumes(*values: float) -> pd.Series:
    idx = pd.bdate_range(end="2026-09-15", periods=len(values))
    return pd.Series(values, index=idx)


def test_spike_above_average_is_included_and_ranked():
    volume_by_ticker = {
        # 20 days of 100k, today 250k -> +150% spike
        "AAA": _volumes(*([100_000] * 20 + [250_000])),
        # 20 days of 100k, today 130k -> +30% spike
        "BBB": _volumes(*([100_000] * 20 + [130_000])),
    }
    rows = vss.compute_spikes(volume_by_ticker)
    assert [r.ticker for r in rows] == ["AAA", "BBB"]
    assert rows[0].spike_pct == pytest.approx(150.0)
    assert rows[0].current_volume == 250_000
    assert rows[0].average_volume == pytest.approx(100_000.0)


def test_at_or_below_average_is_excluded():
    volume_by_ticker = {
        "FLAT": _volumes(*([100_000] * 20 + [100_000])),   # exactly average
        "LOW": _volumes(*([100_000] * 20 + [50_000])),      # below average
    }
    rows = vss.compute_spikes(volume_by_ticker)
    assert rows == []


def test_zero_average_is_excluded_not_a_division_error():
    volume_by_ticker = {"ZERO": _volumes(*([0] * 20 + [1_000]))}
    rows = vss.compute_spikes(volume_by_ticker)
    assert rows == []


def test_insufficient_history_is_skipped():
    volume_by_ticker = {"NEW": _volumes(500_000)}
    rows = vss.compute_spikes(volume_by_ticker)
    assert rows == []


def test_average_window_only_uses_trailing_avg_volume_days():
    # 30 days of 200k far in the past, then 20 days of 100k, then today's
    # 150k spike -- the average must be based on the trailing 20 days
    # (100k), not diluted by the older 200k history.
    volumes = [200_000] * 30 + [100_000] * 20 + [150_000]
    rows = vss.compute_spikes({"AAA": _volumes(*volumes)})
    assert len(rows) == 1
    assert rows[0].average_volume == pytest.approx(100_000.0)
    assert rows[0].spike_pct == pytest.approx(50.0)


class _FakeProvider:
    def __init__(self, data):
        self._data = data

    def download_range(self, tickers, start, end):
        return {t: self._data[t] for t in tickers if t in self._data}, []


def test_scan_volume_spikes_uses_download_range(monkeypatch):
    df = pd.DataFrame({"Volume": [100_000] * 20 + [300_000]},
                       index=pd.bdate_range(end="2026-09-15", periods=21))
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
