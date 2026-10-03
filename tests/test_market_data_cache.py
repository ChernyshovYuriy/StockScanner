"""
tests/test_market_data_cache.py
===============================
Regression tests for the "stale cache tail" bug (found 2026-10).

market_data_cache.sync_tickers() used to (1) pass the requested `end` to
yfinance as-is — but yfinance's `end` is EXCLUSIVE, so that day's bar was
never fetched — and (2) skip the tail top-up entirely unless the newest
cached bar was more than 5 calendar days old. A daily caller's cache
therefore advanced in multi-day jumps:

  * kangaroo_pipeline.py checks only the LAST cached bar, so every skipped
    day was never evaluated — a real Kangaroo Tail on DOL.TO 2026-09-16 was
    missed this way, and the sleeve sat silent for three weeks;
  * the Scanner Board showed prices up to 5 trading days old.

Yahoo Finance is replaced by a fake with yfinance's own semantics (end
exclusive, nothing after "now"), and the clock is pinned with
set_backtest_clock(), so every test is offline and deterministic.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import market_data_cache
from kangaroo_pipeline import _scan_for_new_signals
from market_data_cache import sync_and_load
from time_utils import (
    TSX_TZ,
    last_trading_day_on_or_before,
    previous_trading_day,
    set_backtest_clock,
)


# ─────────────────────────────────────────────────────────────────────────────
# FAKE YAHOO
# ─────────────────────────────────────────────────────────────────────────────

def _trading_days(start: str, end: str) -> list[pd.Timestamp]:
    days, d = [], last_trading_day_on_or_before(pd.Timestamp(end).date())
    first = pd.Timestamp(start).date()
    while d >= first:
        days.append(pd.Timestamp(d))
        d = previous_trading_day(d)
    return sorted(days)


def _flat_bars(start: str, end: str, close: float = 100.0) -> pd.DataFrame:
    idx = _trading_days(start, end)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000},
        index=pd.DatetimeIndex(idx),
    )


class FakeYahoo:
    """Serves `truth` bars like yf.download: [start, end) with `end`
    EXCLUSIVE, and never a bar dated after the pinned clock's date. A bar
    for "today" requested before 16:00 comes back as `partial_today` if set
    (a still-forming session bar)."""

    def __init__(self, truth: dict[str, pd.DataFrame]):
        self.truth = truth
        self.partial_today: dict[str, pd.Series] = {}
        self.calls: list[tuple[list[str], str, str]] = []

    def __call__(self, *_, **__):
        return self  # stands in for the LiveDataProvider(...) constructor

    def download_range(self, tickers, start, end):
        from time_utils import market_now
        now = market_now()
        self.calls.append((list(tickers), start, end))
        data, failed = {}, []
        for t in tickers:
            df = self.truth[t]
            df = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
            df = df[df.index.date <= now.date()].copy()
            today = pd.Timestamp(now.date())
            if t in self.partial_today and today in df.index and now.hour < 16:
                df.loc[today] = self.partial_today[t]
            if df.empty:
                failed.append(t)
            else:
                data[t] = df
        return data, failed


@pytest.fixture
def cache(tmp_path, monkeypatch):
    path = str(tmp_path / "market_cache.db")
    real_connect = market_data_cache._connect
    monkeypatch.setattr(market_data_cache, "_connect", lambda db_path=path: real_connect(db_path))
    yield
    set_backtest_clock(None)


def _install(monkeypatch, truth) -> FakeYahoo:
    fake = FakeYahoo(truth)
    monkeypatch.setattr(market_data_cache, "LiveDataProvider", fake)
    return fake


def _at(day: str, hour: int = 16, minute: int = 30) -> None:
    d = pd.Timestamp(day)
    set_backtest_clock(datetime(d.year, d.month, d.day, hour, minute, tzinfo=TSX_TZ))


def _last_bar(data: dict, ticker: str) -> pd.Timestamp:
    return data[ticker].index.max()


# ─────────────────────────────────────────────────────────────────────────────
# The requested end day is included (yfinance's exclusive end)
# ─────────────────────────────────────────────────────────────────────────────

def test_requested_end_day_bar_is_included(cache, monkeypatch):
    _install(monkeypatch, {"DOL.TO": _flat_bars("2026-08-01", "2026-09-30")})
    _at("2026-09-16")

    data = sync_and_load(["DOL.TO"], start="2026-08-01", end="2026-09-16", quiet=True)

    assert _last_bar(data, "DOL.TO") == pd.Timestamp("2026-09-16")


# ─────────────────────────────────────────────────────────────────────────────
# A daily caller's cache advances exactly one bar per trading day
# ─────────────────────────────────────────────────────────────────────────────

def test_daily_runs_advance_cache_one_bar_per_trading_day(cache, monkeypatch):
    _install(monkeypatch, {"DOL.TO": _flat_bars("2026-06-01", "2026-10-02")})
    _at("2026-09-01")
    sync_and_load(["DOL.TO"], start="2026-06-01", end="2026-09-01", quiet=True)

    for day in _trading_days("2026-09-02", "2026-10-02"):
        _at(day.date().isoformat())
        data = sync_and_load(["DOL.TO"], start="2026-06-01", end=day.date().isoformat(), quiet=True)
        assert _last_bar(data, "DOL.TO") == day, f"run on {day.date()} saw a stale last bar"


# ─────────────────────────────────────────────────────────────────────────────
# No pointless re-downloads when the cache is already current
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("run_day,last_trading_day", [
    ("2026-09-19", "2026-09-18"),  # Saturday -> Friday's bar is the newest
    ("2026-09-20", "2026-09-18"),  # Sunday
    ("2026-10-12", "2026-10-09"),  # Thanksgiving Monday (TSX holiday)
])
def test_weekend_and_holiday_runs_do_not_refetch(cache, monkeypatch, run_day, last_trading_day):
    fake = _install(monkeypatch, {"DOL.TO": _flat_bars("2026-08-01", "2026-10-30")})
    _at(last_trading_day, hour=17)
    sync_and_load(["DOL.TO"], start="2026-08-01", end=last_trading_day, quiet=True)
    fake.calls.clear()

    _at(run_day, hour=12)
    data = sync_and_load(["DOL.TO"], start="2026-08-01", end=run_day, quiet=True)

    assert fake.calls == []
    assert _last_bar(data, "DOL.TO") == pd.Timestamp(last_trading_day)


def test_historical_range_is_fetched_once(cache, monkeypatch):
    """Backtests (run_backtest.py via HistoricalSliceProvider.from_cache)
    ask for a fixed past window — the second run must be served from cache."""
    fake = _install(monkeypatch, {"RY.TO": _flat_bars("2025-01-01", "2025-12-31")})
    _at("2026-10-02")
    first = sync_and_load(["RY.TO"], start="2025-01-01", end="2025-06-30", quiet=True)
    fake.calls.clear()

    second = sync_and_load(["RY.TO"], start="2025-01-01", end="2025-06-30", quiet=True)

    assert fake.calls == []
    assert _last_bar(first, "RY.TO") == _last_bar(second, "RY.TO") == pd.Timestamp("2025-06-30")


def test_future_end_is_capped_at_last_available_bar(cache, monkeypatch):
    fake = _install(monkeypatch, {"DOL.TO": _flat_bars("2026-08-01", "2026-10-30")})
    _at("2026-09-19", hour=12)  # Saturday
    sync_and_load(["DOL.TO"], start="2026-08-01", end="2026-09-25", quiet=True)
    fake.calls.clear()

    data = sync_and_load(["DOL.TO"], start="2026-08-01", end="2026-09-25", quiet=True)

    assert fake.calls == []
    assert _last_bar(data, "DOL.TO") == pd.Timestamp("2026-09-18")


# ─────────────────────────────────────────────────────────────────────────────
# A still-forming intraday bar is never trusted by a later same-day run
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_today_bar_replaced_by_later_same_day_run(cache, monkeypatch):
    truth = _flat_bars("2026-08-01", "2026-09-30")
    truth.loc[pd.Timestamp("2026-09-16"), ["Close", "Volume"]] = [174.56, 5000]
    fake = _install(monkeypatch, {"DOL.TO": truth})
    fake.partial_today["DOL.TO"] = pd.Series(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 99.0, "Volume": 10})

    _at("2026-09-16", hour=11)
    midday = sync_and_load(["DOL.TO"], start="2026-08-01", end="2026-09-16", quiet=True)
    assert midday["DOL.TO"].loc["2026-09-16", "Close"] == 99.0  # sanity: partial was served

    _at("2026-09-16", hour=16, minute=30)
    after_close = sync_and_load(["DOL.TO"], start="2026-08-01", end="2026-09-16", quiet=True)

    assert after_close["DOL.TO"].loc["2026-09-16", "Close"] == 174.56
    assert after_close["DOL.TO"].loc["2026-09-16", "Volume"] == 5000


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: the missed DOL.TO 2026-09-16 Kangaroo Tail
# ─────────────────────────────────────────────────────────────────────────────

def _bars_with_tail_on(tail_day: str) -> pd.DataFrame:
    """Quiet flat history (ATR ~2) plus one textbook bullish Kangaroo Tail:
    a deep flush well below the prior 10-bar low (5.5-pt lower wick, ~2.75
    ATR, 85% of range), tiny body, close near the top, 2x average volume."""
    bars = _flat_bars("2026-06-01", "2026-10-02")
    bars.loc[pd.Timestamp(tail_day)] = {
        "Open": 99.5, "High": 100.5, "Low": 94.0, "Close": 100.0, "Volume": 2000}
    return bars


def test_kangaroo_daily_scan_catches_tail_on_its_own_day(cache, monkeypatch):
    """Replays kangaroo_pipeline.py's 16:30 run every trading day, Sep 1 to
    Oct 2. Before the fix the Sep 16 bar was never the last cached bar on
    any run, so the detector never saw it."""
    _install(monkeypatch, {"DOL.TO": _bars_with_tail_on("2026-09-16")})
    _at("2026-09-01")
    sync_and_load(["DOL.TO"], start="2026-06-01", end="2026-09-01", quiet=True)

    detections = {}
    for day in _trading_days("2026-09-02", "2026-10-02"):
        iso = day.date().isoformat()
        _at(iso)
        bars = sync_and_load(["DOL.TO"], start="2026-06-01", end=iso, quiet=True)
        for intent in _scan_for_new_signals(["DOL.TO"], bars, set()):
            detections[iso] = intent

    assert list(detections) == ["2026-09-16"]
    intent = detections["2026-09-16"]
    assert intent["signal_date"] == "2026-09-16"
    assert intent["entry_price_planned"] == 100.5
    assert intent["stop_price"] == 94.0
