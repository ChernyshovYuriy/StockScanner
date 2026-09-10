"""
Shared synthetic OHLCV fixture builders + test doubles for the Triple Screen
suite (test_triple_screen_*.py). NOT a test module itself -- no test_
functions here, so pytest does not collect this file directly; it only
provides helpers the test_triple_screen_*.py files import.

No network anywhere in this package's tests: every fixture here is
hand-constructed or formula-generated, and FakeDataProvider/FakeScreen below
exist specifically so test_triple_screen_engine.py and
test_triple_screen_batch.py never need a real indicator implementation or a
live data fetch to exercise composition/batch logic in isolation.
"""
import numpy as np
import pandas as pd

from research.triple_screen.data_provider import DataProvider
from research.triple_screen.screens import EntryScreen, TrendScreen, TriggerScreen
from research.triple_screen.types import (
    AlignedBars, Direction, EntryVerdict, PriceData, TimeframeConfig, TrendVerdict, TriggerVerdict,
)


# ── generic bar builder ──────────────────────────────────────────────────────

def make_bars(closes, timeframe="daily", ticker="TEST", start="2024-01-01", freq="D",
              pad=0.5, volume=1_000_000.0) -> PriceData:
    """PriceData from a list/array of close prices. Open = prior Close (first
    bar's Open = its own Close); High/Low padded a fixed amount around
    Open/Close so every bar has a well-defined, OHLC-consistent range.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + pad
    lows = np.minimum(opens, closes) - pad
    idx = pd.date_range(start, periods=n, freq=freq)
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                        "Volume": volume}, index=idx)
    return PriceData(ticker=ticker, timeframe=timeframe, bars=df)


# ── TrendScreen fixtures (Screen 1) ──────────────────────────────────────────
# 60 bars: comfortably past EMA-13's warmup plus the trend-lag comparison
# window, so the trend read is unambiguous regardless of the exact lag Phase
# 3 picks (as long as it's small relative to 60).

def uptrend_bars(n=60, start_price=100.0, daily_return=0.01, **kw) -> PriceData:
    """Strictly, geometrically rising closes -- unambiguous UP fixture: the
    EMA of a monotonically rising series is itself monotonically rising."""
    closes = start_price * (1 + daily_return) ** np.arange(n)
    return make_bars(closes, **kw)


def downtrend_bars(n=60, start_price=100.0, daily_return=0.01, **kw) -> PriceData:
    """Mirror of uptrend_bars: unambiguous DOWN fixture."""
    closes = start_price * (1 - daily_return) ** np.arange(n)
    return make_bars(closes, **kw)


def flat_bars(n=60, price=100.0, **kw) -> PriceData:
    """A perfectly constant close series -- guaranteed FLAT regardless of
    what "flat" tolerance/epsilon Phase 3 picks, since EMA(constant) is
    exactly that constant at every point: ema_now == ema_prior with no
    fuzzy comparison needed (mirrors elder_ray.py's own trend_at(), which
    treats exact equality as flat)."""
    closes = np.full(n, price, dtype=float)
    return make_bars(closes, **kw)


def insufficient_bars(n=5, price=100.0, **kw) -> PriceData:
    """Too short for any reasonable EMA-13 + lag lookback -- for the
    insufficient-data edge-case tests. A screen must return a safe default
    (FLAT / no-pullback / not-triggered), never raise."""
    closes = np.linspace(price, price * 1.05, n)
    return make_bars(closes, **kw)


def bars_with_nan(n=60, start_price=100.0, daily_return=0.01, nan_at=30, **kw) -> PriceData:
    """An otherwise-clean uptrend with one NaN Close in the middle -- a
    screen must not raise on this; the defined-behavior expectation (see
    test_triple_screen_edge_cases.py) is that a NaN bar is treated the same
    as missing/insufficient data for whatever lookback window it falls
    inside, not silently propagated into a NaN verdict.
    """
    pd_ = uptrend_bars(n=n, start_price=start_price, daily_return=daily_return, **kw)
    bars = pd_.bars.copy()
    bars.iloc[nan_at, bars.columns.get_loc("Close")] = float("nan")
    return PriceData(ticker=pd_.ticker, timeframe=pd_.timeframe, bars=bars)


# ── EntryScreen fixtures (Screen 2: 2-period Force Index) ───────────────────
# Force Index(1) = Volume x (Close - Close_prev); Screen 2's reference
# implementation smooths it with a 2-period EMA. Each fixture below is sized
# so the latest bar's sign is unambiguous under that formula regardless of
# the exact smoothing span Phase 3 uses, by making the pullback/rally day's
# move dominate the prior days' by a wide margin.

def pullback_in_uptrend_bars(**kw) -> PriceData:
    """Three up days then one clear down day -- Force Index goes negative on
    the latest bar while the broader trend (fed separately as Direction.UP
    to EntryScreen.evaluate) is still up: pullback present."""
    return make_bars([100, 102, 104, 106, 104], **kw)


def no_pullback_in_uptrend_bars(**kw) -> PriceData:
    """Steady, uninterrupted rise -- Force Index stays positive throughout:
    pullback absent."""
    return make_bars([100, 102, 104, 106, 108], **kw)


def rally_in_downtrend_bars(**kw) -> PriceData:
    """Mirror of pullback_in_uptrend_bars: three down days then one clear up
    day -- counter-rally present."""
    return make_bars([100, 98, 96, 94, 96], **kw)


def no_rally_in_downtrend_bars(**kw) -> PriceData:
    """Steady decline -- counter-rally absent."""
    return make_bars([100, 98, 96, 94, 92], **kw)


# ── TriggerScreen fixtures (Screen 3: prior-bar high/low breakout) ──────────
# Two bars are enough: today's High vs. yesterday's High (UP), or today's Low
# vs. yesterday's Low (DOWN). Built directly (not via make_bars) for exact
# High/Low control.

def _two_bar_frame(o0, h0, l0, c0, o1, h1, l1, c1, ticker="TEST", timeframe="daily") -> PriceData:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame({"Open": [o0, o1], "High": [h0, h1], "Low": [l0, l1],
                        "Close": [c0, c1], "Volume": [1_000_000.0, 1_000_000.0]}, index=idx)
    return PriceData(ticker=ticker, timeframe=timeframe, bars=df)


def breakout_above_prior_high_bars(**kw) -> PriceData:
    """Today's High (107) > yesterday's High (105) -- UP-direction trigger fires."""
    return _two_bar_frame(100, 105, 99, 104, 104, 107, 103, 106, **kw)


def no_breakout_above_prior_high_bars(**kw) -> PriceData:
    """Today's High (104) <= yesterday's High (105) -- UP-direction trigger does not fire."""
    return _two_bar_frame(100, 105, 99, 104, 103, 104, 102, 103, **kw)


def breakdown_below_prior_low_bars(**kw) -> PriceData:
    """Today's Low (93) < yesterday's Low (95) -- DOWN-direction trigger fires."""
    return _two_bar_frame(100, 101, 95, 96, 96, 97, 93, 94, **kw)


def no_breakdown_below_prior_low_bars(**kw) -> PriceData:
    """Today's Low (96) >= yesterday's Low (95) -- DOWN-direction trigger does not fire."""
    return _two_bar_frame(100, 101, 95, 96, 96, 98, 96, 97, **kw)


# ── AlignedBars convenience ───────────────────────────────────────────────────

def aligned(trend: PriceData, entry: PriceData) -> AlignedBars:
    return AlignedBars(trend=trend, entry=entry)


# ── test doubles: fake screens (engine truth-table tests) ───────────────────
# These return a FIXED verdict regardless of input, so
# test_triple_screen_engine.py can exercise SignalEngine's composition logic
# for all 24 (trend, pullback, trigger, position) cells deterministically,
# without depending on real indicator math (that's screens.py's own test
# file's job, above).

class FakeTrendScreen(TrendScreen):
    def __init__(self, direction: Direction):
        self._direction = direction

    def evaluate(self, trend_bars: PriceData) -> TrendVerdict:
        return TrendVerdict(direction=self._direction, indicator_values={"fake": 1.0})


class FakeEntryScreen(EntryScreen):
    def __init__(self, pullback_present: bool):
        self._pullback_present = pullback_present

    def evaluate(self, entry_bars: PriceData, trend: Direction) -> EntryVerdict:
        return EntryVerdict(pullback_present=self._pullback_present, indicator_values={"fake": 1.0})


class FakeTriggerScreen(TriggerScreen):
    def __init__(self, triggered: bool):
        self._triggered = triggered

    def evaluate(self, entry_bars: PriceData, trend: Direction) -> TriggerVerdict:
        return TriggerVerdict(triggered=self._triggered, indicator_values={"fake": 1.0})


# ── test double: fake DataProvider (batch tests) ─────────────────────────────

class FakeDataProvider(DataProvider):
    """Returns pre-registered AlignedBars per ticker; raises KeyError for an
    unregistered ticker (never fetches anything -- no network anywhere in
    this test suite)."""

    def __init__(self, bars_by_ticker: dict[str, AlignedBars]):
        self._bars_by_ticker = bars_by_ticker

    def get_bars(self, ticker: str, timeframes: TimeframeConfig = TimeframeConfig()) -> AlignedBars:
        return self._bars_by_ticker[ticker]
