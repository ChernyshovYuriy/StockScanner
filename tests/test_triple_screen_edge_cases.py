"""
Edge-case tests: insufficient bars, NaN bars, and weekly/daily misalignment
must all produce a *defined* result (never an exception), per the source
build spec's Phase 2 edge-case requirements. Runs against the real
reference_impl screens/engine (not fakes) -- these tests are specifically
about how real indicator math degrades gracefully, not about composition
logic (that's test_triple_screen_engine.py's job).
"""
from research.triple_screen.reference_impl import (
    EMASlopeTrendScreen, ForceIndexEntryScreen, PriorBarTriggerScreen, TripleScreenEngine,
)
from research.triple_screen.types import AlignedBars, Direction, Signal

from triple_screen_fixtures import (
    bars_with_nan, downtrend_bars, flat_bars, insufficient_bars, make_bars, uptrend_bars,
)


# ── insufficient bars -> safe defaults, never an exception ──────────────────

def test_trend_screen_insufficient_bars_returns_flat_not_exception():
    result = EMASlopeTrendScreen().evaluate(insufficient_bars(n=1))
    assert result.direction == Direction.FLAT


def test_entry_screen_insufficient_bars_returns_no_pullback_not_exception():
    result = ForceIndexEntryScreen().evaluate(insufficient_bars(n=1), Direction.UP)
    assert result.pullback_present is False


def test_trigger_screen_insufficient_bars_returns_not_triggered_not_exception():
    result = PriorBarTriggerScreen().evaluate(insufficient_bars(n=1), Direction.UP)
    assert result.triggered is False


def test_engine_end_to_end_insufficient_bars_returns_wait():
    """Too-short bars all the way through the real engine (flat position) --
    FLAT trend -> WAIT per the truth table, without any special-case code
    needed in the engine itself: insufficient data already flows through as
    a FLAT trend verdict."""
    engine = TripleScreenEngine(
        trend_screen=EMASlopeTrendScreen(),
        entry_screen=ForceIndexEntryScreen(),
        trigger_screen=PriorBarTriggerScreen(),
    )
    bars = AlignedBars(trend=insufficient_bars(n=1, timeframe="weekly"),
                        entry=insufficient_bars(n=1, timeframe="daily"))
    result = engine.evaluate(bars, position_open=False)
    assert result.signal == Signal.WAIT


# ── NaN bars -> defined result, never an exception ───────────────────────────

def test_trend_screen_nan_bar_does_not_raise():
    result = EMASlopeTrendScreen().evaluate(bars_with_nan())
    assert isinstance(result.direction, Direction)


def test_entry_screen_nan_bar_does_not_raise():
    bars = bars_with_nan()
    result = ForceIndexEntryScreen().evaluate(bars, Direction.UP)
    assert isinstance(result.pullback_present, bool)


def test_trigger_screen_nan_bar_does_not_raise():
    bars = bars_with_nan()
    result = PriorBarTriggerScreen().evaluate(bars, Direction.UP)
    assert isinstance(result.triggered, bool)


# ── weekly/daily misalignment -> engine handles differing ranges/lengths ────

def test_engine_handles_weekly_daily_date_range_misalignment():
    """Trend (weekly) bars end weeks before entry (daily) bars' last date --
    realistic (weekly lags behind daily) and structurally normal (weekly has
    far fewer rows than daily for the same calendar span). AlignedBars does
    NOT guarantee row-for-row date alignment between the two series (see its
    own docstring) -- each screen only ever looks at its own series."""
    trend_bars = uptrend_bars(n=60, timeframe="weekly", start="2020-01-01", freq="W")
    entry_bars = uptrend_bars(n=200, timeframe="daily", start="2023-06-01", freq="D")
    engine = TripleScreenEngine(
        trend_screen=EMASlopeTrendScreen(),
        entry_screen=ForceIndexEntryScreen(),
        trigger_screen=PriorBarTriggerScreen(),
    )
    result = engine.evaluate(AlignedBars(trend=trend_bars, entry=entry_bars), position_open=False)
    assert result.signal in (Signal.BUY, Signal.WAIT, Signal.HOLD, Signal.SELL)
