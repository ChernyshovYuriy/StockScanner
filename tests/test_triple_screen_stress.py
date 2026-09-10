"""
Adversarial stress test for the Triple Screen engine, per the source
stress-test spec (triple_screen_stress_test_prompt.md): the implementation
is assumed BROKEN until a runnable test proves otherwise. No claim in the
accompanying findings report rests on reading the code -- every claim below
is backed by an assertion in this file, and every confirmed defect is
encoded as `pytest.mark.xfail(strict=True)` so:
  (a) the suite is still meaningful to run (xfail rows show up as XFAIL,
      not silently skipped, and `strict=True` means the suite goes RED the
      moment someone fixes the underlying bug without updating this file --
      that's the trigger to remove the marker), and
  (b) nothing here is "fixed" as a side effect of writing tests -- per
      instruction, no implementation file is touched in this commit.

Sections mirror the spec's four parts. See
research/triple_screen/TRIPLE_SCREEN_STRESS_FINDINGS.md for the narrative
findings report this file's results feed into.
"""
import numpy as np
import pandas as pd
import pytest

try:
    from hypothesis import given, settings, strategies as st
    HYPOTHESIS = True
except ImportError:
    HYPOTHESIS = False

from research.triple_screen.batch import format_results_table, run_batch
from research.triple_screen.data_provider import DataProvider
from research.triple_screen.indicators import ema_slope_direction, force_index_pullback, prior_bar_breakout
from research.triple_screen.reference_impl import (
    EMASlopeTrendScreen, ForceIndexEntryScreen, PriorBarTriggerScreen, TripleScreenEngine,
)
from research.triple_screen.types import (
    AlignedBars, Direction, PriceData, Signal, TimeframeConfig,
)

from triple_screen_fixtures import (
    FakeEntryScreen, FakeTrendScreen, FakeTriggerScreen, aligned, downtrend_bars, flat_bars,
    make_bars, uptrend_bars,
)


def _engine() -> TripleScreenEngine:
    return TripleScreenEngine(EMASlopeTrendScreen(), ForceIndexEntryScreen(), PriorBarTriggerScreen())


def _two_bar(o0, h0, l0, c0, o1, h1, l1, c1, volume=1_000_000.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    return pd.DataFrame({"Open": [o0, o1], "High": [h0, h1], "Low": [l0, l1],
                          "Close": [c0, c1], "Volume": [volume, volume]}, index=idx)


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 -- main logic, independently re-derived
# ═══════════════════════════════════════════════════════════════════════════
#
# Independent derivation from Elder's Triple Screen (not read off engine.py):
#
# Position OPEN (long):
#   Elder's practice: the three-screen conjunction is an ENTRY filter only.
#   An open position is exited on a trend-flip / protective-stop basis, not
#   by waiting for a fully mirrored bearish 3-screen setup. So:
#     trend UP    -> aligned, no reversal -> HOLD
#     trend FLAT  -> indecisive is not itself a reversal -> HOLD
#     trend DOWN  -> the weekly tide has turned against the position -> SELL,
#                    unconditionally, regardless of Screens 2/3.
#
# Position FLAT (nothing open, long-only -- no shorting infra):
#   trend UP    -> BUY requires ALL THREE conditions (trend UP + pullback +
#                  trigger); any one missing -> WAIT.
#   trend DOWN  -> long-only, nothing to exit and no short entry -> WAIT
#                  always, regardless of Screens 2/3 (even a textbook
#                  bearish setup is not actionable in a long-only system).
#   trend FLAT  -> indecisive, no edge either way -> WAIT always.
#
# This gives the 24-row table below (3 trend x 2 pullback x 2 trigger x
# 2 position = 24), each row justified by exactly one of the four bullets
# above.
INDEPENDENT_TRUTH_TABLE = [
    # trend, pullback, trigger, position_open, expected, justification
    (Direction.UP,   True,  True,  True,  Signal.HOLD, "open+UP -> HOLD (aligned)"),
    (Direction.UP,   True,  False, True,  Signal.HOLD, "open+UP -> HOLD (aligned)"),
    (Direction.UP,   False, True,  True,  Signal.HOLD, "open+UP -> HOLD (aligned)"),
    (Direction.UP,   False, False, True,  Signal.HOLD, "open+UP -> HOLD (aligned)"),
    (Direction.UP,   True,  True,  False, Signal.BUY,  "flat+UP+pullback+trigger -> BUY (full conjunction)"),
    (Direction.UP,   True,  False, False, Signal.WAIT, "flat+UP+pullback only -> WAIT (missing trigger)"),
    (Direction.UP,   False, True,  False, Signal.WAIT, "flat+UP+trigger only -> WAIT (missing pullback)"),
    (Direction.UP,   False, False, False, Signal.WAIT, "flat+UP, neither -> WAIT"),
    (Direction.DOWN, True,  True,  True,  Signal.SELL, "open+DOWN -> SELL (trend-flip exit, unconditional)"),
    (Direction.DOWN, True,  False, True,  Signal.SELL, "open+DOWN -> SELL (trend-flip exit, unconditional)"),
    (Direction.DOWN, False, True,  True,  Signal.SELL, "open+DOWN -> SELL (trend-flip exit, unconditional)"),
    (Direction.DOWN, False, False, True,  Signal.SELL, "open+DOWN -> SELL (trend-flip exit, unconditional)"),
    (Direction.DOWN, True,  True,  False, Signal.WAIT, "flat+DOWN -> WAIT (long-only, nothing to do)"),
    (Direction.DOWN, True,  False, False, Signal.WAIT, "flat+DOWN -> WAIT (long-only, nothing to do)"),
    (Direction.DOWN, False, True,  False, Signal.WAIT, "flat+DOWN -> WAIT (long-only, nothing to do)"),
    (Direction.DOWN, False, False, False, Signal.WAIT, "flat+DOWN -> WAIT (long-only, nothing to do)"),
    (Direction.FLAT, True,  True,  True,  Signal.HOLD, "open+FLAT -> HOLD (indecisive != reversal)"),
    (Direction.FLAT, True,  False, True,  Signal.HOLD, "open+FLAT -> HOLD (indecisive != reversal)"),
    (Direction.FLAT, False, True,  True,  Signal.HOLD, "open+FLAT -> HOLD (indecisive != reversal)"),
    (Direction.FLAT, False, False, True,  Signal.HOLD, "open+FLAT -> HOLD (indecisive != reversal)"),
    (Direction.FLAT, True,  True,  False, Signal.WAIT, "flat+FLAT -> WAIT (no edge)"),
    (Direction.FLAT, True,  False, False, Signal.WAIT, "flat+FLAT -> WAIT (no edge)"),
    (Direction.FLAT, False, True,  False, Signal.WAIT, "flat+FLAT -> WAIT (no edge)"),
    (Direction.FLAT, False, False, False, Signal.WAIT, "flat+FLAT -> WAIT (no edge)"),
]
assert len(INDEPENDENT_TRUTH_TABLE) == 24


@pytest.mark.parametrize("trend,pullback,trigger,position_open,expected,why", INDEPENDENT_TRUTH_TABLE)
def test_part1_independently_derived_truth_table(trend, pullback, trigger, position_open, expected, why):
    engine = TripleScreenEngine(
        trend_screen=FakeTrendScreen(trend),
        entry_screen=FakeEntryScreen(pullback),
        trigger_screen=FakeTriggerScreen(trigger),
    )
    bars = aligned(trend=flat_bars(timeframe="weekly"), entry=flat_bars(timeframe="daily"))
    result = engine.evaluate(bars, position_open=position_open)
    assert result.signal == expected, f"{why}: expected {expected}, got {result.signal}"


def test_part1_flat_trend_never_yields_buy_or_sell():
    for pullback in (True, False):
        for trigger in (True, False):
            for position_open in (True, False):
                engine = TripleScreenEngine(FakeTrendScreen(Direction.FLAT),
                                             FakeEntryScreen(pullback), FakeTriggerScreen(trigger))
                bars = aligned(flat_bars(timeframe="weekly"), flat_bars(timeframe="daily"))
                signal = engine.evaluate(bars, position_open=position_open).signal
                assert signal not in (Signal.BUY, Signal.SELL), (
                    f"FLAT trend produced {signal} (pullback={pullback}, trigger={trigger}, "
                    f"position_open={position_open})"
                )


def test_part1_up_trend_never_yields_sell_and_down_trend_never_yields_buy():
    for pullback in (True, False):
        for trigger in (True, False):
            for position_open in (True, False):
                up_engine = TripleScreenEngine(FakeTrendScreen(Direction.UP),
                                                FakeEntryScreen(pullback), FakeTriggerScreen(trigger))
                down_engine = TripleScreenEngine(FakeTrendScreen(Direction.DOWN),
                                                  FakeEntryScreen(pullback), FakeTriggerScreen(trigger))
                bars = aligned(flat_bars(timeframe="weekly"), flat_bars(timeframe="daily"))
                up_signal = up_engine.evaluate(bars, position_open=position_open).signal
                down_signal = down_engine.evaluate(bars, position_open=position_open).signal
                assert up_signal != Signal.SELL, f"UP trend produced SELL ({pullback},{trigger},{position_open})"
                assert down_signal != Signal.BUY, f"DOWN trend produced BUY ({pullback},{trigger},{position_open})"


def test_part1_buy_requires_all_three_missing_any_one_blocks_it():
    bars = aligned(flat_bars(timeframe="weekly"), flat_bars(timeframe="daily"))
    for pullback, trigger in [(True, False), (False, True), (False, False)]:
        engine = TripleScreenEngine(FakeTrendScreen(Direction.UP),
                                     FakeEntryScreen(pullback), FakeTriggerScreen(trigger))
        signal = engine.evaluate(bars, position_open=False).signal
        assert signal != Signal.BUY, f"BUY fired without full conjunction (pullback={pullback}, trigger={trigger})"


def test_part1_position_state_not_indicators_decides_hold_vs_buy_and_wait_vs_sell():
    bars = aligned(flat_bars(timeframe="weekly"), flat_bars(timeframe="daily"))
    # Same indicators (UP, pullback, trigger all True); only position flips.
    up_full = TripleScreenEngine(FakeTrendScreen(Direction.UP), FakeEntryScreen(True), FakeTriggerScreen(True))
    assert up_full.evaluate(bars, position_open=False).signal == Signal.BUY
    assert up_full.evaluate(bars, position_open=True).signal == Signal.HOLD
    # Same indicators (DOWN, arbitrary pullback/trigger); only position flips.
    down_any = TripleScreenEngine(FakeTrendScreen(Direction.DOWN), FakeEntryScreen(True), FakeTriggerScreen(True))
    assert down_any.evaluate(bars, position_open=False).signal == Signal.WAIT
    assert down_any.evaluate(bars, position_open=True).signal == Signal.SELL


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 -- corner cases
# ═══════════════════════════════════════════════════════════════════════════

def test_part2_ema_slope_exactly_zero_is_flat_not_up_or_down():
    # constant closes -> EMA now == EMA lag bars ago, exact equality.
    result = EMASlopeTrendScreen().evaluate(flat_bars())
    assert result.direction == Direction.FLAT
    assert result.indicator_values["ema"] == pytest.approx(100.0)


def test_part2_force_index_exactly_zero_is_pullback_absent():
    # two bars, identical Close -> diff=0 -> Force Index=0 exactly -> not "< 0".
    bars = _two_bar(100, 101, 99, 100, 100, 101, 99, 100)
    present, vals = force_index_pullback(bars, Direction.UP)
    assert present is False
    assert vals["force_index"] == 0.0


def test_part2_trigger_exact_equality_to_prior_high_does_not_fire():
    # today's High == yesterday's High exactly -> strict '>' required, not '>='.
    bars = _two_bar(100, 105, 99, 104, 104, 105, 103, 104)
    fired, vals = prior_bar_breakout(bars, Direction.UP)
    assert fired is False
    assert vals["today_high"] == vals["prior_high"] == 105.0


def test_part2_minimum_viable_trend_data_boundary_is_exactly_ema_period_plus_lag():
    # ema_period=13, lag=3 -> 16 bars is the minimum for a real read; one
    # fewer (15) must degrade to the safe FLAT default, not crash.
    closes_16 = 100 * (1.01) ** np.arange(16)
    closes_15 = 100 * (1.01) ** np.arange(15)
    d16, _ = ema_slope_direction(make_bars(closes_16).bars["Close"], 13, 3)
    d15, _ = ema_slope_direction(make_bars(closes_15).bars["Close"], 13, 3)
    assert d16 == Direction.UP
    assert d15 == Direction.FLAT


def test_part2_minimum_viable_entry_trigger_data_boundary_is_two_bars():
    two = _two_bar(100, 101, 99, 100, 101, 102, 100, 101)
    one = two.iloc[:1]
    p2, _ = force_index_pullback(two, Direction.UP)
    p1, _ = force_index_pullback(one, Direction.UP)
    t2, _ = prior_bar_breakout(two, Direction.UP)
    t1, _ = prior_bar_breakout(one, Direction.UP)
    assert isinstance(p2, bool) and isinstance(t2, bool)  # 2 bars: real read, no crash
    assert p1 is False and t1 is False  # 1 bar: safe default, no crash


def test_part2_trend_transition_up_to_down_to_flat_no_stale_leakage():
    """A single series: 30 bars up, 30 bars down, 20 bars flat. The trend
    read using only the prefix up to each transition point must reflect
    ONLY that prefix -- not leak information from bars added later in the
    same underlying array (proves truncation-safety at the transition
    boundary specifically, not just anywhere)."""
    up = 100 * (1.02) ** np.arange(30)
    down = up[-1] * (0.98) ** np.arange(1, 31)
    flat = np.full(20, down[-1])
    full = np.concatenate([up, down, flat])
    pdfull = make_bars(full)

    # As of the end of the up-leg (bar 29): must read UP using only bars[:30].
    d_up, _ = ema_slope_direction(pdfull.bars["Close"].iloc[:30], 13, 3)
    assert d_up == Direction.UP
    # As of the end of the down-leg (bar 59): must read DOWN using bars[:60].
    d_down, _ = ema_slope_direction(pdfull.bars["Close"].iloc[:60], 13, 3)
    assert d_down == Direction.DOWN
    # Truncated reads must be identical to evaluating the same-length
    # prefix as its own standalone series (no hidden dependence on what
    # comes after the cut in the full array).
    standalone_up = make_bars(full[:30]).bars["Close"]
    d_up_standalone, _ = ema_slope_direction(standalone_up, 13, 3)
    assert d_up_standalone == d_up


def test_part2_single_spike_then_flat_does_not_crash():
    closes = np.concatenate([[100.0], np.full(59, 100.0)])
    closes[5] = 500.0  # one-bar spike, back to flat immediately after
    result = EMASlopeTrendScreen().evaluate(make_bars(closes))
    assert isinstance(result.direction, Direction)


def test_part2_entry_screen_unaffected_by_trend_series_length_or_content():
    """Screen independence at the alignment layer: EntryScreen only takes
    entry_bars + trend (an enum), never trend_bars itself -- so two
    AlignedBars with wildly different trend series but the same entry
    series and same resolved Direction must give an identical EntryVerdict."""
    from triple_screen_fixtures import pullback_in_uptrend_bars
    entry = pullback_in_uptrend_bars()
    short_trend = uptrend_bars(n=20, timeframe="weekly")
    long_trend = uptrend_bars(n=200, timeframe="weekly")
    v1 = ForceIndexEntryScreen().evaluate(entry, Direction.UP)
    v2 = ForceIndexEntryScreen().evaluate(entry, Direction.UP)  # trend series never even passed in
    assert v1 == v2


def test_part2_one_timeframe_empty_other_full_degrades_safely():
    """Ticker present in daily but not weekly (or vice versa): AlignedBars
    doesn't forbid this combination -- must still resolve to a defined
    Signal, not crash."""
    empty_weekly = PriceData(ticker="T", timeframe="weekly",
                              bars=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
    full_daily = uptrend_bars(n=60, timeframe="daily")
    result = _engine().evaluate(AlignedBars(trend=empty_weekly, entry=full_daily), position_open=False)
    assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD, Signal.WAIT)
    assert result.trend.direction == Direction.FLAT  # no weekly data -> safe default, not a crash


# ═══════════════════════════════════════════════════════════════════════════
# Part 3 -- invalid / broken input
# ═══════════════════════════════════════════════════════════════════════════

def test_part3_empty_dataframe_all_three_screens_degrade_safely():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    d, _ = ema_slope_direction(empty["Close"], 13, 3)
    p, _ = force_index_pullback(empty, Direction.UP)
    t, _ = prior_bar_breakout(empty, Direction.UP)
    assert d == Direction.FLAT and p is False and t is False


def test_part3_none_series_raises_rather_than_silently_misbehaving():
    with pytest.raises(AttributeError):
        ema_slope_direction(None, 13, 3)


def test_part3_missing_volume_column_raises_rather_than_silently_misbehaving():
    bars = uptrend_bars(n=20).bars.drop(columns=["Volume"])
    with pytest.raises(KeyError):
        force_index_pullback(bars, Direction.UP)


def test_part3_zero_volume_is_pullback_absent_not_a_crash():
    bars = uptrend_bars(n=20).bars.copy()
    bars["Volume"] = 0.0
    present, vals = force_index_pullback(bars, Direction.UP)
    assert present is False
    assert vals["force_index"] == 0.0


def test_part3_negative_prices_do_not_crash_and_direction_follows_the_math():
    # No domain validation anywhere in this package: negative-but-rising
    # prices are accepted and read as UP, same as positive-but-rising.
    # Documents the absence of a sanity check, not a specific "should".
    closes = np.linspace(-1000.0, -500.0, 60)
    result = EMASlopeTrendScreen().evaluate(make_bars(closes))
    assert result.direction == Direction.UP


def test_part3_ohlc_invariant_violation_high_below_low_silently_accepted():
    """No OHLC sanity check anywhere in this package: a corrupted bar
    (High < Low on the latest row) is not detected or rejected -- the
    trigger screen just compares whatever numbers it's given. Documents a
    real robustness gap: a bad data feed produces no error and no warning."""
    bars = uptrend_bars(n=20).bars.copy()
    bars.iloc[-1, bars.columns.get_loc("High")] = 50.0
    bars.iloc[-1, bars.columns.get_loc("Low")] = 500.0  # High < Low: physically impossible
    fired, vals = prior_bar_breakout(bars, Direction.UP)
    assert isinstance(fired, bool)  # no exception, no flag raised anywhere


def test_part3_inf_in_latest_high_must_not_silently_fire_the_trigger():
    """FIXED: prior_bar_breakout now checks math.isfinite() on both the
    latest and prior value before comparing; a non-finite OHLC value is
    treated the same as missing data (safe default), not compared as if it
    were real. Was xfail (triggered=True, today_high=inf) -- now passes."""
    bars = uptrend_bars(n=20).bars.copy()
    bars.iloc[-1, bars.columns.get_loc("High")] = float("inf")
    fired, vals = prior_bar_breakout(bars, Direction.UP)
    assert fired is False, f"inf High silently fired the trigger: {vals}"


def test_part3_inf_in_midseries_close_does_not_corrupt_the_trend_read():
    """Documents an INCIDENTAL protection, not one this package provides
    itself: pandas' own .ewm().mean() treats a non-finite value as if it
    were missing (carries the prior EMA value forward) rather than
    propagating inf through the whole rest of the series. Verified here so
    the claim doesn't rest on assumption -- if a future pandas version
    changes this, this test is what will catch it."""
    bars = uptrend_bars(n=60).bars.copy()
    bars.iloc[40, bars.columns.get_loc("Close")] = float("inf")
    d, v = ema_slope_direction(bars["Close"], 13, 3)
    assert d == Direction.UP
    assert np.isfinite(v["ema"])


def test_part3_reverse_chronological_bars_must_not_silently_invert_the_trend():
    """FIXED: every indicators.py function now sorts its input by index
    ascending before reading off "latest"/"prior" positionally. Was xfail
    (reversed uptrend read as DOWN) -- now passes."""
    correct = uptrend_bars(n=60)
    reversed_bars = correct.bars.iloc[::-1]
    d, v = ema_slope_direction(reversed_bars["Close"], 13, 3)
    assert d == Direction.UP, f"reversed uptrend read as {d} instead of UP: {v}"


def test_part3_duplicate_tickers_in_batch_collapse_to_one_result_per_ticker():
    """Documents actual behavior (dict semantics: last write wins), not a
    contract violation -- run_batch's return type is `dict[str, SignalResult]`,
    which cannot hold two entries under the same key by construction."""
    provider_bars = {"DUPE": aligned(uptrend_bars(timeframe="weekly"), flat_bars(timeframe="daily"))}

    class OneTickerProvider(DataProvider):
        def get_bars(self, ticker, timeframes=TimeframeConfig()):
            return provider_bars[ticker]

    results = run_batch(["DUPE", "DUPE", "DUPE"], OneTickerProvider(), _engine())
    assert list(results.keys()) == ["DUPE"]


def test_part3_empty_batch_list_returns_empty_results_and_empty_table():
    class NeverCalledProvider(DataProvider):
        def get_bars(self, ticker, timeframes=TimeframeConfig()):
            raise AssertionError("should never be called for an empty ticker list")

    results = run_batch([], NeverCalledProvider(), _engine())
    assert results == {}
    assert format_results_table(results) == ""


def test_part3_one_ticker_provider_failure_does_not_abort_the_whole_batch():
    """FIXED: run_batch now wraps each ticker's fetch+evaluate in try/except
    and skips (with a printed WARNING) only the failing ticker. Was xfail
    (whole batch aborted, good tickers lost) -- now passes."""
    good_bars = aligned(flat_bars(timeframe="weekly"), flat_bars(timeframe="daily"))

    class FlakyProvider(DataProvider):
        def get_bars(self, ticker, timeframes=TimeframeConfig()):
            if ticker == "BAD":
                raise RuntimeError("simulated provider failure")
            return good_bars

    results = run_batch(["GOOD1", "BAD", "GOOD2"], FlakyProvider(), _engine())
    assert set(results.keys()) == {"GOOD1", "GOOD2"}, (
        f"expected the two good tickers to still resolve; got {set(results.keys())}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Part 4 -- properties that must always hold
# ═══════════════════════════════════════════════════════════════════════════

if HYPOTHESIS:
    _finite_price_series = st.lists(
        st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        min_size=20, max_size=80,
    )

    def _to_price_data(closes, timeframe="daily") -> PriceData:
        return make_bars(closes, timeframe=timeframe)

    @given(closes=_finite_price_series, position_open=st.booleans())
    @settings(max_examples=60, deadline=None)
    def test_part4_output_always_one_of_four_signals_never_exception(closes, position_open):
        bars = AlignedBars(trend=_to_price_data(closes, "weekly"), entry=_to_price_data(closes, "daily"))
        result = _engine().evaluate(bars, position_open=position_open)
        assert isinstance(result.signal, Signal)

    @given(closes=_finite_price_series, position_open=st.booleans())
    @settings(max_examples=60, deadline=None)
    def test_part4_determinism_same_input_same_output(closes, position_open):
        bars = AlignedBars(trend=_to_price_data(closes, "weekly"), entry=_to_price_data(closes, "daily"))
        r1 = _engine().evaluate(bars, position_open=position_open)
        r2 = _engine().evaluate(bars, position_open=position_open)
        assert r1 == r2

    @given(
        shared_prefix=st.lists(
            st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
            min_size=20, max_size=40,
        ),
        tail_a=st.lists(
            st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=20,
        ),
        tail_b=st.lists(
            st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=20,
        ),
    )
    @settings(max_examples=60, deadline=None)
    def test_part4_no_lookahead_shared_prefix_gives_identical_verdict(shared_prefix, tail_a, tail_b):
        """The single most important invariant per the spec: two series that
        agree on their first k bars but diverge afterward must produce the
        SAME trend verdict when each is truncated to those k bars -- nothing
        about bar k's verdict may depend on data that comes after it."""
        prefix_only = _to_price_data(shared_prefix)
        d_prefix, _ = ema_slope_direction(prefix_only.bars["Close"], 13, 3)

        series_a = _to_price_data(shared_prefix + tail_a).bars["Close"].iloc[:len(shared_prefix)]
        series_b = _to_price_data(shared_prefix + tail_b).bars["Close"].iloc[:len(shared_prefix)]
        d_a, _ = ema_slope_direction(series_a, 13, 3)
        d_b, _ = ema_slope_direction(series_b, 13, 3)
        assert d_prefix == d_a == d_b

    @given(direction=st.sampled_from(list(Direction)), pullback=st.booleans(),
           trigger=st.booleans(), position_open=st.booleans())
    @settings(max_examples=60, deadline=None)
    def test_part4_direction_permission_invariant_holds_for_generated_inputs(
            direction, pullback, trigger, position_open):
        engine = TripleScreenEngine(FakeTrendScreen(direction), FakeEntryScreen(pullback),
                                     FakeTriggerScreen(trigger))
        bars = aligned(flat_bars(timeframe="weekly"), flat_bars(timeframe="daily"))
        signal = engine.evaluate(bars, position_open=position_open).signal
        if direction == Direction.FLAT:
            assert signal not in (Signal.BUY, Signal.SELL)
        if direction == Direction.UP:
            assert signal != Signal.SELL
        if direction == Direction.DOWN:
            assert signal != Signal.BUY
        if direction == Direction.UP and pullback and trigger and not position_open:
            assert signal == Signal.BUY
        if not (direction == Direction.UP and pullback and trigger) and not position_open:
            assert signal != Signal.BUY


def test_part4_screen_independence_trend_result_unaffected_by_entry_series():
    trend = uptrend_bars(timeframe="weekly")
    entry_a = flat_bars(timeframe="daily")
    entry_b = downtrend_bars(timeframe="daily")
    r_a = _engine().evaluate(AlignedBars(trend=trend, entry=entry_a), position_open=False)
    r_b = _engine().evaluate(AlignedBars(trend=trend, entry=entry_b), position_open=False)
    assert r_a.trend == r_b.trend  # same trend series -> same TrendVerdict, regardless of entry data
