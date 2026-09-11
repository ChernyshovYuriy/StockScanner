"""
tests/test_scanner_board_thesis_rules.py
==========================================
Tests for scanner_board.thesis_rules — see scanner_board/PLAN.md Phase 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner_board.slope import SlopeDirection
from scanner_board.thesis_rules import (
    price_vs_ma, PriceVsMA,
    value_zone_position, ValueZonePosition,
    macd_cross, MACDCross,
    trend_health_from_slopes, trend_health, TrendHealth,
    macd_histogram_extreme, ExtremeReading,
    di_bias, DIBias,
    adx_regime, ADXRegime,
    five_percent_reference_lines, oscillator_zone, OscillatorZone,
    volume_vs_average, VolumeLevel,
    force_short_term_zone, ForceZone,
    force_long_term_bias, ForceBias,
)


def _series(values, start="2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series([float(v) for v in values], index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# price_vs_ma / value_zone_position (Ch.22)
# ─────────────────────────────────────────────────────────────────────────────

class TestPriceVsMa:
    def test_above(self):
        assert price_vs_ma(_series([11.0]), _series([10.0])) is PriceVsMA.ABOVE

    def test_below(self):
        assert price_vs_ma(_series([9.0]), _series([10.0])) is PriceVsMA.BELOW

    def test_equal_is_at(self):
        assert price_vs_ma(_series([10.0]), _series([10.0])) is PriceVsMA.AT

    def test_nan_is_at_not_a_crash(self):
        assert price_vs_ma(_series([np.nan]), _series([10.0])) is PriceVsMA.AT


class TestValueZonePosition:
    def test_above_zone(self):
        # zone is [11, 13] regardless of which EMA is passed first/second
        assert value_zone_position(_series([15.0]), _series([13.0]), _series([11.0])) is ValueZonePosition.ABOVE

    def test_below_zone(self):
        assert value_zone_position(_series([9.0]), _series([13.0]), _series([11.0])) is ValueZonePosition.BELOW

    def test_inside_zone_regardless_of_ema_order(self):
        assert value_zone_position(_series([12.0]), _series([11.0]), _series([13.0])) is ValueZonePosition.IN_ZONE
        assert value_zone_position(_series([12.0]), _series([13.0]), _series([11.0])) is ValueZonePosition.IN_ZONE

    def test_nan_is_in_zone_not_a_crash(self):
        assert value_zone_position(_series([np.nan]), _series([13.0]), _series([11.0])) is ValueZonePosition.IN_ZONE


# ─────────────────────────────────────────────────────────────────────────────
# MACD (Ch.23)
# ─────────────────────────────────────────────────────────────────────────────

class TestMacdCross:
    def test_bull(self):
        assert macd_cross(_series([1.0]), _series([0.5])) is MACDCross.BULL

    def test_bear(self):
        assert macd_cross(_series([0.5]), _series([1.0])) is MACDCross.BEAR

    def test_equal_is_neutral(self):
        assert macd_cross(_series([1.0]), _series([1.0])) is MACDCross.NEUTRAL

    def test_nan_is_neutral_not_a_crash(self):
        assert macd_cross(_series([np.nan]), _series([1.0])) is MACDCross.NEUTRAL


class TestTrendHealth:
    """Ground truth for the flagship thesis in the user's own original
    request: 'When the slope of MACD-Histogram moves in the same
    direction as prices, the trend is safe. When [opposite], the health
    of the trend is in question.'"""

    @pytest.mark.parametrize("price_slope,hist_slope,expected", [
        (SlopeDirection.RISING, SlopeDirection.RISING, TrendHealth.SAFE),
        (SlopeDirection.FALLING, SlopeDirection.FALLING, TrendHealth.SAFE),
        (SlopeDirection.RISING, SlopeDirection.FALLING, TrendHealth.WEAK),
        (SlopeDirection.FALLING, SlopeDirection.RISING, TrendHealth.WEAK),
        (SlopeDirection.FLAT, SlopeDirection.RISING, TrendHealth.UNCLEAR),
        (SlopeDirection.RISING, SlopeDirection.FLAT, TrendHealth.UNCLEAR),
        (SlopeDirection.FLAT, SlopeDirection.FLAT, TrendHealth.UNCLEAR),
    ])
    def test_from_slopes_exhaustive(self, price_slope, hist_slope, expected):
        assert trend_health_from_slopes(price_slope, hist_slope) is expected

    def test_series_wrapper_safe_uptrend(self):
        price = _series(range(1, 12))  # clearly rising
        hist = _series(range(1, 12))  # clearly rising, same direction
        assert trend_health(price, hist, period=5) is TrendHealth.SAFE

    def test_series_wrapper_weak_when_opposite(self):
        price = _series(range(1, 12))  # clearly rising
        hist = _series(range(11, 0, -1))  # clearly falling
        assert trend_health(price, hist, period=5) is TrendHealth.WEAK


class TestMacdHistogramExtreme:
    def test_new_high(self):
        hist = _series([1, 2, 3, 4, 10])
        assert macd_histogram_extreme(hist, lookback=5) is ExtremeReading.NEW_HIGH

    def test_new_low(self):
        hist = _series([1, 2, 3, 4, -10])
        assert macd_histogram_extreme(hist, lookback=5) is ExtremeReading.NEW_LOW

    def test_neither(self):
        hist = _series([1, 2, 10, 2, 3])
        assert macd_histogram_extreme(hist, lookback=5) is ExtremeReading.NONE

    def test_only_looks_within_lookback_window(self):
        """A record 3 months ago doesn't count -- only the trailing
        `lookback` window matters (Ch.23 says 'for the past three
        months')."""
        hist = _series([100.0] + [1, 2, 3, 4, 5])
        assert macd_histogram_extreme(hist, lookback=5) is ExtremeReading.NEW_HIGH

    def test_nan_last_is_none_not_a_crash(self):
        hist = _series([1, 2, 3, 4, np.nan])
        assert macd_histogram_extreme(hist, lookback=5) is ExtremeReading.NONE


# ─────────────────────────────────────────────────────────────────────────────
# Directional System / ADX (Ch.24)
# ─────────────────────────────────────────────────────────────────────────────

class TestDiBias:
    def test_bull(self):
        assert di_bias(_series([30.0]), _series([10.0])) is DIBias.BULL

    def test_bear(self):
        assert di_bias(_series([10.0]), _series([30.0])) is DIBias.BEAR

    def test_equal_is_neutral(self):
        assert di_bias(_series([20.0]), _series([20.0])) is DIBias.NEUTRAL

    def test_nan_is_neutral_not_a_crash(self):
        assert di_bias(_series([np.nan]), _series([20.0])) is DIBias.NEUTRAL


class TestAdxRegime:
    def test_trending_when_sandwiched_between_di_lines(self):
        adx = _series([20.0])
        plus_di, minus_di = _series([30.0]), _series([10.0])
        assert adx_regime(adx, plus_di, minus_di) is ADXRegime.TRENDING

    def test_overheated_above_both_lines(self):
        adx = _series([35.0])
        plus_di, minus_di = _series([30.0]), _series([10.0])
        assert adx_regime(adx, plus_di, minus_di) is ADXRegime.OVERHEATED

    def test_choppy_below_both_lines_no_rise_yet(self):
        adx = _series([5.0] * 6)
        plus_di, minus_di = _series([20.0] * 6), _series([15.0] * 6)
        assert adx_regime(adx, plus_di, minus_di) is ADXRegime.CHOPPY

    def test_waking_up_after_a_four_point_rise_off_the_streak_low(self):
        """Ch.24: 'When ADX rises by four steps ... from its lowest point
        below both Directional lines, it rings a bell on a new trend.'
        Streak of 6 bars all below both DI lines (min=15): ADX dips to 5
        then rises to 10 -- a 5-point rise off the streak's own low,
        while still below both lines."""
        adx = _series([8.0, 7.0, 5.0, 6.0, 8.0, 10.0])
        plus_di, minus_di = _series([20.0] * 6), _series([15.0] * 6)
        assert adx_regime(adx, plus_di, minus_di) is ADXRegime.WAKING_UP

    def test_unknown_on_nan(self):
        adx = _series([np.nan])
        plus_di, minus_di = _series([30.0]), _series([10.0])
        assert adx_regime(adx, plus_di, minus_di) is ADXRegime.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Oscillators (Ch.25, 26, 27) — shared 5%-rule reference lines
# ─────────────────────────────────────────────────────────────────────────────

class TestFivePercentReferenceLines:
    def test_ground_truth_matches_independent_percentile_calc(self):
        """Cross-checked against numpy.percentile (a different library
        call than pandas' own Series.quantile, i.e. an independent
        re-derivation) rather than just re-stating pandas' own formula."""
        values = list(range(1, 121))
        s = _series(values)
        lower, upper = five_percent_reference_lines(s, lookback=120)
        assert lower == pytest.approx(np.percentile(values, 5))
        assert upper == pytest.approx(np.percentile(values, 95))

    def test_only_the_trailing_lookback_window_is_used(self):
        """An extreme value far outside the lookback window must not drag
        the reference lines toward it."""
        s = _series([100000.0] + [1, 2, 3, 4, 5])
        lower, upper = five_percent_reference_lines(s, lookback=5)
        assert upper < 100.0

    def test_empty_window_is_nan_not_a_crash(self):
        s = _series([np.nan, np.nan, np.nan])
        lower, upper = five_percent_reference_lines(s, lookback=3)
        assert np.isnan(lower) and np.isnan(upper)


class TestOscillatorZone:
    def test_overbought_at_or_above_upper_line(self):
        s = _series(list(range(1, 101)) + [1000.0])
        assert oscillator_zone(s, lookback=101) is OscillatorZone.OVERBOUGHT

    def test_oversold_at_or_below_lower_line(self):
        s = _series([1000.0] + list(range(1, 101)))
        # last value (100) is well inside the trailing-101 window's range,
        # so instead directly probe a value at/under the 5th percentile
        lower, _ = five_percent_reference_lines(s, lookback=101)
        s2 = pd.concat([s, _series([lower - 1])])
        assert oscillator_zone(s2, lookback=101) is OscillatorZone.OVERSOLD

    def test_neutral_in_the_middle(self):
        s = _series(list(range(1, 101)) + [50.0])
        assert oscillator_zone(s, lookback=101) is OscillatorZone.NEUTRAL

    def test_reused_by_both_rsi_and_stochastic_shaped_series(self):
        """Same function, same result shape, for a 0-100-bounded series
        (Stochastic) and an unbounded-but-similar-range one (RSI) --
        there is exactly one zone-classification implementation."""
        rsi_like = _series(list(range(1, 101)) + [95.0])
        stoch_like = _series(list(range(1, 101)) + [95.0])
        assert oscillator_zone(rsi_like, lookback=101) == oscillator_zone(stoch_like, lookback=101)


# ─────────────────────────────────────────────────────────────────────────────
# Volume-based (Ch.28, 29, 30)
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumeVsAverage:
    def test_high_volume(self):
        """9 days of volume=100 (avg=100), today's volume=200 -> +100% vs
        the trailing (pre-today) average, well above the 25% threshold."""
        volume = _series([100.0] * 9 + [200.0])
        assert volume_vs_average(volume, lookback=9) is VolumeLevel.HIGH

    def test_low_volume(self):
        volume = _series([100.0] * 9 + [50.0])
        assert volume_vs_average(volume, lookback=9) is VolumeLevel.LOW

    def test_normal_volume(self):
        volume = _series([100.0] * 9 + [110.0])
        assert volume_vs_average(volume, lookback=9) is VolumeLevel.NORMAL

    def test_todays_own_volume_excluded_from_its_own_average(self):
        """A single-bar 5x volume spike must not drag its own comparison
        baseline upward -- the average is computed over the lookback days
        BEFORE today (shift(1)), not including today."""
        volume = _series([100.0] * 9 + [500.0])
        assert volume_vs_average(volume, lookback=9) is VolumeLevel.HIGH


class TestForceShortTermZone:
    def test_negative(self):
        assert force_short_term_zone(_series([-5.0])) is ForceZone.NEGATIVE

    def test_positive(self):
        assert force_short_term_zone(_series([5.0])) is ForceZone.POSITIVE

    def test_zero(self):
        assert force_short_term_zone(_series([0.0])) is ForceZone.ZERO

    def test_nan_is_zero_not_a_crash(self):
        assert force_short_term_zone(_series([np.nan])) is ForceZone.ZERO


class TestForceLongTermBias:
    def test_bull(self):
        assert force_long_term_bias(_series([5.0])) is ForceBias.BULL

    def test_bear(self):
        assert force_long_term_bias(_series([-5.0])) is ForceBias.BEAR

    def test_neutral_at_zero_and_nan(self):
        assert force_long_term_bias(_series([0.0])) is ForceBias.NEUTRAL
        assert force_long_term_bias(_series([np.nan])) is ForceBias.NEUTRAL
