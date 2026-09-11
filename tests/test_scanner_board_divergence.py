"""
tests/test_scanner_board_divergence.py
========================================
Tests for scanner_board.divergence — see scanner_board/PLAN.md Phase 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from scanner_board.divergence import (
    find_pivot_lows,
    find_pivot_highs,
    detect_divergence,
    DivergenceType,
)


def _series(values, start="2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series([float(v) for v in values], index=idx)


def _dip_series(n: int, dips: dict, baseline: float = 100.0) -> pd.Series:
    """A flat `baseline` series with the given {position: value} overrides
    -- used to plant isolated, unambiguous pivots for the divergence tests
    without needing every one of `n` values spelled out by hand."""
    vals = [baseline] * n
    for pos, v in dips.items():
        vals[pos] = v
    return _series(vals)


# ─────────────────────────────────────────────────────────────────────────────
# 1. find_pivot_lows / find_pivot_highs
# ─────────────────────────────────────────────────────────────────────────────

class TestFindPivots:
    """Hypothesis under test: the strict-inequality/interior-only rule is
    loosened somewhere, letting a flat plateau or an edge point count as a
    pivot."""

    def test_ground_truth_hand_traced_alternating_series(self):
        """order=1. Every interior point (index 1..7) alternates strictly
        below/above both its immediate neighbours: 3,2,4,1 are lows at
        indices 1,3,5,7; 5,5,5 are highs at indices 2,4,6. Indices 0 and 8
        are edges -- never evaluated (need `order` bars on both sides)."""
        s = _series([5, 3, 5, 2, 5, 4, 5, 1, 5])
        assert find_pivot_lows(s, order=1) == [1, 3, 5, 7]
        assert find_pivot_highs(s, order=1) == [2, 4, 6]

    def test_flat_plateau_has_no_pivots(self):
        """A completely flat series: no value is *strictly* less/greater
        than its neighbours, so nothing qualifies -- this is the case that
        makes divergence detection correctly report None on a dead market
        instead of an arbitrary pivot choice."""
        s = _series([50.0] * 10)
        assert find_pivot_lows(s, order=2) == []
        assert find_pivot_highs(s, order=2) == []

    def test_boundary_series_shorter_than_window_is_empty_not_a_crash(self):
        s = _series([1.0, 2.0, 3.0])
        assert find_pivot_lows(s, order=5) == []
        assert find_pivot_highs(s, order=5) == []

    def test_boundary_nan_neighbor_excludes_candidate_not_crash(self):
        s = _series([5.0, np.nan, 1.0, np.nan, 5.0])
        assert find_pivot_lows(s, order=1) == []

    @given(
        values=st.lists(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False), min_size=10, max_size=40),
        order=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_every_reported_low_is_a_true_local_minimum(self, values, order):
        """Independent brute-force re-check: every index find_pivot_lows
        returns must actually be strictly below all `order` neighbours on
        each side, re-derived by direct indexing rather than by calling
        the function under test a second time."""
        s = _series(values)
        lows = find_pivot_lows(s, order)
        for i in lows:
            left = values[i - order:i]
            right = values[i + 1:i + 1 + order]
            assert values[i] < min(left)
            assert values[i] < min(right)


# ─────────────────────────────────────────────────────────────────────────────
# 2. detect_divergence
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectDivergence:

    def test_bullish_divergence_detected(self):
        """Price: lower low at bar 30 (85) than bar 10 (90) -- bearish for
        price alone. Indicator: a *higher* (more shallow) low at bar 30
        (35) than bar 10 (20) -- bulls losing steam more slowly than price
        suggests. Gap = 20 bars, right at Lovvorn's own lower bound."""
        price = _dip_series(40, {10: 90.0, 30: 85.0})
        indicator = _dip_series(40, {10: 20.0, 30: 35.0}, baseline=50.0)
        result = detect_divergence(price, indicator, pivot_order=1)
        assert result is DivergenceType.BULLISH

    def test_bearish_divergence_detected(self):
        """Price: higher high at bar 30 (115) than bar 10 (110). Indicator:
        a *lower* top at bar 30 (60) than bar 10 (80) -- bulls weaker than
        the new price high suggests."""
        price = _dip_series(40, {10: 110.0, 30: 115.0})
        indicator = _dip_series(40, {10: 80.0, 30: 60.0}, baseline=50.0)
        result = detect_divergence(price, indicator, pivot_order=1)
        assert result is DivergenceType.BEARISH

    def test_confirming_move_is_not_a_divergence(self):
        """Price makes a lower low AND the indicator also makes a lower
        low -- that CONFIRMS the downtrend (Ch.23's own contrast case),
        it is not a divergence."""
        price = _dip_series(40, {10: 90.0, 30: 85.0})
        indicator = _dip_series(40, {10: 40.0, 30: 30.0}, baseline=50.0)
        assert detect_divergence(price, indicator, pivot_order=1) is DivergenceType.NONE

    def test_gap_too_short_is_not_a_divergence(self):
        """Otherwise-perfect bullish shape, but only 10 bars apart --
        below Lovvorn's 20-bar floor (Ch.23)."""
        price = _dip_series(25, {10: 90.0, 20: 85.0})
        indicator = _dip_series(25, {10: 20.0, 20: 35.0}, baseline=50.0)
        assert detect_divergence(price, indicator, pivot_order=1) is DivergenceType.NONE

    def test_gap_too_long_is_not_a_divergence(self):
        """Otherwise-perfect bullish shape, but 50 bars apart -- above the
        40-bar ceiling (Ch.23)."""
        price = _dip_series(60, {5: 90.0, 55: 85.0})
        indicator = _dip_series(60, {5: 20.0, 55: 35.0}, baseline=50.0)
        assert detect_divergence(price, indicator, pivot_order=1) is DivergenceType.NONE

    def test_fewer_than_two_pivots_is_none_not_a_crash(self):
        price = _dip_series(15, {7: 90.0})
        indicator = _dip_series(15, {7: 20.0}, baseline=50.0)
        assert detect_divergence(price, indicator, pivot_order=1) is DivergenceType.NONE

    def test_centerline_cross_required_but_absent_blocks_the_divergence(self):
        """Ch.23/Ch.30: 'if there is no crossover, there is no
        divergence.' Shape otherwise qualifies as bullish (higher second
        bottom), but the indicator stays negative the whole time between
        the two pivots -- never breaks above the centerline."""
        price = _dip_series(40, {10: 90.0, 30: 85.0})
        indicator = _dip_series(40, {10: -40.0, 30: -10.0}, baseline=-20.0)
        result = detect_divergence(
            price, indicator, pivot_order=1,
            require_centerline_cross=True, centerline=0.0,
        )
        assert result is DivergenceType.NONE

    def test_centerline_cross_required_and_present_allows_the_divergence(self):
        """Same shape as above, but the indicator pokes above zero at bar
        20, between the two pivots -- satisfies the crossover requirement."""
        price = _dip_series(40, {10: 90.0, 30: 85.0})
        indicator = _dip_series(40, {10: -40.0, 20: 5.0, 30: -10.0}, baseline=-20.0)
        result = detect_divergence(
            price, indicator, pivot_order=1,
            require_centerline_cross=True, centerline=0.0,
        )
        assert result is DivergenceType.BULLISH

    def test_centerline_cross_required_bearish_variant(self):
        """Bearish mirror: shape qualifies (lower second top), but the
        indicator must dip below zero between the two pivots too."""
        price = _dip_series(40, {10: 110.0, 30: 115.0})
        indicator = _dip_series(40, {10: 80.0, 30: 60.0}, baseline=20.0)
        blocked = detect_divergence(
            price, indicator, pivot_order=1,
            require_centerline_cross=True, centerline=0.0,
        )
        assert blocked is DivergenceType.NONE

        indicator_with_cross = _dip_series(40, {10: 80.0, 20: -5.0, 30: 60.0}, baseline=20.0)
        allowed = detect_divergence(
            price, indicator_with_cross, pivot_order=1,
            require_centerline_cross=True, centerline=0.0,
        )
        assert allowed is DivergenceType.BEARISH

    def test_nan_indicator_at_a_pivot_is_none_not_a_crash(self):
        """A warm-up NaN in the indicator landing exactly on a price pivot
        (e.g. an EMA/RSI still filling its window) must not raise or be
        silently compared against NaN as if it were a real value."""
        price = _dip_series(40, {10: 90.0, 30: 85.0})
        indicator = _dip_series(40, {30: 35.0}, baseline=50.0)
        indicator.iloc[10] = np.nan
        assert detect_divergence(price, indicator, pivot_order=1) is DivergenceType.NONE
