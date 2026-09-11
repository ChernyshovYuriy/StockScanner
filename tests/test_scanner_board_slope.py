"""
tests/test_scanner_board_slope.py
===================================
Tests for scanner_board.slope.classify_slope — see scanner_board/PLAN.md
Phase 1.

Hypothesis under test: classify_slope silently re-derives its own slope
calculation instead of delegating to
canadian_stock_screener.TechnicalIndicators.linear_regression_slope (the
single-shared-logic rule scanner_board/PLAN.md commits to), or gets the
threshold comparison backwards/off-by-one at the boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from canadian_stock_screener import TechnicalIndicators as TI
from scanner_board.slope import classify_slope, SlopeDirection, FLAT_THRESHOLD


def _series(values, start="2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series([float(v) for v in values], index=idx)


class TestClassifySlope:

    def test_ground_truth_clearly_rising_series(self):
        s = _series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert classify_slope(s, period=5) is SlopeDirection.RISING

    def test_ground_truth_clearly_falling_series(self):
        s = _series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        assert classify_slope(s, period=5) is SlopeDirection.FALLING

    def test_ground_truth_flat_series(self):
        s = _series([50.0] * 10)
        assert classify_slope(s, period=5) is SlopeDirection.FLAT

    def test_boundary_insufficient_data_is_flat_not_a_crash(self):
        """linear_regression_slope returns exactly 0.0 (not NaN, not a
        raised error) when fewer than `period` points are available --
        classify_slope must read that as Flat, the safe/neutral label,
        rather than propagating a crash or a misleading Rising/Falling."""
        s = _series([1.0, 2.0])
        assert classify_slope(s, period=10) is SlopeDirection.FLAT

    def test_boundary_exactly_at_threshold_is_flat_not_rising(self):
        """The comparison is strict (> / <), so a slope that lands exactly
        on the threshold is Flat, not Rising -- this locks that choice so
        a future refactor doesn't silently flip it to >=."""
        s = _series([1.0, 2.0])
        # Force TI.linear_regression_slope to return exactly the threshold
        # via monkeypatching would over-reach into TI's internals; instead
        # call classify_slope with a threshold of 0.0 against a strictly
        # flat (zero-slope) series, which lands exactly on that boundary.
        flat = _series([50.0] * 10)
        assert classify_slope(flat, period=5, threshold=0.0) is SlopeDirection.FLAT

    def test_enum_compares_equal_to_its_string_value(self):
        """SlopeDirection is a str Enum specifically so a rendered table
        cell or a JSON response can compare/serialize it directly."""
        assert SlopeDirection.RISING == "Rising"
        assert SlopeDirection.FALLING == "Falling"
        assert SlopeDirection.FLAT == "Flat"

    @given(
        values=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=15, max_size=15),
        period=st.integers(min_value=2, max_value=14),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_matches_manually_thresholded_underlying_slope(self, values, period):
        """For any input, classify_slope's label must be exactly what you'd
        get by calling TechnicalIndicators.linear_regression_slope yourself
        and thresholding it by hand -- i.e. classify_slope adds no
        independent slope logic of its own."""
        s = _series(values)
        slope = TI.linear_regression_slope(s, period)
        expected = (
            SlopeDirection.RISING if slope > FLAT_THRESHOLD else
            SlopeDirection.FALLING if slope < -FLAT_THRESHOLD else
            SlopeDirection.FLAT
        )
        assert classify_slope(s, period) is expected
