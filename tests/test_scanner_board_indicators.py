"""
tests/test_scanner_board_indicators.py
========================================
Tests for scanner_board.indicators: stochastic(), accumulation_distribution(),
force_index() — see scanner_board/PLAN.md Phase 1.

Scope note: this file only covers indicators genuinely new to this codebase.
Everything scanner_board reuses by import from
canadian_stock_screener.TechnicalIndicators (sma, ema, true_range, atr, ...)
is already covered by tests/test_adversarial_technical_indicators.py; that
existing coverage is not duplicated here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from scanner_board.indicators import (
    stochastic,
    accumulation_distribution,
    force_index,
)


def _series(values, start="2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series([float(v) for v in values], index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# 1. stochastic
# ─────────────────────────────────────────────────────────────────────────────

class TestStochastic:
    """Hypothesis under test: Slow %K/%D get built from the wrong series
    (e.g. Slow %K re-derived from raw Fast %K instead of from Fast %D, or
    the two smoothing passes collapsed into one)."""

    def test_ground_truth_hand_traced(self):
        """5 bars, k_period=2, d_period=2. Per-bar LowN/HighN (2-bar rolling
        window): bar1 undefined (only 1 bar), bar2 window=[8,11],
        bar3=[9,12], bar4=[9,12], bar5=[9,13].
        Fast %K = (C-LowN)/(HighN-LowN)*100:
          bar2=(10-8)/(11-8)*100=66.667, bar3=(11-9)/(12-9)*100=66.667,
          bar4=(10.5-9)/(12-9)*100=50.0, bar5=(12-9)/(13-9)*100=75.0.
        Fast %D = SMA(Fast %K, 2): bar3=(66.667+66.667)/2=66.667,
          bar4=(66.667+50.0)/2=58.333, bar5=(50.0+75.0)/2=62.5.
        Slow %K = Fast %D (same numbers).
        Slow %D = SMA(Slow %K, 2): bar4=(66.667+58.333)/2=62.5,
          bar5=(58.333+62.5)/2=60.4167."""
        high = _series([10.0, 11, 12, 11, 13])
        low = _series([8.0, 9, 10, 9, 10])
        close = _series([9.0, 10, 11, 10.5, 12])
        result = stochastic(high, low, close, k_period=2, d_period=2)

        expected_k = [np.nan, np.nan, 66.666667, 58.333333, 62.5]
        expected_d = [np.nan, np.nan, np.nan, 62.5, 60.416667]
        for got, want in zip(result.percent_k.tolist(), expected_k):
            if np.isnan(want):
                assert np.isnan(got)
            else:
                assert got == pytest.approx(want, abs=1e-4)
        for got, want in zip(result.percent_d.tolist(), expected_d):
            if np.isnan(want):
                assert np.isnan(got)
            else:
                assert got == pytest.approx(want, abs=1e-4)

    def test_boundary_high_equals_low_gives_nan_not_inf(self):
        """A halted/unmoving bar where HighN==LowN makes the %K ratio a
        division by zero -- book's own formula has no guard for this, and
        neither does the standard implementation; the requirement here is
        just that it comes out as pandas' own NaN, not a silent Inf that
        would corrupt every later percentile/zone comparison."""
        flat = _series([50.0] * 5)
        result = stochastic(flat, flat, flat, k_period=2, d_period=2)
        assert not np.isinf(result.percent_k.fillna(0)).any()
        assert not np.isinf(result.percent_d.fillna(0)).any()

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_bounded_zero_to_hundred(self, seed):
        """Fast %K is bounded [0,100] by construction (it's a ratio of a
        sub-range to its own bounding range); Slow %K/%D, being plain SMAs
        of a [0,100]-bounded series, must stay within the same bounds."""
        rng = np.random.default_rng(seed)
        n = 40
        close = np.cumsum(rng.standard_normal(n)) + 100
        high = close + np.abs(rng.standard_normal(n)) * 0.5 + 0.01
        low = close - np.abs(rng.standard_normal(n)) * 0.5 - 0.01
        result = stochastic(_series(high), _series(low), _series(close), k_period=5, d_period=3)
        k, d = result.percent_k.dropna(), result.percent_d.dropna()
        assert (k >= -1e-9).all() and (k <= 100 + 1e-9).all()
        assert (d >= -1e-9).all() and (d <= 100 + 1e-9).all()


# ─────────────────────────────────────────────────────────────────────────────
# 2. accumulation_distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestAccumulationDistribution:
    """Hypothesis under test: a High==Low bar's division by zero silently
    poisons every later cumulative value with NaN or Inf instead of
    contributing zero for that one bar."""

    def test_ground_truth_hand_traced(self):
        """3 bars. Bar1: (11-10)/(12-9)*1000 = 333.333. Bar2: High==Low=11
        -> treated as zero contribution. Bar3: (10-12)/(15-10)*2000=-800.
        Cumulative: [333.333, 333.333, -466.667]."""
        open_ = _series([10.0, 11.0, 12.0])
        high = _series([12.0, 11.0, 15.0])
        low = _series([9.0, 11.0, 10.0])
        close = _series([11.0, 11.0, 10.0])
        volume = _series([1000.0, 500.0, 2000.0])
        out = accumulation_distribution(open_, high, low, close, volume)
        assert out.tolist() == pytest.approx([333.333333, 333.333333, -466.666667], abs=1e-4)

    def test_boundary_high_equals_low_is_zero_contribution_not_crash(self):
        flat_hl = _series([50.0] * 5)
        open_ = _series([10.0, 11.0, 9.0, 12.0, 8.0])
        close = _series([11.0, 10.0, 12.0, 9.0, 13.0])
        volume = _series([100.0] * 5)
        out = accumulation_distribution(open_, flat_hl, flat_hl, close, volume)
        assert not out.isna().any()
        assert not np.isinf(out).any()
        # every bar contributes exactly zero -> the cumulative total never moves
        assert (out == 0.0).all()

    def test_property_output_is_monotonically_defined_running_total(self):
        """A/D's cumulative value at bar i must equal the sum of every
        contribution up to and including bar i -- i.e. it really is a
        running total, not something that resets or looks ahead."""
        rng = np.random.default_rng(11)
        n = 20
        open_ = pd.Series(rng.uniform(9, 11, n))
        close = pd.Series(rng.uniform(9, 11, n))
        high = pd.Series(np.maximum(open_, close) + rng.uniform(0.1, 1, n))
        low = pd.Series(np.minimum(open_, close) - rng.uniform(0.1, 1, n))
        volume = pd.Series(rng.uniform(100, 1000, n))
        out = accumulation_distribution(open_, high, low, close, volume)
        contribution = ((close - open_) / (high - low)) * volume
        assert out.tolist() == pytest.approx(contribution.cumsum().tolist())


# ─────────────────────────────────────────────────────────────────────────────
# 3. force_index
# ─────────────────────────────────────────────────────────────────────────────

class TestForceIndex:
    """Hypothesis under test: `short`/`long` get swapped, or raw isn't
    exactly volume * price-change (e.g. accidentally using pct_change)."""

    def test_ground_truth_hand_traced(self):
        """close=[10,12,11,15], volume=[100,200,150,300].
        raw = volume*close.diff() = [nan, 200*2=400, 150*-1=-150, 300*4=1200].
        short = EMA(raw, span=2) [adjust=False, seeded at first non-NaN]:
          [nan, 400, 33.333, 811.111].
        long = EMA(raw, span=3): [nan, 400, 125.0, 662.5]."""
        close = _series([10.0, 12.0, 11.0, 15.0])
        volume = _series([100.0, 200.0, 150.0, 300.0])
        result = force_index(close, volume, short_period=2, long_period=3)

        assert np.isnan(result.raw.iloc[0])
        assert result.raw.iloc[1:].tolist() == [400.0, -150.0, 1200.0]

        assert np.isnan(result.short.iloc[0])
        assert result.short.iloc[1:].tolist() == pytest.approx([400.0, 33.333333, 811.111111], abs=1e-4)

        assert np.isnan(result.long.iloc[0])
        assert result.long.iloc[1:].tolist() == pytest.approx([400.0, 125.0, 662.5], abs=1e-4)

    def test_property_raw_sign_matches_price_change_direction(self):
        """Ch.30: 'if prices close higher than the close of the previous
        bar, the force is positive... if lower, the force is negative.'"""
        rng = np.random.default_rng(5)
        n = 30
        close = pd.Series(np.cumsum(rng.standard_normal(n)) + 100)
        volume = pd.Series(rng.uniform(100, 1000, n))
        result = force_index(close, volume)
        diff = close.diff()
        for i in range(1, n):
            if diff.iloc[i] > 0:
                assert result.raw.iloc[i] > 0
            elif diff.iloc[i] < 0:
                assert result.raw.iloc[i] < 0
            else:
                assert result.raw.iloc[i] == 0

    def test_boundary_zero_volume_gives_zero_force_not_crash(self):
        close = _series([10.0, 12.0, 8.0])
        volume = _series([0.0, 0.0, 0.0])
        result = force_index(close, volume)
        assert result.raw.iloc[1:].tolist() == [0.0, 0.0]
