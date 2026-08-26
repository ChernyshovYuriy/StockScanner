"""
tests/test_adversarial_technical_indicators.py
================================================
Adversarial test suite for canadian_stock_screener.TechnicalIndicators —
the stateless indicator layer every score_* function and the screener's
composite score are built from. Every method here is assumed BROKEN until
it survives boundary, semantic, numerical, and property-based attacks.

Scope
-----
Data ingestion (yfinance) is out of scope. Every fixture constructs a
pandas.Series directly — no network call anywhere in this file.

Eight static methods on TechnicalIndicators:
  sma, ema, rsi, macd, adx, obv, linear_regression_slope, weekly_resample

Not covered here: ScoreCalculator's score_* methods (composite scoring built
on top of these), which were covered narratively in the preceding audit
passes this session (two genuine bugs already found and fixed there:
score_vam's negative-base complex-number crash, and weekly_resample's own
partial-current-week contamination — the fix is exercised again here as a
regression lock, this time from TechnicalIndicators' own test class rather
than through ScoreCalculator.score_stage2).

─────────────────────────────────────────────────────────────────────────────
THE INPUT STRUCTURE — CONTRACT INFERRED FROM THE CODE
─────────────────────────────────────────────────────────────────────────────

Every method here takes one or more pandas.Series, each expected to be one
column already sliced out of the OHLCV DataFrame contract established by
market_data.validate_ohlcv() (see the previous adversarial suite,
test_adversarial_market_data.py, for that contract in full):
  - index: pandas.DatetimeIndex
  - values: float64
  - Volume series specifically: float64 share/session count, not int

None of the eight methods under test validates or re-sorts its input. They
are pure, stateless, POSITIONAL functions — pandas .rolling()/.ewm() operate
on row order, not on the DatetimeIndex's chronological order. This is a
real, demonstrated contract gap (see TestSma::test_unsorted_input_is_silently_wrong
below) inherited from every caller: HistoricalSliceProvider and DataManager
both hand these functions pre-sorted Series, but TechnicalIndicators itself
enforces nothing.

ASSUMPTIONS these tests bake in:
  A1. "Look-ahead bias" for a rolling/ewm indicator means: the value at
      position i must be reproducible from only positions <= i. Every
      look-ahead test here poisons a FUTURE position with an extreme value
      and asserts the earlier positions are byte-identical to a version of
      the series where that future value never existed.
  A2. Every ewm-based indicator (ema, rsi, macd's three lines, adx) uses
      adjust=False, and pandas' adjust=False semantics with a leading NaN
      in the input (adx's dx series starts NaN, since the first bar's PDI
      and NDI are both trivially 0) skip that NaN, seed at the first
      non-NaN value, then recurse from there — NOT "treat leading NaN as
      zero." TestAdx's ground-truth hand-trace and its independent
      reference-implementation cross-check both encode this explicitly,
      since it's easy to get backwards (see the module's own dev history:
      this exact confusion produced a spurious ~2x error in an early draft
      of this file's own reference implementation, corrected before writing
      any assertion against it).
  A3. RSI is undefined (NaN) for a genuinely flat window (avg_gain==0 AND
      avg_loss==0) — this is a documented, deliberate choice in the source,
      not a bug; a pure-gain window is defined as exactly 100.0, a
      pure-loss window as exactly 0.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from canadian_stock_screener import TechnicalIndicators as TI


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _series(values, start="2024-01-02", freq="B") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values)) if freq == "B" else \
        pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series([float(v) for v in values], index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TechnicalIndicators.sma
# ─────────────────────────────────────────────────────────────────────────────

class TestSma:
    """Hypothesis under test: min_periods silently defaults to 1 (partial
    windows leak in at the start), or the window is off-by-one."""

    def test_ground_truth(self):
        s = _series([1, 2, 3, 4, 5])
        out = TI.sma(s, 3)
        assert out.tolist()[:2] == [None, None] or all(pd.isna(out.iloc[:2]))
        assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0]

    def test_boundary_no_partial_windows_at_start(self):
        """min_periods=period (not 1) — the first (period-1) values must be
        NaN, not a partial-window average."""
        s = _series([10, 20, 30])
        out = TI.sma(s, period=5)
        assert out.isna().all()

    def test_boundary_period_one_equals_series_itself(self):
        s = _series([5, -3, 0, 100])
        out = TI.sma(s, period=1)
        assert out.tolist() == s.tolist()

    def test_boundary_empty_series(self):
        s = _series([])
        out = TI.sma(s, 3)
        assert out.empty

    def test_lookahead_future_poison_value_never_leaks(self):
        s = _series([1, 2, 3, 4, 1_000_000])
        out = TI.sma(s, 3)
        assert out.iloc[3] == (2.0 + 3.0 + 4.0) / 3.0  # window [1,2,3]->idx1..3, unaffected by idx4

    def test_unsorted_input_is_silently_wrong(self):
        """DOCUMENTS the contract gap (assumption in the module docstring):
        sma operates on ROW ORDER, not chronological order. A caller that
        accidentally hands it an unsorted Series gets a silently-computed,
        chronologically-meaningless average with no error of any kind."""
        idx_sorted = pd.date_range("2024-01-01", periods=5)
        s_sorted = pd.Series([1.0, 2, 3, 4, 5], index=idx_sorted)
        s_shuffled = pd.Series([1.0, 2, 3, 4, 5], index=idx_sorted).sample(frac=1, random_state=3)
        out_sorted = TI.sma(s_sorted, 3)
        out_shuffled = TI.sma(s_shuffled, 3)
        # Same values, same length, different (positionally-computed) results —
        # proving no internal sort/validation happens.
        assert out_sorted.tolist()[2:] != out_shuffled.tolist()[2:]

    @given(
        values=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=1, max_size=50),
        period=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_output_length_matches_input(self, values, period):
        s = _series(values)
        out = TI.sma(s, period)
        assert len(out) == len(s)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TechnicalIndicators.ema
# ─────────────────────────────────────────────────────────────────────────────

class TestEma:
    """Hypothesis under test: the smoothing constant doesn't match the
    documented span->alpha conversion (alpha = 2/(span+1))."""

    def test_ground_truth_matches_hand_computed_ewm(self):
        s = _series([10, 12, 14, 12, 10])
        out = TI.ema(s, period=3)
        alpha = 2 / (3 + 1)
        hand = [10.0]
        for v in [12, 14, 12, 10]:
            hand.append(alpha * v + (1 - alpha) * hand[-1])
        assert out.round(6).tolist() == [round(v, 6) for v in hand]

    def test_boundary_single_value_equals_itself(self):
        s = _series([42.0])
        out = TI.ema(s, period=10)
        assert out.iloc[0] == 42.0

    def test_no_nan_produced_unlike_sma(self):
        """Boundary: unlike sma, ewm has no min_periods requirement — every
        position gets a defined value, even the first."""
        s = _series([1, 2, 3])
        out = TI.ema(s, period=50)
        assert not out.isna().any()

    @given(
        values=st.lists(st.floats(min_value=1, max_value=1e6, allow_nan=False), min_size=1, max_size=50),
        period=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_bounded_by_series_min_max(self, values, period):
        """Property: an EMA of positive values is always within
        [min(series), max(series)] — a weighted average can never overshoot
        its own inputs."""
        s = _series(values)
        out = TI.ema(s, period)
        assert out.min() >= s.min() - 1e-9
        assert out.max() <= s.max() + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 3. TechnicalIndicators.rsi
# ─────────────────────────────────────────────────────────────────────────────

class TestRsi:
    """Hypothesis under test: the div-by-zero dodge (avg_loss.replace(0,nan))
    silently produces NaN for a pure-gain window instead of the textbook 100,
    or silently produces a value outside [0,100]."""

    def test_ground_truth_pure_uptrend_is_100(self):
        s = _series([100 + i for i in range(20)])
        assert TI.rsi(s, 14).iloc[-1] == 100.0

    def test_ground_truth_pure_downtrend_is_0(self):
        s = _series([100 - i for i in range(20)])
        assert TI.rsi(s, 14).iloc[-1] == 0.0

    def test_ground_truth_flat_price_is_nan_not_zero_or_crash(self):
        """Documented, deliberate: RSI is undefined for zero movement in
        EITHER direction, not defined as 0, 50, or 100."""
        s = _series([100.0] * 20)
        assert pd.isna(TI.rsi(s, 14).iloc[-1])

    def test_boundary_single_value_is_nan(self):
        s = _series([100.0])
        out = TI.rsi(s, 14)
        assert pd.isna(out.iloc[0])

    def test_lookahead_future_poison_does_not_change_earlier_rsi(self):
        s1 = _series([100 + i for i in range(20)])
        s2 = pd.concat([s1, pd.Series([-999999.0], index=[s1.index[-1] + pd.Timedelta(days=1)])])
        r1 = TI.rsi(s1, 14)
        r2 = TI.rsi(s2, 14)
        assert r1.iloc[-1] == r2.iloc[len(s1) - 1]

    @given(
        n=st.integers(min_value=15, max_value=60),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_bounded_zero_to_hundred(self, n, seed):
        """The classic RSI invariant: for ANY positive-price random walk,
        every non-NaN RSI value must fall in [0, 100]."""
        rng = np.random.default_rng(seed)
        prices = np.cumsum(rng.standard_normal(n)) + 100
        prices = np.maximum(prices, 1.0)  # keep strictly positive
        s = _series(prices)
        out = TI.rsi(s, 14).dropna()
        assert (out >= 0).all() and (out <= 100).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. TechnicalIndicators.macd
# ─────────────────────────────────────────────────────────────────────────────

class TestMacd:
    """Hypothesis under test: histogram isn't actually macd_line minus
    signal_line (an alignment/off-by-one bug in the composition)."""

    def test_ground_truth_composition_matches_independent_ema_calls(self):
        s = _series(list(range(1, 40)), freq="D")
        macd_line, signal_line, hist = TI.macd(s, fast=3, slow=6, signal=2)
        hand_macd = TI.ema(s, 3) - TI.ema(s, 6)
        hand_signal = TI.ema(hand_macd, 2)
        assert (macd_line.round(9) == hand_macd.round(9)).all()
        assert (signal_line.round(9) == hand_signal.round(9)).all()

    def test_property_histogram_is_always_macd_minus_signal(self):
        """Algebraic identity that must hold exactly for ANY input —
        histogram is defined as macd_line - signal_line, not independently
        computed, so this can never legitimately drift."""
        s = _series(np.random.default_rng(1).standard_normal(80).cumsum() + 100, freq="D")
        macd_line, signal_line, hist = TI.macd(s)
        assert ((hist - (macd_line - signal_line)).abs() < 1e-9).all()

    def test_boundary_short_series_still_returns_values_not_all_nan(self):
        """ewm has no min_periods floor — even a 5-bar series against the
        default fast=12/slow=26/signal=9 must produce defined (non-NaN)
        numbers, not blow up."""
        s = _series([10, 11, 9, 12, 10])
        macd_line, signal_line, hist = TI.macd(s)
        assert not macd_line.isna().any()


# ─────────────────────────────────────────────────────────────────────────────
# 5. TechnicalIndicators.adx
# ─────────────────────────────────────────────────────────────────────────────

def _reference_adx(high, low, close, period=14):
    """Independent re-derivation of Wilder's ADX (loop-based, not vectorized
    pandas) — a differential ground-truth oracle for cases too tedious to
    fully hand-trace. Mirrors pandas .ewm(adjust=False)'s leading-NaN
    semantics explicitly (skip NaN, seed at first non-NaN, recurse) — see
    module docstring assumption A2."""
    n = len(high)
    tr, plus_dm, minus_dm = [None] * n, [None] * n, [None] * n
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
            plus_dm[i] = minus_dm[i] = 0.0
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            up, down = high[i] - high[i - 1], low[i - 1] - low[i]
            plus_dm[i] = up if (up > down and up > 0) else 0.0
            minus_dm[i] = down if (down > up and down > 0) else 0.0

    alpha = 1.0 / period

    def ewm_adjust_false(vals):
        out, prev, seeded = [None] * len(vals), None, False
        for i, v in enumerate(vals):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out[i] = float("nan")
                continue
            if not seeded:
                out[i], prev, seeded = v, v, True
            else:
                out[i] = alpha * v + (1 - alpha) * prev
                prev = out[i]
        return out

    atr = ewm_adjust_false(tr)
    pdms, ndms = ewm_adjust_false(plus_dm), ewm_adjust_false(minus_dm)
    pdi = [100 * pdms[i] / atr[i] if atr[i] not in (0,) and not np.isnan(atr[i]) else float("nan") for i in range(n)]
    ndi = [100 * ndms[i] / atr[i] if atr[i] not in (0,) and not np.isnan(atr[i]) else float("nan") for i in range(n)]
    dx = []
    for i in range(n):
        denom = pdi[i] + ndi[i]
        dx.append(100 * abs(pdi[i] - ndi[i]) / denom if denom != 0 and not np.isnan(denom) else float("nan"))
    return ewm_adjust_false(dx)


class TestAdx:
    """Hypothesis under test: the atr==0 division silently produces Inf
    instead of NaN (crashing every downstream comparison), or the
    Wilder-smoothing chain has an off-by-one that only shows up over many
    bars (hence the reference-implementation cross-check, not just a short
    hand-trace)."""

    def test_ground_truth_hand_traced_pure_uptrend_4_bars(self):
        """4 bars, period=2, every bar strictly making new highs with no
        down day at all: NDI stays exactly 0 throughout, so ADX must reach
        exactly 100.0 by the second bar and hold there. Full hand trace in
        the audit; TR=[1,2.5,2.5,2.5], ATR=[1,1.75,2.125,2.3125],
        +DM=[0,2,2,2], smoothed +DM=[0,1.0,1.5,1.75], PDI=[0,57.14,70.59,75.68],
        NDI=[0,0,0,0], DX=[nan,100,100,100], ADX=[nan,100,100,100]."""
        idx = pd.date_range("2024-01-01", periods=4)
        high = pd.Series([10.0, 12, 14, 16], index=idx)
        low = pd.Series([9.0, 10, 12, 14], index=idx)
        close = pd.Series([9.5, 11.5, 13.5, 15.5], index=idx)
        out = TI.adx(high, low, close, period=2)
        assert pd.isna(out.iloc[0])
        assert out.iloc[1:].round(6).tolist() == [100.0, 100.0, 100.0]

    def test_boundary_flat_price_all_nan_not_inf_or_crash(self):
        """High==Low==Close every bar (a halted, unmoving print): TR=0,
        ATR=0, PDI/NDI are 0/0 -> NaN throughout, not a ZeroDivisionError
        and not a silent Inf that would corrupt every downstream comparison
        (score_adx already relies on `.dropna().empty` to detect this)."""
        flat = pd.Series([50.0] * 20)
        out = TI.adx(flat, flat, flat, period=14)
        assert out.isna().all()
        assert not np.isinf(out.fillna(0)).any()

    def test_ground_truth_against_independent_reference_implementation(self):
        """25 bars of random-walk OHLC, cross-checked against a from-scratch
        loop-based Wilder ADX re-derivation (not a re-run of the same
        vectorized code) — the ground-truth technique for an indicator too
        long-chained to hand-trace in full."""
        rng = np.random.default_rng(7)
        n = 25
        close = np.cumsum(rng.standard_normal(n)) + 100
        high = close + np.abs(rng.standard_normal(n)) * 0.7
        low = close - np.abs(rng.standard_normal(n)) * 0.7
        idx = pd.date_range("2024-01-01", periods=n)
        out = TI.adx(pd.Series(high, index=idx), pd.Series(low, index=idx), pd.Series(close, index=idx), period=14)
        ref = _reference_adx(list(high), list(low), list(close), period=14)
        for a, b in zip(out.tolist(), ref):
            if np.isnan(a) and np.isnan(b):
                continue
            assert abs(a - b) < 1e-6

    @given(seed=st.integers(min_value=0, max_value=10_000))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_bounded_zero_to_hundred(self, seed):
        rng = np.random.default_rng(seed)
        n = 40
        close = np.cumsum(rng.standard_normal(n)) + 100
        high = close + np.abs(rng.standard_normal(n)) * 0.5 + 0.01
        low = close - np.abs(rng.standard_normal(n)) * 0.5 - 0.01
        idx = pd.date_range("2024-01-01", periods=n)
        out = TI.adx(pd.Series(high, index=idx), pd.Series(low, index=idx), pd.Series(close, index=idx), period=14)
        out = out.dropna()
        assert (out >= -1e-9).all() and (out <= 100 + 1e-9).all()


# ─────────────────────────────────────────────────────────────────────────────
# 6. TechnicalIndicators.obv
# ─────────────────────────────────────────────────────────────────────────────

class TestObv:
    """Hypothesis under test: a zero-change day silently moves OBV (it
    shouldn't), or OBV isn't strictly the signed cumulative volume."""

    def test_ground_truth_declining_price(self):
        close = _series([20, 19, 18, 17, 16])
        vol = _series([100] * 5)
        out = TI.obv(close, vol)
        assert pd.isna(out.iloc[0])
        assert out.iloc[1:].tolist() == [-100.0, -200.0, -300.0, -400.0]

    def test_ground_truth_flat_day_does_not_move_obv(self):
        close = _series([10, 11, 11, 12])
        vol = _series([100] * 4)
        out = TI.obv(close, vol)
        assert out.iloc[1:].tolist() == [100.0, 100.0, 200.0]  # bar 2 (no change) doesn't move it

    def test_boundary_single_value_is_nan(self):
        close = _series([50.0])
        vol = _series([1000.0])
        out = TI.obv(close, vol)
        assert pd.isna(out.iloc[0])

    def test_property_step_magnitude_equals_that_days_volume(self):
        """Property: whenever price changes, |OBV[i]-OBV[i-1]| == volume[i]
        exactly — OBV can never move by more or less than the day's own
        volume. OBV[0] is NaN (no prior bar to diff against), so the first
        comparable step is day2-vs-day1, i.e. vol.iloc[2:]."""
        close = _series([10, 12, 11, 15, 9])
        vol = _series([100, 250, 80, 400, 60])
        out = TI.obv(close, vol)
        steps = out.diff().dropna().abs()
        assert steps.tolist() == vol.iloc[2:].tolist()

    @given(
        closes=st.lists(st.floats(min_value=1, max_value=1000, allow_nan=False), min_size=2, max_size=40),
    )
    @settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_monotonic_price_gives_monotonic_obv(self, closes):
        """Monotonicity property: a STRICTLY increasing price series
        (dedup + sort ascending, since flat/duplicate values would legally
        pause OBV) must produce a non-decreasing OBV, and vice versa for a
        strictly decreasing series."""
        distinct_sorted = sorted(set(round(c, 6) for c in closes))
        if len(distinct_sorted) < 2:
            return
        close = _series(distinct_sorted)
        vol = _series([100] * len(distinct_sorted))
        out = TI.obv(close, vol).dropna()
        assert (out.diff().dropna() >= 0).all()

        close_dec = _series(list(reversed(distinct_sorted)))
        out_dec = TI.obv(close_dec, vol).dropna()
        assert (out_dec.diff().dropna() <= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 7. TechnicalIndicators.linear_regression_slope
# ─────────────────────────────────────────────────────────────────────────────

class TestLinearRegressionSlope:
    """Hypothesis under test: normalizing by the RAW (signed) mean instead
    of abs(mean) flips the sign for any series whose mean is negative — the
    exact historical bug this function's own comment documents having
    fixed. Regression-locked here independently of ScoreCalculator.score_macd."""

    def test_ground_truth_declining_series_with_negative_mean_stays_negative(self):
        """The specific case that was previously silently WRONG: a clearly
        declining series whose mean happens to be negative (a MACD
        histogram rolling from positive to negative territory is exactly
        this shape) must report a NEGATIVE slope, not a sign-flipped
        positive one."""
        s = _series([-1, -3, -5, -7, -9])
        assert TI.linear_regression_slope(s, period=5) < 0

    def test_ground_truth_rising_series_with_negative_mean_stays_positive(self):
        s = _series([-9, -7, -5, -3, -1])
        assert TI.linear_regression_slope(s, period=5) > 0

    def test_boundary_too_few_points_returns_zero(self):
        s = _series([1, 2, 3])
        assert TI.linear_regression_slope(s, period=5) == 0.0

    def test_boundary_period_one_returns_zero_not_a_crash(self):
        """period=1 requests a 1-point regression via max(2, period)=2, but
        .iloc[-1:] only ever supplies 1 point -> must return 0.0, not divide
        by a zero-length denom or raise."""
        s = _series([1, 2])
        assert TI.linear_regression_slope(s, period=1) == 0.0

    def test_flat_series_slope_is_exactly_zero(self):
        s = _series([100.0] * 10)
        assert TI.linear_regression_slope(s, period=10) == 0.0

    @given(
        start=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        step=st.floats(min_value=0.01, max_value=100, allow_nan=False),
        n=st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_sign_always_matches_direction_regardless_of_mean_sign(self, start, step, n):
        """The core invariant: for ANY strictly monotonic series — including
        ones with a negative mean — a rising series must have positive
        slope and a falling series must have negative slope. This is the
        property whose violation was the actual historical bug."""
        rising = _series([start + i * step for i in range(n)])
        falling = _series([start - i * step for i in range(n)])
        assert TI.linear_regression_slope(rising, period=n) > 0
        assert TI.linear_regression_slope(falling, period=n) < 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. TechnicalIndicators.weekly_resample
# ─────────────────────────────────────────────────────────────────────────────

class TestWeeklyResample:
    """Hypothesis under test: the in-progress final week is folded in as if
    complete (this session's fix — regression-locked here independently of
    ScoreCalculator.score_stage2, which is where it was originally found)."""

    def test_ground_truth_two_full_weeks(self):
        idx = pd.bdate_range("2024-01-01", periods=10)  # Mon 1/1 .. Fri 1/12, two full weeks
        s = pd.Series(range(1, 11), index=idx, dtype=float)
        out = TI.weekly_resample(s)
        assert [d.date().isoformat() for d in out.index] == ["2024-01-05", "2024-01-12"]
        assert out.tolist() == [5.0, 10.0]  # each Friday's own value (last-of-week)

    def test_partial_final_week_is_dropped(self):
        """FIX regression lock: daily data ending mid-week (Tuesday) must
        resample to the prior COMPLETE Friday, not a phantom in-progress
        week labeled with the upcoming Friday."""
        idx = pd.bdate_range("2024-01-01", periods=7)  # Mon1/1-Fri1/5 (full week) + Mon1/8,Tue1/9 (partial)
        s = pd.Series(range(1, 8), index=idx, dtype=float)
        out = TI.weekly_resample(s)
        assert out.index[-1].date().isoformat() == "2024-01-05"
        assert len(out) == 1

    def test_four_day_holiday_shortened_week_is_kept(self):
        """The threshold is 4, not 5, specifically so a single-holiday week
        (a real, common case) isn't discarded as if it were in-progress."""
        idx = pd.bdate_range("2024-01-01", periods=4)  # Mon-Thu only, as if Fri was a holiday
        s = pd.Series(range(1, 5), index=idx, dtype=float)
        out = TI.weekly_resample(s)
        assert len(out) == 1
        assert out.iloc[-1] == 4.0  # Thursday's value stands in for the week

    def test_boundary_single_day_is_dropped_not_kept(self):
        idx = pd.bdate_range("2024-01-01", periods=1)  # just Monday
        s = pd.Series([1.0], index=idx)
        out = TI.weekly_resample(s)
        assert out.empty

    def test_property_output_never_longer_than_a_naive_resample(self):
        """Property: dropping a partial week can only ever shrink the
        output relative to a naive `.resample('W-FRI').last()` with no
        completeness check, never grow it."""
        idx = pd.bdate_range("2024-01-01", periods=17)  # 3 full weeks + 2 extra days
        s = pd.Series(range(1, 18), index=idx, dtype=float)
        naive = s.resample("W-FRI").last()
        out = TI.weekly_resample(s)
        assert len(out) <= len(naive)
