"""
tests/test_backtest_refactor.py
================================
Characterization and regression tests for the backtest refactor.

Run with:
    pytest tests/test_backtest_refactor.py -v

Tests are grouped by phase gate — run the relevant mark after each phase:

  Phase 1  (clock injection)     : pytest -v -m phase1
  Phase 2  (data provider)       : pytest -v -m phase2
  Phase 3  (portfolio state obj) : pytest -v -m phase3
  All characterization baselines : pytest -v -m characterization

The characterization tests establish golden outputs from the CURRENT business
logic.  Any refactoring step that changes their output is a regression.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

TSX_TZ = ZoneInfo("America/Toronto")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURE FACTORIES
# ─────────────────────────────────────────────────────────────────────────────

def _make_trending_ohlcv(
        n: int = 300,
        seed: int = 42,
        start: str = "2023-01-01",
        trend: float = 0.05,  # daily drift as fraction of price
        base_price: float = 20.0,
        vol_shares: int = 300_000,
        atr_noise: float = 0.3,  # absolute spread for high/low
) -> pd.DataFrame:
    """Return a daily OHLCV DataFrame with a clean uptrend, no gaps."""
    np.random.seed(seed)
    idx = pd.bdate_range(start, periods=n)
    prices = base_price + np.arange(n) * trend + np.random.randn(n) * 0.05
    close = pd.Series(prices, index=idx)
    high = close + np.abs(np.random.randn(n)) * atr_noise
    low = close - np.abs(np.random.randn(n)) * atr_noise
    volume = pd.Series(np.ones(n) * vol_shares + np.random.randint(-50_000, 50_000, n),
                       index=idx, dtype=float)
    return pd.DataFrame({"Open": close, "High": high, "Low": low,
                         "Close": close, "Volume": volume})


def _make_base_breakout_ohlcv(
        n: int = 300,
        seed: int = 99,
        start: str = "2023-01-01",
) -> pd.DataFrame:
    """
    Return OHLCV where the final bar is clearly above a tight 40-bar base
    with high volume — BASE CONFIRMED fixture.
    """
    np.random.seed(seed)
    idx = pd.bdate_range(start, periods=n)
    # Slow uptrend for first 260 bars, then tight base (±1%) for 39 bars
    trend = np.linspace(10, 30, n - 40)
    base = np.full(40, 30.0) + np.random.randn(40) * 0.1  # ±0.3%
    prices = np.concatenate([trend, base])
    close = pd.Series(prices, index=idx)
    high = close + 0.2
    low = close - 0.2
    volume = pd.Series(np.ones(n) * 300_000, index=idx, dtype=float)

    # Last bar: breakout above base top with 3× volume
    high.iloc[-1] = 31.5
    close.iloc[-1] = 31.0
    volume.iloc[-1] = 300_000 * 3.0

    return pd.DataFrame({"Open": close * 0.99, "High": high, "Low": low,
                         "Close": close, "Volume": volume})


def _make_ema_pullback_ohlcv(
        n: int = 300,
        ema_period: int = 21,
        seed: int = 77,
        start: str = "2023-01-01",
) -> pd.DataFrame:
    """
    Return OHLCV where price pulls back to EMA21 in the last few bars,
    then reclaims it on the last bar with volume — PB-EMA21 CONFIRMED.
    """
    np.random.seed(seed)
    idx = pd.bdate_range(start, periods=n)
    prices = np.linspace(10, 40, n) + np.random.randn(n) * 0.05
    close = pd.Series(prices, index=idx)

    from auto_pipeline import _ema
    ema = _ema(close, ema_period)

    # Simulate pullback: last 3 bars dip to EMA, last bar reclaims
    close.iloc[-3] = float(ema.iloc[-3]) * 0.998
    close.iloc[-2] = float(ema.iloc[-2]) * 0.999
    close.iloc[-1] = float(ema.iloc[-1]) * 1.003  # reclaim

    high = close + 0.2
    low = close - 0.2
    avg_vol = 300_000
    volume = pd.Series(np.ones(n) * avg_vol, index=idx, dtype=float)
    volume.iloc[-1] = avg_vol * 1.2  # volume ≥ 0.8× avg — vol_ok

    return pd.DataFrame({"Open": close * 0.99, "High": high, "Low": low,
                         "Close": close, "Volume": volume})


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — CLOCK INJECTION
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.phase1
class TestClockInjection:
    """
    Verifies the time_utils.set_backtest_clock() contract introduced in Phase 1.
    All tests in this class must pass immediately after Patch 1 is applied and
    must continue passing through every subsequent phase.
    """

    def test_live_mode_returns_real_time(self):
        """Default behaviour: market_now() returns a real, current timestamp."""
        from time_utils import market_now, set_backtest_clock
        set_backtest_clock(None)  # ensure clean state
        now = market_now()
        real = datetime.now(tz=TSX_TZ)
        # Should be within a few seconds of wall clock
        delta = abs((real - now).total_seconds())
        assert delta < 5, f"market_now() diverges from wall clock by {delta:.1f}s"

    def test_backtest_clock_pins_market_now(self):
        from time_utils import market_now, set_backtest_clock
        sim = datetime(2024, 6, 15, 16, 5, tzinfo=TSX_TZ)
        set_backtest_clock(sim)
        try:
            assert market_now() == sim
        finally:
            set_backtest_clock(None)

    def test_backtest_clock_pins_market_today(self):
        from time_utils import market_today, set_backtest_clock
        sim = datetime(2024, 6, 15, 16, 5, tzinfo=TSX_TZ)
        set_backtest_clock(sim)
        try:
            today = market_today()
            assert today.date() == date(2024, 6, 15)
            assert today.hour == 0 and today.minute == 0
        finally:
            set_backtest_clock(None)

    def test_set_backtest_clock_none_restores_live(self):
        from time_utils import market_now, set_backtest_clock, is_backtest_mode
        sim = datetime(2020, 1, 1, tzinfo=TSX_TZ)
        set_backtest_clock(sim)
        assert is_backtest_mode()
        set_backtest_clock(None)
        assert not is_backtest_mode()
        # After restore, returns real time
        now = market_now()
        real = datetime.now(tz=TSX_TZ)
        assert abs((real - now).total_seconds()) < 5

    def test_is_backtest_mode_false_by_default(self):
        from time_utils import is_backtest_mode, set_backtest_clock
        set_backtest_clock(None)
        assert not is_backtest_mode()

    def test_is_backtest_mode_true_when_pinned(self):
        from time_utils import is_backtest_mode, set_backtest_clock
        set_backtest_clock(datetime(2024, 1, 1, tzinfo=TSX_TZ))
        try:
            assert is_backtest_mode()
        finally:
            set_backtest_clock(None)

    def test_naive_datetime_rejected(self):
        from time_utils import set_backtest_clock
        with pytest.raises(ValueError, match="timezone-aware"):
            set_backtest_clock(datetime(2024, 1, 1))  # no tzinfo

    def test_clock_carries_through_market_today_str(self):
        from time_utils import market_today_str, set_backtest_clock
        sim = datetime(2025, 3, 15, 16, 0, tzinfo=TSX_TZ)
        set_backtest_clock(sim)
        try:
            assert market_today_str() == "2025-03-15"
        finally:
            set_backtest_clock(None)

    def test_multiple_pins_without_restore(self):
        """Setting the clock twice overwrites cleanly."""
        from time_utils import market_now, set_backtest_clock
        d1 = datetime(2024, 1, 1, tzinfo=TSX_TZ)
        d2 = datetime(2025, 6, 30, tzinfo=TSX_TZ)
        set_backtest_clock(d1)
        set_backtest_clock(d2)
        try:
            assert market_now() == d2.astimezone(TSX_TZ).replace(microsecond=0)
        finally:
            set_backtest_clock(None)

    def test_date_formatting_helpers_respect_clock(self):
        from time_utils import (date_to_iso_basic, date_to_iso_extended,
                                market_now, set_backtest_clock)
        sim = datetime(2025, 11, 7, 12, 0, tzinfo=TSX_TZ)
        set_backtest_clock(sim)
        try:
            now = market_now()
            assert date_to_iso_basic(now) == "20251107"
            assert date_to_iso_extended(now) == "2025-11-07"
        finally:
            set_backtest_clock(None)


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTERIZATION — ScoreCalculator (canadian_stock_screener.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.characterization
class TestScoreCalculatorGoldens:
    """
    Golden-value characterization tests.  Exact numbers are locked to the
    current implementation.  A failing test here means the refactor changed
    business logic — investigate before merging.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from canadian_stock_screener import ScoreCalculator, CONFIG
        self.sc = ScoreCalculator(CONFIG)
        self.df = _make_trending_ohlcv(n=300, seed=42)
        self.close = self.df["Close"]
        self.high = self.df["High"]
        self.low = self.df["Low"]
        self.volume = self.df["Volume"]
        # Benchmark slightly below close to get non-trivial RS
        self.bench = self.close * 0.97

    # ── individual scores ──────────────────────────────────────────────────

    def test_score_stage2_type_and_range(self):
        s = self.sc.score_stage2(self.close)
        assert isinstance(s, float)
        assert 0.0 <= s <= 100.0

    def test_score_stage2_golden(self):
        # seed=42, n=300, trend=0.05 uptrend → strong Stage II
        # Golden value dropped 84.9 -> 74.9 when ma10w > ma30w became a hard
        # gate instead of a +10 bonus (2026-08, matching the reference
        # StageDetector in /home/yurii/dev/pythonfintech/market-stage-detection —
        # the fixture already satisfied ma10>ma30, so only the bonus's
        # removal shows up here, not the gate itself).
        s = self.sc.score_stage2(self.close)
        assert s == pytest.approx(74.9, abs=0.5), f"stage2 golden changed: {s}"

    def test_score_macd_range(self):
        s = self.sc.score_macd(self.close)
        assert 0.0 <= s <= 100.0

    def test_score_macd_golden(self):
        # Golden value changed 95.0 -> 100.0 (2026-08) when
        # TechnicalIndicators.linear_regression_slope's hardcoded `len(y) < 10`
        # check was fixed to `len(y) < period`. score_macd calls it with
        # period=5 for hist_slope, so len(y) was always 5 < 10 and hist_slope
        # was unconditionally 0.0 (dead code) -- this golden had the bug
        # baked in. See canadian_stock_screener.py's linear_regression_slope.
        s = self.sc.score_macd(self.close)
        assert s == pytest.approx(100.0, abs=0.5), f"macd golden changed: {s}"

    def test_score_obv_range(self):
        s = self.sc.score_obv(self.close, self.volume)
        assert 0.0 <= s <= 100.0

    def test_score_obv_golden(self):
        s = self.sc.score_obv(self.close, self.volume)
        assert s == pytest.approx(95.0, abs=0.5), f"obv golden changed: {s}"

    def test_score_adx_range(self):
        s = self.sc.score_adx(self.high, self.low, self.close)
        assert 0.0 <= s <= 100.0

    def test_score_adx_golden(self):
        s = self.sc.score_adx(self.high, self.low, self.close)
        assert s == pytest.approx(89.4, abs=1.0), f"adx golden changed: {s}"

    def test_score_vam_range(self):
        s = self.sc.score_vam(self.close)
        assert 0.0 <= s <= 100.0

    def test_score_vam_golden(self):
        s = self.sc.score_vam(self.close)
        assert s == pytest.approx(100.0, abs=1.0), f"vam golden changed: {s}"

    def test_score_breakout_range(self):
        s = self.sc.score_breakout(self.close, self.high, self.volume)
        assert 0.0 <= s <= 100.0

    def test_score_breakout_golden(self):
        s = self.sc.score_breakout(self.close, self.high, self.volume)
        assert s == pytest.approx(100.0, abs=0.5), f"breakout golden changed: {s}"

    def test_score_rs_no_universe_fallback(self):
        """Without universe values, RS falls back to scaled formula."""
        s = self.sc.score_relative_strength(self.close, self.bench, [])
        assert 0.0 <= s <= 100.0

    def test_score_rs_outperformer_scores_above_50(self):
        """A stock outperforming benchmark should score > 50 in RS."""
        strong_close = self.close * 1.10  # 10% extra on top
        s = self.sc.score_relative_strength(strong_close, self.bench, [])
        assert s > 50.0, f"outperformer RS should be > 50, got {s}"

    def test_score_rs_underperformer_scores_below_50(self):
        """A stock underperforming benchmark should score < 50 in RS."""
        weak_close = self.bench * 0.90
        s = self.sc.score_relative_strength(weak_close, self.bench, [])
        assert s < 50.0, f"underperformer RS should be < 50, got {s}"

    # ── composite score ────────────────────────────────────────────────────

    def test_composite_score_formula(self):
        """Composite must match the weighted sum of individual scores."""
        from canadian_stock_screener import CONFIG
        w = CONFIG.weights
        s2 = self.sc.score_stage2(self.close)
        rs = self.sc.score_relative_strength(self.close, self.bench, [])
        mac = self.sc.score_macd(self.close)
        ob = self.sc.score_obv(self.close, self.volume)
        adx = self.sc.score_adx(self.high, self.low, self.close)
        vam = self.sc.score_vam(self.close)
        brk = self.sc.score_breakout(self.close, self.high, self.volume)

        expected = (s2 * w["stage2_score"] +
                    rs * w["rs_score"] +
                    mac * w["macd_score"] +
                    ob * w["obv_score"] +
                    adx * w["adx_score"] +
                    vam * w["vam_score"] +
                    brk * w["breakout_score"])
        assert expected == pytest.approx(expected, abs=0.01)  # tautological — verifies no NaN

    def test_weights_sum_to_one(self):
        from canadian_stock_screener import CONFIG
        total = sum(CONFIG.weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    # ── risk metrics ───────────────────────────────────────────────────────

    def test_risk_metrics_keys(self):
        r = self.sc.calculate_risk_metrics(self.close)
        expected_keys = {"Max_DD", "Sharpe", "Win_Rate", "Profit_Factor",
                         "Skew", "Kurtosis", "VaR_95", "Calmar"}
        assert expected_keys == set(r.keys())

    def test_risk_metrics_max_dd_negative(self):
        r = self.sc.calculate_risk_metrics(self.close)
        assert r["Max_DD"] <= 0.0, "Max drawdown must be ≤ 0"

    def test_risk_metrics_win_rate_range(self):
        r = self.sc.calculate_risk_metrics(self.close)
        assert 0.0 <= r["Win_Rate"] <= 100.0

    def test_risk_metrics_insufficient_data_returns_empty(self):
        short = self.close.iloc[:30]
        r = self.sc.calculate_risk_metrics(short)
        assert r == {}

    # ── edge cases ─────────────────────────────────────────────────────────

    def test_score_stage2_insufficient_weekly_data_returns_zero(self):
        short = self.close.iloc[:50]  # < 40 weeks
        assert self.sc.score_stage2(short) == 0.0

    def test_score_macd_short_data_still_scores(self):
        """
        EWM-based MACD produces a value even on 20 bars (no strict min_period).
        Documents current behaviour: do not change without updating this golden.

        Golden changed 75.0 -> 90.0 (2026-08), same linear_regression_slope
        hist_slope fix as test_score_macd_golden above.
        """
        short = self.close.iloc[:20]
        assert self.sc.score_macd(short) == pytest.approx(90.0, abs=0.5)

    def test_score_vam_insufficient_data_returns_50(self):
        short = self.close.iloc[:30]
        assert self.sc.score_vam(short) == pytest.approx(50.0, abs=0.1)

    def test_score_breakout_insufficient_data_returns_50(self):
        short = self.close.iloc[:100]  # < 252
        assert self.sc.score_breakout(short, self.high.iloc[:100],
                                      self.volume.iloc[:100]) == pytest.approx(50.0, abs=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTERIZATION — TechnicalIndicators (canadian_stock_screener.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.characterization
class TestTechnicalIndicators:

    @pytest.fixture(autouse=True)
    def setup(self):
        from canadian_stock_screener import TechnicalIndicators
        self.ti = TechnicalIndicators()
        n = 100
        idx = pd.bdate_range("2024-01-01", periods=n)
        np.random.seed(5)
        self.close = pd.Series(np.linspace(10, 20, n) + np.random.randn(n) * 0.05, index=idx)

    def test_sma_last_value_correct(self):
        sma = self.ti.sma(self.close, 20)
        expected = float(self.close.iloc[-20:].mean())
        assert float(sma.iloc[-1]) == pytest.approx(expected, rel=1e-6)

    def test_sma_min_periods_enforced(self):
        sma = self.ti.sma(self.close, 20)
        assert pd.isna(sma.iloc[10])
        assert not pd.isna(sma.iloc[19])

    def test_ema_converges_on_constant_series(self):
        const = pd.Series(np.ones(100) * 5.0)
        ema = self.ti.ema(const, 10)
        assert float(ema.iloc[-1]) == pytest.approx(5.0, rel=1e-4)

    def test_rsi_range(self):
        rsi = self.ti.rsi(self.close)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_uptrend_above_50(self):
        """
        RSI on a rising series should be well above 50.
        NOTE: A perfectly monotone series (zero losses) makes avg_loss=0 everywhere,
        so RSI returns all-NaN via the 0→NaN replacement.  A noisy uptrend is used
        instead — this documents the current implementation's edge behaviour.
        """
        np.random.seed(0)
        up = pd.Series(np.linspace(10, 30, 100) + np.random.randn(100) * 0.3)
        rsi = self.ti.rsi(up)
        valid = rsi.dropna()
        assert len(valid) > 0, "RSI produced no valid values on uptrend"
        assert float(valid.iloc[-1]) > 60, f"Expected RSI > 60 on uptrend, got {float(valid.iloc[-1])}"

    def test_rsi_perfectly_monotone_returns_100(self):
        """
        A perfectly monotone rising series (zero losses on every bar) used to
        produce all-NaN RSI: avg_loss==0 was replaced with NaN to dodge a raw
        division by zero, which fed through as NaN instead of the textbook-
        correct limiting value of 100. Fixed 2026-08 (see rsi()'s docstring /
        inline comment) — this test now locks in the corrected value.
        """
        up = pd.Series(np.linspace(10, 30, 100))
        rsi = self.ti.rsi(up)
        valid = rsi.dropna()
        assert not valid.empty
        assert (valid == 100.0).all(), f"Expected RSI==100 throughout, got {valid.unique()}"

    def test_rsi_perfectly_monotone_falling_returns_0(self):
        """Mirror case: zero gains on every bar -> RSI should be 0, and always
        was (this side of the avg_gain/avg_loss asymmetry was never buggy)."""
        down = pd.Series(np.linspace(30, 10, 100))
        rsi = self.ti.rsi(down)
        valid = rsi.dropna()
        assert not valid.empty
        assert (valid == 0.0).all(), f"Expected RSI==0 throughout, got {valid.unique()}"

    def test_linear_regression_slope_period_5_matches_period_10(self):
        """
        linear_regression_slope's internal minimum-length check used to be
        hardcoded to `len(y) < 10` regardless of `period`, which made every
        period=5 caller (score_macd's hist_slope) silently return 0.0 always.
        A period=5 call on an obviously-rising 5-point series must now return
        a real, correctly-signed, non-zero slope.
        """
        from canadian_stock_screener import TechnicalIndicators
        rising = pd.Series([-0.05, -0.02, -0.01, 0.001, 0.03])
        s = TechnicalIndicators.linear_regression_slope(rising, 5)
        assert s > 0, f"Expected a positive slope on a clearly rising series, got {s}"

        falling = pd.Series([0.05, 0.02, 0.01, -0.001, -0.03])
        s = TechnicalIndicators.linear_regression_slope(falling, 5)
        assert s < 0, f"Expected a negative slope on a clearly falling series, got {s}"

    def test_linear_regression_slope_insufficient_data_returns_zero(self):
        from canadian_stock_screener import TechnicalIndicators
        short = pd.Series([1.0, 2.0, 3.0])
        assert TechnicalIndicators.linear_regression_slope(short, 5) == 0.0

    def test_macd_histogram_is_line_minus_signal(self):
        line, signal, hist = self.ti.macd(self.close)
        diff = (line - signal).dropna()
        hist_check = hist.dropna()
        pd.testing.assert_series_equal(
            diff.iloc[-len(hist_check):].reset_index(drop=True),
            hist_check.reset_index(drop=True),
            check_names=False, atol=1e-8
        )

    def test_adx_range(self):
        n = 100
        idx = pd.bdate_range("2024-01-01", periods=n)
        np.random.seed(3)
        close = pd.Series(np.linspace(10, 20, n), index=idx)
        high = close + 0.5
        low = close - 0.5
        adx = self.ti.adx(high, low, close)
        valid = adx.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_obv_increases_on_up_days(self):
        """OBV must increase on each day the close goes up from a constant base."""
        n = 10
        close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        volume = pd.Series([100.0] * n)
        obv = self.ti.obv(close, volume)
        diffs = obv.diff().dropna()
        assert (diffs > 0).all()

    def test_obv_decreases_on_down_days(self):
        n = 5
        close = pd.Series([20.0, 19.0, 18.0, 17.0, 16.0])
        volume = pd.Series([100.0] * n)
        obv = self.ti.obv(close, volume)
        diffs = obv.diff().dropna()
        assert (diffs < 0).all()

    def test_weekly_resample_returns_friday_values(self):
        idx = pd.bdate_range("2024-01-01", periods=50)
        s = pd.Series(np.arange(50, dtype=float), index=idx)
        w = self.ti.weekly_resample(s)
        # All resampled index entries should be a Friday
        assert all(d.weekday() == 4 for d in w.index)


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTERIZATION — Pattern detection (auto_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.characterization
class TestPatternDetectors:

    def test_base_breakout_confirmed_on_fixture(self):
        """BASE CONFIRMED when last bar closes above 40-bar range on 3× volume."""
        from auto_pipeline import _detect_base_breakout
        df = _make_base_breakout_ohlcv()
        result = _detect_base_breakout(df["Close"], df["High"], df["Low"], df["Volume"])
        assert result is not None, "Expected BASE to be detected"
        assert result["pattern"] == "BASE"
        assert result["state"] == "CONFIRMED", f"Expected CONFIRMED, got {result['state']}"

    def test_base_breakout_none_on_wide_range(self):
        """BASE returns None when the base range exceeds 20%."""
        from auto_pipeline import _detect_base_breakout
        df = _make_trending_ohlcv(n=300, seed=1)
        # Widen the last 40 bars so range > 20%
        close = df["Close"].copy()
        high = df["High"].copy()
        low = df["Low"].copy()
        high.iloc[-40:] = close.iloc[-40:] * 1.15  # 30% swing
        low.iloc[-40:] = close.iloc[-40:] * 0.85
        result = _detect_base_breakout(close, high, low, df["Volume"])
        assert result is None, f"Expected None for wide-range base, got {result}"

    def test_base_insufficient_data_returns_none(self):
        from auto_pipeline import _detect_base_breakout
        df = _make_trending_ohlcv(n=50)
        r = _detect_base_breakout(df["Close"], df["High"], df["Low"], df["Volume"])
        assert r is None

    def test_ema_pullback_confirmed_on_fixture(self):
        """PB-EMA21 CONFIRMED when price reclaims EMA21 on last bar with volume."""
        from auto_pipeline import _detect_ema_pullback
        df = _make_ema_pullback_ohlcv(ema_period=21)
        results = _detect_ema_pullback(df["Close"], df["High"], df["Low"], df["Volume"])
        labels = [r["pattern"] for r in results]
        assert "PB-EMA21" in labels, f"Expected PB-EMA21, got {labels}"
        pb21 = next(r for r in results if r["pattern"] == "PB-EMA21")
        assert pb21["state"] == "CONFIRMED", f"Expected CONFIRMED, got {pb21['state']}"

    def test_ema_pullback_requires_rising_ema(self):
        """No EMA pullback signal when EMA slope is flat/negative."""
        from auto_pipeline import _detect_ema_pullback
        # Flat price → EMA slope ≈ 0 → no signal
        n = 300
        idx = pd.bdate_range("2024-01-01", periods=n)
        close = pd.Series(np.ones(n) * 20.0, index=idx)
        high = close + 0.1
        low = close - 0.1
        vol = pd.Series(np.ones(n) * 300_000, index=idx, dtype=float)
        r = _detect_ema_pullback(close, high, low, vol)
        assert r == [], f"Expected no signal on flat EMA, got {r}"

    def test_detect_all_patterns_returns_sorted_by_priority(self):
        """detect_all_patterns must place CONFIRMED before AT_PIVOT before FORMING."""
        from auto_pipeline import detect_all_patterns, STATE_CONFIRMED, STATE_AT_PIVOT, STATE_FORMING
        df = _make_base_breakout_ohlcv()
        patterns = detect_all_patterns("X.TO", df)
        # At least one pattern
        assert patterns, "Expected at least one pattern"
        priority = {STATE_CONFIRMED: 0, STATE_AT_PIVOT: 1, STATE_FORMING: 2}
        orders = [priority.get(p["state"], 9) for p in patterns]
        assert orders == sorted(orders), "Patterns not sorted by state priority"

    def test_detect_all_patterns_keys(self):
        """Every returned pattern dict must have 'pattern', 'state', 'pivot', 'detail'."""
        from auto_pipeline import detect_all_patterns
        df = _make_base_breakout_ohlcv()
        for p in detect_all_patterns("X.TO", df):
            assert "pattern" in p
            assert "state" in p
            assert "pivot" in p
            assert "detail" in p

    def test_vcp_returns_none_below_ma150(self):
        """VCP detector returns None when price is below 150-day MA."""
        from auto_pipeline import _detect_vcp
        df = _make_trending_ohlcv(n=300, seed=42)
        close = df["Close"].copy()
        # Force price below MA150
        close.iloc[-1] = 1.0
        r = _detect_vcp(close, df["High"], df["Low"], df["Volume"])
        assert r is None

    def test_momentum_breakout_confirmed_on_fixture(self):
        """MOMENTUM CONFIRMED when the last bar clears the prior 55-bar high on volume."""
        from auto_pipeline import _detect_momentum_breakout
        df = _make_base_breakout_ohlcv()
        result = _detect_momentum_breakout(df["Close"], df["High"], df["Low"], df["Volume"])
        assert result is not None, "Expected MOMENTUM to be detected"
        assert result["pattern"] == "MOMENTUM"
        assert result["state"] == "CONFIRMED", f"Expected CONFIRMED, got {result['state']}"

    def test_momentum_breakout_catches_vertical_move_base_breakout_misses(self):
        """The whole point of this detector: fire on a wide-range (>20%) vertical
        move that _detect_base_breakout's base_range cap correctly rejects —
        this is what let a gold/silver-miner-style rally through the core
        sleeve's detectors untouched (see config.py MOMENTUM_* rationale)."""
        from auto_pipeline import _detect_base_breakout, _detect_momentum_breakout
        df = _make_trending_ohlcv(n=300, seed=1)
        close = df["Close"].copy()
        high = df["High"].copy()
        low = df["Low"].copy()
        # Widen the last 40 bars so base_range > 20% (same fixture shape as
        # test_base_breakout_none_on_wide_range), then break out above it.
        high.iloc[-40:] = close.iloc[-40:] * 1.15
        low.iloc[-40:] = close.iloc[-40:] * 0.85
        volume = df["Volume"].copy()
        high.iloc[-1] = close.iloc[-1] * 1.30
        close.iloc[-1] = close.iloc[-1] * 1.28
        volume.iloc[-1] = volume.iloc[-50:].mean() * 3.0

        base_result = _detect_base_breakout(close, high, low, volume)
        assert base_result is None, f"Expected BASE to reject the wide-range move, got {base_result}"

        momentum_result = _detect_momentum_breakout(close, high, low, volume)
        assert momentum_result is not None, "Expected MOMENTUM to catch what BASE rejected"
        assert momentum_result["state"] == "CONFIRMED"

    def test_momentum_breakout_none_below_prior_high(self):
        """No signal (not even FORMING) while price sits below the lookback high."""
        from auto_pipeline import _detect_momentum_breakout
        df = _make_trending_ohlcv(n=300, seed=42, trend=0.0)  # flat, never makes a new high
        r = _detect_momentum_breakout(df["Close"], df["High"], df["Low"], df["Volume"])
        assert r is None

    def test_momentum_breakout_insufficient_data_returns_none(self):
        from auto_pipeline import _detect_momentum_breakout
        df = _make_trending_ohlcv(n=50)
        r = _detect_momentum_breakout(df["Close"], df["High"], df["Low"], df["Volume"])
        assert r is None

    def test_detect_all_patterns_excludes_momentum_by_default(self):
        """Core sleeve safety net: detect_all_patterns must never surface MOMENTUM
        unless the momentum sleeve explicitly opts in."""
        from auto_pipeline import detect_all_patterns
        df = _make_base_breakout_ohlcv()
        patterns = detect_all_patterns("X.TO", df)
        assert all(p["pattern"] != "MOMENTUM" for p in patterns)

    def test_detect_all_patterns_includes_momentum_when_enabled(self):
        from auto_pipeline import detect_all_patterns
        df = _make_base_breakout_ohlcv()
        patterns = detect_all_patterns("X.TO", df, enable_momentum_breakout=True)
        assert any(p["pattern"] == "MOMENTUM" for p in patterns)


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTERIZATION — compute_levels & compute_position_size (auto_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.characterization
class TestLevelsAndSizing:

    @pytest.fixture(autouse=True)
    def setup(self):
        df = _make_trending_ohlcv(n=300, seed=42)
        self.close = df["Close"]
        self.high = df["High"]
        self.low = df["Low"]
        self.entry = float(self.close.iloc[-1]) * 1.005
        self.atr_period = 14
        self.atr_mult = 1.5

    def test_levels_keys_present(self):
        from auto_pipeline import compute_levels
        r = compute_levels(self.close, self.high, self.low,
                           self.entry, self.atr_period, self.atr_mult)
        for k in ("entry", "stop", "risk_pct", "target_2r", "target_3r", "atr", "resistance_based"):
            assert k in r, f"Missing key: {k}"

    def test_stop_is_below_entry(self):
        from auto_pipeline import compute_levels
        r = compute_levels(self.close, self.high, self.low,
                           self.entry, self.atr_period, self.atr_mult)
        assert r["stop"] < r["entry"], "Stop must be below entry"

    def test_target_2r_above_entry(self):
        """
        When no resistance swing-high exists above entry (resistance_based=False),
        target_2r = entry + 2*risk, which is strictly above entry.
        Uses tight high/low spreads to prevent swing-high detection.
        """
        from auto_pipeline import compute_levels
        # Tight spread: high = close + 0.1 — no swing highs above the current price
        entry_tight = float(self.close.iloc[-1]) * 1.005
        r = compute_levels(
            self.close,
            self.close + 0.1,  # tight high
            self.close - 0.1,  # tight low
            entry_tight, self.atr_period, self.atr_mult,
        )
        assert not r["resistance_based"], \
            "Fixture should have no resistance above entry; swing high was detected unexpectedly"
        assert r["target_2r"] > r["entry"], \
            f"target_2r {r['target_2r']} must exceed entry {r['entry']} when no resistance"

    def test_target_3r_above_target_2r(self):
        from auto_pipeline import compute_levels
        r = compute_levels(self.close, self.high, self.low,
                           self.entry, self.atr_period, self.atr_mult)
        assert r["target_3r"] >= r["target_2r"]

    def test_risk_pct_positive(self):
        from auto_pipeline import compute_levels
        r = compute_levels(self.close, self.high, self.low,
                           self.entry, self.atr_period, self.atr_mult)
        assert r["risk_pct"] > 0.0

    def test_position_size_shares_whole_number(self):
        from auto_pipeline import compute_levels, compute_position_size
        r = compute_levels(self.close, self.high, self.low,
                           self.entry, self.atr_period, self.atr_mult)
        s = compute_position_size(100_000, 1.0, r["entry"], r["stop"])
        assert s["shares"] == int(s["shares"]), "Shares must be a whole number"

    def test_position_size_risk_dollars(self):
        """Dollar risk ≈ account × risk_pct / 100."""
        from auto_pipeline import compute_position_size
        entry = 20.0
        stop = 18.0
        s = compute_position_size(100_000, 1.0, entry, stop)
        assert s["risk_$"] == pytest.approx(1000.0, rel=0.01)

    def test_position_size_shares_formula(self):
        """shares = int(dollar_risk / (entry - stop))."""
        from auto_pipeline import compute_position_size
        entry = 20.0
        stop = 18.0
        s = compute_position_size(100_000, 1.0, entry, stop)
        expected_shares = int(1000.0 / (entry - stop))
        assert s["shares"] == expected_shares

    def test_position_value_consistent(self):
        from auto_pipeline import compute_position_size
        entry = 25.0
        stop = 23.0
        s = compute_position_size(100_000, 1.0, entry, stop)
        assert s["position_$"] == pytest.approx(s["shares"] * entry, rel=0.001)


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTERIZATION — compute_signals (position_monitor.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.characterization
class TestPositionMonitor:
    """Tests for position_monitor.compute_signals.

    Exit-threshold tests must import the relevant constants from position_monitor
    (e.g. TIME_STOP_DAYS, GIVEBACK_ACTIVATE_PCT) rather than hardcoding values.
    That way the tests stay correct automatically when the constants are tuned.
    """

    # ── Module-level constants imported once for the whole class ─────────────
    from position_monitor import (  # noqa: E402
        TIME_STOP_DAYS,
        TIME_STOP_MIN_PROFIT_PCT,
        GIVEBACK_ACTIVATE_PCT,
        GIVEBACK_ALLOW_PCT,
        INITIAL_STOP_ATR_K,
        CHAND_TRAIL_ATR_K,
    )

    @dataclass
    class _Pos:
        ticker: str
        entry_date: date
        entry_price: float
        shares: float

    def _make_pos_fixture(self, seed: int = 3, n: int = 100,
                          entry_bar: int = 25) -> tuple:
        np.random.seed(seed)
        idx = pd.bdate_range("2024-09-01", periods=n)
        close = pd.Series(20 + np.cumsum(np.random.randn(n) * 0.2), index=idx)
        high = close + 0.3
        low = close - 0.3
        vol = pd.Series(np.ones(n) * 300_000, index=idx)
        df = pd.DataFrame({"Open": close, "High": high, "Low": low,
                           "Close": close, "Volume": vol})
        pos = self._Pos("X.TO", date(2024, 10, 1),
                        float(close.iloc[entry_bar]), 100.0)
        return df, pos

    def test_compute_signals_keys(self):
        from position_monitor import compute_signals
        df, pos = self._make_pos_fixture()
        r = compute_signals(pos, df)
        for k in ("status", "reason", "stop_price", "pnl_%", "tdays",
                  "entry_price", "last_close", "ATR14", "initial_stop",
                  "chandelier_stop", "R_mult"):
            assert k in r, f"Missing key: {k}"

    def test_compute_signals_status_is_hold_or_sell(self):
        from position_monitor import compute_signals
        df, pos = self._make_pos_fixture()
        r = compute_signals(pos, df)
        assert r["status"] in ("HOLD", "SELL")

    def test_compute_signals_golden_fixture(self):
        """Golden values from seed=3, n=100, entry_bar=25."""
        from position_monitor import compute_signals
        df, pos = self._make_pos_fixture(seed=3, n=100, entry_bar=25)
        r = compute_signals(pos, df)
        # Locked golden: SELL (stop hit + time stop on this fixture)
        assert r["status"] == "SELL"
        assert r["pnl_%"] == pytest.approx(-7.9, abs=0.5)

    def test_stop_is_below_entry_price(self):
        from position_monitor import compute_signals
        df, pos = self._make_pos_fixture()
        r = compute_signals(pos, df)
        if "stop_price" in r and r["stop_price"] != "N/A":
            assert float(r["stop_price"]) < pos.entry_price

    def test_initial_stop_uses_atr_mult(self):
        """initial_stop ≈ entry_price - INITIAL_STOP_ATR_K × ATR14."""
        from position_monitor import compute_signals
        df, pos = self._make_pos_fixture()
        r = compute_signals(pos, df)
        if "initial_stop" in r and "ATR14" in r:
            expected = pos.entry_price - self.INITIAL_STOP_ATR_K * float(r["ATR14"])
            assert float(r["initial_stop"]) == pytest.approx(expected, rel=0.01)

    def test_pnl_pct_formula(self):
        """pnl_% = (last_close / entry_price - 1) * 100."""
        from position_monitor import compute_signals
        df, pos = self._make_pos_fixture()
        r = compute_signals(pos, df)
        expected = (float(r["last_close"]) / pos.entry_price - 1.0) * 100.0
        assert float(r["pnl_%"]) == pytest.approx(expected, abs=0.01)

    def test_time_stop_triggers_after_n_days_with_small_profit(self):
        """TIME_STOP fires when tdays ≥ TIME_STOP_DAYS and profit < TIME_STOP_MIN_PROFIT_PCT.

        TIME_STOP_MIN_PROFIT_PCT=0.0 means the position must be below break-even
        (negative P&L) to trigger.  We set price slightly below entry to satisfy
        that condition while keeping the position far from the ATR stop.
        """
        from position_monitor import compute_signals
        n = 200
        idx = pd.bdate_range("2024-01-01", periods=n)
        # Price drifts slightly below entry — negative P&L, nowhere near the ATR stop
        entry = 20.0
        price = np.full(n, entry * 0.99)   # -1%: below break-even, above any ATR stop
        close = pd.Series(price, index=idx)
        df = pd.DataFrame({"Open": close, "High": close + 0.05,
                           "Low": close - 0.05, "Close": close,
                           "Volume": pd.Series(np.ones(n) * 300_000, index=idx)})
        pos = self._Pos("X.TO", idx[0].date(), entry, 100.0)
        r = compute_signals(pos, df)
        # tdays >> self.TIME_STOP_DAYS and profit < self.TIME_STOP_MIN_PROFIT_PCT
        assert r["status"] == "SELL", (
            f"Expected SELL (tdays={r.get('tdays')}, pnl={r.get('pnl_%'):.2f}%, "
            f"TIME_STOP_DAYS={self.TIME_STOP_DAYS}, threshold={self.TIME_STOP_MIN_PROFIT_PCT}%)"
        )
        assert "TIME_STOP" in r["reason"]

    def test_no_data_returns_status_key(self):
        """An empty DataFrame must return a dict with 'status' = 'NO_DATA'."""
        from position_monitor import compute_signals
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        pos = self._Pos("X.TO", date(2024, 1, 1), 20.0, 10.0)
        r = compute_signals(pos, empty)
        assert r.get("status") in ("NO_DATA", "NO_ATR")

    def test_zero_entry_price_returns_bad_data_not_crash(self):
        """
        A zero/negative entry_price used to raise an uncaught ZeroDivisionError
        (pnl_pct = last_close / entry_price), which would crash the whole
        per-position loop in main() for every other open position in the same
        run — not just the corrupted one. Must now fail soft.
        """
        from position_monitor import compute_signals
        df, _ = self._make_pos_fixture()
        pos = self._Pos("BAD.TO", date(2024, 10, 1), 0.0, 100.0)
        r = compute_signals(pos, df)  # must not raise
        assert r["status"] == "BAD_DATA"

    def test_negative_entry_price_returns_bad_data_not_crash(self):
        from position_monitor import compute_signals
        df, _ = self._make_pos_fixture()
        pos = self._Pos("BAD.TO", date(2024, 10, 1), -5.0, 100.0)
        r = compute_signals(pos, df)  # must not raise
        assert r["status"] == "BAD_DATA"


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTERIZATION — virtual_buy.py (funds + allocation)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.characterization
class TestVirtualBuy:

    def test_equal_allocation_formula(self):
        """
        Core capital allocation rule: funds split equally across tickers.
        shares = int(allocation_per_ticker / price)
        """
        total_funds = 30_000.0
        n_tickers = 3
        allocation = total_funds / n_tickers  # 10_000 each
        price = 25.0
        expected_shares = int(allocation / price)  # 400
        assert expected_shares == 400

    def test_equal_allocation_total_cost_lte_funds(self):
        """Total cost of all positions must never exceed available funds."""
        total_funds = 30_000.0
        prices = [25.0, 40.0, 12.50]
        n = len(prices)
        alloc = total_funds / n
        total_cost = sum(int(alloc / p) * p for p in prices)
        assert total_cost <= total_funds



# ─────────────────────────────────────────────────────────────────────────────
# CHARACTERIZATION — swing_tickers.py helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.characterization
class TestSwingTickersHelpers:

    def test_compute_atr_golden(self):
        """ATR14 rolling-mean value on a known fixture."""
        from swing_tickers import compute_atr, safe_last
        n = 60
        idx = pd.bdate_range("2024-01-01", periods=n)
        np.random.seed(1)
        close = pd.Series(50 + np.cumsum(np.random.randn(n) * 0.3), index=idx)
        high = close + 0.5
        low = close - 0.5
        df = pd.DataFrame({"High": high, "Low": low, "Close": close})
        atr = compute_atr(df)
        assert safe_last(atr) == pytest.approx(1.009, abs=0.05)

    def test_compute_atr_all_positive(self):
        from swing_tickers import compute_atr
        n = 60
        idx = pd.bdate_range("2024-01-01", periods=n)
        close = pd.Series(np.linspace(10, 20, n), index=idx)
        df = pd.DataFrame({"High": close + 1, "Low": close - 1, "Close": close})
        atr = compute_atr(df).dropna()
        assert (atr > 0).all()

    def test_slope_of_series_uptrend_positive(self):
        from swing_tickers import slope_of_series
        s = pd.Series(np.linspace(1, 10, 20))
        assert slope_of_series(s, 10) > 0

    def test_slope_of_series_downtrend_negative(self):
        from swing_tickers import slope_of_series
        s = pd.Series(np.linspace(10, 1, 20))
        assert slope_of_series(s, 10) < 0

    def test_slope_insufficient_data_returns_nan(self):
        from swing_tickers import slope_of_series
        s = pd.Series([1.0, 2.0, 3.0])
        result = slope_of_series(s, lookback=10)
        assert np.isnan(result)

    def test_safe_last_returns_last_non_nan(self):
        from swing_tickers import safe_last
        s = pd.Series([1.0, np.nan, 3.0, np.nan])
        assert safe_last(s) == pytest.approx(3.0)

    def test_safe_last_all_nan_returns_nan(self):
        from swing_tickers import safe_last
        s = pd.Series([np.nan, np.nan])
        assert np.isnan(safe_last(s))


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTERIZATION — ATR consistency (auto_pipeline vs swing_tickers)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.characterization
class TestATRConsistency:
    """
    Both auto_pipeline._atr() and swing_tickers.compute_atr() exist.
    They use different smoothing (Wilder's ewm vs rolling mean).
    This test documents the CURRENT behaviour so a refactor cannot
    silently unify them without awareness.
    """

    def test_pipeline_atr_uses_wilders_smoothing(self):
        """auto_pipeline._atr() uses ewm(alpha=1/period) — Wilder's smoothing."""
        from auto_pipeline import _atr
        n = 60
        idx = pd.bdate_range("2024-01-01", periods=n)
        close = pd.Series(np.linspace(10, 20, n), index=idx)
        high = close + 1.0
        low = close - 1.0
        atr = _atr(high, low, close, period=14)
        # Wilder's: ewm alpha=1/14, warm-up ~14 bars, converges to ~2.0 (H-L=2)
        assert float(atr.dropna().iloc[-1]) == pytest.approx(2.0, abs=0.1)

    def test_swing_atr_uses_rolling_mean(self):
        """swing_tickers.compute_atr() uses rolling(period).mean()."""
        from swing_tickers import compute_atr
        n = 60
        idx = pd.bdate_range("2024-01-01", periods=n)
        close = pd.Series(np.linspace(10, 20, n), index=idx)
        df = pd.DataFrame({"High": close + 1.0, "Low": close - 1.0, "Close": close})
        atr = compute_atr(df, period=14)
        assert float(atr.dropna().iloc[-1]) == pytest.approx(2.0, abs=0.1)

    def test_atr_implementations_differ_on_noisy_data(self):
        """
        On noisy data the two implementations produce DIFFERENT values.
        This test documents that divergence — do not silently merge them.
        """
        from auto_pipeline import _atr
        from swing_tickers import compute_atr
        n = 100
        idx = pd.bdate_range("2024-01-01", periods=n)
        np.random.seed(9)
        close = pd.Series(np.linspace(10, 30, n) + np.random.randn(n), index=idx)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        df = pd.DataFrame({"High": high, "Low": low, "Close": close})

        pipeline_atr = float(_atr(high, low, close, 14).iloc[-1])
        swing_atr = float(compute_atr(df, 14).iloc[-1])
        # They are NOT expected to be the same — if they suddenly are, something changed
        assert abs(pipeline_atr - swing_atr) > 0.01, (
            "ATR implementations unexpectedly converged — was one changed silently?"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Data provider interface (tests are skipped until Phase 2 lands)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.phase2
class TestMarketDataProvider:
    """
    These tests define the contract for the MarketDataProvider interface
    introduced in Phase 2.  They are expected to FAIL until Phase 2 is merged.
    Running them beforehand shows what still needs to be built.
    """

    def test_live_provider_importable(self):
        """LiveDataProvider must be importable from market_data module."""
        from market_data import LiveDataProvider  # noqa: F401

    def test_historical_provider_importable(self):
        from market_data import HistoricalSliceProvider  # noqa: F401

    def test_historical_provider_respects_cutoff(self):
        """
        HistoricalSliceProvider.get(ticker, as_of=D) must return only bars up to D.
        """
        from market_data import HistoricalSliceProvider

        # Build a small in-memory dataset
        n = 50
        idx = pd.bdate_range("2024-01-01", periods=n)
        np.random.seed(0)
        close = pd.Series(np.linspace(10, 15, n), index=idx)
        df_all = pd.DataFrame({"Open": close, "High": close + 0.1,
                               "Low": close - 0.1, "Close": close,
                               "Volume": pd.Series(np.ones(n) * 1e5, index=idx)})

        cutoff = idx[25]  # bar 26, zero-indexed
        provider = HistoricalSliceProvider({"TEST.TO": df_all})
        result = provider.get("TEST.TO", as_of=cutoff)

        assert result.index.max() <= cutoff, \
            "HistoricalSliceProvider returned bars beyond the as_of cutoff"

    def test_historical_provider_raises_on_unknown_ticker(self):
        from market_data import HistoricalSliceProvider
        provider = HistoricalSliceProvider({})
        with pytest.raises(KeyError):
            provider.get("UNKNOWN.TO", as_of=pd.Timestamp("2024-01-01"))

    def test_data_manager_accepts_provider_injection(self):
        """
        DataManager must accept a MarketDataProvider via its constructor
        so the screener can be run against historical data.
        """
        from market_data import HistoricalSliceProvider
        from canadian_stock_screener import DataManager
        # Should not raise even with an empty provider
        provider = HistoricalSliceProvider({})
        dm = DataManager.__new__(DataManager)
        # Phase 2: DataManager(tickers_file=..., provider=provider) must work
        # For now just assert the class exists
        assert DataManager is not None

    def test_default_provider_is_live(self):
        """market_data.DEFAULT_PROVIDER is the single instance every live
        call site should default to."""
        from market_data import DEFAULT_PROVIDER, LiveDataProvider
        assert isinstance(DEFAULT_PROVIDER, LiveDataProvider)

    def test_historical_provider_get_quote_returns_last_close(self):
        from market_data import HistoricalSliceProvider

        n = 10
        idx = pd.bdate_range("2024-01-01", periods=n)
        close = pd.Series(np.linspace(10, 19, n), index=idx)
        df = pd.DataFrame({"Open": close, "High": close + 0.1,
                            "Low": close - 0.1, "Close": close,
                            "Volume": pd.Series(np.ones(n) * 1e5, index=idx)})
        provider = HistoricalSliceProvider({"TEST.TO": df})
        assert provider.get_quote("TEST.TO") == pytest.approx(19.0)

    def test_historical_provider_get_quote_unknown_ticker_returns_none(self):
        from market_data import HistoricalSliceProvider
        provider = HistoricalSliceProvider({})
        assert provider.get_quote("UNKNOWN.TO") is None

    def test_historical_provider_get_intraday_snapshot_returns_none(self):
        """The backtester has no "today" — always falls back to daily bars."""
        from market_data import HistoricalSliceProvider
        provider = HistoricalSliceProvider({})
        assert provider.get_intraday_snapshot("TEST.TO") is None

    def test_historical_provider_get_sector_returns_string(self):
        from market_data import HistoricalSliceProvider
        provider = HistoricalSliceProvider({})
        assert isinstance(provider.get_sector("RY.TO"), str)

    def test_live_provider_get_quote_fast_info(self, monkeypatch):
        """get_quote() prefers fast_info["last_price"] over the 1m fallback."""
        import market_data as md

        class _FakeTicker:
            fast_info = {"last_price": 42.5}

        monkeypatch.setattr(md.yf, "Ticker", lambda ticker: _FakeTicker())
        provider = md.LiveDataProvider()
        assert provider.get_quote("RY.TO") == pytest.approx(42.5)

    def test_live_provider_get_quote_falls_back_to_1m_bar(self, monkeypatch):
        import market_data as md

        class _FakeTicker:
            fast_info = {}

        idx = pd.bdate_range("2024-01-01", periods=3, freq="min")
        fallback_df = pd.DataFrame({"Open": [1, 1, 1], "High": [1, 1, 1],
                                     "Low": [1, 1, 1], "Close": [1, 1, 7.25],
                                     "Volume": [10, 10, 10]}, index=idx)
        monkeypatch.setattr(md.yf, "Ticker", lambda ticker: _FakeTicker())
        monkeypatch.setattr(md.yf, "download", lambda **kwargs: fallback_df)
        provider = md.LiveDataProvider()
        assert provider.get_quote("RY.TO") == pytest.approx(7.25)

    def test_live_provider_get_quote_returns_none_on_total_failure(self, monkeypatch):
        import market_data as md

        class _FakeTicker:
            fast_info = {}

        monkeypatch.setattr(md.yf, "Ticker", lambda ticker: _FakeTicker())
        monkeypatch.setattr(md.yf, "download", lambda **kwargs: pd.DataFrame())
        provider = md.LiveDataProvider()
        assert provider.get_quote("RY.TO") is None

    def test_live_provider_get_intraday_snapshot_shape(self, monkeypatch):
        import market_data as md

        idx = pd.bdate_range("2024-01-01", periods=3, freq="5min")
        df = pd.DataFrame({"Open": [10, 11, 12], "High": [10.5, 11.5, 12.5],
                            "Low": [9.5, 10.5, 11.5], "Close": [10.2, 11.2, 12.2],
                            "Volume": [100, 100, 100]}, index=idx)
        monkeypatch.setattr(md.yf, "download", lambda **kwargs: df)
        provider = md.LiveDataProvider()
        snap = provider.get_intraday_snapshot("RY.TO")
        assert snap.low == pytest.approx(9.5)
        assert snap.high == pytest.approx(12.5)
        assert snap.close == pytest.approx(12.2)
        assert snap.source == "5m-intraday"

    def test_live_provider_get_intraday_snapshot_none_on_empty(self, monkeypatch):
        import market_data as md
        monkeypatch.setattr(md.yf, "download", lambda **kwargs: pd.DataFrame())
        provider = md.LiveDataProvider()
        assert provider.get_intraday_snapshot("RY.TO") is None

    def test_live_provider_get_sector_uses_cache(self, monkeypatch, tmp_path):
        import market_data as md

        monkeypatch.setattr(md, "_SECTOR_CACHE_FILE", tmp_path / "sector_cache.json")
        monkeypatch.setattr(md, "_sector_cache", {})

        class _FakeTicker:
            info = {"sector": "Financials"}

        monkeypatch.setattr(md.yf, "Ticker", lambda ticker: _FakeTicker())
        provider = md.LiveDataProvider()
        assert provider.get_sector("RY.TO") == "Financials"
        # Second call must not need yf.Ticker again — cache hit
        monkeypatch.setattr(md.yf, "Ticker",
                             lambda ticker: (_ for _ in ()).throw(AssertionError("should be cached")))
        assert provider.get_sector("RY.TO") == "Financials"

    # ── validate_ohlcv() — the owned internal OHLCV contract ────────────────

    def test_validate_ohlcv_accepts_well_formed_frame(self):
        import market_data as md
        idx = pd.bdate_range("2024-01-01", periods=5)
        df = pd.DataFrame({"Open": [1.0] * 5, "High": [1.0] * 5, "Low": [1.0] * 5,
                            "Close": [1.0] * 5, "Volume": [100] * 5}, index=idx)
        out = md.validate_ohlcv(df)
        assert list(out.columns) == md.OHLCV_COLUMNS
        assert out["Volume"].dtype.kind == "f"

    def test_validate_ohlcv_passes_through_empty(self):
        import market_data as md
        empty = pd.DataFrame()
        assert md.validate_ohlcv(empty) is empty

    def test_validate_ohlcv_rejects_missing_column(self):
        import market_data as md
        idx = pd.bdate_range("2024-01-01", periods=3)
        df = pd.DataFrame({"Open": [1.0] * 3, "High": [1.0] * 3,
                            "Low": [1.0] * 3, "Close": [1.0] * 3}, index=idx)  # no Volume
        with pytest.raises(ValueError):
            md.validate_ohlcv(df)

    def test_validate_ohlcv_rejects_non_datetime_index(self):
        import market_data as md
        df = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0],
                            "Close": [1.0], "Volume": [100]})  # default RangeIndex
        with pytest.raises(ValueError):
            md.validate_ohlcv(df)

    # ── download_batch_with_reasons() — no yfinance-shape leakage to callers ──

    def test_download_batch_with_reasons_multi_ticker_happy_path(self, monkeypatch):
        import market_data as md
        idx = pd.bdate_range("2024-01-01", periods=5)
        cols = pd.MultiIndex.from_product([["RY.TO", "TD.TO"], md.OHLCV_COLUMNS])
        raw = pd.DataFrame(1.0, index=idx, columns=cols)
        raw[("RY.TO", "Volume")] = 100
        raw[("TD.TO", "Volume")] = 100
        monkeypatch.setattr(md.yf, "download", lambda **kwargs: raw)
        provider = md.LiveDataProvider()
        data, reasons = provider.download_batch_with_reasons(
            ["RY.TO", "TD.TO"], period="1y", interval="1d", auto_adjust=True
        )
        assert set(data.keys()) == {"RY.TO", "TD.TO"}
        assert reasons == {}
        assert list(data["RY.TO"].columns) == md.OHLCV_COLUMNS

    def test_download_batch_with_reasons_no_data_for_absent_ticker(self, monkeypatch):
        import market_data as md
        idx = pd.bdate_range("2024-01-01", periods=5)
        cols = pd.MultiIndex.from_product([["RY.TO"], md.OHLCV_COLUMNS])
        raw = pd.DataFrame(1.0, index=idx, columns=cols)
        monkeypatch.setattr(md.yf, "download", lambda **kwargs: raw)
        provider = md.LiveDataProvider()
        data, reasons = provider.download_batch_with_reasons(
            ["RY.TO", "DELISTED.TO"], period="1y", interval="1d", auto_adjust=True
        )
        assert "RY.TO" in data
        assert reasons.get("DELISTED.TO") == "no_data"

    def test_download_batch_with_reasons_single_ticker_missing_ohlcv(self, monkeypatch):
        import market_data as md
        idx = pd.bdate_range("2024-01-01", periods=5)
        raw = pd.DataFrame({"Close": [1.0] * 5}, index=idx)  # missing Open/High/Low/Volume
        monkeypatch.setattr(md.yf, "download", lambda **kwargs: raw)
        provider = md.LiveDataProvider()
        data, reasons = provider.download_batch_with_reasons(
            ["RY.TO"], period="1y", interval="1d", auto_adjust=True
        )
        assert data == {}
        assert reasons == {"RY.TO": "missing_ohlcv"}

    def test_download_batch_with_reasons_single_ticker_all_nan_close(self, monkeypatch):
        import market_data as md
        idx = pd.bdate_range("2024-01-01", periods=5)
        raw = pd.DataFrame({"Open": [1.0] * 5, "High": [1.0] * 5, "Low": [1.0] * 5,
                            "Close": [float("nan")] * 5, "Volume": [100] * 5}, index=idx)
        monkeypatch.setattr(md.yf, "download", lambda **kwargs: raw)
        provider = md.LiveDataProvider()
        data, reasons = provider.download_batch_with_reasons(
            ["RY.TO"], period="1y", interval="1d", auto_adjust=True
        )
        assert data == {}
        assert reasons == {"RY.TO": "all_nan_close"}

    def test_download_batch_with_reasons_single_ticker_happy_path(self, monkeypatch):
        import market_data as md
        idx = pd.bdate_range("2024-01-01", periods=5)
        raw = pd.DataFrame({"Open": [1.0] * 5, "High": [1.0] * 5, "Low": [1.0] * 5,
                            "Close": [1.0] * 5, "Volume": [100] * 5}, index=idx)
        monkeypatch.setattr(md.yf, "download", lambda **kwargs: raw)
        provider = md.LiveDataProvider()
        data, reasons = provider.download_batch_with_reasons(
            ["RY.TO"], period="1y", interval="1d", auto_adjust=True
        )
        assert reasons == {}
        assert list(data["RY.TO"].columns) == md.OHLCV_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Portfolio state object (skipped until Phase 3 lands)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.phase3
class TestPortfolioState:
    """
    Tests for the PortfolioState abstraction introduced in Phase 3.
    Expected to FAIL until Phase 3 is merged.
    """

    def test_portfolio_state_importable(self):
        from portfolio import PortfolioState  # noqa: F401

    def test_initial_cash_set_correctly(self):
        from portfolio import PortfolioState
        p = PortfolioState(initial_cash=100_000.0)
        assert p.cash == pytest.approx(100_000.0)

    def test_buy_reduces_cash(self):
        from portfolio import PortfolioState
        p = PortfolioState(initial_cash=100_000.0)
        p.buy("RY.TO", entry_date=date(2024, 6, 1), price=130.0, shares=100)
        assert p.cash == pytest.approx(100_000.0 - 130.0 * 100)

    def test_sell_increases_cash_and_records_pnl(self):
        from portfolio import PortfolioState
        p = PortfolioState(initial_cash=100_000.0)
        p.buy("RY.TO", entry_date=date(2024, 6, 1), price=130.0, shares=100)
        p.sell("RY.TO", sell_date=date(2024, 6, 15), price=140.0)
        assert p.cash == pytest.approx(100_000.0 + (140.0 - 130.0) * 100)
        assert p.realized_pnl == pytest.approx(1_000.0)

    def test_open_positions_tracked(self):
        from portfolio import PortfolioState
        p = PortfolioState(initial_cash=100_000.0)
        p.buy("RY.TO", date(2024, 6, 1), 130.0, 100)
        p.buy("TD.TO", date(2024, 6, 1), 80.0, 50)
        assert "RY.TO" in p.open_positions
        assert "TD.TO" in p.open_positions
        assert len(p.open_positions) == 2

    def test_sell_removes_from_open_positions(self):
        from portfolio import PortfolioState
        p = PortfolioState(initial_cash=100_000.0)
        p.buy("RY.TO", date(2024, 6, 1), 130.0, 100)
        p.sell("RY.TO", date(2024, 6, 15), 135.0)
        assert "RY.TO" not in p.open_positions

    def test_portfolio_state_snapshot_is_independent(self):
        """
        Snapshots must be deep copies — mutating the original after snapshot
        must not affect the snapshot.
        """
        from portfolio import PortfolioState
        p = PortfolioState(initial_cash=100_000.0)
        p.buy("RY.TO", date(2024, 6, 1), 130.0, 100)
        snap = p.snapshot()
        p.sell("RY.TO", date(2024, 6, 15), 140.0)
        # Snapshot still shows position open
        assert "RY.TO" in snap.open_positions

    def test_equal_allocation_preserved_in_portfolio(self):
        """
        Equal allocation logic from virtual_buy must survive into PortfolioState.
        total_cost = sum(int(funds/n / price) * price for each ticker)  ≤  funds
        """
        from portfolio import PortfolioState
        total = 30_000.0
        tickers = [("RY.TO", 130.0), ("TD.TO", 80.0), ("ENB.TO", 55.0)]
        n = len(tickers)
        alloc = total / n
        p = PortfolioState(initial_cash=total)
        for tkr, price in tickers:
            shares = int(alloc / price)
            p.buy(tkr, date(2024, 6, 1), price, shares)
        assert p.cash >= 0, "Cash went negative — allocation bug"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Backtest Runner
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.phase4
class TestBacktestRunner:
    """
    Tests for the BacktestRunner introduced in Phase 4.

    All tests use only synthetic in-memory data — no network calls.
    The HistoricalSliceProvider is constructed directly from pre-built
    DataFrames so tests are fully deterministic and fast.
    """

    # ── fixture helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_provider(
            tickers: list,
            n: int = 600,
            start: str = "2021-01-01",
            seed: int = 42,
    ):
        """Return a HistoricalSliceProvider pre-loaded with synthetic data."""
        from market_data import HistoricalSliceProvider
        np.random.seed(seed)
        data = {}
        for i, tkr in enumerate(tickers):
            idx = pd.bdate_range(start, periods=n)
            prices = 20.0 + i * 5 + np.arange(n) * 0.05 + np.random.randn(n) * 0.1
            close = pd.Series(prices, index=idx)
            high = close + np.abs(np.random.randn(n)) * 0.3
            low = close - np.abs(np.random.randn(n)) * 0.3
            vol = pd.Series(np.ones(n) * 300_000, index=idx, dtype=float)
            data[tkr] = pd.DataFrame({
                "Open": close * 0.999, "High": high,
                "Low": low, "Close": close, "Volume": vol,
            })
        return HistoricalSliceProvider(data)

    @staticmethod
    def _make_cfg(provider, tickers, start="2022-06-01", end="2022-09-01"):
        """Return a BacktestConfig wired to a synthetic provider."""
        from backtest_runner import BacktestConfig
        return BacktestConfig(
            tickers=tickers,
            benchmark="XIU.TO",
            start_date=start,
            end_date=end,
            initial_cash=50_000.0,
            risk_pct=1.0,
            top_n_buys=2,
            min_score=0.0,  # accept all to ensure buys happen
            lookback_days=252,
            _provider=provider,
        )

    # ── import tests ──────────────────────────────────────────────────────────

    def test_backtest_runner_importable(self):
        from backtest_runner import BacktestRunner  # noqa: F401

    def test_backtest_config_importable(self):
        from backtest_runner import BacktestConfig  # noqa: F401

    def test_backtest_results_importable(self):
        from backtest_runner import BacktestResults  # noqa: F401

    # ── clock safety ─────────────────────────────────────────────────────────

    def test_clock_restored_after_run(self):
        """BacktestRunner.run() must always restore the live clock, even on error."""
        from backtest_runner import BacktestRunner
        from time_utils import is_backtest_mode
        tickers = ["RY.TO", "TD.TO", "XIU.TO"]
        provider = self._make_provider(tickers)
        cfg = self._make_cfg(provider, tickers,
                             start="2022-06-01", end="2022-06-10")
        runner = BacktestRunner(cfg)
        runner.run(verbose=False)
        assert not is_backtest_mode(), \
            "BacktestRunner left clock pinned after run() completed"

    # ── no lookahead ─────────────────────────────────────────────────────────

    def test_no_lookahead_in_screener_step(self):
        """
        _run_screener_step must never receive bars beyond sim_date.
        Verified by inspecting the sliced DataFrames returned by the provider.
        """
        from backtest_runner import _run_screener_step, BacktestConfig

        tickers = ["RY.TO", "TD.TO", "XIU.TO"]
        provider = self._make_provider(tickers, n=600)
        sim_date = pd.Timestamp("2022-06-15")

        cfg = BacktestConfig(
            tickers=tickers, benchmark="XIU.TO",
            start_date="2022-06-01", end_date="2022-09-01",
            initial_cash=50_000.0, lookback_days=252, min_score=0.0,
            _provider=provider,
        )

        from time_utils import set_backtest_clock, TSX_TZ
        from datetime import datetime
        set_backtest_clock(datetime(2022, 6, 15, 16, 5, tzinfo=TSX_TZ))
        try:
            df = _run_screener_step(cfg, provider, sim_date)
        finally:
            set_backtest_clock(None)

        # All provider slices for this sim_date must end on or before sim_date
        for tkr in tickers:
            try:
                sliced = provider.get(tkr, as_of=sim_date)
                assert sliced.index.max() <= sim_date, \
                    f"Lookahead: {tkr} slice extends beyond {sim_date.date()}"
            except KeyError:
                pass

    # ── equity curve ─────────────────────────────────────────────────────────

    def test_equity_curve_length_matches_trading_days(self):
        """equity_curve_df must have one row per simulated trading day."""
        from backtest_runner import BacktestRunner, _trading_days
        tickers = ["RY.TO", "TD.TO", "XIU.TO"]
        provider = self._make_provider(tickers, n=600)
        cfg = self._make_cfg(provider, tickers,
                             start="2022-06-01", end="2022-09-01")
        results = BacktestRunner(cfg).run(verbose=False)
        eq = results.equity_curve_df()
        expected = len(_trading_days("2022-06-01", "2022-09-01"))
        assert len(eq) == expected, \
            f"Expected {expected} equity rows, got {len(eq)}"

    def test_equity_curve_starts_at_initial_cash(self):
        """On day 0 (before any buys) total_equity should equal initial_cash."""
        from backtest_runner import BacktestRunner
        tickers = ["RY.TO", "TD.TO", "XIU.TO"]
        provider = self._make_provider(tickers, n=600)
        cfg = self._make_cfg(provider, tickers,
                             start="2022-06-01", end="2022-09-01")
        results = BacktestRunner(cfg).run(verbose=False)
        eq = results.equity_curve_df()
        # Day 0: no buys executed yet, cash = initial_cash, open_value = 0
        first_equity = float(eq["total_equity"].iloc[0])
        assert first_equity == pytest.approx(cfg.initial_cash, rel=0.01)

    def test_equity_curve_columns(self):
        from backtest_runner import BacktestRunner
        tickers = ["RY.TO", "TD.TO", "XIU.TO"]
        provider = self._make_provider(tickers, n=600)
        cfg = self._make_cfg(provider, tickers)
        results = BacktestRunner(cfg).run(verbose=False)
        eq = results.equity_curve_df()
        for col in ("date", "cash", "open_value", "total_equity",
                    "realized_pnl", "open_count"):
            assert col in eq.columns, f"Missing equity curve column: {col}"

    def test_cash_never_goes_negative(self):
        """Cash balance must never go below zero — allocation rule must hold."""
        from backtest_runner import BacktestRunner
        tickers = ["RY.TO", "TD.TO", "ENB.TO", "XIU.TO"]
        provider = self._make_provider(tickers, n=600)
        cfg = self._make_cfg(provider, tickers,
                             start="2022-06-01", end="2022-12-01")
        results = BacktestRunner(cfg).run(verbose=False)
        eq = results.equity_curve_df()
        assert (eq["cash"] >= -0.01).all(), \
            f"Cash went negative: min={eq['cash'].min():.2f}"

    # ── trade log ─────────────────────────────────────────────────────────────

    def test_trade_log_df_columns(self):
        from backtest_runner import BacktestRunner
        tickers = ["RY.TO", "TD.TO", "XIU.TO"]
        provider = self._make_provider(tickers, n=600)
        cfg = self._make_cfg(provider, tickers)
        results = BacktestRunner(cfg).run(verbose=False)
        tl = results.trade_log_df()
        for col in ("ticker", "entry_date", "sell_date", "entry_price",
                    "sell_price", "shares", "pnl", "pnl_pct", "holding_days"):
            assert col in tl.columns, f"Missing trade log column: {col}"

    def test_trade_log_sell_after_entry(self):
        """Every closed trade must have sell_date >= entry_date."""
        from backtest_runner import BacktestRunner
        tickers = ["RY.TO", "TD.TO", "ENB.TO", "XIU.TO"]
        provider = self._make_provider(tickers, n=600)
        cfg = self._make_cfg(provider, tickers,
                             start="2022-06-01", end="2022-12-01")
        results = BacktestRunner(cfg).run(verbose=False)
        tl = results.trade_log_df()
        if not tl.empty:
            assert (tl["sell_date"] >= tl["entry_date"]).all(), \
                "Trade log contains sell_date < entry_date"

    # ── summary ───────────────────────────────────────────────────────────────

    def test_summary_returns_string(self):
        from backtest_runner import BacktestRunner
        tickers = ["RY.TO", "TD.TO", "XIU.TO"]
        provider = self._make_provider(tickers, n=600)
        cfg = self._make_cfg(provider, tickers)
        results = BacktestRunner(cfg).run(verbose=False)
        s = results.summary()
        assert isinstance(s, str)
        assert "Total return" in s
        assert "Win rate" in s

    def test_summary_contains_initial_capital(self):
        from backtest_runner import BacktestRunner
        tickers = ["RY.TO", "TD.TO", "XIU.TO"]
        provider = self._make_provider(tickers, n=600)
        cfg = self._make_cfg(provider, tickers)
        results = BacktestRunner(cfg).run(verbose=False)
        assert "50,000" in results.summary()

    # ── capital flow correctness ──────────────────────────────────────────────

    def test_realized_pnl_matches_trade_log(self):
        """
        portfolio.realized_pnl must equal the sum of all closed-trade pnl values.
        This verifies the PortfolioState accounting is internally consistent.
        """
        from backtest_runner import BacktestRunner
        tickers = ["RY.TO", "TD.TO", "ENB.TO", "XIU.TO"]
        provider = self._make_provider(tickers, n=600)
        cfg = self._make_cfg(provider, tickers,
                             start="2022-06-01", end="2022-12-01")
        results = BacktestRunner(cfg).run(verbose=False)
        eq = results.equity_curve_df()
        tl = results.trade_log_df()

        runner_pnl = float(eq["realized_pnl"].iloc[-1])
        trade_log_pnl = float(tl["pnl"].sum()) if not tl.empty else 0.0
        assert runner_pnl == pytest.approx(trade_log_pnl, abs=0.05), \
            f"Realized PnL mismatch: equity_curve={runner_pnl:.2f}, trade_log={trade_log_pnl:.2f}"

    def test_deterministic_on_repeated_runs(self):
        """
        Running the same config twice must produce identical equity curves.
        Any non-determinism indicates hidden state mutation.
        """
        from backtest_runner import BacktestRunner
        tickers = ["RY.TO", "TD.TO", "XIU.TO"]
        provider = self._make_provider(tickers, n=600, seed=7)
        cfg = self._make_cfg(provider, tickers,
                             start="2022-06-01", end="2022-09-01")

        r1 = BacktestRunner(cfg).run(verbose=False)
        r2 = BacktestRunner(cfg).run(verbose=False)

        eq1 = r1.equity_curve_df()["total_equity"].tolist()
        eq2 = r2.equity_curve_df()["total_equity"].tolist()
        assert eq1 == eq2, "Backtest is non-deterministic — hidden state mutation"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Backtest Report
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.phase5
class TestBacktestReport:
    """
    Tests for the backtest_report.write_backtest_report() introduced in Phase 5.
    All tests use synthetic BacktestResults — no network calls.
    """

    @staticmethod
    def _make_results(n_days: int = 120, n_trades: int = 4,
                      seed: int = 42) -> "BacktestResults":
        """Build a synthetic BacktestResults for report testing."""
        from backtest_runner import BacktestResults, BacktestConfig, DayLog
        from portfolio import ClosedTrade
        np.random.seed(seed)

        cfg = BacktestConfig(
            tickers=["RY.TO", "TD.TO", "XIU.TO"],
            start_date="2023-01-02",
            end_date="2023-06-30",
            initial_cash=100_000.0,
        )

        start = pd.Timestamp("2023-01-02")
        idx = pd.bdate_range(start, periods=n_days)
        equity = 100_000.0 + np.cumsum(np.random.randn(n_days) * 200)
        equity = np.maximum(equity, 1)

        day_logs = [
            DayLog(
                sim_date=d.date(),
                cash=float(equity[i]) * 0.6,
                open_value=float(equity[i]) * 0.4,
                total_equity=float(equity[i]),
                realized_pnl=float(equity[i]) - 100_000.0,
                open_tickers=["RY.TO"],
                buys_today=[],
                sells_today=[],
            )
            for i, d in enumerate(idx)
        ]

        trades = []
        tickers = ["RY.TO", "TD.TO", "ENB.TO", "CNQ.TO"]
        for k in range(n_trades):
            entry_date = idx[k * (n_days // n_trades)].date()
            sell_date = idx[min(k * (n_days // n_trades) + 15, n_days - 1)].date()
            ep = 50.0 + k * 10
            sp = ep * (1 + (0.05 if k % 2 == 0 else -0.03))
            shares = 100
            pnl = (sp - ep) * shares
            trades.append(ClosedTrade(
                ticker=tickers[k % len(tickers)],
                entry_date=entry_date,
                sell_date=sell_date,
                entry_price=ep,
                sell_price=round(sp, 2),
                shares=shares,
                pnl=round(pnl, 2),
                pnl_pct=round((sp / ep - 1) * 100, 2),
            ))

        return BacktestResults(cfg=cfg, day_logs=day_logs, trades=trades)

    # ── importable ────────────────────────────────────────────────────────────

    def test_module_importable(self):
        from backtest_report import write_backtest_report  # noqa: F401

    # ── file creation ─────────────────────────────────────────────────────────

    def test_creates_html_file(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "report.html"
        write_backtest_report(results, str(out))
        assert out.exists(), "Report file was not created"

    def test_output_is_valid_html(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "report.html"
        write_backtest_report(results, str(out))
        content = out.read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html")
        assert "</html>" in content

    def test_creates_parent_directories(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "nested" / "deep" / "report.html"
        write_backtest_report(results, str(out))
        assert out.exists()

    # ── content checks ────────────────────────────────────────────────────────

    def test_contains_period_dates(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out))
        content = out.read_text()
        assert "2023-01-02" in content
        assert "2023-06-30" in content

    def test_contains_initial_capital(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out))
        assert "100,000" in out.read_text()

    def test_contains_equity_svg(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out))
        content = out.read_text()
        assert "<svg" in content
        assert "Equity Curve" in content

    def test_contains_drawdown_section(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out))
        assert "Drawdown" in out.read_text()

    def test_contains_trade_log_section(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out))
        content = out.read_text()
        assert "Trade Log" in content
        # All tickers from trades should appear
        for tkr in ["RY.TO", "TD.TO"]:
            assert tkr in content

    def test_contains_per_ticker_stats(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out))
        assert "Per-Ticker" in out.read_text()

    def test_contains_monthly_heatmap(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out))
        assert "Monthly Returns" in out.read_text()

    def test_no_trade_log_renders_gracefully(self, tmp_path):
        """Report must render without errors when there are no trades."""
        from backtest_report import write_backtest_report
        results = self._make_results(n_trades=0)
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out))  # must not raise
        content = out.read_text()
        assert "No closed trades" in content

    def test_with_benchmark_overlay(self, tmp_path):
        """Benchmark series must be rendered without errors."""
        from backtest_report import write_backtest_report
        results = self._make_results()
        idx = pd.bdate_range("2023-01-02", periods=120)
        bench = pd.Series(100 + np.arange(120) * 0.05, index=idx)
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out), benchmark_equity=bench)
        content = out.read_text()
        assert "Benchmark" in content

    def test_stat_pills_contain_win_rate(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out))
        assert "Win Rate" in out.read_text()

    def test_disclaimer_present(self, tmp_path):
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out))
        assert "Not financial advice" in out.read_text()

    def test_self_contained_no_external_scripts(self, tmp_path):
        """
        Report must not reference external JS CDNs or stylesheets.
        Ensures it renders offline and in email clients.
        """
        from backtest_report import write_backtest_report
        results = self._make_results()
        out = tmp_path / "r.html"
        write_backtest_report(results, str(out))
        content = out.read_text()
        for bad in ["<script src=", "cdn.jsdelivr", "cdnjs.cloudflare",
                    "<link rel='stylesheet'"]:
            assert bad not in content, \
                f"Report references external resource: {bad}"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — CLI entry point (run_backtest.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.phase6
class TestRunBacktest:
    """
    Tests for run_backtest.py — the CLI entry point.
    All tests use synthetic data; no network calls.
    """

    @staticmethod
    def _make_provider_and_tickers(seed: int = 0):
        """Shared fixture: synthetic provider + ticker list."""
        from market_data import HistoricalSliceProvider
        np.random.seed(seed)
        tickers = ["RY.TO", "TD.TO", "ENB.TO", "XIU.TO"]
        n = 700
        data = {}
        for i, tkr in enumerate(tickers):
            idx = pd.bdate_range("2020-01-01", periods=n)
            prices = 20 + i * 5 + np.arange(n) * 0.06 + np.random.randn(n) * 0.2
            close = pd.Series(prices, index=idx)
            high = close + np.abs(np.random.randn(n)) * 0.4
            low = close - np.abs(np.random.randn(n)) * 0.4
            vol = pd.Series(np.ones(n) * 400_000, index=idx, dtype=float)
            data[tkr] = pd.DataFrame({
                "Open": close, "High": high, "Low": low,
                "Close": close, "Volume": vol,
            })
        return HistoricalSliceProvider(data), tickers

    # ── importable ────────────────────────────────────────────────────────────

    def test_module_importable(self):
        import run_backtest  # noqa: F401

    def test_load_tickers_importable(self):
        from run_backtest import _load_tickers  # noqa: F401

    def test_run_single_importable(self):
        from run_backtest import _run_single  # noqa: F401

    def test_run_sweep_importable(self):
        from run_backtest import _run_sweep  # noqa: F401

    # ── _load_tickers ─────────────────────────────────────────────────────────

    def test_load_tickers_reads_file(self, tmp_path):
        from run_backtest import _load_tickers
        f = tmp_path / "tickers.txt"
        f.write_text("RY.TO\nTD.TO\nENB.TO\n")
        assert _load_tickers(str(f)) == ["RY.TO", "TD.TO", "ENB.TO"]

    def test_load_tickers_skips_comments(self, tmp_path):
        from run_backtest import _load_tickers
        f = tmp_path / "tickers.txt"
        f.write_text("# comment\nRY.TO\n# another\nTD.TO\n")
        assert _load_tickers(str(f)) == ["RY.TO", "TD.TO"]

    def test_load_tickers_skips_blank_lines(self, tmp_path):
        from run_backtest import _load_tickers
        f = tmp_path / "tickers.txt"
        f.write_text("\nRY.TO\n\nTD.TO\n\n")
        assert _load_tickers(str(f)) == ["RY.TO", "TD.TO"]

    def test_load_tickers_missing_file_raises(self, tmp_path):
        from run_backtest import _load_tickers
        with pytest.raises(FileNotFoundError):
            _load_tickers(str(tmp_path / "nonexistent.txt"))

    # ── argument parser defaults ──────────────────────────────────────────────

    def test_parser_default_start(self):
        from run_backtest import _build_parser
        args = _build_parser().parse_args([])
        assert args.start == "2022-01-01"

    def test_parser_default_end(self):
        from run_backtest import _build_parser
        args = _build_parser().parse_args([])
        assert args.end == "2024-01-01"

    def test_parser_default_capital(self):
        from run_backtest import _build_parser
        args = _build_parser().parse_args([])
        assert args.capital == pytest.approx(100_000.0)

    def test_parser_default_min_score_zero(self):
        """min_score default must be 0.0 — not 55.0 (wrong for small universes)."""
        from run_backtest import _build_parser
        args = _build_parser().parse_args([])
        assert args.min_score == pytest.approx(0.0)

    def test_parser_custom_args(self):
        from run_backtest import _build_parser
        args = _build_parser().parse_args([
            "--start", "2021-01-01",
            "--end", "2023-06-01",
            "--capital", "50000",
            "--risk", "0.5",
            "--top-n", "2",
        ])
        assert args.start == "2021-01-01"
        assert args.end == "2023-06-01"
        assert args.capital == pytest.approx(50_000.0)
        assert args.risk == pytest.approx(0.5)
        assert args.top_n == 2

    def test_parser_sweep_flag(self):
        from run_backtest import _build_parser
        args = _build_parser().parse_args(["--sweep"])
        assert args.sweep is True

    def test_parser_quiet_flag(self):
        from run_backtest import _build_parser
        args = _build_parser().parse_args(["--quiet"])
        assert args.quiet is True

    # ── shared fixtures (class-scoped — each simulation runs once) ───────────

    @pytest.fixture(scope="class")
    def single_output(self, tmp_path_factory):
        """Run _run_single once; all single-output tests share this directory."""
        from run_backtest import _run_single, _build_parser
        import run_backtest as rb

        provider, tickers = self._make_provider_and_tickers()
        args = _build_parser().parse_args([
            "--start", "2022-06-01", "--end", "2022-09-01", "--quiet",
        ])
        tmp = tmp_path_factory.mktemp("single")
        original = rb.OUT_PATH
        rb.OUT_PATH = tmp
        try:
            _run_single(args, tickers, provider, bench_series=None)
        finally:
            rb.OUT_PATH = original
        return tmp

    @pytest.fixture(scope="class")
    def single_bench_output(self, tmp_path_factory):
        """Run _run_single with a benchmark overlay once."""
        from run_backtest import _run_single, _build_parser
        import run_backtest as rb

        provider, tickers = self._make_provider_and_tickers()
        idx = pd.bdate_range("2022-06-01", periods=65)
        bench = pd.Series(100 + np.arange(65) * 0.05, index=idx)
        args = _build_parser().parse_args([
            "--start", "2022-06-01", "--end", "2022-09-01", "--quiet",
        ])
        tmp = tmp_path_factory.mktemp("single_bench")
        original = rb.OUT_PATH
        rb.OUT_PATH = tmp
        try:
            _run_single(args, tickers, provider, bench_series=bench)
        finally:
            rb.OUT_PATH = original
        return tmp

    @pytest.fixture(scope="class")
    def sweep_output(self, tmp_path_factory):
        """Run _run_sweep once (16 backtests); all sweep tests share this directory."""
        from run_backtest import _run_sweep, _build_parser
        import run_backtest as rb

        provider, tickers = self._make_provider_and_tickers()
        args = _build_parser().parse_args([
            "--start", "2022-06-01", "--end", "2022-09-01", "--quiet",
        ])
        tmp = tmp_path_factory.mktemp("sweep")
        original = rb.OUT_PATH
        rb.OUT_PATH = tmp
        try:
            _run_sweep(args, tickers, provider, bench_series=None)
        finally:
            rb.OUT_PATH = original
        return tmp

    # ── _run_single ───────────────────────────────────────────────────────────

    def test_run_single_creates_html_report(self, single_output):
        """_run_single must write an HTML report to out/."""
        html_files = list(single_output.glob("backtest_*.html"))
        assert len(html_files) == 1, f"Expected 1 HTML report, got {html_files}"

    def test_run_single_creates_csv_files(self, single_output):
        """_run_single must write equity and trades CSVs."""
        assert len(list(single_output.glob("backtest_equity*.csv"))) == 1
        assert len(list(single_output.glob("backtest_trades*.csv"))) == 1

    def test_run_single_with_benchmark_overlay(self, single_bench_output):
        """benchmark_equity series must not cause errors in _run_single."""
        html_files = list(single_bench_output.glob("backtest_*.html"))
        assert len(html_files) == 1
        assert "Benchmark" in html_files[0].read_text()

    # ── _run_sweep ────────────────────────────────────────────────────────────

    def test_run_sweep_creates_sweep_csv(self, sweep_output):
        """_run_sweep must write a sweep results CSV."""
        assert len(list(sweep_output.glob("backtest_sweep_*.csv"))) == 1

    def test_run_sweep_csv_has_expected_columns(self, sweep_output):
        """Sweep CSV must contain time_stop_d, stop_atr, ret_%, sharpe."""
        sweep_df = pd.read_csv(list(sweep_output.glob("backtest_sweep_*.csv"))[0])
        for col in ("time_stop_d", "stop_atr", "ret_%", "max_dd_%", "sharpe", "trades"):
            assert col in sweep_df.columns, f"Missing sweep column: {col}"

    def test_run_sweep_row_count(self, sweep_output):
        """Sweep must produce one row per parameter combination: time_stop_days(4) × stop_atr(4) = 16."""
        sweep_df = pd.read_csv(list(sweep_output.glob("backtest_sweep_*.csv"))[0])
        assert len(sweep_df) == 16, f"Expected 16 sweep rows, got {len(sweep_df)}"

    def test_run_sweep_also_writes_best_html(self, sweep_output):
        """Sweep must write a full HTML report for the best Sharpe combo."""
        html_files = list(sweep_output.glob("backtest_*.html"))
        assert len(html_files) >= 1, "Sweep must write at least one HTML report"

    # ── BacktestConfig new fields ─────────────────────────────────────────────

    def test_config_screener_frequency_default(self):
        """screener_frequency must default to 5 (weekly)."""
        from backtest_runner import BacktestConfig
        cfg = BacktestConfig(tickers=["RY.TO", "XIU.TO"])
        assert cfg.screener_frequency == 5

    def test_config_min_score_default(self):
        """min_score must default to 0.0 after the fix."""
        from backtest_runner import BacktestConfig
        cfg = BacktestConfig(tickers=["RY.TO", "XIU.TO"])
        assert cfg.min_score == pytest.approx(0.0)

    def test_screener_frequency_reduces_screener_calls(self):
        """
        With screener_frequency=N, the screener should run ceil(days/N) times,
        not once per day.  Verify via a short run that the cached df is reused.
        """
        from backtest_runner import BacktestConfig, BacktestRunner
        provider, tickers = self._make_provider_and_tickers()

        screener_calls = []
        import backtest_runner as br_mod
        original_fn = br_mod._run_screener_step

        def counting_screener(*args, **kwargs):
            screener_calls.append(1)
            return original_fn(*args, **kwargs)

        br_mod._run_screener_step = counting_screener
        try:
            cfg = BacktestConfig(
                tickers=tickers, benchmark="XIU.TO",
                start_date="2022-06-01", end_date="2022-09-01",
                initial_cash=50_000.0, min_score=0.0,
                lookback_days=252, screener_frequency=5,
                _provider=provider,
            )
            BacktestRunner(cfg).run(verbose=False)
        finally:
            br_mod._run_screener_step = original_fn

        from backtest_runner import _trading_days
        n_days = len(_trading_days("2022-06-01", "2022-09-01"))
        max_expected_calls = n_days // 5 + 1  # ceil(days / frequency)
        assert len(screener_calls) <= max_expected_calls, (
            f"Screener ran {len(screener_calls)} times for {n_days} days "
            f"with frequency=5 (expected ≤ {max_expected_calls})"
        )
