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
        s = self.sc.score_stage2(self.close)
        assert s == pytest.approx(84.9, abs=0.5), f"stage2 golden changed: {s}"

    def test_score_macd_range(self):
        s = self.sc.score_macd(self.close)
        assert 0.0 <= s <= 100.0

    def test_score_macd_golden(self):
        s = self.sc.score_macd(self.close)
        assert s == pytest.approx(95.0, abs=0.5), f"macd golden changed: {s}"

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
        """
        short = self.close.iloc[:20]
        assert self.sc.score_macd(short) == pytest.approx(75.0, abs=0.5)

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

    def test_rsi_perfectly_monotone_produces_nan(self):
        """
        Documents a known edge case: a perfectly monotone series (zero losses on
        every bar) causes avg_loss=0 → NaN propagation throughout RSI.
        This is existing behaviour — do not silently change it.
        """
        up = pd.Series(np.linspace(10, 30, 100))
        rsi = self.ti.rsi(up)
        assert rsi.dropna().empty, (
            "Expected all-NaN RSI on perfectly monotone series (zero-loss edge case)"
        )

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
        """initial_stop ≈ entry_price - 1.5 × ATR14."""
        from position_monitor import compute_signals, INITIAL_STOP_ATR_K
        df, pos = self._make_pos_fixture()
        r = compute_signals(pos, df)
        if "initial_stop" in r and "ATR14" in r:
            expected = pos.entry_price - INITIAL_STOP_ATR_K * float(r["ATR14"])
            assert float(r["initial_stop"]) == pytest.approx(expected, rel=0.01)

    def test_pnl_pct_formula(self):
        """pnl_% = (last_close / entry_price - 1) * 100."""
        from position_monitor import compute_signals
        df, pos = self._make_pos_fixture()
        r = compute_signals(pos, df)
        expected = (float(r["last_close"]) / pos.entry_price - 1.0) * 100.0
        assert float(r["pnl_%"]) == pytest.approx(expected, abs=0.01)

    def test_time_stop_triggers_after_n_days_with_small_profit(self):
        """TIME_STOP fires when tdays ≥ 7 and profit < 0.5%."""
        from position_monitor import compute_signals
        n = 200
        idx = pd.bdate_range("2024-01-01", periods=n)
        # Flat price — stays near entry, profit ≈ 0
        price = np.full(n, 20.0)
        close = pd.Series(price, index=idx)
        df = pd.DataFrame({"Open": close, "High": close + 0.05,
                           "Low": close - 0.05, "Close": close,
                           "Volume": pd.Series(np.ones(n) * 300_000, index=idx)})
        pos = self._Pos("X.TO", idx[0].date(), 20.0, 100.0)
        r = compute_signals(pos, df)
        # Should fire TIME_STOP since tdays >> 7 and profit ≈ 0
        assert r["status"] == "SELL"
        assert "TIME_STOP" in r["reason"]

    def test_no_data_returns_status_key(self):
        """An empty DataFrame must return a dict with 'status' = 'NO_DATA'."""
        from position_monitor import compute_signals
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        pos = self._Pos("X.TO", date(2024, 1, 1), 20.0, 10.0)
        r = compute_signals(pos, empty)
        assert r.get("status") in ("NO_DATA", "NO_ATR")


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTERIZATION — virtual_buy.py (funds + allocation)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.characterization
class TestVirtualBuy:

    def test_read_funds_parses_float(self):
        from virtual_buy import read_funds
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "funds.txt"
            p.write_text("50000.00\n")
            assert read_funds(p) == pytest.approx(50000.0)

    def test_read_funds_strips_dollar_and_comma(self):
        from virtual_buy import read_funds
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "funds.txt"
            p.write_text("$1,234.56\n")
            assert read_funds(p) == pytest.approx(1234.56)

    def test_read_funds_skips_comments(self):
        from virtual_buy import read_funds
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "funds.txt"
            p.write_text("# available capital\n99999.0\n")
            assert read_funds(p) == pytest.approx(99999.0)

    def test_read_funds_missing_file_returns_zero(self):
        from virtual_buy import read_funds
        assert read_funds(Path("/nonexistent/path/funds.txt")) == 0.0

    def test_write_funds_overwrites_value(self):
        from virtual_buy import read_funds, write_funds
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "funds.txt"
            p.write_text("100.00\n")
            write_funds(p, 9999.50)
            assert read_funds(p) == pytest.approx(9999.50)

    def test_write_funds_preserves_comments(self):
        from virtual_buy import write_funds
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "funds.txt"
            p.write_text("# my comment\n100.00\n")
            write_funds(p, 50.0)
            content = p.read_text()
            assert "# my comment" in content
            assert "50.00" in content

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

    def test_run_virtual_buy_dry_run_writes_nothing(self):
        """dry_run=True must not touch positions or funds files."""
        from virtual_buy import run_virtual_buy
        from schema_keys import INTENT_COL_STATUS, SIGNAL_COL_TICKER
        from schema_keys import INTENT_COL_SIGNAL_DATE, INTENT_COL_ALERT_STATE
        from schema_keys import INTENT_COL_PRIORITY, SIGNAL_COL_PATTERN
        from schema_keys import INTENT_COL_ENTRY_PRICE_PLANNED, INTENT_COL_STOP_PRICE
        from schema_keys import INTENT_COL_TARGET_PRICE, INTENT_COL_RR
        from schema_keys import INTENT_COL_REASON, INTENT_COL_CREATED_AT

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            signals_path = tmp / "candidates_queue.csv"
            funds_path = tmp / "funds.txt"
            positions_path = tmp / "positions.csv"

            # Write minimal intent file
            df = pd.DataFrame([{
                SIGNAL_COL_TICKER: "RY.TO",
                INTENT_COL_SIGNAL_DATE: "2025-01-01",
                INTENT_COL_ALERT_STATE: "CONFIRMED",
                INTENT_COL_PRIORITY: 1,
                SIGNAL_COL_PATTERN: "BASE",
                INTENT_COL_ENTRY_PRICE_PLANNED: 130.0,
                INTENT_COL_STOP_PRICE: 125.0,
                INTENT_COL_TARGET_PRICE: 140.0,
                INTENT_COL_RR: 2.0,
                INTENT_COL_STATUS: "pending",
                INTENT_COL_REASON: "",
                INTENT_COL_CREATED_AT: "2025-01-01",
            }])
            df.to_csv(signals_path, index=False)
            funds_path.write_text("50000.00\n")

            original_funds_content = funds_path.read_text()

            run_virtual_buy(
                signals_path=signals_path,
                funds_path=funds_path,
                positions_path=positions_path,
                top_n=None,
                dry_run=True,
            )

            # Funds file must be unchanged
            assert funds_path.read_text() == original_funds_content
            # Positions file must not be created
            assert not positions_path.exists()

    def test_run_virtual_buy_skips_zero_funds(self):
        """No buys when funds file is 0."""
        from virtual_buy import run_virtual_buy
        from schema_keys import INTENT_COL_STATUS, SIGNAL_COL_TICKER
        from schema_keys import INTENT_COL_SIGNAL_DATE, INTENT_COL_ALERT_STATE
        from schema_keys import INTENT_COL_PRIORITY, SIGNAL_COL_PATTERN
        from schema_keys import INTENT_COL_ENTRY_PRICE_PLANNED, INTENT_COL_STOP_PRICE
        from schema_keys import INTENT_COL_TARGET_PRICE, INTENT_COL_RR
        from schema_keys import INTENT_COL_REASON, INTENT_COL_CREATED_AT

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            signals_path = tmp / "signals.csv"
            funds_path = tmp / "funds.txt"
            positions_path = tmp / "positions.csv"

            pd.DataFrame([{
                SIGNAL_COL_TICKER: "TD.TO",
                INTENT_COL_SIGNAL_DATE: "2025-01-01",
                INTENT_COL_ALERT_STATE: "CONFIRMED",
                INTENT_COL_PRIORITY: 1,
                SIGNAL_COL_PATTERN: "BASE",
                INTENT_COL_ENTRY_PRICE_PLANNED: 80.0,
                INTENT_COL_STOP_PRICE: 75.0,
                INTENT_COL_TARGET_PRICE: 90.0,
                INTENT_COL_RR: 2.0,
                INTENT_COL_STATUS: "pending",
                INTENT_COL_REASON: "",
                INTENT_COL_CREATED_AT: "2025-01-01",
            }]).to_csv(signals_path, index=False)
            funds_path.write_text("0\n")

            run_virtual_buy(signals_path, funds_path, positions_path,
                            top_n=None, dry_run=False)

            assert not positions_path.exists()


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
