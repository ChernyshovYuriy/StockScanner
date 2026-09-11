"""
tests/test_scanner_board_row.py
=================================
Tests for scanner_board.row.compute_row — see scanner_board/PLAN.md
Phase 4.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from canadian_stock_screener import TechnicalIndicators
from scanner_board import row, store
from scanner_board.weekly import resample_ohlcv_weekly


def _synthetic_daily(n: int, start="2023-01-02", seed: int = 7,
                      drift: float = 0.08) -> pd.DataFrame:
    """A deterministic (seeded), gently uptrending OHLCV series long
    enough to exercise every windowed column (oscillator 5%-rule lookback,
    MACD-H 3-month extreme lookback, divergence pivot spacing, ma200)."""
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(drift + rng.normal(0, 0.3, n))
    high = close + np.abs(rng.normal(0.2, 0.1, n))
    low = close - np.abs(rng.normal(0.2, 0.1, n))
    open_ = close - rng.normal(0, 0.1, n)
    volume = np.abs(rng.normal(100_000, 10_000, n))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


class TestComputeRowSchema:
    """The single-source-of-truth contract between row.py and store.py:
    compute_row()'s own keys must exactly match store.ROW_VALUE_COLUMNS,
    so the schema is never hand-typed a second time and can't silently
    drift from what compute_row actually produces."""

    def test_keys_match_store_schema_exactly(self):
        daily = _synthetic_daily(260)
        weekly = resample_ohlcv_weekly(daily)
        result = row.compute_row("TEST", daily, weekly)
        assert set(result.keys()) - {"ticker"} == set(store.ROW_VALUE_COLUMNS)

    def test_ticker_is_passed_through_verbatim(self):
        daily = _synthetic_daily(260)
        weekly = resample_ohlcv_weekly(daily)
        result = row.compute_row("XIU.TO", daily, weekly)
        assert result["ticker"] == "XIU.TO"

    def test_numeric_columns_are_plain_float_or_none(self):
        """Never a numpy scalar (numpy.float64 et al. don't round-trip
        through sqlite3's type adapter the same way a plain float does),
        and never NaN (None instead, so a JSON/SQL consumer doesn't have
        to special-case NaN != NaN)."""
        daily = _synthetic_daily(260)
        weekly = resample_ohlcv_weekly(daily)
        result = row.compute_row("TEST", daily, weekly)
        for col in store.NUMERIC_COLUMNS:
            value = result[col]
            assert value is None or type(value) is float
            if value is not None:
                assert not np.isnan(value)

    def test_label_columns_are_plain_strings(self):
        daily = _synthetic_daily(260)
        weekly = resample_ohlcv_weekly(daily)
        result = row.compute_row("TEST", daily, weekly)
        for col in store.LABEL_COLUMNS:
            assert isinstance(result[col], str)


class TestComputeRowInsufficientData:
    """A short history (e.g. a recently-listed ticker) must degrade to
    None/neutral labels, never crash -- the same convention every Phase
    1-3 function already established for its own insufficient-data case."""

    def test_short_history_does_not_crash(self):
        daily = _synthetic_daily(5)  # one full trading week, Mon-Fri
        weekly = resample_ohlcv_weekly(daily)
        assert len(weekly) == 1
        result = row.compute_row("TEST", daily, weekly)
        assert set(result.keys()) - {"ticker"} == set(store.ROW_VALUE_COLUMNS)


class TestComputeRowWiring:
    """Precise wiring checks via mocking: confirms compute_row hands each
    downstream function the SERIES the book actually calls for (fast EMA
    vs slow EMA, daily vs weekly), not just that some EMA of some period
    was used somewhere."""

    def test_impulse_uses_the_fast_ema_not_the_slow_one(self):
        """Ch.40: Impulse's inertia measure is the FAST EMA's slope --
        distinct from Triple Screen's SLOW-EMA trend read (Ch.22's Figure
        22.2 caption)."""
        daily = _synthetic_daily(80)
        weekly = resample_ohlcv_weekly(daily)
        expected_daily_fast = TechnicalIndicators.ema(daily["Close"], row.DAILY_VALUE_ZONE_FAST_EMA)
        expected_weekly_fast = TechnicalIndicators.ema(weekly["Close"], row.WEEKLY_VALUE_ZONE_FAST_EMA)

        with patch("scanner_board.row.impulse_color") as mock_impulse:
            from scanner_board.triple_screen import ImpulseColor
            mock_impulse.return_value = ImpulseColor.BLUE
            row.compute_row("TEST", daily, weekly)

        assert mock_impulse.call_count == 2
        daily_call_ema = mock_impulse.call_args_list[0][0][0]
        weekly_call_ema = mock_impulse.call_args_list[1][0][0]
        pd.testing.assert_series_equal(daily_call_ema, expected_daily_fast, check_names=False)
        pd.testing.assert_series_equal(weekly_call_ema, expected_weekly_fast, check_names=False)

    def test_triple_screen_trend_reads_use_the_slow_ema(self):
        """Ch.22's Figure 22.2 caption: 'The slow EMA helps identify the
        trend' -- Triple Screen's weekly/daily trend calls must be fed the
        SLOW EMA of each pair, not the fast one Impulse uses."""
        daily = _synthetic_daily(80)
        weekly = resample_ohlcv_weekly(daily)
        expected_daily_slow = TechnicalIndicators.ema(daily["Close"], row.DAILY_VALUE_ZONE_SLOW_EMA)
        expected_weekly_slow = TechnicalIndicators.ema(weekly["Close"], row.WEEKLY_VALUE_ZONE_SLOW_EMA)

        with patch("scanner_board.row.trend_direction") as mock_trend:
            from scanner_board.triple_screen import TrendDirection
            mock_trend.return_value = TrendDirection.FLAT
            row.compute_row("TEST", daily, weekly)

        assert mock_trend.call_count == 2
        weekly_call_ema = mock_trend.call_args_list[0][0][0]
        daily_call_ema = mock_trend.call_args_list[1][0][0]
        pd.testing.assert_series_equal(weekly_call_ema, expected_weekly_slow, check_names=False)
        pd.testing.assert_series_equal(daily_call_ema, expected_daily_slow, check_names=False)

    def test_value_zone_uses_fast_and_slow_ema_pair(self):
        daily = _synthetic_daily(80)
        weekly = resample_ohlcv_weekly(daily)
        expected_fast = TechnicalIndicators.ema(daily["Close"], row.DAILY_VALUE_ZONE_FAST_EMA)
        expected_slow = TechnicalIndicators.ema(daily["Close"], row.DAILY_VALUE_ZONE_SLOW_EMA)

        with patch("scanner_board.row.value_zone_position") as mock_vz:
            from scanner_board.thesis_rules import ValueZonePosition
            mock_vz.return_value = ValueZonePosition.IN_ZONE
            row.compute_row("TEST", daily, weekly)

        called_price, called_fast, called_slow = mock_vz.call_args_list[0][0]
        pd.testing.assert_series_equal(called_price, daily["Close"], check_names=False)
        pd.testing.assert_series_equal(called_fast, expected_fast, check_names=False)
        pd.testing.assert_series_equal(called_slow, expected_slow, check_names=False)

    def test_ma22_equals_the_same_slow_ema_used_for_the_value_zone(self):
        """ma22 (the user-requested display column) and the daily
        value-zone's slow EMA are the SAME 22-period EMA, computed once --
        not two independent EMA(22) calls that could silently drift apart
        under a future edit."""
        daily = _synthetic_daily(80)
        weekly = resample_ohlcv_weekly(daily)
        result = row.compute_row("TEST", daily, weekly)
        expected = TechnicalIndicators.ema(daily["Close"], 22).iloc[-1]
        assert result["ma22"] == pytest.approx(expected)
