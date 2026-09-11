"""
tests/test_scanner_board_weekly.py
====================================
Tests for scanner_board.weekly.resample_ohlcv_weekly — see
scanner_board/PLAN.md Phase 4.
"""
from __future__ import annotations

import pandas as pd
import pytest

from canadian_stock_screener import TechnicalIndicators
from scanner_board.weekly import resample_ohlcv_weekly


def _daily_ohlcv(n: int, start="2024-01-01") -> pd.DataFrame:
    """12 business days starting Monday 2024-01-01: week1 = Mon-Fri Jan
    1-5 (5 days, full), week2 = Mon-Fri Jan 8-12 (5 days, full), week3 =
    Mon-Tue Jan 15-16 (2 days, a still-in-progress partial week)."""
    idx = pd.bdate_range(start, periods=n)
    i = range(n)
    return pd.DataFrame({
        "Open": [100.0 + x for x in i],
        "High": [110.0 + x for x in i],
        "Low": [90.0 + x for x in i],
        "Close": [105.0 + x for x in i],
        "Volume": [1000.0 + x * 10 for x in i],
    }, index=idx)


class TestResampleOhlcvWeekly:

    def test_ground_truth_two_full_weeks_plus_dropped_partial(self):
        """Week1 (i=0..4): Open=first=100, High=max=114, Low=min=90,
        Close=last=109, Volume=sum(1000..1040 step 10)=5100.
        Week2 (i=5..9): Open=105, High=119, Low=95, Close=114,
        Volume=sum(1050..1090 step 10)=5350.
        Week3 (i=10,11): only 2 trading days -- dropped entirely, not just
        its Close, mirroring TechnicalIndicators.weekly_resample's own
        partial-week rule (reused via the Close-index reindex, not
        re-implemented)."""
        daily = _daily_ohlcv(12)
        weekly = resample_ohlcv_weekly(daily)

        assert len(weekly) == 2
        assert weekly["Open"].tolist() == [100.0, 105.0]
        assert weekly["High"].tolist() == [114.0, 119.0]
        assert weekly["Low"].tolist() == [90.0, 95.0]
        assert weekly["Close"].tolist() == [109.0, 114.0]
        assert weekly["Volume"].tolist() == [5100.0, 5350.0]

    def test_partial_week_drop_reuses_weekly_resamples_own_decision(self):
        """The set of week-ending dates kept must be EXACTLY
        TechnicalIndicators.weekly_resample(daily['Close'])'s own index --
        not independently re-derived."""
        daily = _daily_ohlcv(12)
        weekly = resample_ohlcv_weekly(daily)
        close_only = TechnicalIndicators.weekly_resample(daily["Close"])
        assert weekly.index.equals(close_only.index)
        assert weekly["Close"].tolist() == close_only.tolist()

    def test_four_day_holiday_shortened_final_week_is_kept(self):
        """A final week with exactly 4 trading days (the
        weekly_resample() threshold) is kept, not dropped -- 5 full days
        (week1) + 4 days (week2, one holiday short) = 9 bars."""
        daily = _daily_ohlcv(9)  # week1: 5 days, week2 (partial slice above): 4 days
        weekly = resample_ohlcv_weekly(daily)
        assert len(weekly) == 2
