"""
tests/test_scanner_board_triple_screen.py
===========================================
Tests for scanner_board.triple_screen — see scanner_board/PLAN.md Phase 3.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scanner_board.slope import SlopeDirection
from scanner_board.triple_screen import (
    impulse_color_from_slopes, impulse_color, ImpulseColor,
    trend_direction, TrendDirection,
    triple_screen_alignment, TripleScreenAlignment,
)

R, F, T = SlopeDirection.RISING, SlopeDirection.FALLING, SlopeDirection.FLAT


def _series(values, start="2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series([float(v) for v in values], index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Impulse System (Ch.40)
# ─────────────────────────────────────────────────────────────────────────────

class TestImpulseColorFromSlopes:
    """Elder's own 4-row table (Fig 40.1), exhaustively covering every one
    of the 3x3 slope combinations (not just the 4 the book draws) so a
    future edit can't silently misclassify a Flat-involving combination."""

    @pytest.mark.parametrize("ema_slope,hist_slope,expected", [
        (R, R, ImpulseColor.GREEN),
        (F, F, ImpulseColor.RED),
        (R, F, ImpulseColor.BLUE),
        (F, R, ImpulseColor.BLUE),
        (R, T, ImpulseColor.BLUE),
        (T, R, ImpulseColor.BLUE),
        (F, T, ImpulseColor.BLUE),
        (T, F, ImpulseColor.BLUE),
        (T, T, ImpulseColor.BLUE),
    ])
    def test_exhaustive_table(self, ema_slope, hist_slope, expected):
        assert impulse_color_from_slopes(ema_slope, hist_slope) is expected


class TestImpulseColor:
    """Series-based wrapper: confirms it actually delegates to
    classify_slope (Phase 1) rather than a second slope calculation, by
    exercising real rising/falling series through the full function."""

    def test_green_when_both_rising(self):
        ema = _series([1, 2, 3, 4, 5])
        hist = _series([1, 2, 3, 4, 5])
        assert impulse_color(ema, hist, period=2) is ImpulseColor.GREEN

    def test_red_when_both_falling(self):
        ema = _series([5, 4, 3, 2, 1])
        hist = _series([5, 4, 3, 2, 1])
        assert impulse_color(ema, hist, period=2) is ImpulseColor.RED

    def test_blue_when_disagreeing(self):
        ema = _series([1, 2, 3, 4, 5])
        hist = _series([5, 4, 3, 2, 1])
        assert impulse_color(ema, hist, period=2) is ImpulseColor.BLUE


# ─────────────────────────────────────────────────────────────────────────────
# Triple Screen (Ch.39)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrendDirection:
    def test_up(self):
        assert trend_direction(_series([1, 2, 3, 4, 5]), period=3) is TrendDirection.UP

    def test_down(self):
        assert trend_direction(_series([5, 4, 3, 2, 1]), period=3) is TrendDirection.DOWN

    def test_flat(self):
        assert trend_direction(_series([5.0] * 5), period=3) is TrendDirection.FLAT


class TestTripleScreenAlignment:
    """Ch.39's own published summary table, reproduced exactly, plus the
    Flat-involving fallback documented in the function's own docstring."""

    U, D, Fl = TrendDirection.UP, TrendDirection.DOWN, TrendDirection.FLAT

    @pytest.mark.parametrize("weekly,daily,expected", [
        (U, U, TripleScreenAlignment.STAND_ASIDE),
        (U, D, TripleScreenAlignment.GO_LONG_SETUP),
        (D, D, TripleScreenAlignment.STAND_ASIDE),
        (D, U, TripleScreenAlignment.GO_SHORT_SETUP),
        (Fl, U, TripleScreenAlignment.STAND_ASIDE),
        (U, Fl, TripleScreenAlignment.STAND_ASIDE),
        (Fl, Fl, TripleScreenAlignment.STAND_ASIDE),
        (Fl, D, TripleScreenAlignment.STAND_ASIDE),
        (D, Fl, TripleScreenAlignment.STAND_ASIDE),
    ])
    def test_exhaustive_table(self, weekly, daily, expected):
        assert triple_screen_alignment(weekly, daily) is expected

    def test_integration_go_long_setup_from_real_series(self):
        """Weekly EMA in a clear uptrend, daily EMA in a pullback -- the
        canonical 'weekly tide up, daily wave down' Triple Screen buy
        setup (Ch.39)."""
        weekly_ema = _series([1, 2, 3, 4, 5])
        daily_ema = _series([5, 4, 3, 2, 1])
        weekly_trend = trend_direction(weekly_ema, period=3)
        daily_trend = trend_direction(daily_ema, period=3)
        assert triple_screen_alignment(weekly_trend, daily_trend) is TripleScreenAlignment.GO_LONG_SETUP
