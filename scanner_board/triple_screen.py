"""
scanner_board/triple_screen.py
================================
Impulse System color (Ch.40) and Triple Screen weekly/daily alignment
(Ch.39) — Phase 3, see scanner_board/PLAN.md.

Both are Elder's own published decision tables, reproduced here verbatim
rather than invented — this module is deliberately "no smarter" than the
book. This is the closest thing the Board ever shows to a verdict, and
it's Elder's verdict, not a synthesized one (see scanner_board/PLAN.md's
"Explicitly out of scope" section on why there's no other composite
signal).

Both the Impulse color and the Triple Screen trend read are built on
classify_slope (scanner_board/slope.py) — the same slope classification
already used for the Board's plain MA Slope column — rather than either
one re-deriving its own notion of "rising."
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple

import pandas as pd

from scanner_board.slope import classify_slope, SlopeDirection

# Ch.40: "the relationship between any two neighboring bars" — both the
# Impulse System's inertia (EMA slope) and power (MACD-Histogram slope)
# reads are bar-to-bar, matching MACD-Histogram's own slope definition in
# Ch.23 ("defined by the relationship between any two neighboring bars"),
# not a longer regression window.
IMPULSE_SLOPE_PERIOD = 2


class ImpulseColor(str, Enum):
    GREEN = "Green"
    RED = "Red"
    BLUE = "Blue"


_IMPULSE_TABLE: Dict[Tuple[SlopeDirection, SlopeDirection], ImpulseColor] = {
    (SlopeDirection.RISING, SlopeDirection.RISING): ImpulseColor.GREEN,
    (SlopeDirection.FALLING, SlopeDirection.FALLING): ImpulseColor.RED,
}


def impulse_color_from_slopes(ema_slope: SlopeDirection, macd_hist_slope: SlopeDirection) -> ImpulseColor:
    """Ch.40's own 4-row table, reproduced directly:
      EMA rising & MACD-H rising -> Green (bullish; "shorting prohibited,
        buying or standing aside permitted").
      EMA falling & MACD-H falling -> Red (bearish; "buying prohibited,
        shorting or standing aside permitted").
      Any other combination (including either reading Flat) -> Blue
      (neutral; "nothing is prohibited")."""
    return _IMPULSE_TABLE.get((ema_slope, macd_hist_slope), ImpulseColor.BLUE)


def impulse_color(ema: pd.Series, macd_histogram: pd.Series,
                   period: int = IMPULSE_SLOPE_PERIOD) -> ImpulseColor:
    """Convenience wrapper: classifies both slopes via
    scanner_board.slope.classify_slope (not a re-derived slope check) and
    looks up the color via impulse_color_from_slopes()."""
    ema_slope = classify_slope(ema, period)
    hist_slope = classify_slope(macd_histogram, period)
    return impulse_color_from_slopes(ema_slope, hist_slope)


class TrendDirection(str, Enum):
    UP = "Up"
    DOWN = "Down"
    FLAT = "Flat"


_TREND_FROM_SLOPE = {
    SlopeDirection.RISING: TrendDirection.UP,
    SlopeDirection.FALLING: TrendDirection.DOWN,
    SlopeDirection.FLAT: TrendDirection.FLAT,
}


def trend_direction(ema: pd.Series, period: int) -> TrendDirection:
    """The Up/Down/Flat trend read Triple Screen's own summary table
    (Ch.39) uses for both "Weekly Trend" and "Daily Trend" — built on
    classify_slope, the same slope classification used everywhere else on
    the Board, not a second trend definition. `ema` would typically be the
    weekly or daily fast EMA (Ch.39 switched the first screen's
    trend-following tool over time between weekly MACD-Histogram slope,
    weekly EMA slope, and the Impulse System — the EMA-slope reading is
    used here as the plain Up/Down/Flat primitive both Impulse and Triple
    Screen are built from)."""
    return _TREND_FROM_SLOPE[classify_slope(ema, period)]


class TripleScreenAlignment(str, Enum):
    STAND_ASIDE = "Stand aside"
    GO_LONG_SETUP = "Go long setup"
    GO_SHORT_SETUP = "Go short setup"


_TRIPLE_SCREEN_TABLE: Dict[Tuple[TrendDirection, TrendDirection], TripleScreenAlignment] = {
    (TrendDirection.UP, TrendDirection.UP): TripleScreenAlignment.STAND_ASIDE,
    (TrendDirection.UP, TrendDirection.DOWN): TripleScreenAlignment.GO_LONG_SETUP,
    (TrendDirection.DOWN, TrendDirection.DOWN): TripleScreenAlignment.STAND_ASIDE,
    (TrendDirection.DOWN, TrendDirection.UP): TripleScreenAlignment.GO_SHORT_SETUP,
}


def triple_screen_alignment(weekly_trend: TrendDirection, daily_trend: TrendDirection) -> TripleScreenAlignment:
    """Ch.39's own published summary table, reproduced verbatim:

        Weekly Trend | Daily Trend | Action
        Up           | Up          | Stand aside
        Up           | Down        | Go long setup   (EMA penetration or upside breakout)
        Down         | Down        | Stand aside
        Down         | Up          | Go short setup  (EMA penetration or downside breakout)

    Any combination involving a Flat weekly or daily trend falls through
    to Stand aside — not one of the book's own four rows, but consistent
    with Ch.22's own rule 3: "When the EMA goes flat and only wiggles a
    little, it identifies an aimless, trendless market. Do not trade
    using a trend-following method." """
    return _TRIPLE_SCREEN_TABLE.get((weekly_trend, daily_trend), TripleScreenAlignment.STAND_ASIDE)
