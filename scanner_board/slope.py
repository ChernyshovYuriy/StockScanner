"""
scanner_board/slope.py
=======================
Slope classification (Rising / Falling / Flat) for the Ticker Indicator
Board — Phase 1, see scanner_board/PLAN.md.

Book citation (Ch.22, Moving Averages): "The most important message of a
moving average comes from the direction of its slope... When it rises, it
shows that the crowd is becoming more optimistic — bullish. When it falls,
it shows that the crowd is becoming more pessimistic — bearish." The same
slope-direction reasoning is applied throughout the Board to MACD-Histogram
(Ch.23), ADX (Ch.24), and OBV/A-D/Force Index (Ch.28-30) — this module is
the one place that Rising/Falling/Flat decision gets made, so every column
on the Board means the same thing by "rising."

Single-shared-logic rule: this does NOT re-derive a slope calculation.
canadian_stock_screener.TechnicalIndicators.linear_regression_slope already
computes a mean-normalized linear-regression slope (scale-invariant across
a $5 stock and a $500 stock) and already has 20+ characterization/property
tests locking its edge-case behaviour (too few points, a flat/degenerate
window, mean-sign-independence). classify_slope() is a thin label wrapper
around that single existing implementation, not a second one.
"""
from __future__ import annotations

from enum import Enum

import pandas as pd

from canadian_stock_screener import TechnicalIndicators

# Matches the threshold already used for this exact normalized-slope
# quantity elsewhere in this codebase (ScoreCalculator.score_macd's
# hist_slope check, score_stage2's ma30_slope check) — kept identical
# rather than invented fresh, so "Rising" on the Board means the same
# magnitude of move that "rising" already means in the screener.
FLAT_THRESHOLD = 0.001


class SlopeDirection(str, Enum):
    """str subclass so this compares/serializes as its own value (e.g. in a
    rendered table cell or a JSON response) without an extra .value access."""
    RISING = "Rising"
    FALLING = "Falling"
    FLAT = "Flat"


def classify_slope(series: pd.Series, period: int,
                    threshold: float = FLAT_THRESHOLD) -> SlopeDirection:
    """Classify the recent direction of `series` over its trailing `period`
    bars into Rising / Falling / Flat.

    `series` is typically a moving average, MACD-Histogram, ADX, OBV, A/D,
    or Force Index series — anything the book evaluates "by the direction
    of its slope." `period` is the trailing window handed to
    linear_regression_slope (e.g. Ch.22's own dual-EMA convention: 22/11 or
    26/13 for the moving averages themselves; a shorter window such as 5
    for MACD-Histogram's bar-to-bar slope, Ch.23).

    linear_regression_slope() returns exactly 0.0 both for a genuinely flat
    window AND for insufficient data (fewer than `period` points, or every
    value NaN) — both cases fall through the two threshold checks below and
    classify as Flat, which is the safe/neutral reading for either case.
    """
    slope = TechnicalIndicators.linear_regression_slope(series, period)
    if slope > threshold:
        return SlopeDirection.RISING
    if slope < -threshold:
        return SlopeDirection.FALLING
    return SlopeDirection.FLAT
