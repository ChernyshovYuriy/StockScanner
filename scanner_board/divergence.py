"""
scanner_board/divergence.py
=============================
Shared pivot-based divergence detector for the Ticker Indicator Board —
Phase 2, see scanner_board/PLAN.md.

Divergences are called out repeatedly in the book as the single strongest
signal type for MACD-Histogram (Ch.23), Stochastic (Ch.26), RSI (Ch.27),
OBV and Accumulation/Distribution (Ch.29), and Force Index (Ch.30). This
module is the ONE shared implementation every one of those Phase 3 columns
calls — never five ad-hoc copies of "compare the last two bottoms."

What a divergence is (book definition, Ch.23): price makes a new
low/high, but the indicator fails to confirm it — tracing a *higher*
bottom (bullish) or a *lower* top (bearish) than it did at the previous
price extreme. "Valid divergences are clearly visible... If you need a
ruler to tell whether there is a divergence, assume there is none" (Ch.26)
— which is exactly why this is one tested function rather than five
subjective eyeball calls.

Pivot spacing (Ch.23, Kerry Lovvorn's research): "the most tradable
divergences occur when the distance between the two peaks or the two
bottoms... is between 20 and 40 bars — and the closer to 20, the better."
That is this module's default min/max bar-gap; callers on weekly bars will
want to pass a smaller pair.

The centerline-crossing requirement is INDICATOR-SPECIFIC, not universal —
easy to get wrong by applying it everywhere or nowhere:
  - MACD-Histogram (Ch.23): "the breaking of the centerline between two
    indicator bottoms is an absolute must for a true divergence... If
    there is no crossover, there is no divergence."
  - Force Index (Ch.30): "for a divergence to be legitimate, this
    indicator must make a new peak, then fall below its zero line, and
    then rise above that line again... If there is no crossover, then
    there is no legitimate divergence."
  - RSI (Ch.27), Stochastic (Ch.26), OBV/A-D (Ch.29): the book states no
    such requirement for these — their divergence rule is only about the
    shape of the two pivots.
So `require_centerline_cross` defaults to False here; Phase 3's
thesis_rules.py passes True only for the MACD-Histogram and Force Index
divergence columns.
"""
from __future__ import annotations

from enum import Enum
from typing import List

import numpy as np
import pandas as pd

# Kerry Lovvorn's research, Ch.23 — see module docstring.
DEFAULT_MIN_BARS_BETWEEN_PIVOTS = 20
DEFAULT_MAX_BARS_BETWEEN_PIVOTS = 40


class DivergenceType(str, Enum):
    """str subclass so this compares/serializes as its own value (e.g. in a
    rendered table cell or a JSON response) without an extra .value access."""
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NONE = "None"


def _find_pivots(series: pd.Series, order: int, kind: str) -> List[int]:
    """Positional indices of local extrema in `series`.

    A bar at position i is a pivot low iff its value is *strictly* lower
    than every one of the `order` bars immediately before it AND every one
    of the `order` bars immediately after it (pivot high: strictly higher).
    This is a standard "fractal" pivot definition, deliberately strict —
    a flat plateau produces no pivot at all rather than an ambiguous one.

    Only interior positions (index >= order and index <= n-1-order) are
    ever evaluated, since confirming a pivot needs `order` bars of history
    on BOTH sides. This is a real, deliberate property, not just an
    implementation shortcut: a pivot literally cannot be confirmed until
    `order` bars after it occurs, so the freshest divergence this module
    can ever report lags the current bar by at least `order` bars.
    """
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    pivots: List[int] = []
    for i in range(order, n - order):
        if np.isnan(vals[i]):
            continue
        left = vals[i - order:i]
        right = vals[i + 1:i + 1 + order]
        if np.isnan(left).any() or np.isnan(right).any():
            continue
        if kind == "low":
            if vals[i] < left.min() and vals[i] < right.min():
                pivots.append(i)
        else:
            if vals[i] > left.max() and vals[i] > right.max():
                pivots.append(i)
    return pivots


def find_pivot_lows(series: pd.Series, order: int = 3) -> List[int]:
    """Positional indices of `series`'s local minima. See _find_pivots()."""
    return _find_pivots(series, order, "low")


def find_pivot_highs(series: pd.Series, order: int = 3) -> List[int]:
    """Positional indices of `series`'s local maxima. See _find_pivots()."""
    return _find_pivots(series, order, "high")


def _gap_in_range(p1: int, p2: int, min_bars: int, max_bars: int) -> bool:
    gap = p2 - p1
    return min_bars <= gap <= max_bars


def _crosses(indicator: pd.Series, p1: int, p2: int, centerline: float, above: bool) -> bool:
    """True if `indicator` is on the `above` side of `centerline` at any
    bar strictly between the two pivots p1 and p2 (exclusive of both) —
    the book's "breaking of the centerline between two indicator
    bottoms/tops" test."""
    between = indicator.iloc[p1 + 1:p2]
    if between.empty:
        return False
    return bool((between > centerline).any() if above else (between < centerline).any())


def _check_bullish(price: pd.Series, indicator: pd.Series, lows: List[int],
                    min_bars: int, max_bars: int,
                    require_centerline_cross: bool, centerline: float) -> bool:
    if len(lows) < 2:
        return False
    p1, p2 = lows[-2], lows[-1]
    if not _gap_in_range(p1, p2, min_bars, max_bars):
        return False
    price1, price2 = price.iloc[p1], price.iloc[p2]
    ind1, ind2 = indicator.iloc[p1], indicator.iloc[p2]
    if pd.isna(price1) or pd.isna(price2) or pd.isna(ind1) or pd.isna(ind2):
        return False
    if not (price2 < price1):  # price: a lower low
        return False
    if not (ind2 > ind1):  # indicator: a higher (more shallow) bottom
        return False
    if require_centerline_cross and not _crosses(indicator, p1, p2, centerline, above=True):
        return False
    return True


def _check_bearish(price: pd.Series, indicator: pd.Series, highs: List[int],
                    min_bars: int, max_bars: int,
                    require_centerline_cross: bool, centerline: float) -> bool:
    if len(highs) < 2:
        return False
    p1, p2 = highs[-2], highs[-1]
    if not _gap_in_range(p1, p2, min_bars, max_bars):
        return False
    price1, price2 = price.iloc[p1], price.iloc[p2]
    ind1, ind2 = indicator.iloc[p1], indicator.iloc[p2]
    if pd.isna(price1) or pd.isna(price2) or pd.isna(ind1) or pd.isna(ind2):
        return False
    if not (price2 > price1):  # price: a higher high
        return False
    if not (ind2 < ind1):  # indicator: a lower (weaker) top
        return False
    if require_centerline_cross and not _crosses(indicator, p1, p2, centerline, above=False):
        return False
    return True


def detect_divergence(price: pd.Series, indicator: pd.Series, *,
                       pivot_order: int = 3,
                       min_bars_between_pivots: int = DEFAULT_MIN_BARS_BETWEEN_PIVOTS,
                       max_bars_between_pivots: int = DEFAULT_MAX_BARS_BETWEEN_PIVOTS,
                       require_centerline_cross: bool = False,
                       centerline: float = 0.0) -> DivergenceType:
    """Classify the most recent divergence, if any, between `price` and
    `indicator` (same index/length; `indicator` may be RSI, Stochastic
    %K/%D, MACD-Histogram, OBV, A/D, or Force Index).

    Uses the last two pivot lows in `price` for a Bullish check and the
    last two pivot highs for a Bearish check (independently — see each
    chapter's own definition in the module docstring). Bullish wins ties
    if somehow both patterns are found (in practice price cannot
    simultaneously be tracing both a fresh lower-low and a fresh
    higher-high at the same right edge, so this is only a tie-break rule
    on paper).

    `require_centerline_cross`/`centerline`: see module docstring — pass
    True only for MACD-Histogram and Force Index (Ch.23/Ch.30's explicit
    "no crossover, no divergence" rule); leave False for RSI, Stochastic,
    OBV, and A/D.
    """
    lows = find_pivot_lows(price, pivot_order)
    highs = find_pivot_highs(price, pivot_order)

    if _check_bullish(price, indicator, lows, min_bars_between_pivots,
                       max_bars_between_pivots, require_centerline_cross, centerline):
        return DivergenceType.BULLISH
    if _check_bearish(price, indicator, highs, min_bars_between_pivots,
                       max_bars_between_pivots, require_centerline_cross, centerline):
        return DivergenceType.BEARISH
    return DivergenceType.NONE
