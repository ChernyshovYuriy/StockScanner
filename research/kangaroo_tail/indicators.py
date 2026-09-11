"""
Pure indicator math for the Kangaroo Tail detector. Kept dependency-light
(pandas only) so it's independently unit-testable, same rationale as
research/triple_screen/indicators.py.

Every function here defensively sorts its input by index ascending before
reading off "latest"/"prior" bars -- bars are documented as sorted
ascending wherever they're passed in, but nothing upstream enforces that;
research/triple_screen's own stress testing found a reverse-chronological
feed silently reads as if the oldest bar were newest without this guard
(see TRIPLE_SCREEN_STRESS_FINDINGS.md), so the same defense is applied
here from the start.
"""
import pandas as pd


def wilder_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's smoothed ATR (alpha = 1/period) -- NOT ewm(span=period),
    which gives a faster-decaying, ~30-50% higher-reading average (see
    auto_pipeline.py's _atr() and position_monitor.py's wilder_atr(), both
    of which carry this same fix already; reimplemented independently here
    rather than imported since this research package stays self-contained,
    matching research/triple_screen's own convention).
    """
    clean = bars.sort_index()
    high, low, close = clean["High"], clean["Low"], clean["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


def rolling_prior_low(bars: pd.DataFrame, lookback: int) -> pd.Series:
    """The rolling low of the `lookback` bars strictly BEFORE each bar
    (today's own low never enters its own baseline -- a self-inflating
    "prior low" would make the structure-break check trivially easy to
    pass). Value at index i covers bars [i-lookback, i-1]; the first
    `lookback` rows are NaN (insufficient history).
    """
    clean = bars.sort_index()
    return clean["Low"].shift(1).rolling(lookback).min()
