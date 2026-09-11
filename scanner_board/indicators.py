"""
scanner_board/indicators.py
============================
Phase 1 raw indicator math for the Ticker Indicator Board — see
scanner_board/PLAN.md.

Single-shared-logic rule: every indicator already implemented in
canadian_stock_screener.TechnicalIndicators (sma, ema, rsi, macd,
true_range, atr, adx, obv, linear_regression_slope, weekly_resample) is
reused by IMPORT wherever this module or a future scanner_board module
needs it — never re-derived. Only the three indicators below are new to
this codebase (Stochastic, Accumulation/Distribution, Force Index), and
even those are built out of TechnicalIndicators.sma/.ema for their moving-
average smoothing steps rather than a second moving-average
implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from canadian_stock_screener import TechnicalIndicators


@dataclass(frozen=True)
class StochasticResult:
    """Slow Stochastic (Ch.26). percent_k is the %D of Fast Stochastic,
    which the book's own construction promotes to "%K of Slow Stochastic";
    percent_d is that series smoothed once more."""
    percent_k: pd.Series
    percent_d: pd.Series


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 5, d_period: int = 3) -> StochasticResult:
    """Slow Stochastic oscillator (Ch.26).

    Raw ("Fast") %K = (Close - LowN) / (HighN - LowN) * 100 over a rolling
    N-bar window (book default N=5 — "the standard width of Stochastic's
    time window is 5 days").

    The book gives one specific formula for smoothing %K into %D (a ratio
    of two summed differences) but explicitly hedges: "It can be done in
    several ways, such as: [that formula]." This implementation smooths
    with the plain N-bar simple moving average of %K instead — reusing
    TechnicalIndicators.sma rather than adding a second moving-average
    implementation — which is both what most charting software does and
    avoids the book's sum-of-differences formula needing every term in the
    window to be non-NaN (a stricter warm-up requirement) for no
    book-fidelity cost, since the book itself doesn't insist on its own
    formula.

    Slow Stochastic — which Elder says most traders prefer, for fewer
    whipsaws than Fast Stochastic — treats Fast %D as Slow %K and smooths
    it once more into Slow %D: "The %D of Fast Stochastic becomes the %K of
    Slow Stochastic and is smoothed by repeating step 2."
    """
    low_n = low.rolling(k_period, min_periods=k_period).min()
    high_n = high.rolling(k_period, min_periods=k_period).max()
    fast_k = (close - low_n) / (high_n - low_n) * 100

    fast_d = TechnicalIndicators.sma(fast_k, d_period)
    slow_k = fast_d
    slow_d = TechnicalIndicators.sma(slow_k, d_period)
    return StochasticResult(percent_k=slow_k, percent_d=slow_d)


def accumulation_distribution(open_: pd.Series, high: pd.Series, low: pd.Series,
                               close: pd.Series, volume: pd.Series) -> pd.Series:
    """Accumulation/Distribution (Ch.29): a running total that credits bulls
    or bears with only the fraction of a day's volume proportional to where
    the close fell versus the open, inside that day's own high-low range.

        A/D_today = (Close - Open) / (High - Low) * Volume

    then a cumulative sum ("a running total of each day's A/D creates a
    cumulative A/D indicator").

    A `High == Low` bar (e.g. a halted, unmoving print) makes that ratio a
    division by zero, which the book's own formula does not guard against.
    Treated as a zero contribution for that one bar — not a crash, and not
    an Inf that would poison every later cumulative value — since a bar
    with zero range gives bulls and bears an equally undefined, i.e. zero,
    edge for the day.
    """
    day_range = high - low
    ratio = (close - open_) / day_range.replace(0, np.nan)
    contribution = (ratio * volume).fillna(0.0)
    return contribution.cumsum()


@dataclass(frozen=True)
class ForceIndexResult:
    """Force Index (Ch.30). `raw` is the unsmoothed, jagged series; `short`
    (book default 2-day EMA) "pinpoints entry and exit points"; `long`
    (book default 13-day EMA) "tracks longer-term changes in the force of
    bulls and bears... confirm[s] trends and recognize[s] important
    reversals."""
    raw: pd.Series
    short: pd.Series
    long: pd.Series


def force_index(close: pd.Series, volume: pd.Series,
                 short_period: int = 2, long_period: int = 13) -> ForceIndexResult:
    """Force Index (Ch.30): `Volume_today * (Close_today - Close_yesterday)`.
    "It brings together three essential pieces of information — the
    direction of price change, its extent, and the volume during that
    change." Elder smooths the raw series with two EMAs to make its
    signals usable — reusing TechnicalIndicators.ema rather than a fresh
    moving-average implementation.
    """
    raw = volume * close.diff()
    return ForceIndexResult(
        raw=raw,
        short=TechnicalIndicators.ema(raw, short_period),
        long=TechnicalIndicators.ema(raw, long_period),
    )
