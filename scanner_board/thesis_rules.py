"""
scanner_board/thesis_rules.py
===============================
Book-thesis -> labeled-column functions for the Ticker Indicator Board —
Phase 3, see scanner_board/PLAN.md.

One function per labeled column, each docstring citing its chapter. Every
function reuses Phase 1/2's shared primitives (classify_slope,
detect_divergence, TechnicalIndicators) rather than re-deriving a slope or
divergence check of its own — the single-shared-logic rule scanner_board/
PLAN.md commits to.

Convention shared by every "reading" function below: it takes the full
pd.Series (never a bare scalar) and reads its own latest non-NaN-aware
value(s) — this matches classify_slope's and detect_divergence's own
Series-in signature, so a Phase 4 caller never has to remember which
functions want a scalar and which want a window. Insufficient/NaN data
resolves to each function's own safe, neutral label (Neutral/Normal/Flat/
Unclear) rather than raising — the same convention Phase 1's
classify_slope already established for insufficient data.

Not covered here (deferred, see scanner_board/PLAN.md):
  - Weinstein Stage: reuses the existing canadian_stock_screener.py /
    market-stage-detection StageDetector pattern directly in Phase 4's
    pipeline, not a new thesis_rules function.
  - ATR / ATR% / +-1/2/3 ATR bands (Ch.24): plain arithmetic on
    TechnicalIndicators.atr()'s own output (`ma +- k*atr`), not a labeled
    classification — assembled inline in Phase 4, no dedicated function
    needed.
  - Impulse System color and Triple Screen alignment (Ch.39/40): in the
    sibling module scanner_board/triple_screen.py, since both are
    inherently multi-timeframe/multi-indicator combinations rather than a
    single-series reading.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd

from canadian_stock_screener import TechnicalIndicators
from scanner_board.slope import classify_slope, SlopeDirection


# ─────────────────────────────────────────────────────────────────────────────
# Trend / Moving Averages (Ch.22)
# ─────────────────────────────────────────────────────────────────────────────

class PriceVsMA(str, Enum):
    ABOVE = "Above"
    BELOW = "Below"
    AT = "At"


def price_vs_ma(price: pd.Series, ma: pd.Series) -> PriceVsMA:
    """Ch.22: "When prices rise above a moving average, the crowd is more
    bullish than before. When prices fall below a moving average, the
    crowd is more bearish than before." """
    p, m = price.iloc[-1], ma.iloc[-1]
    if pd.isna(p) or pd.isna(m):
        return PriceVsMA.AT
    if p > m:
        return PriceVsMA.ABOVE
    if p < m:
        return PriceVsMA.BELOW
    return PriceVsMA.AT


class ValueZonePosition(str, Enum):
    ABOVE = "Above"
    IN_ZONE = "In Zone"
    BELOW = "Below"


def value_zone_position(price: pd.Series, fast_ema: pd.Series, slow_ema: pd.Series) -> ValueZonePosition:
    """Ch.22: "Value 'lives' in the zone between the two moving
    averages... When looking to buy a stock, it pays to do it in the
    value zone, rather than overpay and buy above value. Similarly, when
    shorting, it pays to wait for a rally into the value zone." """
    p, f, s = price.iloc[-1], fast_ema.iloc[-1], slow_ema.iloc[-1]
    if pd.isna(p) or pd.isna(f) or pd.isna(s):
        return ValueZonePosition.IN_ZONE
    lo, hi = (f, s) if f <= s else (s, f)
    if p > hi:
        return ValueZonePosition.ABOVE
    if p < lo:
        return ValueZonePosition.BELOW
    return ValueZonePosition.IN_ZONE


# ─────────────────────────────────────────────────────────────────────────────
# MACD (Ch.23)
# ─────────────────────────────────────────────────────────────────────────────

class MACDCross(str, Enum):
    BULL = "Bull"
    BEAR = "Bear"
    NEUTRAL = "Neutral"


def macd_cross(macd_line: pd.Series, signal_line: pd.Series) -> MACDCross:
    """Ch.23: "When the fast MACD line rises above the slow Signal line,
    it shows that bulls dominate the market, and it is better to trade
    from the long side. When the fast line falls below the slow line, it
    shows that bears dominate the market and it pays to trade from the
    short side." """
    m, s = macd_line.iloc[-1], signal_line.iloc[-1]
    if pd.isna(m) or pd.isna(s):
        return MACDCross.NEUTRAL
    if m > s:
        return MACDCross.BULL
    if m < s:
        return MACDCross.BEAR
    return MACDCross.NEUTRAL


class TrendHealth(str, Enum):
    SAFE = "Safe"
    WEAK = "Weak"
    UNCLEAR = "Unclear"


def trend_health_from_slopes(price_slope: SlopeDirection, histogram_slope: SlopeDirection) -> TrendHealth:
    """The flagship thesis this Board was commissioned to encode, Ch.23:
    "When the slope of MACD-Histogram moves in the same direction as
    prices, the trend is safe. When the slope of MACD-Histogram moves in
    a direction opposite to that of prices, the health of the trend is in
    question." """
    if SlopeDirection.FLAT in (price_slope, histogram_slope):
        return TrendHealth.UNCLEAR
    if price_slope == histogram_slope:
        return TrendHealth.SAFE
    return TrendHealth.WEAK


def trend_health(price: pd.Series, histogram: pd.Series, period: int = 5) -> TrendHealth:
    """Convenience wrapper over trend_health_from_slopes(), classifying
    both series via classify_slope (not a re-derived slope check).
    `period` default of 5 matches this codebase's own existing hist_slope
    convention (ScoreCalculator.score_macd's `histogram.iloc[-5:]` slope
    window)."""
    return trend_health_from_slopes(classify_slope(price, period), classify_slope(histogram, period))


class ExtremeReading(str, Enum):
    NEW_HIGH = "New High"
    NEW_LOW = "New Low"
    NONE = "No"


def macd_histogram_extreme(histogram: pd.Series, lookback: int = 63) -> ExtremeReading:
    """Ch.23: "A record peak for the past three months of daily
    MACD-Histogram shows that bulls are very strong and prices are likely
    to rise even higher. A record new low for MACD-Histogram for the past
    three months shows that bears are very strong and lower prices are
    likely ahead." Three months ~= 63 trading days (21 * 3), this
    function's default lookback."""
    last = histogram.iloc[-1]
    window = histogram.iloc[-lookback:].dropna()
    if pd.isna(last) or window.empty:
        return ExtremeReading.NONE
    if last >= window.max():
        return ExtremeReading.NEW_HIGH
    if last <= window.min():
        return ExtremeReading.NEW_LOW
    return ExtremeReading.NONE


# ─────────────────────────────────────────────────────────────────────────────
# Directional System / ADX (Ch.24)
# ─────────────────────────────────────────────────────────────────────────────

class DIBias(str, Enum):
    BULL = "Bull"
    BEAR = "Bear"
    NEUTRAL = "Neutral"


def di_bias(plus_di: pd.Series, minus_di: pd.Series) -> DIBias:
    """Ch.24: "When +DI13 is on top, it shows that the trend is up, and
    when -DI13 is on top, it shows that the trend is down... It pays to
    trade with the upper Directional line." """
    p, m = plus_di.iloc[-1], minus_di.iloc[-1]
    if pd.isna(p) or pd.isna(m):
        return DIBias.NEUTRAL
    if p > m:
        return DIBias.BULL
    if m > p:
        return DIBias.BEAR
    return DIBias.NEUTRAL


class ADXRegime(str, Enum):
    CHOPPY = "Choppy"
    WAKING_UP = "Waking Up"
    TRENDING = "Trending"
    OVERHEATED = "Overheated"
    UNKNOWN = "Unknown"


def _trailing_true_run_start(flags: pd.Series) -> Optional[int]:
    """Positional start index of the contiguous True run in `flags` ending
    at its last position, or None if the last position itself is False
    (or `flags` is empty)."""
    vals = flags.to_numpy()
    n = len(vals)
    if n == 0 or not bool(vals[-1]):
        return None
    i = n - 1
    while i > 0 and bool(vals[i - 1]):
        i -= 1
    return i


def adx_regime(adx: pd.Series, plus_di: pd.Series, minus_di: pd.Series,
               wake_threshold: float = 4.0) -> ADXRegime:
    """Ch.24:
      - ADX below both Directional lines: "a flat, sleepy market. Do not
        use a trend-following system but get ready to trade, because
        major trends emerge from such lulls." -> Choppy.
      - "When ADX rises by four steps (i.e., from 9 to 13) from its
        lowest point below both Directional lines, it 'rings a bell' on a
        new trend." -> Waking Up (a Choppy reading whose ADX has since
        risen `wake_threshold`+ points off the low of its current
        below-both-lines streak, while still below both lines).
      - ADX above both Directional lines: "an overheated market" (a turn
        down from there means "the major trend has stumbled" and it's
        time to take profits) -> Overheated.
      - Otherwise, ADX sandwiched between the two DI lines: an ordinary,
        established Trending market.
    """
    if pd.isna(adx.iloc[-1]) or pd.isna(plus_di.iloc[-1]) or pd.isna(minus_di.iloc[-1]):
        return ADXRegime.UNKNOWN

    last_adx = adx.iloc[-1]
    last_lo = min(plus_di.iloc[-1], minus_di.iloc[-1])
    last_hi = max(plus_di.iloc[-1], minus_di.iloc[-1])

    if last_adx < last_lo:
        below_both = adx < pd.concat([plus_di, minus_di], axis=1).min(axis=1)
        streak_start = _trailing_true_run_start(below_both)
        if streak_start is not None:
            streak_min = adx.iloc[streak_start:].min()
            if last_adx - streak_min >= wake_threshold:
                return ADXRegime.WAKING_UP
        return ADXRegime.CHOPPY
    if last_adx > last_hi:
        return ADXRegime.OVERHEATED
    return ADXRegime.TRENDING


# ─────────────────────────────────────────────────────────────────────────────
# Oscillators (Ch.25, 26, 27) — shared 5%-rule reference lines
# ─────────────────────────────────────────────────────────────────────────────

class OscillatorZone(str, Enum):
    OVERBOUGHT = "Overbought"
    OVERSOLD = "Oversold"
    NEUTRAL = "Neutral"


def five_percent_reference_lines(series: pd.Series, lookback: int = 120) -> tuple:
    """Ch.25: "Place those lines so that they cut across only the highest
    peaks and the lowest valleys of that oscillator for the past six
    months. The proper way to draw those lines is to place them so that
    an oscillator spends only about 5 percent of its time beyond each
    line." Ch.27 repeats this specifically for RSI: "Use the 5 percent
    rule: draw each line at a level beyond which RSI has spent less than
    5 percent of its time in the past 4 to 6 months. Adjust reference
    lines once every three months." `lookback` default of 120 trading
    days sits at the middle of that "4 to 6 months" window. Returns
    (lower, upper) as the 5th/95th percentile of the trailing window —
    dynamic, per-ticker reference lines rather than a hardcoded universal
    80/20 or 70/30.
    """
    window = series.iloc[-lookback:].dropna()
    if window.empty:
        return (float("nan"), float("nan"))
    return (float(window.quantile(0.05)), float(window.quantile(0.95)))


def oscillator_zone(series: pd.Series, lookback: int = 120) -> OscillatorZone:
    """Shared by both the Stochastic Zone and RSI Zone columns (Ch.26/27
    both apply the identical overbought/oversold reasoning to their own
    bounded oscillator) — one implementation, not two copies with
    different hardcoded thresholds. Reference lines from
    five_percent_reference_lines(), not a fixed 80/20 or 70/30."""
    lower, upper = five_percent_reference_lines(series, lookback)
    last = series.iloc[-1]
    if pd.isna(last) or pd.isna(lower) or pd.isna(upper):
        return OscillatorZone.NEUTRAL
    if last >= upper:
        return OscillatorZone.OVERBOUGHT
    if last <= lower:
        return OscillatorZone.OVERSOLD
    return OscillatorZone.NEUTRAL


# ─────────────────────────────────────────────────────────────────────────────
# Volume-based (Ch.28, 29, 30)
# ─────────────────────────────────────────────────────────────────────────────

class VolumeLevel(str, Enum):
    HIGH = "High"
    LOW = "Low"
    NORMAL = "Normal"


def volume_vs_average(volume: pd.Series, lookback: int = 10, threshold_pct: float = 25.0) -> VolumeLevel:
    """Ch.28: "'high volume' for any given market is at least 25 percent
    above its average for the past two weeks, while 'low volume' is at
    least 25 percent below average." Two weeks = 10 trading days, this
    function's default lookback. Reuses TechnicalIndicators.sma for the
    average rather than a fresh rolling-mean implementation; the average
    is computed over the `lookback` days BEFORE today (shift(1)) so
    today's own volume isn't 1/lookback of the baseline it's being
    compared against."""
    avg = TechnicalIndicators.sma(volume, lookback).shift(1)
    last_v, last_avg = volume.iloc[-1], avg.iloc[-1]
    if pd.isna(last_v) or pd.isna(last_avg) or last_avg == 0:
        return VolumeLevel.NORMAL
    pct = (last_v - last_avg) / last_avg * 100
    if pct >= threshold_pct:
        return VolumeLevel.HIGH
    if pct <= -threshold_pct:
        return VolumeLevel.LOW
    return VolumeLevel.NORMAL


class ForceZone(str, Enum):
    NEGATIVE = "Negative"
    POSITIVE = "Positive"
    ZERO = "Zero"


def force_short_term_zone(force_short: pd.Series) -> ForceZone:
    """Ch.30, Short-Term Force Index (2-day EMA): a negative reading
    during an uptrend is where Elder buys pullbacks ("Buy when a 2-day
    EMA of Force Index turns negative during uptrends"); a positive
    reading during a downtrend is where he shorts rallies ("Sell short
    when a 2-day EMA of Force Index turns positive in a downtrend"). This
    column reports only the raw sign — pair it with the row's own trend
    column (e.g. Impulse or Triple Screen) to read it as a pullback-buy
    or pullback-short zone, the way the book itself insists: "as long as
    you trade in the direction of the trend." """
    last = force_short.iloc[-1]
    if pd.isna(last):
        return ForceZone.ZERO
    if last < 0:
        return ForceZone.NEGATIVE
    if last > 0:
        return ForceZone.POSITIVE
    return ForceZone.ZERO


class ForceBias(str, Enum):
    BULL = "Bull"
    BEAR = "Bear"
    NEUTRAL = "Neutral"


def force_long_term_bias(force_long: pd.Series) -> ForceBias:
    """Ch.30, Intermediate-Term Force Index (13-day EMA): "When it rises
    above zero, the bulls are stronger, and when it falls below zero, the
    bears are in charge." """
    last = force_long.iloc[-1]
    if pd.isna(last):
        return ForceBias.NEUTRAL
    if last > 0:
        return ForceBias.BULL
    if last < 0:
        return ForceBias.BEAR
    return ForceBias.NEUTRAL
