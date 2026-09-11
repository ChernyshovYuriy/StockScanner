"""
scanner_board/row.py
======================
Assembles one Ticker Indicator Board row from already-fetched daily and
weekly OHLCV DataFrames — Phase 4, see scanner_board/PLAN.md. Pure
function, no fetching, no DB — kept separate from scanner_pipeline.py so
the column-assembly logic is unit-testable without a live data fetch.

Every computed value funnels through Phase 1-3's shared functions
(classify_slope, detect_divergence, TechnicalIndicators, the
scanner_board.indicators/thesis_rules/triple_screen modules) — this module
adds no new indicator math of its own, only the wiring: which series feeds
which function, and at what period.

Parameter choices this module fixes, each cited:
  - Trend/value-zone EMA pair, daily: 11 (fast) / 22 (slow) — Ch.22: "a
    22-day and an 11-day EMA on a daily chart... keep the difference
    between the two EMAs near 2:1"; "Among the numbers I like are 22
    because there are approximately 22 trading days in a month."
  - Trend/value-zone EMA pair, weekly: 13 (fast) / 26 (slow) — Ch.22: "a
    26-week and a 13-week EMA on a weekly chart."
  - Impulse System's EMA input is the FAST EMA of each pair — Ch.40: "A
    good measure of the inertia of any trading vehicle is the slope of
    its FAST EMA." (Distinct from the next point.)
  - Triple Screen's weekly/daily trend read uses the SLOW EMA of each pair
    — Ch.22's own Figure 22.2 caption: "The slow EMA helps identify the
    trend, while the fast MA sets the boundary of the value zone."
  - MA22/50/200 display columns: the three windows the user asked for by
    name; MA22 doubles as the daily value-zone's slow EMA (same period,
    computed once).
  - Divergence centerline-crossing requirement (Ch.23/Ch.30): applied to
    MACD-Histogram and the 13-day (long-term) Force Index only — see
    scanner_board/divergence.py's own module docstring.

Not computed here (deferred, see scanner_board/PLAN.md): Weinstein Stage
(still deferred to a future pass reusing the existing StageDetector/
score_stage2 pattern) and the +-1/2/3 ATR display bands (trivial
`ma22 +- k*atr` arithmetic the dashboard can derive from the stored
atr/ma22 values directly — not worth persisting six more columns for).

Assumes both `daily` and `weekly` are non-empty (a ticker with fewer than
4 trading days of history resamples to an EMPTY weekly frame — see
scanner_board/weekly.py's partial-week-drop rule — and will raise here).
Resilience to that lives one layer up, in scanner_pipeline.py's per-ticker
try/except, the same pattern triple_screen_tracker_service.py already uses
for a single bad ticker never aborting the whole run — not duplicated
inside every low-level function here.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from canadian_stock_screener import TechnicalIndicators
from scanner_board.divergence import detect_divergence
from scanner_board.indicators import stochastic, accumulation_distribution, force_index
from scanner_board.slope import classify_slope
from scanner_board.thesis_rules import (
    price_vs_ma, value_zone_position, macd_cross, trend_health,
    macd_histogram_extreme, di_bias, adx_regime,
    oscillator_zone, volume_vs_average,
    force_short_term_zone, force_long_term_bias,
)
from scanner_board.triple_screen import impulse_color, trend_direction, triple_screen_alignment

# Daily MA display windows the user asked for by name; MA_DISPLAY_WINDOWS[0]
# (22) doubles as the daily value-zone's slow EMA period (Ch.22).
MA_DISPLAY_WINDOWS = (22, 50, 200)

DAILY_VALUE_ZONE_FAST_EMA = 11
DAILY_VALUE_ZONE_SLOW_EMA = MA_DISPLAY_WINDOWS[0]
WEEKLY_VALUE_ZONE_FAST_EMA = 13
WEEKLY_VALUE_ZONE_SLOW_EMA = 26

# Bar-to-bar, per Ch.40's own definition of both Impulse inputs ("the
# relationship between any two neighboring bars").
IMPULSE_SLOPE_PERIOD = 2

# The trailing window for reading "is the slow (trend) EMA rising or
# falling" — used for the plain MA Slope column and both Triple Screen
# trend reads. Ch.39 doesn't specify an exact window for this; matches the
# existing ma30_slope precedent already in this codebase
# (ScoreCalculator.score_stage2's `linear_regression_slope(ma30w, 10)`).
TREND_SLOPE_PERIOD = 10

# Matches score_macd's existing hist_slope window (`histogram.iloc[-5:]`).
MACD_HIST_SLOPE_PERIOD = 5
# Matches score_adx's existing adx_slope window (`adx_values.iloc[-10:]`).
ADX_TREND_SLOPE_PERIOD = 10

OSCILLATOR_LOOKBACK = 120        # Ch.27's "4 to 6 months" 5%-rule window
MACD_HIST_EXTREME_LOOKBACK = 63  # Ch.23's "past three months" (~21 * 3)
FORCE_SHORT_PERIOD = 2
FORCE_LONG_PERIOD = 13
VOLUME_AVG_LOOKBACK = 10         # Ch.28's "past two weeks"


def _f(value) -> "float | None":
    """Sanitize a pandas/numpy scalar to a plain float, or None for NaN --
    so every value in the returned dict round-trips through SQLite (and
    JSON) cleanly, never a numpy.float64 or a NaN that compares unequal to
    itself."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else value


def compute_row(ticker: str, daily: pd.DataFrame, weekly: pd.DataFrame) -> Dict[str, Any]:
    """One Board row for `ticker`, given its daily and weekly OHLCV bars
    (Open/High/Low/Close/Volume columns, ascending date index — the shape
    market_data_cache.sync_and_load() and
    scanner_board.weekly.resample_ohlcv_weekly() produce). Returns a flat
    dict ready for scanner_board.store.upsert_row().
    """
    open_, high, low, close, volume = (
        daily["Open"], daily["High"], daily["Low"], daily["Close"], daily["Volume"])

    # ── Trend / Moving Averages (Ch.22) ──
    ma22 = TechnicalIndicators.ema(close, MA_DISPLAY_WINDOWS[0])
    ma50 = TechnicalIndicators.ema(close, MA_DISPLAY_WINDOWS[1])
    ma200 = TechnicalIndicators.ema(close, MA_DISPLAY_WINDOWS[2])
    fast_ema = TechnicalIndicators.ema(close, DAILY_VALUE_ZONE_FAST_EMA)
    slow_ema = ma22  # DAILY_VALUE_ZONE_SLOW_EMA == MA_DISPLAY_WINDOWS[0]

    ma_slope = classify_slope(slow_ema, TREND_SLOPE_PERIOD)

    # ── MACD (Ch.23) ──
    macd_line, macd_signal, macd_hist = TechnicalIndicators.macd(close)

    # ── Directional System / ADX (Ch.24) ──
    plus_di, minus_di, adx = TechnicalIndicators.directional_system(high, low, close)
    atr = TechnicalIndicators.atr(high, low, close)

    # ── Oscillators (Ch.25-27) ──
    rsi = TechnicalIndicators.rsi(close)
    stoch = stochastic(high, low, close)

    # ── Volume-based (Ch.28-30) ──
    obv = TechnicalIndicators.obv(close, volume)
    ad = accumulation_distribution(open_, high, low, close, volume)
    force = force_index(close, volume, FORCE_SHORT_PERIOD, FORCE_LONG_PERIOD)

    # ── Weekly trend / Impulse / Triple Screen (Ch.22, 39, 40) ──
    w_close = weekly["Close"]
    w_fast_ema = TechnicalIndicators.ema(w_close, WEEKLY_VALUE_ZONE_FAST_EMA)
    w_slow_ema = TechnicalIndicators.ema(w_close, WEEKLY_VALUE_ZONE_SLOW_EMA)
    _, _, w_macd_hist = TechnicalIndicators.macd(w_close)

    weekly_trend = trend_direction(w_slow_ema, TREND_SLOPE_PERIOD)
    daily_trend = trend_direction(slow_ema, TREND_SLOPE_PERIOD)

    row: Dict[str, Any] = {
        "ticker": ticker,

        "price": _f(close.iloc[-1]),
        "pct_chg": _f((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else None,

        # Trend / Moving Averages
        "ma22": _f(ma22.iloc[-1]), "ma50": _f(ma50.iloc[-1]), "ma200": _f(ma200.iloc[-1]),
        "ma_slope": ma_slope.value,
        "price_vs_ma": price_vs_ma(close, slow_ema).value,
        "value_zone": value_zone_position(close, fast_ema, slow_ema).value,

        # MACD
        "macd_line": _f(macd_line.iloc[-1]), "macd_signal": _f(macd_signal.iloc[-1]),
        "macd_hist": _f(macd_hist.iloc[-1]),
        "macd_cross": macd_cross(macd_line, macd_signal).value,
        "macd_hist_slope": classify_slope(macd_hist, MACD_HIST_SLOPE_PERIOD).value,
        "trend_health": trend_health(close, macd_hist).value,
        "macd_hist_extreme": macd_histogram_extreme(macd_hist, MACD_HIST_EXTREME_LOOKBACK).value,
        "macd_hist_divergence": detect_divergence(
            close, macd_hist, require_centerline_cross=True).value,

        # Directional System / ADX
        "plus_di": _f(plus_di.iloc[-1]), "minus_di": _f(minus_di.iloc[-1]), "adx": _f(adx.iloc[-1]),
        "di_bias": di_bias(plus_di, minus_di).value,
        "adx_trend": classify_slope(adx, ADX_TREND_SLOPE_PERIOD).value,
        "adx_regime": adx_regime(adx, plus_di, minus_di).value,
        "atr": _f(atr.iloc[-1]),
        "atr_pct": _f(atr.iloc[-1] / close.iloc[-1] * 100) if _f(close.iloc[-1]) else None,

        # Oscillators
        "rsi": _f(rsi.iloc[-1]),
        "rsi_zone": oscillator_zone(rsi, OSCILLATOR_LOOKBACK).value,
        "rsi_divergence": detect_divergence(close, rsi).value,
        "stoch_k": _f(stoch.percent_k.iloc[-1]), "stoch_d": _f(stoch.percent_d.iloc[-1]),
        "stoch_zone": oscillator_zone(stoch.percent_k, OSCILLATOR_LOOKBACK).value,
        "stoch_divergence": detect_divergence(close, stoch.percent_k).value,

        # Volume-based
        "volume": _f(volume.iloc[-1]),
        "volume_vs_avg": volume_vs_average(volume, VOLUME_AVG_LOOKBACK).value,
        "obv": _f(obv.iloc[-1]),
        "obv_trend": classify_slope(obv, TREND_SLOPE_PERIOD).value,
        "obv_divergence": detect_divergence(close, obv).value,
        "ad": _f(ad.iloc[-1]),
        "ad_trend": classify_slope(ad, TREND_SLOPE_PERIOD).value,
        "ad_divergence": detect_divergence(close, ad).value,
        "force_short": _f(force.short.iloc[-1]),
        "force_short_zone": force_short_term_zone(force.short).value,
        "force_short_divergence": detect_divergence(close, force.short).value,
        "force_long": _f(force.long.iloc[-1]),
        "force_long_bias": force_long_term_bias(force.long).value,
        "force_long_divergence": detect_divergence(
            close, force.long, require_centerline_cross=True).value,

        # Multi-timeframe (Ch.39, 40)
        "impulse_daily": impulse_color(fast_ema, macd_hist, IMPULSE_SLOPE_PERIOD).value,
        "impulse_weekly": impulse_color(w_fast_ema, w_macd_hist, IMPULSE_SLOPE_PERIOD).value,
        "weekly_trend": weekly_trend.value,
        "daily_trend": daily_trend.value,
        "triple_screen": triple_screen_alignment(weekly_trend, daily_trend).value,
    }
    return row
