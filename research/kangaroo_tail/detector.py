"""
Bullish Kangaroo Tail detection. See __init__.py for the pattern definition
and package-level scope notes.

Every function here returns a safe default (None / False) rather than
raising when there isn't enough clean data for a meaningful read, or when a
computed value comes back NaN/+-inf -- same safe-default contract as
research/triple_screen/indicators.py, for the same reason: an undecided
bar should look like "no signal", not blow up a batch run over many
tickers.
"""
import math

import pandas as pd

from .indicators import rolling_prior_low, wilder_atr
from .types import TailConfig, TailSignal


def _all_finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _check_bar(ticker: str, clean: pd.DataFrame, idx: int,
                atr_series: pd.Series, prior_low_series: pd.Series,
                volume_avg_series: pd.Series, config: TailConfig) -> TailSignal | None:
    """Evaluate bar `idx` of `clean` (already sorted/cleaned) against every
    Kangaroo Tail rule, using indicator series precomputed once over the
    whole frame (see detect_kangaroo_tail()/scan_history() below -- both
    are thin callers of this). Every series here is backward-looking only
    (ATR's own bar aside, which is fine -- an indicator reading AS OF the
    candidate bar is not lookahead; only reading a LATER bar would be), so
    calling this on bar idx never sees information from bar idx+1 onward.
    """
    row = clean.iloc[idx]
    o, h, l, c, v = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), float(row["Volume"])
    atr = float(atr_series.iloc[idx])
    prior_low = float(prior_low_series.iloc[idx])
    if not _all_finite(o, h, l, c, v, atr, prior_low):
        return None
    if atr <= 0:
        return None
    rng = h - l
    if rng <= 0:
        return None

    # Rule 1: genuine structure break -- today's low undercuts the prior
    # N-bar low by a meaningful multiple of ATR, not just a tick.
    break_amount = prior_low - l
    if break_amount < config.break_atr_mult * atr:
        return None

    # Rule 2: closed back into the range, in its upper portion.
    if c <= prior_low:
        return None
    close_position_pct = (c - l) / rng
    if close_position_pct < config.close_back_pct:
        return None

    # Rule 3: lower wick dominates the bar, both relatively and in
    # absolute (ATR) terms.
    lower_wick = min(o, c) - l
    wick_pct = lower_wick / rng
    if wick_pct < config.wick_dominance_pct:
        return None
    if lower_wick < config.wick_atr_mult * atr:
        return None

    # Rule 4: small body.
    body_pct = abs(c - o) / rng
    if body_pct > config.max_body_pct:
        return None

    # Rule 5 (optional): volume kick vs. its own recent baseline.
    volume_ratio = None
    if config.volume_mult is not None:
        avg_volume = float(volume_avg_series.iloc[idx])
        if not _all_finite(avg_volume) or avg_volume <= 0:
            return None
        volume_ratio = v / avg_volume
        if volume_ratio < config.volume_mult:
            return None

    return TailSignal(
        ticker=ticker, date=clean.index[idx],
        open=o, high=h, low=l, close=c, volume=v,
        atr=atr, prior_low=prior_low, break_amount=break_amount,
        close_position_pct=close_position_pct, wick_pct=wick_pct, body_pct=body_pct,
        volume_ratio=volume_ratio,
    )


def _min_bars_required(config: TailConfig) -> int:
    return max(config.lookback, config.atr_period, config.volume_lookback) + 1


def detect_kangaroo_tail(ticker: str, bars: pd.DataFrame, config: TailConfig = TailConfig()) -> TailSignal | None:
    """Evaluate the LAST bar of `bars` as a candidate bullish Kangaroo Tail.
    This is the live/single-point entry: Phase 3's daily scan calls this
    once per ticker per day with only history up to and including today --
    no lookahead by construction, since `bars` simply won't contain
    tomorrow's bar yet.
    """
    clean = bars.sort_index().dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(clean) < _min_bars_required(config):
        return None

    atr_series = wilder_atr(clean, config.atr_period)
    prior_low_series = rolling_prior_low(clean, config.lookback)
    volume_avg_series = clean["Volume"].shift(1).rolling(config.volume_lookback).mean()
    return _check_bar(ticker, clean, len(clean) - 1, atr_series, prior_low_series, volume_avg_series, config)


def scan_history(ticker: str, bars: pd.DataFrame, config: TailConfig = TailConfig()) -> list[TailSignal]:
    """Walk `bars` and collect every Kangaroo Tail detected as of each
    day's close -- i.e. exactly what detect_kangaroo_tail() would have
    returned using only bars up to and including that day, for every day
    in turn. Used by this phase's batch CLI (eyeballing detections against
    a chart) and reusable as-is by Phase 2's historical-verification
    harness, which additionally looks at the bars AFTER each returned
    signal to measure forward outcomes (out of scope for this module --
    scan_history itself never looks past a signal's own bar).
    """
    clean = bars.sort_index().dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    min_bars = _min_bars_required(config)
    if len(clean) < min_bars:
        return []

    atr_series = wilder_atr(clean, config.atr_period)
    prior_low_series = rolling_prior_low(clean, config.lookback)
    volume_avg_series = clean["Volume"].shift(1).rolling(config.volume_lookback).mean()

    signals = []
    for idx in range(min_bars - 1, len(clean)):
        signal = _check_bar(ticker, clean, idx, atr_series, prior_low_series, volume_avg_series, config)
        if signal is not None:
            signals.append(signal)
    return signals


def confirms_next_bar(signal: TailSignal, next_bar: pd.Series, use_midpoint: bool = False) -> bool:
    """The optional second-stage confirmation: did the bar immediately
    AFTER the tail bar close strong enough to trust the reversal, rather
    than firing an alert off the tail bar alone? `use_midpoint=False`
    (default) requires a close above the tail bar's own high -- the
    stricter, standard price-action confirmation threshold; `True` uses
    the tail bar's midpoint instead, a looser bar to clear. Phase 2 tests
    both variants against history to see whether the extra day's wait
    (and which threshold) actually improves the hit rate enough to justify
    delaying the alert -- this function doesn't decide that, it just
    reports the fact for a given `next_bar`.
    """
    threshold = (signal.high + signal.low) / 2 if use_midpoint else signal.high
    next_close = next_bar.get("Close")
    if next_close is None or not math.isfinite(float(next_close)):
        return False
    return float(next_close) > threshold
