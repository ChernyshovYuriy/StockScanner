"""
Pure indicator math for the Triple Screen reference implementation
(reference_impl.py). Kept separate and dependency-light (pandas only, no
screen/engine types beyond Direction) so each function is independently
unit-testable and swappable -- a different TrendScreen implementation could
call entirely different math without touching SignalEngine or the ABCs.

Every function here returns a safe default rather than raising when there
aren't enough clean (non-NaN) bars for a meaningful read -- the contract
this whole package promises for insufficient/missing data (see engine.py's
truth table: an undecided screen just flows through as FLAT/absent/not-
fired, no special-case exception handling needed upstream).

Every function also defensively sorts its input by index ascending before
reading off "latest"/"prior" via positional .iloc -- PriceData.bars is
documented as sorted ascending (see types.py), but nothing upstream
enforces that; without this, a reverse-chronological feed would silently
read as if the oldest bar were the newest (found via adversarial stress
testing, see research/triple_screen/TRIPLE_SCREEN_STRESS_FINDINGS.md).

Screen 3's prior_bar_breakout() requires more than a price cross to fire --
Elder's book (_Come Into My Trading Room_) is explicit that a TRUE breakout
is confirmed by heavy volume and by indicators reaching new extremes in the
trend direction (a false breakout shows light volume and/or price/indicator
divergence); see volume_confirms_breakout() and price_momentum_new_extreme()
below. Added 2026-09 -- the earlier version only checked the price cross.
The extreme/divergence confirmation deliberately uses price momentum, NOT a
second Force Index period, even though Screen 2 already computes Force
Index and Elder himself uses a longer-period Force Index for this exact
purpose: a longer-period EMA of the SAME series Screen 2's short-period EMA
already reads creates a mathematical conflict on the shared latest bar --
the raw value needed today to push a slow EMA to a fresh extreme is always
several times bigger than what keeps a fast EMA of the same series
negative (a fixed ratio set by the two EMA spans), making Screen 2's
pullback and this confirmation all but mutually exclusive on the same bar.
Confirmed via adversarial testing (20,000 synthetic bar sequences, 0 found
both true simultaneously) before switching to a decoupled indicator -- see
TRIPLE_SCREEN_STRESS_FINDINGS.md.
"""
import math

import pandas as pd

from .types import Direction


def ema_slope_direction(closes: pd.Series, ema_period: int = 13, lag: int = 3) -> tuple[Direction, dict]:
    """Screen 1's reference indicator: direction of an `ema_period`-period
    EMA's slope, compared `lag` bars back -- mirrors research/elder_ray.py's
    own trend_at() convention (reimplemented independently here; the two
    tools are otherwise unrelated). Exact-equality ties resolve to FLAT, no
    fuzzy epsilon needed, same as elder_ray.py.
    """
    clean = closes.sort_index().dropna()
    if len(clean) < ema_period + lag:
        indicator_values = {"ema": float(clean.iloc[-1])} if len(clean) else {}
        return Direction.FLAT, indicator_values

    ema = clean.ewm(span=ema_period, adjust=False).mean()
    now, then = float(ema.iloc[-1]), float(ema.iloc[-1 - lag])
    if now > then:
        direction = Direction.UP
    elif now < then:
        direction = Direction.DOWN
    else:
        direction = Direction.FLAT
    return direction, {"ema": now, "ema_prior": then}


def force_index_pullback(bars: pd.DataFrame, trend: Direction, span: int = 2) -> tuple[bool, dict]:
    """Screen 2's reference indicator: Elder's Force Index (Volume x
    delta-Close), smoothed with a `span`-period EMA. "Pullback present" on
    the latest bar means the smoothed value opposes `trend` -- negative in
    an UP trend (bears momentarily won a day), positive in a DOWN trend
    (bulls momentarily won a day, i.e. a counter-rally). FLAT trend has no
    defined pullback: always False.
    """
    if trend == Direction.FLAT:
        return False, {}

    clean = bars.sort_index().dropna(subset=["Close", "Volume"])
    if len(clean) < 2:
        return False, {}

    force_index = clean["Close"].diff() * clean["Volume"]
    smoothed = force_index.ewm(span=span, adjust=False).mean()
    latest = float(smoothed.iloc[-1])

    present = latest < 0 if trend == Direction.UP else latest > 0
    return present, {"force_index": latest}


def volume_confirms_breakout(bars: pd.DataFrame, lookback: int = 20, multiplier: float = 1.5) -> tuple[bool, dict]:
    """Elder's "true breakouts are confirmed by heavy volume" rule (_Come
    Into My Trading Room_): today's volume must clearly exceed its own
    recent baseline -- confirmed when it's more than `multiplier` x the
    average of the `lookback` bars strictly BEFORE today. Today's own
    volume never enters its own baseline (a self-inflating average would
    weaken the check). Insufficient history (< lookback + 1 bars) is
    treated the same as everywhere else in this module: not confirmed,
    never an exception.
    """
    clean = bars.sort_index().dropna(subset=["Volume"])
    if len(clean) < lookback + 1:
        return False, {}

    today_volume = float(clean["Volume"].iloc[-1])
    avg_volume = float(clean["Volume"].iloc[-(lookback + 1):-1].mean())
    if not (math.isfinite(today_volume) and math.isfinite(avg_volume)) or avg_volume <= 0:
        return False, {}

    confirmed = today_volume > avg_volume * multiplier
    return confirmed, {"today_volume": today_volume, f"volume_avg_{lookback}": avg_volume}


def price_momentum_new_extreme(bars: pd.DataFrame, trend: Direction, period: int = 10,
                                lookback: int = 20) -> tuple[bool, dict]:
    """Elder's "true breakouts are confirmed when indicators reach new
    extremes; false breakouts show divergence" rule, via a `period`-bar
    price Rate-of-Change (today's Close minus the Close `period` bars ago)
    -- deliberately a plain price-momentum reading, not a second Force
    Index period (see module docstring for why the two Force Index periods
    conflict on Screen 2's shared latest bar). Confirmed when today's
    Rate-of-Change is itself the highest (UP) or lowest (DOWN) value in the
    trailing `lookback` bars -- a fresh momentum extreme. If price makes a
    new high/low but its own momentum does NOT, that mismatch IS the
    price/indicator divergence the false-breakout warning describes; this
    single check captures both "reaches a new extreme" and "no divergence"
    at once. FLAT trend has no defined direction to be extreme in: always
    False.
    """
    if trend == Direction.FLAT:
        return False, {}

    clean = bars.sort_index().dropna(subset=["Close"])
    if len(clean) < period + lookback:
        return False, {}

    momentum = clean["Close"].diff(period)
    window = momentum.iloc[-lookback:]
    latest = float(window.iloc[-1])
    if not math.isfinite(latest):
        return False, {}

    extreme = float(window.max()) if trend == Direction.UP else float(window.min())
    confirmed = latest == extreme
    return confirmed, {f"price_momentum_{period}": latest, f"price_momentum_{period}_extreme": extreme}


def prior_bar_breakout(bars: pd.DataFrame, trend: Direction,
                        volume_lookback: int = 20, volume_multiplier: float = 1.5,
                        momentum_period: int = 10, momentum_lookback: int = 20) -> tuple[bool, dict]:
    """Screen 3's reference indicator: today's High crossing above
    yesterday's High (UP trend) or today's Low crossing below yesterday's
    Low (DOWN trend) -- but a bare price cross is only a CANDIDATE
    breakout. Elder's book (_Come Into My Trading Room_) is explicit that a
    breakout must additionally be confirmed by heavy volume and by
    indicators reaching new extremes (divergence marks a false breakout);
    a price cross without both confirmations is treated as a false
    breakout, not triggered -- see volume_confirms_breakout() and
    price_momentum_new_extreme() above. FLAT trend has no defined trigger:
    always False.
    """
    if trend == Direction.FLAT:
        return False, {}

    column = "High" if trend == Direction.UP else "Low"
    clean = bars.sort_index().dropna(subset=[column])
    if len(clean) < 2:
        return False, {}

    today, prior = float(clean[column].iloc[-1]), float(clean[column].iloc[-2])
    if not (math.isfinite(today) and math.isfinite(prior)):
        # dropna() only catches NaN, not +/-inf; a non-finite OHLC value is
        # exactly as unusable as a missing one -- same safe-default
        # treatment, not a comparison against a garbage value (see module
        # docstring; found via adversarial stress testing).
        return False, {}
    price_crossed = today > prior if trend == Direction.UP else today < prior
    label = column.lower()
    values = {f"today_{label}": today, f"prior_{label}": prior, "price_crossed": price_crossed}

    volume_confirmed, volume_values = volume_confirms_breakout(bars, volume_lookback, volume_multiplier)
    indicator_confirmed, indicator_values = price_momentum_new_extreme(
        bars, trend, period=momentum_period, lookback=momentum_lookback)
    values.update(volume_values)
    values.update(indicator_values)
    values["volume_confirmed"] = volume_confirmed
    values["indicator_confirmed"] = indicator_confirmed

    fired = price_crossed and volume_confirmed and indicator_confirmed
    return fired, values
