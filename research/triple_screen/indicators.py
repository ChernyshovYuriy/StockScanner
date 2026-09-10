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
"""
import pandas as pd

from .types import Direction


def ema_slope_direction(closes: pd.Series, ema_period: int = 13, lag: int = 3) -> tuple[Direction, dict]:
    """Screen 1's reference indicator: direction of an `ema_period`-period
    EMA's slope, compared `lag` bars back -- mirrors research/elder_ray.py's
    own trend_at() convention (reimplemented independently here; the two
    tools are otherwise unrelated). Exact-equality ties resolve to FLAT, no
    fuzzy epsilon needed, same as elder_ray.py.
    """
    clean = closes.dropna()
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

    clean = bars.dropna(subset=["Close", "Volume"])
    if len(clean) < 2:
        return False, {}

    force_index = clean["Close"].diff() * clean["Volume"]
    smoothed = force_index.ewm(span=span, adjust=False).mean()
    latest = float(smoothed.iloc[-1])

    present = latest < 0 if trend == Direction.UP else latest > 0
    return present, {"force_index": latest}


def prior_bar_breakout(bars: pd.DataFrame, trend: Direction) -> tuple[bool, dict]:
    """Screen 3's reference indicator: today's High crossing above
    yesterday's High (UP trend) or today's Low crossing below yesterday's
    Low (DOWN trend). FLAT trend has no defined trigger: always False.
    """
    if trend == Direction.FLAT:
        return False, {}

    column = "High" if trend == Direction.UP else "Low"
    clean = bars.dropna(subset=[column])
    if len(clean) < 2:
        return False, {}

    today, prior = float(clean[column].iloc[-1]), float(clean[column].iloc[-2])
    fired = today > prior if trend == Direction.UP else today < prior
    label = column.lower()
    return fired, {f"today_{label}": today, f"prior_{label}": prior}
