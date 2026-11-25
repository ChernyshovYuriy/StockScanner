"""
Confirmed upside breakouts from chart patterns with a horizontal boundary, as
published on techcharts.net (Aksel Kibar). A detection is a statement about past
bars only: a level was tested, held, and today closed 3% above it. It is not a
forecast. Rules and where each one comes from: PLAN.md.

Prices must be split-adjusted but NOT dividend-adjusted, as on the source's
charts: dividend adjustment lowers old highs and fabricates breakouts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rules:
    margin: float = 0.03             # source: a daily close 3% beyond the boundary
    max_bars: int = 504              # source: patterns of 3 to 24 months
    min_bars: int = 52               # ... its "3 month" charts start at 2.5 (rounded)
    tolerance: float = 0.02          # a high this near the level tests it
    pullback: float = 0.05           # between two tests the price leaves the level by this much
    min_tests: int = 2               # source: "tested several times"; its charts show 2 or more
    pivot: int = 5                   # a test is the highest bar of 5 on either side
    max_above: float = 0.02          # closes above the level allowed inside the pattern ...
    run_up: int = 10                 # ... not counting the last bars on the way out


DEFAULT = Rules()


@dataclass(frozen=True)
class Breakout:
    bar: int                         # the day of the confirming close
    level: float
    confirm: float                   # level * (1 + margin)
    close: float
    start: int                       # first test
    tests: int
    low: float                       # lowest price of the pattern
    target: float                    # level + height
    last_full_day: int               # last bar wholly below the level
    lfd_low: float

    @property
    def bars(self) -> int:
        return self.bar - self.start


def swing_highs(high: np.ndarray, k: int) -> np.ndarray:
    """Bars that are the highest of k on either side (the first of equals)."""
    n = len(high)
    if n < 2 * k + 1:
        return np.array([], dtype=int)
    wide = pd.Series(high).rolling(2 * k + 1, center=True).max().to_numpy()
    before = pd.Series(high).rolling(k).max().shift(1).to_numpy()
    return np.where((high == wide) & (high > before))[0]


def _tests(p: np.ndarray, high: np.ndarray, low: np.ndarray, level: float, pullback: float) -> List[tuple]:
    """(bar, high) of each separate test: highs with no pullback between them are one."""
    out: List[tuple] = []
    for i in p:
        if out and low[out[-1][0]:i + 1].min() > level * (1 - pullback):
            if high[i] > out[-1][1]:
                out[-1] = (out[-1][0], high[i])
        else:
            out.append((int(i), float(high[i])))
    return out


def ceiling(highs: List[float]) -> float:
    """The highest test; with three or more, one spike above the others is forgiven."""
    top = sorted(highs, reverse=True)
    return top[1] if len(top) >= 3 else top[0]


def detect_at(high: np.ndarray, low: np.ndarray, close: np.ndarray, t: int, rules: Rules = DEFAULT,
              pivots: Optional[np.ndarray] = None) -> Optional[Breakout]:
    """The breakout confirmed by bar t's close, or None. Uses bars up to t only
    (`pivots`, if given, may be of the whole series: those not confirmed by t - 1 are dropped)."""
    r = rules
    a = max(0, t - r.max_bars)
    if t - a < r.min_bars or close[t] <= close[t - r.min_bars:t].max():
        return None
    p = swing_highs(high, r.pivot) if pivots is None else pivots
    p = p[(p >= a + r.pivot) & (p + r.pivot < t)]
    best, best_key = None, None
    for seed in p:
        ref = high[seed]
        if close[t] < ref * (1 + r.margin) * (1 - r.tolerance):
            continue
        over = np.where(close[a:t] >= ref * (1 + r.margin))[0]
        s0 = a + over[-1] + 1 if len(over) else a
        near = p[(p >= s0) & (np.abs(high[p] / ref - 1) <= r.tolerance)]
        tests = _tests(near, high, low, ref, r.pullback)
        n = r.min_tests                                  # the most recent pattern that is long enough
        while n <= len(tests) and t - tests[-n][0] < r.min_bars:
            n += 1
        if n > len(tests):
            continue
        tests = tests[-n:]
        start, level = tests[0][0], ceiling([x for _, x in tests])
        confirm = level * (1 + r.margin)
        inside = close[start:t]
        if close[t] < confirm or inside.max() >= confirm or (inside[:-r.run_up] > level).mean() > r.max_above:
            continue
        spread = float(np.mean([abs(x / level - 1) for _, x in tests]))
        key = (start, level)                         # the most recent pattern, at its highest line
        if best_key is None or key > best_key:
            below = np.where(high[start:t] < level)[0]
            if not len(below):
                continue
            lfd, floor = start + int(below[-1]), float(low[start:t].min())
            best_key = key
            best = Breakout(t, level, confirm, float(close[t]), start, len(tests), floor, 2 * level - floor, lfd, float(low[lfd]))
    return best


def scan(df: pd.DataFrame, rules: Rules = DEFAULT) -> List[Breakout]:
    """Every breakout in a daily OHLC frame, oldest first."""
    h, l, c = (df[x].to_numpy(float) for x in ("High", "Low", "Close"))
    p = swing_highs(h, rules.pivot)
    prior = pd.Series(c).rolling(rules.min_bars).max().shift(1).to_numpy()
    out = [detect_at(h, l, c, int(t), rules, p) for t in np.where(c > prior)[0]]
    return [b for b in out if b is not None]
