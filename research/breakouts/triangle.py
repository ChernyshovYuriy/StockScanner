"""
Ascending triangles and the same pattern without rising lows (PLAN.md, "Ascending
triangle"): a flat top tested at least twice, then a close above it. Reads bars up to
the day only.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from research.breakouts import backtest as bt
from research.breakouts.detector import swing_highs

PIVOT, WIDTH, PULLBACK, OLDEST, YOUNGEST = 3, 0.02, 0.03, 126, 20
INSIDE, MARGIN, RISING, SEPARATE = 1.01, 1.01, 1.01, 20


def detect(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray, t: int, pivots: np.ndarray, margin: float = MARGIN) -> Optional[Dict]:
    """The flat-topped pattern bar t closes out of, or None. `pivots` may be of the whole series."""
    p = pivots[(pivots >= t - OLDEST) & (pivots + PIVOT < t)]
    if len(p) < 2 or c[t] <= c[t - 20:t].max():
        return None
    for level in sorted(set(h[p]), reverse=True):
        if c[t] < level * margin or c[t] > level * 1.15:
            continue
        touches = []
        for i in p[(h[p] <= level) & (h[p] >= level * (1 - WIDTH))]:
            if touches and l[touches[-1]:i + 1].min() > level * (1 - PULLBACK):
                if h[i] > h[touches[-1]]:
                    touches[-1] = i
                continue
            touches.append(int(i))
        if len(touches) < 2 or t - touches[0] < YOUNGEST:
            continue
        s = touches[0]
        if c[s:t].max() > level * INSIDE or h[s:t].max() > level * (1 + WIDTH):
            continue
        lows = [l[a + 1:b].min() for a, b in zip(touches, touches[1:]) if b - a > 1]
        lows += [l[touches[-1] + 1:t].min()] if t - touches[-1] > 1 else []
        if len(lows) < 2:
            continue
        kind = "asc" if all(b >= a * RISING for a, b in zip(lows, lows[1:])) else "fall" if lows[-1] < lows[0] * 0.99 else "flat"
        return {"level": float(level), "bars": t - s, "tests": len(touches), "kind": kind, "first": float(lows[0]), "last_low": float(lows[-1]),
                "vol_x": float(v[t] / v[max(0, t - 50):t].mean()) if v[max(0, t - 50):t].mean() > 0 else np.nan}
    return None


def find_events(m: bt.Market, margin: float = MARGIN) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(events at least SEPARATE sessions apart per name, every detection as a mask)."""
    rows, hits = [], []
    for ticker, d in m.bars.items():
        h, l, c, v = (d[x].to_numpy(float) for x in ("High", "Low", "Close", "Volume"))
        pivots = swing_highs(h, PIVOT)
        prior = pd.Series(c).rolling(20).max().shift(1).to_numpy()
        ok = m.eligible[ticker].reindex(d.index, fill_value=False).to_numpy(bool)
        last = -SEPARATE
        for t in np.where((c > prior) & ok)[0]:
            x = detect(h, l, c, v, int(t), pivots, margin) if t >= OLDEST + PIVOT + 1 else None
            if x and d.index[t] in m.cal:
                hits.append((d.index[t], ticker))
                if t - last >= SEPARATE:
                    rows.append({"ticker": ticker, "date": d.index[t], "close": c[t], **x})
                    last = int(t)
    mask = pd.DataFrame(False, index=m.cal, columns=m.close.columns)
    for day, ticker in hits:
        mask.at[day, ticker] = True
    cols = ["ticker", "date", "close", "level", "bars", "tests", "kind", "first", "last_low", "vol_x"]
    return pd.DataFrame(rows, columns=cols).sort_values(["date", "ticker"]).reset_index(drop=True), mask
