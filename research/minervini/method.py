"""
Minervini's method, mechanically: setups, entries, exits (PLAN.md). Pure functions
on one name's bars; nothing after the day being judged is read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rules:
    above_low: float = 1.30
    below_high: float = 0.75
    min_rank: float = 0.70
    pivot_bars: int = 30
    min_rest: int = 5
    max_depth: float = 0.10
    before: int = 60
    dry_days: int = 5
    chase: float = 1.05
    volume_x: float = 1.4
    max_risk: float = 0.08
    max_hold: int = 252


DEFAULT = Rules()
EXITS = ("t10", "t20", "2R", "3R", "half2R", "half20", "ma50", "ma20", "m21", "m21t10")
ENTRIES = ("stop", "volume")


def arrays(d: pd.DataFrame) -> Dict[str, np.ndarray]:
    a = {k: d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume")}
    c = d["Close"]
    for n in (20, 50, 150, 200):
        a[f"ma{n}"] = c.rolling(n).mean().to_numpy()
    a["vol50"] = d["Volume"].rolling(50).mean().to_numpy()
    a["vol_dry"] = d["Volume"].rolling(DEFAULT.dry_days).mean().to_numpy()
    a["hi252"] = d["High"].rolling(252).max().to_numpy()
    a["lo252"] = d["Low"].rolling(252).min().to_numpy()
    return a


def template(a: Dict[str, np.ndarray], rank: np.ndarray, r: Rules = DEFAULT) -> np.ndarray:
    """The trend template on every bar."""
    c, m50, m150, m200 = a["Close"], a["ma50"], a["ma150"], a["ma200"]
    before = np.concatenate([np.full(21, np.nan), m200[:-21]]) if len(m200) > 21 else np.full(len(m200), np.nan)
    with np.errstate(invalid="ignore"):
        return ((c > m150) & (c > m200) & (m150 > m200) & (m200 > before) & (m50 > m150) & (c > m50)
                & (c >= r.above_low * a["lo252"]) & (c >= r.below_high * a["hi252"]) & (rank >= r.min_rank))


def deepest_fall(high: np.ndarray, low: np.ndarray) -> float:
    """Largest drop from a high to a later bar's low."""
    top = np.maximum.accumulate(high)[:-1]
    return float(np.max(1 - low[1:] / top)) if len(high) > 1 else 0.0


def setup(a: Dict[str, np.ndarray], t: int, r: Rules = DEFAULT) -> Optional[Tuple[float, float]]:
    """(pivot, low of the rest) if bar t ends a tight rest, else None. The trend template is checked elsewhere."""
    s = t - r.pivot_bars + 1
    if s < 0:
        return None
    h, l = a["High"], a["Low"]
    k = s + int(np.argmax(h[s:t + 1]))
    if t - k < r.min_rest:
        return None
    pivot, low = h[k], l[k + 1:t + 1].min()
    depth = 1 - low / pivot
    b = k - r.before
    if depth > r.max_depth or b < 0 or depth >= deepest_fall(h[b:k + 1], l[b:k + 1]) or not a["vol_dry"][t] < a["vol50"][t]:
        return None
    return float(pivot), float(low)


def entry_at(a: Dict[str, np.ndarray], t: int, pivot: float, kind: str, r: Rules = DEFAULT) -> Optional[Tuple[int, float]]:
    """(bar, price) bought after a setup on bar t, or None."""
    o, h = a["Open"], a["High"]
    k = t + 1
    if k >= len(o):
        return None
    if kind == "stop":
        return (k, max(o[k], pivot)) if h[k] > pivot and o[k] <= r.chase * pivot else None
    if a["Close"][k] > pivot and a["Volume"][k] >= r.volume_x * a["vol50"][k - 1] and k + 1 < len(o) and o[k + 1] <= r.chase * pivot:
        return k + 1, o[k + 1]
    return None


def initial_stop(entry: float, low: float, r: Rules = DEFAULT) -> float:
    return max(low * 0.999, entry * (1 - r.max_risk))


def leave(a: Dict[str, np.ndarray], k: int, entry: float, stop: float, how: str, r: Rules = DEFAULT) -> Optional[Dict]:
    """Sell what was bought on bar k at `entry`. Legs of (bar, fraction, price).
    On the entry bar a buy at the open can still be stopped or reach its target; so can one during the day
    (pessimistic for the stop: the low may have come before)."""
    o, h, l, c = a["Open"], a["High"], a["Low"], a["Close"]
    risk = entry - stop
    if risk <= 0:
        return None
    target = {"t10": 1.10 * entry, "m21t10": 1.10 * entry, "t20": 1.20 * entry, "half20": 1.20 * entry,
              "2R": entry + 2 * risk, "half2R": entry + 2 * risk, "3R": entry + 3 * risk}.get(how, np.inf)
    ma = a["ma20"] if how == "ma20" else a["ma50"]
    trail = how in ("half2R", "half20", "ma50", "ma20")
    last = k + (21 if how.startswith("m21") else r.max_hold) - 1
    legs: List[Tuple[int, float, float]] = []
    left, level, sell_open = 1.0, stop, False
    for j in range(k, len(o)):
        if sell_open:
            legs.append((j, left, o[j]))
            return done(legs, entry, stop, k, "average")
        if l[j] <= level:
            legs.append((j, left, level if j == k else min(o[j], level)))
            return done(legs, entry, stop, k, "stop" if level == stop else "breakeven")
        if left == 1.0 and h[j] >= target:
            price = target if j == k else max(o[j], target)
            if not how.startswith("half"):
                legs.append((j, 1.0, price))
                return done(legs, entry, stop, k, "target")
            legs.append((j, 0.5, price))
            left, level = 0.5, max(level, entry)
        if how in ("ma50", "ma20") and h[j] >= entry + 2 * risk:
            level = max(level, entry)
        if j >= last:
            legs.append((j, left, c[j]))
            return done(legs, entry, stop, k, "time")
        if trail and c[j] < ma[j] and (left < 1.0 or not how.startswith("half")):
            sell_open = True
    legs.append((len(o) - 1, left, c[-1]))            # the data end: valued at the last close
    return done(legs, entry, stop, k, "open")


def done(legs: List[Tuple[int, float, float]], entry: float, stop: float, k: int, why: str) -> Dict:
    ret = sum(f * p for _, f, p in legs) / entry - 1
    return {"legs": legs, "ret": ret, "r": ret * entry / (entry - stop), "end": legs[-1][0], "bars": legs[-1][0] - k + 1, "why": why}


def find_trades(a: Dict[str, np.ndarray], ok: np.ndarray, kind: str, how: str, r: Rules = DEFAULT,
                setups: Optional[Dict[int, Tuple[float, float]]] = None) -> List[Dict]:
    """Every trade of one name, one position at a time. `ok`: eligible and template on bar t."""
    setups = all_setups(a, ok, r) if setups is None else setups
    out, free = [], -1
    for t, (pivot, low) in setups.items():
        if t < free:
            continue
        e = entry_at(a, t, pivot, kind, r)
        if e is None or e[0] <= free:
            continue
        k, price = e
        stop = initial_stop(price, low, r)
        x = leave(a, k, price, stop, how, r)
        if x is None:
            continue
        out.append({"t": t, "k": k, "entry": price, "stop": stop, "pivot": pivot, **x})
        free = x["end"]
    return out


def all_setups(a: Dict[str, np.ndarray], ok: np.ndarray, r: Rules = DEFAULT) -> Dict[int, Tuple[float, float]]:
    out = {}
    for t in np.where(ok)[0]:
        s = setup(a, int(t), r)
        if s:
            out[int(t)] = s
    return out


# ── The order as emailed: one week, from the first evening near the pivot (LIVE_PLAN.md) ──

WEEK, NEAR = 5, 0.03


def order(a: Dict[str, np.ndarray], ok: np.ndarray, t: int, pivot: float, low: float, cancel: bool, r: Rules = DEFAULT
          ) -> Tuple[str, int, Optional[float]]:
    """(state, bar it was decided on, price) of an order placed after bar t:
    bought / ran (opened too far above the pivot) / cancelled / expired / waiting."""
    n = len(a["Open"])
    for k in range(t + 1, min(t + WEEK, n - 1) + 1):
        if a["Open"][k] > r.chase * pivot:
            return "ran", k, None
        if a["High"][k] > pivot:
            return "bought", k, float(max(a["Open"][k], pivot))
        if cancel and (a["Low"][k] <= low * 0.999 or not ok[k]):
            return "cancelled", k, None
    return ("expired", t + WEEK, None) if n - 1 >= t + WEEK else ("waiting", n - 1, None)


def near(pivot: float, close: float) -> bool:
    return pivot / close - 1 <= NEAR


def week_candidates(a: Dict[str, np.ndarray], setups: Dict[int, Tuple[float, float]], busy) -> List[Tuple[int, float, float]]:
    """(bar, pivot, low) of each setup that would be emailed: near the pivot, not that pivot before,
    and not while the name's last one is still in its week (`busy(t, pivot, low)`: last bar of it)."""
    out, seen, free = [], set(), -1
    for t, (pivot, low) in setups.items():
        if t < free or pivot in seen or not near(pivot, a["Close"][t]):
            continue
        seen.add(pivot)
        out.append((t, pivot, low))
        free = busy(t, pivot, low)
    return out


def week_trades(a: Dict[str, np.ndarray], ok: np.ndarray, setups: Dict[int, Tuple[float, float]], cancel: bool, how: str = "t20",
                r: Rules = DEFAULT) -> List[Dict]:
    def busy(t, pivot, low):
        state, k, _ = order(a, ok, t, pivot, low, cancel, r)
        return t + WEEK if state in ("bought", "waiting") else k

    out = []
    for t, pivot, low in week_candidates(a, setups, busy):
        state, k, price = order(a, ok, t, pivot, low, cancel, r)
        row = {"t": t, "k": k, "state": state, "pivot": pivot, "low": low}
        if state == "bought":
            stop = initial_stop(price, low, r)
            row.update({"entry": price, "stop": stop, **leave(a, k, price, stop, how, r)})
        out.append(row)
    return out
