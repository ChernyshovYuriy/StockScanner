"""
Monthly top-N portfolio engine: selection with a keep band and sector cap,
costs, and the pass-rule statistics of PLAN.md. No I/O.

Time works in periods. Period k runs from the execution after signal session k
to the execution after signal session k+1. Everything that decides period k's
weights may use data up to signal session k only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

TOP_N, KEEP = 10, 15
BLOCK_MONTHS = 6
N_BOOT = 5000
N_RANDOM = 1000
CASH = "CASH"


@dataclass(frozen=True)
class Panel:
    """Adjusted daily data on one calendar; columns are tickers."""
    calendar: pd.DatetimeIndex
    close: pd.DataFrame              # last close, carried forward a few sessions at most
    execution: pd.DataFrame          # price a trade on that session gets
    dollar_vol: pd.DataFrame         # 20-session average


def build_panel(data: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, max_ffill: int = 5) -> Panel:
    """A session without a trade carries the last close, which is also the price
    a trade would get; after max_ffill such sessions the name is unavailable."""
    def wide(col):
        out = {}
        for t, d in data.items():
            d = d[~d.index.duplicated(keep="last")].sort_index()
            d = d[(d["Close"] > 0) & (d["Volume"] > 0)]
            out[t] = d[col].reindex(calendar)
        return pd.DataFrame(out, index=calendar)
    raw, opens, volume = wide("Close"), wide("Open"), wide("Volume")
    close = raw.ffill(limit=max_ffill)
    dollar = (raw * volume).fillna(0.0).rolling(20).mean()
    execution = opens.where(opens > 0).fillna(close.shift(1).where(raw.isna(), raw))
    return Panel(calendar, close, execution.where(close.notna()), dollar)


def month_end_sessions(calendar: pd.DatetimeIndex) -> List[int]:
    """Last session of each month, except a month still in progress at the end."""
    months = calendar.to_period("M")
    last = pd.Series(np.arange(len(calendar))).groupby(np.asarray(months.asi8)).max().to_list()
    return [int(s) for s in last if s + 1 < len(calendar)]


def period_returns(panel: Panel, sessions: Sequence[int]) -> pd.DataFrame:
    """Return of every ticker over each period: execution after session k to
    execution after session k+1. A name that stops trading inside the period is
    closed at its last price. Rows: signal sessions, the last one excluded."""
    ex = panel.execution
    last = panel.close.ffill()
    rows = []
    for a, b in zip(sessions, sessions[1:]):
        start = ex.iloc[a + 1]
        end = ex.iloc[b + 1].fillna(last.iloc[b + 1])
        rows.append(end / start - 1.0)
    return pd.DataFrame(rows, index=list(sessions[:-1]))


def select(scores: pd.Series, held: Set[str], n: int = TOP_N, keep: int = KEEP,
           sector: Optional[Dict[str, str]] = None, max_per_sector: int = 2) -> List[str]:
    """The n names to hold: current holdings still in the top `keep` first, then
    the best of the rest. Ties go to the alphabetically first name."""
    ranked = scores.dropna().sort_index().sort_values(ascending=False, kind="stable").index.tolist()
    kept = [t for t in ranked[:keep] if t in held]
    kept_set = set(kept)
    order = kept + [t for t in ranked if t not in kept_set]
    out: List[str] = []
    per: Dict[str, int] = {}
    for t in order:
        if len(out) == n:
            break
        s = sector.get(t) if sector is not None else None
        if s is not None and per.get(s, 0) >= max_per_sector:
            continue
        out.append(t)
        if s is not None:
            per[s] = per.get(s, 0) + 1
    return out


def equal_weights(names: Sequence[str], n: int = TOP_N) -> Dict[str, float]:
    """1/n each; unfilled slots are cash."""
    w = {t: 1.0 / n for t in names}
    rest = 1.0 - sum(w.values())
    if rest > 1e-12:
        w[CASH] = w.get(CASH, 0.0) + rest
    return w


def run(weights_for: Callable[[int, Set[str]], Dict[str, float]], returns: pd.DataFrame,
        cost_per_side: float) -> pd.DataFrame:
    """Walk the periods. `weights_for(session, held)` gives target weights (sum
    1). Returns gross and net period returns and turnover. A held name with no
    return (it stopped existing) counts as 0%; trading cash is free."""
    current: Dict[str, float] = {}
    rows = []
    values, column = returns.to_numpy(float), {t: k for k, t in enumerate(returns.columns)}
    for row, s in enumerate(returns.index):
        target = weights_for(int(s), {t for t in current if t != CASH})
        total = sum(target.values())
        if abs(total - 1.0) > 1e-9 or min(target.values()) < -1e-12:
            raise ValueError(f"bad weights at session {s}: {total}")
        traded = sum(abs(target.get(t, 0.0) - current.get(t, 0.0)) for t in set(target) | set(current) if t != CASH)
        r = {t: values[row, column[t]] for t in target if t in column}
        gains = {t: w * (1.0 + (r[t] if t in r and np.isfinite(r[t]) else 0.0)) for t, w in target.items()}
        value = sum(gains.values())
        rows.append({"gross": value - 1.0, "net": value * (1.0 - cost_per_side * traded) - 1.0,
                     "turnover": traded, "names": sum(t != CASH and w > 0 for t, w in target.items())})
        current = {t: g / value for t, g in gains.items()}
    return pd.DataFrame(rows, index=returns.index)


# ── Statistics ───────────────────────────────────────────────────────────────

def cagr(r: pd.Series) -> float:
    return float((1.0 + r).prod() ** (12.0 / len(r)) - 1.0)


def max_drawdown(r: pd.Series) -> float:
    """Worst fall of the month-end equity curve, as a negative number."""
    equity = np.concatenate([[1.0], (1.0 + r.to_numpy()).cumprod()])
    return float((equity / np.maximum.accumulate(equity) - 1.0).min())


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(12.0))


def bootstrap_mean(x: np.ndarray, block: int = BLOCK_MONTHS, n_boot: int = N_BOOT, seed: int = 0,
                   level: float = 0.95) -> tuple:
    """Interval of the annualized mean, moving blocks with wrap-around."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, (n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n] % n
    means = x[idx].mean(axis=1) * 12.0
    a = (1.0 - level) / 2.0
    return float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))


def yearly_excess(r: pd.Series, bench: pd.Series, years: np.ndarray) -> pd.Series:
    grow = lambda x: (1.0 + x).groupby(years).prod() - 1.0
    return grow(r) - grow(bench)


def random_portfolios(eligible: Dict[int, List[str]], returns: pd.DataFrame, n: int = TOP_N, keep: int = KEEP,
                      sector_of: Optional[Callable[[int], Dict[str, str]]] = None,
                      n_random: int = N_RANDOM, seed: int = 0) -> np.ndarray:
    """Gross CAGR of random top-n portfolios: random scores over the names
    eligible each period, with the same keep band and sector cap."""
    out = []
    for k in range(n_random):
        rng = np.random.default_rng(seed + k)

        def pick(s: int, held: Set[str]) -> Dict[str, float]:
            names = eligible[s]
            scores = pd.Series(rng.random(len(names)), index=names)
            return equal_weights(select(scores, held, n, keep, sector_of(s) if sector_of else None), n)
        out.append(cagr(run(pick, returns, 0.0)["gross"]))
    return np.array(out)


def evaluate(net: pd.Series, bench: pd.Series, years: np.ndarray) -> Dict[str, float]:
    """Pass-rule numbers 1, 3 and 4."""
    ex = (net - bench).to_numpy()
    lo, hi = bootstrap_mean(ex)
    y = yearly_excess(net, bench, years)
    dd, bdd = max_drawdown(net), max_drawdown(bench)
    return {
        "months": len(net), "cagr": cagr(net), "bench_cagr": cagr(bench),
        "excess": float(ex.mean() * 12.0), "lo": lo, "hi": hi,
        "years_ahead": float((y > 0).mean()), "max_dd": dd, "bench_dd": bdd,
        "sharpe": sharpe(net), "bench_sharpe": sharpe(bench),
        "c1": bool(ex.mean() > 0 and lo > 0), "c3": bool((y > 0).mean() >= 0.60), "c4": bool(dd >= 1.25 * bdd - 1e-12),
    }
