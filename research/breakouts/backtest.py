"""
What happened after each detected breakout (PLAN.md). No network.

    python -m research.breakouts.backtest dev        # up to 2023-06-30
    python -m research.breakouts.backtest holdout    # once, after the verdict
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from research.breakouts.detector import DEFAULT, Rules, scan

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "breakouts"
XIU = "XIU.TO"
HORIZONS = (20, 60, 120)
MAIN = 60
SEPARATE = 60
COST = 0.005
MIN_PRICE, MIN_DOLLAR, MIN_HISTORY = 1.0, 1_000_000.0, 252
NEW_HIGH = 52
DEV_END, HOLD_START = "2023-06-30", "2024-01-02"
TRADE_BARS, PEERS = 252, 20
WORKERS = max(1, (os.cpu_count() or 2) - 1)


class Market:
    """Panels on XIU.TO's sessions (columns: tickers), plus each name's own bars."""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.cal = data[XIU].index
        self.bars = {t: d[~d.index.duplicated(keep="last")].sort_index() for t, d in data.items() if t != XIU}
        wide = lambda col: pd.DataFrame({t: d[col].reindex(self.cal) for t, d in self.bars.items()})
        close, volume = wide("Close"), wide("Volume")
        self.close, self.open = close.ffill(limit=5), wide("Open")
        self.last = close.ffill()
        dollar = (close * volume).fillna(0.0).rolling(20).mean()
        history = close.notna().cumsum()
        self.eligible = (close >= MIN_PRICE) & (dollar >= MIN_DOLLAR) & (history >= MIN_HISTORY)
        self.new_high = close > close.shift(1).rolling(NEW_HIGH, min_periods=NEW_HIGH).max()
        tr = pd.concat([wide("High") - wide("Low"), (wide("High") - self.last.shift(1)).abs(),
                        (wide("Low") - self.last.shift(1)).abs()], keys=range(3)).groupby(level=1).max()
        self.atr = (tr.rolling(20, min_periods=15).mean() / close).where(close.notna())
        self.xiu_close, self.xiu_open = data[XIU]["Close"], data[XIU]["Open"]

    def forward(self, h: int) -> pd.DataFrame:
        """Return from the next open to the close h sessions after entry. Dead names: the last bar."""
        entry = self.open.shift(-1).where(self.open.shift(-1) > 0)
        return self.last.shift(-(1 + h)) / entry - 1.0

    def xiu_forward(self, h: int) -> pd.Series:
        return self.xiu_close.shift(-(1 + h)) / self.xiu_open.shift(-1) - 1.0


def separate(dates: List[pd.Timestamp], cal: pd.DatetimeIndex, gap: int) -> List[bool]:
    """Keep the first, then only those at least `gap` sessions after the last kept."""
    out, last = [], None
    for d in dates:
        k = cal.get_loc(d)
        keep = last is None or k - last >= gap
        out.append(keep)
        last = k if keep else last
    return out


def find_events(m: Market, rules: Rules = DEFAULT) -> tuple:
    """(separate eligible events, every detection as a (session, ticker) mask)."""
    rows, mask = [], pd.DataFrame(False, index=m.close.index, columns=m.close.columns)
    with ProcessPoolExecutor(WORKERS) as pool:
        scans = dict(zip(m.bars, pool.map(scan, m.bars.values(), [rules] * len(m.bars), chunksize=8)))
    for t, d in m.bars.items():
        found = [(d.index[b.bar], b) for b in scans[t] if d.index[b.bar] in m.cal]
        for day, _ in found:
            mask.at[day, t] = True
        found = [(day, b) for day, b in found if m.eligible.at[day, t]]
        for keep, (day, b) in zip(separate([x for x, _ in found], m.cal, SEPARATE), found):
            if keep:
                rows.append({"ticker": t, "date": day, "level": b.level, "close": b.close, "bars": b.bars, "tests": b.tests,
                             "low": b.low, "target": b.target, "lfd_low": b.lfd_low})
    cols = ["ticker", "date", "level", "close", "bars", "tests", "low", "target", "lfd_low"]
    return pd.DataFrame(rows, columns=cols).sort_values(["date", "ticker"]).reset_index(drop=True), mask


def add_returns(ev: pd.DataFrame, mask: pd.DataFrame, m: Market, horizons=HORIZONS) -> pd.DataFrame:
    """Event returns and the same-day means of (a) eligible names, (b) plain new highs, (c) XIU.
    The event itself is left out of (a)."""
    ev = ev.copy()
    for h in horizons:
        f = m.forward(h).where(m.last.shift(-(1 + h)).notna())
        f.iloc[max(0, len(f) - 1 - h):] = np.nan                       # horizon not over yet
        e = f.where(m.eligible)
        total, n = e.sum(axis=1), e.notna().sum(axis=1)
        plain = f.where(m.eligible & m.new_high & ~mask).mean(axis=1)
        x = m.xiu_forward(h)
        ev[f"r{h}"] = [f.at[d, t] for d, t in zip(ev["date"], ev["ticker"])]
        ev[f"a{h}"] = [(total[d] - r) / (n[d] - 1) if n[d] > 1 else np.nan for d, r in zip(ev["date"], ev[f"r{h}"])]
        ev[f"b{h}"] = plain.reindex(ev["date"]).to_numpy()
        ev[f"x{h}"] = x.reindex(ev["date"]).to_numpy()
    return ev


def block_interval(x: pd.Series, dates: pd.Series, weeks: int = 13, n: int = 5000, seed: int = 0) -> tuple:
    """95% interval of the mean, resampling blocks of calendar weeks."""
    ok = x.notna().to_numpy()
    x, dates = x[ok].to_numpy(), dates[ok]
    if len(x) < 2:
        return np.nan, np.nan
    week = ((dates - dates.min()).dt.days // 7).to_numpy()
    block = week // weeks
    groups = [x[block == b] for b in np.unique(block)]
    sums, sizes = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    pick = np.random.default_rng(seed).integers(0, len(groups), (n, len(groups)))
    means = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summary(ev: pd.DataFrame, h: int = MAIN) -> Dict[str, float]:
    ev = ev[ev[f"r{h}"].notna()]
    ea, eb = ev[f"r{h}"] - ev[f"a{h}"], ev[f"r{h}"] - ev[f"b{h}"]
    la, lb = block_interval(ea, ev["date"]), block_interval(eb, ev["date"])
    years = ea.groupby(ev["date"].dt.year).mean()
    return {"h": h, "events": len(ev), "tickers": ev["ticker"].nunique(), "mean": ev[f"r{h}"].mean(), "median": ev[f"r{h}"].median(),
            "win": (ev[f"r{h}"] > 0).mean(), "ex_a": ea.mean(), "a_lo": la[0], "a_hi": la[1], "ex_a_median": ea.median(),
            "ex_b": eb.mean(), "b_lo": lb[0], "b_hi": lb[1], "xiu": ev[f"x{h}"].mean(),
            "net_vs_xiu": (ev[f"r{h}"] - COST - ev[f"x{h}"]).mean(), "years_ahead": float((years > 0).mean())}


def verdict(s: Dict[str, float], sens_share: float) -> Dict[str, bool]:
    c = {"c1": s["ex_a"] > 0 and s["a_lo"] > 0, "c2": s["ex_b"] > 0 and s["b_lo"] > 0, "c3": s["years_ahead"] >= 0.60,
         "c4": sens_share >= 0.80, "c5": s["net_vs_xiu"] > 0 and s["events"] >= 100 and s["tickers"] >= 50}
    c = {k: bool(v) for k, v in c.items()}
    return {**c, "passed": all(c.values())}


def sensitivity_rules(r: Rules = DEFAULT) -> Dict[str, Rules]:
    """Every setting +/-20%; whole-number settings move by at least 1. min_tests below 2 is not a pattern."""
    out = {}
    for f in fields(Rules):
        v = getattr(r, f.name)
        for label, k in (("-20%", 0.8), ("+20%", 1.2)):
            new = v * k if isinstance(v, float) else (int(round(v * k)) if round(v * k) != v else v + (1 if k > 1 else -1))
            if f.name == "min_tests" and new < 2:
                continue
            out[f"{f.name} {label}"] = replace(r, **{f.name: new})
    return out


# ── The trade as published ───────────────────────────────────────────────────

def trade(bars: pd.DataFrame, k: int, stop: float, target: float, level: Optional[float] = None, floor: Optional[float] = None,
          max_bars: int = TRADE_BARS) -> Optional[Dict]:
    """Enter at the open of bar k. Stop before target within a bar; a gap fills at the
    open. Also the type, read without the stop: what came first, the target or the floor."""
    o, h, l, c = (bars[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
    if k >= len(o) or not (stop < o[k] < target):
        return None
    entry, end = o[k], min(len(o), k + max_bars)
    exit_, why = c[end - 1], "time" if end == k + max_bars else "open"
    for j in range(k, end):
        if l[j] <= stop:
            exit_, why = min(o[j], stop), "stop"
            break
        if h[j] >= target:
            exit_, why = max(o[j], target), "target"
            break
    out = {"entry": entry, "exit": exit_, "why": why, "r": (exit_ - entry) / (entry - stop), "ret": exit_ / entry - 1.0}
    if level is not None:
        kind, touched, broke = "unresolved", False, False
        for j in range(k, end):
            if l[j] < floor:
                kind = "4"
                break
            touched, broke = touched or l[j] <= level, broke or l[j] < stop
            if h[j] >= target:
                kind = "3" if broke else "2" if touched else "1"
                break
        out["type"] = kind
    return out


def trades(ev: pd.DataFrame, m: Market) -> pd.DataFrame:
    rows = []
    for e in ev.itertuples():
        bars = m.bars[e.ticker]
        k = bars.index.get_loc(e.date) + 1
        x = trade(bars, k, e.lfd_low, e.target, e.level, e.low)
        if x is None or x["why"] == "open":
            continue
        atr = m.atr.at[e.date, e.ticker]
        up, down = (e.target / x["entry"] - 1) / atr, (1 - e.lfd_low / x["entry"]) / atr
        near = (m.atr.loc[e.date][m.eligible.loc[e.date]].drop(e.ticker, errors="ignore").dropna() - atr).abs().nsmallest(PEERS).index
        peers = []
        for t in near:
            b = m.bars[t]
            j = b.index.get_loc(e.date) + 1
            if j < len(b):
                p0, a = b["Open"].iat[j], m.atr.at[e.date, t]
                y = trade(b, j, p0 * (1 - down * a), p0 * (1 + up * a))
                if y and y["why"] != "open":
                    peers.append((y["r"], y["ret"], y["why"] == "target"))
        pr = np.mean(peers, axis=0) if peers else [np.nan] * 3
        rows.append({"ticker": e.ticker, "date": e.date, **x, "peer_r": pr[0], "peer_ret": pr[1], "peer_hit": pr[2]})
    return pd.DataFrame(rows)


def run(m: Market, rules: Rules, start: Optional[str], end: Optional[str]) -> tuple:
    ev, mask = find_events(m, rules)
    if start:
        ev = ev[ev["date"] >= start]
    if end:
        ev = ev[ev["date"] <= end]
    ev = ev.reset_index(drop=True)
    return add_returns(ev, mask, m), mask


def main(argv: Optional[List[str]] = None) -> int:
    from research.breakouts import prices
    part = (argv or sys.argv[1:])[0]
    span = (None, DEV_END) if part == "dev" else (HOLD_START, None)
    m = Market(prices.load())
    OUT.mkdir(parents=True, exist_ok=True)
    ev, _ = run(m, DEFAULT, *span)
    ev.to_csv(OUT / f"events_{part}.csv", index=False)
    table = pd.DataFrame([summary(ev, h) for h in HORIZONS])
    pd.set_option("display.width", 220)
    print(table.round(4).T.to_string())
    if part == "dev":
        sens = {k: summary(run(m, r, *span)[0]) for k, r in sensitivity_rules().items()}
        print(pd.DataFrame(sens).T[["events", "ex_a", "ex_b", "net_vs_xiu"]].round(4).to_string())
        share = float(np.mean([s["ex_a"] > 0 for s in sens.values()]))
        print("sensitivity share", share, verdict(summary(ev), share))
        pd.DataFrame(sens).T.to_csv(OUT / "sensitivity_dev.csv")
    tr = trades(ev, m)
    tr.to_csv(OUT / f"trades_{part}.csv", index=False)
    print("trades", len(tr), "target", round((tr["why"] == "target").mean(), 3), "stop", round((tr["why"] == "stop").mean(), 3),
          "mean R", round(tr["r"].mean(), 3), "| peers: target", round(tr["peer_hit"].mean(), 3), "mean R", round(tr["peer_r"].mean(), 3),
          "| R minus peers 95%", [round(v, 3) for v in block_interval(tr["r"] - tr["peer_r"], tr["date"])],
          "| mean ret", round(tr["ret"].mean(), 4), "peers", round(tr["peer_ret"].mean(), 4))
    print(tr["type"].value_counts(normalize=True).round(3).to_dict())
    table.to_csv(OUT / f"summary_{part}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
