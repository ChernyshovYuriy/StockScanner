"""
Minervini's method on the Canadian history (PLAN.md). No network.

    python -m research.minervini.backtest
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.breakouts import backtest as bt
from research.minervini import method as mm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "minervini"
PERIODS = {"A": ("2012-11-01", "2018-11-30"), "B": ("2019-01-02", "2023-06-30"), "C": ("2024-01-02", "2026-11-24")}
COST, PEERS = 0.005, 10
RISK, MAX_WEIGHT, SLOTS = 0.01, 0.125, 8
MIN_TRADES = 100

def strength(m: bt.Market) -> pd.DataFrame:
    """Percentile of the 12-month return (last quarter double) among eligible names."""
    last = m.last
    raw = sum(w * (last / last.shift(n) - 1) for n, w in ((63, 0.4), (126, 0.2), (189, 0.2), (252, 0.2)))
    return raw.where(m.eligible).rank(axis=1, pct=True)

class Study:
    def __init__(self, data: Dict[str, pd.DataFrame], earnings: Optional[pd.DataFrame] = None):
        self.m = bt.Market(data)
        self.rank = strength(self.m)
        self.a, self.ok, self.idx, self.setups = {}, {}, {}, {}
        for t, d in self.m.bars.items():
            a = mm.arrays(d)
            rk = self.rank[t].reindex(d.index).to_numpy()
            el = self.m.eligible[t].reindex(d.index, fill_value=False).to_numpy(bool)
            self.a[t], self.idx[t] = a, d.index
            self.ok[t] = el & mm.template(a, rk)
            self.setups[t] = mm.all_setups(a, self.ok[t])
        self.template = pd.DataFrame({t: pd.Series(self.ok[t], index=self.idx[t]) for t in self.a}).reindex(self.m.cal).fillna(False).astype(bool)
        x = self.m.xiu_close
        a200 = x.rolling(200).mean()
        self.market = (x > a200) & (a200 > a200.shift(21))
        rel = self.m.last.div(x, axis=0)
        self.rel_high = rel >= rel.rolling(252).max()
        self.eps = {t: g for t, g in earnings.dropna(subset=["eps_rep"]).groupby("ticker")} if earnings is not None else {}

    def growth(self, t: str, day: pd.Timestamp) -> Optional[bool]:
        """Latest reported quarter's EPS up at least 25% on the year before; None if not known."""
        g = self.eps.get(t)
        if g is None:
            return None
        ts = pd.DatetimeIndex(g["report_ts"])
        eff = ts.normalize() + pd.to_timedelta((ts.hour >= 16).astype(int), unit="D")
        n = int((eff <= day).sum())
        if n < 5 or (day - eff[n - 1]).days > 100:
            return None
        back = (eff[n - 1] - eff[n - 5]).days
        if not 330 <= back <= 400:
            return None
        new, old = g["eps_rep"].iloc[n - 1], g["eps_rep"].iloc[n - 5]
        return bool(new > 0 and (old <= 0 or new >= 1.25 * old))

    def trades(self, kind: str, how: str) -> pd.DataFrame:
        rows = []
        for t in self.a:
            for x in mm.find_trades(self.a[t], self.ok[t], kind, how, setups=self.setups[t]):
                i = self.idx[t]
                day = i[x["t"]]
                rows.append({"ticker": t, "signal": day, "entry_date": i[x["k"]], "exit_date": i[x["end"]], "entry": x["entry"], "stop": x["stop"],
                             "pivot": x["pivot"], "gross": x["ret"], "net": x["ret"] - COST, "r": (x["ret"] - COST) / (1 - x["stop"] / x["entry"]),
                             "bars": x["bars"], "why": x["why"], "legs": [(i[j], f, p) for j, f, p in x["legs"]],
                             "rank": self.rank.at[day, t], "market": bool(self.market.get(day, False)), "rel_high": bool(self.rel_high.at[day, t]),
                             "k": x["k"], "before": i[x["k"] - 1]})
        cols = ["ticker", "signal", "entry_date"]
        return pd.DataFrame(rows).sort_values(["entry_date", "ticker"]).reset_index(drop=True) if rows else pd.DataFrame(columns=cols)

    def peers(self, tr: pd.DataFrame, how: str) -> pd.DataFrame:
        """Same-day template names of nearest ATR%, bought at the open with the same % stop and exit."""
        out = []
        for e in tr.itertuples():
            day, p = e.before, []
            if day in self.m.cal and e.entry_date in self.m.cal:
                atr = self.m.atr.loc[day]
                near = (atr[self.template.loc[day]].drop(e.ticker, errors="ignore").dropna() - atr[e.ticker]).abs().nsmallest(PEERS).index
                for t in near:
                    k = self.idx[t].searchsorted(e.entry_date)
                    if k < len(self.idx[t]) and self.idx[t][k] == e.entry_date:
                        price = self.a[t]["Open"][k]
                        x = mm.leave(self.a[t], k, price, price * e.stop / e.entry, how)
                        if x:
                            p.append(x["ret"] - COST)
            out.append(np.mean(p) if p else np.nan)
        return tr.assign(peer=out)

def period(tr: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = PERIODS[name]
    return tr[(tr["entry_date"] >= a) & (tr["entry_date"] <= b)].reset_index(drop=True)

def portfolio(tr: pd.DataFrame, m: bt.Market, start: str, end: str) -> Dict[str, float]:
    """Risk-sized book, marked at closes. Positions still open at the end are sold at that close."""
    cal = m.cal[(m.cal >= start) & (m.cal <= end)]
    entries = {d: g.sort_values("rank", ascending=False) for d, g in tr.groupby("entry_date")}
    cash, held, curve, taken, invested = 1.0, {}, [], 0, []
    marks = m.last
    for d in cal:
        before = cash + sum(h["n"] * marks.at[h["seen"], t] if h["seen"] is not None else h["n"] * h["entry"] for t, h in held.items())
        for e in (entries[d].itertuples() if d in entries else ()):
            if len(held) >= SLOTS or e.ticker in held:
                continue
            risk = 1 - e.stop / e.entry
            value = min(before * RISK / risk, before * MAX_WEIGHT, cash / (1 + COST / 2))
            if value <= before * 0.01:
                continue
            cash -= value * (1 + COST / 2)
            held[e.ticker] = {"n": value / e.entry, "full": value / e.entry, "legs": list(e.legs), "entry": e.entry, "seen": None}
            taken += 1
        for t in list(held):
            h = held[t]
            while h["legs"] and h["legs"][0][0] <= d:
                _, f, p = h["legs"].pop(0)
                cash += h["full"] * f * p * (1 - COST / 2)
                h["n"] -= h["full"] * f
            h["seen"] = d
            if not h["legs"]:
                del held[t]
        worth = {t: h["n"] * marks.at[d, t] for t, h in held.items()}
        curve.append(cash + sum(worth.values()))
        invested.append(sum(worth.values()) / curve[-1])
    eq = pd.Series(curve, index=cal)
    years = (cal[-1] - cal[0]).days / 365.25
    x = m.xiu_close.loc[cal]
    dd = lambda s: float((s / s.cummax() - 1).min())
    return {"cagr": eq.iloc[-1] ** (1 / years) - 1, "max_dd": dd(eq), "xiu_cagr": (x.iloc[-1] / x.iloc[0]) ** (1 / years) - 1, "xiu_dd": dd(x),
            "taken": taken, "invested": float(np.mean(invested)), "final": float(eq.iloc[-1])}

def stats(tr: pd.DataFrame) -> Dict[str, float]:
    if not len(tr):
        return {"n": 0}
    w, l = tr["net"][tr["net"] > 0], tr["net"][tr["net"] <= 0]
    lo, hi = bt.block_interval(tr["net"], tr["entry_date"])
    out = {"n": len(tr), "win": len(w) / len(tr), "avg_win": w.mean(), "avg_loss": l.mean(), "net": tr["net"].mean(), "lo": lo, "hi": hi,
           "median": tr["net"].median(), "r": tr["r"].mean(), "bars": tr["bars"].mean(), "no_top10": tr["net"].sort_values().iloc[:-10].mean()}
    if "peer" in tr:
        d = tr["net"] - tr["peer"]
        out["peer"], out["ex"] = tr["peer"].mean(), d.mean()
        out["ex_lo"], out["ex_hi"] = bt.block_interval(d, tr["entry_date"])
    return out

def line(st: Study, tr: pd.DataFrame, p: str) -> Dict[str, float]:
    x = period(tr, p)
    return {**stats(x), **portfolio(x, st.m, *PERIODS[p])}

def verdict(b: Dict[str, float]) -> bool:
    return bool(b["net"] > 0 and b["lo"] > 0 and b["ex"] > 0 and b["ex_lo"] > 0 and b["cagr"] > b["xiu_cagr"])

def choose(table: pd.DataFrame) -> Tuple[str, str]:
    a = table.xs("A", level="period")
    a = a[a["n"] >= MIN_TRADES]
    return a["cagr"].astype(float).idxmax()

FILTERS = {"none": lambda t: np.ones(len(t), bool), "market": lambda t: t["market"].to_numpy(bool), "rank90": lambda t: (t["rank"] >= 0.90).to_numpy(),
           "rel_high": lambda t: t["rel_high"].to_numpy(bool), "eps25": lambda t: (t["eps"] == True).to_numpy()}           # noqa: E712

def main() -> int:
    from research.breakouts import prices
    from research.breakouts.features import load_earnings
    st = Study(prices.load(), load_earnings())
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("setup days", sum(len(s) for s in st.setups.values()), "template days", int(st.template.to_numpy().sum()))
    runs = {(k, h): st.trades(k, h) for k in mm.ENTRIES for h in mm.EXITS}
    table = pd.DataFrame({(k, h, p): line(st, tr, p) for (k, h), tr in runs.items() for p in PERIODS}).T
    table.index.names = ["entry", "exit", "period"]
    table.to_csv(OUT / "grid.csv")
    print(table.round(4).to_string())
    best = choose(table)
    print("CHOSEN on A:", best)
    tr = runs[best]
    tr["eps"] = [st.growth(t, d) for t, d in zip(tr["ticker"], tr["before"])]
    tr = st.peers(tr, best[1])
    filt = pd.DataFrame({(n, p): line(st, tr[f(tr)], p) for n, f in FILTERS.items() for p in PERIODS}).T
    filt.to_csv(OUT / "filters.csv")
    print(filt.round(4).to_string())
    base = filt.loc[("none", "A"), "cagr"]
    kept = [n for n in FILTERS if n != "none" and filt.loc[(n, "A"), "n"] >= MIN_TRADES and filt.loc[(n, "A"), "cagr"] > base]
    print("filters that raise A:", kept)
    for n in ["none", *kept]:
        print(n, {p: verdict(filt.loc[(n, p)]) for p in ("B", "C")})
    tr.drop(columns=["legs"]).to_csv(OUT / "trades.csv", index=False)
    print(tr["why"].value_counts(normalize=True).round(3).to_dict())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
