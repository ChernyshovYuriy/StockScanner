"""
Every price-based sleeve of the project as trades on one market (PLAN.md), before costs.

    python -m research.us.sleeves us|ca
"""
from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import volume_breakout_backtest as vb
from config import KANGAROO_MAX_HOLD_DAYS, KANGAROO_RR_TARGET
from research.breakouts import backtest as bt
from research.breakouts import triangle
from research.breakouts.features import slope200
from research.kangaroo_tail.detector import scan_history
from research.minervini import backtest as bk
from research.us import backtest as us

DIP, TRAIL = 0.20, 0.25


# ── Trades of each sleeve ────────────────────────────────────────────────────

def held(d: pd.DataFrame, k: int, sessions: int, target: float = np.inf) -> Optional[Dict]:
    """Bought at the open of bar k, sold at the close `sessions` later (or at +target%)."""
    if k >= len(d) or d["Open"].iat[k] <= 0:
        return None
    entry, end = d["Open"].iat[k], min(len(d) - 1, k + sessions)
    h, o = d["High"].to_numpy(), d["Open"].to_numpy()
    for j in range(k, end + 1) if np.isfinite(target) else ():
        if h[j] >= entry * (1 + target):
            p = entry * (1 + target) if j == k else max(o[j], entry * (1 + target))
            return {"k": k, "end": j, "entry": entry, "exit": p}
    return {"k": k, "end": end, "entry": entry, "exit": d["Close"].iat[end]}


def kangaroo(d: pd.DataFrame, ok: np.ndarray) -> List[Dict]:
    """The sleeve's own mechanic: buy stop at the tail's high for 10 sessions; stop at its low, target 1.5 times
    the risk, 10 sessions; the stop first."""
    o, h, l, c = (d[x].to_numpy(float) for x in ("Open", "High", "Low", "Close"))
    out, free = [], -1
    for s in scan_history("X", d):
        t = d.index.get_loc(pd.Timestamp(s.date))
        if t <= free or not ok[t] or s.high <= s.low:
            continue
        k = next((j for j in range(t + 1, min(t + KANGAROO_MAX_HOLD_DAYS, len(d) - 1) + 1) if h[j] >= s.high), None)
        if k is None:
            continue
        entry, stop = s.high, s.low
        target, end, price = entry + KANGAROO_RR_TARGET * (entry - stop), None, None
        for j in range(k, min(k + KANGAROO_MAX_HOLD_DAYS, len(d) - 1) + 1):
            if l[j] <= stop:
                end, price = j, stop if j == k else min(o[j], stop)
                break
            if h[j] >= target:
                end, price = j, target if j == k else max(o[j], target)
                break
        if end is None:
            end = min(k + KANGAROO_MAX_HOLD_DAYS, len(d) - 1)
            price = c[end]
        out.append({"t": t, "k": k, "end": end, "entry": max(entry, o[k]) if o[k] > entry else entry, "exit": price, "stop": stop})
        free = end
    return out


def conviction(d: pd.DataFrame, ok: np.ndarray) -> List[Dict]:
    c, o = d["Close"].to_numpy(float), d["Open"].to_numpy(float)
    high = d["Close"].rolling(252).max().to_numpy()
    dip = 1 - c / high >= DIP
    out, free = [], -1
    for t in np.where(dip[1:] & ~dip[:-1] & np.isfinite(high[:-1]))[0] + 1:
        if t <= free or not ok[t] or t + 1 >= len(d):
            continue
        k, peak, end = t + 1, o[t + 1], None
        for j in range(k, len(d)):
            peak = max(peak, c[j])
            if c[j] <= (1 - TRAIL) * peak:
                end = j + 1
                break
        done = end is not None and end < len(d)
        out.append({"t": int(t), "k": int(k), "end": int(end if done else len(d) - 1), "entry": o[k], "exit": o[end] if done else c[-1],
                    "stop": o[k] * (1 - TRAIL), "rank": float(1 - c[t] / high[t])})
        free = end if done else len(d)
    return out


def volume_breakout(variant: str, sessions: int) -> Callable:
    def f(d: pd.DataFrame, ok: np.ndarray) -> List[Dict]:
        _, _, variants = vb.signal_masks(d)
        return [{"t": int(t), **x} for t in np.flatnonzero(vb._cooldown(variants[variant] & ok)) if (x := held(d, int(t) + 1, sessions))]
    return f


def per_name(args) -> List[Dict]:
    ticker, d, ok, fs = args
    return [{"sleeve": n, "ticker": ticker, **x} for n, f in fs.items() for x in f(d, ok)]


def simple(m: bt.Market) -> Dict[str, List[Dict]]:
    """Sleeves that need one name's bars only."""
    vb.MIN_PRICE, vb.MIN_DOLLAR_VOL = 0.0, 0.0                 # eligibility is the market's
    fs = {"kangaroo": kangaroo, "conviction": conviction}
    for v in ("breakout_vol3", "breakout_vol3_quiet", "breakout_vol5_quiet"):
        for s in (20, 60):
            fs[f"{v} {s}"] = volume_breakout(v, s)
    rows = []
    for t, d in m.bars.items():
        if len(d) > 260:
            rows += per_name((t, d, m.eligible[t].reindex(d.index, fill_value=False).to_numpy(bool), fs))
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        out.setdefault(r.pop("sleeve"), []).append(r)
    return out


def breakouts(m: bt.Market) -> Dict[str, List[Dict]]:
    ev, _ = bt.find_events(m)
    out: Dict[str, List[Dict]] = {"breakout 3%, month": [], "breakout 3%, month, 200d rising": [], "breakout 3%, +5% in a month": []}
    for e in ev.itertuples():
        d = m.bars[e.ticker]
        t = d.index.get_loc(e.date)
        x = held(d, t + 1, 20)
        if x:
            out["breakout 3%, month"].append({"ticker": e.ticker, "t": t, **x})
            if slope200(d["Close"].to_numpy(float), t) > 0:
                out["breakout 3%, month, 200d rising"].append({"ticker": e.ticker, "t": t, **x})
            out["breakout 3%, +5% in a month"].append({"ticker": e.ticker, "t": t, **held(d, t + 1, 20, 0.05)})
    return out


def triangles(m: bt.Market) -> Dict[str, List[Dict]]:
    ev, _ = triangle.find_events(m)
    out: Dict[str, List[Dict]] = {}
    for e in ev.itertuples():
        d = m.bars[e.ticker]
        t = d.index.get_loc(e.date)
        for s in (20, 60):
            x = held(d, t + 1, s)
            for n, ok in ((f"triangle {s}", e.kind == "asc"), (f"triangle + volume {s}", e.kind == "asc" and e.vol_x >= us.VOLUME_X)):
                if x and ok:
                    out.setdefault(n, []).append({"ticker": e.ticker, "t": t, **x})
    return out


def frame(rows: List[Dict], m: bt.Market, rank: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in rows:
        i = m.bars[r["ticker"]].index
        day = i[r["t"]]
        out.append({"ticker": r["ticker"], "signal": day, "entry_date": i[r["k"]], "exit_date": i[r["end"]], "entry": r["entry"],
                    "stop": r.get("stop", r["entry"] * 0.92), "net": r["exit"] / r["entry"] - 1, "bars": r["end"] - r["k"] + 1,
                    "rank": r.get("rank", rank.at[day, r["ticker"]]), "legs": [(i[r["end"]], 1.0, r["exit"])]})
    d = pd.DataFrame(out)
    d["r"] = d["net"] / (1 - d["stop"] / d["entry"]).clip(lower=0.01)
    return d.sort_values(["entry_date", "ticker"]).reset_index(drop=True)


# ── Yardsticks ───────────────────────────────────────────────────────────────

def same_days(tr: pd.DataFrame, m: bt.Market) -> np.ndarray:
    """Mean return of every eligible name from the trade's entry open to its exit close."""
    opens, last, elig = m.open.to_numpy(float), m.last.to_numpy(float), m.eligible.to_numpy(bool)
    cal = m.cal
    out = []
    for e in tr.itertuples():
        a, b = cal.searchsorted(e.entry_date), cal.searchsorted(e.exit_date)
        if a == 0 or a >= len(cal) or b >= len(cal):
            out.append(np.nan)
            continue
        ok = elig[a - 1] & (opens[a] > 0)
        out.append(float(np.nanmean(last[b][ok] / opens[a][ok] - 1)) if ok.any() else np.nan)
    return np.array(out)


def judge(tr: pd.DataFrame, m: bt.Market) -> pd.DataFrame:
    tr = tr.assign(peer=same_days(tr, m))
    rows = {}
    for p in us.PERIODS:
        x = us.part(tr, p)
        if len(x) < 2:
            continue
        s = bk.stats(x)
        rows[p] = {k: s[k] for k in ("n", "win", "avg_win", "avg_loss", "net", "median", "bars", "peer", "ex", "ex_lo", "ex_hi", "no_top10")} | {
            k: v for k, v in bk.portfolio(x, m, *us.PERIODS[p]).items() if k in ("cagr", "max_dd", "xiu_cagr", "xiu_dd", "invested")}
    return pd.DataFrame(rows).T


def better(t: pd.DataFrame) -> bool:
    a = t.loc["ALL"]
    return bool(a["ex_lo"] > 0 and a["cagr"] > a["xiu_cagr"] and sum(p in t.index and t.at[p, "cagr"] > t.at[p, "xiu_cagr"] for p in "ABC") >= 2)


def momentum_top10(m: bt.Market) -> pd.Series:
    """Monthly: the ten eligible names with the best 12-1 month return, equal weight; daily returns."""
    ends = pd.Series(m.cal, index=m.cal).groupby(m.cal.to_period("M")).last()
    r = m.last / m.last.shift(1) - 1
    score = m.last.shift(21) / m.last.shift(252) - 1
    parts = []
    for a, b in zip(ends[:-1], ends[1:]):
        pick = score.loc[a].where(m.eligible.loc[a]).nlargest(10).index
        path = (1 + r.loc[(r.index > a) & (r.index <= b), pick].fillna(0)).cumprod()
        book = path.mean(axis=1)
        parts.append(book / book.shift(1).fillna(1.0) - 1 if len(book) else book)
    return pd.concat(parts)


def cagr(s: pd.Series, p: str) -> float:
    s = s[(s.index >= us.PERIODS[p][0]) & (s.index <= us.PERIODS[p][1])]
    return float((1 + s).prod() ** (365.25 / (s.index[-1] - s.index[0]).days) - 1)


def run(m: bt.Market, data: Dict[str, pd.DataFrame], tag: str) -> None:
    bk.COST = 0.0
    st = bk.Study(data, market=m)
    groups = {**simple(m), **breakouts(m), **triangles(m)}
    frames = {k: frame(v, m, st.rank) for k, v in groups.items() if len(v) > 20}
    for name, tr in (("minervini +20%", st.trades("stop", "t20")), ("minervini +10% in a month", st.trades("stop", "m21t10")), ("minervini one-week order", us.week_trades(st))):
        if "exit_date" not in tr:
            tr["exit_date"] = [x[-1][0] for x in tr["legs"]]
        frames[name] = tr
    tables = {k: judge(v, m) for k, v in frames.items()}
    out = pd.concat(tables, names=["sleeve", "period"])
    us.OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(us.OUT / f"sleeves_{tag}.csv")
    show = out.xs("ALL", level="period").copy()
    show["periods_ahead"] = [sum(p in t.index and t.at[p, "cagr"] > t.at[p, "xiu_cagr"] for p in "ABC") for t in tables.values()]
    show["better"] = [better(t) for t in tables.values()]
    print(f"===== {tag}: {len(m.bars)} names; equal weight a year", {p: round(us.equal_weight(m, p), 3) for p in us.PERIODS})
    print(show.sort_values("cagr", ascending=False).round(4).to_string())
    top = momentum_top10(m)
    print("momentum 12-1 top ten a year", {p: round(cagr(top, p), 3) for p in us.PERIODS})


def main() -> int:
    tag = sys.argv[1]
    pd.set_option("display.width", 260, "display.max_columns", 40)
    if tag == "us":
        data = us.load()
        m = us.Market(data)
    else:
        from research.breakouts import prices
        data = prices.load()
        m = bt.Market(data)
    run(m, data, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
