"""
The Canadian rules, unchanged, on US stocks (PLAN.md). No network.

    python -m research.us.backtest
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from research.breakouts import backtest as bt
from research.breakouts import triangle
from research.breakouts.prices import after_seam
from research.minervini import backtest as bk
from research.minervini import method as mm
from research.us import prices as us_prices
from research.us.universe import SP500
from research.vcp import prices as vp

OUT = Path(__file__).resolve().parents[2] / "reports" / "us"
SPY = "SPY"
MIN_PRICE, MIN_DOLLAR, MIN_HISTORY = 5.0, 10_000_000.0, 252
ALL = ("2012-11-01", "2026-11-24")
VOLUME_X, HORIZON = 1.5, 60
PERIODS = {**bk.PERIODS, "ALL": ALL}


def worth_keeping(d: pd.DataFrame) -> bool:
    """Was it ever eligible (a cheap test on its own bars)?"""
    dollar = (d["Close"] * d["Volume"]).rolling(20).mean()
    return bool(((d["Close"] >= MIN_PRICE) & (dollar >= MIN_DOLLAR * 0.5)).iloc[MIN_HISTORY - 1:].any()) if len(d) >= MIN_HISTORY else False


def member_mask(members: pd.DataFrame, cal: pd.DatetimeIndex, columns) -> pd.DataFrame:
    out = pd.DataFrame(False, index=cal, columns=columns)
    for r in members.itertuples():
        if r.ticker in out.columns:
            end = r.end or "2100-01-01"
            out.loc[(cal >= r.start) & (cal < end), r.ticker] = True
    return out


class Market(bt.Market):
    """bt.Market on SPY's sessions with the US eligibility, optionally members only."""

    def __init__(self, data: Dict[str, pd.DataFrame], members: Optional[pd.DataFrame] = None):
        super().__init__({bt.XIU: data[SPY], **{t: d for t, d in data.items() if t != SPY}})
        wide = lambda col: pd.DataFrame({t: d[col].reindex(self.cal) for t, d in self.bars.items()})
        close = wide("Close")
        dollar = (close * wide("Volume")).fillna(0.0).rolling(20).mean()
        self.eligible = (close >= MIN_PRICE) & (dollar >= MIN_DOLLAR) & (close.notna().cumsum() >= MIN_HISTORY)
        if members is not None:
            self.eligible &= member_mask(members, self.cal, close.columns)


def load() -> Dict[str, pd.DataFrame]:
    conn = vp.connect(str(us_prices.DB))
    try:
        data = {t: after_seam(d) for t, d in vp.load_all(conn).items() if t != bt.XIU}
    finally:
        conn.close()
    return {t: d for t, d in data.items() if t == SPY or worth_keeping(d)}


# ── H1 ───────────────────────────────────────────────────────────────────────

def week_trades(st: bk.Study) -> pd.DataFrame:
    """The one-week order as the live scan emails it."""
    rows = []
    for t in st.a:
        i = st.idx[t]
        for x in mm.week_trades(st.a[t], st.ok[t], st.setups[t], cancel=False):
            if x["state"] == "bought":
                rows.append({"ticker": t, "signal": i[x["t"]], "entry_date": i[x["k"]], "entry": x["entry"], "stop": x["stop"], "net": x["ret"] - bk.COST,
                             "r": (x["ret"] - bk.COST) / (1 - x["stop"] / x["entry"]), "bars": x["bars"], "why": x["why"],
                             "legs": [(i[j], f, p) for j, f, p in x["legs"]], "rank": st.rank.at[i[x["t"]], t]})
    return pd.DataFrame(rows).sort_values(["entry_date", "ticker"]).reset_index(drop=True)


def part(tr: pd.DataFrame, p: str) -> pd.DataFrame:
    a, b = PERIODS[p]
    return tr[(tr["entry_date"] >= a) & (tr["entry_date"] <= b)].reset_index(drop=True)


def lines(st: bk.Study, tr: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({p: {**bk.stats(part(tr, p)), **bk.portfolio(part(tr, p), st.m, *PERIODS[p])} for p in PERIODS}).T


def h1(t: pd.DataFrame) -> Dict[str, bool]:
    a = t.loc["ALL"]
    c = {"net": a["net"] > 0 and a["lo"] > 0, "excess": a["ex"] > 0 and a["ex_lo"] > 0, "book": a["cagr"] > a["xiu_cagr"],
         "periods": sum(t.at[p, "cagr"] > t.at[p, "xiu_cagr"] for p in "ABC") >= 2}
    c = {k: bool(v) for k, v in c.items()}
    return {**c, "passed": all(c.values())}


# ── H2 ───────────────────────────────────────────────────────────────────────

def difference(e: pd.Series, flag: pd.Series, dates: pd.Series, n: int = 5000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean of the flagged less mean of the others, 95% interval by 13-week blocks."""
    e, a = e.to_numpy(), flag.to_numpy(bool)
    block = ((dates - dates.min()).dt.days // 91).to_numpy()
    s = np.array([[e[(block == b) & a].sum(), ((block == b) & a).sum(), e[(block == b) & ~a].sum(), ((block == b) & ~a).sum()] for b in np.unique(block)])
    q = s[np.random.default_rng(seed).integers(0, len(s), (n, len(s)))].sum(axis=1)
    d = q[:, 0] / q[:, 1] - q[:, 2] / q[:, 3]
    return float(e[a].mean() - e[~a].mean()), float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))


def h2(ev: pd.DataFrame, h: int = HORIZON) -> Dict[str, float]:
    x = ev[(ev["vol_x"] >= VOLUME_X) & ev[f"r{h}"].notna() & ev[f"a{h}"].notna()].reset_index(drop=True)
    e, asc = x[f"r{h}"] - x[f"a{h}"], x["kind"] == "asc"
    lo, hi = bt.block_interval(e[asc], x["date"][asc])
    d, dlo, dhi = difference(e, asc, x["date"])
    s = e[asc].sort_values()
    return {"n": int(asc.sum()), "up": float((x[f"r{h}"][asc] > 0).mean()), "net": float(x[f"r{h}"][asc].mean() - bt.COST), "ex": float(e[asc].mean()), "lo": lo, "hi": hi,
            "median": float(e[asc].median()), "no_top1pct": float(s.iloc[:-max(1, len(s) // 100)].mean()), "others": int((~asc).sum()), "diff": d, "dlo": dlo, "dhi": dhi,
            "passed": bool(e[asc].mean() > 0 and lo > 0 and e[asc].median() > 0 and d > 0 and dlo > 0)}


def equal_weight(m: bt.Market, p: str) -> float:
    """All eligible names held equally, rebalanced daily, no costs: what the list itself does."""
    r = (m.last / m.last.shift(1) - 1).where(m.eligible.shift(1, fill_value=False)).mean(axis=1)
    s = r[(r.index >= PERIODS[p][0]) & (r.index <= PERIODS[p][1])].fillna(0.0)
    return float((1 + s).prod() ** (365.25 / (s.index[-1] - s.index[0]).days) - 1)


def run(data: Dict[str, pd.DataFrame], members: Optional[pd.DataFrame], tag: str) -> None:
    m = Market(data, members)
    per_day = m.eligible.sum(axis=1)
    print(f"===== {tag}: {len(m.bars)} names, eligible a day {per_day.loc['2013'].mean():.0f} (2013) {per_day.loc['2020'].mean():.0f} (2020) {per_day.loc['2026'].mean():.0f} (2026)")
    if members is not None:
        known = member_mask(members, m.cal, sorted(set(members["ticker"])))
        print("  members with Yahoo prices:", {y: round(float((m.eligible.loc[y].sum(axis=1) / known.loc[y].sum(axis=1)).mean()), 3) for y in ("2013", "2016", "2020", "2024", "2026")})
    print("  equal weight of the list a year:", {p: round(equal_weight(m, p), 3) for p in PERIODS})
    st = bk.Study(data, market=m)
    main_tr = st.peers(st.trades("stop", "t20"), "t20")
    t = lines(st, main_tr)
    t.to_csv(OUT / f"h1_{tag}.csv")
    cols = ["n", "win", "avg_win", "avg_loss", "net", "lo", "hi", "median", "no_top10", "peer", "ex", "ex_lo", "ex_hi", "cagr", "max_dd", "xiu_cagr", "xiu_dd", "taken", "invested"]
    print(t[cols].astype(float).round(4).to_string())
    print("H1", h1(t))
    print("one-week order\n", lines(st, week_trades(st))[[c for c in cols if c not in ("peer", "ex", "ex_lo", "ex_hi")]].astype(float).round(4).to_string())
    grid = {(k, e): lines(st, st.trades(k, e)).loc["ALL"] for k in mm.ENTRIES for e in mm.EXITS}
    g = pd.DataFrame(grid).T[["n", "win", "net", "lo", "hi", "bars", "cagr", "max_dd"]].astype(float)
    g.to_csv(OUT / f"grid_{tag}.csv")
    print(g.round(4).to_string())
    main_tr.drop(columns=["legs"]).to_csv(OUT / f"h1_trades_{tag}.csv.gz", index=False)
    del st
    ev, mask = triangle.find_events(m)
    ev = bt.add_returns(ev, mask, m, (20, HORIZON))
    spans = {"ALL": ev, "A": ev[ev["date"] <= "2018-12-31"], "B": ev[(ev["date"] >= "2019") & (ev["date"] < "2024")], "C": ev[ev["date"] >= "2024"]}
    h = pd.DataFrame({(p, d): h2(x, d) for p, x in spans.items() for d in (20, HORIZON)}).T
    h.to_csv(OUT / f"h2_{tag}.csv")
    print(h.to_string(float_format=lambda v: f"{v:.4f}"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    data = load()
    run(data, None, "all")
    run(data, pd.read_csv(SP500, keep_default_na=False), "sp500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
