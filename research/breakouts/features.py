"""
What the successful breakouts have in common (PLAN.md). Every feature is known at
the breakout's close. No network.

    python -m research.breakouts.features            # periods A and B
    python -m research.breakouts.features holdout    # period C, once
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from research.breakouts import backtest as bt

ROOT = Path(__file__).resolve().parents[2]
A_END, B_START, B_END = "2018-11-30", "2019-01-02", "2023-06-30"
GAIN, LOSS, BIG = 0.05, -0.05, 0.15
SECTORS = ("Materials", "Energy", "Industrials", "Financials", "Real Estate", "Utilities",
           "Information Technology", "Consumer")
YAHOO_SECTOR = {"Basic Materials": "Materials", "Financial Services": "Financials", "Technology": "Information Technology",
                "Healthcare": "Health Care", "Communication Services": "Communication",
                "Consumer Cyclical": "Consumer", "Consumer Defensive": "Consumer",
                "Consumer Discretionary": "Consumer", "Consumer Staples": "Consumer",
                "Telecommunication Services": "Communication"}

FEATURES: Dict[str, str] = {
    # the base
    "bars": "length of the base", "tests": "tests of the line", "depth": "depth of the base",
    "range10": "range of the last 10 days (small = tight)", "squeeze": "ATR 10 days / ATR 60 days before",
    "near_line": "share of last 20 closes within 3% of the line", "right_low": "lowest of last 20 days vs line",
    "rising_lows": "second-half low above first-half low", "run_in": "10-day run before the breakout",
    "cross_days": "days from first close over the line", "prior_up": "6-month move before the base",
    # the breakout day
    "day_ret": "breakout day's gain", "day_atr": "breakout day's gain in ATRs", "gap": "opening gap",
    "ext": "close above the line", "close_pos": "close within the day's range",
    "vol_x": "volume / 50-day average", "vol_dry": "volume 10 days before / 50 days", "up_vol": "share of 50-day volume on up days",
    # risk
    "stop_dist": "distance to the stop", "target_dist": "distance to the target", "target_atr": "target in ATRs",
    # the stock's trend
    "ret21": "1-month return", "ret63": "3-month return", "ret126": "6-month return", "ret252": "12-month return",
    "dist50": "above the 50-day average", "dist200": "above the 200-day average", "ma_gap": "50-day over 200-day average",
    "slope200": "slope of the 200-day average", "overhead": "distance from the 3-year high", "off_low": "above the 12-month low",
    "atr": "volatility (ATR%)", "price": "share price", "dollar": "dollar volume", "past_breakouts": "breakouts in the past year",
    # strength against other names
    "rs21": "1-month strength rank", "rs63": "3-month strength rank", "rs126": "6-month strength rank", "rs252": "12-month strength rank",
    # the market
    "xiu21": "XIU 1 month", "xiu63": "XIU 3 months", "xiu200": "XIU above its 200-day average", "xiu_vol": "XIU volatility",
    "breadth": "names above their 200-day average", "highs": "names at new highs", "cluster": "breakouts in the last 20 days",
    # the sector
    "sec21": "sector 1 month vs market", "sec63": "sector 3 months vs market", "sec_cluster": "sector breakouts in 20 days",
    # the event
    "earn_recent": "reported earnings within 3 days", "earn_soon": "earnings due within 21 days",
    "surprise": "latest earnings surprise", "news_day": "gap >= 2% on >= 2x volume",
    **{f"is_{s}": f"sector {s}" for s in SECTORS},
}
FLAGS = ("earn_recent", "earn_soon", "news_day", *(f"is_{s}" for s in SECTORS))


def _ratio(a: float, b: float) -> float:
    return a / b if b and np.isfinite(b) else np.nan


def slope200(c: np.ndarray, t: int) -> float:
    """Change of the 200-day average over the last 21 sessions (rising when positive)."""
    return c[t - 199:t + 1].mean() / c[t - 220:t - 20].mean() - 1 if t >= 221 else np.nan


def stock_features(d: pd.DataFrame, t: int, level: float, start: int, low: float, target: float, stop: float) -> Dict[str, float]:
    """From the stock's own bars up to t (nothing later is read)."""
    o, h, l, c, v = (d[x].to_numpy(float)[:t + 1] for x in ("Open", "High", "Low", "Close", "Volume"))
    prev = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax(np.fmax(h - l, np.abs(h - prev)), np.abs(l - prev))
    ret = lambda n: c[t] / c[t - n] - 1 if t >= n else np.nan
    ma = lambda n, k=t: c[k - n + 1:k + 1].mean() if k >= n - 1 else np.nan
    atr_before = tr[max(0, t - 20):t].mean()
    half = start + (t - start) // 2
    over = np.where(c[start:t + 1] <= level)[0]
    up = c[t - 49:t + 1] > prev[t - 49:t + 1] if t >= 50 else np.array([])
    return {
        "bars": t - start, "depth": 1 - low / level, "range10": (h[t - 10:t].max() - l[t - 10:t].min()) / level,
        "squeeze": _ratio(tr[t - 10:t].mean(), tr[max(1, t - 60):t].mean()),
        "near_line": float((c[t - 20:t] >= level * 0.97).mean()), "right_low": l[t - 20:t].min() / level - 1,
        "rising_lows": (l[half:t].min() - l[start:half].min()) / (level - low) if level > low else np.nan,
        "run_in": c[t - 1] / c[t - 11] - 1, "cross_days": t - (start + over[-1]) if len(over) else np.nan,
        "prior_up": c[start] / c[start - 126] - 1 if start >= 126 else np.nan,
        "day_ret": c[t] / c[t - 1] - 1, "day_atr": _ratio(c[t] - c[t - 1], atr_before), "gap": o[t] / c[t - 1] - 1,
        "ext": c[t] / level - 1, "close_pos": (c[t] - l[t]) / (h[t] - l[t]) if h[t] > l[t] else np.nan,
        "vol_x": _ratio(v[t], v[t - 50:t].mean()) if t >= 50 else np.nan,
        "vol_dry": _ratio(v[t - 10:t].mean(), v[t - 50:t].mean()) if t >= 50 else np.nan,
        "up_vol": _ratio(v[t - 49:t + 1][up].sum(), v[t - 49:t + 1].sum()) if len(up) else np.nan,
        "stop_dist": 1 - stop / c[t], "target_dist": target / c[t] - 1, "target_atr": _ratio(target - c[t], atr_before),
        "ret21": ret(21), "ret63": ret(63), "ret126": ret(126), "ret252": ret(252),
        "dist50": c[t] / ma(50) - 1, "dist200": c[t] / ma(200) - 1, "ma_gap": ma(50) / ma(200) - 1,
        "slope200": slope200(c, t),
        "overhead": c[t] / h[max(0, t - 756):t].max() - 1 if t >= 252 else np.nan,
        "off_low": c[t] / l[max(0, t - 252):t + 1].min() - 1 if t >= 252 else np.nan,
        "atr": atr_before / c[t - 1], "price": c[t],
        "news_day": float(o[t] / c[t - 1] - 1 >= 0.02 and v[t] >= 2 * v[t - 50:t].mean()) if t >= 50 else np.nan,
    }


def load_sectors() -> Dict[str, str]:
    from research.swing_rank import holdings
    out: Dict[str, str] = {}
    info = ROOT / "cache" / "conviction_watchlist_info_cache.json"
    if info.exists():
        for t, x in json.loads(info.read_text()).items():
            if x and x.get("sector"):
                out[t] = x["sector"]
    for members in holdings.load().values():
        for t, s in members["sector"].items():
            out[holdings.yahoo_symbol(t)] = s
    return {t: YAHOO_SECTOR.get(s, s) for t, s in out.items()}


def load_earnings() -> pd.DataFrame:
    e = pd.read_csv(ROOT / "out" / "earnings_dates_cache.csv", parse_dates=["report_ts"]).dropna(subset=["report_ts"])
    return e.sort_values(["ticker", "report_ts"])


def earnings_features(dates: np.ndarray, surprise: np.ndarray, cal: pd.DatetimeIndex, day: pd.Timestamp) -> Dict[str, float]:
    """`dates` are report times, ascending. A report after 16:00 belongs to the next session."""
    nan = {"earn_recent": np.nan, "earn_soon": np.nan, "surprise": np.nan}
    if not len(dates) or pd.Timestamp(dates[0]) > day - pd.Timedelta(days=100):
        return nan
    ts = pd.DatetimeIndex(dates)
    eff = np.where(ts.hour >= 16, ts.normalize() + pd.Timedelta(days=1), ts.normalize())
    k, pos = cal.get_loc(day), cal.searchsorted(pd.DatetimeIndex(eff))       # session that could react
    past = pos <= k
    s = surprise[past][-1] if past.any() and k - pos[past][-1] <= 63 else np.nan
    return {"earn_recent": float(((k - pos >= 0) & (k - pos <= 2)).any()),
            "earn_soon": float(((pos - k >= 1) & (pos - k <= 21)).any()), "surprise": s}


def build(ev: pd.DataFrame, mask: pd.DataFrame, m: bt.Market, sectors: Dict[str, str], earnings: Optional[pd.DataFrame]) -> pd.DataFrame:
    cal, last, elig = m.cal, m.last, m.eligible
    e = lambda p: p.where(elig)
    rank = {n: e(last / last.shift(n) - 1).rank(axis=1, pct=True) for n in (21, 63, 126, 252)}
    above = e((last > last.rolling(200).mean()).astype(float).where(last.rolling(200).mean().notna()))
    xiu = m.xiu_close
    market = pd.DataFrame({
        "xiu21": xiu / xiu.shift(21) - 1, "xiu63": xiu / xiu.shift(63) - 1, "xiu200": xiu / xiu.rolling(200).mean() - 1,
        "xiu_vol": xiu.pct_change().rolling(20).std(), "breadth": above.mean(axis=1),
        "highs": e(m.new_high.astype(float)).mean(axis=1)})
    seen = mask.rolling(20, min_periods=1).max().where(elig)
    market["cluster"] = seen.mean(axis=1)
    sec_of = pd.Series({t: sectors.get(t) for t in last.columns})
    sec = {}
    for s in sec_of.dropna().unique():
        cols = sec_of.index[sec_of == s]
        sec[s] = pd.DataFrame({f"sec{n}": e(last / last.shift(n) - 1)[cols].mean(axis=1) - e(last / last.shift(n) - 1).mean(axis=1)
                               for n in (21, 63)} | {"sec_cluster": seen[cols].mean(axis=1)})
    dollar = pd.DataFrame({t: (d["Close"] * d["Volume"]).reindex(cal) for t, d in m.bars.items()}).fillna(0.0).rolling(20).mean()
    starts = mask & ~mask.shift(1, fill_value=False).rolling(20, min_periods=1).max().astype(bool)
    reports = {t: g for t, g in earnings.groupby("ticker")} if earnings is not None else {}
    rows = []
    for x in ev.itertuples():
        d, day = m.bars[x.ticker], x.date
        t = d.index.get_loc(day)
        f = stock_features(d, t, x.level, t - x.bars, x.low, x.target, x.lfd_low)
        f["tests"], f["dollar"] = x.tests, dollar.at[day, x.ticker]
        k = cal.get_loc(day)
        f["past_breakouts"] = float(starts[x.ticker].iloc[max(0, k - 252):k].sum())
        for n in rank:
            f[f"rs{n}"] = rank[n].at[day, x.ticker]
        f.update(market.loc[day])
        s = sectors.get(x.ticker)
        for s1 in SECTORS:
            f[f"is_{s1}"] = np.nan if s is None else float(s == s1)
        f.update(sec[s].loc[day] if s else {"sec21": np.nan, "sec63": np.nan, "sec_cluster": np.nan})
        g = reports.get(x.ticker)
        f.update(earnings_features(g["report_ts"].to_numpy() if g is not None else np.array([]),
                                   g["surprise_pct"].to_numpy() if g is not None else np.array([]), cal, day))
        rows.append(f)
    return pd.DataFrame(rows)[list(FEATURES)].astype(float)


# ── Judging ──────────────────────────────────────────────────────────────────

def score(ev: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    return (ev["r20"] - ev["a20"]) / f["atr"]


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() and ry.std() else np.nan


def test_feature(x: pd.Series, y: pd.Series, dates: pd.Series, n: int = 5000, seed: int = 0) -> Dict[str, float]:
    """Rank correlation and the two-sided p of its 13-week block bootstrap (normal approximation)."""
    ok = (x.notna() & y.notna()).to_numpy()
    rx, ry = x[ok].rank().to_numpy(), y[ok].rank().to_numpy()
    if ok.sum() < 100 or not rx.std():
        return {"n": int(ok.sum()), "ic": np.nan, "p": np.nan, "z": np.nan}
    dd = dates[ok]
    block = ((dd - dd.min()).dt.days // 91).to_numpy()
    groups = np.unique(block)
    zx, zy = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
    cols = np.column_stack([zx, zy, zx * zy, zx * zx, zy * zy, np.ones(len(zx))])
    sums = np.array([cols[block == g].sum(axis=0) for g in groups])
    pick = np.random.default_rng(seed).integers(0, len(groups), (n, len(groups)))
    s = sums[pick].sum(axis=1)
    mx, my = s[:, 0] / s[:, 5], s[:, 1] / s[:, 5]
    boot = (s[:, 2] / s[:, 5] - mx * my) / np.sqrt((s[:, 3] / s[:, 5] - mx ** 2) * (s[:, 4] / s[:, 5] - my ** 2))
    ic = float((zx * zy).mean())
    z = ic / float(np.nanstd(boot))
    from math import erfc, sqrt
    return {"n": int(ok.sum()), "ic": ic, "p": erfc(abs(z) / sqrt(2)), "z": z}


def benjamini_hochberg(p: pd.Series, q: float = 0.10) -> pd.Series:
    s = p.dropna().sort_values()
    ok = s.to_numpy() <= q * np.arange(1, len(s) + 1) / len(s)
    cut = s.iloc[np.where(ok)[0][-1]] if ok.any() else -1.0
    return (p <= cut) & p.notna()


def best_third(x: pd.Series, a: pd.Series, sign: float, flag: bool) -> pd.Series:
    """Members of A's best third (by A's cut point), applied to x."""
    if flag:
        return (x == (1.0 if sign > 0 else 0.0)).where(x.notna())
    cut = a.quantile(2 / 3 if sign > 0 else 1 / 3)
    return ((x >= cut) if sign > 0 else (x <= cut)).where(x.notna())


def ranked(x: pd.DataFrame, a: pd.DataFrame) -> np.ndarray:
    """Each column as its percentile among A's values (-0.5..0.5); missing = 0."""
    out = np.zeros(x.shape)
    for j, c in enumerate(x):
        ref = np.sort(a[c].dropna().to_numpy())
        if len(ref):
            v = x[c].to_numpy()
            pct = (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right")) / 2 / len(ref) - 0.5
            out[:, j] = np.where(np.isnan(v), 0.0, pct)
    return out


def ridge(xa: np.ndarray, y: np.ndarray, lam: float = 100.0) -> np.ndarray:
    yc = pd.Series(y).rank(pct=True).to_numpy() - 0.5
    xc = xa - xa.mean(axis=0)
    return np.linalg.solve(xc.T @ xc + lam * np.eye(xa.shape[1]), xc.T @ (yc - yc.mean()))


def tree(x: np.ndarray, y: np.ndarray, depth: int = 3, leaf: int = 60) -> Dict:
    """Regression tree on ranked y; splits at deciles."""
    node = {"value": float(y.mean()), "n": len(y)}
    if depth == 0 or len(y) < 2 * leaf:
        return node
    best = (0.0, None, None)
    for j in range(x.shape[1]):
        for cut in np.unique(np.quantile(x[:, j], np.arange(0.1, 1.0, 0.1))):
            left = x[:, j] <= cut
            n = int(left.sum())
            if n < leaf or len(y) - n < leaf:
                continue
            gain = n * (len(y) - n) / len(y) * (y[left].mean() - y[~left].mean()) ** 2
            if gain > best[0]:
                best = (gain, j, cut)
    if best[1] is None:
        return node
    left = x[:, best[1]] <= best[2]
    return {**node, "j": best[1], "cut": float(best[2]), "lo": tree(x[left], y[left], depth - 1, leaf), "hi": tree(x[~left], y[~left], depth - 1, leaf)}


def leaves(node: Dict, path: tuple = ()) -> List[tuple]:
    if "j" not in node:
        return [(node["value"], node["n"], path)]
    return leaves(node["lo"], path + ((node["j"], "<=", node["cut"]),)) + leaves(node["hi"], path + ((node["j"], ">", node["cut"]),))


def follow(x: np.ndarray, path: tuple) -> np.ndarray:
    ok = np.ones(len(x), bool)
    for j, op, cut in path:
        ok &= (x[:, j] <= cut) if op == "<=" else (x[:, j] > cut)
    return ok


def group(ev: pd.DataFrame, pick: np.ndarray) -> Dict[str, float]:
    g = ev[pick]
    g = g[g["r20"].notna()]
    ex = g["r20"] - g["a20"]
    lo, hi = bt.block_interval(ex, g["date"])
    return {"n": len(g), "gained": (g["r20"] >= GAIN).mean(), "lost": (g["r20"] <= LOSS).mean(), "big": (g["r20"] >= BIG).mean(),
            "up": (g["r20"] > 0).mean(), "net": (g["r20"] - bt.COST).mean(), "ex": ex.mean(), "lo": lo, "hi": hi}


def analyse(ev: pd.DataFrame, f: pd.DataFrame) -> tuple:
    """(features table, combinations table, tree path, periods)."""
    y = score(ev, f)
    a = ((ev["date"] <= A_END) & y.notna()).to_numpy()
    b = ((ev["date"] >= B_START) & (ev["date"] <= B_END) & y.notna()).to_numpy()
    rows = {}
    big, loss = ev["r20"] >= BIG, ev["r20"] <= LOSS
    for c in FEATURES:
        ta, tb = test_feature(f[c][a], y[a], ev["date"][a]), test_feature(f[c][b], y[b], ev["date"][b])
        row = {"what": FEATURES[c], "n_a": ta["n"], "ic_a": ta["ic"], "p_a": ta["p"], "n_b": tb["n"], "ic_b": tb["ic"], "p_b": tb["p"]}
        if np.isfinite(ta["ic"]):
            has = best_third(f[c], f[c][a], ta["ic"], c in FLAGS)
            for name, p in (("a", a), ("b", b)):
                gained, lost = (ev["r20"] >= GAIN)[p], loss[p]
                row[f"gained_{name}"], row[f"lost_{name}"] = has[p][gained].astype(float).mean(), has[p][lost].astype(float).mean()
                g = group(ev, p & (has == True).to_numpy())                                                  # noqa: E712
                row[f"up_{name}"], row[f"net_{name}"], row[f"ex_{name}"], row[f"lo_{name}"], row[f"hi_{name}"] = g["up"], g["net"], g["ex"], g["lo"], g["hi"]
            if c not in FLAGS:
                w = f[c][a & big.to_numpy()].dropna()
                side = ">=" if ta["ic"] > 0 else "<="
                cut = w.quantile(0.10 if ta["ic"] > 0 else 0.90)
                meet = lambda p: ((f[c][p].dropna() >= cut) if side == ">=" else (f[c][p].dropna() <= cut)).mean()
                row.update(side=side, cut90=cut, winners90_a=meet(a & big.to_numpy()), losers90_a=meet(a & loss.to_numpy()),
                           winners90_b=meet(b & big.to_numpy()), losers90_b=meet(b & loss.to_numpy()))
        rows[c] = row
    t = pd.DataFrame(rows).T
    t["bh_a"] = benjamini_hochberg(t["p_a"].astype(float))
    t["same_sign"] = np.sign(t["ic_a"].astype(float)) == np.sign(t["ic_b"].astype(float))
    t["counts"] = t["bh_a"] & t["same_sign"] & (t["p_b"].astype(float) < 0.10)          # two-sided 0.10 = one-sided 0.05
    xa, xall = ranked(f[a], f[a]), ranked(f, f[a])
    w = ridge(xa, y[a].to_numpy())
    s = (xall - xa.mean(axis=0)) @ w
    cut = np.quantile(s[a], 0.8)
    ya = y[a].rank(pct=True).to_numpy()
    path = max(leaves(tree(xa, ya)))[2]
    fits = {"all breakouts": np.ones(len(ev), bool), "ridge best fifth": s >= cut, "tree best leaf": follow(xall, path)}
    comb = pd.DataFrame({(k, p): group(ev, v & mask_) for k, v in fits.items() for p, mask_ in (("A", a), ("B", b))}).T
    ic = test_feature(pd.Series(s, index=ev.index)[b], y[b], ev["date"][b])
    return t, comb, [(list(FEATURES)[j], op, round(c, 2)) for j, op, c in path], {"ridge_ic_b": ic, "n_a": int(a.sum()), "n_b": int(b.sum()),
            "y_ok": float(y.notna().mean()), "weights": pd.Series(w, FEATURES.keys())}


def trend_panels(m: bt.Market) -> Dict[str, pd.DataFrame]:
    """The three counting features for every name and day, for the control."""
    last, a200 = m.last, m.last.rolling(200).mean()
    return {"slope200": a200 / a200.shift(21) - 1, "ma_gap": m.last.rolling(50).mean() / a200 - 1,
            "rs252": (last / last.shift(252) - 1).where(m.eligible).rank(axis=1, pct=True)}


def control(ev: pd.DataFrame, m: bt.Market, mask: pd.DataFrame, member: pd.DataFrame) -> np.ndarray:
    """For each event, the same-day mean month return of eligible names in `member` without a recent detection."""
    f = m.forward(20).where(m.last.shift(-21).notna())
    f.iloc[max(0, len(f) - 21):] = np.nan
    recent = mask.rolling(20, min_periods=1).max().astype(bool)
    return f.where(m.eligible & member & ~recent).mean(axis=1).reindex(ev["date"]).to_numpy()


def rules_table(ev: pd.DataFrame, f: pd.DataFrame, m: bt.Market, mask: pd.DataFrame, names: List[str], periods: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Each rule (best third by A, then all of them) in each period, beside its control."""
    a, y = (ev["date"] <= A_END).to_numpy(), score(ev, f)
    panels, has, member = trend_panels(m), {}, {}
    for c in names:
        sign = test_feature(f[c][a], y[a], ev["date"][a])["ic"]
        cut = f[c][a].quantile(2 / 3 if sign > 0 else 1 / 3)
        has[c] = (best_third(f[c], f[c][a], sign, False) == True).to_numpy()                             # noqa: E712
        member[c] = panels[c] >= cut if sign > 0 else panels[c] <= cut
    has["all of them"], member["all of them"] = np.logical_and.reduce(list(has.values())), np.logical_and.reduce([v.to_numpy() for v in member.values()])
    member["all of them"] = pd.DataFrame(member["all of them"], index=m.close.index, columns=m.close.columns)
    has["every breakout"], member["every breakout"] = np.ones(len(ev), bool), pd.DataFrame(True, index=m.close.index, columns=m.close.columns)
    rows = {}
    for c in has:
        ctl = control(ev, m, mask, member[c])
        for p, ok in periods.items():
            pick = ok & has[c] & ev["r20"].notna().to_numpy()
            g = group(ev, pick)
            r, d = ev["r20"][pick], ev["r20"][pick] - ctl[pick]
            lo, hi = bt.block_interval(d, ev["date"][pick])
            lo_n, hi_n = bt.block_interval(r - bt.COST, ev["date"][pick])
            rows[(c, p)] = {**g, "median": (r - bt.COST).median(), "net_lo": lo_n, "net_hi": hi_n, "xiu": ev["x20"][pick].mean(),
                            "no_top10": (r.sort_values().iloc[:-10] - bt.COST).mean(), "control": ctl[pick].mean() - bt.COST,
                            "vs_control": d.mean(), "c_lo": lo, "c_hi": hi}
    return pd.DataFrame(rows).T


def main(argv: Optional[List[str]] = None) -> int:
    import sys
    from research.breakouts import prices
    part = (argv or sys.argv[1:] or ["dev"])[0]
    m = bt.Market(prices.load())
    ev, mask = bt.run(m, bt.DEFAULT, None, B_END if part == "dev" else None)
    f = build(ev, mask, m, load_sectors(), load_earnings())
    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 46)
    if part == "holdout":
        dev = (ev["date"] <= B_END).to_numpy()
        t = analyse(ev[dev].reset_index(drop=True), f[dev].reset_index(drop=True))[0]
        names = list(t.index[t["counts"].astype(bool)])
        c = ((ev["date"] >= bt.HOLD_START) & ev["r20"].notna()).to_numpy()
        y = score(ev, f)
        tests = pd.DataFrame({k: test_feature(f[k][c], y[c], ev["date"][c]) for k in names}).T
        periods = {"A": (ev["date"] <= A_END).to_numpy(), "B": ((ev["date"] >= B_START) & (ev["date"] <= B_END)).to_numpy(), "C": c}
        table = rules_table(ev, f, m, mask, names, periods)
        tests.to_csv(bt.OUT / "features_holdout.csv")
        table.to_csv(bt.OUT / "features_rules.csv")
        print(tests.round(4).to_string())
        print(table.round(4).to_string())
        return 0
    t, comb, path, info = analyse(ev, f)
    t.to_csv(bt.OUT / "features_result.csv")
    comb.to_csv(bt.OUT / "features_combined.csv")
    num = t.drop(columns=["what", "side"]).apply(pd.to_numeric)
    num.insert(0, "what", t["what"])
    print(num.sort_values("p_a")[["what", "n_a", "ic_a", "p_a", "bh_a", "ic_b", "p_b", "counts", "gained_b", "lost_b", "net_b", "ex_b", "lo_b", "hi_b"]].round(3).to_string())
    print(num.sort_values("losers90_a")[["what", "cut90", "winners90_a", "losers90_a", "winners90_b", "losers90_b"]].round(3).head(25).to_string())
    print(comb.astype(float).round(4).to_string())
    print("tree:", path, "| ridge IC in B", info["ridge_ic_b"], {k: v for k, v in info.items() if k.startswith("n_") or k == "y_ok"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
