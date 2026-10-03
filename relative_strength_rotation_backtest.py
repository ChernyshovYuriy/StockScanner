"""
relative_strength_rotation_backtest.py
=======================================
Research question (2026-10, prompted by MATR.TO's Dec-2025..Oct-2026 run,
which the core/momentum pullback detectors flagged AT_PIVOT but never
bought): does a plain monthly relative-strength rotation -- hold the N
liquid TSX names with the best past return, re-picked every month --
beat holding the whole universe and XIU.TO, out of sample?

Mechanics (no lookahead): the ranking is computed at month-end close a;
the portfolio is bought at the NEXT trading day's close and held to the
next month's equivalent day. Equal weight, ROUND_TRIP_COST charged on
the share of the book that changes. Eligible: price >= MIN_PRICE and
100-day mean dollar volume >= MIN_DOLLAR_VOL at a. A single holding's
monthly return is capped at +500% to blunt bad prints.

Grid: lookback x skip (exclude the most recent month, the classic 12-1
construction) x N x trend filter (close > 200-day SMA). Two answers:
  1. Grid robustness -- how much of the grid beats the equal-weight
     universe. A real effect should not hinge on one cell.
  2. Walk-forward -- each month, pick the cell with the best mean monthly
     excess vs equal-weight over ALL prior months (min MIN_TRAIN_MONTHS),
     hold it for that one month. Only those out-of-sample months count.

SURVIVORSHIP: market_cache.db holds today's tickers only; names that
went to zero or delisted are missing. That inflates absolute returns for
the strategy AND the equal-weight benchmark, so compare against
equal-weight (same biased universe), not XIU.TO. Even that comparison is
biased in an unknown direction (momentum names that later collapsed and
delisted are missing too).

Usage:
    python relative_strength_rotation_backtest.py
"""
from __future__ import annotations

import itertools

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from config import OUT_PATH
from market_data_cache import CACHE_DB_PATH

LOOKBACKS = [63, 126, 189, 252]
SKIPS = [0, 21]
TOP_N = [10, 20, 30]
TREND = [False, True]
MIN_PRICE = 1.0
MIN_DOLLAR_VOL = 250_000
ROUND_TRIP_COST = 0.01
MIN_TRAIN_MONTHS = 12
BENCH = "XIU.TO"

GRID = list(itertools.product(LOOKBACKS, SKIPS, TOP_N, TREND))
GRID_CSV = OUT_PATH / "rs_rotation_grid.csv"
WF_CSV = OUT_PATH / "rs_rotation_walk_forward.csv"


def _load():
    con = duckdb.connect(str(CACHE_DB_PATH), read_only=True)
    d = con.execute("SELECT ticker, date, close, volume FROM ohlcv_daily").df()
    con.close()
    d["date"] = pd.to_datetime(d["date"])
    px = d.pivot(index="date", columns="ticker", values="close").sort_index()
    vol = d.pivot(index="date", columns="ticker", values="volume").sort_index()
    return px, vol


def main() -> None:
    px, vol = _load()
    dv = (px * vol).rolling(100, min_periods=80).mean()
    ma200 = px.rolling(200, min_periods=180).mean()
    idx = px.index
    month_ends = [idx[idx <= m][-1] for m in px.resample("ME").last().index]
    month_ends = [m for m in month_ends if idx.get_loc(m) >= max(LOOKBACKS) + 1]
    month_ends = [m for m in month_ends if idx.get_loc(m) + 1 < len(idx)]
    print(f"{px.shape[1]} tickers, {idx[0].date()}..{idx[-1].date()}, {len(month_ends) - 1} rebalance months")

    rows, picks_prev = [], {g: set() for g in GRID}
    for a, b in zip(month_ends[:-1], month_ends[1:]):
        ia, ib = idx.get_loc(a), idx.get_loc(b)
        buy, sell = idx[ia + 1], idx[ib + 1] if ib + 1 < len(idx) else idx[ib]
        base_ok = (dv.loc[a] >= MIN_DOLLAR_VOL) & (px.loc[a] >= MIN_PRICE) & px.loc[buy].notna() & px.loc[sell].notna()
        fwd = (px.loc[sell] / px.loc[buy] - 1).clip(upper=5.0)
        row = {"month": buy, "ew": fwd[base_ok].mean(), "n_universe": int(base_ok.sum()),
               "xiu": fwd.get(BENCH, np.nan)}
        for g in GRID:
            lb, sk, n, trend = g
            mom = px.iloc[ia - sk] / px.iloc[ia - lb] - 1
            ok = base_ok & mom.notna()
            if trend:
                ok &= px.loc[a] > ma200.loc[a]
            pick = mom[ok].nlargest(n).index
            if len(pick) == 0:
                row[g] = 0.0
                continue
            turnover = len(set(pick) - picks_prev[g]) / n
            picks_prev[g] = set(pick)
            row[g] = fwd[pick].mean() - turnover * ROUND_TRIP_COST
        rows.append(row)
    m = pd.DataFrame(rows).set_index("month")
    years = len(m) / 12

    def cagr(r):
        return (1 + r).prod() ** (1 / years) - 1

    def maxdd(r):
        eq = (1 + r).cumprod()
        return (eq / eq.cummax() - 1).min()

    grid_out = []
    for g in GRID:
        ex = m[g] - m["ew"]
        t, p = stats.ttest_1samp(ex, 0.0)
        grid_out.append({"lookback": g[0], "skip": g[1], "top_n": g[2], "trend": g[3],
                         "cagr%": round(cagr(m[g]) * 100, 1), "maxdd%": round(maxdd(m[g]) * 100, 0),
                         "excess_vs_ew%/mo": round(ex.mean() * 100, 2), "t": round(t, 2), "p": round(p, 3)})
    grid = pd.DataFrame(grid_out).sort_values("excess_vs_ew%/mo", ascending=False)
    print(f"\nEqual-weight universe CAGR {cagr(m['ew']) * 100:.1f}% (maxDD {maxdd(m['ew']) * 100:.0f}%), "
          f"XIU.TO {cagr(m['xiu'].fillna(0)) * 100:.1f}% (maxDD {maxdd(m['xiu'].fillna(0)) * 100:.0f}%)")
    print(f"Grid: {(grid['excess_vs_ew%/mo'] > 0).sum()}/{len(grid)} cells beat equal-weight; "
          f"{(grid['p'] < 0.05).sum()} significant at p<0.05 (expect ~{len(grid) * 0.025:.0f} by chance on the upside)")
    print(tabulate(grid.head(10), headers="keys", tablefmt="simple", showindex=False))
    print("...worst 5:")
    print(tabulate(grid.tail(5), headers="keys", tablefmt="simple", showindex=False))

    wf = []
    for k in range(MIN_TRAIN_MONTHS, len(m)):
        train = m.iloc[:k]
        best = max(GRID, key=lambda g: (train[g] - train["ew"]).mean())
        mo = m.index[k]
        wf.append({"month": mo, "chosen": best, "strat": m[best].iloc[k], "ew": m.loc[mo, "ew"], "xiu": m.loc[mo, "xiu"]})
    w = pd.DataFrame(wf).set_index("month")
    yrs = len(w) / 12
    ex = w["strat"] - w["ew"]
    t, p = stats.ttest_1samp(ex, 0.0)

    def c(r):
        return ((1 + r.fillna(0)).prod() ** (1 / yrs) - 1) * 100

    print(f"\n=== WALK-FORWARD, out-of-sample {w.index[0].date()}..{w.index[-1].date()} ({len(w)} months) ===")
    print(f"strategy CAGR {c(w['strat']):.1f}%  maxDD {maxdd(w['strat']) * 100:.0f}%  |  equal-weight {c(w['ew']):.1f}%  "
          f"maxDD {maxdd(w['ew']) * 100:.0f}%  |  XIU.TO {c(w['xiu']):.1f}%  maxDD {maxdd(w['xiu'].fillna(0)) * 100:.0f}%")
    print(f"monthly excess vs equal-weight {ex.mean() * 100:+.2f}%, beat it in {(ex > 0).mean() * 100:.0f}% of months, t={t:.2f}, p={p:.3f}")
    print("chosen cells:", w["chosen"].astype(str).value_counts().head(5).to_dict())
    by = w.groupby(w.index.year)[["strat", "ew", "xiu"]].apply(lambda g: (1 + g.fillna(0)).prod() - 1) * 100
    print(by.round(1).to_string())

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_CSV, index=False)
    w.assign(chosen=w["chosen"].astype(str)).to_csv(WF_CSV)
    print(f"\nWrote {GRID_CSV} and {WF_CSV}")


if __name__ == "__main__":
    main()
