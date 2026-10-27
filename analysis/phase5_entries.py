"""
analysis/phase5_entries.py -- Phase 5 of the idle-cash plan: rank entry slices
by excess return over XIU after costs, from the Phase 1 trade table
(01_trade_vs_xiu/trades_with_excess.csv; closed trades plus positions open at
fold end, marked to market -- the Phase 1 primary "all" sample).

Slices: setup type (pattern), stop width (stop_bucket), market regime, sector.
The plan's regime split (XIU vs its 200-day at entry) can't vary -- the entry
gate admits only "above" -- so XIU vs its 50-day SMA (prior close) is added as
the nearest testable regime proxy. Sector comes from ref/backtest_sectors.json
(analysis/fetch_backtest_sectors.py) for the backtest, ref/live_sectors.json
for live.

Per slice: trades, mean excess % (realistic and adverse costs), bootstrap 95%
CI (10,000 resamples of trades), a Bonferroni-adjusted CI for the number of
slices examined, total excess $, beat-XIU rate. A slice is flagged only if
its plain 95% CI is above zero with at least MIN_N trades.

Usage:
    python analysis/phase5_entries.py   -> reports/idle_cash/05_entries/slices.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IC = ROOT / "reports" / "idle_cash"
OUT = IC / "05_entries"
N_BOOT = 10_000
SEED = 20261027
MIN_N = 10
DIMS = ("pattern", "stop_bucket", "regime_50d", "sector")


def _xiu_50d() -> pd.Series:
    ref = pd.read_csv(IC / "ref" / "xiu_cash_daily.csv", parse_dates=["date"])
    close = ref[ref.ticker == "XIU.TO"].set_index("date")["Close"].sort_index()
    return (close >= close.rolling(50).mean()).map({True: "XIU above 50d", False: "XIU below 50d"})


def _boot(x: np.ndarray, rng: np.random.Generator, alpha: float) -> tuple[float, float]:
    means = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(IC / "01_trade_vs_xiu" / "trades_with_excess.csv", parse_dates=["entry_date"])
    above = _xiu_50d()
    # Regime as of the prior completed close (entry is at the open).
    t["regime_50d"] = [above.loc[:d - pd.Timedelta(days=1)].iloc[-1] for d in t.entry_date]
    bt_sectors = json.loads((IC / "ref" / "backtest_sectors.json").read_text())
    live_sectors = json.loads((IC / "ref" / "live_sectors.json").read_text())
    t["sector"] = [(bt_sectors if s == "backtest" else live_sectors).get(tk, "Unknown")
                   for s, tk in zip(t.source, t.ticker)]

    rng = np.random.default_rng(SEED)
    rows = []
    for source, g in t.groupby("source"):
        for dim in DIMS:
            for value, s in g.groupby(dim):
                x = s.excess_realistic_pct.to_numpy()
                row = dict(source=source, dimension=dim, value=value, trades=len(s),
                           mean_pct=x.mean(), median_pct=float(np.median(x)),
                           adverse_mean_pct=s.excess_adverse_pct.mean(),
                           excess_total_usd=s.excess_realistic.sum(), beat_xiu_rate=(x > 0).mean())
                if len(s) >= 2:
                    row["ci_lo"], row["ci_hi"] = _boot(x, rng, 0.05)
                rows.append(row)
    df = pd.DataFrame(rows)
    # Bonferroni: one family per source, counting every slice with >= MIN_N trades.
    for source, g in df.groupby("source"):
        k = int((g.trades >= MIN_N).sum())
        alpha = 0.05 / k
        for i in g.index[g.trades >= MIN_N]:
            s = t[(t.source == source) & (t[df.at[i, "dimension"]] == df.at[i, "value"])]
            df.loc[i, ["bonf_lo", "bonf_hi"]] = _boot(s.excess_realistic_pct.to_numpy(), rng, alpha)
            df.loc[i, "family_size"] = k
    df["flag_ci_above_zero"] = (df.trades >= MIN_N) & (df.ci_lo > 0)
    df["flag_bonferroni"] = (df.trades >= MIN_N) & (df.bonf_lo > 0)
    df = df.sort_values(["source", "dimension", "mean_pct"], ascending=[True, True, False])
    df.round(4).to_csv(OUT / "slices.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
