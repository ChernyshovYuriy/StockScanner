"""
Stock rankings on the S&P/TSX Composite as it was on each date (PLAN.md, Phase
3): S1 momentum, S2 low volatility, S3 residual momentum.

A member's returns come from its Yahoo symbol when one fits the index files'
own prices (members.py), otherwise from those prices, so companies that died or
were bought are in the test while they were members.

    python -m research.swing_rank.stocks
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from research.swing_rank import engine as en
from research.swing_rank import members as mb
from research.swing_rank.engine import CASH

ROOT = Path(__file__).resolve().parents[2]
STOCK_DB = ROOT / "cache" / "swing_rank_stocks.db"
OUT = ROOT / "reports" / "swing_rank"
XIU = "XIU.TO"
COST = 0.0020
BAD_SECTORS = {"-", "", "1", "Other", "Funds"}


@dataclass(frozen=True)
class StockConfig:
    formation: int = 12              # months, the last one skipped
    vol_days: int = 252
    regress_months: int = 36
    top_n: int = en.TOP_N
    keep: int = en.KEEP
    max_per_sector: int = 2
    min_scored: int = 150            # first month with this many scored members starts the test


DEFAULT = StockConfig()

# How a name that leaves the files without a Yahoo price is closed in the harsh test.
PINNED = 0.02                        # last month's move this small: a takeover at a fixed price
COLLAPSED = -0.70                    # fell this much over its last 12 file months: bankrupt


class StockData:
    """Everything keyed by company id, on month-end signal sessions."""

    def __init__(self, files: Dict[pd.Timestamp, pd.DataFrame], yahoo: Dict[str, pd.DataFrame]):
        self.table = mb.long_table(files)
        self.table["id"] = mb.company_ids(self.table)
        self.table = mb.collapse(self.table)
        self.table["id"] = mb.link_renames(self.table)
        self.calendar = yahoo[XIU].index
        self.panel = en.build_panel(yahoo, self.calendar)
        self.file_dates = sorted(files)
        self.from_files = mb.file_returns(self.table)
        at_files = mb.month_end_returns(self.panel.close, self.file_dates)
        self.symbol = mb.match_symbols(self.table, self.from_files, at_files.drop(columns=[XIU]))
        self.sessions = [s for s in en.month_end_sessions(self.calendar) if self.calendar[s].to_period("M") >= self.file_dates[0].to_period("M")]
        # By month: a file can be dated on a holiday after the month's last session.
        self._file_of = {s: max(d for d in self.file_dates if d.to_period("M") <= self.calendar[s].to_period("M")) for s in self.sessions}
        self.members = {s: sorted(self.table.loc[self.table["date"] == d, "id"].unique()) for s, d in self._file_of.items()}
        self._sector = self._sectors()
        self.dividend = self._dividends(at_files)
        self._build_returns()

    def _sectors(self) -> Dict[int, pd.Series]:
        """A member's sector at each file; where the file has none, its nearest known one."""
        out = {}
        for cid, g in self.table.drop_duplicates(["id", "date"]).groupby("id"):
            s = g.set_index("date")["sector"].sort_index()
            s = s.where(~s.isin(BAD_SECTORS))
            out[cid] = s.ffill().bfill().fillna("Unknown")
        return out

    def _dividends(self, at_files: pd.DataFrame) -> Dict[str, float]:
        """Average monthly dividend return by sector: how much Yahoo's total returns
        exceed the files' price returns on matched names. Credited to file-priced
        members, whose prices leave dividends out. "" is the overall average."""
        rows = []
        for c, sym in self.symbol.items():
            if c in self.from_files:
                gap = (at_files[sym] - self.from_files[c]).dropna()
                gap = gap[gap.abs() < 0.2]
                rows.append(pd.DataFrame({"sector": self._sector[c].reindex(gap.index).to_numpy(), "gap": gap.to_numpy()}))
        if not rows:
            return {"": 0.0}
        allr = pd.concat(rows)
        out = allr.groupby("sector")["gap"].mean().clip(lower=0.0).to_dict()
        out[""] = max(0.0, float(allr["gap"].mean()))
        return out

    def _dividend(self, c: int) -> float:
        return self.dividend.get(self._sector[c].iloc[-1], self.dividend[""])

    def sector_of(self, s: int) -> Dict[int, str]:
        d = self._file_of[s]
        return {c: self._sector[c][d] for c in self.members[s]}

    def _build_returns(self) -> None:
        ss, cal = self.sessions, self.calendar
        close = self.panel.close
        ids = sorted(self.table["id"].unique())
        month = pd.DataFrame(index=ss, columns=ids, dtype=float)     # close to close, month ending at the session
        for c in ids:
            if c in self.symbol:
                at = close[self.symbol[c]].iloc[ss].to_numpy(float)
                prior = np.concatenate([[np.nan], at[:-1]])
                month[c] = at / prior - 1.0
            elif c in self.from_files:
                fr = self.from_files[c]
                div = self._dividend(c)
                month[c] = [div + fr.get(self._file_of[s], np.nan) if cal[s].to_period("M") == self._file_of[s].to_period("M") else np.nan for s in ss]
        self.month = month
        self.xiu_month = pd.Series(close[XIU].iloc[ss].to_numpy(float), index=ss).pct_change()

        px = en.period_returns(self.panel, ss)
        held = pd.DataFrame(index=px.index, columns=ids, dtype=float)
        unknown = pd.DataFrame(False, index=px.index, columns=ids)
        for c in ids:
            if c in self.symbol:
                held[c] = px[self.symbol[c]]
            else:
                held[c] = month[c].shift(-1).reindex(px.index)
                self._bridge_gaps(c, held, unknown)
        self.period = held
        self.unknown = unknown
        self.xiu = px[XIU]
        self.years = np.array([cal[s + 1].year for s in px.index])

    def _bridge_gaps(self, c: int, held: pd.DataFrame, unknown: pd.DataFrame) -> None:
        """A file-priced member with no return: if it is in the next file after a
        publishing gap, the whole gap's return sits on the gap's first month and
        the rest are 0; otherwise its ending is unknown."""
        g = self.table[self.table["id"] == c].drop_duplicates("date").set_index("date")
        for s in held.index:
            if c not in self.members[s] or np.isfinite(held.loc[s, c]):
                continue
            here = self._file_of[s]
            later = [d for d in self.file_dates if d > here]
            if later and later[0] in g.index and (later[0] - here).days > 40:
                a, b = g.loc[here], g.loc[later[0]]
                first = self.calendar[s].to_period("M") == here.to_period("M")
                f = mb.split_factor(b["price"] / a["price"], b["shares"] / a["shares"])
                held.loc[s, c] = self._dividend(c) + (b["price"] * f / a["price"] - 1.0 if first else 0.0)
            else:
                unknown.loc[s, c] = True

    def returns(self, harsh: bool = False) -> pd.DataFrame:
        """Period returns; unknown endings are 0 (last price), or, if harsh, -100%
        after a collapse, 0 for a price pinned by an offer, -50% otherwise."""
        r = self.period.copy()
        if not harsh:
            return r
        for c in r.columns[self.unknown.any()]:
            for s in r.index[self.unknown[c]]:
                past = self.month.loc[:s, c].dropna()
                if len(past) and (1 + past.iloc[-12:]).prod() - 1 <= COLLAPSED:
                    r.loc[s, c] = -1.0
                elif len(past) and abs(past.iloc[-1]) <= PINNED:
                    r.loc[s, c] = 0.0
                else:
                    r.loc[s, c] = -0.5
        return r

    def coverage(self) -> pd.Series:
        """Share of each month's members with a known return for the period."""
        return pd.Series({s: float(np.mean([np.isfinite(self.period.loc[s, c]) for c in self.members[s]])) for s in self.period.index})


# ── Scores: each looks at month-ends up to the session only ──────────────────

class Scores:
    def __init__(self, data: StockData, cfg: StockConfig = DEFAULT):
        self.d, self.cfg = data, cfg
        daily = data.panel.close.pct_change(fill_method=None)
        self._vol = daily.rolling(cfg.vol_days, min_periods=int(cfg.vol_days * 0.75)).std() * np.sqrt(252)
        self._row = {s: k for k, s in enumerate(data.sessions)}

    def _window(self, s: int, months: int) -> pd.DataFrame:
        """The `months` monthly returns ending one month before s (the last month skipped)."""
        k = self._row[s]
        return self.d.month.iloc[max(0, k - months):k][self.d.members[s]] if k >= months else pd.DataFrame(columns=self.d.members[s], dtype=float)

    def momentum(self, s: int) -> pd.Series:
        w = self._window(s, self.cfg.formation - 1)
        ok = w.notna().sum() == self.cfg.formation - 1
        return ((1 + w).prod() - 1.0).where(ok)

    def low_vol(self, s: int) -> pd.Series:
        out = {}
        k = self._row[s]
        for c in self.d.members[s]:
            if c in self.d.symbol:
                out[c] = -self._vol[self.d.symbol[c]].iat[s]
            else:
                m = self.d.month[c].iloc[max(0, k - 11):k + 1].dropna()
                out[c] = -m.std() * np.sqrt(12) if len(m) >= 9 else np.nan
        return pd.Series(out, dtype=float).replace(0.0, np.nan)

    def residual(self, s: int) -> pd.Series:
        n, f = self.cfg.regress_months, self.cfg.formation - 1
        w = self._window(s, n)
        out = pd.Series(np.nan, index=self.d.members[s], dtype=float)
        if len(w) < n:
            return out
        market = self.d.xiu_month.loc[w.index].to_numpy(float)
        sector = pd.Series(self.d.sector_of(s))
        groups = {g: w[idx].mean(axis=1).to_numpy(float) for g, idx in sector.groupby(sector).groups.items()}
        for c in w.columns:
            y = w[c].to_numpy(float)
            if not np.isfinite(y).all():
                continue
            X = np.column_stack([np.ones(n), market, groups[sector[c]]])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            e = (y - X[:, 1:] @ beta[1:])[-f:]
            sd = e.std(ddof=1)
            if sd > 0:
                out[c] = e.sum() / sd
        return out


def make_strategies(sc: Scores, cfg: StockConfig = DEFAULT):
    def use(score):
        def weights(s: int, held: Set[int]) -> Dict[int, float]:
            names = en.select(score(s), held, cfg.top_n, cfg.keep, sc.d.sector_of(s), cfg.max_per_sector)
            return en.equal_weights(names, cfg.top_n)
        return weights
    return {"S1 momentum 12-1": use(sc.momentum), "S2 low volatility": use(sc.low_vol), "S3 residual momentum": use(sc.residual)}


RELEVANT = {"S1 momentum 12-1": ("formation", "top_n", "keep"), "S2 low volatility": ("vol_days", "top_n", "keep"),
            "S3 residual momentum": ("formation", "regress_months", "top_n", "keep")}


def sensitivity_configs(cfg: StockConfig = DEFAULT) -> Dict[str, StockConfig]:
    out = {}
    for field in ("formation", "vol_days", "regress_months", "top_n", "keep"):
        for label, k in (("-20%", 0.8), ("+20%", 1.2)):
            out[f"{field} {label}"] = replace(cfg, **{field: int(round(getattr(cfg, field) * k))})
    return out


class StockTest:
    def __init__(self, data: StockData, cfg: StockConfig = DEFAULT):
        self.d = data
        sc = Scores(data, cfg)
        ss = [s for s in data.period.index
              if sc.momentum(s).notna().sum() >= cfg.min_scored and sc.residual(s).notna().sum() >= cfg.min_scored * 2 // 3]
        self.start = ss[0]
        self.index = [s for s in data.period.index if s >= self.start]
        self.bench = data.xiu.loc[self.index]
        self.years = data.years[[list(data.period.index).index(s) for s in self.index]]

    def run(self, cfg: StockConfig = DEFAULT, cost: float = COST, harsh: bool = False) -> Dict[str, pd.DataFrame]:
        sc = Scores(self.d, cfg)
        r = self.d.returns(harsh).loc[self.index]
        r[CASH] = 0.0
        return {k: en.run(f, r, cost) for k, f in make_strategies(sc, cfg).items()}

    def universe(self, harsh: bool = False) -> pd.Series:
        """Equal-weight return of all members; unknown endings as in returns()."""
        r = self.d.returns(harsh)
        return pd.Series({s: float(r.loc[s, self.d.members[s]].fillna(0.0).mean()) for s in self.index})


def flags(e: Dict) -> Dict[str, bool]:
    c5 = bool(e["sens_share"] >= 0.80 and e["stress_excess"] > 0)
    c6 = bool(e["harsh_excess"] > 0)
    beats_universe = bool(e["universe_excess"] > 0 and e["universe_lo"] > 0)
    return {"c5": c5, "c6": c6, "beats_universe": beats_universe,
            "passed": bool(e["c1"] and e["c2"] and e["c3"] and e["c4"] and c5 and c6 and beats_universe)}


def report(test: StockTest, n_random: int = en.N_RANDOM) -> pd.DataFrame:
    base, stress, harsh = test.run(), test.run(cost=2 * COST), test.run(harsh=True)
    sens = {k: test.run(c) for k, c in sensitivity_configs().items()}
    r = test.d.returns().loc[test.index]
    r[CASH] = 0.0
    sc = Scores(test.d)
    eligible = {s: sc.momentum(s).dropna().index.tolist() for s in test.index}
    sectors = {s: test.d.sector_of(s) for s in test.index}
    rand = en.random_portfolios(eligible, r, sector_of=sectors.get, n_random=n_random)
    universe = test.universe()
    rows = []
    for name, res in base.items():
        e = en.evaluate(res["net"], test.bench, test.years)
        runs = [en.evaluate(sens[k][name]["net"], test.bench, test.years)["excess"] for k in sens if k.split()[0] in RELEVANT[name]]
        e["random_95"], e["random_median"] = float(np.quantile(rand, 0.95)), float(np.median(rand))
        e["c2"] = bool(en.cagr(res["gross"]) > e["random_95"])
        e["sens_share"] = float(np.mean(np.array(runs) > 0))
        e["stress_excess"] = en.evaluate(stress[name]["net"], test.bench, test.years)["excess"]
        e["harsh_excess"] = en.evaluate(harsh[name]["net"], test.bench, test.years)["excess"]
        u = (res["net"] - universe).to_numpy()
        e["universe_cagr"], e["universe_excess"] = en.cagr(universe), float(u.mean() * 12)
        e["universe_lo"], e["universe_hi"] = en.bootstrap_mean(u)
        e["turnover_per_year"] = float(res["turnover"].mean() * 12)
        half = len(res) // 2
        ex = res["net"] - test.bench
        e["excess_first_half"], e["excess_second_half"] = float(ex.iloc[:half].mean() * 12), float(ex.iloc[half:].mean() * 12)
        e.update(flags(e))
        e["variants_run"] = 3 + len(runs)
        rows.append({"strategy": name, **e})
    return pd.DataFrame(rows)


def load() -> StockData:
    from research.swing_rank import holdings
    from research.vcp import prices
    conn = prices.connect(str(STOCK_DB))
    try:
        yahoo = prices.load_all(conn)
    finally:
        conn.close()
    return StockData(holdings.load(), yahoo)


def main() -> int:
    data = load()
    test = StockTest(data)
    cov = data.coverage().loc[test.index]
    print("companies", data.table["id"].nunique(), "matched to Yahoo", len(data.symbol), "| start",
          data.calendar[test.start].date(), "months", len(test.index), "| coverage min/mean", round(cov.min(), 3), round(cov.mean(), 3),
          "| unknown endings", int(data.unknown.loc[test.index].sum().sum()))
    OUT.mkdir(parents=True, exist_ok=True)
    table = report(test)
    table.to_csv(OUT / "phase3_stocks.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.set_index("strategy").T.to_string())
    (OUT / "phase3_passed.json").write_text(json.dumps(table.loc[table["passed"], "strategy"].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
