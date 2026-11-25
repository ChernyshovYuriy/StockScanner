"""
The book's ETF strategies on iShares Canada funds (PLAN.md, Phase 2): E1, E1b,
E2, E3 and E4. Signals at a month's last close; the engine trades at the next
open. No network: prices from cache/swing_rank_prices.db, fund names from
cache/swing_rank_funds.csv (BlackRock's performance report of 2026-10-31).

    python -m research.swing_rank.etf
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from research.swing_rank import engine as en
from research.swing_rank.engine import CASH, Panel

ROOT = Path(__file__).resolve().parents[2]
PRICES = ROOT / "cache" / "swing_rank_prices.db"
FUNDS = ROOT / "cache" / "swing_rank_funds.csv"
OUT = ROOT / "reports" / "swing_rank"
XIU, BONDS, MONEY = "XIU.TO", "XBB.TO", "CMR.TO"
COST = 0.0010

# Not an equity or commodity fund: bonds, cash and bond-like income funds.
_BOND_RE = re.compile(r"bond|treasury|money market|floating|income|tips|preferred|inflation|government|"
                      r"corporate|mortgage|portfolio|convertible", re.I)


@dataclass(frozen=True)
class EtfConfig:
    formation: int = 252             # 12 months
    ma_days: int = 200
    history: int = 273               # sessions a fund needs before it is eligible
    min_dollar_vol: float = 100_000.0
    top_n: int = en.TOP_N
    keep: int = en.KEEP
    vol_days: int = 252              # E3's volatility
    max_weight: float = 0.20         # E3
    target_vol_days: int = 20        # E4
    vol_threshold: float = 0.10      # E4: trade only when the weight moves by more than this share
    min_universe: int = 20           # first month with this many eligible funds starts the test


DEFAULT = EtfConfig()


def base_name(name: str) -> str:
    """Name shared by a fund's hedged, unhedged and USD versions."""
    return re.sub(r"\s*(\(CAD-Hedged\)|- USD Units)\s*", "", name).strip()


def is_bond(name: str) -> bool:
    return bool(_BOND_RE.search(name))


def is_hedged(name: str) -> bool:
    return "CAD-Hedged" in name


class EtfUniverse:
    """Which funds could have been chosen at a session, from data up to it."""

    def __init__(self, panel: Panel, names: Dict[str, str], cfg: EtfConfig = DEFAULT):
        self.panel, self.cfg = panel, cfg
        self.names = {t: n for t, n in names.items() if t in panel.close.columns and t != MONEY}
        self._count = panel.close.notna().cumsum()

    def eligible(self, s: int, bonds: bool = False) -> List[str]:
        c, p = self.cfg, self.panel
        ok = [t for t, n in self.names.items()
              if (bonds or not is_bond(n)) and np.isfinite(p.close[t].iat[s])
              and self._count[t].iat[s] >= c.history and p.dollar_vol[t].iat[s] >= c.min_dollar_vol]
        unhedged = {base_name(self.names[t]) for t in ok if not is_hedged(self.names[t])}
        return sorted(t for t in ok if not is_hedged(self.names[t]) or base_name(self.names[t]) not in unhedged)


class Signals:
    def __init__(self, panel: Panel, cfg: EtfConfig = DEFAULT):
        c = panel.close
        self.momentum = c / c.shift(cfg.formation) - 1.0
        self.above = c > c.rolling(cfg.ma_days, min_periods=cfg.ma_days).mean()
        daily = c.pct_change(fill_method=None)
        self.vol = daily.rolling(cfg.vol_days, min_periods=cfg.vol_days // 2).std() * np.sqrt(252)
        self.short_vol = daily.rolling(cfg.target_vol_days).std() * np.sqrt(252)
        self.target_vol = self.short_vol.expanding(min_periods=252).mean()


def _top(scores: pd.Series, held: Set[str], cfg: EtfConfig) -> Dict[str, float]:
    return en.equal_weights(en.select(scores, held, cfg.top_n, cfg.keep), cfg.top_n)


def make_strategies(u: EtfUniverse, sig: Signals, cfg: EtfConfig = DEFAULT):
    """{name: weights_for(session, held)}"""
    def e1(s, held):
        return _top(sig.momentum.iloc[s][u.eligible(s)], held, cfg)

    def e1b(s, held):
        m = sig.momentum.iloc[s][u.eligible(s)]
        top = en.select(m, held, cfg.top_n, cfg.keep)
        return en.equal_weights([t for t in top if sig.above[t].iat[s]], cfg.top_n)

    def e2(s, held):
        return e1(s, held) if sig.above[XIU].iat[s] else {BONDS: 1.0}

    def e3(s, held):
        names = [t for t in u.eligible(s, bonds=True) if sig.momentum[t].iat[s] > 0 and sig.above[t].iat[s]]
        score = (sig.momentum.iloc[s][names] / sig.vol.iloc[s][names]).replace([np.inf, -np.inf], np.nan).dropna()
        chosen = en.select(score, held, cfg.top_n, cfg.keep)
        if not chosen:
            return {CASH: 1.0}
        w = score[chosen] / score[chosen].sum()
        w = w.clip(upper=cfg.max_weight)
        out = {t: float(x) for t, x in w.items()}
        rest = 1.0 - sum(out.values())
        if rest > 1e-12:
            out[CASH] = rest
        return out

    last: Dict[str, float] = {}

    def e4(s, held):
        target, now = sig.target_vol[XIU].iat[s], sig.short_vol[XIU].iat[s]
        w = float(min(1.0, target / now))
        if not held:
            last.clear()
        if "w" in last and abs(w / last["w"] - 1.0) <= cfg.vol_threshold:
            w = last["w"]
        last["w"] = w
        return {XIU: w, CASH: 1.0 - w} if w < 1.0 else {XIU: 1.0}

    return {"E1 sector rotation": e1, "E1b rotation + MA filter": e1b, "E2 dual momentum": e2,
            "E3 multi-asset trend": e3, "E4 volatility targeting": e4}


# ── The test ─────────────────────────────────────────────────────────────────

class EtfTest:
    """Panel, periods and benchmark shared by every variant."""

    def __init__(self, data: Dict[str, pd.DataFrame], names: Dict[str, str], cfg: EtfConfig = DEFAULT):
        self.calendar = data[XIU].index
        self.names = names
        self.panel = en.build_panel({t: d for t, d in data.items() if t in names or t in (XIU, BONDS, MONEY)}, self.calendar)
        sessions = en.month_end_sessions(self.calendar)
        u = EtfUniverse(self.panel, names, cfg)
        target = Signals(self.panel, cfg).target_vol[XIU]
        start = next(k for k, s in enumerate(sessions)
                     if len(u.eligible(s)) >= cfg.min_universe and np.isfinite(target.iat[s]))
        self.sessions = sessions[start:]
        r = en.period_returns(self.panel, self.sessions)
        r[CASH] = r[MONEY].fillna(0.0)
        self.returns = r
        self.bench = r[XIU]
        self.years = np.array([self.calendar[s + 1].year for s in r.index])

    def run(self, cfg: EtfConfig = DEFAULT, cost: float = COST) -> Dict[str, pd.DataFrame]:
        u, sig = EtfUniverse(self.panel, self.names, cfg), Signals(self.panel, cfg)
        return {k: en.run(f, self.returns, cost) for k, f in make_strategies(u, sig, cfg).items()}

    def eligible(self, cfg: EtfConfig = DEFAULT, bonds: bool = False) -> Dict[int, List[str]]:
        u = EtfUniverse(self.panel, self.names, cfg)
        return {s: u.eligible(s, bonds) for s in self.returns.index}


def sensitivity_configs(cfg: EtfConfig = DEFAULT) -> Dict[str, EtfConfig]:
    out = {}
    for field in ("formation", "ma_days", "top_n", "keep", "vol_days", "max_weight", "target_vol_days",
                  "vol_threshold", "min_dollar_vol"):
        v = getattr(cfg, field)
        for label, k in (("-20%", 0.8), ("+20%", 1.2)):
            out[f"{field} {label}"] = replace(cfg, **{field: int(round(v * k)) if isinstance(v, int) else v * k})
    return out


# The settings each strategy actually depends on.
RELEVANT = {
    "E1 sector rotation": ("formation", "top_n", "keep", "min_dollar_vol"),
    "E1b rotation + MA filter": ("formation", "top_n", "keep", "min_dollar_vol", "ma_days"),
    "E2 dual momentum": ("formation", "top_n", "keep", "min_dollar_vol", "ma_days"),
    "E3 multi-asset trend": ("formation", "top_n", "keep", "min_dollar_vol", "ma_days", "vol_days", "max_weight"),
    "E4 volatility targeting": ("target_vol_days", "vol_threshold"),
}


def load_data(prices_db: Path = PRICES, funds: Path = FUNDS):
    from research.vcp import prices
    names = pd.read_csv(funds).set_index("ticker")["name"].to_dict()
    conn = prices.connect(str(prices_db))
    try:
        return prices.load_all(conn), names
    finally:
        conn.close()


def flags(e: Dict) -> Dict[str, bool]:
    """Condition 5 and the overall verdict (c2 None: the random test doesn't apply)."""
    c5 = bool(e["sens_share"] >= 0.80 and e["stress_excess"] > 0)
    return {"c5": c5, "passed": bool(e["c1"] and e["c2"] is not False and e["c3"] and e["c4"] and c5)}


def report(test: EtfTest, n_random: int = en.N_RANDOM) -> pd.DataFrame:
    base = test.run()
    stress = test.run(cost=2 * COST)
    sens = {k: test.run(c) for k, c in sensitivity_configs().items()}
    rand = en.random_portfolios(test.eligible(), test.returns, n_random=n_random)
    rand_bonds = en.random_portfolios(test.eligible(bonds=True), test.returns, n_random=n_random)
    rows = []
    for name, res in base.items():
        e = en.evaluate(res["net"], test.bench, test.years)
        pool = rand_bonds if name.startswith("E3") else rand
        runs = [en.evaluate(sens[k][name]["net"], test.bench, test.years)["excess"]
                for k in sens if k.split()[0] in RELEVANT[name]]
        e["random_95"] = float(np.quantile(pool, 0.95))
        e["random_median"] = float(np.median(pool))
        e["c2"] = bool(en.cagr(res["gross"]) > e["random_95"]) if not name.startswith("E4") else None
        e["sens_share"] = float(np.mean(np.array(runs) > 0))
        e["stress_excess"] = en.evaluate(stress[name]["net"], test.bench, test.years)["excess"]
        e["turnover_per_year"] = float(res["turnover"].mean() * 12)
        half = len(res) // 2
        ex = res["net"] - test.bench
        e["excess_first_half"], e["excess_second_half"] = float(ex.iloc[:half].mean() * 12), float(ex.iloc[half:].mean() * 12)
        e.update(flags(e))
        e["variants_run"] = 2 + len(runs)
        rows.append({"strategy": name, **e})
    return pd.DataFrame(rows)


def main() -> int:
    data, names = load_data()
    test = EtfTest(data, names)
    print("first signal", test.calendar[test.sessions[0]].date(), "periods", len(test.returns),
          "funds eligible first/last", len(test.eligible()[test.returns.index[0]]), len(test.eligible()[test.returns.index[-1]]))
    table = report(test)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "phase2_etf.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.set_index("strategy").T.to_string())
    (OUT / "phase2_passed.json").write_text(json.dumps(table.loc[table["passed"], "strategy"].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
