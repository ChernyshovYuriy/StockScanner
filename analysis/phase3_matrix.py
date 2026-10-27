"""
analysis/phase3_matrix.py -- Phase 3 of the idle-cash plan: the comparison
matrix, judged by the pre-registered reports/idle_cash/03_matrix/DECISION_RULES.md.

Reads the strategy runs A/B/C written by
    python analysis/run_walk_forward_trades.py {realistic,adverse,zero}_{A,B,C}
(reports/idle_cash/03_matrix/runs/<scenario>_<id>/), runs the XIU benchmarks
D/E/F (analysis/idle_cash_benchmarks.py) on the same fold days, and writes to
reports/idle_cash/03_matrix/:

    metrics.csv       per variant x cost scenario: CAGR, max DD, Sharpe, Sortino,
                      Calmar, Ulcer, exposure, turnover, costs
    pairs.csv         B-D, C-E, E-D, B-A, C-B: annualized difference, tracking
                      error, IR, block-bootstrap CIs (5/10/20 days), fold wins
    regressions.csv   Jensen's alpha/beta with Newey-West t (excess and raw returns)
    attribution.csv   B and C (realistic): XIU sleeve, stock XIU-matched, stock
                      excess over XIU, cash yield, costs -- cumulative dollars
    decision.json     each rule's inputs and pass/fail
    stitched_returns.csv, equity_curves.png, drawdowns.png, attribution.png

Usage:
    python analysis/phase3_matrix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from backtest_runner import _lookback_start  # noqa: E402
from idle_cash_benchmarks import BENCHMARK, run_fold  # noqa: E402
from market_data import HistoricalSliceProvider  # noqa: E402
from market_data_cache import load_from_cache  # noqa: E402
from run_walk_forward_trades import COST_SCENARIOS, END, LOOKBACK_DAYS, START, _cash_returns  # noqa: E402

DIR = ROOT / "reports" / "idle_cash" / "03_matrix"
RUNS = DIR / "runs"
CAPITAL = 50_000.0
TD = 252
BLOCKS = (5, 10, 20)
N_BOOT = 10_000
SEED = 20261027
SCENARIOS = ("realistic", "adverse", "zero")
XIU_BPS = {"realistic": 2.0, "adverse": 4.0, "zero": 0.0}
STOCK_BPS = {"realistic": 10.0, "adverse": 20.0, "zero": 0.0}
PAIRS = [("B", "D"), ("C", "E"), ("E", "D"), ("B", "A"), ("C", "B")]
STRATS = ("A", "B", "C")
VARIANTS = ("A", "B", "C", "D", "E", "F")


# ── data ─────────────────────────────────────────────────────────────────────

def _load_run(scen: str, vid: str) -> dict:
    d = RUNS / f"{scen}_{vid}"
    out = {k: pd.read_csv(d / f"backtest_{k}.csv") for k in ("equity", "folds", "trades", "open")}
    idle = d / "backtest_idle.csv"
    out["idle"] = pd.read_csv(idle) if idle.exists() else None
    out["equity"]["date"] = pd.to_datetime(out["equity"]["date"])
    return out


def _fold_returns(eq: pd.DataFrame) -> pd.Series:
    """Daily returns, each fold chained from its own $50k start."""
    parts = []
    for _, g in eq.groupby("fold", sort=True):
        e = g.set_index("date")["total_equity"]
        parts.append(e / e.shift(1).fillna(CAPITAL) - 1)
    return pd.concat(parts).sort_index()


def _xiu_returns(xiu: pd.DataFrame, eq: pd.DataFrame) -> pd.Series:
    """XIU total return per strategy day: open->close on a fold's first day, else close->close."""
    c2c = xiu["Close"].pct_change()
    o2c = xiu["Close"] / xiu["Open"] - 1
    parts = []
    for _, g in eq.groupby("fold", sort=True):
        days = pd.DatetimeIndex(g["date"])
        r = c2c.reindex(days)
        r.iloc[0] = o2c.reindex(days).iloc[0]
        parts.append(r)
    return pd.concat(parts).sort_index().fillna(0.0)


# ── metrics ──────────────────────────────────────────────────────────────────

def _metrics(r: pd.Series, rc: pd.Series, eq: pd.DataFrame, traded: float, costs: float) -> dict:
    curve = (1 + r).cumprod()
    dd = curve / curve.cummax() - 1
    years = len(r) / TD
    cagr = curve.iloc[-1] ** (1 / years) - 1
    ex = r - rc
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return dict(
        cagr_pct=cagr * 100,
        total_ret_pct=(curve.iloc[-1] - 1) * 100,
        max_dd_pct=dd.min() * 100,
        sharpe=ex.mean() / ex.std() * np.sqrt(TD),
        sortino=ex.mean() / downside * np.sqrt(TD) if downside > 0 else np.nan,
        calmar=cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        ulcer=float(np.sqrt(((dd * 100) ** 2).mean())),
        exposure_pct=float(((eq["open_value"] + eq["xiu_value"]) / eq["total_equity"]).mean() * 100),
        turnover_x_per_yr=traded / eq["total_equity"].mean() / years,
        costs_usd=costs,
        days=len(r),
    )


def _block_bootstrap(d: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap of the mean of d."""
    n = len(d)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(N_BOOT, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(N_BOOT, -1)[:, :n] % n
    return d[idx].mean(axis=1)


def _newey_west(y: np.ndarray, x: np.ndarray) -> dict:
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    T = len(y)
    L = int(np.floor(4 * (T / 100) ** (2 / 9)))
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, L + 1):
        w = 1 - lag / (L + 1)
        G = Xu[lag:].T @ Xu[:-lag]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    return dict(alpha_daily=beta[0], alpha_ann_pct=beta[0] * TD * 100, beta=beta[1],
                t_alpha=beta[0] / se[0], t_beta=beta[1] / se[1], nw_lags=L, n=T)


# ── attribution ──────────────────────────────────────────────────────────────

def _attribution(run: dict, xiu: pd.DataFrame, scen: str) -> pd.DataFrame:
    eq = run["equity"].copy()
    idle = run["idle"].copy()
    idle["date"] = pd.to_datetime(idle["date"])
    s = STOCK_BPS[scen] / 1e4
    trades, opened = run["trades"], run["open"]
    for df in (trades, opened):
        df["entry_date"] = pd.to_datetime(df["entry_date"])
    buys = pd.concat([trades[["entry_date", "entry_price", "shares"]], opened[["entry_date", "entry_price", "shares"]]])
    buy_notional = (buys.entry_price * buys.shares).groupby(buys.entry_date).sum()
    buy_cost = (buys.entry_price * buys.shares * s / (1 + s)).groupby(buys.entry_date).sum()
    sell_dates = pd.to_datetime(trades["sell_date"])
    sell_cost = (trades.sell_price * trades.shares * s / (1 - s)).groupby(sell_dates).sum() if s else pd.Series(dtype=float)
    stock_cost = buy_cost.add(sell_cost, fill_value=0.0)
    c2c = xiu["Close"].pct_change()
    o2c = xiu["Close"] / xiu["Open"] - 1
    rows = []
    for f, g in eq.groupby("fold", sort=True):
        g = g.set_index("date")
        gi = idle[idle.fold == f].set_index("date").reindex(g.index)
        d_eq = g["total_equity"] - g["total_equity"].shift(1).fillna(CAPITAL)
        prev_open = g["open_value"].shift(1).fillna(0.0)
        r_c2c = c2c.reindex(g.index).fillna(0.0)
        r_c2c.iloc[0] = 0.0                                  # no stock held before the fold's first open
        matched = prev_open * r_c2c + buy_notional.reindex(g.index).fillna(0.0) * o2c.reindex(g.index).fillna(0.0)
        sc = stock_cost.reindex(g.index).fillna(0.0)
        stock_net = d_eq - gi["xiu_pnl"] + gi["xiu_costs"] - gi["cash_yield"]
        stock_gross = stock_net + sc
        rows.append(pd.DataFrame(dict(
            fold=f, xiu_sleeve=gi["xiu_pnl"], stock_xiu_matched=matched, stock_excess=stock_gross - matched,
            cash_yield=gi["cash_yield"], costs=-(gi["xiu_costs"] + sc), total=d_eq)))
    return pd.concat(rows).sort_index()


# ── charts ───────────────────────────────────────────────────────────────────

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4", "F": "#008300"}
LABELS = {"A": "A strategy, cash", "B": "B strategy, XIU always", "C": "C strategy, XIU gated",
          "D": "D XIU buy-and-hold", "E": "E XIU gated", "F": "F 55/45 XIU/cash"}


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _charts(returns: pd.DataFrame, attrib: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = (1 + returns).cumprod() * 100
    for name, data, title, ylabel in (
            ("equity_curves.png", curves, "Stitched out-of-sample growth of 100 (realistic costs)", "value"),
            ("drawdowns.png", (curves / curves.cummax() - 1) * 100, "Drawdown (realistic costs)", "% below peak")):
        fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
        for v in VARIANTS:
            ax.plot(data.index, data[v], color=SERIES[v], linewidth=2, label=LABELS[v])
            if name == "equity_curves.png":
                ax.annotate(f"{v} {data[v].iloc[-1]:.0f}", (data.index[-1], data[v].iloc[-1]),
                            xytext=(6, 0), textcoords="offset points", color=INK2, fontsize=9, va="center")
        _style(ax, title, ylabel)
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
        fig.tight_layout()
        fig.savefig(DIR / name, dpi=120)
        plt.close(fig)

    comps = [("xiu_sleeve", "XIU sleeve", "#2a78d6"), ("stock_xiu_matched", "Stocks: XIU-matched", "#eb6834"),
             ("stock_excess", "Stocks: excess over XIU", "#1baf7a"), ("cash_yield", "Cash yield", "#eda100"),
             ("costs", "Costs", "#e87ba4")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE, sharey=True)
    for ax, (vid, a) in zip(axes, attrib.items()):
        cum = a.drop(columns="fold").cumsum()
        for col, label, color in comps:
            ax.plot(cum.index, cum[col], color=color, linewidth=2, label=label)
        ax.plot(cum.index, cum["total"], color=INK, linewidth=2, linestyle="--", label="Total")
        _style(ax, LABELS[vid], r"cumulative \$ (sum of 16 folds x \$50k)")
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(DIR / "attribution.png", dpi=120)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cash_r = _cash_returns()
    raw = load_from_cache([BENCHMARK], _lookback_start(START, LOOKBACK_DAYS), END)
    provider = HistoricalSliceProvider({BENCHMARK: raw[BENCHMARK]})
    xiu = raw[BENCHMARK].copy()
    xiu.index = pd.to_datetime(xiu.index).tz_localize(None)

    metrics_rows, pair_rows, reg_rows = [], [], []
    stitched, fold_rets, check = {}, {}, {}
    rng = np.random.default_rng(SEED)
    attrib = {}
    for scen in SCENARIOS:
        runs = {v: _load_run(scen, v) for v in STRATS}
        base_eq = runs["A"]["equity"]
        fold_days = {f: [d.date() for d in g["date"]] for f, g in base_eq.groupby("fold", sort=True)}
        for v in ("D", "E", "F"):
            eqs = [run_fold(v, days, provider, cash_r, CAPITAL, XIU_BPS[scen], 0.0).assign(fold=f)
                   for f, days in fold_days.items()]
            eq = pd.concat(eqs)
            eq["date"] = pd.to_datetime(eq["date"])
            last = eq.groupby("fold").last()
            runs[v] = dict(equity=eq, traded=float(last.cum_xiu_traded.sum()), costs=float(last.cum_costs.sum()))
        rets = pd.DataFrame({v: _fold_returns(runs[v]["equity"]) for v in VARIANTS})
        rc = pd.Series({pd.Timestamp(d): r for d, r in cash_r.items()}).reindex(rets.index).fillna(0.0)
        rx = _xiu_returns(xiu, base_eq).reindex(rets.index)
        stitched[scen] = rets
        fold_rets[scen] = pd.DataFrame({v: runs[v]["equity"].groupby("fold").total_equity.last() / CAPITAL - 1
                                        for v in VARIANTS})
        for v in VARIANTS:
            r = runs[v]
            if v in STRATS:
                stock_traded = float((r["trades"].entry_price * r["trades"].shares).sum()
                                     + (r["trades"].sell_price * r["trades"].shares).sum()
                                     + (r["open"].entry_price * r["open"].shares).sum())
                xiu_traded = float(r["idle"]["xiu_traded"].sum()) if r["idle"] is not None else 0.0
                traded, costs = stock_traded + xiu_traded, float(r["folds"].costs.sum())
            else:
                traded, costs = r["traded"], r["costs"]
            metrics_rows.append(dict(scenario=scen, variant=v, **_metrics(rets[v], rc, r["equity"], traded, costs)))
        for a, b in PAIRS:
            d = (rets[a] - rets[b]).to_numpy()
            row = dict(scenario=scen, pair=f"{a}-{b}", ann_diff_pct=d.mean() * TD * 100,
                       tracking_error_pct=d.std(ddof=1) * np.sqrt(TD) * 100,
                       info_ratio=d.mean() / d.std(ddof=1) * np.sqrt(TD),
                       fold_wins=int((fold_rets[scen][a] > fold_rets[scen][b]).sum()), folds=len(fold_rets[scen]))
            for blk in BLOCKS:
                boot = _block_bootstrap(d, blk, rng)
                row[f"ci_lo_ann_pct_b{blk}"] = np.percentile(boot, 2.5) * TD * 100
                row[f"ci_hi_ann_pct_b{blk}"] = np.percentile(boot, 97.5) * TD * 100
            row["ci_above_zero_all_blocks"] = all(row[f"ci_lo_ann_pct_b{blk}"] > 0 for blk in BLOCKS)
            pair_rows.append(row)
        for v in STRATS:
            for basis in ("excess", "raw"):
                y = (rets[v] - rc) if basis == "excess" else rets[v]
                x = (rx - rc) if basis == "excess" else rx
                reg_rows.append(dict(scenario=scen, variant=v, basis=basis, **_newey_west(y.to_numpy(), x.to_numpy())))
        check[scen] = {v: dict(trades=len(runs[v]["trades"]), open_at_end=len(runs[v]["open"]),
                               skipped_buys=int(runs[v]["idle"].groupby("fold").skipped_buys.first().sum())
                               if runs[v]["idle"] is not None else 0)
                       for v in STRATS}
        if scen == "realistic":
            attrib = {v: _attribution(runs[v], xiu, scen) for v in ("B", "C")}

    metrics = pd.DataFrame(metrics_rows)
    pairs = pd.DataFrame(pair_rows)
    regs = pd.DataFrame(reg_rows)
    metrics.round(4).to_csv(DIR / "metrics.csv", index=False)
    pairs.round(4).to_csv(DIR / "pairs.csv", index=False)
    regs.round(5).to_csv(DIR / "regressions.csv", index=False)
    pd.concat({s: stitched[s] for s in SCENARIOS}, axis=1).to_csv(DIR / "stitched_returns.csv")
    pd.concat({v: a.drop(columns="fold").cumsum() for v, a in attrib.items()}, axis=1).round(2).to_csv(
        DIR / "attribution.csv")
    _charts(stitched["realistic"], attrib)

    # ── apply DECISION_RULES.md ──────────────────────────────────────────────
    p1 = pd.read_csv(ROOT / "reports" / "idle_cash" / "01_trade_vs_xiu" / "totals.csv")
    p1 = p1[(p1.source == "backtest") & (p1["sample"] == "all")].set_index("costs")

    def rules(scen: str) -> dict:
        pr = pairs[pairs.scenario == scen].set_index("pair")
        rg = regs[(regs.scenario == scen) & (regs.basis == "excess")].set_index("variant")
        r2 = {"B-D": bool(pr.loc["B-D", "ci_above_zero_all_blocks"]), "C-E": bool(pr.loc["C-E", "ci_above_zero_all_blocks"])}
        cands = [v for v, p in (("B", r2["B-D"]), ("C", r2["C-E"])) if p]
        r3 = {v: bool(rg.loc[v, "alpha_daily"] > 0 and rg.loc[v, "t_alpha"] > 2) for v in cands}
        return dict(rule2=r2, candidates=cands, rule3=r3,
                    rule3_t={v: float(rg.loc[v, "t_alpha"]) for v in STRATS})

    real, adv = rules("realistic"), rules("adverse")
    rule1 = {c: float(p1.loc[c, "excess_mean_pct"]) for c in ("realistic", "adverse")}
    passed3 = [v for v, ok in real["rule3"].items() if ok]
    chosen = max(passed3, key=lambda v: real["rule3_t"][v]) if passed3 else None
    rule4 = bool(chosen and chosen in adv["candidates"] and adv["rule3"].get(chosen))
    keep = rule1["realistic"] > 0 and bool(chosen) and rule4
    m = metrics[metrics.scenario == "realistic"].set_index("variant")
    e_beats_d = bool(m.loc["E", "calmar"] > m.loc["D", "calmar"] and m.loc["E", "max_dd_pct"] > m.loc["D", "max_dd_pct"])
    decision = dict(rule1_phase1_mean_excess_pct=rule1, rule1_pass=rule1["realistic"] > 0,
                    realistic=real, adverse=adv, chosen=chosen, rule4_pass=rule4, keep_swing_layer=keep,
                    otherwise=dict(E_calmar=float(m.loc["E", "calmar"]), D_calmar=float(m.loc["D", "calmar"]),
                                   E_max_dd=float(m.loc["E", "max_dd_pct"]), D_max_dd=float(m.loc["D", "max_dd_pct"]),
                                   E_beats_D=e_beats_d),
                    recommendation=("keep swing layer" if keep else ("hold E" if e_beats_d else "hold XIU (D)")),
                    trade_check=check)
    (DIR / "decision.json").write_text(json.dumps(decision, indent=2, default=float))

    pd.set_option("display.width", 250)
    print(metrics.round(3).to_string(index=False))
    print()
    print(pairs.round(3).to_string(index=False))
    print()
    print(regs.round(4).to_string(index=False))
    print()
    print(json.dumps(decision, indent=2, default=float))


if __name__ == "__main__":
    main()
