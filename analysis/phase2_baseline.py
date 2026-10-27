"""
analysis/phase2_baseline.py -- Phase 2 of the idle-cash plan: how much does
each backtester fix change the unchanged strategy's results?

Reads reports/idle_cash/02_baseline/<variant>/ (written by
analysis/run_walk_forward_trades.py) and writes metrics.csv plus
exit_mix.csv next to them. Every variant runs the same 16 folds, signals and
starting equity ($50k per fold, no compounding across folds).

Usage:
    python analysis/phase2_baseline.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "reports" / "idle_cash" / "02_baseline"
VARIANTS = ["off", "costs", "yield", "conservative", "conservative_gap", "all_on"]
CAPITAL = 50_000.0


def _exit_category(reason) -> str:
    if not isinstance(reason, str) or not reason:
        return "unknown"
    return re.split(r"[(;]", reason, maxsplit=1)[0].strip() or "unknown"


def metrics(variant: str) -> tuple[dict, pd.Series]:
    d = DIR / variant
    folds = pd.read_csv(d / "backtest_folds.csv")
    trades = pd.read_csv(d / "backtest_trades.csv")
    opened = pd.read_csv(d / "backtest_open.csv")
    eq = pd.read_csv(d / "backtest_equity.csv")
    end_eq = eq.groupby("fold").total_equity.last()
    row = {
        "variant": variant,
        "compounded_ret_pct": ((1 + folds.ret_pct / 100).prod() - 1) * 100,
        "mean_fold_ret_pct": folds.ret_pct.mean(),
        "folds_positive": int((folds.ret_pct > 0).sum()),
        "mean_max_dd_pct": folds.max_dd_pct.mean(),
        "worst_max_dd_pct": folds.max_dd_pct.min(),
        "equity_gain_$": float((end_eq - CAPITAL).sum()),
        "closed_trades": len(trades),
        "open_at_fold_end": len(opened),
        "win_rate": float((trades.pnl > 0).mean()),
        "avg_exit_pnl_pct": float(trades.pnl_pct.mean()),
        "realized_$": float(trades.pnl.sum()),
        "costs_$": float(folds.costs.sum()),
        "cash_yield_$": float(folds.cash_yield.sum()),
    }
    mix = trades.exit_reason.map(_exit_category).value_counts().rename(variant)
    return row, mix


def main() -> None:
    rows, mixes = [], []
    for v in VARIANTS:
        if not (DIR / v / "backtest_folds.csv").exists():
            print(f"missing: {v}")
            continue
        r, m = metrics(v)
        rows.append(r)
        mixes.append(m)
    df = pd.DataFrame(rows).set_index("variant")
    base = df.loc["off"]
    for col in ("compounded_ret_pct", "mean_fold_ret_pct", "mean_max_dd_pct", "win_rate", "avg_exit_pnl_pct",
                "equity_gain_$"):
        df[f"d_{col}"] = df[col] - base[col]
    df.round(4).to_csv(DIR / "metrics.csv")
    mix = pd.concat(mixes, axis=1).fillna(0).astype(int)
    mix.to_csv(DIR / "exit_mix.csv")
    pd.set_option("display.width", 220)
    print(df.round(3).T.to_string())
    print()
    print(mix.to_string())


if __name__ == "__main__":
    main()
