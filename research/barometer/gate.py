"""
Barometer Phases 3-4 -- the formal evaluation and the pre-registered
decision gate (PLAN.md). The model under test is the primary one: logistic
regression for the binary targets, ridge for volatility (gradient boosting
was dropped in Phase 2). No days_to_earnings (Phase 2: immaterial).

Gate, as fixed in PLAN.md before any results, with its ambiguous words
pinned down here -- also before the gate was run:
  * "beats" a baseline = lower out-of-sample loss that year (skill > 0;
    significance is reported as years_beat_rule_signif, not required --
    the plan didn't ask for it).
  * every criterion must hold in at least MIN_YEARS of the 4 test years,
    including AUC and calibration (the stricter reading of the plan).
  T1 y_stop_hit, T2 y_up10 and y_dn10 (each separately):
      beats base AND beats rule, AUC >= 0.60, calibration error <= 0.20.
  T3 fwd_rvol: beats the rule (QLIKE).
  T4 y_beat_mkt: per date, mean 20-day return of the top quintile by model
      probability minus the bottom quintile's, averaged over the year's
      dates, above 1% (round-trip costs).

    python -m research.barometer.gate   # -> out/barometer_gate.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from research.barometer import metrics as M
from research.barometer.baselines import (
    BINARY_TARGETS, PRED_COL, TEST_YEARS, _qlike_rows, base_rate_predictions,
    vol_rule_predictions, walk_forward_splits,
)
from research.barometer.dataset import HORIZON, PANEL_PATH
from research.barometer.models import compare, walk_forward_predictions

MODEL = "logistic"
MIN_YEARS = 3
AUC_MIN = 0.60
CALIB_MAX = 0.20
SPREAD_MIN = 0.01
N_GROUPS = 5
OUT_CSV = Path(__file__).resolve().parents[2] / "out" / "barometer_gate.csv"

GATES = {
    "T1": ["y_stop_hit"],
    "T2": ["y_up10", "y_dn10"],
    "T3": ["fwd_rvol"],
    "T4": ["y_beat_mkt"],
}


def row_losses(panel: pd.DataFrame, preds: pd.DataFrame, target: str,
               model: str = MODEL, test_years: Iterable[int] = TEST_YEARS,
               horizon: int = HORIZON) -> pd.DataFrame:
    """Every labelled test row of `target`: year, regime (mkt_above_200),
    outcome, the three forecasts and their per-row losses (Brier, or QLIKE
    for fwd_rvol). Means per year equal compare()'s losses."""
    frames = []
    dates = panel.index.get_level_values("date")
    for year, train_m, test_m in walk_forward_splits(dates, test_years, horizon):
        test = panel[test_m]
        pm = preds.loc[test_m, f"{model}:{target}"].to_numpy()
        lab = test[target].notna().to_numpy() & ~np.isnan(pm)
        if not lab.any():
            continue
        y = test[target].to_numpy()[lab]
        pb = base_rate_predictions(panel[train_m], test)[PRED_COL[target]].to_numpy()[lab]
        pr = vol_rule_predictions(test, horizon)[PRED_COL[target]].to_numpy()[lab]
        pm = pm[lab]
        if target == "fwd_rvol":
            lm, lb, lr = _qlike_rows(y, pm), _qlike_rows(y, pb), _qlike_rows(y, pr)
        else:
            lm, lb, lr = (pm - y) ** 2, (pb - y) ** 2, (pr - y) ** 2
        frames.append(pd.DataFrame({
            "year": year, "regime": test["mkt_above_200"].to_numpy()[lab],
            "y": y, "p_model": pm, "p_base": pb, "p_rule": pr,
            "loss_model": lm, "loss_base": lb, "loss_rule": lr,
        }, index=test.index[lab]))
    return pd.concat(frames) if frames else pd.DataFrame()


def regime_breakdown(rows: pd.DataFrame) -> pd.DataFrame:
    """Pooled over test years, split by market regime (1 = XIU above its
    200-day average): n, skill vs base, skill vs rule."""
    out = []
    for regime, g in rows.groupby("regime"):
        out.append({"regime": "bull" if regime == 1 else "bear", "n": len(g),
                    "skill_vs_base": M.skill(g["loss_model"].mean(), g["loss_base"].mean()),
                    "skill_vs_rule": M.skill(g["loss_model"].mean(), g["loss_rule"].mean())})
    return pd.DataFrame(out)


def group_spread(score: pd.Series, fwd: pd.Series, n_groups: int = N_GROUPS) -> pd.Series:
    """Per date: mean `fwd` of the top 1/n_groups by `score` minus the bottom
    1/n_groups (both indexed by (date, ticker)). Dates with fewer than
    n_groups rows are skipped. Returns a Series indexed by date."""
    df = pd.DataFrame({"s": score, "f": fwd}).dropna()

    def one(g: pd.DataFrame) -> float:
        if len(g) < n_groups:
            return np.nan
        k = len(g) // n_groups
        g = g.sort_values("s", kind="mergesort")
        return float(g["f"].iloc[-k:].mean() - g["f"].iloc[:k].mean())

    return df.groupby(level="date").apply(one).dropna()


def yearly_spreads(panel: pd.DataFrame, preds: pd.DataFrame, model: str = MODEL) -> pd.Series:
    """T4: the year's mean top-minus-bottom quintile spread of fwd_ret,
    ranked by the model's P(beat XIU). Only test rows with a prediction."""
    col = f"{model}:y_beat_mkt"
    has = preds[col].notna() & panel["fwd_ret"].notna()
    per_date = group_spread(preds.loc[has, col], panel.loc[has, "fwd_ret"])
    return per_date.groupby(per_date.index.year).mean()


def evaluate_gate(cmp: pd.DataFrame, spreads: pd.Series, model: str = MODEL,
                  min_years: int = MIN_YEARS) -> pd.DataFrame:
    """One row per (gate, target): how many test years each criterion held,
    and PASS/FAIL. `cmp` is models.compare()'s output, `spreads` is
    yearly_spreads()."""
    c = cmp[cmp["model"] == model]
    rows = []
    for gate, targets in GATES.items():
        for target in targets:
            g = c[c["target"] == target]
            n_years = len(g)
            r = {"gate": gate, "target": target, "years": n_years}
            if gate == "T4":
                s = spreads.reindex(g["year"]).fillna(-np.inf) if n_years else spreads
                r["years_spread_ok"] = int((s > SPREAD_MIN).sum())
                r["mean_spread"] = float(spreads.mean()) if len(spreads) else np.nan
                checks = [r["years_spread_ok"]]
            elif gate == "T3":
                r["years_beat_rule"] = int((g["skill_vs_rule"] > 0).sum())
                checks = [r["years_beat_rule"]]
                r["years_beat_rule_signif"] = _signif(g)
            else:
                r["years_beat_base"] = int((g["skill_vs_base"] > 0).sum())
                r["years_beat_rule"] = int((g["skill_vs_rule"] > 0).sum())
                r["years_auc_ok"] = int((g["auc"] >= AUC_MIN).sum())
                r["years_calib_ok"] = int((g["calib"] <= CALIB_MAX).sum())
                checks = [r["years_beat_base"], r["years_beat_rule"], r["years_auc_ok"], r["years_calib_ok"]]
                r["years_beat_rule_signif"] = _signif(g)
            r["passed"] = bool(n_years >= min_years and all(k >= min_years for k in checks))
            rows.append(r)
    return pd.DataFrame(rows)


def _signif(g: pd.DataFrame) -> int:
    """Years whose win over the rule is significant (HAC t < -2). Reported
    only -- the pre-registered gate counts point-estimate wins."""
    if "t_vs_rule" not in g:
        return 0
    return int(((g["skill_vs_rule"] > 0) & (g["t_vs_rule"] < -2)).sum())


def pooled_calibration(rows: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Model and rule calibration tables side by side, pooled over test years."""
    m = M.calibration_table(rows["y"], rows["p_model"], bins).add_prefix("model_")
    r = M.calibration_table(rows["y"], rows["p_rule"], bins).add_prefix("rule_")
    return pd.concat([m, r], axis=1)


def main(argv: Optional[List[str]] = None) -> int:
    if not PANEL_PATH.exists():
        print(f"{PANEL_PATH} missing -- build it first: python -m research.barometer.dataset")
        return 1
    panel = pd.read_pickle(PANEL_PATH)
    preds = walk_forward_predictions(panel, models=[MODEL])
    cmp = compare(panel, preds, models=[MODEL])
    spreads = yearly_spreads(panel, preds)
    gate = evaluate_gate(cmp, spreads)
    pd.set_option("display.width", 200)
    print("== Gate (logistic / ridge; criteria must hold in >= 3 of 4 test years)")
    print(gate.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n== T4 top-minus-bottom quintile 20-day return spread by year")
    print(spreads.to_string(float_format=lambda v: f"{v:+.4f}"))
    for target in BINARY_TARGETS + ["fwd_rvol"]:
        rows = row_losses(panel, preds, target)
        print(f"\n== {target}: by market regime (pooled 2023-2026)")
        print(regime_breakdown(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if target == "y_stop_hit":
            print("\n== y_stop_hit pooled calibration (model vs rule, deciles)")
            print(pooled_calibration(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    gate.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
