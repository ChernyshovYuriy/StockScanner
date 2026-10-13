"""
Barometer Phase 2 -- fitted models, scored walk-forward against both Phase 1
baselines on exactly the same splits (PLAN.md, Phases 2-3).

Features (build_design()):
  * the 15 stock features, ranked against that day's other rows (percentile
    - 0.5): a same-day cross-section, all known at t's close, so ranking adds
    no lookahead -- and it removes the market-wide level, which the model
    gets separately from:
  * the market features, standardised with TRAINING-set mean/sd only;
  * log(vol_20), raw: the absolute volatility level a rank throws away;
  * the volatility rule's own probability as logit(p) (binary targets except
    y_beat_mkt, where the rule is a constant): the model starts from the
    closed-form answer and learns corrections (drift, regime, calibration);
  * optionally days_to_earnings (scaled, NaN -> clip + a missing flag) --
    it carries mild lookahead, see PLAN.md, so it is switchable.

Models:
  logistic  LogisticRegression (L2, C=1.0, fixed -- not tuned, to avoid
            snooping); the PLAN's primary model.
  gbm       HistGradientBoostingClassifier with fixed, conservative settings;
            kept only if it beats logistic out of sample.
  fwd_rvol  Ridge on log(fwd_rvol) for either model name, mapped back with a
            lognormal mean correction (exp(pred + resid_var/2)) fit on train.

    python -m research.barometer.models              # -> out/barometer_phase2.csv
    python -m research.barometer.models --earnings   # same, with days_to_earnings
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from research.barometer import metrics as M
from research.barometer.baselines import (
    BINARY_TARGETS, PRED_COL, TEST_YEARS, _qlike_rows, base_rate_predictions,
    vol_rule_predictions, walk_forward_splits,
)
from research.barometer.dataset import EARNINGS_CLIP, FEATURE_COLS, HORIZON, MARKET_COLS, PANEL_PATH

MODELS = ("logistic", "gbm")
RULE_TARGETS = ("y_stop_hit", "y_up10", "y_dn10")   # targets where the vol rule has a real view
PROB_CLIP = 1e-4
VOL_FLOOR = 1e-4
RANDOM_STATE = 0
OUT_CSV = Path(__file__).resolve().parents[2] / "out" / "barometer_phase2.csv"


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), PROB_CLIP, 1 - PROB_CLIP)
    return np.log(p / (1 - p))


def cross_sectional_rank(rows: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Per-date percentile rank of each column, centred to (-0.5, 0.5].
    Ties share their average rank. Uses only rows on the same date."""
    cols = list(cols)
    return rows[cols].groupby(level="date").rank(pct=True) - 0.5


def build_design(train: pd.DataFrame, test: pd.DataFrame, target: str,
                 use_earnings: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(X_train, X_test, feature names) for one target. Every statistic fit
    from data (the market-feature standardisation) comes from `train` only."""
    parts_tr, parts_te, names = [], [], []

    def add(name: str, tr, te):
        parts_tr.append(np.asarray(tr, dtype=float))
        parts_te.append(np.asarray(te, dtype=float))
        names.append(name)

    rk_tr, rk_te = cross_sectional_rank(train, FEATURE_COLS), cross_sectional_rank(test, FEATURE_COLS)
    for c in FEATURE_COLS:
        add(f"rank_{c}", rk_tr[c], rk_te[c])
    for c in MARKET_COLS:
        mu, sd = train[c].mean(), train[c].std()
        sd = sd if sd > 0 else 1.0
        add(f"z_{c}", (train[c] - mu) / sd, (test[c] - mu) / sd)
    add("log_vol_20", np.log(np.maximum(train["vol_20"], VOL_FLOOR)),
        np.log(np.maximum(test["vol_20"], VOL_FLOOR)))
    if target in RULE_TARGETS:
        col = PRED_COL[target]
        add("rule_logit", logit(vol_rule_predictions(train)[col]), logit(vol_rule_predictions(test)[col]))
    if use_earnings:
        dte_tr, dte_te = train["days_to_earnings"], test["days_to_earnings"]
        add("dte_scaled", dte_tr.fillna(EARNINGS_CLIP) / EARNINGS_CLIP, dte_te.fillna(EARNINGS_CLIP) / EARNINGS_CLIP)
        add("dte_missing", dte_tr.isna(), dte_te.isna())
    return np.column_stack(parts_tr), np.column_stack(parts_te), names


def make_classifier(model: str):
    if model == "logistic":
        return LogisticRegression(C=1.0, max_iter=2000)
    if model == "gbm":
        # Overlapping H-day labels make ~H neighbouring rows one observation,
        # so a leaf needs ~100 x H rows to hold ~100 independent outcomes
        # (min_samples_leaf=500 overfit pure noise in the tests: -5..-8% vs the rule).
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=100, max_leaf_nodes=8, min_samples_leaf=100 * HORIZON,
            l2_regularization=1.0, early_stopping=False, random_state=RANDOM_STATE)
    raise ValueError(f"unknown model {model!r}")


def fit_predict_binary(model: str, X_train, y_train, X_test) -> np.ndarray:
    """P(y=1) for X_test. A single-class training set predicts that class's rate."""
    y_train = np.asarray(y_train, dtype=float)
    if len(np.unique(y_train)) < 2:
        return np.full(len(X_test), float(y_train.mean()))
    clf = make_classifier(model).fit(X_train, y_train.astype(int))
    return clf.predict_proba(X_test)[:, 1]


def fit_predict_vol(X_train, y_train, X_test) -> np.ndarray:
    """Forecast of fwd_rvol: Ridge on log(vol), back-transformed with the
    lognormal mean correction exp(pred + var(train residuals)/2)."""
    ly = np.log(np.maximum(np.asarray(y_train, dtype=float), VOL_FLOOR))
    reg = Ridge(alpha=1.0).fit(X_train, ly)
    resid_var = float(np.var(ly - reg.predict(X_train)))
    return np.exp(reg.predict(X_test) + resid_var / 2)


def walk_forward_predictions(panel: pd.DataFrame, models: Iterable[str] = MODELS,
                             targets: Optional[Iterable[str]] = None,
                             test_years: Iterable[int] = TEST_YEARS, horizon: int = HORIZON,
                             use_earnings: bool = False) -> pd.DataFrame:
    """Out-of-sample predictions for every test-year row: columns
    '<model>:<target>' (probabilities, or a volatility forecast for
    fwd_rvol). Each year's model sees only that split's training rows with
    a label. Rows outside the test years stay NaN."""
    targets = list(targets or BINARY_TARGETS + ["fwd_rvol"])
    models = list(models)
    out = pd.DataFrame(index=panel.index,
                       columns=[f"{m}:{t}" for m in models for t in targets], dtype=float)
    dates = panel.index.get_level_values("date")
    for year, train_m, test_m in walk_forward_splits(dates, test_years, horizon):
        test = panel[test_m]
        for target in targets:
            train = panel[train_m & panel[target].notna().to_numpy()]
            if train.empty:
                continue
            X_tr, X_te, _ = build_design(train, test, target, use_earnings)
            for m in models:
                if target == "fwd_rvol":
                    pred = fit_predict_vol(X_tr, train[target], X_te)
                else:
                    pred = fit_predict_binary(m, X_tr, train[target], X_te)
                out.loc[test_m, f"{m}:{target}"] = pred
    return out


def compare(panel: pd.DataFrame, preds: pd.DataFrame, models: Iterable[str] = MODELS,
            test_years: Iterable[int] = TEST_YEARS, horizon: int = HORIZON) -> pd.DataFrame:
    """One row per (year, target, model): the model's loss, its skill vs the
    base rate and vs the vol rule, AUC, calibration error, and the
    date-clustered HAC t-stat of its loss minus the rule's (negative = model
    better). Loss = Brier for binary targets, QLIKE for fwd_rvol."""
    rows = []
    dates = panel.index.get_level_values("date")
    for year, train_m, test_m in walk_forward_splits(dates, test_years, horizon):
        test = panel[test_m]
        base = base_rate_predictions(panel[train_m], test)
        rule = vol_rule_predictions(test, horizon)
        for target in BINARY_TARGETS + ["fwd_rvol"]:
            for m in models:
                col = f"{m}:{target}"
                if col not in preds:
                    continue
                pm_all = preds.loc[test_m, col].to_numpy()
                lab = test[target].notna().to_numpy() & ~np.isnan(pm_all)
                if not lab.any():
                    continue
                y, pm = test[target].to_numpy()[lab], pm_all[lab]
                pb = base[PRED_COL[target]].to_numpy()[lab]
                pr = rule[PRED_COL[target]].to_numpy()[lab]
                d = test.index.get_level_values("date")[lab]
                r = {"year": year, "target": target, "model": m, "n": int(lab.sum())}
                if target == "fwd_rvol":
                    lm, lb, lr = _qlike_rows(y, pm), _qlike_rows(y, pb), _qlike_rows(y, pr)
                else:
                    lm, lb, lr = (pm - y) ** 2, (pb - y) ** 2, (pr - y) ** 2
                    r["auc"] = M.auc(y, pm)
                    r["calib"] = M.calibration_error(M.calibration_table(y, pm))
                    r["calib_rule"] = M.calibration_error(M.calibration_table(y, pr))
                r.update(loss=float(lm.mean()), loss_base=float(lb.mean()), loss_rule=float(lr.mean()))
                r["skill_vs_base"] = M.skill(r["loss"], r["loss_base"])
                r["skill_vs_rule"] = M.skill(r["loss"], r["loss_rule"])
                _, r["t_vs_rule"], _ = M.date_clustered_diff(lm, lr, d, horizon)
                _, r["t_vs_base"], _ = M.date_clustered_diff(lm, lb, d, horizon)
                rows.append(r)
    return pd.DataFrame(rows)


def _print(res: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    for target, g in res.groupby("target", sort=False):
        print(f"\n== {target}")
        cols = ["year", "model", "n", "skill_vs_base", "t_vs_base", "skill_vs_rule", "t_vs_rule"]
        if target != "fwd_rvol":
            cols += ["auc", "calib", "calib_rule"]
        print(g[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Barometer Phase 2 models")
    ap.add_argument("--earnings", action="store_true", help="add days_to_earnings (mild lookahead)")
    args = ap.parse_args(argv)
    if not PANEL_PATH.exists():
        print(f"{PANEL_PATH} missing -- build it first: python -m research.barometer.dataset")
        return 1
    panel = pd.read_pickle(PANEL_PATH)
    preds = walk_forward_predictions(panel, use_earnings=args.earnings)
    res = compare(panel, preds)
    res.insert(0, "earnings", args.earnings)
    _print(res)
    out = OUT_CSV.with_name(OUT_CSV.stem + ("_earnings" if args.earnings else "") + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
