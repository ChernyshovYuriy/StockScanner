"""
Barometer Phase 1 -- the two baselines every later model must beat
(PLAN.md, Phases 1 and 4), evaluated walk-forward on the Phase 0 panel.

  base rate        each binary label's frequency in the TRAINING rows; for
                   fwd_rvol, the training mean. Knows nothing about the stock.
  volatility rule  no fitting at all: a driftless random walk with the
                   stock's own trailing 20-day volatility (vol_20).
                   * P(stop hit): reflection principle, P(min <= b) =
                     2*Phi(b / (sigma*sqrt(H))), with the Broadie-Glasserman
                     shift (barrier moved 0.5826*sigma further away) because
                     the stop is checked on daily closes, not continuously.
                   * P(+/-10%): Phi tails of the H-day log return.
                   * fwd_rvol forecast: vol_20 itself (volatility persists).
                   * P(beat XIU): 0.5 -- a random walk has no view.

Walk-forward: for test year Y, train on rows dated at least `embargo + 1`
sessions before Y's first session, so no training label's window (t+1..t+H)
reaches into the test year.

    python -m research.barometer.baselines    # evaluate, print, write out/barometer_phase1.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from research.barometer import metrics as M
from research.barometer.dataset import HORIZON, PANEL_PATH, UP_DOWN_THRESHOLD

BROADIE_GLASSERMAN_BETA = 0.5826   # -zeta(1/2)/sqrt(2*pi): discrete-monitoring barrier shift
TEST_YEARS = (2023, 2024, 2025, 2026)
OUT_CSV = Path(__file__).resolve().parents[2] / "out" / "barometer_phase1.csv"

BINARY_TARGETS = ["y_stop_hit", "y_up10", "y_dn10", "y_beat_mkt"]
PRED_COL = {"y_stop_hit": "p_stop_hit", "y_up10": "p_up10", "y_dn10": "p_dn10",
            "y_beat_mkt": "p_beat_mkt", "fwd_rvol": "f_rvol"}


# ── volatility rule (closed form, no fitting) ────────────────────────────────

def barrier_hit_prob(log_dist, sigma, horizon: int = HORIZON, discrete: bool = True) -> np.ndarray:
    """P(a driftless random walk with daily vol `sigma` touches a barrier at
    log distance `log_dist` (< 0, below) within `horizon` daily steps).
    discrete=True applies the Broadie-Glasserman correction for daily
    (not continuous) monitoring. A barrier at/above the start returns 1."""
    b = np.asarray(log_dist, dtype=float)
    s = np.asarray(sigma, dtype=float)
    if discrete:
        b = b - BROADIE_GLASSERMAN_BETA * s
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * norm.cdf(b / (s * np.sqrt(horizon)))
    p = np.where(np.asarray(log_dist) >= 0, 1.0, p)
    return np.clip(p, 0.0, 1.0)


def tail_probs(sigma, horizon: int = HORIZON, threshold: float = UP_DOWN_THRESHOLD
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(P(H-day return > +threshold), P(H-day return < -threshold)) for a
    driftless random walk with daily log-return vol `sigma`."""
    s = np.asarray(sigma, dtype=float) * np.sqrt(horizon)
    p_up = 1 - norm.cdf(np.log(1 + threshold) / s)
    p_dn = norm.cdf(np.log(1 - threshold) / s)
    return p_up, p_dn


def vol_rule_predictions(rows: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Volatility-rule forecasts for every row, from that row's own features."""
    sigma = rows["vol_20"].to_numpy()
    p_up, p_dn = tail_probs(sigma, horizon)
    return pd.DataFrame({
        "p_stop_hit": barrier_hit_prob(np.log1p(rows["stop_dist_pct"].to_numpy()), sigma, horizon),
        "p_up10": p_up,
        "p_dn10": p_dn,
        "p_beat_mkt": 0.5,
        "f_rvol": sigma,
    }, index=rows.index)


# ── base rate (fitted on training rows only) ─────────────────────────────────

def base_rate_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Constant forecasts for `test`, each the training-label mean."""
    out = pd.DataFrame(index=test.index)
    for target in BINARY_TARGETS + ["fwd_rvol"]:
        out[PRED_COL[target]] = float(train[target].mean())
    return out


# ── walk-forward split ───────────────────────────────────────────────────────

def walk_forward_splits(dates: pd.DatetimeIndex, test_years: Iterable[int] = TEST_YEARS,
                        embargo: int = HORIZON) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """[(year, train_mask, test_mask)] over row `dates`. Train rows are dated
    at least embargo+1 sessions before the test year's first session, so
    their t+1..t+embargo label windows end before it. Years with no rows
    or no eligible training rows are skipped."""
    dates = pd.DatetimeIndex(dates)
    sessions = pd.DatetimeIndex(np.unique(dates.values))
    out = []
    for year in test_years:
        test = np.asarray(dates.year == year)
        if not test.any():
            continue
        first_pos = sessions.searchsorted(dates[test].min())
        cut_pos = first_pos - embargo - 1
        if cut_pos < 0:
            continue
        train = np.asarray(dates <= sessions[cut_pos])
        out.append((year, train, test))
    return out


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate(panel: pd.DataFrame, test_years: Iterable[int] = TEST_YEARS,
             horizon: int = HORIZON) -> pd.DataFrame:
    """One row per (year, target): both baselines scored on that year's
    labelled rows. For binary targets: Brier, skill of the rule vs the base
    rate, AUC, calibration error, and a date-clustered HAC t-stat of the
    Brier difference (negative t = rule better). For fwd_rvol: MSE and
    QLIKE, the base rate being the training-mean volatility."""
    dates = panel.index.get_level_values("date")
    results = []
    for year, train_m, test_m in walk_forward_splits(dates, test_years, horizon):
        train, test = panel[train_m], panel[test_m]
        base = base_rate_predictions(train, test)
        rule = vol_rule_predictions(test, horizon)
        for target in BINARY_TARGETS + ["fwd_rvol"]:
            lab = test[target].notna().to_numpy()
            if not lab.any():
                continue
            y = test[target].to_numpy()[lab]
            pb = base[PRED_COL[target]].to_numpy()[lab]
            pr = rule[PRED_COL[target]].to_numpy()[lab]
            d = test.index.get_level_values("date")[lab]
            row = {"year": year, "target": target, "n": int(lab.sum()),
                   "n_dates": int(pd.Index(d).nunique()), "train_n": int(train[target].notna().sum())}
            if target == "fwd_rvol":
                row.update(mean=float(y.mean()),
                           mse_base=M.mse(y, pb), mse_rule=M.mse(y, pr),
                           qlike_base=M.qlike(y, pb), qlike_rule=M.qlike(y, pr))
                row["skill_rule"] = M.skill(row["qlike_rule"], row["qlike_base"])
                _, row["t_rule_vs_base"], _ = M.date_clustered_diff(
                    _qlike_rows(y, pr), _qlike_rows(y, pb), d, horizon)
            else:
                row.update(rate=float(y.mean()), base_p=float(pb[0]),
                           brier_base=M.brier(y, pb), brier_rule=M.brier(y, pr),
                           auc_rule=M.auc(y, pr),
                           calib_base=M.calibration_error(M.calibration_table(y, pb)),
                           calib_rule=M.calibration_error(M.calibration_table(y, pr)))
                row["skill_rule"] = M.skill(row["brier_rule"], row["brier_base"])
                _, row["t_rule_vs_base"], _ = M.date_clustered_diff(
                    (pr - y) ** 2, (pb - y) ** 2, d, horizon)
            results.append(row)
    return pd.DataFrame(results)


def _qlike_rows(real: np.ndarray, forecast: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Per-row QLIKE terms (M.qlike() is their mean)."""
    x = np.maximum(real, floor) ** 2 / np.maximum(forecast, floor) ** 2
    return x - np.log(x) - 1.0


def _print(res: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    for target, g in res.groupby("target", sort=False):
        print(f"\n== {target}")
        if target == "fwd_rvol":
            cols = ["year", "n", "mean", "qlike_base", "qlike_rule", "skill_rule", "t_rule_vs_base"]
        else:
            cols = ["year", "n", "rate", "base_p", "brier_base", "brier_rule", "skill_rule",
                    "t_rule_vs_base", "auc_rule", "calib_base", "calib_rule"]
        print(g[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main(argv: Optional[List[str]] = None) -> int:
    if not PANEL_PATH.exists():
        print(f"{PANEL_PATH} missing -- build it first: python -m research.barometer.dataset")
        return 1
    panel = pd.read_pickle(PANEL_PATH)
    res = evaluate(panel)
    _print(res)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
