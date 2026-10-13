"""
Barometer scoring -- shared by Phase 1 (baselines) and Phases 3-4 (model
evaluation and the decision gate), so every comparison uses one definition.

Probabilities: brier(), log_loss(), skill() (1 - loss/reference loss),
auc() (rank / Mann-Whitney, ties averaged), calibration_table() +
calibration_error(). Volatility forecasts: mse(), qlike(). Significance:
newey_west_tstat() and date_clustered_diff() -- panel rows on one date are
not independent, and 20-day labels overlap across dates, so a loss
difference is averaged per date first and its t-stat is HAC-corrected with
a lag of the label horizon.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def brier(y, p) -> float:
    """Mean squared error of a probability forecast against 0/1 outcomes."""
    y, p = _arr(y), _arr(p)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p, eps: float = 1e-6) -> float:
    """Binary cross-entropy; p clipped to [eps, 1-eps] so a confident miss is finite."""
    y, p = _arr(y), np.clip(_arr(p), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def skill(loss: float, reference_loss: float) -> float:
    """1 - loss/reference: >0 beats the reference, 0 ties it, <0 loses."""
    if not reference_loss > 0:
        return float("nan")
    return float(1.0 - loss / reference_loss)


def auc(y, p) -> float:
    """Probability a random positive outranks a random negative (ties count
    half). NaN when only one class is present."""
    y, p = _arr(y).astype(bool), _arr(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Quantile bins of p: count, mean predicted, mean observed. A constant
    forecast collapses to a single bin."""
    df = pd.DataFrame({"y": _arr(y), "p": _arr(p)})
    if df["p"].nunique() <= 1:
        df["bin"] = 0
    else:
        df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    return (df.groupby("bin")
              .agg(n=("y", "size"), mean_p=("p", "mean"), mean_y=("y", "mean"))
              .reset_index())


def calibration_error(table: pd.DataFrame) -> float:
    """Count-weighted |observed - predicted| relative to predicted:
    0.20 means outcomes are off from the forecasts by 20% on average."""
    denom = float((table["n"] * table["mean_p"]).sum())
    if not denom > 0:
        return float("nan")
    return float((table["n"] * (table["mean_y"] - table["mean_p"]).abs()).sum() / denom)


def mse(real, forecast) -> float:
    real, forecast = _arr(real), _arr(forecast)
    return float(np.mean((forecast - real) ** 2))


def qlike(real, forecast, floor: float = 1e-8) -> float:
    """QLIKE loss on variances (volatilities are squared here): mean of
    x - ln(x) - 1 with x = realized/forecast variance. Zero only for a
    perfect forecast; robust to noisy realized-volatility proxies."""
    r2 = np.maximum(_arr(real), floor) ** 2
    f2 = np.maximum(_arr(forecast), floor) ** 2
    x = r2 / f2
    return float(np.mean(x - np.log(x) - 1.0))


def newey_west_tstat(x, lag: int) -> Tuple[float, float]:
    """(mean, t-stat) of a time series with a Bartlett-kernel HAC standard
    error. lag=0 reduces to mean / (population sd / sqrt(n))."""
    x = _arr(x)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    mu = x.mean()
    e = x - mu
    var = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    if not var > 0:
        return float(mu), float("nan")
    return float(mu), float(mu / np.sqrt(var / n))


def date_clustered_diff(loss_a, loss_b, dates, lag: int) -> Tuple[float, float, int]:
    """Per-row loss difference (a - b), averaged per date, then a Newey-West
    t-stat over dates. Returns (mean diff, t, number of dates). Negative
    mean = a has the lower loss."""
    d = pd.Series(_arr(loss_a) - _arr(loss_b), index=pd.DatetimeIndex(dates))
    per_date = d.groupby(level=0).mean().sort_index()
    mu, t = newey_west_tstat(per_date.values, lag)
    return mu, t, len(per_date)
