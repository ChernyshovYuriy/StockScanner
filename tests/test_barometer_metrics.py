"""
Barometer scoring (research/barometer/metrics.py). Every later phase's
verdict rests on these numbers, so each is checked against a hand value or
a brute-force reimplementation.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from research.barometer import metrics as M


def test_brier_hand_values():
    assert M.brier([1, 0], [1, 0]) == 0.0
    assert M.brier([1, 0], [0, 1]) == 1.0
    assert M.brier([1, 0, 1, 1], [0.5] * 4) == pytest.approx(0.25)
    assert M.brier([1, 0], [0.8, 0.3]) == pytest.approx((0.04 + 0.09) / 2)


def test_log_loss_hand_value_and_finite_on_confident_miss():
    assert M.log_loss([1, 0], [0.8, 0.2]) == pytest.approx(-np.log(0.8))
    assert np.isfinite(M.log_loss([1, 0], [0.0, 1.0]))


def test_skill_sign_and_degenerate_reference():
    assert M.skill(0.20, 0.25) == pytest.approx(0.2)
    assert M.skill(0.25, 0.25) == 0.0
    assert M.skill(0.30, 0.25) < 0
    assert np.isnan(M.skill(0.1, 0.0))


def _auc_brute(y, p):
    pos = [pi for yi, pi in zip(y, p) if yi]
    neg = [pi for yi, pi in zip(y, p) if not yi]
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def test_auc_known_cases():
    assert M.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert M.auc([0, 1, 0, 1], [0.5] * 4) == 0.5
    assert np.isnan(M.auc([1, 1], [0.2, 0.3]))


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_brute_force_with_ties(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 60)
    p = rng.integers(0, 8, 60) / 8.0          # heavy ties
    assert M.auc(y, p) == pytest.approx(_auc_brute(y, p))


def test_calibration_table_and_error():
    y = [0, 0, 1, 1, 1, 0, 1, 1]
    p = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
    t = M.calibration_table(y, p, bins=2)
    assert t["n"].sum() == 8 and len(t) == 2
    assert t["mean_y"].tolist() == [0.5, 0.75]
    # |0.5-0.1|*4 + |0.75-0.9|*4 over (0.1*4 + 0.9*4)
    assert M.calibration_error(t) == pytest.approx((1.6 + 0.6) / 4.0)


def test_calibration_error_small_for_calibrated_forecasts():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.6, 200_000)
    y = rng.uniform(size=p.size) < p
    assert M.calibration_error(M.calibration_table(y, p)) < 0.02
    assert M.calibration_error(M.calibration_table(y, p * 1.5)) > 0.25   # over-forecasting is caught


def test_constant_forecast_is_one_bin():
    t = M.calibration_table([0, 1, 1, 0], [0.3] * 4)
    assert len(t) == 1 and t["mean_y"].iloc[0] == 0.5


def test_volatility_losses():
    assert M.mse([0.01, 0.02], [0.01, 0.02]) == 0.0
    assert M.qlike([0.01, 0.02], [0.01, 0.02]) == pytest.approx(0.0)
    assert M.qlike([0.02], [0.01]) == pytest.approx(4 - np.log(4) - 1)
    # QLIKE punishes under-forecasting volatility more than over-forecasting by the same factor
    assert M.qlike([0.02], [0.01]) > M.qlike([0.01], [0.02])
    assert np.isfinite(M.qlike([0.0], [0.01]))


def test_newey_west_lag0_is_plain_t():
    x = np.array([1.0, 2.0, 3.0, 6.0])
    mu, t = M.newey_west_tstat(x, lag=0)
    assert mu == 3.0
    assert t == pytest.approx(3.0 / (np.std(x) / 2))


def test_newey_west_widens_for_autocorrelated_series():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 3000)
    x = 0.05 + np.convolve(e, np.ones(20) / 20, mode="same")   # strongly overlapping, like 20d labels
    _, t0 = M.newey_west_tstat(x, 0)
    _, t20 = M.newey_west_tstat(x, 20)
    assert abs(t20) < abs(t0) / 2


def test_date_clustered_diff_averages_within_date_first():
    dates = pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"])
    mu, t, n = M.date_clustered_diff([1, 1, 1, 0], [0, 0, 0, 0], dates, lag=0)
    assert n == 2 and mu == 0.5     # (1 + 0) / 2 dates, not 3/4 rows
