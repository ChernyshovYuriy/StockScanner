"""
Barometer Phase 1 baselines (research/barometer/baselines.py): the closed-
form volatility rule is checked against Monte Carlo, the base rate and the
walk-forward split against leakage, and the whole evaluation end-to-end on a
synthetic panel whose answer is known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.barometer import baselines as B
from research.barometer.dataset import build_panel


def _mc_barrier(b, sigma, h, n=200_000, seed=0):
    """Fraction of daily-step driftless walks whose CLOSE path ends below b at some step."""
    rng = np.random.default_rng(seed)
    paths = np.cumsum(rng.normal(0, sigma, (n, h)), axis=1)
    return float((paths.min(axis=1) < b).mean())


@pytest.mark.parametrize("b, sigma", [(-0.05, 0.015), (-0.07, 0.03), (-0.03, 0.02), (-0.10, 0.012)])
def test_barrier_prob_matches_monte_carlo(b, sigma):
    analytic = float(B.barrier_hit_prob(b, sigma, 20))
    assert analytic == pytest.approx(_mc_barrier(b, sigma, 20), abs=0.01)


def test_discrete_correction_is_needed():
    """Without the Broadie-Glasserman shift the continuous formula clearly overstates daily-close hits."""
    b, sigma = -0.05, 0.02
    mc = _mc_barrier(b, sigma, 20)
    assert float(B.barrier_hit_prob(b, sigma, 20, discrete=False)) - mc > 0.03


def test_barrier_prob_monotonic_and_bounded():
    s = np.full(5, 0.02)
    far = B.barrier_hit_prob(np.array([-0.01, -0.03, -0.05, -0.08, -0.5]), s)
    assert np.all(np.diff(far) < 0) and far[-1] < 1e-6
    vols = B.barrier_hit_prob(np.full(3, -0.05), np.array([0.01, 0.02, 0.04]))
    assert np.all(np.diff(vols) > 0)
    assert B.barrier_hit_prob(0.01, 0.02) == 1.0       # barrier above the start: already "hit"


def test_tail_probs_match_monte_carlo():
    rng = np.random.default_rng(3)
    sigma = 0.025
    r = rng.normal(0, sigma, (200_000, 20)).sum(axis=1)
    p_up, p_dn = B.tail_probs(sigma, 20, 0.10)
    assert float(p_up) == pytest.approx((np.exp(r) - 1 > 0.10).mean(), abs=0.005)
    assert float(p_dn) == pytest.approx((np.exp(r) - 1 < -0.10).mean(), abs=0.005)
    assert float(p_up) > float(p_dn)                   # driftless in logs: +10% (ln 1.1=0.095) is closer than -10% (ln 0.9=-0.105)


def test_vol_rule_uses_only_the_rows_own_features():
    rows = pd.DataFrame({"vol_20": [0.01, 0.03], "stop_dist_pct": [-0.05, -0.05]})
    pred = B.vol_rule_predictions(rows)
    assert pred["p_stop_hit"].iloc[1] > pred["p_stop_hit"].iloc[0]
    assert (pred["p_beat_mkt"] == 0.5).all()
    assert pred["f_rvol"].tolist() == [0.01, 0.03]


def _dates(n=600, start="2022-06-01"):
    return pd.bdate_range(start, periods=n)


def test_walk_forward_split_embargo_and_disjointness():
    sess = _dates()
    dates = pd.DatetimeIndex(np.repeat(sess, 3))       # 3 tickers per date
    splits = B.walk_forward_splits(dates, [2023, 2024], embargo=20)
    assert [y for y, _, _ in splits] == [2023, 2024]
    for year, train, test in splits:
        assert not (train & test).any()
        assert (dates[test].year == year).all()
        first = dates[test].min()
        last_train = dates[train].max()
        gap = sess.get_loc(first) - sess.get_loc(last_train)
        assert gap == 21                                # label window t+1..t+20 ends the session before the test year
        assert len(dates[train]) == 3 * (sess.get_loc(last_train) + 1)


def test_walk_forward_skips_year_without_history():
    dates = pd.DatetimeIndex(_dates(30, "2023-01-02"))
    assert B.walk_forward_splits(dates, [2023], embargo=20) == []


def test_base_rate_is_fit_on_training_rows_only():
    idx = pd.RangeIndex(4)
    train = pd.DataFrame({t: [0.0, 1.0, 0.0, 1.0] for t in B.BINARY_TARGETS} | {"fwd_rvol": [0.01] * 4}, index=idx)
    test = pd.DataFrame({t: [1.0] * 4 for t in B.BINARY_TARGETS} | {"fwd_rvol": [0.5] * 4}, index=idx)
    pred = B.base_rate_predictions(train, test)
    assert (pred["p_stop_hit"] == 0.5).all() and (pred["f_rvol"] == 0.01).all()


# ── end to end on a synthetic panel with a known answer ──────────────────────

@pytest.fixture(scope="module")
def synthetic_panel():
    """40 random-walk stocks with very different but constant volatilities and
    zero drift: the volatility rule's assumptions hold exactly, so it must
    beat the base rate on stop-hit, tails and volatility -- and nothing can
    call direction (beat-the-market AUC stays ~0.5)."""
    cal = pd.bdate_range("2021-01-04", "2024-12-31")
    rng = np.random.default_rng(7)
    data = {"XIU.TO": _walk(rng, cal, 0.008, 30.0)}
    for i in range(40):
        data[f"S{i:02d}.TO"] = _walk(rng, cal, 0.006 + 0.001 * i, 20.0)
    return build_panel(data)


def _walk(rng, cal, sigma, start):
    c = start * np.exp(np.cumsum(rng.normal(0, sigma, len(cal))))
    return pd.DataFrame({"Open": c, "High": c * (1 + sigma / 2), "Low": c * (1 - sigma / 2),
                         "Close": c, "Volume": 1e6}, index=cal)


def test_evaluate_rule_beats_base_rate_when_its_model_is_true(synthetic_panel):
    res = B.evaluate(synthetic_panel, test_years=[2023, 2024])
    assert set(res["year"]) == {2023, 2024}
    # Stop-hit separates least: the stop sits at 1.5x ATR, which already scales
    # with each stock's own volatility, so only the 7% cap makes risk differ.
    for target in ["y_stop_hit", "y_dn10", "y_up10"]:
        g = res[res["target"] == target]
        assert (g["skill_rule"] > 0.03).all(), target
        assert (g["auc_rule"] > 0.60).all(), target
        assert (g["t_rule_vs_base"] < -2).all(), target         # lower Brier, significantly
    vol = res[res["target"] == "fwd_rvol"]
    assert (vol["skill_rule"] > 0.3).all()
    beat = res[res["target"] == "y_beat_mkt"]
    assert (beat["auc_rule"] == 0.5).all()                      # constant 0.5: all ties, no ranking at all
    assert (beat["skill_rule"].abs() < 0.03).all()


def test_evaluate_rows_and_counts(synthetic_panel):
    res = B.evaluate(synthetic_panel, test_years=[2024])
    assert len(res) == 5
    stop = res[res["target"] == "y_stop_hit"].iloc[0]
    labelled = synthetic_panel["y_stop_hit"].notna() & (synthetic_panel.index.get_level_values("date").year == 2024)
    assert stop["n"] == labelled.sum()
    assert 0 < stop["rate"] < 1 and 0 <= stop["calib_rule"]
