"""
Barometer Phase 2 models (research/barometer/models.py): feature design fit
on training rows only, model fitting, walk-forward predictions that never see
their own test year, and two end-to-end experiments with known answers --
a planted signal the model must find, and pure noise it must not "find".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.barometer import models as Mo
from research.barometer.dataset import FEATURE_COLS, MARKET_COLS, build_panel


def _walk(rng, cal, sigma, start, drift=0.0):
    c = start * np.exp(np.cumsum(rng.normal(drift, sigma, len(cal))))
    return pd.DataFrame({"Open": c, "High": c * (1 + sigma / 2), "Low": c * (1 - sigma / 2),
                         "Close": c, "Volume": 1e6}, index=cal)


CAL = pd.bdate_range("2021-01-04", "2024-12-31")


@pytest.fixture(scope="module")
def noise_panel():
    """Driftless random walks, varied vol: nothing beyond volatility is knowable."""
    rng = np.random.default_rng(11)
    data = {"XIU.TO": _walk(rng, CAL, 0.008, 30.0)}
    for i in range(40):
        data[f"N{i:02d}.TO"] = _walk(rng, CAL, 0.008 + 0.0006 * i, 20.0)
    return build_panel(data)


@pytest.fixture(scope="module")
def drift_panel():
    """Each stock has its own persistent daily drift (-0.4%..+0.38%), same vol
    (1%): momentum features reveal it, the driftless vol rule cannot. Strong on
    purpose -- knowing the TRUE drift ranks beat-the-market at AUC ~0.78 here;
    a weaker one (+/-0.3% on 1.5% vol) caps out near 0.65, too close to noise
    to test the machinery. Start at 500 so falling names stay above $2."""
    rng = np.random.default_rng(12)
    data = {"XIU.TO": _walk(rng, CAL, 0.008, 30.0)}
    for i in range(40):
        data[f"D{i:02d}.TO"] = _walk(rng, CAL, 0.010, 500.0, drift=-0.004 + 0.0002 * i)
    return build_panel(data)


# ── small pieces ─────────────────────────────────────────────────────────────

def test_logit_values_and_clipping():
    assert Mo.logit(0.5) == pytest.approx(0.0)
    assert Mo.logit(0.8) == pytest.approx(-Mo.logit(0.2))
    assert np.isfinite(Mo.logit([0.0, 1.0])).all()


def test_cross_sectional_rank_is_per_date_with_ties():
    idx = pd.MultiIndex.from_tuples(
        [("2024-01-02", "A"), ("2024-01-02", "B"), ("2024-01-02", "C"),
         ("2024-01-03", "A"), ("2024-01-03", "B")], names=["date", "ticker"])
    df = pd.DataFrame({"x": [3.0, 1.0, 3.0, 100.0, -100.0]}, index=idx)
    r = Mo.cross_sectional_rank(df, ["x"])["x"].tolist()
    # day 1: B lowest (1/3), A and C tie at (2+3)/2 / 3; day 2 ranks only among itself
    assert r == pytest.approx([2.5 / 3 - 0.5, 1 / 3 - 0.5, 2.5 / 3 - 0.5, 1.0 - 0.5, 0.5 - 0.5])


def _two_years(panel):
    d = panel.index.get_level_values("date")
    return panel[d.year == 2022], panel[d.year == 2023]


def test_design_is_fit_on_train_only(noise_panel):
    train, test = _two_years(noise_panel)
    X_tr, X_te, names = Mo.build_design(train, test, "y_stop_hit")
    shocked = test.copy()
    shocked[MARKET_COLS] = shocked[MARKET_COLS] * 10 + 5
    X_tr2, X_te2, _ = Mo.build_design(train, shocked, "y_stop_hit")
    np.testing.assert_array_equal(X_tr, X_tr2)              # test data never moves the training design
    j = names.index("z_mkt_vol_20")
    mu, sd = train["mkt_vol_20"].mean(), train["mkt_vol_20"].std()
    np.testing.assert_allclose(X_te[:, j], (test["mkt_vol_20"] - mu) / sd)


def test_design_columns_per_target_and_earnings(noise_panel):
    train, test = _two_years(noise_panel)
    _, _, n_stop = Mo.build_design(train, test, "y_stop_hit")
    _, _, n_beat = Mo.build_design(train, test, "y_beat_mkt")
    assert "rule_logit" in n_stop and "rule_logit" not in n_beat   # the rule is a constant there
    assert len(n_beat) == len(FEATURE_COLS) + len(MARKET_COLS) + 1
    X_tr, X_te, n_e = Mo.build_design(train, test, "y_beat_mkt", use_earnings=True)
    assert n_e[-2:] == ["dte_scaled", "dte_missing"]
    assert (X_te[:, -1] == 1).all() and (X_te[:, -2] == 1).all()   # no earnings data: missing flag, clip value
    assert np.isfinite(X_tr).all() and np.isfinite(X_te).all()


@pytest.mark.parametrize("model", Mo.MODELS)
def test_classifiers_find_a_planted_signal(model):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20_000, 3))
    y = rng.uniform(size=20_000) < 1 / (1 + np.exp(-(1.5 * X[:, 0] - 0.5)))
    p = Mo.fit_predict_binary(model, X[:15_000], y[:15_000], X[15_000:])
    assert ((p > 0) & (p < 1)).all()
    from research.barometer.metrics import auc
    assert auc(y[15_000:], p) > 0.75


def test_single_class_training_predicts_its_rate():
    p = Mo.fit_predict_binary("logistic", np.zeros((10, 2)), np.zeros(10), np.zeros((3, 2)))
    assert (p == 0).all()
    with pytest.raises(ValueError):
        Mo.make_classifier("svm")


def test_vol_forecast_is_unbiased_after_lognormal_correction():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40_000, 1))
    vol = np.exp(-4.0 + 0.4 * x[:, 0] + rng.normal(0, 0.3, 40_000))
    f = Mo.fit_predict_vol(x[:30_000], vol[:30_000], x[30_000:])
    assert f.mean() == pytest.approx(vol[30_000:].mean(), rel=0.02)
    # without the correction the forecast would sit ~4.6% low (exp(0.3**2/2))
    assert abs(f.mean() / np.exp(-4.0 + 0.4 * x[30_000:, 0]).mean() - np.exp(0.045)) < 0.01


# ── walk-forward: never sees its own test year ───────────────────────────────

def test_predictions_ignore_labels_from_test_year_and_embargo(noise_panel):
    years = [2023]
    base = Mo.walk_forward_predictions(noise_panel, ["logistic"], ["y_stop_hit"], years)
    d = noise_panel.index.get_level_values("date")
    sessions = pd.DatetimeIndex(np.unique(d.values))
    first_2023 = sessions[sessions.year == 2023][0]
    embargo_start = sessions[sessions.get_loc(first_2023) - 20]
    tampered = noise_panel.copy()
    hide = np.asarray(d >= embargo_start)
    tampered.loc[hide, "y_stop_hit"] = 1 - tampered.loc[hide, "y_stop_hit"]   # flip every label it must not see
    after = Mo.walk_forward_predictions(tampered, ["logistic"], ["y_stop_hit"], years)
    pd.testing.assert_frame_equal(base, after)
    assert base.loc[d.year != 2023].isna().all().all()
    assert base.loc[(d.year == 2023)].notna().all().all()


def test_predictions_do_depend_on_training_labels(noise_panel):
    years = [2023]
    base = Mo.walk_forward_predictions(noise_panel, ["logistic"], ["y_stop_hit"], years)
    tampered = noise_panel.copy()
    early = np.asarray(noise_panel.index.get_level_values("date").year == 2022)
    tampered.loc[early, "y_stop_hit"] = 1 - tampered.loc[early, "y_stop_hit"]
    after = Mo.walk_forward_predictions(tampered, ["logistic"], ["y_stop_hit"], years)
    assert not np.allclose(base.dropna().values, after.dropna().values)


# ── end to end: find the signal, don't invent one ────────────────────────────

@pytest.fixture(scope="module")
def drift_result(drift_panel):
    preds = Mo.walk_forward_predictions(drift_panel, test_years=[2023, 2024])
    return Mo.compare(drift_panel, preds, test_years=[2023, 2024])


@pytest.fixture(scope="module")
def noise_result(noise_panel):
    preds = Mo.walk_forward_predictions(noise_panel, test_years=[2023, 2024])
    return Mo.compare(noise_panel, preds, test_years=[2023, 2024])


@pytest.mark.parametrize("model", Mo.MODELS)
@pytest.mark.parametrize("target", ["y_beat_mkt", "y_stop_hit"])
def test_models_find_planted_drift_the_rule_cannot_see(drift_result, model, target):
    g = drift_result[(drift_result["model"] == model) & (drift_result["target"] == target)]
    assert len(g) == 2
    assert (g["skill_vs_rule"] > 0.10).all(), g
    assert (g["t_vs_rule"] < -3).all(), g
    assert (g["auc"] > 0.70).all(), g


@pytest.mark.parametrize("model", Mo.MODELS)
@pytest.mark.parametrize("target", ["y_up10", "y_dn10"])
def test_models_rank_planted_drift_tails(drift_result, model, target):
    """Tails are ranked well; their calibration can drift when the market
    features leave their training range (seen 2024, dn10), so only ranking is
    guaranteed here -- calibration is judged on real data in Phase 3."""
    g = drift_result[(drift_result["model"] == model) & (drift_result["target"] == target)]
    assert (g["auc"] > 0.75).all(), g


@pytest.mark.parametrize("model", Mo.MODELS)
@pytest.mark.parametrize("target", ["y_stop_hit", "y_up10", "y_dn10", "y_beat_mkt"])
def test_models_do_not_invent_signal_in_noise(noise_result, model, target):
    """On driftless walks the rule's assumptions are true: a model may match it
    but must not beat it beyond noise, and must not lose badly either."""
    g = noise_result[(noise_result["model"] == model) & (noise_result["target"] == target)]
    assert (g["skill_vs_rule"] < 0.03).all(), g
    assert (g["skill_vs_rule"] > -0.05).all(), g


def test_compare_shape_and_vol_rows(noise_result):
    assert set(noise_result["model"]) == set(Mo.MODELS)
    assert len(noise_result) == 2 * 5 * len(Mo.MODELS)
    vol = noise_result[noise_result["target"] == "fwd_rvol"]
    assert vol["auc"].isna().all() and (vol["skill_vs_base"] > 0.2).all()
