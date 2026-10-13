"""
Barometer Phases 3-4 (research/barometer/gate.py): each gate rule on
hand-built tables, the spread/regime/calibration helpers on hand values,
row-level losses consistent with models.compare(), and the full gate on two
synthetic worlds whose verdict is known in advance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.barometer import gate as G
from research.barometer import models as Mo
from research.barometer.dataset import build_panel

YEARS = [2023, 2024, 2025, 2026]


def _cmp(target, skill_base, skill_rule, auc=None, calib=None, model="logistic"):
    n = len(skill_base)
    return pd.DataFrame({"year": YEARS[:n], "target": target, "model": model,
                         "skill_vs_base": skill_base, "skill_vs_rule": skill_rule,
                         "auc": auc if auc is not None else [np.nan] * n,
                         "calib": calib if calib is not None else [np.nan] * n})


def _full_cmp(**over):
    good = dict(skill_base=[.05] * 4, skill_rule=[.02] * 4, auc=[.7] * 4, calib=[.1] * 4)
    parts = []
    for t in ["y_stop_hit", "y_up10", "y_dn10", "y_beat_mkt"]:
        parts.append(_cmp(t, **over.get(t, good)))
    parts.append(_cmp("fwd_rvol", [.5] * 4, over.get("fwd_rvol", [.3] * 4)))
    return pd.concat(parts, ignore_index=True)


def _verdict(res, target):
    return bool(res.loc[res["target"] == target, "passed"].iloc[0])


GOOD_SPREAD = pd.Series([0.02] * 4, index=YEARS)


def test_gate_passes_when_every_criterion_holds_3_of_4():
    good = dict(skill_base=[.05, .05, .05, -.01], skill_rule=[.02, .02, -.01, .02],
                auc=[.7, .7, .55, .7], calib=[.1, .3, .1, .1])
    res = G.evaluate_gate(_full_cmp(y_stop_hit=good), GOOD_SPREAD)
    assert _verdict(res, "y_stop_hit")
    assert set(res["gate"]) == {"T1", "T2", "T3", "T4"} and len(res) == 5


@pytest.mark.parametrize("field, values", [
    ("skill_base", [.05, .05, -.01, -.01]),   # beats base only 2/4
    ("skill_rule", [.02, -.01, -.01, .02]),   # beats rule only 2/4
    ("auc", [.7, .59, .59, .7]),              # AUC 0.59 is below 0.60
    ("calib", [.1, .21, .25, .1]),            # calibration off by >20% in 2 years
])
def test_gate_fails_on_any_single_criterion(field, values):
    spec = dict(skill_base=[.05] * 4, skill_rule=[.02] * 4, auc=[.7] * 4, calib=[.1] * 4)
    spec[field] = values
    res = G.evaluate_gate(_full_cmp(y_stop_hit=spec), GOOD_SPREAD)
    assert not _verdict(res, "y_stop_hit")
    assert _verdict(res, "y_up10")           # other targets untouched


def test_skill_of_exactly_zero_is_not_a_win():
    spec = dict(skill_base=[.05] * 4, skill_rule=[0.0, 0.0, .02, .02], auc=[.7] * 4, calib=[.1] * 4)
    assert not _verdict(G.evaluate_gate(_full_cmp(y_up10=spec), GOOD_SPREAD), "y_up10")


def test_t2_labels_are_judged_separately():
    bad = dict(skill_base=[.05] * 4, skill_rule=[-.01] * 4, auc=[.7] * 4, calib=[.1] * 4)
    res = G.evaluate_gate(_full_cmp(y_dn10=bad), GOOD_SPREAD)
    assert _verdict(res, "y_up10") and not _verdict(res, "y_dn10")


def test_t3_needs_only_to_beat_the_rule():
    assert _verdict(G.evaluate_gate(_full_cmp(fwd_rvol=[.3, .3, .3, -.1]), GOOD_SPREAD), "fwd_rvol")
    assert not _verdict(G.evaluate_gate(_full_cmp(fwd_rvol=[.3, -.1, .3, -.1]), GOOD_SPREAD), "fwd_rvol")


def test_t4_uses_spread_above_costs():
    assert _verdict(G.evaluate_gate(_full_cmp(), GOOD_SPREAD), "y_beat_mkt")
    thin = pd.Series([0.02, 0.009, 0.01, 0.03], index=YEARS)     # 1% is not "above" 1%
    res = G.evaluate_gate(_full_cmp(), thin)
    assert not _verdict(res, "y_beat_mkt")
    assert res.loc[res["target"] == "y_beat_mkt", "years_spread_ok"].iloc[0] == 2


def test_gate_fails_with_too_few_test_years():
    two = pd.concat([_cmp(t, [.05] * 2, [.02] * 2, [.7] * 2, [.1] * 2)
                     for t in ["y_stop_hit", "y_up10", "y_dn10", "y_beat_mkt"]]
                    + [_cmp("fwd_rvol", [.5] * 2, [.3] * 2)], ignore_index=True)
    assert not G.evaluate_gate(two, GOOD_SPREAD.iloc[:2])["passed"].any()


def test_gate_ignores_other_models():
    """Logistic fails stop-hit, a GBM row set passes it: the verdict must be
    logistic's alone (mixing the rows in would give 4 passing years)."""
    bad = dict(skill_base=[-1] * 4, skill_rule=[-1] * 4, auc=[.5] * 4, calib=[.9] * 4)
    cmp = pd.concat([_full_cmp(y_stop_hit=bad), _full_cmp().assign(model="gbm")])
    res = G.evaluate_gate(cmp, GOOD_SPREAD)
    assert not _verdict(res, "y_stop_hit")
    assert res.loc[res["target"] == "y_stop_hit", "years"].iloc[0] == 4


# ── helpers on hand values ───────────────────────────────────────────────────

def _mi(dates, n):
    return pd.MultiIndex.from_tuples([(pd.Timestamp(d), f"T{i}") for d in dates for i in range(n)],
                                     names=["date", "ticker"])


def test_group_spread_hand_values():
    idx = _mi(["2024-01-02"], 10)
    score = pd.Series(np.arange(10.0), index=idx)
    fwd = pd.Series(np.arange(10.0) * 0.01, index=idx)
    s = G.group_spread(score, fwd, 5)
    assert s.iloc[0] == pytest.approx(0.085 - 0.005)       # top (8,9) minus bottom (0,1)
    assert G.group_spread(-score, fwd, 5).iloc[0] == pytest.approx(-0.08)


def test_group_spread_per_date_and_skips_thin_dates():
    idx = _mi(["2024-01-02", "2024-01-03"], 10).append(_mi(["2024-01-04"], 3))
    score = pd.Series(np.r_[np.arange(10.0), np.arange(10.0), [1, 2, 3]], index=idx)
    fwd = pd.Series(np.r_[np.arange(10.0) * 0.01, np.zeros(10), [0, 0, 1]], index=idx)
    s = G.group_spread(score, fwd, 5)
    assert list(s.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert s.tolist() == pytest.approx([0.08, 0.0])


def test_regime_breakdown_hand_values():
    rows = pd.DataFrame({"regime": [1, 1, 0, 0],
                         "loss_model": [0.1, 0.1, 0.3, 0.3],
                         "loss_base": [0.2, 0.2, 0.2, 0.2],
                         "loss_rule": [0.1, 0.1, 0.15, 0.15]})
    rb = G.regime_breakdown(rows).set_index("regime")
    assert rb.loc["bull", "skill_vs_base"] == pytest.approx(0.5)
    assert rb.loc["bull", "skill_vs_rule"] == pytest.approx(0.0)
    assert rb.loc["bear", "skill_vs_rule"] == pytest.approx(-1.0)
    assert rb["n"].tolist() == [2, 2] or rb["n"].sum() == 4


def test_pooled_calibration_side_by_side():
    rows = pd.DataFrame({"y": [0, 1, 0, 1], "p_model": [.2, .8, .2, .8], "p_rule": [.5] * 4})
    t = G.pooled_calibration(rows, bins=2)
    assert t["model_mean_y"].dropna().tolist() == [0.0, 1.0]
    assert t["rule_n"].dropna().tolist() == [4]


# ── end to end on synthetic worlds ───────────────────────────────────────────

CAL = pd.bdate_range("2019-01-01", "2024-12-31")   # 4 test years (2021-2024) with history before


def _walk(rng, sigma, start, drift=0.0):
    c = start * np.exp(np.cumsum(rng.normal(drift, sigma, len(CAL))))
    return pd.DataFrame({"Open": c, "High": c * (1 + sigma / 2), "Low": c * (1 - sigma / 2),
                         "Close": c, "Volume": 1e6}, index=CAL)


def _world(drift: bool, seed: int):
    rng = np.random.default_rng(seed)
    data = {"XIU.TO": _walk(rng, 0.008, 30.0)}
    for i in range(30):
        if drift:
            data[f"D{i:02d}.TO"] = _walk(rng, 0.010, 1000.0, drift=-0.004 + 0.00027 * i)
        else:
            data[f"N{i:02d}.TO"] = _walk(rng, 0.008 + 0.0008 * i, 20.0)
    return build_panel(data)


def _run_gate(panel, years):
    preds = Mo.walk_forward_predictions(panel, models=["logistic"], test_years=years)
    cmp = Mo.compare(panel, preds, models=["logistic"], test_years=years)
    return G.evaluate_gate(cmp, G.yearly_spreads(panel, preds)), preds, cmp


@pytest.fixture(scope="module")
def drift_gate():
    panel = _world(True, 21)
    return (panel,) + _run_gate(panel, [2021, 2022, 2023, 2024])


@pytest.fixture(scope="module")
def noise_gate():
    panel = _world(False, 22)
    return (panel,) + _run_gate(panel, [2021, 2022, 2023, 2024])


def test_gate_passes_t4_when_direction_is_truly_predictable(drift_gate):
    _, res, _, _ = drift_gate
    row = res[res["target"] == "y_beat_mkt"].iloc[0]
    assert row["passed"] and row["mean_spread"] > 0.03


def test_gate_on_pure_noise_finds_no_direction_and_no_large_edge(noise_gate):
    """Driftless walks with constant per-stock volatility. Nothing ranks
    direction (T4 fails, spread ~0), the +/-10% tails fail. Stop-hit is NOT
    asserted to fail: the rule estimates each stock's (constant) vol from 20
    days only, the model also sees vol_60 and the vol level, so it holds a
    genuinely better estimate of the same truth -- a small real edge (seen:
    +0.5..+1.6% Brier skill vs the rule), not overfitting. What must hold is
    that any such edge stays small."""
    _, res, _, cmp = noise_gate
    for target in ["y_up10", "y_dn10", "y_beat_mkt"]:
        assert not _verdict(res, target), target
    assert abs(res.loc[res["target"] == "y_beat_mkt", "mean_spread"].iloc[0]) < SPREAD_NOISE
    stop = cmp[cmp["target"] == "y_stop_hit"]
    assert (stop["skill_vs_rule"] < 0.03).all()
    assert _verdict(res, "fwd_rvol")       # the better vol estimate is exactly what T3 measures


SPREAD_NOISE = 0.01


def test_row_losses_match_compare(noise_gate):
    panel, _, preds, cmp = noise_gate
    for target in ["y_stop_hit", "fwd_rvol"]:
        rows = G.row_losses(panel, preds, target, test_years=[2021, 2022, 2023, 2024])
        per_year = rows.groupby("year")[["loss_model", "loss_rule", "loss_base"]].mean()
        c = cmp[cmp["target"] == target].set_index("year")
        np.testing.assert_allclose(per_year["loss_model"], c["loss"])
        np.testing.assert_allclose(per_year["loss_rule"], c["loss_rule"])
        np.testing.assert_allclose(per_year["loss_base"], c["loss_base"])
        assert set(rows["regime"].unique()) <= {0.0, 1.0}


def test_significant_wins_are_counted_but_do_not_change_the_verdict():
    cmp = _full_cmp()
    cmp["t_vs_rule"] = -1.0
    stop = cmp["target"] == "y_stop_hit"
    cmp.loc[stop, "t_vs_rule"] = [-3.0, -2.5, -1.0, 0.5]
    res = G.evaluate_gate(cmp, GOOD_SPREAD)
    row = res[res["target"] == "y_stop_hit"].iloc[0]
    assert row["years_beat_rule_signif"] == 2 and row["passed"]
    assert res.loc[res["target"] == "y_up10", "years_beat_rule_signif"].iloc[0] == 0
