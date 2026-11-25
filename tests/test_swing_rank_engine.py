"""Swing rank -- portfolio engine. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.swing_rank import engine as en
from research.swing_rank.engine import (
    CASH, Panel, bootstrap_mean, build_panel, cagr, equal_weights, evaluate, max_drawdown, month_end_sessions,
    period_returns, random_portfolios, run, select, sharpe, yearly_excess,
)

CAL = pd.bdate_range("2020-01-01", periods=300)


def frame(close, open_=None, volume=1000.0, index=CAL):
    c = np.asarray(close, float)
    return pd.DataFrame({"Open": c if open_ is None else open_, "High": c, "Low": c, "Close": c,
                         "Volume": np.broadcast_to(np.asarray(volume, float), c.shape)}, index=index[:len(c)])


# ── panel ────────────────────────────────────────────────────────────────────

def test_panel_execution_is_the_open():
    p = build_panel({"A": frame(np.arange(300) + 10.0, np.arange(300) + 9.5)}, CAL)
    assert p.execution["A"].iloc[7] == 16.5 and p.close["A"].iloc[7] == 17.0


def test_panel_untraded_session_uses_previous_close():
    d = frame(np.arange(300) + 10.0, np.arange(300) + 9.5)
    d.iloc[50, 4] = 0
    p = build_panel({"A": d}, CAL)
    assert p.close["A"].iloc[50] == 59.0 and p.execution["A"].iloc[50] == 59.0
    assert p.execution["A"].iloc[51] == 60.5


def test_panel_two_untraded_sessions_in_a_row():
    d = frame(np.arange(300) + 10.0, np.arange(300) + 9.5)
    d.iloc[50:52, 4] = 0
    assert build_panel({"A": d}, CAL).execution["A"].iloc[51] == 59.0


def test_panel_missing_open_falls_back_to_that_close():
    d = frame(np.arange(300) + 10.0, np.arange(300) + 9.5)
    d.iloc[50, 0] = np.nan
    assert build_panel({"A": d}, CAL).execution["A"].iloc[50] == 60.0


def test_panel_name_disappears_after_five_sessions():
    p = build_panel({"A": frame(np.full(100, 5.0))}, CAL)
    assert p.close["A"].iloc[104] == 5.0 and np.isnan(p.close["A"].iloc[105]) and np.isnan(p.execution["A"].iloc[105])


def test_panel_dollar_volume_counts_dead_days_as_zero():
    d = frame(np.full(300, 2.0), volume=1000.0)
    d.iloc[90:100, 4] = 0
    p = build_panel({"A": d}, CAL)
    assert p.dollar_vol["A"].iloc[99] == 1000.0 and p.dollar_vol["A"].iloc[89] == 2000.0


# ── periods ──────────────────────────────────────────────────────────────────

def test_month_end_sessions():
    cal = pd.bdate_range("2020-01-01", "2020-04-15")
    assert [cal[s].strftime("%m%d") for s in month_end_sessions(cal)] == ["0131", "0228", "0331"]
    cal = pd.bdate_range("2020-01-01", "2020-03-31")                     # no session after the last one yet
    assert [cal[s].strftime("%m%d") for s in month_end_sessions(cal)] == ["0131", "0228"]


def test_period_returns_run_from_next_open_to_next_open():
    o = np.arange(300) + 100.0
    p = build_panel({"A": frame(o + 0.5, o)}, CAL)
    r = period_returns(p, [10, 30, 55])
    assert r.index.tolist() == [10, 30]
    assert r.loc[10, "A"] == pytest.approx(131 / 111 - 1) and r.loc[30, "A"] == pytest.approx(156 / 131 - 1)


def test_period_return_of_a_dead_name_is_its_last_price():
    c = np.full(40, 10.0); c[39] = 4.0
    p = build_panel({"A": frame(c), "B": frame(np.full(300, 1.0))}, CAL)
    assert period_returns(p, [20, 60, 90]).loc[20, "A"] == pytest.approx(-0.6)
    assert np.isnan(period_returns(p, [20, 60, 90]).loc[60, "A"])


# ── selection ────────────────────────────────────────────────────────────────

S = pd.Series({f"T{k:02d}": 100.0 - k for k in range(30)})


def test_select_top_n():
    assert select(S, set()) == [f"T{k:02d}" for k in range(10)]


def test_keep_band():
    got = select(S, {"T12", "T14", "T15", "T20"})
    assert set(got) == {"T12", "T14"} | {f"T{k:02d}" for k in range(8)}
    assert "T15" not in got                                               # 16th: out


def test_kept_name_is_not_listed_twice():
    got = select(S, {"T03", "T14"})
    assert len(got) == len(set(got)) == 10


def test_fallen_holding_can_still_be_bought_if_nothing_better():
    assert select(S[:4], {"T03"}, n=10, keep=2) == ["T00", "T01", "T02", "T03"]


def test_nan_scores_are_never_held():
    s = S.copy(); s["T00"] = np.nan
    assert "T00" not in select(s, {"T00"})
    assert select(pd.Series({"A": np.nan, "B": 1.0}), {"A"}) == ["B"]


def test_ties_are_alphabetical():
    assert select(pd.Series({"B": 1.0, "A": 1.0, "C": 1.0}), set(), n=2) == ["A", "B"]


def test_sector_cap_applies_to_kept_and_new():
    sec = {t: ("X" if int(t[1:]) < 6 else f"S{t}") for t in S.index}
    got = select(S, {"T05", "T04", "T03"}, sector=sec)
    assert [t for t in got if sec[t] == "X"] == ["T03", "T04"]
    assert len(got) == 10


# ── running ──────────────────────────────────────────────────────────────────

def test_equal_weights_leave_cash():
    w = equal_weights(["A", "B"], 10)
    assert w["A"] == 0.1 and w[CASH] == pytest.approx(0.8)
    assert CASH not in equal_weights(list("abcdefghij"), 10)


def test_run_returns_costs_and_drift_by_hand():
    R = pd.DataFrame({"A": [0.10, 0.20], "B": [-0.10, 0.0], CASH: [0.01, 0.01]}, index=[1, 2])
    out = run(lambda s, held: {"A": 0.5, "B": 0.25, CASH: 0.25}, R, 0.01)
    v1 = 0.5 * 1.1 + 0.25 * 0.9 + 0.25 * 1.01
    assert out.loc[1, "gross"] == pytest.approx(v1 - 1) and out.loc[1, "turnover"] == pytest.approx(0.75)
    assert out.loc[1, "net"] == pytest.approx(v1 * (1 - 0.0075) - 1)
    t2 = abs(0.5 - 0.55 / v1) + abs(0.25 - 0.225 / v1)
    assert out.loc[2, "turnover"] == pytest.approx(t2) and out.loc[2, "gross"] == pytest.approx(0.1 + 0.0025)
    assert out["names"].tolist() == [2, 2]


def test_run_sells_are_charged():
    R = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]}, index=[1, 2])
    out = run(lambda s, held: {"A": 1.0} if s == 1 else {"B": 1.0}, R, 0.01)
    assert out["turnover"].tolist() == [1.0, 2.0] and out.loc[2, "net"] == pytest.approx(-0.02)


def test_run_passes_what_was_held():
    seen = []
    R = pd.DataFrame({"A": [0.0] * 3, "B": [0.0] * 3}, index=[1, 2, 3])
    run(lambda s, held: (seen.append(set(held)), {"A": 0.5, CASH: 0.5})[1], R, 0.0)
    assert seen == [set(), {"A"}, {"A"}]


def test_run_missing_return_is_zero():
    R = pd.DataFrame({"A": [np.nan]}, index=[1])
    assert run(lambda s, h: {"A": 0.5, "ZZ": 0.5}, R, 0.0).loc[1, "gross"] == 0.0


@pytest.mark.parametrize("w", [{"A": 0.9}, {"A": 1.2, "B": -0.2}])
def test_run_rejects_bad_weights(w):
    with pytest.raises(ValueError):
        run(lambda s, h: w, pd.DataFrame({"A": [0.0], "B": [0.0]}, index=[1]), 0.0)


# ── statistics ───────────────────────────────────────────────────────────────

def test_cagr_drawdown_sharpe_by_hand():
    assert cagr(pd.Series([0.01] * 24)) == pytest.approx(1.01 ** 12 - 1)
    assert max_drawdown(pd.Series([0.1, -0.2, 0.1, -0.1])) == pytest.approx(0.8 * 1.1 * 0.9 - 1)
    assert max_drawdown(pd.Series([-0.3, 0.5])) == pytest.approx(-0.3)
    r = pd.Series([0.01, 0.03, -0.02, 0.04])
    assert sharpe(r) == pytest.approx(r.mean() / r.std() * 12 ** 0.5)


def test_yearly_excess():
    r, b = pd.Series([0.1, 0.1, 0.0]), pd.Series([0.0, 0.05, 0.02])
    y = yearly_excess(r, b, np.array([2020, 2020, 2021]))
    assert y.to_dict() == pytest.approx({2020: 1.21 - 1.05, 2021: -0.02})


def test_bootstrap_interval_coverage():
    rng = np.random.default_rng(1)
    hits = 0
    for k in range(200):
        e = rng.normal(0, 0.03, 204)
        x = 0.004 + e
        lo, hi = bootstrap_mean(x, n_boot=400, seed=k)
        hits += lo <= 0.048 <= hi
    assert 0.88 <= hits / 200 <= 0.985


def test_bootstrap_uses_blocks():
    x = np.repeat(np.tile([0.05, -0.05], 20), 6)                         # runs of six months
    wide = np.subtract(*bootstrap_mean(x, block=6)[::-1])
    narrow = np.subtract(*bootstrap_mean(x, block=1)[::-1])
    assert wide > 1.3 * narrow


def test_bootstrap_level():
    x = np.random.default_rng(0).normal(0, 0.02, 120)
    a, b = bootstrap_mean(x), bootstrap_mean(x, level=0.995)
    assert b[0] < a[0] < a[1] < b[1]


def test_evaluate_conditions():
    years = np.repeat(np.arange(2010, 2020), 12)
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.007, 0.035, 120))
    good = evaluate(bench + 0.01, bench, years)
    assert good["c1"] and good["c3"] and good["c4"] and good["excess"] == pytest.approx(0.12)
    bad = evaluate(bench - 0.01, bench, years)
    assert not bad["c1"] and not bad["c3"] and bad["years_ahead"] == 0.0
    noisy = evaluate(bench + pd.Series(rng.normal(0.0005, 0.05, 120)), bench, years)
    assert not noisy["c1"]
    lucky = bench + pd.Series(rng.normal(0, 0.05, 120))
    lucky += 0.002 - (lucky - bench).mean()              # positive on average, but inside the noise
    v = evaluate(lucky, bench, years)
    assert v["excess"] == pytest.approx(0.024) and v["lo"] < 0 and not v["c1"]
    lev = evaluate(bench * 2, bench, years)
    assert not lev["c4"]


def test_years_condition_is_sixty_percent():
    years = np.repeat(np.arange(2010, 2020), 12)
    bench = pd.Series(np.zeros(120))
    for wins, want in ((6, True), (5, False)):
        net = pd.Series(np.where(np.repeat(np.arange(10), 12) < wins, 0.01, -0.001))
        assert evaluate(net, bench, years)["c3"] is want


def test_drawdown_condition_is_125_percent():
    years = np.zeros(4, int)
    bench = pd.Series([0.0, -0.20, 0.0, 0.0])
    assert evaluate(pd.Series([0.0, -0.25, 0.0, 0.1]), bench, years)["c4"]
    assert not evaluate(pd.Series([0.0, -0.2501, 0.0, 0.1]), bench, years)["c4"]


# ── random portfolios and an end-to-end sanity check ─────────────────────────

def test_random_portfolios_centre_on_the_universe_and_vary():
    rng = np.random.default_rng(2)
    names = [f"N{k:02d}" for k in range(40)]
    R = pd.DataFrame(rng.normal(0.01, 0.05, (60, 40)), columns=names, index=range(60))
    out = random_portfolios({s: names for s in range(60)}, R, n_random=60)
    ew = cagr(R.mean(axis=1))
    assert len(set(out)) == 60 and abs(out.mean() - ew) < 0.02 and out.std() > 0.002


def test_random_portfolios_respect_eligibility_and_sector():
    R = pd.DataFrame({"A": [0.1] * 6, "B": [0.1] * 6, "C": [0.1] * 6, "BAD": [-0.9] * 6}, index=range(6))
    out = random_portfolios({s: ["A", "B", "C"] for s in range(6)}, R, n=2, keep=2, n_random=5)
    assert out == pytest.approx([1.1 ** 12 - 1] * 5)
    out = random_portfolios({s: ["A", "B", "C"] for s in range(6)}, R, n=3, keep=3, n_random=3,
                            sector_of=lambda s: {"A": "x", "B": "x", "C": "x"})
    assert out == pytest.approx([(1 + 0.1 * 2 / 3) ** 12 - 1] * 3)


def test_a_perfect_score_beats_every_random_portfolio():
    rng = np.random.default_rng(3)
    names = [f"N{k:02d}" for k in range(40)]
    R = pd.DataFrame(rng.normal(0.005, 0.05, (60, 40)), columns=names, index=range(60))
    best = run(lambda s, h: equal_weights(select(R.loc[s], h, keep=10)), R, 0.0)["gross"]
    assert cagr(best) > random_portfolios({s: names for s in range(60)}, R, n_random=40).max() + 0.5
