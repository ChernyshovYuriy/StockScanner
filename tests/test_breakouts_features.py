"""What successful breakouts share: hand-checked features, no look-ahead, planted and absent signals. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.breakouts import backtest as bt
from research.breakouts import features as fx

CAL = pd.bdate_range("2012-01-02", periods=900)


def walk(seed, n=900, p0=50.0):
    g = np.random.default_rng(seed)
    c = p0 * np.exp(np.cumsum(g.normal(0.0005, 0.015, n)))
    o = c * np.exp(g.normal(0, 0.004, n))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * (1 + g.uniform(0, .01, n)), "Low": np.minimum(o, c) * (1 - g.uniform(0, .01, n)),
                         "Close": c, "Volume": g.integers(500_000, 2_000_000, n).astype(float)}, index=CAL[:n])


def market(n=12, end=900):
    return bt.Market({"XIU.TO": walk(99).iloc[:end], **{f"S{k}.TO": walk(k).iloc[:end] for k in range(n)}})


def event(m, t="S1.TO", k=600):
    d = m.bars[t]
    return pd.DataFrame([{"ticker": t, "date": d.index[k], "level": d["Close"].iloc[k] / 1.04, "close": d["Close"].iloc[k], "bars": 80, "tests": 3,
                          "low": d["Low"].iloc[k - 80:k].min(), "target": d["Close"].iloc[k] * 1.1, "lfd_low": d["Low"].iloc[k - 2]}])


def masked(m):
    g = np.random.default_rng(5)
    return pd.DataFrame(g.random(m.close.shape) < 0.01, index=m.close.index, columns=m.close.columns)


EARN = pd.DataFrame({"ticker": "S1.TO", "report_ts": CAL[[300, 360, 420, 480, 540, 599, 615]] + pd.Timedelta(hours=8), "surprise_pct": [1., 2, 3, 4, 5, 6, 7]})
SECT = {f"S{k}.TO": "Energy" if k < 6 else "Materials" for k in range(11)}


def test_every_feature_is_built_and_named():
    m = market()
    f = fx.build(event(m), masked(m), m, SECT, EARN)
    assert list(f.columns) == list(fx.FEATURES)
    assert f.drop(columns=["cross_days", "rising_lows"]).notna().all(axis=None)        # these two need a real level


def test_nothing_after_the_breakout_day_is_read():
    m = market()
    ev, mk = event(m), masked(m)
    short = market(end=601)
    earn = EARN[EARN["report_ts"] <= CAL[600]]
    a, b = fx.build(ev, mk, m, SECT, EARN), fx.build(ev, mk.iloc[:601], short, SECT, earn)
    drop = ["earn_soon"]                                  # a scheduled date is known ahead: the one exception
    pd.testing.assert_frame_equal(a.drop(columns=drop), b.drop(columns=drop))


def test_future_bars_changed_to_garbage_change_nothing():
    m = market()
    data = {t: d.copy() for t, d in m.bars.items()}
    for d in data.values():
        d.iloc[601:] = d.iloc[601:] * 7
    x = walk(99); x.iloc[601:] *= 3
    n = bt.Market({"XIU.TO": x, **data})
    pd.testing.assert_frame_equal(fx.build(event(m), masked(m), m, SECT, EARN), fx.build(event(m), masked(m), n, SECT, EARN))


def hand():
    n = 300
    c = np.full(n, 100.0)
    d = pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1000.0}, index=CAL[:n]).copy()
    d.iloc[290] = [103, 108, 102, 106, 3000]
    d.iloc[291] = [1, 1, 1, 1, 1]                         # must not be read
    return d


def test_hand_checked_values():
    f = fx.stock_features(hand(), 290, 102.0, 200, 95.0, 118.0, 99.0)
    want = {"bars": 90, "depth": 1 - 95 / 102, "range10": 2 / 102, "squeeze": 1.0, "near_line": 1.0, "right_low": 99 / 102 - 1, "rising_lows": 0.0,
            "run_in": 0.0, "cross_days": 1, "prior_up": 0.0, "day_ret": 0.06, "day_atr": 3.0, "gap": 0.03, "ext": 106 / 102 - 1, "close_pos": 4 / 6,
            "vol_x": 3.0, "vol_dry": 1.0, "up_vol": 3000 / (49 * 1000 + 3000), "stop_dist": 1 - 99 / 106, "target_dist": 118 / 106 - 1, "target_atr": 6.0,
            "ret21": 0.06, "ret63": 0.06, "ret126": 0.06, "ret252": 0.06, "dist50": 106 / 100.12 - 1, "dist200": 106 / 100.03 - 1,
            "ma_gap": 100.12 / 100.03 - 1, "slope200": 0.0003, "overhead": 106 / 101 - 1, "off_low": 106 / 99 - 1, "atr": 0.02, "price": 106, "news_day": 1.0}
    assert set(want) == set(f)
    for k, v in want.items():
        assert f[k] == pytest.approx(v), k


def test_news_day_needs_both_gap_and_volume():
    d = hand()
    d.loc[d.index[290], "Volume"] = 1900
    assert fx.stock_features(d, 290, 102.0, 200, 95.0, 118.0, 99.0)["news_day"] == 0
    d = hand()
    d.loc[d.index[290], "Open"] = 101.9
    assert fx.stock_features(d, 290, 102.0, 200, 95.0, 118.0, 99.0)["news_day"] == 0


def test_rising_lows_and_cross_days():
    d = hand()
    d.loc[d.index[210], "Low"] = 90
    d.loc[d.index[260], "Low"] = 94
    d.loc[d.index[288:290], ["Close", "High"]] = 103
    f = fx.stock_features(d, 290, 102.0, 200, 90.0, 118.0, 99.0)
    assert f["rising_lows"] == pytest.approx(4 / 12) and f["cross_days"] == 3


def test_short_history_gives_missing_not_wrong():
    f = fx.stock_features(hand().iloc[190:], 100, 102.0, 20, 95.0, 118.0, 99.0)
    for k in ("ret126", "ret252", "dist200", "overhead", "off_low", "prior_up", "slope200"):
        assert np.isnan(f[k]), k


@pytest.mark.parametrize("at,hour,recent,soon", [(600, 8, 1, 0), (600, 17, 0, 1), (599, 17, 1, 0), (598, 8, 1, 0), (597, 8, 0, 0),
                                                 (601, 8, 0, 1), (621, 8, 0, 1), (622, 8, 0, 0)])
def test_earnings_windows(at, hour, recent, soon):
    dates = np.array([CAL[100], CAL[at] + pd.Timedelta(hours=hour)], dtype="datetime64[ns]")
    f = fx.earnings_features(dates, np.array([1.0, 9.0]), CAL, CAL[600])
    assert (f["earn_recent"], f["earn_soon"]) == (recent, soon)


def test_surprise_is_the_latest_published_and_not_stale():
    f = lambda at: fx.earnings_features(np.array([CAL[100], CAL[at]], dtype="datetime64[ns]"), np.array([1.0, 9.0]), CAL, CAL[600])["surprise"]
    assert f(537) == 9 and np.isnan(f(536)) and np.isnan(f(601))


def test_no_coverage_is_missing():
    assert np.isnan(fx.earnings_features(np.array([]), np.array([]), CAL, CAL[600])["earn_recent"])
    assert np.isnan(fx.earnings_features(np.array([CAL[560]]), np.array([1.]), CAL, CAL[600])["earn_recent"])


def test_panel_values():
    m = market()
    ev, mk = event(m), masked(m)
    f = fx.build(ev, mk, m, SECT, EARN).iloc[0]
    k, day = 600, CAL[600]
    c = pd.DataFrame({t: d["Close"] for t, d in m.bars.items()})
    r = c.iloc[k] / c.iloc[k - 63] - 1
    assert f["rs63"] == pytest.approx((r.rank() / len(r))["S1.TO"])
    en = [t for t, s in SECT.items() if s == "Energy"]
    assert f["sec63"] == pytest.approx(r[en].mean() - r.mean())
    assert f["is_Energy"] == 1 and f["is_Materials"] == 0
    seen = mk.iloc[k - 19:k + 1].any()
    assert f["cluster"] == pytest.approx(seen.mean()) and f["sec_cluster"] == pytest.approx(seen[en].mean())
    assert f["breadth"] == pytest.approx((c.iloc[k] > c.iloc[k - 199:k + 1].mean()).mean())
    assert f["highs"] == pytest.approx((c.iloc[k] > c.iloc[k - 52:k].max()).mean())
    x = m.xiu_close
    assert f["xiu200"] == pytest.approx(x.iloc[k] / x.iloc[k - 199:k + 1].mean() - 1) and f["xiu21"] == pytest.approx(x.iloc[k] / x.iloc[k - 21] - 1)
    assert f["tests"] == 3 and f["earn_recent"] == 1 and f["earn_soon"] == 1 and f["surprise"] == 6


def test_unknown_sector_is_missing_not_zero():
    m = market()
    f = fx.build(event(m, "S11.TO"), masked(m), m, SECT, EARN).iloc[0]
    assert f[[c for c in fx.FEATURES if c.startswith(("is_", "sec"))]].isna().all()


def test_past_breakouts_counts_separate_earlier_ones_only():
    m = market()
    mk = pd.DataFrame(False, index=m.close.index, columns=m.close.columns)
    mk.loc[CAL[[400, 401, 500, 600]], "S1.TO"] = True       # 400-401 are one; 600 is the event itself
    mk.loc[CAL[300], "S1.TO"] = True                         # over a year ago
    assert fx.build(event(m), mk, m, SECT, EARN)["past_breakouts"].iloc[0] == 2


# ── Judging ──────────────────────────────────────────────────────────────────

def study(n=1600, planted=0.0, seed=1):
    g = np.random.default_rng(seed)
    dates = pd.Series(np.sort(g.choice(pd.bdate_range("2012-11-01", "2023-06-30"), n)))
    f = pd.DataFrame(g.normal(size=(n, len(fx.FEATURES))), columns=list(fx.FEATURES))
    for c in fx.FLAGS:
        f[c] = (f[c] > 0.5).astype(float)
    f["atr"] = 0.03 * np.exp(0.2 * f["atr"])
    r = 0.08 * g.standard_t(4, n) * f["atr"] / 0.03 + planted * f["vol_x"] * 0.1
    ev = pd.DataFrame({"ticker": "X", "date": dates, "r20": r, "a20": 0.005})
    return ev, f


def test_planted_feature_is_found_in_both_periods_and_by_both_combinations():
    ev, f = study(planted=0.4)
    t, comb, path, info = fx.analyse(ev, f)
    assert t.loc["vol_x", "counts"] and t.loc["vol_x", "gained_b"] > t.loc["vol_x", "lost_b"] + 0.1
    assert path[0][0] == "vol_x" and comb.loc[("ridge best fifth", "B"), "lo"] > 0
    assert info["weights"].abs().idxmax() == "vol_x" and info["ridge_ic_b"]["p"] < 0.01


def test_noise_is_rejected():
    hits = [int(fx.analyse(*study(seed=s))[0]["counts"].sum()) for s in range(10, 16)]
    assert sum(hits) <= 2


def test_a_volatility_effect_alone_does_not_count_as_good():
    ev, f = study(seed=3)
    assert not fx.analyse(ev, f)[0].loc["atr", "counts"]


def test_period_b_is_never_used_to_fit():
    ev, f = study(planted=0.4)
    t1, c1, p1, i1 = fx.analyse(ev, f)
    e2 = ev.copy()
    late = e2["date"] > fx.A_END
    e2.loc[late, "r20"] = -e2.loc[late, "r20"]
    g2 = f.copy(); g2.loc[late] *= 3
    t2, c2, p2, i2 = fx.analyse(e2, g2)
    assert p1 == p2 and i1["weights"].equals(i2["weights"]) and (t1["ic_a"].astype(float) == t2["ic_a"].astype(float)).all()
    pd.testing.assert_frame_equal(c1.xs("A", level=1), c2.xs("A", level=1))
    assert (t1["gained_a"].astype(float) == t2["gained_a"].astype(float)).all()
    assert (t1["cut90"].dropna().astype(float) == t2["cut90"].dropna().astype(float)).all()


def test_score_is_excess_in_volatility_units():
    ev = pd.DataFrame({"r20": [0.05], "a20": [0.01]})
    assert fx.score(ev, pd.DataFrame({"atr": [0.02]})).iloc[0] == pytest.approx(2.0)


def test_slope_compares_the_same_average_a_month_apart():
    d = hand()
    d.loc[d.index[:200], ["Open", "High", "Low", "Close"]] = 80.0
    f = fx.stock_features(d, 290, 102.0, 200, 95.0, 118.0, 99.0)
    now, then = (109 * 80 + 90 * 100 + 106) / 200, (130 * 80 + 70 * 100) / 200
    assert f["slope200"] == pytest.approx(now / then - 1)


def test_rank_correlation_matches_scipy():
    from scipy.stats import spearmanr
    g = np.random.default_rng(0)
    x, y = pd.Series(g.normal(size=500)), pd.Series(g.normal(size=500))
    d = pd.Series(pd.bdate_range("2015-01-01", periods=500))
    t = fx.test_feature(x, y, d)
    assert t["ic"] == pytest.approx(spearmanr(x, y).statistic)
    assert 0.3 < t["p"] / spearmanr(x, y).pvalue < 3 or t["p"] > 0.3


def test_benjamini_hochberg():
    p = pd.Series([0.001, 0.03, 0.04, 0.2, np.nan])
    assert fx.benjamini_hochberg(p, 0.10).tolist() == [True, True, True, False, False]
    assert not fx.benjamini_hochberg(pd.Series([0.2, 0.9]), 0.10).any()


def test_best_third_uses_periods_a_cut_point():
    a = pd.Series(np.arange(1, 10, dtype=float))
    x = pd.Series([6.0, 7.0, 100.0, np.nan])
    assert fx.best_third(x, a, 1, False).tolist()[:3] == [False, True, True] and fx.best_third(x, a, 1, False).isna().iloc[3]
    assert fx.best_third(pd.Series([3.0, 4.0]), a, -1, False).tolist() == [True, False]
    assert fx.best_third(pd.Series([1.0, 0.0]), a, -1, True).tolist() == [False, True]


def test_tree_and_paths():
    x = np.column_stack([np.linspace(-.5, .5, 400), np.tile([0.0, 1.0], 200)])
    y = (x[:, 0] > 0).astype(float) + 0.2 * x[:, 1]
    tr = fx.tree(x, y, 2, 60)
    v, n, path = max(fx.leaves(tr))
    assert path[0][0] == 0 and path[0][1] == ">" and abs(path[0][2]) < 0.01 and path[1][0] == 1
    assert fx.follow(x, path).sum() == n and v == pytest.approx(1.2, abs=0.01)
    assert sum(q[1] for q in fx.leaves(tr)) == 400


def test_small_leaves_are_not_made():
    g = np.random.default_rng(0)
    assert all(q[1] >= 60 for q in fx.leaves(fx.tree(g.normal(size=(500, 4)), g.normal(size=500))))


def test_ranked_is_against_a_and_missing_is_neutral():
    a = pd.DataFrame({"z": np.arange(10.0)})
    r = fx.ranked(pd.DataFrame({"z": [-5.0, 4.5, 99.0, np.nan]}), a)[:, 0]
    assert r.tolist() == [-0.5, 0.0, 0.5, 0.0]


def test_group_numbers():
    ev = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=4), "r20": [0.2, 0.05, -0.05, 0.0], "a20": 0.01})
    g = fx.group(ev, np.ones(4, bool))
    assert (g["gained"], g["lost"], g["big"], g["up"]) == (0.5, 0.25, 0.25, 0.5)
    assert g["net"] == pytest.approx(0.045) and g["ex"] == pytest.approx(0.04)


def test_trend_panels_equal_the_event_features():
    m = market()
    f, p = fx.build(event(m), masked(m), m, SECT, EARN).iloc[0], fx.trend_panels(m)
    for c in p:
        assert p[c].at[CAL[600], "S1.TO"] == pytest.approx(f[c]), c


def test_control_is_same_day_members_without_a_recent_breakout():
    m = market()
    ev, mk = event(m), pd.DataFrame(False, index=m.close.index, columns=m.close.columns)
    mk.loc[CAL[600], "S1.TO"] = mk.loc[CAL[581], "S2.TO"] = mk.loc[CAL[580], "S3.TO"] = True
    member = pd.DataFrame(True, index=m.close.index, columns=m.close.columns)
    member["S4.TO"] = False
    r = [m.bars[f"S{k}.TO"]["Close"].iloc[621] / m.bars[f"S{k}.TO"]["Open"].iloc[601] - 1 for k in (0, 3, 5, 6, 7, 8, 9, 10, 11)]
    assert fx.control(ev, m, mk, member)[0] == pytest.approx(np.mean(r))
    late = ev.assign(date=CAL[880])
    assert np.isnan(fx.control(late, m, mk, member)[0])
    assert np.isfinite(fx.control(ev.assign(date=CAL[878]), m, mk, member)[0])


def test_rules_table_takes_its_cut_from_period_a_only():
    m = market()
    g = np.random.default_rng(0)
    n = 300
    days = pd.Series(np.sort(g.choice(CAL[300:870], n)))
    ev = pd.DataFrame({"ticker": "S1.TO", "date": days, "r20": g.normal(0, .05, n), "a20": 0.0, "x20": 0.0})
    f = pd.DataFrame({c: g.normal(size=n) for c in ("slope200", "ma_gap", "rs252")}).assign(atr=0.02)
    ev["r20"] += 0.03 * f["slope200"]
    cut = fx.A_END
    fx.A_END = str(CAL[600].date())
    try:
        a = (ev["date"] <= fx.A_END).to_numpy()
        per = {"A": a, "C": ~a}
        t1 = fx.rules_table(ev, f, m, masked(m), ["slope200", "ma_gap"], per)
        f2 = f.copy(); f2.loc[~a] += 50
        t2 = fx.rules_table(ev, f2, m, masked(m), ["slope200", "ma_gap"], per)
    finally:
        fx.A_END = cut
    assert t1.loc[("slope200", "A"), "n"] == pytest.approx(a.sum() / 3, abs=1) and t1.loc[("slope200", "A"), "net"] == t2.loc[("slope200", "A"), "net"]
    assert t2.loc[("slope200", "C"), "n"] == (~a).sum() and t1.loc[("every breakout", "C"), "n"] == (~a).sum()
    assert t1.loc[("all of them", "A"), "n"] < t1.loc[("slope200", "A"), "n"]
    got = {}
    real = fx.control
    for k, g2 in (("1", f * 0.03), ("2", f2 * 0.03)):                       # the control's range may not depend on later events either
        seen = []
        fx.control, fx.A_END = lambda *x: (seen.append(x[3]), real(*x))[1], str(CAL[600].date())
        try:
            fx.rules_table(ev, g2, m, masked(m), ["slope200"], {"A": a})
        finally:
            fx.control, fx.A_END = real, cut
        got[k] = seen[0]
    pd.testing.assert_frame_equal(got["1"], got["2"])
