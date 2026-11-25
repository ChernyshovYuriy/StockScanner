"""Swing rank Phase 2 -- the book's ETF strategies. Synthetic data only."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research.swing_rank import engine as en
from research.swing_rank import etf
from research.swing_rank.engine import CASH
from research.swing_rank.etf import (
    BONDS, MONEY, XIU, DEFAULT, EtfTest, EtfUniverse, Signals, base_name, is_bond, is_hedged, make_strategies,
    report, sensitivity_configs,
)

N = 1000
CAL = pd.bdate_range("2015-01-01", periods=N)
CFG = replace(DEFAULT, min_universe=5)


def frame(close, volume=100_000.0):
    c = np.asarray(close, float)
    return pd.DataFrame({"Open": c * 0.999, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                         "Volume": np.full(len(c), volume)}, index=CAL[:len(c)])


def walk(seed, drift=0.0003, vol=0.01, start=20.0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(drift, vol, N)))


def market(n=20, seed=0):
    data = {XIU: frame(walk(seed + 1000)), BONDS: frame(walk(seed + 1001, 0.0001, 0.002)),
            MONEY: frame(50 * np.exp(0.00005 * np.arange(N)))}
    names = {XIU: "iShares S&P/TSX 60 Index ETF", BONDS: "iShares Core Canadian Universe Bond Index ETF",
             MONEY: "iShares Premium Money Market ETF"}
    for k in range(n):
        data[f"F{k:02d}.TO"] = frame(walk(seed + k, 0.0002 * (k - n / 2) / n))
        names[f"F{k:02d}.TO"] = f"iShares Fund {k:02d} Index ETF"
    return data, names


def parts(data, names, cfg=CFG):
    panel = en.build_panel(data, CAL)
    return panel, EtfUniverse(panel, names, cfg), Signals(panel, cfg)


# ── names ────────────────────────────────────────────────────────────────────

def test_base_name_joins_fund_versions():
    a = {base_name(x) for x in ("iShares Core S&P 500 Index ETF", "iShares Core S&P 500 Index ETF (CAD-Hedged)",
                                "iShares Core S&P 500 Index ETF - USD Units")}
    assert a == {"iShares Core S&P 500 Index ETF"}


@pytest.mark.parametrize("name,bond", [
    ("iShares Core Canadian Universe Bond Index ETF", True), ("iShares Floating Rate Index ETF", True),
    ("iShares 0-5 Year TIPS Bond Index ETF", True), ("iShares Core Equity ETF Portfolio", True),
    ("iShares Gold Bullion ETF", False), ("iShares Semiconductor Index ETF", False),
    ("iShares S&P/TSX Capped REIT Index ETF", False), ("iShares Premium Money Market ETF", True),
    ("iShares Core MSCI Canadian Quality Dividend Index ETF", False)])
def test_is_bond(name, bond):
    assert is_bond(name) is bond


# ── eligibility ──────────────────────────────────────────────────────────────

def test_eligibility_history_volume_and_class():
    data, names = market(4)
    data["NEW.TO"], names["NEW.TO"] = frame(walk(5)).iloc[300:], "iShares New Index ETF"
    data["THIN.TO"], names["THIN.TO"] = frame(walk(6), 100.0), "iShares Thin Index ETF"
    _, u, _ = parts(data, names)
    got = u.eligible(500)
    assert "NEW.TO" not in got and "THIN.TO" not in got and BONDS not in got and MONEY not in got and XIU in got
    assert "NEW.TO" in u.eligible(572) and "NEW.TO" not in u.eligible(571)        # 273 sessions
    assert BONDS in u.eligible(500, bonds=True) and MONEY not in u.eligible(500, bonds=True)


def test_hedged_twin_only_until_the_unhedged_one_is_eligible():
    data, names = market(3)
    data["H.TO"], names["H.TO"] = frame(walk(7)), "iShares Core S&P 500 Index ETF (CAD-Hedged)"
    data["U.TO"], names["U.TO"] = frame(walk(8)).iloc[300:], "iShares Core S&P 500 Index ETF"
    _, u, _ = parts(data, names)
    assert "H.TO" in u.eligible(500) and "U.TO" not in u.eligible(500)
    assert "H.TO" not in u.eligible(700) and "U.TO" in u.eligible(700)


def test_eligibility_dies_with_the_fund():
    data, names = market(3)
    data["F00.TO"] = data["F00.TO"].iloc[:600]
    _, u, _ = parts(data, names)
    assert "F00.TO" in u.eligible(599) and "F00.TO" not in u.eligible(606)


# ── signals, by hand ─────────────────────────────────────────────────────────

def test_signals_by_hand():
    data, names = market(2)
    _, _, sig = parts(data, names)
    c = data["F00.TO"]["Close"]
    assert sig.momentum["F00.TO"].iat[600] == pytest.approx(c.iloc[600] / c.iloc[348] - 1)
    for s in range(200, N):
        assert bool(sig.above["F00.TO"].iat[s]) == (c.iloc[s] > c.iloc[s - 199:s + 1].mean())
    r = c.pct_change()
    assert sig.vol["F00.TO"].iat[600] == pytest.approx(r.iloc[349:601].std() * 252 ** 0.5)
    x = data[XIU]["Close"].pct_change()
    assert sig.short_vol[XIU].iat[600] == pytest.approx(x.iloc[581:601].std() * 252 ** 0.5)
    past = (x.rolling(20).std() * 252 ** 0.5).iloc[:601]
    assert sig.target_vol[XIU].iat[600] == pytest.approx(past.mean())


def test_signals_never_use_the_future():
    data, names = market(6)
    _, u, sig = parts(data, names)
    wild = {t: d.copy() for t, d in data.items()}
    rng = np.random.default_rng(1)
    for d in wild.values():
        d.iloc[601:, :4] *= np.exp(np.cumsum(rng.normal(0, 0.2, len(d) - 601)))[:, None]
        d.iloc[601:, 4] = 1.0
    _, u2, sig2 = parts(wild, names)
    a, b = make_strategies(u, sig, CFG), make_strategies(u2, sig2, CFG)
    for k in a:
        assert a[k](600, set()) == b[k](600, set()), k
    for f in ("momentum", "above", "vol", "short_vol", "target_vol"):
        pd.testing.assert_frame_equal(getattr(sig, f).iloc[:601], getattr(sig2, f).iloc[:601])


# ── strategies ───────────────────────────────────────────────────────────────

def strategies(n=20, seed=0, cfg=CFG):
    data, names = market(n, seed)
    panel, u, sig = parts(data, names, cfg)
    return data, panel, u, sig, make_strategies(u, sig, cfg)


def test_e1_holds_the_ten_strongest():
    _, _, u, sig, st = strategies()
    w = st["E1 sector rotation"](700, set())
    want = sig.momentum.iloc[700][u.eligible(700)].nlargest(10).index
    assert set(w) == set(want) and all(x == pytest.approx(0.1) for x in w.values())


def test_e1_keep_band():
    _, _, u, sig, st = strategies()
    ranked = sig.momentum.iloc[700][u.eligible(700)].nlargest(16).index.tolist()
    w = st["E1 sector rotation"](700, {ranked[14], ranked[15]})
    assert ranked[14] in w and ranked[15] not in w


def test_e1b_weak_funds_become_cash():
    _, _, u, sig, st = strategies()
    w, plain = st["E1b rotation + MA filter"](700, set()), st["E1 sector rotation"](700, set())
    bad = [t for t in plain if not sig.above[t].iat[700]]
    s = next(s for s in range(300, 1000, 7) if any(not sig.above[t].iat[s] for t in st["E1 sector rotation"](s, set())))
    w, plain = st["E1b rotation + MA filter"](s, set()), st["E1 sector rotation"](s, set())
    bad = [t for t in plain if not sig.above[t].iat[s]]
    assert bad and set(w) == (set(plain) - set(bad)) | {CASH} and w[CASH] == pytest.approx(0.1 * len(bad))


def test_e2_switches_to_bonds_below_the_average():
    _, _, u, sig, st = strategies()
    up = next(s for s in range(300, 1000) if sig.above[XIU].iat[s])
    down = next(s for s in range(300, 1000) if not sig.above[XIU].iat[s])
    assert st["E2 dual momentum"](up, set()) == st["E1 sector rotation"](up, set())
    assert st["E2 dual momentum"](down, set()) == {BONDS: 1.0}


def test_e3_by_hand():
    _, _, u, sig, st = strategies()
    s = 800
    ok = [t for t in u.eligible(s, bonds=True) if sig.momentum[t].iat[s] > 0 and sig.above[t].iat[s]]
    score = pd.Series({t: sig.momentum[t].iat[s] / sig.vol[t].iat[s] for t in ok}).nlargest(10)
    w = st["E3 multi-asset trend"](s, set())
    raw = score / score.sum()
    for t in score.index:
        assert w[t] == pytest.approx(min(raw[t], 0.2))
    assert sum(w.values()) == pytest.approx(1.0) and w.get(CASH, 0) == pytest.approx(1 - raw.clip(upper=0.2).sum())


def test_e3_can_hold_bonds():
    data, names = market(4)
    data[BONDS] = frame(20 * np.exp(0.0004 * np.arange(N) + np.random.default_rng(0).normal(0, 0.001, N)))
    _, u, sig = parts(data, names)
    assert make_strategies(u, sig, CFG)["E3 multi-asset trend"](700, set())[BONDS] == pytest.approx(0.2)


def test_e3_can_be_all_cash():
    data, names = market(4)
    for t in data:
        data[t] = frame(np.linspace(50, 20, N))
    panel, u, sig = parts(data, names)
    assert make_strategies(u, sig, CFG)["E3 multi-asset trend"](700, set()) == {CASH: 1.0}


def test_e4_weight_threshold_and_cap():
    _, _, u, sig, st = strategies()
    f = st["E4 volatility targeting"]
    want = min(1.0, sig.target_vol[XIU].iat[500] / sig.short_vol[XIU].iat[500])
    assert f(500, set()).get(XIU) == pytest.approx(want)
    seen, w = [], want
    for s in range(501, 900, 5):
        raw = min(1.0, sig.target_vol[XIU].iat[s] / sig.short_vol[XIU].iat[s])
        if abs(raw / w - 1) > 0.10:
            w = raw
        got = f(s, {XIU})
        assert got[XIU] == pytest.approx(w) and sum(got.values()) == pytest.approx(1.0) and got[XIU] <= 1
        seen.append(w)
    assert len(set(seen)) < len(seen) and len(set(seen)) > 5


def test_e4_restarts_clean():
    _, _, u, sig, st = strategies()
    f = st["E4 volatility targeting"]
    raw = lambda s: min(1.0, sig.target_vol[XIU].iat[s] / sig.short_vol[XIU].iat[s])
    a, b = next((a, b) for a in range(400, 900) for b in range(a + 1, 900) if 0.02 < abs(raw(b) / raw(a) - 1) < 0.08)
    assert f(a, set())[XIU] == pytest.approx(raw(a))
    assert f(b, {XIU})[XIU] == pytest.approx(raw(a))                  # inside the threshold: unchanged
    assert f(b, set())[XIU] == pytest.approx(raw(b))                  # a fresh start forgets it


# ── the whole test object ────────────────────────────────────────────────────

def test_etf_test_periods_start_when_enough_funds_exist():
    data, names = market(20)
    for k in range(12):
        data[f"F{k:02d}.TO"] = data[f"F{k:02d}.TO"].iloc[200:]
    t = EtfTest(data, names, replace(DEFAULT, min_universe=20))
    first = t.sessions[0]
    assert len(t.eligible(replace(DEFAULT))[first]) >= 20 and first >= 200 + 272
    assert (t.returns.index == t.sessions[:-1]).all()


def test_cash_earns_the_money_market_fund():
    data, names = market(8)
    t = EtfTest(data, names, CFG)
    s0, s1 = t.sessions[0], t.sessions[1]
    o = data[MONEY]["Open"]
    assert t.returns.loc[s0, CASH] == pytest.approx(o.iloc[s1 + 1] / o.iloc[s0 + 1] - 1)


def test_planted_momentum_is_found_and_noise_is_not():
    data, names = market(30)
    rng = np.random.default_rng(0)
    for k in range(30):                                  # persistent drifts: past winners keep winning
        drift = 0.0018 if k < 10 else -0.0004
        data[f"F{k:02d}.TO"] = frame(20 * np.exp(np.cumsum(rng.normal(drift, 0.008, N))))
    t = EtfTest(data, names, CFG)
    r = t.run()["E1 sector rotation"]["net"]
    e = en.evaluate(r, t.bench, t.years)
    assert e["excess"] > 0.10 and e["lo"] > 0
    passes = 0
    for seed in range(5):
        d2, n2 = market(30, seed=100 * seed + 7)
        for k in range(30):
            d2[f"F{k:02d}.TO"] = frame(walk(9000 + 100 * seed + k, 0.0003, 0.01))
        t2 = EtfTest(d2, n2, CFG)
        passes += en.evaluate(t2.run()["E1 sector rotation"]["net"], d2 and t2.returns[[f"F{k:02d}.TO" for k in range(30)]].mean(axis=1), t2.years)["c1"]
    assert passes <= 1


def test_costs_reduce_returns():
    data, names = market(20)
    t = EtfTest(data, names, CFG)
    a, b = t.run(cost=0.0), t.run(cost=0.01)
    for k in a:
        assert en.cagr(b[k]["net"]) < en.cagr(a[k]["net"])
        assert en.cagr(b[k]["gross"]) == pytest.approx(en.cagr(a[k]["gross"]))


def test_sensitivity_configs_change_one_field_each():
    c = sensitivity_configs()
    assert len(c) == 18 and all(v != DEFAULT for v in c.values())
    assert c["formation -20%"].formation == 202 and c["top_n +20%"].top_n == 12 and c["keep -20%"].keep == 12
    assert set(x for v in etf.RELEVANT.values() for x in v) <= {k.split()[0] for k in c}
    assert set(etf.RELEVANT) == set(strategies(6)[4])


def test_report_flags_are_consistent():
    data, names = market(20)
    t = EtfTest(data, names, CFG)
    r = report(t, n_random=20).set_index("strategy")
    assert len(r) == 5 and set(r["months"]) == {len(t.returns)}
    for k, x in r.iterrows():
        want = x["c1"] and (x["c2"] is not False) and x["c3"] and x["c4"] and x["c5"]
        assert bool(x["passed"]) == bool(want)
        assert x["variants_run"] == 2 + 2 * len(etf.RELEVANT[k])
        assert x["c5"] == (x["sens_share"] >= 0.8 and x["stress_excess"] > 0)
    assert r.loc["E4 volatility targeting", "c2"] is None
    e1 = t.run()["E1 sector rotation"]
    assert r.loc["E1 sector rotation", "excess"] == pytest.approx((e1["net"] - t.bench).mean() * 12)
    h = len(e1) // 2
    assert r.loc["E1 sector rotation", "excess_second_half"] == pytest.approx((e1["net"] - t.bench).iloc[h:].mean() * 12)
    assert r.loc["E1 sector rotation", "excess_first_half"] == pytest.approx((e1["net"] - t.bench).iloc[:h].mean() * 12)


OK = dict(c1=True, c2=True, c3=True, c4=True, sens_share=0.80, stress_excess=0.01)


@pytest.mark.parametrize("change,c5,passed", [
    ({}, True, True), ({"c2": None}, True, True), ({"c1": False}, True, False), ({"c2": False}, True, False),
    ({"c3": False}, True, False), ({"c4": False}, True, False), ({"sens_share": 0.79}, False, False),
    ({"stress_excess": 0.0}, False, False)])
def test_flags(change, c5, passed):
    assert etf.flags({**OK, **change}) == {"c5": c5, "passed": passed}


def test_random_pool_for_e3_includes_bonds(monkeypatch):
    data, names = market(20)
    t = EtfTest(data, names, CFG)
    monkeypatch.setattr(en, "random_portfolios", lambda el, r, n_random: np.full(3, 9.0 if BONDS in el[t.sessions[0]] else -9.0))
    r = report(t, n_random=3).set_index("strategy")
    assert r.loc["E3 multi-asset trend", "random_95"] == 9.0 and not r.loc["E3 multi-asset trend", "c2"]
    assert r.loc["E1 sector rotation", "random_95"] == -9.0 and r.loc["E1 sector rotation", "c2"]


def test_relevant_settings():
    base = {"formation", "top_n", "keep", "min_dollar_vol"}
    assert set(etf.RELEVANT["E1 sector rotation"]) == base
    assert set(etf.RELEVANT["E1b rotation + MA filter"]) == set(etf.RELEVANT["E2 dual momentum"]) == base | {"ma_days"}
    assert set(etf.RELEVANT["E3 multi-asset trend"]) == base | {"ma_days", "vol_days", "max_weight"}
    assert set(etf.RELEVANT["E4 volatility targeting"]) == {"target_vol_days", "vol_threshold"}


def test_report_stress_uses_double_cost():
    data, names = market(20)
    t = EtfTest(data, names, CFG)
    r = report(t, n_random=3).set_index("strategy")
    s = t.run(cost=0.002)["E1 sector rotation"]["net"]
    assert r.loc["E1 sector rotation", "stress_excess"] == pytest.approx((s - t.bench).mean() * 12)
