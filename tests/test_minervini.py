"""Minervini method: hand-checked setups, fills, exits, book. No network."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.minervini import backtest as bk
from research.minervini import method as mm

CAL = pd.bdate_range("2018-01-01", periods=700)


def arr(o, h, l, c, v=None):
    n = len(c)
    x = {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v if v is not None else [1000.0] * n}
    return mm.arrays(pd.DataFrame({k: np.asarray(y, float) for k, y in x.items()}))


# ── The owner's example ──────────────────────────────────────────────────────

def tgtx():
    return pd.read_csv(Path(__file__).parent / "data" / "tgtx.csv.gz", index_col=0, parse_dates=True)


def test_the_owners_example_is_found():
    d = tgtx()
    a, t = mm.arrays(d), d.index.get_loc("2026-11-23")
    assert mm.template(a, np.full(len(d), 0.98))[t]
    pivot, low = mm.setup(a, t)
    assert pivot == pytest.approx(68.03, abs=.001) and low == pytest.approx(61.50, abs=.001)
    k, price = mm.entry_at(a, t, pivot, "stop")
    assert d.index[k] == pd.Timestamp("2026-11-24") and price == pivot
    assert mm.initial_stop(price, low) == pytest.approx(68.03 * 0.92)


def test_example_does_not_use_the_next_day():
    d = tgtx()
    t = d.index.get_loc("2026-11-23")
    assert mm.setup(mm.arrays(d.iloc[:t + 1]), t) == mm.setup(mm.arrays(d), t)


# ── Template and setup ───────────────────────────────────────────────────────

def rising(n=400):
    return 10 * 1.003 ** np.arange(n)


def test_template_each_condition_matters():
    c = rising()
    a = arr(c, c * 1.01, c * .99, c)
    full = np.ones(400)
    assert mm.template(a, full)[300] and not mm.template(a, full)[250]            # 252 bars of history needed
    assert mm.template(a, full * .70)[-1] and not mm.template(a, full * .69)[-1] and not mm.template(a, full * np.nan)[-1]
    for key, val in (("ma150", 1e9), ("ma200", 1e9), ("ma50", 1e9), ("lo252", a["Close"][-1] / 1.29), ("hi252", a["Close"][-1] / 0.74)):
        b = dict(a); b[key] = a[key].copy(); b[key][-1] = val
        assert not mm.template(b, full)[-1], key
    for key, val in (("ma50", a["ma150"][-1] - .01), ("ma150", a["ma200"][-1] - .01), ("ma200", a["ma200"][-22] - .01)):
        b = dict(a); b[key] = a[key].copy(); b[key][-1] = val
        assert not mm.template(b, full)[-1], key


def rest(depth=0.06, prior=0.12, age=8, dry=True, n=200):
    """Rise to a pivot of 100 `age` bars before the last bar, after a fall of `prior`."""
    h = np.full(n, 90.0); l = np.full(n, 89.0)
    k = n - 1 - age
    h[k - 20], l[k - 20] = 98, 97
    l[k - 10] = 98 * (1 - prior); h[k - 10] = l[k - 10] + 1
    if prior < .08:
        l[:] = np.maximum(l, 98 * (1 - prior)); h = np.maximum(h, l + .5); h[k - 20] = 98
    h[k], l[k] = 100, 98
    h[k + 1:], l[k + 1:] = 97, 96
    l[n - 2] = 100 * (1 - depth)
    c = (h + l) / 2
    v = np.full(n, 1000.0)
    v[-5:] = 900 if dry else 1100
    return arr(c, h, l, c, v)


def test_setup_found_with_pivot_and_low():
    assert mm.setup(rest(), 199) == pytest.approx((100, 94))


@pytest.mark.parametrize("kw,ok", [(dict(depth=.10), True), (dict(depth=.101, prior=.2), False), (dict(depth=.095, prior=.085), False), (dict(depth=.09, prior=.085), True),
                                   (dict(age=5), True), (dict(age=4), False), (dict(dry=False), False)])
def test_setup_limits(kw, ok):
    assert bool(mm.setup(rest(**{"prior": .12, **kw}), 199)) == ok


def test_pivot_is_at_most_29_sessions_old():
    a = rest(age=29)
    assert mm.setup(a, 199)[0] == 100
    b = rest(age=30)
    s = mm.setup(b, 199)
    assert s is None or s[0] < 100


def test_deepest_fall():
    assert mm.deepest_fall(np.array([10, 12, 11, 13.]), np.array([9, 11, 9, 12.])) == pytest.approx(1 - 9 / 12)
    assert mm.deepest_fall(np.array([10., 12]), np.array([5, 11.])) < 0      # a low before the high is not a fall from it


def test_setup_needs_history():
    assert mm.setup({k: v[-60:] for k, v in rest().items()}, 59) is None


# ── Entry ────────────────────────────────────────────────────────────────────

def two(o, h, c=100., v=1000.0, o2=101.0):
    n = 60
    a = {k: np.array(v) for k, v in arr([99.0] * n, [99.5] * n, [98.0] * n, [99.0] * n, [1000.0] * n).items()}
    a["Open"][51], a["High"][51], a["Close"][51], a["Volume"][51], a["Open"][52] = o, h, c, v, o2
    return a


@pytest.mark.parametrize("o,h,want", [(99, 101, (51, 100)), (102, 103, (51, 102)), (99, 100, None), (105.0, 106, (51, 105.0)), (105.1, 106, None)])
def test_stop_entry(o, h, want):
    assert mm.entry_at(two(o, h), 50, 100.0, "stop") == want


@pytest.mark.parametrize("c,v,o2,want", [(101, 1400, 101, (52, 101)), (101, 1399, 101, None), (100, 2000, 101, None), (101, 1400, 105.1, None)])
def test_volume_entry(c, v, o2, want):
    assert mm.entry_at(two(99, 102, c, v, o2), 50, 100.0, "volume") == want


def test_no_entry_after_the_data():
    a = two(99, 101)
    assert mm.entry_at(a, 59, 98.0, "stop") is None and mm.entry_at(two(99, 102, 101, 5000), 58, 98., "volume") is None


def test_initial_stop_is_the_nearer_of_the_two():
    assert mm.initial_stop(100, 95) == pytest.approx(94.905) and mm.initial_stop(100, 85) == pytest.approx(92)


# ── Exit ─────────────────────────────────────────────────────────────────────

def path(bars, ma50=0.0, ma20=0.0):
    o, h, l, c = (np.array(x, float) for x in zip(*bars))
    n = len(o)
    return {"Open": o, "High": h, "Low": l, "Close": c, "ma50": np.full(n, ma50), "ma20": np.full(n, ma20)}


FLAT = (100, 101, 99, 100)


@pytest.mark.parametrize("how,bar,price", [("t10", (105, 111, 104, 110), 110), ("t20", (105, 121, 104, 110), 120), ("2R", (105, 117, 104, 110), 116),
                                           ("3R", (105, 125, 104, 110), 124), ("m21t10", (105, 111, 104, 110), 110), ("t10", (113, 114, 112, 113), 113)])
def test_targets(how, bar, price):
    x = mm.leave(path([FLAT, bar, FLAT]), 0, 100, 92, how)
    assert x["why"] == "target" and x["ret"] == pytest.approx(price / 100 - 1) and x["end"] == 1 and x["r"] == pytest.approx((price - 100) / 8)


@pytest.mark.parametrize("how", mm.EXITS)
def test_stop_before_target_and_gap(how):
    x = mm.leave(path([FLAT, (100, 150, 92, 100)]), 0, 100, 92, how)
    assert x["why"] == "stop" and x["ret"] == pytest.approx(-.08) and x["r"] == pytest.approx(-1)
    assert mm.leave(path([FLAT, (85, 86, 84, 85)]), 0, 100, 92, how)["ret"] == pytest.approx(-.15)


def test_entry_bar_can_stop_or_reach_the_target():
    assert mm.leave(path([(98, 101, 91, 100), FLAT]), 0, 100, 92, "t10")["end"] == 0
    assert mm.leave(path([(98, 112, 97, 100), FLAT]), 0, 100, 92, "t10")["ret"] == pytest.approx(.10)


def test_half_then_breakeven_or_average():
    x = mm.leave(path([FLAT, (110, 117, 109, 115), (104, 105, 99, 100)]), 0, 100, 92, "half2R")
    assert x["why"] == "breakeven" and x["ret"] == pytest.approx(.5 * .16 + 0) and [f for _, f, _ in x["legs"]] == [.5, .5]
    y = path([FLAT, (110, 117, 109, 115), (114, 115, 108, 109), (108, 110, 107, 108)])
    y["ma50"][2] = 112
    z = mm.leave(y, 0, 100, 92, "half2R")
    assert z["why"] == "average" and z["ret"] == pytest.approx(.5 * .16 + .5 * .08) and z["end"] == 3
    assert mm.leave(path([FLAT, (110, 121, 109, 115), FLAT]), 0, 100, 92, "half20")["legs"][0][2] == 120


def test_half_exits_ignore_the_average_before_the_first_half():
    assert mm.leave(path([FLAT] * 5, ma50=200), 0, 100, 92, "half2R")["why"] == "open"


@pytest.mark.parametrize("how", ["ma50", "ma20"])
def test_average_exit_next_open_and_breakeven_after_2r(how):
    y = path([FLAT, FLAT, (97, 99, 96, 98)])
    y[how][1] = 100.5
    x = mm.leave(y, 0, 100, 92, how)
    assert x["why"] == "average" and x["end"] == 2 and x["ret"] == pytest.approx(-.03)
    z = mm.leave(path([FLAT, (110, 116, 109, 112), (101, 102, 99, 100)]), 0, 100, 92, how)
    assert z["why"] == "breakeven" and z["ret"] == pytest.approx(0)
    assert mm.leave(path([FLAT, (110, 115.9, 109, 112), (101, 102, 99, 100)]), 0, 100, 92, how)["why"] == "open"


def test_each_average_is_its_own():
    y = path([FLAT] * 4, ma50=200, ma20=0)
    assert mm.leave(y, 0, 100, 92, "ma20")["why"] == "open" and mm.leave(y, 0, 100, 92, "ma50")["why"] == "average"


def test_month_is_21_sessions_and_the_rest_252():
    b = [FLAT] * 400
    assert mm.leave(path(b), 5, 100, 92, "m21")["end"] == 25 and mm.leave(path(b), 5, 100, 92, "m21t10")["end"] == 25
    x = mm.leave(path(b), 5, 100, 92, "t20")
    assert x["end"] == 256 and x["why"] == "time" and x["bars"] == 252


def test_end_of_data_is_valued_not_dropped():
    x = mm.leave(path([FLAT, (103, 104, 102, 103)]), 0, 100, 92, "t20")
    assert x["why"] == "open" and x["ret"] == pytest.approx(.03)


def test_no_risk_no_trade():
    assert mm.leave(path([FLAT]), 0, 92, 92, "t10") is None


# ── Trades of one name ───────────────────────────────────────────────────────

def test_one_position_at_a_time():
    n = 120
    a = arr([99.0] * n, [100.5] * n, [98.5] * n, [100.0] * n)
    setups = {t: (100.0, 97.0) for t in range(60, 110)}
    out = mm.find_trades(a, np.ones(n, bool), "stop", "m21", setups=setups)
    assert [x["k"] for x in out] == [61, 82, 103] and [x["end"] for x in out] == [81, 102, 119]
    assert all(x["t"] == x["k"] - 1 and x["entry"] == 100 for x in out)


def test_setups_only_where_allowed():
    a = rest()
    ok = np.zeros(200, bool)
    assert mm.all_setups(a, ok) == {}
    ok[199] = True
    assert list(mm.all_setups(a, ok)) == [199]


# ── Market side ──────────────────────────────────────────────────────────────

def frame(seed, n=700, drift=.0008):
    g = np.random.default_rng(seed)
    c = 50 * np.exp(np.cumsum(g.normal(drift, .012, n)))
    o = c * np.exp(g.normal(0, .003, n))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * (1 + g.uniform(0, .006, n)), "Low": np.minimum(o, c) * (1 - g.uniform(0, .006, n)),
                         "Close": c, "Volume": g.uniform(5e5, 2e6, n)}, index=CAL[:n])


def data(n=700):
    return {"XIU.TO": frame(99).iloc[:n], **{f"S{k}.TO": frame(k, drift=.0002 * k).iloc[:n] for k in range(12)}}


def test_strength_rank_by_hand():
    st = bk.Study(data())
    c = pd.DataFrame({t: d["Close"] for t, d in st.m.bars.items()})
    r = sum(w * (c.iloc[600] / c.iloc[600 - n] - 1) for n, w in ((63, .4), (126, .2), (189, .2), (252, .2)))
    pd.testing.assert_series_equal(st.rank.iloc[600], r.rank() / len(r), check_names=False)


def test_signals_do_not_look_ahead():
    full, cut = bk.Study(data()), bk.Study(data(600))
    n = 0
    for t in full.setups:
        want = {k: v for k, v in full.setups[t].items() if k < 600}
        assert cut.setups[t] == want
        n += len(want)
    assert n > 20


def test_trades_before_the_cut_are_the_same():
    f, c = bk.Study(data()).trades("stop", "m21"), bk.Study(data(600)).trades("stop", "m21")
    f, c = f[f["exit_date"] < CAL[590]], c[c["exit_date"] < CAL[590]]
    assert len(f) > 3
    pd.testing.assert_frame_equal(f.reset_index(drop=True), c.reset_index(drop=True))


def test_costs_are_charged_and_r_is_net():
    x = bk.Study(data()).trades("stop", "m21").iloc[0]
    assert x["net"] == pytest.approx(x["gross"] - .005) and x["r"] == pytest.approx(x["net"] / (1 - x["stop"] / x["entry"]))


def test_market_and_relative_filters_by_hand():
    st = bk.Study(data())
    x, d = st.m.xiu_close, CAL[650]
    k = 650
    want = x.iloc[k] > x.iloc[k - 199:k + 1].mean() and x.iloc[k - 199:k + 1].mean() > x.iloc[k - 220:k - 20].mean()
    assert st.market.loc[d] == want
    rel = st.m.bars["S3.TO"]["Close"] / x
    assert st.rel_high.at[d, "S3.TO"] == (rel.iloc[650] >= rel.iloc[399:651].max())


def test_growth():
    ts = pd.to_datetime(["2019-02-10", "2019-05-10", "2019-08-10", "2019-11-10", "2020-02-10 17:00"], format="mixed")
    e = pd.DataFrame({"ticker": "S1.TO", "report_ts": ts, "eps_rep": [1.0, 9, 9, 9, 1.25]})
    g = lambda eps, day: bk.Study.growth(type("X", (), {"eps": {"S1.TO": e.assign(eps_rep=eps)}})(), "S1.TO", pd.Timestamp(day))
    base = [1.0, 9, 9, 9, 1.25]
    assert g(base, "2020-02-11") is True and g(base, "2020-02-10") is None and g(base, "2020-06-01") is None
    assert g([1.0, 9, 9, 9, 1.24], "2020-02-11") is False and g([-1, 9, 9, 9, .1], "2020-02-11") is True and g([-1, 9, 9, 9, -.1], "2020-02-11") is False


def test_peers_are_same_day_template_names_without_itself():
    st = bk.Study(data())
    tr = st.trades("stop", "m21").iloc[[2]].reset_index(drop=True)
    e = tr.iloc[0]
    names = [t for t in st.a if t != e["ticker"] and st.template.at[e["before"], t]]
    names = sorted(names, key=lambda t: abs(st.m.atr.at[e["before"], t] - st.m.atr.at[e["before"], e["ticker"]]))[:10]
    want = []
    for t in names:
        k = st.idx[t].get_loc(e["entry_date"])
        p = st.a[t]["Open"][k]
        want.append(mm.leave(st.a[t], k, p, p * e["stop"] / e["entry"], "m21")["ret"] - .005)
    assert names and st.peers(tr, "m21")["peer"].iloc[0] == pytest.approx(np.mean(want))


# ── Book ─────────────────────────────────────────────────────────────────────

class M:
    def __init__(self, prices):
        self.cal = CAL[:len(next(iter(prices.values())))]
        self.last = pd.DataFrame({t: np.asarray(p, float) for t, p in prices.items()}, index=self.cal)
        self.xiu_close = pd.Series(np.linspace(100, 110, len(self.cal)), index=self.cal)


def trade(t="A", entry=100., stop=92., k=1, legs=((3, 1.0, 110.),), rank=.9):
    return {"ticker": t, "entry_date": CAL[k], "entry": entry, "stop": stop, "rank": rank, "legs": [(CAL[j], f, p) for j, f, p in legs]}


def test_book_sizes_by_risk_and_charges_cost(monkeypatch):
    m = M({"A": [100] * 5})
    p = bk.portfolio(pd.DataFrame([trade(stop=80.)]), m, str(CAL[0]), str(CAL[4]))
    assert p["final"] == pytest.approx(1 - .05 * 1.0025 + .05 * 1.1 * .9975)                  # 1% / 20% = 5% of equity
    p = bk.portfolio(pd.DataFrame([trade(stop=96.)]), m, str(CAL[0]), str(CAL[4]))
    assert p["final"] == pytest.approx(1 - .125 * 1.0025 + .125 * 1.1 * .9975)               # capped


def test_book_takes_the_strongest_eight():
    names = [f"N{k}" for k in range(10)]
    m = M({t: [100] * 5 for t in names})
    tr = pd.DataFrame([trade(t, rank=.9 + k / 1000, stop=99., legs=((3, 1., 100 + k),)) for k, t in enumerate(names)])
    p = bk.portfolio(tr, m, str(CAL[0]), str(CAL[4]))
    assert p["taken"] == 8
    v = .125
    spent, gain = 0.0, 0.0
    for k in range(9, 1, -1):
        x = min(v, (1 - spent) / 1.0025)
        spent += x * 1.0025; gain += x * (1 + k / 100) * .9975
    assert p["final"] == pytest.approx(1 - spent + gain)


def test_book_marks_daily_for_drawdown_and_half_sales():
    m = M({"A": [100, 100, 50, 100, 100, 100]})
    p = bk.portfolio(pd.DataFrame([trade(legs=((3, .5, 120.), (4, .5, 100.)))]), m, str(CAL[0]), str(CAL[5]))
    assert p["max_dd"] == pytest.approx(-(.0625 + .125 * .0025))
    assert p["final"] == pytest.approx(1 - .125 * 1.0025 + .125 * (.5 * 1.2 + .5) * .9975)


def test_position_open_at_the_end_is_counted_at_market():
    m = M({"A": [100, 100, 104]})
    p = bk.portfolio(pd.DataFrame([trade(legs=((9, 1., 500.),))]), m, str(CAL[0]), str(CAL[2]))
    assert p["final"] == pytest.approx(1 - .125 * 1.0025 + .125 * 1.04)


def test_verdict_needs_all():
    good = dict(net=.01, lo=.001, ex=.01, ex_lo=.001, cagr=.2, xiu_cagr=.1)
    assert bk.verdict(good)
    for k, v in (("net", -1), ("lo", -.1), ("ex", -1), ("ex_lo", -.1), ("cagr", .05)):
        assert not bk.verdict({**good, k: v}), k


def test_choice_uses_period_a_only():
    ix = pd.MultiIndex.from_product([["stop"], ["t10", "t20", "ma50"], ["A", "B", "C"]], names=["entry", "exit", "period"])
    t = pd.DataFrame({"n": 200, "cagr": [.05, 9, 9, .08, -1, -1, .5, 9, 9]}, index=ix)
    t.loc[("stop", "ma50", "A"), "n"] = 99
    assert bk.choose(t) == ("stop", "t20")


def test_periods_do_not_overlap():
    tr = pd.DataFrame({"entry_date": pd.to_datetime(["2018-11-30", "2018-12-10", "2019-01-02", "2023-06-30", "2023-09-01", "2024-01-02"])})
    assert [len(bk.period(tr, p)) for p in "ABC"] == [1, 2, 1]


def test_the_pivot_bars_own_low_is_not_part_of_the_rest():
    a = {k: np.array(v) for k, v in rest().items()}
    a["Low"][199 - 8] = 91.0
    assert mm.setup(a, 199) == pytest.approx((100, 94))


def test_a_setup_made_while_holding_is_not_used():
    n = 140
    a = {k: np.array(v) for k, v in arr([99.0] * n, [99.5] * n, [98.5] * n, [99.0] * n).items()}
    for k in (90, 111):
        a["Close"][k], a["Volume"][k] = 101.0, 9000.0
    out = mm.find_trades(a, np.ones(n, bool), "volume", "m21", setups={89: (100.0, 95.0), 110: (100.0, 95.0)})
    assert [x["k"] for x in out] == [91] and out[0]["end"] == 111                  # bought at 91, held to 111; the setup of 110 is inside


def test_only_eligible_template_days_count():
    d = data()
    d["S11.TO"] = d["S11.TO"].assign(Volume=100.0)                                # illiquid
    st = bk.Study(d)
    assert not st.ok["S11.TO"].any() and st.rank["S11.TO"].isna().all()
    for t in ("S9.TO", "S10.TO"):
        a = mm.template(st.a[t], st.rank[t].to_numpy())
        assert (st.ok[t] == (a & st.m.eligible[t].to_numpy())).all() and st.ok[t].any() and not st.ok[t].all()


def test_growth_needs_five_reports():
    ts = pd.to_datetime(["2019-05-10", "2019-08-10", "2019-11-10", "2020-02-10"])
    e = pd.DataFrame({"ticker": "S1.TO", "report_ts": ts, "eps_rep": [1.0, 1, 1, 2]})
    assert bk.Study.growth(type("X", (), {"eps": {"S1.TO": e}})(), "S1.TO", pd.Timestamp("2020-02-11")) is None


def test_book_never_holds_more_than_eight_even_when_small():
    names = [f"N{k}" for k in range(10)]
    tr = pd.DataFrame([trade(t, stop=80.) for t in names])
    assert bk.portfolio(tr, M({t: [100] * 5 for t in names}), str(CAL[0]), str(CAL[4]))["taken"] == 8


def test_half_sold_is_no_longer_marked():
    m = M({"A": [100, 100, 100, 100, 200, 200]})
    p = bk.portfolio(pd.DataFrame([trade(legs=((2, .5, 100.), (99, .5, 1.)))]), m, str(CAL[0]), str(CAL[5]))
    assert p["final"] == pytest.approx(1 - .125 * 1.0025 + .0625 * .9975 + .0625 * 2)
