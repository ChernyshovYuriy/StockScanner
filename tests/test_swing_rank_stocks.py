"""Swing rank Phase 3 -- stock rankings on point-in-time members. Synthetic."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research.swing_rank import engine as en
from research.swing_rank import stocks as st
from research.swing_rank.engine import CASH
from research.swing_rank.stocks import DEFAULT, Scores, StockConfig, StockData, StockTest, flags, make_strategies, report

CAL = pd.bdate_range("2015-01-01", "2020-12-31")
N = len(CAL)
MONTHS = [d for d in pd.date_range("2016-01-01", "2020-11-30", freq="BME")]
SECTORS = ["Energy", "Financials", "Materials", "Utilities", "Industrials", "Tech", "Health", "Staples"]
CFG = replace(DEFAULT, min_scored=20)


def frame(c, calendar=CAL, seed=0):
    c = np.asarray(c, float)
    o = c * (1 + np.random.default_rng(seed).normal(0, 0.001, len(c)))
    return pd.DataFrame({"Open": o, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6}, index=calendar[:len(c)])


def walk(seed, drift=0.0003, vol=0.015):
    return 20 * np.exp(np.cumsum(np.random.default_rng(seed).normal(drift, vol, N)))


def build(n=40, drift=None, drop_yahoo=(), gone_after=None, skip_files=(), vol=0.015, seed=0, months=MONTHS):
    """n companies C00.. on Yahoo and in the files. drop_yahoo: no Yahoo prices.
    gone_after: {number: last file date it is in}."""
    prices = {k: walk(seed + k, (drift or {}).get(k, 0.0003), vol) for k in range(n)}
    yahoo = {"XIU.TO": frame(walk(seed + 999, 0.0003, 0.008), seed=999)}
    for k in range(n):
        if k not in drop_yahoo:
            yahoo[f"C{'ABCDEFGHIJ'[k // 10]}{'ABCDEFGHIJ'[k % 10]}.TO"] = frame(prices[k], seed=k)
    files = {}
    for d in months:
        if d in skip_files:
            continue
        rows = []
        for k in range(n):
            if gone_after and k in gone_after and d > gone_after[k]:
                continue
            t = f"C{'ABCDEFGHIJ'[k // 10]}{'ABCDEFGHIJ'[k % 10]}"
            rows.append((t, f"COMPANY {k}", SECTORS[k % 8], 1000.0, float(prices[k][CAL.get_loc(d)]), 1.0, "Toronto"))
        files[d] = pd.DataFrame(rows, columns=["ticker", "name", "sector", "shares", "price", "weight", "exchange"]).set_index("ticker")
    data = StockData(files, yahoo)
    cid = {int(n.split()[1]): i for n, i in zip(data.table["name"], data.table["id"])}
    return data, cid, prices, yahoo


def test_members_and_sessions():
    data, cid, *_ = build(gone_after={3: MONTHS[20]})
    s = data.sessions[20]
    assert data.calendar[s] == MONTHS[20] and cid[3] in data.members[s] and cid[3] not in data.members[data.sessions[21]]
    assert len(data.symbol) == 40


def test_month_and_period_returns_by_hand():
    data, cid, prices, yahoo = build(drop_yahoo={5})
    k = 30
    a, b, c = (CAL.get_loc(MONTHS[j]) for j in (k - 1, k, k + 1))
    s = data.sessions[k]
    assert data.month.loc[s, cid[7]] == pytest.approx(prices[7][b] / prices[7][a] - 1)
    assert data.month.loc[s, cid[5]] == pytest.approx(prices[5][b] / prices[5][a] - 1)       # from the files
    o = yahoo[data.symbol[cid[7]]]["Open"]
    assert data.period.loc[s, cid[7]] == pytest.approx(o.iloc[c + 1] / o.iloc[b + 1] - 1)
    assert data.period.loc[s, cid[5]] == pytest.approx(prices[5][c] / prices[5][b] - 1)
    x = yahoo["XIU.TO"]["Open"]
    assert data.xiu.loc[s] == pytest.approx(x.iloc[c + 1] / x.iloc[b + 1] - 1)
    assert cid[5] not in data.symbol


def test_file_priced_names_are_credited_their_sectors_dividend():
    """Yahoo's total returns run 0.3% a month above the files for Energy names only."""
    data, cid, prices, yahoo = build(drop_yahoo={5, 8})
    lift = np.ones(N)
    for j in range(1, len(MONTHS)):
        lift[CAL.get_loc(MONTHS[j]):] *= 1.003
    for k in range(0, 40, 8):
        if k != 8:
            yahoo[data.symbol[cid[k]]].iloc[:, :4] *= lift[:, None]
    for k in range(3, 40, 8):                            # Utilities run 0.2% below: no negative dividends
        yahoo[data.symbol[cid[k]]].iloc[:, :4] *= (lift ** (-2 / 3))[:, None]
    files = {d: g.set_index("ticker").drop(columns=["date", "id"]) for d, g in data.table.groupby("date")}
    d2 = StockData(files, yahoo)
    assert d2.dividend["Utilities"] == 0.0
    assert d2.dividend["Energy"] == pytest.approx(0.003, rel=0.05) and d2.dividend["Financials"] == pytest.approx(0.0, abs=1e-9)
    s = data.sessions[30]
    assert d2.month.loc[s, cid[8]] - data.month.loc[s, cid[8]] == pytest.approx(d2.dividend["Energy"])
    assert d2.period.loc[s, cid[8]] - data.period.loc[s, cid[8]] == pytest.approx(d2.dividend["Energy"])
    assert d2.month.loc[s, cid[5]] == pytest.approx(data.month.loc[s, cid[5]])
    assert 0 <= d2.dividend[""] < d2.dividend["Energy"]


def test_renamed_company_is_not_an_unknown_ending():
    data, cid, prices, yahoo = build(drop_yahoo={5})
    t = data.table
    files = {}
    for d, g in t.groupby("date"):
        g = g.set_index("ticker").drop(columns=["date", "id"])
        if d > MONTHS[30]:
            g = g.rename(index={"CAF": "NEWT"})
            g.loc["NEWT", "name"] = "BRAND NEW NAME"
        g["shares"] = [1000.0 + 37 * k for k in range(len(g))]
        files[d] = g
    d2 = StockData(files, yahoo)
    assert d2.table["id"].nunique() == 40 and d2.unknown.sum().sum() == 0


def test_file_dated_on_a_holiday_after_the_last_session():
    """Good Friday 2018: the file says 30 March, the last session was the 29th."""
    holiday = MONTHS[26]
    data, cid, prices, yahoo = build(drop_yahoo={5})
    files = {d: g.set_index("ticker").drop(columns=["date", "id"]) for d, g in data.table.groupby("date")}
    files[holiday].loc["CAF", "price"] = prices[5][CAL.get_loc(holiday) - 1]
    d2 = StockData(files, {t: d.drop(holiday) for t, d in yahoo.items()})
    s = d2.sessions[26]
    assert d2.calendar[s] == CAL[CAL.get_loc(holiday) - 1] and d2._file_of[s] == holiday
    assert d2.unknown.sum().sum() == 0 and d2.coverage().min() == 1.0


def test_company_listed_twice_is_not_read_as_a_split():
    """Two half-size rows in one file, then one full row after the price halves: a real fall."""
    data, cid, prices, yahoo = build(drop_yahoo={5})
    files = {d: g.set_index("ticker").drop(columns=["date", "id"]) for d, g in data.table.groupby("date")}
    k = MONTHS[30]
    twin = files[k].loc[["CAF"]].rename(index={"CAF": "CAFOLD"})
    files[k].loc["CAF", "shares"] = twin["shares"] = 500.0
    files[k] = pd.concat([files[k], twin])
    for d in MONTHS[31:]:
        files[d].loc["CAF", "price"] *= 0.5
    d2 = StockData(files, yahoo)
    c = int(d2.table.loc[d2.table["name"] == "COMPANY 5", "id"].iloc[0])
    assert d2.table["id"].nunique() == 40
    assert d2.month.loc[d2.sessions[31], c] == pytest.approx(0.5 * (1 + data.month.loc[data.sessions[31], cid[5]]) - 1, abs=0.01)


def test_unknown_ending_only_for_file_priced_names_that_leave():
    data, cid, *_ = build(drop_yahoo={5, 6}, gone_after={5: MONTHS[30], 7: MONTHS[30]})
    s = data.sessions[30]
    assert data.unknown.loc[s, cid[5]] and data.unknown[cid[5]].sum() == 1
    assert not data.unknown[cid[6]].iloc[:-1].any()
    assert not data.unknown[cid[7]].any() and np.isfinite(data.period.loc[s, cid[7]])


def test_publishing_gap_is_bridged_not_unknown():
    skip = MONTHS[25:28]
    data, cid, prices, _ = build(drop_yahoo={5}, skip_files=skip)
    c = cid[5]
    s = data.sessions
    a, b = CAL.get_loc(MONTHS[24]), CAL.get_loc(MONTHS[28])
    assert not data.unknown[c].iloc[:-1].any()
    assert data.period.loc[s[24], c] == pytest.approx(prices[5][b] / prices[5][a] - 1)
    assert data.period.loc[s[25:28], c].tolist() == [0.0, 0.0, 0.0]
    assert data.members[s[26]] == data.members[s[24]]                      # the last earlier file stands in
    assert data.calendar[s[26]] == MONTHS[26]


def test_leaving_during_a_gap_is_unknown():
    data, cid, *_ = build(drop_yahoo={5}, skip_files=MONTHS[25:28], gone_after={5: MONTHS[24]})
    assert data.unknown.loc[data.sessions[24], cid[5]]


def test_harsh_endings():
    last = MONTHS[30]
    data, cid, prices, _ = build(drop_yahoo={1, 2, 3}, gone_after={1: last, 2: last, 3: last}, drift={2: -0.008})
    s = data.sessions[30]
    data.month.loc[s, cid[3]] = 0.015                    # pinned by an offer
    data.month.loc[s, cid[1]] = 0.05
    kind, harsh = data.returns(), data.returns(harsh=True)
    assert np.isnan(kind.loc[s, cid[1]])                 # the engine counts this as 0
    assert harsh.loc[s, cid[2]] == -1.0 and harsh.loc[s, cid[3]] == 0.0 and harsh.loc[s, cid[1]] == -0.5
    others = [c for c in harsh if c not in (cid[1], cid[2], cid[3])]
    pd.testing.assert_frame_equal(kind[others], harsh[others])


def test_harsh_ending_ignores_what_happened_later():
    data, cid, *_ = build(drop_yahoo={1}, gone_after={1: MONTHS[30]})
    s = data.sessions[30]
    data.month.loc[s, cid[1]] = 0.05
    data.month.loc[data.sessions[35], cid[1]] = -0.95
    assert data.returns(harsh=True).loc[s, cid[1]] == -0.5


def test_coverage():
    data, cid, *_ = build(n=40, drop_yahoo={5}, gone_after={5: MONTHS[30]})
    c = data.coverage()
    assert c.loc[data.sessions[30]] == pytest.approx(39 / 40) and c.loc[data.sessions[20]] == 1.0


def test_sector_filled_from_neighbours_when_missing():
    data, cid, *_ = build()
    t = data.table
    t.loc[(t["id"] == cid[4]) & (t["date"] <= MONTHS[10]), "sector"] = "-"
    data._sector = data._sectors()
    assert data.sector_of(data.sessions[2])[cid[4]] == SECTORS[4]


# ── scores ───────────────────────────────────────────────────────────────────

def test_momentum_skips_the_last_month():
    data, cid, prices, _ = build(drop_yahoo={5})
    sc = Scores(data, CFG)
    s = data.sessions[30]
    a, b = CAL.get_loc(MONTHS[18]), CAL.get_loc(MONTHS[29])
    m = sc.momentum(s)
    assert m[cid[7]] == pytest.approx(prices[7][b] / prices[7][a] - 1) and m[cid[5]] == pytest.approx(prices[5][b] / prices[5][a] - 1)


def test_momentum_needs_all_eleven_months():
    data, cid, *_ = build(drop_yahoo={5})
    data.month.loc[data.sessions[25], cid[5]] = np.nan
    m = Scores(data, CFG).momentum(data.sessions[30])
    assert np.isnan(m[cid[5]]) and np.isfinite(m[cid[6]])


def test_scores_cover_members_only():
    data, cid, *_ = build(gone_after={3: MONTHS[20]})
    sc = Scores(data, CFG)
    for f in (sc.momentum, sc.low_vol, sc.residual):
        assert cid[3] not in f(data.sessions[40]).index and len(f(data.sessions[40])) == 39


def test_low_vol_by_hand():
    data, cid, prices, _ = build(drop_yahoo={5})
    s = data.sessions[30]
    v = Scores(data, CFG).low_vol(s)
    r = pd.Series(prices[7]).pct_change().iloc[s - 251:s + 1]
    assert v[cid[7]] == pytest.approx(-r.std() * 252 ** 0.5)
    m = data.month[cid[5]].iloc[19:31]
    assert v[cid[5]] == pytest.approx(-m.std() * 12 ** 0.5)


def test_residual_by_hand():
    data, cid, *_ = build()
    s = data.sessions[50]
    got = Scores(data, CFG).residual(s)
    w = data.month.iloc[14:50]
    sec = data.sector_of(s)
    c = cid[9]
    peers = [x for x in data.members[s] if sec[x] == sec[c]]
    X = np.column_stack([np.ones(36), data.xiu_month.iloc[14:50], w[peers].mean(axis=1)])
    b = np.linalg.lstsq(X, w[c].to_numpy(), rcond=None)[0]
    e = w[c].to_numpy() - X[:, 1] * b[1] - X[:, 2] * b[2]
    assert got[c] == pytest.approx(e[-11:].sum() / e[-11:].std(ddof=1))


def test_residual_needs_full_history():
    data, cid, *_ = build()
    sc = Scores(data, CFG)
    assert sc.residual(data.sessions[30]).isna().all() and sc.residual(data.sessions[40]).notna().all()
    data.month.loc[data.sessions[20], cid[8]] = np.nan
    assert np.isnan(sc.residual(data.sessions[40])[cid[8]])


def test_scores_never_use_the_future():
    data, cid, prices, yahoo = build(drop_yahoo={5})
    k = 45
    cut = CAL.get_loc(MONTHS[k])
    rng = np.random.default_rng(0)
    wild = {t: d.copy() for t, d in yahoo.items()}
    for d in wild.values():
        d.iloc[cut + 1:, :4] *= np.exp(np.cumsum(rng.normal(0, 0.1, N - cut - 1)))[:, None]
    files = {}
    for d, g in data.table.groupby("date"):
        g = g.set_index("ticker").drop(columns=["date", "id"])
        if d > MONTHS[k]:
            g["price"] *= np.exp(rng.normal(0, 0.3, len(g)))
        files[d] = g
    other = StockData(files, wild)
    other.symbol = data.symbol                           # identity is matched on all history, by design
    other._build_returns()
    s = data.sessions[k]
    a, b = Scores(data, CFG), Scores(other, CFG)
    for f in ("momentum", "low_vol", "residual"):
        pd.testing.assert_series_equal(getattr(a, f)(s), getattr(b, f)(s))


# ── strategies and the test ──────────────────────────────────────────────────

def test_strategies_pick_the_best_within_the_sector_cap():
    data, cid, *_ = build()
    sc = Scores(data, CFG)
    s = data.sessions[45]
    for name, f in (("S1 momentum 12-1", sc.momentum), ("S2 low volatility", sc.low_vol), ("S3 residual momentum", sc.residual)):
        w = make_strategies(sc, CFG)[name](s, set())
        sec = data.sector_of(s)
        assert len(w) == 10 and max(pd.Series([sec[c] for c in w]).value_counts()) <= 2
        assert f(s).idxmax() in w


def test_low_vol_prefers_calm_names():
    data, cid, *_ = build()
    sc = Scores(data, CFG)
    s = data.sessions[45]
    v = sc.low_vol(s)
    assert v.idxmax() == v.abs().idxmin()


def test_stock_test_start_needs_all_scores():
    data, *_ = build()
    t = StockTest(data, CFG)
    assert data.sessions.index(t.start) == 37                              # 36 earlier monthly returns, the first session has none
    assert len(t.index) == len(t.bench) == len(t.years)


def test_planted_momentum_is_found():
    data, *_ = build(n=40, drift={k: 0.0025 for k in range(8)}, vol=0.01, seed=50)
    t = StockTest(data, CFG)
    e = en.evaluate(t.run()["S1 momentum 12-1"]["net"], t.universe(), t.years)
    assert e["excess"] > 0.10 and e["lo"] > 0


def test_noise_is_not_found():
    hits = 0
    for seed in range(6):
        data, *_ = build(n=40, seed=1000 * (seed + 1))
        t = StockTest(data, CFG)
        for r in t.run().values():
            hits += en.evaluate(r["net"], t.universe(), t.years)["c1"]
    assert hits <= 3                                                        # 18 tests at 2.5%, short samples


def test_harsh_run_is_never_better_and_universe_matches_by_hand():
    data, cid, *_ = build(drop_yahoo=set(range(5)), gone_after={k: MONTHS[44] for k in range(5)})
    t = StockTest(data, CFG)
    for k, r in t.run().items():
        assert en.cagr(t.run(harsh=True)[k]["net"]) <= en.cagr(r["net"]) + 1e-12
    s = data.sessions[44]
    r = data.period.loc[s, data.members[s]].fillna(0.0)
    assert t.universe().loc[s] == pytest.approx(r.mean())
    assert t.universe(harsh=True).loc[s] < t.universe().loc[s]


def test_universe_is_that_months_members_only():
    data, cid, *_ = build(gone_after={k: MONTHS[40] for k in range(10)})
    t = StockTest(data, CFG)
    s = data.sessions[45]
    assert len(data.members[s]) == 30 and np.isfinite(data.period.loc[s, cid[3]])
    assert t.universe().loc[s] == pytest.approx(data.period.loc[s, data.members[s]].mean())


def test_harsh_run_hits_a_holding_that_vanishes():
    """The strongest name is file-priced and disappears while held."""
    data, cid, *_ = build(drop_yahoo={0}, gone_after={0: MONTHS[44]}, drift={0: 0.004}, vol=0.01)
    t = StockTest(data, CFG)
    data.month.loc[data.sessions[44], cid[0]] = 0.06
    s = data.sessions[44]
    kind, harsh = t.run()["S1 momentum 12-1"], t.run(harsh=True)["S1 momentum 12-1"]
    assert harsh.loc[s, "gross"] == pytest.approx(kind.loc[s, "gross"] - 0.05, abs=0.02)
    r = report(t, n_random=3).set_index("strategy")
    assert r.loc["S1 momentum 12-1", "harsh_excess"] < r.loc["S1 momentum 12-1", "excess"] - 0.005


def test_costs_matter():
    data, *_ = build()
    t = StockTest(data, CFG)
    a, b = t.run(cost=0.0), t.run(cost=0.01)
    assert all(en.cagr(b[k]["net"]) < en.cagr(a[k]["net"]) for k in a)


OK = dict(c1=True, c2=True, c3=True, c4=True, sens_share=0.8, stress_excess=0.01, harsh_excess=0.01, universe_excess=0.01, universe_lo=0.001)


@pytest.mark.parametrize("change", [{}, {"c1": False}, {"c2": False}, {"c3": False}, {"c4": False}, {"sens_share": 0.79},
                                    {"stress_excess": 0.0}, {"harsh_excess": 0.0}, {"universe_lo": 0.0}, {"universe_excess": -0.1, "universe_lo": 0.1}])
def test_flags(change):
    assert flags({**OK, **change})["passed"] is (change == {})


def test_report_numbers():
    data, *_ = build(drop_yahoo={1}, gone_after={1: MONTHS[50]})
    t = StockTest(data, CFG)
    r = report(t, n_random=10).set_index("strategy")
    assert set(r.index) == set(st.RELEVANT)
    s1 = t.run()["S1 momentum 12-1"]
    assert r.loc["S1 momentum 12-1", "excess"] == pytest.approx((s1["net"] - t.bench).mean() * 12)
    assert r.loc["S1 momentum 12-1", "universe_excess"] == pytest.approx((s1["net"] - t.universe()).mean() * 12)
    assert r.loc["S1 momentum 12-1", "stress_excess"] == pytest.approx((t.run(cost=0.004)["S1 momentum 12-1"]["net"] - t.bench).mean() * 12)
    assert r.loc["S1 momentum 12-1", "harsh_excess"] == pytest.approx((t.run(harsh=True)["S1 momentum 12-1"]["net"] - t.bench).mean() * 12)
    h = len(s1) // 2
    assert r.loc["S1 momentum 12-1", "excess_second_half"] == pytest.approx((s1["net"] - t.bench).iloc[h:].mean() * 12)
    assert r.loc["S1 momentum 12-1", "variants_run"] == 9 and r.loc["S3 residual momentum", "variants_run"] == 11
    for k, x in r.iterrows():
        assert bool(x["passed"]) == flags(x.to_dict())["passed"]


def test_sensitivity_configs():
    c = st.sensitivity_configs()
    assert len(c) == 10 and c["formation -20%"].formation == 10 and c["formation +20%"].formation == 14
    assert c["regress_months -20%"].regress_months == 29 and c["vol_days +20%"].vol_days == 302
    assert st.RELEVANT == {"S1 momentum 12-1": ("formation", "top_n", "keep"), "S2 low volatility": ("vol_days", "top_n", "keep"),
                           "S3 residual momentum": ("formation", "regress_months", "top_n", "keep")}
