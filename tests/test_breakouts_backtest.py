"""Breakout backtest: hand-checked numbers on synthetic markets. No network."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from research.breakouts import backtest as bt
from research.breakouts import prices
from research.breakouts.detector import DEFAULT

CAL = pd.bdate_range("2019-01-01", periods=900)


@pytest.fixture(autouse=True)
def one_process(monkeypatch):
    class Serial:
        def __init__(self, n): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def map(self, f, *xs, chunksize=1): return map(f, *xs)
    monkeypatch.setattr(bt, "ProcessPoolExecutor", Serial)


def bars(close, idx=CAL, open_=None, vol=1e6, wick=0.002):
    c = np.asarray(close, float)
    o = c if open_ is None else np.asarray(open_, float)
    return pd.DataFrame({"Open": o, "High": np.maximum(c, o) * (1 + wick), "Low": np.minimum(c, o) * (1 - wick), "Close": c, "Volume": vol}, index=idx[:len(c)])


def flat(p=100.0, n=len(CAL)):
    return bars(np.full(n, p))


def shape(at, after=0.0, n=len(CAL)):
    """Two tests of 100 and a breakout to 104 on bar `at`, then drifting."""
    x = [0, at - 80, at - 60, at - 40, at - 20, at - 2, at - 1, at]
    c = np.interp(np.arange(n), x, [80, 100, 88, 100, 88, 100, 101, 104])
    c[at + 1:] = 104 * (1 + after) ** np.arange(1, n - at)
    return c


# ── Panels and returns ───────────────────────────────────────────────────────

def test_eligible_needs_price_volume_and_history():
    m = bt.Market({"XIU.TO": flat(), "OK.TO": flat(), "CHEAP.TO": flat(0.9), "THIN.TO": bars(np.full(900, 100.0), vol=9_000)})
    e = m.eligible
    assert not e["OK.TO"].iloc[250] and e["OK.TO"].iloc[251:].all()
    assert not e["CHEAP.TO"].any() and not e["THIN.TO"].any() and "XIU.TO" not in e


def test_not_eligible_on_a_day_without_a_trade():
    d = flat().drop(CAL[400])
    assert not bt.Market({"XIU.TO": flat(), "A.TO": d}).eligible["A.TO"].iloc[400]


def test_forward_is_next_open_to_close_h_sessions_after_entry():
    o, c = np.arange(100, 1000, dtype=float), np.arange(100, 1000, dtype=float) + 0.5
    m = bt.Market({"XIU.TO": bars(c, open_=o), "A.TO": bars(c, open_=o)})
    assert m.forward(20)["A.TO"].iloc[10] == pytest.approx(c[31] / o[11] - 1)
    assert m.xiu_forward(20).iloc[10] == pytest.approx(c[31] / o[11] - 1)


def test_a_dead_name_is_closed_at_its_last_bar():
    m = bt.Market({"XIU.TO": flat(), "A.TO": bars(np.linspace(100, 50, 500))})
    last = m.bars["A.TO"]["Close"].iloc[-1]
    assert m.forward(60)["A.TO"].iloc[480] == pytest.approx(last / m.bars["A.TO"]["Open"].iloc[481] - 1)


def test_new_high_excludes_today_from_the_comparison():
    c = np.full(900, 100.0)
    c[300] = 101
    m = bt.Market({"XIU.TO": flat(), "A.TO": bars(c)})
    assert m.new_high["A.TO"].sum() == 1 and m.new_high["A.TO"].iloc[300]


def test_atr_share():
    m = bt.Market({"XIU.TO": flat(), "A.TO": bars(np.full(900, 100.0), wick=0.01)})
    assert m.atr["A.TO"].iloc[-1] == pytest.approx(0.02)


def test_separate():
    assert bt.separate([CAL[0], CAL[59], CAL[60], CAL[100], CAL[120]], CAL, 60) == [True, False, True, False, True]


def market(extra=(), at=400):
    data = {"XIU.TO": flat(), "B.TO": bars(shape(at)), **{f"F{k}.TO": flat() for k in range(5)}}
    data.update(extra)
    return bt.Market(data)


def test_events_are_the_detections_that_are_eligible():
    ev, mask = bt.find_events(market())
    assert ev[["ticker", "date"]].values.tolist() == [["B.TO", CAL[400]]]
    e = ev.iloc[0]
    assert e["bars"] == 80 and e["level"] == pytest.approx(100.2) and e["close"] == 104 and e["low"] == pytest.approx(88 * .998)
    assert e["target"] == pytest.approx(200.4 - 88 * .998) and mask.to_numpy().sum() == 1
    early, mask = bt.find_events(market(at=200))
    assert early.empty and mask.at[CAL[200], "B.TO"]


def test_same_day_means():
    up = flat()
    up.loc[CAL[401]:, ["Open", "High", "Low", "Close"]] *= 1.0
    up.loc[CAL[402]:, ["Open", "High", "Low", "Close"]] *= 1.2          # +20% after the entry open
    nh = np.full(900, 100.0)
    nh[400:] = 100.5                                                   # a plain new high on the day
    nh[402:] = 100.5 * 1.05
    m = market({"UP.TO": up, "NH.TO": bars(nh)})
    ev, mask = bt.find_events(m)
    out = bt.add_returns(ev, mask, m)
    e = out.iloc[0]
    assert e["r60"] == pytest.approx(0.0) and e["x60"] == 0.0
    assert e["a60"] == pytest.approx((0.2 + 0.05 + 0 * 5) / 7)         # the event itself is not in its own comparison
    assert e["b60"] == pytest.approx(0.05)                             # only the plain new high


def test_horizon_not_over_is_missing():
    ev, mask = bt.find_events(market(at=870))
    out = bt.add_returns(ev, mask, market(at=870))
    assert np.isfinite(out["r20"].iloc[0]) and np.isnan(out["r60"].iloc[0]) and np.isnan(out["r120"].iloc[0])
    m = market(at=838)                                                 # entry 839, exit 899: the last bar
    ev, mask = bt.find_events(m)
    assert np.isfinite(bt.add_returns(ev, mask, m)["r60"].iloc[0])
    m = market(at=839)
    ev, mask = bt.find_events(m)
    assert np.isnan(bt.add_returns(ev, mask, m)["r60"].iloc[0])


def test_summary_by_hand():
    ev = pd.DataFrame({"ticker": ["A", "B", "A", "C"], "date": pd.to_datetime(["2020-01-06", "2020-06-01", "2021-01-04", "2021-06-01"]),
                       "r60": [0.10, -0.05, 0.02, np.nan], "a60": [0.02, 0.01, 0.03, 0.0], "b60": [0.01, 0.0, 0.0, 0.0], "x60": [0.03, 0.0, 0.0, 0.0]})
    s = bt.summary(ev)
    assert s["events"] == 3 and s["tickers"] == 2 and s["mean"] == pytest.approx(0.07 / 3) and s["median"] == 0.02
    assert s["win"] == pytest.approx(2 / 3) and s["ex_a"] == pytest.approx((0.08 - 0.06 - 0.01) / 3)
    assert s["ex_b"] == pytest.approx((0.09 - 0.05 + 0.02) / 3) and s["ex_a_median"] == pytest.approx(-0.01)
    assert s["net_vs_xiu"] == pytest.approx((0.07 - 0.05 + 0.02) / 3 - 0.005) and s["years_ahead"] == 0.5


def test_block_interval():
    d = pd.Series(pd.date_range("2015-01-05", periods=400, freq="7D"))
    x = pd.Series(np.random.default_rng(0).normal(0.02, 0.1, 400))
    lo, hi = bt.block_interval(x, d)
    assert lo < x.mean() < hi and hi - lo == pytest.approx(4 * 0.1 / 20, rel=0.35)
    clumped = pd.Series(np.repeat(np.random.default_rng(1).normal(0.02, 0.1, 31), 13)[:400])
    lo2, hi2 = bt.block_interval(clumped, d)
    assert hi2 - lo2 > 2 * (hi - lo)
    x[5] = np.nan
    assert np.isfinite(bt.block_interval(x, d)).all()


GOOD = {"ex_a": .02, "a_lo": .001, "ex_b": .02, "b_lo": .001, "years_ahead": .6, "net_vs_xiu": .01, "events": 100, "tickers": 50}


def test_verdict():
    assert bt.verdict(GOOD, 0.8)["passed"]
    for k, v in [("a_lo", -.001), ("b_lo", -.001), ("ex_a", -.1), ("years_ahead", .59), ("net_vs_xiu", 0.0), ("events", 99), ("tickers", 49)]:
        assert not bt.verdict({**GOOD, k: v}, 0.8)["passed"], k
    assert not bt.verdict(GOOD, 0.79)["passed"]


def test_sensitivity_rules():
    r = bt.sensitivity_rules()
    assert len(r) == 17 and r["margin +20%"].margin == pytest.approx(0.036) and r["min_bars -20%"].min_bars == 42
    assert r["pivot -20%"].pivot == 4 and r["pivot +20%"].pivot == 6 and r["min_tests +20%"].min_tests == 3 and "min_tests -20%" not in r
    assert all(v != DEFAULT for v in r.values())


# ── The trade ────────────────────────────────────────────────────────────────

def tb(rows):
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])


@pytest.mark.parametrize("rows,exit_,why", [
    ([(100, 101, 99, 100), (100, 112, 99, 111)], 110, "target"),
    ([(100, 101, 99, 100), (115, 116, 114, 115)], 115, "target"),      # gap through
    ([(100, 101, 94, 96)], 95, "stop"),
    ([(100, 101, 99, 100), (90, 91, 89, 90)], 90, "stop"),             # gap through
    ([(100, 111, 94, 100)], 95, "stop"),                               # both in one bar
    ([(100, 101, 99, 100), (100, 101, 99, 100.5)], 100.5, "open"),
])
def test_trade_exits(rows, exit_, why):
    x = bt.trade(tb(rows), 0, 95, 110)
    assert x["exit"] == exit_ and x["why"] == why and x["r"] == pytest.approx((exit_ - 100) / 5) and x["ret"] == pytest.approx(exit_ / 100 - 1)


def test_trade_times_out_at_the_close():
    x = bt.trade(tb([(100, 101, 99, 100 + k / 10) for k in range(10)]), 0, 95, 110, max_bars=5)
    assert x["why"] == "time" and x["exit"] == 100.4


def test_no_trade_outside_stop_and_target():
    assert bt.trade(tb([(94, 95, 93, 94)]), 0, 95, 110) is None and bt.trade(tb([(111, 112, 110, 111)]), 0, 95, 110) is None
    assert bt.trade(tb([(100, 101, 99, 100)]), 1, 95, 110) is None


@pytest.mark.parametrize("lows,kind", [((101, 101), "1"), ((99, 101), "2"), ((94, 101), "3"), ((79, 101), "4"), ((101, 102), "1"), ((100, 101), "2")])
def test_types(lows, kind):
    rows = [(102, 103, lows[0], 102), (102, 111, lows[1], 110)]
    assert bt.trade(tb(rows), 0, 95, 110, level=100, floor=80)["type"] == kind


def test_type_unresolved():
    assert bt.trade(tb([(102, 103, 101, 102)] * 3), 0, 95, 110, level=100, floor=80)["type"] == "unresolved"


def test_trades_and_peers():
    """The event drifts up 0.1% a day from the open after the signal; the peers do nothing."""
    m = market({f"P{k}.TO": bars(np.full(900, 50.0)) for k in range(30)} | {"B.TO": bars(shape(400, 0.001))})
    ev, _ = bt.find_events(m)
    x = bt.trades(ev, m).iloc[0]
    assert x["entry"] == pytest.approx(104 * 1.001) and x["why"] == "target" and x["type"] == "1"
    assert x["r"] == pytest.approx((ev["target"].iloc[0] - 104.104) / (104.104 - ev["lfd_low"].iloc[0]))
    assert x["peer_r"] == pytest.approx(0.0, abs=1e-9) and x["peer_hit"] == 0


def test_peers_get_the_same_stop_and_target_in_atrs():
    """Peers rise 15% at once: far past the 8% target, which they sell at the open."""
    c = np.full(900, 50.0)
    c[405:] = 57.5
    m = market({f"{p}{k}.TO": bars(c) for p in "PF" for k in range(30)})
    ev, _ = bt.find_events(m)
    x = bt.trades(ev, m).iloc[0]
    atr, own = m.atr.at[CAL[400], "B.TO"], m.atr.at[CAL[400], "P0.TO"]
    down = (1 - ev["lfd_low"].iloc[0] / 104) / atr * own
    assert x["peer_hit"] == 1 and x["peer_ret"] == pytest.approx(0.15) and x["peer_r"] == pytest.approx(0.15 / down)


def test_trades_still_open_are_left_out():
    ev, _ = bt.find_events(market(at=700))
    assert len(ev) == 1 and bt.trades(ev, market(at=700)).empty


def test_ineligible_names_are_in_no_comparison():
    nh = np.full(900, 100.0)
    nh[400:] = 100.5
    nh[402:] = 300
    m = market({"THIN.TO": bars(nh, vol=100)})
    ev, mask = bt.find_events(m)
    out = bt.add_returns(ev, mask, m).iloc[0]
    assert out["a60"] == pytest.approx(0.0) and np.isnan(out["b60"])


def test_interval_is_95_percent():
    d = pd.Series(pd.date_range("2015-01-05", periods=2000, freq="7D"))
    x = pd.Series(np.random.default_rng(0).normal(0.0, 0.1, 2000))
    lo, hi = bt.block_interval(x, d)
    assert hi - lo == pytest.approx(2 * 1.96 * 0.1 / 2000 ** 0.5, rel=0.2)


# ── End to end ───────────────────────────────────────────────────────────────

def many(after, n=40):
    rng = np.random.default_rng(3)
    data = {"XIU.TO": flat()}
    for k in range(n):
        at = 300 + 12 * k
        noise = np.exp(np.cumsum(rng.normal(0, 0.004, len(CAL))))
        data[f"B{k}.TO"] = bars(np.where(np.arange(900) <= at, shape(at), shape(at, after) * noise / noise[at]))
        data[f"F{k}.TO"] = bars(100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(CAL)))))
    return bt.Market(data)


def test_a_planted_edge_is_found_and_no_edge_is_not():
    ev, _ = bt.run(many(0.002), DEFAULT, None, None)
    s = bt.summary(ev)
    assert s["events"] >= 35 and s["ex_a"] > 0.05 and s["a_lo"] > 0
    ev, _ = bt.run(many(0.0), DEFAULT, None, None)
    s = bt.summary(ev)
    assert s["a_lo"] < 0.005 and abs(s["ex_a"]) < 0.02


def test_run_keeps_to_its_dates():
    m = many(0.0)
    a, _ = bt.run(m, DEFAULT, None, "2020-12-31")
    b, _ = bt.run(m, DEFAULT, "2021-01-01", None)
    every, _ = bt.run(m, DEFAULT, None, None)
    assert len(a) and len(b) and len(a) + len(b) == len(every) and a["date"].max() < b["date"].min()


def test_later_bars_do_not_change_events():
    m = many(0.0)
    cut = CAL[600]
    short = bt.Market({"XIU.TO": flat().loc[:cut], **{t: d.loc[:cut] for t, d in m.bars.items()}})
    full, _ = bt.find_events(m)
    part, _ = bt.find_events(short)
    pd.testing.assert_frame_equal(full[full["date"] <= cut].reset_index(drop=True), part)


def test_development_returns_end_before_the_holdout_starts():
    gap = len(pd.bdate_range(bt.DEV_END, bt.HOLD_START)) - 2
    assert gap > max(bt.HORIZONS) + 1


# ── Download adapter ─────────────────────────────────────────────────────────

def test_sync_unadjusted(tmp_path):
    from research.vcp import prices as vp
    asked = []

    def fetch(tickers, period, interval, auto_adjust):
        asked.append((tickers, period, interval, auto_adjust))
        d = flat().iloc[:10].copy()
        d.iloc[3, 0] = np.nan
        d.iloc[4, 3] = 0.0
        return {"A.TO": d}, {"B.TO": "no_data"}
    conn = vp.connect(str(tmp_path / "x.db"))
    prices.sync_unadjusted(["A.TO", "B.TO"], str(CAL[1].date()), str(CAL[8].date()), conn=conn, fetch=fetch)
    got = vp.load_all(conn)
    conn.close()
    assert asked == [(["A.TO", "B.TO"], "15y", "1d", False)]
    assert list(got) == ["A.TO"] and got["A.TO"].index.tolist() == [CAL[k] for k in (1, 2, 5, 6, 7, 8)]


def test_history_before_an_unadjusted_consolidation_is_dropped():
    c = np.array([10, 10, 25, 24, 8, 8, 19.9, 20.0])
    assert prices.after_seam(bars(c)).index[0] == CAL[2]
    assert len(prices.after_seam(bars(np.array([10, 2, 2, 4.9])))) == 4
    assert prices.after_seam(bars(np.array([1, 3, 1, 3, 3]))).index[0] == CAL[3]


def test_recent_lists_only_recent_liquid_breakouts():
    from research.breakouts.recent import recent
    data = {"XIU.TO": flat(), "NEW.TO": bars(shape(898)), "OLD.TO": bars(shape(700)), "THIN.TO": bars(shape(898), vol=100)}
    out = recent(data, 5)
    assert out["ticker"].tolist() == ["NEW.TO"] and out["date"].iloc[0] == CAL[898].date().isoformat()
    assert out["line"].iloc[0] == 100.2 and out["confirm_at"].iloc[0] == pytest.approx(103.206) and out["above_pct"].iloc[0] == 3.8
    assert recent(data, 2)["ticker"].tolist() == ["NEW.TO"] and recent(data, 1).empty
    assert recent(data, 5, ["THIN.TO"])["ticker"].tolist() == ["THIN.TO"]
