"""Sleeves as trades: hand-checked entries and exits, yardsticks. No network."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.breakouts import backtest as bt
from research.us import sleeves as sl

CAL = pd.bdate_range("2019-01-01", periods=600)


def bars(rows, n0=0, base=(100, 101, 99, 100)):
    o, h, l, c = zip(*([base] * n0 + list(rows)))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1e6}, index=CAL[:n0 + len(rows)], dtype=float)


def test_held():
    d = bars([(100, 101, 99, 100), (102, 104, 101, 103), (103, 112, 102, 110), (110, 111, 100, 101)])
    assert sl.held(d, 1, 2) == {"k": 1, "end": 3, "entry": 102, "exit": 101}
    x = sl.held(d, 1, 2, 0.05)
    assert x["end"] == 2 and x["exit"] == pytest.approx(107.1)
    assert sl.held(bars([(100, 101, 99, 100), (100, 106, 99, 105)]), 1, 5, 0.05)["exit"] == pytest.approx(105)
    assert sl.held(bars([(100, 101, 99, 100), (100, 101, 99, 100), (110, 111, 109, 110)]), 1, 5, 0.05)["exit"] == 110        # gap over
    assert sl.held(d, 4, 2) is None and sl.held(d, 2, 99)["end"] == 3


@pytest.fixture
def tail(monkeypatch):
    def put(d, at, high=102.0, low=98.0):
        monkeypatch.setattr(sl, "scan_history", lambda *_: [SimpleNamespace(date=d.index[at], high=high, low=low)])
    return put


OK = np.ones(600, bool)
Q = (100, 101, 99, 100)


@pytest.mark.parametrize("after,k,end,entry,exit", [
    ([(101, 103, 100.5, 102), (103, 109, 102, 108)], 1, 2, 102, 108), ([(101, 103, 100.5, 102), (101, 102, 97, 98)], 1, 2, 102, 98),
    ([(101, 103, 97, 99)], 1, 1, 102, 98), ([(101, 103, 100.5, 102), (95, 110, 94, 100)], 1, 2, 102, 95), ([(101, 103, 100.5, 102)] + [(101, 102.5, 100.5, 101)] * 14, 1, 11, 102, 101),
    ([(103, 104, 102.5, 103), (103, 104, 102.5, 103)], 1, 2, 103, 103)])
def test_kangaroo(tail, after, k, end, entry, exit):
    d = bars(after, n0=51)
    tail(d, 50)
    x, = sl.kangaroo(d, OK)
    assert (x["k"], x["end"], x["entry"], x["exit"]) == (50 + k, 50 + end, pytest.approx(entry), pytest.approx(exit))


def test_kangaroo_expires_and_needs_eligibility(tail):
    d = bars([Q] * 10 + [(101, 103, 100, 102)] * 3, n0=51)
    tail(d, 50)
    assert sl.kangaroo(d, OK) == []
    d = bars([Q] * 9 + [(101, 103, 100, 102)] * 3, n0=51)
    tail(d, 50)
    assert len(sl.kangaroo(d, OK)) == 1 and sl.kangaroo(d, np.zeros(600, bool)) == []


def flat_to(prices):
    c = np.asarray(prices, float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6}, index=CAL[:len(c)])


def test_conviction():
    c = [100.0] * 260 + [81, 79.9, 79, 100, 120, 91, 90, 88, 88]
    d = flat_to(c)
    x, = sl.conviction(d, OK)
    assert x["t"] == 261 and x["k"] == 262 and x["entry"] == 79 and x["end"] == 267 and x["exit"] == 88 and x["rank"] == pytest.approx(.201)
    y, = sl.conviction(flat_to([100.0] * 260 + [79, 78, 85, 84]), OK)                        # never stopped: valued at the end
    assert y["end"] == 263 and y["exit"] == 84


def test_frame_and_same_days():
    data = {"XIU.TO": flat_to([10.0] * 400), "A": flat_to([100.0] * 300 + [110.0] * 100), "B": flat_to([50.0] * 310 + [100.0] * 90)}
    m = bt.Market(data)
    rank = pd.DataFrame(.5, index=m.cal, columns=m.close.columns)
    tr = sl.frame([{"ticker": "A", "t": 299, "k": 300, "end": 320, "entry": 110.0, "exit": 121.0, "stop": 104.5}], m, rank)
    r = tr.iloc[0]
    assert r["net"] == pytest.approx(.1) and r["r"] == pytest.approx(2.0) and r["bars"] == 21 and r["legs"] == [(CAL[320], 1.0, 121.0)]
    assert sl.same_days(tr, m)[0] == pytest.approx((0.0 + 1.0) / 2)


def test_momentum_top10_holds_last_months_winners():
    data = {"XIU.TO": flat_to([10.0] * 600)}
    for k in range(12):
        data[f"S{k}"] = flat_to(100 * 1.001 ** (np.arange(600) * k / 6))
    m = bt.Market(data)
    s = sl.momentum_top10(m)
    s = s[s.index >= CAL[300]]
    r = (m.last / m.last.shift(1) - 1)[[f"S{k}" for k in range(2, 12)]].mean(axis=1)
    assert s.iloc[0:200].mean() == pytest.approx(r.loc[s.index[0:200]].mean(), rel=.05)


def test_better_needs_all_three():
    t = pd.DataFrame({p: {"ex_lo": .01, "cagr": .2, "xiu_cagr": .1} for p in ("A", "B", "C", "ALL")}).T
    assert sl.better(t)
    for p, c, v in (("ALL", "ex_lo", -.01), ("ALL", "cagr", .05)):
        u = t.copy(); u.at[p, c] = v
        assert not sl.better(u)
    u = t.copy(); u.at["A", "cagr"] = u.at["B", "cagr"] = .05
    assert not sl.better(u)
