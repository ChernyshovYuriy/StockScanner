"""US test: eligibility, membership, verdicts; the triangle detector. No network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.breakouts import backtest as bt
from research.breakouts import triangle as tri
from research.breakouts.detector import swing_highs
from research.us import backtest as us

CAL = pd.bdate_range("2019-01-01", periods=600)


def flat(price=100.0, vol=1e6, n=600):
    c = np.full(n, price)
    return pd.DataFrame({"Open": c, "High": c + .1, "Low": c - .1, "Close": c, "Volume": vol}, index=CAL[:n])


# ── Market ───────────────────────────────────────────────────────────────────

def test_us_eligibility():
    m = us.Market({"SPY": flat(), "OK": flat(), "CHEAP": flat(4.99, 1e8), "THIN": flat(100, 99_000), "EDGE": flat(5.0, 2e6)})
    e = m.eligible
    assert not e["OK"].iloc[250] and e["OK"].iloc[251:].all() and e["EDGE"].iloc[251:].all()
    assert not e["CHEAP"].any() and not e["THIN"].any() and "SPY" not in e and m.xiu_close.equals(flat()["Close"])


def test_members_only_while_members():
    mem = pd.DataFrame({"ticker": ["OK", "OK", "GONE"], "start": ["2000-01-01", "2020-10-01", "2000-01-01"], "end": ["2020-03-02", "", "2015-01-01"]})
    e = us.Market({"SPY": flat(), "OK": flat(), "NEVER": flat()}, mem).eligible
    assert not e["NEVER"].any()
    for d, want in (("2020-02-28", True), ("2020-03-02", False), ("2020-09-30", False), ("2020-10-01", True)):
        assert e.at[pd.Timestamp(d), "OK"] == want, d


def test_worth_keeping_never_drops_an_eligible_name():
    g = np.random.default_rng(0)
    for k in range(40):
        d = flat(float(g.uniform(3, 8)), float(g.uniform(.2e6, 6e6)))
        d = d.iloc[int(g.integers(0, 400)):]
        if us.Market({"SPY": flat(), "A": d}).eligible["A"].any():
            assert us.worth_keeping(d)
    assert not us.worth_keeping(flat(n=251)) and not us.worth_keeping(flat(4.0, 1e9)) and not us.worth_keeping(flat(100, 1_000))


def test_difference_by_hand():
    e = pd.Series([.1, .2, .05, .05])
    d, lo, hi = us.difference(e, pd.Series([True, True, False, False]), pd.Series(pd.to_datetime(["2020-01-01"] * 4)))
    assert d == pytest.approx(.1) and lo == pytest.approx(.1) and hi == pytest.approx(.1)


# ── Verdicts ─────────────────────────────────────────────────────────────────

def table(**k):
    base = dict(net=.02, lo=.01, ex=.01, ex_lo=.001, cagr=.2, xiu_cagr=.1)
    t = pd.DataFrame({p: dict(base) for p in ("A", "B", "C", "ALL")}).T
    for (p, c), v in k.get("set", {}).items():
        t.at[p, c] = v
    return t


@pytest.mark.parametrize("change,ok", [({}, True), ({("ALL", "lo"): -.1}, False), ({("ALL", "ex_lo"): -.1}, False), ({("ALL", "cagr"): .05}, False),
                                       ({("A", "cagr"): .05}, True), ({("A", "cagr"): .05, ("B", "cagr"): .05}, False), ({("A", "lo"): -.1}, True)])
def test_h1(change, ok):
    assert us.h1(table(set=change))["passed"] is ok


def ev(n=600, asc=.04, other=0.0, skew=False, seed=0):
    g = np.random.default_rng(seed)
    kind = np.where(g.random(n) < .5, "asc", "flat")
    r = g.normal(0, .05, n) + np.where(kind == "asc", asc, other)
    if skew:
        r = np.where(kind == "asc", np.where(g.random(n) < .1, 2.0, -.02), r)
    return pd.DataFrame({"kind": kind, "vol_x": 2.0, "date": np.sort(g.choice(CAL, n)), "r60": r, "a60": 0.0})


def test_h2_needs_everything():
    assert us.h2(ev())["passed"]
    assert not us.h2(ev(asc=0))["passed"]                       # no excess
    assert not us.h2(ev(other=.04))["passed"]                   # others do as well
    s = us.h2(ev(skew=True))
    assert s["ex"] > 0 and s["median"] < 0 and not s["passed"]  # a few monsters
    quiet = ev()
    quiet.loc[quiet["kind"] == "asc", "vol_x"] = np.where(np.arange((quiet["kind"] == "asc").sum()) % 2, 1.49, 1.5)
    assert us.h2(quiet)["n"] < us.h2(ev())["n"]


# ── Triangle ─────────────────────────────────────────────────────────────────

def shape(lows=(88, 92), top=100.0, end=102.0, gap=20, n=300, vx=1.0):
    """`lows` between and after the touches of `top`; closes at `end` on the last bar."""
    k = len(lows)
    xs, ys = [0], [60.0]
    at = n - 1 - gap * k - 2
    for low in lows:
        xs += [at, at + gap // 2]
        ys += [top, low]
        at += gap
    xs += [n - 2, n - 1]
    ys += [99.0, end]
    c = np.interp(np.arange(n), xs, ys)
    v = np.full(n, 1000.0)
    v[-1] = 1000 * vx
    return c + .001, c - .001, c, v


def found(*a, t=None, **k):
    h, l, c, v = shape(*a, **k)
    t = len(c) - 1 if t is None else t
    return tri.detect(h, l, c, v, t, swing_highs(h, 3))


@pytest.mark.parametrize("lows,kind", [((88, 92), "asc"), ((88, 88.2), "flat"), ((92, 88), "fall"), ((85, 88, 92), "asc"), ((88, 86, 92), "flat"), ((88, 88.88), "asc"), ((88, 87.5), "flat")])
def test_kinds(lows, kind):
    x = found(lows)
    assert x["kind"] == kind and x["tests"] == len(lows) and x["level"] == pytest.approx(100.001) and x["first"] == pytest.approx(lows[0], abs=.01)


def test_limits():
    assert found(end=101.02) and not found(end=100.9)            # 1% over the level
    assert not found(end=116.0)                                  # not the day it broke out
    assert found(lows=(88, 92), gap=10, n=300) and not found(lows=(88, 92), gap=8)             # at least 20 sessions
    assert not found(lows=(88, 92), gap=64)                      # at most 126
    assert not found(lows=(97.5, 98))                            # no pullback between the touches
    assert found(vx=2.0)["vol_x"] == pytest.approx(2.0)


@pytest.mark.parametrize("field,value,spoils", [("c", 101.5, True), ("c", 100.9, False), ("h", 102.5, True), ("h", 101.9, False)])
def test_one_bar_above_inside(field, value, spoils):
    h, l, c, v = shape()
    k = len(c) - 1 - 42 + 8                                      # in the first dip
    if field == "c":
        c[k] = value
    h[k] = max(value, c[k])
    x = tri.detect(h, l, c, v, len(c) - 1, swing_highs(h, 3))
    assert (x is None) is spoils


def test_touches_are_within_two_percent():
    h, l, c, v = shape()
    k = len(c) - 1 - 42                                          # the first touch lowered to 97.5
    h[k - 12:k + 8] = np.minimum(h[k - 12:k + 8], 97.5)
    c[k - 12:k + 8] = np.minimum(c[k - 12:k + 8], 97.5)
    assert tri.detect(h, l, c, v, len(c) - 1, swing_highs(h, 3)) is None


def test_triangle_does_not_look_ahead():
    n = 0
    for seed in range(25):
        g = np.random.default_rng(seed)
        c = 50 * np.exp(np.cumsum(g.normal(0.0005, .012, 900)))
        h, l, v = c * (1 + g.uniform(0, .01, 900)), c * (1 - g.uniform(0, .01, 900)), g.uniform(1, 2, 900)
        p = swing_highs(h, 3)
        for t in range(150, 900, 3):
            a = tri.detect(h, l, c, v, t, p)
            assert a == tri.detect(h[:t + 1], l[:t + 1], c[:t + 1], v[:t + 1], t, swing_highs(h[:t + 1], 3))
            n += a is not None
    assert n > 20


def test_find_events_eligible_and_separate():
    h, l, c, v = shape(n=300)
    d = pd.DataFrame({"Open": c, "High": h, "Low": l, "Close": c, "Volume": v * 1e4}, index=CAL[:300])
    thin = d.assign(Volume=1.0)
    m = bt.Market({"XIU.TO": flat(n=300), "A.TO": d, "B.TO": thin})
    e, mask = tri.find_events(m)
    assert e["ticker"].tolist() == ["A.TO"] and e["date"].iloc[0] == CAL[299] and mask.to_numpy().sum() == 1
