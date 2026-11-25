"""The evening Minervini scan and its dashboard tab. No network, no real database, no mail."""
from __future__ import annotations

import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import minervini_dashboard_data as dash
from research.minervini import method as mm
from research.minervini import scan

CAL = pd.bdate_range(end="2026-11-25", periods=420)


# ── The week's order ─────────────────────────────────────────────────────────

def bars(after, n=60):
    """`after`: (o, h, l, c) of the sessions after the signal bar n-1. Pivot 100, rest low 96."""
    rows = [(98, 99, 97, 98)] * n + list(after)
    o, h, l, c = (np.array(x, float) for x in zip(*rows))
    return {"Open": o, "High": h, "Low": l, "Close": c, "ma20": c, "ma50": c}


Q = (98, 99.5, 97, 98)
OK = np.ones(200, bool)


@pytest.mark.parametrize("after,want", [
    ([], ("waiting", 59, None)), ([Q] * 4, ("waiting", 63, None)), ([Q] * 5, ("expired", 64, None)), ([Q] * 6, ("expired", 64, None)),
    ([Q, (99, 101, 98, 100)], ("bought", 61, 100.0)), ([Q] * 4 + [(99, 101, 98, 100)], ("bought", 64, 100.0)),
    ([Q] * 5 + [(99, 101, 98, 100)], ("expired", 64, None)), ([(103, 104, 102, 103)], ("bought", 60, 103.0)),
    ([(105, 106, 104, 105)], ("bought", 60, 105.0)), ([(105.1, 106, 104, 105)], ("ran", 60, None)), ([(99, 100, 98, 99)], ("waiting", 60, None))])
def test_order_states(after, want):
    assert mm.order(bars(after), OK, 59, 100.0, 96.0, False) == want


def test_cancel_variant():
    low = [Q, (97, 98, 95.9, 96), Q]
    assert mm.order(bars(low), OK, 59, 100.0, 96.0, True) == ("cancelled", 61, None)
    assert mm.order(bars(low), OK, 59, 100.0, 96.0, False)[0] == "waiting"
    ok = OK.copy()
    ok[60] = False
    assert mm.order(bars([Q, Q]), ok, 59, 100.0, 96.0, True) == ("cancelled", 60, None)
    assert mm.order(bars([(99, 101, 95, 100)]), OK, 59, 100.0, 96.0, True)[0] == "bought"


def test_near_is_three_percent():
    assert mm.near(102.9, 100.0) and not mm.near(103.1, 100.0)


def test_week_candidates_one_per_name_and_pivot():
    a = {"Close": np.full(60, 98.0)}
    setups = {10: (100., 95.), 11: (100., 95.), 12: (99., 95.), 15: (99., 95.), 20: (120., 95.), 30: (100., 95.), 40: (99.5, 95.)}
    got = mm.week_candidates(a, setups, lambda t, p, l: t + 5)
    assert [(t, p) for t, p, _ in got] == [(10, 100.), (15, 99.), (40, 99.5)]           # 12 too soon, 20 too far, 30 the same pivot


# ── What is followed ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("after,state,bar", [
    ([Q], "waiting", None), ([(99, 101, 98, 100)], "bought", 60), ([(99, 101, 98, 100), (100, 101, 95, 96)], "stopped", 61),
    ([(99, 101, 98, 100), (110, 121, 109, 118)], "target", 61), ([Q] * 5, "expired", 64), ([(106, 107, 105, 106)], "ran", 60)])
def test_follow(after, state, bar):
    x = scan.follow(bars(after), 59, 100.0, 96.0)
    assert (x["state"], x["bar"]) == (state, bar)
    assert x["stop"] == pytest.approx(95.904) and x["target"] == pytest.approx(120)


def test_follow_uses_the_actual_fill():
    x = scan.follow(bars([(104, 105, 103.9, 104)]), 59, 100.0, 90.0)
    assert x["entry"] == 104 and x["stop"] == pytest.approx(104 * .92) and x["target"] == pytest.approx(124.8)


def test_levels():
    assert scan.levels(100, 96) == pytest.approx((95.904, 120)) and scan.levels(100, 80) == pytest.approx((92, 120))


# ── The evening run ──────────────────────────────────────────────────────────

def frame(close, vol):
    c = np.asarray(close, float)
    return pd.DataFrame({"Open": c, "High": c * 1.001, "Low": c * .999, "Close": c, "Volume": np.asarray(vol, float)}, index=CAL[-len(c):])


def leader(n=420, rest=8, dip=.05, gap=.02):
    """Steady rise with a 12% shake-out, a top, then a quiet rest `gap` under it."""
    up = 20 * 1.004 ** np.arange(n - rest)
    up[-40:-30] *= np.linspace(1, .88, 10)
    up[-30:-20] *= np.linspace(.88, 1, 10)
    top = up[-1]
    tail = np.full(rest, top * (1 - gap))
    tail[rest // 2] = top * (1 - dip)
    vol = np.full(n, 1e6)
    vol[-6:] = 7e5
    return frame(np.r_[up, tail], vol)


def market(**names):
    flat = {f"F{k}.TO": frame(50 + np.sin(np.arange(420) / (5 + k)), np.full(420, 1e6)) for k in range(6)}
    return {"XIU.TO": frame(30 * 1.0003 ** np.arange(420), np.full(420, 1e6)), **flat, **names}


class Mail:
    def __init__(self, ok=True):
        self.ok, self.sent = ok, []

    def __call__(self, s, b):
        self.sent.append((s, b))
        return self.ok


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    now = {"at": datetime(2026, 11, 25, 18, 20)}
    monkeypatch.setattr(scan, "market_now", lambda: now["at"])
    return now


def put(tmp_path, data, session="2026-11-25"):
    cut = {t: d.loc[:session] for t, d in data.items()}
    pd.to_pickle({"session": session, "bars": cut}, tmp_path / "bars.pkl")


def go(tmp_path, mail, **k):
    return scan.auto(path=tmp_path / "m.db", bars_path=tmp_path / "bars.pkl", send=mail, **k)


def rows(tmp_path):
    c = scan.connect(tmp_path / "m.db")
    out = pd.read_sql("SELECT * FROM candidates ORDER BY signal, ticker", c)
    c.close()
    return out


def extend(d, closes, vol=7e5):
    idx = pd.bdate_range(d.index[-1], periods=len(closes) + 1)[1:]
    c = np.asarray(closes, float)
    return pd.concat([d, pd.DataFrame({"Open": c, "High": c * 1.001, "Low": c * .999, "Close": c, "Volume": float(vol)}, index=idx)])


def all_to(data, day):
    n = len(pd.bdate_range("2026-11-25", day)) - 1
    return {t: (d if d.index[-1] >= pd.Timestamp(day) else extend(d, [d["Close"].iloc[-1]] * n, 1e6)) for t, d in data.items()}


def at(clock, tmp_path, data, day, last):
    clock["at"] = datetime.fromisoformat(day + " 18:20")
    put(tmp_path, all_to(data, last), day)


def test_a_candidate_is_sent_once(tmp_path):
    put(tmp_path, market(**{"A.TO": leader()}))
    mail = Mail()
    assert go(tmp_path, mail) == 0 and go(tmp_path, mail) == 0
    (subject, body), = mail.sent
    r, = rows(tmp_path).itertuples()
    assert subject == "Minervini candidates 2026-11-25: A.TO"
    assert f"buy above {r.pivot:,.2f}" in body and f"LOSS POINT   {r.stop:,.2f}" in body and f"TAKE PROFIT  {r.target:,.2f}" in body
    assert "30% of those won" in body and r.state == "waiting" and r.expires == "2026-12-02"
    assert r.pivot == pytest.approx(leader()["High"].iloc[-9]) and r.target == pytest.approx(1.2 * r.pivot)


def test_no_candidate_sends_nothing(tmp_path):
    mail = Mail()
    put(tmp_path, market())
    assert go(tmp_path, mail) == 0 and mail.sent == []


@pytest.mark.parametrize("kw", [dict(gap=.04), dict(rest=4), dict(dip=.15)])
def test_not_a_candidate(tmp_path, kw):
    mail = Mail()
    put(tmp_path, market(**{"A.TO": leader(**kw)}))
    go(tmp_path, mail)
    assert mail.sent == []


def test_stale_name_is_not_a_candidate(tmp_path):
    mail = Mail()
    put(tmp_path, market(**{"A.TO": leader().iloc[:-1]}))
    go(tmp_path, mail)
    assert mail.sent == []


def test_the_week_is_followed_without_more_mail(tmp_path, clock):
    mail, a = Mail(), leader()
    put(tmp_path, market(**{"A.TO": a}))
    go(tmp_path, mail)
    p = rows(tmp_path)["pivot"].iloc[0]
    a = extend(a, [p * .985, p * 1.01])
    at(clock, tmp_path, market(**{"A.TO": a}), "2026-11-26", "2026-11-27")
    go(tmp_path, mail)
    assert rows(tmp_path)["state"].tolist() == ["waiting"]
    at(clock, tmp_path, market(**{"A.TO": a}), "2026-11-27", "2026-11-27")
    go(tmp_path, mail)
    r = rows(tmp_path).iloc[0]
    assert (r["state"], r["state_date"]) == ("bought", "2026-11-27")
    assert r["entry"] == pytest.approx(p * 1.01) and r["last"] == pytest.approx(p * 1.01) and len(mail.sent) == 1


def test_after_its_week_a_row_is_left_alone(tmp_path, clock):
    mail, a = Mail(), leader()
    put(tmp_path, market(**{"A.TO": a}))
    go(tmp_path, mail)
    p = rows(tmp_path)["pivot"].iloc[0]
    a = extend(a, [p * .985] * 5 + [p * 1.02])
    for day in ("2026-12-02", "2026-12-03"):
        at(clock, tmp_path, market(**{"A.TO": a}), day, "2026-12-03")
        go(tmp_path, mail)
        r = rows(tmp_path).iloc[0]
        assert r["state"] == "expired" and r["last_date"] == "2026-12-02"


def test_a_split_is_followed(tmp_path, clock):
    a = leader()
    put(tmp_path, market(**{"A.TO": a}))
    go(tmp_path, Mail())
    p = rows(tmp_path)["pivot"].iloc[0]
    b = extend(a, [p * 1.01])
    b[["Open", "High", "Low", "Close"]] /= 2
    at(clock, tmp_path, market(**{"A.TO": b}), "2026-11-26", "2026-11-26")
    go(tmp_path, Mail())
    r = rows(tmp_path).iloc[0]
    assert r["pivot"] == pytest.approx(p / 2) and r["state"] == "bought" and r["entry"] == pytest.approx(p * 1.01 / 2)


def test_is_new_rules(tmp_path):
    conn = scan.connect(tmp_path / "m.db")

    def add(signal, pivot, state):
        conn.execute("DELETE FROM candidates")
        conn.execute("INSERT INTO candidates (ticker, signal, pivot, low, close, state, stop, target, found_at) "
                     "VALUES ('A.TO', ?, ?, 1, 1, ?, 1, 1, '')", (signal, pivot, state))

    c = {"ticker": "A.TO", "signal": "2026-11-25", "pivot": 100.0}
    day = lambda k: CAL[-1 - k].date().isoformat()
    for k, pivot, state, want in [(3, 90, "waiting", False), (4, 90, "bought", False), (4, 90, "stopped", False), (5, 90, "bought", True),
                                  (3, 90, "expired", True), (3, 90, "ran", True), (29, 100, "expired", False), (30, 100, "expired", True),
                                  (10, 100.05, "expired", True), (10, 100.00001, "expired", False)]:
        add(day(k), pivot, state)
        assert scan.is_new(conn, CAL, c) is want, (k, pivot, state)
    add(day(3), 90, "waiting")
    assert scan.is_new(conn, CAL, {**c, "ticker": "B.TO"})
    conn.close()


def test_others_of_the_week_are_listed(tmp_path, clock):
    mail, a = Mail(), leader()
    put(tmp_path, market(**{"A.TO": a}))
    go(tmp_path, mail)
    b = leader(rest=4)                                          # one session short of a rest on the 25th
    data = market(**{"A.TO": extend(a, [a["Close"].iloc[-1]]), "B.TO": extend(b, [b["Close"].iloc[-1]])})
    at(clock, tmp_path, data, "2026-11-26", "2026-11-26")
    go(tmp_path, mail)
    subject, body = mail.sent[1]
    assert subject == "Minervini candidates 2026-11-26: B.TO"
    assert "A.TO of 2026-11-25" in body and "waiting for the entry" in body and "buy above" in body


def test_missing_or_old_bars(tmp_path, clock):
    mail = Mail()
    assert go(tmp_path, mail) == 2 and mail.sent == []                       # 18:20: quiet, the 21:40 run tries again
    put(tmp_path, market(), "2026-11-24")
    clock["at"] = datetime(2026, 11, 25, 21, 40)
    assert go(tmp_path, mail) == 2 and "NOT done" in mail.sent[0][0] and "2026-11-24" in mail.sent[0][1]
    assert rows(tmp_path).empty


def test_failed_mail_is_repeated(tmp_path):
    put(tmp_path, market(**{"A.TO": leader()}))
    assert go(tmp_path, Mail(ok=False)) == 2
    mail = Mail()
    assert go(tmp_path, mail) == 0 and len(mail.sent) == 1 and go(tmp_path, mail) == 0 and len(mail.sent) == 1


def test_a_crash_leaves_nothing_half_done(tmp_path, monkeypatch):
    put(tmp_path, market(**{"A.TO": leader(), "B.TO": leader(rest=9)}))
    calls = []

    def second_fails(*a):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return True

    monkeypatch.setattr(scan, "is_new", second_fails)
    assert go(tmp_path, Mail()) == 2
    assert rows(tmp_path).empty and scan.connect(tmp_path / "m.db").execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_dry_run_leaves_nothing(tmp_path, capsys):
    put(tmp_path, market(**{"A.TO": leader()}))
    mail = Mail()
    assert go(tmp_path, mail, dry_run=True) == 0 and mail.sent == [] and "ENTRY" in capsys.readouterr().out
    assert not (tmp_path / "m.db").exists()


# ── The tab ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tab(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "MINERVINI_DB_PATH", tmp_path / "m.db")
    return tmp_path


def test_tab_without_a_database(tab):
    assert dash.build_minervini_state() == {"rows": [], "session": None}


def test_tab_shows_one_week(tab):
    conn = scan.connect(tab / "m.db")
    for k in range(8):
        conn.execute("INSERT INTO candidates (ticker, signal, pivot, low, close, rank, rest, state, stop, target, last, found_at) "
                     "VALUES (?, ?, 100, 95, 98, .9, 8, 'waiting', 95, 120, 98, '')", (f"T{k}.TO", CAL[-1 - k].date().isoformat()))
    conn.execute("INSERT INTO runs VALUES ('2026-11-25', '')")
    conn.commit()
    conn.close()
    s = dash.build_minervini_state()
    assert s["session"] == "2026-11-25" and [r["ticker"] for r in s["rows"]] == [f"T{k}.TO" for k in range(6)]


@pytest.mark.parametrize("state,entry,last,result,distance", [
    ("waiting", None, 98, None, 100 / 98 * 100 - 100), ("bought", 102., 104., 104 / 102 * 100 - 100, None),
    ("target", 100., 121., 20., None), ("stopped", 100., 90., -5., None), ("expired", None, 98., None, None)])
def test_tab_numbers(state, entry, last, result, distance):
    r = dash.describe({"state": state, "entry": entry, "pivot": 100., "stop": 95., "target": 120., "last": last})
    assert r["result_pct"] == (None if result is None else pytest.approx(result))
    assert r["distance_pct"] == (None if distance is None else pytest.approx(distance))
    assert r["limit"] == 105 and r["words"]


def test_every_state_has_words():
    assert set(dash.STATES) == set(scan.STATES)
