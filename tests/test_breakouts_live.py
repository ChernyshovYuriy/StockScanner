"""The daily breakout scan: when it runs, what it finds, what it emails, how it fails. No network, no real database, no email."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from research.breakouts import live
from research.breakouts.detector import scan
from research.breakouts.prices import after_seam
from research.vcp.prices import Pacer, YahooBlocked

CAL = pd.bdate_range(end="2026-11-25", periods=700)
TODAY = CAL[-1]


def bars(close, vol=1e6, idx=CAL):
    c = np.asarray(close, float)
    return pd.DataFrame({"Open": c, "High": c * 1.002, "Low": c * 0.998, "Close": c, "Volume": vol}, index=idx[-len(c):])


def breakout(back=0, n=700, base=80.0, scale=1.0):
    """Two tests of 100, then a close at 104 `back` sessions before the last; flat after."""
    at = n - 1 - back
    c = np.interp(np.arange(n), [0, at - 100, at - 80, at - 60, at - 40, at - 20, at - 2, at - 1, at], [base, base, 88, 100, 88, 100, 99, 100, 104])
    return c * scale


def quiet(n=700, p=100.0):
    return p + np.sin(np.arange(n) / 9.0)


class Yahoo:
    def __init__(self, data, dead_after=None):
        self.data, self.calls, self.dead_after = {"XIU.TO": bars(quiet()), **data}, [], dead_after

    def __call__(self, names):
        self.calls.append(list(names))
        if self.dead_after is not None and len(self.calls) > self.dead_after:
            return {}
        return {t: self.data[t] for t in names if t in self.data}


class Mail:
    def __init__(self, ok=True):
        self.ok, self.sent = ok, []

    def __call__(self, subject, body):
        self.sent.append((subject, body))
        return self.ok


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    now = {"at": datetime(2026, 11, 25, 17, 45)}
    monkeypatch.setattr(live, "market_now", lambda: now["at"])
    return now


def pacer():
    return Pacer(1000, sleep=lambda s: None)


def go(tmp_path, y, mail, seed=None, full=(), dry=False):
    return live.auto(dry, tmp_path / "b.db", y, lambda: list(full), mail, pacer(), seed if seed is not None else [t for t in y.data if t != "XIU.TO"])


# ── When ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("now,want", [((2026, 11, 25, 17, 45), "2026-11-25"), ((2026, 11, 25, 16, 29), "2026-11-24"), ((2026, 11, 25, 16, 30), "2026-11-25"),
                                      ((2026, 11, 28, 12, 0), "2026-11-27"), ((2026, 11, 30, 9, 0), "2026-11-27"), ((2026, 12, 25, 18, 0), "2026-12-24")])
def test_session_at(now, want):
    assert live.session_at(datetime(*now)).isoformat() == want


# ── Finding ──────────────────────────────────────────────────────────────────

def test_a_breakout_is_emailed_the_same_evening_once(tmp_path):
    y, mail = Yahoo({"A.TO": bars(breakout()), "Q.TO": bars(quiet())}), Mail()
    assert go(tmp_path, y, mail) == 0
    (subject, body), = mail.sent
    assert subject == "Breakouts 2026-11-25: A.TO" and "A.TO   200-day average RISING" in body and "Q.TO" not in body and "57% up after a month" in body
    n = len(y.calls)
    assert go(tmp_path, y, mail) == 0
    assert len(mail.sent) == 1 and len(y.calls) == n           # the second slot asks Yahoo for nothing


def test_a_quiet_day_sends_nothing(tmp_path):
    mail = Mail()
    assert go(tmp_path, Yahoo({"Q.TO": bars(quiet())}), mail) == 0 and mail.sent == []


def test_yesterdays_breakout_is_not_news_on_the_first_run(tmp_path):
    mail = Mail()
    go(tmp_path, Yahoo({"A.TO": bars(breakout(back=1))}), mail)
    assert mail.sent == []


def test_a_missed_session_is_made_up_and_marked_late(tmp_path, clock):
    mail, y = Mail(), Yahoo({"A.TO": bars(breakout(back=1)), "B.TO": bars(breakout(back=0))})
    conn = live.connect(tmp_path / "b.db")
    live.save(conn, CAL[-3].date(), [], {}, datetime(2026, 11, 23, 18))
    conn.close()
    go(tmp_path, y, mail)
    (subject, body), = mail.sent
    assert "=== 2026-11-24  (found late" in body and "=== 2026-11-25 ===" in body and body.index("B.TO") < body.index("A.TO")


def test_catch_up_is_limited(tmp_path):
    mail, y = Mail(), Yahoo({"A.TO": bars(breakout(back=5)), "B.TO": bars(breakout(back=4))})
    conn = live.connect(tmp_path / "b.db")
    live.save(conn, CAL[-40].date(), [], {}, datetime(2026, 10, 1, 18))
    conn.close()
    go(tmp_path, y, mail)
    assert "B.TO" in mail.sent[0][0] and "A.TO" not in mail.sent[0][0]


def test_falling_average_is_marked_and_listed_after_rising(tmp_path):
    mail = Mail()
    fall = breakout(base=160.0)
    go(tmp_path, Yahoo({"A.TO": bars(fall), "Z.TO": bars(breakout())}), mail)
    subject, body = mail.sent[0]
    assert subject == "Breakouts 2026-11-25: Z.TO (+1 not rising)" and "A.TO   200-day average FALLING" in body and body.index("Z.TO  ") < body.index("A.TO  ")


def test_only_falling_is_said_in_the_subject(tmp_path):
    mail = Mail()
    go(tmp_path, Yahoo({"A.TO": bars(breakout(base=160.0))}), mail)
    assert mail.sent[0][0] == "Breakouts 2026-11-25: 1 not rising"


@pytest.mark.parametrize("frame", [bars(breakout(), vol=9_000), bars(breakout(scale=0.009), vol=1e9), bars(breakout(n=251))])
def test_illiquid_cheap_or_new_names_are_not_reported(tmp_path, frame):
    mail = Mail()
    go(tmp_path, Yahoo({"A.TO": frame, "Q.TO": bars(quiet())}), mail)
    assert mail.sent == []


def test_a_name_with_252_bars_counts(tmp_path):
    mail = Mail()
    go(tmp_path, Yahoo({"A.TO": bars(breakout(n=252))}), mail)
    assert len(mail.sent) == 1


def test_short_history_is_not_called_falling():
    r = pd.Series({"ticker": "A.TO", "rising": None, "slope200": np.nan, "close": 104., "line": 100., "confirm": 103., "tests": 2, "bars": 63, "depth": .12,
                   "day_ret": .04, "vol_x": 1.5, "atr": .02, "lfd_low": 98., "target": 112., "previous": None})
    text = live.describe(next(r.to_frame().T.itertuples()))
    assert "not known" in text and "FALLING" not in text


def test_numbers_in_the_email():
    mail, y = Mail(), Yahoo({"A.TO": bars(breakout())})
    found, _ = live.scan_all(["A.TO"], TODAY.date(), [], set(), y, pacer(), lambda m: None)
    r, = found
    b = scan(y.data["A.TO"])[-1]
    assert (r["date"], r["close"], r["line"], r["confirm"], r["lfd_low"], r["target"]) == ("2026-11-25", 104.0, b.level, b.confirm, b.lfd_low, b.target)
    text = live.describe(next(pd.DataFrame(found).itertuples()))
    assert f"closed 104.00, {104 / b.level - 1:+.1%} over the line {b.level:.2f} (confirmed at {b.confirm:.2f})" in text
    assert "volume 1.0x" in text and f"measured target {b.target:.2f} ({b.target / 104 - 1:+.1%})" in text


def test_an_earlier_breakout_is_mentioned(tmp_path):
    c = breakout(back=0)
    real, calls = live.scan, []
    d = bars(c)
    class B: pass
    def fake(x):
        out = real(x)
        e = B(); e.__dict__.update(out[-1].__dict__); e.__dict__["bar"] = len(x) - 1 - 30
        o = B(); o.__dict__.update(out[-1].__dict__); o.__dict__["bar"] = len(x) - 1 - 61
        return [o, e, out[-1]]
    live.scan = fake
    try:
        r, = live.find(d, CAL, [TODAY])
    finally:
        live.scan = real
    assert r["previous"] == CAL[-31].date().isoformat()


def test_bars_after_the_session_are_ignored(tmp_path, clock):
    clock["at"] = datetime(2026, 11, 25, 10, 0)                # during the 25th: the 24th is the session
    mail = Mail()
    go(tmp_path, Yahoo({"A.TO": bars(breakout(back=0)), "B.TO": bars(breakout(back=1))}), mail)
    assert mail.sent[0][0] == "Breakouts 2026-11-24: B.TO"


def test_tail_gives_the_same_breakouts_as_the_whole_history():
    n, seen = 1500, 0
    idx = pd.bdate_range(end="2026-11-25", periods=n)
    for seed in range(150):
        g = np.random.default_rng(seed)
        c = 50 * np.exp(np.cumsum(g.normal(0.0006, 0.012, n)))
        d = pd.DataFrame({"Open": c, "High": c * (1 + g.uniform(0, .01, n)), "Low": c * (1 - g.uniform(0, .01, n)), "Close": c, "Volume": 1e6}, index=idx)
        days = list(idx[-55:])
        whole = {d.index[b.bar]: (b.level, b.target, b.lfd_low, b.tests, b.bars) for b in scan(d) if d.index[b.bar] in days}
        part = {pd.Timestamp(r["date"]): (r["line"], r["target"], r["lfd_low"], r["tests"], r["bars"]) for r in live.find(d, idx, days)}
        assert part == whole
        seen += len(whole)
    assert seen >= 5


def test_live_eligibility_equals_the_backtests():
    from research.breakouts.backtest import Market
    g = np.random.default_rng(0)
    data = {"XIU.TO": bars(quiet())}
    for k in range(6):
        data[f"S{k}.TO"] = bars(breakout(scale=[1, .012, .0099, .5, 1, 1][k]), vol=[1e4, 1e6, 1e6, 2e4, 9.7e3, 1e6][k]).drop(CAL[[-7, -12]] if k % 2 else [])
    m = Market(data)
    for t in m.bars:
        hit = live.find(data[t], CAL, [TODAY])
        assert bool(hit) == bool(m.eligible.at[TODAY, t]), t


# ── Failures ─────────────────────────────────────────────────────────────────

def test_blocked_by_yahoo_stops_emails_and_is_retried(tmp_path):
    names = {f"S{k:02d}.TO": bars(quiet()) for k in range(40)}
    names["S05.TO"] = bars(breakout())
    mail, y = Mail(), Yahoo(names, dead_after=2)
    assert go(tmp_path, y, mail) == 2
    assert len(y.calls) == 4 and y.calls[-1] == ["XIU.TO"]      # benchmark, chunk, dead chunk, probe: nothing more
    assert mail.sent[0][0] == "Breakouts 2026-11-25: scan NOT done"
    y.dead_after = None
    assert go(tmp_path, y, mail) == 0 and mail.sent[1][0] == "Breakouts 2026-11-25: S05.TO"


def test_no_benchmark_at_all(tmp_path):
    mail = Mail()
    assert go(tmp_path, Yahoo({"A.TO": bars(breakout())}, dead_after=0), mail) == 2 and "YahooBlocked" in mail.sent[0][1]


def test_benchmark_without_todays_bar_waits(tmp_path):
    mail, y = Mail(), Yahoo({"A.TO": bars(breakout())})
    y.data["XIU.TO"] = y.data["XIU.TO"].iloc[:-1]
    assert go(tmp_path, y, mail) == 2 and "no bar for 2026-11-25" in mail.sent[0][1] and len(y.calls) == 1


def test_stale_liquid_names_are_a_failure_not_a_quiet_day(tmp_path):
    data = {f"S{k}.TO": bars(quiet()) for k in range(8)} | {f"T{k}.TO": bars(quiet(699), idx=CAL[:-1]) for k in range(1)}
    assert go(tmp_path, Yahoo(data), Mail()) == 2               # 8 of 9
    data["S8.TO"] = bars(quiet())
    assert go(tmp_path / "x", Yahoo(data), Mail()) == 0         # 9 of 10


def test_missing_known_liquid_names_count_as_stale(tmp_path):
    data = {f"S{k}.TO": bars(quiet()) for k in range(10)}
    y = Yahoo(data)
    conn = live.connect(tmp_path / "b.db")
    live.save(conn, CAL[-2].date(), [], {**{t: 1e8 for t in data}, "L1.TO": 1e8, "L2.TO": 1e8}, datetime(2026, 11, 24, 18))
    conn.close()
    assert go(tmp_path, y, Mail()) == 2                         # 10 of 12


def test_failed_email_is_sent_next_time(tmp_path):
    y, mail = Yahoo({"A.TO": bars(breakout())}), Mail(ok=False)
    assert go(tmp_path, y, mail) == 2
    good, n = Mail(), len(y.calls)
    assert go(tmp_path, y, good) == 0 and good.sent[0][0] == "Breakouts 2026-11-25: A.TO" and len(y.calls) == n
    assert go(tmp_path, y, good) == 0 and len(good.sent) == 1


def test_full_list_down_still_scans_known_names(tmp_path):
    def boom():
        raise OSError("down")
    mail = Mail()
    assert live.auto(False, tmp_path / "b.db", Yahoo({"A.TO": bars(breakout())}), boom, mail, pacer(), ["A.TO"]) == 0 and len(mail.sent) == 1


def test_dry_run_writes_and_sends_nothing(tmp_path, capsys):
    mail, y = Mail(), Yahoo({"A.TO": bars(breakout())})
    assert go(tmp_path, y, mail, dry=True) == 0
    assert mail.sent == [] and "A.TO   200-day" in capsys.readouterr().out
    conn = live.connect(tmp_path / "b.db")
    assert live.finished(conn) == [] and live.unsent(conn).empty
    conn.close()


def test_garbage_bars_are_dropped():
    d = bars(quiet())
    d.iloc[-3, 3] = np.nan
    d.iloc[-4, 3] = 0.0
    assert len(live.clean(d, TODAY.date())) == len(d) - 2
    s = bars(np.r_[np.full(300, 1.0), np.full(400, 10.0)])       # an unadjusted consolidation
    assert len(live.clean(s, TODAY.date())) == 400 and len(after_seam(s)) == 400


# ── Names ────────────────────────────────────────────────────────────────────

def test_rotation_reaches_every_name_and_liquidity_moves_names(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "ROTATE", 2)
    conn = live.connect(tmp_path / "b.db")
    full, day = ["A.TO", "B.TO", "C.TO", "D.TO", "E.TO"], date(2026, 11, 2)
    assert live.choose(conn, ["A.TO"], full) == ["A.TO", "B.TO", "C.TO"]
    live.save(conn, day, [], {"A.TO": 5e6, "B.TO": 1e3, "C.TO": 4e6}, datetime(2026, 11, 2, 18))
    assert live.choose(conn, ["A.TO"], full) == ["A.TO", "C.TO", "D.TO", "E.TO"]       # C joined, D and E are next
    live.save(conn, date(2026, 11, 3), [], {"A.TO": 1e3, "C.TO": 4e6, "D.TO": 1., "E.TO": 1.}, datetime(2026, 11, 3, 18))
    assert live.choose(conn, ["A.TO"], full) == ["C.TO", "B.TO", "A.TO"]                 # A dried up; B waited longest
    conn.close()


def test_a_passing_failure_keeps_a_name_and_a_long_one_drops_it(tmp_path):
    conn = live.connect(tmp_path / "b.db")
    live.save(conn, date(2026, 11, 2), [], {"A.TO": 5e6}, datetime(2026, 11, 2, 18))
    live.save(conn, date(2026, 11, 10), [], {"A.TO": None}, datetime(2026, 11, 10, 18))
    assert live.liquid_names(conn) == {"A.TO"} and "A.TO" in live.choose(conn, [], [])
    live.save(conn, date(2026, 11, 17), [], {"A.TO": None}, datetime(2026, 11, 17, 18))
    assert live.liquid_names(conn) == set()
    conn.close()


def test_seed_name_without_data_stays_and_a_rotated_one_moves_to_the_back(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "ROTATE", 1)
    conn = live.connect(tmp_path / "b.db")
    live.save(conn, date(2026, 11, 2), [], {"A.TO": None, "B.TO": None}, datetime(2026, 11, 2, 18))
    assert live.choose(conn, ["A.TO"], ["B.TO", "C.TO"]) == ["A.TO", "C.TO"]
    conn.close()


def test_benchmark_is_never_a_name(tmp_path):
    conn = live.connect(tmp_path / "b.db")
    assert live.choose(conn, ["XIU.TO", "A.TO"], []) == ["A.TO"]
    conn.close()


def test_seed_file_is_the_backtests_list():
    seed = live.SEED.read_text().split()
    assert len(seed) == len(set(seed)) == 513 and all(t.endswith((".TO", ".V", ".CN")) for t in seed)


def test_scan_never_exceeds_the_pace(tmp_path):
    now, waits = [0.0], []
    def sleep(s):
        waits.append(s); now[0] += s
    p = Pacer(0.4, sleep=sleep, clock=lambda: now[0])
    names = {f"S{k:02d}.TO": bars(quiet()) for k in range(30)}
    live.scan_all(sorted(names), TODAY.date(), [], set(), Yahoo(names), p, lambda m: None)
    assert now[0] >= 21 / 0.4 - 1e-6


def test_fetch_is_unadjusted(monkeypatch):
    seen = {}
    class P:
        def download_batch_with_reasons(self, t, **k):
            seen.update(k); return {"A.TO": 1}, {}
    monkeypatch.setattr(live, "LiveDataProvider", P)
    assert live.yahoo(["A.TO"]) == {"A.TO": 1} and seen == {"period": "3y", "interval": "1d", "auto_adjust": False}


def test_dollar_volume_is_twenty_sessions_with_gaps_as_zero():
    d = pd.DataFrame({"Close": 2.0, "Volume": np.arange(700.0)}, index=CAL)
    assert live.dollar_volume(d, CAL, CAL[-1]) == pytest.approx(2 * np.arange(680, 700).mean())
    assert live.dollar_volume(d.drop(CAL[-3]), CAL, CAL[-1]) == pytest.approx(2 * (np.arange(680, 700).sum() - 697) / 20)
    assert live.dollar_volume(d, CAL, CAL[-2]) == pytest.approx(2 * np.arange(679, 699).mean())


def test_price_is_judged_on_the_breakout_day():
    c = breakout(back=1, scale=0.99 / 104)
    c[-1] = 1.02
    assert live.find(bars(c, vol=1e8), CAL, list(CAL[-2:])) == []


def test_a_recorded_breakout_is_never_rewritten(tmp_path):
    y = Yahoo({"A.TO": bars(breakout())})
    found, dollars = live.scan_all(["A.TO"], TODAY.date(), [], set(), y, pacer(), lambda m: None)
    conn = live.connect(tmp_path / "b.db")
    live.save(conn, TODAY.date(), found, dollars, datetime(2026, 11, 25, 18))
    conn.execute("UPDATE breakouts SET emailed = 1")
    live.save(conn, TODAY.date(), found, dollars, datetime(2026, 11, 25, 21))
    assert live.unsent(conn).empty
    conn.close()


@pytest.mark.parametrize("stored", [False, True])
def test_no_earlier_breakout_is_not_printed_as_nan(tmp_path, stored):
    y = Yahoo({"A.TO": bars(breakout()), "B.TO": bars(breakout())})
    found, dollars = live.scan_all(["A.TO", "B.TO"], TODAY.date(), [], set(), y, pacer(), lambda m: None)
    found[1]["previous"] = "2026-11-09"
    rows = pd.DataFrame(found)
    if stored:
        conn = live.connect(tmp_path / "b.db")
        live.save(conn, TODAY.date(), found, dollars, datetime(2026, 11, 25, 18))
        rows = live.unsent(conn)
        conn.close()
    body = live.message(rows, TODAY.date())[1]
    assert body.count("also broke out on") == 1 and "also broke out on 2026-11-09" in body and "nan" not in body
