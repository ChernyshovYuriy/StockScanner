"""Swing rank -- forward record of S1. Synthetic, no network."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

import time_utils
from research.swing_rank import engine as en
from research.swing_rank import forward as fw
from research.swing_rank import holdings
from research.swing_rank.forward import NotNow
from research.swing_rank.stocks import Scores, StockData

TZ = time_utils.TSX_TZ
CAL = pd.bdate_range("2023-01-02", "2026-12-04")
CAL = CAL[[d.strftime("%Y-%m-%d") not in time_utils.TSX_HOLIDAYS for d in CAL]]
SECTORS = ["Energy", "Financials", "Materials", "Utilities", "Industrials", "Tech", "Health", "Staples"]
N = 40


def at(text):
    return datetime.fromisoformat(text).replace(tzinfo=TZ)


@pytest.fixture(autouse=True)
def clock():
    yield lambda text: time_utils.set_backtest_clock(at(text))
    time_utils.set_backtest_clock(None)


def frame(c, idx=CAL):
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6}, index=idx)


def ticker(k):
    return f"C{'ABCDEFGHIJ'[k // 10]}{'ABCDEFGHIJ'[k % 10]}"


def world(seed=0):
    walk = lambda s: 20 * np.exp(np.cumsum(np.random.default_rng(s).normal(0.0003, 0.015, len(CAL))))
    yahoo = {"XIU.TO": frame(walk(seed + 999))}
    for k in range(N):
        yahoo[ticker(k) + ".TO"] = frame(walk(seed + k))
    files = {}
    ends = [d for d in CAL[:-1] if CAL[CAL.get_loc(d) + 1].month != d.month]
    for d in ends:
        rows = [(ticker(k), f"COMPANY {k}", SECTORS[k % 8], 1000.0, float(yahoo[ticker(k) + ".TO"].at[d, "Close"]), 1.0, "Toronto") for k in range(N)]
        files[d] = pd.DataFrame(rows, columns=holdings.COLUMNS).set_index("ticker")
    return files, yahoo


WORLD = world()


def data_at(signal, files=None, yahoo=None):
    files, yahoo = files or WORLD[0], yahoo or WORLD[1]
    return StockData({d: m for d, m in files.items() if d <= pd.Timestamp(signal)}, fw.with_next_session(yahoo, signal))


# ── When ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("now,expected", [
    ("2026-11-25 12:00", "2026-10-30"),       # mid-month: the month before
    ("2026-11-30 16:20", "2026-10-30"),       # last session, bars not final yet
    ("2026-11-30 16:30", "2026-11-30"),
    ("2026-12-01 09:00", "2026-11-30"),
    ("2026-12-01 18:00", "2026-11-30"),
    ("2026-10-31 10:00", "2026-10-30"),       # Saturday
    ("2026-12-31 20:00", "2026-12-31"),
    ("2027-01-01 12:00", "2026-12-31"),       # holiday
    ("2027-01-31 12:00", "2027-01-29"),       # Sunday, month ended on Friday
])
def test_signal_session(now, expected):
    assert fw.signal_session(at(now)).isoformat() == expected


@pytest.mark.parametrize("now,expected", [
    ("2026-11-30 17:00", "2026-12-01"), ("2026-12-01 09:29", "2026-12-01"), ("2026-12-01 09:30", "2026-12-02"),
    ("2026-12-04 18:00", "2026-12-07"), ("2026-12-05 09:00", "2026-12-07"), ("2026-12-24 18:00", "2026-12-29"),
    ("2026-12-25 08:00", "2026-12-29"),
])
def test_entry_session(now, expected):
    assert fw.entry_session(at(now)).isoformat() == expected


def test_sessions_late():
    s = date(2026, 11, 30)
    assert [fw.sessions_late(s, date(2026, 12, d)) for d in (1, 2, 4, 7, 8, 9)] == [0, 1, 3, 4, 5, 6]


def test_a_list_cannot_be_recorded_long_after_its_month():
    empty = pd.DataFrame(columns=fw.COLUMNS)
    assert fw.check_time(at("2026-12-07 18:00"), empty) == date(2026, 11, 30)          # entry Dec 8, 5 late
    with pytest.raises(NotNow, match="late"):
        fw.check_time(at("2026-12-08 18:00"), empty)
    with pytest.raises(NotNow, match="late"):
        fw.check_time(at("2026-11-25 18:00"), empty)


def test_a_month_is_recorded_once():
    with pytest.raises(NotNow, match="already"):
        fw.check_time(at("2026-12-01 08:00"), pd.DataFrame({"signal_date": ["2026-11-30"]}))


# ── What ─────────────────────────────────────────────────────────────────────

def test_next_session_is_empty_and_nothing_after_the_signal_survives():
    cut = fw.with_next_session(WORLD[1], date(2026, 11, 30))
    assert cut["XIU.TO"].index[-1] == pd.Timestamp("2026-12-01") and cut["XIU.TO"].iloc[-1].isna().all()
    assert all(d.index.max() == pd.Timestamp("2026-11-30") for t, d in cut.items() if t != "XIU.TO")


def test_missing_benchmark_bar_stops():
    y = {**WORLD[1], "XIU.TO": WORLD[1]["XIU.TO"].loc[:"2026-11-27"]}
    with pytest.raises(NotNow, match="no bar"):
        fw.with_next_session(y, date(2026, 11, 30))


@pytest.mark.parametrize("day", ["2026-11-30", "2026-07-31", "2025-12-31"])
def test_list_equals_the_backtest_at_that_session(day):
    """Built from data that ends at the signal, equal to the choice the backtest makes with everything after it."""
    full = StockData(*WORLD)
    s = int(full.calendar.get_loc(pd.Timestamp(day)))
    want = en.select(Scores(full).momentum(s), set(), 10, 15, full.sector_of(s), 2)
    name = full.table.drop_duplicates("id").set_index("id")["ticker"]
    got = fw.make_list(data_at(date.fromisoformat(day)), date.fromisoformat(day), set())
    assert got["ticker"].tolist() == [name[c] for c in want] and len(got) == 10
    assert got["rank"].tolist() == list(range(1, 11))
    assert got["close"].tolist() == [round(WORLD[1][t].at[pd.Timestamp(day), "Close"], 4) for t in got["symbol"]]
    assert set(got["signal_date"]) == {day}


def test_future_bars_do_not_matter():
    d = date(2026, 7, 31)
    noisy = {t: f.copy() for t, f in WORLD[1].items()}
    for t in noisy:
        noisy[t].loc["2026-08-01":, ["Open", "High", "Low", "Close"]] *= 7.0
    pd.testing.assert_frame_equal(fw.make_list(data_at(d), d, set()), fw.make_list(data_at(d, yahoo=noisy), d, set()))


def record_of(rows, entry="2026-11-02"):
    return rows.assign(recorded_at="x", entry_date=entry)[fw.COLUMNS]


def test_held_names_use_the_keep_band():
    d0, d1 = date(2026, 10, 30), date(2026, 11, 30)
    data = data_at(d1)
    s = int(data.calendar.get_loc(pd.Timestamp(d1)))
    score = Scores(data).momentum(s).sort_values(ascending=False)
    tick = data.table[data.table["date"] == pd.Timestamp(d0)].set_index("id")
    eleventh_to_15th = score.index[10:15]
    rows = fw.make_list(data_at(d0), d0, set()).iloc[:5].copy()
    rows["ticker"] = [tick.at[c, "ticker"] for c in eleventh_to_15th]
    held = fw.held_ids(data, record_of(rows))
    assert held == set(eleventh_to_15th)
    got = fw.make_list(data, d1, held)["ticker"].tolist()
    assert got == [tick.at[c, "ticker"] for c in en.select(score, held, 10, 15, data.sector_of(s), 2)]
    assert got != fw.make_list(data, d1, set())["ticker"].tolist()


def test_held_uses_the_last_list_only_and_survives_a_rename():
    files = {d: m.copy() for d, m in WORLD[0].items()}
    d0, d1 = date(2026, 10, 30), date(2026, 11, 30)
    m = files[pd.Timestamp(d1)]
    files[pd.Timestamp(d1)] = m.rename(index={"CAD": "NEW"})
    rows = fw.make_list(data_at(d0), d0, set()).iloc[:1].copy()
    old = record_of(rows.assign(signal_date="2026-09-30", file_date="2026-09-30", ticker="CAA"))
    new = record_of(rows.assign(ticker="CAD"))
    data = data_at(d1, files=files)
    now = data.table[data.table["date"] == pd.Timestamp(d1)].set_index("ticker")["id"]
    assert fw.held_ids(data, pd.concat([old, new])) == {int(now["NEW"])}


def test_held_name_missing_from_its_file_is_reported_not_fatal(capsys):
    d0 = date(2026, 10, 30)
    rows = fw.make_list(data_at(d0), d0, set()).iloc[:1].assign(ticker="GONE")
    assert fw.held_ids(data_at(d0), record_of(rows)) == set()
    assert "GONE" in capsys.readouterr().out
    assert fw.held_ids(data_at(d0), pd.DataFrame(columns=fw.COLUMNS)) == set()


def test_fresh_share():
    y = {"A.TO": frame(np.ones(len(CAL))), "B.TO": frame(np.ones(len(CAL))).loc[:"2026-11-27"]}
    assert fw.fresh_share(y, ["A.TO", "B.TO", "NONE.TO", "C.TO"], date(2026, 11, 30)) == 0.25


def csv_text(day, n=200):
    head = f'Fund Holdings as of,"{day:%b %d, %Y}"\n\nTicker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,Price,Location,Exchange,Currency\n'
    return head + "".join(f'"T{k}","NAME {k}","Energy","Equity","1","1","1","100","10.0","Canada","Toronto","CAD"\n' for k in range(n))


def test_member_file_steps_back_to_the_latest_published_day():
    asked = []

    def get(day):
        asked.append(day)
        return csv_text(pd.Timestamp("2026-11-27")) if day == "20261127" else 'Fund Holdings as of,"-"\n'
    as_of, m = fw.member_file(date(2026, 11, 30), get=get, sleep=lambda s: None)
    assert as_of == pd.Timestamp("2026-11-27") and len(m) == 200 and m.index[0] == "T0"
    assert asked == ["20261130", "20261127"]


@pytest.mark.parametrize("get", [lambda d: "", lambda d: csv_text(pd.Timestamp("2026-10-30")),
                                 lambda d: csv_text(pd.Timestamp(d), n=20)])
def test_member_file_refuses_wrong_dates_and_thin_files(get):
    with pytest.raises(NotNow):
        fw.member_file(date(2026, 11, 30), get=get, sleep=lambda s: None)


def test_build_never_reads_past_the_signal(monkeypatch, tmp_path):
    """The member files and the price request stop at the signal; a month-end file
    not published yet is replaced by the latest list, and is not cached."""
    d = date(2026, 11, 30)
    files = {k: v for k, v in WORLD[0].items() if k != pd.Timestamp(d)}
    seen = {}
    monkeypatch.setattr(fw, "CACHE", tmp_path)
    monkeypatch.setattr(holdings, "fetch", lambda last: seen.update(last=last))
    monkeypatch.setattr(holdings, "load", lambda: {**files, pd.Timestamp("2026-12-31"): WORLD[0][pd.Timestamp(d)]})
    monkeypatch.setattr(fw, "member_file", lambda s: (pd.Timestamp("2026-11-27"), WORLD[0][pd.Timestamp(d)]))
    monkeypatch.setattr(fw, "fetch_prices", lambda symbols, db, end: seen.update(end=end, n=len(symbols)) or WORLD[1])
    data = fw.build(d)
    assert seen == {"last": pd.Timestamp("2026-11-29"), "end": d, "n": N}
    assert max(data.file_dates) == pd.Timestamp("2026-11-27")
    assert data.calendar[-2] == pd.Timestamp(d)
    monkeypatch.setattr(fw, "fetch_prices", lambda *a: {t: f.loc[:"2026-11-27"] if t != "XIU.TO" else f for t, f in WORLD[1].items()})
    with pytest.raises(NotNow, match="0%"):
        fw.build(d)
    monkeypatch.setattr(fw, "member_file", lambda s: (pd.Timestamp("2026-11-18"), WORLD[0][pd.Timestamp(d)]))
    with pytest.raises(NotNow, match="newest member list"):
        fw.build(d)


# ── Recording ────────────────────────────────────────────────────────────────

def test_record_writes_one_month_and_never_rewrites(tmp_path, clock):
    path = tmp_path / "r.csv"
    clock("2026-10-30 20:00")
    first = fw.record(path=path, build_fn=data_at)
    rec = fw.load_record(path)
    assert list(rec.columns) == fw.COLUMNS and len(rec) == 10
    assert set(rec["entry_date"]) == {"2026-11-02"} and set(rec["recorded_at"]) == {"2026-10-30T20:00:00-04:00"}
    assert rec["ticker"].tolist() == first["ticker"].tolist()
    before = path.read_text()
    with pytest.raises(NotNow, match="already"):
        fw.record(path=path, build_fn=data_at)
    assert path.read_text() == before
    clock("2026-12-01 09:45")                       # a day late: entry is the next open
    fw.record(path=path, build_fn=data_at)
    assert path.read_text().startswith(before)
    rec = fw.load_record(path)
    assert len(rec) == 20 and rec["entry_date"].iloc[-1] == "2026-12-02" and rec["signal_date"].iloc[-1] == "2026-11-30"
    held = fw.held_ids(data_at(date(2026, 11, 30)), rec.iloc[:10])
    assert rec["ticker"].iloc[10:].tolist() == fw.make_list(data_at(date(2026, 11, 30)), date(2026, 11, 30), held)["ticker"].tolist()


def test_entry_is_set_after_the_download_ended(tmp_path, clock):
    clock("2026-12-01 09:20")

    def slow(signal):
        clock("2026-12-01 09:40")
        return data_at(signal)
    fw.record(path=tmp_path / "r.csv", build_fn=slow)
    assert set(fw.load_record(tmp_path / "r.csv")["entry_date"]) == {"2026-12-02"}


def test_append_refuses_a_recorded_month(tmp_path):
    rows = fw.make_list(data_at(date(2026, 10, 30)), date(2026, 10, 30), set())
    fw.append(tmp_path / "r.csv", rows, at("2026-10-30 20:00"), date(2026, 11, 2))
    with pytest.raises(NotNow, match="already"):
        fw.append(tmp_path / "r.csv", rows, at("2026-10-30 21:00"), date(2026, 11, 2))


def test_dry_run_writes_nothing_any_day(tmp_path, clock):
    clock("2026-11-25 12:00")
    assert len(fw.record(dry_run=True, path=tmp_path / "r.csv", build_fn=data_at)) == 10
    assert not (tmp_path / "r.csv").exists()
    with pytest.raises(NotNow):
        fw.record(path=tmp_path / "r.csv", build_fn=data_at)


def test_too_few_names_is_not_recorded(tmp_path, clock, monkeypatch):
    clock("2026-10-30 20:00")
    monkeypatch.setattr(fw, "make_list", lambda *a: pd.DataFrame({"signal_date": ["2026-10-30"] * 9}))
    with pytest.raises(NotNow, match="only 9"):
        fw.record(path=tmp_path / "r.csv", build_fn=data_at)


# ── Results ──────────────────────────────────────────────────────────────────

def rec(*months):
    """months: (entry date, symbols)"""
    return pd.DataFrame([{"entry_date": e, "rank": k, "symbol": s} for e, names in months for k, s in enumerate(names, 1)])


def prices_of(**paths):
    idx = pd.DatetimeIndex(["2026-11-30", "2026-12-01", "2026-12-02", "2026-12-03", "2026-12-04"])
    out = {}
    for t, c in paths.items():
        c = np.array(c, float)
        out[t + ".TO"] = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c + 1, "Volume": 1.0}, index=idx)
    return out


NAMES = [f"N{k}" for k in range(10)]


def flat(**special):
    return prices_of(XIU=[100, 100, 100, 100, 100], **{n: [100] * 5 for n in NAMES}, **special)


def test_results_by_hand():
    d = flat()
    d["N0.TO"]["Open"] = [100, 100, 100, 120, 120]
    d["N0.TO"]["Close"] = [100, 100, 100, 120, 132]
    for n in NAMES[1:]:
        d[n + ".TO"]["Close"] = 100.0
    d["XIU.TO"]["Open"], d["XIU.TO"]["Close"] = [100, 100, 100, 101, 101], [100, 100, 100, 101, 102]
    sym = [n + ".TO" for n in NAMES]
    out = fw.results(rec(("2026-12-01", sym), ("2026-12-03", sym)), d, cost=0.002)
    assert out["complete"].tolist() == [True, False] and out["entry"].tolist() == ["2026-12-01", "2026-12-03"]
    assert out["net"].iloc[0] == pytest.approx(1.02 * (1 - 0.002) - 1)
    traded = (0.12 / 1.02 - 0.1) + 9 * (0.1 - 0.1 / 1.02)
    assert out["net"].iloc[1] == pytest.approx(1.01 * (1 - 0.002 * traded) - 1)
    assert out["xiu"].tolist() == pytest.approx([0.01, 102 / 101 - 1])
    assert out["excess"].iloc[0] == pytest.approx(out["net"].iloc[0] - 0.01)
    assert out["missing"].tolist() == ["", ""]


def test_results_before_the_first_entry_and_future_entry():
    sym = [n + ".TO" for n in NAMES]
    assert fw.results(rec(("2026-12-07", sym)), flat()).empty
    out = fw.results(rec(("2026-12-01", sym), ("2026-12-07", sym)), flat())
    assert len(out) == 1 and not out["complete"].iloc[0]


def test_results_name_the_missing():
    sym = [n + ".TO" for n in NAMES[:9]] + ["DEAD.TO"]
    out = fw.results(rec(("2026-12-01", sym)), flat())
    assert out["missing"].iloc[0] == "DEAD.TO"


def test_a_name_that_stops_trading_is_closed_at_its_last_price():
    d = flat(STOP=[100, 100, 80, 80, 80])
    d["STOP.TO"] = d["STOP.TO"].iloc[:3]
    d["STOP.TO"]["Close"] = [100, 100, 80]
    for n in NAMES:
        d[n + ".TO"]["Close"] = 100.0
    sym = [n + ".TO" for n in NAMES[:9]] + ["STOP.TO"]
    out = fw.results(rec(("2026-12-01", sym)), d, cost=0.0)
    assert out["net"].iloc[0] == pytest.approx(-0.02)


def test_entries_and_exits_are_at_the_open():
    idx = pd.bdate_range("2026-12-01", periods=6)
    base = pd.DataFrame({"Open": [100, 110, 120, 130, 140, 150], "High": 1.0, "Low": 1.0, "Close": [105, 115, 125, 135, 145, 155],
                         "Volume": 1.0}, index=idx).astype(float)
    sym = [n + ".TO" for n in NAMES]
    d = {t: base.copy() for t in sym + ["XIU.TO"]}
    out = fw.results(rec((str(idx[0].date()), sym), (str(idx[2].date()), sym)), d, cost=0.0)
    assert out["net"].tolist() == pytest.approx([120 / 100 - 1, 155 / 120 - 1])
    assert out["xiu"].tolist() == pytest.approx([120 / 100 - 1, 155 / 120 - 1])


def test_a_long_dead_name_still_gets_its_last_price():
    idx = pd.bdate_range("2026-12-01", periods=15)
    one = pd.DataFrame({c: 100.0 for c in ("Open", "High", "Low", "Close", "Volume")}, index=idx)
    d = {n + ".TO": one for n in NAMES[:9]}
    d["XIU.TO"] = one
    d["STOP.TO"] = one.iloc[:3].assign(Close=[100.0, 100.0, 50.0])
    out = fw.results(rec((str(idx[0].date()), [n + ".TO" for n in NAMES[:9]] + ["STOP.TO"])), d, cost=0.0)
    assert out["net"].iloc[0] == pytest.approx(-0.05) and out["missing"].iloc[0] == ""


def test_held_ticker_means_the_company_that_had_it_then():
    """CAA belonged to another company three years earlier."""
    files = {d: m.copy() for d, m in WORLD[0].items()}
    d0 = date(2026, 10, 30)
    early = sorted(files)[:16]                      # gone for more than 400 days in between
    old = files[early[0]].loc[["CAA"]].assign(name="OLD OWNER", price=999.0)
    for d in early:
        files[d] = files[d].drop(index="CAA")
    files[early[0]] = pd.concat([old, files[early[0]]])
    data = data_at(d0, files=files)
    then = data.table[data.table["ticker"] == "CAA"].sort_values("date")["id"]
    assert then.iloc[0] != then.iloc[-1]
    rows = fw.make_list(data_at(d0), d0, set()).iloc[:1].assign(ticker="CAA")
    assert fw.held_ids(data, record_of(rows)) == {int(then.iloc[-1])}


def test_list_shows_the_labels_of_the_signal_month():
    files = {d: m.copy() for d, m in WORLD[0].items()}
    d0 = date(2026, 10, 30)
    files[pd.Timestamp(d0)] = files[pd.Timestamp(d0)].assign(name=lambda m: m["name"] + " RENAMED")
    got = fw.make_list(data_at(d0, files=files), d0, set())
    assert got["name"].str.endswith("RENAMED").all()


def test_load_prices_equals_load_all(tmp_path):
    from research.vcp import prices
    import market_data_cache as mdc
    conn = prices.connect(str(tmp_path / "p.db"))
    for t in ("XIU.TO", "CAA.TO"):
        mdc._upsert(conn, t, WORLD[1][t])
    a, b = prices.load_all(conn), fw.load_prices(conn)
    conn.close()
    assert list(a) == list(b) == ["CAA.TO", "XIU.TO"]
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t], check_dtype=False, check_index_type=False)


# ── On the timer ─────────────────────────────────────────────────────────────

@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setattr(fw, "CACHE", tmp_path)
    (tmp_path / "swing_rank_fwd_20261130.db").write_text("x")
    (tmp_path / "swing_rank_prices.db").write_text("keep")
    mails = []

    class Box:
        path, cache = tmp_path / "r.csv", tmp_path
        sent = mails

        def run(self, record_fn=None, score_fn=lambda: "SCORE"):
            return fw.auto(self.path, record_fn or (lambda: fw.record(path=self.path, build_fn=data_at)), score_fn,
                           lambda s, b: mails.append((s, b)) or True)
    return Box()


@pytest.mark.parametrize("now", ["2026-11-25 18:30", "2026-11-27 20:15", "2026-12-08 18:30"])
def test_auto_is_silent_when_nothing_is_due(box, clock, now):
    clock(now)
    assert box.run(record_fn=lambda: 1 / 0) == 0
    assert box.sent == [] and not box.path.exists() and (box.cache / "swing_rank_fwd_20261130.db").exists()


def test_auto_records_emails_once_and_cleans_up(box, clock):
    clock("2026-11-30 18:30")
    assert box.run() == 0
    rec = fw.load_record(box.path)
    assert len(rec) == 10 and set(rec["entry_date"]) == {"2026-12-01"}
    (subject, body), = box.sent
    assert subject == "Swing rank S1: 2026-11-30" and "2026-12-01" in body and "SCORE" in body and "did not pass" in body
    assert all(t in body for t in rec["symbol"])
    assert [p.name for p in box.cache.glob("*.db")] == ["swing_rank_prices.db"]
    clock("2026-11-30 20:15")
    assert box.run(record_fn=lambda: 1 / 0) == 0 and len(box.sent) == 1


def test_auto_failure_is_emailed_and_retried(box, clock):
    clock("2026-11-30 18:30")

    def blocked():
        raise fw.prices.YahooBlocked("no bars")
    assert box.run(record_fn=blocked) == 2
    assert box.sent[0][0] == "Swing rank S1: list NOT recorded" and "YahooBlocked: no bars" in box.sent[0][1]
    assert not box.path.exists() and (box.cache / "swing_rank_fwd_20261130.db").exists()
    clock("2026-11-30 20:15")
    assert box.run() == 0 and len(fw.load_record(box.path)) == 10


def test_a_failing_score_does_not_lose_the_list(box, clock):
    clock("2026-11-30 18:30")
    assert box.run(score_fn=lambda: 1 / 0) == 0
    assert len(fw.load_record(box.path)) == 10 and "not available: ZeroDivisionError" in box.sent[0][1]


def test_score_text(tmp_path, monkeypatch, clock):
    clock("2026-12-04 18:00")
    path = tmp_path / "r.csv"
    assert fw.score_text(path) == "Nothing recorded yet."
    monkeypatch.setattr(fw, "fetch_prices", lambda symbols, db, end: flat())
    sym = [n + ".TO" for n in NAMES]
    rec(("2026-12-07", sym)).to_csv(path, index=False)
    monkeypatch.setattr(fw, "load_record", lambda p: rec(("2026-12-07", sym)))
    assert "not started" in fw.score_text(path)
    monkeypatch.setattr(fw, "load_record", lambda p: rec(("2026-12-01", sym), ("2026-12-03", sym)))
    assert "1 complete months: S1 -0.20%, XIU +0.00%" in fw.score_text(path)


def test_fetch_prices_asks_again_for_names_without_data(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fw.prices, "download", lambda conn, symbols, start, end, retry=False: calls.append((symbols, end, retry)))
    fw.fetch_prices(["B.TO", "A.TO"], tmp_path / "p.db", date(2026, 11, 30))
    assert calls == [(["A.TO", "B.TO"], "2026-11-30", False), (["A.TO", "B.TO"], "2026-11-30", True)]


def test_the_default_record_is_the_state_file():
    assert fw.RECORD.parts[-2:] == ("data", "swing_rank_forward.csv")
