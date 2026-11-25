"""VCP study Phase 1 -- the paced, resumable price download. No network."""
from __future__ import annotations

import pandas as pd
import pytest

import market_data_cache as mdc
from research.vcp import prices
from research.vcp.prices import (
    BENCHMARK, CHUNK, NO_DATA, OK, Pacer, YahooBlocked, candidate_tickers, connect, download,
    load_all, statuses,
)

START, END = "2024-01-02", "2024-03-01"


def _bars(start=START, end=END, price=10.0):
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({"Open": price, "High": price + 1, "Low": price - 1, "Close": price,
                         "Volume": 1000}, index=idx)


class FakeYahoo:
    """Stands in for market_data_cache.sync_tickers: writes bars for known
    names, silently nothing for unknown ones, nothing at all once blocked."""

    def __init__(self, known, block_after=None):
        self.known, self.block_after, self.calls = set(known) | {BENCHMARK}, block_after, []

    def __call__(self, tickers, start, end, conn=None, force_refresh=False, quiet=False):
        self.calls.append((list(tickers), start, end, force_refresh))
        if self.block_after is not None and len(self.calls) > self.block_after:
            return
        for t in tickers:
            if t in self.known:
                mdc._upsert(conn, t, _bars(start, end))

    def asked(self):
        return [t for c in self.calls for t in c[0]]


class FakeClock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def sleep(self, s):
        self.slept.append(s)
        self.now += s

    def pacer(self, per_second=0.4):
        return Pacer(per_second, sleep=self.sleep, clock=lambda: self.now)


@pytest.fixture
def conn(tmp_path):
    c = connect(str(tmp_path / "p.db"))
    yield c
    c.close()


def _run(conn, tickers, yahoo, clock=None, **kw):
    clock = clock or FakeClock()
    return download(conn, tickers, START, END, sync_fn=yahoo, pacer=clock.pacer(),
                    progress=lambda m: None, **kw)


def _names(n, prefix="AA"):
    return [f"{prefix}{i:03d}.TO" for i in range(n)]


# ── candidates ───────────────────────────────────────────────────────────────

def test_candidates_keep_common_shares_only():
    raw = ["ry.to", " SHOP.TO ", "ABC.V", "XYZ.CN", "TECK-B.TO", "RCI-A.TO",      # keep
           "REI-UN.TO", "ABC-U.TO", "GPH-WTA.V", "X-RT.TO", "FC-DBM.TO", "TD-PFA.TO", "BN-PA.TO",
           "AAPL.NE", "AAPL", "", "XIU.TO"]
    assert candidate_tickers(raw) == ["ABC.V", "RCI-A.TO", "RY.TO", "SHOP.TO", "TECK-B.TO", "XYZ.CN"]


def test_candidates_are_deduplicated():
    assert candidate_tickers(["RY.TO", "ry.to", "RY.TO "]) == ["RY.TO"]


# ── pacing ───────────────────────────────────────────────────────────────────

def test_pacer_never_exceeds_rate():
    clock = FakeClock()
    p = clock.pacer(0.4)
    for _ in range(5):
        p.wait(10)
    assert clock.now == pytest.approx(4 * 10 / 0.4)     # the first booking is free, then 25 s each


def test_pacer_gives_no_credit_for_idle_time():
    clock = FakeClock()
    p = clock.pacer(0.4)
    p.wait(1)
    clock.now += 10_000
    p.wait(10)
    before = clock.now
    p.wait(1)
    assert clock.now - before == pytest.approx(25.0)


def test_whole_download_stays_under_rate(conn):
    clock, names = FakeClock(), _names(45)
    yahoo = FakeYahoo(names)
    _run(conn, names, yahoo, clock)
    booked_before_last = len(yahoo.asked()) - len(yahoo.calls[-1][0])
    assert clock.now >= booked_before_last / 0.4 - 1e-6
    assert max(len(c[0]) for c in yahoo.calls) <= CHUNK


def test_probe_requests_are_paced_too(conn):
    clock = FakeClock()
    yahoo = FakeYahoo([])
    _run(conn, _names(30), yahoo, clock)
    assert clock.now >= (len(yahoo.asked()) - 1) / 0.4 - 1e-6


# ── status, resume, retry ────────────────────────────────────────────────────

def test_records_ok_and_no_data(conn):
    names = _names(12)
    out = _run(conn, names, FakeYahoo(names[:7]))
    s = statuses(conn)
    assert [t for t in names if s[t] == OK] == names[:7]
    assert [t for t in names if s[t] == NO_DATA] == names[7:]
    assert out == {OK: 8, NO_DATA: 5}                   # + benchmark


def test_second_run_asks_for_nothing(conn):
    names = _names(12)
    _run(conn, names, FakeYahoo(names[:7]))
    again = FakeYahoo(names)
    _run(conn, names, again)
    assert again.calls == []


def test_retry_asks_only_for_no_data(conn):
    names = _names(12)
    _run(conn, names, FakeYahoo(names[:7]))
    again = FakeYahoo(names)
    _run(conn, names, again, retry=True)
    assert again.asked() == names[7:]
    assert set(statuses(conn).values()) == {OK}


def test_every_fetch_forces_the_full_window(conn):
    names = _names(3)
    yahoo = FakeYahoo(names)
    _run(conn, names, yahoo)
    assert all(c[1:] == (START, END, True) for c in yahoo.calls)


# ── being blocked ────────────────────────────────────────────────────────────

def test_block_stops_and_does_not_mark_names(conn):
    names = _names(50)
    yahoo = FakeYahoo(names, block_after=3)             # benchmark + 2 chunks answer
    with pytest.raises(YahooBlocked):
        _run(conn, names, yahoo)
    s = statuses(conn)
    assert sorted(s) == sorted(names[:20] + [BENCHMARK])
    assert NO_DATA not in s.values()
    assert len(yahoo.calls) == 5                        # failed chunk + probe, then nothing more


def test_resume_after_block_finishes(conn):
    names = _names(50)
    with pytest.raises(YahooBlocked):
        _run(conn, names, FakeYahoo(names, block_after=3))
    again = FakeYahoo(names)
    out = _run(conn, names, again)
    assert again.asked() == names[20:]
    assert out == {OK: 51, NO_DATA: 0}


def test_blocked_probe_leaves_benchmark_bars_intact(conn):
    names = _names(50)
    with pytest.raises(YahooBlocked):
        _run(conn, names, FakeYahoo(names, block_after=3))
    pd.testing.assert_frame_equal(load_all(conn)[BENCHMARK], _bars().astype(
        {"Open": float, "High": float, "Low": float, "Close": float, "Volume": "int64"}),
        check_freq=False, check_names=False, check_index_type=False, check_dtype=False)


def test_probe_is_not_fooled_by_stored_bars(conn):
    """The benchmark's old rows are in the file; a blocked probe must still count as blocked."""
    names = _names(10)
    _run(conn, names, FakeYahoo(names))
    with pytest.raises(YahooBlocked):
        _run(conn, names + _names(10, "ZZ"), FakeYahoo(names, block_after=0))


def test_dead_chunk_with_live_yahoo_is_no_data(conn):
    names = _names(20)
    out = _run(conn, names, FakeYahoo([]))
    assert out == {OK: 1, NO_DATA: 20}


def test_successful_probe_leaves_benchmark_complete(conn):
    _run(conn, _names(20), FakeYahoo([]))
    assert len(load_all(conn)[BENCHMARK]) == len(_bars())


def test_no_benchmark_at_start_is_a_block(conn):
    with pytest.raises(YahooBlocked):
        _run(conn, _names(5), FakeYahoo(_names(5), block_after=0))
    assert statuses(conn) == {}


def test_exception_in_sync_propagates_and_keeps_earlier_names(conn):
    names = _names(30)
    inner = FakeYahoo(names)

    def sync(tickers, *a, **k):
        if names[10] in tickers:
            raise RuntimeError("boom")
        inner(tickers, *a, **k)

    with pytest.raises(RuntimeError):
        _run(conn, names, sync)
    assert sorted(statuses(conn)) == sorted(names[:10] + [BENCHMARK])


# ── isolation and loading ────────────────────────────────────────────────────

def test_uses_its_own_file():
    assert prices.DB_PATH.endswith("cache/vcp_prices.db")
    assert "market_cache" not in prices.DB_PATH


def test_default_paths_never_touch_market_cache(conn, tmp_path, monkeypatch):
    shared = tmp_path / "market_cache.db"
    monkeypatch.setattr(mdc, "CACHE_DB_PATH", str(shared))
    _run(conn, _names(5), FakeYahoo(_names(5)))
    load_all(conn)
    assert not shared.exists()


def test_load_all_round_trips(conn):
    names = _names(3)
    _run(conn, names, FakeYahoo(names))
    data = load_all(conn)
    assert sorted(data) == sorted(names + [BENCHMARK])
    df = data[names[1]]
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df["High"].iloc[0] == 11.0 and df["Low"].iloc[0] == 9.0 and len(df) == len(_bars())
