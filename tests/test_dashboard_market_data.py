"""dashboard_market_data.py: one batched Yahoo call per kind, memoized,
with today's daily row rebuilt from the intraday snapshot."""
from datetime import date

import pandas as pd
import pytest

import dashboard_market_data as dmd
from market_data import TodayBar

TODAY = pd.Timestamp("2026-11-20")
_real_prefetch = dmd.prefetch  # captured at import, before conftest's no-op patch


def _bars(last_day):
    days = pd.bdate_range(end=last_day, periods=40)
    return pd.DataFrame({"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.0, "Volume": 1000.0}, index=days)


@pytest.fixture
def env(monkeypatch):
    # undo conftest's no-op so the real prefetch runs against fakes
    monkeypatch.setattr(dmd, "prefetch", _real_prefetch)
    calls = {"range": [], "snaps": [], "single": []}
    state = {"open": True, "data": {}, "snaps": {}}

    def download_range(tickers, start, end):
        calls["range"].append(list(tickers))
        return {t: state["data"][t] for t in tickers if t in state["data"]}, [t for t in tickers if t not in state["data"]]

    def get_intraday_snapshots(tickers):
        calls["snaps"].append(list(tickers))
        return {t: state["snaps"][t] for t in tickers if t in state["snaps"]}

    def load_one(ticker, start):
        calls["single"].append(ticker)
        return _bars("2026-09-04")

    monkeypatch.setattr(dmd.DEFAULT_PROVIDER, "download_range", download_range)
    monkeypatch.setattr(dmd.DEFAULT_PROVIDER, "get_intraday_snapshots", get_intraday_snapshots)
    monkeypatch.setattr(dmd, "_load_one", load_one)
    monkeypatch.setattr(dmd, "is_market_open", lambda: state["open"])
    monkeypatch.setattr(dmd, "market_today", lambda: TODAY.to_pydatetime())
    return calls, state


def test_one_batched_call_each_then_memoized(env):
    calls, state = env
    state["data"] = {"AAA.TO": _bars(TODAY), "BBB.TO": _bars(TODAY)}
    state["snaps"] = {"AAA.TO": TodayBar(low=8.5, close=12.0, high=12.5, source="5m-intraday")}
    starts = {"AAA.TO": date(2026, 1, 1), "BBB.TO": date(2026, 2, 1)}

    dmd.prefetch(starts, intraday=list(starts))
    dmd.prefetch(starts, intraday=list(starts))  # within both TTLs: no new calls

    assert calls["range"] == [["AAA.TO", "BBB.TO"]]
    assert calls["snaps"] == [["AAA.TO", "BBB.TO"]]
    assert dmd.intraday_snapshot("AAA.TO").close == 12.0
    assert dmd.intraday_snapshot("BBB.TO") is None

    df = dmd.load_daily_bars("AAA.TO", date(2026, 1, 1))
    last = df.iloc[-1]
    assert df.index[-1] == TODAY and len(df) == 40
    assert (last["Close"], last["High"], last["Low"]) == (12.0, 12.5, 8.5)
    # no snapshot -> the memo's own today row is kept unchanged
    assert dmd.load_daily_bars("BBB.TO", date(2026, 2, 1)).iloc[-1]["Close"] == 10.0
    assert calls["single"] == []


def test_failed_ticker_falls_back_to_csv_loader(env):
    calls, state = env
    dmd.prefetch({"SES.TO": date(2026, 1, 1)})
    assert calls["single"] == ["SES.TO"]
    assert dmd.load_daily_bars("SES.TO", date(2026, 1, 1)).index[-1] == pd.Timestamp("2026-09-04")


def test_refetch_after_close_and_for_earlier_start(env):
    calls, state = env
    state["data"] = {"AAA.TO": _bars(TODAY)}
    dmd.prefetch({"AAA.TO": date(2026, 1, 1)})
    state["open"] = False
    dmd.prefetch({"AAA.TO": date(2026, 1, 1)})   # fetched mid-session, now closed
    dmd.prefetch({"AAA.TO": date(2026, 1, 1)})   # fresh closed-market memo
    dmd.prefetch({"AAA.TO": date(2025, 1, 1)})   # needs earlier history
    assert len(calls["range"]) == 3
