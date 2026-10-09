"""
Point-in-time backtest universe (2026-10).

The live CAN_TICKERS_URL list is rebuilt every week from the raw list by
swing_tickers.py's filters. Backtests used to screen TODAY's copy of that
list over past years — names selected on today's readings (look-ahead).
With BacktestConfig.universe_filter set, the backtest instead screens, each
week, only the raw-list names that passed the filters on the previous
week's close.
"""
from __future__ import annotations

import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import backtest_runner
import run_backtest
import time_utils
from backtest_runner import BacktestConfig, _point_in_time_universe, _run_screener_step
from config import BACKTEST_RAW_TICKERS_URL, CAN_TICKERS_URL
from market_data import HistoricalSliceProvider
from swing_tickers import Thresholds
from time_utils import TSX_TZ, set_backtest_clock

DAYS = pd.bdate_range(end="2024-06-28", periods=300)  # ends on a Friday


def _bars(volume, start=50.0, drift=0.001) -> pd.DataFrame:
    close = start * (1 + drift) ** np.arange(len(DAYS))  # steady uptrend, ~1% daily range
    vol = np.broadcast_to(np.asarray(volume, dtype=float), len(DAYS)).copy()
    return pd.DataFrame({"Open": close, "High": close * 1.005, "Low": close * 0.995,
                         "Close": close, "Volume": vol}, index=DAYS)


def _late_volume():
    # Thin (fails the $1M/day liquidity filter) until Wednesday 2024-06-19,
    # then heavy from that close on.
    v = np.full(len(DAYS), 1_000.0)
    v[DAYS >= pd.Timestamp("2024-06-19")] = 5_000_000.0
    return v


@pytest.fixture
def provider():
    return HistoricalSliceProvider({
        "XIU.TO": _bars(1_000_000, start=30.0),
        "GOOD.TO": _bars(100_000),        # $5M/day, uptrend, quiet: passes
        "THIN.TO": _bars(1_000),          # $50k/day: fails liquidity
        "LATE.TO": _bars(_late_volume()), # becomes liquid mid-week
    })


def _cfg(**kw):
    base = dict(tickers=["XIU.TO", "GOOD.TO", "THIN.TO", "LATE.TO"], benchmark="XIU.TO",
                universe_filter=Thresholds())
    base.update(kw)
    return BacktestConfig(**base)


def test_universe_applies_swing_filters(provider):
    assert _point_in_time_universe(_cfg(), provider, pd.Timestamp("2024-06-18")) == ["GOOD.TO"]


def test_no_look_ahead_within_the_week(provider):
    # LATE.TO qualifies on Wednesday's close; the list for that week was
    # built from the previous Friday, so it only joins the following week.
    cfg = _cfg()
    assert "LATE.TO" not in _point_in_time_universe(cfg, provider, pd.Timestamp("2024-06-20"))
    assert "LATE.TO" in _point_in_time_universe(cfg, provider, pd.Timestamp("2024-06-24"))


def test_bars_after_the_week_start_are_never_read(provider):
    # Corrupt every bar from Monday 2024-06-24 on: the 06-24 week's list must
    # be built from 06-21 and earlier only, so it can't change.
    cfg = _cfg()
    before = _point_in_time_universe(cfg, provider, pd.Timestamp("2024-06-24"))
    for t, df in provider._data.items():
        df.loc[df.index >= pd.Timestamp("2024-06-24"), ["Close", "High", "Low", "Open"]] = 0.01
    assert _point_in_time_universe(_cfg(), provider, pd.Timestamp("2024-06-26")) == before


def test_one_rebuild_per_week(provider, monkeypatch):
    calls = []
    import swing_tickers
    orig = swing_tickers.analyze_symbol
    monkeypatch.setattr(swing_tickers, "analyze_symbol", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    cfg = _cfg()
    for d in ("2024-06-17", "2024-06-18", "2024-06-21"):
        _point_in_time_universe(cfg, provider, pd.Timestamp(d))
    assert len(calls) == 3  # 3 tickers, one rebuild for the whole week


def test_caller_clock_is_restored(provider):
    pinned = datetime(2024, 6, 18, 16, 5, tzinfo=TSX_TZ)
    set_backtest_clock(pinned)
    try:
        _point_in_time_universe(_cfg(), provider, pd.Timestamp("2024-06-18"))
        assert time_utils._backtest_now == pinned
    finally:
        set_backtest_clock(None)
    _point_in_time_universe(_cfg(), provider, pd.Timestamp("2024-06-11"))
    assert time_utils._backtest_now is None


def test_screener_step_screens_only_the_point_in_time_names(provider, monkeypatch):
    import canadian_stock_screener as css
    seen = {}

    class _DM(css.DataManager):
        def __init__(self, tickers_source, provider=None):
            seen["tickers"] = list(tickers_source)
            super().__init__(tickers_source=tickers_source, provider=provider)

    monkeypatch.setattr(css, "DataManager", _DM)
    _run_screener_step(_cfg(), provider, pd.Timestamp("2024-06-20"))
    assert seen["tickers"] == ["GOOD.TO"]

    _run_screener_step(_cfg(universe_filter=None), provider, pd.Timestamp("2024-06-20"))
    assert seen["tickers"] == ["GOOD.TO", "THIN.TO", "LATE.TO"]  # legacy: everything


def test_empty_universe_screens_nothing(provider):
    cfg = _cfg(tickers=["XIU.TO", "THIN.TO"])
    assert _run_screener_step(cfg, provider, pd.Timestamp("2024-06-20")).empty


# ── run_backtest.py: point-in-time is the default ───────────────────────────

def _cli(monkeypatch, argv):
    seen = {}

    class _P:
        def __len__(self):
            return 1

    monkeypatch.setattr(run_backtest, "_load_tickers", lambda src: (seen.setdefault("src", src), ["AAA.TO"])[1])
    monkeypatch.setattr(run_backtest, "_fetch_provider", lambda **kw: (_P(), None))
    monkeypatch.setattr(run_backtest, "_run_single", lambda args, *a, **k: seen.setdefault("filter", run_backtest._universe_filter(args)))
    monkeypatch.setattr(sys, "argv", ["run_backtest.py", *argv])
    run_backtest.main()
    return seen


def test_cli_defaults_to_raw_list_screened_point_in_time(monkeypatch):
    seen = _cli(monkeypatch, [])
    assert seen["src"] == BACKTEST_RAW_TICKERS_URL
    assert isinstance(seen["filter"], Thresholds)


def test_cli_static_universe_restores_legacy(monkeypatch):
    seen = _cli(monkeypatch, ["--static-universe"])
    assert seen["src"] == CAN_TICKERS_URL
    assert seen["filter"] is None


def test_cli_custom_tickers_kept(monkeypatch):
    seen = _cli(monkeypatch, ["--tickers", "my_list.txt"])
    assert seen["src"] == "my_list.txt"
    assert isinstance(seen["filter"], Thresholds)
