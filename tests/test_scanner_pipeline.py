"""
tests/test_scanner_pipeline.py
================================
Offline integration tests for scanner_pipeline.run_pipeline — see
scanner_board/PLAN.md Phase 4. market_data_cache.sync_and_load is always
monkeypatched to a fixture, never a real network call.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

import scanner_pipeline
from scanner_board import store
from time_utils import TSX_TZ, set_backtest_clock


@pytest.fixture(autouse=True)
def _reset_backtest_clock():
    yield
    set_backtest_clock(None)


def _pin_clock(d: date) -> None:
    set_backtest_clock(datetime(d.year, d.month, d.day, 17, 15, tzinfo=TSX_TZ))


# 2024-06-10 is a Monday, not a TSX holiday.
A_TRADING_DAY = date(2024, 6, 10)
A_SATURDAY = date(2024, 6, 8)


def _synthetic_daily(n: int, end: str, seed: int = 1) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(0.05 + rng.normal(0, 0.3, n))
    high = close + np.abs(rng.normal(0.2, 0.1, n))
    low = close - np.abs(rng.normal(0.2, 0.1, n))
    open_ = close - rng.normal(0, 0.1, n)
    volume = np.abs(rng.normal(100_000, 10_000, n))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


@pytest.fixture()
def conn(tmp_path):
    c = store.connect(tmp_path / "scanner_pipeline_test.db")
    yield c
    c.close()


class TestNotATradingDay:
    def test_skips_without_calling_sync_and_load(self, monkeypatch, conn):
        called = []
        monkeypatch.setattr(scanner_pipeline, "sync_and_load", lambda *a, **kw: called.append(1) or {})
        _pin_clock(A_SATURDAY)
        scanner_pipeline.run_pipeline("run1", tickers=["AAA.TO"], conn=conn)
        assert called == []
        assert store.latest_rows(conn) == []


class TestComputeAndPersist:
    def test_computes_and_persists_the_universe(self, monkeypatch, conn):
        daily = _synthetic_daily(260, end="2024-06-07")
        monkeypatch.setattr(scanner_pipeline, "sync_and_load",
                             lambda tickers, **kw: {"AAA.TO": daily})
        _pin_clock(A_TRADING_DAY)

        scanner_pipeline.run_pipeline("run1", tickers=["AAA.TO"], conn=conn)

        got = store.rows_for_run_date(conn, A_TRADING_DAY.isoformat())
        assert len(got) == 1
        assert got[0]["ticker"] == "AAA.TO"
        assert got[0]["price"] is not None

    def test_dry_run_computes_but_writes_nothing(self, monkeypatch, conn, capsys):
        daily = _synthetic_daily(260, end="2024-06-07")
        monkeypatch.setattr(scanner_pipeline, "sync_and_load",
                             lambda tickers, **kw: {"AAA.TO": daily})
        _pin_clock(A_TRADING_DAY)

        scanner_pipeline.run_pipeline("run1", dry_run=True, tickers=["AAA.TO"], conn=conn)

        assert store.latest_rows(conn) == []
        out = capsys.readouterr().out
        assert "AAA.TO" in out

    def test_too_short_history_is_skipped_not_crashed(self, monkeypatch, conn):
        short_daily = _synthetic_daily(5, end="2024-06-07")  # below MIN_DAILY_BARS
        monkeypatch.setattr(scanner_pipeline, "sync_and_load",
                             lambda tickers, **kw: {"AAA.TO": short_daily})
        _pin_clock(A_TRADING_DAY)

        scanner_pipeline.run_pipeline("run1", tickers=["AAA.TO"], conn=conn)

        assert store.rows_for_run_date(conn, A_TRADING_DAY.isoformat()) == []

    def test_ticker_missing_from_sync_result_is_skipped_not_crashed(self, monkeypatch, conn):
        """sync_and_load() returning nothing for a delisted/broken ticker
        (e.g. yfinance failure) must not abort the run for every other
        ticker in the universe."""
        good_daily = _synthetic_daily(260, end="2024-06-07")
        monkeypatch.setattr(scanner_pipeline, "sync_and_load",
                             lambda tickers, **kw: {"GOOD.TO": good_daily})  # BAD.TO absent
        _pin_clock(A_TRADING_DAY)

        scanner_pipeline.run_pipeline("run1", tickers=["GOOD.TO", "BAD.TO"], conn=conn)

        got = store.rows_for_run_date(conn, A_TRADING_DAY.isoformat())
        assert {r["ticker"] for r in got} == {"GOOD.TO"}

    def test_one_tickers_compute_row_failure_does_not_block_the_others(self, monkeypatch, conn):
        good_daily = _synthetic_daily(260, end="2024-06-07")
        bad_daily = _synthetic_daily(260, end="2024-06-07", seed=2)
        monkeypatch.setattr(
            scanner_pipeline, "sync_and_load",
            lambda tickers, **kw: {"GOOD.TO": good_daily, "BAD.TO": bad_daily})

        real_compute_row = scanner_pipeline.row_mod.compute_row

        def flaky_compute_row(ticker, daily, weekly):
            if ticker == "BAD.TO":
                raise ValueError("synthetic failure")
            return real_compute_row(ticker, daily, weekly)

        monkeypatch.setattr(scanner_pipeline.row_mod, "compute_row", flaky_compute_row)
        _pin_clock(A_TRADING_DAY)

        scanner_pipeline.run_pipeline("run1", tickers=["GOOD.TO", "BAD.TO"], conn=conn)

        got = store.rows_for_run_date(conn, A_TRADING_DAY.isoformat())
        assert {r["ticker"] for r in got} == {"GOOD.TO"}
