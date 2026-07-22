"""
tests/test_dashboard_positions.py
==================================
Unit tests for dashboard_positions.build_live_positions() — the read-only
live-position pipeline used by the web dashboard's Monitor page. Mirrors
position_monitor.py's own pipeline but never sells anything.
"""

from __future__ import annotations

import pandas as pd
import pytest

import dashboard_positions
import db as db_module
from db import init_db, insert_position, set_cash


@pytest.fixture(autouse=True)
def db(tmp_path):
    path = tmp_path / "trading.db"
    init_db(path)
    yield path
    db_module.DB_PATH = tmp_path / "reset.db"


@pytest.fixture(autouse=True)
def reset_cache():
    """The TTL cache is module-level state — reset it around every test."""
    dashboard_positions._cache["rows"] = None
    dashboard_positions._cache["ts"] = 0.0
    yield
    dashboard_positions._cache["rows"] = None
    dashboard_positions._cache["ts"] = 0.0


def _fake_df(n: int = 40) -> pd.DataFrame:
    idx = pd.bdate_range(start="2026-01-01", periods=n)
    close = pd.Series([100.0 + i * 0.5 for i in range(n)], index=idx)
    return pd.DataFrame({
        "Open": close,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": 1000,
    }, index=idx)


class _TodayBar:
    def __init__(self, close: float = 110.0):
        self.close = close
        self.low = close - 1
        self.high = close + 1
        self.source = "5m-intraday"


def test_build_live_positions_empty_when_no_positions():
    set_cash(0.0)
    assert dashboard_positions.build_live_positions() == []


def test_build_live_positions_no_data_row_when_insufficient_bars(monkeypatch):
    set_cash(0.0)
    insert_position("RY.TO", "2026-01-05", 100.0, 10)

    monkeypatch.setattr("dashboard_positions.load_or_fetch_data", lambda ticker, start: pd.DataFrame())
    monkeypatch.setattr("dashboard_positions.is_market_open", lambda: False)

    rows = dashboard_positions.build_live_positions()

    assert len(rows) == 1
    assert rows[0]["status"] == "NO_DATA"


def test_build_live_positions_returns_compute_signals_row(monkeypatch):
    set_cash(0.0)
    df = _fake_df()
    entry_date = df.index[10].date().isoformat()
    insert_position("RY.TO", entry_date, 105.0, 10)

    monkeypatch.setattr("dashboard_positions.load_or_fetch_data", lambda ticker, start: df)
    monkeypatch.setattr("dashboard_positions.is_market_open", lambda: False)

    rows = dashboard_positions.build_live_positions()

    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "RY.TO"
    assert row["status"] in ("HOLD", "SELL")
    assert "pnl_%" in row
    assert "stop_price" in row


def test_build_live_positions_fetches_intraday_snapshot_when_market_open(monkeypatch):
    set_cash(0.0)
    df = _fake_df()
    entry_date = df.index[10].date().isoformat()
    insert_position("RY.TO", entry_date, 105.0, 10)

    calls = []
    monkeypatch.setattr("dashboard_positions.load_or_fetch_data", lambda ticker, start: df)
    monkeypatch.setattr("dashboard_positions.is_market_open", lambda: True)
    monkeypatch.setattr(
        "dashboard_positions.fetch_intraday_snapshot",
        lambda ticker: calls.append(ticker) or _TodayBar(),
    )

    dashboard_positions.build_live_positions()

    assert calls == ["RY.TO"]


def test_build_live_positions_skips_intraday_snapshot_when_market_closed(monkeypatch):
    set_cash(0.0)
    df = _fake_df()
    entry_date = df.index[10].date().isoformat()
    insert_position("RY.TO", entry_date, 105.0, 10)

    calls = []
    monkeypatch.setattr("dashboard_positions.load_or_fetch_data", lambda ticker, start: df)
    monkeypatch.setattr("dashboard_positions.is_market_open", lambda: False)
    monkeypatch.setattr(
        "dashboard_positions.fetch_intraday_snapshot",
        lambda ticker: calls.append(ticker) or _TodayBar(),
    )

    dashboard_positions.build_live_positions()

    assert calls == []


def test_build_live_positions_caches_within_ttl(monkeypatch):
    set_cash(0.0)
    df = _fake_df()
    entry_date = df.index[10].date().isoformat()
    insert_position("RY.TO", entry_date, 105.0, 10)

    call_count = {"n": 0}

    def _counting_loader(ticker, start):
        call_count["n"] += 1
        return df

    monkeypatch.setattr("dashboard_positions.load_or_fetch_data", _counting_loader)
    monkeypatch.setattr("dashboard_positions.is_market_open", lambda: False)

    dashboard_positions.build_live_positions()
    dashboard_positions.build_live_positions()

    assert call_count["n"] == 1
