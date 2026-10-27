"""
Every dashboard position view shows a position whose price data stopped
updating as NO_DATA ("Stale data: last bar ...") instead of a HOLD/SELL
computed on the cached bar — the same check position_monitor.py and the
sleeve monitors apply (SES.TO, 2026-09, read HOLD on a 7-week-old price).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import dashboard_positions
import db as db_module
import kangaroo_dashboard_data
import macro_dashboard_data
import momentum_dashboard_data
from db import init_db, insert_position, set_cash
from time_utils import TSX_TZ, set_backtest_clock

NOW = datetime(2026, 5, 14, 15, 50, tzinfo=TSX_TZ)  # Thursday; previous trading day 05-13

VIEWS = {
    "core": (dashboard_positions, "_build_live_positions", None),
    "momentum": (momentum_dashboard_data, "_build_momentum_positions", "MOMENTUM_DB_PATH"),
    "macro": (macro_dashboard_data, "_build_macro_positions", "MACRO_DB_PATH"),
    "kangaroo": (kangaroo_dashboard_data, "_build_kangaroo_positions", "KANGAROO_DB_PATH"),
}


@pytest.fixture(autouse=True)
def clock():
    set_backtest_clock(NOW)
    yield
    set_backtest_clock(None)


@pytest.fixture(autouse=True)
def _reset_db_path(tmp_path):
    yield
    db_module.DB_PATH = tmp_path / "reset.db"


def _build(monkeypatch, tmp_path, name: str, last_bar_day: str) -> dict:
    module, builder, db_attr = VIEWS[name]
    days = pd.bdate_range(end=last_bar_day, periods=60)
    df = pd.DataFrame({"Open": 40.0, "High": 40.5, "Low": 39.5, "Close": 40.0, "Volume": 1_000}, index=days)

    path = tmp_path / f"{name}.db"
    init_db(path)
    set_cash(1_000.0)
    insert_position("AAA.TO", days[55].date().isoformat(), 40.0, 10, stop_price=30.0, target_price=60.0)

    if db_attr is not None:
        monkeypatch.setattr(module, db_attr, path)
    monkeypatch.setattr(module, "load_or_fetch_data", lambda t, start: df)
    monkeypatch.setattr(module, "fetch_intraday_snapshot", lambda t: None)
    monkeypatch.setattr(module, "is_market_open", lambda: False)
    rows = getattr(module, builder)()
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("name", VIEWS)
def test_stale_position_shows_no_data(monkeypatch, tmp_path, name):
    row = _build(monkeypatch, tmp_path, name, last_bar_day="2026-05-06")
    assert row["status"] == "NO_DATA"
    assert "Stale data: last bar 2026-05-06" in row["reason"]


@pytest.mark.parametrize("name", VIEWS)
def test_previous_sessions_bar_is_evaluated(monkeypatch, tmp_path, name):
    row = _build(monkeypatch, tmp_path, name, last_bar_day="2026-05-13")
    assert row["status"] == "HOLD"
