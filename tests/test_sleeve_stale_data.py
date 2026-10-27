"""
Every sleeve monitor flags a position whose price data stopped updating as
NO_DATA and alerts, instead of evaluating (HOLD forever) on the cached bar —
the check position_monitor.py gained after SES.TO (2026-09) read HOLD on a
7-week-old price. Offline: data fetch, lock, report and email all patched.
"""

from __future__ import annotations

import sys
from datetime import datetime

import pandas as pd
import pytest

import db as db_module
import kangaroo_monitor
import macro_monitor
import momentum_monitor
import position_monitor
from db import get_open_positions, init_db, insert_position, set_cash
from time_utils import TSX_TZ, set_backtest_clock

NOW = datetime(2026, 5, 14, 15, 50, tzinfo=TSX_TZ)  # Thursday; previous trading day 05-13

SLEEVES = {
    "momentum": (momentum_monitor, "MOMENTUM_DB_PATH", "Momentum Monitor"),
    "macro": (macro_monitor, "MACRO_DB_PATH", "Macro Monitor"),
    "kangaroo": (kangaroo_monitor, "KANGAROO_DB_PATH", "Kangaroo Monitor"),
}


class _Lock:
    def close(self):
        pass


@pytest.fixture(autouse=True)
def clock():
    set_backtest_clock(NOW)
    yield
    set_backtest_clock(None)


@pytest.fixture(autouse=True)
def _reset_db_path(tmp_path):
    yield
    db_module.DB_PATH = tmp_path / "reset.db"


def _run(monkeypatch, tmp_path, name: str, last_bar_day: str) -> tuple[list, pd.DataFrame]:
    """Run one sleeve's monitor in pre-close with a position held well above
    its stop, the live snapshot unavailable, and daily bars ending on
    `last_bar_day`. Returns (alerts sent, the run's log CSV)."""
    module, db_attr, _ = SLEEVES[name]
    days = pd.bdate_range(end=last_bar_day, periods=60)
    df = pd.DataFrame({"Open": 40.0, "High": 40.5, "Low": 39.5, "Close": 40.0, "Volume": 1_000}, index=days)

    path = tmp_path / f"{name}.db"
    init_db(path)
    set_cash(1_000.0)
    insert_position("AAA.TO", days[55].date().isoformat(), 40.0, 10, stop_price=30.0, target_price=60.0)

    sent: list = []
    monkeypatch.setattr(module, db_attr, path)
    monkeypatch.setattr(module, "acquire_lock", lambda service: (None, _Lock()))
    monkeypatch.setattr(module, "log", lambda *a, **k: None)
    monkeypatch.setattr(module, "is_market_open", lambda: True)
    monkeypatch.setattr(module, "fetch_intraday_snapshot", lambda t: None)
    monkeypatch.setattr(module, "load_or_fetch_data", lambda t, start: df)
    monkeypatch.setattr(module, "LOGS_PATH", tmp_path)
    monkeypatch.setattr(module, "append_positions_report", lambda *a, **k: None)
    monkeypatch.setattr(module, "__run_send_report", lambda: None)
    monkeypatch.setattr(module, "send_stale_data_alert", lambda stale, **k: sent.append((stale, k)))
    monkeypatch.setattr(position_monitor, "send_transaction_email", lambda *a, **k: None)
    if module is macro_monitor:
        monkeypatch.setattr(module, "get_macro_regime",
                            lambda: {"label": "neutral", "composite": 0, "votes": {}, "detail": {}, "fetched_at": "x"})
    monkeypatch.setattr(sys, "argv", [f"{name}_monitor.py", "--mode", "pre-close"])
    module.main()
    log = pd.read_csv(tmp_path / f"{name}_monitor_20260514.csv")
    return sent, log


@pytest.mark.parametrize("name", SLEEVES)
def test_stale_position_is_no_data_and_alerted(monkeypatch, tmp_path, name):
    sent, log = _run(monkeypatch, tmp_path, name, last_bar_day="2026-05-06")
    assert log.loc[0, "status"] == "NO_DATA"
    assert "2026-05-06" in log.loc[0, "reason"]
    assert len(get_open_positions()) == 1
    assert len(sent) == 1
    stale, kwargs = sent[0]
    assert [s["ticker"] for s in stale] == ["AAA.TO"]
    assert kwargs["title"] == SLEEVES[name][2]
    assert f"--sleeve {name}" in kwargs["sell_cmd"]


@pytest.mark.parametrize("name", SLEEVES)
def test_previous_sessions_bar_is_evaluated_normally(monkeypatch, tmp_path, name):
    sent, log = _run(monkeypatch, tmp_path, name, last_bar_day="2026-05-13")
    assert log.loc[0, "status"] == "HOLD"
    assert sent == []


def test_alert_names_the_sleeve(monkeypatch):
    sent = []
    monkeypatch.setattr(position_monitor, "send_text_email", lambda subject, body: sent.append((subject, body)))
    stale = [{"ticker": "AAA.TO", "last_bar": "2026-05-06", "entry_date": "2026-04-01",
              "shares": 10.0, "last_close": 40.0}]
    position_monitor.send_stale_data_alert(stale, title="Momentum Monitor",
                                           sell_cmd="python manual_sell.py TICKER --sleeve momentum --price X")
    subject, body = sent[0]
    assert subject == "Momentum Monitor: stale price data (1)"
    assert "AAA.TO" in body and "--sleeve momentum" in body
