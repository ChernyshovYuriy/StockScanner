"""
The 16:30 pipelines (main.py, momentum_pipeline.py, kangaroo_pipeline.py) must
not run on a TSX holiday: their timers fire Mon-Fri, and a holiday run's
save_intents() deleted the previous session's still-PENDING intents
(main.py, 2026-10-12, Thanksgiving).
"""

from __future__ import annotations

import runpy
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from time_utils import TSX_TZ, set_backtest_clock

ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = str(ROOT / "main.py")
MOMENTUM_PY = str(ROOT / "momentum_pipeline.py")
HOLIDAY = datetime(2026, 10, 12, 16, 30, tzinfo=TSX_TZ)  # Thanksgiving
TRADING_DAY = datetime(2026, 10, 13, 16, 30, tzinfo=TSX_TZ)


class _Lock:
    def close(self):
        pass


def _run_main(monkeypatch, tmp_path, when: datetime) -> list:
    """Run main.py as __main__ at `when`, offline; return the steps it reached."""
    calls: list = []

    class _Screener:
        def __init__(self, *a, **k):
            calls.append("screener")

        def run(self, force_refresh=False):
            return pd.DataFrame()

    monkeypatch.setattr("concurrent_utils.acquire_lock", lambda service: (tmp_path / "lock", _Lock()))
    monkeypatch.setattr("log_utils.log", lambda *a, **k: None)
    monkeypatch.setattr("market_data.DEFAULT_PROVIDER.get", lambda *a, **k: None)  # regime -> default BULL
    monkeypatch.setattr("canadian_stock_screener.DataManager", lambda *a, **k: None)
    monkeypatch.setattr("canadian_stock_screener.StockScreener", _Screener)
    monkeypatch.setattr("auto_pipeline.run_pipeline", lambda cfg: calls.append("pipeline"))
    monkeypatch.setattr("send_report.send_report", lambda cfg: calls.append("report"))
    monkeypatch.setattr("sys.argv", ["main.py"])
    set_backtest_clock(when)
    try:
        runpy.run_path(MAIN_PY, run_name="__main__")
    except SystemExit as exc:
        calls.append(f"exit {exc.code}")
    finally:
        set_backtest_clock(None)
    return calls


def test_holiday_skips_everything(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, HOLIDAY)
    assert calls == ["exit 0"]


def test_trading_day_runs_the_pipeline(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, TRADING_DAY)
    assert calls == ["screener", "pipeline", "report"]


class _Stop(Exception):
    """Raised by the patched init_db() to end a trading-day run at its first DB touch."""


def _run_sleeve(monkeypatch, tmp_path, when: datetime, run) -> list:
    """Run a sleeve pipeline at `when`; init_db() is its first DB/network
    step after the holiday gate, so record it and stop there."""
    calls: list = []

    def _init_db(*a, **k):
        calls.append("init_db")
        raise _Stop

    monkeypatch.setattr("concurrent_utils.acquire_lock", lambda service: (tmp_path / "lock", _Lock()))
    monkeypatch.setattr("log_utils.log", lambda *a, **k: None)
    monkeypatch.setattr("db.init_db", _init_db)
    set_backtest_clock(when)
    try:
        run(monkeypatch)
    except SystemExit as exc:
        calls.append(f"exit {exc.code}")
    except _Stop:
        pass
    finally:
        set_backtest_clock(None)
    return calls


def _momentum(monkeypatch):
    runpy.run_path(MOMENTUM_PY, run_name="__main__")


def _kangaroo(monkeypatch):
    import kangaroo_pipeline as kp
    monkeypatch.setattr(kp, "acquire_lock", lambda service: (Path("lock"), _Lock()))
    monkeypatch.setattr(kp, "log", lambda *a, **k: None)
    monkeypatch.setattr(kp, "init_db", __import__("db").init_db)
    kp.main()


@pytest.mark.parametrize("run", [_momentum, _kangaroo], ids=["momentum", "kangaroo"])
def test_sleeve_pipeline_skips_holiday(monkeypatch, tmp_path, run):
    assert _run_sleeve(monkeypatch, tmp_path, HOLIDAY, run) == ["exit 0"]


@pytest.mark.parametrize("run", [_momentum, _kangaroo], ids=["momentum", "kangaroo"])
def test_sleeve_pipeline_runs_on_trading_day(monkeypatch, tmp_path, run):
    assert _run_sleeve(monkeypatch, tmp_path, TRADING_DAY, run) == ["init_db"]
