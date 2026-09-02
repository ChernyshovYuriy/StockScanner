"""Offline tests for macro_monitor.py (no network — load_or_fetch_data,
fetch_intraday_snapshot, compute_signals, and get_macro_regime are all
monkeypatched; send_transaction_email is monkeypatched at its source in
position_monitor.py, since execute_virtual_sells() calls it from there, so a
real .env's Gmail credentials are never touched by the test suite)."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

import db as db_module
import macro_monitor
import position_monitor
from db import get_open_positions_df, init_db, insert_position, set_cash
from schema_keys import POSITION_COL_REASON, POSITION_COL_STATUS


class _FakeLockFile:
    def close(self):
        pass


@pytest.fixture(autouse=True)
def _no_lock(monkeypatch):
    monkeypatch.setattr(macro_monitor, "acquire_lock", lambda service: (None, _FakeLockFile()))


@pytest.fixture(autouse=True)
def _no_report_io(monkeypatch, tmp_path):
    monkeypatch.setattr(macro_monitor, "LOGS_PATH", tmp_path / "logs")
    monkeypatch.setattr(macro_monitor, "append_positions_report", lambda *a, **k: None)
    monkeypatch.setattr(macro_monitor, "__run_send_report", lambda: None)


@pytest.fixture(autouse=True)
def _no_email(monkeypatch):
    # execute_virtual_sells() (imported unchanged from position_monitor.py)
    # calls send_transaction_email via ITS OWN import binding, not
    # macro_monitor's — must be patched at the source.
    monkeypatch.setattr(position_monitor, "send_transaction_email", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _reset_db_path(tmp_path):
    yield
    db_module.DB_PATH = tmp_path / "reset.db"


def _regime(label, composite=0):
    return {"label": label, "composite": composite, "votes": {}, "detail": {}, "fetched_at": "x"}


def _dummy_df():
    idx = pd.date_range("2026-01-01", periods=30, freq="D")
    return pd.DataFrame({"High": 11.0, "Low": 9.5, "Close": 10.0}, index=idx)


def _fake_compute_signals_factory(status, reason="OK"):
    def _fake(pos, df, today_bar=None, exit_params=None, planned_stop=None):
        assert exit_params is None  # macro_monitor must never override ExitParams
        return {
            "ticker": pos.ticker, "entry_date": pos.entry_date.isoformat(),
            "entry_price": pos.entry_price, "shares": pos.shares,
            "last_close": 10.5, "last_low": 10.0, "pnl_%": 5.0, "pnl_$": 50.0,
            "stop_price": pos.stop_price, "status": status, "reason": reason,
        }
    return _fake


def _seed_position(tmp_path, ticker="AAA", entry_price=10.0, shares=100.0, stop_price=9.0):
    macro_path = tmp_path / "macro.db"
    init_db(macro_path)
    set_cash(1_000.0)
    insert_position(ticker, "2026-08-01", entry_price, shares, pattern="vcp", stop_price=stop_price)
    return macro_path


def _run(monkeypatch, macro_path, *, mode, market_open, regime_label, status, reason="OK", regime_composite=0):
    monkeypatch.setattr(macro_monitor, "MACRO_DB_PATH", macro_path)
    monkeypatch.setattr(macro_monitor, "is_market_open", lambda: market_open)
    monkeypatch.setattr(macro_monitor, "get_macro_regime", lambda: _regime(regime_label, regime_composite))
    monkeypatch.setattr(macro_monitor, "load_or_fetch_data", lambda ticker, start: _dummy_df())
    monkeypatch.setattr(macro_monitor, "fetch_intraday_snapshot", lambda t: None)
    monkeypatch.setattr(macro_monitor, "compute_signals", _fake_compute_signals_factory(status, reason))
    monkeypatch.setattr(sys, "argv", ["macro_monitor.py", "--mode", mode])
    macro_monitor.main()


# ── _apply_regime_flip (pure, isolated) ────────────────────────────────────

def test_apply_regime_flip_forces_hold_rows_to_sell_when_risk_off():
    rows = [{POSITION_COL_STATUS: "HOLD", POSITION_COL_REASON: "OK", "ticker": "AAA"}]
    macro_monitor._apply_regime_flip(rows, _regime("risk_off", composite=-2))
    assert rows[0][POSITION_COL_STATUS] == "SELL"
    assert "macro_regime_flip" in rows[0][POSITION_COL_REASON]


def test_apply_regime_flip_is_noop_when_not_risk_off():
    for label in ("risk_on", "neutral"):
        rows = [{POSITION_COL_STATUS: "HOLD", POSITION_COL_REASON: "OK", "ticker": "AAA"}]
        macro_monitor._apply_regime_flip(rows, _regime(label))
        assert rows[0][POSITION_COL_STATUS] == "HOLD"
        assert rows[0][POSITION_COL_REASON] == "OK"


def test_apply_regime_flip_leaves_existing_sell_reason_untouched():
    rows = [{POSITION_COL_STATUS: "SELL", POSITION_COL_REASON: "STOP_HIT(...)", "ticker": "AAA"}]
    macro_monitor._apply_regime_flip(rows, _regime("risk_off", composite=-1))
    assert rows[0][POSITION_COL_REASON] == "STOP_HIT(...)"  # not overwritten


def test_apply_regime_flip_leaves_no_data_rows_untouched():
    rows = [{POSITION_COL_STATUS: "NO_DATA", POSITION_COL_REASON: "Insufficient bars (3)", "ticker": "AAA"}]
    macro_monitor._apply_regime_flip(rows, _regime("risk_off", composite=-1))
    assert rows[0][POSITION_COL_STATUS] == "NO_DATA"


# ── main() integration ──────────────────────────────────────────────────────

def test_normal_exit_path_reuses_core_compute_signals_reason_unchanged(tmp_path, monkeypatch):
    macro_path = _seed_position(tmp_path)
    _run(monkeypatch, macro_path, mode="pre-close", market_open=True,
         regime_label="risk_on", status="HOLD")

    positions = get_open_positions_df()
    assert list(positions["ticker"]) == ["AAA"]  # not sold — regime is risk_on, status stayed HOLD


def test_regime_flip_force_liquidates_all_hold_positions(tmp_path, monkeypatch):
    macro_path = _seed_position(tmp_path)
    _run(monkeypatch, macro_path, mode="pre-close", market_open=True,
         regime_label="risk_off", regime_composite=-2, status="HOLD")

    assert get_open_positions_df().empty  # force-liquidated despite compute_signals saying HOLD


def test_regime_flip_does_not_override_no_data_positions(tmp_path, monkeypatch):
    macro_path = _seed_position(tmp_path)
    monkeypatch.setattr(macro_monitor, "MACRO_DB_PATH", macro_path)
    monkeypatch.setattr(macro_monitor, "is_market_open", lambda: True)
    monkeypatch.setattr(macro_monitor, "get_macro_regime", lambda: _regime("risk_off", composite=-1))
    # Empty df -> the per-position loop flags NO_DATA *before* compute_signals
    # is ever called, matching the real "insufficient bars" path.
    monkeypatch.setattr(macro_monitor, "load_or_fetch_data", lambda ticker, start: pd.DataFrame())
    monkeypatch.setattr(macro_monitor, "fetch_intraday_snapshot", lambda t: None)
    monkeypatch.setattr(sys, "argv", ["macro_monitor.py", "--mode", "pre-close"])

    macro_monitor.main()

    # NO_DATA rows are never in sell_rows (no last_close) -- position untouched.
    assert list(get_open_positions_df()["ticker"]) == ["AAA"]


def test_regime_flip_execution_suppressed_when_market_closed(tmp_path, monkeypatch):
    macro_path = _seed_position(tmp_path)
    _run(monkeypatch, macro_path, mode="pre-close", market_open=False,
         regime_label="risk_off", regime_composite=-2, status="HOLD")

    # Status override still happens in-memory (visible in the report/log),
    # but execute_sells is gated on market_open -- no DB write.
    assert list(get_open_positions_df()["ticker"]) == ["AAA"]


def test_regime_flip_execution_suppressed_in_post_close_mode(tmp_path, monkeypatch):
    macro_path = _seed_position(tmp_path)
    _run(monkeypatch, macro_path, mode="post-close", market_open=True,
         regime_label="risk_off", regime_composite=-2, status="HOLD")

    assert list(get_open_positions_df()["ticker"]) == ["AAA"]


def test_main_calls_init_db_before_any_other_db_call(tmp_path, monkeypatch):
    macro_path = _seed_position(tmp_path)
    order = []
    real_init_db = macro_monitor.init_db
    real_get_cash = macro_monitor.get_cash

    def spy_init_db(path):
        order.append("init_db")
        return real_init_db(path)

    def spy_get_cash():
        order.append("get_cash")
        return real_get_cash()

    monkeypatch.setattr(macro_monitor, "init_db", spy_init_db)
    monkeypatch.setattr(macro_monitor, "get_cash", spy_get_cash)
    monkeypatch.setattr(macro_monitor, "MACRO_DB_PATH", macro_path)
    monkeypatch.setattr(macro_monitor, "is_market_open", lambda: True)
    monkeypatch.setattr(macro_monitor, "get_macro_regime", lambda: _regime("risk_on"))
    monkeypatch.setattr(macro_monitor, "load_or_fetch_data", lambda ticker, start: _dummy_df())
    monkeypatch.setattr(macro_monitor, "fetch_intraday_snapshot", lambda t: None)
    monkeypatch.setattr(macro_monitor, "compute_signals", _fake_compute_signals_factory("HOLD"))
    monkeypatch.setattr(sys, "argv", ["macro_monitor.py", "--mode", "pre-close"])

    macro_monitor.main()

    assert order[0] == "init_db"
    assert "get_cash" in order
