"""
Idle-cash parking in the live core sleeve (idle_cash.py, config.IDLE_CASH_*):
uninvested cash is held in XIU.TO, sold to fund buys, parked again at the
pre-close run. See reports/idle_cash/FINAL_REPORT.md.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import db as db_module
from db import (get_cash, get_open_positions, get_parked, get_transactions, init_db, insert_position, park_trade,
                save_intents, set_cash)
from idle_cash import funds_estimate, park_excess, parked_shares, raise_cash

PARK = "XIU.TO"


@pytest.fixture(autouse=True)
def db(tmp_path):
    init_db(tmp_path / "trading.db")
    yield
    db_module.DB_PATH = tmp_path / "reset.db"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from time_utils import TSX_TZ, set_backtest_clock
    monkeypatch.setattr("virtual_buy.send_transaction_email", lambda **_: None)
    monkeypatch.setattr("virtual_buy.get_sector", lambda ticker: "Unknown")
    set_backtest_clock(datetime(2026, 5, 14, 11, 0, tzinfo=TSX_TZ))
    yield
    set_backtest_clock(None)


# ── db.park_trade ────────────────────────────────────────────────────────────

def test_park_trade_moves_cash_and_records_the_ledger():
    set_cash(10_000.0)
    park_trade(PARK, "2026-05-14", 40.0, 200)
    park_trade(PARK, "2026-05-15", 41.0, -50)
    assert get_parked() == {PARK: {"shares": 150.0, "last_price": 41.0}}
    assert get_cash() == pytest.approx(10_000.0 - 200 * 40.0 + 50 * 41.0)
    tx = get_transactions()
    assert list(tx.side) == ["BUY", "SELL"] and list(tx.reason) == ["PARK", "UNPARK"]
    # The dashboard's ledger identity still recovers the starting capital.
    buys, sells = tx.loc[tx.side == "BUY", "amount"].sum(), tx.loc[tx.side == "SELL", "amount"].sum()
    assert get_cash() + buys - sells == pytest.approx(10_000.0)
    assert get_open_positions() == []                 # never a monitored position


def test_park_trade_refuses_to_sell_more_than_held():
    set_cash(10_000.0)
    park_trade(PARK, "2026-05-14", 40.0, 10)
    with pytest.raises(ValueError):
        park_trade(PARK, "2026-05-14", 40.0, -11)
    assert parked_shares() == 10.0


# ── idle_cash ────────────────────────────────────────────────────────────────

def test_park_excess_keeps_the_buffer_and_skips_small_amounts():
    set_cash(10_000.0)
    assert park_excess(40.0, "2026-05-14", equity=10_000.0) == int(9_900 / 40.0)
    assert get_cash() == pytest.approx(10_000.0 - 247 * 40.0)
    assert park_excess(40.0, "2026-05-14", equity=10_000.0) == 0      # 120 left: below the 3% threshold


def test_raise_cash_sells_just_enough_plus_the_buffer():
    set_cash(10_000.0)
    park_trade(PARK, "2026-05-14", 40.0, 240)                         # cash 400
    sold = raise_cash(2_000.0, 40.0, "2026-05-15", equity=10_000.0)
    assert sold == 43                                                  # ceil((2,000 - 400 + 100) / 40)
    assert get_cash() == pytest.approx(400.0 + 43 * 40.0)
    assert raise_cash(500.0, 40.0, "2026-05-15", equity=10_000.0) == 0  # cash already covers it


def test_funds_estimate_includes_the_parked_holding():
    set_cash(10_000.0)
    park_trade(PARK, "2026-05-14", 40.0, 200)
    assert funds_estimate() == pytest.approx(10_000.0)


# ── virtual_buy ──────────────────────────────────────────────────────────────

def _intent(**kw) -> dict:
    base = dict(ticker="RY.TO", signal_date="2026-05-14", alert_state="CONFIRMED", priority=1, pattern="VCP",
                entry_price_planned=42.50, stop_price=40.00, target_price=47.50, rr=2.0)
    return {**base, **kw}


def _quotes(prices: dict):
    return patch("virtual_buy.fetch_latest_price", side_effect=lambda t: prices.get(t))


def test_virtual_buy_sizes_from_cash_plus_parked_and_sells_parked_to_fund():
    from virtual_buy import run_virtual_buy
    set_cash(10_000.0)
    park_trade(PARK, "2026-05-13", 40.0, 247)                         # cash 120, parked 9,880
    save_intents([_intent()])
    with _quotes({"RY.TO": 42.50, PARK: 40.0}):
        run_virtual_buy(top_n=None, dry_run=False)
    pos = get_open_positions()
    assert pos[0]["shares"] == 29                                      # same as with 10,000 in cash
    sold = 247 - parked_shares()
    assert sold == 31                                                  # ceil((1,232.50 - 120 + 100) / 40)
    assert get_cash() == pytest.approx(120.0 + 31 * 40.0 - 29 * 42.50)
    assert get_cash() >= 0


def test_virtual_buy_without_a_parked_quote_uses_cash_only():
    from virtual_buy import run_virtual_buy
    set_cash(10_000.0)
    park_trade(PARK, "2026-05-13", 40.0, 247)                         # cash 120
    save_intents([_intent()])
    with _quotes({"RY.TO": 42.50}):
        run_virtual_buy(top_n=None, dry_run=False)
    assert get_open_positions() == []                                  # 120 can't buy a 42.50 share at 1% risk
    assert parked_shares() == 247


def test_virtual_buy_dry_run_sells_nothing():
    from virtual_buy import run_virtual_buy
    set_cash(10_000.0)
    park_trade(PARK, "2026-05-13", 40.0, 247)
    save_intents([_intent()])
    with _quotes({"RY.TO": 42.50, PARK: 40.0}):
        run_virtual_buy(top_n=None, dry_run=True)
    assert parked_shares() == 247 and get_open_positions() == []


# ── position_monitor.park_idle_cash ──────────────────────────────────────────

def test_monitor_parks_cash_above_the_buffer_valuing_positions_live():
    from position_monitor import park_idle_cash
    set_cash(10_000.0)
    insert_position("RY.TO", "2026-05-01", 40.0, 100, cash_delta=-4_000.0)   # cash 6,000
    snap = SimpleNamespace(close=40.0)
    rows = [{"ticker": "RY.TO", "last_close": 50.0}]                   # equity 6,000 + 5,000 = 11,000
    with patch("position_monitor.fetch_intraday_snapshot", return_value=snap):
        n = park_idle_cash(rows, dry_run=False)
    assert n == int((6_000 - 110) / 40.0)
    assert parked_shares() == n


def test_monitor_parks_nothing_without_a_quote_or_on_dry_run():
    from position_monitor import park_idle_cash
    set_cash(10_000.0)
    with patch("position_monitor.fetch_intraday_snapshot", return_value=None):
        assert park_idle_cash([], dry_run=False) == 0
    with patch("position_monitor.fetch_intraday_snapshot", return_value=SimpleNamespace(close=40.0)):
        assert park_idle_cash([], dry_run=True) > 0
    assert parked_shares() == 0 and get_cash() == 10_000.0


# ── dashboard ────────────────────────────────────────────────────────────────

def test_dashboard_equity_and_initial_capital_include_the_parked_holding(monkeypatch):
    import dashboard_app
    set_cash(10_000.0)
    park_trade(PARK, "2026-05-13", 40.0, 200)                         # cash 2,000
    monkeypatch.setattr("dashboard_app.build_live_positions", lambda: [])
    monkeypatch.setattr("dashboard_app.build_parked_holdings", lambda: [
        {"ticker": PARK, "shares": 200.0, "price": 42.0, "value": 8_400.0, "price_source": "5m-intraday"}])
    html = dashboard_app.create_app().test_client().get("/").data.decode()
    assert "Parked (XIU.TO)" in html
    assert "$10000.00" in html                                         # initial capital
    assert "$10400.00" in html                                         # equity: 2,000 cash + 8,400 parked
