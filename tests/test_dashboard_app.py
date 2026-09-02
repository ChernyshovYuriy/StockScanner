"""
tests/test_dashboard_app.py
============================
Route-level tests for the Flask web dashboard. build_live_positions() and
sell_position() are mocked here (they're covered by their own unit tests in
test_dashboard_positions.py / test_manual_sell.py) — these tests only verify
routing, rendering, and HTTP status mapping.
"""

from __future__ import annotations

import pytest

import db as db_module
from db import get_all_trades, get_transactions, init_db, insert_position, set_cash
from position_monitor import execute_virtual_sells


@pytest.fixture(autouse=True)
def db(tmp_path):
    path = tmp_path / "trading.db"
    init_db(path)
    yield path
    db_module.DB_PATH = tmp_path / "reset.db"


@pytest.fixture(autouse=True)
def no_emails(monkeypatch):
    monkeypatch.setattr("position_monitor.send_transaction_email", lambda **_: None)


@pytest.fixture
def client():
    import dashboard_app
    return dashboard_app.create_app().test_client()


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}


def test_monitor_page_renders_when_no_positions(client):
    set_cash(1000.0)
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"No open positions" in resp.data


def test_monitor_page_renders_open_position(client, monkeypatch):
    set_cash(1000.0)
    row = {
        "ticker": "RY.TO", "entry_date": "2026-05-01", "entry_price": 42.5,
        "shares": 100.0, "last_close": 45.0, "pnl_%": 5.88, "pnl_$": 250.0,
        "stop_price": 40.0, "R_mult": 1.0, "tdays": 5, "status": "HOLD", "reason": "OK",
    }
    monkeypatch.setattr("dashboard_app.build_live_positions", lambda: [row])

    resp = client.get("/")

    assert resp.status_code == 200
    assert b"RY.TO" in resp.data
    assert b"Sell" in resp.data


def test_monitor_page_shows_initial_capital_and_total_equity(client, monkeypatch):
    # 10,000 starting capital, spend 4,250 on a position that's now worth 4,500.
    set_cash(10_000.0)
    insert_position("RY.TO", "2026-05-01", 42.50, 100, cash_delta=-4_250.0)
    row = {
        "ticker": "RY.TO", "entry_date": "2026-05-01", "entry_price": 42.5,
        "shares": 100.0, "last_close": 45.0, "pnl_%": 5.88, "pnl_$": 250.0,
        "stop_price": 40.0, "R_mult": 1.0, "tdays": 5, "status": "HOLD", "reason": "OK",
    }
    monkeypatch.setattr("dashboard_app.build_live_positions", lambda: [row])

    resp = client.get("/")
    html = resp.data.decode()

    assert resp.status_code == 200
    assert "$10000.00" in html.replace(",", "")  # initial capital recovered from the ledger
    assert "$5750.00" in html.replace(",", "")   # cash after the buy
    assert "$4500.00" in html.replace(",", "")   # positions value (100 * 45.00)
    assert "$10250.00" in html.replace(",", "")  # total equity (5750 cash + 4500 value)
    assert "+250.00" in html                     # total return since inception (+2.50%)


def test_history_page_renders_trades_and_transactions(client):
    set_cash(0.0)
    insert_position("RY.TO", "2026-05-01", 42.50, 100)
    execute_virtual_sells([{
        "ticker": "RY.TO", "entry_date": "2026-05-01", "entry_price": 42.50,
        "shares": 100.0, "last_close": 45.00, "pnl_$": 250.0, "pnl_%": 5.88,
        "reason": "MANUAL_SELL",
    }])

    resp = client.get("/history")

    assert resp.status_code == 200
    assert b"RY.TO" in resp.data
    assert b"MANUAL_SELL" in resp.data
    assert len(get_all_trades()) == 1
    assert len(get_transactions()) == 2  # BUY + SELL


def test_sell_endpoint_happy_path(client, monkeypatch):
    monkeypatch.setattr(
        "dashboard_app.sell_position",
        lambda ticker, price=None: {"ok": True, "ticker": ticker, "price": 45.0, "source": "5m-intraday",
                                     "pnl_dollars": 250.0, "pnl_pct": 5.88, "funds_state": {}},
    )

    resp = client.post("/api/positions/RY.TO/sell")

    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True


@pytest.mark.parametrize("error,expected_status", [
    ("locked", 409),
    ("no_position", 404),
    ("no_price", 503),
])
def test_sell_endpoint_maps_errors_to_status_codes(client, monkeypatch, error, expected_status):
    monkeypatch.setattr(
        "dashboard_app.sell_position",
        lambda ticker, price=None: {"ok": False, "ticker": ticker, "error": error, "message": "nope"},
    )

    resp = client.post("/api/positions/RY.TO/sell")

    assert resp.status_code == expected_status
    assert resp.get_json()["ok"] is False


def test_sell_endpoint_forwards_manual_price(client, monkeypatch):
    received = {}

    def _fake_sell(ticker, price=None):
        received["ticker"] = ticker
        received["price"] = price
        return {"ok": True, "ticker": ticker, "price": price, "source": "manual",
                "pnl_dollars": -50.0, "pnl_pct": -6.7, "funds_state": {}}

    monkeypatch.setattr("dashboard_app.sell_position", _fake_sell)

    resp = client.post("/api/positions/BLN.TO/sell", json={"price": 8.50})

    assert resp.status_code == 200
    assert received == {"ticker": "BLN.TO", "price": 8.50}
    assert resp.get_json()["source"] == "manual"


@pytest.mark.parametrize("bad_price", ["not-a-number", -5, 0])
def test_sell_endpoint_rejects_invalid_manual_price(client, monkeypatch, bad_price):
    monkeypatch.setattr(
        "dashboard_app.sell_position",
        lambda ticker, price=None: pytest.fail("sell_position should not be called with a bad price"),
    )

    resp = client.post("/api/positions/BLN.TO/sell", json={"price": bad_price})

    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False


def test_demand_page_renders_when_no_signals(client, monkeypatch):
    monkeypatch.setattr("dashboard_app.build_demand_signals_by_ticker", lambda: {})
    resp = client.get("/demand")
    assert resp.status_code == 200
    assert b"No signals stored yet" in resp.data


def test_demand_page_renders_signals_grouped_by_ticker(client, monkeypatch):
    rows = {
        "MU": [
            {"source": "edgar_insider", "signal_type": "insider_buy", "direction": "bullish",
             "strength": 0.8, "lag_days": 5, "date": "2026-06-05"},
            {"source": "options_flow", "signal_type": "call_put_skew", "direction": "bearish",
             "strength": 0.3, "lag_days": 0, "date": "2026-06-05"},
        ],
    }
    monkeypatch.setattr("dashboard_app.build_demand_signals_by_ticker", lambda: rows)

    resp = client.get("/demand")
    html = resp.data.decode()

    assert resp.status_code == 200
    assert "MU" in html
    assert "insider_buy" in html
    assert "call_put_skew" in html
    assert "badge-bullish" in html
    assert "badge-bearish" in html
