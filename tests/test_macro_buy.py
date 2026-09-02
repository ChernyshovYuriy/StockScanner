"""Offline tests for macro_buy.py (no network — fetch_latest_price and
get_macro_regime are monkeypatched; send_transaction_email is monkeypatched
to a no-op so a real .env's Gmail credentials, if configured, are never
touched by the test suite)."""

from __future__ import annotations

from datetime import timedelta

import duckdb
import pytest

import db as db_module
import macro_buy
from db import get_open_positions_df, init_db, save_intents, set_cash
from time_utils import market_today


@pytest.fixture(autouse=True)
def _no_email(monkeypatch):
    monkeypatch.setattr(macro_buy, "send_transaction_email", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _market_open(monkeypatch):
    monkeypatch.setattr(macro_buy, "is_market_open", lambda: True)


@pytest.fixture(autouse=True)
def _reset_db_path(tmp_path):
    yield
    db_module.DB_PATH = tmp_path / "reset.db"


def _regime(label="risk_on"):
    return {"label": label, "composite": 2 if label == "risk_on" else (-2 if label == "risk_off" else 0),
            "votes": {}, "detail": {}, "fetched_at": "2026-09-02T00:00:00"}


def _seed(tmp_path, core_rows, macro_cash=10_000.0):
    """Seed a core trading.db with `core_rows` intents, then switch db.py's
    global DB_PATH to a fresh macro.db with `macro_cash` seeded — mirrors
    the exact sequencing macro_buy.py's __main__ block performs (read core,
    then own writes against MACRO_DB_PATH)."""
    core_path = tmp_path / "trading.db"
    init_db(core_path)
    save_intents(core_rows)

    macro_path = tmp_path / "macro.db"
    init_db(macro_path)
    set_cash(macro_cash)

    return core_path, macro_path


def _intent_row(ticker, rr, signal_date, entry=10.0, stop=9.0, target=12.0):
    return {
        "ticker": ticker, "signal_date": signal_date, "alert_state": "CONFIRMED",
        "priority": 1, "pattern": "vcp", "entry_price_planned": entry,
        "stop_price": stop, "target_price": target, "rr": rr,
    }


def _today_iso():
    return market_today().date().isoformat()


def test_buy_ranks_candidates_by_rr_descending(tmp_path, monkeypatch):
    rows = [
        _intent_row("AAA", rr=1.0, signal_date=_today_iso()),
        _intent_row("BBB", rr=3.0, signal_date=_today_iso()),
    ]
    core_path, macro_path = _seed(tmp_path, rows)
    monkeypatch.setattr(macro_buy, "CORE_DB_PATH", core_path)
    monkeypatch.setattr(macro_buy, "MACRO_MAX_POSITIONS", 1)
    monkeypatch.setattr(macro_buy, "get_macro_regime", lambda: _regime("risk_on"))
    monkeypatch.setattr(macro_buy, "fetch_latest_price", lambda t: 10.0)

    macro_buy.run_macro_buy(dry_run=False)

    positions = get_open_positions_df()
    assert list(positions["ticker"]) == ["BBB"]  # higher rr wins the single slot


@pytest.mark.parametrize("label", ["neutral", "risk_off"])
def test_buy_skips_all_when_regime_not_risk_on(tmp_path, monkeypatch, label):
    rows = [_intent_row("AAA", rr=3.0, signal_date=_today_iso())]
    core_path, macro_path = _seed(tmp_path, rows)
    monkeypatch.setattr(macro_buy, "CORE_DB_PATH", core_path)
    monkeypatch.setattr(macro_buy, "get_macro_regime", lambda: _regime(label))
    monkeypatch.setattr(macro_buy, "fetch_latest_price", lambda t: 10.0)

    macro_buy.run_macro_buy(dry_run=False)

    assert get_open_positions_df().empty


def test_buy_buys_up_to_max_positions_only(tmp_path, monkeypatch):
    rows = [
        _intent_row("AAA", rr=3.0, signal_date=_today_iso()),
        _intent_row("BBB", rr=2.0, signal_date=_today_iso()),
        _intent_row("CCC", rr=1.0, signal_date=_today_iso()),
    ]
    core_path, macro_path = _seed(tmp_path, rows)
    monkeypatch.setattr(macro_buy, "CORE_DB_PATH", core_path)
    monkeypatch.setattr(macro_buy, "MACRO_MAX_POSITIONS", 2)
    monkeypatch.setattr(macro_buy, "get_macro_regime", lambda: _regime("risk_on"))
    monkeypatch.setattr(macro_buy, "fetch_latest_price", lambda t: 10.0)

    macro_buy.run_macro_buy(dry_run=False)

    tickers = set(get_open_positions_df()["ticker"])
    assert tickers == {"AAA", "BBB"}


def test_buy_sizing_uses_risk_and_cap_minimum(tmp_path, monkeypatch):
    # entry=10, stop=9 -> per_share_risk=1. cash=10_000, MAX_POSITIONS=1 ->
    # max_position_value=10_000, so shares_by_cap = 10_000/10 = 1000.
    # RISK_PER_TRADE_PCT=1.0 -> dollar_risk=100, shares_by_risk = 100/1 = 100.
    # min(100, 1000) = 100 -- the risk cap binds.
    rows = [_intent_row("AAA", rr=2.0, signal_date=_today_iso(), entry=10.0, stop=9.0)]
    core_path, macro_path = _seed(tmp_path, rows, macro_cash=10_000.0)
    monkeypatch.setattr(macro_buy, "CORE_DB_PATH", core_path)
    monkeypatch.setattr(macro_buy, "MACRO_MAX_POSITIONS", 1)
    monkeypatch.setattr(macro_buy, "MACRO_RISK_PER_TRADE_PCT", 1.0)
    monkeypatch.setattr(macro_buy, "get_macro_regime", lambda: _regime("risk_on"))
    monkeypatch.setattr(macro_buy, "fetch_latest_price", lambda t: 10.0)

    macro_buy.run_macro_buy(dry_run=False)

    positions = get_open_positions_df()
    assert len(positions) == 1
    assert positions.iloc[0]["shares"] == 100.0


def test_buy_never_mutates_core_intents_intent_status(tmp_path, monkeypatch):
    rows = [_intent_row("AAA", rr=3.0, signal_date=_today_iso())]
    core_path, macro_path = _seed(tmp_path, rows)
    monkeypatch.setattr(macro_buy, "CORE_DB_PATH", core_path)
    monkeypatch.setattr(macro_buy, "get_macro_regime", lambda: _regime("risk_on"))
    monkeypatch.setattr(macro_buy, "fetch_latest_price", lambda t: 10.0)

    macro_buy.run_macro_buy(dry_run=False)

    # Confirm this sleeve actually bought (otherwise the invariant is vacuous)
    assert not get_open_positions_df().empty

    conn = duckdb.connect(str(core_path), read_only=True)
    try:
        status = conn.execute("SELECT intent_status FROM intents WHERE ticker = 'AAA'").fetchone()[0]
    finally:
        conn.close()
    assert status == "PENDING"


def test_buy_skips_stale_intents(tmp_path, monkeypatch):
    stale_date = (market_today().date() - timedelta(days=30)).isoformat()
    rows = [_intent_row("AAA", rr=3.0, signal_date=stale_date)]
    core_path, macro_path = _seed(tmp_path, rows)
    monkeypatch.setattr(macro_buy, "CORE_DB_PATH", core_path)
    monkeypatch.setattr(macro_buy, "get_macro_regime", lambda: _regime("risk_on"))
    monkeypatch.setattr(macro_buy, "fetch_latest_price", lambda t: 10.0)

    macro_buy.run_macro_buy(dry_run=False)

    assert get_open_positions_df().empty


def test_buy_dry_run_writes_nothing(tmp_path, monkeypatch):
    rows = [_intent_row("AAA", rr=3.0, signal_date=_today_iso())]
    core_path, macro_path = _seed(tmp_path, rows)
    monkeypatch.setattr(macro_buy, "CORE_DB_PATH", core_path)
    monkeypatch.setattr(macro_buy, "get_macro_regime", lambda: _regime("risk_on"))
    monkeypatch.setattr(macro_buy, "fetch_latest_price", lambda t: 10.0)

    macro_buy.run_macro_buy(dry_run=True)

    assert get_open_positions_df().empty


def test_read_core_intents_never_calls_db_init(tmp_path, monkeypatch):
    rows = [_intent_row("AAA", rr=3.0, signal_date=_today_iso())]
    core_path = tmp_path / "trading.db"
    init_db(core_path)
    save_intents(rows)

    calls = []
    monkeypatch.setattr(macro_buy, "init_db", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(macro_buy, "CORE_DB_PATH", core_path)

    df = macro_buy._read_core_intents(market_today().date() - timedelta(days=1))

    assert calls == []
    assert not df.empty
