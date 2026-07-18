"""
tests/test_integration.py
=========================
Integration tests verifying that the three scheduled services read from
and write to the database correctly.

Each test gets an isolated temporary database via the `db` fixture.
Network calls (yfinance price fetches) are mocked so tests run offline
and deterministically.

Coverage:
  virtual_buy.run_virtual_buy   — happy path, dry_run, no funds, already
                                   owned, portfolio full, gap-up filter
  position_monitor              — parse_positions_from_db, execute_virtual_sells
                                   happy path and dry_run
  auto_pipeline                 — load_signal_db / save_signal_db round-trip
  cross-service                 — pipeline writes intents → virtual_buy reads
                                   and executes them
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

import db as db_module
from auto_pipeline import load_signal_db, save_signal_db
from db import (
    get_all_trades,
    get_cash,
    get_open_positions,
    get_open_positions_df,
    get_transactions,
    init_db,
    insert_position,
    load_pending_intents,
    save_intents,
    set_cash,
)
from position_monitor import Position, execute_virtual_sells, parse_positions_from_db
from schema_keys import (
    POSITION_COL_ENTRY_DATE,
    POSITION_COL_ENTRY_PRICE,
    POSITION_COL_LAST_CLOSE,
    POSITION_COL_PNL_DOLLARS,
    POSITION_COL_PNL_PCT,
    POSITION_COL_REASON,
    POSITION_COL_SHARES,
    SIGNAL_COL_TICKER,
)
from virtual_buy import run_virtual_buy


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def db(tmp_path):
    path = tmp_path / "trading.db"
    init_db(path)
    yield path
    db_module.DB_PATH = tmp_path / "reset.db"


@pytest.fixture(autouse=True)
def no_emails(monkeypatch):
    """Prevent any test from sending real emails via Gmail."""
    monkeypatch.setattr("virtual_buy.send_transaction_email", lambda **_: None)
    monkeypatch.setattr("position_monitor.send_transaction_email", lambda **_: None)


@pytest.fixture(autouse=True)
def market_open():
    """Pin the clock to a known TSX session so run_virtual_buy's market-hours
    guard is satisfied — 2026-05-14 is a Thursday, 11:00 ET, not a holiday."""
    from time_utils import set_backtest_clock, TSX_TZ
    from datetime import datetime
    set_backtest_clock(datetime(2026, 5, 14, 11, 0, tzinfo=TSX_TZ))
    yield
    set_backtest_clock(None)


def _intent(**overrides) -> dict:
    base = {
        "ticker": "RY.TO",
        "signal_date": "2026-05-14",
        "alert_state": "CONFIRMED",
        "priority": 1,
        "pattern": "VCP",
        "entry_price_planned": 42.50,
        "stop_price": 40.00,
        "target_price": 47.50,
        "rr": 2.0,
    }
    base.update(overrides)
    return base


def _sell_row(ticker="RY.TO", entry_price=42.50, shares=100.0,
              sell_price=45.00, pnl_dollars=250.0, pnl_pct=5.88,
              reason="GIVEBACK") -> dict:
    """Minimal sell-row dict matching what compute_signals() returns."""
    return {
        SIGNAL_COL_TICKER: ticker,
        POSITION_COL_ENTRY_DATE: "2026-05-01",
        POSITION_COL_ENTRY_PRICE: entry_price,
        POSITION_COL_SHARES: shares,
        POSITION_COL_LAST_CLOSE: sell_price,
        POSITION_COL_PNL_DOLLARS: pnl_dollars,
        POSITION_COL_PNL_PCT: pnl_pct,
        POSITION_COL_REASON: reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# virtual_buy — happy path
# ─────────────────────────────────────────────────────────────────────────────

def test_virtual_buy_inserts_position_and_deducts_cash():
    set_cash(10_000.0)
    save_intents([_intent(ticker="RY.TO", entry_price_planned=42.50, stop_price=40.00)])

    with patch("virtual_buy.fetch_latest_price", return_value=42.50):
        run_virtual_buy(top_n=None, dry_run=False)

    positions = get_open_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "RY.TO"
    # risk_based sizing: dollar_risk=100, per_share_risk=2.50 → 40 shares by risk,
    # capped to int(1250/42.50)=29 by max_position_value → cost = 29 * 42.50 = 1232.50
    assert get_cash() == pytest.approx(10_000.0 - 1232.50)


def test_virtual_buy_marks_intent_executed():
    set_cash(10_000.0)
    save_intents([_intent()])

    with patch("virtual_buy.fetch_latest_price", return_value=42.50):
        run_virtual_buy(top_n=None, dry_run=False)

    assert load_pending_intents().empty


def test_virtual_buy_records_buy_transaction():
    set_cash(10_000.0)
    save_intents([_intent(ticker="RY.TO", pattern="VCP")])

    with patch("virtual_buy.fetch_latest_price", return_value=42.50):
        run_virtual_buy(top_n=None, dry_run=False)

    tx = get_transactions()
    assert len(tx) == 1
    assert tx.iloc[0]["side"] == "BUY"
    assert tx.iloc[0]["ticker"] == "RY.TO"
    assert tx.iloc[0]["reason"] == "VCP"


def test_virtual_buy_respects_top_n():
    set_cash(10_000.0)
    save_intents([
        _intent(ticker="RY.TO",  priority=1),
        _intent(ticker="TD.TO",  priority=2),
        _intent(ticker="ENB.TO", priority=3),
    ])

    with patch("virtual_buy.fetch_latest_price", return_value=42.50):
        run_virtual_buy(top_n=1, dry_run=False)

    positions = get_open_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "RY.TO"


def test_virtual_buy_skips_already_owned_ticker():
    set_cash(10_000.0)
    insert_position("RY.TO", "2026-05-01", 42.50, 100)
    save_intents([_intent(ticker="RY.TO")])

    with patch("virtual_buy.fetch_latest_price", return_value=42.50):
        run_virtual_buy(top_n=None, dry_run=False)

    # Position count unchanged (still just the original one)
    assert len(get_open_positions()) == 1
    # Cash unchanged
    assert get_cash() == 10_000.0


# ─────────────────────────────────────────────────────────────────────────────
# virtual_buy — early exits
# ─────────────────────────────────────────────────────────────────────────────

def test_virtual_buy_does_nothing_when_no_intents():
    set_cash(10_000.0)

    run_virtual_buy(top_n=None, dry_run=False)

    assert get_open_positions() == []
    assert get_cash() == 10_000.0


def test_virtual_buy_does_nothing_when_cash_is_zero():
    set_cash(0.0)
    save_intents([_intent()])

    with patch("virtual_buy.fetch_latest_price", return_value=42.50):
        run_virtual_buy(top_n=None, dry_run=False)

    assert get_open_positions() == []


def test_virtual_buy_does_nothing_when_portfolio_full():
    from config import MAX_POSITIONS
    set_cash(100_000.0)
    for i in range(MAX_POSITIONS):
        insert_position(f"T{i}.TO", "2026-05-01", 10.0, 100)
    save_intents([_intent(ticker="NEW.TO")])

    with patch("virtual_buy.fetch_latest_price", return_value=42.50):
        run_virtual_buy(top_n=None, dry_run=False)

    assert len(get_open_positions()) == MAX_POSITIONS
    assert get_cash() == 100_000.0


# ─────────────────────────────────────────────────────────────────────────────
# virtual_buy — dry_run
# ─────────────────────────────────────────────────────────────────────────────

def test_virtual_buy_dry_run_writes_nothing():
    set_cash(10_000.0)
    save_intents([_intent()])

    with patch("virtual_buy.fetch_latest_price", return_value=42.50):
        run_virtual_buy(top_n=None, dry_run=True)

    assert get_open_positions() == []
    assert get_cash() == 10_000.0
    assert get_transactions().empty        # no BUY recorded
    assert not load_pending_intents().empty  # intent still pending


# ─────────────────────────────────────────────────────────────────────────────
# virtual_buy — gap-up filter
# ─────────────────────────────────────────────────────────────────────────────

def test_virtual_buy_skips_intent_on_gap_up():
    gap_pct = 2.0
    set_cash(10_000.0)
    planned = 42.50
    gapped_price = planned * (1 + gap_pct / 100 + 0.01)  # just over the limit
    save_intents([_intent(entry_price_planned=planned)])

    with patch("virtual_buy.fetch_latest_price", return_value=gapped_price), \
         patch("virtual_buy.GAP_FILTER_PCT", gap_pct):
        run_virtual_buy(top_n=None, dry_run=False)

    assert get_open_positions() == []
    assert get_cash() == 10_000.0


def test_virtual_buy_executes_when_price_within_gap_filter():
    gap_pct = 2.0
    set_cash(10_000.0)
    planned = 42.50
    within_price = planned * (1 + gap_pct / 100 - 0.005)  # just under the limit
    save_intents([_intent(entry_price_planned=planned, stop_price=40.00)])

    with patch("virtual_buy.fetch_latest_price", return_value=within_price), \
         patch("virtual_buy.GAP_FILTER_PCT", gap_pct):
        run_virtual_buy(top_n=None, dry_run=False)

    assert len(get_open_positions()) == 1


def test_virtual_buy_ignores_gap_up_when_filter_disabled():
    set_cash(10_000.0)
    planned = 42.50
    gapped_price = planned * 1.10  # far above any gap limit
    save_intents([_intent(entry_price_planned=planned, stop_price=40.00)])

    with patch("virtual_buy.fetch_latest_price", return_value=gapped_price), \
         patch("virtual_buy.GAP_FILTER_PCT", None):
        run_virtual_buy(top_n=None, dry_run=False)

    assert len(get_open_positions()) == 1


# ─────────────────────────────────────────────────────────────────────────────
# position_monitor — parse_positions_from_db
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_positions_from_db_returns_empty_when_no_positions():
    result = parse_positions_from_db()
    assert result == []


def test_parse_positions_from_db_returns_position_objects():
    insert_position("RY.TO", "2026-05-01", 42.50, 100)
    insert_position("TD.TO", "2026-05-03", 85.00, 50)

    positions = parse_positions_from_db()
    assert len(positions) == 2
    assert all(isinstance(p, Position) for p in positions)

    tickers = {p.ticker for p in positions}
    assert tickers == {"RY.TO", "TD.TO"}


def test_parse_positions_from_db_correct_field_types():
    insert_position("RY.TO", "2026-05-01", 42.50, 100)
    pos = parse_positions_from_db()[0]

    assert isinstance(pos.entry_date, date)
    assert isinstance(pos.entry_price, float)
    assert isinstance(pos.shares, float)
    assert pos.entry_price == 42.50
    assert pos.shares == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# position_monitor — execute_virtual_sells
# ─────────────────────────────────────────────────────────────────────────────

def test_execute_virtual_sells_removes_position_from_db():
    set_cash(0.0)
    insert_position("RY.TO", "2026-05-01", 42.50, 100)

    execute_virtual_sells([_sell_row("RY.TO", shares=100.0, sell_price=45.00)])

    assert get_open_positions() == []


def test_execute_virtual_sells_adds_proceeds_to_cash():
    set_cash(1_000.0)
    insert_position("RY.TO", "2026-05-01", 42.50, 100)

    execute_virtual_sells([_sell_row("RY.TO", shares=100.0, sell_price=45.00)])

    assert get_cash() == pytest.approx(1_000.0 + 45.00 * 100, rel=1e-4)


def test_execute_virtual_sells_inserts_trade_record():
    set_cash(0.0)
    insert_position("RY.TO", "2026-05-01", 42.50, 100)

    execute_virtual_sells([_sell_row("RY.TO", pnl_dollars=250.0, reason="GIVEBACK")])

    trades = get_all_trades()
    assert len(trades) == 1
    assert trades.iloc[0]["ticker"] == "RY.TO"
    assert trades.iloc[0]["pnl_dollars"] == 250.0
    assert trades.iloc[0]["reason"] == "GIVEBACK"


def test_execute_virtual_sells_records_sell_transaction():
    set_cash(0.0)
    insert_position("RY.TO", "2026-05-01", 42.50, 100)

    execute_virtual_sells([_sell_row("RY.TO", sell_price=45.00)])

    sell_txs = get_transactions()[get_transactions()["side"] == "SELL"]
    assert len(sell_txs) == 1
    assert sell_txs.iloc[0]["ticker"] == "RY.TO"
    assert sell_txs.iloc[0]["price"] == 45.00


def test_execute_virtual_sells_multiple_positions():
    set_cash(0.0)
    insert_position("RY.TO",  "2026-05-01", 42.50, 100)
    insert_position("TD.TO",  "2026-05-01", 85.00, 50)
    insert_position("ENB.TO", "2026-05-01", 50.00, 80)

    execute_virtual_sells([
        _sell_row("RY.TO", shares=100.0, sell_price=45.00),
        _sell_row("TD.TO", shares=50.0,  sell_price=83.00, pnl_dollars=-100.0),
    ])

    remaining = {p["ticker"] for p in get_open_positions()}
    assert remaining == {"ENB.TO"}
    assert len(get_all_trades()) == 2


def test_execute_virtual_sells_dry_run_writes_nothing():
    set_cash(5_000.0)
    insert_position("RY.TO", "2026-05-01", 42.50, 100)

    execute_virtual_sells([_sell_row("RY.TO")], dry_run=True)

    assert len(get_open_positions()) == 1  # position still there
    assert get_cash() == 5_000.0           # cash unchanged
    assert get_all_trades().empty
    sell_txs = get_transactions()[get_transactions()["side"] == "SELL"]
    assert sell_txs.empty                  # no SELL transaction recorded


def test_execute_virtual_sells_empty_list_returns_zeros():
    result = execute_virtual_sells([])
    assert result["funds_before"] == 0.0
    assert result["funds_after"] == 0.0
    assert result["funds_gained"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# auto_pipeline — signal DB round-trip
# ─────────────────────────────────────────────────────────────────────────────

def _make_signals_df(**overrides) -> pd.DataFrame:
    row = {
        "ticker": "TD.TO", "pattern": "VCP", "state": "AT_PIVOT",
        "first_seen": "2026-05-01", "last_seen": "2026-05-14",
        "days_in_state": 3, "consecutive_screener_days": 5, "screener_days": 5,
        "entry": 85.0, "stop": 81.0, "target_2r": 93.0, "target_3r": 97.0,
        "risk_pct": 4.7, "pivot_price": 84.5, "detail": "tight range",
        "alert_sent": False,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_save_and_load_signal_db_round_trip():
    df = _make_signals_df(ticker="TD.TO", state="AT_PIVOT")
    save_signal_db(df)

    loaded = load_signal_db()
    assert len(loaded) == 1
    assert loaded.iloc[0]["ticker"] == "TD.TO"
    assert loaded.iloc[0]["state"] == "AT_PIVOT"


def test_load_signal_db_returns_empty_dataframe_when_no_signals():
    df = load_signal_db()
    assert df.empty


def test_save_signal_db_replaces_previous():
    save_signal_db(_make_signals_df(ticker="TD.TO"))
    save_signal_db(_make_signals_df(ticker="RY.TO"))

    loaded = load_signal_db()
    assert len(loaded) == 1
    assert loaded.iloc[0]["ticker"] == "RY.TO"


def test_load_signal_db_parses_dates_as_date_objects():
    save_signal_db(_make_signals_df(first_seen="2026-05-01", last_seen="2026-05-14"))

    loaded = load_signal_db()
    assert isinstance(loaded.iloc[0]["first_seen"], date)
    assert isinstance(loaded.iloc[0]["last_seen"], date)


# ─────────────────────────────────────────────────────────────────────────────
# cross-service: pipeline intents → virtual_buy execution
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_intents_consumed_by_virtual_buy():
    """
    Simulate the end-of-day flow:
      auto_pipeline saves confirmed intents → virtual_buy reads and executes them.
    """
    set_cash(10_000.0)

    # auto_pipeline writes confirmed intents to DB
    save_intents([
        _intent(ticker="RY.TO",  priority=1, entry_price_planned=42.50, stop_price=40.00),
        _intent(ticker="TD.TO",  priority=2, entry_price_planned=85.00, stop_price=81.00),
    ])

    # virtual_buy runs next morning and fills them
    prices = {"RY.TO": 42.50, "TD.TO": 85.00}
    with patch("virtual_buy.fetch_latest_price", side_effect=lambda t: prices[t]):
        run_virtual_buy(top_n=None, dry_run=False)

    positions = get_open_positions()
    tickers = {p["ticker"] for p in positions}
    assert tickers == {"RY.TO", "TD.TO"}
    assert load_pending_intents().empty
    assert get_cash() < 10_000.0
    assert len(get_transactions()) == 2


def test_pipeline_intents_then_sell_full_lifecycle():
    """
    Full lifecycle: buy intent → position → sell → trade record + cash restored.
    """
    set_cash(10_000.0)
    save_intents([_intent(ticker="RY.TO", entry_price_planned=42.50, stop_price=40.00)])

    with patch("virtual_buy.fetch_latest_price", return_value=42.50):
        run_virtual_buy(top_n=None, dry_run=False)

    cash_after_buy = get_cash()
    assert cash_after_buy < 10_000.0

    # Read what was actually bought so the sell row is consistent.
    pos = get_open_positions()[0]
    actual_shares = pos["shares"]
    actual_entry  = pos["entry_price"]
    sell_price    = 45.00
    proceeds      = round(actual_shares * sell_price, 2)
    pnl_dollars   = round((sell_price - actual_entry) * actual_shares, 2)
    pnl_pct       = round((sell_price / actual_entry - 1) * 100, 2)

    execute_virtual_sells([_sell_row("RY.TO",
                                     shares=actual_shares,
                                     sell_price=sell_price,
                                     pnl_dollars=pnl_dollars,
                                     pnl_pct=pnl_pct)])

    assert get_open_positions() == []
    assert get_cash() == pytest.approx(cash_after_buy + proceeds)
    assert len(get_all_trades()) == 1

    tx = get_transactions()
    assert list(tx["side"]) == ["BUY", "SELL"]  # ledger has both sides
