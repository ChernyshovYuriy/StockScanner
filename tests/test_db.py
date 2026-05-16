"""
tests/test_db.py
================
Tests for db.py — happy path and failure path.

Each test gets an isolated temporary database via the `db` fixture,
which calls init_db() and sets the module-level DB_PATH so all
db functions use the temp file for the duration of the test.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

import db as db_module
from db import (
    delete_position,
    get_all_trades,
    get_cash,
    get_open_positions,
    get_open_positions_df,
    get_transactions,
    init_db,
    insert_position,
    insert_trade,
    load_pending_intents,
    load_signals,
    mark_intent_executed,
    mark_intent_skipped,
    save_intents,
    save_signals,
    set_cash,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def db(tmp_path):
    """Create a fresh isolated database for every test."""
    path = tmp_path / "trading.db"
    init_db(path)
    yield path
    # Reset to a dummy path so a stale DB_PATH never bleeds into the next test
    db_module.DB_PATH = tmp_path / "reset.db"


# ─────────────────────────────────────────────────────────────────────────────
# init_db
# ─────────────────────────────────────────────────────────────────────────────

def test_init_db_creates_file(db):
    assert db.exists()


def test_init_db_is_idempotent(db):
    """Calling init_db a second time must not raise or corrupt the schema."""
    init_db(db)
    set_cash(42.0)
    init_db(db)
    assert get_cash() == 42.0


# ─────────────────────────────────────────────────────────────────────────────
# account
# ─────────────────────────────────────────────────────────────────────────────

def test_get_cash_before_set_returns_zero():
    assert get_cash() == 0.0


def test_set_and_get_cash():
    set_cash(10_000.0)
    assert get_cash() == 10_000.0


def test_set_cash_upserts():
    set_cash(10_000.0)
    set_cash(9_500.0)
    assert get_cash() == 9_500.0


def test_set_cash_rounds_to_four_decimals():
    set_cash(1234.56789)
    assert get_cash() == 1234.5679


# ─────────────────────────────────────────────────────────────────────────────
# positions
# ─────────────────────────────────────────────────────────────────────────────

def test_insert_and_get_position():
    insert_position("RY.TO", "2026-05-01", 42.50, 100)
    rows = get_open_positions()
    assert len(rows) == 1
    assert rows[0]["ticker"] == "RY.TO"
    assert rows[0]["entry_price"] == 42.50
    assert rows[0]["shares"] == 100.0


def test_insert_position_uppercases_ticker():
    insert_position("ry.to", "2026-05-01", 42.50, 100)
    assert get_open_positions()[0]["ticker"] == "RY.TO"


def test_get_open_positions_df_columns_when_empty():
    df = get_open_positions_df()
    assert df.empty
    assert set(df.columns) == {"ticker", "entry_date", "entry_price", "shares"}


def test_get_open_positions_df_returns_dataframe():
    insert_position("TD.TO", "2026-05-01", 85.0, 50)
    df = get_open_positions_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "TD.TO"


def test_positions_ordered_by_entry_date():
    insert_position("ENB.TO", "2026-05-03", 50.0, 80)
    insert_position("RY.TO",  "2026-05-01", 42.5, 100)
    tickers = [r["ticker"] for r in get_open_positions()]
    assert tickers == ["RY.TO", "ENB.TO"]


def test_delete_position_removes_it():
    insert_position("RY.TO", "2026-05-01", 42.50, 100)
    delete_position("RY.TO")
    assert get_open_positions() == []


def test_delete_position_case_insensitive():
    insert_position("RY.TO", "2026-05-01", 42.50, 100)
    delete_position("ry.to")
    assert get_open_positions() == []


def test_delete_nonexistent_position_does_not_raise():
    delete_position("MISSING.TO")  # must not raise


def test_insert_duplicate_ticker_raises():
    insert_position("RY.TO", "2026-05-01", 42.50, 100)
    with pytest.raises(Exception):
        insert_position("RY.TO", "2026-05-02", 43.00, 50)


# ─────────────────────────────────────────────────────────────────────────────
# trades
# ─────────────────────────────────────────────────────────────────────────────

def _insert_sample_trade(ticker="RY.TO"):
    insert_trade(ticker, "2026-05-01", 42.50, 100, "2026-05-10", 45.00, 4500.0, 250.0, 5.88, "GIVEBACK")


def test_insert_and_get_trade():
    _insert_sample_trade()
    df = get_all_trades()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "RY.TO"
    assert row["pnl_dollars"] == 250.0
    assert row["reason"] == "GIVEBACK"


def test_get_all_trades_empty():
    df = get_all_trades()
    assert df.empty


def test_trade_pnl_rounds_to_two_decimals():
    insert_trade("RY.TO", "2026-05-01", 42.50, 100, "2026-05-10", 45.00, 4500.0, 250.123, 5.8811, "STOP_HIT")
    row = get_all_trades().iloc[0]
    assert row["pnl_dollars"] == 250.12
    assert row["pnl_pct"] == 5.88


def test_multiple_trades_ordered_by_sell_date():
    insert_trade("TD.TO",  "2026-05-01", 85.0, 50, "2026-05-12", 88.0, 4400.0, 150.0, 3.5, "GIVEBACK")
    insert_trade("RY.TO",  "2026-05-01", 42.5, 100, "2026-05-10", 45.0, 4500.0, 250.0, 5.9, "STOP_HIT")
    tickers = list(get_all_trades()["ticker"])
    assert tickers == ["RY.TO", "TD.TO"]


# ─────────────────────────────────────────────────────────────────────────────
# transactions (unified ledger)
# ─────────────────────────────────────────────────────────────────────────────

def test_insert_position_creates_buy_transaction():
    insert_position("RY.TO", "2026-05-01", 42.50, 100, pattern="VCP")
    df = get_transactions()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["side"] == "BUY"
    assert row["ticker"] == "RY.TO"
    assert row["price"] == 42.50
    assert row["shares"] == 100.0
    assert row["amount"] == 4250.0
    assert row["reason"] == "VCP"
    assert pd.isna(row["pnl_dollars"])


def test_insert_trade_creates_sell_transaction():
    _insert_sample_trade()
    df = get_transactions()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["side"] == "SELL"
    assert row["ticker"] == "RY.TO"
    assert row["price"] == 45.00
    assert row["pnl_dollars"] == 250.0
    assert row["reason"] == "GIVEBACK"


def test_transactions_contain_both_buy_and_sell():
    insert_position("RY.TO", "2026-05-01", 42.50, 100, pattern="VCP")
    _insert_sample_trade()
    df = get_transactions()
    assert len(df) == 2
    assert list(df["side"]) == ["BUY", "SELL"]


def test_transactions_empty_initially():
    assert get_transactions().empty


def test_buy_without_pattern_has_null_reason():
    insert_position("RY.TO", "2026-05-01", 42.50, 100)
    row = get_transactions().iloc[0]
    assert row["reason"] is None or pd.isna(row["reason"])


def test_sell_transaction_pnl_can_be_negative():
    insert_trade("RY.TO", "2026-05-01", 42.50, 100, "2026-05-10", 40.00, 4000.0, -250.0, -5.88, "STOP_HIT")
    row = get_transactions().iloc[0]
    assert row["pnl_dollars"] == -250.0


# ─────────────────────────────────────────────────────────────────────────────
# signals
# ─────────────────────────────────────────────────────────────────────────────

def _make_signals_df(**overrides):
    row = {
        "ticker": "TD.TO", "pattern": "VCP", "state": "AT_PIVOT",
        "first_seen": "2026-05-01", "last_seen": "2026-05-14",
        "days_in_state": 3, "consecutive_screener_days": 5, "screener_days": 5,
        "entry": 85.0, "stop": 81.0, "target_2r": 93.0, "target_3r": 97.0,
        "risk_pct": 4.7, "pivot_price": 84.5, "detail": "tight range", "alert_sent": False,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_load_signals_empty():
    df = load_signals()
    assert df.empty


def test_save_and_load_signals():
    save_signals(_make_signals_df())
    df = load_signals()
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "TD.TO"
    assert df.iloc[0]["state"] == "AT_PIVOT"


def test_save_signals_replaces_all():
    save_signals(_make_signals_df(ticker="TD.TO"))
    save_signals(_make_signals_df(ticker="RY.TO"))
    df = load_signals()
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "RY.TO"


def test_save_empty_signals_clears_table():
    save_signals(_make_signals_df())
    save_signals(pd.DataFrame())
    assert load_signals().empty


def test_signals_alert_sent_preserved_as_bool():
    save_signals(_make_signals_df(alert_sent=True))
    assert load_signals().iloc[0]["alert_sent"] == True


def test_signals_optional_fields_can_be_null():
    save_signals(_make_signals_df(entry=None, stop=None, detail=None))
    row = load_signals().iloc[0]
    assert pd.isna(row["entry"])
    assert pd.isna(row["stop"])


# ─────────────────────────────────────────────────────────────────────────────
# intents
# ─────────────────────────────────────────────────────────────────────────────

def _make_intent(**overrides):
    intent = {
        "ticker": "ENB.TO", "signal_date": "2026-05-14",
        "alert_state": "CONFIRMED", "priority": 1, "pattern": "VCP",
        "entry_price_planned": 50.0, "stop_price": 47.5,
        "target_price": 55.0, "rr": 2.0,
    }
    intent.update(overrides)
    return intent


def test_load_pending_intents_empty():
    assert load_pending_intents().empty


def test_save_and_load_pending_intents():
    save_intents([_make_intent()])
    df = load_pending_intents()
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "ENB.TO"
    assert df.iloc[0]["intent_status"] == "PENDING"


def test_save_intents_replaces_pending_only():
    save_intents([_make_intent(ticker="ENB.TO", priority=1)])
    pending = load_pending_intents()
    mark_intent_executed(int(pending.iloc[0]["id"]), 50.25, 99)
    save_intents([_make_intent(ticker="SU.TO", priority=1)])
    df = load_pending_intents()
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "SU.TO"


def test_pending_intents_ordered_by_priority():
    save_intents([
        _make_intent(ticker="SU.TO",  priority=2),
        _make_intent(ticker="ENB.TO", priority=1),
    ])
    tickers = list(load_pending_intents()["ticker"])
    assert tickers == ["ENB.TO", "SU.TO"]


def test_intents_ticker_uppercased():
    save_intents([_make_intent(ticker="enb.to")])
    assert load_pending_intents().iloc[0]["ticker"] == "ENB.TO"


def test_mark_intent_executed():
    save_intents([_make_intent()])
    intent_id = int(load_pending_intents().iloc[0]["id"])
    mark_intent_executed(intent_id, 50.25, 99)
    assert load_pending_intents().empty


def test_mark_intent_skipped():
    save_intents([_make_intent()])
    intent_id = int(load_pending_intents().iloc[0]["id"])
    mark_intent_skipped(intent_id, "gap-up filter")
    assert load_pending_intents().empty


def test_mark_intent_executed_on_unknown_id_does_not_raise():
    mark_intent_executed(99999, 50.0, 10)  # must not raise


def test_mark_intent_skipped_on_unknown_id_does_not_raise():
    mark_intent_skipped(99999, "reason")  # must not raise


def test_executed_and_skipped_intents_not_returned_as_pending():
    save_intents([
        _make_intent(ticker="ENB.TO", priority=1),
        _make_intent(ticker="SU.TO",  priority=2),
    ])
    pending = load_pending_intents()
    mark_intent_executed(int(pending.iloc[0]["id"]), 50.0, 100)
    mark_intent_skipped(int(pending.iloc[1]["id"]), "no funds")
    assert load_pending_intents().empty


# ─────────────────────────────────────────────────────────────────────────────
# CORNER CASES
# ─────────────────────────────────────────────────────────────────────────────

# account ────────────────────────────────────────────────────────────────────

def test_set_cash_zero():
    set_cash(0.0)
    assert get_cash() == 0.0


def test_set_cash_negative():
    # No schema guard — negative cash is the caller's problem, not the DB's.
    set_cash(-500.0)
    assert get_cash() == -500.0


# trades ─────────────────────────────────────────────────────────────────────

def test_insert_trade_for_ticker_not_in_positions():
    # No foreign key from trades → positions, so this must succeed.
    insert_trade("GHOST.TO", "2026-05-01", 10.0, 100, "2026-05-05", 11.0, 1100.0, 100.0, 10.0, "STOP_HIT")
    assert len(get_all_trades()) == 1


def test_same_ticker_can_be_traded_multiple_times():
    insert_trade("RY.TO", "2026-05-01", 42.5, 100, "2026-05-10", 45.0, 4500.0, 250.0, 5.9, "GIVEBACK")
    insert_trade("RY.TO", "2026-05-15", 44.0, 100, "2026-05-20", 43.0, 4300.0, -100.0, -2.3, "STOP_HIT")
    df = get_all_trades()
    assert len(df) == 2
    assert list(df["ticker"]) == ["RY.TO", "RY.TO"]


# transactions ───────────────────────────────────────────────────────────────

def test_transaction_amount_is_price_times_shares():
    insert_position("RY.TO", "2026-05-01", 42.123, 137, pattern="VCP")
    row = get_transactions().iloc[0]
    assert row["amount"] == round(42.123 * 137, 2)


def test_transactions_full_lifecycle_buy_then_sell():
    insert_position("RY.TO", "2026-05-01", 42.50, 100, pattern="VCP")
    insert_trade("RY.TO", "2026-05-01", 42.50, 100, "2026-05-10", 45.00, 4500.0, 250.0, 5.88, "GIVEBACK")
    df = get_transactions()
    assert list(df["side"]) == ["BUY", "SELL"]
    assert list(df["ticker"]) == ["RY.TO", "RY.TO"]


def test_transactions_multiple_tickers_same_date_ordered_by_recorded_at():
    # Both inserted on the same trade_date — order must be stable (recorded_at).
    insert_position("AAA.TO", "2026-05-01", 10.0, 100)
    insert_position("ZZZ.TO", "2026-05-01", 20.0, 50)
    df = get_transactions()
    assert list(df["ticker"]) == ["AAA.TO", "ZZZ.TO"]


# atomicity ──────────────────────────────────────────────────────────────────

def test_insert_position_rolls_back_if_transaction_write_fails():
    # Simulate _record_transaction raising after the positions row is written.
    # The whole _connect() transaction must roll back — no position should persist.
    with patch("db._record_transaction", side_effect=RuntimeError("injected")):
        with pytest.raises(RuntimeError, match="injected"):
            insert_position("RY.TO", "2026-05-01", 42.50, 100)
    assert get_open_positions() == []


def test_insert_trade_rolls_back_if_transaction_write_fails():
    with patch("db._record_transaction", side_effect=RuntimeError("injected")):
        with pytest.raises(RuntimeError, match="injected"):
            insert_trade("RY.TO", "2026-05-01", 42.5, 100, "2026-05-10", 45.0, 4500.0, 250.0, 5.9, "GIVEBACK")
    assert get_all_trades().empty
    assert get_transactions().empty


# intents ────────────────────────────────────────────────────────────────────

def test_save_empty_intents_preserves_executed_history():
    save_intents([_make_intent(ticker="ENB.TO")])
    intent_id = int(load_pending_intents().iloc[0]["id"])
    mark_intent_executed(intent_id, 50.0, 100)
    save_intents([])  # new pipeline run finds nothing — must not delete EXECUTED rows
    # Verify executed row still exists by querying directly
    import duckdb
    conn = duckdb.connect(str(db_module.DB_PATH))
    count = conn.execute("SELECT COUNT(*) FROM intents WHERE intent_status = 'EXECUTED'").fetchone()[0]
    conn.close()
    assert count == 1


def test_save_intents_ignores_extra_keys_in_dict():
    intent = _make_intent()
    intent["unexpected_field"] = "should be ignored"
    save_intents([intent])  # must not raise
    assert len(load_pending_intents()) == 1


# signals ────────────────────────────────────────────────────────────────────

def test_save_signals_with_missing_optional_columns():
    # DataFrame with only the required columns — optional ones absent.
    df = pd.DataFrame([{
        "ticker": "TD.TO", "pattern": "VCP", "state": "FORMING",
        "first_seen": "2026-05-01", "last_seen": "2026-05-01",
        "days_in_state": 1, "consecutive_screener_days": 1, "screener_days": 1,
        "alert_sent": False,
        # entry, stop, target_2r, target_3r, risk_pct, pivot_price, detail — absent
    }])
    save_signals(df)  # must not raise
    row = load_signals().iloc[0]
    assert pd.isna(row["entry"])
    assert pd.isna(row["stop"])


def test_save_signals_nan_in_integer_columns_treated_as_zero():
    # Simulates what happens when signal_history.csv has empty cells in int cols.
    df = _make_signals_df()
    df["days_in_state"] = float("nan")
    df["screener_days"] = float("nan")
    save_signals(df)
    row = load_signals().iloc[0]
    assert row["days_in_state"] == 0
    assert row["screener_days"] == 0
