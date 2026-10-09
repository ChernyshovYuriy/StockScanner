"""
tests/test_manual_sell.py
==========================
Unit tests for manual_sell.sell_position() — the core operation reused by
both the CLI (manual_sell.py's main()) and the web dashboard.
"""

from __future__ import annotations

import pytest

import db as db_module
from concurrent_utils import acquire_lock
from db import get_all_trades, get_cash, get_open_positions, init_db, insert_position
from manual_sell import sell_position


@pytest.fixture(autouse=True)
def db(tmp_path):
    path = tmp_path / "trading.db"
    init_db(path)
    yield path
    db_module.DB_PATH = tmp_path / "reset.db"


@pytest.fixture(autouse=True)
def no_emails(monkeypatch):
    """Prevent any test from sending real emails via Gmail."""
    monkeypatch.setattr("position_monitor.send_transaction_email", lambda **_: None)


def test_sell_position_happy_path(monkeypatch):
    set_cash_and_position(monkeypatch)

    result = sell_position("RY.TO")

    assert result["ok"] is True
    assert result["ticker"] == "RY.TO"
    assert result["price"] == 45.00
    assert get_open_positions() == []
    trades = get_all_trades()
    assert len(trades) == 1
    assert trades.iloc[0]["reason"] == "MANUAL_SELL"


def test_sell_position_credits_cash(monkeypatch):
    set_cash_and_position(monkeypatch, cash=1_000.0)

    sell_position("RY.TO")

    assert get_cash() == pytest.approx(1_000.0 + 45.00 * 100, rel=1e-4)


def test_sell_position_normalises_ticker_case(monkeypatch):
    set_cash_and_position(monkeypatch)

    result = sell_position("ry.to")

    assert result["ok"] is True
    assert result["ticker"] == "RY.TO"


def test_sell_position_unknown_ticker():
    from db import set_cash
    set_cash(0.0)

    result = sell_position("NOPE.TO")

    assert result["ok"] is False
    assert result["error"] == "no_position"


def test_sell_position_no_price_available(monkeypatch):
    from db import set_cash
    set_cash(0.0)
    insert_position("RY.TO", "2026-05-01", 42.50, 100)

    monkeypatch.setattr("manual_sell.fetch_intraday_snapshot", lambda ticker: None)
    monkeypatch.setattr("manual_sell.download_ohlc", lambda ticker, start: _empty_df())

    result = sell_position("RY.TO")

    assert result["ok"] is False
    assert result["error"] == "no_price"
    # Position must be untouched when no price could be found.
    assert len(get_open_positions()) == 1


def test_sell_position_returns_locked_when_lock_held():
    from db import set_cash
    set_cash(0.0)
    insert_position("RY.TO", "2026-05-01", 42.50, 100)

    _, lock_file = acquire_lock("manual_sell")
    try:
        result = sell_position("RY.TO")
        assert result["ok"] is False
        assert result["error"] == "locked"
        # Nothing should have been touched.
        assert len(get_open_positions()) == 1
    finally:
        lock_file.close()


def test_sell_position_releases_lock_on_exception(monkeypatch):
    set_cash_and_position(monkeypatch)
    monkeypatch.setattr(
        "manual_sell.execute_virtual_sells",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError):
        sell_position("RY.TO")

    # The lock must have been released despite the exception — otherwise
    # every subsequent sell (CLI or web) would be permanently blocked.
    _, lock_file = acquire_lock("manual_sell")
    lock_file.close()


def test_sell_position_dry_run_writes_nothing(monkeypatch):
    set_cash_and_position(monkeypatch, cash=5_000.0)

    result = sell_position("RY.TO", dry_run=True)

    assert result["ok"] is True
    assert len(get_open_positions()) == 1
    assert get_cash() == 5_000.0
    assert get_all_trades().empty


def test_sell_position_uses_manual_price_when_given(monkeypatch):
    """A ticker with no live quote (e.g. Yahoo Finance has stopped carrying
    it) must still be sellable via an explicit price override, bypassing
    get_market_price() entirely."""
    from db import set_cash
    set_cash(0.0)
    insert_position("BLN.TO", "2026-06-23", 9.00, 76)

    def _boom(*args, **kwargs):
        raise AssertionError("get_market_price() should not be called when price is given")

    monkeypatch.setattr("manual_sell.fetch_intraday_snapshot", _boom)
    monkeypatch.setattr("manual_sell.download_ohlc", _boom)

    result = sell_position("BLN.TO", price=8.50)

    assert result["ok"] is True
    assert result["price"] == 8.50
    assert result["source"] == "manual"
    assert get_open_positions() == []
    trades = get_all_trades()
    assert trades.iloc[0]["sell_price"] == 8.50


def test_cli_momentum_sleeve_sells_from_momentum_db_and_prints_json(monkeypatch, tmp_path, capsys):
    """--sleeve momentum --json (what the dashboard's momentum Sell button
    runs) sells from the momentum DB, labels the email Momentum, and prints
    only the JSON result on stdout."""
    import json
    import manual_sell
    from db import set_cash

    momentum_path = tmp_path / "momentum.db"
    init_db(momentum_path)
    set_cash(0.0)
    insert_position("AEM.TO", "2026-09-01", 100.0, 10)
    monkeypatch.setattr("config.MOMENTUM_DB_PATH", momentum_path)
    labels = []
    monkeypatch.setattr("position_monitor.send_transaction_email",
                        lambda **kw: labels.append(kw.get("label")))
    monkeypatch.setattr("sys.argv", ["manual_sell.py", "AEM.TO", "--sleeve", "momentum",
                                     "--json", "--price", "110"])

    with pytest.raises(SystemExit) as exc:
        manual_sell.main()

    assert exc.value.code == 0
    result = json.loads(capsys.readouterr().out.strip())
    assert result["ok"] is True and result["price"] == 110.0
    assert get_open_positions() == []
    assert get_cash() == 1_100.0
    assert labels == ["Momentum"]


@pytest.mark.parametrize("sleeve,db_attr,label", [
    ("kangaroo", "KANGAROO_DB_PATH", "Kangaroo Tail"),
    ("macro", "MACRO_DB_PATH", "Macro"),
])
def test_cli_kangaroo_and_macro_sleeves_sell_from_their_db(monkeypatch, tmp_path, capsys, sleeve, db_attr, label):
    import json
    import manual_sell
    from db import set_cash

    path = tmp_path / f"{sleeve}.db"
    init_db(path)
    set_cash(0.0)
    insert_position("CNQ.TO", "2026-09-01", 40.0, 10)
    monkeypatch.setattr(f"config.{db_attr}", path)
    labels = []
    monkeypatch.setattr("position_monitor.send_transaction_email",
                        lambda **kw: labels.append(kw.get("label")))
    monkeypatch.setattr("sys.argv", ["manual_sell.py", "CNQ.TO", "--sleeve", sleeve, "--json", "--price", "41"])

    with pytest.raises(SystemExit) as exc:
        manual_sell.main()

    assert exc.value.code == 0
    assert json.loads(capsys.readouterr().out.strip())["ok"] is True
    assert get_open_positions() == []
    assert get_cash() == 410.0
    assert labels == [label]


def test_momentum_label_uses_its_own_lock():
    """A momentum sell must not be blocked by a core sell in progress."""
    set_cash_and_position_no_price()
    lock_path, lock_file = acquire_lock("manual_sell")
    try:
        result = sell_position("RY.TO", price=45.0, label="Momentum")
    finally:
        lock_file.close()
    assert result["ok"] is True


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _empty_df():
    import pandas as pd
    return pd.DataFrame()


def set_cash_and_position(monkeypatch, cash: float = 0.0, price: float = 45.00):
    from db import set_cash
    set_cash(cash)
    insert_position("RY.TO", "2026-05-01", 42.50, 100)
    monkeypatch.setattr(
        "manual_sell.fetch_intraday_snapshot",
        lambda ticker: _TodayBar(price),
    )


def set_cash_and_position_no_price():
    from db import set_cash
    set_cash(0.0)
    insert_position("RY.TO", "2026-05-01", 42.50, 100)


class _TodayBar:
    def __init__(self, close: float):
        self.close = close
        self.low = close
        self.high = close
        self.source = "5m-intraday"
