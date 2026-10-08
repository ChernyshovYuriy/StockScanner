"""
The 2026-10 core-sleeve fixes (see test_monitor_fixes.py), applied to the
sleeves that copied the same code:

  * momentum_buy / macro_buy / backtest_runner "live" sizing: size on the
    fill price, skip a fill at/below the stop.
  * momentum / macro / kangaroo monitors: never sell on a price that isn't
    from today (drop_stale_sell_rows).
"""
from __future__ import annotations

from datetime import date, datetime

import duckdb
import pandas as pd
import pytest

import db as db_module
import kangaroo_monitor
import macro_buy
import macro_monitor
import momentum_buy
import momentum_monitor
import position_monitor
from backtest_runner import BacktestConfig, _execute_buys
from db import get_cash, get_open_positions_df, init_db, insert_position, save_intents, set_cash
from market_data import HistoricalSliceProvider
from portfolio import PortfolioState
from time_utils import TSX_TZ, set_backtest_clock


@pytest.fixture(autouse=True)
def clock():
    set_backtest_clock(datetime(2026, 5, 14, 15, 50, tzinfo=TSX_TZ))  # Thursday
    yield
    set_backtest_clock(None)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(position_monitor, "send_transaction_email", lambda *a, **k: None)
    monkeypatch.setattr(momentum_buy, "send_transaction_email", lambda *a, **k: None)
    monkeypatch.setattr(macro_buy, "send_transaction_email", lambda *a, **k: None)
    monkeypatch.setattr(momentum_buy, "is_market_open", lambda: True)
    monkeypatch.setattr(macro_buy, "is_market_open", lambda: True)
    yield
    db_module.DB_PATH = tmp_path / "reset.db"


def _intent(ticker="AAA", entry=42.50, stop=35.00, rr=2.0) -> dict:
    return {"ticker": ticker, "signal_date": "2026-05-13", "alert_state": "CONFIRMED",
            "priority": 1, "pattern": "vcp", "entry_price_planned": entry,
            "stop_price": stop, "target_price": entry + 2 * (entry - stop), "rr": rr}


# ─────────────────────────────────────────────────────────────────────────────
# Sizing on the fill price
# ─────────────────────────────────────────────────────────────────────────────
# Risk budget: 10_000 × 1% = $100. Planned 42.50/35.00 = 7.50/share -> 13
# shares; filled at 45.00 the real risk is 10.00/share -> 10 shares. One slot
# (cap = 10_000), so the cap never binds.

def _momentum(tmp_path, monkeypatch, fill):
    init_db(tmp_path / "momentum.db")
    set_cash(10_000.0)
    save_intents([_intent()])
    monkeypatch.setattr(momentum_buy, "MOMENTUM_MAX_POSITIONS", 1)
    monkeypatch.setattr(momentum_buy, "MOMENTUM_RISK_PER_TRADE_PCT", 1.0)
    monkeypatch.setattr(momentum_buy, "fetch_latest_price", lambda t: fill)
    momentum_buy.run_momentum_buy(top_n=None, dry_run=False)


def _macro(tmp_path, monkeypatch, fill):
    core = tmp_path / "trading.db"
    init_db(core)
    save_intents([_intent()])
    init_db(tmp_path / "macro.db")
    set_cash(10_000.0)
    monkeypatch.setattr(macro_buy, "CORE_DB_PATH", core)
    monkeypatch.setattr(macro_buy, "MACRO_MAX_POSITIONS", 1)
    monkeypatch.setattr(macro_buy, "MACRO_RISK_PER_TRADE_PCT", 1.0)
    monkeypatch.setattr(macro_buy, "get_macro_regime",
                        lambda: {"label": "risk_on", "composite": 2, "votes": {}, "detail": {}, "fetched_at": "x"})
    monkeypatch.setattr(macro_buy, "fetch_latest_price", lambda t: fill)
    macro_buy.run_macro_buy(dry_run=False)


@pytest.mark.parametrize("run", [_momentum, _macro], ids=["momentum", "macro"])
def test_gap_up_fill_sized_on_actual_risk(tmp_path, monkeypatch, run):
    run(tmp_path, monkeypatch, 45.00)
    pos = get_open_positions_df()
    assert list(pos["shares"]) == [10]
    assert (45.00 - 35.00) * pos.iloc[0]["shares"] <= 100.0


@pytest.mark.parametrize("run", [_momentum, _macro], ids=["momentum", "macro"])
@pytest.mark.parametrize("fill", [35.00, 33.00])
def test_fill_at_or_below_stop_skipped(tmp_path, monkeypatch, run, fill):
    run(tmp_path, monkeypatch, fill)
    assert get_open_positions_df().empty
    assert get_cash() == pytest.approx(10_000.0)


def test_momentum_marks_below_stop_intent_skipped(tmp_path, monkeypatch):
    _momentum(tmp_path, monkeypatch, 34.00)
    con = duckdb.connect(str(tmp_path / "momentum.db"), read_only=True)
    row = con.execute("SELECT intent_status, intent_reason FROM intents").fetchone()
    con.close()
    assert row == ("SKIPPED", "price_below_stop")


def _backtest_buy(next_open: float):
    days = pd.bdate_range("2026-05-01", periods=10)
    df = pd.DataFrame({"Open": 42.50, "High": 43.0, "Low": 42.0, "Close": 42.50,
                       "Volume": 1_000}, index=days)
    df.iloc[-1, df.columns.get_loc("Open")] = next_open
    portfolio = PortfolioState(10_000.0)
    cfg = BacktestConfig(tickers=["AAA"], sizing="live", max_positions=1, risk_pct=1.0)
    bought = _execute_buys([{"ticker": "AAA", "entry": 42.50, "stop": 35.00}], portfolio,
                           HistoricalSliceProvider({"AAA": df}), days[-2], days[-1].date(), cfg)
    return bought, portfolio


def test_backtest_live_sizing_uses_fill_price():
    bought, portfolio = _backtest_buy(45.00)
    assert bought == ["AAA"]
    assert portfolio.open_positions["AAA"].shares == 10


def test_backtest_live_sizing_skips_fill_below_stop():
    bought, portfolio = _backtest_buy(34.00)
    assert bought == []
    assert portfolio.cash == pytest.approx(10_000.0)


# ─────────────────────────────────────────────────────────────────────────────
# No stale-price sells
# ─────────────────────────────────────────────────────────────────────────────

class _Lock:
    def close(self):
        pass


def _bars(last_day: str) -> pd.DataFrame:
    """60 daily bars at 40, last one collapsing to 30 (far through any stop)."""
    days = pd.bdate_range(end=last_day, periods=60)
    closes = [40.0] * 59 + [30.0]
    return pd.DataFrame({"Open": closes, "High": [c + 0.5 for c in closes],
                         "Low": [c - 0.5 for c in closes], "Close": closes,
                         "Volume": 1_000}, index=days)


def _run_monitor(module, db_attr, monkeypatch, tmp_path, last_day, fake_signals=None):
    df = _bars(last_day)
    db_path = tmp_path / "sleeve.db"
    init_db(db_path)
    set_cash(1_000.0)
    insert_position("AAA", df.index[50].date().isoformat(), 40.0, 10,
                    stop_price=37.0, target_price=50.0)

    monkeypatch.setattr(module, db_attr, db_path)
    monkeypatch.setattr(module, "acquire_lock", lambda service: (None, _Lock()))
    monkeypatch.setattr(module, "log", lambda *a, **k: None)
    monkeypatch.setattr(module, "is_market_open", lambda: True)
    monkeypatch.setattr(module, "fetch_intraday_snapshot", lambda t: None)
    monkeypatch.setattr(module, "load_or_fetch_data", lambda t, start: df)
    monkeypatch.setattr(module, "LOGS_PATH", tmp_path)
    monkeypatch.setattr(module, "append_positions_report", lambda *a, **k: None)
    monkeypatch.setattr(module, "__run_send_report", lambda: None)
    if fake_signals is not None:
        monkeypatch.setattr(module, "compute_signals", fake_signals)
    monkeypatch.setattr("sys.argv", [f"{module.__name__}.py", "--mode", "pre-close"])
    module.main()


def _macro_monitor(monkeypatch, tmp_path, last_day):
    monkeypatch.setattr(macro_monitor, "get_macro_regime",
                        lambda: {"label": "risk_on", "composite": 2, "votes": {}, "detail": {}, "fetched_at": "x"})
    _run_monitor(macro_monitor, "MACRO_DB_PATH", monkeypatch, tmp_path, last_day)


MONITORS = {
    "momentum": lambda mp, tp, d: _run_monitor(momentum_monitor, "MOMENTUM_DB_PATH", mp, tp, d),
    "macro": _macro_monitor,
    "kangaroo": lambda mp, tp, d: _run_monitor(kangaroo_monitor, "KANGAROO_DB_PATH", mp, tp, d),
}


@pytest.mark.parametrize("name", list(MONITORS))
def test_monitor_does_not_sell_on_yesterdays_bar(monkeypatch, tmp_path, name):
    MONITORS[name](monkeypatch, tmp_path, "2026-05-13")
    assert list(get_open_positions_df()["ticker"]) == ["AAA"]
    assert get_cash() == pytest.approx(1_000.0)


@pytest.mark.parametrize("name", list(MONITORS))
def test_monitor_still_sells_on_todays_bar(monkeypatch, tmp_path, name):
    MONITORS[name](monkeypatch, tmp_path, "2026-05-14")
    assert get_open_positions_df().empty
    assert get_cash() > 1_000.0
