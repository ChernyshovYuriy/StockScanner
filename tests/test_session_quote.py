"""
Buy fills must be priced from TODAY's session (2026-10 fix).

The buy services used get_quote(): fast_info["last_price"] is yesterday's
close until a ticker's first trade of the day, and its period="1d" 1-minute
fallback can be yesterday's session — so a thin name that hadn't traded by
09:45 was bought at a price that never traded today. They now use
get_session_quote(), which returns None in that case (skipped as
no_price_data). report.py, a display that runs any time, keeps get_quote().
"""
from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd
import pytest

import db as db_module
import macro_buy
import market_data as md
import momentum_buy
import report
import virtual_buy
from db import get_cash, get_open_positions_df, init_db, save_intents, set_cash
from time_utils import TSX_TZ, set_backtest_clock


@pytest.fixture(autouse=True)
def clock():
    set_backtest_clock(datetime(2026, 5, 14, 9, 45, tzinfo=TSX_TZ))  # Thursday, buy slot
    yield
    set_backtest_clock(None)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    for mod in (virtual_buy, momentum_buy, macro_buy):
        monkeypatch.setattr(mod, "send_transaction_email", lambda *a, **k: None)
        monkeypatch.setattr(mod, "is_market_open", lambda: True)
    monkeypatch.setattr(virtual_buy, "get_sector", lambda t: "Unknown")
    # get_quote's fast_info path would answer with yesterday's close — make
    # sure nothing on the buy path can reach it.
    monkeypatch.setattr(md.LiveDataProvider, "get_quote",
                        lambda self, t: pytest.fail("buy path used get_quote()"))
    yield
    db_module.DB_PATH = tmp_path / "reset.db"


def _one_min(day: str, closes, tz=TSX_TZ) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:30", periods=len(closes), freq="1min", tz=tz)
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": closes, "Volume": 100}, index=idx)


def _yf(monkeypatch, df):
    monkeypatch.setattr(md.yf, "download", lambda **kw: df)


# ── get_session_quote ───────────────────────────────────────────────────────

def test_session_quote_rejects_yesterdays_bars(monkeypatch):
    _yf(monkeypatch, _one_min("2026-05-13", [10.0, 10.5]))
    assert md.LiveDataProvider().get_session_quote("RY.TO") is None


def test_session_quote_takes_todays_last_close(monkeypatch):
    _yf(monkeypatch, pd.concat([_one_min("2026-05-13", [10.0, 99.0]), _one_min("2026-05-14", [11.0, 11.2])]))
    assert md.LiveDataProvider().get_session_quote("RY.TO") == pytest.approx(11.2)


def test_session_quote_converts_utc_index(monkeypatch):
    # 03:00 UTC on the 14th is still the 13th in Toronto; 13:40 UTC is 09:40 ET.
    idx = pd.DatetimeIndex(["2026-05-14 03:00", "2026-05-14 13:40"], tz="UTC")
    _yf(monkeypatch, pd.DataFrame({"Close": [5.0, 6.0]}, index=idx))
    assert md.LiveDataProvider().get_session_quote("RY.TO") == pytest.approx(6.0)


@pytest.mark.parametrize("result", [pd.DataFrame(), None])
def test_session_quote_none_when_no_data(monkeypatch, result):
    _yf(monkeypatch, result)
    assert md.LiveDataProvider().get_session_quote("RY.TO") is None


def test_session_quote_none_on_error(monkeypatch):
    def boom(**kw):
        raise RuntimeError("network down")
    monkeypatch.setattr(md.yf, "download", boom)
    assert md.LiveDataProvider().get_session_quote("RY.TO") is None


def test_historical_provider_has_no_session():
    assert md.HistoricalSliceProvider({}).get_session_quote("RY.TO") is None


# ── buy services end to end (only yfinance itself is faked) ──────────────────

def _intent(ticker="AAA"):
    return {"ticker": ticker, "signal_date": "2026-05-13", "alert_state": "CONFIRMED",
            "priority": 1, "pattern": "vcp", "entry_price_planned": 10.0,
            "stop_price": 9.0, "target_price": 12.0, "rr": 2.0}


def _run_core(tmp_path, monkeypatch):
    init_db(tmp_path / "trading.db")
    set_cash(10_000.0)
    save_intents([_intent()])
    virtual_buy.run_virtual_buy(top_n=None, dry_run=False)
    return tmp_path / "trading.db"


def _run_momentum(tmp_path, monkeypatch):
    init_db(tmp_path / "momentum.db")
    set_cash(10_000.0)
    save_intents([_intent()])
    momentum_buy.run_momentum_buy(top_n=None, dry_run=False)
    return tmp_path / "momentum.db"


def _run_macro(tmp_path, monkeypatch):
    core = tmp_path / "trading.db"
    init_db(core)
    save_intents([_intent()])
    init_db(tmp_path / "macro.db")
    set_cash(10_000.0)
    monkeypatch.setattr(macro_buy, "CORE_DB_PATH", core)
    monkeypatch.setattr(macro_buy, "get_macro_regime",
                        lambda: {"label": "risk_on", "composite": 2, "votes": {}, "detail": {}, "fetched_at": "x"})
    macro_buy.run_macro_buy(dry_run=False)
    return None


RUNS = {"core": _run_core, "momentum": _run_momentum, "macro": _run_macro}


@pytest.mark.parametrize("name", list(RUNS))
def test_no_buy_when_ticker_has_not_traded_today(tmp_path, monkeypatch, name):
    _yf(monkeypatch, _one_min("2026-05-13", [10.0, 10.1]))  # only yesterday's session
    db_path = RUNS[name](tmp_path, monkeypatch)
    assert get_open_positions_df().empty
    assert get_cash() == pytest.approx(10_000.0)
    if db_path is not None:  # macro never marks the core's intents
        con = duckdb.connect(str(db_path), read_only=True)
        row = con.execute("SELECT intent_status, intent_reason FROM intents").fetchone()
        con.close()
        assert row == ("SKIPPED", "no_price_data")


@pytest.mark.parametrize("name", list(RUNS))
def test_buys_at_todays_price(tmp_path, monkeypatch, name):
    _yf(monkeypatch, pd.concat([_one_min("2026-05-13", [10.0]), _one_min("2026-05-14", [10.2, 10.3])]))
    RUNS[name](tmp_path, monkeypatch)
    pos = get_open_positions_df()
    assert list(pos["ticker"]) == ["AAA"]
    assert pos.iloc[0]["entry_price"] == pytest.approx(10.3)


# ── report.py stays on the plain quote (works off-hours) ────────────────────

def test_report_uses_plain_quote(monkeypatch):
    monkeypatch.setattr(report.DEFAULT_PROVIDER, "get_quote", lambda t: 12.0)
    monkeypatch.setattr(report.DEFAULT_PROVIDER, "get_session_quote",
                        lambda t: pytest.fail("report must not need a session price"))
    out = report._mark_to_market(pd.DataFrame({"ticker": ["AAA"], "entry_price": [10.0], "shares": [5.0]}))
    assert out.iloc[0]["last"] == pytest.approx(12.0)
    assert out.iloc[0]["unrealized_pnl"] == pytest.approx(10.0)
