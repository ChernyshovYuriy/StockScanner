"""
Regression tests for three core-sleeve (Monitor) bugs fixed 2026-10:

  1. virtual_buy sized on the PLANNED entry, not the fill — a gap-up fill
     carried more than RISK_PER_TRADE_PCT of risk, and a fill at/below the
     stop was still bought.
  2. compute_signals' chandelier trail used today's ATR only, so a
     volatility spike LOWERED an armed trailing stop.
  3. A pre-close run could sell at a stale (yesterday's) price: the intraday
     snapshot wasn't checked to be today's session, and the daily-bar/cache
     fallback wasn't either.
"""
from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import pytest

import db as db_module
import position_monitor as pm
from config import MAX_POSITIONS, RISK_PER_TRADE_PCT
from db import get_cash, get_open_positions, init_db, insert_position, save_intents, set_cash
from position_monitor import Position, compute_signals, drop_stale_sell_rows
from time_utils import TSX_TZ, set_backtest_clock
from virtual_buy import run_virtual_buy

TODAY = date(2026, 5, 14)  # a Thursday, not a TSX holiday


@pytest.fixture(autouse=True)
def db(tmp_path):
    path = tmp_path / "trading.db"
    init_db(path)
    yield path
    db_module.DB_PATH = tmp_path / "reset.db"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("virtual_buy.send_transaction_email", lambda **_: None)
    monkeypatch.setattr("position_monitor.send_transaction_email", lambda **_: None)
    monkeypatch.setattr("virtual_buy.get_sector", lambda ticker: "Unknown")


@pytest.fixture(autouse=True)
def clock():
    set_backtest_clock(datetime(2026, 5, 14, 15, 50, tzinfo=TSX_TZ))
    yield
    set_backtest_clock(None)


def _intent(**overrides) -> dict:
    base = {
        "ticker": "RY.TO", "signal_date": "2026-05-13", "alert_state": "CONFIRMED",
        "priority": 1, "pattern": "VCP", "entry_price_planned": 42.50,
        "stop_price": 35.00, "target_price": 57.50, "rr": 2.0,
    }
    base.update(overrides)
    return base


def _intent_row(db_path, ticker="RY.TO") -> tuple:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            "SELECT intent_status, intent_reason FROM intents WHERE ticker = ?", [ticker]
        ).fetchone()
    finally:
        con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1 — size on the fill price
# ─────────────────────────────────────────────────────────────────────────────

def test_gap_up_fill_is_sized_on_actual_risk():
    # $100 risk budget (1% of 10k). Planned 42.50/35.00 = 7.50/share -> 13 shares;
    # filled at 45.00 the real risk is 10.00/share -> must be 10 shares.
    # (Cap: 10_000 / 8 slots = 1250 -> 27 shares at 45, not binding.)
    set_cash(10_000.0)
    save_intents([_intent()])
    with patch("virtual_buy.fetch_latest_price", return_value=45.00):
        run_virtual_buy(top_n=None, dry_run=False)

    pos = get_open_positions()
    assert len(pos) == 1
    assert pos[0]["shares"] == 10
    assert pos[0]["stop_price"] == pytest.approx(35.00)
    assert (45.00 - 35.00) * pos[0]["shares"] <= 10_000.0 * RISK_PER_TRADE_PCT / 100


@pytest.mark.parametrize("fill", [35.10, 38.00, 42.50, 44.00, 47.75, 52.00])
def test_realized_risk_never_exceeds_budget(fill):
    set_cash(10_000.0)
    save_intents([_intent()])
    with patch("virtual_buy.fetch_latest_price", return_value=fill):
        run_virtual_buy(top_n=None, dry_run=False)

    pos = get_open_positions()
    assert len(pos) == 1
    budget = 10_000.0 * RISK_PER_TRADE_PCT / 100
    assert (fill - 35.00) * pos[0]["shares"] <= budget + 1e-9
    assert fill * pos[0]["shares"] <= 10_000.0 / MAX_POSITIONS + 1e-9


@pytest.mark.parametrize("fill", [35.00, 34.20])
def test_fill_at_or_below_stop_is_skipped(db, fill):
    set_cash(10_000.0)
    save_intents([_intent()])
    with patch("virtual_buy.fetch_latest_price", return_value=fill):
        run_virtual_buy(top_n=None, dry_run=False)

    assert get_open_positions() == []
    assert get_cash() == pytest.approx(10_000.0)
    assert _intent_row(db) == ("SKIPPED", "price_below_stop")


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2 — the chandelier trail only ratchets up
# ─────────────────────────────────────────────────────────────────────────────

def _bars(highs, lows, closes, start="2026-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"Open": closes, "High": highs, "Low": lows,
                         "Close": closes, "Volume": 1_000}, index=idx)


def test_volatility_spike_does_not_lower_armed_trailing_stop():
    # 40 quiet bars at 100 (ATR ~1), entry, steady run-up to 120 (arms the
    # chandelier at +8%), then one bar with a 15-point range and no new high.
    closes = [100.0] * 40 + [100.0 + 2 * i for i in range(1, 11)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    closes.append(119.0); highs.append(120.4); lows.append(105.0)  # spike bar
    df = _bars(highs, lows, closes)
    pos = Position("TEST.TO", df.index[40].date(), 100.0, 10, stop_price=93.0)

    before = compute_signals(pos, df.iloc[:-1], planned_stop=93.0)
    after = compute_signals(pos, df, planned_stop=93.0)

    # The spike really did widen ATR enough that the old today-only formula
    # would have dropped the stop...
    old_formula = df["High"].iloc[40:].max() - pm.CHAND_TRAIL_ATR_K * after["ATR14"]
    assert after["ATR14"] > before["ATR14"]
    assert old_formula < before["stop_price"]
    # ...but the stop must not move down.
    assert after["stop_price"] >= before["stop_price"]


def test_trailing_stop_is_monotonic_day_by_day():
    # Random walk with drift and a volatility regime change; replay it one bar
    # at a time like the daily monitor does and check the stop never falls
    # once it is above the initial stop.
    rng = np.random.default_rng(7)
    n_pre, n_post = 40, 80
    vol = np.r_[np.full(n_pre + 30, 0.01), np.full(n_post - 30, 0.04)]
    rets = rng.normal(0.004, vol)
    closes = 50.0 * np.cumprod(1 + rets)
    highs = closes * (1 + np.abs(rng.normal(0, vol)))
    lows = closes * (1 - np.abs(rng.normal(0, vol)))
    df = _bars(highs, lows, closes)
    entry = float(closes[n_pre])
    pos = Position("TEST.TO", df.index[n_pre].date(), entry, 10, stop_price=entry * 0.93)

    stops = [compute_signals(pos, df.iloc[:k], planned_stop=pos.stop_price)["stop_price"]
             for k in range(n_pre + 1, len(df) + 1)]
    assert max(stops) > entry * 0.93  # the trail actually armed in this path
    assert all(b >= a - 1e-9 for a, b in zip(stops, stops[1:]))


def test_unarmed_position_still_rides_initial_stop():
    closes = [100.0] * 40 + [101.0, 102.0, 103.0]  # peak +3%, below the 8% arm
    df = _bars([c + 0.5 for c in closes], [c - 0.5 for c in closes], closes)
    pos = Position("TEST.TO", df.index[40].date(), 100.0, 10, stop_price=93.0)
    assert compute_signals(pos, df, planned_stop=93.0)["stop_price"] == pytest.approx(93.0)


# ─────────────────────────────────────────────────────────────────────────────
# Bug 3 — never sell at a stale price
# ─────────────────────────────────────────────────────────────────────────────

def _five_min(day: str, closes) -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:30", periods=len(closes), freq="5min", tz=TSX_TZ)
    return pd.DataFrame({"Open": closes, "High": [c + 0.1 for c in closes],
                         "Low": [c - 0.1 for c in closes], "Close": closes,
                         "Volume": 100}, index=idx)


def test_snapshot_rejects_yesterdays_session(monkeypatch):
    import market_data as md
    monkeypatch.setattr(md.yf, "download", lambda **kw: _five_min("2026-05-13", [10.0, 11.0]))
    assert md.LiveDataProvider().get_intraday_snapshot("RY.TO") is None


def test_snapshot_uses_only_todays_bars(monkeypatch):
    import market_data as md
    both = pd.concat([_five_min("2026-05-13", [5.0, 50.0]), _five_min("2026-05-14", [10.0, 11.0])])
    monkeypatch.setattr(md.yf, "download", lambda **kw: both)
    snap = md.LiveDataProvider().get_intraday_snapshot("RY.TO")
    assert snap.low == pytest.approx(9.9)
    assert snap.high == pytest.approx(11.1)
    assert snap.close == pytest.approx(11.0)


def test_snapshot_handles_utc_index(monkeypatch):
    # 2026-05-14 19:45 UTC is 15:45 ET the same day; 03:00 UTC on the 14th is
    # still the 13th in Toronto and must be dropped.
    import market_data as md
    idx = pd.DatetimeIndex(["2026-05-14 03:00", "2026-05-14 19:45"], tz="UTC")
    df = pd.DataFrame({"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
                       "Close": [1.0, 2.0], "Volume": 1}, index=idx)
    monkeypatch.setattr(md.yf, "download", lambda **kw: df)
    snap = md.LiveDataProvider().get_intraday_snapshot("RY.TO")
    assert snap.low == pytest.approx(2.0)


def test_drop_stale_sell_rows():
    rows = [{"ticker": "A.TO", "last_date": "2026-05-14"},
            {"ticker": "B.TO", "last_date": "2026-05-13"},
            {"ticker": "C.TO"}]
    assert [r["ticker"] for r in drop_stale_sell_rows(rows, TODAY)] == ["A.TO"]


def _run_pre_close(monkeypatch, tmp_path, last_bar_day: str):
    """Run position_monitor.main() in pre-close mode, offline, with one
    position far below its stop and the snapshot unavailable, so the price
    comes from the daily bars whose last bar is `last_bar_day`."""
    days = pd.bdate_range(end=last_bar_day, periods=60)
    closes = [40.0] * 59 + [30.0]
    df = pd.DataFrame({"Open": closes, "High": [c + 0.5 for c in closes],
                       "Low": [c - 0.5 for c in closes], "Close": closes,
                       "Volume": 1_000}, index=days)

    set_cash(1_000.0)
    insert_position("RY.TO", days[50].date().isoformat(), 40.0, 10, stop_price=37.0)

    class _Lock:
        def close(self):
            pass

    monkeypatch.setattr(pm, "acquire_lock", lambda service: (tmp_path / "lock", _Lock()))
    monkeypatch.setattr(pm, "init_db", lambda: None)  # keep the test DB, not data/trading.db
    monkeypatch.setattr(pm, "log", lambda *a, **k: None)
    monkeypatch.setattr(pm, "is_market_open", lambda: True)
    monkeypatch.setattr(pm, "fetch_intraday_snapshot", lambda t: None)
    monkeypatch.setattr(pm, "load_or_fetch_data", lambda t, start: df)
    monkeypatch.setattr(pm, "LOGS_PATH", tmp_path)
    monkeypatch.setattr(pm, "_write_position_report", lambda **k: None)
    monkeypatch.setattr(pm, "__run_send_report", lambda: None)
    monkeypatch.setattr("sys.argv", ["position_monitor.py", "--mode", "pre-close"])
    pm.main()


def test_pre_close_does_not_sell_on_yesterdays_bar(monkeypatch, tmp_path):
    _run_pre_close(monkeypatch, tmp_path, last_bar_day="2026-05-13")
    assert len(get_open_positions()) == 1
    assert get_cash() == pytest.approx(1_000.0)


def test_pre_close_still_sells_on_todays_bar(monkeypatch, tmp_path):
    _run_pre_close(monkeypatch, tmp_path, last_bar_day="2026-05-14")
    assert get_open_positions() == []
    assert get_cash() == pytest.approx(1_000.0 + 10 * 30.0)
