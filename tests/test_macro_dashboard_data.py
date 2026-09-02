"""Offline tests for macro_dashboard_data.py (no network — load_or_fetch_data
/ fetch_intraday_snapshot / compute_signals / get_macro_regime are all
monkeypatched)."""

from __future__ import annotations

import pandas as pd
import pytest

import db as db_module
import macro_dashboard_data
from db import init_db, insert_position, set_cash


@pytest.fixture(autouse=True)
def _reset_cache():
    """build_macro_positions() is TTL-cached at module scope — reset before
    and after every test so one test's rows never leak into the next."""
    macro_dashboard_data._cache["rows"] = None
    macro_dashboard_data._cache["ts"] = 0.0
    yield
    macro_dashboard_data._cache["rows"] = None
    macro_dashboard_data._cache["ts"] = 0.0


@pytest.fixture(autouse=True)
def _reset_db_path(tmp_path):
    yield
    db_module.DB_PATH = tmp_path / "reset.db"


def _dummy_df():
    idx = pd.date_range("2026-01-01", periods=30, freq="D")
    return pd.DataFrame({"High": 11.0, "Low": 9.5, "Close": 10.0}, index=idx)


def test_get_macro_cash_reads_from_own_db_not_default_db_path(tmp_path, monkeypatch):
    macro_path = tmp_path / "macro.db"
    init_db(macro_path)
    set_cash(4242.0)

    # Point db.py's global DB_PATH somewhere else entirely, to prove
    # get_macro_cash() never relies on it.
    db_module.DB_PATH = tmp_path / "unrelated.db"
    monkeypatch.setattr(macro_dashboard_data, "MACRO_DB_PATH", macro_path)

    assert macro_dashboard_data.get_macro_cash() == 4242.0


def test_get_macro_cash_returns_zero_when_db_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(macro_dashboard_data, "MACRO_DB_PATH", tmp_path / "does_not_exist.db")
    assert macro_dashboard_data.get_macro_cash() == 0.0


def test_build_macro_positions_never_touches_db_module_global(tmp_path, monkeypatch):
    macro_path = tmp_path / "macro.db"
    init_db(macro_path)
    set_cash(1_000.0)
    insert_position("AAA", "2026-08-01", 10.0, 100.0, pattern="vcp", stop_price=9.0)

    sentinel = tmp_path / "sentinel.db"
    db_module.DB_PATH = sentinel

    monkeypatch.setattr(macro_dashboard_data, "MACRO_DB_PATH", macro_path)
    monkeypatch.setattr(macro_dashboard_data, "load_or_fetch_data", lambda ticker, start: _dummy_df())
    monkeypatch.setattr(macro_dashboard_data, "fetch_intraday_snapshot", lambda t: None)
    monkeypatch.setattr(macro_dashboard_data, "is_market_open", lambda: False)

    rows = macro_dashboard_data.build_macro_positions()

    assert db_module.DB_PATH == sentinel  # untouched by a dashboard read
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAA"


def test_best_effort_price_uses_stale_cache_when_not_intraday():
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    result = macro_dashboard_data._best_effort_price("AAA", df, use_intraday=False)
    assert result == (3.0, "stale-cache")


def test_best_effort_price_returns_none_when_no_data():
    assert macro_dashboard_data._best_effort_price("AAA", pd.DataFrame(), use_intraday=False) is None


def test_get_current_regime_passthrough(monkeypatch):
    fake = {"label": "risk_on", "composite": 2, "votes": {}, "detail": {}, "fetched_at": "x"}
    monkeypatch.setattr(macro_dashboard_data, "get_macro_regime", lambda: fake)
    assert macro_dashboard_data.get_current_regime() == fake
