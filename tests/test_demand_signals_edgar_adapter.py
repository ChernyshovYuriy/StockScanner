"""Offline tests for demand_signals.edgar_adapter (no network)."""

import edgar.core
from demand_signals import edgar_adapter
from edgar import store as edgar_store


def _seed_insider_buys(conn):
    edgar_store.save_insider_buys(conn, 723125, [
        {"accession": "acc-1", "owner": "DOE JANE", "shares": 3000.0, "price": 9.5,
         "date": "2026-06-05", "is_officer": False, "is_director": True},
    ])
    edgar_store.save_insider_buys(conn, 999999, [
        {"accession": "acc-2", "owner": "SMITH BOB", "shares": 100.0, "price": 5.0,
         "date": "2026-06-01", "is_officer": True, "is_director": False},
    ])


def test_normalize_recent_insider_buys_maps_cik_to_ticker(tmp_path, monkeypatch):
    conn = edgar_store.connect(tmp_path / "edgar.db")
    _seed_insider_buys(conn)
    monkeypatch.setattr(edgar.core, "load_cik_to_ticker", lambda: {723125: "MU", 999999: "KEY"})

    signals = edgar_adapter.normalize_recent_insider_buys(conn)

    assert len(signals) == 2
    tickers = {s.ticker for s in signals}
    assert tickers == {"MU", "KEY"}
    assert all(s.source == "edgar_insider" and s.signal_type == "insider_buy" for s in signals)
    assert all(s.direction == "bullish" for s in signals)
    assert all(s.us_ticker == s.ticker for s in signals)
    assert all(s.lag_days == 5 for s in signals)


def test_normalize_skips_ciks_with_no_resolvable_ticker(tmp_path, monkeypatch):
    conn = edgar_store.connect(tmp_path / "edgar.db")
    _seed_insider_buys(conn)
    monkeypatch.setattr(edgar.core, "load_cik_to_ticker", lambda: {723125: "MU"})  # 999999 missing

    signals = edgar_adapter.normalize_recent_insider_buys(conn)
    assert len(signals) == 1
    assert signals[0].ticker == "MU"


def test_normalize_since_date_filters(tmp_path, monkeypatch):
    conn = edgar_store.connect(tmp_path / "edgar.db")
    _seed_insider_buys(conn)
    monkeypatch.setattr(edgar.core, "load_cik_to_ticker", lambda: {723125: "MU", 999999: "KEY"})

    signals = edgar_adapter.normalize_recent_insider_buys(conn, since_date="2026-06-03")
    assert len(signals) == 1
    assert signals[0].ticker == "MU"


def test_normalize_strength_scales_with_buy_value(tmp_path, monkeypatch):
    conn = edgar_store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(edgar_adapter, "EDGAR_MIN_BUY_VALUE", 250_000)
    # $3,000,000 buy -> value / (10 * floor) = 3_000_000 / 2_500_000 = 1.2 -> capped at 1.0
    edgar_store.save_insider_buys(conn, 723125, [
        {"accession": "acc-big", "owner": "BIG BUYER", "shares": 100_000.0, "price": 30.0,
         "date": "2026-06-05", "is_officer": True, "is_director": False},
    ])
    monkeypatch.setattr(edgar.core, "load_cik_to_ticker", lambda: {723125: "MU"})

    signals = edgar_adapter.normalize_recent_insider_buys(conn)
    assert signals[0].strength == 1.0


def test_normalize_empty_table_returns_empty_list(tmp_path, monkeypatch):
    conn = edgar_store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(edgar.core, "load_cik_to_ticker", lambda: {})
    assert edgar_adapter.normalize_recent_insider_buys(conn) == []
