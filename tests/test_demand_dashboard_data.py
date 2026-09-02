"""Offline tests for demand_dashboard_data (no network)."""

import demand_dashboard_data as ddd
from demand_signals import store
from demand_signals.schema import DemandSignal


def _signal(**overrides):
    fields = dict(
        ticker="MU", us_ticker="MU", date="2026-06-05",
        source="edgar_insider", signal_type="insider_buy",
        direction="bullish", strength=0.5, lag_days=5,
        detail={"owner": "DOE JANE"}, fetched_at="2026-06-05T12:00:00",
    )
    fields.update(overrides)
    return DemandSignal(**fields)


def test_read_returns_empty_when_db_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(ddd, "DEMAND_DB_PATH", tmp_path / "does_not_exist.db")
    assert ddd._read_demand_signals() == []


def test_read_returns_stored_signals(tmp_path, monkeypatch):
    db_path = tmp_path / "demand_signals.db"
    conn = store.connect(db_path)
    store.save_signal(conn, _signal())
    monkeypatch.setattr(ddd, "DEMAND_DB_PATH", db_path)

    rows = ddd._read_demand_signals()
    assert len(rows) == 1
    assert rows[0]["ticker"] == "MU"
    assert rows[0]["detail"] == {"owner": "DOE JANE"}


def test_build_by_ticker_groups_correctly(tmp_path, monkeypatch):
    db_path = tmp_path / "demand_signals.db"
    conn = store.connect(db_path)
    store.save_signals(conn, [
        _signal(ticker="MU", signal_type="insider_buy"),
        _signal(ticker="MU", source="options_flow", signal_type="call_put_skew"),
        _signal(ticker="KEY", signal_type="insider_buy"),
    ])
    monkeypatch.setattr(ddd, "DEMAND_DB_PATH", db_path)

    by_ticker = ddd._build_demand_signals_by_ticker()
    assert set(by_ticker.keys()) == {"MU", "KEY"}
    assert len(by_ticker["MU"]) == 2
    assert len(by_ticker["KEY"]) == 1


def test_build_demand_signals_by_ticker_caches_within_ttl(tmp_path, monkeypatch):
    db_path = tmp_path / "demand_signals.db"
    conn = store.connect(db_path)
    store.save_signal(conn, _signal())
    monkeypatch.setattr(ddd, "DEMAND_DB_PATH", db_path)
    monkeypatch.setattr(ddd, "DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS", 999)
    ddd._cache["ts"] = 0.0
    ddd._cache["rows"] = None

    first = ddd.build_demand_signals_by_ticker()
    # A second signal saved after the first read must NOT appear yet --
    # confirms the TTL cache is actually being served, not re-read every call.
    store.save_signal(conn, _signal(ticker="KEY"))
    second = ddd.build_demand_signals_by_ticker()

    assert first == second
    assert "KEY" not in second
