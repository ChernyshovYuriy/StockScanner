"""Offline integration test for demand_signals_service.run_collector (no network)."""

import demand_signals_service
from demand_signals import darkpool, store
from demand_signals.options_flow import OptionsSnapshot
from edgar import store as edgar_store


class _FakeProvider:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def snapshot(self, us_ticker):
        return self._snapshot


def test_run_collector_combines_all_three_sources(tmp_path, monkeypatch):
    demand_conn = store.connect(tmp_path / "demand_signals.db")
    edgar_conn = edgar_store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(demand_signals_service.store, "connect", lambda *a, **k: demand_conn)
    monkeypatch.setattr(demand_signals_service.edgar_store, "connect", lambda *a, **k: edgar_conn)

    edgar_store.set_watchlist(edgar_conn, [("MU", 723125)])
    edgar_store.save_insider_buys(edgar_conn, 723125, [
        {"accession": "acc-1", "owner": "DOE JANE", "shares": 3000.0, "price": 9.5,
         "date": "2026-06-05", "is_officer": False, "is_director": True},
    ])

    monkeypatch.setattr(demand_signals_service, "load_cik_to_ticker", lambda: {723125: "MU"})
    monkeypatch.setattr(demand_signals_service, "get_us_ticker", lambda t: "MU")

    monkeypatch.setattr(darkpool, "fetch_weekly_ats_volume",
                         lambda us_ticker, weeks=8: [{"week_start": "2026-06-01", "shares": 50_000}])
    monkeypatch.setattr(darkpool, "_total_weekly_volume", lambda us_ticker, week: 1_000_000)

    snap = OptionsSnapshot(us_ticker="MU", as_of_date="2026-06-05",
                            call_volume=300, put_volume=50, call_oi=100, put_oi=100)
    monkeypatch.setattr(demand_signals_service, "YahooOptionsProvider",
                         lambda: _FakeProvider(snap))

    demand_signals_service.run_collector("rid", dry_run=False)

    stored = store.signals_for_ticker(demand_conn, "MU")
    sources = {s.source for s in stored}
    assert sources == {"edgar_insider", "finra_darkpool", "options_flow"}


def test_run_collector_skips_tickers_with_no_us_line(tmp_path, monkeypatch):
    demand_conn = store.connect(tmp_path / "demand_signals.db")
    edgar_conn = edgar_store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(demand_signals_service.store, "connect", lambda *a, **k: demand_conn)
    monkeypatch.setattr(demand_signals_service.edgar_store, "connect", lambda *a, **k: edgar_conn)

    edgar_store.set_watchlist(edgar_conn, [("JUNIOR", 555555)])
    monkeypatch.setattr(demand_signals_service, "load_cik_to_ticker", lambda: {555555: "JUNIOR.V"})
    monkeypatch.setattr(demand_signals_service, "get_us_ticker", lambda t: None)  # no US line

    darkpool_calls = {"n": 0}
    monkeypatch.setattr(darkpool, "fetch_weekly_ats_volume",
                         lambda *a, **k: darkpool_calls.__setitem__("n", darkpool_calls["n"] + 1) or [])

    demand_signals_service.run_collector("rid", dry_run=False)

    assert darkpool_calls["n"] == 0  # never even attempted for an uncovered ticker
    assert store.signals_for_ticker(demand_conn, "JUNIOR.V") == []


def test_run_collector_dry_run_does_not_write(tmp_path, monkeypatch, capsys):
    demand_conn = store.connect(tmp_path / "demand_signals.db")
    edgar_conn = edgar_store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(demand_signals_service.store, "connect", lambda *a, **k: demand_conn)
    monkeypatch.setattr(demand_signals_service.edgar_store, "connect", lambda *a, **k: edgar_conn)
    monkeypatch.setattr(demand_signals_service, "load_cik_to_ticker", lambda: {})

    demand_signals_service.run_collector("rid", dry_run=True)

    assert store.signals_for_ticker(demand_conn, "MU") == []


def test_run_collector_options_flow_failure_does_not_sink_the_run(tmp_path, monkeypatch):
    """One ticker's options-flow fetch raising must not stop the rest of
    the run (same "one bad filing must not sink the run" convention as
    edgar_service.py's activist-parse guard)."""
    demand_conn = store.connect(tmp_path / "demand_signals.db")
    edgar_conn = edgar_store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(demand_signals_service.store, "connect", lambda *a, **k: demand_conn)
    monkeypatch.setattr(demand_signals_service.edgar_store, "connect", lambda *a, **k: edgar_conn)

    edgar_store.set_watchlist(edgar_conn, [("MU", 723125)])
    monkeypatch.setattr(demand_signals_service, "load_cik_to_ticker", lambda: {723125: "MU"})
    monkeypatch.setattr(demand_signals_service, "get_us_ticker", lambda t: "MU")
    monkeypatch.setattr(darkpool, "fetch_weekly_ats_volume", lambda *a, **k: [])

    class _RaisingProvider:
        def snapshot(self, us_ticker):
            raise RuntimeError("yfinance blew up")

    monkeypatch.setattr(demand_signals_service, "YahooOptionsProvider", lambda: _RaisingProvider())

    # Must not raise.
    demand_signals_service.run_collector("rid", dry_run=False)
