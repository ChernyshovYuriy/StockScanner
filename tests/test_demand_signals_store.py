"""Offline tests for demand_signals.store persistence (no network)."""

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


def test_save_and_query_single_signal_roundtrips(tmp_path):
    conn = store.connect(tmp_path / "demand_signals.db")
    s = _signal()
    store.save_signal(conn, s)

    got = store.signals_for_ticker(conn, "MU")
    assert got == [s]


def test_save_signals_batch(tmp_path):
    conn = store.connect(tmp_path / "demand_signals.db")
    signals = [
        _signal(source="edgar_insider", signal_type="insider_buy"),
        _signal(source="finra_darkpool", signal_type="darkpool_ratio"),
        _signal(source="options_flow", signal_type="call_put_skew"),
    ]
    store.save_signals(conn, signals)

    got = store.signals_for_ticker(conn, "MU")
    assert len(got) == 3
    assert {s.source for s in got} == {"edgar_insider", "finra_darkpool", "options_flow"}


def test_same_key_upsert_replaces_not_duplicates(tmp_path):
    """Same (ticker, date, source, signal_type): a re-run replaces the row
    (e.g. FINRA restating a prior week), it doesn't accumulate duplicates."""
    conn = store.connect(tmp_path / "demand_signals.db")
    store.save_signal(conn, _signal(strength=0.3))
    store.save_signal(conn, _signal(strength=0.9))

    got = store.signals_for_ticker(conn, "MU")
    assert len(got) == 1
    assert got[0].strength == 0.9


def test_different_tickers_are_independent(tmp_path):
    conn = store.connect(tmp_path / "demand_signals.db")
    store.save_signals(conn, [_signal(ticker="MU", us_ticker="MU"),
                               _signal(ticker="KEY", us_ticker="KEY")])
    assert len(store.signals_for_ticker(conn, "MU")) == 1
    assert len(store.signals_for_ticker(conn, "KEY")) == 1
    assert store.signals_for_ticker(conn, "AMD") == []


def test_since_date_filters_older_signals(tmp_path):
    conn = store.connect(tmp_path / "demand_signals.db")
    store.save_signals(conn, [
        _signal(date="2026-05-01", signal_type="old"),
        _signal(date="2026-06-05", signal_type="new"),
    ])
    got = store.signals_for_ticker(conn, "MU", since_date="2026-06-01")
    assert len(got) == 1
    assert got[0].signal_type == "new"


def test_signals_for_ticker_newest_first(tmp_path):
    conn = store.connect(tmp_path / "demand_signals.db")
    store.save_signals(conn, [
        _signal(date="2026-06-01", signal_type="a"),
        _signal(date="2026-06-08", signal_type="b"),
        _signal(date="2026-06-05", signal_type="c"),
    ])
    got = store.signals_for_ticker(conn, "MU")
    assert [s.date for s in got] == ["2026-06-08", "2026-06-05", "2026-06-01"]
