"""
Offline integration tests for triple_screen_tracker_service.run_collector
(no network -- provider is always a fake/stub; every real fetch call would
be a test bug). `send_text_email` is monkeypatched everywhere so a test run
can never actually send a live email through the user's configured Gmail
account.
"""
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

import triple_screen_tracker_service
from research.triple_screen.types import PriceData
from time_utils import TSX_TZ, set_backtest_clock
from triple_screen_tracker import store

from triple_screen_fixtures import FakeDataProvider, aligned, buy_ready_entry_bars, uptrend_bars


@pytest.fixture(autouse=True)
def _no_real_email(monkeypatch):
    monkeypatch.setattr(triple_screen_tracker_service, "send_text_email", lambda subject, body: True)


@pytest.fixture(autouse=True)
def _reset_backtest_clock():
    yield
    set_backtest_clock(None)


def _pin_clock(ts: pd.Timestamp) -> None:
    set_backtest_clock(datetime(ts.year, ts.month, ts.day, 17, 0, tzinfo=TSX_TZ))


def _entry_bars(closes, end_date, ticker="T") -> PriceData:
    """Daily OHLCV bars ending exactly on the last business day on/before
    `end_date` (freq="B" so tests fully control which weekday "today" is,
    regardless of calendar-date arithmetic)."""
    n = len(closes)
    idx = pd.date_range(end=end_date, periods=n, freq="B")
    opens = [closes[0]] + list(closes[:-1])
    highs = [max(o, c) + 0.5 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.5 for o, c in zip(opens, closes)]
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                        "Volume": 1_000_000.0}, index=idx)
    return PriceData(ticker=ticker, timeframe="daily", bars=df)


def _buy_ready_entry_bars(end_date, ticker="T") -> PriceData:
    """The one daily bar shape that satisfies both Screen 2's pullback and
    Screen 3's true-breakout confirmation on the identical latest bar -- see
    triple_screen_fixtures.buy_ready_entry_bars (numerically verified, not
    hand-guessed; the two are structurally in tension). Buy price (last
    close) is 101.0."""
    return buy_ready_entry_bars(trend="UP", end_date=end_date, ticker=ticker)


def _next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.bdate_range(start=ts, periods=2)[-1]


def _a_saturday() -> date:
    d = date(2024, 6, 3)  # arbitrary weekday anchor
    return d + timedelta(days=(5 - d.weekday()) % 7)


# ── fresh BUY signal ──────────────────────────────────────────────────────

def test_fresh_buy_signal_creates_tracked_record_at_close_price(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    buy_bars = _buy_ready_entry_bars(end_date="2024-06-07", ticker="BUY1")
    today = buy_bars.bars.index[-1]
    _pin_clock(today)

    provider = FakeDataProvider({
        "BUY1": aligned(trend=uptrend_bars(timeframe="weekly"), entry=buy_bars),
    })
    triple_screen_tracker_service.run_collector("run1", tickers=["BUY1"], provider=provider, conn=conn)

    assert store.open_tracked_tickers(conn) == {"BUY1"}
    record = store.open_records(conn)[0]
    assert record["buy_price"] == 101.0
    assert record["buy_date"] == today.strftime("%Y-%m-%d")
    history = store.price_history_for(conn, record["id"])
    assert history == [{"date": today.strftime("%Y-%m-%d"), "close_price": 101.0}]


def test_no_buy_signal_creates_nothing(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    flat = _entry_bars([100, 100, 100, 100], end_date="2024-06-07", ticker="FLAT1")
    today = flat.bars.index[-1]
    _pin_clock(today)

    provider = FakeDataProvider({
        "FLAT1": aligned(trend=uptrend_bars(timeframe="weekly"), entry=flat),
    })
    triple_screen_tracker_service.run_collector("run1", tickers=["FLAT1"], provider=provider, conn=conn)

    assert store.all_tracked(conn) == []


# ── "if a record exists, skip it" ────────────────────────────────────────

def test_already_open_ticker_is_excluded_from_the_buy_scan(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    buy_bars = _buy_ready_entry_bars(end_date="2024-06-07", ticker="AAPL")
    today = buy_bars.bars.index[-1]
    store.create_tracked(conn, "AAPL", "2024-06-06", 50.0, created_at="2024-06-06T17:00:00")
    _pin_clock(today)

    # Even though this fixture is a fully-formed BUY setup, AAPL already has
    # an OPEN record -- it must not create a second one.
    provider = FakeDataProvider({
        "AAPL": aligned(trend=uptrend_bars(timeframe="weekly"), entry=buy_bars),
    })
    triple_screen_tracker_service.run_collector("run1", tickers=["AAPL"], provider=provider, conn=conn)

    all_rows = store.all_tracked(conn)
    assert len(all_rows) == 1
    assert all_rows[0]["buy_price"] == 50.0  # original record untouched


# ── daily price roll-forward ─────────────────────────────────────────────

def test_open_position_price_appended_daily(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    day1_bars = _entry_bars([100, 101, 102], end_date="2024-06-05", ticker="AAPL")
    day1 = day1_bars.bars.index[-1]
    tid = store.create_tracked(conn, "AAPL", day1.strftime("%Y-%m-%d"), 102.0,
                                created_at=f"{day1.strftime('%Y-%m-%d')}T17:00:00")
    store.append_price(conn, tid, day1.strftime("%Y-%m-%d"), 102.0)

    day2 = _next_business_day(day1)
    day2_bars = _entry_bars([102, 103], end_date=day2, ticker="AAPL")
    _pin_clock(day2)

    provider = FakeDataProvider({
        "AAPL": aligned(trend=uptrend_bars(timeframe="weekly"), entry=day2_bars),
    })
    triple_screen_tracker_service.run_collector("run2", tickers=[], provider=provider, conn=conn)

    history = store.price_history_for(conn, tid)
    assert [h["date"] for h in history] == [day1.strftime("%Y-%m-%d"), day2.strftime("%Y-%m-%d")]
    assert history[-1]["close_price"] == 103.0
    assert store.open_tracked_tickers(conn) == {"AAPL"}  # still above buy price, still open


# ── zero-tolerance sell rule ──────────────────────────────────────────────

def test_price_below_buy_closes_the_record(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    tid = store.create_tracked(conn, "AAPL", "2024-06-05", 100.0, created_at="2024-06-05T17:00:00")

    day2_bars = _entry_bars([100, 95], end_date="2024-06-06", ticker="AAPL")
    day2 = day2_bars.bars.index[-1]
    _pin_clock(day2)

    provider = FakeDataProvider({
        "AAPL": aligned(trend=uptrend_bars(timeframe="weekly"), entry=day2_bars),
    })
    triple_screen_tracker_service.run_collector("run2", tickers=[], provider=provider, conn=conn)

    assert store.open_tracked_tickers(conn) == set()
    row = store.all_tracked(conn)[0]
    assert row["status"] == "SOLD"
    assert row["sell_price"] == 95.0
    assert row["sell_date"] == day2.strftime("%Y-%m-%d")
    assert row["id"] == tid


def test_price_exactly_equal_to_buy_stays_open_not_sold(tmp_path):
    """Zero-tolerance means strictly below, not below-or-equal."""
    conn = store.connect(tmp_path / "ts.db")
    store.create_tracked(conn, "AAPL", "2024-06-05", 100.0, created_at="2024-06-05T17:00:00")

    day2_bars = _entry_bars([100, 100], end_date="2024-06-06", ticker="AAPL")
    day2 = day2_bars.bars.index[-1]
    _pin_clock(day2)

    provider = FakeDataProvider({
        "AAPL": aligned(trend=uptrend_bars(timeframe="weekly"), entry=day2_bars),
    })
    triple_screen_tracker_service.run_collector("run2", tickers=[], provider=provider, conn=conn)

    assert store.open_tracked_tickers(conn) == {"AAPL"}


# ── per-ticker isolation ─────────────────────────────────────────────────

class _RaisingProvider:
    """Raises for every ticker except one -- proves a single bad ticker
    doesn't abort the whole run (same isolation guarantee as
    research/triple_screen/batch.run_batch)."""

    def __init__(self, good_ticker, good_bars):
        self._good_ticker = good_ticker
        self._good_bars = good_bars

    def get_bars(self, ticker, timeframes=None):
        if ticker == self._good_ticker:
            return self._good_bars
        raise RuntimeError(f"boom: {ticker}")


def test_one_ticker_provider_failure_does_not_abort_the_run(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    buy_bars = _buy_ready_entry_bars(end_date="2024-06-07", ticker="GOOD")
    today = buy_bars.bars.index[-1]
    _pin_clock(today)

    provider = _RaisingProvider("GOOD", aligned(trend=uptrend_bars(timeframe="weekly"), entry=buy_bars))
    # Must not raise, despite BAD1/BAD2 blowing up.
    triple_screen_tracker_service.run_collector(
        "run1", tickers=["BAD1", "GOOD", "BAD2"], provider=provider, conn=conn)

    assert store.open_tracked_tickers(conn) == {"GOOD"}


# ── staleness guard ───────────────────────────────────────────────────────

def test_stale_bar_data_is_not_recorded_as_todays_price(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    stale_bars = _buy_ready_entry_bars(end_date="2024-06-07", ticker="STALE")
    stale_date = stale_bars.bars.index[-1]
    # "Today" is one business day AFTER the fixture's latest bar -- the
    # fetched data is stale relative to "today" and must be skipped, not
    # silently recorded as if it were today's price.
    today = _next_business_day(stale_date)
    _pin_clock(today)

    provider = FakeDataProvider({
        "STALE": aligned(trend=uptrend_bars(timeframe="weekly"), entry=stale_bars),
    })
    triple_screen_tracker_service.run_collector("run1", tickers=["STALE"], provider=provider, conn=conn)

    assert store.all_tracked(conn) == []


# ── dry-run ───────────────────────────────────────────────────────────────

def test_dry_run_writes_nothing(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    buy_bars = _buy_ready_entry_bars(end_date="2024-06-07", ticker="BUY1")
    today = buy_bars.bars.index[-1]
    _pin_clock(today)

    provider = FakeDataProvider({
        "BUY1": aligned(trend=uptrend_bars(timeframe="weekly"), entry=buy_bars),
    })
    triple_screen_tracker_service.run_collector(
        "run1", dry_run=True, tickers=["BUY1"], provider=provider, conn=conn)

    assert store.all_tracked(conn) == []
    assert store.already_sent(conn, today.strftime("%Y-%m-%d")) is False


# ── non-trading-day short circuit ────────────────────────────────────────

def test_non_trading_day_short_circuits_before_any_scan(tmp_path):
    conn = store.connect(tmp_path / "ts.db")
    saturday = _a_saturday()
    set_backtest_clock(datetime(saturday.year, saturday.month, saturday.day, 17, 0, tzinfo=TSX_TZ))

    class _ExplodingProvider:
        def get_bars(self, ticker, timeframes=None):
            raise AssertionError("provider must not be called on a non-trading day")

    triple_screen_tracker_service.run_collector(
        "run1", tickers=["AAPL"], provider=_ExplodingProvider(), conn=conn)

    assert store.all_tracked(conn) == []


# ── same-day rerun email idempotency ─────────────────────────────────────

def test_same_day_rerun_does_not_send_a_second_email(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "ts.db")
    buy_bars = _buy_ready_entry_bars(end_date="2024-06-07", ticker="BUY1")
    today = buy_bars.bars.index[-1]
    _pin_clock(today)

    send_calls = []
    monkeypatch.setattr(triple_screen_tracker_service, "send_text_email",
                         lambda subject, body: send_calls.append(1) or True)

    provider = FakeDataProvider({
        "BUY1": aligned(trend=uptrend_bars(timeframe="weekly"), entry=buy_bars),
    })
    triple_screen_tracker_service.run_collector("run1", tickers=["BUY1"], provider=provider, conn=conn)
    triple_screen_tracker_service.run_collector("run2", tickers=["BUY1"], provider=provider, conn=conn)

    assert len(send_calls) == 1
