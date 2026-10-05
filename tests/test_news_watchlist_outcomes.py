"""
Offline tests for news_watchlist/outcomes.py (forward-return scoring of
press releases) and news_watchlist_service.score_release_outcomes(). No
network -- bar data is always a synthetic DataFrame.
"""
from datetime import date, datetime

import pandas as pd
import pytest

import news_watchlist_service
from news_watchlist import outcomes, store
from press_release_tracker import store as pr_store
from press_release_tracker.feeds import FeedItem
from time_utils import TSX_TZ, set_backtest_clock


def _bars(start, n, open_=10.0, step=0.0, skip=()):
    """n business-day bars from start; Open = open_ + i*step, Close = Open + 0.5.
    Dates in `skip` are dropped (a day the name didn't trade)."""
    idx = pd.bdate_range(start, periods=n)
    opens = [open_ + i * step for i in range(n)]
    df = pd.DataFrame({"Open": opens, "High": opens, "Low": opens,
                       "Close": [o + 0.5 for o in opens], "Volume": 1000}, index=idx)
    return df[[d.date() not in skip for d in df.index]]


# Benchmark: flat 100 open / 100 close, so bench_ret is 0 and easy to reason about.
def _bench(start, n):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0,
                         "Close": 100.0, "Volume": 1}, index=idx)


TODAY = date(2026, 12, 31)


# ── entry rule ───────────────────────────────────────────────────────────

def test_premarket_release_enters_at_that_days_open():
    bars = _bars("2026-09-14", 10, open_=10.0, step=1.0)  # Mon 14th open=10, Tue 15th open=11
    pub = datetime(2026, 9, 15, 8, 0, tzinfo=TSX_TZ)
    out = outcomes.compute_outcome(bars, _bench("2026-09-14", 10), pub, TODAY)
    assert out["entry_date"] == "2026-09-15"
    assert out["entry_open"] == 11.0
    assert out["prior_close"] == 10.5


def test_intraday_release_enters_at_next_sessions_open():
    """Published after 09:30 -- that day's open already happened, so the
    first tradeable open is the next session's (no lookahead)."""
    bars = _bars("2026-09-14", 10, open_=10.0, step=1.0)
    pub = datetime(2026, 9, 15, 9, 30, tzinfo=TSX_TZ)
    out = outcomes.compute_outcome(bars, _bench("2026-09-14", 10), pub, TODAY)
    assert out["entry_date"] == "2026-09-16"
    assert out["prior_close"] == 11.5  # the release day's own close


def test_weekend_release_enters_monday_open():
    bars = _bars("2026-09-14", 10)
    pub = datetime(2026, 9, 19, 14, 0, tzinfo=TSX_TZ)  # Saturday afternoon
    out = outcomes.compute_outcome(bars, _bench("2026-09-14", 10), pub, TODAY)
    assert out["entry_date"] == "2026-09-21"


def test_utc_pubdate_is_converted_to_toronto_time():
    """13:00 GMT is 09:00 ET (EDT) -- pre-market, so same-day entry."""
    pub = outcomes.parse_published("Tue, 15 Sep 2026 13:00:00 GMT")
    assert pub.hour == 9 and pub.minute == 0
    out = outcomes.compute_outcome(_bars("2026-09-14", 10), _bench("2026-09-14", 10), pub, TODAY)
    assert out["entry_date"] == "2026-09-15"


def test_unparseable_pubdate_is_none():
    assert outcomes.parse_published("") is None
    assert outcomes.parse_published("not a date") is None


# ── horizons ─────────────────────────────────────────────────────────────

def test_horizon_returns_and_completion():
    bars = _bars("2026-09-01", 80, open_=10.0, step=0.0)  # flat: close = 10.5 every day
    pub = datetime(2026, 9, 1, 8, 0, tzinfo=TSX_TZ)
    out = outcomes.compute_outcome(bars, _bench("2026-09-01", 80), pub, TODAY)
    for h in outcomes.HORIZONS:
        assert out[f"ret_{h}d"] == pytest.approx(0.05)
        assert out[f"bench_ret_{h}d"] == pytest.approx(0.0)
    assert out["complete"] is True


def test_one_day_horizon_is_entry_day_open_to_close():
    bars = _bars("2026-09-01", 10, open_=10.0, step=1.0)
    pub = datetime(2026, 9, 1, 8, 0, tzinfo=TSX_TZ)
    out = outcomes.compute_outcome(bars, _bench("2026-09-01", 10), pub, TODAY)
    assert out["ret_1d"] == pytest.approx(10.5 / 10.0 - 1)
    assert out["ret_5d"] == pytest.approx(14.5 / 10.0 - 1)  # 5th session's close


def test_unfinished_horizons_are_none_and_not_complete():
    bars = _bars("2026-09-01", 10)
    pub = datetime(2026, 9, 1, 8, 0, tzinfo=TSX_TZ)
    out = outcomes.compute_outcome(bars, _bench("2026-09-01", 10), pub, TODAY)
    assert out["ret_5d"] is not None
    assert out["ret_20d"] is None and out["ret_60d"] is None
    assert out["complete"] is False


def test_todays_bar_is_never_used():
    """Today's bar may still be filling -- a release published pre-market
    today has no finished entry session yet."""
    bars = _bars("2026-09-14", 5)
    pub = datetime(2026, 9, 18, 8, 0, tzinfo=TSX_TZ)
    assert outcomes.compute_outcome(bars, _bench("2026-09-14", 5), pub, date(2026, 9, 18)) is None


def test_horizon_counts_benchmark_sessions_when_the_stock_skips_a_day():
    """A thin name that didn't trade on the 5th session still exits on
    that session's date -- at its last close before it."""
    skip = {date(2026, 9, 7)}  # 5th session from Tue 1st
    bars = _bars("2026-09-01", 10, open_=10.0, step=1.0, skip=skip)
    pub = datetime(2026, 9, 1, 8, 0, tzinfo=TSX_TZ)
    out = outcomes.compute_outcome(bars, _bench("2026-09-01", 10), pub, TODAY)
    assert out["ret_5d"] == pytest.approx(13.5 / 10.0 - 1)  # Fri 4th's close


# ── symbol candidates / report ───────────────────────────────────────────

def test_symbol_candidates():
    assert outcomes.symbol_candidates("OMI.V") == ["OMI.V"]
    assert outcomes.symbol_candidates("aya") == ["AYA.TO", "AYA.V", "AYA.CN", "AYA"]


def test_dedupe_keeps_one_event_per_symbol_and_day_preferring_the_analysed_one():
    rows = [
        {"guid": "fr", "yahoo_ticker": "X.V", "entry_date": "2026-09-01", "verdict": None},
        {"guid": "en", "yahoo_ticker": "X.V", "entry_date": "2026-09-01", "verdict": "bullish"},
        {"guid": "other", "yahoo_ticker": "X.V", "entry_date": "2026-09-02", "verdict": None},
    ]
    assert sorted(r["guid"] for r in outcomes.dedupe_events(rows)) == ["en", "other"]


def test_summarize_groups_excess_returns():
    rows = [
        {"verdict": "bullish", "ret_1d": 0.10, "bench_ret_1d": 0.01},
        {"verdict": "bullish", "ret_1d": 0.04, "bench_ret_1d": 0.01},
        {"verdict": "bearish", "ret_1d": -0.05, "bench_ret_1d": 0.0},
    ]
    s = outcomes.summarize(rows, "verdict")
    assert s["bullish"][1]["n"] == 2
    assert s["bullish"][1]["mean"] == pytest.approx(0.06)
    assert s["bearish"][1]["beat"] == 0.0
    assert s["bullish"][5]["n"] == 0


def test_format_report_handles_empty_input():
    assert "0 events" in outcomes.format_report([])


# ── service integration ──────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clock():
    set_backtest_clock(datetime(2026, 9, 21, 17, 10, tzinfo=TSX_TZ))  # Monday
    yield
    set_backtest_clock(None)


@pytest.fixture(autouse=True)
def _no_real_email(monkeypatch):
    monkeypatch.setattr(news_watchlist_service, "send_text_email", lambda s, b: True)


def _seed(pr_db_path, guid, ticker, pubdate, verdict=None):
    conn = pr_store.connect(pr_db_path)
    pr_store.mark_seen(conn, FeedItem(guid=guid, feed_url="f", title=guid, link=None,
                                      pubdate=pubdate, description="", categories=[]),
                       first_seen_at="2026-09-01T00:00:00")
    pr_store.save_parsed(conn, guid, {"ticker": ticker, "company": "Co", "category": "financing",
                                      "materiality": "high", "summary": "s"},
                         model="m", parsed_at="2026-09-01T00:00:00")
    if verdict:
        pr_store.save_analysis(conn, guid, ticker, {"verdict": verdict,
                                                    "dilution": {"level": "significant"}},
                               model="m", analyzed_at="2026-09-01T00:00:00")
    conn.close()


def _fetcher(data, calls=None):
    def fetch(symbols, start, end):
        if calls is not None:
            calls.append(list(symbols))
        return {s: data[s] for s in symbols if s in data}
    return fetch


def test_service_scores_every_release_and_resolves_suffix(tmp_path):
    pr_db = tmp_path / "pr.db"
    _seed(pr_db, "g1", "ABC", "Tue, 01 Sep 2026 12:00:00 GMT", verdict="bearish")
    conn = store.connect(tmp_path / "nw.db")
    data = {"ABC.V": _bars("2026-08-20", 22), "XIU.TO": _bench("2026-08-20", 22)}

    news_watchlist_service.run_collector("r", mode="update-prices", conn=conn,
                                         press_release_db_path=pr_db, price_fetcher=lambda t: (None, None, None),
                                         bar_fetcher=_fetcher(data))

    [row] = store.list_scored_outcomes(conn)
    assert row["yahoo_ticker"] == "ABC.V"
    assert row["entry_date"] == "2026-09-01"
    assert row["verdict"] == "bearish" and row["dilution_level"] == "significant"
    assert row["status"] == "pending"  # 60d horizon not in yet
    assert row["ret_5d"] is not None and row["ret_60d"] is None


def test_complete_rows_are_not_refetched(tmp_path):
    set_backtest_clock(datetime(2026, 12, 21, 17, 10, tzinfo=TSX_TZ))
    pr_db = tmp_path / "pr.db"
    _seed(pr_db, "g1", "ABC.V", "Tue, 01 Sep 2026 12:00:00 GMT")
    conn = store.connect(tmp_path / "nw.db")
    data = {"ABC.V": _bars("2026-08-20", 90), "XIU.TO": _bench("2026-08-20", 90)}
    calls = []

    news_watchlist_service.score_release_outcomes("r", conn, pr_db, _fetcher(data, calls))
    assert store.list_scored_outcomes(conn)[0]["status"] == "complete"
    assert news_watchlist_service.score_release_outcomes("r", conn, pr_db, _fetcher(data, calls)) == []
    assert len(calls) == 1


def test_unresolvable_ticker_waits_out_the_grace_period_then_settles(tmp_path):
    pr_db = tmp_path / "pr.db"
    _seed(pr_db, "g1", "NOPE", "Tue, 15 Sep 2026 12:00:00 GMT")
    conn = store.connect(tmp_path / "nw.db")
    data = {"XIU.TO": _bench("2026-08-20", 30)}

    assert news_watchlist_service.score_release_outcomes("r", conn, pr_db, _fetcher(data)) == []
    assert store.final_outcome_guids(conn) == set()

    set_backtest_clock(datetime(2026, 10, 15, 17, 10, tzinfo=TSX_TZ))
    news_watchlist_service.score_release_outcomes("r", conn, pr_db, _fetcher(data))
    assert store.final_outcome_guids(conn) == {"g1"}


def test_missing_benchmark_writes_nothing(tmp_path):
    pr_db = tmp_path / "pr.db"
    _seed(pr_db, "g1", "ABC.V", "Tue, 01 Sep 2026 12:00:00 GMT")
    conn = store.connect(tmp_path / "nw.db")
    data = {"ABC.V": _bars("2026-08-20", 22)}
    assert news_watchlist_service.score_release_outcomes("r", conn, pr_db, _fetcher(data)) == []
    assert store.list_scored_outcomes(conn) == []


def test_dry_run_writes_nothing(tmp_path):
    pr_db = tmp_path / "pr.db"
    _seed(pr_db, "g1", "ABC.V", "Tue, 01 Sep 2026 12:00:00 GMT")
    conn = store.connect(tmp_path / "nw.db")
    data = {"ABC.V": _bars("2026-08-20", 22), "XIU.TO": _bench("2026-08-20", 22)}
    rows = news_watchlist_service.score_release_outcomes("r", conn, pr_db, _fetcher(data), dry_run=True)
    assert len(rows) == 1
    assert store.list_scored_outcomes(conn) == []
