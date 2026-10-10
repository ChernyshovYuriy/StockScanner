"""Offline tests for demand_signals.darkpool (no network)."""

import demand_signals.darkpool as darkpool

# Representative FINRA ATS weekly-by-venue rows (field names confirmed
# against FINRA's otcMarket/weeklySummary dataset: issueSymbolIdentifier,
# weekStartDate, totalWeeklyShareQuantity, summaryTypeCode). A symbol trades
# across multiple ATS venues in the same week, hence >1 row per week.
FIXTURE_ATS_RECORDS = [
    {"issueSymbolIdentifier": "MU", "weekStartDate": "2026-05-18",
     "totalWeeklyShareQuantity": 100_000, "summaryTypeCode": "ATS_W_SMBL_FIRM"},
    {"issueSymbolIdentifier": "MU", "weekStartDate": "2026-05-18",
     "totalWeeklyShareQuantity": 50_000, "summaryTypeCode": "ATS_W_SMBL_FIRM"},
    {"issueSymbolIdentifier": "MU", "weekStartDate": "2026-05-25",
     "totalWeeklyShareQuantity": 200_000, "summaryTypeCode": "ATS_W_SMBL_FIRM"},
]


# ── _parse_ats_records ──────────────────────────────────────────────────────

def test_parse_ats_records_sums_per_venue_rows_into_weekly_totals():
    parsed = darkpool._parse_ats_records(FIXTURE_ATS_RECORDS)
    assert parsed == [
        {"week_start": "2026-05-18", "shares": 150_000},
        {"week_start": "2026-05-25", "shares": 200_000},
    ]


def test_parse_ats_records_skips_rows_missing_fields():
    records = [{"issueSymbolIdentifier": "MU"}]  # no weekStartDate/shares
    assert darkpool._parse_ats_records(records) == []


def test_parse_ats_records_empty_input():
    assert darkpool._parse_ats_records([]) == []


# ── _get_access_token ────────────────────────────────────────────────────────

def test_get_access_token_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.setattr(darkpool, "FINRA_CLIENT_ID", "")
    monkeypatch.setattr(darkpool, "FINRA_CLIENT_SECRET", "")
    assert darkpool._get_access_token() is None


def test_fetch_weekly_ats_volume_returns_empty_when_unconfigured(monkeypatch):
    monkeypatch.setattr(darkpool, "FINRA_CLIENT_ID", "")
    monkeypatch.setattr(darkpool, "FINRA_CLIENT_SECRET", "")
    assert darkpool.fetch_weekly_ats_volume("MU") == []


# ── _consecutive_rising ──────────────────────────────────────────────────────

def test_consecutive_rising_true_for_strictly_increasing_window():
    ratios = [0.10, 0.12, 0.15, 0.20]
    assert darkpool._consecutive_rising(ratios, idx=3, weeks=3) is True


def test_consecutive_rising_false_when_window_dips():
    ratios = [0.10, 0.20, 0.15, 0.20]
    assert darkpool._consecutive_rising(ratios, idx=3, weeks=3) is False


def test_consecutive_rising_false_when_not_enough_history():
    ratios = [0.10, 0.12]
    assert darkpool._consecutive_rising(ratios, idx=1, weeks=3) is False


# ── build_signals ────────────────────────────────────────────────────────────

def test_build_signals_flags_rising_trend_as_bullish(monkeypatch):
    # Three consecutive weeks, ratio rising each week: 100k/1M, 150k/1M, 250k/1M.
    ats_weekly = [
        {"week_start": "2026-05-04", "shares": 100_000},
        {"week_start": "2026-05-11", "shares": 150_000},
        {"week_start": "2026-05-18", "shares": 250_000},
    ]
    monkeypatch.setattr(darkpool, "_weekly_volumes",
                        lambda us_ticker, weeks: {w: 1_000_000 for w in weeks})
    monkeypatch.setattr(darkpool, "RISING_WEEKS", 3)

    signals = darkpool.build_signals("MU", "MU", ats_weekly, fetched_at="2026-05-20T00:00:00")

    assert len(signals) == 3
    assert signals[-1].signal_type == "darkpool_ratio_rising"
    assert signals[-1].direction == "bullish"
    assert signals[-1].source == "finra_darkpool"
    assert signals[-1].lag_days == 14
    assert signals[0].direction == "neutral"  # not enough history yet at week 1


def test_build_signals_flat_ratio_is_neutral(monkeypatch):
    ats_weekly = [
        {"week_start": "2026-05-04", "shares": 100_000},
        {"week_start": "2026-05-11", "shares": 100_000},
        {"week_start": "2026-05-18", "shares": 100_000},
    ]
    monkeypatch.setattr(darkpool, "_weekly_volumes",
                        lambda us_ticker, weeks: {w: 1_000_000 for w in weeks})
    monkeypatch.setattr(darkpool, "RISING_WEEKS", 3)

    signals = darkpool.build_signals("MU", "MU", ats_weekly, fetched_at="2026-05-20T00:00:00")
    assert all(s.direction == "neutral" for s in signals)
    assert all(s.signal_type == "darkpool_ratio" for s in signals)


def test_build_signals_skips_weeks_with_no_total_volume(monkeypatch):
    ats_weekly = [{"week_start": "2026-05-04", "shares": 100_000}]
    monkeypatch.setattr(darkpool, "_weekly_volumes", lambda us_ticker, weeks: {})

    signals = darkpool.build_signals("MU", "MU", ats_weekly, fetched_at="2026-05-20T00:00:00")
    assert signals == []


def test_build_signals_carries_ticker_and_us_ticker_through(monkeypatch):
    monkeypatch.setattr(darkpool, "_weekly_volumes",
                        lambda us_ticker, weeks: {w: 1_000_000 for w in weeks})
    signals = darkpool.build_signals(
        "SLF.TO", "SLF", [{"week_start": "2026-05-04", "shares": 50_000}],
        fetched_at="2026-05-20T00:00:00",
    )
    assert signals[0].ticker == "SLF.TO"
    assert signals[0].us_ticker == "SLF"


def test_weekly_volumes_sums_each_week_from_one_download(monkeypatch):
    import pandas as pd
    import market_data

    calls = []
    idx = pd.to_datetime(["2026-05-04", "2026-05-05", "2026-05-08", "2026-05-11", "2026-05-15"])
    df = pd.DataFrame({"Volume": [100, 200, 300, 1_000, 2_000]}, index=idx)

    class _Provider:
        def get(self, ticker, as_of, start_dt=None, end_dt=None):
            calls.append((ticker, start_dt, end_dt))
            return df

    monkeypatch.setattr(market_data, "LiveDataProvider", _Provider)

    totals = darkpool._weekly_volumes("MU", ["2026-05-11", "2026-05-04", "2026-05-18"])

    assert calls == [("MU", "2026-05-04", "2026-05-24")]
    assert totals == {"2026-05-04": 600, "2026-05-11": 3_000}  # 05-18 has no bars


def test_weekly_volumes_fetch_failure_is_empty(monkeypatch):
    import market_data

    class _Provider:
        def get(self, *a, **k):
            raise KeyError("no data")

    monkeypatch.setattr(market_data, "LiveDataProvider", _Provider)
    assert darkpool._weekly_volumes("MU", ["2026-05-04"]) == {}
