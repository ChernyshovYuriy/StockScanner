"""Offline tests for demand_signals.short_volume (no network)."""

from datetime import date

import demand_signals.short_volume as short_volume

FIXTURE_FILE_TEXT = (
    "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    "20260901|A|484453.041438|3132|810522.796681|B,Q,N\n"
    "20260901|MU|100000|0|1000000|B,Q,N\n"
    "20260901|AA|636645.790693|1945|1469237.467092|B,Q,N\n"
)


# ── _parse_symbol_row ────────────────────────────────────────────────────────

def test_parse_symbol_row_finds_matching_symbol():
    assert short_volume._parse_symbol_row(FIXTURE_FILE_TEXT, "MU") == (100000.0, 1000000.0)


def test_parse_symbol_row_missing_symbol_returns_none():
    assert short_volume._parse_symbol_row(FIXTURE_FILE_TEXT, "ZZZZ") is None


def test_parse_symbol_row_skips_header():
    # A symbol literally named "Symbol" (the header row) must never match.
    assert short_volume._parse_symbol_row(FIXTURE_FILE_TEXT, "Symbol") is None


# ── _fetch_daily_file ─────────────────────────────────────────────────────────

def test_fetch_daily_file_skips_weekends_without_a_request(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(short_volume._client, "request",
                         lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    saturday = date(2026, 8, 29)
    assert short_volume._fetch_daily_file(saturday) is None
    assert called["n"] == 0


def test_fetch_daily_file_returns_none_on_request_failure(monkeypatch):
    import requests

    def _raise(*a, **k):
        raise requests.RequestException("404")

    monkeypatch.setattr(short_volume._client, "request", _raise)
    monday = date(2026, 8, 31)
    assert short_volume._fetch_daily_file(monday) is None


# ── fetch_daily_short_volume ─────────────────────────────────────────────────

def test_fetch_daily_short_volume_builds_ascending_ratio_series(monkeypatch):
    # Three business days of fixture data, most recent first as returned by
    # the calendar walk-back, oldest-first once returned to the caller.
    files = {
        "2026-09-01": "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
                      "20260901|MU|300000|0|1000000|B,Q,N\n",
        "2026-08-31": "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
                      "20260831|MU|200000|0|1000000|B,Q,N\n",
        "2026-08-28": "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
                      "20260828|MU|100000|0|1000000|B,Q,N\n",
    }
    monkeypatch.setattr(short_volume, "_fetch_daily_file",
                         lambda d: files.get(d.isoformat()))
    monkeypatch.setattr(short_volume, "date", _FrozenDate)

    rows = short_volume.fetch_daily_short_volume("MU", days=5)

    assert [r["date"] for r in rows] == ["2026-08-28", "2026-08-31", "2026-09-01"]
    assert [r["ratio"] for r in rows] == [0.1, 0.2, 0.3]


def test_fetch_daily_short_volume_skips_days_with_no_total_volume(monkeypatch):
    monkeypatch.setattr(short_volume, "_fetch_daily_file",
                         lambda d: "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
                                   f"{d.strftime('%Y%m%d')}|MU|0|0|0|B\n")
    monkeypatch.setattr(short_volume, "date", _FrozenDate)

    rows = short_volume.fetch_daily_short_volume("MU", days=3)
    assert rows == []


class _FrozenDate(date):
    """A date subclass whose .today() is pinned, so the calendar walk-back
    in fetch_daily_short_volume() is deterministic under test."""

    @classmethod
    def today(cls):
        return date(2026, 9, 2)  # Wednesday


# ── _consecutive_trend ────────────────────────────────────────────────────────

def test_consecutive_trend_true_for_strictly_falling_window():
    ratios = [0.30, 0.25, 0.20, 0.15]
    assert short_volume._consecutive_trend(ratios, idx=3, days=3, rising=False) is True


def test_consecutive_trend_true_for_strictly_rising_window():
    ratios = [0.10, 0.15, 0.20, 0.25]
    assert short_volume._consecutive_trend(ratios, idx=3, days=3, rising=True) is True


def test_consecutive_trend_false_when_window_reverses():
    ratios = [0.10, 0.20, 0.15, 0.20]
    assert short_volume._consecutive_trend(ratios, idx=3, days=3, rising=True) is False


def test_consecutive_trend_false_when_not_enough_history():
    ratios = [0.10, 0.08]
    assert short_volume._consecutive_trend(ratios, idx=1, days=3, rising=False) is False


# ── build_signals ─────────────────────────────────────────────────────────────

def test_build_signals_flags_falling_trend_as_covering_bullish(monkeypatch):
    monkeypatch.setattr(short_volume, "TREND_DAYS", 3)
    rows = [
        {"date": "2026-08-28", "short_shares": 300_000, "total_shares": 1_000_000, "ratio": 0.30},
        {"date": "2026-08-31", "short_shares": 200_000, "total_shares": 1_000_000, "ratio": 0.20},
        {"date": "2026-09-01", "short_shares": 100_000, "total_shares": 1_000_000, "ratio": 0.10},
    ]
    signals = short_volume.build_signals("MU", "MU", rows, fetched_at="2026-09-02T00:00:00")

    assert len(signals) == 3
    assert signals[-1].signal_type == "short_volume_covering"
    assert signals[-1].direction == "bullish"
    assert signals[-1].source == "finra_short_volume"
    assert signals[-1].lag_days == 1
    assert signals[0].direction == "neutral"  # not enough history yet on day 1


def test_build_signals_flags_rising_trend_as_pressure_bearish(monkeypatch):
    monkeypatch.setattr(short_volume, "TREND_DAYS", 3)
    rows = [
        {"date": "2026-08-28", "short_shares": 100_000, "total_shares": 1_000_000, "ratio": 0.10},
        {"date": "2026-08-31", "short_shares": 200_000, "total_shares": 1_000_000, "ratio": 0.20},
        {"date": "2026-09-01", "short_shares": 300_000, "total_shares": 1_000_000, "ratio": 0.30},
    ]
    signals = short_volume.build_signals("MU", "MU", rows, fetched_at="2026-09-02T00:00:00")

    assert signals[-1].signal_type == "short_volume_pressure"
    assert signals[-1].direction == "bearish"


def test_build_signals_flat_ratio_is_neutral(monkeypatch):
    monkeypatch.setattr(short_volume, "TREND_DAYS", 3)
    rows = [
        {"date": "2026-08-28", "short_shares": 100_000, "total_shares": 1_000_000, "ratio": 0.10},
        {"date": "2026-08-31", "short_shares": 100_000, "total_shares": 1_000_000, "ratio": 0.10},
        {"date": "2026-09-01", "short_shares": 100_000, "total_shares": 1_000_000, "ratio": 0.10},
    ]
    signals = short_volume.build_signals("MU", "MU", rows, fetched_at="2026-09-02T00:00:00")
    assert all(s.direction == "neutral" for s in signals)
    assert all(s.signal_type == "short_volume_ratio" for s in signals)


def test_build_signals_carries_ticker_and_us_ticker_through():
    rows = [{"date": "2026-08-28", "short_shares": 50_000, "total_shares": 1_000_000, "ratio": 0.05}]
    signals = short_volume.build_signals(
        "SLF.TO", "SLF", rows, fetched_at="2026-09-02T00:00:00",
    )
    assert signals[0].ticker == "SLF.TO"
    assert signals[0].us_ticker == "SLF"
