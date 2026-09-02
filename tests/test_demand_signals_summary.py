"""Offline tests for demand_signals.summary (no network, no DB)."""

from demand_signals.summary import summarize_all, summarize_ticker


def _row(source, signal_type, direction, date="2026-08-10"):
    return {"source": source, "signal_type": signal_type, "direction": direction, "date": date}


def test_no_rows_is_no_data():
    summary = summarize_ticker([])
    assert summary["label"] == "no data"


def test_plain_bullish_skew_alone_is_mild_bullish():
    rows = [_row("options_flow", "call_put_skew", "bullish", date="2026-09-02")]
    summary = summarize_ticker(rows)
    assert summary["label"] == "mild bullish"
    assert "no unusual volume, no dark-pool trend, no insider buys" in summary["reasons"]


def test_plain_bearish_skew_alone_is_mild_bearish():
    rows = [_row("options_flow", "call_put_skew", "bearish", date="2026-09-02")]
    summary = summarize_ticker(rows)
    assert summary["label"] == "mild bearish"


def test_neutral_skew_alone_is_neutral():
    rows = [_row("options_flow", "call_put_skew", "neutral", date="2026-09-02")]
    summary = summarize_ticker(rows)
    assert summary["label"] == "neutral"


def test_darkpool_rising_plus_bullish_skew_is_bullish_with_month_in_reason():
    rows = [
        _row("finra_darkpool", "darkpool_ratio_rising", "bullish", date="2026-07-20"),
        _row("options_flow", "call_put_skew", "bullish", date="2026-09-02"),
    ]
    summary = summarize_ticker(rows)
    assert summary["label"] == "bullish"
    assert any("July" in r for r in summary["reasons"])


def test_unusual_volume_both_sides_combines_into_one_reason():
    rows = [
        _row("options_flow", "unusual_call_volume", "bullish", date="2026-09-02"),
        _row("options_flow", "unusual_put_volume", "bearish", date="2026-09-02"),
        _row("options_flow", "call_put_skew", "bullish", date="2026-09-02"),
    ]
    summary = summarize_ticker(rows)
    assert summary["label"] == "bullish"  # skew breaks the call/put tie
    assert "unusual volume on both calls and puts today" in summary["reasons"]
    # each side isn't ALSO reported separately
    assert not any(r == "unusual call volume today" for r in summary["reasons"])
    assert not any(r == "unusual put volume today" for r in summary["reasons"])


def test_unusual_call_alone_is_bullish():
    rows = [_row("options_flow", "unusual_call_volume", "bullish", date="2026-09-02")]
    summary = summarize_ticker(rows)
    assert summary["label"] == "bullish"


def test_unusual_put_alone_is_bearish():
    rows = [_row("options_flow", "unusual_put_volume", "bearish", date="2026-09-02")]
    summary = summarize_ticker(rows)
    assert summary["label"] == "bearish"


def test_insider_buy_alone_is_bullish_and_counted():
    rows = [
        _row("edgar_insider", "insider_buy", "bullish", date="2026-08-01"),
        _row("edgar_insider", "insider_buy", "bullish", date="2026-08-15"),
    ]
    summary = summarize_ticker(rows)
    assert summary["label"] == "bullish"
    assert "2 insider buys on record" in summary["reasons"]


def test_conflicting_elevated_signals_are_mixed_not_neutral():
    # rising dark pool (+1) directly offset by a bearish skew (-1); the
    # ticker DOES have elevated evidence, so "mixed" (not the plain
    # "neutral" reserved for no elevated evidence at all).
    rows = [
        _row("finra_darkpool", "darkpool_ratio_rising", "bullish", date="2026-07-20"),
        _row("options_flow", "call_put_skew", "bearish", date="2026-09-02"),
    ]
    summary = summarize_ticker(rows)
    assert summary["label"] == "mixed"


def test_summarize_all_wraps_each_ticker():
    by_ticker = {
        "AAPL": [_row("options_flow", "call_put_skew", "bullish", date="2026-09-02")],
        "MU": [_row("finra_darkpool", "darkpool_ratio_rising", "bullish", date="2026-07-27")],
    }
    result = summarize_all(by_ticker)
    assert result["AAPL"]["label"] == "mild bullish"
    assert result["MU"]["label"] == "bullish"
