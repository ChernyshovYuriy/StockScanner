"""Offline tests for the normalized DemandSignal record (no network)."""

import pytest

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


def test_valid_signal_constructs():
    s = _signal()
    assert s.ticker == "MU" and s.strength == 0.5


@pytest.mark.parametrize("bad_source", ["insider", "finra", "", None])
def test_invalid_source_rejected(bad_source):
    with pytest.raises(ValueError):
        _signal(source=bad_source)


@pytest.mark.parametrize("bad_direction", ["up", "down", "", None])
def test_invalid_direction_rejected(bad_direction):
    with pytest.raises(ValueError):
        _signal(direction=bad_direction)


@pytest.mark.parametrize("bad_strength", [-0.01, 1.01, 2.0, -5.0])
def test_strength_out_of_range_rejected(bad_strength):
    with pytest.raises(ValueError):
        _signal(strength=bad_strength)


def test_strength_boundaries_are_valid():
    assert _signal(strength=0.0).strength == 0.0
    assert _signal(strength=1.0).strength == 1.0


def test_negative_lag_days_rejected():
    with pytest.raises(ValueError):
        _signal(lag_days=-1)


def test_to_row_serializes_detail_to_json_text():
    row = _signal(detail={"owner": "DOE JANE", "shares": 1000}).to_row()
    assert isinstance(row["detail"], str)
    assert "DOE JANE" in row["detail"]


def test_from_row_round_trips_to_row():
    original = _signal(detail={"owner": "DOE JANE", "shares": 1000.0})
    restored = DemandSignal.from_row(original.to_row())
    assert restored == original


def test_from_row_handles_empty_detail_text():
    row = _signal().to_row()
    row["detail"] = ""
    restored = DemandSignal.from_row(row)
    assert restored.detail == {}
