"""Offline tests for demand_signals.ticker_map (no network)."""

from demand_signals.ticker_map import get_us_ticker, is_us_covered


def test_known_interlisted_can_ticker_resolves():
    assert get_us_ticker("SLF.TO") == "SLF"
    assert get_us_ticker("TD.TO") == "TD"


def test_us_ticker_maps_to_itself():
    assert get_us_ticker("AAPL") == "AAPL"
    assert get_us_ticker("MU") == "MU"


def test_can_only_ticker_not_in_map_returns_none():
    """A TSX-V/CSE junior with no US line -- the SEDI extension-point gap."""
    assert get_us_ticker("SOME-JUNIOR.V") is None
    assert get_us_ticker("UNKNOWN-NAME.TO") is None


def test_is_us_covered_matches_get_us_ticker():
    assert is_us_covered("SLF.TO") is True
    assert is_us_covered("UNKNOWN-NAME.TO") is False
    assert is_us_covered("AAPL") is True
