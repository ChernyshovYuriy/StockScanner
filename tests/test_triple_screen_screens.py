"""
Isolated screen tests against the planned Phase 3 reference implementations
(research.triple_screen.reference_impl) -- pin down the concrete indicator
choice for each screen, since the interfaces (Phase 1) deliberately leave
"which indicator" open ("a documented, swappable parameter"). These tests
ARE that documentation, made executable.

Phase 3 must supply:
  - EMASlopeTrendScreen(TrendScreen): slope of a 13-period EMA (ema_period,
    lag constructor params, defaults 13/3 -- mirrors research/elder_ray.py's
    own trend_at() convention for consistency across the research package,
    reimplemented independently rather than imported, since the two tools
    are otherwise unrelated). indicator_values must include "ema" (float).
  - ForceIndexEntryScreen(EntryScreen): 2-period-EMA-smoothed Force Index
    (Volume x delta-Close). Pullback/rally present iff the latest smoothed
    value's sign opposes `trend` (negative in an UP trend, positive in a
    DOWN trend).
  - PriorBarTriggerScreen(TriggerScreen): today's High > yesterday's High
    (UP direction) or today's Low < yesterday's Low (DOWN direction).

These imports will fail until Phase 3 creates reference_impl.py -- expected;
that module's whole job is to make this file pass.
"""
import pytest

from research.triple_screen.reference_impl import (
    EMASlopeTrendScreen, ForceIndexEntryScreen, PriorBarTriggerScreen,
)
from research.triple_screen.types import Direction

from triple_screen_fixtures import (
    breakdown_below_prior_low_bars, breakout_above_prior_high_bars, downtrend_bars, flat_bars,
    no_breakdown_below_prior_low_bars, no_breakout_above_prior_high_bars,
    no_pullback_in_uptrend_bars, no_rally_in_downtrend_bars, pullback_in_uptrend_bars,
    rally_in_downtrend_bars, uptrend_bars,
)


# ── TrendScreen (Screen 1) ────────────────────────────────────────────────────

def test_trend_screen_up_fixture_returns_up():
    result = EMASlopeTrendScreen().evaluate(uptrend_bars())
    assert result.direction == Direction.UP
    assert "ema" in result.indicator_values


def test_trend_screen_down_fixture_returns_down():
    result = EMASlopeTrendScreen().evaluate(downtrend_bars())
    assert result.direction == Direction.DOWN
    assert "ema" in result.indicator_values


def test_trend_screen_flat_fixture_returns_flat():
    result = EMASlopeTrendScreen().evaluate(flat_bars())
    assert result.direction == Direction.FLAT
    assert "ema" in result.indicator_values


# ── EntryScreen (Screen 2) ────────────────────────────────────────────────────

def test_entry_screen_pullback_present_in_uptrend():
    result = ForceIndexEntryScreen().evaluate(pullback_in_uptrend_bars(), Direction.UP)
    assert result.pullback_present is True
    assert result.indicator_values


def test_entry_screen_pullback_absent_in_uptrend():
    result = ForceIndexEntryScreen().evaluate(no_pullback_in_uptrend_bars(), Direction.UP)
    assert result.pullback_present is False


def test_entry_screen_rally_present_in_downtrend():
    result = ForceIndexEntryScreen().evaluate(rally_in_downtrend_bars(), Direction.DOWN)
    assert result.pullback_present is True


def test_entry_screen_rally_absent_in_downtrend():
    result = ForceIndexEntryScreen().evaluate(no_rally_in_downtrend_bars(), Direction.DOWN)
    assert result.pullback_present is False


# ── TriggerScreen (Screen 3) ──────────────────────────────────────────────────

def test_trigger_screen_fires_on_breakout_above_prior_high():
    result = PriorBarTriggerScreen().evaluate(breakout_above_prior_high_bars(), Direction.UP)
    assert result.triggered is True
    assert result.indicator_values


def test_trigger_screen_does_not_fire_without_breakout():
    result = PriorBarTriggerScreen().evaluate(no_breakout_above_prior_high_bars(), Direction.UP)
    assert result.triggered is False


def test_trigger_screen_fires_on_breakdown_below_prior_low():
    result = PriorBarTriggerScreen().evaluate(breakdown_below_prior_low_bars(), Direction.DOWN)
    assert result.triggered is True


def test_trigger_screen_does_not_fire_without_breakdown():
    result = PriorBarTriggerScreen().evaluate(no_breakdown_below_prior_low_bars(), Direction.DOWN)
    assert result.triggered is False
