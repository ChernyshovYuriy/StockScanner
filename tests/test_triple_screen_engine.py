"""
Exhaustive truth-table test for SignalEngine -- the core correctness
contract (engine.py's docstring is the authority; this file is its
executable form). All 3 x 2 x 2 x 2 = 24 (trend, pullback, trigger,
position_open) combinations are enumerated explicitly below, no
partial coverage.

Uses FakeTrendScreen/FakeEntryScreen/FakeTriggerScreen (fixed-verdict test
doubles, see triple_screen_fixtures.py) wired into the planned Phase 3
TripleScreenEngine(SignalEngine) -- this decouples the composition logic
under test from any real indicator math (that's test_triple_screen_screens.
py's job). The bars passed through are irrelevant to a fake screen's
verdict, so a single reusable AlignedBars fixture suffices for every case.

TripleScreenEngine's planned constructor:
    TripleScreenEngine(trend_screen: TrendScreen, entry_screen: EntryScreen,
                        trigger_screen: TriggerScreen)
"""
import pytest

from research.triple_screen.reference_impl import TripleScreenEngine
from research.triple_screen.types import Direction, Signal

from triple_screen_fixtures import (
    FakeEntryScreen, FakeTrendScreen, FakeTriggerScreen, aligned, flat_bars,
)

# Bars content doesn't matter here -- fakes ignore their input entirely.
_BARS = aligned(trend=flat_bars(timeframe="weekly"), entry=flat_bars(timeframe="daily"))

# (trend, pullback_present, trigger_fired, position_open) -> expected Signal
# See engine.py's docstring for the two rules this table is the resolved
# form of: an OPEN position's fate depends on trend alignment alone; a FLAT
# position's fate depends on the full three-screen conjunction alone.
TRUTH_TABLE = [
    # trend=UP, position=open -> HOLD unconditionally
    (Direction.UP, True, True, True, Signal.HOLD),
    (Direction.UP, True, False, True, Signal.HOLD),
    (Direction.UP, False, True, True, Signal.HOLD),
    (Direction.UP, False, False, True, Signal.HOLD),
    # trend=UP, position=flat -> BUY only on the full conjunction
    (Direction.UP, True, True, False, Signal.BUY),
    (Direction.UP, True, False, False, Signal.WAIT),
    (Direction.UP, False, True, False, Signal.WAIT),
    (Direction.UP, False, False, False, Signal.WAIT),
    # trend=DOWN, position=open -> SELL unconditionally (trend-flip exit)
    (Direction.DOWN, True, True, True, Signal.SELL),
    (Direction.DOWN, True, False, True, Signal.SELL),
    (Direction.DOWN, False, True, True, Signal.SELL),
    (Direction.DOWN, False, False, True, Signal.SELL),
    # trend=DOWN, position=flat -> WAIT always (long-only: nothing to do)
    (Direction.DOWN, True, True, False, Signal.WAIT),
    (Direction.DOWN, True, False, False, Signal.WAIT),
    (Direction.DOWN, False, True, False, Signal.WAIT),
    (Direction.DOWN, False, False, False, Signal.WAIT),
    # trend=FLAT, position=open -> HOLD unconditionally
    (Direction.FLAT, True, True, True, Signal.HOLD),
    (Direction.FLAT, True, False, True, Signal.HOLD),
    (Direction.FLAT, False, True, True, Signal.HOLD),
    (Direction.FLAT, False, False, True, Signal.HOLD),
    # trend=FLAT, position=flat -> WAIT always
    (Direction.FLAT, True, True, False, Signal.WAIT),
    (Direction.FLAT, True, False, False, Signal.WAIT),
    (Direction.FLAT, False, True, False, Signal.WAIT),
    (Direction.FLAT, False, False, False, Signal.WAIT),
]

assert len(TRUTH_TABLE) == 24, "exhaustive means all 3x2x2x2 cells, not a sample"


@pytest.mark.parametrize("trend,pullback,trigger,position_open,expected", TRUTH_TABLE)
def test_truth_table_cell(trend, pullback, trigger, position_open, expected):
    engine = TripleScreenEngine(
        trend_screen=FakeTrendScreen(trend),
        entry_screen=FakeEntryScreen(pullback),
        trigger_screen=FakeTriggerScreen(trigger),
    )
    result = engine.evaluate(_BARS, position_open=position_open)
    assert result.signal == expected, (
        f"trend={trend} pullback={pullback} trigger={trigger} "
        f"position_open={position_open}: expected {expected}, got {result.signal}"
    )


def test_result_carries_all_three_intermediate_verdicts_for_audit():
    engine = TripleScreenEngine(
        trend_screen=FakeTrendScreen(Direction.UP),
        entry_screen=FakeEntryScreen(True),
        trigger_screen=FakeTriggerScreen(True),
    )
    result = engine.evaluate(_BARS, position_open=False)
    assert result.signal == Signal.BUY
    assert result.trend.direction == Direction.UP
    assert result.entry.pullback_present is True
    assert result.trigger.triggered is True
