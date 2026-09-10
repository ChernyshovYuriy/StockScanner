"""
SignalEngine composes the three screens plus a position-awareness input into
one final Signal per ticker. Phase 1: interface only.

Signal semantics -- fully resolved, no remaining ambiguity (all 3x2x2x2 = 24
(trend, pullback, trigger, position) cells are covered by the two rules
below; see the source build spec's own "Signal semantics" section for the
original wording this refines).

`position_open` is a bare boolean with no entry price or stop level in this
interface -- there is no way to model Elder's actual trailing/protective
stop here. Trend alignment is used as its proxy instead, which is why the
two branches below never mix conditions: an OPEN position's fate is decided
by trend alignment alone; a FLAT position's fate is decided by the full
three-screen conjunction alone.

  - Position OPEN (implicitly long -- this build never opens a short; a
    future short-enabled variant would need a directional position type,
    not a bool):
      - trend UP or FLAT  -> HOLD (still aligned, or at least not reversed;
        an indecisive tide is not itself an exit trigger).
      - trend DOWN        -> SELL, unconditionally, regardless of Screens
        2/3 -- the trend flip alone is the exit trigger (Elder's own
        practice: Triple Screen's three-screen conjunction is an ENTRY
        filter only; exits are governed by a stop that tightens the moment
        the weekly tide turns against you, not by waiting for a mirrored,
        fully-confirmed opposite setup). SELL here means "exit long", never
        "open short" -- long-only TSX equity context.
  - Position FLAT (nothing open):
      - trend UP and pullback present and trigger fired -> BUY. All three
        conditions are required; any one missing -> WAIT. This conjunction
        is the ONLY signal path in the whole table that requires more than
        trend + position -- deliberate: entries are gated hard, exits (SELL
        above) react fast, matching the asymmetry in the book.
      - trend DOWN  -> WAIT always, regardless of Screens 2/3. Long-only:
        there is no position to exit and no short entry to open, so even a
        fully-formed bearish setup is not actionable here.
      - trend FLAT  -> WAIT always.
"""
from abc import ABC, abstractmethod

from .screens import EntryScreen, TrendScreen, TriggerScreen
from .types import AlignedBars, SignalResult


class SignalEngine(ABC):
    """Composes a TrendScreen + EntryScreen + TriggerScreen + a position-
    awareness input into the final SignalResult. Takes bars already resolved
    by a DataProvider (or handed directly) -- no implicit data fetching
    here either; a concrete implementation's constructor is where the three
    screen instances are wired in.
    """

    @abstractmethod
    def evaluate(self, bars: AlignedBars, position_open: bool) -> SignalResult:
        """Evaluate one ticker's already-fetched, aligned bars against
        `position_open` (True if this engine's caller currently holds a
        position in this ticker) and return the composed SignalResult --
        the final Signal plus all three screens' individual verdicts, per
        the semantics documented above.
        """
        raise NotImplementedError
