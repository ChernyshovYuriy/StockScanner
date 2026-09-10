"""
Elder's Triple Screen system for swing trading (2-10 day holds).

Timeframe mapping for this build: Screen 1 (Trend) = weekly, Screens 2/3
(Entry pullback / Trigger) = daily -- configurable via TimeframeConfig
(see types.py), not hardcoded, so the same engine can later run a different
mapping (e.g. daily trend / hourly entry for an intraday variant) without
changing any screen or engine code.

Working order (enforced across commits, per the source build spec): Phase 1
is interfaces only -- no logic, every abstract method body raises
NotImplementedError. Phase 2 (tests against these interfaces, using a fake
DataProvider, no network) and Phase 3 (implementation) are deliberately not
started yet; each phase stops for review before the next begins.

Phase 1 (this commit) -- see:
  types.py          PriceData, AlignedBars, TimeframeConfig, Direction,
                     Signal, and each screen's verdict type.
  data_provider.py   DataProvider interface.
  screens.py         TrendScreen / EntryScreen / TriggerScreen interfaces.
  engine.py          SignalEngine interface + the exact BUY/SELL/HOLD/WAIT
                     semantics it must encode.
"""
