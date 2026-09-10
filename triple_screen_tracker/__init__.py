"""
triple_screen_tracker — a 7th, fully isolated StockScanner service: a daily
paper-tracking experiment built on research/triple_screen.

Not a paper-trading sleeve like momentum/macro (no cash, no share sizing) --
purpose-built to answer one research question: "after a Triple Screen BUY
signal, how does price actually behave until it first closes below entry?"
Each trading day, the universe (CAN_TICKERS_URL) is scanned for fresh BUY
signals; every one not already tracked gets "bought" at that day's close and
recorded. Every already-tracked (OPEN) ticker gets that day's close appended
to its price history, and is "sold" (closed, kept as history) the first day
its close drops below the recorded buy price -- a zero-tolerance rule, not a
simulated stop, so the dataset shows the true first-touch-below-entry point
for later review of where the *real* exit point should be.

Own SQLite DB (data/triple_screen_tracker.db, see store.py) -- kept separate
from trading.db/momentum.db/macro.db/edgar.db/demand_signals.db, matching
this repo's "services stay independent" precedent (see CLAUDE.md).
"""
