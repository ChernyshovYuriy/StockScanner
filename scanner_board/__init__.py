"""
scanner_board — the Ticker Indicator Board.

An isolated, read-only research service: one row per ticker, many columns,
each column a raw indicator reading or a named, book-cited rule from
Dr. Alexander Elder's *The New Trading for a Living*. See PLAN.md for the
full design and phase breakdown. No capital, no positions, no buy/sell —
this package never touches db.py, config.py's trading parameters, or any
data/*.db used by an actual sleeve.
"""
