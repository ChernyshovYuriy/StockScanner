"""
news_watchlist
================
The 11th StockScanner service: a follow-through tracker for
press_release_tracker's catches. No capital, no positions, no auto-buy --
same "pure collector" shape as press_release_tracker/scanner_board, not a
paper-trading sleeve.

Grew out of the user's own observation: press_release_tracker already
catches real catalysts fast, but most of them are penny names whose
day-to-month follow-through gets forgotten within a week of the triggering
email. The fix isn't to automate the trade decision -- it's to stop losing
the candidate itself. So this package automates the bookkeeping only:

  inbox      -- auto-seeded daily from press_release_tracker's own parsed
                results (data/press_releases.db, read read-only here --
                this package never writes back to it, same isolation
                precedent as demand_signals/edgar_adapter.py reading
                edgar.db). Every parsed item with a non-null ticker not
                already in this watchlist lands here, stamped with that
                day's price.
  watching   -- an item explicitly confirmed off the inbox. ONLY watching
                items get a daily price-history row appended -- price
                tracking is gated on a human decision, not automatic for
                everything the feed produces.
  dismissed  -- explicitly rejected (from either state above). Kept, not
                deleted, so the record survives.

The triage step (is this one actually worth following) stays a manual,
one-click action -- the same judgment call the user already makes today,
just against a persistent dashboard list instead of a scrolling email
inbox. Own SQLite DB (data/news_watchlist.db), unrelated schema to every
other data/*.db file.
"""
