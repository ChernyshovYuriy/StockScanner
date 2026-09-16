"""
press_release_tracker/
=======================
News-wire aggregator/parser for the press-release tracker service (see
press_release_service.py) -- the 10th StockScanner service. Same
isolation shape as edgar/ and demand_signals/: own SQLite store, no
capital or positions, structurally independent of every trading sleeve.

feeds.py    -- mechanical RSS fetch + parse (no interpretation).
llm_parser.py -- turns one item's title/description into structured
                 fields (ticker, company, category, materiality, summary)
                 via the OpenAI API; degrades gracefully when unconfigured.
store.py    -- SQLite persistence (seen_items dedupe + parsed_releases).
digest.py   -- plain-text email digest of newly-seen items.
"""
