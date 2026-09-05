"""
conviction_watchlist -- standalone personal tool for the user's real RBC
account (Margin + TFSA), NOT one of this repo's paper-trading sleeves. Never
touches db.py, root config.py, or any of the data/*.db files.

Grew out of a 2026-09-05 research session that tested a large family of
price-only mean-reversion "buy dip, sell rip" rules on AAPL/NVDA/AMD and
found every variant either lost to plain buy-and-hold or only "won" via
in-sample overfitting that failed walk-forward validation (see the
dip-bounce-grid-rejected-2026-09 memory entry, and the standalone
dip_grid_backtest.py script that produced it). The user's actual practice is
"buy-and-hold, sized for conviction" -- this package makes that practice
systematic instead of manual (premarket/news-watching), without automating
any actual buy/sell decision:

  - quality_filter.py         : filters the CAN ticker universe (fetched
                                 from config.TICKERS_URL) down to large-cap,
                                 profitable, main-board (.TO) names -- a
                                 fixed quality bar, no sector exclusion (see
                                 that file's docstring for why)
  - entry_screener.py         : flags a quality ticker as a buy candidate
                                 once it's >= config.DIP_PCT_OFF_HIGH off its
                                 52-week high
  - trailing_stop_monitor.py  : flags a SELL once a held position has
                                 fallen config.TRAILING_STOP_PCT from its
                                 peak since entry -- no profit-target exit,
                                 by design

Holdings are tracked in config.HOLDINGS_FILE (data/conviction_holdings.json),
updated by hand as the user actually trades in the real account -- nothing
in this package places, simulates, or records a trade itself.
"""
