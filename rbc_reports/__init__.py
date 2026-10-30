"""
rbc_reports/
============
RBC research tracker -- organizes the user's daily "RBC Insight Alert"
digest emails (subscription: RBC Capital Markets - All Reports) so the
reports can be searched, filtered per ticker and, in later steps, tagged
and matched against prices. Built for the user's REAL account decisions
(same spirit as conviction_watchlist/): nothing here places, simulates or
records a trade.

Step 1 (this package today): the user saves each digest email's HTML into
config.RBC_REPORTS_DIR; ingest.py parses every report item out of it
(parser.py) into data/rbc_reports.db (store.py); the dashboard's /rbc tab
(rbc_dashboard_data.py) shows the feed, a per-ticker timeline and the
most-covered tickers.

A digest only carries the first 100 reports of the day ("100 of 238
Reports" in its footer) -- the rest need an RBC Insight login. Each
digest's shown/total counts are stored so the dashboard can say how much
was missed.
"""
