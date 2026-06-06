"""
Layer 3: daily-index scanner.

EDGAR publishes a daily index of every filing. Instead of polling tickers one
at a time, we read the daily index, keep only forms we care about, and keep
only filers on our watchlist (by CIK). This is the spine for "what did big
money / insiders / activists do today".

Form types and why they matter (timeliness in business/calendar days):
  4      insider transactions          ~within 5 days   -- freshest ownership
  3/5    insider init/annual           context
  SC 13D activist >5% stake            ~within 10 days  -- real catalyst
  SC 13G passive >5% stake             periodic         -- big holder crossing
  8-K    material event                ~within 4 days   -- the actual catalysts
  13F-HR institutional holdings (qtr)  ~within 45 days  -- stale; context only

HONEST CEILING: every one of these is a LAGGED DISCLOSURE of a decision already
made. You are following footprints, never live position. Real-time flow is not
public.
"""

from edgar.core import get, load_cik_to_ticker

FORMS_OF_INTEREST = {
    "4": "insider_txn",
    "3": "insider_init",
    "5": "insider_annual",
    "SC 13D": "activist_stake",
    "SC 13D/A": "activist_stake_amend",
    "SC 13G": "passive_stake",
    "SC 13G/A": "passive_stake_amend",
    "8-K": "material_event",
    "13F-HR": "institutional_holdings",
}


def _daily_index_url(year, quarter, yyyymmdd):
    return (f"https://www.sec.gov/Archives/edgar/daily-index/"
            f"{year}/QTR{quarter}/master.{yyyymmdd}.idx")


def fetch_daily_index(d):
    """
    d: datetime.date. Returns list of dicts for the day's filings.
    The master .idx is pipe-delimited: CIK|Name|Form|Date|Filename
    """
    q = (d.month - 1) // 3 + 1
    url = _daily_index_url(d.year, q, d.strftime("%Y%m%d"))
    text = get(url, cache_key=f"idx_{d.strftime('%Y%m%d')}.idx", is_json=False)
    rows = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        cik, name, form, fdate, fname = parts
        rows.append({"cik": int(cik), "name": name, "form": form.strip(),
                     "date": fdate, "filename": fname})
    return rows


def scan_day(d, watchlist_ciks=None, forms=None):
    """
    Filter a day's filings to forms of interest and (optionally) watchlist CIKs.

    watchlist_ciks: set of int CIKs. None = all filers (large; use for testing).
    forms: iterable of form strings. None = FORMS_OF_INTEREST keys.
    Returns list of hits with a 'category' label and resolved ticker if known.
    """
    forms = set(forms) if forms else set(FORMS_OF_INTEREST)
    index = fetch_daily_index(d)
    cik2tic = load_cik_to_ticker()
    hits = []
    for row in index:
        if row["form"] not in forms:
            continue
        if watchlist_ciks is not None and row["cik"] not in watchlist_ciks:
            continue
        hits.append({**row,
                     "category": FORMS_OF_INTEREST.get(row["form"], "other"),
                     "ticker": cik2tic.get(row["cik"]),
                     "url": "https://www.sec.gov/Archives/" + row["filename"]})
    return hits


def scan_range(start, end, watchlist_ciks=None, forms=None):
    """Scan an inclusive date range (skips weekends; EDGAR has no weekend idx)."""
    from datetime import timedelta
    hits, d = [], start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            try:
                hits.extend(scan_day(d, watchlist_ciks, forms))
            except Exception as e:
                hits.append({"date": d.isoformat(), "error": str(e)})
        d += timedelta(days=1)
    return hits
