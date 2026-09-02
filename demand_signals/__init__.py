"""
demand_signals — a 5th, structurally independent StockScanner service: "real
buyer demand" tracker. Normalizes four sources into one schema, screenable
per ticker (see demand_signals/schema.py):

  edgar_insider      — insider open-market/private purchases (edgar_adapter.py
                       reads them from edgar.db; this package doesn't re-fetch
                       or duplicate edgar/'s own SEC EDGAR logic)
  finra_darkpool     — FINRA ATS weekly off-exchange volume (darkpool.py)
  finra_short_volume — FINRA's daily consolidated short-sale-volume file, a
                       falling/rising short-volume ratio read as short
                       sellers stepping back or piling in (short_volume.py)
  options_flow       — an unusual-activity proxy from a pluggable options-chain
                       provider, one free implementation to start (options_flow.py)

US-market sources; see ticker_map.py for the CAN-interlisting gap and its
SEDI extension point.

HONEST CEILING — data limitations, read before trusting a signal:

  FINRA ATS: published WEEKLY, itself lagged ~2 weeks behind the week it
  covers. This is a confirmation signal, never a live trigger -- the same
  "every filing is a lagged disclosure" framing edgar/ already uses for SEC
  filings, just with a longer lag.

  FINRA daily short-sale volume: published the next business day (T+1) --
  no OAuth needed either, unlike the ATS weekly summary's Query API. Still
  short-sale *volume*, not signed buy/sell flow -- the covering/pressure
  read is inferred from the trend, not observed directly.

  Options flow (free provider): a SNAPSHOT of the current option chain
  (volume/open-interest/bid-ask at request time), not a trade-by-trade tape.
  No sweep or block-trade detection -- that needs a paid feed (e.g. Unusual
  Whales), which is exactly why this is a pluggable Provider interface
  (options_flow.py) rather than one hardcoded implementation. The free
  endpoint (Yahoo, via yfinance) is also unofficial/undocumented: no SLA,
  can change or break without notice, unlike SEC/FINRA's documented APIs.

  Coverage gap: a Canadian-only name (no US interlisting) gets NO signal
  from finra_darkpool or options_flow until a SEDI source exists (see
  ticker_map.py). Callers must treat "no demand_signals row" as "not
  covered", never as "confirmed quiet" -- there is no neutral/all-clear
  reading from an uncovered ticker.
"""
