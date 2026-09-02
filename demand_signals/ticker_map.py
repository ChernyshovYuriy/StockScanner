"""
Canadian symbol -> US symbol mapping, for interlisted names only.

finra_darkpool.py and options_flow.py are US-market sources; a TSX-listed
ticker only gets a signal from them if it also trades on a US exchange
under a (usually different) symbol. Free/automated cross-listing lookups
are unreliable, so this is a small hand-curated table -- same convention
this repo already uses for other short, rarely-changing lists (config.py's
EDGAR_FORMS, the EDGAR watchlist) rather than fragile inference.

Extend CAN_TO_US as interlisted names come up in practice; there's no need
to pre-populate the whole TSX.

EXTENSION POINT -- Canadian-only names (no US line, e.g. most TSX-V/CSE
juniors) get no signal here at all. get_us_ticker() returning None IS that
gap, not a bug: a future SEDI (Canadian insider filings) source is the
domestic-only counterpart to edgar_insider and would plug in as a 4th
`source` value in schema.py, keyed on the CAN ticker directly (no mapping
needed, since SEDI already speaks in CAN symbols/issuers).
"""

# A starter set of common TSX/US interlistings. CAN symbol (as the screener
# already uses it, e.g. "SLF.TO") -> its US-listed symbol.
CAN_TO_US = {
    "SLF.TO": "SLF",
    "TD.TO": "TD",
    "RY.TO": "RY",
    "BMO.TO": "BMO",
    "BNS.TO": "BNS",
    "CM.TO": "CM",
    "SU.TO": "SU",
    "CNQ.TO": "CNQ",
    "ENB.TO": "ENB",
    "TRP.TO": "TRP",
    "BAM.TO": "BAM",
    "MFC.TO": "MFC",
    "GIB-A.TO": "GIB",
    "CP.TO": "CP",
    "CNI.TO": "CNI",  # CNR.TO on TSX; CNI on NYSE
    "ABX.TO": "GOLD",  # Barrick
    "AEM.TO": "AEM",
    "WPM.TO": "WPM",
    "FNV.TO": "FNV",
    "SHOP.TO": "SHOP",
}


def get_us_ticker(ticker: str) -> str | None:
    """The US-listed symbol for `ticker`, or None if there isn't one (a
    US ticker maps to itself; a Canadian-only name maps to None -- see the
    module docstring's SEDI extension point for that gap)."""
    if not ticker.endswith(".TO") and not ticker.endswith(".V") and not ticker.endswith(".CN"):
        return ticker  # already a US-style symbol, nothing to map
    return CAN_TO_US.get(ticker)


def is_us_covered(ticker: str) -> bool:
    """True if `ticker` resolves to a US line (finra_darkpool/options_flow
    can cover it); False for a Canadian-only name."""
    return get_us_ticker(ticker) is not None
