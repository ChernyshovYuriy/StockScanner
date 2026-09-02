"""Offline tests for the 13D/G body parser, storage, and digest enrichment."""

from edgar.activist import accession_from_url, parse_activist
from edgar.digest import build_digest
from edgar import store

# Representative full-submission SC 13D text (SEC-HEADER + cover page).
FIXTURE_13D = """<SEC-HEADER>0001104659-26-012345.hdr.sgml : 20260605
ACCESSION NUMBER:\t\t0001104659-26-012345
CONFORMED SUBMISSION TYPE:\tSC 13D
FILED AS OF DATE:\t\t20260605

SUBJECT COMPANY:

\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\t\tTARGET MINING CORP
\t\tCENTRAL INDEX KEY:\t\t\t0000111111

FILED BY:

\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\t\tACTIVIST CAPITAL LP
\t\tCENTRAL INDEX KEY:\t\t\t0000222222
</SEC-HEADER>
<DOCUMENT>
<TYPE>SC 13D
CUSIP No. 12345678
(11) Aggregate Amount Beneficially Owned by Each Reporting Person: 4,500,000
(13) Percent of Class Represented by Amount in Row (11): 7.8%
Item 4. Purpose of Transaction. The Reporting Persons intend to engage management.
</DOCUMENT>
"""


# ── parser ───────────────────────────────────────────────────────────────────

def test_parse_activist_extracts_filer_subject_pct():
    p = parse_activist(FIXTURE_13D)
    assert p["filer"] == "ACTIVIST CAPITAL LP"
    assert p["subject"] == "TARGET MINING CORP"
    assert p["pct"] == 7.8


def test_parse_activist_missing_fields_are_none():
    p = parse_activist("no header, no cover page here")
    assert p["filer"] is None and p["subject"] is None and p["pct"] is None


# Representative GROUP-filed SC 13D: two "FILED BY:" blocks (one per co-filer)
# and two "Percent of Class" cover pages (one per reporting person).
FIXTURE_13D_GROUP = """<SEC-HEADER>0001104659-26-054321.hdr.sgml : 20260605
ACCESSION NUMBER:\t\t0001104659-26-054321
CONFORMED SUBMISSION TYPE:\tSC 13D
FILED AS OF DATE:\t\t20260605

SUBJECT COMPANY:

\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\t\tTARGET MINING CORP
\t\tCENTRAL INDEX KEY:\t\t\t0000111111

FILED BY:

\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\t\tACTIVIST CAPITAL LP
\t\tCENTRAL INDEX KEY:\t\t\t0000222222

FILED BY:

\tCOMPANY DATA:
\t\tCOMPANY CONFORMED NAME:\t\t\tCO-FILER PARTNERS FUND LP
\t\tCENTRAL INDEX KEY:\t\t\t0000333333
</SEC-HEADER>
<DOCUMENT>
<TYPE>SC 13D
CUSIP No. 12345678
Reporting Person: ACTIVIST CAPITAL LP
(11) Aggregate Amount Beneficially Owned by Each Reporting Person: 3,000,000
(13) Percent of Class Represented by Amount in Row (11): 5.2%
Reporting Person: CO-FILER PARTNERS FUND LP
(11) Aggregate Amount Beneficially Owned by Each Reporting Person: 4,500,000
(13) Percent of Class Represented by Amount in Row (11): 7.8%
Item 4. Purpose of Transaction. The Reporting Persons intend to engage management.
</DOCUMENT>
"""


def test_parse_activist_group_filing_keeps_every_co_filer():
    """Regression: a jointly-filed 13D lists one FILED BY block per co-filer;
    first-match-only silently dropped every co-filer but the first."""
    p = parse_activist(FIXTURE_13D_GROUP)
    assert p["filer"] == "ACTIVIST CAPITAL LP; CO-FILER PARTNERS FUND LP"
    assert p["subject"] == "TARGET MINING CORP"


def test_parse_activist_group_filing_keeps_largest_stake():
    """Regression: a jointly-filed 13D has one cover page (and percent) per
    reporting person; first-match-only could report whichever happens to
    appear first rather than the group's largest reported stake."""
    p = parse_activist(FIXTURE_13D_GROUP)
    assert p["pct"] == 7.8


def test_accession_from_url():
    url = "https://www.sec.gov/Archives/edgar/data/111111/0001104659-26-012345.txt"
    assert accession_from_url(url) == "0001104659-26-012345"


# ── storage (incl. raw evidence text) ────────────────────────────────────────

def test_save_activist_filing_roundtrip_and_idempotent(tmp_path):
    conn = store.connect(tmp_path / "edgar.db")
    row = {
        "accession": "0001104659-26-012345", "cik": 111111, "ticker": "TGT",
        "form": "SC 13D", "filer": "ACTIVIST CAPITAL LP",
        "subject": "TARGET MINING CORP", "pct": 7.8, "date": "2026-06-05",
        "url": "https://sec.gov/x", "raw_text": FIXTURE_13D,
    }
    store.save_activist_filing(conn, row)

    got = conn.execute(
        "SELECT filer, pct, form, raw_text FROM activist_filings WHERE accession=?",
        ("0001104659-26-012345",),
    ).fetchone()
    assert got == ("ACTIVIST CAPITAL LP", 7.8, "SC 13D", FIXTURE_13D)

    # Re-scan of the same filing must not duplicate it.
    store.save_activist_filing(conn, row)
    assert conn.execute("SELECT COUNT(*) FROM activist_filings").fetchone()[0] == 1


# ── digest enrichment ────────────────────────────────────────────────────────

def test_digest_shows_filer_and_pct():
    activist = [{"ticker": "TGT", "cik": 1, "form": "SC 13D", "date": "2026-06-05",
                 "url": "https://sec.gov/x", "filer": "ACTIVIST CAPITAL LP", "pct": 7.8}]
    _, body = build_digest("2026-06-05", [], activist)
    assert "by ACTIVIST CAPITAL LP" in body
    assert "7.8%" in body
