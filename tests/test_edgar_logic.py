"""Offline tests for parsing/selection logic (no network)."""

from edgar import store
from edgar.fundamentals import _select
from edgar.insiders import parse_form4, open_market_buys, cluster_flag


def test_eps_prefers_quarterly_over_annual():
    facts = {"facts": {"us-gaap": {"EarningsPerShareDiluted": {"units": {"USD/shares": [
        {"val": 17.0, "start": "2025-02-26", "end": "2026-02-26"},
        {"val": 4.2, "start": "2025-11-28", "end": "2026-02-26"},
    ]}}}}}
    pick = _select(facts, "eps_diluted", ["EarningsPerShareDiluted"])
    assert pick["val"] == 4.2
    assert 80 <= pick["days"] <= 100


def test_form4_isolates_open_market_buys():
    xml = """<doc xmlns="x"><reportingOwner><reportingOwnerId>
    <rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>0</isOfficer>
    </reportingOwnerRelationship></reportingOwner><nonDerivativeTable>
    <nonDerivativeTransaction><transactionDate><value>2026-05-01</value></transactionDate>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>3000</value></transactionShares>
    <transactionPricePerShare><value>9.5</value></transactionPricePerShare>
    <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
    </transactionAmounts></nonDerivativeTransaction>
    <nonDerivativeTransaction><transactionDate><value>2026-05-02</value></transactionDate>
    <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>1000</value></transactionShares>
    <transactionPricePerShare><value>9.9</value></transactionPricePerShare>
    <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
    </transactionAmounts></nonDerivativeTransaction></nonDerivativeTable></doc>"""
    parsed = parse_form4(xml)
    buys = open_market_buys([parsed])
    assert len(buys) == 1
    assert buys[0]["code"] == "P"
    assert buys[0]["shares"] == 3000.0


def test_cluster_flag_needs_two_distinct():
    one = [{"owner": "A"}]
    two = [{"owner": "A"}, {"owner": "B"}]
    assert cluster_flag(one) is False
    assert cluster_flag(two) is True


def test_open_market_buys_carries_the_form4_accession():
    """Regression: the buy dict must carry the real Form 4 accession id, not
    just filing_date -- store.save_insider_buys keys the DB on it."""
    activity = [{
        "owner": "DOE JANE", "is_officer": False, "is_director": True,
        "filing_date": "2026-06-05", "accession": "0001234567-26-000001",
        "transactions": [{"code": "P", "shares": 3000.0, "price": 9.5,
                           "date": "2026-06-01", "direction": "A"}],
    }]
    buys = open_market_buys(activity)
    assert buys[0]["accession"] == "0001234567-26-000001"


def test_save_insider_buys_roundtrip_stores_real_accession(tmp_path):
    """Regression: save_insider_buys previously put filing_date into the
    accession column (a column/value mismatch), corrupting the field the
    table's PRIMARY KEY relies on for uniqueness."""
    conn = store.connect(tmp_path / "edgar.db")
    activity = [{
        "owner": "DOE JANE", "is_officer": False, "is_director": True,
        "filing_date": "2026-06-05", "accession": "0001234567-26-000001",
        "transactions": [{"code": "P", "shares": 3000.0, "price": 9.5,
                           "date": "2026-06-01", "direction": "A"}],
    }]
    buys = open_market_buys(activity)
    store.save_insider_buys(conn, 723125, buys)

    row = conn.execute(
        "SELECT cik, accession, owner, shares, price, txn_date FROM insider_buys"
    ).fetchone()
    assert row == (723125, "0001234567-26-000001", "DOE JANE", 3000.0, 9.5, "2026-06-01")
