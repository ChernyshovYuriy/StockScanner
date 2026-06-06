"""Offline tests for parsing/selection logic (no network)."""

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
