"""
Layer 2b: insider transactions (Form 4).

The signal worth computing is clustered open-market PURCHASES (code 'P') by
multiple insiders -- not sells, which are mostly noise.
"""

import re
import xml.etree.ElementTree as ET

from edgar.core import fetch_submissions, get


def _recent_form4_accessions(cik_int, limit=15):
    subs = fetch_submissions(cik_int)
    recent = subs.get("filings", {}).get("recent", {})
    out = []
    for form, accn, doc, fdate in zip(recent.get("form", []),
                                      recent.get("accessionNumber", []),
                                      recent.get("primaryDocument", []),
                                      recent.get("filingDate", [])):
        if form == "4":
            out.append((accn, doc, fdate))
        if len(out) >= limit:
            break
    return out


def _form4_xml_url(cik_int, accession, primary_doc):
    return (f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik_int)}/{accession.replace('-', '')}/{primary_doc}")


def _txt(node, path):
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else None


def parse_form4(xml_text):
    xml_text = re.sub(r'\sxmlns="[^"]+"', "", xml_text, count=1)
    root = ET.fromstring(xml_text)
    owner = _txt(root, ".//reportingOwner/reportingOwnerId/rptOwnerName")
    is_dir = _txt(root, ".//reportingOwner/reportingOwnerRelationship/isDirector")
    is_off = _txt(root, ".//reportingOwner/reportingOwnerRelationship/isOfficer")
    txns = []
    for t in root.findall(".//nonDerivativeTransaction"):
        txns.append({
            "code": _txt(t, ".//transactionCoding/transactionCode"),
            "shares": _f(_txt(t, ".//transactionAmounts/transactionShares/value")),
            "price": _f(_txt(t, ".//transactionAmounts/transactionPricePerShare/value")),
            "date": _txt(t, ".//transactionDate/value"),
            "direction": _txt(t, ".//transactionAmounts/transactionAcquiredDisposedCode/value"),
        })
    return {"owner": owner, "is_director": is_dir == "1",
            "is_officer": is_off == "1", "transactions": txns}


def _f(s):
    return float(s) if s else None


def get_recent_insider_activity(cik_int, limit=15):
    results = []
    for accn, doc, fdate in _recent_form4_accessions(cik_int, limit=limit):
        if not doc or not doc.endswith(".xml"):
            continue
        try:
            xml = get(_form4_xml_url(cik_int, accn, doc),
                      cache_key=f"form4_{accn}.xml", is_json=False)
            parsed = parse_form4(xml)
            parsed.update({"accession": accn, "filing_date": fdate})
            results.append(parsed)
        except Exception as e:
            results.append({"accession": accn, "error": str(e)})
    return results


def open_market_buys(activity):
    buys = []
    for f in activity:
        for t in f.get("transactions", []):
            if t.get("code") == "P":
                buys.append({"owner": f.get("owner"),
                             "is_officer": f.get("is_officer"),
                             "is_director": f.get("is_director"),
                             "filing_date": f.get("filing_date"), **t})
    return buys


def cluster_flag(buys, min_distinct=2):
    return len({b["owner"] for b in buys}) >= min_distinct
