"""
Layer 2a: fundamentals.

Resolves each metric through a fallback list of candidate XBRL tags and returns
latest values. Handles the real-world gotchas found in live testing:

  1. EPS period-mixing: EPS is reported under one tag in multiple period
     framings (quarterly vs trailing/annual). We select flow metrics by period
     LENGTH (prefer a ~quarter), not just latest end date.

  2. Stale stock values: companyfacts contains EVERY historical value a company
     ever tagged. Sorting point-in-time metrics by `end` and taking newest can
     surface an ancient entry (e.g. long_term_debt from 2012) when the current
     tag doesn't match. We anchor stock metrics to the company's latest
     balance-sheet date and REJECT entries far outside it.

  3. Tag coverage: the same concept is tagged differently across filers. Tag
     lists are widened, and `liabilities` falls back to the accounting identity
     (assets - equity) when no direct tag is present.
"""

from datetime import date

from edgar.core import fetch_company_facts

# Flow metrics are period-bounded (start..end); the rest are point-in-time.
FLOW_METRICS = {"revenue", "net_income", "eps_diluted"}

# How many days a stock-metric value may trail the latest balance-sheet date
# before we treat it as stale and reject it. One reporting gap of slack.
STALE_TOLERANCE_DAYS = 120

METRIC_TAGS = {
    "revenue": ["Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss",
                   "ProfitLoss",
                   "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities",
                    "LiabilitiesAndStockholdersEquity"],  # see identity fallback
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "eps_diluted": ["EarningsPerShareDiluted",
                    "EarningsPerShareBasicAndDiluted"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "long_term_debt": ["LongTermDebtNoncurrent",
                       "LongTermDebt",
                       "LongTermDebtAndCapitalLeaseObligations",
                       "DebtInstrumentCarryingAmount"],
}


def _parse(d):
    return date.fromisoformat(d)


def _period_days(entry):
    if "start" in entry and "end" in entry:
        return (_parse(entry["end"]) - _parse(entry["start"])).days
    return None


def _candidates(facts, tags):
    """All usable entries across candidate tags, newest end first."""
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for tag in tags:
        if tag not in usgaap:
            continue
        for unit_key, entries in usgaap[tag].get("units", {}).items():
            for e in entries:
                if e.get("val") is None or "end" not in e:
                    continue
                rows.append({"tag": tag, "unit": unit_key, "val": e["val"],
                             "end": e["end"], "start": e.get("start"),
                             "form": e.get("form"), "fy": e.get("fy"),
                             "fp": e.get("fp"), "days": _period_days(e)})
    rows.sort(key=lambda r: r["end"], reverse=True)
    return rows


def _latest_balance_date(facts):
    """Company's most recent point-in-time reporting date (anchor for staleness).
    Assets is the most reliably-tagged stock metric, so we anchor on it."""
    rows = _candidates(facts, METRIC_TAGS["assets"])
    return rows[0]["end"] if rows else None


def _select(facts, metric, tags, anchor=None):
    rows = _candidates(facts, tags)
    if not rows:
        return None

    if metric in FLOW_METRICS:
        # Prefer a ~quarterly window so trailing/annual figures don't land in a
        # quarterly comparison. Fall back to latest if no quarterly entry.
        quarterly = [r for r in rows if r["days"] and 80 <= r["days"] <= 100]
        return quarterly[0] if quarterly else rows[0]

    # Stock metric: reject anything stale relative to the anchor balance date.
    if anchor:
        a = _parse(anchor)
        fresh = [r for r in rows
                 if abs((a - _parse(r["end"])).days) <= STALE_TOLERANCE_DAYS]
        if fresh:
            return fresh[0]
        return None  # only stale junk exists -> report as missing, not 2012
    return rows[0]


def get_fundamentals(cik_int, pin_period=True):
    """
    Return {entity, metrics:{metric:{val,end,...}|None}, period_flags, target_period}.

    Stock metrics are anchored to the latest balance-sheet date; stale matches
    are dropped rather than reported. `liabilities` falls back to assets-equity.
    """
    facts = fetch_company_facts(cik_int)
    anchor = _latest_balance_date(facts)
    out = {"entity": facts.get("entityName"), "metrics": {},
           "period_flags": [], "target_period": anchor}

    for metric, tags in METRIC_TAGS.items():
        out["metrics"][metric] = _select(facts, metric, tags, anchor=anchor)

    # Liabilities identity fallback: total liabilities = assets - equity.
    liab = out["metrics"].get("liabilities")
    if liab is None or liab["tag"] == "LiabilitiesAndStockholdersEquity":
        a = out["metrics"].get("assets")
        e = out["metrics"].get("equity")
        if a and e and a["end"] == e["end"]:
            out["metrics"]["liabilities"] = {
                "val": a["val"] - e["val"], "end": a["end"],
                "tag": "DERIVED(assets-equity)", "unit": a["unit"],
                "start": None, "days": None, "form": a.get("form"),
                "fy": a.get("fy"), "fp": a.get("fp")}

    # Flag stock metrics whose period diverges from the anchor.
    if pin_period and anchor:
        for metric, v in out["metrics"].items():
            if metric in FLOW_METRICS or v is None:
                continue
            if v["end"] != anchor:
                out["period_flags"].append(
                    f"{metric} from {v['end']} (target {anchor})")
    return out
