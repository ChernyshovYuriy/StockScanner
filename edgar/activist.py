"""
Activist / passive stake filings (SC 13D / 13G) — body parsing.

v1 extracts the filer (the "big money"), percent of class owned, and the subject
company from the full submission text, and KEEPS THE RAW filing text so a later
analysis layer (e.g. an LLM) can re-read evidence the regexes miss. 13D = activist
intent (stronger signal); 13G = passive. The form string itself carries the
13D-vs-13G distinction, so only filer + percent need parsing.

Honest ceiling: every filing is a LAGGED disclosure — a research trigger, never a
price predictor or financial advice. The deterministic parse is best-effort; the
retained raw text is the durable evidence.
"""
import re

from edgar.core import get

# Cap stored raw text so a filing with huge exhibits can't bloat the DB. The
# cover page (filer + percent) sits near the top, so trailing exhibits are
# expendable for the evidence we care about.
_RAW_TEXT_CAP = 1_000_000


def accession_from_url(url):
    """Derive the accession id from a daily-index submission URL/filename."""
    tail = (url or "").rstrip("/").split("/")[-1]
    return tail[:-4] if tail.endswith(".txt") else tail


def dedup_by_accession(hits):
    """Collapse the per-CIK duplicate index rows for one filing (same accession).

    A 13D is listed under both the filer and the subject-company CIK; keep one,
    preferring the row that resolved a ticker (the subject company).
    """
    by_acc = {}
    for h in hits:
        acc = accession_from_url(h.get("url", ""))
        cur = by_acc.get(acc)
        if cur is None or (not cur.get("ticker") and h.get("ticker")):
            by_acc[acc] = h
    return list(by_acc.values())


def _header_company(text, label):
    """COMPANY CONFORMED NAME within the SEC-HEADER block following `label`.

    Falls back to None when the block or name isn't found.
    """
    start = re.search(re.escape(label) + r"\s*:", text, re.I)
    if not start:
        return None
    block = text[start.end(): start.end() + 1500]
    name = re.search(r"COMPANY CONFORMED NAME\s*:\s*(.+)", block, re.I)
    return name.group(1).strip() if name else None


def _percent_of_class(text):
    """Best-effort percent-of-class from the cover page. None if not found."""
    m = re.search(
        r"Percent\s+of\s+Class.{0,200}?(\d{1,3}(?:\.\d+)?)\s*%",
        text, re.I | re.S,
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_activist(text):
    """Extract {filer, subject, pct} from a full-submission 13D/G text (best-effort)."""
    return {
        "filer": _header_company(text, "FILED BY"),
        "subject": _header_company(text, "SUBJECT COMPANY"),
        "pct": _percent_of_class(text),
    }


def fetch_activist_filing(url, accession=None):
    """Fetch + parse a 13D/G filing; return structured fields + capped raw text.

    Network call (rate-limited + cached by edgar.core.get). Raises on fetch
    failure — callers should guard so one bad filing doesn't sink the run.
    """
    accession = accession or accession_from_url(url)
    text = get(url, cache_key=f"activist_{accession}.txt", is_json=False)
    parsed = parse_activist(text)
    parsed["accession"] = accession
    parsed["raw_text"] = text[:_RAW_TEXT_CAP]
    return parsed
