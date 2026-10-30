"""
rbc_reports/parser.py
=====================
Parses one "RBC Insight Alert" daily digest (the email's HTML, as saved
from Gmail) into a Digest of ReportItems. Pure function, no I/O.

The digest is a fixed email template: every report starts with its own
`padding-top:7px` time cell ("6:24pm ET", or "25-Oct-2026 7:33pm ET" for
an item from an earlier day), then holds (in order) the source ("| RBC Capital Markets"), the bold title
("Company - headline"), a ResearchViewer link, an optional "Our view: ..."
teaser, and three 198px-wide cells: tickers, analysts, PDF size.

An item time without a date belongs to the digest's own day, unless it is
later than the digest's "as of" time (sent ~7pm ET, covering the reports
since the previous one) -- then it is from the previous evening.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Bloomberg-style exchange codes RBC appends to a ticker when one report
# covers several exchanges ("ATD CN CASY US MUSA US"). A code always
# follows the symbol it belongs to.
EXCHANGE_CODES = frozenset({
    "US", "CN", "LN", "AU", "GR", "IM", "SM", "SW", "FP", "BB", "NA", "ID",
    "PL", "NO", "SS", "DC", "FH", "JP", "JT", "HK", "AB", "AV", "NZ",
})

_BLOCK_SPLIT = "padding-top:7px;font-family:Arial;font-size:11px"
_AS_OF_RE = re.compile(r"as of \w+, (\d{1,2} \w+ \d{4}) (\d{1,2}:\d{2}[ap]m) ET")
_SUBSCRIPTION_RE = re.compile(r"Daily Digest for (.+?) as of ")
_COUNT_RE = re.compile(r"\b(\d+) of (\d+) Reports\b")
_TIME_RE = re.compile(r">\s*(?:(\d{1,2}-\w{3}-\d{4}) )?(\d{1,2}:\d{2}[ap]m) ET")
_SOURCE_RE = re.compile(r"\|\s*([^<]+?)\s*</font>")
_TITLE_RE = re.compile(r"<font[^>]*font-size:13px[^>]*>(.*?)</font>", re.S)
_LINK_RE = re.compile(r'href="(https://www\.rbcinsight\.com/WM/share/ResearchViewer/\?[^"]+)"')
_VIEW_RE = re.compile(r"padding-top:2px[^>]*>\s*<font[^>]*>(.*?)</font>", re.S)
_CELL_RE = re.compile(r'<td width="198"[^>]*>\s*<font[^>]*>(.*?)</font>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class ReportItem:
    published_at: str          # "YYYY-MM-DDTHH:MM", ET
    source: str
    title: str
    company: str
    headline: str
    link: str | None
    view: str | None
    tickers_raw: str
    tickers: tuple             # ((symbol, exchange-or-""), ...)
    analysts: str
    pdf_size: str

    @property
    def key(self) -> str:
        """Dedup key -- the report's own ResearchViewer link, falling back
        to time+title for an item without one."""
        return self.link or f"{self.published_at}|{self.title}"


@dataclass
class Digest:
    as_of: str                 # "YYYY-MM-DDTHH:MM", ET
    subscription: str          # the alert's name, e.g. "RBC Capital Markets - All Reports"
    shown: int | None
    total: int | None
    items: list = field(default_factory=list)


def _text(fragment: str) -> str:
    return " ".join(html.unescape(_TAG_RE.sub(" ", fragment)).split())


def _parse_time(day: datetime, hhmm: str) -> datetime:
    t = datetime.strptime(hhmm, "%I:%M%p")
    return day.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)


def parse_tickers(raw: str) -> tuple:
    """Splits a ticker cell into (symbol, exchange) pairs; exchange is ""
    when RBC gave none. A trailing "..." (RBC truncating a long list) is
    dropped."""
    tokens = [t for t in raw.split() if t != "..."]
    out, i = [], 0
    while i < len(tokens):
        sym = tokens[i]
        if i + 1 < len(tokens) and tokens[i + 1] in EXCHANGE_CODES:
            out.append((sym, tokens[i + 1]))
            i += 2
        else:
            out.append((sym, ""))
            i += 1
    return tuple(dict.fromkeys(out))


def is_digest(page: str) -> bool:
    return "Daily Digest for RBC" in page or "RBC Insight Alert" in page


def parse_digest(page: str) -> Digest:
    """Raises ValueError if the page has no "as of" header (not a digest)."""
    text = _text(page)
    m = _AS_OF_RE.search(text)
    if not m:
        raise ValueError("no 'as of ... ET' digest header found")
    as_of = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%d %B %Y %I:%M%p")
    c = _COUNT_RE.search(text)
    sub = _SUBSCRIPTION_RE.search(text)
    digest = Digest(
        as_of=as_of.strftime("%Y-%m-%dT%H:%M"),
        subscription=sub.group(1).strip() if sub else "",
        shown=int(c.group(1)) if c else None,
        total=int(c.group(2)) if c else None,
    )

    for block in page.split(_BLOCK_SPLIT)[1:]:
        tm, title_m = _TIME_RE.search(block), _TITLE_RE.search(block)
        if not tm or not title_m:
            continue
        if tm.group(1):
            published = _parse_time(datetime.strptime(tm.group(1), "%d-%b-%Y"), tm.group(2))
        else:
            published = _parse_time(as_of, tm.group(2))
            if published > as_of:
                published -= timedelta(days=1)
        title = _text(title_m.group(1))
        company, _, headline = title.partition(" - ")
        src = _SOURCE_RE.search(block)
        link = _LINK_RE.search(block)
        view = _VIEW_RE.search(block)
        cells = [_text(c) for c in _CELL_RE.findall(block)] + ["", "", ""]
        digest.items.append(ReportItem(
            published_at=published.strftime("%Y-%m-%dT%H:%M"),
            source=_text(src.group(1)) if src else "",
            title=title,
            company=company.strip(),
            headline=headline.strip(),
            link=html.unescape(link.group(1)) if link else None,
            view=(_text(view.group(1)) or None) if view else None,
            tickers_raw=cells[0],
            tickers=parse_tickers(cells[0]),
            analysts=cells[1],
            pdf_size=cells[2],
        ))
    return digest
