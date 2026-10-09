"""
press_release_tracker/article.py
==================================
Fetches a press release's FULL article text from its wire page. The RSS
<description> llm_parser.py sees is only a 200-700 character teaser, so
the actual figures (revenue, net loss, cash, shares issued, financing
price) never reach it -- analyst.py needs the whole body.

Mechanical only, same fetch-vs-interpret split as feeds.py: no
interpretation here. Stdlib html.parser (no BeautifulSoup dependency).
GlobeNewswire marks the body with itemprop="articleBody", TMX Newsfile
with <article id="release">; a page without either marker returns None
rather than guessing at a body from page chrome.
Tables are kept row-by-row ("cell | cell | cell") since an earnings
release's numbers mostly live in them.

Fetches are paced one by one per host
(config.PRESS_RELEASE_ARTICLE_MIN_INTERVAL_SECONDS): both wires block an IP
that pulls pages in bursts.
"""
from __future__ import annotations

import io
import re
import subprocess
import time
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urlparse

import requests

try:
    from config import PRESS_RELEASE_USER_AGENT as USER_AGENT
except Exception:
    USER_AGENT = "StockScanner-PressRelease/0.1 (chernyshov.yuriy@gmail.com)"
try:
    from config import PRESS_RELEASE_ARTICLE_MIN_INTERVAL_SECONDS as MIN_INTERVAL_SECONDS
except Exception:
    MIN_INTERVAL_SECONDS = 10.0

_last_fetch: dict = {}  # host -> time.monotonic() of its last fetch

_BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "ul", "ol"}
_SKIP_TAGS = {"script", "style", "noscript"}
_VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "col", "source", "wbr"}


class _BodyExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._depth = 0          # open-element depth inside the article body; 0 = outside
        self._skip = 0
        self._row: Optional[list] = None
        self._cell: Optional[list] = None
        self.parts: list[str] = []
        self.found = False

    def handle_starttag(self, tag, attrs):
        if self._depth == 0:
            a = dict(attrs)
            if a.get("itemprop") == "articleBody" or (tag == "article" and a.get("id") == "release"):
                self._depth = 1
                self.found = True
            return
        if tag not in _VOID_TAGS:
            self._depth += 1
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self._depth == 0:
            return
        if tag not in _VOID_TAGS:
            self._depth -= 1
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            cells = [c for c in self._row if c]
            if cells:
                self.parts.append("\n" + " | ".join(cells) + "\n")
            self._row = None
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._depth == 0 or self._skip:
            return
        if self._cell is not None:
            self._cell.append(data)
        else:
            self.parts.append(data)


def extract_article_text(html: str) -> Optional[str]:
    """The article body's plain text (tables row-by-row), or None if the
    page has neither body marker (see module docstring)."""
    parser = _BodyExtractor()
    parser.feed(html)
    if not parser.found:
        return None
    text = "".join(parser.parts)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", text) or None


def wait_turn(url: str) -> None:
    """Sleep until url's host is due another request (MIN_INTERVAL_SECONDS
    since its last one), then record this one. Every wire page or PDF
    fetch goes through here."""
    host = urlparse(url).netloc
    last = _last_fetch.get(host)
    if last is not None:
        wait = MIN_INTERVAL_SECONDS - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    _last_fetch[host] = time.monotonic()


def pdf_to_text(data: bytes, timeout: int = 60) -> Optional[str]:
    """Plain text of a PDF via poppler's pdftotext (installed on the Pi),
    or None if it's missing, fails, or finds no text (e.g. a scanned
    image)."""
    try:
        out = subprocess.run(["pdftotext", "-q", "-enc", "UTF-8", "-", "-"], input=data,
                             capture_output=True, timeout=timeout, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [" ".join(line.split()) for line in out.decode("utf-8", errors="replace").splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text or None


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_to_text(data: bytes) -> Optional[str]:
    """Plain text of a Word .docx (one line per paragraph, table cells
    included), or None if it isn't one or has no text."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = ET.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        return None
    lines = [" ".join("".join(t.text or "" for t in p.iter(_W + "t")).split()) for p in root.iter(_W + "p")]
    text = "\n".join(line for line in lines if line)
    return text or None


def document_to_text(data: bytes) -> Optional[str]:
    """A release file's text by its leading bytes: PDF (pdf_to_text) or
    Word .docx (docx_to_text) -- the CSE serves both (~98% / ~2%)."""
    if data[:5] == b"%PDF-":
        return pdf_to_text(data)
    if data[:2] == b"PK":
        return docx_to_text(data)
    return None


def fetch_article_text(link: str, timeout: int = 30) -> Optional[str]:
    """Fetch link and extract its article body. None on any fetch failure
    or an unrecognised page -- the caller then skips the analysis for that
    item, never crashes the run over one page."""
    if not link:
        return None
    wait_turn(link)
    try:
        resp = requests.get(link, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    return extract_article_text(resp.text)
