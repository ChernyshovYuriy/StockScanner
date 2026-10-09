"""
CSE company news (2026-10): the CSE's public company-news JSON (the whole
archive, newest first) read from a byte-limited prefix, turned into normal
feed items, gated like Newsfile (only 'high' reaches the user), never
article-fetched. Plus: the parser's ".CSE" suffix resolves to Yahoo's ".CN".
"""
import json
import shutil
from email.utils import parsedate_to_datetime

import pytest
import requests

import config
import news_watchlist_service
import press_release_service
from news_watchlist import outcomes
from press_release_tracker import analyst, cse_news, feeds, store

URL = config.PRESS_RELEASE_CSE_NEWS_FEED


def _entry(i, title="Hertz Energy Announces Private Placement", lang="en", symbol="HZ",
           date="2026-10-09T08:46:44-04:00"):
    return {"id": i, "date": date, "title": title, "language": lang, "slug": "s",
            "mainSymbol": symbol, "symbols": [symbol], "fileUrl": "https://pdf?X-Amz-Expires=300"}


def _doc(entries):
    return json.dumps({"totalItems": 98449, "list": entries})


# ── parsing ─────────────────────────────────────────────────────────────────

def test_truncated_prefix_yields_only_complete_entries():
    text = _doc([_entry(3), _entry(2), _entry(1)])
    cut = text[: text.index('"id": 1') + 5]  # mid-way through the third entry
    assert [e["id"] for e in cse_news.parse_entries(cut)] == [3, 2]


def test_full_document_is_parsed_to_the_end():
    assert [e["id"] for e in cse_news.parse_entries(_doc([_entry(2), _entry(1)]))] == [2, 1]


def test_unrecognised_response_raises_value_error():
    with pytest.raises(ValueError):
        cse_news.parse_entries("<html>login</html>")


def test_feed_item_fields():
    it = cse_news.to_feed_item(_entry(299369), URL)
    assert it.guid == "cse-news:299369"
    assert it.feed_url == URL
    assert it.link == "https://thecse.com/s/HZ"
    assert "(CSE: HZ)" in it.description
    # RSS-style date, as news_watchlist parses it
    assert parsedate_to_datetime(it.pubdate).isoformat() == "2026-10-09T08:46:44-04:00"


def test_french_dropped_neutral_kept():
    assert cse_news.to_feed_item(_entry(1, lang="fr"), URL) is None
    assert cse_news.to_feed_item(_entry(1, lang="neutral"), URL) is not None


def test_missing_symbol_or_date_still_an_item():
    it = cse_news.to_feed_item(_entry(1, symbol=None, date=None), URL)
    assert it.pubdate == "" and "CSE:" not in it.description


# ── fetching: a Range request, and never more than max_bytes read ───────────

class _Resp:
    def __init__(self, body, status=206):
        self.body, self.status_code, self.closed, self.read = body, status, False, 0

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            self.read += chunk_size
            yield self.body[i:i + chunk_size]

    def close(self):
        self.closed = True


def test_fetch_sends_range_and_stops_reading_at_the_limit(monkeypatch):
    # A server that ignores Range and sends the whole (big) archive.
    body = _doc([_entry(i) for i in range(5000, 0, -1)]).encode()
    seen = {}

    def fake_get(url, headers, timeout, stream):
        seen.update(headers=headers, stream=stream)
        seen["resp"] = _Resp(body, status=200)
        return seen["resp"]

    monkeypatch.setattr(cse_news.requests, "get", fake_get)
    items = cse_news.fetch_cse_items(URL, max_bytes=50_000)
    assert seen["headers"]["Range"] == "bytes=0-49999" and seen["stream"]
    assert seen["resp"].read < 50_000 + 16384 * 2 and seen["resp"].closed
    assert 0 < len(items) < 5000
    assert items[0].guid == "cse-news:5000"


def test_feeds_dispatches_the_cse_url(monkeypatch):
    monkeypatch.setattr(cse_news.requests, "get",
                        lambda url, **k: _Resp(_doc([_entry(7)]).encode()))
    assert [i.guid for i in feeds.fetch_feed_items(URL)] == ["cse-news:7"]


def test_login_redirect_is_logged_and_other_feeds_still_run(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", [URL, "https://feed-b"])
    rss = _Resp(b"", 200)
    rss.content = b"<rss><channel><item><guid>b1</guid><title>T</title></item></channel></rss>"
    # cse_news and feeds share one requests module: route by URL
    monkeypatch.setattr(requests, "get",
                        lambda url, **k: _Resp(b"<html>login</html>", 200) if url == URL else rss)
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release", lambda *a: None)
    monkeypatch.setattr(press_release_service, "send_text_email", lambda s, b: True)
    press_release_service.run_collector("r", conn=conn)
    assert [r[0] for r in conn.execute("SELECT guid FROM seen_items")] == ["b1"]


# ── service: gated like Newsfile, never article-fetched ─────────────────────

def test_cse_config_is_high_only():
    assert URL in config.PRESS_RELEASE_FEEDS
    assert URL in config.PRESS_RELEASE_HIGH_ONLY_FEEDS


def _run_cse_item(tmp_path, monkeypatch, materiality, category="financing"):
    conn = store.connect(tmp_path / "pr.db")
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_FEEDS", [URL])
    monkeypatch.setattr(press_release_service, "PRESS_RELEASE_HIGH_ONLY_FEEDS", frozenset({URL}))
    monkeypatch.setattr(press_release_service.feeds, "fetch_feed_items",
                        lambda url: [cse_news.to_feed_item(_entry(1), URL)])
    monkeypatch.setattr(press_release_service.llm_parser, "parse_release",
                        lambda *a: {"ticker": "HZ.CSE", "company": "Hertz", "category": category,
                                    "materiality": materiality, "summary": "s"})
    monkeypatch.setattr(press_release_service.article, "fetch_article_text",
                        lambda link: pytest.fail("a CSE item's link is the company page, never read"))
    calls = {"pdf": [], "analyst": [], "terms": [], "sent": []}
    monkeypatch.setattr(press_release_service.cse_news, "fetch_release_text",
                        lambda item: calls["pdf"].append(item.guid) or calls.get("pdf_text", "PDF BODY"))
    monkeypatch.setattr(press_release_service.analyst, "analyze_release",
                        lambda ticker, company, title, link, body=None: calls["analyst"].append(body) or None)
    monkeypatch.setattr(press_release_service.financing, "extract_terms",
                        lambda ticker, title, link, body=None, context=None: calls["terms"].append(body) or None)
    monkeypatch.setattr(press_release_service, "send_text_email",
                        lambda s, b: calls["sent"].append(b) or True)
    return conn, calls


def test_high_cse_item_is_emailed_and_analysed_from_its_pdf(tmp_path, monkeypatch):
    conn, calls = _run_cse_item(tmp_path, monkeypatch, "high")
    press_release_service.run_collector("r", conn=conn)
    assert calls["pdf"] == ["cse-news:1"]
    assert calls["analyst"] == ["PDF BODY"] and calls["terms"] == ["PDF BODY"]
    assert calls["sent"] and store.unemailed(conn) == []


def test_non_high_cse_item_is_filed_with_no_pdf_read(tmp_path, monkeypatch):
    conn, calls = _run_cse_item(tmp_path, monkeypatch, "medium")
    press_release_service.run_collector("r", conn=conn)
    assert calls["pdf"] == [] and calls["analyst"] == [] and calls["terms"] == []
    assert not calls["sent"] and store.unemailed(conn) == []


def test_unreadable_pdf_skips_analysis_and_financing(tmp_path, monkeypatch):
    conn, calls = _run_cse_item(tmp_path, monkeypatch, "high")
    monkeypatch.setattr(press_release_service.cse_news, "fetch_release_text",
                        lambda item: calls["pdf"].append(item.guid) or None)
    press_release_service.run_collector("r", conn=conn)
    assert calls["pdf"] == ["cse-news:1"]
    assert calls["analyst"] == [] and calls["terms"] == []  # nothing to read; company page never fetched
    assert calls["sent"]  # still emailed, just without the analyst block


# ── the release PDF: fresh link from the prefix, paced, pdftotext ───────────

def test_release_text_uses_a_fresh_link_from_the_prefix(monkeypatch):
    fresh = dict(_entry(42), fileUrl="https://webfiles-primary.thecse.com/pdf/42?sig=fresh")
    got = []

    def fake_get(url, headers=None, timeout=None, stream=False):
        got.append(url)
        if url == URL:
            return _Resp(_doc([_entry(43), fresh]).encode())
        r = _Resp(b"%PDF-bytes", 200)
        r.content = b"%PDF-bytes"
        return r

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(cse_news.article, "wait_turn", lambda url: got.append("wait:" + url))
    monkeypatch.setattr(cse_news.article, "document_to_text", lambda data: "text of " + data.decode())
    item = cse_news.to_feed_item(_entry(42), URL)
    assert cse_news.fetch_release_text(item) == "text of %PDF-bytes"
    assert got == [URL, "wait:" + fresh["fileUrl"], fresh["fileUrl"]]


def test_release_text_none_when_release_left_the_prefix(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **k: _Resp(_doc([_entry(43)]).encode()))
    assert cse_news.fetch_release_text(cse_news.to_feed_item(_entry(42), URL)) is None


def test_release_text_none_on_pdf_fetch_failure(monkeypatch):
    def fake_get(url, **k):
        if url == URL:
            return _Resp(_doc([_entry(42)]).encode())
        return _Resp(b"", 403)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(cse_news.article, "wait_turn", lambda url: None)
    assert cse_news.fetch_release_text(cse_news.to_feed_item(_entry(42), URL)) is None


def _tiny_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


@pytest.mark.skipif(not shutil.which("pdftotext"), reason="poppler pdftotext not installed")
def test_pdf_to_text_extracts_the_text():
    from press_release_tracker import article
    assert article.pdf_to_text(_tiny_pdf("Hertz Energy closes placement")) == "Hertz Energy closes placement"


def test_pdf_to_text_none_without_pdftotext_or_on_garbage(monkeypatch):
    from press_release_tracker import article
    monkeypatch.setattr(article.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("pdftotext")))
    assert article.pdf_to_text(b"%PDF") is None


@pytest.mark.skipif(not shutil.which("pdftotext"), reason="poppler pdftotext not installed")
def test_pdf_to_text_none_for_a_non_pdf():
    from press_release_tracker import article
    assert article.pdf_to_text(b"<html>not a pdf</html>") is None


# ── ".CSE" resolves to Yahoo's ".CN" everywhere a price is looked up ────────

def test_outcomes_candidates_map_cse_to_cn():
    assert outcomes.symbol_candidates("hz.cse") == ["HZ.CN"]
    assert outcomes.symbol_candidates("KTO.V") == ["KTO.V"]


def test_analyst_candidates_map_cse_to_cn():
    assert analyst._candidates("HZ.CSE") == ["HZ.CN"]


def test_watchlist_price_resolver_maps_cse_to_cn(monkeypatch):
    asked = []
    monkeypatch.setattr(news_watchlist_service, "get_market_price",
                        lambda t: asked.append(t) or ((0.5, "daily-close") if t == "HZ.CN" else (None, None)))
    assert news_watchlist_service._resolve_market_price("HZ.CSE") == (0.5, "daily-close", "HZ.CN")
    assert asked == ["HZ.CN"]


def test_bare_cse_ticker_resolves_to_cn_before_a_us_lookalike(monkeypatch):
    # The parser writes a CSE item's ticker bare ("SX"); a bare US "SX" must
    # not win over the CSE listing.
    asked = []
    monkeypatch.setattr(news_watchlist_service, "get_market_price",
                        lambda t: asked.append(t) or ((1.0, "daily-close") if t in ("SX.CN", "SX") else (None, None)))
    assert news_watchlist_service._resolve_market_price("SX") == (1.0, "daily-close", "SX.CN")
    assert asked == ["SX.TO", "SX.V", "SX.CN"]
    assert analyst._candidates("SX") == ["SX.TO", "SX.V", "SX.CN", "SX"]


def _tiny_docx(paragraphs) -> bytes:
    import io, zipfile
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{p[:5]}</w:t></w:r><w:r><w:t>{p[5:]}</w:t></w:r></w:p>" for p in paragraphs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f'<w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>')
    return buf.getvalue()


def test_docx_release_text():
    from press_release_tracker import article
    data = _tiny_docx(["Asep Medical Announces LIFE Financing", "Proceeds of $1.5 million"])
    assert article.document_to_text(data) == "Asep Medical Announces LIFE Financing\nProceeds of $1.5 million"


@pytest.mark.skipif(not shutil.which("pdftotext"), reason="poppler pdftotext not installed")
def test_document_to_text_routes_pdf_and_rejects_unknown():
    from press_release_tracker import article
    assert article.document_to_text(_tiny_pdf("Hertz closes")) == "Hertz closes"
    assert article.document_to_text(b"<html>login</html>") is None
    assert article.docx_to_text(b"PK not really a zip") is None
