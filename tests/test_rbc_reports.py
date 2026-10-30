"""
Offline tests for rbc_reports/ (parser, store, ingest) and
rbc_dashboard_data.py. The digest fixture below is a trimmed, synthetic
copy of the real RBC Insight email template -- same tags/styles the
parser keys off, made-up content.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from rbc_dashboard_data import build_rbc_state
from rbc_reports import store
from rbc_reports.ingest import ingest_dir
from rbc_reports.parser import parse_digest, parse_tickers


def _item(time, title, tickers, analysts, view="", link_id="X1", border=True):
    border_css = "border-top:solid 1px #e5e5e5" if border else ""
    return f"""
<table border="0" cellpadding="0" cellspacing="0" width="610"><tr>
<td width="610" style="width:610px;{border_css}">
<table border="0" cellpadding="0" cellspacing="0" width="610">
<tr><td width="610" style="width:610px;padding-top:7px;font-family:Arial;font-size:11px;color:#333333">
<font color="#333333" face="Arial" size="2" style="font-family:Arial;font-size:11px;color:#333333">{time} </font>
<font color="#666666" face="Arial" size="2" style="font-family:Arial;font-size:11px;color:#666666">
| RBC Capital Markets</font>
</td></tr>
<tr><td width="610" style="width:610px;padding-top:4px;font-family:Arial;font-size:12px;color:#000000"><font color="#000000" face="Arial" size="2" style="font-family:Arial;font-size:12px;color:#000000"><font color="#000000" face="Arial" size="1" style="font-family:Arial;font-size:13px;color:#000000;text-decoration:none;font-weight:bold">{title}</font>
<a style="font-family:Arial;font-size:13px;color:#0063a9" href="https://www.rbcinsight.com/WM/share/ResearchViewer/?{link_id}==&amp;a=a" target="_blank"> - Click to view</a></font>
</td></tr>
<tr><td width="610" style="width:610px;padding-top:2px;font-family:Arial;font-size:11px;color:#666666"><font color="#666666" face="Arial" size="2" style="font-family:Arial;font-size:11px;color:#666666">{view}</font>
</td></tr>
<tr><td width="610" style="width:610px;padding-top:5px;padding-bottom:9px"><table border="0" cellpadding="0" cellspacing="0" width="610"><tr>
<td width="198" align="left" valign="top" style="width:198px"><font color="#333333" face="Arial" size="2">{tickers}</font></td>
<td width="8"><font> </font></td>
<td width="198" align="left" valign="top" style="width:198px"><font color="#333333" face="Arial" size="2">{analysts}</font></td>
<td width="8"><font> </font></td>
<td width="198" align="right" valign="top" style="width:198px"><font color="#666666" face="Arial" size="2">PDF -
310k </font></td>
</tr></table></td></tr>
</table></td></tr></table>"""


def _digest(items, as_of="Thursday, 29 October 2026 07:01pm ET", shown=4, total=9):
    return f"""<html><body>
<b>RBC Insight Alert - Subscription: RBC Capital Markets - All Reports</b>
<div>Daily Digest for RBC Capital Markets - All Reports</div>
<div>as of {as_of}</div>
{''.join(items)}
<div><b>{shown}</b> of <b>{total}</b> Reports</div>
</body></html>"""


SAMPLE = _digest([
    _item("6:24pm ET", "WESCO International, Inc. - Strong 3Q26 Beat &amp; Raise", "WCC", "Jeffrey Reive",
          view="\nOur view: Outperform-rated WESCO posted another beat-and-raise...", link_id="A1", border=False),
    _item("5:17pm ET", "Life Time Group Holdings, Inc. - Firing on all cylinders",
          "LTH AW CN BROS US CAVA US ...", "Logan Reich", link_id="A2"),
    _item("8:05pm ET", "Goldman Sachs Group Inc. - Late note from yesterday evening", "GS", "Gerard Cassidy",
          link_id="A3"),
    _item("25-Oct-2026 7:33pm ET", "Industrial Metals Weekly - Copper down on US inventories", "",
          "Sam Crittenden", link_id="A4"),
])


# ── parser ────────────────────────────────────────────────────────────────

def test_parse_digest_reads_header_counts_and_every_item():
    d = parse_digest(SAMPLE)
    assert d.as_of == "2026-10-29T19:01"
    assert d.subscription == "RBC Capital Markets - All Reports"
    assert (d.shown, d.total) == (4, 9)
    assert [i.company for i in d.items] == [
        "WESCO International, Inc.", "Life Time Group Holdings, Inc.",
        "Goldman Sachs Group Inc.", "Industrial Metals Weekly",
    ]


def test_parse_digest_item_fields():
    first = parse_digest(SAMPLE).items[0]
    assert first.published_at == "2026-10-29T18:24"
    assert first.headline == "Strong 3Q26 Beat & Raise"
    assert first.view.startswith("Our view: Outperform-rated WESCO")
    assert first.link == "https://www.rbcinsight.com/WM/share/ResearchViewer/?A1==&a=a"
    assert first.tickers == (("WCC", ""),)
    assert first.analysts == "Jeffrey Reive"
    assert first.pdf_size == "PDF - 310k"


def test_raw_email_style_formatting_parses_the_same():
    """The original email (what gmail_fetch reads) writes styles with
    spaces and line breaks; Gmail's saved page compacts them."""
    import re
    loose = re.sub(r'style="([^"]*)"',
                   lambda m: 'style="' + m.group(1).replace(";", "; \n\t\t").replace(":", ": ") + '"',
                   SAMPLE)
    assert loose != SAMPLE
    assert parse_digest(loose).items == parse_digest(SAMPLE).items


def test_item_without_view_has_none():
    assert parse_digest(SAMPLE).items[1].view is None


def test_time_after_the_digest_belongs_to_the_previous_evening():
    assert parse_digest(SAMPLE).items[2].published_at == "2026-10-28T20:05"


def test_explicit_date_on_an_item_is_used():
    item = parse_digest(SAMPLE).items[3]
    assert item.published_at == "2026-10-25T19:33"
    assert item.tickers == ()


def test_non_digest_page_raises():
    with pytest.raises(ValueError):
        parse_digest("<html><body>hello</body></html>")


def test_parse_tickers_pairs_exchange_codes_and_drops_ellipsis():
    assert parse_tickers("LTH AW CN BROS US CAVA US ...") == (
        ("LTH", ""), ("AW", "CN"), ("BROS", "US"), ("CAVA", "US"))
    assert parse_tickers("SPGI MCO") == (("SPGI", ""), ("MCO", ""))
    assert parse_tickers("") == ()


# ── store / ingest ────────────────────────────────────────────────────────

def test_save_digest_is_idempotent(tmp_path):
    conn = store.connect(tmp_path / "rbc.db")
    d = parse_digest(SAMPLE)
    assert store.save_digest(conn, d, "a.html", "2026-10-29T19:30") == 4
    assert store.save_digest(conn, d, "a-copy.html", "2026-10-29T19:40") == 0
    assert conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM digests").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM report_tickers").fetchone()[0] == 6


def test_ingest_dir_skips_non_digest_files(tmp_path):
    folder = tmp_path / "reports"
    folder.mkdir()
    (folder / "digest.html").write_text(SAMPLE)
    (folder / "other.html").write_text("<html>not a digest</html>")
    (folder / "report.pdf").write_bytes(b"%PDF-1.4")
    conn = store.connect(tmp_path / "rbc.db")
    s = ingest_dir(folder, conn=conn)
    assert (s["files"], s["digests"], s["reports_new"]) == (2, 1, 4)
    assert s["skipped"] == ["other.html"]


def test_ingest_dir_skips_files_already_ingested(tmp_path):
    folder = tmp_path / "reports"
    folder.mkdir()
    (folder / "digest.html").write_text(SAMPLE)
    conn = store.connect(tmp_path / "rbc.db")
    ingest_dir(folder, conn=conn)
    s = ingest_dir(folder, conn=conn)
    assert (s["already"], s["digests"], s["reports_new"]) == (1, 0, 0)


def test_ingest_dir_dry_run_writes_nothing(tmp_path):
    folder = tmp_path / "reports"
    folder.mkdir()
    (folder / "digest.html").write_text(SAMPLE)
    s = ingest_dir(folder, dry_run=True)
    assert (s["digests"], s["reports_seen"], s["reports_new"]) == (1, 4, 0)


# ── dashboard data ────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "rbc.db"
    conn = store.connect(path)
    store.save_digest(conn, parse_digest(SAMPLE), "a.html", "2026-10-29T19:30")
    conn.close()
    return path


_NOW = datetime(2026, 10, 29, 20, 0)


def test_state_feed_is_newest_first_with_tickers(db_path):
    s = build_rbc_state(days=7, db_path=db_path, now=_NOW)
    assert [r["published_at"] for r in s["reports"]] == [
        "2026-10-29T18:24", "2026-10-29T17:17", "2026-10-28T20:05", "2026-10-25T19:33"]
    assert [t["symbol"] for t in s["reports"][1]["tickers"]] == ["LTH", "AW", "BROS", "CAVA"]
    assert s["digests"] == [{"date": "2026-10-29", "subscription": "RBC Capital Markets - All Reports",
                             "shown": 4, "total": 9, "missed": 5}]


def test_state_days_window(db_path):
    s = build_rbc_state(days=1, db_path=db_path, now=_NOW)
    assert s["total_reports"] == 3


def test_state_ticker_timeline_ignores_days(db_path):
    s = build_rbc_state(days=1, ticker="cava", db_path=db_path, now=datetime(2026, 12, 1))
    assert s["ticker"] == "CAVA"
    assert [r["company"] for r in s["reports"]] == ["Life Time Group Holdings, Inc."]


def test_state_text_query(db_path):
    s = build_rbc_state(days=None, query="copper", db_path=db_path)
    assert [r["company"] for r in s["reports"]] == ["Industrial Metals Weekly"]


def test_state_top_tickers(db_path):
    s = build_rbc_state(days=None, db_path=db_path)
    assert {t["symbol"] for t in s["top_tickers"]} == {"WCC", "GS"}


def test_state_missing_db_is_empty(tmp_path):
    s = build_rbc_state(db_path=tmp_path / "nope.db")
    assert s["reports"] == [] and s["digests"] == []


# ── gmail_fetch (fake IMAP, never the real mailbox) ───────────────────────

from email.message import EmailMessage

from rbc_reports import gmail_fetch


def _raw_email(html_body):
    msg = EmailMessage()
    msg["From"] = "alerts@insight.rbc.com"
    msg["Subject"] = "RBC Insight Alert - Subscription: RBC Capital Markets - All Reports"
    msg.set_content("plain part")
    msg.add_alternative(html_body, subtype="html")
    return msg.as_bytes()


class _FakeIMAP:
    """Records calls; serves `messages` (id -> raw bytes)."""
    def __init__(self, messages):
        self.messages, self.calls = messages, []

    def __call__(self, host):
        self.calls.append(("connect", host))
        return self

    def login(self, user, pw):
        self.calls.append(("login", user))

    def select(self, mailbox, readonly=False):
        self.calls.append(("select", mailbox, readonly))
        return "OK", [b"3"]

    def search(self, charset, criteria):
        self.calls.append(("search", criteria))
        return "OK", [b" ".join(self.messages)]

    def fetch(self, msg_id, spec):
        self.calls.append(("fetch", msg_id, spec))
        return "OK", [(b"1 (BODY[] {n}", self.messages[msg_id]), b")"]

    def logout(self):
        self.calls.append(("logout",))


@pytest.fixture
def gmail_creds(monkeypatch):
    monkeypatch.setattr(gmail_fetch, "GMAIL_USER", "me@gmail.com")
    monkeypatch.setattr(gmail_fetch, "GMAIL_APP_PASSWORD", "app-pass")


def test_fetch_saves_new_digests_read_only(tmp_path, gmail_creds):
    other = SAMPLE.replace("29 October 2026 07:01pm", "28 October 2026 07:01pm")
    fake = _FakeIMAP({b"1": _raw_email(SAMPLE), b"2": _raw_email(other),
                      b"3": _raw_email("<html>not a digest</html>")})
    r = gmail_fetch.fetch_digests(days=7, folder=tmp_path, imap_factory=fake)
    assert r["messages"] == 3 and r["skipped"] == 1
    assert sorted(r["saved"]) == [
        "rbc_digest_20261028_1901_rbc-capital-markets-all-reports.html",
        "rbc_digest_20261029_1901_rbc-capital-markets-all-reports.html",
    ]
    assert ("select", '"[Gmail]/All Mail"', True) in fake.calls
    assert all(c[2] == "(BODY.PEEK[])" for c in fake.calls if c[0] == "fetch")
    search = next(c[1] for c in fake.calls if c[0] == "search")
    assert 'FROM "alerts@insight.rbc.com"' in search and "SINCE" in search
    assert fake.calls[-1] == ("logout",)
    # the saved file parses back to the same digest
    saved = (tmp_path / r["saved"][1]).read_text()
    assert len(parse_digest(saved).items) == 4


def test_html_part_without_charset_is_decoded_as_utf8():
    """RBC's HTML part has no MIME charset; its body is UTF-8."""
    raw = (b"From: alerts@insight.rbc.com\r\nMIME-Version: 1.0\r\n"
           b'Content-Type: multipart/alternative; boundary="b"\r\n\r\n'
           b"--b\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: 8bit\r\n\r\nx\r\n"
           b"--b\r\nContent-Type: text/html\r\nContent-Transfer-Encoding: 8bit\r\n\r\n"
           + "<p>Athabasca\u2019s FY\u201927</p>".encode("utf-8") + b"\r\n--b--\r\n")
    assert gmail_fetch._html_body(raw) == "<p>Athabasca\u2019s FY\u201927</p>"


def test_fetch_skips_already_saved_and_all_has_no_since(tmp_path, gmail_creds):
    fake = _FakeIMAP({b"1": _raw_email(SAMPLE)})
    gmail_fetch.fetch_digests(days=None, folder=tmp_path, imap_factory=fake)
    r = gmail_fetch.fetch_digests(days=None, folder=tmp_path, imap_factory=fake)
    assert (r["saved"], r["existing"]) == ([], 1)
    assert all("SINCE" not in c[1] for c in fake.calls if c[0] == "search")


def test_fetch_dry_run_writes_nothing(tmp_path, gmail_creds):
    fake = _FakeIMAP({b"1": _raw_email(SAMPLE)})
    r = gmail_fetch.fetch_digests(days=7, folder=tmp_path, dry_run=True, imap_factory=fake)
    assert len(r["saved"]) == 1
    assert list(tmp_path.iterdir()) == []


def test_fetch_unconfigured_never_connects(tmp_path, monkeypatch):
    monkeypatch.setattr(gmail_fetch, "GMAIL_USER", "")
    fake = _FakeIMAP({})
    r = gmail_fetch.fetch_digests(folder=tmp_path, imap_factory=fake)
    assert r["configured"] is False and fake.calls == []
