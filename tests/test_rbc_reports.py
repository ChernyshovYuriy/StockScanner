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


def test_digest_that_yielded_no_reports_is_re_read(tmp_path):
    folder = tmp_path / "reports"
    folder.mkdir()
    (folder / "digest.html").write_text(SAMPLE)
    conn = store.connect(tmp_path / "rbc.db")
    empty = parse_digest(SAMPLE)
    empty.items = []
    store.save_digest(conn, empty, "digest.html", "2026-10-29T19:30")
    s = ingest_dir(folder, conn=conn)
    assert (s["already"], s["reports_new"]) == (0, 4)


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


# ── tagger (fake OpenAI client, never the real API) ───────────────────────

import json

from rbc_reports import tagger


class _FakeClient:
    """Answers each batch from `answer(lines)` -> list of label dicts;
    records every user message."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []
        self.chat = self
        self.completions = self

    def create(self, model, messages, **kw):
        lines = [json.loads(l) for l in messages[1]["content"].splitlines()]
        self.calls.append(lines)
        content = json.dumps({"reports": self.answer(lines)})
        msg = type("M", (), {"content": content})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def _bullish(lines):
    return [{"id": l["id"], "report_type": "company", "action": "results_beat", "tone": "Bullish",
             "rating": "Outperform", "theme": "industrials"} for l in lines]


def test_tagger_labels_every_untagged_report_once(db_path):
    conn = store.connect(db_path)
    client = _FakeClient(_bullish)
    r = tagger.tag_untagged(conn, client=client)
    assert r == {"configured": True, "pending": 4, "tagged": 4}
    assert "WESCO" in client.calls[0][0]["title"]          # newest first
    row = conn.execute("SELECT report_type, action, tone, rating, theme FROM report_tags "
                       "WHERE report_key LIKE '%A1%'").fetchone()
    assert row == ("company", "results_beat", "bullish", "Outperform", "industrials")
    assert tagger.tag_untagged(conn, client=client)["pending"] == 0
    assert tagger.tag_untagged(conn, client=client, retag=True)["tagged"] == 4


def test_tagger_batches_and_limit(db_path, monkeypatch):
    monkeypatch.setattr(tagger, "RBC_TAG_BATCH_SIZE", 3)
    client = _FakeClient(_bullish)
    assert tagger.tag_untagged(store.connect(db_path), client=client)["tagged"] == 4
    assert sorted(len(c) for c in client.calls) == [1, 3]
    conn = store.connect(db_path)
    assert tagger.tag_untagged(conn, client=client, retag=True, limit=2)["pending"] == 2


def test_tagger_unknown_values_fall_back_and_bad_ids_are_ignored(db_path):
    def odd(lines):
        return [{"id": 0, "report_type": "essay", "action": "maybe", "tone": "euphoric",
                 "rating": "null", "theme": "crypto"}, {"id": 99, "tone": "bullish"}, "junk"]
    conn = store.connect(db_path)
    assert tagger.tag_untagged(conn, client=_FakeClient(odd))["tagged"] == 1
    assert conn.execute("SELECT report_type, action, tone, rating, theme FROM report_tags").fetchone() == (
        "company", "none", "neutral", None, "other")


def test_tagger_failed_call_leaves_reports_untagged(db_path):
    def boom(lines):
        raise RuntimeError("API down")
    conn = store.connect(db_path)
    assert tagger.tag_untagged(conn, client=_FakeClient(boom))["tagged"] == 0
    assert len(store.untagged_reports(conn)) == 4


def test_tagger_unconfigured_does_nothing(db_path, monkeypatch):
    monkeypatch.setattr(tagger, "OPENAI_API_KEY", "")
    r = tagger.tag_untagged(store.connect(db_path))
    assert r == {"configured": False, "pending": 4, "tagged": 0}


def _tag(db_path, labels):
    conn = store.connect(db_path)
    keys = {k.split("?")[1][:2]: k for (k,) in conn.execute("SELECT report_key FROM reports")}
    store.save_tags(conn, {keys[k]: {"report_type": "company", "rating": None, **v}
                           for k, v in labels.items()}, "test", "2026-10-29T20:00")
    conn.close()


def test_state_tag_filters_labels_and_themes(db_path):
    _tag(db_path, {
        "A1": {"action": "results_beat", "tone": "bullish", "theme": "industrials"},
        "A2": {"action": "pt_cut", "tone": "bearish", "theme": "restaurants_leisure"},
        "A3": {"action": "upgrade", "tone": "bullish", "theme": "banks"},
    })
    s = build_rbc_state(days=None, db_path=db_path)
    assert s["tagged"] and s["untagged"] == 1
    assert s["reports"][0]["tone"] == "bullish" and s["reports"][-1]["tone"] is None
    assert {t["theme"]: (t["count"], t["bullish"], t["bearish"]) for t in s["themes"]} == {
        "industrials": (1, 1, 0), "restaurants_leisure": (1, 0, 1), "banks": (1, 1, 0)}
    company = lambda **kw: [r["company"][:6] for r in build_rbc_state(days=None, db_path=db_path, **kw)["reports"]]
    assert company(tone="bullish") == ["WESCO ", "Goldma"]
    assert company(action="positive") == ["WESCO ", "Goldma"]
    assert company(action="negative") == ["Life T"]
    assert company(theme="banks", tone="bullish") == ["Goldma"]
    assert len(company(tone="nonsense")) == 4               # unknown value ignored


def test_state_db_without_tag_table_reads_untagged(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE report_tags")
    conn.commit()
    conn.close()
    s = build_rbc_state(days=None, db_path=db_path)
    assert not s["tagged"] and len(s["reports"]) == 4 and s["themes"] == []
    assert build_rbc_state(days=None, tone="bullish", db_path=db_path)["reports"] == []


# ── prices (step 3; fake Yahoo, never the network) ─────────────────────────

import pandas as pd

from rbc_reports import prices


class _NoPace:
    def wait(self, n=1):
        pass


def test_names_match():
    assert prices.names_match("Goldman Sachs Group Inc.", "The Goldman Sachs Group, Inc.")
    assert prices.names_match("Bausch + Lomb", "Bausch + Lomb Corporation")
    assert prices.names_match("Toronto-Dominion Bank", "The Toronto-Dominion Bank")
    assert not prices.names_match("Loblaw Companies Limited", "Loews Corporation")
    assert not prices.names_match("First Horizon Corporation", "First Industrial Realty Trust, Inc.")
    assert not prices.names_match("", "Anything")


def test_yahoo_candidates():
    assert prices.yahoo_candidates("GS", "") == ["GS", "GS.TO", "GS.V"]
    assert prices.yahoo_candidates("GS", "US") == ["GS"]
    assert prices.yahoo_candidates("CTC.A", "CN") == ["CTC-A.TO", "CTC-A.V"]
    assert prices.yahoo_candidates("AV.", "") == ["AV", "AV.TO", "AV.V"]
    assert prices.yahoo_candidates("ANTO", "LN") == []


def test_resolve_symbols_verifies_company_name(db_path):
    names = {"WCC": "WESCO International, Inc.", "LTH": "Life Time Group Holdings, Inc.",
             "GS": "Loews-ish Unrelated Corp", "GS.TO": "Goldman Sachs Group Inc."}
    calls = []

    def info(sym):
        calls.append(sym)
        if sym not in names:
            raise KeyError(sym)
        return {"longName": names[sym], "currency": "CAD" if sym.endswith(".TO") else "USD"}
    conn = store.connect(db_path)
    r = prices.resolve_symbols(conn, info_fn=info, pacer=_NoPace())
    assert r == {"checked": 2, "resolved": 2, "stopped": False}       # LTH's note is a 4-ticker list
    got = dict(conn.execute("SELECT symbol, yahoo_symbol FROM symbol_map").fetchall())
    assert got == {"WCC": "WCC", "GS": "GS.TO"}
    calls.clear()
    assert prices.resolve_symbols(conn, info_fn=info, pacer=_NoPace())["checked"] == 0
    assert calls == []


def test_resolve_symbols_skips_depositary_receipts(db_path):
    info = {"WCC": {"longName": "WESCO International, Inc.", "quoteType": "ETF"},
            "WCC.TO": {"longName": "WESCO International, Inc.", "quoteType": "EQUITY"}}
    conn = store.connect(db_path)
    prices.resolve_symbols(conn, info_fn=lambda s: info.get(s, {}), pacer=_NoPace())
    assert conn.execute("SELECT yahoo_symbol FROM symbol_map WHERE symbol='WCC'").fetchone() == ("WCC.TO",)


def test_resolve_symbols_records_misses_and_stops_on_rate_limit(db_path):
    class YFRateLimitError(Exception):
        pass

    def limited(sym):
        raise YFRateLimitError("Too Many Requests")
    conn = store.connect(db_path)
    assert prices.resolve_symbols(conn, info_fn=limited, pacer=_NoPace())["stopped"]
    assert conn.execute("SELECT COUNT(*) FROM symbol_map").fetchone()[0] == 0
    r = prices.resolve_symbols(conn, info_fn=lambda s: {}, pacer=_NoPace())
    assert r["checked"] == 2 and r["resolved"] == 0
    assert prices.resolve_symbols(conn, info_fn=lambda s: {}, pacer=_NoPace())["checked"] == 0
    assert prices.resolve_symbols(conn, info_fn=lambda s: {}, pacer=_NoPace(), retry=True)["checked"] == 2


def _bars(closes, end="2026-10-29"):
    idx = pd.bdate_range(end=end, periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=idx)


def test_price_context_dip_and_turning_up():
    # A year falling from 100 to 60, then a 25-day bounce to 70.
    closes = [100 - 40 * i / 229 for i in range(230)] + [60 + 10 * i / 24 for i in range(25)]
    ctx = prices.price_context(_bars(closes))
    assert ctx["close"] == 70 and ctx["low_52w"] == 60
    assert ctx["pct_off_high"] < -0.2 and 0 < ctx["range_pos"] < 0.5
    assert ctx["turning_up"] == 1 and ctx["ret_20d"] > 0
    falling = prices.price_context(_bars([100 - i * 0.2 for i in range(255)]))
    assert falling["turning_up"] == 0 and falling["above_sma50"] == 0
    assert prices.price_context(_bars([10.0] * 30)) is None


def test_base_close_uses_last_close_before_publication():
    df = _bars([1.0, 2.0, 3.0], end="2026-10-29")          # 27th, 28th, 29th
    assert prices.base_close(df, "2026-10-29T18:24") == ("2026-10-29", 3.0)
    assert prices.base_close(df, "2026-10-29T05:00") == ("2026-10-28", 2.0)
    assert prices.base_close(df, "2026-10-01T05:00") is None


def _priced_db(db_path):
    """SAMPLE with tags + a symbol map: WCC bullish (beaten down, turning
    up), GS bullish upgrade (near its high), LTH bearish, A4 unresolved."""
    _tag(db_path, {
        "A1": {"action": "results_beat", "tone": "bullish", "theme": "industrials"},
        "A2": {"action": "pt_cut", "tone": "bearish", "theme": "restaurants_leisure"},
        "A3": {"action": "upgrade", "tone": "neutral", "theme": "banks"},
    })
    conn = store.connect(db_path)
    for sym, ys, company in (("WCC", "WCC", "WESCO International, Inc."),
                             ("GS", "GS", "Goldman Sachs Group Inc."),
                             ("LTH", "LTH", "Life Time Group Holdings, Inc.")):
        store.save_symbol(conn, sym, "", company, ys, sym, "USD", "2026-10-29")
    return conn


def test_sync_prices_writes_context_and_base_close(db_path):
    conn = _priced_db(db_path)
    dip = [100 - 40 * i / 229 for i in range(230)] + [60 + 10 * i / 24 for i in range(25)]
    bars = {"WCC": _bars(dip), "GS": _bars([50 + i * 0.1 for i in range(255)])}
    synced = []
    r = prices.sync_prices(conn, now=datetime(2026, 10, 29, 19, 30), pacer=_NoPace(), cache_conn=object(),
                           sync_fn=lambda chunk, *a, **k: synced.extend(chunk),
                           load_fn=lambda syms, *a, **k: {s: bars[s] for s in syms if s in bars})
    assert sorted(synced) == ["GS", "SPY", "WCC", "XIU.TO"]  # + benchmarks; LTH's 4-ticker list isn't a company note
    # WCC's 6:24pm note enters at the next session's close (not in the bars yet);
    # GS's previous-evening note enters at the 29th's close.
    assert r == {"symbols": 2, "with_context": 2, "scored": 1, "stopped": False}
    assert dict(conn.execute("SELECT yahoo_symbol, turning_up FROM price_context").fetchall()) == {
        "WCC": 1, "GS": 1}
    base = conn.execute("SELECT base_date, base_close FROM report_prices WHERE yahoo_symbol='WCC'").fetchone()
    assert base == ("2026-10-29", 70.0)                     # 6:24pm note -> that day's close


def test_sync_prices_stops_when_a_chunk_gets_no_bars(db_path):
    conn = _priced_db(db_path)
    r = prices.sync_prices(conn, now=datetime(2026, 10, 29, 19, 30), pacer=_NoPace(), cache_conn=object(),
                           sync_fn=lambda *a, **k: None, load_fn=lambda *a, **k: {})
    assert r["stopped"] and r["with_context"] == 0


def test_due_today_once_per_trading_day(db_path):
    conn = store.connect(db_path)
    fri = datetime(2026, 10, 30, 19, 30)
    assert prices.due_today(conn, now=fri)
    store.set_meta(conn, "prices_last_run", "2026-10-30")
    assert not prices.due_today(conn, now=fri)
    assert not prices.due_today(conn, now=datetime(2026, 10, 31, 19, 30))   # Saturday
    assert prices.due_today(conn, now=datetime(2026, 11, 2, 19, 30))        # Monday


def test_state_candidates(db_path):
    conn = _priced_db(db_path)
    store.save_price_context(conn, {
        "WCC": {"as_of": "2026-10-29", "close": 70.0, "high_52w": 100.0, "low_52w": 60.0,
                "pct_off_high": -0.3, "range_pos": 0.25, "ret_20d": 0.1, "above_sma50": 1, "turning_up": 1},
        "GS": {"as_of": "2026-10-29", "close": 75.0, "high_52w": 76.0, "low_52w": 50.0,
               "pct_off_high": -0.013, "range_pos": 0.96, "ret_20d": 0.04, "above_sma50": 1, "turning_up": 0},
    }, "2026-10-29T19:40")
    store.save_report_prices(conn, [(k, "WCC", "2026-10-29", 50.0) for (k,) in conn.execute(
        "SELECT report_key FROM reports WHERE title LIKE 'WESCO%'")])
    conn.close()
    s = build_rbc_state(days=7, db_path=db_path, now=_NOW)
    assert [c["symbol"] for c in s["candidates"]] == ["WCC"]           # GS too close to its high
    wcc = s["candidates"][0]
    assert wcc["turning_up"] and round(wcc["since_note"], 2) == 0.4 and wcc["tone"] == "bullish"
    assert s["candidates_all"] == 2 and s["candidates_unpriced"] == 0
    s = build_rbc_state(days=7, db_path=db_path, now=_NOW, show_all_candidates=True)
    assert [c["symbol"] for c in s["candidates"]] == ["WCC", "GS"]     # most beaten-down first
    t = build_rbc_state(ticker="WCC", db_path=db_path, now=_NOW)
    assert t["ticker_price"]["pct_off_high"] == -0.3


def test_same_symbol_for_two_companies_maps_separately(tmp_path):
    page = _digest([
        _item("6:00pm ET", "Cullen/Frost Bankers, Inc. - 3Q beat", "CFR", "A", link_id="C1", border=False),
        _item("5:00pm ET", "Compagnie Financiere Richemont SA - H1 preview", "CFR", "B", link_id="C2"),
    ], shown=2, total=2)
    conn = store.connect(tmp_path / "x.db")
    store.save_digest(conn, parse_digest(page), "x.html", "2026-10-29T19:30")
    info = {"CFR": {"longName": "Cullen/Frost Bankers, Inc.", "currency": "USD"}}
    r = prices.resolve_symbols(conn, info_fn=lambda s: info.get(s, {}), pacer=_NoPace())
    assert r == {"checked": 2, "resolved": 1, "stopped": False}
    assert dict(conn.execute("SELECT company, yahoo_symbol FROM symbol_map").fetchall()) == {
        "Cullen/Frost Bankers, Inc.": "CFR", "Compagnie Financiere Richemont SA": None}



# ── outcomes (step 5) ─────────────────────────────────────────────────────

from rbc_reports import outcomes


def test_entry_is_first_tradeable_close():
    df = _bars([1.0, 2.0, 3.0], end="2026-10-29")          # 27th, 28th, 29th
    assert outcomes.entry_index(df, "2026-10-27T18:24") == 1   # evening -> next session
    assert outcomes.entry_index(df, "2026-10-27T05:00") == 0   # pre-open -> that day
    assert outcomes.entry_index(df, "2026-10-29T19:00") is None


def test_compute_outcome_returns_vs_benchmark_and_unfilled_horizons():
    closes = [100.0] * 100 + [100 + i for i in range(1, 31)]    # flat, then +1/day
    df = _bars(closes)
    bench = _bars([200.0] * 120 + [202.0] * 10)
    published = df.index[99].strftime("%Y-%m-%dT18:00")       # evening of the last flat day
    o = outcomes.compute_outcome(df, bench, published)
    assert o["entry_date"] == df.index[100].strftime("%Y-%m-%d") and o["entry_close"] == 101.0
    assert o["ret_5"] == round(106 / 101 - 1, 4) and o["bench_5"] == 0.0
    assert o["ret_20"] == round(121 / 101 - 1, 4) and o["bench_20"] == 0.01
    assert o["ret_60"] is None and o["bench_60"] is None
    assert o["off_high"] == 0.0                                 # judged from bars up to entry only


def test_benchmark_for():
    assert outcomes.benchmark_for("GS") == "SPY"
    assert outcomes.benchmark_for("CS.TO") == "XIU.TO"


def _event(sym, day, bullish, tone, off_high, turning, r20, b20, published=None):
    return {"yahoo_symbol": sym, "entry_date": day, "published_at": published or day + "T18:00",
            "bullish": bullish, "tone": tone, "off_high": off_high, "turning_up": turning,
            "ret_5": None, "bench_5": None, "ret_20": r20, "bench_20": b20, "ret_60": None, "bench_60": None}


def test_summarize_groups_and_dedups():
    events = [
        _event("A", "2026-09-01", True, "bullish", -0.3, 1, 0.10, 0.02),
        _event("A", "2026-09-01", True, "bullish", -0.3, 1, 0.10, 0.02, "2026-09-01T19:00"),  # same trade
        _event("B", "2026-09-02", True, "bullish", -0.25, 0, -0.05, 0.01),
        _event("C", "2026-09-02", True, "neutral", -0.05, 1, 0.03, 0.01),   # positive action, near high
        _event("D", "2026-09-03", False, "bearish", -0.4, 0, -0.08, 0.0),
    ]
    rows = {r["label"]: r for r in outcomes.summarize(events)}
    rule = rows["Bullish, beaten down, turning up (the Candidates rule)"]
    assert rule["events"] == 1 and rule["by_horizon"][20]["n"] == 1
    assert round(rule["by_horizon"][20]["avg_excess"], 4) == 0.08
    assert rows["Bullish, beaten down"]["by_horizon"][20]["hit_rate"] == 0.5
    assert rows["Bullish, not beaten down"]["events"] == 1
    assert rows["All bullish"]["events"] == 3
    assert rows["Bearish"]["by_horizon"][20]["avg_excess"] == -0.08
    assert rows["Neutral"]["events"] == 0
    assert rule["by_horizon"][5] is None


def test_state_track_record(db_path):
    conn = _priced_db(db_path)
    keys = {t[:5]: k for k, t in conn.execute("SELECT report_key, title FROM reports")}
    base = {"benchmark": "SPY", "entry_close": 10.0, "ret_5": 0.02, "bench_5": 0.01,
            "ret_20": None, "bench_20": None, "ret_60": None, "bench_60": None}
    store.save_outcomes(conn, {
        keys["WESCO"]: {**base, "yahoo_symbol": "WCC", "entry_date": "2026-10-30", "off_high": -0.3, "turning_up": 1},
        keys["Goldm"]: {**base, "yahoo_symbol": "GS", "entry_date": "2026-10-29", "off_high": -0.01, "turning_up": 0},
    }, "2026-10-30T19:40")
    conn.close()
    s = build_rbc_state(days=7, db_path=db_path, now=_NOW)
    rows = {r["label"]: r for r in s["track_record"]}
    assert s["track_events"] == 2 and s["track_first"] == "2026-10-29"
    assert rows["Bullish, beaten down, turning up (the Candidates rule)"]["by_horizon"][5]["n"] == 1
    assert rows["All bullish"]["by_horizon"][5]["n"] == 2       # GS: neutral tone but an upgrade
