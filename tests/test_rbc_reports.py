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
    assert s["digests"] == [{"date": "2026-10-29", "shown": 4, "total": 9, "missed": 5}]


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
