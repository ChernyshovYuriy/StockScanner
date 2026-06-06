"""Offline tests for edgar_report — emails latest fresh 13D from scan_hits."""

import edgar_report
from edgar import store
from edgar.digest import build_digest


def _seed_scan(conn):
    """Seed scan_hits: filing a1 appears under both filer + subject CIK (one
    has the ticker), an older filing a2, and a 13G that must be ignored."""
    store.save_scan_hits(conn, [
        {"cik": 10, "ticker": None, "form": "SC 13D", "category": "activist_stake",
         "date": "2026-06-05", "url": "https://sec.gov/data/10/a1.txt"},
        {"cik": 11, "ticker": "AAA", "form": "SC 13D", "category": "activist_stake",
         "date": "2026-06-05", "url": "https://sec.gov/data/11/a1.txt"},
        {"cik": 20, "ticker": "BBB", "form": "SC 13D", "category": "activist_stake",
         "date": "2026-06-04", "url": "https://sec.gov/data/20/a2.txt"},
        {"cik": 30, "ticker": "CCC", "form": "SC 13G", "category": "passive_stake",
         "date": "2026-06-05", "url": "https://sec.gov/data/30/g1.txt"},
    ])


def _fake_fetch(url, accession=None):
    return {"filer": "ACME LP", "pct": 6.1, "subject": "S",
            "accession": "acc-" + url.split("/")[-1], "raw_text": "x"}


def test_latest_activist_from_scan_hits_dedups_enriches_and_persists(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "edgar.db")
    _seed_scan(conn)
    monkeypatch.setattr(edgar_report, "fetch_activist_filing", _fake_fetch)

    hits = edgar_report.latest_activist(conn, 10)
    # a1 deduped to its ticker'd subject row (AAA), then a2 (BBB); 13G excluded;
    # newest first.
    assert [h.get("ticker") for h in hits] == ["AAA", "BBB"]
    assert all(h.get("filer") == "ACME LP" for h in hits)
    # Parsed bodies were persisted as evidence.
    assert conn.execute("SELECT COUNT(*) FROM activist_filings").fetchone()[0] == 2


def test_report_digest_includes_filers(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "edgar.db")
    _seed_scan(conn)
    monkeypatch.setattr(edgar_report, "fetch_activist_filing", _fake_fetch)
    subject, body = build_digest("latest", [], edgar_report.latest_activist(conn, 10))
    assert "2 activist" in subject
    assert "ACME LP" in body and "6.1%" in body


def test_report_empty_db_is_quiet(tmp_path, monkeypatch, capsys):
    conn = store.connect(tmp_path / "edgar.db")          # no scan_hits seeded
    monkeypatch.setattr(edgar_report.store, "connect", lambda *a, **k: conn)
    monkeypatch.setattr(edgar_report, "send_text_email",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not send")))
    edgar_report.run_report(20, dry_run=False)
    assert "No stored records" in capsys.readouterr().out
