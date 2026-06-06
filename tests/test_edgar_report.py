"""Offline tests for edgar_report — the read-only 'email latest N records' command."""

import edgar_report
from edgar import store
from edgar.digest import build_digest


def _seed(conn):
    store.save_activist_filing(conn, {
        "accession": "a1", "cik": 1, "ticker": "AAA", "form": "SC 13D",
        "filer": "FUND ONE LP", "subject": "AAA INC", "pct": 7.0,
        "date": "2026-06-05", "url": "u1", "raw_text": "x"})
    store.save_activist_filing(conn, {
        "accession": "a2", "cik": 2, "ticker": "BBB", "form": "SC 13D",
        "filer": "FUND TWO LP", "subject": "BBB INC", "pct": 5.5,
        "date": "2026-06-04", "url": "u2", "raw_text": "y"})


def test_latest_activist_orders_newest_first_and_limits(tmp_path):
    conn = store.connect(tmp_path / "edgar.db")
    _seed(conn)
    rows = edgar_report.latest_activist(conn, 1)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAA"          # 2026-06-05 newer than 2026-06-04
    assert len(edgar_report.latest_activist(conn, 10)) == 2


def test_report_digest_includes_stored_filers(tmp_path):
    conn = store.connect(tmp_path / "edgar.db")
    _seed(conn)
    subject, body = build_digest("latest", [], edgar_report.latest_activist(conn, 10))
    assert "2 activist" in subject
    assert "FUND ONE LP" in body and "FUND TWO LP" in body


def test_report_empty_db_is_quiet(tmp_path, monkeypatch, capsys):
    conn = store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(edgar_report.store, "connect", lambda *a, **k: conn)
    # No records, and email must never be attempted.
    monkeypatch.setattr(edgar_report, "send_text_email",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not send")))
    edgar_report.run_report(20, dry_run=False)
    assert "No stored records" in capsys.readouterr().out
