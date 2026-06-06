"""Offline test for edgar_service.run_collector event-loop logic (no network)."""

from datetime import datetime

import edgar_service
from edgar import store
from time_utils import TSX_TZ, set_backtest_clock


def test_run_collector_flags_today_activist_and_dedups(tmp_path, monkeypatch):
    # Redirect the store to a throwaway DB.
    tmp_conn = store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(edgar_service.store, "connect", lambda *a, **k: tmp_conn)

    today = "2026-06-05"
    canned = [
        {"cik": 1, "ticker": "XYZ", "form": "SC 13D", "category": "activist_stake",
         "date": today, "url": "u1"},                       # today's 13D -> flagged
        {"cik": 2, "ticker": "OLD", "form": "SC 13D", "category": "activist_stake",
         "date": "2026-06-01", "url": "u2"},                # older -> stored, not emailed
        {"cik": 3, "ticker": "MU", "form": "4", "category": "insider_txn",
         "date": today, "url": "u3"},                       # not watchlisted -> no insider flag
        {"date": "2026-06-04", "error": "404"},             # missing index -> tolerated
    ]
    monkeypatch.setattr(edgar_service, "scan_range", lambda *a, **k: canned)
    monkeypatch.setattr(edgar_service, "load_cik_to_ticker", lambda *a, **k: {})
    monkeypatch.setattr(
        edgar_service, "fetch_activist_filing",
        lambda url, accession=None: {
            "filer": "ACME PARTNERS LP", "pct": 6.1, "subject": "XYZ INC",
            "accession": "acc-xyz", "raw_text": "<full 13D text>",
        },
    )

    captured = {}
    monkeypatch.setattr(
        edgar_service, "send_text_email",
        lambda subject, body: captured.update(subject=subject, body=body) or True,
    )

    set_backtest_clock(datetime(2026, 6, 5, 18, 30, tzinfo=TSX_TZ))
    try:
        edgar_service.run_collector("rid", dry_run=False)
    finally:
        set_backtest_clock(None)

    # Today's fresh 13D emailed (enriched with filer/pct); the older one and the
    # market-wide Form 4 (no watchlist) excluded.
    assert captured, "expected a digest to be sent"
    assert "0 insider buys, 1 activist" in captured["subject"]
    assert "XYZ" in captured["body"]
    assert "by ACME PARTNERS LP" in captured["body"]
    assert "6.1%" in captured["body"]
    assert "OLD" not in captured["body"]

    # All non-error hits were stored (raw market-wide log), the flagged 13D body
    # was persisted, and today is marked sent.
    stored = tmp_conn.execute("SELECT COUNT(*) FROM scan_hits").fetchone()[0]
    assert stored == 3
    assert tmp_conn.execute("SELECT COUNT(*) FROM activist_filings").fetchone()[0] == 1
    assert store.already_sent(tmp_conn, today) is True
