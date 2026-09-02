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


def test_run_collector_across_two_days_both_get_emailed(tmp_path, monkeypatch):
    """Integration regression: this whole review started from a bug where the
    live collector went quiet forever after its first day of real operation.
    Two full run_collector() invocations, one real DB/cache carried across
    them (as in live daily runs), each day's own fresh 13D must still get
    emailed -- day 2 must not be silently blocked or merged into day 1."""
    tmp_conn = store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(edgar_service.store, "connect", lambda *a, **k: tmp_conn)
    monkeypatch.setattr(edgar_service, "load_cik_to_ticker", lambda *a, **k: {})
    monkeypatch.setattr(
        edgar_service, "fetch_activist_filing",
        lambda url, accession=None: {
            "filer": f"FILER FOR {url}", "pct": 5.0, "subject": "SUBJ",
            "accession": url, "raw_text": "<text>",
        },
    )
    captured = []
    monkeypatch.setattr(
        edgar_service, "send_text_email",
        lambda subject, body: captured.append({"subject": subject, "body": body}) or True,
    )

    day1, day2 = "2026-06-05", "2026-06-08"   # Fri -> Mon, mirrors a live gap

    monkeypatch.setattr(edgar_service, "scan_range", lambda *a, **k: [
        {"cik": 1, "ticker": "XYZ", "form": "SC 13D", "category": "activist_stake",
         "date": day1, "url": "u-day1"},
    ])
    set_backtest_clock(datetime(2026, 6, 5, 21, 30, tzinfo=TSX_TZ))
    try:
        edgar_service.run_collector("rid1", dry_run=False)
    finally:
        set_backtest_clock(None)

    monkeypatch.setattr(edgar_service, "scan_range", lambda *a, **k: [
        {"cik": 2, "ticker": "ABC", "form": "SC 13D", "category": "activist_stake",
         "date": day2, "url": "u-day2"},
    ])
    set_backtest_clock(datetime(2026, 6, 8, 21, 30, tzinfo=TSX_TZ))
    try:
        edgar_service.run_collector("rid2", dry_run=False)
    finally:
        set_backtest_clock(None)

    assert len(captured) == 2, "expected a separate email each day, not 0 or 1"
    assert "XYZ" in captured[0]["body"] and "ABC" not in captured[0]["body"]
    assert "ABC" in captured[1]["body"] and "XYZ" not in captured[1]["body"]
    assert store.already_sent(tmp_conn, day1) is True
    assert store.already_sent(tmp_conn, day2) is True


def test_run_collector_falls_back_when_todays_index_is_missing(tmp_path, monkeypatch):
    """Regression: today's master.idx often isn't published yet when this runs.

    scan_range returns only yesterday's hits (today's fetch failed -> "error"
    entry, no rows dated today). The collector must still flag yesterday's
    13D instead of silently treating the run as a quiet day forever.
    """
    tmp_conn = store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(edgar_service.store, "connect", lambda *a, **k: tmp_conn)

    today = "2026-06-05"
    yesterday = "2026-06-04"
    canned = [
        {"cik": 1, "ticker": "XYZ", "form": "SC 13D", "category": "activist_stake",
         "date": yesterday, "url": "u1"},                   # yesterday's 13D, backfilled
        {"date": today, "error": "404"},                    # today's index not published yet
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

    assert captured, "expected yesterday's 13D to still be emailed"
    assert "1 activist" in captured["subject"]
    assert "XYZ" in captured["body"]

    # The flag lands under yesterday's date (the actual filing date), not today's.
    assert store.already_sent(tmp_conn, yesterday) is True
    assert store.already_sent(tmp_conn, today) is False


def _seed_watchlist_form4_day(tmp_conn, monkeypatch, today, activity):
    """Shared setup: one watchlisted CIK with a Form 4 on `today`, backed by
    the given `activity` (edgar.insiders.get_recent_insider_activity shape)."""
    store.set_watchlist(tmp_conn, [("MU", 723125)])
    canned = [
        {"cik": 723125, "ticker": "MU", "form": "4", "category": "insider_txn",
         "date": today, "url": "u1"},
    ]
    monkeypatch.setattr(edgar_service, "scan_range", lambda *a, **k: canned)
    monkeypatch.setattr(edgar_service, "load_cik_to_ticker", lambda *a, **k: {723125: "MU"})
    monkeypatch.setattr(edgar_service, "get_recent_insider_activity", lambda cik: activity)


def test_run_collector_filters_buys_below_materiality_floor(tmp_path, monkeypatch):
    """Regression: EDGAR_MIN_BUY_VALUE was defined in config but never
    applied -- every open-market buy, however small, got emailed."""
    tmp_conn = store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(edgar_service.store, "connect", lambda *a, **k: tmp_conn)
    monkeypatch.setattr(edgar_service, "EDGAR_MIN_BUY_VALUE", 10_000)

    today = "2026-06-05"
    activity = [{
        "owner": "DOE JANE", "is_officer": False, "is_director": True,
        "filing_date": today, "accession": "acc-1",
        "transactions": [{"code": "P", "shares": 100.0, "price": 5.0,
                           "date": today, "direction": "A"}],   # $500 -> below floor
    }]
    _seed_watchlist_form4_day(tmp_conn, monkeypatch, today, activity)
    monkeypatch.setattr(
        edgar_service, "send_text_email",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not send: below floor")),
    )

    set_backtest_clock(datetime(2026, 6, 5, 21, 30, tzinfo=TSX_TZ))
    try:
        edgar_service.run_collector("rid", dry_run=False)
    finally:
        set_backtest_clock(None)

    # Below the floor -> quiet day (no email), but still stored as evidence.
    assert store.already_sent(tmp_conn, today) is False
    assert tmp_conn.execute("SELECT COUNT(*) FROM insider_buys").fetchone()[0] == 1


def test_run_collector_flags_cluster_of_distinct_insiders(tmp_path, monkeypatch):
    """Regression: cluster_flag() (2+ distinct insiders buying) was
    implemented and unit-tested but never wired into the live digest."""
    tmp_conn = store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(edgar_service.store, "connect", lambda *a, **k: tmp_conn)
    monkeypatch.setattr(edgar_service, "EDGAR_MIN_BUY_VALUE", 1_000)

    today = "2026-06-05"
    activity = [
        {"owner": "DOE JANE", "is_officer": False, "is_director": True,
         "filing_date": today, "accession": "acc-1",
         "transactions": [{"code": "P", "shares": 1000.0, "price": 5.0,
                            "date": today, "direction": "A"}]},   # $5,000
        {"owner": "SMITH BOB", "is_officer": True, "is_director": False,
         "filing_date": today, "accession": "acc-2",
         "transactions": [{"code": "P", "shares": 2000.0, "price": 10.0,
                            "date": today, "direction": "A"}]},   # $20,000
    ]
    _seed_watchlist_form4_day(tmp_conn, monkeypatch, today, activity)

    captured = {}
    monkeypatch.setattr(
        edgar_service, "send_text_email",
        lambda subject, body: captured.update(subject=subject, body=body) or True,
    )

    set_backtest_clock(datetime(2026, 6, 5, 21, 30, tzinfo=TSX_TZ))
    try:
        edgar_service.run_collector("rid", dry_run=False)
    finally:
        set_backtest_clock(None)

    assert captured, "expected a digest to be sent"
    assert "2 insider buys," in captured["subject"]
    assert "DOE JANE" in captured["body"] and "SMITH BOB" in captured["body"]
    assert captured["body"].count(">> CLUSTER") == 2


def test_run_collector_single_insider_buy_has_no_cluster_marker(tmp_path, monkeypatch):
    """Negative case: one insider buying alone must not read as a cluster."""
    tmp_conn = store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(edgar_service.store, "connect", lambda *a, **k: tmp_conn)
    monkeypatch.setattr(edgar_service, "EDGAR_MIN_BUY_VALUE", 1_000)

    today = "2026-06-05"
    activity = [{
        "owner": "DOE JANE", "is_officer": False, "is_director": True,
        "filing_date": today, "accession": "acc-1",
        "transactions": [{"code": "P", "shares": 1000.0, "price": 5.0,
                           "date": today, "direction": "A"}],   # $5,000
    }]
    _seed_watchlist_form4_day(tmp_conn, monkeypatch, today, activity)

    captured = {}
    monkeypatch.setattr(
        edgar_service, "send_text_email",
        lambda subject, body: captured.update(subject=subject, body=body) or True,
    )

    set_backtest_clock(datetime(2026, 6, 5, 21, 30, tzinfo=TSX_TZ))
    try:
        edgar_service.run_collector("rid", dry_run=False)
    finally:
        set_backtest_clock(None)

    assert captured, "expected a digest to be sent"
    assert ">> CLUSTER" not in captured["body"]


def test_run_collector_flags_new_insider_form3(tmp_path, monkeypatch):
    """Regression: Form 3 (new Section 16 filer) was excluded from
    EDGAR_FORMS entirely, so a new insider joining a watchlisted company
    never surfaced -- even with no buy attached to size it."""
    tmp_conn = store.connect(tmp_path / "edgar.db")
    monkeypatch.setattr(edgar_service.store, "connect", lambda *a, **k: tmp_conn)
    store.set_watchlist(tmp_conn, [("MU", 723125)])

    today = "2026-06-05"
    canned = [
        {"cik": 723125, "ticker": "MU", "form": "3", "category": "insider_init",
         "date": today, "url": "u1"},
    ]
    monkeypatch.setattr(edgar_service, "scan_range", lambda *a, **k: canned)
    monkeypatch.setattr(edgar_service, "load_cik_to_ticker", lambda *a, **k: {723125: "MU"})

    captured = {}
    monkeypatch.setattr(
        edgar_service, "send_text_email",
        lambda subject, body: captured.update(subject=subject, body=body) or True,
    )

    set_backtest_clock(datetime(2026, 6, 5, 21, 30, tzinfo=TSX_TZ))
    try:
        edgar_service.run_collector("rid", dry_run=False)
    finally:
        set_backtest_clock(None)

    assert captured, "expected a digest to be sent for a lone Form 3"
    assert "1 new insider" in captured["subject"]
    assert "NEW INSIDERS" in captured["body"] and "MU" in captured["body"]
